# coding=utf-8
# Copyright 2024 The HunYuan team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom attention metadata, 2D RoPE and image KV-cache helpers used by the
HunyuanImage-3 model during image generation.

Ported from the official HunyuanImage-3 model repository
(`modeling_hunyuan_image_3.py`).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================
# 1. Custom attention meta.
# =============================================================

@dataclass
class HunYuanImageAttentionMeta:
    query_lens: list[int]
    seq_lens: list[int]
    num_image_tokens: int
    first_step: bool


def create_hunyuan_image_attention_meta(
    attention_mask: torch.Tensor, num_image_tokens: int, first_step: bool
) -> HunYuanImageAttentionMeta:
    b, _, q_len1, seq_len = attention_mask.shape
    return HunYuanImageAttentionMeta(
        query_lens=[q_len1] * b,
        seq_lens=[seq_len] * b,
        num_image_tokens=num_image_tokens,
        first_step=first_step,
    )


# =============================================================
# 2. RoPE helpers
# =============================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False):
    """Applies Rotary Position Embedding to the query and key tensors.

    Supports both full-dim (cos/sin last dim == q last dim) and half-dim
    (cos/sin last dim == q last dim // 2) layouts.  The half-dim layout
    applies each rotation angle to a PAIR of consecutive dimensions,
    matching the original HunyuanImage-3 model's ``rotated_half`` mode.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    head_dim = q.shape[-1]
    cos_dim = cos.shape[-1]

    if cos_dim == head_dim:
        # Standard full-dim RoPE (original path)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        if mla:
            b, h, s, d = q.shape
            q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            b, h, s, d = k.shape
            k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        # Half-dim RoPE: cos/sin has head_dim//2 elements.
        # Reshape q/k into pairs and apply each angle to a pair.
        assert cos_dim == head_dim // 2, (
            f"cos last dim {cos_dim} must be head_dim//2 ({head_dim // 2})"
        )
        # q: [..., head_dim] -> [..., head_dim//2, 2]
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(*q_shape[:-1], head_dim // 2, 2)
        k = k.reshape(*k_shape[:-1], head_dim // 2, 2)
        # cos/sin: [..., cos_dim] -> [..., cos_dim, 1] for broadcasting
        # against the pair dimension (size 2) of q/k.
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        q_embed = (q * cos) + (
            torch.stack((-q[..., 1], q[..., 0]), dim=-1) * sin
        )
        k_embed = (k * cos) + (
            torch.stack((-k[..., 1], k[..., 0]), dim=-1) * sin
        )
        # Restore original shape
        q_embed = q_embed.reshape(q_shape)
        k_embed = k_embed.reshape(k_shape)

    return q_embed, k_embed


class HunYuanRotary2DEmbedder:
    """
    A RoPE wrapper specifically designed for HunYuan-Image attention.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.custom_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None

    def _prepare_cos_sin(
        self,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
        device: torch.device,
    ):
        if first_step:
            cos_input, sin_input = custom_pos_emb
            cos = cos_input.to(device)
            sin = sin_input.to(device)
            self.custom_pos_emb = None
        else:
            if self.custom_pos_emb is None:
                cos_input, sin_input = custom_pos_emb
                cos = cos_input.to(device)
                sin = sin_input.to(device)
                self.custom_pos_emb = (cos, sin)
            else:
                cos, sin = self.custom_pos_emb
        return cos, sin

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        attn_meta: HunYuanImageAttentionMeta | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attn_meta is None:
            return q, k

        first_step = attn_meta.first_step
        device = q.device
        cos, sin = self._prepare_cos_sin(custom_pos_emb, first_step, device)

        total_tokens = q.shape[0]
        bs = len(attn_meta.query_lens)
        q_len = total_tokens // bs

        # Cast to float32 for RoPE precision (matching vllm-omni)
        q = q.reshape(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2).to(torch.float32)
        k = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).to(torch.float32)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = (
            q.transpose(1, 2)
            .reshape(total_tokens, self.num_heads * self.head_dim)
            .to(torch.bfloat16)
        )
        k = (
            k.transpose(1, 2)
            .reshape(total_tokens, self.num_kv_heads * self.head_dim)
            .to(torch.bfloat16)
        )

        return q, k


# =============================================================
# 3. 2D RoPE construction (ported from official model repo)
# =============================================================

def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device="cpu"):
    """Get n-D meshgrid with start, stop and num."""
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)
    return grid


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    """Build 2D RoPE cos/sin tables.

    Text positions use sequential 1D indices (y = x = index).
    Image positions use 2D grid indices derived from the image layout.

    Returns
    -------
    cos: torch.Tensor with shape [seq_len, n_elem]
    sin: torch.Tensor with shape [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d, 2]

    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L, device=device))
                x_sections.append(torch.arange(last_pos, L, device=device))
            elif h is None:
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                continue
            else:
                pass
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w), device=device)
            grid = grid.reshape(2, -1)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            last_pos = L + w * h
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)  # [seq_len, 1, 2]

    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 1)
    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos
    return cos, sin


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    """Build batched 2D RoPE cos/sin tables."""
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list
    return stacked_cos, stacked_sin


class CachedRoPE:
    """Cache for 2D RoPE cos/sin tables to avoid recomputation across diffusion steps."""

    def __init__(self, rope_theta: float, head_dim: int, rope_type: str = "2d"):
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.rope_type = rope_type
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_image_info = None

    def __call__(self, seq_len, device, rope_image_info=None, position_ids=None):
        if (self.seq_len != seq_len) or (rope_image_info is not None and self.rope_image_info != rope_image_info):
            if self.rope_type in ["2d", "default"]:
                self.cos_cache, self.sin_cache = build_batch_2d_rope(
                    image_infos=rope_image_info,
                    seq_len=seq_len,
                    n_elem=self.head_dim,
                    device=device,
                    base=self.rope_theta,
                )
            else:
                raise NotImplementedError(f"rope_type `{self.rope_type}` not supported")
            self.seq_len = seq_len
            self.rope_image_info = rope_image_info

        if position_ids is None:
            cos, sin = self.cos_cache, self.sin_cache
        else:
            assert position_ids.dim() == 2, f"{position_ids.shape=}"
            head_size = self.cos_cache.size(-1)
            cos = torch.gather(self.cos_cache, dim=1, index=position_ids.unsqueeze(-1).expand(-1, -1, head_size))
            sin = torch.gather(self.sin_cache, dim=1, index=position_ids.unsqueeze(-1).expand(-1, -1, head_size))

        return cos, sin


# =============================================================
# 4. Timestep embedding helpers (ported from official model repo)
# =============================================================

def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# =============================================================
# 5. Custom attention impl (KV cache for image tokens).
# =============================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ImageKVCacheManager:
    """
    Manages specialized caching and updating of KV-Cache for image tokens.
    """

    def __init__(self, image_token_len: int = 4097):
        self.image_token_len: int = image_token_len
        self.image_kv_cache_map: tuple[torch.Tensor, torch.Tensor] | None = None

    def _save_image_kv_caches(self, key, value, seq_len):
        bs, q_len, num_kv_heads, head_dim = key.shape
        assert q_len == seq_len

        key = key.reshape(-1, num_kv_heads, head_dim)
        value = value.reshape(-1, num_kv_heads, head_dim)

        cached_prompt_len = seq_len - self.image_token_len - 1
        cached_key = [key[:cached_prompt_len], key[seq_len - 1 : seq_len]]
        cached_value = [value[:cached_prompt_len], value[seq_len - 1 : seq_len]]

        if bs > 1:
            assert bs == 2
            cached_key.append(key[seq_len : seq_len + cached_prompt_len])
            cached_key.append(key[-1:])
            cached_value.append(value[seq_len : seq_len + cached_prompt_len])
            cached_value.append(value[-1:])

        cached_key = torch.cat(cached_key, dim=0)
        cached_value = torch.cat(cached_value, dim=0)
        self.image_kv_cache_map = (cached_key, cached_value)

    def _update_image_kv_caches(self, key, value, seq_len):
        cached_key, cached_value = self.image_kv_cache_map
        bs, q_len, num_kv_heads, head_dim = key.shape

        cached_prompt_len = cached_key.shape[0] // bs - 1
        assert (cached_prompt_len + 1) == (seq_len - q_len)

        key = key.reshape(-1, num_kv_heads, head_dim)
        value = value.reshape(-1, num_kv_heads, head_dim)

        new_key = [
            cached_key[:cached_prompt_len],
            key[:q_len],
            cached_key[cached_prompt_len : cached_prompt_len + 1],
        ]
        new_value = [
            cached_value[:cached_prompt_len],
            value[:q_len],
            cached_value[cached_prompt_len : cached_prompt_len + 1],
        ]

        if bs > 1:
            assert bs == 2
            new_key.append(
                cached_key[cached_prompt_len + 1 : cached_prompt_len + 1 + cached_prompt_len]
            )
            new_key.append(key[q_len:])
            new_key.append(cached_key[-1:])
            new_value.append(
                cached_value[cached_prompt_len + 1 : cached_prompt_len + 1 + cached_prompt_len]
            )
            new_value.append(value[q_len:])
            new_value.append(cached_value[-1:])

        new_key = torch.cat(new_key, dim=0)
        new_value = torch.cat(new_value, dim=0)
        new_key = new_key.reshape(bs, seq_len, num_kv_heads, head_dim)
        new_value = new_value.reshape(bs, seq_len, num_kv_heads, head_dim)

        return new_key.contiguous(), new_value.contiguous()

    def __call__(self, query, key, value, attn_metadata, attention_mask=None):
        assert attn_metadata is not None
        self.image_token_len = attn_metadata.num_image_tokens
        first_step = attn_metadata.first_step

        total_tokens = query.shape[0]
        bs = len(attn_metadata.query_lens)
        q_len = total_tokens // bs

        head_num_per_rank = query.shape[1]
        kv_head_num_per_rank = key.shape[1]
        repeat_num = head_num_per_rank // kv_head_num_per_rank
        head_dim = query.shape[2]

        query = query.reshape(bs, q_len, head_num_per_rank, head_dim)
        key = key.reshape(bs, q_len, kv_head_num_per_rank, head_dim)
        value = value.reshape(bs, q_len, kv_head_num_per_rank, head_dim)

        # Full attention every step – no KV cache needed.
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        key = repeat_kv(key, repeat_num)
        value = repeat_kv(value, repeat_num)

        attention_mask = attention_mask.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(total_tokens, head_num_per_rank, head_dim)
        return attn_output
