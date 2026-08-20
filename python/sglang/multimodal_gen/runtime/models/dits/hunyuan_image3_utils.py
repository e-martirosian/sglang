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

Ported from the vLLM implementation
(`vllm/model_executor/models/hunyuan_image3_utils.py`).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# 1. custom attention meta.
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


# 2. custom Rope2D impl.
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if mla:
        b, h, s, d = q.shape
        q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
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

        bs = len(attn_metadata.query_lens)
        q_len = attn_meta.query_lens[0]
        assert hidden_states.shape[0] == bs * q_len

        q = q.reshape(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = (
            q.transpose(1, 2)
            .reshape(hidden_states.shape[0], self.num_heads * self.head_dim)
            .to(torch.bfloat16)
        )
        k = (
            k.transpose(1, 2)
            .reshape(hidden_states.shape[0], self.num_kv_heads * self.head_dim)
            .to(torch.bfloat16)
        )

        return q, k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 3. custom attention impl.
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

        bs = len(attn_metadata.query_lens)
        q_len = attn_metadata.query_lens[0]
        seq_len = attn_metadata.seq_lens[0]
        assert query.shape[0] == bs * q_len

        head_num_per_rank = query.shape[1]
        kv_head_num_per_rank = key.shape[1]
        repeat_num = head_num_per_rank // kv_head_num_per_rank
        head_dim = query.shape[2]

        query = query.reshape(bs, q_len, head_num_per_rank, head_dim)
        key = key.reshape(bs, q_len, kv_head_num_per_rank, head_dim)
        value = value.reshape(bs, q_len, kv_head_num_per_rank, head_dim)

        if first_step:
            self.image_kv_cache_map = None
            self._save_image_kv_caches(key, value, seq_len)
        else:
            key, value = self._update_image_kv_caches(key, value, seq_len)

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
        attn_output = attn_output.reshape(bs * q_len, head_num_per_rank, head_dim)
        return attn_output
