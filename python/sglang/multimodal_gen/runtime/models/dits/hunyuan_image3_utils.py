# SPDX-License-Identifier: Apache-2.0
"""Native input-preparation utilities for HunyuanImage-3.0.

Natively ported from the official ``modeling_hunyuan_image_3.py`` (shipped
with the ``tencent/HunyuanImage-3.0-Instruct`` checkpoint):

* 2D multimodal RoPE construction and caching
  (``build_2d_rope`` / ``build_batch_2d_rope`` / ``CachedRoPE``)
* ``HunyuanStaticCache`` - preallocated KV cache with inplace
  ``index_copy_`` updates and per-batch cache positions
* small helpers (``repeat_kv``, ``timestep_embedding``, rotary apply)

Numerics are kept identical to the official implementation on purpose.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


# =======================================================
#     Helper Functions
# =======================================================


def get_device(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    elif isinstance(tensor, list):
        return get_device(tensor[0])
    else:
        raise ValueError(f"Unsupported type for get_device: {type(tensor)}")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings (cos then sin concatenated)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# =======================================================
#     Multi-Dimensional RoPE
# =======================================================


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device="cpu"):
    """Get n-D meshgrid with start, stop and num.

    Args:
        start: If len(args) == 0, start is num; if len(args) == 1, start is
            start and args[0] is stop (step 1); if len(args) == 2, start is
            start, args[0] is stop, args[1] is num.
        dim: Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid: [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), (
            f"num should be int, but got {num}"
        )
        num = num_int
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)  # Left-Top
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom
        num = _to_tuple(args[1], dim=dim)  # Target Size
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch impl of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [H, W]
    grid = torch.stack(grid, dim=0)  # [dim, H, W]

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
    """Build the 2D multimodal RoPE for one sample.

    Reference: https://kexue.fm/archives/10352

    Start from 1, we have
        beta_y = L + (wh - h)/2
        beta_x = L + (wh - w)/2

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
    )
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start  # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L, device=device))
                x_sections.append(torch.arange(last_pos, L, device=device))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size>
                # <ratio> <timestep> <eoi> tokens.
                y_sections.append(
                    torch.arange(sec_slice.start, sec_slice.stop, device=device)
                )
                x_sections.append(
                    torch.arange(sec_slice.start, sec_slice.stop, device=device)
                )
                continue
            else:
                # Interleave data has overlapped positions for noised image
                # and the successive clean image.
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd(
                (beta_y, beta_x), (beta_y + h, beta_x + w), device=device
            )  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(
            torch.arange(last_pos, sample_seq_lens[sample_id], device=device)
        )
        x_sections.append(
            torch.arange(last_pos, sample_seq_lens[sample_id], device=device)
        )

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)  # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

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
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len,
            n_elem,
            image_infos=image_info,
            device=device,
            base=base,
            base_rescale_factor=base_rescale_factor,
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    ``cos``/``sin`` have shape [batch_size, seq_len, head_dim]; with
    ``unsqueeze_dim=1`` they broadcast onto
    [batch_size, heads, seq_len, head_dim] q/k.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CachedRoPE(object):
    """A 2D RoPE is determined by rope_image_info and seq_len."""

    def __init__(self, config):
        self.config = config
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_image_info = None

    def __call__(self, seq_len, device, rope_image_info=None, position_ids=None):
        """Get cached RoPE for given seq_len and rope_image_info.

        If cache miss, compute and cache it.

        Args:
            seq_len (int): The sequence length.
            device (torch.device): The device to store the RoPE.
            rope_image_info (list): list of lists of (slice, (height, width)).
            position_ids (torch.Tensor): The input positions.

        Returns:
            The RoPE cos and sin tensors.
        """
        if (self.seq_len != seq_len) or (
            rope_image_info is not None
            and self.rope_image_info != rope_image_info
        ):
            # Cache miss, compute RoPE
            if self.config.rope_type in ["2d", "default"]:
                self.cos_cache, self.sin_cache = build_batch_2d_rope(
                    image_infos=rope_image_info,
                    seq_len=seq_len,
                    n_elem=self.config.attention_head_dim,
                    device=device,
                    base=self.config.rope_theta,
                )
                self.seq_len = seq_len
                self.rope_image_info = rope_image_info
            else:
                raise NotImplementedError(
                    f"rope_type `{self.config.rope_type}` not supported"
                )

        if position_ids is None:
            # Typically for training
            cos, sin = self.cos_cache, self.sin_cache
        else:
            # Typically for inference
            assert position_ids.dim() == 2, f"{position_ids.shape=}"
            head_size = self.cos_cache.size(-1)
            cos = torch.gather(
                self.cos_cache,
                dim=1,
                index=position_ids.unsqueeze(-1).expand(-1, -1, head_size),
            )
            sin = torch.gather(
                self.sin_cache,
                dim=1,
                index=position_ids.unsqueeze(-1).expand(-1, -1, head_size),
            )

        return cos, sin


# =======================================================
#     Static KV Cache
# =======================================================


class _StaticCacheLayer:
    """Holds the preallocated key/value buffers of one layer."""

    def __init__(self):
        self.keys = None
        self.values = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        max_batch_size: int,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        out_shape = (
            max_batch_size,
            key_states.shape[1],
            max_cache_len,
            key_states.shape[-1],
        )
        self.keys = torch.zeros(
            out_shape, dtype=dtype, device=key_states.device
        )
        self.values = torch.zeros(
            out_shape, dtype=dtype, device=key_states.device
        )


class HunyuanStaticCache:
    """A static cache supporting dynamic extension of the cache and inplace
    updates, ported from the official ``HunyuanStaticCache`` (which extends
    ``transformers.StaticCache``).

    Supports batched (2-D) ``cache_position`` updates via per-batch
    ``index_copy_``.
    """

    def __init__(
        self,
        config=None,
        max_batch_size: int = 1,
        max_cache_len: int = 1,
        dtype: torch.dtype = None,
        dynamic: bool = False,
        device=None,
        **kwargs,
    ):
        self.dynamic = dynamic
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(
            config, "text_config", config
        ).num_hidden_layers
        self.layers = [_StaticCacheLayer() for _ in range(num_layers)]
        self.seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with the new key/value states for ``layer_idx``.

        ``cache_kwargs["cache_position"]`` tells where to write. A 1-D
        position tensor is shared across the batch; a 2-D tensor holds
        per-batch positions.
        """
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        layer = self.layers[layer_idx]
        if layer.keys is None:
            layer.lazy_initialization(
                key_states, self.max_batch_size, self.max_cache_len, self.dtype
            )
        k_out = layer.keys
        v_out = layer.values

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # `tensor.index_copy_(dim, index, tensor)` is equivalent to
            # `tensor[:, :, index] = tensor` but avoids copies.
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                assert cache_position.dim() == 2, (
                    f"multiple batch dims not yet {cache_position.shape=}"
                )
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(
                        unbatched_dim, cache_position[i], key_states[i]
                    )
                    v_out[i].index_copy_(
                        unbatched_dim, cache_position[i], value_states[i]
                    )

                if self.dynamic:
                    assert len(cache_position) == 1
                    end = cache_position[0, -1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

            self.seen_tokens = max(
                self.seen_tokens, int(cache_position.max().item()) + 1
            )

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seen_tokens

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self):
        for layer in self.layers:
            layer.keys = None
            layer.values = None
        self.seen_tokens = 0
