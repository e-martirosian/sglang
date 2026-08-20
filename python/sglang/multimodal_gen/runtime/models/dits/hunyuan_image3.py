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
"""HunyuanImage-3 model for sglang multimodal_gen diffusion pipeline.

This is the AR transformer backbone + forward_block denoising interface for
HunyuanImage-3. It lives in multimodal_gen (not srt) because HunyuanImage-3
is a diffusion model, not an LLM serving model.

Ported from the vLLM implementation
(`vllm/model_executor/models/hunyuan_image3.py`).

Uses multimodal_gen layers for TP parallelism, attention, RoPE and
embeddings. The fused-MoE stack (FusedMoE/TopK and its parallel context)
still comes from srt because multimodal_gen does not yet provide a fused
MoE layer of its own.
"""

import re
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

# The fused-MoE stack has no multimodal_gen equivalent yet (mm's own
# MoE blocks also delegate to srt), so these imports stay on srt.
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.runtime_context import get_parallel

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_rank,
    get_tp_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import HunyuanImage3DitConfig

from .hunyuan_image3_utils import (
    HunYuanImageAttentionMeta,
    HunYuanRotary2DEmbedder,
    ImageKVCacheManager,
    create_hunyuan_image_attention_meta,
)

# Weight names belonging to the non-AR parts of the HunyuanImage-3 checkpoint
# (VAE, ViT, diffusion head, ...). These are skipped.
UNEXPECTED_KEYWORDS = [
    "vae",
    "vision_aligner",
    "vision_model",
    "final_layer",
    "patch_embed",
    "timestep_emb",
    "time_embed",
    "time_embed_2",
    "guidance_emb",
    "timestep_r_emb",
]


def _is_moe(config: PretrainedConfig) -> bool:
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        if all(isinstance(e, int) for e in num_experts):
            return max(num_experts) > 1
        return False
    return False


def _get_cla_factor(config: PretrainedConfig) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


def _get_layer_value(config: PretrainedConfig, field: str, layer_id: int, default=None):
    value = getattr(config, field, default)
    if isinstance(value, list):
        assert layer_id >= 0 and len(value) > layer_id, f"{field}[{layer_id}] missing"
        return value[layer_id]
    return value


class HunYuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _get_head_dim(config: PretrainedConfig, hidden_size: int, num_heads: int) -> int:
    if getattr(config, "head_dim", None):
        return config.head_dim
    if hasattr(config, "attention_head_dim"):
        return config.attention_head_dim
    return hidden_size // num_heads


def _make_rope(config: PretrainedConfig, head_dim: int, rope_theta, rope_scaling, max_position):
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "default"
    return get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=True,
    )


class HunYuanAttention(nn.Module):
    """Self-attention of a master layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = _get_head_dim(config, hidden_size, self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = _make_rope(
            config, self.head_dim, rope_theta, rope_scaling, max_position_embeddings
        )
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
        )

        self.image_attn = ImageKVCacheManager(image_token_len=4097)
        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            self.query_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        kv_states=None,
        attn_meta=None,
        attention_mask=None,
        custom_pos_emb=None,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if attn_meta is not None:
            assert positions is None
            q, k = self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb, attn_meta)
        else:
            q, k = self.rotary_emb(positions, q, k)
        ori_k = k

        if self.use_qk_norm:
            q = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

        if attn_meta is not None:
            attn_output = self.image_attn(q, k, v, attn_meta, attention_mask=attention_mask)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            attn_output = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


class HunYuanCrossAttention(nn.Module):
    """CLA follower layer: owns only q_proj, attends to master K/V."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = _get_head_dim(config, hidden_size, self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.q_proj = ColumnParallelLinear(
            hidden_size, hidden_size, bias=bias, quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=bias,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = _make_rope(
            config, self.head_dim, rope_theta, rope_scaling, max_position_embeddings
        )
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
        )

        self.image_attn = ImageKVCacheManager(image_token_len=4097)
        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            self.query_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self, positions, hidden_states, forward_batch,
        kv_states=None, attn_meta=None, attention_mask=None, custom_pos_emb=None,
    ):
        assert kv_states is not None
        ori_k, v = kv_states
        k = ori_k

        q, _ = self.q_proj(hidden_states)
        if attn_meta is not None:
            assert positions is None
            q, _ = self.image_rope2d_emb(
                q, torch.empty_like(k), hidden_states, custom_pos_emb, attn_meta
            )
        else:
            k_tmp = torch.empty_like(k)
            q, _ = self.rotary_emb(positions, q, k_tmp)

        if self.use_qk_norm:
            q = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

        if attn_meta is not None:
            attn_output = self.image_attn(q, k, v, attn_meta, attention_mask=attention_mask)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            attn_output = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


class HunYuanSparseMoeBlock(nn.Module):
    def __init__(
        self, config: PretrainedConfig, layer_id: int,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ):
        super().__init__()
        assert layer_id >= 0
        # Diffusion shards experts over TP only; there is no EP/DP axis.
        self.tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        self.n_routed_experts = config.num_experts

        top_k = _get_layer_value(config, "moe_topk", layer_id)
        intermediate_size = _get_layer_value(config, "intermediate_size", layer_id, 0)
        if getattr(config, "moe_intermediate_size", None) is not None:
            intermediate_size = _get_layer_value(config, "moe_intermediate_size", layer_id)

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False,
            quant_config=None, prefix=f"{prefix}.gate",
        )
        self.topk = TopK(
            top_k=top_k,
            renormalize=getattr(config, "norm_topk_prob", top_k > 1),
            scoring_func="softmax",
        )

        if getattr(config, "use_mixed_mlp_moe", 0) > 0:
            num_shared_expert = _get_layer_value(config, "num_shared_expert", layer_id)
            shared_intermediate_size = _get_layer_value(config, "intermediate_size", layer_id)
            self.shared_mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.shared_mlp",
                reduce_results=False,
            )
        else:
            self.shared_mlp = None

        # srt's FusedMoE/dispatcher read srt's MoE parallel context at
        # construction, which is never initialized for diffusion (no srt
        # scheduler). Publish the diffusion topology via override: experts
        # are TP-sharded (moe_tp == TP) and EP stays degenerate (size 1).
        with get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=self.tp_size,
            moe_tp_rank=tp_rank,
        ):
            self.experts = FusedMoE(
                num_experts=self.n_routed_experts, top_k=top_k,
                hidden_size=config.hidden_size, intermediate_size=intermediate_size,
                reduce_results=False, layer_id=layer_id,
                quant_config=quant_config, prefix=f"{prefix}.experts",
            )

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if self.shared_mlp is not None:
            final_hidden_states = final_hidden_states + self.shared_mlp(hidden_states)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(is_tp_path=True):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, layer_id: int,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ) -> None:
        super().__init__()
        assert layer_id >= 0
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = _get_layer_value(config, "intermediate_size", layer_id, 0)
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling = dict(rope_scaling)
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        cla_factor = _get_cla_factor(config)
        is_cross_attn = layer_id % cla_factor != 0
        attn_kwargs = dict(
            config=config, hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config, bias=attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        if is_cross_attn:
            self.self_attn = HunYuanCrossAttention(**attn_kwargs)
        else:
            self.self_attn = HunYuanAttention(**attn_kwargs)

        if _is_moe(config):
            self.mlp = HunYuanSparseMoeBlock(
                config=config, layer_id=layer_id, quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = HunYuanMLP(
                hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                hidden_act=config.hidden_act, quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False), prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions, hidden_states, forward_batch, residual,
        kv_states=None, attn_meta=None, attention_mask=None, custom_pos_emb=None,
    ):
        if attention_mask is not None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
                attn_meta=attn_meta, attention_mask=attention_mask,
                custom_pos_emb=custom_pos_emb,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
            )
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, ori_kv_states


class HunyuanImage3Model(nn.Module):
    def __init__(
        self, config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            HunyuanImage3DecoderLayer(
                config=config, layer_id=i, quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                positions, hidden_states, forward_batch, residual, prev_kv_states,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_block(
        self, hidden_states, attention_mask, custom_pos_emb,
        attn_meta=None, num_image_tokens=None, first_step=False,
        residual=None,
    ):
        if attn_meta is None:
            assert num_image_tokens is not None
            attn_meta = create_hunyuan_image_attention_meta(
                attention_mask, num_image_tokens, first_step
            )

        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                None, hidden_states, None, residual,
                prev_kv_states, attn_meta, attention_mask, custom_pos_emb,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        return hidden_states.contiguous()

    def _split_qkv_weight(self, qkv):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size
        attention_head_dim = _get_head_dim(self.config, self.config.hidden_size, num_attention_heads)

        qkv = qkv.reshape(num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size)
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))


class HunyuanImage3ForCausalMM(CachableDiT):
    """Top-level HunyuanImage-3 model for diffusion pipeline."""

    def __init__(
        self, config: HunyuanImage3DitConfig, prefix: str = "", **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.config = config
        hf_config = config.arch_config
        self.hf_config = hf_config

        self.model = HunyuanImage3Model(
            hf_config, prefix=f"{prefix}.model",
        )
        self.unpadded_vocab_size = hf_config.vocab_size
        # multimodal_gen has no dedicated LM-head layer; the vocab-parallel
        # embedding shares its layout and only `.weight` is consumed downstream.
        self.lm_head = VocabParallelEmbedding(
            self.unpadded_vocab_size, hf_config.hidden_size,
            org_num_embeddings=self.unpadded_vocab_size,
            prefix=f"{prefix}.lm_head",
        )
        if getattr(hf_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        """DiT-style forward for denoising stage."""
        return hidden_states

    def forward_block(
        self, hidden_states, attention_mask, custom_pos_emb,
        num_image_tokens=None, first_step=False,
    ):
        return self.model.forward_block(
            hidden_states, attention_mask, custom_pos_emb,
            num_image_tokens=num_image_tokens, first_step=first_step,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.hf_config.num_attention_heads
        num_kv_heads = getattr(
            self.hf_config, "num_key_value_heads", self.hf_config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj", ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self.model._split_qkv_weight,
            ),
        ]

        cla_factor = _get_cla_factor(self.hf_config)

        expert_params_mapping = []
        if _is_moe(self.hf_config):
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.hf_config.num_experts,
            )
        expert_weights_remapping = {
            "gate_proj": ("gate_and_up_proj", 1, 2),
            "up_proj": ("gate_and_up_proj", 0, 2),
        }

        params_dict = dict(self.named_parameters())
        loaded_params: set = set()

        for name, loaded_weight in weights:
            if any(keyword in name for keyword in UNEXPECTED_KEYWORDS):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if getattr(self.hf_config, "tie_word_embeddings", False) and "lm_head.weight" in name:
                continue

            if name.endswith("wte.weight"):
                name = name.replace("wte.weight", "embed_tokens.weight")
            if name.endswith("ln_f.weight"):
                name = name.replace("ln_f.weight", "norm.weight")
            if "mlp.gate.wg." in name:
                name = name.replace("wg.", "")

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                if weight_name == ".q_proj" and cla_factor > 1:
                    match = re.search(r"layers\.(\d+)", name)
                    if match and int(match.group(1)) % cla_factor != 0:
                        continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for param_name, weight_name, den, split_param, func in split_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                param = params_dict[name]
                weight_loader = param.weight_loader
                chunk = func(loaded_weight) if func is not None else loaded_weight
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    weight_loader(param, chunk[offset:new_offset], shard_id)
                    offset = new_offset
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            is_expert_weight = False
            found_num = 0
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                offset = 0
                den = 1
                for mapped_weight_substr, origin_weight_info in expert_weights_remapping.items():
                    if mapped_weight_substr in weight_name:
                        origin_weight_name, offset, den = origin_weight_info
                        weight_name = weight_name.replace(mapped_weight_substr, origin_weight_name)
                        break
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                found_num += 1
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                weight_loader(
                    param, loaded_weight[offset * units : offset * units + units],
                    name_mapped, shard_id=shard_id, expert_id=expert_id,
                )
                loaded_params.add(name_mapped)
                is_found = True
                if found_num == den:
                    break
            if is_found:
                continue
            if is_expert_weight:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = [HunyuanImage3ForCausalMM]
