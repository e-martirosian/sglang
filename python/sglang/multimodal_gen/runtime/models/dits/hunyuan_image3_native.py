# SPDX-License-Identifier: Apache-2.0
"""Native implementation of the HunyuanImage-3.0 autoregressive model.

Replaces the official ``HunyuanImage3ForCausalMM`` (``trust_remote_code``)
with an sglang-native implementation:

* MoE AR backbone (32 layers, GQA 32/8, fused interleaved QKV, 2D RoPE +
  qk-norm, eager MoE with a shared expert) using sglang TP-parallel layers
* flow-matching image head (UNet patch embed / final layer + timestep
  embedders)
* ``CachedRoPE`` multimodal 2D rotary embeddings

Module/parameter names are kept identical to the official safetensors
checkpoint (``model.layers.N.self_attn.qkv_proj``, ``mlp.gate.wg``,
``mlp.experts.E.*``, ``patch_embed.model.*``, ``final_layer.*``,
``{time_embed,time_embed_2,timestep_emb}.mlp.{0,2}``, ``lm_head``,
``model.wte``/``model.ln_f``) so weights load with an identity name mapping.

Excluded on purpose (not needed for T2I): SigLIP2 vision encoder,
vision_aligner, guidance/timestep-r embeddings (cfg_distilled / meanflow
off in the released checkpoint), Taylor cache, flashinfer fused MoE.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_inputs import (
    ConditionalSliceVocabLogitsProcessor,
    HunyuanImage3InputPreparationMixin,
    StageTransitionLogitsProcessor,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_utils import (
    CachedRoPE,
    apply_rotary_pos_emb,
    get_device,
    repeat_kv,
    timestep_embedding,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]


def _per_layer(value, layer_idx):
    """Config values may be scalars or per-layer lists."""
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


# =======================================================
#     Native config
# =======================================================


@dataclass
class HunyuanImage3NativeConfig:
    """Architecture hyper-parameters of HunyuanImage-3.0-Instruct."""

    vocab_size: int = 133120
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_head_dim: int = 128
    intermediate_size: int = 3072
    moe_intermediate_size: Any = 3072
    num_experts: Any = 64
    moe_topk: Any = 8
    num_shared_expert: Any = 1
    moe_layer_num_skipped: int = 0
    use_mixed_mlp_moe: bool = True
    hidden_act: str = "silu"
    norm_type: str = "rms"
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    use_qk_norm: bool = True
    use_rotary_pos_emb: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "2d"
    max_position_embeddings: int = 22800
    pad_token_id: int = 128009
    cfg_distilled: bool = False
    use_meanflow: bool = False
    patch_size: int = 1
    patch_embed_hidden_dim: int = 1024
    img_proj_type: str = "unet"
    vae_latent_channels: int = 32
    image_base_size: int = 1024
    vae_downsample_factor: Any = (16, 16)
    cond_image_type: str = "vae_vit"
    cond_token_attn_type: str = "joint_full"

    @classmethod
    def from_hf_config(cls, cfg) -> "HunyuanImage3NativeConfig":
        """Build from the checkpoint's ``config.json`` (dict or HF config)."""
        get = cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)
        vae_cfg = get("vae", {}) or {}
        latent = (
            vae_cfg.get("latent_channels", 32)
            if isinstance(vae_cfg, dict)
            else getattr(vae_cfg, "latent_channels", 32)
        )
        kwargs = {}
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "attention_head_dim",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts",
            "moe_topk",
            "num_shared_expert",
            "moe_layer_num_skipped",
            "use_mixed_mlp_moe",
            "hidden_act",
            "norm_type",
            "rms_norm_eps",
            "attention_bias",
            "attention_dropout",
            "mlp_bias",
            "use_qk_norm",
            "use_rotary_pos_emb",
            "rope_theta",
            "rope_type",
            "max_position_embeddings",
            "pad_token_id",
            "cfg_distilled",
            "use_meanflow",
        ):
            value = get(name, None)
            if value is not None:
                kwargs[name] = value
        for name in ("patch_size", "patch_embed_hidden_dim", "img_proj_type"):
            value = get(name, None)
            if value is not None:
                kwargs[name] = value
        for name in (
            "image_base_size",
            "vae_downsample_factor",
            "cond_image_type",
            "cond_token_attn_type",
        ):
            value = get(name, None)
            if value is not None:
                kwargs[name] = value
        # config.json stores the pad token id under `pad_id`
        pad_id = get("pad_id", None)
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        kwargs["vae_latent_channels"] = latent
        return cls(**kwargs)


# =======================================================
#     Modules for Image Generation (image head)
# =======================================================


def _normalization(channels, **kwargs):
    return nn.GroupNorm(32, channels, **kwargs)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """A residual block with adaptive group norm (timestep-conditioned)."""

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        assert dims == 2, "only 2D convolutions are supported in the native port"
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            _normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, **factory_kwargs),
        )

        self.out_layers = nn.Sequential(
            _normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            )
        else:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, 1, **factory_kwargs
            )

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """patch_embed: latent patches -> transformer tokens."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            ]
        )

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=(
                            hidden_channels if (i + 1) * 2 != self.patch_size else out_channels
                        ),
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        # 'b c h w -> b (h w) c'
        x = x.flatten(2).transpose(1, 2)
        return x, token_h, token_w


class UNetUp(nn.Module):
    """final_layer: transformer tokens -> latent patches."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
        out_norm=False,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    _normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),
                )
            )
        else:
            self.model.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            )

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        # 'b (h w) c -> b c h w'
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], token_h, token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# =======================================================
#     Modules for Transformer Backbone
# =======================================================


def _rms_norm_fp32(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with fp32 accumulate and input-dtype output.

    Mirrors the official HF ``RMSNorm`` (fp32 variance computation, result
    cast back to the input dtype).  A pure elementwise implementation is
    used because the NPU ``aclnnRmsNorm`` kernel rejects mixed x/weight
    dtypes (e.g. fp32 activations with bf16 weights, which happen after the
    fp32 RoPE promotion).
    """
    orig_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + norm.variance_epsilon)
    return (x32 * norm.weight.float()).to(orig_dtype)


class HunyuanMLP(nn.Module):
    """SwiGLU MLP; also used for MoE experts and the shared expert."""

    def __init__(
        self,
        config: HunyuanImage3NativeConfig,
        layer_idx=None,
        is_shared_mlp=False,
        is_moe=False,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_act = config.hidden_act
        assert self.hidden_act == "silu", "only silu (SwiGLU) is supported"

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            self.intermediate_size = _per_layer(config.moe_intermediate_size, layer_idx)
            if is_shared_mlp:
                self.intermediate_size *= _per_layer(config.num_shared_expert, layer_idx)

        # SwiGLU: gate and up projections packed into one linear
        self.gate_and_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.intermediate_size] * 2,
            bias=config.mlp_bias,
            gather_output=False,
            prefix=f"{prefix}.gate_and_up_proj" if prefix else "gate_and_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            input_is_parallel=True,
            reduce_results=True,
            prefix=f"{prefix}.down_proj" if prefix else "down_proj",
        )

    def forward(self, x):
        gate_and_up, _ = self.gate_and_up_proj(x)
        x1, x2 = gate_and_up.chunk(2, dim=-1)
        out, _ = self.down_proj(x1 * F.silu(x2))
        return out


class HunyuanTopKGate(nn.Module):
    """fp32 router with renormalized top-k softmax weights."""

    def __init__(self, config: HunyuanImage3NativeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.moe_topk = _per_layer(config.moe_topk, layer_idx)
        num_experts = _per_layer(config.num_experts, layer_idx)
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        # easy_topk
        gates = F.softmax(logits, dim=1)
        topk_weight_1, expert_index = torch.topk(gates, self.moe_topk)
        weight_sums = topk_weight_1.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight_1 / weight_sums
        return topk_weight, expert_index


class HunyuanMoE(nn.Module):
    """Eager MoE: fp32 gating + per-expert loop, plus the shared expert."""

    def __init__(self, config: HunyuanImage3NativeConfig, layer_idx: Optional[int] = None, prefix: str = ""):
        super().__init__()
        self.moe_topk = _per_layer(config.moe_topk, layer_idx)
        self.num_experts = _per_layer(config.num_experts, layer_idx)
        self.use_mixed_mlp_moe = config.use_mixed_mlp_moe
        if self.use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(
                config, layer_idx=layer_idx, is_shared_mlp=True, prefix=f"{prefix}.shared_mlp"
            )
        self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
        self.experts = nn.ModuleList(
            [
                HunyuanMLP(
                    config, layer_idx=layer_idx, is_moe=True, prefix=f"{prefix}.experts.{i}"
                )
                for i in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        input_hidden_states = hidden_states

        if self.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        # Gate in fp32, immune to any surrounding autocast.
        with torch.autocast(current_platform.device_type, enabled=False):
            topk_weights, topk_idx = self.gate(hidden_states)
        # Cast back to the input dtype
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Flatten for easier indexing
        flat_topk_idx = topk_idx.view(-1)
        hidden_states_flat = input_hidden_states.view(-1, hidden_size)
        hidden_states_repeated = hidden_states_flat.repeat_interleave(self.moe_topk, dim=0)

        # Forward through experts
        expert_outputs = torch.zeros_like(hidden_states_repeated)
        for i in range(self.num_experts):
            expert_mask = flat_topk_idx == i
            selected_inputs = hidden_states_repeated[expert_mask]
            expert_output = self.experts[i](selected_inputs)
            expert_outputs[expert_mask] = expert_output

        # Weighted sum of expert outputs
        combined_output = (
            expert_outputs.view(bsz * seq_len, self.moe_topk, hidden_size)
            * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
        combined_output = combined_output.to(hidden_states.dtype).view(bsz, seq_len, hidden_size)

        if self.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output


class HunyuanImage3SDPAAttention(nn.Module):
    """SDPA attention with fused interleaved QKV, 2D RoPE and qk-norm.

    The fused ``qkv_proj`` output follows the official interleaved layout:
    per kv-head, ``[num_kv_groups q-heads, 1 k-head, 1 v-head]``; the column
    parallel sharding therefore partitions per kv-head block so that every
    rank keeps complete (q, k, v) groups.
    """

    def __init__(self, config: HunyuanImage3NativeConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim: int = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # One partition per kv-head block: [groups q, 1 k, 1 v] * head_dim
        block_size = self.head_dim * (self.num_key_value_groups + 2)
        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=config.attention_bias,
            gather_output=False,
            output_sizes=[block_size] * self.num_key_value_heads,
            prefix=f"{prefix}.qkv_proj" if prefix else "qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.hidden_size_q,
            self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            reduce_results=True,
            prefix=f"{prefix}.o_proj" if prefix else "o_proj",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        tp_size = self.qkv_proj.tp_size
        local_kv_heads = self.num_key_value_heads // tp_size
        local_q_heads = self.num_heads // tp_size

        qkv_states, _ = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(
            bsz, q_len, local_kv_heads, self.num_key_value_groups + 2, self.head_dim
        )
        query_states, key_states, value_states = torch.split(
            qkv_states, [self.num_key_value_groups, 1, 1], dim=3
        )

        query_states = query_states.reshape(bsz, q_len, local_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, local_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, local_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            # RoPE is applied before qk_norm, matching the official model
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # The fp32 RoPE cos/sin above promotes q/k to fp32; bring them back
        # to the activation dtype first: the NPU rms_norm kernel requires
        # matching x/weight dtypes, and the official qk-norm semantics are
        # "compute in fp32, return the input dtype" (see _rms_norm_fp32).
        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if self.use_qk_norm:
            query_states = _rms_norm_fp32(self.query_layernorm, query_states)
            key_states = _rms_norm_fp32(self.key_layernorm, key_states)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA memory-efficient backend needs contiguous inputs with masks
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output, _ = self.o_proj(attn_output)

        return attn_output


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config: HunyuanImage3NativeConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HunyuanImage3SDPAAttention(
            config, layer_idx=layer_idx, prefix=f"{prefix}.self_attn"
        )

        num_experts = _per_layer(config.num_experts, layer_idx)
        if num_experts > 1 and layer_idx >= config.moe_layer_num_skipped:
            self.mlp = HunyuanMoE(config, layer_idx=layer_idx, prefix=f"{prefix}.mlp")
        else:
            self.mlp = HunyuanMLP(config, layer_idx=layer_idx, prefix=f"{prefix}.mlp")

        assert config.norm_type in ("rms", "hf_rms"), (
            f"norm_type {config.norm_type} not supported in the native port"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            custom_pos_emb=custom_pos_emb,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = _rms_norm_fp32(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunyuanImage3Model(nn.Module):
    """Inner AR backbone: wte + decoder layers + ln_f.

    ``ln_f`` is intentionally NOT applied here; the outer model applies it
    only on the text (logits) path, matching the official implementation.
    """

    def __init__(self, config: HunyuanImage3NativeConfig, prefix: str = "model"):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.wte = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.wte"
        )
        self.layers = nn.ModuleList(
            [
                HunyuanImage3DecoderLayer(config, layer_idx, prefix=f"{prefix}.layers.{layer_idx}")
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                custom_pos_emb=custom_pos_emb,
            )
        # Do ln_f outside of the model for compatibility with image generation.
        return hidden_states


# =======================================================
#     Outer multimodal model
# =======================================================


@dataclass
class CausalMMOutputWithPast:
    """Output mirroring the official ``CausalMMOutputWithPast`` fields.

    The official one subclasses transformers ``ModelOutput``, which supports
    dict-like access; the pipeline stages rely on
    ``model_output["diffusion_prediction"]``, so reproduce that here.
    """

    logits: Optional[torch.Tensor] = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None
    diffusion_prediction: Optional[torch.Tensor] = None

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()


class HunyuanImage3ForCausalMM(HunyuanImage3InputPreparationMixin, nn.Module):
    """Native HunyuanImage-3.0 AR + image-head model (T2I path).

    Parameter names match the official safetensors checkpoint exactly, so
    weights load with an identity name mapping; see
    :meth:`map_param_name`.

    Input preparation (``prepare_model_inputs`` / ``generate_text`` / ...)
    comes from :class:`HunyuanImage3InputPreparationMixin` and needs
    ``self._tokenizer`` / ``self.image_processor`` / ``self.generation_config``
    wired up at load time.
    """

    # Identity mapping: checkpoint names == module names.
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    # Official nested-class names used by the pipeline stages.
    _ConditionalSliceVocabLogitsProcessor = ConditionalSliceVocabLogitsProcessor
    _StageTransitionLogitsProcessor = StageTransitionLogitsProcessor

    def __init__(self, config: HunyuanImage3NativeConfig):
        super().__init__()
        self.config = config
        assert not config.cfg_distilled, "cfg_distilled checkpoints are not supported"
        assert not config.use_meanflow, "meanflow checkpoints are not supported"
        assert config.img_proj_type == "unet", (
            f"img_proj_type {config.img_proj_type} not supported"
        )

        # image generation related
        self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)
        self.patch_embed = UNetDown(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.vae_latent_channels,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.hidden_size,
        )
        self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)
        self.final_layer = UNetUp(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.vae_latent_channels,
            out_norm=True,
        )
        self.time_embed_2 = TimestepEmbedder(hidden_size=config.hidden_size)

        # transformer backbone + linear head
        self.model = HunyuanImage3Model(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,
            prefix="lm_head",
        )

        self.pad_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Bookkeeping set by the pipeline stages
        self.num_image_tokens = None
        self.num_special_tokens = None
        self.post_token_len = None
        # Initialize cached rope, supporting automatic cache update
        self.cached_rope = CachedRoPE(config)

        # Generic bookkeeping attributes expected by pipeline utilities.
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.vae_latent_channels

    @staticmethod
    def map_param_name(name: str) -> Tuple[str, Any, Any]:
        """Identity checkpoint-name mapping for the state-dict loader."""
        return name, None, None

    # ------------------------------------------------------------------
    # Generic properties
    # ------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Input-embedding instantiation
    # ------------------------------------------------------------------
    def instantiate_vae_image_tokens(
        self,
        hidden_states: torch.Tensor,
        timesteps: BatchRaggedTensor,
        images: BatchRaggedImages,
        image_mask: torch.Tensor,
    ):
        """Instantiate the VAE image embeddings into the input embedding
        sequence (scatter on the first step, build from scratch afterwards).

        Args:
            hidden_states: input sequence, (batch_size, seq_len, n_embd)
            images: a 4-D tensor, or a list of 4-D tensors / 3-D tensors
            timesteps: a 1-D tensor, or a list of 1-D tensors
            image_mask: (batch_size, seq_len)
        """
        if hidden_states is None:
            # Only for inference in non-first step image generation
            t_emb = self.time_embed(timesteps)
            image_emb = self.patch_embed(images, t_emb)[0]
            timestep_emb = self.timestep_emb(timesteps).reshape(
                images.size(0), -1, self.config.hidden_size
            )
            hidden_states = torch.cat([timestep_emb, image_emb], dim=1)
            return hidden_states

        bsz, seqlen, n_embd = hidden_states.shape
        assert isinstance(images, (torch.Tensor, list)), (
            f"images should be BatchRaggedImages, got {type(images)}"
        )

        if isinstance(images, torch.Tensor):
            assert images.ndim == 4, f"images should be a 4-D tensor, got {images.ndim}-D tensor"
            assert isinstance(timesteps, torch.Tensor), (
                f"timesteps should be 1-D tensor, got {type(timesteps)}"
            )

            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
            t_emb = self.time_embed(timesteps)  # (bsz, n_embd)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )

        else:  # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
            for i, (image_i, t_i) in enumerate(zip(images, timesteps)):
                t_i_emb = self.time_embed(t_i)  # (n_i, n_embd)

                if isinstance(image_i, torch.Tensor):
                    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)

                elif isinstance(image_i, list):
                    image_i_seq_list = []
                    for j in range(len(image_i)):
                        image_ij = image_i[j].unsqueeze(0)
                        assert image_ij.ndim == 4, (
                            f"image_ij should have size of (1, C, H, W), got {list(image_ij.size())}"
                        )
                        image_i_seq_j = self.patch_embed(image_ij, t_i_emb[j : j + 1])[0]
                        image_i_seq_list.append(image_i_seq_j)
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)

                else:
                    raise TypeError(
                        f"image_i should be a torch.Tensor or a list, got {type(image_i)}"
                    )

                image_i_index = index[i : i + 1].masked_select(image_mask[i : i + 1].bool()).reshape(1, -1)
                hidden_states[i : i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd),
                )

        return hidden_states

    def instantiate_continuous_tokens(
        self,
        hidden_states: torch.Tensor,
        timesteps: Optional[BatchRaggedTensor] = None,
        timesteps_index: Optional[BatchRaggedTensor] = None,
    ):
        bsz, seqlen, n_embd = hidden_states.shape

        if isinstance(timesteps, list):
            for i, timestep in enumerate(timesteps):
                timestep_src = self.timestep_emb(timestep)  # (n, n_embd)
                hidden_states[i : i + 1].scatter_(
                    dim=1,
                    index=timesteps_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_src.reshape(1, -1, n_embd),
                )
        else:
            timesteps_src = self.timestep_emb(timesteps.reshape(-1))  # (bsz * n, n_embd)
            hidden_states.scatter_(
                dim=1,
                index=timesteps_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_src.reshape(bsz, -1, n_embd),
            )

        return hidden_states

    def get_image_tokens_hw(self, images: BatchRaggedImages):
        assert isinstance(images, (torch.Tensor, list)), (
            f"images should be BatchRaggedImages, got {type(images)}"
        )
        if isinstance(images, torch.Tensor):
            token_h = images.shape[-2] // self.config.patch_size
            token_w = images.shape[-1] // self.config.patch_size
        else:
            token_h, token_w = [], []
            for image_i in images:
                assert isinstance(image_i, (torch.Tensor, list)), (
                    f"image_i should be a tensor or a list of tensors, got {type(image_i)}"
                )
                if isinstance(image_i, torch.Tensor):
                    token_h.append(image_i.shape[-2] // self.config.patch_size)
                    token_w.append(image_i.shape[-1] // self.config.patch_size)
                else:
                    token_h.append([])
                    token_w.append([])
                    for j in range(len(image_i)):
                        token_h[-1].append(image_i[j].shape[-2] // self.config.patch_size)
                        token_w[-1].append(image_i[j].shape[-1] // self.config.patch_size)
        return token_h, token_w

    def ragged_final_layer(self, hidden_states, image_mask, timesteps, token_h, token_w, first_step=None):
        n_embd = hidden_states.size(-1)
        if isinstance(timesteps, torch.Tensor):
            # Only one target image.
            t_emb = self.time_embed_2(timesteps)
            if first_step is False:
                # only for gen_image non-first-step inference
                image_output = hidden_states[:, self.num_special_tokens :, :]
            else:  # first_step is True or None
                image_output = hidden_states.masked_select(
                    image_mask.unsqueeze(-1).bool()
                ).reshape(-1, token_h * token_w, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            # Multiple target images (interleave data).
            sections = image_mask.sum(1).tolist()
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()
            ).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_i, token_h_i, token_w_i in zip(
                image_output, timesteps, token_h, token_w
            ):
                t_emb_i = self.time_embed_2(t_i)
                if isinstance(token_h_i, int):
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    subsections = [
                        token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)
                    ]
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(
                            image_output_ij[None], t_emb_i[j : j + 1], token_h_i[j], token_w_i[j]
                        )
                        pred_i.append(pred_ij)
                    pred.append(pred_i)
        return pred

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # bsz x seqlen
        attention_mask: Optional[torch.Tensor] = None,  # bsz x 1 x S x S
        rope_image_info: Optional[list] = None,
        return_dict: bool = True,
        # for gen images
        images: Optional[BatchRaggedImages] = None,
        image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
        timesteps: Optional[BatchRaggedTensor] = None,
        timesteps_index: Optional[BatchRaggedTensor] = None,
        # only for inference
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        mode: Optional[str] = None,
        first_step: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        gen_timestep_scatter_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalMMOutputWithPast:
        # Sanity Check of Inputs
        if mode == "gen_image":
            assert images is not None, "`images` should be provided in `gen_image` mode."
            assert timesteps is not None, "`timesteps` should be provided in `gen_image` mode."
            if first_step:
                assert image_mask is not None, (
                    "`image_mask` should be provided in `gen_image` mode at the first step."
                )
                assert timesteps_index is not None, (
                    "`timesteps_index` should be provided in `gen_image` mode at the first step."
                )
        if input_ids is None and images is None:
            raise ValueError("Either input_ids or images should be provided.")
        if input_ids is not None:
            device = input_ids.device
        else:
            device = get_device(images)
        if self.training:
            seqlen = input_ids.size(1)
        else:
            # For inference, always set seqlen to maximum length to simplify
            # the rope cache handling
            seqlen = self.config.max_position_embeddings
        assert self.config.max_position_embeddings >= seqlen, (
            f"Cannot forward sequence of length {seqlen}, "
            f"max position embeddings is only {self.config.max_position_embeddings}."
        )

        # Calculate multimodal 2d rope
        cos, sin = self.cached_rope(
            seqlen, device, rope_image_info=rope_image_info, position_ids=position_ids
        )
        # === Map token ids to embeddings ===
        if input_ids is not None:
            hidden_states = self.model.wte(input_ids)  # (bsz, seqlen, n_embd)
        else:
            hidden_states = None  # only for non-first step inference

        # === Input layers ===
        if images is not None:
            hidden_states = self.instantiate_vae_image_tokens(
                hidden_states, timesteps, images, image_mask
            )

        if timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(
                hidden_states, timesteps, timesteps_index
            )

        if mode == "gen_text":
            first_step = True

        hidden_states = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            custom_pos_emb=(cos, sin),
        )

        # === Output layers ===
        # -- image tokens
        if images is not None:
            token_h, token_w = self.get_image_tokens_hw(images)
            hidden_states = hidden_states.to(device=get_device(images))
            diff_pred = self.ragged_final_layer(
                hidden_states, image_mask, timesteps, token_h, token_w, first_step
            )
        else:
            diff_pred = None
        # -- text tokens
        if input_ids is None or mode == "gen_image":
            logits = None
        else:
            hidden_states = self.model.ln_f(hidden_states)
            logits, _ = self.lm_head(hidden_states)  # (bsz, seqlen, vocab_size)

        return CausalMMOutputWithPast(
            logits=logits.float() if logits is not None else None,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            diffusion_prediction=diff_pred,
        )
