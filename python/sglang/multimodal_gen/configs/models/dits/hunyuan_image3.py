# SPDX-License-Identifier: Apache-2.0
"""DiT/architecture configuration for HunyuanImage-3.0.

HunyuanImage-3.0 (``tencent/HunyuanImage-3.0-Instruct``) is a unified
autoregressive multimodal model (``HunyuanImage3ForCausalMM``): an 80B-total /
13B-active MoE transformer understands text+image and generates images via a
flow-matching head over continuous VAE latents (50 Euler steps, CFG 2.5,
flow shift 3.0), conditioned on the AR context. The latents are decoded to
pixels by the bundled ``AutoencoderKLConv3D`` VAE.

Values below mirror the checkpoint's ``config.json``
(``model_type=hunyuan_image_3_moe``); at load time ``update_model_arch``
overwrites them from that file.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class HunyuanImage3ArchConfig(DiTArchConfig):
    # Unified vocabulary: text BPE tokens + image/media specials.
    vocab_size: int = 133120

    # Transformer backbone (MoE).
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    # MoE routing: 64 routed experts + 1 shared expert per layer, top-8.
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 3072
    num_shared_experts: int = 1

    # 2D RoPE; the image latents live on a spatial grid.
    rope_type: str = "2d"
    rope_theta: float = 10000.0
    max_position_embeddings: int = 22800
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # Flow-matching generation head.
    vae_latent_channels: int = 32
    vae_downsample_factor: int = 16
    default_diff_infer_steps: int = 50
    default_diff_guidance_scale: float = 2.5
    default_flow_shift: float = 3.0

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class HunyuanImage3DitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HunyuanImage3ArchConfig)

    prefix: str = "hunyuanimage3"
