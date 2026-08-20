from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _build_hunyuan_image3_param_names_mapping() -> dict:
    """Map HunyuanImage-3 checkpoint names to the sglang model namespace.

    Source keys (official checkpoint) -> target keys (sglang model):
        model.wte.weight                          -> model.embed_tokens.weight
        model.ln_f.weight                         -> model.norm.weight
        model.layers.X.mlp.gate.wg.weight         -> model.layers.X.mlp.gate.weight

    Tensor-level conversions (interleaved qkv_proj, fused gate_and_up_proj,
    per-expert packing) are applied by the model's
    ``preprocess_loaded_state_dict``. Non-AR components (VAE / ViT / diffusion
    head) are loaded by separate pipeline modules and skipped here.
    """
    return {
        # GPT-style backbone names in the checkpoint.
        r"^model\.wte\.(.*)$": r"model.embed_tokens.\1",
        r"^model\.ln_f\.(.*)$": r"model.norm.\1",
        # The MoE router is stored as `mlp.gate.wg` in the checkpoint.
        r"^(.*\.mlp\.gate)\.wg\.(.*)$": r"\1.\2",
        # Non-AR components are loaded separately.
        r"^(vae|vision_model|vision_aligner|final_layer|patch_embed"
        r"|time_embed(?:_\d+)?|timestep_emb)\..*$": "",
    }


@dataclass
class HunyuanImage3ArchConfig(DiTArchConfig):
    """Architecture config for HunyuanImage-3 AR transformer backbone."""

    # AR transformer params (from HF config)
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_head_dim: int = 128
    vocab_size: int = 133120
    intermediate_size: int = 3072

    # MoE params
    num_experts: int = 64
    moe_topk: int = 8
    num_shared_expert: int = 1
    use_mixed_mlp_moe: bool = True
    norm_topk_prob: bool = True

    # CLA (Cross-Layer Attention)
    use_cla: bool = False
    cla_share_factor: int = 2

    # QK Norm
    use_qk_norm: bool = True

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling_type: str = "custom"
    max_position_embeddings: int = 22800

    # Image generation
    vae_downsample_factor: tuple[int, int] = (16, 16)
    latent_channels: int = 32

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(
        default_factory=_build_hunyuan_image3_param_names_mapping
    )

    def __post_init__(self):
        super().__post_init__()
        self.num_layers = self.num_hidden_layers
        self.in_channels = self.latent_channels
        self.out_channels = self.latent_channels
        self.num_channels_latents = self.latent_channels


@dataclass
class HunyuanImage3DitConfig(DiTConfig):
    """DiT config for HunyuanImage-3."""

    arch_config: DiTArchConfig = field(default_factory=HunyuanImage3ArchConfig)

    prefix: str = "hunyuan_image3"
