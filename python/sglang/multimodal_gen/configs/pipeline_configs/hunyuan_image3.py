# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for HunyuanImage-3.0 unified AR T2I."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


@dataclass
class HunyuanImage3PipelineConfig(SpatialImagePipelineConfig):
    """Configuration for the HunyuanImage-3.0 pipeline.

    HunyuanImage-3.0 is a unified autoregressive model: the MoE transformer
    builds the text/image context AR-ly, then generates the image with a
    flow-matching head over continuous VAE latents (the same transformer acts
    as the denoiser). The latents are decoded by the bundled
    ``AutoencoderKLConv3D`` VAE (32 latent channels, 16x spatial, 4x temporal).
    There is no standalone text encoder (text understanding lives inside the
    AR model), so the text encoder config is a placeholder that is never
    loaded. CFG is applied inside the denoising stage by the model's own
    cond/uncond input preparation, not by the framework's guidance machinery.
    """

    # The official VAE runs fp32 weights with fp16 autocast at decode time.
    vae_precision: str = "fp32"

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False
    vae_sp: bool = False

    # Text conditioning happens inside the AR transformer; nothing to load.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )

    dit_config: DiTConfig = field(default_factory=HunyuanImage3DitConfig)
    vae_config: VAEConfig = field(
        default_factory=lambda: VAEConfig(
            arch_config=VAEArchConfig(
                scaling_factor=0.562679178327931,
                temporal_compression_ratio=4,
                spatial_compression_ratio=16,
            )
        )
    )

    enable_autocast: bool = False

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()

    def supports_dynamic_batching(self):
        server_args = get_global_server_args()
        return server_args.srt_encoder_url is not None

    def supports_native_grouped_requests(self):
        return False

    def supports_sequential_dit_inference(self):
        return True

    def supports_sequential_multi_output_inference(self):
        return current_platform.is_npu()
