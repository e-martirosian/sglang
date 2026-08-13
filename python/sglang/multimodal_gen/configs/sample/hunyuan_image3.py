# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for HunyuanImage-3.0 generation.

HunyuanImage-3.0 generates images with a flow-matching head conditioned on the
autoregressive context: 50 Euler steps (8 for the Distil checkpoint), CFG 2.5,
flow shift 3.0. Defaults mirror the checkpoint's ``generation_config.json``.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# The VAE downsamples spatially by 16; requested resolutions are aligned to a
# multiple of this factor before building the latent grid.
HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT = 32


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    """Sampling parameters for HunyuanImage-3.0 text-to-image generation."""

    negative_prompt: str = ""

    num_frames: int = 1
    # Flow-matching denoising steps (upstream default; 8 for -Distil).
    num_inference_steps: int = 50
    # Classifier-free guidance scale for the diffusion head.
    guidance_scale: float = 2.5
    # Timestep-shift for the flow-matching scheduler.
    flow_shift: float = 3.0

    # Sampling knobs for the AR text stage (think/recaption).
    temperature: float = 0.6
    top_k: int = 1024
    top_p: float = 0.95
    max_new_tokens: int = 2048

    # Task routing, same semantics as upstream's --bot-task: "image" for
    # direct generation, "recaption" / "think" / "think_recaption" for prompt
    # enhancement before generation. None = use the checkpoint's default.
    bot_task: str | None = None
    # System prompt selection: None/"dynamic"/"en_vanilla"/"en_recaption"/
    # "en_think_recaption"/"en_unified"/"custom".
    use_system_prompt: str | None = None
    # Custom system prompt text (only with use_system_prompt="custom").
    system_prompt: str | None = None

    def _adjust(self, server_args):
        requested_width = self.width
        requested_height = self.height
        if self.width is not None and self.height is not None:
            self.width, self.height = align_hunyuan_image3_resolution(
                self.width, self.height
            )
            if (self.width, self.height) != (
                requested_width,
                requested_height,
            ):
                logger.warning(
                    "HunyuanImage-3.0 requires dimensions divisible by %s; adjusted "
                    "requested resolution from %sx%s to %sx%s",
                    HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
                    requested_width,
                    requested_height,
                    self.width,
                    self.height,
                )
        super()._adjust(server_args)


def align_hunyuan_image3_dimension(value: int) -> int:
    """Round a HunyuanImage-3.0 dimension up to a supported multiple."""
    return max(
        HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
        (value + HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT - 1)
        // HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT
        * HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
    )


def align_hunyuan_image3_resolution(width: int, height: int) -> tuple[int, int]:
    return (
        align_hunyuan_image3_dimension(width),
        align_hunyuan_image3_dimension(height),
    )
