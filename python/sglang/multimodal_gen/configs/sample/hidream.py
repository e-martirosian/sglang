from dataclasses import dataclass, field
from typing import List, Optional

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HiDreamSamplingParams(SamplingParams):
    """Sampling parameters for HiDream-O1-Image Dev variant."""

    num_inference_steps: int = 28
    num_frames: int = 1
    height: int = 2048
    width: int = 2048
    guidance_scale: float = 5.0
    shift: float = 3.0
    seed: int = 32
    negative_prompt: str = " "

    # Scheduler: "default" (UniPC), "flash", or "flow_match"
    scheduler_name: str = "default"

    # Noise schedule (used by flash scheduler)
    noise_scale_start: float = 7.5
    noise_scale_end: float = 7.5
    noise_clip_std: float = 0.0

    # Reference images for I2I/TI2I
    ref_images: Optional[List[str]] = None
    layout_bboxes: Optional[str] = None
    keep_original_aspect: bool = False

    # Model type: "dev" or "full"
    model_type: str = "dev"
