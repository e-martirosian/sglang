from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class HiDreamPipelineConfig(PipelineConfig):
    """HiDream-O1-Image unified VLM diffusion pipeline configuration.

    HiDream is a unified model: the Qwen3VL backbone handles text encoding,
    vision encoding, and diffusion denoising in a single forward pass.
    No separate text encoder, VAE, or scheduler modules are needed.
    """

    task_type: ModelTaskType = ModelTaskType.T2I
    model_precision: str = "bf16"
    should_use_guidance: bool = True
    supports_cfg_parallel: bool = False

    # HiDream-specific defaults
    num_inference_steps: int = 28
    guidance_scale: float = 5.0
    shift: float = 3.0

    def supports_dynamic_batching(self):
        return True

    def estimate_request_cost(self, batch) -> float:
        patch_size = 32
        image_tokens = (int(batch.height) // patch_size) * (
            int(batch.width) // patch_size
        )
        cfg_branches = 2 if float(batch.guidance_scale) > 1 else 1
        return float(
            image_tokens
            * int(batch.num_inference_steps)
            * cfg_branches
            * int(batch.num_outputs_per_prompt)
        )

    def supports_disaggregation(self) -> bool:
        return False


def register():
    from sglang.multimodal_gen.configs.sample.hidream import (
        HiDreamSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=HiDreamSamplingParams,
        pipeline_config_cls=HiDreamPipelineConfig,
        hf_model_paths=[
            "HiDream-ai/HiDream-O1-Image-Dev-2604",
        ],
        model_detectors=[
            lambda hf_id: "hidream" in hf_id.lower(),
        ],
    )
