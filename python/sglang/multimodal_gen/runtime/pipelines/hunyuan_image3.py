from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3AR,
    HunyuanImage3BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    """Pipeline for HunyuanImage-3 text-to-image generation."""

    pipeline_name = "HunyuanImage3Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "vision_language_encoder",
        "processor",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # Stage 1: AR token generation (text + image tokens)
        self.add_stage(
            HunyuanImage3AR(
                processor=self.get_module("processor"),
                vision_language_encoder=self.get_module("vision_language_encoder"),
            ),
            "hunyuan_image3_ar",
        )

        # Stage 2: Prepare latents and conditioning before denoising
        self.add_stage(
            HunyuanImage3BeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "hunyuan_image3_before_denoising_stage",
        )

        # Stage 3: Denoising loop (forward_block)
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Stage 4: VAE decoding
        self.add_standard_decoding_stage()


EntryClass = [HunyuanImage3Pipeline]
