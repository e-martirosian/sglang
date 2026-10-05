"""HiDream-O1-Image unified VLM diffusion pipeline.

HiDream is a unified model: the Qwen3VL backbone handles text encoding,
vision encoding, and diffusion denoising in a single forward pass.
No separate text encoder, VAE, or scheduler modules are needed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.hidream import (
    HiDreamPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hidream import HiDreamSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream import (
    HiDreamGenerationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HiDreamPipeline(ComposedPipelineBase):
    """Pipeline for HiDream-O1-Image unified VLM diffusion model."""

    pipeline_name = "HiDreamPipeline"
    pipeline_config_cls = HiDreamPipelineConfig
    sampling_params_cls = HiDreamSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "HiDreamPipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None and {"model", "processor"} <= set(loaded_modules):
            return loaded_modules

        from transformers import AutoProcessor

        from sglang.multimodal_gen.runtime.models.dits.hidream import (
            HiDreamForCausalLM,
        )

        model_path = self.model_path
        logger.info("Loading HiDream model from %s", model_path)

        # Load processor (tokenizer + image processor)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load model using the DiT config
        from sglang.multimodal_gen.configs.models.dits.hidream import (
            HiDreamDitConfig,
        )
        import json
        import os

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)

        dit_config = HiDreamDitConfig()
        model = HiDreamForCausalLM(dit_config, hf_config)

        # Load weights
        from safetensors.torch import load_file

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for wf in weight_files:
                shard = load_file(os.path.join(model_path, wf))
                state_dict.update(shard)
        else:
            # Single file fallback
            from glob import glob

            safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
            state_dict = {}
            for sf in safetensor_files:
                state_dict.update(load_file(sf))

        # Map checkpoint keys to model keys
        # Checkpoint uses "model." prefix for the inner model
        mapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[6:]  # Remove "model." prefix
            mapped_state_dict[new_key] = value

        missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading HiDream: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys when loading HiDream: %s", unexpected[:10])

        model = model.to(torch.bfloat16).eval()
        logger.info("HiDream model loaded successfully")

        return {"model": model, "processor": processor}

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(
            HiDreamGenerationStage(
                model=self.get_module("model"),
                processor=self.get_module("processor"),
            ),
            "hidream_generation_stage",
        )


EntryClass = HiDreamPipeline
