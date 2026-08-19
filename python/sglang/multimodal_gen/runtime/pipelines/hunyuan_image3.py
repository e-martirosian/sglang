# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 unified autoregressive text-to-image pipeline.

Pipeline shape: InputValidation -> AR -> Denoise -> Decode

* AR stage: builds the autoregressive context (chat template, optional
  think/recaption and image-ratio stages, cond/uncond stacking).
* Denoise stage: flow-matching loop where the unified MoE transformer itself
  acts as the denoiser (50 Euler steps, CFG 2.5, flow shift 3.0).
* Decode stage: ``AutoencoderKLConv3D`` VAE decodes latents to pixels.

The checkpoint (``tencent/HunyuanImage-3.0-Instruct``) is a flat HF repo with
``trust_remote_code`` model code and sharded safetensors at the root (no
``model_index.json``), so :meth:`_load_config` synthesizes the module index
and :meth:`load_modules` loads the official implementation directly, the same
way upstream ``run_image_gen.py`` does.

Everything is eager PyTorch (SDPA attention, eager MoE; no Triton /
FlashInfer), so the pipeline runs on both CUDA and NPU.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DitConfig,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HunyuanImage3ARTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3ARStage,
    HunyuanImage3DecodeStage,
    HunyuanImage3DenoiseStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HunyuanImage3Pipeline(ComposedPipelineBase):
    """HunyuanImage-3.0 unified AR T2I pipeline.

    Pipeline shape: InputValidation -> AR -> Denoise -> Decode
    """

    pipeline_name = "HunyuanImage3Pipeline"
    _required_config_modules = ["transformer", "tokenizer", "vae"]

    def _load_config(self) -> dict[str, Any]:
        """Synthesize a model_index for the non-diffusers checkpoint layout."""
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "transformer": ["sglang", "HunyuanImage3ARTransformer"],
            "tokenizer": ["transformers", "HunyuanImage3TokenizerFast"],
            "vae": ["sglang", "AutoencoderKLConv3D"],
        }

    @staticmethod
    def _update_dit_config_from_checkpoint(
        server_args: ServerArgs, model_path: str
    ) -> None:
        """Overwrite the placeholder arch config from the root config.json."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                hf_config = json.load(f)
            dit_config = HunyuanImage3DitConfig()
            dit_config.update_model_arch(hf_config)
            server_args.pipeline_config.dit_config = dit_config
        except Exception as e:
            logger.warning("Failed to refresh HunyuanImage-3 dit config: %s", e)

    # Module loading override
    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load the official HunyuanImage-3.0 model (AR backbone + VAE)."""
        model_path = maybe_download_model(server_args.model_path)

        logger.info("Loading HunyuanImage-3.0 from %s", model_path)

        # Loads the remote-code HunyuanImage3ForCausalMM (transformer + VAE +
        # vision encoder) and binds the tokenizer, exactly like upstream.
        transformer = HunyuanImage3ARTransformer.from_official_pretrained(
            model_path,
        )

        components: dict[str, Any] = {
            "transformer": transformer,
            # Tokenizer and VAE live inside the official model; expose them
            # under the usual module names for the stages.
            "tokenizer": transformer.tokenizer,
            "vae": transformer.vae,
        }

        logger.info("All HunyuanImage-3.0 components loaded successfully")
        return components

    # Pipeline lifecycle
    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            stage_name="hunyuan_image3_ar",
            stage=HunyuanImage3ARStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            ),
        )
        self.add_stage(
            stage_name="hunyuan_image3_denoise",
            stage=HunyuanImage3DenoiseStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            ),
        )
        self.add_stage(
            stage_name="hunyuan_image3_decode",
            stage=HunyuanImage3DecodeStage(
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )


EntryClass = [HunyuanImage3Pipeline]
