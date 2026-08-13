# SPDX-License-Identifier: Apache-2.0
"""Model adapter for HunyuanImage-3.0.

The public checkpoint (``tencent/HunyuanImage-3.0-Instruct``) ships its model
code alongside the weights (``trust_remote_code``): ``HunyuanImage3ForCausalMM``
in ``modeling_hunyuan_image_3.py`` bundles the MoE AR backbone, the
flow-matching image head (UNet patch embed / final layer + timestep embeds),
the SigLIP2 vision encoder, and the ``AutoencoderKLConv3D`` VAE.

Rather than re-implementing an 80B model inside sglang, this adapter loads the
official implementation exactly the way upstream does (``AutoModelForCausalLM``
+ ``load_tokenizer``) and exposes the pieces the pipeline stages need:

* :meth:`prepare_model_inputs` – chat-template + tokenization + KV-cache setup
* :meth:`generate_text` – AR generation for the think/recaption stage
* :meth:`denoise_forward` – one flow-matching denoiser call
* ``vae`` / ``tokenizer`` / ``image_processor`` accessors

Everything the stages call is eager PyTorch with SDPA attention and the eager
MoE implementation (no FlashInfer / FlashAttention / Triton), so it runs on
both CUDA and NPU.
"""

from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_npu = current_platform.is_npu()


class HunyuanImage3ARTransformer(nn.Module):
    """Adapter wrapping the official HunyuanImage-3.0 remote-code model."""

    _aliases = [
        "HunyuanImage3ForCausalMM",
        "HunyuanImage3Model",
        "HunyuanImage3Transformer",
    ]

    def __init__(self, inner_model: nn.Module) -> None:
        super().__init__()
        self.inner = inner_model
        config = inner_model.config
        # Generic bookkeeping attributes expected by pipeline utilities.
        self.hidden_size = getattr(config, "hidden_size", 4096)
        self.num_attention_heads = getattr(config, "num_attention_heads", 32)
        self.num_channels_latents = config.vae.get("latent_channels", 32)
        self.vae_scale_factor = config.vae_downsample_factor

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_official_pretrained(
        cls,
        model_path: str,
        attn_implementation: str = "sdpa",
        moe_impl: str = "eager",
        torch_dtype: Any = torch.bfloat16,
    ) -> "HunyuanImage3ARTransformer":
        """Load the checkpoint exactly like upstream ``run_image_gen.py``.

        Notes:
        * Upstream requires the local directory name to contain no dots
          (transformers remote-code import limitation).
        * ``device_map="auto"`` shards the 80B weights across the visible
          accelerators; on NPU this relies on torch_npu device registration.
        * ``moe_impl="eager"`` keeps the MoE dispatch in pure PyTorch
          (FlashInfer is CUDA-only), which is what the official eager path and
          our NPU target need.
        """
        from transformers import AutoModelForCausalLM

        if "." in str(model_path).split("/")[-1]:
            logger.warning(
                "HunyuanImage-3.0 model directory %r contains a dot; upstream "
                "notes this can break trust_remote_code loading. Consider "
                "renaming the directory (e.g. HunyuanImage-3-Instruct).",
                model_path,
            )

        kwargs = dict(
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            moe_impl=moe_impl,
            moe_drop_tokens=True,
        )
        logger.info(
            "Loading official HunyuanImage-3.0 model from %s (attn=%s, moe=%s)",
            model_path,
            attn_implementation,
            moe_impl,
        )
        inner = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        # Upstream binds the HunyuanImage3TokenizerFast this way.
        inner.load_tokenizer(model_path)
        inner.eval()
        return cls(inner)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------
    @property
    def config(self):
        return self.inner.config

    @property
    def generation_config(self):
        return self.inner.generation_config

    @property
    def vae(self) -> nn.Module:
        return self.inner.vae

    @property
    def tokenizer(self):
        return self.inner.tokenizer

    @property
    def image_processor(self):
        return self.inner.image_processor

    @property
    def dtype(self):
        return self.inner.dtype

    @property
    def device(self):
        return self.inner.device

    # ------------------------------------------------------------------
    # Generation plumbing (delegates to the official implementation)
    # ------------------------------------------------------------------
    def resolve_get_system_prompt(self):
        """Fetch ``get_system_prompt`` from the loaded remote-code package."""
        import importlib
        import sys

        module = sys.modules[type(self.inner).__module__]
        fn = getattr(module, "get_system_prompt", None)
        if fn is None:
            sp_module = importlib.import_module(module.__package__ + ".system_prompt")
            fn = sp_module.get_system_prompt
        return fn

    def prepare_model_inputs(self, **kwargs) -> dict[str, Any]:
        """Build model inputs (chat template, tokens, KV cache) for a mode."""
        return self.inner.prepare_model_inputs(**kwargs)

    @torch.no_grad()
    def generate_text(self, **model_inputs) -> torch.Tensor:
        """Run the AR text stage (think / recaption / img_ratio)."""
        return self.inner.generate(**model_inputs, decode_text=False)

    def prepare_denoise_inputs(self, input_ids, model_kwargs, latents, timesteps):
        """Build one denoiser call's inputs, mirroring upstream pipeline."""
        return self.inner.prepare_inputs_for_generation(
            input_ids,
            images=latents,
            timesteps=timesteps,
            **model_kwargs,
        )

    @torch.no_grad()
    def denoise_forward(self, model_inputs: dict[str, Any], first_step: bool):
        """One flow-matching denoiser call; returns the raw model output."""
        device_type = current_platform.device_type
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=True
        ):
            return self.inner(**model_inputs, first_step=first_step)

    def update_denoise_kwargs(self, model_output, model_kwargs):
        """Advance KV-cache bookkeeping between denoising steps."""
        return self.inner._update_model_kwargs_for_generation(
            model_output, model_kwargs
        )

    def prepare_attention_mask(self, input_ids, model_kwargs):
        """Build the mixed causal/full attention mask for denoising."""
        return self.inner._prepare_attention_mask_for_generation(
            input_ids, self.inner.generation_config, model_kwargs=model_kwargs
        )

    def forward(self, *args, **kwargs):
        """BaseDiT-compatible passthrough (used only for diagnostics)."""
        return self.inner(*args, **kwargs)


EntryClass = [HunyuanImage3ARTransformer]
