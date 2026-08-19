# SPDX-License-Identifier: Apache-2.0
"""Model adapter for HunyuanImage-3.0.

The public checkpoint (``tencent/HunyuanImage-3.0-Instruct``) is a unified
autoregressive text-to-image model: an 80B MoE AR backbone plus a
flow-matching image head, with an ``AutoencoderKLConv3D`` VAE.  All of it is
implemented natively in sglang:

* :mod:`...models.dits.hunyuan_image3_native` – the AR MoE backbone + image
  head (``HunyuanImage3ForCausalMM``)
* :mod:`...models.dits.hunyuan_image3_inputs` – input preparation
  (``prepare_model_inputs`` / ``generate_text`` / attention masks / KV cache)
* :mod:`...models.vaes.autoencoder_kl_conv3d` – the VAE

Only the official ``HunyuanImage3TokenizerFast`` (chat template / special
tokens) and the ``system_prompt`` text module are kept from the checkpoint's
remote code.  Weights stream from the sharded safetensors with an identity
name mapping; ``vision_model.*`` / ``vision_aligner.*`` (SigLIP2, unused for
T2I) are skipped.

The adapter exposes the pieces the pipeline stages need:

* :meth:`prepare_model_inputs` – chat-template + tokenization + KV-cache setup
* :meth:`generate_text` – AR generation for the think/recaption stage
* :meth:`denoise_forward` – one flow-matching denoiser call
* ``vae`` / ``tokenizer`` / ``image_processor`` accessors

Everything the stages call is eager PyTorch with SDPA attention and the eager
MoE implementation (no FlashInfer / FlashAttention / Triton), so it runs on
both CUDA and NPU.
"""

import glob
import importlib.util
import json
import os
from itertools import chain
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_inputs import (
    HunyuanImage3NativeImageProcessor,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_native import (
    HunyuanImage3ForCausalMM,
    HunyuanImage3NativeConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_conv3d import (
    AutoencoderKLConv3D,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_npu = current_platform.is_npu()


def _load_official_tokenizer(model_path: str):
    """Instantiate the official ``HunyuanImage3TokenizerFast``.

    ``AutoTokenizer.from_pretrained`` cannot be used: the checkpoint's
    ``tokenizer_config.json`` declares ``tokenizer_class =
    PreTrainedTokenizerFast`` with no ``auto_map``, so AutoTokenizer would
    return the generic fast tokenizer instead of the custom class (which
    defines ``Resolution`` / ``ImageInfo`` / the gen-image chat template).
    Load ``tokenization_hunyuan_image_3.py`` from the model directory
    directly and register it in ``sys.modules`` so the native input
    preparation can resolve the module's classes.
    """
    import sys

    tk_path = os.path.join(model_path, "tokenization_hunyuan_image_3.py")
    if not os.path.exists(tk_path):
        raise FileNotFoundError(
            f"tokenization_hunyuan_image_3.py not found in {model_path}"
        )
    module_name = "tokenization_hunyuan_image_3"
    tk_module = sys.modules.get(module_name)
    if tk_module is None:
        spec = importlib.util.spec_from_file_location(module_name, tk_path)
        tk_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = tk_module
        spec.loader.exec_module(tk_module)
    tokenizer_cls = tk_module.HunyuanImage3TokenizerFast
    return tokenizer_cls.from_pretrained(model_path)


class _NativeConfigView:
    """Official-config-compatible view over the native config dataclass.

    The pipeline stages were written against the official HF config, which
    exposes ``config.vae`` (a dict) and ``config.vae_downsample_factor``;
    this proxy adds those on top of :class:`HunyuanImage3NativeConfig`.
    """

    def __init__(self, native_config):
        object.__setattr__(self, "_native", native_config)
        vae_ds = native_config.vae_downsample_factor
        if isinstance(vae_ds, (list, tuple)):
            vae_ds = vae_ds[0]
        object.__setattr__(
            self,
            "vae",
            {
                "latent_channels": native_config.vae_latent_channels,
                "scaling_factor": None,
                "shift_factor": None,
            },
        )
        object.__setattr__(self, "vae_downsample_factor", vae_ds)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_native"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_native"), name, value)


class HunyuanImage3ARTransformer(nn.Module):
    """Adapter wrapping the native HunyuanImage-3.0 model."""

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
        self.num_channels_latents = getattr(config, "vae_latent_channels", 32)
        vae_ds = config.vae_downsample_factor
        self.vae_scale_factor = vae_ds if isinstance(vae_ds, int) else vae_ds[0]
        # The denoise stage reads ``config.vae["latent_channels"]`` /
        # ``config.vae_downsample_factor`` with the official config layout;
        # expose that view on top of the native config.
        self._config_view = _NativeConfigView(config)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_native_pretrained(
        cls,
        model_path: str,
        server_args: Any = None,
        torch_dtype: Any = torch.bfloat16,
    ) -> "HunyuanImage3ARTransformer":
        """Load the checkpoint into the native sglang implementation.

        * AR backbone + image head stream from the sharded safetensors with
          an identity parameter-name mapping (``vision_model.*`` /
          ``vision_aligner.*`` are skipped – SigLIP2 is unused for T2I).
        * The VAE (``vae.*`` tensors) loads into the native
          ``AutoencoderKLConv3D`` in fp32, mirroring the official
          ``vae_dtype=float32`` setup.
        * The tokenizer is the official ``HunyuanImage3TokenizerFast``
          (trust_remote_code basic functionality, kept by design).
        """
        device = get_local_torch_device()

        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            hf_config = json.load(f)
        config = HunyuanImage3NativeConfig.from_hf_config(hf_config)

        logger.info(
            "Loading native HunyuanImage-3.0 from %s (dtype=%s)",
            model_path,
            torch_dtype,
        )

        # --- AR backbone + image head ---------------------------------
        with set_default_torch_dtype(torch_dtype), torch.device("meta"):
            inner = HunyuanImage3ForCausalMM(config)

        safetensors_list = sorted(
            glob.glob(os.path.join(model_path, "*.safetensors"))
        )
        if not safetensors_list:
            raise FileNotFoundError(
                f"No safetensors files found in {model_path}"
            )

        def _backbone_key_filter(name: str) -> bool:
            # Skip the SigLIP2 vision tower / aligner (not implemented) and
            # the VAE (loaded separately into the native VAE).
            return not name.startswith(
                ("vision_model.", "vision_aligner.", "vae.")
            )

        weight_iterator = safetensors_weights_iterator(
            safetensors_list, key_filter=_backbone_key_filter
        )
        load_model_from_full_model_state_dict(
            inner,
            weight_iterator,
            device,
            torch_dtype,
            strict=False,
            cpu_offload=bool(getattr(server_args, "dit_cpu_offload", False)),
            param_names_mapping=get_param_names_mapping(inner.param_names_mapping),
        )
        for name, p in chain(inner.named_parameters(), inner.named_buffers()):
            if p.is_meta:
                raise RuntimeError(
                    f"Unexpected param or buffer {name} on meta device."
                )
            if isinstance(p, nn.Parameter):
                p.requires_grad = False
        inner.eval()

        # --- VAE -------------------------------------------------------
        vae_cfg = hf_config.get("vae", {}) or {}
        vae = AutoencoderKLConv3D(
            in_channels=vae_cfg.get("in_channels", 3),
            out_channels=vae_cfg.get("out_channels", 3),
            latent_channels=vae_cfg.get("latent_channels", 32),
            block_out_channels=tuple(vae_cfg.get("block_out_channels", (128, 256, 512, 1024, 1024))),
            layers_per_block=vae_cfg.get("layers_per_block", 2),
            ffactor_spatial=vae_cfg.get("ffactor_spatial", 16),
            ffactor_temporal=vae_cfg.get("ffactor_temporal", 4),
            sample_size=vae_cfg.get("sample_size", 384),
            sample_tsize=vae_cfg.get("sample_tsize", 96),
            scaling_factor=vae_cfg.get("scaling_factor", None),
            shift_factor=vae_cfg.get("shift_factor", None),
            downsample_match_channel=vae_cfg.get("downsample_match_channel", True),
            upsample_match_channel=vae_cfg.get("upsample_match_channel", True),
        ).to(dtype=torch.float32)
        vae_iterator = (
            (name[len("vae."):], tensor)
            for name, tensor in safetensors_weights_iterator(
                safetensors_list,
                key_filter=lambda n: n.startswith("vae."),
            )
        )
        load_model_from_full_model_state_dict(
            vae,
            vae_iterator,
            device,
            torch.float32,
            strict=True,
            param_names_mapping=get_param_names_mapping({}),
        )
        vae.eval()
        inner.vae = vae

        # --- tokenizer / image processor / generation config ----------
        tokenizer = _load_official_tokenizer(model_path)
        inner._tokenizer = tokenizer
        inner.image_processor = HunyuanImage3NativeImageProcessor(config, tokenizer)

        gen_config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            with open(gen_config_path, "r", encoding="utf-8") as f:
                gen_config = SimpleNamespace(**json.load(f))
        else:
            gen_config = SimpleNamespace()
        # Defaults the stages / input preparation rely on.
        for attr, default in (
            ("max_length", config.max_position_embeddings),
            ("max_new_tokens", 2048),
            ("do_sample", True),
            ("top_k", 1024),
            ("top_p", 0.95),
            ("temperature", 0.6),
            ("sequence_template", "instruct"),
            ("diff_infer_steps", 50),
            ("diff_guidance_scale", 2.5),
            ("flow_shift", 3.0),
            ("drop_think", False),
        ):
            if not hasattr(gen_config, attr):
                setattr(gen_config, attr, default)
        inner.generation_config = gen_config
        inner.model_path = model_path

        total_params = sum(p.numel() for p in inner.parameters())
        logger.info(
            "Native HunyuanImage-3.0 loaded: %.2fB backbone params", total_params / 1e9
        )
        return cls(inner)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------
    @property
    def config(self):
        return self._config_view

    @property
    def generation_config(self):
        return self.inner.generation_config

    @property
    def vae(self) -> nn.Module:
        return self.inner.vae

    @property
    def tokenizer(self):
        return self.inner._tokenizer

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
    # Generation plumbing (delegates to the native implementation)
    # ------------------------------------------------------------------
    def resolve_get_system_prompt(self):
        """Load ``get_system_prompt`` from the checkpoint's system_prompt.py.

        The module is plain text templating with no imports, so loading it
        directly from the model directory avoids any remote-code dependency
        beyond the tokenizer.
        """
        model_path = getattr(self.inner, "model_path", None)
        if model_path is None:
            raise RuntimeError(
                "model_path is not set; cannot resolve get_system_prompt."
            )
        sp_path = os.path.join(model_path, "system_prompt.py")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"system_prompt.py not found in {model_path}")
        spec = importlib.util.spec_from_file_location(
            "hunyuan_image3_system_prompt", sp_path
        )
        sp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp_module)
        return sp_module.get_system_prompt

    def prepare_model_inputs(self, **kwargs) -> dict[str, Any]:
        """Build model inputs (chat template, tokens, KV cache) for a mode."""
        return self.inner.prepare_model_inputs(**kwargs)

    @torch.no_grad()
    def generate_text(self, **model_inputs) -> torch.Tensor:
        """Run the AR text stage (think / recaption / img_ratio)."""
        return self.inner.generate_text(**model_inputs)

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

    def compute_post_token_len(self, model_inputs: dict[str, Any]) -> int | None:
        """Compute ``post_token_len`` from the tokenizer output.

        Mirrors the logic in the upstream ``generate()`` method: count the
        number of tokens after the last ``<img>`` token in the sequence.
        """
        tokenizer_output = model_inputs.get("tokenizer_output")
        if tokenizer_output is None:
            return None
        tokens = tokenizer_output.tokens[0]
        img_token_id = self.inner._tokenizer.encode("<img>")[0]
        indices = torch.where(tokens == img_token_id)[0]
        if indices.shape[0] > 0:
            last_idx = indices[-1]
            return int(tokens.shape[0] - 1 - last_idx)
        return None

    def compute_num_special_tokens(self, model_inputs: dict[str, Any]) -> int:
        """Compute ``num_special_tokens`` from batch_gen_image_info.

        Counts the special tokens inserted for the denoiser: timestep,
        guidance (cfg_distilled), and timestep_r (meanflow).
        """
        batch_gen_image_info = model_inputs.get("batch_gen_image_info")
        if batch_gen_image_info and len(batch_gen_image_info) > 0:
            info = batch_gen_image_info[0]
            count = 0
            if getattr(info, "add_timestep_token", False):
                count += 1
            if getattr(info, "add_guidance_token", False):
                count += 1
            if getattr(info, "add_timestep_r_token", False):
                count += 1
            return count
        # Fallback: check model config for what tokens are expected
        config = self.inner.config
        count = 1  # timestep token is always present
        if getattr(config, "cfg_distilled", False):
            count += 1
        if getattr(config, "use_meanflow", False):
            count += 1
        return count

    def forward(self, *args, **kwargs):
        """BaseDiT-compatible passthrough (used only for diagnostics)."""
        return self.inner(*args, **kwargs)


EntryClass = [HunyuanImage3ARTransformer]
