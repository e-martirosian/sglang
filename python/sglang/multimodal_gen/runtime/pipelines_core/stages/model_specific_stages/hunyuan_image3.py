"""Native AR stage for HunyuanImage-3 text-to-image generation.

Implements the diffusion sampling loop directly using the sglang backbone's
``forward_block`` interface, without relying on the official HF shell model.
"""

import os
import sys
from functools import partial
from typing import Any, Optional

import torch
from einops import rearrange

from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    align_hunyuan_image3_resolution,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Default sampling parameters (from generation_config.json)
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 2.5
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant to generate an image from user's description."
)


def _build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    image_slices: list[list[slice]],
    device: torch.device,
) -> torch.Tensor:
    """Build 4D causal attention mask with full attention at image positions.

    Args:
        batch_size: batch size (may be doubled for CFG).
        seq_len: total sequence length.
        image_slices: per-batch list of slice objects marking image token
            positions that should use full (non-causal) attention.
        device: target device.
    """
    # Causal (lower-triangular) mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(0)
    # Enable full attention at image positions
    for slices in image_slices:
        for s in slices:
            mask[s, s] = True
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).contiguous()


def _build_rope_image_info(
    tokenizer_output: Any,
    batch_size: int,
) -> list[list[tuple[slice, tuple[int, int]]]]:
    """Extract 2D-RoPE image info from the tokenizer output.

    Returns a per-batch list of ``[(slice, (token_h, token_w)), ...]`` tuples
    describing where image tokens sit in the sequence and their spatial layout.
    """
    gen_slices = getattr(tokenizer_output, "gen_image_slices", None)
    gen_image_info = getattr(tokenizer_output, "batch_gen_image_info", None)

    rope_image_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for b in range(batch_size):
        batch_info: list[tuple[slice, tuple[int, int]]] = []
        if gen_slices is not None and gen_image_info is not None:
            info = gen_image_info[b] if isinstance(gen_image_info, list) else gen_image_info
            slices = gen_slices[b] if isinstance(gen_slices[0], list) else gen_slices
            if info is not None and slices:
                token_h = getattr(info, "token_height", None)
                token_w = getattr(info, "token_width", None)
                if token_h is not None and token_w is not None:
                    for s in slices:
                        batch_info.append((s, (token_h, token_w)))
        rope_image_info.append(batch_info)
    return rope_image_info


class HunyuanImage3AR(PipelineStage):
    """Native AR stage for HunyuanImage-3 text-to-image generation.

    Runs the flow-matching diffusion loop directly using the sglang backbone
    (``forward_block``) and the diffusion I/O modules (``patch_embed``,
    ``timestep_emb``, ``time_embed``, ``final_layer``, ``time_embed_2``)
    that live on the AR model.

    Only direct image generation (text-to-image) is supported.

    Args:
        ar_model: The sglang-loaded HunyuanImage-3 backbone with diffusion
            I/O modules, providing ``forward_block``.
        vae: The pipeline-loaded VAE module (used for config only; decode
            happens in the decoding stage).
        tokenizer: Standard HF tokenizer (may be unused if we load the
            custom tokenizer ourselves).
        processor: The repo's HunyuanImage3ImageProcessor.
        scheduler: Flow-matching Euler scheduler.
        model_path: Path to the model repository (for loading the custom
            tokenizer).
    """

    def __init__(
        self,
        ar_model,
        vae=None,
        tokenizer=None,
        processor=None,
        scheduler=None,
        model_path: str = "",
    ):
        super().__init__()
        self.ar_model = ar_model
        self._vae = vae
        self._tokenizer = tokenizer
        self._processor = processor
        self._scheduler = scheduler
        self._model_path = model_path
        self._custom_tokenizer = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if not isinstance(self.ar_model, torch.nn.Module):
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                memory_intensive=True,
            )
        ]

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    # ------------------------------------------------------------------
    # Tokenizer / processor resolution
    # ------------------------------------------------------------------

    def _resolve_custom_tokenizer(self, server_args: ServerArgs):
        """Load the custom HunyuanImage3 tokenizer (once)."""
        if self._custom_tokenizer is not None:
            return self._custom_tokenizer

        model_path = self._model_path
        if not model_path:
            raise ValueError(
                "HunyuanImage3AR requires a model_path to load the custom tokenizer."
            )

        # Try loading the custom tokenizer class from the model repo
        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            tokenizer_cls = get_class_from_dynamic_module(
                "tokenization_hunyuan_image_3.HunyuanImage3TokenizerFast",
                model_path,
                revision=server_args.revision,
            )
            self._custom_tokenizer = tokenizer_cls.from_pretrained(
                model_path,
                revision=server_args.revision,
                trust_remote_code=server_args.trust_remote_code,
            )
            logger.info("Loaded custom HunyuanImage3 tokenizer from %s", model_path)
        except Exception as e:
            logger.warning(
                "Failed to load custom tokenizer from %s: %s. "
                "Falling back to AutoTokenizer.",
                model_path,
                e,
            )
            from transformers import AutoTokenizer

            self._custom_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=server_args.revision,
                trust_remote_code=server_args.trust_remote_code,
            )
        return self._custom_tokenizer

    def _get_image_info_class(self, tokenizer):
        """Extract the ImageInfo class from the tokenizer's module namespace.

        The tokenizer and processor may load ``tokenization_hunyuan_image_3.py``
        from different ``transformers_modules`` cache directories, producing
        different Python classes with the same name.  To avoid ``isinstance``
        failures inside the tokenizer, we always use the ``ImageInfo`` class
        that lives in the *tokenizer's* module.
        """
        tok_mod = sys.modules.get(tokenizer.__class__.__module__)
        if tok_mod is not None and hasattr(tok_mod, "ImageInfo"):
            return tok_mod.ImageInfo
        # Fallback: search parent modules
        parts = tokenizer.__class__.__module__.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            mod = sys.modules.get(parent)
            if mod is not None and hasattr(mod, "ImageInfo"):
                return mod.ImageInfo
        return None

    def _rebuild_image_info(self, image_info, ImageInfoCls):
        """Re-create *image_info* as an instance of *ImageInfoCls*.

        Copies all instance attributes so the tokenizer's ``isinstance`` check
        succeeds even when the processor and tokenizer loaded
        ``tokenization_hunyuan_image_3.py`` from different cache directories.
        """
        if isinstance(image_info, ImageInfoCls):
            return image_info
        # Create a bare instance and copy all attributes from the source.
        new_info = ImageInfoCls.__new__(ImageInfoCls)
        new_info.__dict__.update(image_info.__dict__)
        return new_info

    def _resolve_processor(self, server_args: ServerArgs):
        """Return the image processor, loading it lazily if needed."""
        if self._processor is not None:
            return self._processor

        model_path = self._model_path
        if not model_path or not server_args.trust_remote_code:
            return None

        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            hf_config_obj = server_args.hf_config if hasattr(server_args, "hf_config") else None
            if hf_config_obj is None:
                from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
                    get_hf_config,
                )
                hf_config_obj = get_hf_config(
                    model_path,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
            processor_cls = get_class_from_dynamic_module(
                "image_processor.HunyuanImage3ImageProcessor",
                model_path,
                revision=server_args.revision,
            )
            self._processor = processor_cls(hf_config_obj)
        except Exception as e:
            logger.warning("Failed to load image processor: %s", e)
        return self._processor

    # ------------------------------------------------------------------
    # Backbone forward (with TP broadcast for determinism)
    # ------------------------------------------------------------------

    def _backbone_forward(
        self,
        num_image_tokens: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
    ) -> torch.Tensor:
        """Run one backbone pass through the sglang forward_block."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size).contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        cos = cos.contiguous()
        sin = sin.contiguous()

        # Broadcast from rank 0 for deterministic TP collectives
        if model_parallel_is_initialized():
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                hidden_states = tp_group.broadcast(hidden_states, src=0)
                attention_mask = tp_group.broadcast(attention_mask, src=0)
                cos = tp_group.broadcast(cos, src=0)
                sin = tp_group.broadcast(sin, src=0)

        output = self.ar_model.forward_block(
            hidden_states,
            attention_mask,
            (cos, sin),
            num_image_tokens=num_image_tokens,
            first_step=first_step,
        )
        return output.view(batch_size, seq_len, hidden_size)

    # ------------------------------------------------------------------
    # Diffusion I/O helpers
    # ------------------------------------------------------------------

    def _instantiate_vae_tokens_first_step(
        self,
        hidden_states: torch.Tensor,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter VAE image embeddings + timestep embeddings into text hidden states.

        Used on the first diffusion step when hidden_states contains text
        token embeddings.
        """
        bsz, seqlen, n_embd = hidden_states.shape
        # Timestep conditioning for patch_embed
        t_emb = self.ar_model.time_embed(timesteps)
        # VAE latent → sequence embedding
        image_seq, token_h, token_w = self.ar_model.patch_embed(images, t_emb)
        # Scatter image embeddings at image_mask positions
        image_scatter_index = (
            torch.arange(seqlen, device=hidden_states.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        image_scatter_index = image_scatter_index.masked_select(image_mask.bool()).reshape(bsz, -1)
        hidden_states = hidden_states.clone()
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_seq,
        )
        return hidden_states

    def _instantiate_timestep_tokens(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        timestep_index: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter timestep embeddings into hidden_states at timestep_index positions."""
        bsz, seqlen, n_embd = hidden_states.shape
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(bsz, -1, n_embd)
        index = (
            torch.arange(seqlen, device=hidden_states.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        ts_scatter_index = index.masked_select(timestep_index.bool()).reshape(bsz, -1)
        hidden_states = hidden_states.clone()
        hidden_states.scatter_(
            dim=1,
            index=ts_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=timestep_emb,
        )
        return hidden_states

    def _build_non_first_step_input(
        self, timesteps: torch.Tensor, images: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """Build hidden states for non-first diffusion steps (no text tokens).

        Concatenates [timestep_emb, patch_embed(latents, time_embed(t))].
        """
        t_emb = self.ar_model.time_embed(timesteps)
        image_emb, _, _ = self.ar_model.patch_embed(images, t_emb)
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(
            batch_size, -1, self.ar_model.config.hidden_size
            if hasattr(self.ar_model.config, "hidden_size")
            else image_emb.shape[-1]
        )
        return torch.cat([timestep_emb, image_emb], dim=1)

    def _extract_diffusion_pred(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
        token_h: int,
        token_w: int,
        first_step: bool,
        num_special_tokens: int,
    ) -> torch.Tensor:
        """Extract the noise prediction from backbone output via final_layer."""
        n_embd = hidden_states.size(-1)
        t_emb = self.ar_model.time_embed_2(timesteps)

        if first_step:
            # Select image positions using the mask
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()
            ).reshape(-1, token_h * token_w, n_embd)
        else:
            # Non-first step: skip the timestep token (position 0)
            image_output = hidden_states[:, 1:, :]

        pred = self.ar_model.final_layer(image_output, t_emb, token_h, token_w)
        return pred

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the native diffusion loop and store the final latents."""
        # 1. Resolve tokenizer and processor
        tokenizer = self._resolve_custom_tokenizer(server_args)
        processor = self._resolve_processor(server_args)

        # 2. Determine image resolution
        width, height = align_hunyuan_image3_resolution(batch.width, batch.height)
        if processor is not None:
            image_info = processor.build_gen_image_info(f"{height}x{width}")
            height = image_info.image_height
            width = image_info.image_width
            token_h = image_info.token_height
            token_w = image_info.token_width
            # Ensure ImageInfo uses the tokenizer's module class so that
            # isinstance checks inside the tokenizer succeed.
            ImageInfoCls = self._get_image_info_class(tokenizer)
            if ImageInfoCls is not None:
                image_info = self._rebuild_image_info(image_info, ImageInfoCls)
        else:
            # Fallback: compute from VAE downsample factor
            vae_factor = getattr(
                self.ar_model.hf_config, "vae_downsample_factor", [16, 16]
            )
            if isinstance(vae_factor, (list, tuple)):
                vae_h = vae_factor[0]
                vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
            else:
                vae_h = vae_w = int(vae_factor)
            token_h = height // vae_h
            token_w = width // vae_w
            image_info = None

        num_image_tokens = token_h * token_w
        # Derive device from model weights to avoid mismatches between
        # get_local_torch_device() and the actual model placement.
        device = self.ar_model.model.embed_tokens.weight.device

        # 3. Build input sequence using the custom tokenizer
        batch_size = 1
        guidance_scale = float(
            getattr(batch, "guidance_scale", None) or _DEFAULT_GUIDANCE_SCALE
        )
        do_cfg = guidance_scale > 1.0
        cfg_factor = 2 if do_cfg else 1

        # Build tokenizer inputs
        prompts = [batch.prompt] * cfg_factor
        tokenizer_kwargs: dict[str, Any] = dict(
            batch_prompt=prompts,
            mode="gen_image",
            bot_task="image",
            sequence_template="instruct",
            cfg_factor=cfg_factor,
            image_base_size=getattr(
                processor, "vae_reso_group", None
            ) and processor.vae_reso_group.base_size,
        )

        # Provide gen image info if the tokenizer supports it
        if image_info is not None:
            tokenizer_kwargs["batch_gen_image_info"] = [image_info] * cfg_factor

        tokenizer_output_dict = tokenizer.apply_chat_template(**tokenizer_kwargs)
        # The output format: dict with 'output' and 'sections'
        if isinstance(tokenizer_output_dict, dict):
            tokenizer_output = tokenizer_output_dict.get("output", tokenizer_output_dict)
        else:
            tokenizer_output = tokenizer_output_dict

        # Extract tensors from tokenizer output
        if hasattr(tokenizer_output, "tokens"):
            input_ids = tokenizer_output.tokens.to(device)
        elif isinstance(tokenizer_output, torch.Tensor):
            input_ids = tokenizer_output.to(device)
        else:
            input_ids = tokenizer_output["tokens"].to(device)

        actual_batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Image mask
        if hasattr(tokenizer_output, "gen_image_mask"):
            image_mask = tokenizer_output.gen_image_mask.to(device)
        else:
            image_mask = tokenizer_output.get("gen_image_mask")
            if image_mask is not None:
                image_mask = image_mask.to(device)

        # Timestep scatter index
        if hasattr(tokenizer_output, "gen_timestep_scatter_index"):
            timestep_index = tokenizer_output.gen_timestep_scatter_index.to(device)
        else:
            timestep_index = tokenizer_output.get("gen_timestep_scatter_index")
            if timestep_index is not None:
                timestep_index = timestep_index.to(device)

        # 4. Build attention mask (4D causal + full attn at image positions)
        image_slices = getattr(tokenizer_output, "gen_image_slices", [[] for _ in range(actual_batch_size)])
        if not isinstance(image_slices[0], list):
            image_slices = [image_slices]
        attention_mask = _build_causal_attention_mask(
            actual_batch_size, seq_len, image_slices, device
        )

        # 5. Build 2D RoPE image info and compute cached cos/sin
        rope_image_info = _build_rope_image_info(tokenizer_output, actual_batch_size)
        head_dim = self.ar_model.cached_rope.head_dim
        cos, sin = self.ar_model.cached_rope(seq_len, device, rope_image_info=rope_image_info)

        # 6. Set up the diffusion scheduler
        num_inference_steps = int(
            getattr(batch, "num_inference_steps", None) or _DEFAULT_NUM_INFERENCE_STEPS
        )
        scheduler = self._scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        # 7. Prepare noise latents
        hf_config = self.ar_model.hf_config
        if hasattr(hf_config, "vae") and isinstance(hf_config.vae, dict):
            latent_channels = hf_config.vae["latent_channels"]
        else:
            latent_channels = getattr(hf_config, "latent_channels", 32)

        vae_factor = getattr(hf_config, "vae_downsample_factor", [16, 16])
        if isinstance(vae_factor, (list, tuple)):
            vae_h = vae_factor[0]
            vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
        else:
            vae_h = vae_w = int(vae_factor)

        latent_h = height // vae_h
        latent_w = width // vae_w

        generator = torch.Generator(device=device)
        if batch.seed is not None:
            generator.manual_seed(batch.seed)

        latents = torch.randn(
            actual_batch_size,
            latent_channels,
            latent_h,
            latent_w,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # Backbone forward agent (bound to num_image_tokens for KV cache)
        backbone_fn = partial(self._backbone_forward, num_image_tokens)

        # 8. Diffusion sampling loop
        for step_idx, t in enumerate(timesteps):
            first_step = step_idx == 0

            # Scale model input for scheduler
            latent_model_input = scheduler.scale_model_input(latents, t)

            # Prepare timestep tensor
            t_expand = t.repeat(actual_batch_size).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                if first_step:
                    # Embed text tokens
                    hidden_states = self.ar_model.model.get_input_embeddings(input_ids)
                    # Scatter VAE image embeddings at image positions
                    hidden_states = self._instantiate_vae_tokens_first_step(
                        hidden_states, latent_model_input, t_expand, image_mask,
                    )
                    # Scatter timestep embedding
                    if timestep_index is not None:
                        hidden_states = self._instantiate_timestep_tokens(
                            hidden_states, t_expand, timestep_index,
                        )
                else:
                    # No text tokens: build from scratch
                    hidden_states = self._build_non_first_step_input(
                        t_expand, latent_model_input, actual_batch_size,
                    )

                # Run backbone
                backbone_out = backbone_fn(
                    hidden_states, attention_mask, (cos, sin), first_step,
                )

                # Extract diffusion prediction
                pred = self._extract_diffusion_pred(
                    backbone_out, t_expand, image_mask,
                    token_h, token_w, first_step,
                    num_special_tokens=seq_len - num_image_tokens,
                )

            pred = pred.float()

            # Classifier-free guidance
            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # Keep single-batch latents
                pred = pred[:1]

            # Scheduler step
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]

            if do_cfg:
                latents = latents[:1]

            # After first step, text tokens are no longer needed
            if first_step:
                input_ids = None
                # Update attention mask for shorter sequence (non-first steps)
                # Non-first steps use a different sequence length, but the
                # forward_block handles this via the attn_meta mechanism.

        # 9. Store latents for the decoding stage
        # Apply VAE scaling/shift (inverse of what the decoding stage will do)
        vae_config = self._vae.config if self._vae is not None else None
        scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0) or 1.0)
        shift_factor = getattr(vae_config, "shift_factor", None)
        shift = float(shift_factor) if shift_factor else 0.0

        # The decoding stage expects: latents = (raw_latents - shift) * scaling_factor
        # so that decode does: raw_latents = latents / scaling_factor + shift
        batch.latents = ((latents.float() - shift) * scaling_factor).to(torch.bfloat16)

        logger.info(
            "HunyuanImage3AR produced latents %s for %dx%d image",
            tuple(batch.latents.shape),
            height,
            width,
        )
        return batch
