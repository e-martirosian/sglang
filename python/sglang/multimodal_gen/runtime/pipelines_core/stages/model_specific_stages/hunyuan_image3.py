# SPDX-License-Identifier: Apache-2.0
"""Model-specific pipeline stages for HunyuanImage-3.0 unified AR T2I.

The stages mirror the official ``HunyuanImage3ForCausalMM.generate_image``
flow from https://github.com/Tencent-Hunyuan/HunyuanImage-3.0 :

1. :class:`HunyuanImage3ARStage` – resolves the system prompt / bot task,
   optionally runs the AR text stage (think / recaption / image-ratio
   prediction), then builds the ``mode="gen_image"`` model inputs (chat
   template, tokens, cond/uncond batching, static KV cache).
2. :class:`HunyuanImage3DenoiseStage` – runs the flow-matching loop with the
   unified transformer as the denoiser (``HunyuanImage3Text2ImagePipeline``
   semantics: Euler solver, SD3-style timestep shift, CFG).
3. :class:`HunyuanImage3DecodeStage` – decodes the final latents to pixels
   with the bundled ``AutoencoderKLConv3D`` VAE.

Everything is eager PyTorch (SDPA attention, eager MoE; no Triton /
FlashInfer) so it runs on both CUDA and NPU.
"""

from typing import Any

import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_npu = current_platform.is_npu()


class _FlowMatchDiscreteScheduler:
    """Minimal Euler flow-matching scheduler.

    Mirrors the upstream ``FlowMatchDiscreteScheduler`` (shift + reverse +
    euler solver only), which is all the HunyuanImage-3.0 pipeline uses.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.reverse = reverse
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.timesteps_full: torch.Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        if self.shift != 1.0:
            # SD3-style timestep shift.
            sigmas = (self.shift * sigmas) / (1 + (self.shift - 1) * sigmas)
        if not self.reverse:
            sigmas = 1 - sigmas
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(
            dtype=torch.float32, device=device
        )
        self.timesteps_full = (sigmas * self.num_train_timesteps).to(
            dtype=torch.float32, device=device
        )

    def get_timestep_r(self, step_index: int) -> torch.Tensor:
        return self.timesteps_full[step_index + 1]


class HunyuanImage3ARStage(PipelineStage):

    def __init__(self, transformer, pipeline=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.pipeline = pipeline

    @property
    def parallelism_type(self) -> StageParallelismType:
        # The model is sharded across all visible devices via device_map="auto"
        # inside the single process, so every rank runs the same AR stage.
        return StageParallelismType.REPLICATED

    def _resolve_task_params(self, batch: Req, server_args: ServerArgs):
        """Resolve bot_task / use_system_prompt like upstream generate_image.

        Priority: sampling_params (per-request) > server_args (CLI) > generation_config (model default).
        """
        model = self.transformer
        sampling_params = batch.sampling_params
        gen_config = model.generation_config

        use_system_prompt = (
            getattr(sampling_params, "use_system_prompt", None)
            or getattr(server_args, "use_system_prompt", None)
            or getattr(gen_config, "use_system_prompt", None)
        )
        bot_task = (
            getattr(sampling_params, "bot_task", None)
            or getattr(server_args, "bot_task", None)
            or getattr(gen_config, "bot_task", "image")
        )
        custom_system_prompt = getattr(sampling_params, "system_prompt", None)

        get_system_prompt = model.resolve_get_system_prompt()
        system_prompt = get_system_prompt(use_system_prompt, bot_task, custom_system_prompt)
        system_prompt = system_prompt.strip() if system_prompt is not None else ""
        return bot_task, system_prompt

    def _resolve_image_size(self, batch: Req):
        if batch.height and batch.width:
            return (batch.height, batch.width)
        return "auto"

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        model = self.transformer
        inner = model.inner
        sampling_params = batch.sampling_params

        prompt = batch.prompt
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise ValueError(
                    "HunyuanImage-3.0 currently supports a single prompt per request."
                )
            prompt = prompt[0]

        bot_task, system_prompt = self._resolve_task_params(batch, server_args)
        image_size = self._resolve_image_size(batch)
        need_ratio = image_size == "auto" or bot_task == "img_ratio"
        max_new_tokens = getattr(sampling_params, "max_new_tokens", 2048)
        seed = getattr(sampling_params, "seed", None)

        tkw = model.tokenizer
        cot_text = None
        # TI2I conditioning images are not wired yet (T2I only); upstream
        # reuses the encoded cond images across stages via this cache.
        batch_cond_images_cache = None

        if bot_task in ["think", "recaption", "think_recaption"]:
            cot_text, image_size, batch_cond_images_cache = self._run_text_stage(
                model=model,
                inner=inner,
                tkw=tkw,
                prompt=prompt,
                bot_task=bot_task,
                system_prompt=system_prompt,
                image_size=image_size,
                need_ratio=need_ratio,
                max_new_tokens=max_new_tokens,
                batch_cond_images_cache=batch_cond_images_cache,
            )
        elif need_ratio:
            image_size, batch_cond_images_cache = self._run_ratio_stage(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                seed=seed,
                batch_cond_images_cache=batch_cond_images_cache,
            )

        # Final stage: build the gen_image model inputs (chat template,
        # cond/uncond stacking, static KV cache, 2D RoPE info).
        model_inputs = model.prepare_model_inputs(
            prompt=prompt,
            image=None,
            cot_text=cot_text,
            message_list=None,
            system_prompt=system_prompt,
            seed=seed,
            image_size=image_size,
            mode="gen_image",
            batch_cond_images=batch_cond_images_cache,
            infer_align_image_size=False,
        )

        batch.extra["model_inputs"] = model_inputs
        batch.extra["image_size"] = image_size
        batch.extra["cot_text"] = cot_text
        batch.latents = None
        return batch

    def _run_text_stage(
        self,
        model,
        inner,
        tkw,
        prompt: str,
        bot_task: str,
        system_prompt: str,
        image_size,
        need_ratio: bool,
        max_new_tokens: int,
        batch_cond_images_cache,
    ):
        """Upstream think / recaption / think_recaption AR text stage."""
        first_bot_task = bot_task.split("_")[0]
        stage_transitions = []

        if first_bot_task == "think" and "recaption" in bot_task:
            stage_transitions.append(
                (
                    tkw.end_of_think_token_id,
                    [tkw.convert_tokens_to_ids(tkw.recaption_token)],
                )
            )

        if need_ratio:
            answer_prefix_tokens = []
            if (
                getattr(model.generation_config, "sequence_template", "pretrain")
                == "instruct"
            ):
                answer_prefix_tokens = [
                    tkw.convert_tokens_to_ids(tkw.answer_token)
                ]
            image_base_size = model.image_processor.vae_reso_group.base_size
            if "recaption" in bot_task:
                transition_id = tkw.end_of_recaption_token_id
            else:
                transition_id = tkw.end_of_think_token_id
            stage_transitions.append(
                (
                    transition_id,
                    answer_prefix_tokens
                    + [tkw.boi_token_id, tkw.size_token_id(image_base_size)],
                )
            )
            final_stop_tokens = list(
                range(tkw.start_ratio_token_id, tkw.end_ratio_token_id + 1)
            )
            for start, end in getattr(tkw, "ratio_token_other_slices", []):
                final_stop_tokens.extend(range(start, end))
        else:
            if "recaption" in bot_task:
                final_stop_tokens = [tkw.end_of_recaption_token_id]
            else:
                final_stop_tokens = [
                    tkw.end_of_think_token_id,
                    tkw.end_of_recaption_token_id,
                ]

        model_inputs = model.prepare_model_inputs(
            prompt=prompt,
            image=None,
            message_list=None,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            mode="gen_text",
            bot_task=first_bot_task,
            batch_cond_images=batch_cond_images_cache,
            infer_align_image_size=False,
        )
        batch_cond_images_cache = model_inputs["batch_cond_images"]

        logits_processor = None
        if need_ratio:
            from transformers import LogitsProcessorList

            image_base_size = model.image_processor.vae_reso_group.base_size
            logits_processor = LogitsProcessorList(
                [
                    inner._ConditionalSliceVocabLogitsProcessor(
                        trigger_token_ids=[tkw.size_token_id(image_base_size)],
                        vocab_start=tkw.start_ratio_token_id,
                        vocab_end=tkw.end_ratio_token_id + 1,
                        other_slices=getattr(tkw, "ratio_token_other_slices", []),
                        force_greedy=True,
                    )
                ]
            )

        input_length = model_inputs["input_ids"].shape[1]
        if stage_transitions:
            model_inputs["stage_transitions"] = stage_transitions
            model_inputs["final_stop_tokens"] = final_stop_tokens
        if logits_processor is not None:
            model_inputs["logits_processor"] = logits_processor
       
        outputs = model.generate_text(**model_inputs)
        generated_tokens = outputs[:, input_length:]

        if "recaption" in bot_task:
            end_token_id = tkw.end_of_recaption_token_id
        else:
            end_token_id = tkw.end_of_think_token_id
        end_positions = (generated_tokens[0] == end_token_id).nonzero(as_tuple=False)
        if end_positions.numel() > 0:
            end_pos = end_positions[0].item()
            cot_tokens = generated_tokens[0, : end_pos + 1]
        else:
            cot_tokens = generated_tokens[0]
        cot_text_gen = tkw.decode(cot_tokens)

        if first_bot_task == "think":
            cot_text = [tkw.think_token + cot_text_gen]
        else:
            cot_text = [tkw.recaption_token + cot_text_gen]

        if getattr(model.generation_config, "drop_think", False) and (
            tkw.think_token in cot_text[0]
        ):
            if tkw.recaption_token in cot_text[0]:
                recaption_part = cot_text[0].split(tkw.recaption_token)[1]
                if tkw.end_of_recaption_token in recaption_part:
                    recaption_part = recaption_part.split(
                        tkw.end_of_recaption_token
                    )[0]
                cot_text = [
                    tkw.recaption_token
                    + recaption_part
                    + tkw.end_of_recaption_token
                ]

        if need_ratio:
            ratio_token_id = outputs[0, -1].item()
            ratio_index = inner._get_ratio_index_from_token(ratio_token_id, tkw)
            reso = model.image_processor.vae_reso_group[ratio_index]
            image_size = (reso.height, reso.width)

        return cot_text, image_size, batch_cond_images_cache

    def _run_ratio_stage(
        self,
        model,
        prompt: str,
        system_prompt: str,
        seed,
        batch_cond_images_cache,
    ):
        """Upstream standalone image-ratio prediction stage (image_size=auto)."""
        model.image_processor.build_img_ratio_slice_logits_proc(model.tokenizer)
        model_inputs = model.prepare_model_inputs(
            prompt=prompt,
            image=None,
            cot_text=None,
            message_list=None,
            max_new_tokens=1,
            system_prompt=system_prompt,
            seed=seed,
            mode="gen_text",
            bot_task="img_ratio",
            batch_cond_images=batch_cond_images_cache,
            infer_align_image_size=False,
        )
        batch_cond_images_cache = model_inputs["batch_cond_images"]
        model_inputs["do_sample"] = False
        model_inputs["logits_processor"] = (
            model.image_processor.img_ratio_slice_logits_processor
        )
        outputs = model.generate_text(**model_inputs)
        ratio_index = outputs[0, -1].item()
        reso = model.image_processor.vae_reso_group[ratio_index]
        return (reso.height, reso.width), batch_cond_images_cache


class HunyuanImage3DenoiseStage(PipelineStage):
    """Flow-matching denoising with the unified transformer as the denoiser.

    Mirrors the sampling loop of the upstream
    ``HunyuanImage3Text2ImagePipeline.__call__``: Euler solver over
    ``linspace(1, 0)`` sigmas with SD3-style shift, classifier-free guidance
    on the cond/uncond pair, and per-step KV-cache updates.
    """

    def __init__(self, transformer, pipeline=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.pipeline = pipeline

    @property
    def parallelism_type(self) -> StageParallelismType:
        # The model is sharded across all visible devices via device_map="auto"
        # inside the single process, so every rank runs the same denoising
        # loop on the same shard.
        return StageParallelismType.REPLICATED

    def _prepare_latents(
        self,
        batch: Req,
        image_size: tuple[int, int],
        device: torch.device,
        generator,
    ) -> torch.Tensor:
        model = self.transformer
        config = model.config
        latent_channels = config.vae["latent_channels"]
        downsample = config.vae_downsample_factor
        if isinstance(downsample, (list, tuple)):
            factor_h, factor_w = downsample[0], downsample[1]
        else:
            factor_h = factor_w = downsample
        height, width = image_size
        shape = (1, latent_channels, height // factor_h, width // factor_w)
        if isinstance(generator, (list, tuple)):
            generator = generator[0] if generator else None
        return torch.randn(
            shape, generator=generator, device=device, dtype=torch.bfloat16
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        model = self.transformer.inner.model
        sampling_params = batch.sampling_params
        device = get_local_torch_device()

        model_inputs = batch.extra.get("model_inputs")
        if model_inputs is None:
            raise ValueError(
                "HunyuanImage3DenoiseStage requires 'model_inputs' from the AR stage."
            )
        image_size = batch.extra["image_size"]
        if not isinstance(image_size, (tuple, list)):
            raise ValueError(
                "HunyuanImage-3.0 could not resolve a concrete image size "
                "(got 'auto' without a ratio stage)."
            )

        gen_config = self.transformer.generation_config
        num_inference_steps = int(
            sampling_params.num_inference_steps
            or getattr(gen_config, "diff_infer_steps", 50)
        )
        guidance_scale = float(
            sampling_params.guidance_scale
            or getattr(gen_config, "diff_guidance_scale", 2.5)
        )
        flow_shift = float(
            getattr(sampling_params, "flow_shift", None)
            or getattr(gen_config, "flow_shift", 3.0)
        )
        cfg_distilled = bool(getattr(model.config, "cfg_distilled", False))
        meanflow = bool(getattr(model.config, "use_meanflow", False))
        do_cfg = guidance_scale > 1.0
        cfg_factor = 1 if cfg_distilled else (1 + int(do_cfg))

        scheduler = _FlowMatchDiscreteScheduler(shift=flow_shift, reverse=True)
        scheduler.set_timesteps(num_inference_steps, device)
        timesteps = scheduler.timesteps

        model_kwargs = dict(model_inputs)
        generator = model_kwargs.get("generator")
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.transformer.prepare_attention_mask(input_ids, model_kwargs)
        model_kwargs["attention_mask"] = attention_mask.to(device)

        latents = self._prepare_latents(batch, image_size, device, generator)

        # Compute post_token_len and num_special_tokens from model_inputs,
        # mirroring the upstream generate() method logic.
        post_token_len = self.transformer.compute_post_token_len(model_inputs)
        num_special_tokens = self.transformer.compute_num_special_tokens(model_inputs)
        self.transformer.inner.post_token_len = post_token_len
        self.transformer.inner.num_special_tokens = num_special_tokens

        # Also set num_image_tokens from batch_gen_image_info if available
        batch_gen_image_info = model_kwargs.get("batch_gen_image_info")
        if batch_gen_image_info and len(batch_gen_image_info) > 0:
            self.transformer.inner.num_image_tokens = (
                batch_gen_image_info[0].image_token_length
            )

        logger.info(
            "HunyuanImage-3.0 denoising: %s steps, guidance %.2f, flow_shift %.2f, "
            "cfg_distilled=%s, latent_shape=%s, post_token_len=%s, "
            "num_special_tokens=%s",
            num_inference_steps,
            guidance_scale,
            flow_shift,
            cfg_distilled,
            tuple(latents.shape),
            post_token_len,
            num_special_tokens,
        )

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * cfg_factor)

            if meanflow:
                r = scheduler.get_timestep_r(i)
                r_expand = r.repeat(latent_model_input.shape[0])
            else:
                r_expand = None
            model_kwargs["timesteps_r"] = r_expand

            t_expand = t.repeat(latent_model_input.shape[0])

            if cfg_distilled:
                model_kwargs["guidance"] = torch.tensor(
                    [1000.0 * guidance_scale], device=device, dtype=torch.bfloat16
                )

            denoise_inputs = self.transformer.prepare_denoise_inputs(
                input_ids, model_kwargs, latent_model_input, t_expand
            )

            model_output = self.transformer.denoise_forward(
                denoise_inputs, first_step=(i == 0)
            )
            pred = model_output["diffusion_prediction"].to(dtype=torch.float32)

            if do_cfg and not cfg_distilled:
                pred_cond, pred_uncond = pred.chunk(2)
                # Upstream ClassifierFreeGuidance (non-original formulation):
                # pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # Euler step on the flow-matching trajectory.
            sigma = scheduler.sigmas[i]
            sigma_next = scheduler.sigmas[i + 1]
            latents = (
                latents.to(torch.float32) + pred * (sigma_next - sigma)
            ).to(torch.bfloat16)

            if i != len(timesteps) - 1:
                model_kwargs = self.transformer.update_denoise_kwargs(model_output, model_kwargs)
                input_ids = None

        batch.latents = latents.to(torch.float32)
        return batch


class HunyuanImage3DecodeStage(PipelineStage):
    """Decode HunyuanImage-3.0 latents to pixels with AutoencoderKLConv3D.

    Mirrors the upstream post-loop handling: un-scale by the VAE scaling
    factor, add the temporal dim expected by the 3D conv VAE, decode under
    fp16 autocast, and normalize to [0, 1].
    """

    def __init__(self, vae, pipeline=None) -> None:
        super().__init__()
        self.vae = vae
        self.pipeline = pipeline
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        latents = batch.latents
        if latents is None:
            raise ValueError(
                "HunyuanImage3DecodeStage requires latents from the denoise stage."
            )
        vae = self.vae
        latents = latents.to(get_local_torch_device())

        if getattr(vae.config, "scaling_factor", None):
            latents = latents / vae.config.scaling_factor
        if getattr(vae.config, "shift_factor", None):
            latents = latents + vae.config.shift_factor

        has_temporal_factor = hasattr(vae, "ffactor_temporal")
        if has_temporal_factor:
            latents = latents.unsqueeze(2)

        device_type = current_platform.device_type
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=True):
            frames = vae.decode(latents, return_dict=False)[0]

        if has_temporal_factor:
            assert frames.shape[2] == 1, (
                "image should have shape [B, C, T, H, W] with T == 1"
            )
            frames = frames.squeeze(2)

        # VAE output is [-1, 1]; normalize to [0, 1] like upstream denormalize.
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = server_args.pipeline_config.post_decoding(frames, server_args)

        return OutputBatch(
            output=frames,
            metrics=batch.metrics,
            noise_pred=None,
            usage=batch.usage,
        )
