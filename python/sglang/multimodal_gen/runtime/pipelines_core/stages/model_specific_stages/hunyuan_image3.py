import contextlib
import gc
from functools import partial

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    align_hunyuan_image3_resolution,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
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
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Submodules of the official shell model that actually run on the local
# accelerator (embedding stack, diffusion I/O projections, conditioning
# encoders). The transformer backbone ("model.layers") is never executed:
# every backbone call is routed into the sglang backbone via forward_block.
_SHELL_ACCELERATOR_MODULES = (
    "vision_model",
    "vision_aligner",
    "timestep_emb",
    "patch_embed",
    "time_embed",
    "time_embed_2",
    "final_layer.model",
    "model.wte",
    "model.ln_f",
    "lm_head",
)

# AR submodules whose weights the sglang backbone (ar_model) already holds.
# Whenever the backbone is unsharded and resident on the local accelerator,
# the shell shares these tensors instead of re-reading the checkpoint, so
# the AR model is only loaded once.
_SHELL_AR_SHARED_MODULES = ("model.wte", "model.ln_f", "lm_head")


def _shell_inner_forward(agent, *args, **kwargs):
    """Replacement for the shell's inner model forward.

    Mirrors the vLLM orchestrator's ``ext_forward``: pulls the prepared
    denoising inputs out of the kwargs and delegates the backbone pass to
    the forward agent (which routes into the sglang backbone).
    """
    assert len(args) == 0, "args should be empty"
    hidden_states = kwargs.pop("inputs_embeds")
    attention_mask = kwargs.pop("attention_mask", None)
    position_ids = kwargs.pop("position_ids", None)
    custom_pos_emb = kwargs.pop("custom_pos_emb", None)
    first_step = kwargs.pop("first_step", False)
    hidden_states = agent(
        hidden_states, position_ids, attention_mask, custom_pos_emb, first_step
    )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )


class HunyuanImage3AR(PipelineStage):
    """
    AR stage for HunyuanImage-3 text-to-image generation.

    Following the vLLM HunyuanImage-3 orchestrator design, this stage keeps
    an official HF "shell" model for prompt preparation, timestep/embedding
    handling and the diffusion sampling loop, while every backbone forward
    is rerouted (via a patched inner forward) into the sglang-loaded,
    TP/FSDP-sharded backbone's ``forward_block``. The stage stops before
    VAE decode: the final latents are handed to the standard decoding stage.

    Only direct image generation (bot_task="image") is supported; CoT /
    recaption / ratio modes are not implemented.

    Args:
        ar_model: The sglang-loaded HunyuanImage-3 backbone, providing
            ``forward_block``.
        vae: The pipeline-loaded VAE module. Shared with the shell so its
            weights are not loaded a second time.
    """

    def __init__(self, ar_model, vae=None):
        super().__init__()
        self.ar_model = ar_model
        self._vae = vae
        self._shell = None

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
        # All TP ranks must enter forward_block together; running the stage
        # on the main rank only would deadlock the backbone collectives.
        return StageParallelismType.REPLICATED

    # --- official shell model -------------------------------------------------

    def _ensure_shell(self, server_args: ServerArgs):
        """Build (once) the official HF shell model with a selective device map."""
        if self._shell is not None:
            return self._shell
        if not server_args.trust_remote_code:
            raise ValueError(
                "HunyuanImage-3 AR stage requires --trust-remote-code to load "
                "the official modeling code from the model repository."
            )
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_path = maybe_download_model(server_args.model_path)
        shell_cls = get_class_from_dynamic_module(
            "modeling_hunyuan_image_3.HunyuanImage3ForCausalMM",
            model_path,
            revision=server_args.revision,
        )

        device = get_local_torch_device()
        device_str = str(device)
        # Reuse the weights already loaded by the pipeline instead of
        # reading them from the checkpoint a second time.
        shared_ar = self._shareable_ar_tensors(device)
        share_vae = self._shareable_vae(device)
        if shared_ar:
            logger.info(
                "Sharing AR weights with the shell model (skipping re-load): %s",
                sorted(shared_ar),
            )
        if share_vae:
            logger.info("Sharing the pipeline VAE module with the shell model")

        # Note: the vae must share the accelerator device with the model;
        # diffusers' DiffusionPipeline.device rejects mixed-device modules,
        # even though decode itself is intercepted and never executed.
        device_map = {
            "model.layers": "meta",
            "vae": "meta" if share_vae else device_str,
        }
        for name in _SHELL_ACCELERATOR_MODULES:
            device_map[name] = "meta" if name in shared_ar else device_str

        load_kwargs = dict(
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            moe_impl="eager",
            low_cpu_mem_usage=True,
        )
        logger.info("Building HunyuanImage-3 shell model from %s", model_path)
        try:
            shell = shell_cls.from_pretrained(
                model_path, device_map=device_map, **load_kwargs
            )
        except Exception as e:
            logger.warning(
                "Meta placement for model.layers failed (%s); retrying with "
                "CPU placement.",
                e,
            )
            device_map["model.layers"] = "cpu"
            shell = shell_cls.from_pretrained(
                model_path, device_map=device_map, **load_kwargs
            )
        self._bind_shared_shell_weights(shell, shared_ar, device)
        if share_vae:
            shell.vae = self._vae
        shell.load_tokenizer(model_path)

        # The backbone layers are never executed: every backbone call is
        # routed into the sglang backbone via forward_block. Drop them to
        # reclaim memory.
        shell.model.layers = torch.nn.ModuleList()
        gc.collect()

        shell.eval()
        self._shell = shell
        logger.info("HunyuanImage-3 shell model ready (backbone delegated to sglang)")
        return shell

    # --- weight sharing with pipeline-loaded components ----------------------

    @staticmethod
    def _same_device(tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device.type == device.type and (
            tensor.device.index or 0
        ) == (device.index or 0)

    def _shareable_ar_tensors(
        self, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """AR embedding/norm/head tensors already resident in the sglang backbone.

        Sharing them with the shell avoids loading the AR weights a second
        time. This is only safe when the backbone is not sharded (TP/FSDP)
        and lives on the local accelerator; otherwise an empty dict is
        returned and the shell loads these modules from the checkpoint.
        """
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            DTensor = ()  # type: ignore[assignment]

        ar = self.ar_model
        shared: dict[str, torch.Tensor] = {}
        if not isinstance(ar, torch.nn.Module):
            return shared
        try:
            candidates = {
                "model.wte": ar.model.embed_tokens.weight,
                "model.ln_f": ar.model.norm.weight,
                "lm_head": ar.lm_head.weight,
            }
            vocab_size = getattr(ar, "unpadded_vocab_size", None)
        except AttributeError:
            return shared

        for name, tensor in candidates.items():
            if not isinstance(tensor, torch.Tensor) or isinstance(tensor, DTensor):
                continue
            if not self._same_device(tensor, device):
                continue
            # VocabParallelEmbedding may pad the vocab dimension; the shell
            # expects the exact vocab size, so fall back to checkpoint load.
            if name in ("model.wte", "lm_head") and (
                vocab_size is None or tensor.shape[0] != vocab_size
            ):
                continue
            shared[name] = tensor
        return shared

    def _shareable_vae(self, device: torch.device) -> bool:
        """Whether the pipeline-loaded VAE can be shared with the shell."""
        vae = self._vae
        if not isinstance(vae, torch.nn.Module):
            return False
        try:
            param = next(vae.parameters())
        except StopIteration:
            return False
        return self._same_device(param, device)

    def _bind_shared_shell_weights(
        self,
        shell: torch.nn.Module,
        shared_ar: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Materialize meta-placed shell params from the sglang backbone."""
        if not shared_ar:
            return
        from accelerate.utils import set_module_tensor_to_device

        for name, tensor in shared_ar.items():
            param_name = f"{name}.weight"
            meta_param = shell.get_parameter(param_name)
            if meta_param.shape != tensor.shape:
                raise RuntimeError(
                    f"Cannot share AR weight {param_name} with the shell: "
                    f"shape {tuple(tensor.shape)} does not match the shell's "
                    f"{tuple(meta_param.shape)}."
                )
            set_module_tensor_to_device(
                shell, param_name, device, value=tensor.data
            )

    # --- backbone routing -----------------------------------------------------

    @contextlib.contextmanager
    def _ext_forward_context(self, shell, agent):
        """Temporarily route the shell's inner forward to the agent."""
        inner = shell.model
        origin_forward = inner.forward
        inner.forward = partial(_shell_inner_forward, agent)
        try:
            yield
        finally:
            inner.forward = origin_forward

    def _forward_agent(
        self,
        num_image_tokens,
        hidden_states,
        position_ids,
        attention_mask,
        custom_pos_emb,
        first_step,
    ):
        """Run one backbone pass through the sglang forward_block."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size).contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        cos = cos.contiguous()
        sin = sin.contiguous()

        # All ranks replicate the shell deterministically, but broadcasting
        # from rank 0 guarantees bitwise-identical inputs to the TP
        # collectives (same guard as the vLLM orchestrator).
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

    # --- forward ---------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the official diffusion loop and store the final latents."""
        shell = self._ensure_shell(server_args)

        # Sampling params already align the resolution; align again defensively.
        width, height = align_hunyuan_image3_resolution(batch.width, batch.height)
        image_info = shell.image_processor.build_gen_image_info(f"{height}x{width}")
        batch.height = image_info.image_height
        batch.width = image_info.image_width
        image_size = f"{image_info.image_height}x{image_info.image_width}"

        num_image_tokens = image_info.image_token_length + sum(
            int(bool(getattr(image_info, flag, False)))
            for flag in (
                "add_timestep_token",
                "add_guidance_token",
                "add_timestep_r_token",
            )
        )

        # Honor per-request sampling params on top of the official
        # generation_config.json defaults.
        generation_config = shell.generation_config
        if getattr(batch, "num_inference_steps", None):
            generation_config.diff_infer_steps = int(batch.num_inference_steps)
        if getattr(batch, "guidance_scale", None):
            generation_config.diff_guidance_scale = float(batch.guidance_scale)

        # Intercept the VAE decode at the end of the official pipeline: keep
        # the final latents for the sglang decoding stage and hand back a
        # dummy pixel tensor so the official postprocessing passes harmlessly.
        captured: dict[str, torch.Tensor] = {}
        original_decode = shell.vae.decode

        def _capture_decode(latents, *args, **kwargs):
            captured["latents"] = latents
            dummy = torch.zeros(
                (
                    latents.shape[0],
                    3,
                    latents.shape[2],
                    image_info.image_height,
                    image_info.image_width,
                ),
                device=latents.device,
                dtype=latents.dtype,
            )
            return (dummy,)

        agent = partial(self._forward_agent, num_image_tokens)
        with self._ext_forward_context(shell, agent):
            shell.vae.decode = _capture_decode
            try:
                shell.generate_image(
                    prompt=batch.prompt,
                    seed=batch.seed,
                    image_size=image_size,
                    bot_task="image",
                    output_type="np",
                    verbose=0,
                )
            finally:
                shell.vae.decode = original_decode

        if "latents" not in captured:
            raise RuntimeError(
                "HunyuanImage-3 AR stage did not capture latents; the official "
                "pipeline never reached VAE decode."
            )

        # The captured latents are already unscaled/shifted for decode; invert
        # that so the decoding stage's scale_and_shift reproduces the official
        # decode input exactly.
        vae_config = shell.vae.config
        scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0) or 1.0)
        shift_factor = getattr(vae_config, "shift_factor", None)
        shift = float(shift_factor) if shift_factor else 0.0
        latents = (captured["latents"].float() - shift) * scaling_factor
        batch.latents = latents.to(torch.bfloat16)

        logger.info(
            "HunyuanImage3AR produced latents %s for %dx%d image",
            tuple(batch.latents.shape),
            batch.height,
            batch.width,
        )
        return batch
