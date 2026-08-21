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
    get_local_torch_device,
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
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


def _tensor_stats(t: torch.Tensor) -> str:
    """Compact tensor stats for debug logging."""
    if t is None:
        return "None"
    tf = t.float().detach()
    return (
        f"shape={tuple(tf.shape)} dtype={t.dtype} "
        f"min={tf.min().item():.6f} max={tf.max().item():.6f} "
        f"mean={tf.mean().item():.6f} std={tf.std().item():.6f}"
    )


# Default sampling parameters (from generation_config.json)
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 2.5

# System prompts for HunyuanImage-3 (from vllm_omni system_prompt.py)
_SYSTEM_PROMPTS = {
    "en_unified": """You are an advanced multimodal model whose core mission is to analyze user intent and generate high-quality text and images.

#### Four Core Capabilities
1.  **Text-to-Text (T2T):** Generate coherent text responses from text prompts.
2.  **Text-to-Image (T2I):** Generate high-quality images from text prompts.
3.  **Text & Image to Text (TI2T):** Generate accurate text responses based on a combination of images and text.
4.  **Text & Image to Image (TI2I):** Generate modified images based on a reference image and editing instructions.

---
### Image Generation Protocol (for T2I & TI2I)
You will operate in one of two modes, determined by the user's starting tag:
#### **<recaption> Mode (Prompt Rewriting)**:
*   **Trigger:** Input begins with `<recaption>`.
*   **Task:** Immediately rewrite the user's text into a structured, objective, and detail-rich professional-grade prompt.
*   **Output:** Output only the rewritten prompt within `<recaption>` tags: `<recaption>Rewritten professional-grade prompt</recaption>`

#### **<think> Mode (Think + Rewrite)**:
*   **Trigger:** Input begins with `<think>`.
*   **Task:** First, conduct a structured analysis of the request within `<think>` tags. Then, output the professional prompt, rewritten based on the analysis, within `<recaption>` tags.
*   **Output:** Strictly adhere to the format: `<think>Analysis process</think><recaption>Rewritten prompt</recaption>`

---
### Execution Standards and Guidelines
#### **`<think>` Phase: Analysis Guidelines**
**For T2I (New Image Generation):**
Deconstruct the user's request into the following core visual components:
*   **Subject:** Key features of the main character/object, including appearance, pose, expression, and emotion.
*   **Composition:** Camera angle, lens type, and layout.
*   **Environment/Background:** The setting, time of day, weather, and background elements.
*   **Lighting:** Technical details such as light source type, direction, and quality.
*   **Color Palette:** The dominant hues and overall color scheme.
*   **Style/Quality:** The artistic style, clarity, depth of field, and other technical details.
*   **Text:** Identify any text to be rendered in the image, including its content, style, and position.
*   **Details:** Small elements that add narrative depth and realism.

**For TI2I (Image Editing):**
Adopt a task-diagnostic approach:
1.  **Diagnose Task:** Identify the edit type and analyze key requirements.
2.  **Prioritize Analysis:**
    *   **Adding:** Analyze the new element's position and appearance, ensuring seamless integration with the original image's lighting, shadows, and style.
    *   **Removing:** Identify the target for removal and determine how to logically fill the resulting space using surrounding textures and lighting.
    *   **Modifying:** Analyze what to change and what it should become, while emphasizing which elements must remain unchanged.
    *   **Style Transfer:** Deconstruct the target style into specific features (e.g., brushstrokes, color palette) and apply them to the original image.
    *   **Text Editing:** Ensure correct content and format. Consider the text's visual style (e.g., font, color, material) and how it adapts to the surface's perspective, curvature, and lighting.
    *   **Reference Editing:** Extract specific visual elements (e.g., appearance, posture, composition, lines, depth) from the reference image to generate an image that aligns with the text description while also incorporating the referenced content.
    *   **Inferential Editing:** Identify vague requests (e.g., "make it more professional") and translate them into concrete visual descriptions.

#### `<recaption>` Phase: Professional-Grade Prompt Generation Rules
**General Rewriting Principles (for T2I & TI2I):**
1.  **Structure & Logic:** Start with a global description. Use positional words (e.g., "foreground", "background") to define the layout.
2.  **Absolute Objectivity:** Avoid subjective terms. Convey aesthetics through precise descriptions of color, light, shadow, and materials.
3.  **Physical & Logical Consistency:** Ensure all descriptions adhere to the laws of physics and common sense.
4.  **Fidelity to User Intent:** Preserve the user's core concepts, subjects, and attributes. Text to be rendered in the image **must be enclosed in double quotes ("")**.
5.  **Camera & Resolution:** Translate camera parameters into descriptions of visual effects. Convert resolution information into natural language.

**T2I-Specific Guidelines:**
*   **Style Adherence & Inference:** Strictly follow the specified style. If none is given, infer the most appropriate style and detail it using professional terminology.
*   **Style Detailing:**
    *   **Photography/Realism:** Use professional photography terms to describe lighting, lens effects, and material textures.
    *   **Painting/Illustration:** Specify the art movement or medium's characteristics.
    *   **UI/Design:** Objectively describe the final product. Define layout, elements, and typography. Text content must be specific and unambiguous.

**TI2I-Specific Guidelines:**
*   **Preserve Unchanged Elements:** Emphasize elements that **remain unchanged**. Unless explicitly instructed, never alter a character's identity/appearance, the core background, camera angle, or overall style.
*   **Clear Editing Instructions:**
    *   **Replacement:** Use the logic "**replace B with A**," and provide a detailed description of A.
    *   **Addition:** Clearly state what to add, where, and what it looks like.
*   **Unambiguous Referencing:** Avoid vague references (e.g., "that person"). Use specific descriptions of appearance.
""",
    "en_vanilla": "You are a helpful assistant to generate an image from user's description.",
}


def _get_system_prompt(sys_type: str) -> str | None:
    """Get system prompt based on sys_type.
    
    Args:
        sys_type: System prompt type. Options:
            - "none": No system prompt
            - "en_unified": Unified English system prompt (default)
            - "en_vanilla": Simple English system prompt
            - "auto": Auto-select (currently maps to en_unified)
    
    Returns:
        System prompt string or None if sys_type is "none".
    """
    if sys_type == "none":
        return None
    elif sys_type in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[sys_type]
    elif sys_type == "auto":
        # Auto-select based on task (default to en_unified for image generation)
        return _SYSTEM_PROMPTS["en_unified"]
    else:
        logger.warning(
            f"Unknown sys_type '{sys_type}', falling back to 'en_unified'."
        )
        return _SYSTEM_PROMPTS["en_unified"]



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
        # Derive reshape dims from actual output (may differ from input
        # batch_size after TP broadcast).
        actual_batch = attention_mask.shape[0]
        actual_seq_len = output.shape[0] // actual_batch
        return output.view(actual_batch, actual_seq_len, hidden_size)

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
        """Scatter timestep embeddings into hidden_states at timestep_index positions.

        The ``timestep_emb`` module produces one embedding per batch element,
        but ``timestep_index`` may mark multiple sequence positions that should
        all receive the *same* embedding vector.
        """
        bsz, seqlen, n_embd = hidden_states.shape
        # One embedding per batch element → [bsz, 1, n_embd]
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(bsz, -1, n_embd)
        index = (
            torch.arange(seqlen, device=hidden_states.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        ts_scatter_index = index.masked_select(timestep_index.bool()).reshape(bsz, -1)
        num_positions = ts_scatter_index.shape[1]
        # Expand the single embedding to fill all marked positions
        timestep_emb = timestep_emb.expand(-1, num_positions, -1)
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

        # 2b. Ensure the AR model lives on the compute device.
        # When cpu_offload is enabled the pipeline loads weights on CPU;
        # we must move them to the accelerator before running inference.
        device = get_local_torch_device()
        model_device = self.ar_model.model.embed_tokens.weight.device
        if model_device.type == "cpu":
            logger.info("Moving AR model from CPU to %s", device)
            self.ar_model.to(device)
        else:
            device = model_device

        # 3. Build input sequence using the custom tokenizer
        _debug = os.environ.get("HUNYUAN_DEBUG", "0") == "1"
        batch_size = 1
        guidance_scale = float(
            getattr(batch, "guidance_scale", None) or _DEFAULT_GUIDANCE_SCALE
        )
        do_cfg = guidance_scale > 1.0
        cfg_factor = 2 if do_cfg else 1

        # Get bot_task and sys_type from batch (with defaults)
        bot_task = getattr(batch, "bot_task", "image")
        sys_type = getattr(batch, "sys_type", "en_unified")
        
        # Handle "none" bot_task (convert to "image" for tokenizer compatibility)
        if bot_task == "none":
            bot_task = "image"

        # Build tokenizer inputs
        # The base tokenizer supports cfg_factor natively (via batch_gen_infer).
        # When cfg_factor=2, it internally creates:
        #   - conditioned branch: real prompt (uncond_p=0.0)
        #   - unconditioned branch: prompt text replaced with <cfg> tokens (uncond_p=1.0)
        # This matches the vllm-omni TokenizerWrapper behaviour.
        tokenizer_kwargs: dict[str, Any] = dict(
            batch_prompt=[batch.prompt],
            mode="gen_image",
            bot_task=bot_task,
            sequence_template="instruct",
            cfg_factor=cfg_factor,
            image_base_size=getattr(
                processor, "vae_reso_group", None
            ) and processor.vae_reso_group.base_size,
        )

        # Add system prompt based on sys_type
        system_prompt = _get_system_prompt(sys_type)
        if system_prompt is not None:
            tokenizer_kwargs["batch_system_prompt"] = [system_prompt]

        # Provide gen image info if the tokenizer supports it
        if image_info is not None:
            tokenizer_kwargs["batch_gen_image_info"] = [image_info]

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

        if _debug:
            # Verify the two CFG branches have different tokens
            if actual_batch_size >= 2:
                branch0 = input_ids[0].cpu()
                branch1 = input_ids[1].cpu()
                diff_positions = (branch0 != branch1).nonzero(as_tuple=True)[0]
                logger.info(
                    "[DEBUG] input_ids: shape=%s, branch0 vs branch1: %d/%d tokens differ",
                    tuple(input_ids.shape), len(diff_positions), seq_len,
                )
                if len(diff_positions) > 0:
                    logger.info(
                        "[DEBUG]   first 10 diff positions: %s",
                        diff_positions[:10].tolist(),
                    )
                    for pos in diff_positions[:3]:
                        logger.info(
                            "[DEBUG]   pos %d: branch0=%d branch1=%d",
                            pos.item(), branch0[pos].item(), branch1[pos].item(),
                        )
                else:
                    logger.warning("[DEBUG]   WARNING: both branches have IDENTICAL tokens!")
            else:
                logger.warning("[DEBUG] input_ids batch_size=%d, expected 2 for CFG", actual_batch_size)

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

        # Non-first-step attention mask: shorter sequence [timestep_tok, image_toks...]
        non_first_seq_len = 1 + num_image_tokens
        non_first_image_slices = [
            [slice(1, non_first_seq_len)]
            for _ in range(actual_batch_size)
        ]
        non_first_attention_mask = _build_causal_attention_mask(
            actual_batch_size, non_first_seq_len, non_first_image_slices, device
        )

        # 5. Build 2D RoPE image info and compute cached cos/sin
        rope_image_info = _build_rope_image_info(tokenizer_output, actual_batch_size)
        cos, sin = self.ar_model.cached_rope(seq_len, device, rope_image_info=rope_image_info)

        # Pre-build RoPE for non-first steps (shorter sequence: 1 timestep + image tokens).
        non_first_rope_info: list[list[tuple[slice, tuple[int, int]]]] = [
            [(slice(1, non_first_seq_len), (token_h, token_w))]
            for _ in range(actual_batch_size)
        ]
        non_first_cos, non_first_sin = self.ar_model.cached_rope(
            non_first_seq_len, device, rope_image_info=non_first_rope_info
        )

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

        # Generate base noise with batch_size=1, then duplicate for CFG
        # (matching vllm-omni which uses the SAME noise for cond/uncond)
        latents = torch.randn(
            1,
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
            # Duplicate latents for CFG (same noise for both branches)
            if do_cfg:
                latent_model_input = torch.cat([latent_model_input] * cfg_factor, dim=0)

            # Prepare timestep tensor – match latent batch size
            latent_bs = latent_model_input.shape[0]
            t_expand = t.repeat(latent_bs).to(device)

            with torch.autocast(device_type=current_platform.device_type, dtype=torch.bfloat16, enabled=True):
                if first_step:
                    # Embed text tokens
                    hidden_states = self.ar_model.model.get_input_embeddings(input_ids)
                    if _debug:
                        logger.info("[DEBUG] step%d text_emb: %s", step_idx, _tensor_stats(hidden_states))
                        # Compare text embeddings at differing positions
                        if actual_batch_size >= 2:
                            diff_pos_list = (input_ids[0].cpu() != input_ids[1].cpu()).nonzero(as_tuple=True)[0]
                            if len(diff_pos_list) > 0:
                                emb0 = hidden_states[0, diff_pos_list].float().detach()
                                emb1 = hidden_states[1, diff_pos_list].float().detach()
                                diff_norm = (emb0 - emb1).norm(dim=-1)
                                logger.info(
                                    "[DEBUG]   text_emb at %d diff positions: "
                                    "mean_diff_norm=%.6f max_diff_norm=%.6f",
                                    len(diff_pos_list),
                                    diff_norm.mean().item(),
                                    diff_norm.max().item(),
                                )
                    # Scatter VAE image embeddings at image positions
                    hidden_states = self._instantiate_vae_tokens_first_step(
                        hidden_states, latent_model_input, t_expand, image_mask,
                    )
                    if _debug:
                        logger.info("[DEBUG] step%d after_vae_scatter: %s", step_idx, _tensor_stats(hidden_states))
                    # Scatter timestep embedding
                    if timestep_index is not None:
                        hidden_states = self._instantiate_timestep_tokens(
                            hidden_states, t_expand, timestep_index,
                        )
                    if _debug:
                        logger.info("[DEBUG] step%d after_ts_scatter: %s", step_idx, _tensor_stats(hidden_states))
                else:
                    # No text tokens: build from scratch
                    hidden_states = self._build_non_first_step_input(
                        t_expand, latent_model_input, actual_batch_size,
                    )
                    if _debug:
                        logger.info("[DEBUG] step%d non_first_input: %s", step_idx, _tensor_stats(hidden_states))

                # Select the correct RoPE and attention mask for this step
                if first_step:
                    step_cos, step_sin = cos, sin
                    step_attn_mask = attention_mask
                else:
                    step_cos, step_sin = non_first_cos, non_first_sin
                    step_attn_mask = non_first_attention_mask

                if _debug:
                    logger.info("[DEBUG] step%d backbone_in: hidden=%s cos=%s", step_idx, _tensor_stats(hidden_states), _tensor_stats(step_cos))

                # Run backbone
                backbone_out = backbone_fn(
                    hidden_states, step_attn_mask, (step_cos, step_sin), first_step,
                )

                if _debug:
                    logger.info("[DEBUG] step%d backbone_out: %s", step_idx, _tensor_stats(backbone_out))

                # Extract diffusion prediction
                pred = self._extract_diffusion_pred(
                    backbone_out, t_expand, image_mask,
                    token_h, token_w, first_step,
                    num_special_tokens=seq_len - num_image_tokens,
                )

                if _debug:
                    logger.info("[DEBUG] step%d pred_before_cfg: %s", step_idx, _tensor_stats(pred))

            pred = pred.float()

            # Classifier-free guidance
            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                if _debug:
                    logger.info("[DEBUG] step%d pred_cond: %s", step_idx, _tensor_stats(pred_cond))
                    logger.info("[DEBUG] step%d pred_uncond: %s", step_idx, _tensor_stats(pred_uncond))
                    logger.info("[DEBUG] step%d cond-uncond diff: %s", step_idx, _tensor_stats(pred_cond - pred_uncond))
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                if _debug:
                    logger.info("[DEBUG] step%d pred_after_cfg: %s", step_idx, _tensor_stats(pred))

            # Scheduler step (latents is always batch_size=1)
            latent_dtype = latents.dtype
            latents = scheduler.step(pred, t, latents, return_dict=False)[0].to(dtype=latent_dtype)

            if _debug:
                logger.info("[DEBUG] step%d latents: %s", step_idx, _tensor_stats(latents))

            # After first step, text tokens are no longer needed
            if first_step:
                input_ids = None
                # Update attention mask for shorter sequence (non-first steps)
                # Non-first steps use a different sequence length, but the
                # forward_block handles this via the attn_meta mechanism.

        # 9. Store latents for the decoding stage.
        # The denoising loop produces latents in the VAE-encoded space.
        # The decoding stage's ``scale_and_shift`` will convert them to
        # raw VAE space (``latents / scaling_factor + shift_factor``)
        # before calling ``vae.decode``.  We only need to add the temporal
        # dimension expected by the 3D VAE: [B, C, H, W] -> [B, C, 1, H, W].
        batch.latents = latents.to(torch.bfloat16).unsqueeze(2)

        logger.info(
            "HunyuanImage3AR produced latents %s for %dx%d image",
            tuple(batch.latents.shape),
            height,
            width,
        )
        return batch
