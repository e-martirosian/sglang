# SPDX-License-Identifier: Apache-2.0
"""Native input preparation for HunyuanImage-3.0 unified AR T2I.

Ports the input-preparation layer of the official
``HunyuanImage3ForCausalMM`` (``modeling_hunyuan_image_3.py``) to sglang,
for the T2I path only (no conditioning images / SigLIP2):

* :class:`HunyuanImage3NativeImageProcessor` – resolution-group / ratio
  table, ``build_gen_image_info``, ``prepare_full_attn_slices`` and the
  image-ratio vocab-slice logits processor.
* :class:`HunyuanImage3InputPreparationMixin` – ``preprocess_inputs`` /
  ``prepare_model_inputs`` / ``build_batch_rope_image_info``, the mixed
  causal+full attention mask, ``prepare_inputs_for_generation``,
  ``_update_model_kwargs_for_generation`` and a native ``generate_text``
  sampling loop replacing HF ``GenerationMixin.generate``.

The chat template itself stays in the official ``HunyuanImage3TokenizerFast``
(kept via transformers); ``ImageInfo`` / ``Resolution`` / ``ResolutionGroup``
are fetched from that tokenizer's module because the tokenizer asserts
``isinstance(info, ImageInfo)`` on the objects we build here.
"""

import random
import sys
from typing import Any, Optional

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_utils import (
    HunyuanStaticCache,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


def _default(value, fallback):
    return value if value is not None else fallback


def to_device(x, device):
    """Move a tensor (or nested dict/list of tensors) to ``device``;
    ``None`` passes through."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def get_tokenization_module(tokenizer=None):
    """Locate the official tokenization module
    (``transformers_modules.<...>.tokenization_hunyuan_image_3``).

    ``type(tokenizer).__module__`` cannot be trusted: depending on the
    transformers version the tokenizer class resolves to an internal
    transformers module (e.g. ``tokenization_utils_tokenizers``) that does
    not define ``Resolution`` / ``ImageInfo``.  The official module is
    registered in ``sys.modules`` by the adapter's
    ``_load_official_tokenizer`` (and by transformers' remote-code loading),
    so scan for it directly.
    """
    for module in sys.modules.values():
        if module is None:
            continue
        name = getattr(module, "__name__", "") or ""
        if "tokenization_hunyuan_image_3" in name and (
            hasattr(module, "ImageInfo") and hasattr(module, "Resolution")
        ):
            return module
    raise RuntimeError(
        "The official HunyuanImage-3 tokenization module is not loaded; the "
        "native input preparation requires the official tokenizer to be "
        "loaded with trust_remote_code=True."
    )


# =======================================================
#     Logits processors (verbatim ports of the official ones)
# =======================================================


class SliceVocabLogitsProcessor:
    """Restrict probabilities to a vocab slice (modality constraint).

    Note: like the official ``SliceVocabLogitsProcessor`` this *returns the
    sliced scores* (reduced vocab dim), so the sampling loop must map sampled
    indices back through the slice.
    """

    def __init__(self, vocab_start: int = None, vocab_end: int = None, **kwargs):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, (
                f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
            )
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = kwargs.get("other_slices", [])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_processed = scores[:, self.vocab_start : self.vocab_end]
        for other_slice in self.other_slices:
            scores_processed = torch.cat(
                [scores_processed, scores[:, other_slice[0] : other_slice[1]]], dim=-1
            )
        return scores_processed

    def __repr__(self):
        return (
            f"SliceVocabLogitsWarper(vocab_start={self.vocab_start}, "
            f"vocab_end={self.vocab_end}, other_slices={self.other_slices})"
        )


class StageTransitionLogitsProcessor:
    """Force the token sequences injected on stage transitions
    (e.g. ``<end_of_think>`` -> ``<answer><boi><size>``)."""

    def __init__(self, stage_transitions: list[tuple[int, list[int]]], batch_size: int):
        self.transition_map = {
            stop_id: list(append_ids) for stop_id, append_ids in stage_transitions
        }
        self.pending_tokens = [[] for _ in range(batch_size)]
        self.completed = [set() for _ in range(batch_size)]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        last_tokens = input_ids[:, -1]
        device = scores.device
        min_score = torch.finfo(scores.dtype).min

        for i in range(batch_size):
            last_token = last_tokens[i].item()

            # Consume pending tokens if the last token matches the head.
            if self.pending_tokens[i] and last_token == self.pending_tokens[i][0]:
                self.pending_tokens[i].pop(0)

            # If pending tokens remain, force the next token.
            if self.pending_tokens[i]:
                scores[i].fill_(min_score)
                scores[i, self.pending_tokens[i][0]] = 0
                continue

            # Trigger stage transition if needed.
            if last_token in self.transition_map and last_token not in self.completed[i]:
                self.completed[i].add(last_token)
                next_tokens = self.transition_map[last_token]
                if next_tokens:
                    self.pending_tokens[i] = list(next_tokens)
                    scores[i].fill_(min_score)
                    scores[i, self.pending_tokens[i][0]] = 0

            scores[i] = scores[i].to(device)

        return scores


class ConditionalSliceVocabLogitsProcessor:
    """After a trigger token (e.g. the ``<size>`` token) restrict sampling to
    the ratio-token slice; optionally forces greedy on that slice."""

    def __init__(
        self,
        trigger_token_ids: list[int],
        vocab_start: int,
        vocab_end: int,
        other_slices: Optional[list[tuple[int, int]]] = None,
        force_greedy: bool = False,
    ):
        self.trigger_token_ids = set(trigger_token_ids)
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = other_slices or []
        self.force_greedy = force_greedy

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        last_tokens = input_ids[:, -1]
        min_score = torch.finfo(scores.dtype).min
        for i in range(scores.size(0)):
            if last_tokens[i].item() not in self.trigger_token_ids:
                continue
            original_scores = scores[i].clone()
            scores[i].fill_(min_score)
            scores[i, self.vocab_start : self.vocab_end] = original_scores[
                self.vocab_start : self.vocab_end
            ]
            for start, end in self.other_slices:
                scores[i, start:end] = original_scores[start:end]
            if self.force_greedy:
                max_token_id = scores[i].argmax().item()
                scores[i].fill_(min_score)
                scores[i, max_token_id] = 0
        return scores


# =======================================================
#     Minimal image processor (T2I only)
# =======================================================


class HunyuanImage3NativeImageProcessor:
    """T2I-only port of ``HunyuanImage3ImageProcessor``.

    Keeps ``vae_reso_group`` (resolution/ratio table), ``build_gen_image_info``,
    ``prepare_full_attn_slices`` and ``build_img_ratio_slice_logits_processor``;
    drops everything related to conditioning-image encoding / SigLIP2.

    ``Resolution`` / ``ResolutionGroup`` come from the official tokenizer
    module so the ratio table is exactly the upstream one.
    """

    def __init__(self, config, tokenizer) -> None:
        self._tokenizer_ref = tokenizer
        tk_module = get_tokenization_module(tokenizer)
        Resolution = tk_module.Resolution
        ResolutionGroup = tk_module.ResolutionGroup

        self.config = config
        self.vae_reso_group = ResolutionGroup(
            base_size=config.image_base_size,
            step=None,
            align=16,
            extra_resolutions=[
                Resolution("1024x768"),
                Resolution("1280x720"),
                Resolution("768x1024"),
                Resolution("720x1280"),
            ],
        )
        self.img_ratio_slice_logits_processor = None
        # token grid factor = vae spatial downsample * AR patch size
        down_factor = config.vae_downsample_factor
        if isinstance(down_factor, (list, tuple)):
            down_factor = down_factor[0]
        self.vae_h_factor = down_factor * config.patch_size
        self.vae_w_factor = down_factor * config.patch_size
        self.cond_image_type = getattr(config, "cond_image_type", "vae_vit")
        self.cond_token_attn_type = getattr(config, "cond_token_attn_type", "joint_full")

    def build_gen_image_info(
        self, image_size, add_guidance_token=False, add_timestep_r_token=False
    ):
        """Parse an image size spec and build the official ``ImageInfo``."""
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif "x" in image_size:
                image_size = [int(s) for s in image_size.split("x")]
            elif ":" in image_size:
                image_size = [int(s) for s in image_size.split(":")]
                assert len(image_size) == 2, (
                    f"`image_size` should be in the format of 'W:H', got {image_size}."
                )
                # Note that ratio is width:height
                image_size = [image_size[1], image_size[0]]
            else:
                raise ValueError(
                    "`image_size` should be in the format of 'HxW', 'W:H' or "
                    f"<img_ratio_i>, got {image_size}."
                )
            assert len(image_size) == 2, (
                f"`image_size` should be in the format of 'HxW', got {image_size}."
            )
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(
                isinstance(s, int) for s in image_size
            ), (
                "`image_size` should be a tuple of two integers or a string in the "
                f"format of 'HxW', got {image_size}."
            )
        else:
            raise ValueError(
                "`image_size` should be a tuple of two integers or a string in the "
                f"format of 'WxH', got {image_size}."
            )
        image_width, image_height = self.vae_reso_group.get_target_size(
            image_size[1], image_size[0]
        )
        token_height = image_height // self.vae_h_factor
        token_width = image_width // self.vae_w_factor
        base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(
            image_size[1], image_size[0]
        )
        ImageInfo = self._image_info_cls()
        image_info = ImageInfo(
            image_type="gen_image",
            image_width=image_width,
            image_height=image_height,
            token_width=token_width,
            token_height=token_height,
            base_size=base_size,
            ratio_index=ratio_idx,
            add_guidance_token=add_guidance_token,
            add_timestep_r_token=add_timestep_r_token,
        )
        return image_info

    def _image_info_cls(self):
        # The tokenizer asserts isinstance(info, ImageInfo) with *its own*
        # ImageInfo class, so resolve the class lazily from the loaded
        # official tokenization module.
        return get_tokenization_module(self._tokenizer_ref).ImageInfo

    def prepare_full_attn_slices(self, output, batch_idx=None, with_gen=True):
        """Determine full-attention image slices according to the strategies.

        T2I has no conditioning images, so the cond part is always empty and
        only ``gen_image_slices`` contribute; the official branching is kept
        for faithfulness.
        """

        def _pick(slices):
            return slices[batch_idx] if batch_idx is not None else slices

        if self.cond_image_type == "vae":
            cond_choices = dict(
                causal=[],
                full=_pick(output.vae_image_slices),
            )
        elif self.cond_image_type == "vit":
            cond_choices = dict(
                causal=[],
                full=_pick(output.vit_image_slices),
            )
        elif self.cond_image_type == "vae_vit":
            cond_choices = {
                "causal": [],
                "full": _pick(output.vae_image_slices) + _pick(output.vit_image_slices),
                "joint_full": _pick(output.joint_image_slices),
                "full_causal": _pick(output.vae_image_slices),
            }
        else:
            raise ValueError(f"Unknown cond_image_type: {self.cond_image_type}")
        slices = cond_choices[self.cond_token_attn_type]

        if with_gen:
            slices = slices + _pick(output.gen_image_slices)
        return slices

    def build_img_ratio_slice_logits_processor(self, tokenizer):
        if self.img_ratio_slice_logits_processor is None:
            self.img_ratio_slice_logits_processor = [
                SliceVocabLogitsProcessor(
                    vocab_start=tokenizer.start_ratio_token_id,
                    vocab_end=tokenizer.end_ratio_token_id + 1,
                    other_slices=getattr(tokenizer, "ratio_token_other_slices", []),
                )
            ]


# =======================================================
#     Input preparation mixin for the native CausalMM model
# =======================================================


class HunyuanImage3InputPreparationMixin:
    """Methods bound onto the native ``HunyuanImage3ForCausalMM``.

    Requires ``self.config``, ``self._tokenizer``, ``self.image_processor``
    and ``self.generation_config`` to be wired up (done at load time).
    """

    # ------------------------------------------------------------------
    # Small helpers (verbatim ports)
    # ------------------------------------------------------------------
    @staticmethod
    def check_inputs(prompt=None, image=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError(
                "`prompt` and `message_list` cannot be provided at the same time."
            )

    @staticmethod
    def _validate_and_batchify_text(text, name, check_batch_size=None):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), (
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        )
        if isinstance(text, str):
            text = [text]
        if check_batch_size is not None:
            assert len(text) == check_batch_size, (
                f"Input `{name}` should have the same batch size as other inputs"
                f"({check_batch_size}), got {len(text)}."
            )
        return text

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(
                    f"Length of seed must be equal to the batch_size({batch_size}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        return seeds

    # ------------------------------------------------------------------
    # RoPE image info
    # ------------------------------------------------------------------
    def build_batch_rope_image_info(self, output, sections):
        # Rope 1D. No need to build rope_image_info
        if self.config.rope_type == "default":
            return None

        # Rope 2D
        assert self.config.rope_type == "2d", (
            f"Rope type {self.config.rope_type} not supported by method "
            "'build_batch_rope_image_info'."
        )
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            rope_2d_image_slices = []
            rope_2d_image_shapes = []
            image_idx = 0

            for section in sections_i:
                if section["type"] in ["gen_image", "cond_vae_image", "cond_vit_image"]:
                    assert image_idx < len(image_slices), (
                        f"Image index {image_idx} out of range for image slices "
                        f"with length {len(image_slices)}."
                    )
                    rope_2d_image_slices.append(image_slices[image_idx])
                    rope_2d_image_shapes.append(
                        (section["token_height"], section["token_width"])
                    )
                    image_idx += 1

                elif section["type"] == "cond_joint_image":
                    assert image_idx + 1 < len(image_slices), (
                        f"Image index {image_idx + 1} out of range for image slices "
                        f"with length {len(image_slices)}."
                    )
                    assert len(section["token_height"]) == len(section["token_width"]), (
                        "token_height and token_width should have the same length, "
                        f"but got {len(section['token_height'])} and "
                        f"{len(section['token_width'])}"
                    )

                    if self.image_processor.cond_token_attn_type in ["full", "joint_full"]:
                        rope_2d_image_slices.extend(
                            [image_slices[image_idx], image_slices[image_idx + 1]]
                        )
                        rope_2d_image_shapes.extend(
                            list(zip(section["token_height"], section["token_width"]))
                        )
                    elif self.image_processor.cond_token_attn_type == "full_causal":
                        rope_2d_image_slices.append(image_slices[image_idx])
                        rope_2d_image_shapes.append(
                            (section["token_height"][0], section["token_width"][0])
                        )
                    elif self.image_processor.cond_token_attn_type == "causal":
                        pass
                    else:
                        raise NotImplementedError(
                            f"cond_token_attn_type "
                            f"{self.image_processor.cond_token_attn_type} not supported "
                            "by method 'build_batch_rope_image_info'."
                        )
                    image_idx += 2

            rope_image_info.append(list(zip(rope_2d_image_slices, rope_2d_image_shapes)))

        return rope_image_info

    # ------------------------------------------------------------------
    # Preprocess / prepare
    # ------------------------------------------------------------------
    def preprocess_inputs(
        self,
        prompt: "str | list[str]" = None,
        image=None,
        cot_text=None,
        message_list=None,
        cfg_factor=1,
        bot_task="auto",
        system_prompt=None,
        max_new_tokens=None,
        mode="gen_text",
        image_size="auto",
        infer_align_image_size=False,
        device=None,
        **kwargs,
    ):
        """T2I port of the official ``preprocess_inputs``.

        Conditioning images are not supported; token sequence assembly is
        delegated to the official tokenizer's ``apply_chat_template``.
        """
        # 1. Sanity check
        self.check_inputs(prompt, image, message_list)
        if image is not None:
            raise NotImplementedError(
                "Conditioning images (I2I) are not supported by the native path."
            )

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt

        batch_cond_images = kwargs.get("batch_cond_images", None)
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)
            if batch_cond_images is not None and any(batch_cond_images):
                raise NotImplementedError(
                    "Conditioning images (I2I) are not supported by the native path."
                )
            if mode == "gen_image":
                batch_gen_image_info = [
                    self.image_processor.build_gen_image_info(
                        image_size,
                        add_guidance_token=self.config.cfg_distilled,
                        add_timestep_r_token=self.config.use_meanflow,
                    )
                    for _ in range(batch_size)
                ]
            else:
                batch_gen_image_info = [None] * batch_size
            # Convert OpenAI message list into inner message list (text only)
            batch_message_list = [
                self._prepare_message_list_text_only(message_list_, gen_image_info)
                for message_list_, gen_image_info in zip(
                    batch_message_list, batch_gen_image_info
                )
            ]

        #   -- 2.2 Prompt, cot text, system prompt
        else:
            batch_prompt = self._validate_and_batchify_text(batch_prompt, "prompt")
            batch_size = len(batch_prompt)
            batch_cot_text = self._validate_and_batchify_text(
                batch_cot_text, "cot_text", batch_size
            )
            batch_system_prompt = self._validate_and_batchify_text(
                batch_system_prompt, "system_prompt", batch_size
            )
            if mode == "gen_image":
                batch_gen_image_info = [
                    self.image_processor.build_gen_image_info(
                        image_size,
                        add_guidance_token=self.config.cfg_distilled,
                        add_timestep_r_token=self.config.use_meanflow,
                    )
                    for _ in range(batch_size)
                ]
            else:
                batch_gen_image_info = [None] * batch_size

        # Apply batched prompt to build the input sequence with associated info.
        # If `drop_think` enabled, always drop <tool_call> parts in the context.
        drop_think = kwargs.get(
            "drop_think", getattr(self.generation_config, "drop_think", False)
        )
        out = self._tokenizer.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_images=batch_cond_images,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get("max_length", self.generation_config.max_length),
            bot_task=bot_task,
            image_base_size=(
                None
                if mode == "gen_text" and bot_task == "auto"
                else self.image_processor.vae_reso_group.base_size
            ),
            sequence_template=getattr(self.generation_config, "sequence_template", "pretrain"),
            cfg_factor=cfg_factor,
            drop_think=drop_think,
        )
        out["batch_size"] = batch_size
        out["batch_cond_images"] = batch_cond_images
        out["batch_gen_image_info"] = batch_gen_image_info

        # 8. Define stop tokens by tasks
        tkw = self._tokenizer
        if bot_task == "auto":
            stop_token_id = dict(
                auto=self._tokenizer.conversation.stop_token_ids,
            )
        else:
            if image_size == "auto":
                extra_auto_stops = [tkw.ratio_token_id(i) for i in range(33)]
            else:
                extra_auto_stops = [tkw.boi_token_id]
            stop_token_id = dict(
                auto=self._tokenizer.conversation.stop_token_ids + extra_auto_stops,
                recaption=[tkw.end_of_recaption_token_id],
                think=[tkw.end_of_think_token_id, tkw.end_of_recaption_token_id],
                img_ratio=extra_auto_stops,
            )
        out["stop_token_id"] = stop_token_id

        return out

    @staticmethod
    def _prepare_message_list_text_only(message_list, gen_image_info):
        """OpenAI-style messages -> internal format, text content only."""
        inner_message_list = []
        for message in message_list:
            content = message["content"]
            if isinstance(content, str):
                inner_message_list.append(
                    dict(role=message["role"], type="text", content=content)
                )
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        inner_message_list.append(
                            dict(role=message["role"], type="text", content=item["text"])
                        )
                    elif item["type"] == "image":
                        raise NotImplementedError(
                            "Conditioning images (I2I) are not supported by the native path."
                        )
                    else:
                        raise NotImplementedError(
                            f"Message content type {item['type']} not supported."
                        )
            else:
                raise ValueError(
                    f"Message content should be str or list, but got {type(content)}."
                )

        if gen_image_info is not None:
            inner_message_list.append(
                dict(role="assistant", type="gen_image", content=gen_image_info)
            )
        return inner_message_list

    def prepare_model_inputs(
        self,
        prompt: "str | list[str]" = None,
        image=None,
        mode="gen_text",
        system_prompt=None,
        cot_text=None,
        image_size="auto",
        message_list=None,
        device=None,
        max_new_tokens=None,
        **kwargs,
    ):
        device = _default(device, self.device)

        # 1. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        if self.config.cfg_distilled:
            cfg_factor["gen_image"] = 1

        bot_task = kwargs.pop("bot_task", "auto")

        out = kwargs.pop("tokenizer_output", None)
        if out is None:
            out = self.preprocess_inputs(
                prompt=prompt,
                image=image,
                mode=mode,
                system_prompt=system_prompt,
                cot_text=cot_text,
                image_size=image_size,
                message_list=message_list,
                cfg_factor=cfg_factor[mode],
                bot_task=bot_task,
                **kwargs,
            )
        output, sections = out["output"], out["sections"]

        batch_size = out["batch_size"]
        batch_cond_images = out["batch_cond_images"]
        batch_gen_image_info = out["batch_gen_image_info"]
        stop_token_id = out["stop_token_id"]

        #   -- seed
        seeds = self.prepare_seed(seed=kwargs.get("seed"), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 4. Conditional images are not supported (T2I only), nothing to encode.

        # 5. Build position embeddings
        rope_image_info = self.build_batch_rope_image_info(output, sections)

        # 6. Build kv cache
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token
            # length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + _default(
                max_new_tokens, self.generation_config.max_length
            )
        cache = HunyuanStaticCache(
            config=self.config,
            max_batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=self.dtype,
            dynamic=mode == "gen_text",
            device=self.device,
        )

        # 7. Build position ids
        batch_position_ids = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device
        )[None].expand(
            batch_size * cfg_factor[mode], -1
        )  # use expand to share indices to save memory

        # 8. Define stop tokens by tasks
        tkw = self._tokenizer
        if mode == "gen_image":
            eos_token_id = None  # no eos needed for image generation
        else:
            if bot_task == "auto":
                stop_token_id = dict(
                    auto=self._tokenizer.conversation.stop_token_ids,
                )
            else:
                if image_size == "auto":
                    extra_auto_stops = tkw.get_all_ratio_token_ids()
                else:
                    extra_auto_stops = [tkw.boi_token_id]
                stop_token_id = dict(
                    auto=self._tokenizer.conversation.stop_token_ids + extra_auto_stops,
                    recaption=[tkw.end_of_recaption_token_id],
                    think=[tkw.end_of_think_token_id, tkw.end_of_recaption_token_id],
                    img_ratio=extra_auto_stops,
                )
            eos_token_id = stop_token_id[bot_task]

        # 9. Build model input kwargs
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            position_ids=batch_position_ids,
            past_key_values=cache,
            mode=mode,
            rope_image_info=rope_image_info,
            image_mask=to_device(output.gen_image_mask, device),
            timesteps_index=to_device(output.gen_timestep_scatter_index, device),
            guidance_index=to_device(output.guidance_scatter_index, device),
            timesteps_r_index=to_device(output.gen_timestep_r_scatter_index, device),
            cond_vae_images=None,
            cond_vae_image_mask=to_device(output.vae_image_mask, device),
            cond_timesteps=None,
            cond_timesteps_index=to_device(output.cond_timestep_scatter_index, device),
            cond_vit_images=None,
            cond_vit_image_mask=to_device(output.vit_image_mask, device),
            cond_vit_image_kwargs=None,
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            batch_cond_images=batch_cond_images,
            # generation config
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, device),
        )

        return model_input_kwargs

    # ------------------------------------------------------------------
    # Generation-loop plumbing (verbatim ports)
    # ------------------------------------------------------------------
    def _prepare_attention_mask_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        generation_config,
        model_kwargs: "dict[str, Any]",
    ) -> Optional[torch.Tensor]:
        """Create the 4-D bool attention mask (b, 1, S, S): causal tril with
        full attention inside the image slices."""
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_full_attn_slices = [
            self.image_processor.prepare_full_attn_slices(tokenizer_output, i)
            for i in range(bsz)
        ]

        attention_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=inputs_tensor.device
        ).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_full_attn_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        tokenizer_output=None,
        batch_gen_image_info=None,
        batch_cond_images=None,
        infer_align_image_size=False,
        generator=None,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids")
        # if `inputs_embeds` are passed, we only want to use them in the 1st
        # generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            assert position_ids is not None, "position_ids must be provided in kwargs."
            # in decode steps
            if input_ids is not None and input_ids.shape[1] != position_ids.shape[1]:
                input_ids = torch.gather(input_ids, dim=1, index=position_ids)
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "rope_image_info": kwargs["rope_image_info"],
                "mode": kwargs["mode"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timesteps": kwargs.get("timesteps"),
                "timesteps_index": kwargs.get("timesteps_index"),
                "timesteps_r": kwargs.get("timesteps_r"),
                "timesteps_r_index": kwargs.get("timesteps_r_index"),
                "guidance": kwargs.get("guidance"),
                "guidance_index": kwargs.get("guidance_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_timesteps": kwargs.get("cond_timesteps"),
                "cond_timesteps_index": kwargs.get("cond_timesteps_index"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "cond_vit_image_kwargs": kwargs.get("cond_vit_image_kwargs"),
                "cache_dic": kwargs.get("cache_dic"),
                "gen_timestep_scatter_index": kwargs.get("gen_timestep_scatter_index"),
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: "dict[str, Any]",
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> "dict[str, Any]":
        """Run after each forward step; updates kwargs for the next step."""
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "rope_image_info": model_kwargs["rope_image_info"],
        }

        # update past_key_values keeping its naming used in model code
        if getattr(outputs, "past_key_values", None) is not None:
            updated_model_kwargs["past_key_values"] = outputs.past_key_values

        if "tokenizer_output" in model_kwargs:
            # After prefill step
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a
                # real_pos to index the valid end position of the sequence. If
                # tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["position_ids"] = real_pos
            else:
                # inputs_pos
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = (
                    torch.arange(seq_len, device=image_mask.device)
                    .unsqueeze(0)
                    .repeat(bsz, 1)
                )
                position_ids = index.masked_select(image_mask.bool()).reshape(bsz, -1)
                timestep_position_ids = index[
                    torch.arange(bsz), model_kwargs["timesteps_index"][:, -1]
                ].unsqueeze(-1)
                pos_cat_list = [
                    timestep_position_ids,
                ]
                if self.config.cfg_distilled:
                    guidance_position_ids = index[
                        torch.arange(bsz), model_kwargs["guidance_index"][:, -1]
                    ].unsqueeze(-1)
                    pos_cat_list.append(guidance_position_ids)
                if self.config.use_meanflow:
                    timestep_r_position_ids = index[
                        torch.arange(bsz), model_kwargs["timesteps_r_index"][:, -1]
                    ].unsqueeze(-1)
                    pos_cat_list.append(timestep_r_position_ids)
                pos_cat_list.append(position_ids)
                updated_model_kwargs["position_ids"] = torch.cat(pos_cat_list, dim=1)

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                    model_kwargs["attention_mask"], updated_model_kwargs["position_ids"]
                ):
                    mask_list.append(
                        torch.index_select(
                            attention_mask_i, dim=1, index=position_ids_i.reshape(-1)
                        )
                    )
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs[
                    "gen_timestep_scatter_index"
                ]
        else:
            # After decode steps
            if mode == "gen_text":
                # Now we are in the decode steps.
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
                # Remove attention mask to use full attention of 1 x seqlen in
                # decode steps
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs[
                    "gen_timestep_scatter_index"
                ]
        return updated_model_kwargs

    def _get_ratio_index_from_token(self, ratio_token_id: int, tokenizer) -> int:
        if hasattr(tokenizer, "get_all_ratio_token_ids"):
            ratio_token_ids = tokenizer.get_all_ratio_token_ids()
            try:
                ratio_index = ratio_token_ids.index(ratio_token_id)
            except ValueError as exc:
                raise ValueError(f"Unknown ratio token id {ratio_token_id}") from exc
        else:
            ratio_index = ratio_token_id - tokenizer.ratio_token_id(0)
        if ratio_index < 0 or ratio_index >= len(self.image_processor.vae_reso_group):
            raise ValueError(
                f"ratio_index {ratio_index} out of range for vae_reso_group"
            )
        return ratio_index

    # ------------------------------------------------------------------
    # Native text generation loop (replaces HF GenerationMixin.generate)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        generator=None,
    ) -> torch.Tensor:
        """Greedy / temperature+top-k+top-p sampling (HF semantics)."""
        if not do_sample:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(int(top_k), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token
            # above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits.float(), dim=-1)
        if generator is not None:
            # The NPU ``aclnnMultinomial`` kernel does not accept a device
            # generator (and rejects non-fp32 inputs); sample on CPU where
            # the generator is honored, then move the indices back.
            probs = probs.cpu()
        samples = torch.multinomial(probs, num_samples=1, generator=generator)
        return samples.squeeze(-1).to(logits.device)

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        mode: str = "gen_text",
        rope_image_info=None,
        tokenizer_output=None,
        eos_token_id=None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        logits_processor=None,
        stage_transitions: Optional[list[tuple[int, list[int]]]] = None,
        final_stop_tokens: Optional[list[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Native AR text generation for the think / recaption / img_ratio
        stages, replacing HF ``GenerationMixin.generate``.

        Returns the full token sequence ``(batch, input_len + n_generated)``
        like the official ``generate(..., decode_text=False)``.
        """
        assert mode == "gen_text", f"generate_text supports mode gen_text, got {mode}"
        assert input_ids is not None, "`input_ids` must be provided."
        gen_config = self.generation_config
        do_sample = _default(do_sample, getattr(gen_config, "do_sample", True))
        top_k = _default(top_k, getattr(gen_config, "top_k", 1024))
        top_p = _default(top_p, getattr(gen_config, "top_p", 0.95))
        temperature = _default(temperature, getattr(gen_config, "temperature", 0.6))
        max_new_tokens = _default(
            max_new_tokens, getattr(gen_config, "max_new_tokens", 2048)
        )

        if stage_transitions is not None:
            if final_stop_tokens is None:
                raise ValueError(
                    "`final_stop_tokens` must be provided when `stage_transitions` is set."
                )
            if logits_processor is None:
                logits_processor = []
            else:
                logits_processor = list(logits_processor)
            logits_processor.append(
                StageTransitionLogitsProcessor(stage_transitions, input_ids.shape[0])
            )
            eos_token_id = final_stop_tokens

        stop_ids = []
        if eos_token_id is not None:
            stop_ids = (
                list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            )

        device = input_ids.device
        bsz, seq_len = input_ids.shape

        # real_pos is the first <pad> position, i.e. the position where the
        # first generated token will be written; with an unpadded sequence it
        # equals the sequence length.
        if tokenizer_output is not None and getattr(tokenizer_output, "real_pos", None) is not None:
            sample_positions = to_device(tokenizer_output.real_pos, device) - 1
        else:
            sample_positions = torch.full(
                (bsz,), seq_len - 1, dtype=torch.long, device=device
            )

        model_kwargs: "dict[str, Any]" = dict(kwargs)
        # Prefill attention: the official path builds the 4-D causal mask even
        # for gen_text (T2I has no full-attn slices in text stages, so this
        # is a pure causal tril); decode steps run without a mask.
        attention_mask = self._prepare_attention_mask_for_generation(
            input_ids, gen_config, {"tokenizer_output": tokenizer_output}
        )
        model_kwargs.update(
            position_ids=position_ids,
            past_key_values=past_key_values,
            mode=mode,
            rope_image_info=rope_image_info,
            tokenizer_output=tokenizer_output,
            attention_mask=attention_mask,
        )

        generators = kwargs.get("generator")
        if generators is not None and not isinstance(generators, (list, tuple)):
            generators = [generators] * bsz

        # Track whether a vocab-slice processor reduced the logits, so sampled
        # indices can be mapped back to full-vocab token ids.
        valid_token_ids = None

        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        for _ in range(int(max_new_tokens)):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ):
                outputs = self(**model_inputs)

            logits = outputs.logits  # (bsz, q_len, vocab)
            next_logits = logits[torch.arange(bsz, device=device), sample_positions, :]

            if logits_processor is not None:
                for processor in logits_processor:
                    next_logits = processor(input_ids, next_logits)

            if next_logits.shape[-1] != self.vocab_size:
                if valid_token_ids is None:
                    # Only SliceVocabLogitsProcessor shrinks the vocab; the
                    # reduced layout is [main slice, *other_slices].
                    valid_token_ids = []
                    for processor in logits_processor or []:
                        if isinstance(processor, SliceVocabLogitsProcessor):
                            valid_token_ids.extend(
                                range(processor.vocab_start, processor.vocab_end)
                            )
                            for start, end in processor.other_slices:
                                valid_token_ids.extend(range(start, end))
                    valid_token_ids = torch.tensor(
                        valid_token_ids, dtype=torch.long, device=device
                    )
                assert next_logits.shape[-1] == valid_token_ids.shape[0], (
                    f"Reduced logits dim {next_logits.shape[-1]} does not match "
                    f"vocab slice size {valid_token_ids.shape[0]}"
                )

            sample_idx = self._sample_next_token(
                next_logits,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                generator=generators[0] if generators is not None else None,
            )
            if valid_token_ids is not None:
                next_tokens = valid_token_ids[sample_idx]
            else:
                next_tokens = sample_idx
            # Keep pad for already-finished sequences.
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, self.pad_id), next_tokens
            )

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            # The next token is sampled from the position the just-appended
            # token was written to (== the next write position).
            sample_positions = model_kwargs["position_ids"].reshape(bsz)

            if stop_ids:
                finished = finished | torch.isin(next_tokens, torch.tensor(stop_ids, device=device))
                if bool(finished.all()):
                    break

        return input_ids
