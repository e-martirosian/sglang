"""HiDream-O1-Image generation stage: preprocessing, denoising, and decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Constants
TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
T_EPS = 0.001

PREDEFINED_RESOLUTIONS = [
    (2048, 2048), (2304, 1728), (1728, 2304), (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496), (3104, 1312), (1312, 3104), (2304, 1792),
    (1792, 2304),
]

DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
    737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]


# ---------------------------------------------------------------------------
# Flow matching scheduler (simplified UniPC-style)
# ---------------------------------------------------------------------------

class FlowMatchScheduler:
    """Simple flow matching scheduler for HiDream denoising."""

    def __init__(self, shift: float = 3.0, use_dynamic_shifting: bool = False):
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set up the timestep schedule."""
        timesteps = torch.tensor(DEFAULT_TIMESTEPS[:num_inference_steps],
                                 device=device, dtype=torch.long)
        self.timesteps = timesteps
        sigmas = [t.item() / 1000.0 for t in timesteps]
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device)

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising step using flow matching Euler method."""
        # Find current step index
        step_idx = (self.timesteps == timestep.long().item()).nonzero(as_tuple=True)[0][0].item()
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1] if step_idx + 1 < len(self.sigmas) else 0.0

        # Flow matching: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_pred
        # model_output is -v (negative velocity), so v = -model_output
        dt = sigma_next - (sigma.item() if isinstance(sigma, torch.Tensor) else sigma)
        result = sample + dt * (-model_output)
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    """Find the closest predefined resolution."""
    img_ratio = width / height
    best_res = None
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        ratio = w / h
        diff = abs(ratio - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    return best_res


def get_rope_index_fix_point(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    skip_vision_start_token: list[int] | None = None,
    fix_point: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D RoPE position IDs for the HiDream model.

    Ported from the official HiDream-O1-Image utils.py.
    """
    if skip_vision_start_token is None:
        skip_vision_start_token = [1]

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens and remain_images > 0 else len(input_tokens) + 1
                ed_video = input_tokens.index(video_token_id, st) if video_token_id in input_tokens and remain_videos > 0 else len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


def build_t2i_text_sample(
    prompt: str, height: int, width: int,
    tokenizer: Any, processor: Any, model_config: Any,
) -> dict[str, torch.Tensor]:
    """Build text sample for text-to-image generation."""
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


def decode_patches_to_image(
    z: torch.Tensor, h_patches: int, w_patches: int, patch_size: int = PATCH_SIZE,
) -> Image.Image:
    """Convert patch tensor to PIL image."""
    img = (z.float() + 1) / 2
    img = img.reshape(1, h_patches, w_patches, 3, patch_size, patch_size)
    img = img.permute(0, 3, 1, 4, 2, 5).reshape(1, 3, h_patches * patch_size, w_patches * patch_size)
    arr = np.round(np.clip(img[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


# ---------------------------------------------------------------------------
# Generation stage
# ---------------------------------------------------------------------------

class HiDreamGenerationStage(PipelineStage):
    """Full generation stage: preprocessing -> denoising -> decoding."""

    def __init__(self, model: torch.nn.Module, processor: Any):
        super().__init__()
        self.model = model
        self.processor = processor

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        device = next(self.model.parameters()).device
        dtype = torch.bfloat16

        # Extract parameters from batch
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        heights = batch.height if isinstance(batch.height, list) else [batch.height]
        widths = batch.width if isinstance(batch.width, list) else [batch.width]
        num_steps = int(batch.num_inference_steps) if hasattr(batch, 'num_inference_steps') else 28
        guidance_scale = float(batch.guidance_scale) if hasattr(batch, 'guidance_scale') else 5.0
        shift = float(batch.shift) if hasattr(batch, 'shift') else 3.0
        seeds = batch.seed if isinstance(batch.seed, list) else [batch.seed]

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        model_config = self.model.config  # HiDreamArchConfig

        results = []
        for prompt, height, width, seed in zip(prompts, heights, widths, seeds):
            height, width = int(height), int(width)

            # Snap to closest predefined resolution
            best_w, best_h = find_closest_resolution(width, height)
            if best_w != width or best_h != height:
                logger.info("Resolution snapped from %dx%d to %dx%d", width, height, best_w, best_h)
                width, height = best_w, best_h

            h_patches = height // PATCH_SIZE
            w_patches = width // PATCH_SIZE

            # Build text samples
            cond_sample = build_t2i_text_sample(
                prompt, height, width, tokenizer, self.processor, model_config
            )
            uncond_sample = None
            if guidance_scale > 1.0:
                uncond_sample = build_t2i_text_sample(
                    " ", height, width, tokenizer, self.processor, model_config
                )

            # Move to device
            def to_device(s):
                return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}

            cond_sample = to_device(cond_sample)
            if uncond_sample is not None:
                uncond_sample = to_device(uncond_sample)

            # Initialize noise
            torch.manual_seed(int(seed) + 1)
            noise = 8.0 * torch.randn(
                (1, 3, height, width),
                generator=torch.Generator("cpu").manual_seed(int(seed) + 1),
            ).to(device, dtype)
            z = noise.reshape(1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
            z = z.permute(0, 2, 4, 1, 3, 5).reshape(1, h_patches * w_patches, -1)

            # Build scheduler
            sched = FlowMatchScheduler(shift=shift)
            sched.set_timesteps(num_steps, device)

            tgt_image_len = h_patches * w_patches

            # Denoising loop
            for step_idx, step_t in enumerate(sched.timesteps):
                t_pixeldit = 1.0 - step_t.float() / 1000.0
                sigma = (step_t.float() / 1000.0).clamp_min(T_EPS)

                # Conditional forward
                with torch.autocast(device_type=device.type if isinstance(device.type, str) else "cpu",
                                    dtype=dtype, cache_enabled=False):
                    x_pred_cond = self._forward_once(
                        cond_sample, z.clone(), t_pixeldit.to(device),
                    )
                v_cond = (x_pred_cond.float() - z.float()) / sigma

                if uncond_sample is not None:
                    with torch.autocast(device_type=device.type if isinstance(device.type, str) else "cpu",
                                        dtype=dtype, cache_enabled=False):
                        x_pred_uncond = self._forward_once(
                            uncond_sample, z.clone(), t_pixeldit.to(device),
                        )
                    v_uncond = (x_pred_uncond.float() - z.float()) / sigma
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond

                model_output = -v_guided
                z = sched.step(model_output.float(), step_t.to(dtype=torch.float32), z.float()).to(dtype)

            # Decode to image
            img = decode_patches_to_image(z, h_patches, w_patches)
            results.append(img)

        # Return result
        output = OutputBatch()
        output.images = results
        return output

    def _forward_once(
        self, sample: dict, z_in: torch.Tensor, t_pixeldit: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass through the model."""
        device = z_in.device
        outputs = self.model(
            input_ids=sample["input_ids"],
            position_ids=sample["position_ids"],
            vinputs=z_in,
            timestep=t_pixeldit.reshape(-1).to(device),
            token_types=sample["token_types"],
        )
        x_pred = outputs["x_pred"]
        vinput_mask = sample["vinput_mask"]
        return x_pred[0, vinput_mask[0]].unsqueeze(0)
