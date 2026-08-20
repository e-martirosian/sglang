import inspect
import time
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
    align_hunyuan_image3_resolution,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
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
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    get_module_dtype,
)

logger = init_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    accepts_timesteps = "timesteps" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )
    accepts_sigmas = "sigmas" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )

    if timesteps is not None and sigmas is not None:
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"Scheduler {scheduler.__class__} does not support custom timesteps or sigmas."
            )
        scheduler.set_timesteps(
            timesteps=timesteps, sigmas=sigmas, device=device, **kwargs
        )
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    return timesteps, num_inference_steps


class HunyuanImage3AR(PipelineStage):
    """
    AR (autoregressive) stage for HunyuanImage-3 token generation.

    This stage handles the AR model that generates text and image tokens.

    Args:
        processor: Processor for the AR model.
        vision_language_encoder: The AR model that generates image tokens.
    """

    def __init__(
        self,
        processor,
        vision_language_encoder,
    ):
        super().__init__()
        self.processor = processor
        self.vision_language_encoder = vision_language_encoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if not isinstance(self.vision_language_encoder, torch.nn.Module):
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vision_language_encoder",
                memory_intensive=True,
            )
        ]

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    async def process_batch(self, batch: Req, server_args: ServerArgs) -> Req:
        """Process a single batch through AR generation."""
        # TODO: Implement AR token generation
        # This is a placeholder - actual implementation requires:
        # 1. Tokenize prompt using custom tokenizer
        # 2. Run AR model to generate text + image tokens
        # 3. Extract image tokens for denoising
        logger.warning("HunyuanImage3AR.process_batch is not yet implemented")
        return batch

    async def __call__(self, batch_iterator: Iterator[Req], server_args: ServerArgs):
        """Process batches through AR generation."""
        async for batch in batch_iterator:
            batch = await self.process_batch(batch, server_args)
            yield batch


class HunyuanImage3BeforeDenoisingStage(PipelineStage):
    """
    Pre-denoising stage for HunyuanImage-3.

    Prepares latents, conditioning, and scheduler timesteps before the denoising loop.
    """

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler,
    ):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        uses = []
        if isinstance(self.vae, torch.nn.Module):
            uses.append(
                ComponentUse(
                    self._component_stage_name(stage_name),
                    "vae",
                    memory_intensive=True,
                )
            )
        if isinstance(self.text_encoder, torch.nn.Module):
            uses.append(
                ComponentUse(
                    self._component_stage_name(stage_name),
                    "text_encoder",
                    memory_intensive=True,
                )
            )
        return uses

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare initial latents for denoising."""
        # Calculate latent dimensions
        vae_scale_factor = 16  # HunyuanImage-3 VAE scale factor
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        shape = (
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        latents = randn_tensor(
            shape=shape, generator=generator, device=device, dtype=dtype
        )
        return latents

    async def process_batch(self, batch: Req, server_args: ServerArgs) -> Req:
        """Process a single batch before denoising."""
        device = get_local_torch_device()
        dtype = get_module_dtype(self.transformer)

        # Get dimensions
        height = batch.height
        width = batch.width

        # Align dimensions
        height, width = align_hunyuan_image3_resolution(height, width)
        batch.height = height
        batch.width = width

        # Prepare latents
        num_channels_latents = 32  # HunyuanImage-3 latent channels
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=batch.generator,
        )
        batch.latents = latents

        # Prepare timesteps
        num_inference_steps = batch.num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
        )
        batch.timesteps = timesteps
        batch.num_inference_steps = num_inference_steps

        return batch

    async def __call__(self, batch_iterator: Iterator[Req], server_args: ServerArgs):
        """Process batches before denoising."""
        async for batch in batch_iterator:
            batch = await self.process_batch(batch, server_args)
            yield batch
