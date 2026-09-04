"""Resolution-contract tests for HunyuanImage-3 image generation and editing."""

from types import SimpleNamespace

from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_image3 import (
    HunyuanImage3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    resolve_hunyuan_image3_output_resolution,
)


def test_edit_without_explicit_size_follows_reference_aspect_ratio():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1280,
        height=720,
        explicit_fields=set(),
        reference_size=(1000, 333),
    )

    assert (width, height) == (1008, 336)


def test_explicit_generation_size_takes_precedence_over_reference_image():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields={"width", "height"},
        reference_size=(1000, 333),
    )

    assert (width, height) == (1040, 592)


def test_one_explicit_dimension_prevents_reference_size_override():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=640,
        height=720,
        explicit_fields={"width"},
        reference_size=(333, 1000),
    )

    assert (width, height) == (640, 720)


def test_text_to_image_without_reference_uses_aligned_request_size():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields=set(),
    )

    assert (width, height) == (1040, 592)


def test_pipeline_config_delegates_condition_image_sizing_to_native_stage():
    config = HunyuanImage3PipelineConfig()
    reference = Image.new("RGB", (1000, 333))

    assert config.calculate_condition_image_size(reference, 1280, 720) is None
    assert config.prepare_calculated_size(reference) is None


def test_input_validation_does_not_apply_generic_32_pixel_edit_resize():
    reference = Image.new("RGB", (1000, 333))
    batch = Req(
        sampling_params=SamplingParams(prompt="edit", width=1008, height=336),
        condition_image=reference,
    )
    batch.extra["explicit_fields"] = ["width", "height"]
    server_args = SimpleNamespace(pipeline_config=HunyuanImage3PipelineConfig())

    InputValidationStage().preprocess_condition_image(
        batch,
        server_args,
        condition_image_width=reference.width,
        condition_image_height=reference.height,
    )

    assert batch.condition_image[0].size == (1000, 333)
    assert (batch.width, batch.height) == (1008, 336)
