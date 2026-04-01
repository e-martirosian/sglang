import unittest
from types import SimpleNameSpace

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import JANUS_PRO_1B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

MODELS = [
    SimpleNameSpace(
        model=JANUS_PRO_1B_WEIGHTS_PATH,
        mmmu_accuracy=0.2,
        mmmu_perf=62,
        task="mmmu_val",
        dataset="mmmu",
        server=SimpleNameSpace(tp=4),
        llms_eval_args=[],
        bench_serving_args=["--num-prompts", "2"],
        server_args=[
            "--tp-size",
            "4",
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "32",
            "--enable-multimodal",
            "--log-level",
            "info",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ],
    )
]


class TestVLMModels(TestLMModels):
    models = MODELS


if __name__ == "__main__":
    unittest.main()
