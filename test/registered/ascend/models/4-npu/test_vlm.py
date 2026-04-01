import unittest
from types import SimpleNamespace

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import JANUS_PRO_1B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

COMMON_SERVER_ARGS = [
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
        ]

COMMON_MODEL_ARGS = {
    "tp": 4,
    "task": "mmmu_val",
    "dataset": "mmmu",
}

MODELS = [
    SimpleNamespace(
        model_name=JANUS_PRO_1B_WEIGHTS_PATH,
        accuracy=0.2,
        out_throughput=62,
        llms_eval_args=[],
        bench_serving_args=["--num-prompts", "2"],
        server_args=COMMON_SERVER_ARGS,
        **COMMON_MODEL_ARGS
    )
]


class TestVLMModels(TestLMModels):
    models = MODELS

    def test_vlm_mmmu_benchmakr(self):
        for model in self.model_names:
            self._run_vlm_mmmu_test(model, "./logs")

if __name__ == "__main__":
    unittest.main()
