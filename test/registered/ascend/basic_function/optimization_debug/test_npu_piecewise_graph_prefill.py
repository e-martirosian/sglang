import unittest

from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    run_bench_one_batch,
)

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


TOKENS_TO_CAPTURE = [i for i in range(128, 4096, 128)]


class TestPiecewiseGraphPrefillCorrectness(GSM8KAscendMixin, CustomTestCase):
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--cuda-graph-bs",
        128,
        "--enforce-piecewise-cuda-graph",
        "--piecewise-cuda-graph-tokens",
        *TOKENS_TO_CAPTURE,
    ]
    model = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.84
    num_questions = 1319


class TestPiecewiseGraphPrefillBenchmark(CustomTestCase):
    latency = 0.045
    model = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH

    def test_latency(self):
        print(f"##=== Testing prefill latency: {self.model} ===##")
        prefill_latency, _, _ = run_bench_one_batch(
            self.model,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--enforce-piecewise-cuda-graph",
                "--piecewise-cuda-graph-tokens",
            ]
            + TOKENS_TO_CAPTURE,
        )
        self.assertLess(prefill_latency, self.latency)


if __name__ == "__main__":
    unittest.main()
