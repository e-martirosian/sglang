import unittest

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import (
    MINIMAX_M2_WEIGHTS_PATH,
    QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_235B_A22B_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class Test8NPUMMMU(TestLMModels):
    dataset = "mmmu"
    task = "mmmu_val"
    batch_size = 32
    tp = 8
    llms_eval_acc_tag = "mmmu_acc,none"


class Test8NPUGSM8K(TestLMModels):
    dataset = "gsm8k"
    task = "gsm8k_val"
    batch_size = 32
    tp = 8
    llms_eval_args = ["--num-fewshot", "5"]


class TestGLM4Models(Test8NPUMMMU, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-4.5V"
    accuracy = 0.2
    server_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.7,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        8,
    ]


class TestQwen25VL72B(Test8NPUMMMU, CustomTestCase):
    model = QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2
    server_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.6,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        8,
    ]


class TestDbrx(Test8NPUGSM8K, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/AI-ModelScope/dbrx-instruct"
    accuracy = 0.735
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
    ]


class TestMiniMaxM2(Test8NPUGSM8K, CustomTestCase):
    model = MINIMAX_M2_WEIGHTS_PATH
    accuracy = 0.9
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
    ]


class TestQwen3235BA22BW8A8(Test8NPUGSM8K, CustomTestCase):
    model = QWEN3_235B_A22B_W8A8_WEIGHTS_PATH
    accuracy = 0.955
    timeout_for_server_launch = 3000
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
        "--quantization",
        "modelslim",
    ]


if __name__ == "__main__":
    unittest.main()
