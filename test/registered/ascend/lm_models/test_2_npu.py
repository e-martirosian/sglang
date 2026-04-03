import unittest

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import (
    C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH,
    C4AI_COMMAND_R_V01_WEIGHTS_PATH,
    LING_LITE_WEIGHTS_PATH,
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
    QWEN3_30B_MODELSLIM_INT4_WEIGHTS_PATH,
    QWQ_32B_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class Test2NPUGSM8K(TestLMModels):
    dataset = "gsm8k"
    task = "gsm8k_val"
    batch_size = 32
    tp = 2
    llms_eval_args = ["--num-fewshot", "5"]


class TestC4AI(Test2NPUGSM8K, CustomTestCase):
    model = C4AI_COMMAND_R_V01_WEIGHTS_PATH
    accuracy = 0.55
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chat-template",
        C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH,
        "--tp-size",
        "2",
        "--dtype",
        "bfloat16",
    ]


class TestLingLite(Test2NPUGSM8K, CustomTestCase):
    model = LING_LITE_WEIGHTS_PATH
    accuracy = 0.75
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
    ]


class TestQwen317BGPTQInt8(Test2NPUGSM8K, CustomTestCase):
    model = QWEN3_30B_MODELSLIM_INT4_WEIGHTS_PATH
    accuracy = 0.85
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
    ]


class TestQwen330B(Test2NPUGSM8K, CustomTestCase):
    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
    ]


class TestQWQ32BW8A8(Test2NPUGSM8K, CustomTestCase):
    model = QWQ_32B_W8A8_WEIGHTS_PATH
    accuracy = 0.59
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
        "--quantization",
        "modelslim",
    ]


if __name__ == "__main__":
    unittest.main()
