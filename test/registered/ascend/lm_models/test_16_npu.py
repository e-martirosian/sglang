import unittest

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH,
    QWEN3_VL_235B_A22B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class Test16NPUMMMU(TestLMModels):
    dataset = "mmmu"
    task = "mmmu_val"
    batch_size = 32
    tp = 16
    llms_eval_acc_tag = "mmmu_acc,none"


class Test16NPUGSM8K(TestLMModels):
    dataset = "gsm8k"
    task = "gsm8k_val"
    batch_size = 32
    tp = 16
    llms_eval_args = ["--num-fewshot", "5"]


class TestQwen3VL235BA22B(Test16NPUMMMU, CustomTestCase):
    model = QWEN3_VL_235B_A22B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2
    server_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        16,
    ]
    time_out = 3000


class TestDeepSeekV32(Test16NPUGSM8K, CustomTestCase):
    model = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    accuracy = 0.5
    time_out = 3000
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
    ]


class TestGrok2(Test16NPUGSM8K, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/huihui-ai/grok-2"
    accuracy = 0.91
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--tokenizer-path",
        "/root/.cache/modelscope/hub/models/huihui-ai/grok-2/tokenizer.tok.json",
        "--tp-size",
        "16",
    ]


class TestQwen3Coder480BA35B(Test16NPUGSM8K, CustomTestCase):
    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH
    accuracy = 0.94
    time_out = 3000
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
    ]


if __name__ == "__main__":
    unittest.main()
