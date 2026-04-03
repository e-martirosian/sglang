import os
import unittest

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import (
    AFM_4_5B_BASE_WEIGHTS_PATH,
    BAICHUAN2_13B_CHAT_WEIGHTS_PATH,
    CHATGLM2_6B_WEIGHTS_PATH,
    EXAONE_3_5_7_8B_INSTRUCT_WEIGHTS_PATH,
    GEMMA_3_4B_IT_WEIGHTS_PATH,
    GLM_4_9B_CHAT_WEIGHTS_PATH,
    GRANITE_3_0_3B_A800M_INSTRUCT_WEIGHTS_PATH,
    GRANITE_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    INTERNLM2_7B_WEIGHTS_PATH,
    LLAMA_2_7B_WEIGHTS_PATH,
    MIMO_7B_RL_WEIGHTS_PATH,
    MINICPM3_4B_WEIGHTS_PATH,
    MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH,
    PERSIMMON_8B_CHAT_WEIGHTS_PATH,
    PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_1_7B_GPTQ_INT8_WEIGHTS_PATH,
    SMOLLM_1_7B_WEIGHTS_PATH,
    STABLELM_2_1_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class Test1NPUMMMU(TestLMModels):
    dataset = "mmmu"
    task = "mmmu_val"
    batch_size = 32
    tp = 1
    llms_eval_acc_tag = "mmmu_acc,none"


class Test1NPUGSM8K(TestLMModels):
    dataset = "gsm8k"
    task = "gsm8k_val"
    batch_size = 32
    tp = 1
    llms_eval_args = ["--num-fewshot", "5"]


class TestLlama3211BVisionInstruct(Test1NPUMMMU, CustomTestCase):
    model = (
        "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-11B-Vision-Instruct"
    )
    accuracy = 0.2
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]


class TestAFM(Test1NPUGSM8K, CustomTestCase):
    model = AFM_4_5B_BASE_WEIGHTS_PATH
    accuracy = 0.375


class TestBaichuan(Test1NPUGSM8K, CustomTestCase):
    model = BAICHUAN2_13B_CHAT_WEIGHTS_PATH
    accuracy = 0.48
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--max-running-requests",
        "128",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
    ]
    llms_eval_args = ["--num-fewshots", "1"]


class TestChatGlm2(Test1NPUGSM8K, CustomTestCase):
    model = CHATGLM2_6B_WEIGHTS_PATH
    accuracy = 0.25
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
    ]


class TestEXAONE(Test1NPUGSM8K, CustomTestCase):
    model = EXAONE_3_5_7_8B_INSTRUCT_WEIGHTS_PATH
    # Allow 1% tolerance for the accuracy threshold
    accuracy = round(0.8 * 0.99, 3)
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
    ]


class TestGemma34B(Test1NPUGSM8K, CustomTestCase):
    model = GEMMA_3_4B_IT_WEIGHTS_PATH
    accuracy = 0.7
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
    ]


class TestGLM49BChat(Test1NPUGSM8K, CustomTestCase):
    model = GLM_4_9B_CHAT_WEIGHTS_PATH
    accuracy = 0.77


class TestGranite(Test1NPUGSM8K, CustomTestCase):
    model = GRANITE_3_0_3B_A800M_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.38


class TestGranite(Test1NPUGSM8K, CustomTestCase):
    model = GRANITE_3_1_8B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.695


class TestInternlm2(Test1NPUGSM8K, CustomTestCase):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    model = INTERNLM2_7B_WEIGHTS_PATH
    accuracy = 0.585


class TestLlama(Test1NPUGSM8K, CustomTestCase):
    model = LLAMA_2_7B_WEIGHTS_PATH
    accuracy = 0.18


class TestMiMo7BRL(Test1NPUGSM8K, CustomTestCase):
    model = MIMO_7B_RL_WEIGHTS_PATH
    accuracy = 0.75


class TestMiniCPM3(Test1NPUGSM8K, CustomTestCase):
    model = MINICPM3_4B_WEIGHTS_PATH
    accuracy = 0.69
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "128",
        "--chunked-prefill-size",
        "-1",
    ]


class TestMistral7B(Test1NPUGSM8K, CustomTestCase):
    model = MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
    accuracy = 0.375


class TestPersimmon8BChat(Test1NPUGSM8K, CustomTestCase):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    model = PERSIMMON_8B_CHAT_WEIGHTS_PATH
    accuracy = 0.17


class TestPhi4(Test1NPUGSM8K, CustomTestCase):
    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.8


class TestQwen306B(Test1NPUGSM8K, CustomTestCase):
    model = QWEN3_0_6B_WEIGHTS_PATH
    accuracy = 0.38
    server_args = [
        "--chunked-prefill-size",
        256,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


class TestQwen317BGPTQInt8(Test1NPUGSM8K, CustomTestCase):
    model = QWEN3_1_7B_GPTQ_INT8_WEIGHTS_PATH
    accuracy = 0.65
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--quantization",
        "gptq",
    ]


class TestSmolLM(Test1NPUGSM8K, CustomTestCase):
    model = SMOLLM_1_7B_WEIGHTS_PATH
    accuracy = 0.05
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
    ]


class TestStablelm(Test1NPUGSM8K, CustomTestCase):
    model = STABLELM_2_1_6B_WEIGHTS_PATH
    accuracy = 0.195
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        1,
        "--enable-torch-compile",
    ]


if __name__ == "__main__":
    unittest.main()
