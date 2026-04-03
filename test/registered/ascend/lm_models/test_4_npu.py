import unittest

from sglang.test.ascend.lm_utils import TestLMModels
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_VL2_WEIGHTS_PATH,
    GEMMA_3_4B_IT_WEIGHTS_PATH,
    JANUS_PRO_1B_WEIGHTS_PATH,
    JANUS_PRO_7B_WEIGHTS_PATH,
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
    MIMO_VL_7B_RL_WEIGHTS_PATH,
    MINICPM_O_2_6_WEIGHTS_PATH,
    MINICPM_V_2_6_WEIGHTS_PATH,
    MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_WEIGHTS_PATH,
    PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_32B_WEIGHTS_PATH,
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class Test4NPUMMMU(TestLMModels):
    dataset = "mmmu"
    task = "mmmu_val"
    batch_size = 32
    tp = 4
    llms_eval_acc_tag = "mmmu_acc,none"


class Test4NPUGSM8K(TestLMModels):
    dataset = "gsm8k"
    task = "gsm8k_val"
    batch_size = 32
    tp = 4
    llms_eval_args = ["--num-fewshot", "5"]


class TestDeepseekVl2(Test4NPUMMMU, CustomTestCase):
    model = DEEPSEEK_VL2_WEIGHTS_PATH
    accuracy = 0.2


class TestGemma34bModels(Test4NPUMMMU, CustomTestCase):
    model = GEMMA_3_4B_IT_WEIGHTS_PATH
    accuracy = 0.2


class TestJanusPro1B(Test4NPUMMMU, CustomTestCase):
    model = JANUS_PRO_1B_WEIGHTS_PATH
    accuracy = 0.2


class TestJanusPro7B(Test4NPUMMMU, CustomTestCase):
    model = JANUS_PRO_7B_WEIGHTS_PATH
    accuracy = 0.2


class TestMiMoModels(Test4NPUMMMU, CustomTestCase):
    model = MIMO_VL_7B_RL_WEIGHTS_PATH
    accuracy = 0.2


class TestMiniCPMModelsO(Test4NPUMMMU, CustomTestCase):
    model = MINICPM_O_2_6_WEIGHTS_PATH
    accuracy = 0.2


class TestMiniCPMModelsV(Test4NPUMMMU, CustomTestCase):
    model = MINICPM_V_2_6_WEIGHTS_PATH
    accuracy = 0.2


class TestMistralModels(Test4NPUMMMU, CustomTestCase):
    model = MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_WEIGHTS_PATH
    accuracy = 0.2


class TestPhi4Multimodal(Test4NPUMMMU, CustomTestCase):
    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2


class TestQwen25VL3B(Test4NPUMMMU, CustomTestCase):
    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2


class TestQwen3VL4B(Test4NPUMMMU, CustomTestCase):
    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2


class TestQwen3VL8B(Test4NPUMMMU, CustomTestCase):
    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2


class TestQwen3VL30BA3B(Test4NPUMMMU, CustomTestCase):
    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.2


class TestKimiVLA3BInstruct(Test4NPUGSM8K, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/Kimi/Kimi-VL-A3B-Instruct"
    accuracy = 0.66
    server_args = [
        "--trust-remote-code",
        "--max-running-requests",
        2048,
        "--mem-fraction-static",
        0.7,
        "--attention-backend",
        "ascend",
        "--tp-size",
        "4",
        "--disable-cuda-graph",
    ]


class TestLlama4(Test4NPUGSM8K, CustomTestCase):
    model = LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.9
    server_args = [
        "--chat-template",
        "llama-4",
        "--tp-size",
        4,
        "--mem-fraction-static",
        "0.9",
        "--context-length",
        "8192",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]


class TestQwen332B(Test4NPUGSM8K, CustomTestCase):
    model = QWEN3_32B_WEIGHTS_PATH
    accuracy = 0.86
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
