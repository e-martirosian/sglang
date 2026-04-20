from types import SimpleNamespace

from sglang.test.run_eval import run_eval


class TestMMLU:
    accuracy_mmlu = 0.00

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], self.accuracy_mmlu)
