from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
from sglang.test.run_eval import run_eval


class TestMMLU:
    accuracy_mmlu = 0.00

    def test_mmlu(self):
        model_metrics = {
            "params": self.other_args,
            "accuracy": "-",
            "accuracy_threshold": self.accuracy_mmlu,
            "output_throughput": "-",
            "output_throughput_threshold": "N/A",
            "latency": "-",
            "latency_threshold": "N/A",
        }

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmlu",
                num_examples=128,
                num_threads=32,
            )
            print("Starting mmlu test...")
            metrics = run_eval(args)
            model_metrics["accuracy"] = metrics["score"]
            self.assertGreater(metrics["score"], self.accuracy_mmlu)
        except Exception as e:
            model_metrics["error"] = e
            self.fail(f"Test failed for {self.model}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})
