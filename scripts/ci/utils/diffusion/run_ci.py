import subprocess
from datetime import datetime
from sglang.multimodal_gen.test.run_suite import SUITES
import os

def run_suit(cmd, log_path):
    with open(log_path, "a", encoding="utf-8") as log:
        print(f"\n\n===== RUN START {datetime.now()} =====\n\n")

        print(f"\n--- Command {cmd} ---\n")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                text=True,
                capture_output=True
            )

            if result.returncode != 0:
                print(f"Testing failes. See log file {log_path}")
            else:
                print("Test passed.")
            log.write("STDOUT:\n")
            log.write(result.stdout if result.stdout else "[No output]\n")

            log.write("\nSTDERR:\n")
            log.write(result.stderr if result.stderr else "[No errors]\n")

            log.write(f"\nReturn code: {result.returncode}\n")

        except Exception as e:
            print(f"ERROR running command: {e}\n")
            log.write(f"ERROR running command: {e}\n")

        print(f"\n===== RUN END {datetime.now()} =====\n")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    for suit in SUITES.keys():
        cmd = f"python -m sglang.multimodal_gen.test.run_suite_npu --suite {suit} --continue-on-error"
        run_suit(cmd, f"logs/log_{suit}.txt")
    print(f"Execution complete.")
