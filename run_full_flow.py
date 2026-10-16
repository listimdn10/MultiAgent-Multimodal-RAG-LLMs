import subprocess
import sys
import time


SCRIPTS = [
    "invoke.py",
    "crew_run.py",
    "consensus_agent.py",
]


def run_script(script_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start_time
        print(f"\nFailed: {script_name} (exit code {exc.returncode}) after {elapsed:.2f}s")
        raise

    elapsed = time.time() - start_time
    print(f"\nCompleted: {script_name} in {elapsed:.2f}s")


def main() -> None:
    total_start = time.time()

    for script_name in SCRIPTS:
        run_script(script_name)

    total_elapsed = time.time() - total_start
    print(f"\nFull flow completed in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
