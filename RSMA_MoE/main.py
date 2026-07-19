from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import ExperimentConfig
from experiment_runner import run_experiment


def main() -> None:
    run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
