from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import ExperimentConfig 
from experiment_runner import run_experiment
from sensitivity_analysis import run_all_sweeps


def main() -> None:
    config = ExperimentConfig()
    run_experiment(config)
    run_all_sweeps(config.num_runs, base_config=config)


if __name__ == "__main__":
    main()
