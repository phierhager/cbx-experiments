from cbx_experiments.runner import Runner
from cbx_experiments.experiment_generation import create_experiment_config
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "config",
        type=str,
        help="The path to the YAML configuration file.",
        default=r".\config\default.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_config = create_experiment_config(args.file_path)
    runner = Runner(experiment_config)
    experiment_result = runner.run_experiment()
