from __future__ import annotations

import argparse
from pathlib import Path

from onwordly.experiments.arithmetic import ABLATION_REGIMES, run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Onwordly arithmetic mechanism ablation"
    )
    parser.add_argument(
        "--manifest",
        default="experiments/002-adaptive-ablation/manifest.json",
        help="Path to experiment manifest",
    )
    parser.add_argument(
        "--output",
        default="results/002-adaptive-ablation",
        help="Directory for frozen datasets and result JSON",
    )
    args = parser.parse_args()

    manifest = ArithmeticExperimentManifest.from_json(args.manifest)
    run_experiment(manifest, output_dir=Path(args.output), regimes=ABLATION_REGIMES)


if __name__ == "__main__":
    main()
