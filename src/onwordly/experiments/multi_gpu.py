"""Run independent arithmetic regimes in parallel, one process per GPU.

Regimes share nothing (each loads a fresh model), so they can run on separate
GPUs at once. Each worker process sees exactly one GPU via CUDA_VISIBLE_DEVICES
and writes its regimes' result files; the parent then calls ``run_experiment``
for every regime, which loads those files (regime-level resume) and writes the
combined summary. With fewer than two GPUs this falls back to a single process.

Per-regime wall time and peak memory stay comparable because every worker has a
whole GPU to itself; workers do share the host CPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from onwordly.experiments.arithmetic import DEFAULT_REGIMES, run_experiment
from onwordly.experiments.manifest import ArithmeticExperimentManifest


def gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def assign_regimes(regimes: Sequence[str], devices: int) -> list[list[str]]:
    """Round-robin regimes over devices, preserving each device's order."""
    buckets: list[list[str]] = [[] for _ in range(devices)]
    for index, regime in enumerate(regimes):
        buckets[index % devices].append(regime)
    return [bucket for bucket in buckets if bucket]


def run_parallel(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    devices: int | None = None,
) -> dict[str, object]:
    manifest = ArithmeticExperimentManifest.from_json(manifest_path)
    available = gpu_count() if devices is None else devices
    if available >= 2:
        workers = []
        for device, subset in enumerate(assign_regimes(regimes, available)):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
            command = [
                sys.executable, "-m", "onwordly.experiments.multi_gpu",
                "--manifest", str(manifest_path),
                "--output", str(output_dir),
                "--regimes", *subset,
            ]
            print(f"GPU {device}: {', '.join(subset)}", flush=True)
            workers.append((device, subset, subprocess.Popen(command, env=env)))
        failed = [(device, subset) for device, subset, process in workers if process.wait() != 0]
        if failed:
            raise RuntimeError(f"regime workers failed: {failed}")
    # Loads every finished regime from disk; runs anything still missing here.
    return run_experiment(manifest, output_dir=output_dir, regimes=regimes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a subset of arithmetic regimes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regimes", nargs="+", required=True)
    args = parser.parse_args(argv)
    manifest = ArithmeticExperimentManifest.from_json(args.manifest)
    run_experiment(manifest, output_dir=args.output, regimes=args.regimes)


if __name__ == "__main__":
    main()
