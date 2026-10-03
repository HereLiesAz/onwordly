"""CPU end-to-end smoke of every experiment through the real Hugging Face adapter.

Uses each experiment's smoke manifest with a tiny model (default
``sshleifer/tiny-gpt2``) so the adapter, token accounting, serialization,
reports and run-plan dispatch are exercised without a GPU. Nothing is learned;
accuracy figures from this script are meaningless by design.

    python scripts/smoke_all.py --out /tmp/onwordly-smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from onwordly.experiments.kaggle import KaggleRunPlan, execute_run_plan

SMOKE_MANIFESTS = {
    "001": "experiments/001-arithmetic-curriculum/smoke-manifest.json",
    # 002 shares 001's arithmetic manifest schema and has no smoke manifest of its own.
    "002": "experiments/001-arithmetic-curriculum/smoke-manifest.json",
    "003": "experiments/003-symbolic-transformations/smoke-manifest.json",
    "004": "experiments/004-string-manipulation/smoke-manifest.json",
    "005": "experiments/005-program-execution/smoke-manifest.json",
    "006": "experiments/006-formal-logic/smoke-manifest.json",
    "007": "experiments/007-program-process-supervision/smoke-manifest.json",
}
SUITE_EXPERIMENTS = {"001", "003", "004", "005", "006"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--experiments", nargs="*", default=sorted(SMOKE_MANIFESTS))
    parser.add_argument("--suite-seeds", default="3303,4404")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in args.suite_seeds.split(","))
    outcomes: dict[str, dict[str, object]] = {}

    for experiment in args.experiments:
        manifest = json.loads(Path(SMOKE_MANIFESTS[experiment]).read_text())
        manifest["model_name"] = args.model
        manifest_path = root / f"manifest-{experiment}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        modes = ["single"] + (["suite"] if experiment in SUITE_EXPERIMENTS else [])
        for mode in modes:
            key = f"{experiment}/{mode}"
            started = time.perf_counter()
            try:
                plan = KaggleRunPlan(
                    experiment=experiment,
                    mode=mode,
                    manifest=str(manifest_path),
                    seeds=seeds,
                )
                output = execute_run_plan(plan, root / mode)
                outcomes[key] = {"ok": True, "output": str(output)}
            except Exception as exc:  # noqa: BLE001 - smoke must report every failure
                outcomes[key] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            outcomes[key]["seconds"] = round(time.perf_counter() - started, 1)
            print(key, "ok" if outcomes[key]["ok"] else outcomes[key]["error"], flush=True)

    (root / "smoke.json").write_text(json.dumps(outcomes, indent=2))
    return 0 if all(value["ok"] for value in outcomes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
