"""Publish finished results to GitHub straight from a Kaggle or Colab run.

Copies the results tree (minus bulky frozen datasets and archives) into
``results/kaggle/<stamp>/`` on a new branch ``kaggle-results/<stamp>`` and
pushes it. Main is never written; open a PR from the printed link.

The token needs contents:write on the repository. On Kaggle store it as a
secret named ``GITHUB_TOKEN`` (Add-ons → Secrets); it is read at call time and
never printed or written to disk outside the temporary clone's push URL.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SKIP_SUFFIXES = (".jsonl", ".tar.gz", ".zip", ".partial")


def kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - missing secret is reported by the caller
        return None


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def publish_results(
    results_dir: str | Path,
    *,
    repository: str = "HereLiesAz/onwordly",
    token: str | None = None,
    remote_url: str | None = None,
    stamp: str | None = None,
    label: str = "",
    base: str = "main",
) -> str:
    """Push ``results_dir`` to a new branch and return the branch name."""
    source = Path(results_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"no results at {source}")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"kaggle-results/{stamp}"
    if remote_url is None:
        token = token or os.environ.get("GITHUB_TOKEN") or kaggle_secret("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set (env var or Kaggle secret)")
        remote_url = f"https://x-access-token:{token}@github.com/{repository}.git"

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "repo"
        _run(["git", "clone", "--quiet", "--depth", "1", "--branch", base, remote_url, str(clone)])
        _run(["git", "checkout", "-q", "-b", branch], cwd=clone)
        target = clone / "results" / "kaggle" / stamp
        for path in sorted(source.rglob("*")):
            if path.is_dir() or path.name.endswith(SKIP_SUFFIXES):
                continue
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if not target.exists():
            raise RuntimeError(f"nothing to publish under {source}")
        _run(["git", "add", "-A"], cwd=clone)
        message = f"Kaggle results {stamp}" + (f": {label}" if label else "")
        _run(
            ["git", "-c", "user.name=onwordly-kaggle", "-c", "user.email=onwordly-kaggle@users.noreply.github.com",
             "commit", "-q", "-m", message],
            cwd=clone,
        )
        _run(["git", "push", "-q", "origin", branch], cwd=clone)
    print(f"published {source} -> {repository}@{branch}", flush=True)
    print(f"open a PR: https://github.com/{repository}/compare/{branch}?expand=1", flush=True)
    return branch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--repository", default="HereLiesAz/onwordly")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    publish_results(args.results_dir, repository=args.repository, label=args.label)


if __name__ == "__main__":
    main()
