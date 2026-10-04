import subprocess
from pathlib import Path

import pytest

from onwordly.publish import publish_results


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_publish_pushes_results_branch_without_bulky_files(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    _git("init", "-q", "--bare", str(remote))
    _git("init", "-q", "-b", "main", str(seed))
    (seed / "README.md").write_text("x\n")
    _git("add", ".", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=seed)
    _git("push", "-q", str(remote), "main", cwd=seed)

    results = tmp_path / "results" / "008-corrective-language-game"
    results.mkdir(parents=True)
    (results / "RESULTS.md").write_text("# r\n")
    (results / "summary.json").write_text("{}\n")
    (results / "static-train.jsonl").write_text("{}\n")
    (tmp_path / "results" / "008.tar.gz").write_bytes(b"x")

    branch = publish_results(tmp_path / "results", remote_url=str(remote), stamp="S1", label="008")
    assert branch == "kaggle-results/S1"
    files = _git("ls-tree", "-r", "--name-only", branch, cwd=remote).split()
    assert "results/kaggle/S1/008-corrective-language-game/RESULTS.md" in files
    assert "results/kaggle/S1/008-corrective-language-game/summary.json" in files
    assert not any(name.endswith((".jsonl", ".tar.gz")) for name in files)
    assert "README.md" in files


def test_publish_requires_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    (tmp_path / "r").mkdir()
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        publish_results(tmp_path / "r")
