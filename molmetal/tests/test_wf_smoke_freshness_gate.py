# Licensed under the Apache License, Version 2.0.
"""Regression tests for the AGENTS.md §5.7 smoke-freshness gate (BLK-I4 F2).

Built around a NEGATIVE CONTROL, like the launcher-hardening suite: the gate must
FAIL a trainer whose flag surface changed with no smoke artifact, and PASS once a
fresh artifact exists. A gate that passes everything is worse than no gate.

The integration cases run against a throwaway git repo so `HEAD` is real — the
gate's whole mechanism is a diff against HEAD, and faking that would test nothing.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_smoke_freshness_gate.py"

_spec = importlib.util.spec_from_file_location("_smoke_freshness_gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_smoke_freshness_gate"] = gate
_spec.loader.exec_module(gate)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFlagSurface:
    def test_single_line_registrations(self):
        src = 'p.add_argument("--epochs", type=int)\np.add_argument("--lr")\n'
        assert gate.flag_surface(src) == {"--epochs", "--lr"}

    def test_multi_line_registration_is_caught(self):
        src = textwrap.dedent(
            """
            p.add_argument(
                "--update",
                choices=["gru", "residual"],
                default="gru",
            )
            """
        )
        assert gate.flag_surface(src) == {"--update"}

    def test_short_flags_are_ignored(self):
        src = 'p.add_argument("-x", "--xlong")\n'
        assert gate.flag_surface(src) == {"--xlong"}

    def test_prose_mentioning_a_flag_does_not_count(self):
        src = "# see --not-a-real-flag for details\n"
        assert gate.flag_surface(src) == set()


class TestArtifactFreshness:
    def _make(self, tmp_path: Path, *, rc: int = 0, metrics=None, age: float = 1.0):
        trainer = tmp_path / "train_x.py"
        trainer.write_text("x = 1\n")
        artifact = tmp_path / "train_x.json"
        artifact.write_text(
            json.dumps({"smoke": {"returncode": rc, "metrics": metrics if metrics is not None else {"auc": 0.8}}})
        )
        # age > 0 => artifact newer than trainer (both written this instant, so
        # nudge the artifact mtime forward explicitly).
        import os

        st = artifact.stat()
        os.utime(artifact, (st.st_atime + age, st.st_mtime + age))
        return trainer, artifact

    def test_missing_artifact_fails(self, tmp_path):
        trainer = tmp_path / "train_x.py"
        trainer.write_text("x=1\n")
        ok, why = gate.artifact_is_fresh_and_passing(tmp_path / "nope.json", trainer)
        assert ok is False and "no smoke artifact" in why

    def test_stale_artifact_fails(self, tmp_path):
        trainer, artifact = self._make(tmp_path, age=-1.0)  # artifact OLDER
        ok, why = gate.artifact_is_fresh_and_passing(artifact, trainer)
        assert ok is False and "OLDER" in why

    def test_nonzero_returncode_fails(self, tmp_path):
        trainer, artifact = self._make(tmp_path, rc=1)
        ok, why = gate.artifact_is_fresh_and_passing(artifact, trainer)
        assert ok is False and "returncode" in why

    def test_empty_metrics_is_accepted(self, tmp_path):
        """F2's literal ask is a fresh artifact, not a metrics payload. The CFM
        trainer writes per_epoch_metrics.json (not <out-dir>/metrics.json), so a
        real CFM smoke can carry an empty metrics dict — it must still pass."""
        trainer, artifact = self._make(tmp_path, metrics={})
        ok, _ = gate.artifact_is_fresh_and_passing(artifact, trainer)
        assert ok is True

    def test_fresh_end_to_end_passes(self, tmp_path):
        trainer, artifact = self._make(tmp_path)
        ok, why = gate.artifact_is_fresh_and_passing(artifact, trainer)
        assert ok is True

    def test_decoder_verdict_pass_is_not_required(self, tmp_path):
        """A classification trainer's artifact has verdict.pass=False (no
        decode_ratio); the gate must still accept it."""
        trainer, artifact = self._make(tmp_path)
        doc = json.loads(artifact.read_text())
        doc["verdict"] = {"pass": False}
        artifact.write_text(json.dumps(doc))
        import os

        st = artifact.stat()
        os.utime(artifact, (st.st_atime + 10, st.st_mtime + 10))
        ok, _ = gate.artifact_is_fresh_and_passing(artifact, trainer)
        assert ok is True


# ---------------------------------------------------------------------------
# Integration against a throwaway git repo (real HEAD)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )


def _trainer_src(extra_flags: str = "") -> str:
    return (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--epochs", type=int)\n'
        f"{extra_flags}"
    )


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """A tmp git repo with molmetal/scripts/train_a.py, HEAD = no extra flag."""
    (tmp_path / "molmetal" / "scripts").mkdir(parents=True)
    (tmp_path / "molmetal" / "reports" / "smoke_artifacts").mkdir(parents=True)
    trainer = tmp_path / "molmetal" / "scripts" / "train_a.py"
    trainer.write_text(_trainer_src())
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "SMOKE_DIR", tmp_path / "molmetal" / "reports" / "smoke_artifacts")
    return tmp_path, trainer


class TestEvaluateIntegration:
    def test_unchanged_trainer_needs_no_smoke(self, tmp_repo):
        _, _ = tmp_repo
        rep = gate.evaluate()
        assert rep["ok"] is True
        assert rep["n_changed"] == 0

    def test_changed_flag_without_artifact_fails(self, tmp_repo):
        _, trainer = tmp_repo
        trainer.write_text(_trainer_src('p.add_argument("--update")\n'))
        rep = gate.evaluate()
        assert rep["ok"] is False
        assert rep["n_changed"] == 1
        assert rep["findings"][0]["added_flags"] == ["--update"]

    def test_changed_flag_with_fresh_artifact_passes(self, tmp_repo):
        repo, trainer = tmp_repo
        trainer.write_text(_trainer_src('p.add_argument("--update")\n'))
        artifact = repo / "molmetal" / "reports" / "smoke_artifacts" / "train_a.json"
        artifact.write_text(json.dumps({"smoke": {"returncode": 0, "metrics": {"auc": 0.8}}}))
        import os

        st = artifact.stat()
        os.utime(artifact, (st.st_atime + 5, st.st_mtime + 5))
        rep = gate.evaluate()
        assert rep["ok"] is True
        assert rep["n_changed"] == 1

    def test_removed_flag_also_counts_as_changed(self, tmp_repo):
        repo, trainer = tmp_repo
        # HEAD has --epochs; remove it in the working tree.
        trainer.write_text("import argparse\np = argparse.ArgumentParser()\n")
        rep = gate.evaluate()
        assert rep["n_changed"] == 1
        assert rep["findings"][0]["removed_flags"] == ["--epochs"]

    def test_new_trainer_counts_all_flags_as_added(self, tmp_repo):
        repo, _ = tmp_repo
        (repo / "molmetal" / "scripts" / "train_b.py").write_text(_trainer_src('p.add_argument("--new")\n'))
        rep = gate.evaluate()
        names = {f["trainer"] for f in rep["findings"]}
        assert "molmetal/scripts/train_b.py" in names
