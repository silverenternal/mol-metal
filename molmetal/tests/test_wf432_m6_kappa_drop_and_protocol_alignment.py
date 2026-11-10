"""Wf-432-A3 + Wf-432-A4 protocol validator tests.

CPU-only tests; no GPU; no paper/*; no wet-lab; no sudo.
Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule + §5.6 atomic write protocol.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # molmetal/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /home/hugo/codes/mol-metal/
SCRIPT_M6 = REPO_ROOT / "scripts/wf432_m6_kappa_paper_drop_decision.py"
SCRIPT_ALIGN = REPO_ROOT / "scripts/wf432_protocol_alignment_validator.py"


# =============================================================================
# Wf-432-A3 m6_kappa drop decision tests
# =============================================================================


def test_m6_kappa_validate_subcommand_exits_zero() -> None:
    """m6_kappa drop decision validate exits 0 when block present."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_M6), "validate"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"validate failed: stderr={result.stderr}"
    data = json.loads(result.stdout)
    assert data["validation"] == "PASS"
    assert data["decision_block_found"] is True
    assert len(data["top3_alternatives"]) == 3


def test_m6_kappa_emit_latex_subcommand() -> None:
    """m6_kappa emit-latex writes a valid LaTeX file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out = tmp / "latex.tex"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_M6), "emit-latex", "--out", str(out)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"emit-latex failed: {result.stderr}"
        assert out.exists()
        content = out.read_text()
        assert r"\begin{table}[h]" in content
        assert r"\begin{tabular}" in content
        assert "Pearson r" in content
        # 4 NEW §6 paragraphs per Wf-410-A1 §3.1.6 + Wf-375-A4 §3 + Wf-377-A1 §4.4 + Wf-424-A15 §4
        assert r"\subsection{m6\_kappa dropped from §3 main-result}" in content
        assert "9 paths exhausted" in content


def test_m6_kappa_rebuild_decision_idempotent(tmp_path) -> None:
    """m6_kappa rebuild-decision is idempotent (running twice doesn't duplicate state).

    Wf-A123 isolation: the command writes ``metrics/by_metric/m6_kappa.json``.
    It is run against a COPY of the published artifact under ``tmp_path``,
    with the script's ``REPO_ROOT`` redirected there, so the published tree
    is never written (the previous version copied a ``.bak.wf432a3_test``
    file INTO the published directory and left it behind).
    """
    import importlib.util
    import shutil

    published = PROJECT_ROOT / "metrics" / "by_metric" / "m6_kappa.json"
    if not published.exists():
        import pytest

        pytest.skip(f"{published} missing — nothing to rebuild")

    sandbox = tmp_path / "sandbox"
    target = sandbox / "metrics" / "by_metric" / "m6_kappa.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(published, target)

    spec = importlib.util.spec_from_file_location(
        "wf432_m6_kappa_paper_drop_decision", str(SCRIPT_M6)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Redirect ONLY the write target (the command reads/writes nothing else).
    module.REPO_ROOT = sandbox

    assert module.main(["rebuild-decision"]) == 0

    # Confirm wf424a15_drop_decision block exists + has 9 paths in breakdown
    with open(target) as f:
        data = json.load(f)
    assert "wf424a15_drop_decision" in data
    assert data["wf424a15_drop_decision"]["paths_exhausted_breakdown"]["total"] == 9

    # Run again, verify same state
    assert module.main(["rebuild-decision"]) == 0
    with open(target) as f:
        data2 = json.load(f)
    assert data2["wf424a15_drop_decision"]["paths_exhausted_breakdown"]["total"] == 9


# =============================================================================
# Wf-432-A4 protocol alignment validator tests
# =============================================================================


def test_protocol_alignment_help_exits_zero() -> None:
    """wf432_protocol_alignment_validator.py --help exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_ALIGN), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "Wf-432-A4" in result.stdout


def test_protocol_alignment_list_protocols_returns_4() -> None:
    """list-protocols returns 4 protocol templates."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_ALIGN), "list-protocols"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"list-protocols failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert set(data.keys()) == {"Krasnov", "Rusanov", "DrugOOD", "TransDiffSBDD"}


def test_protocol_alignment_emit_protocol_writes_file() -> None:
    """emit-protocol writes a valid baseline-protocol JSON."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out = tmp / "Krasnov_protocol.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ALIGN),
                "emit-protocol",
                "--protocol",
                "Krasnov",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"emit-protocol failed: {result.stderr}"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["protocol"] == "Krasnov"
        assert "dataset_pattern" in data
        assert "split_protocol" in data
        assert "metric_definition" in data


def test_protocol_alignment_validate_dataset_align() -> None:
    """validate: dataset aligned + metric aligned → DATASET+METRIC alignment level."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = tmp / "trainer.json"
        trainer.write_text(
            json.dumps(
                {
                    "metric": "ru_auc",
                    "metric_definition": "mean ROC-AUC for Ru subset",
                    "data_source": "metalcytotox_5metal_multitask_v4.csv",
                    "split_protocol": "5-fold CV",
                }
            )
        )
        baseline = tmp / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "protocol": "Krasnov",
                    "dataset": "MetalCytoToxDB",
                    "dataset_pattern": ".*metalcytotox.*",
                    "split_protocol": "5-fold CV",
                    "metric_definition": "mean ROC-AUC for Ru subset",
                }
            )
        )
        out = tmp / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ALIGN),
                "validate",
                "--trainer-results",
                str(trainer),
                "--baseline-protocol",
                str(baseline),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out.read_text())
        assert data["alignment_level"] == "FULL"
        assert data["checks"]["dataset_aligned"] is True
        assert data["checks"]["split_protocol_aligned"] is True
        assert data["checks"]["metric_definition_aligned"] is True


def test_protocol_alignment_validate_dataset_only() -> None:
    """validate: dataset aligned + split different → DATASET_ALIGN."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = tmp / "trainer.json"
        trainer.write_text(
            json.dumps(
                {
                    "metric": "ru_auc",
                    "data_source": "metalcytotox_5metal_multitask_v4.csv",
                    "split_protocol": "scaffold-split 70/10/20 (Wf-261 strict)",
                }
            )
        )
        baseline = tmp / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "protocol": "Krasnov",
                    "dataset": "MetalCytoToxDB",
                    "dataset_pattern": ".*metalcytotox.*",
                    "split_protocol": "5-fold CV",
                    "metric_definition": "mean ROC-AUC for Ru subset",
                }
            )
        )
        out = tmp / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ALIGN),
                "validate",
                "--trainer-results",
                str(trainer),
                "--baseline-protocol",
                str(baseline),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out.read_text())
        assert (
            data["alignment_level"] == "DATASET + METRIC ALIGN"
            or data["alignment_level"] == "DATASET ALIGN"
        )
        assert data["checks"]["dataset_aligned"] is True
        assert data["checks"]["split_protocol_aligned"] is False


def test_protocol_alignment_validate_not_align() -> None:
    """validate: dataset doesn't match pattern → NOT_ALIGN."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = tmp / "trainer.json"
        trainer.write_text(
            json.dumps(
                {
                    "metric": "some_metric",
                    "data_source": "some_random_dataset.csv",
                }
            )
        )
        baseline = tmp / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "protocol": "Krasnov",
                    "dataset": "MetalCytoToxDB",
                    "dataset_pattern": ".*metalcytotox.*",
                    "split_protocol": "5-fold CV",
                    "metric_definition": "mean ROC-AUC for Ru subset",
                }
            )
        )
        out = tmp / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ALIGN),
                "validate",
                "--trainer-results",
                str(trainer),
                "--baseline-protocol",
                str(baseline),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out.read_text())
        assert data["alignment_level"] == "NOT_ALIGN"
        assert data["checks"]["dataset_aligned"] is False


if __name__ == "__main__":
    test_m6_kappa_validate_subcommand_exits_zero()
    test_m6_kappa_emit_latex_subcommand()
    test_m6_kappa_rebuild_decision_idempotent()
    test_protocol_alignment_help_exits_zero()
    test_protocol_alignment_list_protocols_returns_4()
    test_protocol_alignment_emit_protocol_writes_file()
    test_protocol_alignment_validate_dataset_align()
    test_protocol_alignment_validate_dataset_only()
    test_protocol_alignment_validate_not_align()
    print("OK: 9/9 Wf-432-A3+A4 tests PASS")
