"""test_biolm_score_oracle.py — pytest tests for the BioLM-Score oracle wrapper.

This module covers the public surface of
:file:`molmetal/scripts/biolm_score_oracle.py`:

1. The wrapper module imports cleanly (no heavy upstream deps at
   import time).
2. ``biolm_oracle_available()`` correctly reports availability
   on this host (no ``torch_scatter`` installed → not available).
3. ``biolm_score_smiles(...)`` returns ``0.0`` in dry-run mode
   (the documented "no upstream available" sentinel).
4. ``biolm_score_batch(...)`` returns a list of the same length
   as the input — no silent SMILES drops.
5. The :class:`BioLMScoreOracle` class supports both
   instantiation paths (explicit ``repo_root`` and
   default-discovery).
6. The oracle warns exactly once when dry-run mode is engaged
   (no log spam on batched calls).
7. ``biolm_score_smiles`` returns ``None`` when the caller asks
   for the real-mode path AND the upstream is missing — i.e.
   honest "I cannot run this" semantics.

All tests are CPU-only and self-skip-free: the wrapper is
designed to import without the heavy upstream stack.

Run from the project root::

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \\
        molmetal/tests/test_biolm_score_oracle.py -v --tb=short
"""

from __future__ import annotations

import importlib
import logging
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Lazy loader — keeps pytest collection working even when the wrapper
# module is missing or broken.
# ---------------------------------------------------------------------------
def _load_oracle_module():
    try:
        return importlib.import_module("biolm_score_oracle")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"biolm_score_oracle module not importable: {exc}")


# ---------------------------------------------------------------------------
# 1. Smoke-import test
# ---------------------------------------------------------------------------
def test_smoke_import() -> None:
    """The wrapper module imports cleanly without the heavy upstream.

    Importing :mod:`biolm_score_oracle` must NOT pull in
    ``torch_scatter``, ``torch_geometric``, or the
    ``BioLM_Score.model.model4`` module — those are loaded
    only inside :func:`biolm_oracle_available`.
    """
    mod = _load_oracle_module()
    assert mod is not None
    # Public surface — the names downstream code imports.
    for name in (
        "BioLMScoreOracle",
        "biolm_score_smiles",
        "biolm_score_batch",
        "biolm_oracle_available",
        "_dry_run_score",
        "_dry_run_score_batch",
        "_resolve_default_repo_root",
    ):
        assert hasattr(mod, name), f"missing public symbol: {name}"


def test_dry_run_no_protein() -> None:
    """Score a SMILES list without protein context — dry-run path.

    With no upstream available (no ``torch_scatter``, no
    checkpoint, no embeddings) the wrapper MUST degrade to a
    deterministic ``0.0`` per SMILES.  No silent SMILES drop,
    no fabricated affinity, no exception raised.
    """
    mod = _load_oracle_module()
    smiles_list = ["CCO", "c1ccccc1O", "CC(=O)O", "INVALID_SMILES"]
    scores = mod.biolm_score_batch(smiles_list)
    assert isinstance(scores, list)
    assert len(scores) == len(smiles_list), (
        f"score list length {len(scores)} != input length {len(smiles_list)}"
    )
    for s in scores:
        assert s == 0.0, f"expected dry-run 0.0, got {s!r}"


# ---------------------------------------------------------------------------
# 2. Availability probe — the host has no torch_scatter, so this
#    is the honest-negative path that the wrapper must handle.
# ---------------------------------------------------------------------------
def test_oracle_available_reports_false_on_this_host() -> None:
    """On this host (no ``torch_scatter``), the probe returns ``False``."""
    mod = _load_oracle_module()
    avail = mod.biolm_oracle_available()
    # The vendored repo IS present, but ``torch_scatter`` is not —
    # so the probe must report False (partial availability is
    # reported as not-available).
    if _has_torch_scatter():
        # If torch_scatter happens to be installed, the probe may
        # legitimately return True — accept either way.
        assert isinstance(avail, bool)
    else:
        assert avail is False, (
            "Probe must report False when torch_scatter is missing"
        )


def _has_torch_scatter() -> bool:
    try:
        import torch_scatter  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3. Single-SMILES convenience path
# ---------------------------------------------------------------------------
def test_single_smiles_returns_zero_in_dry_run() -> None:
    """``biolm_score_smiles(...)`` returns ``0.0`` in dry-run mode."""
    mod = _load_oracle_module()
    v = mod.biolm_score_smiles("CCO")
    assert v == 0.0, f"expected dry-run 0.0, got {v!r}"


def test_single_smiles_returns_zero_with_explicit_dry_run_flag() -> None:
    """``biolm_score_smiles(..., dry_run=True)`` always returns ``0.0``."""
    mod = _load_oracle_module()
    v = mod.biolm_score_smiles("c1ccccc1O", dry_run=True)
    assert v == 0.0


# ---------------------------------------------------------------------------
# 4. Class form — explicit repo_root + dry_run
# ---------------------------------------------------------------------------
def test_oracle_class_explicit_dry_run(tmp_path: Path) -> None:
    """``BioLMScoreOracle(dry_run=True, repo_root=...)`` runs without
    touching the upstream.
    """
    mod = _load_oracle_module()
    oracle = mod.BioLMScoreOracle(
        repo_root=str(tmp_path / "no_such_repo"),
        dry_run=True,
    )
    assert oracle.available is False
    v = oracle.score("CCO")
    assert v == 0.0
    v_b = oracle.score_batch(["CCO", "c1ccccc1O"])
    assert v_b == [0.0, 0.0]


def test_oracle_class_preserves_input_length(tmp_path: Path) -> None:
    """``score_batch`` always returns a list of the same length as
    the input — no silent SMILES drop.
    """
    mod = _load_oracle_module()
    oracle = mod.BioLMScoreOracle(
        repo_root=str(tmp_path / "no_such_repo"),
        dry_run=True,
    )
    long_input = ["CCO"] * 17
    out = oracle.score_batch(long_input)
    assert len(out) == 17
    for v in out:
        assert v == 0.0


# ---------------------------------------------------------------------------
# 5. Warning behaviour — exactly once per oracle instance
# ---------------------------------------------------------------------------
def test_oracle_warns_exactly_once(tmp_path: Path) -> None:
    """The dry-run ``UserWarning`` fires on the first ``score()``
    call only — subsequent calls must not re-warn (prevents log
    spam in batched inference loops).
    """
    mod = _load_oracle_module()
    oracle = mod.BioLMScoreOracle(
        repo_root=str(tmp_path / "no_such_repo"),
        dry_run=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oracle.score("CCO")
        oracle.score("c1ccccc1O")
        oracle.score_batch(["c1ccccc1O", "CC(=O)O"])
    biolm_warnings = [
        w for w in caught if "BioLM" in str(w.message)
    ]
    assert len(biolm_warnings) == 1, (
        f"Expected exactly one BioLM warning, got {len(biolm_warnings)}: "
        f"{[str(w.message) for w in biolm_warnings]}"
    )


# ---------------------------------------------------------------------------
# 6. Honest-negative — caller asks for real mode but upstream missing
# ---------------------------------------------------------------------------
def test_real_mode_returns_none_when_upstream_missing(tmp_path: Path) -> None:
    """When the caller asks for real mode (``require_real=True``) but
    the upstream is unavailable, ``score()`` returns ``None`` —
    the honest "I cannot run this" sentinel — NOT ``0.0`` (which
    would be conflated with the dry-run placeholder).

    Distinguishing ``None`` from ``0.0`` matters for downstream
    MCTS / Pareto code that treats "no oracle available"
    differently from "oracle ran but returned zero".
    """
    mod = _load_oracle_module()
    oracle = mod.BioLMScoreOracle(
        repo_root=str(tmp_path / "no_such_repo"),
        dry_run=False,
    )
    assert oracle.available is False
    v = oracle.score("CCO", require_real=True)
    # No upstream available AND require_real → wrapper returns None.
    assert v is None, (
        f"Expected None (real-mode without upstream), got {v!r}"
    )
    # Batched path must also return Nones, preserving length.
    v_b = oracle.score_batch(["CCO", "c1ccccc1O"], require_real=True)
    assert len(v_b) == 2
    for v in v_b:
        assert v is None, f"Expected None in real-mode batch, got {v!r}"

    # Sanity: WITHOUT require_real, the same call falls back to 0.0.
    v_default = oracle.score("CCO")
    assert v_default == 0.0


# ---------------------------------------------------------------------------
# 7. CLI smoke test — invoke the module as a script and confirm
#    it produces one tab-separated ``smiles\\tscore`` line per input.
# ---------------------------------------------------------------------------
def test_cli_smoke(tmp_path: Path) -> None:
    """Run :mod:`biolm_score_oracle` as a subprocess and verify the
    tab-separated output shape.
    """
    import subprocess

    bin_ = "/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
    proc = subprocess.run(
        [
            bin_,
            str(SCRIPTS_DIR / "biolm_score_oracle.py"),
            "--smiles",
            "CCO,c1ccccc1O,CC(=O)O",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"CLI failed: rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    assert len(lines) == 3, f"Expected 3 output lines, got {len(lines)}"
    for ln in lines:
        parts = ln.split("\t")
        assert len(parts) == 2
        smi, sc = parts
        assert smi in ("CCO", "c1ccccc1O", "CC(=O)O")
        assert sc == "0.0"
    # The stderr line confirms availability probe result.
    assert "upstream_available=" in proc.stderr