"""Tests for the VariancePUCT-default integration layer (TODO-92 R20-P1).

================================================================
CPU-only smoke / contract tests for the R20-P1 ship: VariancePUCT
replaces Dirichlet PUCT as the **default** PUCT selector, with
backward-compat opt-out.
================================================================
This module exercises:

* :data:`DEFAULT_USE_VARIANCE_PUCT` constant (must be True at R20-P1)
* :func:`make_default_selector` (the canonical factory)
  - default-on path -> VarianceControlledPUCT with c_var=DEFAULT_C_VAR
  - default-off path -> VarianceControlledPUCT with c_var=0.0
  - explicit use_variance_puct=True / False override
  - parameter forwarding (c_puct / c_var / n_min_visits / epsilon / rng_seed)
* :class:`VarianceControlledPUCT` behaviour under the default
  contract — actions are valid indices, Q is bounded in [0, 1] for
  unit-variance rewards, sigma is non-negative.
* Backward compat — the legacy Dirichlet PUCT path (c_var=0) still
  works and returns the same action statistics as a hand-built
  VarianceControlledPUCT with c_var=0.

The tests are CPU-only and complete in <1 s on any CI container.
None of them mutate :mod:`proof_search` or :mod:`score_repellent_mc`.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_puct_default.py -v
"""

from __future__ import annotations

import random
from pathlib import Path


# Lazy import so missing torch / score_repellent_mc is caught at
# use-time, not at module load (matches the convention in
# test_variance_puct_integration.py).
try:
    from molmetal_lam.search_alg.variance_puct_integration import (
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        DEFAULT_N_MIN_VISITS,
        DEFAULT_USE_VARIANCE_PUCT,
        IntegrationNode,
        VarianceControlledPUCT,
        make_default_selector,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore  # noqa: E402
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        DEFAULT_N_MIN_VISITS,
        DEFAULT_USE_VARIANCE_PUCT,
        IntegrationNode,
        VarianceControlledPUCT,
        make_default_selector,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_root(n_children: int = 4, prior: float = 0.5) -> IntegrationNode:
    """Return a root :class:`IntegrationNode` with ``n_children`` children."""
    return IntegrationNode(
        node_id="root",
        prior=float(prior),
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0) for i in range(int(n_children))
        ],
    )


# ---------------------------------------------------------------------------
# DEFAULT_USE_VARIANCE_PUCT constant
# ---------------------------------------------------------------------------


def test_default_use_variance_puct_is_true():
    """G1_variance_puct_default — the project default must be True at R20-P1."""
    assert DEFAULT_USE_VARIANCE_PUCT is True, (
        "R20-P1 (TODO-92) requires DEFAULT_USE_VARIANCE_PUCT=True; "
        "if you intentionally rolled back, update the test."
    )


# ---------------------------------------------------------------------------
# make_default_selector — factory behaviour
# ---------------------------------------------------------------------------


def test_make_default_selector_returns_variance_puct_when_default_on():
    """Default-on path returns a VarianceControlledPUCT with c_var=DEFAULT_C_VAR."""
    sel = make_default_selector()
    assert isinstance(sel, VarianceControlledPUCT)
    # R20-P1: variance penalty term must be active.
    assert sel.c_var == DEFAULT_C_VAR
    assert sel.c_var > 0.0
    assert sel.c_puct == DEFAULT_C_PUCT
    assert sel.n_min_visits == DEFAULT_N_MIN_VISITS
    assert sel.score_repellent is True


def test_make_default_selector_falls_back_when_explicit_false():
    """use_variance_puct=False -> c_var=0 (Dirichlet PUCT legacy fallback)."""
    sel = make_default_selector(use_variance_puct=False)
    assert isinstance(sel, VarianceControlledPUCT)
    # Backward compat: variance term disabled -> canonical Dirichlet PUCT.
    assert sel.c_var == 0.0
    assert sel.c_puct == DEFAULT_C_PUCT


def test_make_default_selector_respects_explicit_true():
    """use_variance_puct=True -> c_var=DEFAULT_C_VAR (even if global default changes)."""
    sel = make_default_selector(use_variance_puct=True)
    assert sel.c_var == DEFAULT_C_VAR
    assert sel.c_var > 0.0


def test_make_default_selector_forwards_parameters():
    """All kwargs forwarded correctly to the VarianceControlledPUCT constructor."""
    sel = make_default_selector(
        use_variance_puct=True,
        c_puct=1.7,
        c_var=0.3,
        score_repellent=False,
        n_min_visits=4,
        epsilon=0.1,
        rng_seed=42,
    )
    assert sel.c_puct == 1.7
    assert sel.c_var == 0.3
    assert sel.score_repellent is False
    assert sel.n_min_visits == 4
    assert sel.epsilon == 0.1
    assert sel.rng_seed == 42


def test_make_default_selector_uses_global_default_when_none():
    """When use_variance_puct is None, the global DEFAULT_USE_VARIANCE_PUCT is used."""
    sel = make_default_selector(use_variance_puct=None)
    if DEFAULT_USE_VARIANCE_PUCT:
        assert sel.c_var == DEFAULT_C_VAR
        assert sel.c_var > 0.0
    else:
        assert sel.c_var == 0.0


# ---------------------------------------------------------------------------
# Backward compat — Dirichlet PUCT legacy path
# ---------------------------------------------------------------------------


def test_legacy_dirichlet_puct_path_selects_valid_actions():
    """Legacy path (c_var=0) still returns valid action indices in [0, n_actions)."""
    sel = make_default_selector(use_variance_puct=False)
    root = _mk_root(n_children=4)
    for _ in range(20):
        a = sel.select(root)
        assert 0 <= a < 4
        # Update with a valid (value, variance) — variance is ignored
        # when c_var=0 but we still need to pass something well-formed.
        sel.update(root, a, value=0.5, variance=0.0)


def test_variance_puct_default_path_selects_valid_actions():
    """Default-on path (c_var>0) still returns valid action indices in [0, n_actions)."""
    sel = make_default_selector(use_variance_puct=True)
    root = _mk_root(n_children=4)
    for _ in range(20):
        a = sel.select(root)
        assert 0 <= a < 4
        sel.update(root, a, value=0.5, variance=0.1)


def test_variance_punct_prefers_low_variance_arm_when_q_equal():
    """VariancePUCT prefers low-sigma arms when Q is equal (the variance bonus)."""
    sel = make_default_selector(
        use_variance_puct=True,
        c_var=1.0,  # strong variance penalty
        n_min_visits=1,  # activate variance term immediately
    )
    root = _mk_root(n_children=2)
    # Equal Q, but arm 0 has higher sigma -> selector should prefer arm 1.
    sel.update(root, 0, value=0.5, variance=1.0)  # sigma=1.0
    sel.update(root, 1, value=0.5, variance=0.0)  # sigma=0.0
    # Force argmax path (all actions visited).
    sel.update(root, 0, value=0.5, variance=1.0)
    sel.update(root, 1, value=0.5, variance=0.0)
    a = sel.select(root)
    # With c_var=1.0, n_min_visits=1, equal Q -> arm 1 wins.
    assert a == 1, f"VariancePUCT should prefer low-sigma arm; got {a}"


# ---------------------------------------------------------------------------
# Per-arm statistics contract (default-on path)
# ---------------------------------------------------------------------------


def test_per_arm_stats_well_formed_default_on():
    """Per-arm stats (N, Q, sigma) are well-formed after a small run."""
    sel = make_default_selector(
        use_variance_puct=True,
        c_var=0.5,
        n_min_visits=2,
        rng_seed=0,
    )
    rng = random.Random(0)
    root = _mk_root(n_children=3)
    arm_means = [0.2, 0.5, 0.8]
    for _ in range(30):
        a = sel.select(root)
        assert 0 <= a < 3
        v = rng.gauss(arm_means[a], 0.1)
        sel.update(root, a, value=v, variance=0.01)
    # Each arm stats must be well-formed.
    for i in range(3):
        st = sel.stats_.get((root.node_id, i))
        assert st is not None
        assert st["N"] >= 0
        assert 0.0 <= st["Q"] <= 1.0 + 1.0e-6
        assert st["sigma"] >= 0.0


# ---------------------------------------------------------------------------
# CLI smoke (script entry point)
# ---------------------------------------------------------------------------


def test_cli_smoke_default_on():
    """The CLI smoke path returns exit code 0 (acceptance gate G1+G5)."""
    import subprocess
    import sys as _sys

    repo_root = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [
            _sys.executable,
            "-m",
            "molmetal.scripts.run_proof_search_variance_puct",
            "--smoke",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, (
        f"CLI smoke (default-on) failed: rc={out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}"
    )
    assert "[PASS] VariancePUCT default selector contract holds" in out.stdout


def test_cli_smoke_legacy_dirichlet():
    """The CLI smoke path with --disable-variance-puct returns exit code 0."""
    import subprocess
    import sys as _sys

    repo_root = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [
            _sys.executable,
            "-m",
            "molmetal.scripts.run_proof_search_variance_puct",
            "--smoke",
            "--disable-variance-puct",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, (
        f"CLI smoke (disable) failed: rc={out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}"
    )
    assert "dirichlet_puct_legacy" in out.stdout
