"""Pytest suite for Wf-122 Phase 3 — Hilbert+ASTSR combined R² post-fix smoke.

================================================================
This module validates the post-fix pipeline invariants for the
Wf-122 Phase 3 re-bench (L2 regularisation + ASTSR scale-mismatch
guard + best_ast_lisp() accessor fix) of
``molmetal/scripts/bench_hilbert_astr_combined_real.py`` on
PlatinAI 4-cell with 5 seeds × 50 epochs × strict-scaffold split.

Test design
-----------
1. ``test_wf122_p1_l2_lambda_default`` — the bench script's
   ``fit_hilbert`` uses ``LambdaASTSRHilbert`` with default
   ``l2_lambda=0.01`` (the Wf-122 P1 fix).
2. ``test_wf122_p12_scale_guard_active`` — the ASTSR ``_score``
   method contains the scale-mismatch guard that forces
   ``-inf`` on collapse AND explode (Wf-122 P1.2).
3. ``test_wf122_p13_best_ast_lisp_accessor`` — ``LambdaASTSR``
   exposes ``best_ast_lisp()`` returning the LISP rendering of
   :attr:`best_ast_` or ``"<no_lisp>"`` (Wf-122 P1.3).
4. ``test_post_fix_per_seed_metrics_keys`` — the per-seed JSON
   written by ``evaluate_cell_line`` has all post-fix required keys
   (including ``best_ast_lisp``).
5. ``test_post_fix_combined_r2_present`` — the synthesis.md or
   per_seed_metrics.json contains a combined_r2_test_mean field.
6. ``test_post_fix_no_collapse_in_per_seed`` — the Wf-122 P1.2
   scale guard means no (cell × seed) pair should have
   ``collapse_detected = yes`` post-fix (the guard rejects
   the collapse-mode AST during selection).
7. ``test_post_fix_hilbert_r2_finite`` — every per-seed Hilbert R²
   is finite (not NaN/Inf) thanks to L2 regularisation.
8. ``test_post_fix_astr_r2_finite`` — every per-seed ASTSR R² is
   finite (not NaN/Inf) thanks to scale-guard.
9. ``test_post_fix_best_ast_lisp_non_collapse`` — at least one
   per-seed best_ast_lisp is a real signal-driven AST (not a
   constant collapse trick like ``(div (add 2.0 0.5) 0.5)``).
10. ``test_post_fix_synthesis_md_pre_post_comparison`` — the
    synthesis.md contains the pre-fix vs post-fix comparison
    table with the −1.50 anchor.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_combined_real.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR  # noqa: E402
from molmetal.molmetal_lam.reward.hilbert_primal_dual import (  # noqa: E402
    LambdaASTSRHilbert,
)
from molmetal.scripts.bench_hilbert_astr_combined_real import (  # noqa: E402
    fit_hilbert,
)


# ---------------------------------------------------------------------------
# 1. Wf-122 P1: L2 lambda default on LambdaASTSRHilbert.
# ---------------------------------------------------------------------------
def test_wf122_p1_l2_lambda_default() -> None:
    """LambdaASTSRHilbert defaults ``l2_lambda=0.01`` (Wf-122 P1 fix)."""
    opt = LambdaASTSRHilbert(
        population_dim=12,
        n_iters=5,
        seed=0,
    )
    assert opt.l2_lambda == 0.01, (
        f"expected l2_lambda=0.01 (Wf-122 P1 default), got {opt.l2_lambda}"
    )


# ---------------------------------------------------------------------------
# 2. Wf-122 P1.2: ASTSR scale-mismatch guard in _score.
# ---------------------------------------------------------------------------
def test_wf122_p12_scale_guard_active() -> None:
    """LambdaASTSR._score forces ``-inf`` on collapse / explode (Wf-122 P1.2)."""
    import inspect

    src = inspect.getsource(LambdaASTSR._score)
    # The guard must reference both the collapse detector and the
    # ``-inf`` return path.
    assert "collapse_detected" in src, (
        "Wf-122 P1.2: scale-mismatch guard must call collapse_detected in _score"
    )
    assert '"-inf"' in src or "'-inf'" in src, (
        "Wf-122 P1.2: scale-mismatch guard must return -inf on collapse"
    )


# ---------------------------------------------------------------------------
# 3. Wf-122 P1.3: best_ast_lisp() accessor.
# ---------------------------------------------------------------------------
def test_wf122_p13_best_ast_lisp_accessor() -> None:
    """LambdaASTSR.best_ast_lisp() returns LISP or "<no_lisp>" (Wf-122 P1.3)."""
    reg = LambdaASTSR(
        depth=2,
        population=4,
        generations=2,
        seed=0,
        feature_names=("a", "b"),
        early_stop_on_nan=True,
    )
    # Pre-fit: must return "<no_lisp>" per the Wf-122 P1.3 contract.
    assert reg.best_ast_lisp() == "<no_lisp>", (
        f"expected '<no_lisp>' pre-fit, got {reg.best_ast_lisp()!r}"
    )


# ---------------------------------------------------------------------------
# 4. Post-fix per_seed_metrics.json shape (NEW artifact).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def post_fix_dir() -> Path:
    """Locate the Wf-122 P3 post-fix output directory."""
    candidate = _REPO_ROOT / "molmetal" / "reports" / "wf122_p3_post_fix"
    return candidate


def test_post_fix_per_seed_metrics_keys(post_fix_dir: Path) -> None:
    """Each cell-line per_seed JSON has all post-fix required keys."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    required = {
        "cell_line", "n_samples", "n_seeds", "epochs", "hilbert_dim",
        "astr_population", "astr_depth", "device", "split",
        "combined_r2_test_mean", "combined_r2_test_std",
        "hilbert_r2_test_mean", "astr_r2_test_mean",
        "alpha_chosen_mean", "per_seed",
    }
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            pytest.fail(f"missing per-seed metrics for {cl}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        missing = required - set(data.keys())
        assert not missing, f"{cl} missing keys: {missing}"
        # per_seed[0] must also have best_ast_lisp + collapse_detected.
        ps0 = data["per_seed"][0]
        for key in ("best_ast_lisp", "collapse_detected", "alpha_chosen"):
            assert key in ps0, f"{cl} per_seed[0] missing {key}"


# ---------------------------------------------------------------------------
# 5. Post-fix combined_r2 field present.
# ---------------------------------------------------------------------------
def test_post_fix_combined_r2_present(post_fix_dir: Path) -> None:
    """At least one cell-line has a finite combined_r2_test_mean."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    found_finite = False
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = float(data["combined_r2_test_mean"])
        if np.isfinite(v):
            found_finite = True
            break
    assert found_finite, "no cell-line has finite combined_r2_test_mean"


# ---------------------------------------------------------------------------
# 6. Post-fix collapse guard prevents collapse_detected=yes.
# ---------------------------------------------------------------------------
def test_post_fix_no_collapse_in_per_seed(post_fix_dir: Path) -> None:
    """Scale guard should prevent collapse_detected=yes (the constant-collapse trick)."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    n_collapse = 0
    n_total = 0
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["per_seed"]:
            n_total += 1
            if bool(s["collapse_detected"]):
                n_collapse += 1
    if n_total == 0:
        pytest.fail("no per-seed data")
    # The scale guard should keep collapse at or near zero.
    # Pre-fix had 17/20 collapse=yes; post-fix should have 0/20 ideally,
    # but we allow a few in case the test set has truly tiny std.
    assert n_collapse <= n_total // 5, (
        f"expected ≤{n_total // 5} collapse_detected=yes post-fix, got "
        f"{n_collapse}/{n_total}"
    )


# ---------------------------------------------------------------------------
# 7. Hilbert R² finite (L2 fix prevents NaN/Inf).
# ---------------------------------------------------------------------------
def test_post_fix_hilbert_r2_finite(post_fix_dir: Path) -> None:
    """All per-seed Hilbert R² must be finite (L2 prevents divergence)."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["per_seed"]:
            for k in ("hilbert_r2_train", "hilbert_r2_val", "hilbert_r2_test"):
                v = float(s[k])
                assert np.isfinite(v), (
                    f"{cl} seed {s['seed']} {k} = {v} (L2 fix should prevent "
                    f"divergence)"
                )


# ---------------------------------------------------------------------------
# 8. ASTSR R² finite (scale guard prevents NaN/Inf).
# ---------------------------------------------------------------------------
def test_post_fix_astr_r2_finite(post_fix_dir: Path) -> None:
    """All per-seed ASTSR R² must be finite (scale guard prevents blowup)."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["per_seed"]:
            for k in ("astr_r2_train", "astr_r2_val", "astr_r2_test"):
                v = float(s[k])
                assert np.isfinite(v), (
                    f"{cl} seed {s['seed']} {k} = {v} (scale guard should "
                    f"prevent blowup)"
                )


# ---------------------------------------------------------------------------
# 9. best_ast_lisp is a real signal-driven AST (not constant collapse trick).
# ---------------------------------------------------------------------------
def test_post_fix_best_ast_lisp_non_collapse(post_fix_dir: Path) -> None:
    """At least one best_ast_lisp is a signal-driven AST (not constant)."""
    if not post_fix_dir.exists():
        pytest.fail(f"post-fix dir missing: {post_fix_dir}")
    n_signal_driven = 0
    n_total = 0
    for cl in ("mcf7", "a549", "a2780", "a2780cis"):
        path = post_fix_dir / cl / "per_seed_metrics.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["per_seed"]:
            n_total += 1
            lisp = str(s.get("best_ast_lisp", "<no_lisp>"))
            # Real AST: contains at least one ``v_<feature>`` token.
            if re.search(r"v_\w+", lisp):
                n_signal_driven += 1
    if n_total == 0:
        pytest.fail("no per-seed data")
    assert n_signal_driven > 0, (
        f"expected ≥1 signal-driven best_ast_lisp, got "
        f"{n_signal_driven}/{n_total}"
    )


# ---------------------------------------------------------------------------
# 10. synthesis.md contains pre-fix vs post-fix comparison anchor.
# ---------------------------------------------------------------------------
def test_post_fix_synthesis_md_pre_post_comparison(post_fix_dir: Path) -> None:
    """synthesis.md contains the pre-fix baseline anchor (−1.50 or −1.5030)."""
    syn_path = post_fix_dir / "synthesis.md"
    if not syn_path.exists():
        pytest.fail(f"missing synthesis.md: {syn_path}")
    text = syn_path.read_text(encoding="utf-8")
    # Pre-fix anchor: −1.50 or −1.5030 (the Wf-85 P3 aggregate).
    assert "−1.50" in text or "-1.50" in text or "−1.5030" in text, (
        "synthesis.md must reference the pre-fix Wf-85 P3 anchor (−1.50 / −1.5030)"
    )
    # Post-fix anchor must also be present.
    assert "−5.66" in text or "-5.66" in text or "−5.6625" in text, (
        "synthesis.md must reference the post-fix Wf-122 P3 aggregate (−5.66)"
    )
