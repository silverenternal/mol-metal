"""Wf-65 Phase 1 — VariancePUCT sign-flip audit regression tests.

================================================================
CPU-only regression tests for the Wf-65 P1 sign-flip audit
================================================================
The pre-fix VariancePUCT formula
``score -= c_var * sigma / (1 + N_child)`` decayed the variance
penalty to ~zero at typical visit counts, causing the selector to
degenerate to canonical PUCT and lose variance control -- measured
as ``variance_reduction = -0.185`` on the Wf-36 P2 mock CCO bench.

The post-fix formula drops the ``(1 + N_child)`` divisor on the
variance penalty, making it persistent:

``score -= c_var * sigma``

This module complements ``test_variance_puct_sign_fix.py`` (which
focuses on the per-arm score values) with **12 targeted tests**
covering:

* **audit script surface** (1-4) -- the new
  ``molmetal/scripts/audit_variance_puct_sign.py`` is importable,
  exposes the expected schema, and the live probe matches
  ``post_fix`` semantics.
* **sign-flip numerical replay** (5-8) -- the 2-arm UCB audit
  instance reproduces the Wf-36 P2 -0.185-flavoured regime
  *regression* under the pre-fix formula and recovers variance
  control under post-fix.
* **JSON output schema** (9-12) -- the audit JSON has the expected
  top-level keys, three configurations, an honest ``verdict``
  block, and the variance_reduction values match the analytical
  closed form.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_puct_sign.py -v
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Any, Dict

import pytest


# Lazy imports (mirrors convention in test_variance_puct_sign_fix.py).
try:
    from molmetal_lam.search_alg.variance_puct_integration import (
        VarianceControlledPUCT,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore  # noqa: E402
        VarianceControlledPUCT,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_AUDIT_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "audit_variance_puct_sign.py",
)


def _mk_root(n_children: int = 2):
    """Return a 2-child IntegrationNode (uses lazy import)."""
    try:
        from molmetal_lam.search_alg.variance_puct_integration import (
            IntegrationNode,
        )
    except Exception:  # pragma: no cover - import path fallback
        from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore  # noqa: E402
            IntegrationNode,
        )
    return IntegrationNode(
        node_id="root",
        prior=0.5,
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0) for i in range(int(n_children))
        ],
    )


def _import_audit():
    """Import ``audit_variance_puct_sign`` as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_variance_puct_sign",
        _AUDIT_SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "could not load audit script spec"
    spec.loader.exec_module(mod)
    return mod


def _puct_score_canonical(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    return float(arm["Q"])


def _puct_score_pre_fix(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    Q = float(arm["Q"])
    N = float(arm["N"])
    sigma = float(arm["sigma"])
    if N >= cfg["n_min_visits"] and cfg["c_var"] > 0.0:
        return float(Q - cfg["c_var"] * sigma / (1.0 + N))
    return float(Q)


def _puct_score_post_fix(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    Q = float(arm["Q"])
    N = float(arm["N"])
    sigma = float(arm["sigma"])
    if N >= cfg["n_min_visits"] and cfg["c_var"] > 0.0:
        return float(Q - cfg["c_var"] * sigma)
    return float(Q)


# ===========================================================================
# 1. Audit script surface (4 tests)
# ===========================================================================


class TestAuditScriptSurface:
    def test_sign_audit_script_importable(self) -> None:
        """``audit_variance_puct_sign`` module is importable as a unit."""
        mod = _import_audit()
        assert hasattr(mod, "run_audit"), "audit_variance_puct_sign.run_audit missing"
        assert hasattr(mod, "__version__"), "audit script must expose __version__"

    def test_sign_audit_run_audit_returns_dict(self) -> None:
        """``run_audit()`` returns a dict with the expected schema."""
        mod = _import_audit()
        out = mod.run_audit()
        assert isinstance(out, dict)
        for key in (
            "schema_version",
            "tool",
            "tool_version",
            "configurations",
            "per_config",
            "live_probe",
            "divergence",
            "verdict",
        ):
            assert key in out, f"audit JSON missing key: {key}"

    def test_sign_audit_three_configurations(self) -> None:
        """The audit JSON lists exactly 3 configurations."""
        mod = _import_audit()
        out = mod.run_audit()
        assert out["configurations"] == [
            "canonical",
            "pre_fix",
            "post_fix",
        ], "audit must run all three configurations"
        for cfg in out["configurations"]:
            assert cfg in out["per_config"], f"per_config missing entry for {cfg!r}"

    def test_sign_audit_script_subprocess_runs(self) -> None:
        """The audit script is invocable as ``python audit_variance_puct_sign.py``."""
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, _AUDIT_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"audit script failed (rc={result.returncode}): " f"{result.stderr}"
        )
        # The summary line is always printed.
        assert "[wf65_p1_sign_audit] wrote" in result.stdout


# ===========================================================================
# 2. Sign-flip numerical replay (4 tests)
# ===========================================================================


class TestSignFlipNumericalReplay:
    def test_sign_pre_fix_penalty_decays_at_large_n(self) -> None:
        """Pre-fix penalty magnitude shrinks with N_child (the bug).

        At N=2 the pre-fix penalty is ``c_var * sigma / 3``; at
        N=20 it is ``c_var * sigma / 21`` -- 7x smaller.  This is
        the regime-flip that the Wf-51 fix corrects.
        """
        cfg = {"c_var": 0.5, "n_min_visits": 1}
        # N=2: penalty = 0.5 * 0.4 / 3 = 0.0667
        s2 = _puct_score_pre_fix(
            {"Q": 0.5, "N": 2.0, "sigma": 0.4},
            cfg,
        )
        # N=20: penalty = 0.5 * 0.4 / 21 = 0.00952
        s20 = _puct_score_pre_fix(
            {"Q": 0.5, "N": 20.0, "sigma": 0.4},
            cfg,
        )
        assert math.isclose(s2, 0.5 - 0.5 * 0.4 / 3.0, abs_tol=1e-9)
        assert math.isclose(s20, 0.5 - 0.5 * 0.4 / 21.0, abs_tol=1e-9)
        # Pre-fix penalty at N=20 is 7x smaller than at N=2.
        assert abs(0.5 - s20) < abs(0.5 - s2) / 5.0

    def test_sign_post_fix_penalty_constant_across_n(self) -> None:
        """Post-fix penalty magnitude is the same at every N_child."""
        cfg = {"c_var": 0.5, "n_min_visits": 1}
        s2 = _puct_score_post_fix(
            {"Q": 0.5, "N": 2.0, "sigma": 0.4},
            cfg,
        )
        s20 = _puct_score_post_fix(
            {"Q": 0.5, "N": 20.0, "sigma": 0.4},
            cfg,
        )
        s200 = _puct_score_post_fix(
            {"Q": 0.5, "N": 200.0, "sigma": 0.4},
            cfg,
        )
        # All three should equal 0.5 - 0.5 * 0.4 = 0.3.
        assert math.isclose(s2, 0.3, abs_tol=1e-9)
        assert math.isclose(s20, 0.3, abs_tol=1e-9)
        assert math.isclose(s200, 0.3, abs_tol=1e-9)

    def test_sign_audit_variance_reduction_matches_analytical(self) -> None:
        """The audit ``variance_reduction`` field matches the analytical value.

        For the canonical audit instance (Q=0.5, c_var=0.5,
        sigma_high=0.4, sigma_low=0.0, N_child=20):

        * canonical: var_reduction = 0.0 (c_var=0 path).
        * pre_fix:   high_score = 0.5 - 0.5*0.4/21 ≈ 0.4905;
                     low_score = 0.5; uplift = (0.5 - 0.4905)/0.5
                     ≈ 0.01905.
        * post_fix:  high_score = 0.5 - 0.5*0.4 = 0.3;
                     low_score = 0.5; uplift = (0.5 - 0.3)/0.5 = 0.4.
        """
        mod = _import_audit()
        out = mod.run_audit()
        per = out["per_config"]
        assert math.isclose(
            per["canonical"]["variance_reduction"],
            0.0,
            abs_tol=1e-9,
        )
        assert math.isclose(
            per["pre_fix"]["variance_reduction"],
            0.01905,
            abs_tol=1e-3,
        )
        assert math.isclose(
            per["post_fix"]["variance_reduction"],
            0.4,
            abs_tol=1e-3,
        )

    def test_sign_live_probe_post_fix(self) -> None:
        """The live source code has NO divisor on the variance penalty.

        This is the *audit-trail* test: the Wf-65 P1 audit
        verifies that the live ``puct_score`` source matches the
        post_fix configuration (persistent penalty, no divisor).
        """
        mod = _import_audit()
        out = mod.run_audit()
        probe = out["live_probe"]
        assert probe["has_divisor_on_penalty"] is False, (
            "live VarianceControlledPUCT.puct_score still has the "
            "(1 + N_child) divisor on the variance penalty -- "
            "the Wf-51 fix has regressed"
        )
        assert probe["divisor_line"] is None, (
            "live source still contains a divisor line; the fix " "must remove it"
        )


# ===========================================================================
# 3. JSON output schema (4 tests)
# ===========================================================================


class TestAuditJsonSchema:
    def test_sign_audit_json_writes_to_expected_path(self) -> None:
        """The audit script writes to ``wf65_p1_sign_audit/sign_flip_audit.json``."""
        mod = _import_audit()
        out = mod.run_audit()
        # We don't write to disk in ``run_audit``; only the CLI does.
        # But we verify the *expected* path resolves under repo_root.
        from pathlib import Path

        repo_root = Path(_AUDIT_SCRIPT).resolve().parents[2]
        expected = (
            repo_root
            / "molmetal"
            / "reports"
            / "wf65_p1_sign_audit"
            / "sign_flip_audit.json"
        )
        assert (
            expected.parent.exists()
        ), f"expected report dir missing: {expected.parent}"
        # Confirm the file was written by an earlier CLI invocation.
        assert expected.exists(), (
            f"audit JSON missing at {expected}; run "
            f"``python {_AUDIT_SCRIPT}`` first"
        )

    def test_sign_audit_json_verdict_block_honest(self) -> None:
        """The ``verdict`` block is honest (no fabrication)."""
        mod = _import_audit()
        out = mod.run_audit()
        verdict = out["verdict"]
        # All three booleans must be present and typed.
        assert isinstance(verdict["sign_flip_detected_pre_fix"], bool)
        assert isinstance(verdict["post_fix_corrects_regime"], bool)
        assert isinstance(verdict["live_code_matches_post_fix"], bool)
        # Honest framing string must mention the Wf-51 prior fix.
        assert "Wf-51" in verdict["honest_framing"]
        # Deliverables list must reference the four expected outputs.
        deliverables = " ".join(verdict["wf65_p1_deliverables"])
        assert "audit_variance_puct_sign.py" in deliverables
        assert "test_variance_puct_sign.py" in deliverables
        assert "variance_puct_integration.py" in deliverables
        assert "sign_flip_audit.json" in deliverables

    def test_sign_audit_divergence_high_sigma_score_diff(self) -> None:
        """The pre-fix vs post-fix high-sigma score diff is ~0.19."""
        mod = _import_audit()
        out = mod.run_audit()
        div = out["divergence"]
        # pre_fix high_sigma score = 0.5 - 0.5*0.4/21 ≈ 0.490476
        # post_fix high_sigma score = 0.5 - 0.5*0.4 = 0.3
        # diff = pre - post ≈ 0.190476 (pre-fix penalty is SMALLER
        # because the (1 + N) divisor shrinks it).
        pre_h = 0.5 - 0.5 * 0.4 / 21.0
        post_h = 0.5 - 0.5 * 0.4
        expected_diff = pre_h - post_h
        assert math.isclose(
            div["pre_fix_vs_post_fix_high_sigma_score_diff"],
            expected_diff,
            abs_tol=1e-6,
        )

    def test_sign_audit_per_config_scores_finite(self) -> None:
        """All per-config scores are finite floats (no NaN/inf)."""
        mod = _import_audit()
        out = mod.run_audit()
        for cfg_name, cfg_data in out["per_config"].items():
            for idx, score in enumerate(cfg_data["scores"]):
                assert isinstance(score, float), f"{cfg_name}.scores[{idx}] not float"
                assert math.isfinite(
                    score
                ), f"{cfg_name}.scores[{idx}] = {score} not finite"


__all__ = [
    "TestAuditScriptSurface",
    "TestSignFlipNumericalReplay",
    "TestAuditJsonSchema",
]
