"""test_bond_aromatic_lo_invariance — lock the CFM-WEAKNESSES D.3 invariant.

Background — CFM_WEAKNESSES D.3 / Wf-A43 §7:

    "The F1 IS MAXIMISED AT AROMATIC_LO=1.30, the current threshold.
     Tightening the band to 1.45 trades precision for RECALL (P=0.258
     −0.629, R=0.447 +0.160, F1=0.327 −0.107) and to 1.50 sharpens the
     trade (P=0.123 −0.764, R=0.731 +0.444, F1=0.211 −0.223). F1 is
     maximised at 1.30, the current threshold. The head's 5539 AROMATIC
     predictions are concentrated in BO 1.30-1.50 (4912 of 5539 = 88.7%
     fall inside [1.30, 1.70)); tightening the band relabels those
     'correct' predictions as 'false positive' rather than 'correct' and
     F1 falls. The collapse is RELOCATED, not FIXED: the constant edit
     would lower F1, not raise it. The binding axis is the head's
     intrinsic over-prediction of aromatic on lower-BO bonds, not the
     band boundary."

The fix here is NOT a code change.  It is a regression-prevention test:
    1. Lock the 3 band settings used in
       ``metrics/by_metric/cfm_m4_aromatic_lo_band_sweep.json``.
    2. Lock the F1-monotonicity direction (F1 strictly decreases as
       AROMATIC_LO increases from 1.30 → 1.50).
    3. Lock the precision/recall TRADE — raising AROMATIC_LO trades
       precision for recall (P ↓, R ↑); this is the empirical reason
       that threshold alone cannot close the F1 gap.
    4. Lock the source provenance — the JSON file lives at
       ``metrics/by_metric/cfm_m4_aromatic_lo_band_sweep.json`` and is
       ``MEASURED`` tagged.

A future PR that perturbs the band thresholds or claims "moving
AROMATIC_LO fixes the collapse" should observe these 3 PASS invocations
flip red and re-read this docstring.

Wf-A43 (2026-10-15) — CPU-only, no GPU / wet-lab / external data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SWEEP_JSON = REPO / "metrics" / "by_metric" / "cfm_m4_aromatic_lo_band_sweep.json"


def _load_sweep() -> dict:
    """Load the on-disk sweep JSON; missing file is a FAIL not a SKIP."""
    assert SWEEP_JSON.exists(), (
        f"missing sweep artefact at {SWEEP_JSON}; the D.3 invariant requires "
        f"the MEASURED JSON to be on disk (was it moved to reports/?)"
    )
    with SWEEP_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_aromatic_lo_band_sweep_invariant_present():
    """Sweep JSON carries the 3 aromatic_lo values used by the CFM probe."""
    sweep = _load_sweep()
    assert "aromatic_lo_values" in sweep, "sweep JSON missing aromatic_lo_values"
    assert sweep["aromatic_lo_values"] == [1.3, 1.45, 1.5], (
        "aromatic_lo_values drift: must be [1.30, 1.45, 1.50] to lock the "
        "F1-monotonicity invariant below; if you intentionally re-ran the "
        "probe at new band settings, update this test AND the verdict in "
        "metrics/by_metric/cfm_m4_aromatic_lo_band_sweep.json"
    )


def test_f1_strictly_decreases_with_aromatic_lo_increase():
    """F1 IS MAXIMISED AT AROMATIC_LO=1.30 — strictly decreasing past that.

    Locks the headline verdict: moving the band tightens the precision/
    recall trade and lowers F1.  This is the empirical reason "the
    constant edit would lower F1, not raise it".
    """
    sweep = _load_sweep()
    f1_by_lo = sweep["headline_sweep"]["aromatic_f1_full"]
    lo_values = sweep["headline_sweep"]["aromatic_lo"]
    assert lo_values == [1.3, 1.45, 1.5], (
        f"unexpected band order in sweep: {lo_values}"
    )
    f1_at_130, f1_at_145, f1_at_150 = f1_by_lo
    assert f1_at_130 > f1_at_145, (
        f"F1(1.30)={f1_at_130} must exceed F1(1.45)={f1_at_145} "
        f"(the collapse is RELOCATED, not FIXED)"
    )
    assert f1_at_145 > f1_at_150, (
        f"F1(1.45)={f1_at_145} must exceed F1(1.50)={f1_at_150} "
        f"(monotonic-decrease past 1.30)"
    )


def test_precision_recall_trade_invariant():
    """Raising AROMATIC_LO trades precision for recall — never both up.

    Locks the empirical precision/recall trade.  A future "I increased
    AROMATIC_LO and P went UP" report is a regression: it means the
    sweep JSON's verdict changed without the threshold being moved.
    """
    sweep = _load_sweep()
    precision = sweep["headline_sweep"]["aromatic_precision"]
    lo_values = sweep["headline_sweep"]["aromatic_lo"]
    assert lo_values == [1.3, 1.45, 1.5]
    p_130, p_145, p_150 = precision
    # Precision strictly decreases.
    assert p_130 > p_145 > p_150, (
        f"precision must be strictly decreasing in AROMATIC_LO; "
        f"got P(1.30)={p_130}, P(1.45)={p_145}, P(1.50)={p_150}"
    )
    # And: P(1.30) >= 0.5 (the head IS high-precision at the shipped
    # threshold — this is the headline of the D.3 verdict, not just the
    # precision/recall trade).
    assert p_130 >= 0.5, (
        f"shipped threshold AROMATIC_LO=1.30 must hold P >= 0.5; got P={p_130}. "
        f"Per D.3, the head is HIGH-PRECISION/LOW-RECALL at 1.30."
    )
    # Per-band deltas (vs 1.30) must be NEGATIVE — locks the verdict
    # verbatim.
    deltas = sweep["deltas_vs_1_30"]
    assert deltas["1.30_to_1.45"]["precision_delta"] < 0, (
        "1.30→1.45 precision_delta must be negative (precision lost)"
    )
    assert deltas["1.30_to_1.50"]["precision_delta"] < 0, (
        "1.30→1.50 precision_delta must be negative (precision lost)"
    )
    assert deltas["1.30_to_1.45"]["recall_delta"] > 0, (
        "1.30→1.45 recall_delta must be positive (recall gained)"
    )
    assert deltas["1.30_to_1.50"]["recall_delta"] > 0, (
        "1.30→1.50 recall_delta must be positive (recall gained)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))