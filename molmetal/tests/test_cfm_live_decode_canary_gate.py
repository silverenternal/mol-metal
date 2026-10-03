"""HOLE-1 verification tests — the cfm decode canary verifies TODAY's code.

HOLE-1 fix (2026-10-03) in ``molmetal/scripts/pre_flight_check.py``:

  * ``p1_cfm_smoke_decoded`` is relabelled HISTORICAL — it reads a
    FROZEN, UNTRACKED Wf-188 artifact whose source checkpoint changed
    six days after the artifact was written, so it cannot verify
    today's trainer / decoder.  The gate result is kept (name + status
    semantics unchanged), but its evidence now carries an explicit
    HISTORICAL note.
  * ``p1_cfm_live_decode_canary`` (registered in the ``cfm`` pipeline,
    P1) EXECUTES today's ``train_cfm_yuelbond_wasserstein.py`` +
    YuelBond decoder via the proven Wf-385-A3 3-epoch CPU smoke and
    requires loss decrease (0.3% floor) + ``decode_ratio_rdkit_strict
    > 0`` (fail-closed).

These tests are CPU-only, network-free, GPU-free, and do NOT run the
real smoke (the Wf-385-A3 script itself is covered by
``test_wf385a3_scaffold_wire_in_gate.py``; the end-to-end live PASS was
MEASURED at fix time: status=PASS, wall 6.5 s, decode_ratio=1.000).
The stub-based tests below prove the wiring FAILs loudly when the
canary reports a decode collapse / crash / missing script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ARTIFACT = (
    REPO_ROOT / "molmetal" / "reports" / "wf188_m14_full_metalcytotox" / "result.json"
)

STUB_VERDICT_FAIL = """import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--out-dir", required=True)
p.add_argument("--python-bin", default="")
p.add_argument("--timeout", type=float, default=0.0)
p.add_argument("--skip-smoke", action="store_true")
a, _ = p.parse_known_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
(out / "gate_result.json").write_text(json.dumps({
    "verdict": "FAIL",
    "reason": "simulated: decode_ratio_rdkit_strict collapsed to 0.0",
    "modules": [], "metrics": {}}), encoding="utf-8")
print("verdict=FAIL")
sys.exit(0)
"""

STUB_RC1 = """import sys
print("simulated trainer crash")
sys.exit(1)
"""


def _pf():
    from molmetal.scripts import pre_flight_check

    return pre_flight_check


class TestHistoricalAnchorRelabel:
    """``gate_cfm_smoke_decoded_historical`` keeps name/status but the
    evidence must read as HISTORICAL (not a live-code verification)."""

    def test_pass_on_fresh_mtime_copy_carries_historical_note(self, tmp_path):
        if not FROZEN_ARTIFACT.is_file():
            pytest.skip("frozen Wf-188 artifact absent (untracked)")
        pf = _pf()
        copy = tmp_path / "result.json"
        copy.write_bytes(FROZEN_ARTIFACT.read_bytes())
        os.utime(copy, None)  # AnchorGuard freshness: copy is "now"
        res = pf.gate_cfm_smoke_decoded_historical(
            "p1_cfm_smoke_decoded",
            "P1",
            "cfm",
            copy,
        )
        assert res.status == "PASS", res.evidence
        assert res.name == "p1_cfm_smoke_decoded"
        assert res.priority == "P1" and res.pipeline == "cfm"
        assert "HISTORICAL" in res.evidence
        assert "does NOT exercise today's code" in res.evidence
        assert "p1_cfm_live_decode_canary" in res.evidence
        assert "days vs artifact mtime" in res.evidence

    def test_fails_on_tampered_decode_ratio(self, tmp_path):
        if not FROZEN_ARTIFACT.is_file():
            pytest.skip("frozen Wf-188 artifact absent (untracked)")
        pf = _pf()
        data = json.loads(FROZEN_ARTIFACT.read_text(encoding="utf-8"))
        data["gate_4_decode_ratio"] = 0.0
        bad = tmp_path / "tampered.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        res = pf.gate_cfm_smoke_decoded_historical(
            "p1_cfm_smoke_decoded",
            "P1",
            "cfm",
            bad,
        )
        assert res.status == "FAIL"
        assert "NOT >= 0.5" in res.evidence
        assert "HISTORICAL" in res.evidence

    def test_fails_on_missing_file_with_historical_context(self, tmp_path):
        pf = _pf()
        res = pf.gate_cfm_smoke_decoded_historical(
            "p1_cfm_smoke_decoded",
            "P1",
            "cfm",
            tmp_path / "nope.json",
        )
        assert res.status == "FAIL"
        assert "HISTORICAL" in res.evidence


class TestLiveCanaryWiring:
    """``p1_cfm_live_decode_canary`` is registered for ``cfm`` and
    propagates FAIL from the underlying Wf-385-A3 gate."""

    def test_registered_in_cfm_pipeline(self):
        pf = _pf()
        cfm_names = [n for n, _ in pf.build_gates("cfm")]
        assert "p1_cfm_live_decode_canary" in cfm_names
        # Deliberately cfm-only: the `all` pipeline already executes the
        # identical smoke under `p3_wf385_scaffold_loss_wire_in_verified`
        # (asserted below), so an `all` registration would run the same
        # 3-epoch smoke twice per pre-flight.  If this assertion breaks,
        # bump EXPECTED_GATE_COUNTS in the same edit.
        all_names = [n for n, _ in pf.build_gates("all")]
        assert "p1_cfm_live_decode_canary" not in all_names
        assert "p3_wf385_scaffold_loss_wire_in_verified" in all_names

    def test_fail_verdict_propagates(self, tmp_path):
        pf = _pf()
        stub = tmp_path / "stub_verdict_fail.py"
        stub.write_text(STUB_VERDICT_FAIL, encoding="utf-8")
        res = pf.gate_cfm_live_decode_canary(
            "p1_cfm_live_decode_canary",
            "P1",
            "cfm",
            stub,
            out_dir=tmp_path / "out_verdict",
        )
        assert res.status == "FAIL"
        assert "decode_ratio" in res.evidence
        assert "live cfm decode canary" in res.evidence

    def test_nonzero_rc_fails(self, tmp_path):
        pf = _pf()
        stub = tmp_path / "stub_rc1.py"
        stub.write_text(STUB_RC1, encoding="utf-8")
        res = pf.gate_cfm_live_decode_canary(
            "p1_cfm_live_decode_canary",
            "P1",
            "cfm",
            stub,
            out_dir=tmp_path / "out_rc1",
        )
        assert res.status == "FAIL"
        assert "rc=1" in res.evidence

    def test_missing_script_fails(self, tmp_path):
        pf = _pf()
        res = pf.gate_cfm_live_decode_canary(
            "p1_cfm_live_decode_canary",
            "P1",
            "cfm",
            tmp_path / "no_such_gate.py",
            out_dir=tmp_path / "out_missing",
        )
        assert res.status == "FAIL"
        assert "not on disk" in res.evidence
