# SPDX-License-Identifier: Apache-2.0
"""Tests for the narrative-claims provenance checker (AGENTS.md §5.2).

The checker exists because hand-enforced traceability failed: during the
2026-10-10 audit a stale per-cell NCI-60 value (PC3's 0.870 labelled A549)
had been copied from ALIGNED_COMPARISON.md into the pinned narrative, and
n_distinct's headline "20" turned out to be a saturated n_top_k cap rather
than a measurement. Neither was visible to a reader who trusted the prose.

So the tests here are mostly about the checker itself being trustworthy — a
provenance gate that cannot fail is worse than no gate, because it manufactures
confidence. Two properties are pinned:

  1. it PASSES on the real claims file, against real data;
  2. it FAILS when a stated value drifts, and when a field path stops
     resolving — verified by mutating a COPY, never the committed file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLAIMS = REPO_ROOT / "registry" / "narrative_claims.yaml"
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"


@pytest.fixture(scope="module")
def claims_module():
    import sys

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import verify_narrative_claims as mod

    return mod


def test_claims_file_exists_and_parses(claims_module):
    claims = claims_module._load_claims(CLAIMS)
    assert len(claims) >= 20, f"expected a substantive claims file, got {len(claims)}"
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids)), "duplicate claim ids"


def test_real_claims_all_verify(claims_module):
    rep = claims_module.verify(CLAIMS)
    drift = [r for r in rep["rows"] if r["status"] == "DRIFT"]
    unver = [r for r in rep["rows"] if r["status"] == "UNVERIFIABLE"]
    assert not drift, "claims drifted from disk:\n" + "\n".join(
        f"  {r['id']}: stated={r['stated']} disk={r.get('disk')}" for r in drift
    )
    assert not unver, "claims unverifiable (source or field missing):\n" + "\n".join(
        f"  {r['id']}: {r['reason']}" for r in unver
    )
    assert rep["verdict"] == "PASS"


def test_checker_detects_value_drift(claims_module, tmp_path):
    """A stated value that no longer matches disk must be reported as DRIFT."""
    import yaml

    doc = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    target = doc["claims"][0]
    target["value"] = float(target["value"]) + 0.5
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(doc), encoding="utf-8")

    rep = claims_module.verify(mutated)
    drifted = {r["id"] for r in rep["rows"] if r["status"] == "DRIFT"}
    assert target["id"] in drifted, (
        "checker did not flag an injected drift — the gate cannot be trusted"
    )
    assert rep["verdict"] == "DRIFT"


def test_checker_detects_missing_field(claims_module, tmp_path):
    """A field path that stops resolving must be UNVERIFIABLE, not silently ok."""
    import yaml

    doc = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    doc["claims"][0]["field"] = "definitely.not.a.real.field"
    mutated = tmp_path / "badfield.yaml"
    mutated.write_text(yaml.safe_dump(doc), encoding="utf-8")

    rep = claims_module.verify(mutated)
    unver = {r["id"] for r in rep["rows"] if r["status"] == "UNVERIFIABLE"}
    assert doc["claims"][0]["id"] in unver


def test_checker_rejects_malformed_claims_file(claims_module, tmp_path):
    """A malformed file is a CONFIG error, distinct from drift."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("claims:\n  - id: x\n", encoding="utf-8")  # missing keys
    with pytest.raises(claims_module.ConfigError):
        claims_module._load_claims(bad)


def test_capped_claim_is_flagged_as_capped():
    """n_distinct's headline must carry an explicit cap warning.

    This is the claim that was most misleading: 20 is aggregate.n_top_k_cap,
    so rendering it as "20 vs gate 3 = 6.7x" presents a ceiling as a
    measurement. If someone drops the note, this fails.
    """
    import yaml

    doc = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in doc["claims"]}
    capped = by_id["n_distinct_capped"]
    assert capped["field"].endswith("best_measured_with_all_fixes")
    assert "cap" in capped["note"].lower(), (
        "n_distinct_capped lost its cap warning — the narrative will present a "
        "saturated ceiling as a measurement"
    )


def test_nci60_cell_count_claim_exists():
    """The 8-vs-9 cell discrepancy must stay asserted, not drift silently.

    It moved 8 -> 9 on 2026-10-12: DU145's per-seed AUC was null under the old
    splitter and is measured (n_test=7) under the fixed one.  Pin the claim
    against the artifact it cites, so a future edit to either is caught.
    """
    import yaml

    doc = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in doc["claims"]}
    claim = by_id["nci60_cells_with_auc"]
    assert claim["value"] == 9
    assert claim["field"] == "n_cells_with_measured_auc"

    artifact = json.loads(
        (REPO_ROOT / claim["source"]).read_text(encoding="utf-8")
    )
    assert artifact[claim["field"]] == claim["value"]
    # The cells that are measured but unresolvable are named, not hidden.
    unresolvable = {
        c
        for c, v in artifact["per_cell_line"].items()
        if v["n_test"] < 10
    }
    assert unresolvable == {"DU145", "SKOV3"}


def test_preflight_registers_the_gate():
    """The checker must be wired into pre-flight, not merely exist.

    An unwired provenance script is the same failure mode as an ungated rule:
    it looks like enforcement and enforces nothing.
    """
    src = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    assert "p3_narrative_claims_check" in src
    assert "verify_narrative_claims.py" in src


def test_expected_gate_counts_include_the_new_gate():
    """Gate-count drift must not be silently tolerated."""
    import re

    src = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    block = re.search(r"EXPECTED_GATE_COUNTS = \{(.*?)\n\}", src, re.S)
    assert block, "EXPECTED_GATE_COUNTS not found"
    import sys

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import pre_flight_check as pf

    # SUPERSEDED (2026-11-04, Wf-A269 stale-pins sweep — PRESERVED VERBATIM
    # per AGENTS.md §5.5 rule #6).  The old extraction read the FIRST
    # `"all": N` inside the captured block, comment lines included:
    #     m = re.search(r'"all": (\d+)', block.group(1))
    # THAT PREMISE IS FALSE.  The Wf-A269 wave-19 repair bumped the live
    # constant to 133 and, per AGENTS.md §5.5 rule #6, preserved the
    # superseded rendering VERBATIM as a COMMENT directly above it:
    #     # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #     #     "all": 132,
    #     "all": 133,
    # so the old regex read the PRESERVED historical rendering (132) as the
    # live value and FAILED against len(build_gates("all")) == 133 — a
    # §5.5-COMPLIANT edit produced a RED test.
    # REPAIRED on the CONSUMER side: the comment lines are dropped before
    # the live value is read.  The producer is NOT touched (the preserved
    # rendering stays exactly as it is) and the guard keeps its teeth — the
    # LIVE constant is still compared against len(build_gates("all")).
    live_block = "\n".join(
        line
        for line in block.group(1).splitlines()
        if not line.lstrip().startswith("#")
    )
    m = re.search(r'"all": (\d+)', live_block)
    assert m is not None, "no live \"all\" count found in EXPECTED_GATE_COUNTS"
    assert int(m.group(1)) == len(pf.build_gates("all")), (
        "EXPECTED_GATE_COUNTS['all'] does not match build_gates('all') — "
        "update it and TODO/completed/321d_preflight_gate_audit.md in the same edit"
    )
