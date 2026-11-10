"""Wf-A204 (2026-11-02) -- the PRODUCER<->ARTEFACT REPRODUCIBILITY regression repair.

THE DEFECT THIS WAVE REPAIRS
----------------------------
``molmetal/scripts/wf_a155_ligand_connected_rate.py`` emits the dated
rule-#6 record ``comparability.superseded_frontier_2026_11_02_wfa194``
for EVERY pool (Wf-A199, 2026-11-02).  The c7-derived artefact already
carried it (hand-patched by Wf-A194); the c8-derived PUBLISHED metric
did NOT, so it was BEHIND its own producer and a producer re-run no
longer reproduced it.

THE FIX
-------
Branch (a) REGENERATE: the record is POOL-INDEPENDENT -- its eight
leaves are static strings with no pool token in any of them, and the
producer's own copy is byte-identical to the c7 artefact's copy -- so it
belongs in every pool's artefact.  Scoping the emission to one pool
would make a pool-independent record pool-dependent and would drop the
rule-#6 history from the c8 artefact.  The published metric is therefore
re-published from the producer, so that it is byte-identical to a
producer re-run.

WHAT THIS FILE GUARDS
---------------------
1. BYTE-IDENTITY (MUST-BITE): a producer re-run on the c8 pool yields a
   file byte-identical to the published metric.  If the published file
   drifts, this fails.
2. The guard against test 1 being vacuous: a perturbed copy of the
   published bytes is NOT byte-identical.
3. The dated 194 record is POOL-INDEPENDENT: the c8 metric's copy and
   the c7 artefact's copy are byte-identical.
4. The dated correction record carries the superseded rendering VERBATIM
   and names the leaf it corrects (AGENTS.md §5.5 rule #6: additive and
   dated, never a silent edit).
5. The dated records carry NO numeric leaf.
6. The NUMERIC-IDENTITY proof: the recorded ``narrative_self_consistency``
   block is exactly what the guard computes on the artefact with the
   dated records removed -- so attaching them moved NO numeric leaf.
7. The MEASURED payload numbers did not move.

CPU-only.  No GPU is touched, queried or trained on: the pool CSV is
read from disk and RDKit parses it on the CPU.  The producer is run as a
SUBPROCESS so that its internals are not imported by this test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO = Path(__file__).resolve().parents[2]

EMITTER_REL = "molmetal/scripts/wf_a155_ligand_connected_rate.py"
EMITTER = REPO / EMITTER_REL

PUBLISHED_REL = "metrics/by_metric/wfa155_ligand_connected_rate.json"
PUBLISHED = REPO / PUBLISHED_REL

C7_REL = (
    "molmetal/reports/wfa162_c7_pool_ligand_connected_rate/"
    "ligand_connected_rate_c7.json"
)
C7_ARTEFACT = REPO / C7_REL

POOL_C8_REL = (
    "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
)
POOL_C8 = REPO / POOL_C8_REL

#: The dated rule-#6 record the Wf-A199 repair made emitter-emitted.
DATED_194_KEY = "superseded_frontier_2026_11_02_wfa194"

#: The dated correction of the 194 record's ``producer_parity`` leaf.
DATED_204_KEY = "superseded_producer_parity_2026_11_02_wfa204"

#: The comparability sub-keys the emitter drops from the guard subject.
COMPARABILITY_DATED_KEYS = (
    "superseded_frontier_2026_11_01_wfa183",
    "superseded_frontier_2026_11_01_wfa189",
    DATED_194_KEY,
)

#: The top-level dated key the emitter drops from the guard subject.
TOP_LEVEL_DATED_KEYS = (DATED_204_KEY,)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run_emitter(out_path: Path) -> bytes:
    """Run the producer as a SUBPROCESS and return the bytes it wrote."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            str(EMITTER),
            "--pool-csv",
            str(POOL_C8),
            "--out",
            str(out_path),
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"the producer failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return out_path.read_bytes()


def _leaf_map(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_leaf_map(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_leaf_map(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _string_leaves_under(node: Any) -> List[Any]:
    leaves = _leaf_map(node)
    return list(leaves.values())


# ---------------------------------------------------------------------------
# 1. THE MUST-BITE TEST
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
@pytest.mark.skipif(not POOL_C8.is_file(), reason="c8 pool not on disk")
def test_the_published_metric_is_byte_identical_to_a_producer_rerun(tmp_path: Path) -> None:
    """A producer re-run REPRODUCES the published metric, byte for byte.

    This is the property the Wf-A199 emitter change BROKE for the c8
    metric: the producer emitted the dated 194 record for every pool while
    the published c8 metric did not carry it, so the published file was
    behind its producer.
    """
    produced = _run_emitter(tmp_path / "rerun.json")
    published = PUBLISHED.read_bytes()
    assert produced == published, (
        "the published metric is NOT byte-identical to a producer re-run on "
        "the same pool -- it is behind (or ahead of) its producer.\n"
        f"published: {len(published)} bytes, sha256 "
        f"{__import__('hashlib').sha256(published).hexdigest()}\n"
        f"re-run   : {len(produced)} bytes, sha256 "
        f"{__import__('hashlib').sha256(produced).hexdigest()}"
    )


@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
@pytest.mark.skipif(not POOL_C8.is_file(), reason="c8 pool not on disk")
def test_the_byte_identity_check_can_fail(tmp_path: Path) -> None:
    """The guard above must be capable of failing (no vacuous PASS).

    Two independent bites are checked: (a) the producer is DETERMINISTIC --
    two independent re-runs agree byte for byte, so the equality asserted
    above is not luck; (b) a drifted NUMERIC leaf in the published metric is
    detected as a difference, which is exactly the failure mode this wave
    is about.
    """
    first = _run_emitter(tmp_path / "run_a.json")
    second = _run_emitter(tmp_path / "run_b.json")
    assert first == second, "the producer is not deterministic"

    published = PUBLISHED.read_bytes()
    doc = json.loads(published)
    doc["denominators"]["n_analyzed"] = 2113  # a drifted numeric leaf
    drifted = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert drifted != published, "the drift did not change a byte"
    assert drifted != first, (
        "a drifted numeric leaf is NOT detected by the byte comparison -- "
        "the byte-identity guard cannot fail"
    )


# ---------------------------------------------------------------------------
# 2. THE DATED 194 RECORD IS POOL-INDEPENDENT
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_dated_194_record_is_pool_independent() -> None:
    """The 194 record documents a property of the COMPARABILITY block, not
    of one pool: the c8 metric's copy and the c7 artefact's copy are
    byte-identical.  That is why the emission is not pool-scoped."""
    c8 = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    c7 = json.loads(C7_ARTEFACT.read_text(encoding="utf-8"))
    assert DATED_194_KEY in c8["comparability"], sorted(c8["comparability"])
    assert DATED_194_KEY in c7["comparability"], sorted(c7["comparability"])
    assert json.dumps(c8["comparability"][DATED_194_KEY], ensure_ascii=False) == (
        json.dumps(c7["comparability"][DATED_194_KEY], ensure_ascii=False)
    ), (
        "the c8 metric's dated 194 record differs from the c7 artefact's -- "
        "the record is supposed to be POOL-INDEPENDENT, so it must not "
        "mention the pool at all"
    )


@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
def test_the_dated_194_record_names_no_pool() -> None:
    """No pool token appears in any of the 194 record's leaves."""
    doc = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    record = doc["comparability"][DATED_194_KEY]
    blob = json.dumps(record, ensure_ascii=False)
    for token in ("cfm_c8", "cfm_c7", "2112", "12668"):
        assert token not in blob, f"the dated 194 record names the pool token {token!r}"


# ---------------------------------------------------------------------------
# 3. THE DATED CORRECTION OF THE `producer_parity` LEAF
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
def test_the_dated_correction_reaches_the_artefact_and_preserves_the_old_leaf() -> None:
    """The 194 record's ``producer_parity`` leaf says the dated key is NOT
    emitter-emitted and that a producer re-run would DROP it.  That was
    true when Wf-A194 hand-patched the artefact; after the Wf-A199 emitter
    repair it is false for the artefact it lives in.  Rule #6 forbids
    editing the leaf, so the correction is a DATED SIBLING -- and it must
    REACH the artefact, or the artefact still publishes the false claim.
    """
    doc = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert DATED_204_KEY in doc, (
        f"the artefact carries no dated correction record ({DATED_204_KEY}); "
        "the superseded `producer_parity` leaf is still uncorrected in the "
        "published metric"
    )
    correction = doc[DATED_204_KEY]
    assert correction["date"] == "2026-11-02", correction
    assert correction["wave"] == "Wf-A204", correction
    assert (
        correction["field"]
        == "comparability.superseded_frontier_2026_11_02_wfa194.producer_parity"
    ), correction["field"]
    superseded = doc["comparability"][DATED_194_KEY]["producer_parity"]
    assert correction["rendering"] == superseded, (
        "the dated correction does NOT preserve the superseded rendering "
        "VERBATIM (AGENTS.md §5.5 rule #6)"
    )


@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
def test_the_dated_records_carry_no_numeric_leaf() -> None:
    """Every leaf of both dated records is a STRING -- so attaching them
    cannot move a numeric leaf of the artefact."""
    doc = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    records = {
        DATED_194_KEY: doc["comparability"][DATED_194_KEY],
        DATED_204_KEY: doc[DATED_204_KEY],
    }
    for name, record in records.items():
        for leaf in _string_leaves_under(record):
            assert isinstance(leaf, str), f"{name} carries a non-string leaf {leaf!r}"


# ---------------------------------------------------------------------------
# 4. THE NUMERIC-IDENTITY PROOF
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
def test_the_recorded_guard_block_is_the_pre_dated_record_subject() -> None:
    """The durable numeric-identity proof.

    The recorded ``narrative_self_consistency`` block is the guard's own
    output computed on the artefact BEFORE the dated records were
    attached.  Recomputing it on the published artefact with the dated
    records removed must reproduce the recorded block EXACTLY -- which is
    what it means for the re-publication to have moved no numeric leaf.
    """
    from molmetal.scripts import wf_a155_ligand_connected_rate as tool

    doc = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    recorded = doc["narrative_self_consistency"]
    subject = dict(doc)
    subject.pop("narrative_self_consistency", None)
    for key in TOP_LEVEL_DATED_KEYS:
        subject.pop(key, None)
    subject["comparability"] = {
        k: v
        for k, v in doc["comparability"].items()
        if k not in COMPARABILITY_DATED_KEYS
    }
    recomputed = tool._narrative_self_consistency(subject)
    assert recomputed == recorded, (
        "the recorded guard block is NOT what the guard computes on the "
        "pre-dated-record subject -- the re-publication moved a numeric leaf"
    )
    assert recorded["pass"] is True, recorded


@pytest.mark.skipif(not PUBLISHED.is_file(), reason="published metric not on disk")
def test_the_measured_payload_numbers_did_not_move() -> None:
    """The MEASURED payload of the metric is unchanged by the re-publication."""
    doc = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert doc["denominators"]["n_analyzed"] == 2112
    assert doc["denominators"]["n_rows_in_csv"] == 2112
    assert doc["denominators"]["n_with_free_halide_counterion"] == 60
    assert doc["metal_inclusive"]["n_single_fragment"] == 529
    assert doc["metal_inclusive"]["single_fragment_rate"] == 0.25047348484848486
    assert doc["ligand_only"]["n_single_fragment"] == 60
    assert doc["ligand_only"]["single_fragment_rate"] == 0.028409090909090908
    assert doc["source_pool"] == POOL_C8_REL
