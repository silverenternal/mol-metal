#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A120 / I2 -- the SA-yield audit must TIER its numbers by cohort scale.

The defect this file pins: ``metrics/by_metric/cfm_sa_guided.json`` published
ONE ``combined`` rate (16/2396 = 0.0067) over three GENERATION pools whose
training cohorts differ by two orders of magnitude -- the n_train=26715
full-corpus panel of record (66 rows, 0 SA<=4.5 hits) and two n_train=256
pools (2330 rows, 16 hits).  Every hit came from the small cohorts, so the
number was a property of the mixture and nothing in the artifact said so.

Three properties are pinned here:

1. the cohort-scale rule has ONE owner (``molmetal/scripts/cfm_cohort_scale.py``)
   with fixed boundaries;
2. the audit publishes the rate PER COHORT TIER (``combined_by_tier``), and
   the reconstruction pools are not in it;
3. the historical ``combined`` block keeps its numbers but is LABELLED
   ``mixed_cohort_scale`` / ``comparable=false``; and the paper table's SA row
   reads the panel of record's OWN rate (``by_pool/panel_of_record``) with
   the cohort named in the label.

CPU-only, RDKit-only, no GPU.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import cfm_cohort_scale as scale  # noqa: E402
from molmetal.scripts import cfm_sa_yield_audit as sa  # noqa: E402
from molmetal.scripts import paper_metric_tables as pmt  # noqa: E402

ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "cfm_sa_guided.json"

GEN_HEADER = ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
RECON_HEADER = ["epoch", "smiles", "sanitized", "metal_z", "n_fragments"]
EASY = "CCO"
HARD = "C1CC2CCC3(CCC4(CC(C3)CC4)CC2)CC1"


def _write_pool(path: Path, header, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _generation_pool(tmp_path: Path, name: str, n_rows: int = 3) -> Path:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "epoch": 0,
                "smiles": EASY if i == 0 else HARD,
                "n_frags_raw": 1,
                "largest_frag_frac": 1.0,
            }
        )
    return _write_pool(tmp_path / name / "generated_smiles.csv", GEN_HEADER, rows)


def _reconstruction_pool(tmp_path: Path, name: str) -> Path:
    return _write_pool(
        tmp_path / name / "generated_smiles.csv",
        RECON_HEADER,
        [
            {
                "epoch": 0,
                "smiles": HARD,
                "sanitized": 1,
                "metal_z": "77",
                "n_fragments": 1,
            },
            {
                "epoch": 0,
                "smiles": HARD,
                "sanitized": 1,
                "metal_z": "77",
                "n_fragments": 1,
            },
        ],
    )


def _census(tmp_path: Path, rows) -> Path:
    path = tmp_path / "census" / "cfm_run_census.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metric": "cfm_run_census", "runs": rows}))
    return path


@pytest.fixture(autouse=True)
def _no_corpus_scan(monkeypatch):
    """The novelty half of the gate reads an 18k-row corpus; stub it.

    The cohort-scale tiering is what these tests are about; the Morgan
    fingerprints of the training corpus are not.
    """
    monkeypatch.setattr(
        sa,
        "novelty_of",
        lambda smiles, train_path: (
            {"n": len(smiles), "mean_novelty": 0.5, "stubbed": True} if smiles else None
        ),
    )


# ---------------------------------------------------------------------------
# (a) the rule owner: boundaries
# ---------------------------------------------------------------------------


def test_tier_of_boundaries():
    assert scale.tier_of(26715) == "full_corpus"
    assert scale.tier_of(26714) == "mid"
    assert scale.tier_of(1024) == "mid"
    assert scale.tier_of(1023) == "small"
    assert scale.tier_of(None) == "unknown"


def test_the_rule_owner_exposes_the_thresholds_it_states():
    assert scale.FULL_CORPUS_N_TRAIN == 26715
    assert scale.SMALL_TRAIN_CUTOFF == 1024
    assert scale.TIERS == ("full_corpus", "mid", "small", "unknown")
    assert "26715" in scale.TIER_RULE
    assert "1024" in scale.TIER_RULE


def test_the_census_supplies_n_train_and_the_tier(tmp_path, monkeypatch):
    census = _census(
        tmp_path,
        [
            {"run": "big_run", "n_train": 26715},
            {"run": "small_run", "n_train": 256},
            {"run": "null_run", "n_train": None},
        ],
    )
    monkeypatch.setattr(scale, "CENSUS_PATH", census)
    assert scale.cohort_scale_of_run("big_run") == (26715, "full_corpus")
    assert scale.cohort_scale_of_run("small_run") == (256, "small")
    # A recorded-but-null cohort is UNKNOWN, not a guess.
    assert scale.cohort_scale_of_run("null_run") == (None, "unknown")
    # A run the census never heard of is UNKNOWN too.
    assert scale.cohort_scale_of_run("absent_run") == (None, "unknown")
    assert "not in" in scale.n_train_source("absent_run")


def test_the_c7_rule_and_the_new_owner_agree():
    """The rule has ONE owner; until c7 is wired to import it, the two copies
    must not disagree about any cohort size."""
    from molmetal.scripts import c7_scale_matched_pools as c7

    for n_train in (None, 0, 16, 256, 1023, 1024, 4096, 26714, 26715, 100000):
        assert scale.tier_of(n_train) == c7.tier_of(n_train)


# ---------------------------------------------------------------------------
# (b) + (c) the audit publishes the tiered view
# ---------------------------------------------------------------------------


def test_the_audit_splits_the_generation_pools_by_cohort_tier(tmp_path, monkeypatch):
    big = _generation_pool(tmp_path, "big_run", n_rows=4)
    small = _generation_pool(tmp_path, "small_run", n_rows=6)
    recon = _reconstruction_pool(tmp_path, "recon_run")
    census = _census(
        tmp_path,
        [
            {"run": "big_run", "n_train": 26715},
            {"run": "small_run", "n_train": 256},
            {"run": "recon_run", "n_train": 26715},
        ],
    )
    monkeypatch.setattr(scale, "CENSUS_PATH", census)
    monkeypatch.setattr(
        sa,
        "DEFAULT_POOLS",
        [("panel_of_record", str(big)), ("small_run_pool", str(small))],
    )
    monkeypatch.setattr(sa, "RECONSTRUCTION_POOLS", [("recon_run", str(recon))])
    out = tmp_path / "cfm_sa_guided.json"
    assert sa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())

    by_tier = payload["combined_by_tier"]
    # The two GENERATION cohorts are separate blocks, each with its own
    # denominator.
    assert set(by_tier) == {"full_corpus", "small"}
    assert by_tier["full_corpus"]["pools"] == ["panel_of_record"]
    assert by_tier["small"]["pools"] == ["small_run_pool"]
    assert by_tier["full_corpus"]["n_scored"] == 4
    assert by_tier["small"]["n_scored"] == 6
    assert by_tier["full_corpus"]["n_train"] == 26715
    assert by_tier["small"]["n_train"] == 256

    # The RECONSTRUCTION pool is NOT in the tier blocks: the tier axis is
    # the GENERATION cohorts only, and the reconstruction pools keep their
    # own combined_by_kind block.
    assert "recon_run" not in [
        label for block in by_tier.values() for label in block["pools"]
    ]
    assert payload["combined_by_kind"]["reconstruction"]["pools"] == ["recon_run"]

    # Every pool row carries its own tier and n_train.
    rows = {p["label"]: p for p in payload["pools"]}
    assert rows["panel_of_record"]["tier"] == "full_corpus"
    assert rows["panel_of_record"]["n_train"] == 26715
    assert rows["small_run_pool"]["tier"] == "small"
    assert rows["small_run_pool"]["n_train"] == 256
    assert rows["recon_run"]["tier"] == "full_corpus"


def test_the_combined_block_is_labelled_as_a_mixed_cohort_scale(tmp_path, monkeypatch):
    big = _generation_pool(tmp_path, "big_run", n_rows=4)
    small = _generation_pool(tmp_path, "small_run", n_rows=6)
    census = _census(
        tmp_path,
        [{"run": "big_run", "n_train": 26715}, {"run": "small_run", "n_train": 256}],
    )
    monkeypatch.setattr(scale, "CENSUS_PATH", census)
    monkeypatch.setattr(
        sa,
        "DEFAULT_POOLS",
        [("panel_of_record", str(big)), ("small_run_pool", str(small))],
    )
    monkeypatch.setattr(sa, "RECONSTRUCTION_POOLS", [])
    out = tmp_path / "cfm_sa_guided.json"
    assert sa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())

    combined = payload["combined"]
    assert combined["scale_label"] == "mixed_cohort_scale"
    assert combined["comparable"] is False
    assert combined["cohort_scales_included"] == {
        "panel_of_record": 26715,
        "small_run_pool": 256,
    }
    # The note says what the denominator mixes, in numbers.
    assert "26715" in combined["note"] and "256" in combined["note"]
    assert "mixed" in combined["note"].lower()


def test_the_by_pool_view_carries_each_pool_own_rate(tmp_path, monkeypatch):
    big = _generation_pool(tmp_path, "big_run", n_rows=4)
    census = _census(tmp_path, [{"run": "big_run", "n_train": 26715}])
    monkeypatch.setattr(scale, "CENSUS_PATH", census)
    monkeypatch.setattr(sa, "DEFAULT_POOLS", [("panel_of_record", str(big))])
    monkeypatch.setattr(sa, "RECONSTRUCTION_POOLS", [])
    out = tmp_path / "cfm_sa_guided.json"
    assert sa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())

    block = payload["by_pool"]["panel_of_record"]
    assert block["n_scored"] == 4
    assert block["n_sa_le_gate"] == 1
    assert block["rate_sa_le_gate"] == 0.25
    assert block["wilson_ci95"] == sa.wilson(1, 4)
    assert block["tier"] == "full_corpus"
    assert block["n_train"] == 26715
    assert block["pool_path"].endswith("generated_smiles.csv")


# ---------------------------------------------------------------------------
# the SHIPPED artifact
# ---------------------------------------------------------------------------


def test_the_shipped_artifact_keeps_its_historical_combined_numbers():
    payload = json.loads(ARTIFACT.read_text())
    combined = payload["combined"]
    # Unchanged on purpose: no existing consumer breaks.
    assert combined["n_scored"] == 2396
    assert combined["n_sa_le_gate"] == 16
    assert combined["rate_sa_le_gate"] == 0.0067
    assert combined["wilson_ci95"] == [0.0041, 0.0108]
    # ... and now labelled.
    assert combined["scale_label"] == "mixed_cohort_scale"
    assert combined["comparable"] is False


def test_the_shipped_artifact_splits_the_generation_pools_by_tier():
    payload = json.loads(ARTIFACT.read_text())
    by_tier = payload["combined_by_tier"]
    assert set(by_tier) == {"full_corpus", "small"}
    # The full-corpus panel of record: 0 hits in 66 molecules.
    assert by_tier["full_corpus"]["n_scored"] == 66
    assert by_tier["full_corpus"]["n_sa_le_gate"] == 0
    assert by_tier["full_corpus"]["rate_sa_le_gate"] == 0.0
    assert by_tier["full_corpus"]["n_train"] == 26715
    # The two n_train=256 pools: 16 hits in 2330 molecules.
    assert by_tier["small"]["n_scored"] == 2330
    assert by_tier["small"]["n_sa_le_gate"] == 16
    assert by_tier["small"]["n_train"] == 256
    # The reconstruction pools are NOT in the tier blocks.
    assert "recon_fullcorpus_b64" not in [
        label for block in by_tier.values() for label in block["pools"]
    ]


def test_the_shipped_by_pool_view_is_what_the_paper_table_reads():
    payload = json.loads(ARTIFACT.read_text())
    block = payload["by_pool"]["panel_of_record"]
    assert block["rate_sa_le_gate"] == 0.0
    assert block["n_scored"] == 66
    assert block["n_sa_le_gate"] == 0
    assert block["tier"] == "full_corpus"
    assert block["n_train"] == 26715
    # The small-cohort pools are NOT comparable to the panel of record.
    assert payload["by_pool"]["pcgrad_on"]["tier"] == "small"
    assert payload["by_pool"]["pcgrad_on"]["comparable"] is False


# ---------------------------------------------------------------------------
# (d) the paper table row
# ---------------------------------------------------------------------------


def test_the_t7_sa_row_names_its_cohort_and_reads_the_by_pool_path():
    table = pmt.table_generation()
    rows = [r for r in table.rows if "SA<=4.5" in r.label]
    assert len(rows) == 1
    row = rows[0]
    assert "panel of record" in row.label
    assert "66 survivors" in row.label
    assert row.cells[0].file == "cfm_sa_guided.json"
    assert row.cells[0].pointer == "/by_pool/panel_of_record/rate_sa_le_gate"
    # The mixed-cohort pointer is gone from the table.
    assert all(
        c.pointer != "/combined/rate_sa_le_gate" for r in table.rows for c in r.cells
    )


def test_the_t7_sa_row_resolves_against_the_shipped_artifact(tmp_path):
    cache = pmt.ArtifactCache(pmt.METRICS_DIR)
    table = pmt.table_generation()
    row = next(r for r in table.rows if "SA<=4.5" in r.label)
    text, status, where = row.cells[0].value(cache)
    assert status == "ok", where
    assert text == "0.0000"
