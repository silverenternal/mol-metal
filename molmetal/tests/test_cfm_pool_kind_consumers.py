# Licensed under the Apache License, Version 2.0.
"""Wf-A114 CLUSTER P — the last two places a RECONSTRUCTION pool and a
GENERATION pool could be mixed, compared or averaged together.

``molmetal/scripts/cfm_pool_kind.py`` owns the rule.  A pool whose CSV header
carries ``n_frags_raw`` is the trainer's free-generation output; a pool whose
header carries ``metal_z`` is decided by its CONTENT, because
``wf_cfm_decode_dump_smiles.py`` wrote ONE header in TWO eras -- before
2026-10-12 it ran only the bond head on REAL tmQM geometries (a
reconstruction), from 2026-10-12 it called the trainer's ODE sampler (free
generation).  The two kinds are not comparable: a reconstruction pool
inherits the chemistry it started from.

Two consumers were still mixing them:

* ``cfm_sa_yield_audit.py`` pooled all three of its pools into ONE gate rate
  (``n_total = sum(n_scored)``), so a reconstruction pool's molecules sat in
  the generator's denominator;
* ``cfm_generation_metrics.py`` published a reconstruction pool's numbers
  under the artifact name ``cfm_generation_metrics`` as "generation-side
  metrics" (novelty / diversity / IntDiv1 / metal compliance).

These tests pin the three properties that make the fix safe:

1. the two kinds are classified correctly and are NEVER pooled into one rate;
2. an UNKNOWN pool (a header with neither marker, or a ``metal_z`` pool
   whose content signals disagree) is REFUSED with a reason, never silently
   bucketed into a kind;
3. the payload carries the kind, the reason and the source of the rule NEXT
   TO the numbers, so a reader cannot take a decoder-fidelity number for a
   generator number.

THE REGRESSION TEST is the one that pins the exact defect: the pool of
record (``cfm_fullcorpus_2026_10_10``, 66 rows) is FREE GENERATION, and the
two full-corpus pools of 2026-10-11 are RECONSTRUCTIONS.

The real on-disk pool is asserted by its CLASSIFICATION only — a chemistry
number is not this test's business.

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

from molmetal.scripts import cfm_generation_metrics as genm  # noqa: E402
from molmetal.scripts import cfm_sa_yield_audit as sa  # noqa: E402
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    CONTENT_MARKER,
    KIND_GENERATION,
    KIND_RECONSTRUCTION,
    KIND_UNKNOWN,
    classify_pool,
)

REPORTS = REPO_ROOT / "molmetal" / "reports"
POOL_KIND_SOURCE = "molmetal/scripts/cfm_pool_kind.py"

#: An era-A reconstruction pool: ``metal_z`` in the header AND the CONTENT
#: signals of the bond-head path (``sanitized == 1``, a populated
#: ``metal_z``).  A ``metal_z`` header ALONE is not decidable -- one script
#: wrote one header in two eras -- so a synthetic pool must carry the era-A
#: content to be a reconstruction at all.
RECON_HEADER = ["epoch", "smiles", "sanitized", "metal_z", "n_fragments"]
#: A generation pool carries the trainer's free-generation marker.
GEN_HEADER = ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
#: Neither marker: the pool's provenance is not established.
UNKNOWN_HEADER = ["epoch", "smiles", "charge"]

#: A trivially small molecule (SA well under the 4.5 gate) and a cage
#: polycycle (SA well over it), so a synthetic pool can have a non-zero gate
#: count without borrowing a number from the real corpus.
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


def _generation_pool(tmp_path: Path, name: str = "gen") -> Path:
    return _write_pool(
        tmp_path / name / "generated_smiles.csv",
        GEN_HEADER,
        [
            {"epoch": 0, "smiles": EASY, "n_frags_raw": 1, "largest_frag_frac": 1.0},
            {"epoch": 0, "smiles": HARD, "n_frags_raw": 1, "largest_frag_frac": 1.0},
            {"epoch": 0, "smiles": HARD, "n_frags_raw": 1, "largest_frag_frac": 1.0},
        ],
    )


def _reconstruction_pool(tmp_path: Path, name: str = "recon") -> Path:
    """An era-A reconstruction pool (bond head on a real tmQM geometry)."""
    return _write_pool(
        tmp_path / name / "generated_smiles.csv",
        RECON_HEADER,
        [
            {"epoch": 0, "smiles": HARD, "sanitized": 1, "metal_z": "77", "n_fragments": 1},
            {"epoch": 0, "smiles": HARD, "sanitized": 1, "metal_z": "77", "n_fragments": 1},
        ],
    )


def _unknown_pool(tmp_path: Path, name: str = "unknown") -> Path:
    return _write_pool(
        tmp_path / name / "generated_smiles.csv",
        UNKNOWN_HEADER,
        [{"epoch": 0, "smiles": EASY, "charge": "0"}],
    )


@pytest.fixture(autouse=True)
def _no_corpus_scan(monkeypatch):
    """The novelty half of the SA gate reads an 18k-row corpus; stub it.

    The pool-kind separation is what these tests are about; the Morgan
    fingerprints of the training corpus are not.
    """
    monkeypatch.setattr(
        sa,
        "novelty_of",
        lambda smiles, train_path: (
            {"n": len(smiles), "mean_novelty": 0.5, "stubbed": True}
            if smiles
            else None
        ),
    )


# --- the classification -----------------------------------------------------


def test_a_synthetic_generation_pool_is_classified_as_generation(tmp_path):
    kind = classify_pool(_generation_pool(tmp_path))
    assert kind.kind == KIND_GENERATION


def test_a_synthetic_reconstruction_pool_is_classified_as_reconstruction(tmp_path):
    kind = classify_pool(_reconstruction_pool(tmp_path))
    assert kind.kind == KIND_RECONSTRUCTION


def test_a_pool_with_neither_marker_is_unknown(tmp_path):
    kind = classify_pool(_unknown_pool(tmp_path))
    assert kind.kind == KIND_UNKNOWN
    # The reason names the markers that were looked for.
    assert "metal_z" in kind.reason and "n_frags_raw" in kind.reason


# --- the SA audit never pools the two kinds ---------------------------------


def test_the_sa_audit_never_pools_a_reconstruction_pool_with_a_generation_pool(
    tmp_path, monkeypatch
):
    gen = _generation_pool(tmp_path, "gen")
    recon = _reconstruction_pool(tmp_path, "recon")
    monkeypatch.setattr(
        sa,
        "DEFAULT_POOLS",
        [("gen", str(gen)), ("recon", str(recon))],
    )
    # The real reconstruction pools on disk are not part of this scenario.
    monkeypatch.setattr(sa, "RECONSTRUCTION_POOLS", [])
    out = tmp_path / "cfm_sa_guided.json"
    assert sa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())

    by_kind = payload["combined_by_kind"]
    assert set(by_kind) == {KIND_GENERATION, KIND_RECONSTRUCTION}

    # Each block holds ONE pool, and only that pool.
    assert by_kind[KIND_GENERATION]["pools"] == ["gen"]
    assert by_kind[KIND_RECONSTRUCTION]["pools"] == ["recon"]
    assert by_kind[KIND_GENERATION]["n_scored"] == 3
    assert by_kind[KIND_RECONSTRUCTION]["n_scored"] == 2

    # The headline rate is the GENERATION one, and it says so.
    assert payload["combined"]["scope"] == "generation_only"
    assert payload["combined"]["pools_included"] == ["gen"]
    assert payload["combined"]["pools_excluded"] == ["recon"]
    assert payload["combined"]["n_scored"] == 3

    # The test is NOT vacuous: the mixed rate would be a different number.
    mixed_n = 3 + 2
    mixed_gate = int(by_kind[KIND_GENERATION]["n_sa_le_gate"]) + int(
        by_kind[KIND_RECONSTRUCTION]["n_sa_le_gate"]
    )
    assert mixed_gate / mixed_n != payload["combined"]["rate_sa_le_gate"]
    assert payload["combined"]["rate_sa_le_gate"] == round(
        int(by_kind[KIND_GENERATION]["n_sa_le_gate"]) / 3, 4
    )

    # Every pool row carries its kind and the reason for it.
    rows = {p["label"]: p for p in payload["pools"]}
    assert rows["gen"]["pool_kind"] == KIND_GENERATION
    assert rows["recon"]["pool_kind"] == KIND_RECONSTRUCTION
    assert rows["recon"]["pool_kind_source"] == POOL_KIND_SOURCE
    assert rows["recon"]["verdict_scope"] == "decoder_fidelity_on_real_geometries"


def test_the_sa_audit_never_drops_a_pool_without_a_known_kind():
    # A pool that does not declare a known kind must not vanish into a
    # denominator: the aggregation refuses it instead of quietly reducing
    # the set the rate is computed over.
    # The control: a properly classified pool is aggregated normally.
    by_kind = sa.combined_by_kind(
        [{"label": "gen", "pool_kind": KIND_GENERATION, "n_scored": 3}]
    )
    assert by_kind[KIND_GENERATION]["n_scored"] == 3

    with pytest.raises(ValueError):
        sa.combined_by_kind(
            [{"label": "mystery", "pool_kind": KIND_UNKNOWN, "n_scored": 3}]
        )
    with pytest.raises(ValueError):
        sa.combined_by_kind([{"label": "mystery", "n_scored": 3}])


def test_the_sa_audit_refuses_an_unknown_pool_with_a_reason(tmp_path, monkeypatch):
    gen = _generation_pool(tmp_path, "gen")
    unknown = _unknown_pool(tmp_path, "unknown")
    monkeypatch.setattr(
        sa,
        "DEFAULT_POOLS",
        [("gen", str(gen)), ("unknown", str(unknown))],
    )
    monkeypatch.setattr(sa, "RECONSTRUCTION_POOLS", [])
    out = tmp_path / "cfm_sa_guided.json"
    rc = sa.main(["--out", str(out)])
    payload = json.loads(out.read_text())

    # The refusal is loud (exit 2, the repo's SKIP code).
    assert rc == 2

    # It is recorded, with a reason.
    assert payload["n_refused_pools"] == 1
    refusal = payload["refused_pools"][0]
    assert refusal["label"] == "unknown"
    assert refusal["refusal"] == "pool_kind_unknown"
    assert refusal["reason"]
    assert "metal_z" in refusal["reason"] and "n_frags_raw" in refusal["reason"]

    # It was NOT bucketed: it is in no kind's pool list and in no rate.
    assert payload["combined_by_kind"][KIND_GENERATION]["pools"] == ["gen"]
    assert payload["combined_by_kind"][KIND_RECONSTRUCTION]["pools"] == []
    assert "unknown" not in [
        label
        for block in payload["combined_by_kind"].values()
        for label in block["pools"]
    ]
    # It was not scored either: the generation denominator is the generation
    # pool alone.
    assert payload["combined"]["n_scored"] == 3


# --- the generation-metrics artifact refuses to publish a reconstruction -----


def _tiny_train(tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["smiles", "split"])
        writer.writeheader()
        for smiles in ("CCO", "c1ccccc1", "CC(=O)O"):
            writer.writerow({"smiles": smiles, "split": "train"})
    return path


def _generation_metrics_deps_present() -> bool:
    base = REPO_ROOT / "metrics" / "by_metric"
    return all(
        (base / name).is_file()
        for name in (
            "cfm_free_generation_panel_seed0.json",
            "m7_diversity.json",
            "cfm_vina_generated.json",
        )
    )


@pytest.mark.skipif(
    not _generation_metrics_deps_present(),
    reason="the generation-metrics script reads three sibling metric artifacts",
)
def test_the_generation_metrics_artifact_refuses_to_publish_a_reconstruction_pool(
    tmp_path,
):
    pool = _reconstruction_pool(tmp_path, "recon")
    out = tmp_path / "cfm_generation_metrics.json"
    assert genm.main(
        [
            "--generated",
            str(pool),
            "--train",
            str(_tiny_train(tmp_path)),
            "--out",
            str(out),
        ]
    ) == 0
    payload = json.loads(out.read_text())

    # The kind, the reason and the source of the rule travel WITH the numbers.
    assert payload["pool_kind"]["kind"] == KIND_RECONSTRUCTION
    # A content-decided pool's marker is the CONTENT signal, not the header
    # marker: the header alone cannot decide a metal_z pool.
    assert payload["pool_kind"]["marker"] == CONTENT_MARKER
    assert payload["pool_kind"]["source"] == POOL_KIND_SOURCE
    assert payload["pool_kind"]["reason"]
    assert payload["pool_kind"]["verdict_scope"] == "decoder_fidelity_on_real_geometries"

    # The generation-side metric set is NOT published ...
    assert payload["generation_side"]["published"] is False
    assert payload["generation_side"]["reason"]
    assert payload["subject"]["generation_side_published"] is False
    assert payload["subject"]["generation_side_refusal_reason"]

    # ... and there is no generation-side verdict.
    assert payload["verdict"] == "MEASURED_NOT_A_CLAIM"
    assert payload["verdict_detail"]["gate_met"] is None

    # Every measured block says which kind it belongs to.
    assert payload["novelty"]["scope"] == "decoder_fidelity_on_real_geometries"
    assert payload["cohort"]["scope"] == "decoder_fidelity_on_real_geometries"
    assert payload["metal_compliance"]["scope"] == "decoder_fidelity_on_real_geometries"
    assert payload["checks"]["intdiv1"]["scope"] == "decoder_fidelity_on_real_geometries"


@pytest.mark.skipif(
    not _generation_metrics_deps_present(),
    reason="the generation-metrics script reads three sibling metric artifacts",
)
def test_the_generation_metrics_artifact_publishes_a_generation_pool(tmp_path):
    pool = _generation_pool(tmp_path, "gen")
    out = tmp_path / "cfm_generation_metrics.json"
    assert genm.main(
        [
            "--generated",
            str(pool),
            "--train",
            str(_tiny_train(tmp_path)),
            "--out",
            str(out),
        ]
    ) == 0
    payload = json.loads(out.read_text())

    assert payload["pool_kind"]["kind"] == KIND_GENERATION
    assert payload["generation_side"]["published"] is True
    assert payload["verdict"] in {"PASS", "NEGATIVE"}
    assert payload["novelty"]["scope"] == "generator_fragmentation"
    assert payload["checks"]["intdiv1"]["scope"] == "generator_fragmentation"


@pytest.mark.skipif(
    not _generation_metrics_deps_present(),
    reason="the generation-metrics script reads three sibling metric artifacts",
)
def test_the_generation_metrics_artifact_refuses_an_unknown_pool(tmp_path):
    pool = _unknown_pool(tmp_path, "unknown")
    out = tmp_path / "cfm_generation_metrics.json"
    rc = genm.main(
        [
            "--generated",
            str(pool),
            "--train",
            str(_tiny_train(tmp_path)),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    payload = json.loads(out.read_text())

    # Refused with a reason, and NOT bucketed into a kind.
    assert payload["refusal"]["refusal"] == "pool_kind_unknown"
    assert payload["refusal"]["reason"]
    assert payload["pool_kind"]["kind"] == KIND_UNKNOWN
    assert payload["pool_kind"]["source"] == POOL_KIND_SOURCE
    assert payload["generation_side"]["published"] is False

    # No metric is published from an unclassified pool.
    assert "novelty" not in payload
    assert "metal_compliance" not in payload
    assert "checks" not in payload


# --- the real pools on disk -------------------------------------------------


def test_the_real_fullcorpus_pool_is_classified_as_generation():
    # THE REGRESSION TEST FOR THE DEFECT (Wf-A114D).
    #
    # MEASURED 2026-10-30: the default pool of BOTH consumers
    # (cfm_fullcorpus_2026_10_10) carries 'metal_z' in its header -- but the
    # header alone does not decide the kind, because the decode-dump script
    # wrote ONE header in TWO eras.  Every one of its 66 rows carries
    # 'sanitized = 0' and an EMPTY 'metal_z', which is the structural
    # signature of the 2026-10-12+ ODE path, so the pool is FREE GENERATION.
    # The project's own panel artifact agrees: 192 attempts, 66 valid,
    # gen_validity 0.344.
    pool = REPORTS / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
    assert pool.is_file()
    kind = classify_pool(pool)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == CONTENT_MARKER
    assert "era B" in kind.reason


def test_the_two_2026_10_11_full_corpus_pools_are_classified_as_reconstruction():
    # The two pools written BEFORE the 2026-10-12 fix: the bond head applied
    # to real tmQM geometries, never the sampler.  Their rows carry the
    # bond head's own 'sanitized = 1' and the cloud's metal, which is the
    # era-A signature.
    for name in ("cfm_fullcorpus_b64_2026_10_11", "cfm_objective_terms_2026_10_11"):
        pool = REPORTS / name / "generated_smiles.csv"
        assert pool.is_file()
        kind = classify_pool(pool)
        assert kind.kind == KIND_RECONSTRUCTION
        assert kind.marker == CONTENT_MARKER
        assert "era A" in kind.reason


def test_the_two_pcgrad_pools_are_classified_as_generation():
    for name in ("cfm_pcgrad_on_2026_10_14", "cfm_pcgrad_off_2026_10_15"):
        pool = REPORTS / name / "generated_smiles.csv"
        assert pool.is_file()
        assert classify_pool(pool).kind == KIND_GENERATION


def test_the_sa_audit_pools_are_the_ones_the_defect_named():
    # The defect was the SA audit pooling THESE three pools under one rate.
    # Under the corrected rule ALL THREE are generation pools (the pool of
    # record is the ODE path), so the headline rate is the generator's rate
    # over all 2,396 molecules -- and the reconstruction pools sit in their
    # own block.
    kinds = {
        label: classify_pool(REPO_ROOT / rel).kind
        for label, rel in sa.DEFAULT_POOLS
    }
    assert kinds["panel_of_record"] == KIND_GENERATION
    assert kinds["pcgrad_on"] == KIND_GENERATION
    assert kinds["pcgrad_off"] == KIND_GENERATION


def test_the_sa_audit_scores_the_two_real_reconstruction_pools_separately():
    # The reconstruction pools are REPORTED (their SA numbers are the
    # decoder's fidelity on real chemistry), never pooled with the
    # generator's numbers.
    kinds = {
        label: classify_pool(REPO_ROOT / rel).kind
        for label, rel in sa.RECONSTRUCTION_POOLS
    }
    assert kinds == {
        "recon_fullcorpus_b64": KIND_RECONSTRUCTION,
        "recon_objective_terms": KIND_RECONSTRUCTION,
    }
