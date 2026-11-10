#!/usr/bin/env python3
"""Wf-A164 — the ``wfa142_fragflex_cminus1_probe`` narrative must be MEASURED.

The defect class (Wf-A155 -> Wf-A162 -> Wf-A163): a measurement tool embeds
POOL-SPECIFIC claims as string literals inside a ``tag: MEASURED`` artifact.
For the tool's DEFAULT pool the literals are TRUE; for ANY OTHER pool — or
any other ``--cohort`` — the tool emitted FALSE statements.  This probe is
the worst instance of the class found so far, because its flags select a
COHORT, not just a pool: ``--cohort`` picks a subset of the pool, so the
cohort's row count is a property of BOTH the pool and the cohort.

What this test asserts (CPU-only, no GPU, no decode):

1.  A synthetic pool under a DIFFERENT directory name, with a DIFFERENT
    epoch count and DIFFERENT per-epoch row counts, produces an artifact
    whose NARRATIVE carries NONE of the default pool's literals ("704",
    "12,669", "18 epochs", "epoch 17") and DOES carry the synthetic
    cohort's own numbers.
2.  The ``--cohort`` HELP TEXT carries none of those literals either (a
    help string is prose a reader will quote).
3.  NEGATIVE CONTROL: restoring the old hard-coded literal makes the run
    FAIL (exit 1, ``status: FAIL``, the token reported as unbacked) — i.e.
    the guard is what makes the repair safe, not a cosmetic edit.
4.  The guard resolves a LIST-valued narrative path to its ELEMENTS (the
    reference guard's hole): a number smuggled into a caveat is caught.
5.  ``cohort_detail`` is re-derivable from ``cohort_facts`` alone.

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        molmetal/tests/test_wfa164_fragflex_cminus1_narrative.py -q
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from molmetal.scripts import wfa142_fragflex_cminus1_probe as probe  # noqa: E402

#: The literals the pre-repair tool hard-coded.  They are TRUE for the
#: default pool (``molmetal/reports/cfm_c7_fullgen_2026_10_20``, 12,669
#: rows, 18 epochs, 704 rows in epoch 17) and FALSE for everything else.
STALE_LITERALS = ("704", "12,669", "18 epochs", "epoch 17")

_SMILES = (
    "CCO.c1ccccc1",
    "CCN.CC(=O)O",
    "c1ccncc1.CCO",
    "CCOC(=O)C.Clc1ccccc1",
)


def _write_pool(directory: Path, epochs) -> Path:
    """Write a synthetic pool CSV under ``directory`` (a DIFFERENT name)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generated_smiles.csv"
    rows = []
    for epoch in epochs:
        for smiles in _SMILES:
            rows.append(
                {
                    "epoch": epoch,
                    "smiles": smiles,
                    "n_frags_raw": smiles.count(".") + 1,
                    "largest_frag_frac": 0.5,
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "smiles", "n_frags_raw", "largest_frag_frac"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run(pool_csv: Path, out_json: Path, cohort: str, *extra: str) -> int:
    """Run the tool with the corpus pointed at a non-existent file.

    The corpus is irrelevant to the narrative defect and reading the real
    26,632-row corpus would dominate the test's wall time.
    """
    return probe.main(
        [
            "--pool",
            str(pool_csv),
            "--corpus",
            str(out_json.parent / "no_such_corpus.csv"),
            "--out",
            str(out_json),
            "--cohort",
            cohort,
            *extra,
        ]
    )


def _narrative(artifact) -> str:
    """All prose leaves of the artifact, joined."""
    return "\n".join(probe.narrative_text(artifact).values())


# ---------------------------------------------------------------------------
# 1. A different pool + a different cohort -> no default-pool literal
# ---------------------------------------------------------------------------
def test_stratified_cohort_on_synthetic_pool_has_no_stale_literal(tmp_path):
    pool = _write_pool(
        tmp_path / "synthpool_wfa164", epochs=["0", "1", "2", "17"]
    )
    out = tmp_path / "probe.json"
    rc = _run(pool, out, "all_stratified", "--per-epoch-limit", "2")
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "PASS"

    prose = _narrative(artifact)
    for literal in STALE_LITERALS:
        assert literal not in prose, (
            f"stale default-pool literal {literal!r} survived in the "
            f"narrative:\n{prose}"
        )

    # The synthetic cohort's OWN numbers are present.
    assert "2 rows per epoch" in artifact["cohort_detail"]
    assert "4 epochs" in artifact["cohort_detail"]
    assert "8 rows" in artifact["cohort_detail"]
    facts = artifact["cohort_facts"]
    assert facts["n_rows_in_pool"] == 16
    assert facts["n_epochs_in_pool"] == 4
    assert facts["n_rows_in_cohort_selected"] == 8
    assert facts["n_rows_processed"] == 8
    assert artifact["n_rows_in_cohort"] == 8
    assert artifact["pool_n_rows_total"] == 16


def test_all_cohort_on_synthetic_pool_has_no_stale_literal(tmp_path):
    pool = _write_pool(tmp_path / "synthpool_wfa164_b", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    rc = _run(pool, out, "all")
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "PASS"
    prose = _narrative(artifact)
    for literal in STALE_LITERALS:
        assert literal not in prose, f"stale literal {literal!r} in:\n{prose}"
    assert artifact["cohort_detail"] == "every row of every epoch (16 rows)"
    assert artifact["n_rows_in_cohort"] == 16


def test_epoch17_cohort_reports_its_own_row_count(tmp_path):
    """The epoch-17 cohort of a 4-row-per-epoch pool is 4 rows, not 704."""
    pool = _write_pool(tmp_path / "synthpool_wfa164_c", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    rc = _run(pool, out, "epoch17")
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "PASS"
    prose = _narrative(artifact)
    assert "704" not in prose
    assert "12,669" not in prose
    assert "18 epochs" not in prose
    # "epoch 17" IS correct here: the synthetic pool carries an epoch 17.
    assert artifact["cohort_detail"] == "epoch 17 rows of the pool (4 rows)"
    assert artifact["n_rows_in_cohort"] == 4


def test_epoch17_cohort_on_pool_without_epoch17_is_empty_not_704(tmp_path):
    """A pool with NO epoch 17 must not claim epoch-17 rows."""
    pool = _write_pool(tmp_path / "synthpool_wfa164_d", epochs=["0", "1", "2", "3"])
    out = tmp_path / "probe.json"
    rc = _run(pool, out, "epoch17")
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "PASS"
    assert artifact["n_rows_in_cohort"] == 0
    prose = _narrative(artifact)
    assert "704" not in prose
    assert "no rows for that epoch" in artifact["cohort_detail"]


# ---------------------------------------------------------------------------
# 2. The --cohort HELP TEXT is prose a reader will quote
# ---------------------------------------------------------------------------
def test_cohort_help_text_carries_no_stale_literal():
    help_text = probe.build_parser().format_help()
    for literal in STALE_LITERALS:
        assert literal not in help_text, (
            f"stale default-pool literal {literal!r} survived in the "
            f"--cohort help text:\n{help_text}"
        )
    assert "--per-epoch-limit" in help_text


# ---------------------------------------------------------------------------
# 3. NEGATIVE CONTROL — the old hard-coded literal must FAIL the run
# ---------------------------------------------------------------------------
def test_negative_control_old_literal_fails_the_run(tmp_path, monkeypatch):
    """Restoring the pre-repair literal makes the run FAIL (exit 1)."""
    pool = _write_pool(tmp_path / "synthpool_wfa164_e", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    monkeypatch.setattr(
        probe,
        "derive_cohort_detail",
        lambda facts: "epoch 17 rows of the pool (704 rows)",
    )
    rc = _run(pool, out, "epoch17")
    assert rc == 1, "the old literal must NOT pass the guard"
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "FAIL"
    assert "704" in artifact["narrative_self_consistency"]["unbacked_numbers"]
    assert artifact["cohort_detail_recheck"]["matches"] is True


def test_negative_control_doctored_artifact_is_flagged(tmp_path):
    """A hand-edited cohort_detail is caught by the guard itself."""
    pool = _write_pool(tmp_path / "synthpool_wfa164_f", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    assert _run(pool, out, "epoch17") == 0
    artifact = json.loads(out.read_text())
    artifact["cohort_detail"] = "epoch 17 rows of the pool (704 rows)"
    guard = probe._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    assert "704" in guard["unbacked_numbers"]
    assert guard["unbacked_numbers"]["704"] == ["cohort_detail"]


def test_recheck_catches_a_detail_that_disagrees_with_the_facts(tmp_path):
    """cohort_detail must be re-derivable from cohort_facts alone."""
    pool = _write_pool(tmp_path / "synthpool_wfa164_g", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    assert _run(pool, out, "epoch17") == 0
    artifact = json.loads(out.read_text())
    facts = dict(artifact["cohort_facts"])
    facts["n_rows_processed"] = 999
    assert probe.derive_cohort_detail(facts) != artifact["cohort_detail"]
    assert artifact["cohort_detail_recheck"]["matches"] is True


# ---------------------------------------------------------------------------
# 4. The guard must resolve a LIST-valued narrative path to its ELEMENTS
# ---------------------------------------------------------------------------
def test_guard_resolves_list_valued_narrative_to_elements(tmp_path):
    pool = _write_pool(tmp_path / "synthpool_wfa164_h", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    assert _run(pool, out, "epoch17") == 0
    artifact = json.loads(out.read_text())

    resolved = probe._resolve(artifact, "caveats")
    assert resolved, "the caveats list must resolve to leaf paths"
    assert all(r.startswith("caveats[") for r in resolved)
    assert len(resolved) == len(artifact["caveats"])

    # A number smuggled into a LIST element must be caught.  With a naive
    # "path + '.'" prefix the list elements would be classified as machine
    # fields and the number would go unchecked.
    artifact["caveats"].append("the pool held 999999 rows")
    guard = probe._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    assert "999999" in guard["unbacked_numbers"]


def test_guard_covers_the_population_statement_and_the_method_strings(tmp_path):
    pool = _write_pool(tmp_path / "synthpool_wfa164_i", epochs=["0", "1", "2", "17"])
    out = tmp_path / "probe.json"
    assert _run(pool, out, "epoch17") == 0
    artifact = json.loads(out.read_text())
    guard = artifact["narrative_self_consistency"]
    assert guard["pass"] is True
    assert guard["n_prose_numbers"] > 0
    assert guard["narrative_paths"] == list(probe.NARRATIVE_PATHS)
    # The method string's "C-1" is backed by the machine method field.
    assert "C-1" in artifact["method"]["edges_added_rule"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
