# SPDX-License-Identifier: Apache-2.0
"""Wf-A133 -- the fragmentation REPORTING MATRIX tool, asserted against disk.

WHAT THIS TEST IS FOR
---------------------
``molmetal/scripts/wfa133_fragmentation_reporting_matrix.py`` computes the
fragmentation reporting matrix from files already on disk, so the matrix can
be produced (and later registered) without a GPU.  A reporting tool that
drifts from the artifacts it claims to summarise is worse than no tool, so
this file asserts the tool's cells against BOTH:

  * the pool artifact ``molmetal/reports/cfm_c7_fullgen_2026_10_20/
    generated_smiles.csv`` (12 669 rows), and
  * the published metrics ``metrics/by_metric/c7_fullgen_pool_quality.json``
    and ``metrics/by_metric/cfm_fragmentation_audit.json``.

THE KNOWN VALUES (the ones the brief fixes)
--------------------------------------------
  fragmented_ratio          0.9149104112400347
  measured_rate             0.08508958875996528   <- see the note below
  largest_fragment_frac_mean 0.7395141725353019
  n_input                   12669
  n_scored                  12668

THE ONE-ULP NOTE (measured, not fudged)
---------------------------------------
The brief's ``measured_rate`` is 0.08508958875996528.  The value ON DISK --
both in ``metrics/by_metric/cfm_fragmentation_audit.json``
(``single_molecule_rate_over_pool_rows``) and as the exact IEEE-754 double
for ``1078 / 12669`` -- is 0.08508958875996527.  The two differ by exactly
one unit in the last place (0x1.5c86e68f44285p-4 vs 0x1.5c86e68f44286p-4,
1.39e-17 apart).  This test therefore asserts:

  * the tool's value equals the PUBLISHED artifact's value EXACTLY
    (0.08508958875996527), and
  * the tool's value is within one ULP of the brief's literal
    (0.08508958875996528),

and it does NOT quietly round either side to make them agree.

CPU-only: the repo's ``molmetal/tests/conftest.py`` blanks
``CUDA_VISIBLE_DEVICES`` at import time unless
``MOLMETAL_TEST_ALLOW_GPU=1`` is set, so this file cannot take VRAM.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts import wfa133_fragmentation_reporting_matrix as matrix  # noqa: E402

POOL_CSV = (
    REPO / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20" / "generated_smiles.csv"
)
RUN_DIR = REPO / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20"
C7_ARTIFACT = REPO / "metrics" / "by_metric" / "c7_fullgen_pool_quality.json"
AUDIT_ARTIFACT = REPO / "metrics" / "by_metric" / "cfm_fragmentation_audit.json"

#: The values the brief fixes, verbatim.
KNOWN_FRAGMENTED_RATIO = 0.9149104112400347
KNOWN_MEASURED_RATE = 0.08508958875996528
KNOWN_LARGEST_FRAGMENT_FRAC_MEAN = 0.7395141725353019
KNOWN_N_INPUT = 12669
KNOWN_N_SCORED = 12668

#: The value ON DISK for the same rate -- one ULP below the brief's literal.
PUBLISHED_MEASURED_RATE = 0.08508958875996527


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    """The matrix, built ONCE (the pool pass costs ~30 s on CPU)."""
    if not POOL_CSV.is_file():
        pytest.skip(f"pool artifact missing: {POOL_CSV}")
    return matrix.build_matrix(RUN_DIR, reference_path=None)


@pytest.fixture(scope="module")
def c7_artifact() -> Dict[str, Any]:
    if not C7_ARTIFACT.is_file():
        pytest.skip(f"published artifact missing: {C7_ARTIFACT}")
    return json.loads(C7_ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audit_row() -> Dict[str, Any]:
    if not AUDIT_ARTIFACT.is_file():
        pytest.skip(f"published artifact missing: {AUDIT_ARTIFACT}")
    audit = json.loads(AUDIT_ARTIFACT.read_text(encoding="utf-8"))
    for row in audit.get("runs") or []:
        if row.get("run") == "cfm_c7_fullgen_2026_10_20":
            return row
    pytest.skip("the audit artifact carries no cfm_c7_fullgen_2026_10_20 row")


# ---------------------------------------------------------------------------
# THE DENOMINATOR LADDER
# ---------------------------------------------------------------------------
def test_denominator_ladder(payload: Dict[str, Any]) -> None:
    """drawn / pool rows / parsed / valid / connected, and their order."""
    counts = payload["cells"]["full_pool"]["counts"]
    assert counts["n_pool_rows"] == KNOWN_N_INPUT
    assert counts["n_parsed"] == KNOWN_N_SCORED
    assert counts["n_valid"] == KNOWN_N_SCORED
    assert counts["n_connected"] == 1078
    # The trainer's own epoch-level draws: 18 epochs x 704.
    assert counts["n_drawn_samples"] == 12672
    # The ladder is monotone: samples drawn >= survivors >= parsed.
    assert counts["n_drawn_samples"] >= counts["n_pool_rows"] >= counts["n_parsed"]
    assert payload["denominators"]["n_drawn_samples"] == 12672
    assert "events.jsonl" in payload["denominators"]["n_drawn_samples_source"]


# ---------------------------------------------------------------------------
# THE KNOWN VALUES
# ---------------------------------------------------------------------------
def test_known_fragmented_ratio(payload: Dict[str, Any]) -> None:
    """The brief's fragmented_ratio, exactly."""
    raw = payload["cells"]["full_pool"]["raw"]
    assert raw["fragmented_ratio_over_pool_rows"]["value"] == KNOWN_FRAGMENTED_RATIO
    assert raw["fragmented_ratio_over_pool_rows"]["num"] == 1078
    assert raw["fragmented_ratio_over_pool_rows"]["den"] == KNOWN_N_INPUT
    assert raw["fragmented_ratio_over_pool_rows"]["denominator"] == "n_pool_rows"


def test_known_measured_rate(payload: Dict[str, Any]) -> None:
    """The brief's measured_rate, and the one-ULP difference from disk."""
    raw = payload["cells"]["full_pool"]["raw"]
    cell = raw["success_rate_over_pool_rows"]
    assert cell["value"] == PUBLISHED_MEASURED_RATE
    assert cell["num"] == 1078
    assert cell["den"] == KNOWN_N_INPUT
    # The brief's literal is ONE ULP above the value on disk.
    assert math.isclose(cell["value"], KNOWN_MEASURED_RATE, rel_tol=1e-15)
    assert abs(cell["value"] - KNOWN_MEASURED_RATE) <= 1e-16


def test_known_largest_fragment_frac_mean(payload: Dict[str, Any]) -> None:
    """The brief's largest_fragment_frac_mean, exactly."""
    cell = payload["cells"]["full_pool"]["raw"]["largest_component_frac_mean"]
    assert cell["value"] == KNOWN_LARGEST_FRAGMENT_FRAC_MEAN
    assert cell["den"] == KNOWN_N_SCORED


def test_known_n_input_and_n_scored(payload: Dict[str, Any]) -> None:
    counts = payload["cells"]["full_pool"]["counts"]
    assert counts["n_pool_rows"] == KNOWN_N_INPUT
    assert counts["n_parsed"] == KNOWN_N_SCORED
    assert counts["n_unparseable"] == 1
    assert counts["n_empty"] == 0


def test_known_mean_n_frags(payload: Dict[str, Any]) -> None:
    """The brief's mean_n_frags 10.244 (the full precision is 10.244316...)."""
    cell = payload["cells"]["full_pool"]["raw"]["n_fragments_mean"]
    assert round(cell["value"], 3) == 10.244


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE PUBLISHED ARTIFACTS
# ---------------------------------------------------------------------------
def test_agrees_with_the_published_fragmentation_audit(
    payload: Dict[str, Any], audit_row: Dict[str, Any]
) -> None:
    """Every published audit number the tool re-derives, cell for cell."""
    raw = payload["cells"]["full_pool"]["raw"]
    assert audit_row["fragmented_ratio"] == KNOWN_FRAGMENTED_RATIO
    assert (
        raw["fragmented_ratio_over_pool_rows"]["value"] == audit_row["fragmented_ratio"]
    )
    assert audit_row["largest_fragment_frac_mean"] == KNOWN_LARGEST_FRAGMENT_FRAC_MEAN
    assert (
        raw["largest_component_frac_mean"]["value"]
        == audit_row["largest_fragment_frac_mean"]
    )
    assert raw["n_fragments_mean"]["value"] == audit_row["n_fragments_mean"]
    assert (
        raw["connectivity_conditional"]["value"] == audit_row["single_fragment_ratio"]
    )
    assert (
        raw["success_rate_over_pool_rows"]["value"]
        == audit_row["single_molecule_rate_over_pool_rows"]
    )
    assert (
        raw["success_rate_end_to_end"]["value"]
        == audit_row["single_molecule_rate_end_to_end"]
    )
    assert payload["cells"]["full_pool"]["counts"]["n_drawn_samples"] == int(
        audit_row["samples_drawn"]
    )
    assert audit_row["n_rows_in_csv"] == KNOWN_N_INPUT
    assert audit_row["n_scored"] == KNOWN_N_SCORED
    assert audit_row["n_single"] == 1078


def test_agrees_with_the_published_c7_pool_quality(
    payload: Dict[str, Any], c7_artifact: Dict[str, Any]
) -> None:
    """The tool's overall numbers match the published per-pool quality."""
    overall = c7_artifact["overall"]
    raw = payload["cells"]["full_pool"]["raw"]
    counts = payload["cells"]["full_pool"]["counts"]
    assert overall["n_rows"] == counts["n_pool_rows"]
    assert overall["n_parse"] == counts["n_parsed"]
    assert overall["validity"] == round(raw["validity"]["value"], 4)
    assert overall["connected_rate"] == round(
        raw["connectivity_conditional"]["value"], 4
    )
    assert overall["single_fragment_rate"] == round(
        raw["connectivity_conditional"]["value"], 4
    )
    assert overall["mean_n_frags"] == round(raw["n_fragments_mean"]["value"], 3)
    assert overall["mean_largest_frag_frac"] == round(
        raw["largest_component_frac_mean"]["value"], 4
    )
    assert overall["share_n_frags_2_to_6"] == round(
        raw["share_n_frags_2_to_6"]["value"], 4
    )
    assert overall["share_n_frags_ge21"] == round(raw["share_n_frags_ge21"]["value"], 4)
    assert overall["three_ring_share"] == round(raw["three_ring_share"]["value"], 4)
    assert overall["six_ring_share"] == round(raw["six_ring_share"]["value"], 4)
    assert overall["second_frag_is_single_atom"] == round(
        raw["second_frag_is_single_atom"]["value"], 4
    )
    assert overall["n_unique_smiles"] == raw["uniqueness"]["num"]


def test_agrees_with_the_published_final_epoch(
    payload: Dict[str, Any], c7_artifact: Dict[str, Any]
) -> None:
    """The tool's final-epoch slice matches the published epoch-17 row."""
    epoch = c7_artifact["per_epoch"]["17"]
    final = payload["cells"]["final_epoch"]
    raw = final["raw"]
    assert epoch["n_rows"] == final["counts"]["n_pool_rows"] == 704
    assert epoch["n_parse"] == final["counts"]["n_parsed"] == 704
    assert epoch["connected_rate"] == round(raw["connectivity_conditional"]["value"], 4)
    assert epoch["mean_n_frags"] == round(raw["n_fragments_mean"]["value"], 3)
    assert epoch["mean_largest_frag_frac"] == round(
        raw["largest_component_frac_mean"]["value"], 4
    )
    assert epoch["second_frag_is_single_atom"] == round(
        raw["second_frag_is_single_atom"]["value"], 4
    )


# ---------------------------------------------------------------------------
# THE MATRIX'S OWN CONTRACT
# ---------------------------------------------------------------------------
def test_every_cell_names_its_denominator(payload: Dict[str, Any]) -> None:
    """A rate without its denominator is unreadable."""
    for slice_name, block in payload["cells"].items():
        for arm in ("raw", "post_processed"):
            for key, cell in block[arm].items():
                assert cell[
                    "denominator"
                ], f"{slice_name}/{arm}/{key} has no denominator"
                assert cell["arm"] == ("RAW" if arm == "raw" else "POST-PROCESSED")
                assert cell["source"], f"{slice_name}/{arm}/{key} has no source"


def test_the_population_is_named(payload: Dict[str, Any]) -> None:
    """The pool is a trajectory, and the slices say which rows they hold."""
    pop = payload["population"]
    assert pop["pool_epochs"] == list(range(18))
    assert pop["run_of_record_epochs"] == list(range(4, 18))
    assert pop["killed_pre_resume_epochs"] == [0, 1, 2, 3]
    assert pop["n_pool_rows"] == KNOWN_N_INPUT
    ro = payload["cells"]["run_of_record"]["counts"]
    assert ro["n_pool_rows"] == 9853
    assert ro["n_parsed"] == 9852
    assert ro["n_connected"] == 952
    final = payload["cells"]["final_epoch"]["counts"]
    assert final["n_pool_rows"] == 704
    assert final["n_connected"] == 86


def test_post_processed_connectivity_is_flagged_a_tautology(
    payload: Dict[str, Any],
) -> None:
    """LargestFragmentChooser returns ONE fragment: 1.0 by construction."""
    post = payload["cells"]["full_pool"]["post_processed"]
    cell = post["single_component_after_selection"]
    assert cell["value"] == 1.0
    assert cell.get("tautology") is True
    assert cell["arm"] == "POST-PROCESSED"


def test_post_processed_arm_is_labelled_and_measured(payload: Dict[str, Any]) -> None:
    """The POST-PROCESSED arm is the field's pipeline minus the UFF step."""
    post = payload["cells"]["full_pool"]["post_processed"]
    validity = post["validity_after_selection_min2heavy"]
    assert validity["value"] > 0.7
    assert validity["num"] == 10005
    assert validity["den"] == KNOWN_N_SCORED
    assert "LargestFragmentChooser" in validity["note"]
    assert "OMITTED" in payload["tool"]["postprocessing_uff_step"]
    assert "UFF" in " ".join(payload["notes"])
    # The same numerator over the SURVIVOR denominator, published beside it.
    over_rows = post["validity_after_selection_min2heavy_over_pool_rows"]
    assert over_rows["num"] == validity["num"] == 10005
    assert over_rows["den"] == KNOWN_N_INPUT
    assert over_rows["value"] == 10005 / 12669
    # The mass the repair discards -- the honest cost of post-processing.
    assert post["heavy_atom_mass_discarded_mean"]["value"] > 0.2
    assert post["heavy_atom_mass_discarded_median"]["value"] is not None


def test_unavailable_cells_are_named_not_omitted(payload: Dict[str, Any]) -> None:
    """Every uncomputable cell carries a reason."""
    named = {item["cell"] for item in payload["unavailable_cells"]}
    for expected in (
        "posebusters_valid",
        "strain_or_internal_energy",
        "postproc_validity_with_uff_on_model_geometry",
        "bonds_dropped_histogram",
        "novelty_and_intdiv1",
        "downstream_properties_on_connected_subset",
        "docking_and_redock_rms",
    ):
        assert expected in named
    for item in payload["unavailable_cells"]:
        assert item["reason"], item["cell"]
        assert item["blocker"], item["cell"]


def test_the_tool_declares_itself_cpu_only(payload: Dict[str, Any]) -> None:
    tool = payload["tool"]
    assert tool["cpu_only"] is True
    assert tool["gpu_used"] is False
    assert tool["checkpoint_loaded"] is False
    assert tool["writes_under_metrics_or_registry"] is False
    assert "score_smiles_pool" in tool["fragment_definition_owner"]


def test_the_cross_checks_pass(payload: Dict[str, Any]) -> None:
    """The pool's own columns agree with the re-derivation."""
    checks = payload["cross_checks"]
    assert checks["pool_n_frags_raw_column"]["n_mismatch"] == 0
    assert checks["pool_n_frags_raw_column"]["n_checked"] == KNOWN_N_SCORED
    assert checks["pool_largest_frag_frac_column"]["n_mismatch"] == 0
    assert checks["two_pass_component_agreement"]["n_mismatch"] == 0
    consistency = checks["slice_derivation_consistency"]
    assert consistency["checked"] is True
    assert consistency["n_parsed_rederived"] == consistency["n_parsed_from_slice"]
    assert consistency["n_single_rederived"] == consistency["n_single_from_slice"]
    assert (
        consistency["largest_fragment_frac_mean_rederived"]
        == (consistency["largest_fragment_frac_mean_from_slice"])
    )


def test_the_field_band_is_published_not_measured(payload: Dict[str, Any]) -> None:
    """The band is quoted, and labelled as PUBLISHED."""
    band = payload["field_band"]
    assert band["tag"] == "PUBLISHED"
    assert band["worst_published_connected"] == 0.374
    assert band["citations"]
    assert "NOT re-fetched" in band["provenance"]


# ---------------------------------------------------------------------------
# THE CLI
# ---------------------------------------------------------------------------
def _metrics_snapshot() -> Dict[str, int]:
    """name -> mtime_ns for the published trees (cheap, no hashing)."""
    out: Dict[str, int] = {}
    for root in ("metrics/by_metric", "metrics/paper_tables"):
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_file():
                out[f"{root}/{path.name}"] = path.stat().st_mtime_ns
    return out


def test_cli_writes_json_to_a_named_path(
    tmp_path: Path, capsys: Any, monkeypatch: Any, payload: Dict[str, Any]
) -> None:
    """The CLI writes the matrix where the CALLER names -- and nowhere else."""
    monkeypatch.setattr(matrix, "build_matrix", lambda *a, **k: payload)
    out = tmp_path / "matrix.json"
    rc = matrix.main(
        [
            "--run-dir",
            str(RUN_DIR),
            "--reference",
            "none",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["metric"] == "wfa133_fragmentation_reporting_matrix"
    assert written["tag"] == "MEASURED"
    assert (
        written["cells"]["full_pool"]["raw"]["fragmented_ratio_over_pool_rows"]["value"]
        == KNOWN_FRAGMENTED_RATIO
    )
    capsys.readouterr()


def test_cli_refuses_to_write_under_metrics(
    capsys: Any, monkeypatch: Any, payload: Dict[str, Any]
) -> None:
    """The tool publishes NOTHING into the project's SSOT."""
    monkeypatch.setattr(matrix, "build_matrix", lambda *a, **k: payload)
    before = _metrics_snapshot()
    rc = matrix.main(
        [
            "--run-dir",
            str(RUN_DIR),
            "--reference",
            "none",
            "--out",
            str(REPO / "metrics" / "by_metric" / "wfa133_should_not_exist.json"),
        ]
    )
    assert rc == 2
    after = _metrics_snapshot()
    assert before == after
    assert not (
        REPO / "metrics" / "by_metric" / "wfa133_should_not_exist.json"
    ).exists()
    capsys.readouterr()


def test_cli_without_out_writes_nothing(
    capsys: Any, monkeypatch: Any, payload: Dict[str, Any]
) -> None:
    """Default: stdout only."""
    monkeypatch.setattr(matrix, "build_matrix", lambda *a, **k: payload)
    before = _metrics_snapshot()
    rc = matrix.main(["--run-dir", str(RUN_DIR), "--reference", "none"])
    assert rc == 0
    assert before == _metrics_snapshot()
    capsys.readouterr()


def test_text_rendering_keeps_every_cell(payload: Dict[str, Any]) -> None:
    """The human-readable rendering loses nothing."""
    text = matrix.render_text(payload)
    for slice_name, block in payload["cells"].items():
        assert slice_name in text
        for arm in ("raw", "post_processed"):
            for key in block[arm]:
                assert key in text
    assert "UNAVAILABLE CELLS" in text
    assert "FIELD BAND" in text
