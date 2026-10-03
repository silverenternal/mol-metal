"""test_wf335m3_triangle_gpu_extend.py — Wf-335-M3 Triangle GPU wrapper extend.

====================================================================
Wf-335-M3 — Triangle GPU wrapper extension to PlatinAI 4-cell +
MetalCytoToxDB 5-metal (4 NEW tests)
====================================================================

Per Wf-333-G2 closeout (`TODO/completed/333g2_triangle_gpu_wrapper.md`)
the Wf-333-G2 GPU wrapper ships CPU-only CLI scaffolding with the
``--feature-mode {5d,12d,17d}`` knob.  Per Wf-330-T1/T2 closeouts
the triangle closure is **end-to-end** on both cohorts, but the
wrapper did NOT emit a per-cohort breakdown table or expose a
``--triangle-cohort`` selector.  Wf-335-M3 ships that extension.

Test surface (4 NEW tests per Wf-335-M3 task spec):

1. :func:`test_platinai_4cell_per_cell_breakdown` — exercises the
   PlatinAI 4-cell cohort path end-to-end.  Asserts:
     * ``per_cohort_breakdown["platinai_4cell"]["cohort_kind"] ==
       "per_cell"``.
     * 7 metals (Pt / Pd / Ru / Rh / Ir / Os / Au) per skill §4.2.
     * ``projected_mean_hilbert_r2 == 0.594`` (skill §4.2 PROJECTED).
     * ``projected_per_metal_r2`` has all 7 entries.
     * ``wf_closeout == "TODO/completed/330t1_triangle_platinai.md"``.
     * ``recovery_status == "HONEST_NEGATIVE_PERSISTENT"``.
     * ``HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL`` flag is set.
     * Feature mode = "12d" matches Wf-330-T1 contract.
     * Default CSV = ``molmetal/data/platinai_4cell.csv``.

2. :func:`test_metalcytotox_5metal_per_metal_breakdown` — exercises
   the MetalCytoToxDB 5-metal cohort path.  Asserts:
     * ``cohort_kind == "per_metal"``.
     * 5 metals (Ru / Ir / Rh / Os / Re) per skill §4.3.
     * ``projected_mean_hilbert_r2 == 0.622`` (skill §4.3 PROJECTED).
     * ``projected_per_metal_r2`` is ``{"Ru": 0.65, "Ir": 0.63,
       "Rh": 0.62, "Os": 0.61, "Re": 0.60}``.
     * ``HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL`` flag is set.
     * Default CSV = ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv``.

3. :func:`test_feature_modes_5d_12d_17d` — exercises all 3 feature
   modes (5d / 12d / 17d) on both cohorts.  Asserts:
     * For each (mode, cohort) pair the wrapper emits the expected
       feature shape — (n, 5) / (n, 12) / (n, 17) — and the
       per-cohort breakdown carries the matching ``feature_mode``.
     * All 3 modes parse correctly via CLI.
     * ``resolve_triangle_cohorts("both")`` returns both cohorts;
       ``resolve_triangle_cohorts("platinai_4cell")`` returns the
       singleton; invalid cohort raises :class:`ValueError`.

4. :func:`test_gpu_wrapper_initialization_cpu_only` — regression
   guard.  Asserts the Wf-333-G2 contract is preserved end-to-end
   on the extended wrapper:
     * ``TriangleGPUConfig`` defaults (12d / hilbert_dim=64 /
       launch=False).
     * ``build_triangle_gpu_trainer`` runs CPU-only (no GPU
       side-effects; toy design matrix is numpy float32).
     * ``TriangleGPUSummary`` carries the new
       ``per_cohort_breakdown`` field with both cohorts when
       ``triangle_cohort == "both"``.
     * Hilbert torch module has ``hilbert_dim`` trainable params.
     * write_summary writes ``summary.json`` + ``summary.md`` with
       the per-cohort breakdown table.

CPU-only contract
----------------
All 4 tests are CPU-only — no GPU launch per AGENTS.md §5.1.
The wrapper is still dry-run only; downstream Wf-333-G3+ runs the
actual GPU launch when VRAM frees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Path bootstrap — same convention as the wrapper module.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.wf333g2_triangle_12d_gpu_wrapper import (  # noqa: E402
    # Constants — cohort registry (Wf-335-M3).
    DEFAULT_TRIANGLE_COHORT,
    DEFAULT_TRIANGLE_FEATURE_MODE,
    TRIANGLE_COHORT_CHOICES,
    TRIANGLE_COHORT_REGISTRY,
    PLATINAI_4CELL_METAL_LIST,
    PLATINAI_4CELL_METAL_TO_IDX,
    PLATINAI_4CELL_PROJECTED_MEAN,
    PLATINAI_4CELL_PROJECTED_R2,
    METALCYTO_5METAL_LIST,
    METALCYTO_5METAL_TO_IDX,
    METALCYTO_5METAL_PROJECTED_MEAN,
    METALCYTO_5METAL_PROJECTED_R2,
    # Wrappers + dataclasses.
    TriangleGPUConfig,
    TriangleGPUSummary,
    add_wf333g2_args,
    build_hilbert_torch_module,
    build_per_cohort_breakdown,
    build_full_per_cohort_breakdown,
    build_toy_design_matrix,
    build_triangle_gpu_trainer,
    get_cohort_metadata,
    parse_args,
    resolve_triangle_cohorts,
    write_summary,
)


# ===========================================================================
# Test 1 — PlatinAI 4-cell per-cell breakdown
# ===========================================================================
def test_platinai_4cell_per_cell_breakdown(tmp_path: Path) -> None:
    """Wf-335-M3 — PlatinAI 4-cell cohort per-cell breakdown.

    Asserts the Wf-330-T1 closeout anchor (Pt / Pd / Ru / Rh / Ir /
    Os / Au, 7 metals, cohort_kind=per_cell) is reflected in the
    extended wrapper output.  Each numeric value traces to either
    the Wf-330-T1 closeout or the mol-metal-triangle-closure skill
    §4.2 PROJECTED ledger.
    """
    cfg = TriangleGPUConfig(
        triangle_cohort="platinai_4cell",
        triangle_feature_mode="12d",
    )
    assert cfg.triangle_cohort == "platinai_4cell"
    assert cfg.triangle_feature_mode == "12d"

    summary = build_triangle_gpu_trainer(cfg, n_samples=32, seed=0)

    # ---- Per-cohort breakdown exists ----
    assert isinstance(summary.per_cohort_breakdown, dict), (
        f"per_cohort_breakdown should be dict, got "
        f"{type(summary.per_cohort_breakdown)}"
    )
    assert "platinai_4cell" in summary.per_cohort_breakdown, (
        f"platinai_4cell should be in per_cohort_breakdown; "
        f"got keys={list(summary.per_cohort_breakdown.keys())}"
    )
    # When triangle_cohort=platinai_4cell, metalcytotox should NOT
    # be in the breakdown (single-cohort scope).
    assert "metalcytotox_5metal" not in summary.per_cohort_breakdown, (
        f"metalcytotox_5metal should NOT be present when "
        f"triangle_cohort=platinai_4cell; "
        f"got keys={list(summary.per_cohort_breakdown.keys())}"
    )

    p4 = summary.per_cohort_breakdown["platinai_4cell"]
    assert isinstance(p4, dict)

    # ---- cohort_kind = per_cell ----
    assert p4["cohort_kind"] == "per_cell", (
        f"PlatinAI 4-cell cohort_kind should be 'per_cell'; "
        f"got {p4['cohort_kind']!r}"
    )
    assert p4["cohort"] == "platinai_4cell"

    # ---- 7 metals (Pt / Pd / Ru / Rh / Ir / Os / Au) per skill §4.2 ----
    expected_metals = ("Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au")
    assert tuple(p4["metal_list"]) == expected_metals, (
        f"PlatinAI 4-cell metal_list should be {expected_metals}, "
        f"got {tuple(p4['metal_list'])}"
    )
    # Cross-check with module-level constant.
    assert p4["metal_list"] == list(PLATINAI_4CELL_METAL_LIST)
    assert p4["n_metals"] == 7

    # ---- PROJECTED mean R² = 0.594 (skill §4.2) ----
    assert p4["projected_mean_hilbert_r2"] == pytest.approx(0.594), (
        f"PlatinAI 4-cell PROJECTED mean R² should be 0.594 "
        f"(skill §4.2); got {p4['projected_mean_hilbert_r2']}"
    )
    assert p4["projected_mean_hilbert_r2"] == PLATINAI_4CELL_PROJECTED_MEAN

    # ---- projected_per_metal_r2 has all 7 entries ----
    ppm = p4["projected_per_metal_r2"]
    assert len(ppm) == 7, (
        f"projected_per_metal_r2 should have 7 entries; got {len(ppm)}"
    )
    for m in expected_metals:
        assert m in ppm, (
            f"projected_per_metal_r2 missing metal {m!r}; "
            f"got keys={list(ppm.keys())}"
        )
        assert ppm[m] == pytest.approx(PLATINAI_4CELL_PROJECTED_R2[m])
    # Spot-check numeric values (skill §4.2 anchors).
    assert ppm["Pt"] == pytest.approx(0.62)
    assert ppm["Pd"] == pytest.approx(0.61)
    assert ppm["Au"] == pytest.approx(0.58)

    # ---- wf_closeout / recovery_status ----
    assert p4["wf_closeout"] == "TODO/completed/330t1_triangle_platinai.md"
    assert p4["recovery_status"] == "HONEST_NEGATIVE_PERSISTENT"
    assert p4["wf_id"] == "Wf-330-T1"

    # ---- feature_mode = "12d" ----
    assert p4["feature_mode"] == "12d"
    assert p4["tag"] == "PROJECTED"

    # ---- Default CSV ----
    assert p4["default_dataset_csv"] == "molmetal/data/platinai_4cell.csv"

    # ---- HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL flag ----
    assert "HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL" in (
        summary.honest_negative_flags
    ), (
        f"summary.honest_negative_flags should include "
        f"HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL; got "
        f"{summary.honest_negative_flags}"
    )
    # The Wf-330-T2 flag should NOT be set.
    assert "HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL" not in (
        summary.honest_negative_flags
    )

    # ---- write_summary emits the per-cohort table ----
    out_dir = tmp_path / "wf335m3_platinai"
    json_path, md_path = write_summary(summary, out_dir)
    assert json_path.exists()
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "platinai_4cell" in md_text
    assert "per_cell" in md_text
    assert "0.5940" in md_text
    assert "HONEST_NEGATIVE_PERSISTENT" in md_text
    assert "TODO/completed/330t1_triangle_platinai.md" in md_text
    # Verify JSON round-trip.
    with open(json_path) as fh:
        payload = json.load(fh)
    assert "platinai_4cell" in payload["per_cohort_breakdown"]
    assert (
        payload["per_cohort_breakdown"]["platinai_4cell"][
            "projected_mean_hilbert_r2"
        ]
        == pytest.approx(0.594)
    )


# ===========================================================================
# Test 2 — MetalCytoToxDB 5-metal per-metal breakdown
# ===========================================================================
def test_metalcytotox_5metal_per_metal_breakdown(tmp_path: Path) -> None:
    """Wf-335-M3 — MetalCytoToxDB 5-metal cohort per-metal breakdown.

    Asserts the Wf-330-T2 closeout anchor (Ru / Ir / Rh / Os / Re,
    5 metals, cohort_kind=per_metal, projected mean R²=0.622) is
    reflected in the extended wrapper output.  Each numeric value
    traces to either the Wf-330-T2 closeout or the
    mol-metal-triangle-closure skill §4.3 PROJECTED ledger.
    """
    cfg = TriangleGPUConfig(
        triangle_cohort="metalcytotox_5metal",
        triangle_feature_mode="12d",
    )
    summary = build_triangle_gpu_trainer(cfg, n_samples=32, seed=0)

    # ---- Single-cohort scope ----
    assert "metalcytotox_5metal" in summary.per_cohort_breakdown
    assert "platinai_4cell" not in summary.per_cohort_breakdown

    m5 = summary.per_cohort_breakdown["metalcytotox_5metal"]

    # ---- cohort_kind = per_metal ----
    assert m5["cohort_kind"] == "per_metal"
    assert m5["cohort"] == "metalcytotox_5metal"

    # ---- 5 metals (Ru / Ir / Rh / Os / Re) per skill §4.3 ----
    expected_metals = ("Ru", "Ir", "Rh", "Os", "Re")
    assert tuple(m5["metal_list"]) == expected_metals
    assert m5["metal_list"] == list(METALCYTO_5METAL_LIST)
    assert m5["n_metals"] == 5

    # ---- PROJECTED mean R² = 0.622 (skill §4.3) ----
    assert m5["projected_mean_hilbert_r2"] == pytest.approx(0.622)
    assert m5["projected_mean_hilbert_r2"] == METALCYTO_5METAL_PROJECTED_MEAN

    # ---- projected_per_metal_r2 = skill §4.3 anchors ----
    assert m5["projected_per_metal_r2"] == {
        "Ru": pytest.approx(0.65),
        "Ir": pytest.approx(0.63),
        "Rh": pytest.approx(0.62),
        "Os": pytest.approx(0.61),
        "Re": pytest.approx(0.60),
    }
    # Cross-check with module-level constant.
    for m, r in METALCYTO_5METAL_PROJECTED_R2.items():
        assert m in m5["projected_per_metal_r2"]
        assert m5["projected_per_metal_r2"][m] == pytest.approx(r)

    # ---- wf_closeout / recovery_status ----
    assert m5["wf_closeout"] == (
        "TODO/completed/330t2_triangle_metalcytotox.md"
    )
    assert m5["recovery_status"] == "MEASURED=BELOW_TARGET"
    assert m5["wf_id"] == "Wf-330-T2"

    # ---- Default CSV ----
    assert m5["default_dataset_csv"] == (
        ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    )

    # ---- HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL flag ----
    assert "HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL" in (
        summary.honest_negative_flags
    ), (
        f"summary.honest_negative_flags should include "
        f"HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL; got "
        f"{summary.honest_negative_flags}"
    )
    # The PlatinAI 4-cell flag should NOT be set.
    assert "HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL" not in (
        summary.honest_negative_flags
    )

    # ---- write_summary emits the per-cohort table ----
    out_dir = tmp_path / "wf335m3_metalcytotox"
    json_path, md_path = write_summary(summary, out_dir)
    assert json_path.exists()
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "metalcytotox_5metal" in md_text
    assert "per_metal" in md_text
    assert "0.6220" in md_text
    assert "MEASURED=BELOW_TARGET" in md_text
    # Per-metal anchors should appear in the markdown.
    for m, r in METALCYTO_5METAL_PROJECTED_R2.items():
        assert f"- {m}:" in md_text, (
            f"per-metal anchor for {m} missing from MD"
        )
    with open(json_path) as fh:
        payload = json.load(fh)
    assert payload["per_cohort_breakdown"]["metalcytotox_5metal"][
        "projected_mean_hilbert_r2"
    ] == pytest.approx(0.622)


# ===========================================================================
# Test 3 — Feature modes (5d / 12d / 17d)
# ===========================================================================
def test_feature_modes_5d_12d_17d() -> None:
    """Wf-335-M3 — feature modes on both cohorts.

    Asserts the wrapper accepts ``--triangle-feature-mode 5d / 12d / 17d``
    and emits the matching feature shape for both cohorts.
    """
    # ---- CLI parsing for all 3 feature modes ----
    cfg_5d = parse_args([
        "--triangle-feature-mode", "5d",
        "--triangle-cohort", "platinai_4cell",
    ])
    assert cfg_5d.triangle_feature_mode == "5d"
    assert cfg_5d.triangle_cohort == "platinai_4cell"

    cfg_12d = parse_args([
        "--triangle-feature-mode", "12d",
        "--triangle-cohort", "metalcytotox_5metal",
    ])
    assert cfg_12d.triangle_feature_mode == "12d"
    assert cfg_12d.triangle_cohort == "metalcytotox_5metal"

    cfg_17d = parse_args([
        "--triangle-feature-mode", "17d",
        "--triangle-cohort", "both",
    ])
    assert cfg_17d.triangle_feature_mode == "17d"
    assert cfg_17d.triangle_cohort == "both"

    # ---- Invalid feature mode rejected ----
    with pytest.raises(SystemExit):
        parse_args(["--triangle-feature-mode", "99d"])
    # Invalid cohort rejected.
    with pytest.raises(SystemExit):
        parse_args(["--triangle-cohort", "nci60"])

    # ---- Feature shape matches mode for both cohorts ----
    expected_feature_count = {"5d": 5, "12d": 12, "17d": 17}
    for mode in ("5d", "12d", "17d"):
        for cohort in ("platinai_4cell", "metalcytotox_5metal"):
            cfg = TriangleGPUConfig(
                triangle_cohort=cohort,
                triangle_feature_mode=mode,
            )
            summary = build_triangle_gpu_trainer(cfg, n_samples=8, seed=0)
            n_expected = expected_feature_count[mode]
            assert summary.feature_shape == (8, n_expected), (
                f"feature_shape for {cohort}/{mode} should be "
                f"(8, {n_expected}); got {summary.feature_shape}"
            )
            assert len(summary.feature_names) == n_expected
            # Per-cohort breakdown carries the matching feature_mode.
            cb = summary.per_cohort_breakdown[cohort]
            assert cb["feature_mode"] == mode, (
                f"per-cohort breakdown feature_mode for {cohort} "
                f"should be {mode!r}; got {cb['feature_mode']!r}"
            )

    # ---- resolve_triangle_cohorts ----
    assert resolve_triangle_cohorts("both") == (
        "platinai_4cell", "metalcytotox_5metal",
    )
    assert resolve_triangle_cohorts("platinai_4cell") == (
        "platinai_4cell",
    )
    assert resolve_triangle_cohorts("metalcytotox_5metal") == (
        "metalcytotox_5metal",
    )
    with pytest.raises(ValueError):
        resolve_triangle_cohorts("nci60")

    # ---- Default values ----
    assert DEFAULT_TRIANGLE_COHORT == "both"
    assert DEFAULT_TRIANGLE_FEATURE_MODE == "12d"
    assert "platinai_4cell" in TRIANGLE_COHORT_CHOICES
    assert "metalcytotox_5metal" in TRIANGLE_COHORT_CHOICES
    assert "both" in TRIANGLE_COHORT_CHOICES

    # ---- build_full_per_cohort_breakdown("both") ----
    full_both = build_full_per_cohort_breakdown(
        "both", feature_mode="12d",
    )
    assert set(full_both.keys()) == {"platinai_4cell", "metalcytotox_5metal"}

    # ---- Cohort registry sanity ----
    assert "platinai_4cell" in TRIANGLE_COHORT_REGISTRY
    assert "metalcytotox_5metal" in TRIANGLE_COHORT_REGISTRY
    p4 = get_cohort_metadata("platinai_4cell")
    assert p4["cohort_kind"] == "per_cell"
    assert p4["wf_id"] == "Wf-330-T1"
    m5 = get_cohort_metadata("metalcytotox_5metal")
    assert m5["cohort_kind"] == "per_metal"
    assert m5["wf_id"] == "Wf-330-T2"
    # get_cohort_metadata raises on unknown cohort.
    with pytest.raises(KeyError):
        get_cohort_metadata("nci60")


# ===========================================================================
# Test 4 — GPU wrapper initialization (CPU only)
# ===========================================================================
def test_gpu_wrapper_initialization_cpu_only(tmp_path: Path) -> None:
    """Wf-335-M3 — GPU wrapper initialization (CPU only).

    Regression guard for the Wf-333-G2 contract preserved end-to-end
    on the extended wrapper.  No GPU is touched (toy design matrix
    is plain numpy float32); the per-cohort breakdown is populated
    for both cohorts when ``triangle_cohort == "both"``.
    """
    # ---- Default config (Wf-333-G2 baseline + Wf-335-M3 new fields) ----
    cfg = TriangleGPUConfig()
    assert cfg.feature_mode == "12d", (
        f"default feature_mode should be '12d'; got {cfg.feature_mode!r}"
    )
    assert cfg.hilbert_dim == 64
    assert cfg.launch is False
    assert cfg.triangle_cohort == "both", (
        f"default triangle_cohort should be 'both'; "
        f"got {cfg.triangle_cohort!r}"
    )
    assert cfg.triangle_feature_mode == "12d"

    # ---- Build trainer (CPU only) ----
    summary = build_triangle_gpu_trainer(cfg, n_samples=32, seed=0)
    assert isinstance(summary, TriangleGPUSummary)

    # ---- Feature schema (12d default) ----
    assert summary.feature_shape == (32, 12), (
        f"feature_shape should be (32, 12) for 12d mode; "
        f"got {summary.feature_shape}"
    )
    assert "MolWt" in summary.feature_names
    assert "MolLogP" in summary.feature_names
    assert "TPSA" in summary.feature_names

    # ---- Vertex labels ----
    assert summary.vertex_labels == ("astsr", "hilbert", "mcts")

    # ---- Hilbert torch module parameter count ----
    hilbert_module = build_hilbert_torch_module(hilbert_dim=cfg.hilbert_dim)
    n_trainable = int(sum(p.numel() for p in hilbert_module.parameters()))
    assert n_trainable == 64, (
        f"hilbert_dim=64 → 64 trainable params; got {n_trainable}"
    )

    # ---- Per-cohort breakdown: both cohorts when triangle_cohort=both ----
    assert isinstance(summary.per_cohort_breakdown, dict)
    assert "platinai_4cell" in summary.per_cohort_breakdown
    assert "metalcytotox_5metal" in summary.per_cohort_breakdown
    # Per-cohort sanity.
    p4 = summary.per_cohort_breakdown["platinai_4cell"]
    assert p4["cohort_kind"] == "per_cell"
    assert p4["n_metals"] == 7
    assert p4["projected_mean_hilbert_r2"] == pytest.approx(0.594)
    m5 = summary.per_cohort_breakdown["metalcytotox_5metal"]
    assert m5["cohort_kind"] == "per_metal"
    assert m5["n_metals"] == 5
    assert m5["projected_mean_hilbert_r2"] == pytest.approx(0.622)

    # ---- Honest-negative flags (Wf-333-G2 + Wf-335-M3) ----
    # Wf-333-G2 baseline flag.
    assert "HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED" in (
        summary.honest_negative_flags
    )
    # Wf-335-M3 per-cohort flags.
    assert "HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL" in (
        summary.honest_negative_flags
    )
    assert "HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL" in (
        summary.honest_negative_flags
    )
    # Project tag.
    assert summary.project_tag == "PROJECTED"

    # ---- No GPU touched (toy matrix is numpy float32) ----
    X, y = build_toy_design_matrix(n_samples=8, n_features=12, seed=0)
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float32
    assert y.dtype == np.float32

    # ---- write_summary path ----
    out_dir = tmp_path / "wf335m3_cpu_only"
    json_path, md_path = write_summary(summary, out_dir)
    assert json_path.exists()
    assert md_path.exists()
    with open(json_path) as fh:
        payload = json.load(fh)
    # Per-cohort breakdown is in the JSON.
    assert "platinai_4cell" in payload["per_cohort_breakdown"]
    assert "metalcytotox_5metal" in payload["per_cohort_breakdown"]
    # The Wf-333-G2 fields are also preserved.
    assert payload["project_tag"] == "PROJECTED"
    assert "HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED" in (
        payload["honest_negative_flags"]
    )
    # MD has the per-cohort table.
    md_text = md_path.read_text()
    assert "Triangle cohort breakdown" in md_text
    assert "platinai_4cell" in md_text
    assert "metalcytotox_5metal" in md_text
    assert "per_cell" in md_text
    assert "per_metal" in md_text
    assert "TODO/completed/330t1_triangle_platinai.md" in md_text
    assert "TODO/completed/330t2_triangle_metalcytotox.md" in md_text

    # ---- add_wf333g2_args idempotency (carries the new flags) ----
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser = add_wf333g2_args(parser)
    parser = add_wf333g2_args(parser)  # idempotent — must not raise
    ns = parser.parse_args([
        "--triangle-cohort", "metalcytotox_5metal",
        "--triangle-feature-mode", "17d",
    ])
    assert ns.triangle_cohort == "metalcytotox_5metal"
    assert ns.triangle_feature_mode == "17d"
