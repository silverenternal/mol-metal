"""Tests for Wf-130 Phase 1 (R20-P5) — Catherine 6/6 audit on ALL 30 d-block metals.

==============================================================================

Pytest suite for the Wf-130 P1 Catherine 6/6 audit.  Verifies the
audit script imports, the 30-metal default matches the canonical list,
the per-metal output structure is correct, the 6/6 PASS verdict is
emitted for every metal on the synthetic path, and the CLI smoke runs
end-to-end.

Test coverage
=============

* **Constants + module surface (3 tests)** — DEFAULT_30_METALS,
  CATHERINE_PROPERTIES, helpers.

* **Per-metal audit JSON structure (3 tests)** — schema sanity for
  the on-disk per-metal files.

* **Per-metal 6/6 PASS tests (3 tests)** — sweeping the synthetic path
  produces a 6/6 PASS verdict on all 30 d-block metals, parity with
  the Wf-57 / Wf-68 ship surfaces, and FAIL propagation when a
  property is broken.

* **CLI smoke + end-to-end (3 tests)** — the audit CLI runs against
  tmp_path, writes 30 per-metal directories + the combined JSON +
  synthesis.md, and the per-property pass-rate is uniform 30/30.

* **Honest-framing tests (3 tests)** — synthetic flag is propagated
  through the JSON; the synthesis.md headline verdict matches the
  per-metal verdicts; the Wf-68 back-link is recorded.

**Total**: ~15 pytest items, all designed to PASS on a vanilla Python
install (no torch / RDKit / cuda requirements).  This is the Wf-130
P1 CPU-only ship target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo-root path injection — matches the audit script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts import audit_catherine_30metals as w130


# ---------------------------------------------------------------------------
# Loader shim — load catherine_properties directly to bypass the
# torch-dependent molmetal.math_skills.__init__ chain on slim envs.
# ---------------------------------------------------------------------------


def _load_catherine_properties():
    """Load ``catherine_properties.py`` directly without triggering math_skills.__init__."""
    import importlib.util

    cp_path = (
        Path(__file__).resolve().parent.parent
        / "math_skills"
        / "catherine_properties.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_catherine_properties_wf130", str(cp_path)
    )
    cp = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, cp)
    spec.loader.exec_module(cp)
    return cp


_cp = _load_catherine_properties()
DBLOCK_30_METALS = _cp.DBLOCK_30_METALS
PER_PROPERTY_VERIFIERS = _cp.PER_PROPERTY_VERIFIERS
sweep_metals = _cp.sweep_metals
sweep_summary_stats = _cp.sweep_summary_stats
synthetic_closure_summary = _cp.synthetic_closure_summary
verify_all_catherine_properties = _cp.verify_all_catherine_properties


# ---------------------------------------------------------------------------
# 1. Constants + module surface
# ---------------------------------------------------------------------------


def test_audit_script_imports():
    """The audit module imports without error and exposes the public API."""
    assert hasattr(w130, "main")
    assert hasattr(w130, "_run_audit_for_metal")
    assert hasattr(w130, "_write_per_metal")
    assert hasattr(w130, "_render_synthesis")
    assert hasattr(w130, "_parse_args")


def test_default_30_metals_constant_matches_canonical():
    """The audit ships a 30-metal default list matching the canonical 30-set."""
    assert len(w130.DEFAULT_30_METALS) == 30
    assert set(w130.DEFAULT_30_METALS) == set(DBLOCK_30_METALS)


def test_catherine_properties_constant_is_six():
    """The audit ships all 6 Catherine properties P1..P6."""
    assert len(w130.CATHERINE_PROPERTIES) == 6
    assert set(w130.CATHERINE_PROPERTIES) == set(PER_PROPERTY_VERIFIERS)
    assert "P1_reflexivity" in w130.CATHERINE_PROPERTIES
    assert "P6_composability" in w130.CATHERINE_PROPERTIES


def test_utcnow_iso_format():
    """``_utcnow_iso`` returns a Z-suffixed ISO 8601 string."""
    ts = w130._utcnow_iso()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts) >= 20


# ---------------------------------------------------------------------------
# 2. Per-metal audit JSON structure
# ---------------------------------------------------------------------------


def test_write_per_metal_creates_dir_and_files(tmp_path):
    """``_write_per_metal`` writes the per-metal catherine_audit.json."""
    summary = synthetic_closure_summary()
    summary["synthetic"] = True
    verdict_obj = verify_all_catherine_properties("Pt", summary)
    verdict = {
        "metal": "Pt",
        "six_of_six_pass": verdict_obj.six_of_six_pass,
        "n_pass": verdict_obj.n_pass,
        "n_fail": verdict_obj.n_fail,
        "passed_properties": verdict_obj.passed_property_ids(),
        "failed_properties": verdict_obj.failed_property_ids(),
        "per_property": {
            pid: {
                "passed": r.passed,
                "n_samples": r.n_samples,
                "n_violations": r.n_violations,
                "wall_time_s": r.wall_time_s,
                "notes": r.notes,
            }
            for pid, r in verdict_obj.per_property.items()
        },
        "wall_time_s": verdict_obj.wall_time_s,
    }
    metadata = {
        "wf": "Wf-130",
        "phase": "Phase 1",
        "r20_plan": "R20-P5",
        "n_seeds": 5,
        "max_depth": 1,
        "max_products": 64,
    }
    row = w130._write_per_metal(
        tmp_path / "Pt",
        "Pt",
        summary,
        verdict,
        metadata=metadata,
        helpers_available=False,
    )
    assert row["metal"] == "Pt"
    assert row["six_of_six_pass"] is True
    assert row["synthetic_only"] is True
    assert (tmp_path / "Pt" / "catherine_audit.json").exists()

    payload = json.loads(
        (tmp_path / "Pt" / "catherine_audit.json").read_text()
    )
    assert payload["metal"] == "Pt"
    assert payload["six_of_six_pass"] is True
    assert payload["n_pass"] == 6
    assert payload["n_fail"] == 0
    assert payload["metadata"]["wf"] == "Wf-130"
    assert payload["metadata"]["r20_plan"] == "R20-P5"
    assert payload["metadata"]["synthetic_only_run"] is True
    assert set(payload["per_property"].keys()) == set(
        w130.CATHERINE_PROPERTIES
    )


def test_write_per_metal_records_bench_summary(tmp_path):
    """``_write_per_metal`` records the bench summary that fed the verifiers."""
    summary = synthetic_closure_summary()
    summary["synthetic"] = True
    verdict_obj = verify_all_catherine_properties("Ir", summary)
    verdict = {
        "metal": "Ir",
        "six_of_six_pass": verdict_obj.six_of_six_pass,
        "n_pass": verdict_obj.n_pass,
        "n_fail": verdict_obj.n_fail,
        "passed_properties": verdict_obj.passed_property_ids(),
        "failed_properties": verdict_obj.failed_property_ids(),
        "per_property": {
            pid: {
                "passed": r.passed,
                "n_samples": r.n_samples,
                "n_violations": r.n_violations,
                "wall_time_s": r.wall_time_s,
                "notes": r.notes,
            }
            for pid, r in verdict_obj.per_property.items()
        },
        "wall_time_s": verdict_obj.wall_time_s,
    }
    w130._write_per_metal(
        tmp_path / "Ir",
        "Ir",
        summary,
        verdict,
        metadata={"n_seeds": 1, "max_depth": 1, "max_products": 32},
        helpers_available=False,
    )
    payload = json.loads(
        (tmp_path / "Ir" / "catherine_audit.json").read_text()
    )
    bench = payload["bench_summary"]
    assert bench["metal"] == "Ir"
    assert bench["synthetic"] is True
    assert bench["witness_rate"] == 1.0
    assert bench["well_typed_rate"] == 1.0
    assert bench["coord_violations"] == 0
    assert bench["reachability_rate"] == 1.0


def test_write_per_metal_propagates_fail_verdict(tmp_path):
    """When a property FAILs, the per-metal JSON records it honestly."""
    summary = synthetic_closure_summary(witness_rate=0.4)  # breaks P2
    verdict_obj = verify_all_catherine_properties("Hg", summary)
    verdict = {
        "metal": "Hg",
        "six_of_six_pass": verdict_obj.six_of_six_pass,
        "n_pass": verdict_obj.n_pass,
        "n_fail": verdict_obj.n_fail,
        "passed_properties": verdict_obj.passed_property_ids(),
        "failed_properties": verdict_obj.failed_property_ids(),
        "per_property": {
            pid: {
                "passed": r.passed,
                "n_samples": r.n_samples,
                "n_violations": r.n_violations,
                "wall_time_s": r.wall_time_s,
                "notes": r.notes,
            }
            for pid, r in verdict_obj.per_property.items()
        },
        "wall_time_s": verdict_obj.wall_time_s,
    }
    w130._write_per_metal(
        tmp_path / "Hg",
        "Hg",
        summary,
        verdict,
        metadata={"n_seeds": 1, "max_depth": 1, "max_products": 32},
        helpers_available=False,
    )
    payload = json.loads(
        (tmp_path / "Hg" / "catherine_audit.json").read_text()
    )
    assert payload["six_of_six_pass"] is False
    assert payload["n_pass"] == 5
    assert payload["n_fail"] == 1
    assert "P2_transitivity" in payload["failed_properties"]
    assert payload["per_property"]["P2_transitivity"]["passed"] is False


# ---------------------------------------------------------------------------
# 3. Per-metal 6/6 PASS tests — synthetic path on all 30 metals
# ---------------------------------------------------------------------------


def _synthetic_summaries_for(metals):
    return {m: synthetic_closure_summary() for m in metals}


def test_sweep_all_30_metals_synthetic_pass():
    """Synthetic sweep over ALL 30 d-block metals yields 30/30 6/6 PASS.

    This is the Wf-130 P1 ship target — 30/30 metals × 6/6 properties
    PASS on the synthetic CPU-only path.
    """
    metals = list(DBLOCK_30_METALS)
    summaries = _synthetic_summaries_for(metals)
    verdicts = sweep_metals(summaries, metals)
    stats = sweep_summary_stats(verdicts)
    assert stats["metals_total"] == 30
    assert stats["metals_pass_6of6"] == 30
    assert stats["metals_fail_6of6"] == 0
    for pid in PER_PROPERTY_VERIFIERS:
        assert stats["properties_pass_count"][pid] == 30
        assert stats["properties_pass_rate"][pid] == 1.0


def test_sweep_lifts_to_30_metals_including_la_ir_pt_au_hg():
    """The lift from 25 to 30 metals covers La / Ir / Pt / Au / Hg."""
    metals = ("La", "Ir", "Pt", "Au", "Hg")
    summaries = _synthetic_summaries_for(metals)
    verdicts = sweep_metals(summaries, metals)
    assert set(verdicts.keys()) == set(metals)
    for m in metals:
        assert verdicts[m].six_of_six_pass, (
            f"{m} should PASS 6/6 on synthetic path"
        )


def test_verify_all_catherine_properties_propagates_violation():
    """When a property FAILs, the verdict reflects it."""
    s = synthetic_closure_summary(witness_rate=0.3)
    v = verify_all_catherine_properties("Au", s)
    assert v.six_of_six_pass is False
    assert "P2_transitivity" in v.failed_property_ids()


# ---------------------------------------------------------------------------
# 4. CLI smoke + end-to-end
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    """The default args satisfy the Wf-130 P1 ship target = 30 metals."""
    args = w130._parse_args([])
    assert args.metals == 30
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.max_depth == 1
    assert args.max_products == 64
    assert args.synthetic_only is False
    assert args.metals_list is None


def test_audit_sweep_30_metals_synthetic_runs(tmp_path):
    """End-to-end audit: 30 metals × 5 seeds, synthetic CPU-only."""
    out_dir = tmp_path / "wf130_smoke"
    rc = w130.main([
        "--metals", "30",
        "--seeds", "0", "1", "2", "3", "4",
        "--synthetic-only",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    # Combined sweep JSON + synthesis.md exist.
    assert (out_dir / "catherine_audit_30metals.json").exists()
    assert (out_dir / "synthesis.md").exists()
    payload = json.loads(
        (out_dir / "catherine_audit_30metals.json").read_text()
    )
    assert payload["metadata"]["n_metals"] == 30
    assert payload["metadata"]["n_seeds"] == 5
    assert payload["metadata"]["synthetic_only"] is True
    # All 30 metals present.
    assert set(payload["metals"].keys()) == set(w130.DEFAULT_30_METALS)
    # All 30 metals 6/6 PASS on the synthetic path.
    n_pass = sum(
        1 for m in payload["metals"].values()
        if m["six_of_six_pass"]
    )
    assert n_pass == 30, (
        f"Wf-130 P1 ship target requires 30/30 metals × 6/6 PASS; "
        f"got {n_pass}/30"
    )


def test_audit_sweep_writes_30_metal_dirs(tmp_path):
    """End-to-end: 30 per-metal directories with catherine_audit.json."""
    out_dir = tmp_path / "wf130_30dirs"
    rc = w130.main([
        "--metals", "30",
        "--seeds", "0",
        "--synthetic-only",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    metal_dirs = sorted(d.name for d in out_dir.iterdir() if d.is_dir())
    assert len(metal_dirs) == 30
    # Every per-metal JSON has 6 properties recorded.
    for metal in w130.DEFAULT_30_METALS:
        p = out_dir / metal / "catherine_audit.json"
        assert p.exists(), f"missing per-metal audit for {metal}"
        d = json.loads(p.read_text())
        assert d["six_of_six_pass"] is True
        assert d["n_pass"] == 6
        assert set(d["per_property"].keys()) == set(
            w130.CATHERINE_PROPERTIES
        )


# ---------------------------------------------------------------------------
# 5. Honest framing + back-link tests
# ---------------------------------------------------------------------------


def test_synthesis_md_renders_headline_verdict(tmp_path):
    """``_render_synthesis`` produces a Markdown with the 30/30 headline."""
    per_metal_rows = [
        {"metal": m, "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True}
        for m in w130.DEFAULT_30_METALS
    ]
    for r in per_metal_rows:
        d = tmp_path / r["metal"]
        d.mkdir()
        (d / "catherine_audit.json").write_text(json.dumps({
            "metal": r["metal"],
            "six_of_six_pass": True,
            "n_pass": 6,
            "n_fail": 0,
            "passed_properties": list(w130.CATHERINE_PROPERTIES),
            "failed_properties": [],
            "per_property": {
                pid: {"passed": True, "n_samples": 1, "n_violations": 0,
                      "wall_time_s": 0.0, "notes": ""}
                for pid in w130.CATHERINE_PROPERTIES
            },
            "wall_time_s": 0.0,
        }))
    md = w130._render_synthesis(
        metadata={
            "n_metals": 30, "n_seeds": 5, "max_depth": 1,
            "real_bfs_available": False,
            "timestamp": "2026-09-27T00:00:00Z",
        },
        per_metal_rows=per_metal_rows,
        out_dir=tmp_path,
    )
    assert "Wf-130 Phase 1" in md
    assert "30 / 30 metals" in md
    assert "Headline verdict" in md
    assert "Per-property pass-rate" in md
    # All 30 metals appear in the per-metal table.
    for m in w130.DEFAULT_30_METALS:
        assert m in md, f"missing {m} from synthesis.md"


def test_synthesis_md_records_back_link_to_wf68():
    """``_render_synthesis`` records the Wf-68 back-link in the prose."""
    per_metal_rows = [
        {"metal": "Pt", "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True},
    ]
    md = w130._render_synthesis(
        metadata={
            "n_metals": 1, "n_seeds": 1, "max_depth": 1,
            "real_bfs_available": False,
            "timestamp": "2026-09-27T00:00:00Z",
        },
        per_metal_rows=per_metal_rows,
        out_dir=Path("/tmp"),
    )
    assert "Wf-68" in md
    assert "first 25 metals" in md
    assert "all 30" in md


def test_synthesis_md_honest_framing_when_synthetic(tmp_path):
    """``_render_synthesis`` notes the synthetic fallback in honest framing."""
    per_metal_rows = [
        {"metal": "Pt", "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True},
        {"metal": "Au", "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True},
    ]
    md = w130._render_synthesis(
        metadata={
            "n_metals": 2, "n_seeds": 1, "max_depth": 1,
            "real_bfs_available": False,
            "timestamp": "2026-09-27T00:00:00Z",
        },
        per_metal_rows=per_metal_rows,
        out_dir=tmp_path,
    )
    assert "Honest framing" in md
    assert "synthetic fallback" in md.lower() or "synthetic" in md.lower()