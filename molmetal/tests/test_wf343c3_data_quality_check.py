"""Wf-343c-I3 — tests for the pre-retrain data quality check script.

================================================================
Six focused tests covering the public contract of
``molmetal/scripts/wf343c3_data_quality_check.py``:

  1. **Build report shape** — ``build_report()`` returns exactly 5
     LoaderReport rows; each has 5 CheckResult entries.
  2. **CSV-loads check status** — every loader whose underlying CSV /
     JSON is on disk returns ``PASS`` for the ``csv_loads`` check.
  3. **Per-metal balance for multi-metal loaders** — the multi-metal
     loaders (``load_binding_db``, ``metalcytotox_5metal_multitask_v4``)
     return ``PASS`` for ``per_metal_balance`` with the relaxed 1 %
     floor (MEASURED: Ru dominates, Os/Re/Rh sit at 1-5 %).
  4. **Scaffold-split check** — ``metalcytotox_5metal_multitask_v4``
     returns ``PASS`` for ``scaffold_split`` (pre-computed
     ``scaffold_id`` + ``split`` columns on the source CSV).
  5. **Label-range check** — every loader returns ``PASS`` for
     ``label_range`` (the MEASURED pIC50 floors are all ≥ 2.5).
  6. **Per-check / per-loader / n_tests / integration counts** — the
     report contract surfaces the right ``per_check_count`` (5),
     ``per_loader_status`` list, and ``n_tests = 5 * 5 = 25`` check
     rows total.  Also pins the JSON report schema (file written).

All tests are **skip-if-unavailable** for the DrugOOD / metalcytotox
loaders (their source artefacts are not on every CI machine) but
otherwise exercise the full path.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wf343c3_data_quality_check import (  # noqa: E402
    CANONICAL_COLUMNS,
    CheckResult,
    DEFAULT_OUT_DIR,
    LoaderReport,
    PIC50_SOFT_FLOOR,
    build_report,
    write_report_json,
    write_summary_md,
)

# Skip conditions for on-disk data
from molmetal.data.binding_db import DEFAULT_CSV as BINDING_DB_CSV  # noqa: E402
from molmetal.data.nci60 import DEFAULT_CSV as NCI60_CSV  # noqa: E402
from molmetal.data.drugood_loader import (  # noqa: E402
    DEFAULT_DRUGOOD_TARGET_ROOT,
)

_DRUGOOD_JSON = DEFAULT_DRUGOOD_TARGET_ROOT / "lbap_refined_ic50_assay.json"
_DRUGOOD_AVAILABLE = _DRUGOOD_JSON.exists()
_DRUGOOD_SKIP = f"DrugOOD JSON not found at {_DRUGOOD_JSON}"

_METALCYTO_V4 = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
_METALCYTO_AVAILABLE = _METALCYTO_V4.exists()
_METALCYTO_SKIP = f"MetalCytoTox v4 CSV not found at {_METALCYTO_V4}"

_PLATINAI_4CELL = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
_PLATINAI_AVAILABLE = _PLATINAI_4CELL.exists()
_PLATINAI_SKIP = f"PlatinAI 4-cell CSV not found at {_PLATINAI_4CELL}"

_BD_AVAILABLE = BINDING_DB_CSV.exists()
_BD_SKIP = f"BindingDB CSV not found at {BINDING_DB_CSV}"

_NCI_AVAILABLE = NCI60_CSV.exists()
_NCI_SKIP = f"NCI-60 CSV not found at {NCI60_CSV}"

EXPECTED_LOADERS = (
    "load_binding_db",
    "load_nci60",
    "load_drugood_train",
    "metalcytotox_5metal_multitask_v4",
    "platinai_4cell",
)

EXPECTED_CHECK_NAMES = (
    "csv_loads",
    "per_metal_balance",
    "scaffold_split",
    "class_balance",
    "label_range",
)


# ---------------------------------------------------------------------------
# 1. Build report shape
# ---------------------------------------------------------------------------
def test_wf343c3_build_report_shape() -> None:
    """``build_report()`` returns 5 LoaderReports × 5 CheckResults each."""
    reports = build_report()
    assert isinstance(reports, list)
    assert len(reports) == 5, f"expected 5 loaders, got {len(reports)}"
    observed_loaders = tuple(r.loader for r in reports)
    assert observed_loaders == EXPECTED_LOADERS, (
        f"loader set drifted: {observed_loaders} != {EXPECTED_LOADERS}"
    )
    for rep in reports:
        assert isinstance(rep, LoaderReport)
        assert len(rep.checks) == 5, (
            f"{rep.loader}: expected 5 checks, got {len(rep.checks)}"
        )
        observed_checks = tuple(c.name for c in rep.checks)
        assert observed_checks == EXPECTED_CHECK_NAMES, (
            f"{rep.loader}: check order drifted {observed_checks} != "
            f"{EXPECTED_CHECK_NAMES}"
        )
        for c in rep.checks:
            assert isinstance(c, CheckResult)
            assert c.status in ("PASS", "FAIL", "SKIP"), (
                f"{rep.loader}.{c.name}: bad status {c.status}"
            )


# ---------------------------------------------------------------------------
# 2. CSV-loads check status
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _BD_AVAILABLE, reason=_BD_SKIP)
@pytest.mark.skipif(not _NCI_AVAILABLE, reason=_NCI_SKIP)
@pytest.mark.skipif(not _DRUGOOD_AVAILABLE, reason=_DRUGOOD_SKIP)
@pytest.mark.skipif(not _METALCYTO_AVAILABLE, reason=_METALCYTO_SKIP)
@pytest.mark.skipif(not _PLATINAI_AVAILABLE, reason=_PLATINAI_SKIP)
def test_wf343c3_csv_loads_check_passes_for_all_loaders() -> None:
    """Every loader's CSV / JSON file is loadable + canonical schema."""
    reports = build_report()
    for rep in reports:
        c = next(c for c in rep.checks if c.name == "csv_loads")
        assert c.status == "PASS", (
            f"{rep.loader} csv_loads should be PASS, got {c.status}: "
            f"{c.detail}"
        )
        assert "n_rows" in c.metrics, (
            f"{rep.loader} csv_loads missing n_rows metric"
        )
        assert c.metrics["n_rows"] > 0, (
            f"{rep.loader} csv_loads reports 0 rows"
        )


# ---------------------------------------------------------------------------
# 3. Per-metal balance for multi-metal loaders
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _BD_AVAILABLE, reason=_BD_SKIP)
@pytest.mark.skipif(not _METALCYTO_AVAILABLE, reason=_METALCYTO_SKIP)
def test_wf343c3_per_metal_balance_passes_for_multi_metal_loaders() -> None:
    """Multi-metal loaders (BindingDB + MetalCytoTox v4) pass the 1 % floor."""
    reports = build_report()
    for loader_name in ("load_binding_db", "metalcytotox_5metal_multitask_v4"):
        rep = next(r for r in reports if r.loader == loader_name)
        c = next(c for c in rep.checks if c.name == "per_metal_balance")
        assert c.status == "PASS", (
            f"{loader_name} per_metal_balance should be PASS, got "
            f"{c.status}: {c.detail}"
        )
        n_metals = c.metrics.get("n_metals", 0)
        assert n_metals >= 5, (
            f"{loader_name} per_metal_balance n_metals={n_metals} < 5"
        )
        fracs = c.metrics.get("fractions", {})
        for m, f in fracs.items():
            assert f > 0, (
                f"{loader_name} metal {m} has zero fraction"
            )


# ---------------------------------------------------------------------------
# 4. Scaffold-split check (only MetalCytoTox has pre-computed split)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _METALCYTO_AVAILABLE, reason=_METALCYTO_SKIP)
def test_wf343c3_scaffold_split_check_passes_for_metalcytotox_v4() -> None:
    """MetalCytoTox v4 has pre-computed ``scaffold_id`` + ``split`` cols → PASS."""
    reports = build_report()
    rep = next(r for r in reports if r.loader == "metalcytotox_5metal_multitask_v4")
    c = next(c for c in rep.checks if c.name == "scaffold_split")
    assert c.status == "PASS", (
        f"metalcytotox_v4 scaffold_split should be PASS, got {c.status}: "
        f"{c.detail}"
    )
    split_counts = c.metrics.get("split_counts", {})
    for split_name in ("train", "val", "test"):
        assert split_counts.get(split_name, 0) > 0, (
            f"metalcytotox_v4 split {split_name!r} has 0 rows: "
            f"{split_counts}"
        )


# ---------------------------------------------------------------------------
# 5. Label-range check
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _BD_AVAILABLE, reason=_BD_SKIP)
@pytest.mark.skipif(not _NCI_AVAILABLE, reason=_NCI_SKIP)
@pytest.mark.skipif(not _DRUGOOD_AVAILABLE, reason=_DRUGOOD_SKIP)
@pytest.mark.skipif(not _METALCYTO_AVAILABLE, reason=_METALCYTO_SKIP)
@pytest.mark.skipif(not _PLATINAI_AVAILABLE, reason=_PLATINAI_SKIP)
def test_wf343c3_label_range_check_passes_for_all_loaders() -> None:
    """Every loader's pIC50 floor sits at or above the SOFT_FLOOR (= 2.5)."""
    reports = build_report()
    for rep in reports:
        c = next(c for c in rep.checks if c.name == "label_range")
        assert c.status == "PASS", (
            f"{rep.loader} label_range should be PASS, got {c.status}: "
            f"{c.detail}"
        )
        pmin = c.metrics.get("min")
        pstd = c.metrics.get("std")
        assert pmin is not None and math.isfinite(float(pmin))
        assert pmin >= PIC50_SOFT_FLOOR, (
            f"{rep.loader} pIC50 min={pmin} < soft-floor {PIC50_SOFT_FLOOR}"
        )
        assert pstd is not None and float(pstd) > 0, (
            f"{rep.loader} pIC50 has zero variance"
        )


# ---------------------------------------------------------------------------
# 6. Per-check / per-loader / n_tests / integration counts
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _BD_AVAILABLE, reason=_BD_SKIP)
@pytest.mark.skipif(not _NCI_AVAILABLE, reason=_NCI_SKIP)
@pytest.mark.skipif(not _DRUGOOD_AVAILABLE, reason=_DRUGOOD_SKIP)
@pytest.mark.skipif(not _METALCYTO_AVAILABLE, reason=_METALCYTO_SKIP)
@pytest.mark.skipif(not _PLATINAI_AVAILABLE, reason=_PLATINAI_SKIP)
def test_wf343c3_per_check_per_loader_n_tests_integration_status(tmp_path) -> None:
    """End-to-end: report.json has 5 loaders × 5 checks; status mix is sane."""
    reports = build_report()

    # Contract 1 — per_check_count = 5
    for rep in reports:
        assert len(rep.checks) == 5, (
            f"{rep.loader}: per_check_count = {len(rep.checks)} != 5"
        )

    # Contract 2 — per_loader_status covers PASS / FAIL / SKIP only
    statuses = tuple(r.status for r in reports)
    for st in statuses:
        assert st in ("PASS", "FAIL", "SKIP"), (
            f"loader status drifted: {st}"
        )

    # Contract 3 — n_tests (total check rows) = loaders × checks = 25
    n_tests = sum(len(r.checks) for r in reports)
    assert n_tests == 25, f"n_tests = {n_tests} != 25"

    # Contract 4 — JSON report round-trips with the expected schema.
    out_dir = out_dir_for(tmp_path)
    written_json = write_report_json(reports, out_dir)
    assert written_json.exists()
    payload = json.loads(written_json.read_text(encoding="utf-8"))
    assert payload["loader_count"] == 5
    assert payload["checks_per_loader"] == 5
    assert len(payload["loader_reports"]) == 5
    for lr in payload["loader_reports"]:
        assert set(lr.keys()) >= {
            "loader", "csv_path", "status",
            "pass_count", "fail_count", "skip_count", "checks",
        }
        assert len(lr["checks"]) == 5
        assert lr["pass_count"] + lr["fail_count"] + lr["skip_count"] == 5

    # Contract 5 — summary.md round-trips with the expected Markdown table.
    written_md = write_summary_md(reports, out_dir)
    assert written_md.exists()
    body = written_md.read_text(encoding="utf-8")
    for loader_name in EXPECTED_LOADERS:
        assert loader_name in body, (
            f"summary.md missing loader row {loader_name!r}"
        )
    for cn in EXPECTED_CHECK_NAMES:
        assert cn in body, f"summary.md missing check column {cn!r}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def out_dir_for(tmp_path) -> Path:
    """Map pytest's tmp_path to a sub-directory of DEFAULT_OUT_DIR."""
    out = DEFAULT_OUT_DIR / ("test_" + tmp_path.name)
    return out


# Suppress unused-import lint for ``pd`` (used implicitly via loader output).
_ = pd  # type: ignore[has-type]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
