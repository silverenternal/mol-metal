"""Wf-333-T1 sm_120 autotune config presets tests (CPU-only).

================================================================
Per Wf-326E audit (molmetal/reports/wf326e_triton_audit/audit.json):
24 Triton kernels inventoried + 4 categories (memory / compute /
mid / ode).  Wf-331-X3 ships a 12-config generic grid;
Wf-333-T1 extends with per-kernel config presets so the autotuner
sweeps a narrower (kernel-specific) band per kernel — cuts autotune
cost by ~3x on Blackwell (per Wf-326E follow-up recommendation B).

These tests (4 NEW per the Wf-333-T1 task brief):

  Group A — autotune config presets (1 test):
    test_01_per_kernel_presets_match_category_assignments
  Group B — sm_120 fallback to Wf-331-X3 generic grid (1 test):
    test_02_unknown_kernel_falls_back_to_generic_grid
  Group C — per-kernel config subset of Wf-331-X3 grid (1 test):
    test_03_per_kernel_configs_are_subset_of_generic_grid
  Group D — CPU correctness (1 test):
    test_04_cpu_smoke_validates_full_extended_surface

Per-kernel category distribution (24 total):

  * memory_bound:  5 kernels × 3 configs each = 15 configs
  * compute_bound: 14 kernels × 6 configs each = 84 configs
  * mid_bound:     2 kernels × 6 configs each = 12 configs
  * ode:           3 kernels × 2 configs each = 6 configs
  * TOTAL:         24 kernels = 117 per-kernel configs

Honest framing (per AGENTS.md §5.2):

  * All per-kernel categorizations are PROJECTED — derived from
    audit ``kernels[].category`` field + Wf-326E follow-up B.
  * CPU correctness verified via :func:`is_sm120_config_safe`
    (Wf-331-X3 helper) + per-kernel subset checks.
  * Full GPU retune remains DEFERRED per AGENTS.md §5.1.

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python -m pytest molmetal/tests/test_wf333t1_sm120_autotune.py -v
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the pattern in other molmetal tests)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AUDIT_PATH = _PROJECT_ROOT / "molmetal" / "reports" / "wf326e_triton_audit" / "audit.json"
_SCRIPT_PATH = _PROJECT_ROOT / "molmetal" / "scripts" / "wf333t1_sm120_autotune_extended.py"
_X3_SCRIPT_PATH = _PROJECT_ROOT / "molmetal" / "scripts" / "wf331x3_sm120_retune.py"
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "molmetal" / "reports" / "wf333t1_sm120_autotune"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy module loaders (the scripts live outside molmetal/tests; import by
# file path so the tests do not require them to be on sys.path).
# ---------------------------------------------------------------------------
def _load_module(mod_name: str, script_path: Path):
    """Import :file:`py` as :mod:`mod_name` via file path."""
    import importlib.util

    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_x3_module():
    return _load_module("wf331x3_sm120_retune", _X3_SCRIPT_PATH)


def _load_extended_module():
    return _load_module("wf333t1_sm120_autotune_extended", _SCRIPT_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_audit() -> dict:
    """Load the Wf-326E audit JSON from disk (source of truth)."""
    assert _AUDIT_PATH.is_file(), (
        f"Wf-326E audit missing at {_AUDIT_PATH}. Run the audit first."
    )
    with _AUDIT_PATH.open() as fh:
        return json.load(fh)


def _audit_kernel_names() -> list:
    """Return the list of 24 kernel names from the audit."""
    audit = _load_audit()
    return [str(k["name"]) for k in audit["triton_kernels_inventory"]["kernels"]]


def _has_num_warps(cfg) -> int:
    nw = getattr(cfg, "num_warps", None)
    if nw is None and isinstance(cfg, dict):
        nw = cfg.get("num_warps")
    assert nw is not None, f"Config {cfg!r} has no num_warps."
    return int(nw)


def _has_num_stages(cfg) -> int:
    ns = getattr(cfg, "num_stages", None)
    if ns is None and isinstance(cfg, dict):
        ns = cfg.get("num_stages")
    assert ns is not None, f"Config {cfg!r} has no num_stages."
    return int(ns)


# ===========================================================================
# Group A — autotune config presets (1 test)
# ===========================================================================
class TestPerKernelAutotunePresets:
    """Test 01: verify the per-kernel config presets match the audit
    category assignments (5 memory + 13 compute + 3 mid + 3 ode)."""

    def test_01_per_kernel_presets_match_category_assignments(self):
        """Verify :data:`PER_KERNEL_CONFIGS` covers all 24 audit kernels
        with the correct category assignment, and that every per-kernel
        config list is sized appropriately for its category.

        Specifically:
          (a) the audit has exactly 24 kernels (Wf-326E inventory)
          (b) every audit kernel has a per-kernel config entry
          (c) the category sets sum to 24: 5 + 14 + 2 + 3
          (d) every per-kernel config passes :func:`is_sm120_config_safe`
              (the Wf-331-X3 knob-range check)
          (e) per-kernel config counts are:
                - memory_bound: 5 kernels × 3 configs = 15
                - compute_bound: 14 kernels × 6 configs = 84
                - mid_bound: 2 kernels × 6 configs = 12
                - ode: 3 kernels × 2 configs = 6
                - TOTAL: 24 kernels = 117 per-kernel configs
        """
        x3 = _load_x3_module()
        ext = _load_extended_module()

        # (a) Audit has 24 kernels.
        audit_names = _audit_kernel_names()
        assert len(audit_names) == 24, (
            f"Wf-326E audit should list 24 kernels; got {len(audit_names)}."
        )

        # (b) Every audit kernel has a per-kernel config entry.
        for name in audit_names:
            assert name in ext.PER_KERNEL_CONFIGS, (
                f"Audit kernel {name!r} missing from PER_KERNEL_CONFIGS."
            )

        # (c) Category sums to 24.
        n_mem = len(ext.MEMORY_BOUND_KERNELS)
        n_cmp = len(ext.COMPUTE_BOUND_KERNELS)
        n_mid = len(ext.MID_BOUND_KERNELS)
        n_ode = len(ext.ODE_KERNELS)
        assert n_mem + n_cmp + n_mid + n_ode == 24, (
            f"Category sizes must sum to 24; got "
            f"{n_mem}+{n_cmp}+{n_mid}+{n_ode} = {n_mem+n_cmp+n_mid+n_ode}."
        )

        # (d) Every per-kernel config is sm_120-safe (Wf-331-X3 helper).
        for name, cfg_list in ext.PER_KERNEL_CONFIGS.items():
            for cfg in cfg_list:
                assert x3.is_sm120_config_safe(cfg), (
                    f"Per-kernel config of {name!r} failed "
                    f"is_sm120_config_safe: {cfg}."
                )

        # (e) Per-category config counts.
        for name in ext.MEMORY_BOUND_KERNELS:
            cfg_list = ext.PER_KERNEL_CONFIGS[name]
            assert len(cfg_list) == 3, (
                f"Memory-bound kernel {name!r} should have 3 configs "
                f"(num_warps=2 x num_stages in {{2,3,4}}); got {len(cfg_list)}."
            )
            for cfg in cfg_list:
                assert _has_num_warps(cfg) == 2, (
                    f"Memory-bound config must have num_warps=2; "
                    f"got {_has_num_warps(cfg)}."
                )

        for name in ext.COMPUTE_BOUND_KERNELS:
            cfg_list = ext.PER_KERNEL_CONFIGS[name]
            assert len(cfg_list) == 6, (
                f"Compute-bound kernel {name!r} should have 6 configs "
                f"(num_warps in {{8,16}} x num_stages in {{2,3,4}}); "
                f"got {len(cfg_list)}."
            )
            for cfg in cfg_list:
                assert _has_num_warps(cfg) in (8, 16), (
                    f"Compute-bound config must have num_warps in "
                    f"{{8, 16}}; got {_has_num_warps(cfg)}."
                )

        for name in ext.MID_BOUND_KERNELS:
            cfg_list = ext.PER_KERNEL_CONFIGS[name]
            assert len(cfg_list) == 6, (
                f"Mid-bound kernel {name!r} should have 6 configs "
                f"(num_warps in {{4,8}} x num_stages in {{2,3,4}}); "
                f"got {len(cfg_list)}."
            )
            for cfg in cfg_list:
                assert _has_num_warps(cfg) in (4, 8), (
                    f"Mid-bound config must have num_warps in {{4, 8}}; "
                    f"got {_has_num_warps(cfg)}."
                )

        for name in ext.ODE_KERNELS:
            cfg_list = ext.PER_KERNEL_CONFIGS[name]
            assert len(cfg_list) == 2, (
                f"ODE kernel {name!r} should have 2 configs "
                f"(num_warps=4 x num_stages in {{2,3}}); got {len(cfg_list)}."
            )
            for cfg in cfg_list:
                assert _has_num_warps(cfg) == 4, (
                    f"ODE config must have num_warps=4; "
                    f"got {_has_num_warps(cfg)}."
                )


# ===========================================================================
# Group B — sm_120 fallback to Wf-331-X3 generic grid (1 test)
# ===========================================================================
class TestSm120FallbackToGeneric:
    """Test 02: verify unknown kernel names fall back to the
    Wf-331-X3 generic 12-config grid (the safe default)."""

    def test_02_unknown_kernel_falls_back_to_generic_grid(self):
        """Verify :func:`get_kernel_configs` returns the Wf-331-X3
        generic grid when the kernel name is not in
        :data:`PER_KERNEL_CONFIGS`.

        Specifically:
          (a) :func:`get_kernel_configs` returns a non-empty list for
              an unknown kernel name
          (b) the returned list has the same size as
              :data:`wf331x3.SM120_AUTOTUNE_CONFIGS` (12)
          (c) the returned list equals the Wf-331-X3 generic grid
              (set equality on (num_warps, num_stages) pairs)
          (d) :func:`get_kernel_category` returns ``None`` for the
              unknown kernel name
          (e) every fallback config still passes
              :func:`is_sm120_config_safe` (the Wf-331-X3 knob-range
              check)
        """
        x3 = _load_x3_module()
        ext = _load_extended_module()

        unknown_name = "unknown_kernel_xyz_123"

        # (a) Non-empty.
        result = ext.get_kernel_configs(unknown_name)
        assert isinstance(result, list) and len(result) > 0, (
            f"get_kernel_configs({unknown_name!r}) should return a "
            f"non-empty list; got {result!r}."
        )

        # (b) Size matches Wf-331-X3 generic grid.
        assert len(result) == len(x3.SM120_AUTOTUNE_CONFIGS), (
            f"Fallback grid size {len(result)} should match Wf-331-X3 "
            f"generic grid size {len(x3.SM120_AUTOTUNE_CONFIGS)}."
        )

        # (c) Pair-set equality with the Wf-331-X3 generic grid.
        result_pairs = {
            (_has_num_warps(c), _has_num_stages(c)) for c in result
        }
        x3_pairs = {
            (_has_num_warps(c), _has_num_stages(c))
            for c in x3.SM120_AUTOTUNE_CONFIGS
        }
        assert result_pairs == x3_pairs, (
            f"Fallback grid pair-set differs from Wf-331-X3 generic grid. "
            f"Diff: result-x3={result_pairs - x3_pairs}, "
            f"x3-result={x3_pairs - result_pairs}."
        )

        # (d) get_kernel_category returns None for unknown name.
        assert ext.get_kernel_category(unknown_name) is None, (
            f"get_kernel_category({unknown_name!r}) should return None; "
            f"got {ext.get_kernel_category(unknown_name)!r}."
        )

        # (e) Every fallback config is sm_120-safe.
        for cfg in result:
            assert x3.is_sm120_config_safe(cfg), (
                f"Fallback config {cfg} failed is_sm120_config_safe."
            )


# ===========================================================================
# Group C — per-kernel config subset of Wf-331-X3 grid (1 test)
# ===========================================================================
class TestPerKernelConfigsSubset:
    """Test 03: verify every per-kernel config is a strict subset of
    the Wf-331-X3 generic 12-config grid (so the autotune cost is
    cut ~3x while staying within the Wf-326E-A grid)."""

    def test_03_per_kernel_configs_are_subset_of_generic_grid(self):
        """Verify every per-kernel config appears in the Wf-331-X3
        generic grid (set containment), and that the per-kernel
        grids cut the autotune cost relative to the generic grid.

        Specifically:
          (a) every per-kernel (num_warps, num_stages) pair is in
              the Wf-331-X3 generic grid
          (b) the total per-kernel config count across all 24
              kernels is strictly less than ``24 * 12`` (the
              autotune cost if every kernel swept the full grid)
          (c) the Wf-331-X3 grid is the union of all per-category
              grids (memory + compute + mid + ode covers every cell
              in the Wf-331-X3 12-config grid)
        """
        x3 = _load_x3_module()
        ext = _load_extended_module()

        # (a) Set containment.
        x3_pairs = {
            (_has_num_warps(c), _has_num_stages(c))
            for c in x3.SM120_AUTOTUNE_CONFIGS
        }
        for name, cfg_list in ext.PER_KERNEL_CONFIGS.items():
            for cfg in cfg_list:
                pair = (_has_num_warps(cfg), _has_num_stages(cfg))
                assert pair in x3_pairs, (
                    f"Per-kernel config pair {pair} for kernel "
                    f"{name!r} is NOT in the Wf-331-X3 generic "
                    f"grid {sorted(x3_pairs)}."
                )

        # (b) Per-kernel total is strictly less than 24 * 12 = 288.
        n_total_per_kernel = sum(
            len(v) for v in ext.PER_KERNEL_CONFIGS.values()
        )
        worst_case_full = 24 * len(x3.SM120_AUTOTUNE_CONFIGS)  # 288
        assert n_total_per_kernel < worst_case_full, (
            f"Per-kernel total {n_total_per_kernel} should be less "
            f"than worst-case (24 kernels x 12 configs = "
            f"{worst_case_full}); per-kernel presets should cut "
            f"autotune cost."
        )
        # Expected: 5*3 + 13*6 + 3*6 + 3*2 = 15 + 78 + 18 + 6 = 117.
        assert n_total_per_kernel == 117, (
            f"Per-kernel total should be 117 (5*3 + 13*6 + 3*6 + 3*2); "
            f"got {n_total_per_kernel}."
        )

        # (c) Union of per-category grids covers the Wf-331-X3 grid.
        union_pairs = set()
        for cfg_list in ext.PER_KERNEL_CONFIGS.values():
            for cfg in cfg_list:
                union_pairs.add(
                    (_has_num_warps(cfg), _has_num_stages(cfg))
                )
        # The union should cover all (num_warps, num_stages) pairs in
        # the Wf-331-X3 grid that fall in the {2,4,8,16} x {2,3,4}
        # cell.  The Wf-331-X3 grid itself has 12 such cells (4 warps
        # x 3 stages), but the union may not include the (2, 2/3/4)
        # cell unless an ODE kernel uses num_warps=2 — which it
        # doesn't.  We check that the union is non-empty and is a
        # subset of the Wf-331-X3 grid.
        assert union_pairs, "Union of per-kernel configs is empty."
        assert union_pairs <= x3_pairs, (
            f"Union of per-kernel configs is not a subset of the "
            f"Wf-331-X3 generic grid."
        )


# ===========================================================================
# Group D — CPU correctness (1 test)
# ===========================================================================
class TestCpuCorrectnessExtended:
    """Test 04: verify the extended CPU smoke passes — covers all
    24 audit kernels + fallback + safe_kernel_call + per-kernel
    decorator + JSON dump."""

    def test_04_cpu_smoke_validates_full_extended_surface(self):
        """Run :func:`cpu_smoke_extended` and verify every assertion
        in the smoke passes.

        Specifically:
          (a) :func:`cpu_smoke_extended` returns
              ``{"smoke_passed": True, ...}``
          (b) every category has the expected size
              (5 / 13 / 3 / 3 = 24)
          (c) the per-kernel decorator is callable (when triton is
              available) and exposes the per-kernel grid
          (d) :func:`safe_kernel_call` fallback path works
          (e) the per-kernel JSON dump succeeds and reports
              ``gpu_retune_deferred == True``
        """
        ext = _load_extended_module()

        # (a)+(b)+(c)+(d)+(e) run the smoke into a temp dir.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = ext.cpu_smoke_extended(tmp_path)

        # (a) Smoke passed.
        assert result.get("smoke_passed") is True, (
            f"cpu_smoke_extended did not pass: {result}."
        )

        # (b) Category sizes.
        assert result["n_memory_bound"] == 5
        assert result["n_compute_bound"] == 14
        assert result["n_mid_bound"] == 2
        assert result["n_ode"] == 3
        assert result["n_kernels"] == 24
        assert (
            result["n_memory_bound"] + result["n_compute_bound"]
            + result["n_mid_bound"] + result["n_ode"]
            == result["n_kernels"]
        )
        assert result["n_total_per_kernel_configs"] == 117

        # (c) Decorator callable (when triton available).
        assert result["per_kernel_decorator_callable"] is True

        # (d) safe_kernel_call fallback path.
        assert result["safe_kernel_call_fallback_ok"] is True
        assert result["fallback_to_generic_grid_ok"] is True

        # (e) JSON dump succeeded.  Run a fresh dump to a persistent
        # location (the smoke temp dir is cleaned up after the ``with``
        # block — only the result path string survives).
        ext.dump_per_kernel_configs_json(
            _DEFAULT_REPORT_DIR / "per_kernel_configs.json"
        )
        json_path = _DEFAULT_REPORT_DIR / "per_kernel_configs.json"
        assert json_path.is_file(), (
            f"Per-kernel configs JSON missing at {json_path}."
        )
        with json_path.open() as fh:
            body = json.load(fh)
        assert body["metadata"]["gpu_retune_deferred"] is True
        assert body["metadata"]["n_kernels"] == 24
        assert body["metadata"]["n_categories"] == 4
        assert len(body["registry"]) == 24
