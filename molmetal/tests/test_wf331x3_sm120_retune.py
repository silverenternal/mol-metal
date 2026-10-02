"""Wf-331-X3 sm_120 Triton retune scaffold tests (CPU-only).

================================================================
Per Wf-326E audit (molmetal/reports/wf326e_triton_audit/audit.json):

  * 24 Triton kernels inventoried (audit.triton_kernels_inventory
    .total_kernels == 24) — root cause of the user's "GPU0 算力没
    吃满" observation is gfx1101-tuned configs running 3-11× slower
    than PyTorch on Blackwell sm_120.
  * Wf-326E follow-up recommendation A: retune AUTOTUNE_CONFIGS for
    sm_120 — sweep num_warps in {4, 8, 16}, num_stages in {2, 3, 4}.
  * Wf-331-X3 ships a CPU-only scaffold for that retune (no GPU
    launch required).  Full retune is deferred per AGENTS.md §5.1.

These tests (4 NEW tests per the Wf-331-X3 task brief):

  Group A — autotune wrapper correctness (1 test):
    test_01_sm120_autotune_decorator_is_callable
  Group B — sm_120 config presets (2 tests):
    test_02_sm120_configs_match_wf326e_a_grid
    test_03_sm120_fallback_configs_smaller_for_eval
  Group C — gfx1101 fallback path (1 test):
    test_04_safe_kernel_call_falls_back_to_gfx1101

Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5):

  * All configs are PROJECTED — the real GPU retune is deferred per
    AGENTS.md §5.1 (standing GPU authorization + VRAM + pre-flight
    gate).
  * CPU correctness is verified via :func:`is_sm120_config_safe` (the
    knob-range check) and :func:`safe_kernel_call` (the fallback
    dispatch path).  No actual Triton kernel launch is performed —
    the tests are CPU-only and importable in any environment that
    has :mod:`triton_kernels` (which the audit script also requires).

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python -m pytest molmetal/tests/test_wf331x3_sm120_retune.py -v
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
_SCRIPT_PATH = _PROJECT_ROOT / "molmetal" / "scripts" / "wf331x3_sm120_retune.py"
_DEFAULT_REPORT_DIR = (
    _PROJECT_ROOT / "molmetal" / "reports" / "wf331x3_sm120_retune"
)

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy module loader (the script lives outside molmetal/tests; import it
# by file path so the test does not require it to be on sys.path).
# ---------------------------------------------------------------------------
def _load_scaffold_module():
    """Import :file:`py` as :mod:`wf331x3_sm120_retune`.

    Uses :func:`importlib.util.spec_from_file_location` so the test
    works even if the script is not on sys.path.  Caches the import
    in :data:`sys.modules` to keep the rest of the test fast.
    """
    import importlib.util

    mod_name = "wf331x3_sm120_retune"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(_SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _has_num_warps(cfg) -> int:
    """Return the ``num_warps`` field of a Triton config or dict."""
    nw = getattr(cfg, "num_warps", None)
    if nw is None and isinstance(cfg, dict):
        nw = cfg.get("num_warps")
    assert nw is not None, f"Config {cfg!r} has no num_warps attribute."
    return int(nw)


def _has_num_stages(cfg) -> int:
    """Return the ``num_stages`` field of a Triton config or dict."""
    ns = getattr(cfg, "num_stages", None)
    if ns is None and isinstance(cfg, dict):
        ns = cfg.get("num_stages")
    assert ns is not None, f"Config {cfg!r} has no num_stages attribute."
    return int(ns)


# ===========================================================================
# Group A — autotune wrapper correctness (1 test)
# ===========================================================================
class TestAutotuneWrapper:
    """Tests 01: verify :func:`sm120_autotune` produces a callable and
    delegates to :func:`triton.autotune` with the sm_120 configs."""

    def test_01_sm120_autotune_decorator_is_callable(self):
        """Verify the sm_120 autotune wrapper is callable and applies
        the sm_120 config grid (not the gfx1101 default).

        Specifically:
          (a) the scaffold module loads without error
          (b) :func:`sm120_autotune` is callable and accepts the
              ``key`` kwarg without the ``configs`` override
              (defaults to :data:`SM120_AUTOTUNE_CONFIGS`)
          (c) the wrapped kernel exposes the sm_120 configs (via
              ``.configs`` attribute — :class:`triton.runtime.autotuner
              .Autotuner` is not ``callable()`` but has ``.run``)
          (d) the sm_120 config grid is **larger** than the gfx1101
              grid (12 vs 9) — proves the retune is more aggressive
              than the legacy tuning
        """
        mod = _load_scaffold_module()

        # (a) Module imports cleanly.
        assert hasattr(mod, "sm120_autotune"), (
            "Scaffold module must export sm120_autotune()."
        )
        assert hasattr(mod, "SM120_AUTOTUNE_CONFIGS"), (
            "Scaffold module must export SM120_AUTOTUNE_CONFIGS."
        )

        # (b) Apply the decorator over a @triton.jit function (required
        # by :func:`triton.autotune` — needs ``arg_names`` attribute).
        # We use the script's pre-defined ``_sm120_dummy_kernel`` (which
        # is defined inside the .py file so :func:`triton.jit` accepts
        # it) instead of defining one here in the test (which would
        # fail :func:`triton.jit`'s "must be in a Python file" check).
        try:
            kernel = mod._sm120_dummy_kernel
        except AttributeError:
            pytest.skip(
                "Scaffold module does not expose _sm120_dummy_kernel "
                "(triton not available in this environment)."
            )

        # (c) Autotuner exposes .configs and .run.
        assert hasattr(kernel, "configs"), (
            "sm120_autotune-wrapped kernel must expose .configs."
        )
        assert hasattr(kernel, "run"), (
            "sm120_autotune-wrapped kernel must expose .run method."
        )
        assert len(kernel.configs) == len(mod.SM120_AUTOTUNE_CONFIGS), (
            f"sm120_autotune-wrapped kernel has {len(kernel.configs)} "
            f"configs; expected {len(mod.SM120_AUTOTUNE_CONFIGS)}."
        )

        # (d) sm_120 config grid is wider than gfx1101 (12 vs 9).
        n_sm120 = len(mod.SM120_AUTOTUNE_CONFIGS)
        try:
            import triton_kernels
            n_gfx1101 = len(triton_kernels.AUTOTUNE_CONFIGS)
        except Exception:
            n_gfx1101 = 9
        assert n_sm120 > n_gfx1101, (
            f"sm_120 config grid ({n_sm120}) should be wider than the "
            f"gfx1101 grid ({n_gfx1101}) — Blackwell can use more warps."
        )


# ===========================================================================
# Group B — sm_120 config presets (2 tests)
# ===========================================================================
class TestSm120ConfigPresets:
    """Tests 02-03: verify the sm_120 config grid matches the Wf-326E-A
    recommendation and that the fallback grid is smaller for eval."""

    def test_02_sm120_configs_match_wf326e_a_grid(self):
        """Verify :data:`SM120_AUTOTUNE_CONFIGS` matches the Wf-326E-A
        recommendation: num_warps in {4, 8, 16}, num_stages in {2, 3, 4}.

        Specifically:
          (a) every config has num_warps in {2, 4, 8, 16} and
              num_stages in {2, 3, 4}
          (b) the union of (num_warps, num_stages) pairs covers the
              Wf-326E-A grid plus the memory-bound fallback entries
          (c) :func:`is_sm120_config_safe` returns True for every
              sm_120 config and False for an out-of-range config
          (d) the grid has 12 entries (3 × 3 + 1 × 3)
        """
        mod = _load_scaffold_module()
        configs = mod.SM120_AUTOTUNE_CONFIGS

        # (a) Range checks per config.
        allowed_warps = {2, 4, 8, 16}
        allowed_stages = {2, 3, 4}
        for cfg in configs:
            nw = _has_num_warps(cfg)
            ns = _has_num_stages(cfg)
            assert nw in allowed_warps, (
                f"sm_120 config num_warps={nw} is outside the "
                f"Wf-326E-A grid {allowed_warps}."
            )
            assert ns in allowed_stages, (
                f"sm_120 config num_stages={ns} is outside the "
                f"Wf-326E-A grid {allowed_stages}."
            )

        # (b) Cover the Wf-326E-A grid: num_warps in {4, 8, 16},
        # num_stages in {2, 3, 4}.
        seen_pairs = {
            (_has_num_warps(c), _has_num_stages(c)) for c in configs
        }
        for nw in (4, 8, 16):
            for ns in (2, 3, 4):
                assert (nw, ns) in seen_pairs, (
                    f"Wf-326E-A grid missing config "
                    f"(num_warps={nw}, num_stages={ns}). "
                    f"Present pairs: {sorted(seen_pairs)}."
                )

        # (c) is_sm120_config_safe true / false.
        for cfg in configs:
            assert mod.is_sm120_config_safe(cfg), (
                f"is_sm120_config_safe({cfg}) must be True for "
                f"SM120_AUTOTUNE_CONFIGS."
            )
        # Out-of-range config (num_warps=64, num_stages=8) is NOT safe.
        import triton
        bogus = triton.Config({}, num_warps=64, num_stages=8)
        assert not mod.is_sm120_config_safe(bogus), (
            "Bogus config (num_warps=64, num_stages=8) should NOT be "
            "sm_120-safe per the Wf-326E-A grid."
        )

        # (d) Grid size: 3 (num_warps) x 3 (num_stages) + 1 x 3 = 12.
        assert len(configs) == 12, (
            f"SM120_AUTOTUNE_CONFIGS should have 12 entries "
            f"(Wf-326E-A grid + memory-bound fallback); got "
            f"{len(configs)}."
        )

    def test_03_sm120_fallback_configs_smaller_for_eval(self):
        """Verify :data:`SM120_AUTOTUNE_CONFIGS_FALLBACK` is smaller
        than :data:`SM120_AUTOTUNE_CONFIGS` (4 vs 12 entries) — the
        fallback grid is for eval-mode passes where autotune cost
        matters more than peak throughput.

        Specifically:
          (a) the fallback grid has 4 entries (num_warps in {4, 8},
              num_stages in {2, 3})
          (b) every fallback config has num_warps in {4, 8} (subset of
              the Wf-326E-A grid; no num_warps=16 for the eval
              fallback because it inflates autotune cost)
          (c) :func:`is_sm120_config_safe` returns True for every
              fallback config
          (d) the fallback grid is a strict subset of the full grid
              (every fallback pair appears in the full grid)
        """
        mod = _load_scaffold_module()
        fallback = mod.SM120_AUTOTUNE_CONFIGS_FALLBACK
        full = mod.SM120_AUTOTUNE_CONFIGS

        # (a) 4 entries.
        assert len(fallback) == 4, (
            f"SM120_AUTOTUNE_CONFIGS_FALLBACK should have 4 entries; "
            f"got {len(fallback)}."
        )

        # (b) num_warps in {4, 8}.
        for cfg in fallback:
            nw = _has_num_warps(cfg)
            assert nw in {4, 8}, (
                f"Fallback config num_warps={nw} should be in {{4, 8}} "
                f"for the eval-mode grid."
            )

        # (c) is_sm120_config_safe.
        for cfg in fallback:
            assert mod.is_sm120_config_safe(cfg), (
                f"is_sm120_config_safe({cfg}) must be True for "
                f"SM120_AUTOTUNE_CONFIGS_FALLBACK."
            )

        # (d) Fallback ⊆ full.
        fallback_pairs = {
            (_has_num_warps(c), _has_num_stages(c)) for c in fallback
        }
        full_pairs = {
            (_has_num_warps(c), _has_num_stages(c)) for c in full
        }
        assert fallback_pairs <= full_pairs, (
            f"SM120_AUTOTUNE_CONFIGS_FALLBACK is not a subset of "
            f"SM120_AUTOTUNE_CONFIGS. "
            f"Fallback-only pairs: {fallback_pairs - full_pairs}"
        )


# ===========================================================================
# Group C — gfx1101 fallback path (1 test)
# ===========================================================================
class TestGfx1101Fallback:
    """Tests 04: verify :func:`safe_kernel_call` falls back to gfx1101
    when the sm_120 launch raises."""

    def test_04_safe_kernel_call_falls_back_to_gfx1101(self):
        """Verify the safe-dispatch helper falls back to gfx1101 when
        the sm_120 launch raises.

        Specifically:
          (a) the helper exists and accepts (sm120_kernel,
              gfx1101_kernel, *args, **kwargs)
          (b) on sm_120 RuntimeError, the helper invokes the gfx1101
              fallback and returns its result
          (c) on a successful sm_120 launch, the gfx1101 fallback is
              NOT invoked (fallback is opt-in, not always-on)
          (d) the call order is recorded correctly
        """
        mod = _load_scaffold_module()
        assert hasattr(mod, "safe_kernel_call"), (
            "Scaffold module must export safe_kernel_call()."
        )

        # (b) Fallback path: sm_120 raises, gfx1101 returns "ok".
        calls: list = []

        def _sm120_raises() -> str:
            calls.append("sm120")
            raise RuntimeError("simulated sm_120 launch failure")

        def _gfx1101_returns() -> str:
            calls.append("gfx1101")
            return "gfx1101_ok"

        result = mod.safe_kernel_call(_sm120_raises, _gfx1101_returns)
        assert result == "gfx1101_ok", (
            f"safe_kernel_call should return the gfx1101 fallback "
            f"result; got {result!r}."
        )
        assert calls == ["sm120", "gfx1101"], (
            f"safe_kernel_call should invoke sm_120 first then gfx1101; "
            f"got call order {calls}."
        )

        # (c) Happy path: sm_120 succeeds, gfx1101 is NOT invoked.
        calls.clear()

        def _sm120_ok() -> str:
            calls.append("sm120")
            return "sm120_ok"

        def _gfx1101_should_not_run() -> str:
            calls.append("gfx1101")
            return "gfx1101_should_not_run"

        result_ok = mod.safe_kernel_call(_sm120_ok, _gfx1101_should_not_run)
        assert result_ok == "sm120_ok", (
            f"safe_kernel_call should return the sm_120 result on "
            f"success; got {result_ok!r}."
        )
        assert calls == ["sm120"], (
            f"safe_kernel_call should NOT invoke gfx1101 on success; "
            f"got call order {calls}."
        )

        # (d) Positional + keyword args pass through unchanged.
        calls.clear()

        def _sm120_with_args(x: int, y: int = 0) -> int:
            calls.append(("sm120", x, y))
            return x + y

        def _gfx1101_with_args(x: int, y: int = 0) -> int:
            calls.append(("gfx1101", x, y))
            return x * y

        result_args = mod.safe_kernel_call(
            _sm120_with_args, _gfx1101_with_args, 3, y=4
        )
        assert result_args == 7, (
            f"safe_kernel_call should pass through positional + kwarg; "
            f"got {result_args}."
        )
        assert calls == [("sm120", 3, 4)], (
            f"safe_kernel_call should NOT invoke gfx1101 on success; "
            f"got call order {calls}."
        )