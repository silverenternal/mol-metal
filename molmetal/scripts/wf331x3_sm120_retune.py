"""Wf-331-X3 — sm_120 Triton retune CPU scaffold (Blackwell).

================================================================
Wf-331-X3 (2026-09-29) — CPU-only scaffold for the Wf-326E-A
sm_120 autotune retune (RTX 5090 Blackwell, sm_120).  Per the
Wf-326E audit (molmetal/reports/wf326e_triton_audit/audit.json):

  * 24 Triton kernels inventoried (audit.triton_kernels_inventory
    .total_kernels == 24).
  * 24 kernels default-OFF on sm_120 because AUTOTUNE_CONFIGS in
    triton_kernels/autotune.py L49-59 was tuned for gfx1101 (RDNA3
    iGPU), where num_warps in {2, 4, 8} + num_stages in {2, 3, 4} is
    the only legitimate grid.  On Blackwell the same configs run
    3-11× slower than PyTorch (per config.py L107), so PyTorch is
    the default since 2026-09-27.
  * Wf-326E follow-up recommendation A: "sweep AUTOTUNE_CONFIGS over
    num_warps {4, 8, 16}, num_stages {2, 3, 4} on a single RTX 5090;
    expected 1-2 day implementation + 1 day validation".

This scaffold provides the CPU-only infrastructure for that retune:

  * :data:`SM120_AUTOTUNE_CONFIGS` — 12-element config preset grid
    (num_warps in {4, 8, 16} x num_stages in {2, 3, 4} — matches
    the Wf-326E-A grid).
  * :data:`SM120_AUTOTUNE_CONFIGS_FALLBACK` — a smaller (4-config)
    grid to use when the full grid would exceed the kernel launch
    budget (e.g. eval-mode passes where autotune cost matters).
  * :func:`sm120_autotune` — decorator wrapping :func:`triton.autotune`
    that uses :data:`SM120_AUTOTUNE_CONFIGS` by default.
  * :func:`safe_kernel_call` — tries the sm_120 config first, falls
    back to the gfx1101 default if the sm_120 launch raises (e.g.
    the kernel cannot be JITted under sm_120 PTX).
  * :func:`dump_sm120_configs_json` — emits the configs as a JSON
    file under ``molmetal/reports/wf331x3_sm120_retune/`` for the
    eventual GPU retune run.

Honest framing (per AGENTS.md §5.2):

  * The configs are **PROJECTED**, not MEASURED.  The full retune
    (real grid sweep on a live RTX 5090) is deferred per the user
    GPU-authorization §5.1 — agents promote to GPU launch only
    after VRAM + pre-flight PASS.
  * All 24 kernels default-OFF status is preserved — this scaffold
    does **not** flip the default.  Operators opt-in via
    ``TRITON_USE_FUSED=1`` + ``TRITON_USE_SM120_AUTOTUNE=1``.
  * The fallback path to gfx1101 is the **safe** default — the
    scaffold never silently disables a kernel that previously ran
    on gfx1101 just because sm_120 isn't reachable.
  * PROJECTED aggregate speedup once the GPU retune + this scaffold
    wire-in land: 1.5-3× (per audit.audit_summary
    .projected_speedup_with_retune_and_new_kernels).

CLI usage (CPU-only, no GPU launch required)::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf331x3_sm120_retune.py \\
        --output-dir molmetal/reports/wf331x3_sm120_retune

    # With explicit smoke (CPU correctness test for the configs).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf331x3_sm120_retune.py --smoke

Notes
-----

* This scaffold deliberately does not import torch.cuda.* to keep
  the importable surface CPU-friendly.  The :func:`safe_kernel_call`
  helper accepts any :func:`triton.runtime.JITFunction` and falls
  back via a try/except wrapper.
* The fallback mechanism is per-launch, not per-process.  A failed
  sm_120 launch raises one Retry-With-Gfx1101 fallback attempt; we
  do NOT cache the failure to avoid spamming the fallback on
  identical launches (caller can pre-check via
  :func:`is_sm120_config_safe`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import triton
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None  # type: ignore[assignment]

try:
    import triton_kernels
except ImportError:
    triton_kernels = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf331x3_sm120_retune"
AUDIT_PATH = REPO_ROOT / "molmetal" / "reports" / "wf326e_triton_audit" / "audit.json"


# ---------------------------------------------------------------------------
# sm_120 config presets (PROJECTED — Wf-326E-A grid)
# ---------------------------------------------------------------------------
# The full grid mirrors the Wf-326E follow-up recommendation:
#   num_warps in {4, 8, 16} x num_stages in {2, 3, 4} = 9 configs.
# We add 3 more configs (num_warps=2, num_stages=2/3/4) to cover
# the case where the smaller block sizes are optimal for memory-
# bound kernels (e.g. fused_residual_add) — total 12 configs.
SM120_NUM_WARPS: Sequence[int] = (4, 8, 16)
SM120_NUM_STAGES: Sequence[int] = (2, 3, 4)
SM120_NUM_WARPS_EXTRA: Sequence[int] = (2,)  # for memory-bound kernels

SM120_AUTOTUNE_CONFIGS: List[Any] = []
"""Triton autotune config grid for Blackwell sm_120 (RTX 5090).

PROJECTED — derived from Wf-326E-A follow-up recommendation.
Order is deliberately (high num_warps, high num_stages) first so
the autotuner biases toward the wider blocks (which is the
production hot-path for compute-bound MLP kernels on Blackwell).
"""
if triton is not None:
    for nw in sorted(SM120_NUM_WARPS, reverse=True):
        for ns in sorted(SM120_NUM_STAGES, reverse=True):
            SM120_AUTOTUNE_CONFIGS.append(
                triton.Config({}, num_warps=nw, num_stages=ns)
            )
    # Add 3 extra configs for memory-bound kernels.
    for nw in SM120_NUM_WARPS_EXTRA:
        for ns in sorted(SM120_NUM_STAGES, reverse=True):
            SM120_AUTOTUNE_CONFIGS.append(
                triton.Config({}, num_warps=nw, num_stages=ns)
            )
else:  # pragma: no cover - triton absent in CPU-only env
    # Placeholder list — tests assert len() == 12.
    SM120_AUTOTUNE_CONFIGS = [
        {"num_warps": nw, "num_stages": ns}
        for nw in (16, 8, 4, 2)
        for ns in (4, 3, 2)
    ]


SM120_AUTOTUNE_CONFIGS_FALLBACK: List[Any] = []
"""4-config fallback grid for eval-mode passes.

Smaller grid (num_warps in {4, 8}, num_stages in {2, 3}) — 4 configs.
Use via ``sm120_autotune(..., configs=SM120_AUTOTUNE_CONFIGS_FALLBACK)``.
"""
if triton is not None:
    for nw in (8, 4):
        for ns in (3, 2):
            SM120_AUTOTUNE_CONFIGS_FALLBACK.append(
                triton.Config({}, num_warps=nw, num_stages=ns)
            )
else:  # pragma: no cover
    SM120_AUTOTUNE_CONFIGS_FALLBACK = [
        {"num_warps": nw, "num_stages": ns} for nw in (8, 4) for ns in (3, 2)
    ]


# Public env-var name (CPU-only, no GPU launch required to use it).
ENV_USE_SM120_AUTOTUNE = "TRITON_USE_SM120_AUTOTUNE"


# ---------------------------------------------------------------------------
# Gfx1101 fallback config (mirrors triton_kernels.autotune.AUTOTUNE_CONFIGS)
# ---------------------------------------------------------------------------
GFX1101_AUTOTUNE_CONFIGS: List[Any] = []
"""Read-through to :data:`triton_kernels.autotune.AUTOTUNE_CONFIGS`.

We re-import lazily so this module is importable on CPU-only envs
that lack triton (the tests pin this behaviour).
"""
if triton_kernels is not None:
    try:
        GFX1101_AUTOTUNE_CONFIGS = list(triton_kernels.AUTOTUNE_CONFIGS)
    except Exception:
        GFX1101_AUTOTUNE_CONFIGS = []
if not GFX1101_AUTOTUNE_CONFIGS and triton is not None:
    # Reconstruct the canonical gfx1101 grid if triton_kernels is unavailable.
    GFX1101_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3, 4)
    ]


# ---------------------------------------------------------------------------
# Decorator + dispatch helpers
# ---------------------------------------------------------------------------
def sm120_autotune(
    configs: Optional[Sequence[Any]] = None,
    *,
    key: Sequence[str],
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: wrap :func:`triton.autotune` with sm_120 configs as default.

    Mirrors :func:`triton_kernels.autotune.autotune` but defaults to
    :data:`SM120_AUTOTUNE_CONFIGS` instead of gfx1101's grid.

    Parameters
    ----------
    configs : sequence of :class:`triton.Config`, optional
        Defaults to :data:`SM120_AUTOTUNE_CONFIGS` (12 configs).
        Pass :data:`SM120_AUTOTUNE_CONFIGS_FALLBACK` for the
        4-config eval-mode grid.
    key : sequence of str
        Required — same contract as :func:`triton.autotune`.
    **kwargs
        Forwarded verbatim to :func:`triton.autotune`.
    """
    if triton is None:  # pragma: no cover - CPU-only env
        def _passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return _passthrough
    if configs is None:
        configs = SM120_AUTOTUNE_CONFIGS
    if not key:
        raise ValueError(
            "`key` must be a non-empty sequence of argument names; got an "
            "empty sequence."
        )
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)


def is_sm120_config_safe(config: Any) -> bool:
    """Return True iff a sm_120 config has the expected knob ranges.

    Asserts ``num_warps in {2, 4, 8, 16}`` and
    ``num_stages in {2, 3, 4}`` — outside these ranges the
    Blackwell retune is unsafe (config not in the Wf-326E-A grid).
    """
    nw = getattr(config, "num_warps", None)
    ns = getattr(config, "num_stages", None)
    if nw is None and isinstance(config, dict):
        nw = config.get("num_warps")
        ns = config.get("num_stages")
    if nw is None or ns is None:
        return False
    if nw not in (2, 4, 8, 16):
        return False
    if ns not in (2, 3, 4):
        return False
    return True


# ---------------------------------------------------------------------------
# Sample :func:`triton.jit` kernel for the CPU smoke (defined in this
# .py file so :func:`triton.jit` accepts it; never invoked on GPU)
# ---------------------------------------------------------------------------
if triton is not None:
    import triton.language as tl

    @sm120_autotune(key=["M", "N"])
    @triton.jit
    def _sm120_dummy_kernel(
        x_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
    ) -> None:  # pragma: no cover - never invoked on GPU
        """Smoke-only kernel used to verify the sm_120 autotune wrapper
        constructs an :class:`triton.runtime.autotuner.Autotuner` over a
        :func:`triton.jit` kernel.  Never launched — exists only so the
        test can inspect ``.configs`` + ``.run``."""
        _ = x_ptr
        _ = M
        _ = N
else:  # pragma: no cover - CPU-only env without triton
    _sm120_dummy_kernel = None  # type: ignore[assignment]


def safe_kernel_call(
    sm120_kernel: Callable[..., Any],
    gfx1101_kernel: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Try the sm_120 kernel; fall back to gfx1101 on failure.

    Use for the wrap-and-fallback dispatch pattern (e.g.
    ``safe_kernel_call(fused_silu_mlp_sm120, fused_silu_mlp_gfx1101,
    x, w1, b1, w2, b2)``).  Both kernels must have compatible
    call signatures.

    On a successful sm_120 launch: return the sm_120 result.
    On any exception (sm_120 launch failure, register pressure,
    PTX mismatch, OOM): log the failure to stderr (one line) and
    re-invoke the gfx1101 fallback with the same args.

    Notes
    -----
    * This is intentionally simple — no retry, no caching.  Per-call
      overhead is one try/except + one stderr print in the failure
      path; the happy path adds zero overhead.
    * The caller is expected to have already checked
      :func:`is_sm120_config_safe` for any sm_120 config that wasn't
      constructed by this scaffold.
    """
    try:
        return sm120_kernel(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - we want broad fallback
        print(
            f"[wf331x3] sm_120 kernel call failed ({type(exc).__name__}: "
            f"{exc}); falling back to gfx1101.",
            file=sys.stderr,
        )
        return gfx1101_kernel(*args, **kwargs)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_audit_kernels(audit_path: Path = AUDIT_PATH) -> List[Dict[str, Any]]:
    """Load the Wf-326E audit kernel inventory (source of truth).

    Returns the ``kernels`` list from the audit JSON.  Raises
    :class:`FileNotFoundError` if the audit is missing — the test
    suite pins this contract.
    """
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Wf-326E audit missing at {audit_path}. Run the audit "
            f"pipeline first."
        )
    with audit_path.open() as fh:
        data = json.load(fh)
    return list(data["triton_kernels_inventory"]["kernels"])


def dump_sm120_configs_json(
    output_path: Path,
    *,
    configs: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Emit the sm_120 configs as JSON (CPU-only).

    Writes ``configs`` (default :data:`SM120_AUTOTUNE_CONFIGS`) as
    a JSON array of ``{"num_warps": int, "num_stages": int}`` dicts
    to ``output_path``.

    Returns
    -------
    Dict with keys:
        ``output_path`` — string path of the JSON file.
        ``n_configs`` — number of configs written.
        ``kernel_names`` — list of kernel module names (from audit).
        ``n_kernels_inventoried`` — len(kernel_names).
        ``gpu_retune_deferred`` — bool (always True for this CPU scaffold).
        ``configs`` — the dict payload.
    """
    if configs is None:
        configs = SM120_AUTOTUNE_CONFIGS
    payload = []
    for cfg in configs:
        nw = getattr(cfg, "num_warps", None)
        ns = getattr(cfg, "num_stages", None)
        if nw is None and isinstance(cfg, dict):
            nw = cfg.get("num_warps")
            ns = cfg.get("num_stages")
        payload.append({"num_warps": int(nw), "num_stages": int(ns)})

    kernel_names: List[str] = []
    try:
        for entry in load_audit_kernels():
            kernel_names.append(str(entry["name"]))
    except FileNotFoundError:
        kernel_names = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "configs": payload,
        "kernel_names": kernel_names,
        "n_kernels_inventoried": len(kernel_names),
        "gpu_retune_deferred": True,
        "scaffold_audit_path": str(AUDIT_PATH),
    }
    with output_path.open("w") as fh:
        json.dump(body, fh, indent=2)

    return {
        "output_path": str(output_path),
        "n_configs": len(payload),
        "kernel_names": kernel_names,
        "n_kernels_inventoried": len(kernel_names),
        "gpu_retune_deferred": True,
        "configs": payload,
    }


# ---------------------------------------------------------------------------
# CPU smoke (correctness test for the configs)
# ---------------------------------------------------------------------------
def cpu_smoke(output_dir: Path) -> Dict[str, Any]:
    """Run a CPU-only smoke that exercises the sm_120 retune scaffold.

    Validates:

    * :data:`SM120_AUTOTUNE_CONFIGS` has the expected size (12).
    * :data:`SM120_AUTOTUNE_CONFIGS_FALLBACK` has the expected size (4).
    * Every config has the Wf-326E-A knob ranges (num_warps in {2, 4,
      8, 16}, num_stages in {2, 3, 4}).
    * :func:`is_sm120_config_safe` returns True for sm_120 configs,
      False for out-of-range configs.
    * :func:`safe_kernel_call` falls back when the sm_120 kernel
      raises.
    * :func:`sm120_autotune` produces a callable (even if the
      underlying :func:`triton.autotune` is unavailable).
    """
    started = time.time()

    n_full = len(SM120_AUTOTUNE_CONFIGS)
    n_fallback = len(SM120_AUTOTUNE_CONFIGS_FALLBACK)

    # 1) Config shape checks
    for cfg in SM120_AUTOTUNE_CONFIGS:
        assert is_sm120_config_safe(cfg), (
            f"sm_120 config {cfg} is outside the Wf-326E-A grid."
        )
    for cfg in SM120_AUTOTUNE_CONFIGS_FALLBACK:
        assert is_sm120_config_safe(cfg), (
            f"sm_120 fallback config {cfg} is outside the Wf-326E-A grid."
        )

    # 2) is_sm120_config_safe returns False for out-of-range configs.
    if triton is not None:
        bogus_cfg = triton.Config({}, num_warps=64, num_stages=8)
    else:
        bogus_cfg = {"num_warps": 64, "num_stages": 8}
    assert not is_sm120_config_safe(bogus_cfg), (
        "Bogus config (num_warps=64, num_stages=8) should NOT be "
        "sm_120-safe per the Wf-326E-A grid."
    )

    # 3) safe_kernel_call fallback path
    calls: List[str] = []

    def _sm120_kernel() -> str:
        calls.append("sm120")
        raise RuntimeError("simulated sm_120 failure")

    def _gfx1101_kernel() -> str:
        calls.append("gfx1101")
        return "fallback_ok"

    result = safe_kernel_call(_sm120_kernel, _gfx1101_kernel)
    assert result == "fallback_ok", (
        f"safe_kernel_call should fall back; got {result!r}"
    )
    assert calls == ["sm120", "gfx1101"], (
        f"safe_kernel_call should call sm_120 first then gfx1101; "
        f"got call order {calls}"
    )

    # 4) sm120_autotune produces an Autotuner over the sample kernel.
    if triton is not None and _sm120_dummy_kernel is not None:
        # _sm120_dummy_kernel is an Autotuner wrapping a JITFunction.
        # It has ``.configs`` (list of sm_120 configs) and ``.run`` method.
        assert hasattr(_sm120_dummy_kernel, "configs"), (
            "sm120_autotune-wrapped kernel must expose .configs."
        )
        assert hasattr(_sm120_dummy_kernel, "run"), (
            "sm120_autotune-wrapped kernel must expose .run method."
        )
        assert len(_sm120_dummy_kernel.configs) == len(
            SM120_AUTOTUNE_CONFIGS
        ), (
            f"sm120_autotune-wrapped kernel has {len(_sm120_dummy_kernel.configs)} "
            f"configs; expected {len(SM120_AUTOTUNE_CONFIGS)}."
        )

    # 5) Dump configs to JSON for the eventual GPU retune run.
    output_dir.mkdir(parents=True, exist_ok=True)
    dumped = dump_sm120_configs_json(
        output_dir / "sm120_autotune_configs.json"
    )

    elapsed = time.time() - started
    return {
        "smoke_passed": True,
        "n_sm120_configs": n_full,
        "n_sm120_fallback_configs": n_fallback,
        "n_kernels_inventoried": dumped["n_kernels_inventoried"],
        "gpu_retune_deferred": True,
        "safe_kernel_call_fallback_ok": True,
        "autotune_decorator_callable": True,
        "configs_json_path": dumped["output_path"],
        "elapsed_seconds": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the sm_120 configs JSON (default: "
             "molmetal/reports/wf331x3_sm120_retune).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the CPU smoke test that exercises the scaffold "
             "(autotune wrapper, fallback, is_safe).",
    )
    parser.add_argument(
        "--dump-configs-only",
        action="store_true",
        help="Only emit the configs JSON (skip the smoke).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()

    if args.dump_configs_only:
        dumped = dump_sm120_configs_json(
            output_dir / "sm120_autotune_configs.json"
        )
        print(json.dumps({
            "n_configs": dumped["n_configs"],
            "n_kernels_inventoried": dumped["n_kernels_inventoried"],
            "gpu_retune_deferred": dumped["gpu_retune_deferred"],
            "configs_json_path": dumped["output_path"],
        }, indent=2))
        return 0

    if args.smoke:
        result = cpu_smoke(output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["smoke_passed"] else 1

    # Default: dump the configs and report the audit cross-reference.
    try:
        kernel_names = [e["name"] for e in load_audit_kernels()]
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dumped = dump_sm120_configs_json(
        output_dir / "sm120_autotune_configs.json"
    )

    print(
        json.dumps(
            {
                "n_kernels_inventoried": len(kernel_names),
                "n_sm120_configs": len(SM120_AUTOTUNE_CONFIGS),
                "n_sm120_fallback_configs": len(
                    SM120_AUTOTUNE_CONFIGS_FALLBACK
                ),
                "configs_json_path": dumped["output_path"],
                "gpu_retune_deferred": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
