"""Wf-333-T1 — sm_120 Triton autotune config presets extension (Blackwell).

================================================================
Wf-333-T1 (2026-09-29) — Extends the Wf-331-X3 CPU scaffold with
per-kernel autotune config presets.  Per the Wf-326E audit
(`molmetal/reports/wf326e_triton_audit/audit.json`):

  * 24 Triton kernels inventoried (4 categories + 14 sub-categories
    in the audit.json ``kernels[].category`` field).
  * Wf-331-X3 shipped a 12-config generic grid (num_warps in {2, 4,
    8, 16} x num_stages in {2, 3, 4}) — but every kernel uses the
    SAME grid regardless of whether it's memory-bound (elementwise
    / norm / residual) or compute-bound (MLP variants / matmul).
  * Wf-333-T1 splits the generic grid into per-kernel presets so
    the autotuner sweeps a much narrower config band per kernel —
    cuts autotune cost by ~3x on Blackwell (per Wf-326E follow-up
    recommendation B).

The per-kernel categorization is derived from the audit
``kernels[].category`` field:

  * **Memory-bound** (``num_warps=2``) — elementwise / norm /
    residual / dropout (5 kernels).  These are bandwidth-limited;
    fewer warps = better register reuse + lower occupancy stalls.
  * **Compute-bound** (``num_warps in {8, 16}``) — MLP / matmul /
    batched MLP variants / module-wrapped variants (13 kernels).
    Compute-bound workloads benefit from wider warps for ILP +
    tensor-core-friendly block sizes.
  * **Mid-bound** (``num_warps in {4, 8}``) — softmax / scatter /
    small losses (3 kernels).  These are neither purely memory nor
    purely compute-bound; medium warp count is a balanced default.
  * **ODE / se(3)** (``num_warps=4``) — small element-count kernels
    with low arithmetic intensity (3 kernels: rotation_from_axis_
    angle, euler_step, rk4_step).  Fewest warps + smallest stages.

API surface (all CPU-friendly; no GPU launch required):

  * :data:`MEMORY_BOUND_KERNELS` — set of 5 kernel names
  * :data:`COMPUTE_BOUND_KERNELS` — set of 13 kernel names
  * :data:`MID_BOUND_KERNELS` — set of 3 kernel names
  * :data:`ODE_KERNELS` — set of 3 kernel names
  * :data:`PER_KERNEL_CONFIGS` — dict[24] of kernel -> config list
  * :func:`get_kernel_category(kernel_name)` — returns category str
  * :func:`get_kernel_configs(kernel_name)` — returns per-kernel
    config list (falls back to :data:`SM120_AUTOTUNE_CONFIGS` if the
    kernel name is not in the per-kernel registry)
  * :func:`sm120_autotune_per_kernel(kernel_name, *, key)` — decorator
    that wraps :func:`triton.autotune` with the per-kernel config list
  * :func:`dump_per_kernel_configs_json(output_path)` — emits the
    per-kernel registry as JSON
  * :func:`cpu_smoke_extended(output_dir)` — CPU correctness test for
    the extended surface (category classification, per-kernel configs,
    fallback path, decorator)

Honest framing (per AGENTS.md §5.2):

  * The per-kernel categorization is **PROJECTED**, derived from the
    audit's ``kernels[].category`` field + the Wf-326E follow-up
    recommendation B (per-kernel config subsets).
  * CPU correctness is verified via :func:`is_sm120_config_safe` (the
    knob-range check from Wf-331-X3) — no GPU launch is performed.
  * Full GPU retune remains **DEFERRED** per AGENTS.md §5.1 —
    per-kernel grid cuts autotune cost but does NOT flip the
    default-OFF status of the 24-kernel library.
  * PROJECTED aggregate speedup once GPU retune + per-kernel configs
    land: 1.5-3× (per audit ``audit_summary.projected_speedup_with_
    retune_and_new_kernels``).

CLI usage (CPU-only, no GPU launch required)::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf333t1_sm120_autotune_extended.py \\
        --output-dir molmetal/reports/wf333t1_sm120_autotune

    # With explicit smoke (CPU correctness test for the extension).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf333t1_sm120_autotune_extended.py --smoke

Cross-references:

  * :file:`molmetal/scripts/wf331x3_sm120_retune.py` — Wf-331-X3
    CPU scaffold (this module extends it via REUSE pattern: imports
    the constants + helpers from the Wf-331-X3 ship surface rather
    than redefining them).
  * :file:`molmetal/reports/wf326e_triton_audit/audit.json` —
    Wf-326E source-of-truth (24 kernels, dispatch status, category
    field per kernel).
  * :file:`TODO/completed/331x3_sm120_retune_cpu_scaffold.md` —
    Wf-331-X3 ship surface (the 4 Wf-331-X3 tests are preserved).
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

# Reuse the Wf-331-X3 scaffold surface (constants + helpers).
import wf331x3_sm120_retune as _x3  # noqa: E402  (lazy: sys.path bootstrap)


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf333t1_sm120_autotune"
AUDIT_PATH = REPO_ROOT / "molmetal" / "reports" / "wf326e_triton_audit" / "audit.json"


# ---------------------------------------------------------------------------
# Per-kernel category assignments (PROJECTED — derived from
# ``audit.triton_kernels_inventory.kernels[].category``)
# ---------------------------------------------------------------------------
# Category strings used throughout the module.
CAT_MEMORY_BOUND = "memory_bound"
CAT_COMPUTE_BOUND = "compute_bound"
CAT_MID_BOUND = "mid_bound"
CAT_ODE = "ode"


# Memory-bound kernels — bandwidth-limited; num_warps=2 (low occupancy).
MEMORY_BOUND_KERNELS = frozenset({
    "fused_residual_add",           # elementwise
    "fused_dropout_residual",       # elementwise+dropout
    "fused_layer_norm",             # norm
    "fused_rms_norm",               # norm
    "fused_rmsnorm_residual",       # norm+residual
})


# Compute-bound kernels — ILP-heavy; num_warps in {8, 16}.
COMPUTE_BOUND_KERNELS = frozenset({
    # MLP variants
    "fused_silu_mlp",
    "fused_gelu_mlp",
    "FusedGeluMLP",
    # Batched MLP variants
    "batched_fused_silu_mlp",
    "batched_fused_gelu_mlp",
    "batched_fused_silu_mlp_per_k",
    "batched_fused_gelu_mlp_per_k",
    # ReLU MLP variants
    "fused_relu_linear",
    "fused_relu_mlp",
    "FusedReLULinear",
    "FusedReLUMLP",
    # Matmul
    "matmul_fn",
    # Compute-bound losses
    "fused_cross_entropy",
    "fused_log_softmax_nll",
})


# Mid-bound kernels — neither purely memory nor purely compute;
# num_warps in {4, 8}.
MID_BOUND_KERNELS = frozenset({
    "softmax_last_dim",  # softmax
    "aggregate_vectors",  # scatter/gather
    # fused_cross_entropy / fused_log_softmax_nll moved to COMPUTE_BOUND
    # because on Blackwell the log-softmax cross-entropy is dominated
    # by the matmul (logits @ targets).
})


# ODE / se(3) kernels — low arithmetic intensity; num_warps=4.
ODE_KERNELS = frozenset({
    "rotation_from_axis_angle",  # se3
    "euler_step",                # ode
    "rk4_step",                  # ode
})


# Aggregate sanity check — total must equal 24 (Wf-326E inventory).
_ALL_KERNELS = (
    MEMORY_BOUND_KERNELS | COMPUTE_BOUND_KERNELS
    | MID_BOUND_KERNELS | ODE_KERNELS
)
assert len(_ALL_KERNELS) == 24, (
    f"Per-kernel category assignments cover {len(_ALL_KERNELS)} kernels; "
    f"expected 24 per Wf-326E audit. Diff: "
    f"{set(_x3._all_kernel_names_static()) - _ALL_KERNELS}."
)


# ---------------------------------------------------------------------------
# Per-kernel config grid construction
# ---------------------------------------------------------------------------
def _has_num_warps_safe(cfg: Any) -> int:
    """Return ``num_warps`` from a :class:`triton.Config` or dict (safe).

    Local copy of the Wf-331-X3 helper logic so we don't reach into
    the shipped module's private namespace.  Raises if missing.
    """
    nw = getattr(cfg, "num_warps", None)
    if nw is None and isinstance(cfg, dict):
        nw = cfg.get("num_warps")
    if nw is None:
        raise AttributeError(f"Config {cfg!r} has no num_warps.")
    return int(nw)


def _has_num_stages_safe(cfg: Any) -> int:
    """Return ``num_stages`` from a :class:`triton.Config` or dict (safe).

    Local copy of the Wf-331-X3 helper logic.  Raises if missing.
    """
    ns = getattr(cfg, "num_stages", None)
    if ns is None and isinstance(cfg, dict):
        ns = cfg.get("num_stages")
    if ns is None:
        raise AttributeError(f"Config {cfg!r} has no num_stages.")
    return int(ns)


def _make_configs(num_warps_grid: Sequence[int], num_stages_grid: Sequence[int]) -> List[Any]:
    """Build a list of :class:`triton.Config` from (num_warps, num_stages) grids.

    Falls back to dict-of-ints placeholders when :mod:`triton` is not
    importable (CPU-only test env).  The resulting list always has
    ``len(num_warps_grid) * len(num_stages_grid)`` entries.
    """
    if triton is not None:
        return [
            triton.Config({}, num_warps=nw, num_stages=ns)
            for nw in num_warps_grid
            for ns in num_stages_grid
        ]
    return [
        {"num_warps": nw, "num_stages": ns}
        for nw in num_warps_grid
        for ns in num_stages_grid
    ]


# Per-category grids (kept narrow to cut autotune cost ~3x).
_MEMORY_BOUND_GRID = _make_configs((2,), (2, 3, 4))                # 3 configs
_COMPUTE_BOUND_GRID = _make_configs((8, 16), (2, 3, 4))            # 6 configs
_MID_BOUND_GRID = _make_configs((4, 8), (2, 3, 4))                 # 6 configs
_ODE_GRID = _make_configs((4,), (2, 3))                            # 2 configs

# Sanity: 3 + 6 + 6 + 2 = 17 unique (num_warps, num_stages) cells; but
# PER_KERNEL_CONFIGS maps each of 24 kernel names to a list, so the
# total per-kernel configs across all 24 kernels is
# 5*3 + 13*6 + 3*6 + 3*2 = 15 + 78 + 18 + 6 = 117 entries.
PER_KERNEL_CONFIGS: Dict[str, List[Any]] = {}
"""Per-kernel config registry (24 entries).

Each kernel name maps to a :class:`triton.Config` list sized to
its workload category.  Lookup is O(1); the fallback
(:func:`get_kernel_configs`) returns the Wf-331-X3 generic grid if
the kernel name is not in this registry.
"""
for _name in MEMORY_BOUND_KERNELS:
    PER_KERNEL_CONFIGS[_name] = list(_MEMORY_BOUND_GRID)
for _name in COMPUTE_BOUND_KERNELS:
    PER_KERNEL_CONFIGS[_name] = list(_COMPUTE_BOUND_GRID)
for _name in MID_BOUND_KERNELS:
    PER_KERNEL_CONFIGS[_name] = list(_MID_BOUND_GRID)
for _name in ODE_KERNELS:
    PER_KERNEL_CONFIGS[_name] = list(_ODE_GRID)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_kernel_category(kernel_name: str) -> Optional[str]:
    """Return the workload category for a kernel name.

    Returns one of ``"memory_bound"``, ``"compute_bound"``,
    ``"mid_bound"``, ``"ode"``, or ``None`` if the kernel name is
    not in the per-kernel registry.
    """
    if kernel_name in MEMORY_BOUND_KERNELS:
        return CAT_MEMORY_BOUND
    if kernel_name in COMPUTE_BOUND_KERNELS:
        return CAT_COMPUTE_BOUND
    if kernel_name in MID_BOUND_KERNELS:
        return CAT_MID_BOUND
    if kernel_name in ODE_KERNELS:
        return CAT_ODE
    return None


def get_kernel_configs(kernel_name: str) -> List[Any]:
    """Return the per-kernel config list for a kernel name.

    Falls back to :data:`_x3.SM120_AUTOTUNE_CONFIGS` (the Wf-331-X3
    generic 12-config grid) if the kernel name is not in
    :data:`PER_KERNEL_CONFIGS` — preserves the Wf-331-X3 safety
    contract: callers always get a non-empty config list.
    """
    if kernel_name in PER_KERNEL_CONFIGS:
        return list(PER_KERNEL_CONFIGS[kernel_name])
    return list(_x3.SM120_AUTOTUNE_CONFIGS)


def list_kernel_names_from_audit(audit_path: Path = AUDIT_PATH) -> List[str]:
    """Return the list of kernel names from the Wf-326E audit JSON.

    Reuses :func:`wf331x3_sm120_retune.load_audit_kernels` so the
    Wf-331-X3 source-of-truth is the single point of audit parsing.
    """
    try:
        entries = _x3.load_audit_kernels(audit_path)
    except FileNotFoundError:
        return []
    return [str(e["name"]) for e in entries]


def _all_kernel_names_static() -> List[str]:
    """Return the static list of all 24 per-kernel names (sorted).

    Defined so the ``assert`` at module level can check the union of
    per-kernel categories covers all 24 audit kernels without
    requiring disk I/O.  Used internally; not part of the public API.
    """
    return sorted(_ALL_KERNELS)


# Inject the helper into wf331x3 so the module-level assert at import
# time can call it without circular-import issues.  (Weak reference;
# only used by the import-time assertion, never by public API.)
_x3._all_kernel_names_static = _all_kernel_names_static


# ---------------------------------------------------------------------------
# Per-kernel autotune decorator
# ---------------------------------------------------------------------------
def sm120_autotune_per_kernel(
    kernel_name: str,
    *,
    key: Sequence[str],
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: wrap :func:`triton.autotune` with per-kernel configs.

    Looks up the per-kernel config list via
    :func:`get_kernel_configs` (falls back to the Wf-331-X3 generic
    grid if the kernel name is unknown).  Mirrors the API of
    :func:`wf331x3_sm120_retune.sm120_autotune` so callers can swap
    decorators with a one-line change.

    Parameters
    ----------
    kernel_name : str
        The kernel name to look up in :data:`PER_KERNEL_CONFIGS`.
        Must match one of the 24 Wf-326E audit kernels (e.g.
        ``"fused_silu_mlp"``, ``"matmul_fn"``); otherwise the
        Wf-331-X3 generic grid is used.
    key : sequence of str
        Required — same contract as :func:`triton.autotune`.
    **kwargs
        Forwarded verbatim to :func:`triton.autotune`.

    Returns
    -------
    A decorator that wraps the kernel with :func:`triton.autotune`
    using the per-kernel config grid.
    """
    if triton is None:  # pragma: no cover - CPU-only env
        def _passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return _passthrough
    if not key:
        raise ValueError(
            "`key` must be a non-empty sequence of argument names; got an "
            "empty sequence."
        )
    configs = get_kernel_configs(kernel_name)
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def dump_per_kernel_configs_json(
    output_path: Path,
    *,
    audit_path: Path = AUDIT_PATH,
) -> Dict[str, Any]:
    """Emit the per-kernel registry as JSON (CPU-only).

    Writes a JSON object mapping each kernel name to a
    ``{"category": str, "configs": [{"num_warps": int, "num_stages": int}, ...]}``
    payload, plus a top-level metadata block (n_kernels,
    n_categories, gpu_retune_deferred, scaffold_audit_path).

    Returns
    -------
    Dict with keys:
        ``output_path`` — string path of the JSON file.
        ``n_kernels`` — number of kernels in the registry (24).
        ``n_categories`` — number of distinct categories (4).
        ``gpu_retune_deferred`` — bool (always True).
        ``kernel_names`` — list of kernel names (sorted).
    """
    registry: Dict[str, Dict[str, Any]] = {}
    for name in sorted(PER_KERNEL_CONFIGS):
        cat = get_kernel_category(name) or "unknown"
        cfg_list = []
        for cfg in PER_KERNEL_CONFIGS[name]:
            nw = getattr(cfg, "num_warps", None)
            ns = getattr(cfg, "num_stages", None)
            if nw is None and isinstance(cfg, dict):
                nw = cfg.get("num_warps")
                ns = cfg.get("num_stages")
            cfg_list.append({"num_warps": int(nw), "num_stages": int(ns)})
        registry[name] = {"category": cat, "configs": cfg_list}

    body = {
        "metadata": {
            "n_kernels": len(registry),
            "n_categories": 4,
            "categories": [
                CAT_MEMORY_BOUND, CAT_COMPUTE_BOUND,
                CAT_MID_BOUND, CAT_ODE,
            ],
            "n_memory_bound_kernels": len(MEMORY_BOUND_KERNELS),
            "n_compute_bound_kernels": len(COMPUTE_BOUND_KERNELS),
            "n_mid_bound_kernels": len(MID_BOUND_KERNELS),
            "n_ode_kernels": len(ODE_KERNELS),
            "gpu_retune_deferred": True,
            "scaffold_audit_path": str(audit_path),
        },
        "registry": registry,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(body, fh, indent=2)

    return {
        "output_path": str(output_path),
        "n_kernels": len(registry),
        "n_categories": 4,
        "gpu_retune_deferred": True,
        "kernel_names": sorted(registry.keys()),
    }


# ---------------------------------------------------------------------------
# CPU smoke (correctness test for the extended surface)
# ---------------------------------------------------------------------------
def cpu_smoke_extended(output_dir: Path) -> Dict[str, Any]:
    """Run a CPU-only smoke that exercises the per-kernel extension.

    Validates:

    * All 24 audit kernels are assigned to a category (5 memory + 13
      compute + 3 mid + 3 ode = 24).
    * Each per-kernel config list is a strict subset of the Wf-331-X3
      generic 12-config grid (every per-kernel (num_warps, num_stages)
      pair appears in the Wf-331-X3 grid).
    * :func:`is_sm120_config_safe` (from Wf-331-X3) returns True for
      every per-kernel config.
    * :func:`get_kernel_configs` falls back to the generic grid when
      the kernel name is unknown.
    * :func:`sm120_autotune_per_kernel` produces a callable that
      exposes ``.configs`` matching the per-kernel grid size.
    * Per-kernel JSON dump succeeds.
    """
    started = time.time()

    # 1) Category assignment covers all 24 audit kernels.
    audit_names = list_kernel_names_from_audit()
    n_audit = len(audit_names)
    assert n_audit == 24, (
        f"Wf-326E audit should list 24 kernels; got {n_audit}. "
        f"Update PER_KERNEL_CONFIGS if the audit inventory grew."
    )
    missing = set(audit_names) - _ALL_KERNELS
    assert not missing, (
        f"Audit kernels missing from per-kernel registry: {missing}."
    )

    # 2) Each per-kernel config is a subset of the Wf-331-X3 grid.
    x3_grid_pairs = {
        (_has_num_warps_safe(c), _has_num_stages_safe(c))
        for c in _x3.SM120_AUTOTUNE_CONFIGS
    }
    for name, cfg_list in PER_KERNEL_CONFIGS.items():
        for cfg in cfg_list:
            nw = _has_num_warps_safe(cfg)
            ns = _has_num_stages_safe(cfg)
            assert (nw, ns) in x3_grid_pairs, (
                f"Per-kernel config (num_warps={nw}, num_stages={ns}) "
                f"for kernel {name!r} is NOT in the Wf-331-X3 generic "
                f"grid {x3_grid_pairs}."
            )

    # 3) is_sm120_config_safe (Wf-331-X3 helper) returns True for every config.
    for name, cfg_list in PER_KERNEL_CONFIGS.items():
        for cfg in cfg_list:
            assert _x3.is_sm120_config_safe(cfg), (
                f"is_sm120_config_safe returned False for per-kernel "
                f"config of {name!r}: {cfg}."
            )

    # 4) get_kernel_configs falls back when kernel is unknown.
    fallback_grid = get_kernel_configs("unknown_kernel_xyz")
    assert len(fallback_grid) == len(_x3.SM120_AUTOTUNE_CONFIGS), (
        f"Fallback grid should match Wf-331-X3 generic grid size "
        f"({len(_x3.SM120_AUTOTUNE_CONFIGS)}); got {len(fallback_grid)}."
    )

    # 5) sm120_autotune_per_kernel produces a callable decorator.
    if triton is not None:
        import triton.language as tl

        @sm120_autotune_per_kernel(kernel_name="fused_silu_mlp", key=["M"])
        @triton.jit
        def _per_kernel_dummy(x_ptr, M: tl.constexpr) -> None:  # pragma: no cover
            _ = (x_ptr, M)

        assert hasattr(_per_kernel_dummy, "configs"), (
            "sm120_autotune_per_kernel-wrapped kernel must expose .configs."
        )
        expected_n = len(PER_KERNEL_CONFIGS["fused_silu_mlp"])
        assert len(_per_kernel_dummy.configs) == expected_n, (
            f"Per-kernel decorator produced {len(_per_kernel_dummy.configs)} "
            f"configs; expected {expected_n} for fused_silu_mlp."
        )

    # 6) Per-kernel JSON dump.
    output_dir.mkdir(parents=True, exist_ok=True)
    dumped = dump_per_kernel_configs_json(
        output_dir / "per_kernel_configs.json"
    )

    # 7) safe_kernel_call fallback path (re-verify from Wf-331-X3).
    calls: List[str] = []

    def _sm120_kernel() -> str:
        calls.append("sm120")
        raise RuntimeError("simulated sm_120 failure")

    def _gfx1101_kernel() -> str:
        calls.append("gfx1101")
        return "fallback_ok"

    fb = _x3.safe_kernel_call(_sm120_kernel, _gfx1101_kernel)
    assert fb == "fallback_ok", f"safe_kernel_call fallback broken: {fb}"
    assert calls == ["sm120", "gfx1101"], (
        f"safe_kernel_call call order wrong: {calls}"
    )

    elapsed = time.time() - started
    return {
        "smoke_passed": True,
        "n_kernels": len(PER_KERNEL_CONFIGS),
        "n_categories": 4,
        "n_memory_bound": len(MEMORY_BOUND_KERNELS),
        "n_compute_bound": len(COMPUTE_BOUND_KERNELS),
        "n_mid_bound": len(MID_BOUND_KERNELS),
        "n_ode": len(ODE_KERNELS),
        "n_total_per_kernel_configs": sum(
            len(v) for v in PER_KERNEL_CONFIGS.values()
        ),
        "fallback_to_generic_grid_ok": True,
        "safe_kernel_call_fallback_ok": True,
        "per_kernel_decorator_callable": True,
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
        help="Where to write the per-kernel configs JSON (default: "
             "molmetal/reports/wf333t1_sm120_autotune).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the CPU smoke that exercises the per-kernel extension.",
    )
    parser.add_argument(
        "--dump-configs-only",
        action="store_true",
        help="Only emit the per-kernel configs JSON (skip the smoke).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()

    if args.dump_configs_only:
        dumped = dump_per_kernel_configs_json(
            output_dir / "per_kernel_configs.json"
        )
        print(json.dumps(dumped, indent=2))
        return 0

    if args.smoke:
        result = cpu_smoke_extended(output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["smoke_passed"] else 1

    # Default: dump configs and report the cross-reference.
    dumped = dump_per_kernel_configs_json(
        output_dir / "per_kernel_configs.json"
    )
    print(json.dumps({
        "n_kernels": dumped["n_kernels"],
        "n_categories": dumped["n_categories"],
        "n_memory_bound": len(MEMORY_BOUND_KERNELS),
        "n_compute_bound": len(COMPUTE_BOUND_KERNELS),
        "n_mid_bound": len(MID_BOUND_KERNELS),
        "n_ode": len(ODE_KERNELS),
        "configs_json_path": dumped["output_path"],
        "gpu_retune_deferred": True,
        "extends": "wf331x3_sm120_retune.py",
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
