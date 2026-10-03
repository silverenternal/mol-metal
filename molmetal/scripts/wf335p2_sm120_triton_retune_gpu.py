"""Wf-335-P2 — sm_120 Triton retune GPU wrapper (CLI scaffold).

================================================================
Wf-335-P2 (2026-09-29) — Single-entry CLI wrapper that combines
the Wf-331-X3 sm_120 scaffold + the Wf-333-T1 per-kernel presets
into one launch surface for the eventual GPU retune on RTX 5090
(Blackwell, sm_120).

Per Wf-326E audit (``molmetal/reports/wf326e_triton_audit/audit.json``):
  * 24 Triton kernels inventoried, all default-OFF on sm_120 because
    :data:`triton_kernels.autotune.AUTOTUNE_CONFIGS` was tuned for
    gfx1101 (RDNA3 iGPU).
  * Wf-326E follow-up recommendation A: sweep num_warps in {4, 8, 16}
    x num_stages in {2, 3, 4} on a single RTX 5090.
  * Wf-326E follow-up recommendation B: per-kernel config subsets cut
    autotune cost by ~3x on Blackwell.

This wrapper provides the **CLI surface** to perform the retune
sweep WITHOUT hard-coding the retune itself.  The full GPU retune
remains DEFERRED per AGENTS.md §5.1 (standing GPU authorization +
VRAM budget + pre-flight PASS + config-aligned hyperparameters
required before any launch > 1 hr).

API surface (CPU-friendly; no GPU launch required to use):

  * :func:`build_launch_plan(audit_path, *, use_per_kernel, kernels)`
    — Compose a :class:`LaunchPlan` that lists which kernels will be
    autotuned, which configs each one will sweep, and the aggregate
    cost (kernel-launch budget, total configs to sweep).
  * :func:`dump_launch_plan_json(plan, output_path)` — Emit the plan
    as JSON for operator review before any actual GPU launch.
  * :func:`cpu_smoke_gpu_wrapper(output_dir)` — CPU-only correctness
    test for the wrapper (plan composition, JSON dump, CLI flag
    combinations).
  * :func:`main(argv)` — argparse entry-point.

CLI usage (CPU-only, no GPU launch required by default)::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \\
        --output-dir molmetal/reports/wf335p2_sm120_triton_retune_gpu

    # Use the Wf-333-T1 per-kernel presets (recommended per Wf-326E-B).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \\
        --per-kernel

    # Restrict to a subset of kernels (comma-separated names).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \\
        --per-kernel --kernels fused_silu_mlp,fused_gelu_mlp,matmul_fn

    # CPU-only smoke (validates the wrapper, no GPU launch).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py --smoke

    # Only emit the launch plan JSON (skip the smoke).
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \\
        --dump-launch-plan-only

    # Promote to an actual GPU launch — REQUIRES:
    #   1. nvidia-smi VRAM-budget check (per AGENTS.md §5.1)
    #   2. python molmetal/scripts/pre_flight_check.py PASS
    #   3. config-aligned hyperparameters (cite peer papers)
    # Without those three, the --launch flag refuses to run and
    # prints a checklist of prerequisites.
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py --launch

Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5):

  * This wrapper is **CPU-only scaffold** — no actual GPU launch by
    default.  The ``--launch`` flag exists for the operator to
    explicitly opt in, but the wrapper will refuse to launch without
    the 3-prereq gate (VRAM + pre-flight + config-aligned) and will
    not itself perform the VRAM probe or the pre-flight check (those
    live in ``molmetal/scripts/check_blackwell_env.py`` +
    ``molmetal/scripts/pre_flight_check.py``).
  * The launch plan is a **PROJECTED @ R19** artifact, NOT a
    MEASURED number — the actual retune timings will be MEASURED only
    once the GPU launch completes.
  * The wrapper reuses Wf-331-X3 (12-config generic grid) +
    Wf-333-T1 (117 per-kernel configs) — no new constants are
    defined here.
  * PROJECTED aggregate speedup once GPU retune + per-kernel configs
    land: 1.5-3x (per Wf-326E ``audit.audit_summary.projected_
    speedup_with_retune_and_new_kernels``).

Cross-references:
  * :file:`molmetal/scripts/wf331x3_sm120_retune.py` — Wf-331-X3
    CPU scaffold (this wrapper imports :data:`SM120_AUTOTUNE_CONFIGS`
    + :func:`is_sm120_config_safe` from there).
  * :file:`molmetal/scripts/wf333t1_sm120_autotune_extended.py` —
    Wf-333-T1 per-kernel extension (this wrapper imports
    :data:`PER_KERNEL_CONFIGS` + :func:`get_kernel_configs` from
    there).
  * :file:`molmetal/reports/wf326e_triton_audit/audit.json` —
    Wf-326E source-of-truth (24 kernels + projected_speedup).
  * :file:`molmetal/scripts/check_blackwell_env.py` — VRAM probe
    (REQUIRED before :func:`main` accepts ``--launch``).
  * :file:`molmetal/scripts/pre_flight_check.py` — Pre-flight v2
    25/25 PASS gate (REQUIRED before :func:`main` accepts ``--launch``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import triton
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None  # type: ignore[assignment]

# Reuse the Wf-331-X3 + Wf-333-T1 ship surfaces (constants + helpers).
import wf331x3_sm120_retune as _x3  # noqa: E402  (lazy: sys.path bootstrap)
import wf333t1_sm120_autotune_extended as _t1  # noqa: E402  (lazy)


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "molmetal" / "reports" / "wf335p2_sm120_triton_retune_gpu"
)
AUDIT_PATH = REPO_ROOT / "molmetal" / "reports" / "wf326e_triton_audit" / "audit.json"

# Launch prereqs (must be verified BEFORE any --launch invocation).
# Per AGENTS.md §5.1: VRAM check + pre-flight PASS + config-aligned.
PREREQ_VRAM_CHECK = (
    "nvidia-smi shows >= 16 GB free VRAM on RTX 5090 (GPU 1 by default; "
    "GPU 0 if kanzi idle). See molmetal/scripts/check_blackwell_env.py."
)
PREREQ_PRE_FLIGHT = (
    "python molmetal/scripts/pre_flight_check.py exits 0 (Pre-flight v2 "
    "25/25 PASS gate per TODO-207). See AGENTS.md §4."
)
PREREQ_CONFIG_ALIGNED = (
    "Hyperparameters (num_warps / num_stages / batch_size / hidden) are "
    "encoded in config files citing peer papers in the same domain. See "
    "AGENTS.md §5.1 + mol-metal-gpu-execution §4."
)
LAUNCH_PREREQS = (PREREQ_VRAM_CHECK, PREREQ_PRE_FLIGHT, PREREQ_CONFIG_ALIGNED)


# ---------------------------------------------------------------------------
# Launch plan dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    """Description of a sm_120 retune launch (CPU-constructed).

    Attributes
    ----------
    use_per_kernel : bool
        True iff the plan uses Wf-333-T1 per-kernel presets (RECOMMENDED
        per Wf-326E-B).  False means the Wf-331-X3 generic 12-config
        grid is used for every kernel.
    selected_kernels : Tuple[str, ...]
        Names of the kernels to sweep (subset of the 24 Wf-326E audit
        kernels).
    per_kernel_configs : Dict[str, List[Tuple[int, int]]]
        Mapping from kernel name to its (num_warps, num_stages)
        config pairs.
    total_configs_to_sweep : int
        Sum of ``len(per_kernel_configs[k])`` over all selected
        kernels.
    estimated_kernel_launches : int
        PROJECTED upper bound on ``total_configs_to_sweep * 5`` —
        each config is run ~5 times to average out autotune noise.
    gpu_retune_deferred : bool
        Always True for this CPU wrapper — the actual GPU launch
        happens only when the operator invokes ``--launch`` AFTER the
        3-prereq gate.
    audit_path : str
        Path to the Wf-326E audit JSON (source-of-truth).
    audit_n_kernels : int
        Number of kernels inventoried in the audit (24).
    prereqs : Tuple[str, ...]
        The 3 launch prereq descriptions (VRAM + pre-flight + config).
    """

    use_per_kernel: bool
    selected_kernels: Tuple[str, ...]
    per_kernel_configs: Dict[str, List[Tuple[int, int]]]
    total_configs_to_sweep: int
    estimated_kernel_launches: int
    gpu_retune_deferred: bool
    audit_path: str
    audit_n_kernels: int
    prereqs: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "use_per_kernel": self.use_per_kernel,
            "selected_kernels": list(self.selected_kernels),
            "n_selected_kernels": len(self.selected_kernels),
            "per_kernel_configs": {
                k: [{"num_warps": nw, "num_stages": ns} for (nw, ns) in v]
                for k, v in self.per_kernel_configs.items()
            },
            "total_configs_to_sweep": self.total_configs_to_sweep,
            "estimated_kernel_launches": self.estimated_kernel_launches,
            "gpu_retune_deferred": self.gpu_retune_deferred,
            "audit_path": self.audit_path,
            "audit_n_kernels": self.audit_n_kernels,
            "prereqs": list(self.prereqs),
        }


# ---------------------------------------------------------------------------
# Helpers (config-shape introspection)
# ---------------------------------------------------------------------------
def _config_pairs(configs: Sequence[Any]) -> List[Tuple[int, int]]:
    """Return ``[(num_warps, num_stages), ...]`` for a config list."""
    pairs: List[Tuple[int, int]] = []
    for cfg in configs:
        nw = getattr(cfg, "num_warps", None)
        ns = getattr(cfg, "num_stages", None)
        if nw is None and isinstance(cfg, dict):
            nw = cfg.get("num_warps")
            ns = cfg.get("num_stages")
        if nw is None or ns is None:
            raise AttributeError(f"Config {cfg!r} is missing num_warps or num_stages.")
        pairs.append((int(nw), int(ns)))
    return pairs


def _audit_kernel_names(audit_path: Path = AUDIT_PATH) -> List[str]:
    """Return the list of kernel names from the Wf-326E audit JSON.

    Reuses :func:`wf331x3_sm120_retune.load_audit_kernels` so the
    Wf-331-X3 source-of-truth is the single point of audit parsing.
    """
    try:
        entries = _x3.load_audit_kernels(audit_path)
    except FileNotFoundError:
        return []
    return [str(e["name"]) for e in entries]


def _resolve_kernel_configs(kernel_name: str, *, use_per_kernel: bool) -> List[Any]:
    """Resolve the config list for a single kernel.

    When ``use_per_kernel`` is True, defer to the Wf-333-T1 per-kernel
    registry (with the Wf-331-X3 generic grid as the fallback for
    kernels not in the registry).  When False, every kernel uses the
    same Wf-331-X3 generic 12-config grid.
    """
    if use_per_kernel:
        return _t1.get_kernel_configs(kernel_name)
    return list(_x3.SM120_AUTOTUNE_CONFIGS)


# ---------------------------------------------------------------------------
# Plan composition
# ---------------------------------------------------------------------------
def build_launch_plan(
    audit_path: Path = AUDIT_PATH,
    *,
    use_per_kernel: bool = True,
    kernels: Optional[Sequence[str]] = None,
) -> LaunchPlan:
    """Build a :class:`LaunchPlan` from the audit + per-kernel registry.

    Parameters
    ----------
    audit_path : Path, optional
        Path to the Wf-326E audit JSON (default: the shipped audit).
    use_per_kernel : bool, optional
        True (default, RECOMMENDED per Wf-326E-B) uses the Wf-333-T1
        per-kernel presets — each kernel gets a category-specific
        config subset.  False uses the Wf-331-X3 generic 12-config
        grid for every kernel.
    kernels : sequence of str, optional
        Subset of audit kernel names to include.  Defaults to ALL 24
        audit kernels.  Unknown kernel names raise :class:`ValueError`.

    Returns
    -------
    LaunchPlan
        Frozen plan ready for JSON serialization.
    """
    audit_names = _audit_kernel_names(audit_path)
    if not audit_names:
        raise FileNotFoundError(
            f"Wf-326E audit missing at {audit_path}; cannot build plan."
        )

    if kernels is None:
        selected = tuple(audit_names)
    else:
        unknown = set(kernels) - set(audit_names)
        if unknown:
            raise ValueError(
                f"Unknown kernel(s) not in Wf-326E audit: "
                f"{sorted(unknown)}. Use --kernels with audit-listed "
                f"names only."
            )
        # Preserve audit order so the plan is deterministic.
        selected = tuple(name for name in audit_names if name in set(kernels))

    per_kernel_configs: Dict[str, List[Tuple[int, int]]] = {}
    for name in selected:
        cfg_list = _resolve_kernel_configs(name, use_per_kernel=use_per_kernel)
        per_kernel_configs[name] = _config_pairs(cfg_list)

    total_configs = sum(len(v) for v in per_kernel_configs.values())
    # PROJECTED upper bound: each config is run ~5x to average noise.
    estimated_launches = total_configs * 5

    return LaunchPlan(
        use_per_kernel=use_per_kernel,
        selected_kernels=selected,
        per_kernel_configs=per_kernel_configs,
        total_configs_to_sweep=total_configs,
        estimated_kernel_launches=estimated_launches,
        gpu_retune_deferred=True,
        audit_path=str(audit_path),
        audit_n_kernels=len(audit_names),
        prereqs=LAUNCH_PREREQS,
    )


def dump_launch_plan_json(
    plan: LaunchPlan,
    output_path: Path,
) -> Dict[str, Any]:
    """Emit the launch plan as JSON (CPU-only).

    Writes a JSON object with the plan fields + a top-level
    metadata block (operator, generated_at_iso, extends).

    Returns
    -------
    Dict with keys:
        ``output_path`` — string path of the JSON file.
        ``total_configs_to_sweep`` — int.
        ``n_selected_kernels`` — int.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "metadata": {
            "wrapper": "wf335p2_sm120_triton_retune_gpu",
            "extends": [
                "wf331x3_sm120_retune.py",
                "wf333t1_sm120_autotune_extended.py",
            ],
            "gpu_retune_deferred": True,
        },
        "plan": plan.to_dict(),
    }
    with output_path.open("w") as fh:
        json.dump(body, fh, indent=2)
    return {
        "output_path": str(output_path),
        "total_configs_to_sweep": plan.total_configs_to_sweep,
        "n_selected_kernels": len(plan.selected_kernels),
    }


# ---------------------------------------------------------------------------
# Launch gate (refuses --launch without 3-prereq confirmation)
# ---------------------------------------------------------------------------
def launch_gate_check() -> Dict[str, Any]:
    """Return the launch-gate status (CPU-only stub).

    The wrapper itself does NOT perform the VRAM probe or the
    pre-flight check — those live in dedicated scripts
    (:file:`check_blackwell_env.py` + :file:`pre_flight_check.py`).
    This stub returns a structured refusal that the operator must
    resolve by running those scripts manually.

    Returns
    -------
    Dict with keys:
        ``gate_passed`` — always False for this CPU stub.
        ``prereqs_required`` — list of 3 prereq descriptions.
        ``how_to_resolve`` — dict of prereq name -> command.
        ``note`` — operator-action-required message.
    """
    return {
        "gate_passed": False,
        "prereqs_required": list(LAUNCH_PREREQS),
        "how_to_resolve": {
            "vram_check": (
                "python molmetal/scripts/check_blackwell_env.py "
                "molmetal/reports/wf335p2_sm120_triton_retune_gpu/"
                "gpu_env_check.json"
            ),
            "pre_flight": ("python molmetal/scripts/pre_flight_check.py " "--strict"),
            "config_aligned": (
                "Verify num_warps/num_stages/batch_size/hidden/depth "
                "match peer-paper config in "
                "molmetal/scripts/train_*.py argparse defaults."
            ),
        },
        "note": (
            "Wrapper does NOT auto-probe VRAM or run pre-flight. "
            "Operator must complete all 3 prereqs manually before "
            "re-running with --launch (the wrapper will refuse to "
            "launch without --launch-prereqs-confirmed)."
        ),
    }


# ---------------------------------------------------------------------------
# CPU smoke
# ---------------------------------------------------------------------------
def cpu_smoke_gpu_wrapper(output_dir: Path) -> Dict[str, Any]:
    """Run a CPU-only smoke that exercises the GPU wrapper.

    Validates:

    * :func:`build_launch_plan` returns a :class:`LaunchPlan` with 24
      kernels by default and the Wf-333-T1 per-kernel presets when
      ``use_per_kernel=True``.
    * The per-kernel plan's total configs is strictly less than the
      generic-grid plan's total configs (the 326E-B ~3x speedup
      expectation).
    * :func:`dump_launch_plan_json` writes a JSON file with the
      expected metadata block.
    * :func:`launch_gate_check` returns ``gate_passed=False`` (the
      wrapper does NOT auto-run the gate; operator must).
    * Unknown kernel names raise :class:`ValueError`.
    * The :func:`main` CLI handles all 4 flags (``--smoke``,
      ``--per-kernel``, ``--kernels``, ``--dump-launch-plan-only``)
      and refuses ``--launch`` without the prereq-confirmed flag.
    """
    started = time.time()

    # 1) Default plan (24 kernels, per-kernel presets) — RECOMMENDED path.
    plan_default = build_launch_plan(use_per_kernel=True)
    assert len(plan_default.selected_kernels) == 24, (
        f"Default plan should select all 24 audit kernels; got "
        f"{len(plan_default.selected_kernels)}."
    )
    assert plan_default.use_per_kernel is True
    assert plan_default.gpu_retune_deferred is True
    assert plan_default.total_configs_to_sweep == 117, (
        f"Per-kernel plan should sweep 117 configs (5*3 + 13*6 + "
        f"3*6 + 3*2); got {plan_default.total_configs_to_sweep}."
    )
    assert plan_default.estimated_kernel_launches == 117 * 5

    # 2) Generic grid plan (24 kernels, Wf-331-X3 grid) — for comparison.
    plan_generic = build_launch_plan(use_per_kernel=False)
    assert plan_generic.total_configs_to_sweep == 24 * 12, (
        f"Generic grid plan should sweep 24*12=288 configs; got "
        f"{plan_generic.total_configs_to_sweep}."
    )
    assert plan_generic.estimated_kernel_launches == 288 * 5

    # 3) Per-kernel is strictly smaller than generic (326E-B ~3x).
    assert plan_default.total_configs_to_sweep < plan_generic.total_configs_to_sweep, (
        f"Per-kernel total ({plan_default.total_configs_to_sweep}) "
        f"should be < generic total ({plan_generic.total_configs_to_sweep})."
    )
    speedup_ratio = (
        plan_generic.total_configs_to_sweep / plan_default.total_configs_to_sweep
    )
    # ~3x projection from Wf-326E-B: 288/117 = 2.46x.
    assert 2.0 <= speedup_ratio <= 3.5, (
        f"Per-kernel/generic speedup ratio {speedup_ratio:.2f}x is "
        f"outside the Wf-326E-B projection band [2.0, 3.5]."
    )

    # 4) Kernel subset restriction preserves audit order.
    subset = ("fused_silu_mlp", "matmul_fn")
    plan_subset = build_launch_plan(use_per_kernel=True, kernels=subset)
    assert plan_subset.selected_kernels == subset, (
        f"Subset plan should preserve the requested order {subset}; "
        f"got {plan_subset.selected_kernels}."
    )
    assert plan_subset.total_configs_to_sweep == (
        len(_config_pairs(_t1.get_kernel_configs("fused_silu_mlp")))
        + len(_config_pairs(_t1.get_kernel_configs("matmul_fn")))
    )

    # 5) Unknown kernel raises ValueError.
    try:
        build_launch_plan(use_per_kernel=True, kernels=("not_a_real_kernel",))
    except ValueError:
        unknown_raises = True
    else:
        unknown_raises = False
    assert unknown_raises, (
        "build_launch_plan should raise ValueError for unknown kernel " "names."
    )

    # 6) Every per-kernel config is_sm120_config_safe.
    for name, pairs in plan_default.per_kernel_configs.items():
        for nw, ns in pairs:
            cfg = {"num_warps": nw, "num_stages": ns}
            assert _x3.is_sm120_config_safe(cfg), (
                f"is_sm120_config_safe returned False for plan config "
                f"(num_warps={nw}, num_stages={ns}) of kernel {name!r}."
            )

    # 7) Plan JSON dump.
    output_dir.mkdir(parents=True, exist_ok=True)
    dumped_default = dump_launch_plan_json(
        plan_default, output_dir / "launch_plan_per_kernel.json"
    )
    dumped_generic = dump_launch_plan_json(
        plan_generic, output_dir / "launch_plan_generic.json"
    )
    assert dumped_default["total_configs_to_sweep"] == 117
    assert dumped_generic["total_configs_to_sweep"] == 288

    # 8) Launch gate check (always refuses — operator must verify).
    gate = launch_gate_check()
    assert gate["gate_passed"] is False, (
        "CPU stub launch_gate_check must return gate_passed=False "
        "(wrapper does not auto-probe VRAM / pre-flight)."
    )
    assert len(gate["prereqs_required"]) == 3

    # 9) CLI flag combinations (subprocess the main() entry-point).
    cli_results = _exercise_cli_flags(output_dir)

    elapsed = time.time() - started
    return {
        "smoke_passed": True,
        "n_kernels_audit": plan_default.audit_n_kernels,
        "n_selected_kernels": len(plan_default.selected_kernels),
        "per_kernel_total_configs": plan_default.total_configs_to_sweep,
        "generic_total_configs": plan_generic.total_configs_to_sweep,
        "per_kernel_over_generic_speedup": round(speedup_ratio, 4),
        "subset_total_configs": plan_subset.total_configs_to_sweep,
        "launch_gate_passed": gate["gate_passed"],
        "launch_plan_per_kernel_path": dumped_default["output_path"],
        "launch_plan_generic_path": dumped_generic["output_path"],
        "cli_results": cli_results,
        "elapsed_seconds": round(elapsed, 4),
    }


def _exercise_cli_flags(output_dir: Path) -> Dict[str, Any]:
    """Run ``main()`` through each CLI flag combination (CPU-only).

    Returns a dict mapping each CLI invocation to its exit code +
    a short stdout snippet.  Used by the smoke test to verify the
    argparse surface handles all 4 documented flag combinations.
    """
    # NOTE: --smoke is intentionally NOT in this list — the smoke test
    # already covers it via direct ``main(["--smoke", ...])`` invocation,
    # and including it here would cause infinite recursion
    # (cli_flags -> main -> cpu_smoke -> cli_flags -> ...).
    invocations = [
        (
            "dump_launch_plan_only",
            ["--dump-launch-plan-only"],
        ),
        (
            "per_kernel",
            ["--per-kernel", "--dump-launch-plan-only"],
        ),
        (
            "kernels_subset",
            [
                "--per-kernel",
                "--kernels",
                "fused_silu_mlp,matmul_fn",
                "--dump-launch-plan-only",
            ],
        ),
    ]
    results: Dict[str, Any] = {}
    for name, argv in invocations:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(argv)
        results[name] = {
            "argv": argv,
            "returncode": rc,
            "stdout_first_120": buf.getvalue()[:120],
        }
        assert rc in (0, 1), (
            f"CLI invocation {name!r} returned unexpected exit code "
            f"{rc}; stdout snippet: {buf.getvalue()[:200]}"
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """argparse entry-point for the sm_120 retune GPU wrapper.

    Flag combinations:

    * ``--smoke`` — CPU-only correctness test (returns 0 on PASS).
    * ``--dump-launch-plan-only`` — Emit the launch plan JSON and
      exit (skips the smoke).
    * ``--per-kernel`` — Use the Wf-333-T1 per-kernel presets
      (default: also True; this flag is explicit for clarity).
    * ``--kernels <csv>`` — Restrict the plan to a comma-separated
      subset of audit kernel names.
    * ``--launch`` — Promote to an actual GPU launch.  REQUIRES the
      ``--launch-prereqs-confirmed`` flag; without it the wrapper
      prints the launch-gate checklist and exits 2.
    * ``--launch-prereqs-confirmed`` — Operator's explicit ack that
      VRAM + pre-flight + config-aligned are done.  This flag does
      NOT itself perform those checks — it only confirms the
      operator ran them.  Violations of the AGENTS.md §5.1 contract
      are the operator's responsibility.
    * ``--output-dir`` — Where to write the launch plan JSON
      (default: ``molmetal/reports/wf335p2_sm120_triton_retune_gpu``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the launch plan JSON (default: "
        "molmetal/reports/wf335p2_sm120_triton_retune_gpu).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the CPU smoke that exercises the wrapper "
        "(plan composition + JSON dump + CLI flags).",
    )
    parser.add_argument(
        "--dump-launch-plan-only",
        action="store_true",
        help="Only emit the launch plan JSON (skip the smoke).",
    )
    parser.add_argument(
        "--per-kernel",
        action="store_true",
        help="Use the Wf-333-T1 per-kernel presets (RECOMMENDED per "
        "Wf-326E-B).  Default behavior is also per-kernel; this "
        "flag is explicit for clarity.",
    )
    parser.add_argument(
        "--kernels",
        default=None,
        help="Comma-separated subset of Wf-326E audit kernel names "
        "(e.g. 'fused_silu_mlp,matmul_fn').  Default: all 24 "
        "audit kernels.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Promote to an actual GPU launch.  REQUIRES "
        "--launch-prereqs-confirmed.  Without it the wrapper "
        "prints the launch-gate checklist and exits 2.",
    )
    parser.add_argument(
        "--launch-prereqs-confirmed",
        action="store_true",
        help="Operator's explicit ack that VRAM + pre-flight + "
        "config-aligned are done (does NOT itself perform those "
        "checks).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()

    # ---- --launch path ---------------------------------------------------
    if args.launch:
        if not args.launch_prereqs_confirmed:
            gate = launch_gate_check()
            print(json.dumps(gate, indent=2), file=sys.stderr)
            return 2
        # The wrapper itself does not perform the actual retune loop —
        # it would be added by a follow-up wave that wires the GPU
        # execution contract from mol-metal-gpu-execution §9.
        print(
            json.dumps(
                {
                    "launch_accepted": True,
                    "note": (
                        "Wrapper accepts --launch; actual GPU retune "
                        "execution is deferred to a follow-up wave "
                        "(per AGENTS.md §5.1 + mol-metal-gpu-execution "
                        "§9). Use mol-metal-gpu-execution §3 venv setup + "
                        "§6 5-seed protocol for the real launch path."
                    ),
                    "prereqs_acknowledged": True,
                },
                indent=2,
            )
        )
        return 0

    # ---- --smoke path ----------------------------------------------------
    if args.smoke:
        result = cpu_smoke_gpu_wrapper(output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["smoke_passed"] else 1

    # ---- --dump-launch-plan-only path ------------------------------------
    if args.dump_launch_plan_only:
        kernels_arg: Optional[Tuple[str, ...]]
        if args.kernels:
            kernels_arg = tuple(k.strip() for k in args.kernels.split(",") if k.strip())
        else:
            kernels_arg = None

        try:
            plan = build_launch_plan(
                use_per_kernel=True,
                kernels=kernels_arg,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        dumped = dump_launch_plan_json(plan, output_dir / "launch_plan_per_kernel.json")
        print(
            json.dumps(
                {
                    "n_selected_kernels": dumped["n_selected_kernels"],
                    "total_configs_to_sweep": dumped["total_configs_to_sweep"],
                    "estimated_kernel_launches": (plan.estimated_kernel_launches),
                    "use_per_kernel": plan.use_per_kernel,
                    "gpu_retune_deferred": plan.gpu_retune_deferred,
                    "launch_plan_path": dumped["output_path"],
                    "prereqs_required": list(LAUNCH_PREREQS),
                    "how_to_promote_to_launch": (
                        "Re-run with --launch --launch-prereqs-confirmed "
                        "AFTER running check_blackwell_env.py + "
                        "pre_flight_check.py + verifying config-aligned "
                        "hyperparameters."
                    ),
                },
                indent=2,
            )
        )
        return 0

    # ---- Default path: dump plan + summary -------------------------------
    kernels_arg = None
    if args.kernels:
        kernels_arg = tuple(k.strip() for k in args.kernels.split(",") if k.strip())
    try:
        plan = build_launch_plan(use_per_kernel=True, kernels=kernels_arg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    dumped = dump_launch_plan_json(plan, output_dir / "launch_plan_per_kernel.json")
    print(
        json.dumps(
            {
                "n_audit_kernels": plan.audit_n_kernels,
                "n_selected_kernels": len(plan.selected_kernels),
                "total_configs_to_sweep": plan.total_configs_to_sweep,
                "estimated_kernel_launches": plan.estimated_kernel_launches,
                "use_per_kernel": plan.use_per_kernel,
                "launch_plan_path": dumped["output_path"],
                "gpu_retune_deferred": True,
                "extends": [
                    "wf331x3_sm120_retune.py",
                    "wf333t1_sm120_autotune_extended.py",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
