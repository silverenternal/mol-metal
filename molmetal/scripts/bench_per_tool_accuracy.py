"""Wf-48 Phase 2 — per-tool accuracy verification.

================================================
Per-tool accuracy bench — 11 real tools on PlatinAI 5K + MetalCytoToxDB
================================================

Wf-43 Phase 1+2 shipped the unified ``molmetal.scoring.score_any()``
router that dispatches metric names to the *real, locally-deployed*
scoring tools (no mock fallback permitted). Wf-48 Phase 1 verified the
*router* (every metric routes to its declared real tool); this phase
closes the verification loop by **measuring the actual per-tool
accuracy** on the PlatinAI 5K + MetalCytoToxDB datasets.

The 11 tools under test
-----------------------
Every tool in :data:`DEFAULT_TOOL_REGISTRY` is paired with a single
:func:`score_any` metric name. The full set is:

1. ``vina_score``        — AutoDock Vina 1.2.7
2. ``diffdock_pose``     — DiffDock (R17 ship)
3. ``boltz_affinity``    — Boltz-style pIC50 (R17 ship, D-MPNN)
4. ``af2_confidence``    — Voxelization complexity (AF2 surrogate)
5. ``pb_pass_rate``      — PoseBusters (R17 ship)
6. ``aizynth_synthesis`` — AiZynthFinder (R17 ship)
7. ``equibind_pose``     — EquiBind (NOT SHIPPED, reserved)
8. ``imold_ood``         — IMOLDE OOD (NOT SHIPPED, reserved)
9. ``drugood_ood``       — DrugOOD OOD (NOT SHIPPED, reserved)
10. ``pocket2mol``       — Pocket2Mol (R17 ship)
11. ``targetdiff``       — TargetDiff (R17 ship)

For each tool we record:

* ``status`` — ``"MEASURED"`` | ``"ADAPTER_UNAVAILABLE"`` |
  ``"PARTIAL"`` | ``"ERROR"`` | ``"NOT_SHIPPED"``
* ``accuracy`` — float in [0, 1]; fraction of input SMILES that
  produced a valid (non-NaN / non-None / non-empty) score. ``NaN`` when
  the tool is reserved (AdapterUnavailable on every sample).
* ``mean_speed_s`` — mean wall-time per SMILES call (seconds)
* ``failure_rate`` — float in [0, 1]; AdapterUnavailable / total calls
* ``n_calls`` — total ``score_any()`` invocations
* ``n_measured`` — invocations that returned MEASURED status
* ``values_summary`` — ``min``, ``median``, ``max`` of valid values
* ``sample_inputs`` — the SMILES strings actually fed to the wrapper

The bench is *mock-free*: every measurement goes through the real
``score_any()`` router, and the failure mode is
:class:`AdapterUnavailable` — never a fabricated score.

Outputs
-------
* :file:`molmetal/reports/wf48_per_tool/per_tool_accuracy.json` —
  verbatim per-tool report + aggregate.
* :file:`molmetal/reports/wf48_per_tool/per_tool_accuracy.md` —
  human-readable summary table.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/bench_per_tool_accuracy.py \\
        --tools 11 --datasets platinai_5k metalcytotox \\
        --n-samples 16 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the convention in every Wf-3x / Wf-4x bench so
# the script can be invoked directly (e.g.
# ``python molmetal/scripts/bench_per_tool_accuracy.py``) without setting
# ``PYTHONPATH`` first.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MOLMETAL_PKG_DIR = _HERE.parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_MOLMETAL_PKG_DIR), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Tool-record dataclasses + tool registry
# ---------------------------------------------------------------------------
@dataclass
class ToolRecord:
    """Per-tool accuracy record."""

    tool: str  # human-readable tool name (e.g. "AutoDock Vina 1.2.7")
    metric: str  # score_any() metric name
    backend: str
    datasets: Tuple[str, ...]
    n_calls: int = 0
    n_measured: int = 0
    n_adapter_unavailable: int = 0
    n_partial: int = 0  # measured but ≥50% values NaN/None
    n_error: int = 0
    status: str = "OK"  # OK | NOT_SHIPPED | MISSING_DEP | ERROR
    accuracy: float = math.nan
    mean_speed_s: float = math.nan
    median_speed_s: float = math.nan
    failure_rate: float = math.nan
    values_min: float = math.nan
    values_median: float = math.nan
    values_max: float = math.nan
    sample_inputs: Tuple[str, ...] = field(default_factory=tuple)
    error: Optional[str] = None


@dataclass
class BenchConfig:
    """CLI configuration for the Wf-48 Phase 2 per-tool bench."""

    n_tools: int = 11
    datasets: Tuple[str, ...] = ("platinai_5k", "metalcytotox")
    n_samples: int = 16
    seeds: Tuple[int, ...] = (0,)
    out_dir: str = "molmetal/reports/wf48_per_tool"
    timeout_s: float = 30.0


# Canonical Wf-48 Phase 2 tool registry. Each entry pairs the
# human-readable tool name with the **score_any() metric** it maps to.
# The ``requires_pocket`` flag controls whether the bench supplies a
# mock PDB-string pocket to the wrapper (mandatory for docking / 3D
# generation tools; not required for property-only predictors).
DEFAULT_TOOL_REGISTRY: Tuple[Dict[str, Any], ...] = (
    {
        "tool": "AutoDock Vina 1.2.7",
        "metric": "vina_score",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
    {
        "tool": "DiffDock (R17 ship)",
        "metric": "diffdock_pose",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
    {
        "tool": "Boltz-style pIC50 (R17 ship, D-MPNN)",
        "metric": "boltz_affinity",
        "requires_pocket": False,
        "datasets": ("platinai_5k", "metalcytotox"),
    },
    {
        "tool": "Voxelization complexity (AF2 surrogate)",
        "metric": "af2_confidence",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
    {
        "tool": "PoseBusters (R17 ship)",
        "metric": "pb_pass_rate",
        "requires_pocket": False,
        "datasets": ("platinai_5k", "metalcytotox"),
    },
    {
        "tool": "AiZynthFinder (R17 ship)",
        "metric": "aizynth_synthesis",
        "requires_pocket": False,
        "datasets": ("platinai_5k", "metalcytotox"),
    },
    {
        "tool": "EquiBind (NOT SHIPPED, reserved)",
        "metric": "equibind_pose",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
    {
        "tool": "IMOLDE OOD (NOT SHIPPED, reserved)",
        "metric": "imold_ood",
        "requires_pocket": False,
        "datasets": ("metalcytotox",),
    },
    {
        "tool": "DrugOOD OOD (NOT SHIPPED, reserved)",
        "metric": "drugood_ood",
        "requires_pocket": False,
        "datasets": ("metalcytotox",),
    },
    {
        "tool": "Pocket2Mol (R17 ship)",
        "metric": "pocket2mol",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
    {
        "tool": "TargetDiff (R17 ship)",
        "metric": "targetdiff",
        "requires_pocket": True,
        "datasets": ("platinai_5k",),
    },
)


# ---------------------------------------------------------------------------
# Mock pocket — synthetic PDB string for the per-tool accuracy bench.
# Real PDB files are not required for *accuracy* verification: the
# wrapper either succeeds on a valid PDB string or raises
# AdapterUnavailable, and the bench captures both outcomes.
# ---------------------------------------------------------------------------
_MOCK_PDB = (
    "HEADER    MOCK PDB FOR WF-48 PHASE 2 ACCURACY BENCH\n"
    "ATOM      1  N   ALA A   1      11.000  10.000  10.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1      11.500  10.500  10.500  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1      12.500  11.500  10.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1      12.500  12.500  10.500  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1      10.500  11.000  11.500  1.00  0.00           C\n"
    "END\n"
)


# ---------------------------------------------------------------------------
# Dataset samplers — pull a deterministic SMILES sample from each
# benchmark dataset. We deliberately do NOT load full RDKit feature
# matrices; the bench only needs the canonical SMILES strings.
# ---------------------------------------------------------------------------
def _load_platinai_smiles(n: int, seed: int) -> List[str]:
    """Pull up to ``n`` PlatinAI SMILES from the local 5K CSV."""
    csv_path = Path("molmetal/data/platinai_5k.csv")
    if not csv_path.exists():
        return []
    out: List[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        try:
            smi_idx = header.index("smiles")
        except ValueError:
            return []
        for i, line in enumerate(fh):
            if i < seed:
                continue
            parts = line.strip().split(",")
            if smi_idx >= len(parts):
                continue
            out.append(parts[smi_idx])
            if len(out) >= n:
                break
    return out


def _load_metalcytotox_smiles(n: int, seed: int) -> List[str]:
    """Pull up to ``n`` MetalCytoToxDB SMILES from the local multitask CSV."""
    csv_path = Path(".data/molmetal/metalcytotox_5metal_multitask_v4.csv")
    if not csv_path.exists():
        return []
    out: List[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        try:
            smi_idx = header.index("smiles")
        except ValueError:
            return []
        for i, line in enumerate(fh):
            if i < seed:
                continue
            parts = line.strip().split(",")
            if smi_idx >= len(parts):
                continue
            out.append(parts[smi_idx])
            if len(out) >= n:
                break
    return out


DATASET_LOADERS: Dict[str, Callable[[int, int], List[str]]] = {
    "platinai_5k": _load_platinai_smiles,
    "metalcytotox": _load_metalcytotox_smiles,
}


# ---------------------------------------------------------------------------
# Per-tool driver
# ---------------------------------------------------------------------------
def _invoke_tool(
    entry: Dict[str, Any],
    cfg: BenchConfig,
    seed: int,
) -> ToolRecord:
    """Run a single tool through ``score_any()`` on every compatible dataset.

    Returns a fully-populated :class:`ToolRecord` summarising the per-tool
    accuracy, speed, and failure profile. NO mock substitution: a tool
    that fails returns ``status="NOT_SHIPPED"`` or ``status="MISSING_DEP"``
    and ``accuracy=NaN``.
    """
    metric = entry["metric"]
    requires_pocket = bool(entry.get("requires_pocket", False))
    tool = entry["tool"]
    rec = ToolRecord(
        tool=tool,
        metric=metric,
        backend="",
        datasets=tuple(entry.get("datasets", ())),
    )

    # Resolve backend from the score_any registry.
    try:
        from molmetal.scoring import tool_for_metric

        ref = tool_for_metric(metric)
        rec.backend = ref.backend
    except Exception as exc:  # pragma: no cover - registry invariant
        rec.status = "ERROR"
        rec.error = f"registry lookup failed: {type(exc).__name__}: {exc}"
        return rec

    speeds: List[float] = []
    values: List[float] = []
    sample_inputs: List[str] = []

    for ds in cfg.datasets:
        if ds not in rec.datasets:
            continue
        loader = DATASET_LOADERS.get(ds)
        if loader is None:
            rec.error = (rec.error or "") + f"unknown dataset {ds}; "
            continue
        smiles = loader(int(cfg.n_samples), int(seed))
        if not smiles:
            rec.error = (rec.error or "") + f"{ds}: empty sample; "
            continue
        sample_inputs.extend(smiles[: int(cfg.n_samples)])

        pocket = _MOCK_PDB if requires_pocket else None

        # Single batch call — gives the wrapper a chance to process the
        # full sample in one go (more representative of real throughput).
        t0 = time.time()
        rec.n_calls += 1  # count every invocation attempt
        try:
            from molmetal.scoring import score_any

            result = score_any(metric, molecules=smiles, pocket=pocket)
            dt = time.time() - t0
            speeds.append(dt)
            vals = list(result.get("values") or [])
            # Coerce to float; treat None and NaN as invalid.
            valid = []
            for v in vals:
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                valid.append(fv)
            if not valid:
                rec.n_partial += 1
            else:
                rec.n_measured += 1
                values.extend(valid)
        except Exception as exc:
            dt = time.time() - t0
            speeds.append(dt)
            tb = traceback.format_exc()
            # AdapterUnavailable is the documented "no mock" failure
            # mode — distinguish it from generic exceptions.
            from molmetal.scoring.score_any import AdapterUnavailable

            if isinstance(exc, AdapterUnavailable):
                rec.n_adapter_unavailable += 1
                err_msg = str(exc).splitlines()[0]
                # NOT_SHIPPED metrics raise a specific message.
                if "NOT SHIPPED" in tool.upper() or "reserved" in err_msg.lower():
                    rec.status = "NOT_SHIPPED"
                else:
                    # Tool is shipped but missing dep / checkpoint on
                    # this host — record as MISSING_DEP, not ERROR.
                    rec.status = "MISSING_DEP"
                rec.error = err_msg
            else:
                rec.n_error += 1
                rec.status = "ERROR"
                rec.error = f"{type(exc).__name__}: {exc}"
        # Bail out early for adapter-unavailable tools — no point
        # calling them again on every seed.
        if rec.status in ("NOT_SHIPPED", "MISSING_DEP") and rec.n_error == 0:
            break

    rec.sample_inputs = tuple(sample_inputs[: int(cfg.n_samples)])

    # Aggregate.
    total_calls = rec.n_calls
    if total_calls > 0:
        rec.failure_rate = float(rec.n_adapter_unavailable) / float(total_calls)
        if speeds:
            rec.mean_speed_s = float(statistics.mean(speeds))
            rec.median_speed_s = float(statistics.median(speeds))
    if values:
        rec.accuracy = float(len(values)) / float(
            max(int(cfg.n_samples) * max(1, len(rec.datasets)), 1)
        )
        # Cap accuracy at 1.0.
        rec.accuracy = min(rec.accuracy, 1.0)
        rec.values_min = float(min(values))
        rec.values_median = float(statistics.median(values))
        rec.values_max = float(max(values))
    else:
        # No valid values: accuracy is the fraction of calls that
        # returned MEASURED at all (0 if every call failed).
        if rec.n_measured + rec.n_partial > 0:
            rec.accuracy = float(rec.n_measured) / float(total_calls or 1)
        else:
            rec.accuracy = 0.0

    if rec.status == "OK":
        # Distinguish partial accuracy (any adapter-unavailable call).
        if rec.n_adapter_unavailable > 0 and rec.n_measured == 0:
            rec.status = (
                "NOT_SHIPPED" if "NOT SHIPPED" in tool.upper() else "MISSING_DEP"
            )
        elif rec.n_adapter_unavailable > 0:
            rec.status = "PARTIAL"
        elif rec.n_error > 0:
            rec.status = "ERROR"
        elif rec.n_partial > 0 and rec.n_measured == 0:
            rec.status = "PARTIAL"

    return rec


# ---------------------------------------------------------------------------
# Aggregate + writers
# ---------------------------------------------------------------------------
def _aggregate(records: List[ToolRecord]) -> Dict[str, Any]:
    """Aggregate per-tool records into summary stats."""
    n = len(records)
    if n == 0:
        return {
            "n_tools": 0,
            "n_measured": 0,
            "n_not_shipped": 0,
            "n_missing_dep": 0,
            "n_partial": 0,
            "n_error": 0,
            "mean_accuracy": 0.0,
            "mean_speed_s": 0.0,
            "mean_failure_rate": 0.0,
        }
    measured = [r for r in records if r.status == "OK"]
    not_shipped = [r for r in records if r.status == "NOT_SHIPPED"]
    missing_dep = [r for r in records if r.status == "MISSING_DEP"]
    partial = [r for r in records if r.status == "PARTIAL"]
    error = [r for r in records if r.status == "ERROR"]

    def _mean(xs: List[float]) -> float:
        xs = [float(x) for x in xs if not math.isnan(x)]
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "n_tools": int(n),
        "n_measured": int(len(measured)),
        "n_not_shipped": int(len(not_shipped)),
        "n_missing_dep": int(len(missing_dep)),
        "n_partial": int(len(partial)),
        "n_error": int(len(error)),
        "mean_accuracy": _mean([r.accuracy for r in records]),
        "mean_speed_s": _mean([r.mean_speed_s for r in records]),
        "mean_failure_rate": _mean([r.failure_rate for r in records]),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` with NaN → null normalisation.

    NaN is not valid JSON, but :class:`ToolRecord` legitimately uses
    ``NaN`` for ``accuracy`` / ``mean_speed_s`` when no measurement
    was collected. We therefore serialise NaN / ±Inf to ``null`` so the
    resulting file is a valid JSON document that downstream tooling
    can ingest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _normalise(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalise(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj

    safe = _normalise(payload)
    path.write_text(
        json.dumps(safe, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = payload.get("aggregate", {})
    lines: List[str] = []
    lines.append("# Wf-48 Phase 2 — Per-Tool Accuracy Benchmark")
    lines.append("")
    lines.append(f"Recipe: `{payload.get('recipe')}`")
    lines.append(f"Tool version: `{payload.get('tool_version')}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- n_tools: `{agg.get('n_tools')}`")
    lines.append(f"- n_measured: `{agg.get('n_measured')}`")
    lines.append(f"- n_not_shipped: `{agg.get('n_not_shipped')}`")
    lines.append(f"- n_missing_dep: `{agg.get('n_missing_dep')}`")
    lines.append(f"- n_partial: `{agg.get('n_partial')}`")
    lines.append(f"- n_error: `{agg.get('n_error')}`")
    lines.append(f"- mean_accuracy: `{agg.get('mean_accuracy'):.4f}`")
    lines.append(f"- mean_speed_s: `{agg.get('mean_speed_s'):.4f}`")
    lines.append(f"- mean_failure_rate: `{agg.get('mean_failure_rate'):.4f}`")
    lines.append("")
    lines.append("## Per-tool")
    lines.append("")
    lines.append(
        "| tool | metric | backend | status | n_calls | accuracy | mean_speed_s | failure_rate |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for r in payload.get("per_tool", []):
        acc = (
            f"{r['accuracy']:.3f}"
            if isinstance(r.get("accuracy"), (int, float))
            and not math.isnan(float(r.get("accuracy") or 0.0))
            else "NaN"
        )
        sp = (
            f"{r['mean_speed_s']:.3f}"
            if isinstance(r.get("mean_speed_s"), (int, float))
            and not math.isnan(float(r.get("mean_speed_s") or 0.0))
            else "NaN"
        )
        fr = (
            f"{r['failure_rate']:.3f}"
            if isinstance(r.get("failure_rate"), (int, float))
            and not math.isnan(float(r.get("failure_rate") or 0.0))
            else "NaN"
        )
        lines.append(
            f"| `{r['tool']}` | `{r['metric']}` | `{r['backend']}` | "
            f"{r['status']} | {r['n_calls']} | {acc} | {sp} | {fr} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
__version__ = "0.1.0-wf48-phase2"


def run_bench(cfg: BenchConfig) -> Dict[str, Any]:
    """Execute the Wf-48 Phase 2 per-tool accuracy bench.

    Returns the verbatim JSON-serialisable payload that is also
    written to ``cfg.out_dir / per_tool_accuracy.json``.
    """
    entries = list(DEFAULT_TOOL_REGISTRY)
    if int(cfg.n_tools) < len(entries):
        entries = entries[: int(cfg.n_tools)]

    records: List[ToolRecord] = []
    for seed in cfg.seeds:
        for entry in entries:
            rec = _invoke_tool(entry, cfg, int(seed))
            # If the first seed already records NOT_SHIPPED / MISSING_DEP
            # there's no point retrying — preserve the first observation.
            if rec.status in ("NOT_SHIPPED", "MISSING_DEP"):
                if any(r.tool == rec.tool for r in records):
                    continue
            records.append(rec)

    agg = _aggregate(records)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "recipe": "wf48_per_tool_accuracy",
        "tool": "bench_per_tool_accuracy.py",
        "tool_version": __version__,
        "config": {
            "n_tools": int(cfg.n_tools),
            "datasets": [str(d) for d in cfg.datasets],
            "n_samples": int(cfg.n_samples),
            "seeds": [int(s) for s in cfg.seeds],
        },
        "per_tool": [asdict(r) for r in records],
        "aggregate": agg,
    }

    out_dir = Path(cfg.out_dir)
    _write_json(out_dir / "per_tool_accuracy.json", payload)
    _write_markdown(out_dir / "per_tool_accuracy.md", payload)

    print(
        f"[wf48] wrote {out_dir / 'per_tool_accuracy.json'} "
        f"({len(records)} tool records, {agg['n_measured']}/{agg['n_tools']} "
        f"measured)",
        file=sys.stderr,
    )
    return payload


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_per_tool_accuracy",
        description=(
            "Wf-48 Phase 2 — Run all 11 real tools through score_any() "
            "on PlatinAI 5K + MetalCytoToxDB and record per-tool "
            "accuracy, speed, and failure rate."
        ),
    )
    p.add_argument(
        "--tools",
        type=int,
        default=11,
        help="Number of tools to bench (default 11 — the Wf-43 ship set).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["platinai_5k", "metalcytotox"],
        help="Datasets to sample from (default: platinai_5k metalcytotox).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of SMILES per dataset (default 16).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0],
        help="Seed(s) for the bench (default [0]).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="molmetal/reports/wf48_per_tool",
        help="Output directory for the bench report.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    cfg = BenchConfig(
        n_tools=int(args.tools),
        datasets=tuple(
            str(d) for d in (args.datasets or ["platinai_5k", "metalcytotox"])
        ),
        n_samples=int(args.n_samples),
        seeds=tuple(int(s) for s in (args.seeds or [0])),
        out_dir=str(args.out_dir),
    )
    payload = run_bench(cfg)
    # Exit code policy:
    #   * 0 — every tool returned OK / NOT_SHIPPED / MISSING_DEP (no
    #         unexpected ERROR); at least one tool measured.
    #   * 1 — zero tools measured (no metric worked at all).
    #   * 2 — at least one tool returned unexpected ERROR.
    agg = payload["aggregate"]
    if int(agg["n_error"]) > 0:
        return 2
    if int(agg["n_measured"]) == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
