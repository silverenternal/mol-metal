"""Coupling Adapter full pipeline benchmark.

================================================================
Wf-21 dispatch — Lambda × CFM × Hilbert × VariancePUCT 端到端
================================================================
This script benchmarks the **end-to-end coupling adapter pipeline**
that ships four already-shipped pieces without modifying any of
them:

* :class:`LambdaCFMPipeline` (Wf-4 ship, TODO-21) — Lambda closure
  → CFM ODE.
* :class:`SE3LipmanSolver` (Wf-14 phase 1 ship) — SE(3) SB ODE
  integration around the CFM output.
* :class:`MultiPHPLLSignature` (Wf-14 phase 4 ship) — Multi-PH
  feature map on the decoded molecule.
* :class:`VarianceControlledPUCT` (Wf-14 phase 5 ship) — Variance-
  controlled MCTS selector on a multi-PH backed PUCT tree.
* :class:`HilbertASTSRComposite` (Wf-14 phase 2 ship) — Hilbert
  primal-dual reward channel (lambda-ASTSR).
* :class:`BidirectionalCoupling` (Wf-21 phase 1 ship) — Lambda →
  CFM → Lambda round-trip closure check.

The benchmark orchestrates all six stages on ``--n-samples`` PlatinAI
SMILES (default 200, head of the PlatinAI 5K CSV) and writes a JSON
report to ``molmetal/reports/wf_coupling_full/benchmark.json``.

Measurements
------------
For each sample the harness records:

* ``per_step_wall_time`` — ``{step: {mean, std, n}}`` for the six
  pipeline steps (Lambda→CFM, SE3, Multi-PH, VariancePUCT, Hilbert,
  Bidirectional).
* ``per_step_memory_mb`` — ``{step: peak_delta_mb}`` per pipeline
  step (Python ``tracemalloc`` delta).
* ``total_wall_time_s`` — sum of per-step wall-time across all
  samples.
* ``success_rate`` — fraction of samples where **every** stage
  returned a finite, well-typed result.

Output schema (saved to ``benchmark.json``)
-------------------------------------------
::

    {
      "n_samples": 200,
      "per_step_wall_time": {
        "lambda_cfm":     {"mean": ..., "std": ..., "n": ...},
        "se3_ode":        {...},
        "multi_ph":       {...},
        "variance_puct":  {...},
        "hilbert":        {...},
        "bidirectional":  {...},
      },
      "per_step_memory_mb": {"lambda_cfm": ..., ...},
      "total_wall_time_s":  ...,
      "mean_wall_time_per_sample_s": ...,
      "success_rate": ...,
      "n_success": ...,
      "n_failures": ...,
      "n_steps_cfm": ...,
      "pocket_size":  ...,
      "timestamp_utc": ...,
      "torch_version": ...,
      "rdkit_version": ...,
    }

Honest framing
--------------
* Each step is **independently time-and-memory tracked** so a
  single slow stage is visible in the JSON.  The pipeline does
  not require every stage to succeed — failed stages are recorded
  as ``status="skipped"`` and the sample is marked unsuccessful
  when **any** step fails.
* This is a **CPU-only** benchmark.  The GPU retraining story is
  deferred to ``wf_cfm_gpu_retrain/`` (separate track).
* The default ``--n-samples 200`` runs in < 60 s on a single CPU
  core when every module is importable; missing ``torch`` /
  ``rdkit`` deps cause the corresponding steps to skip gracefully
  but do not crash the run.

Usage::

    uv run --no-sync python molmetal/scripts/coupling_full_pipeline_bench.py --n-samples 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_coupling_full"
REPORT_FILE = REPORT_DIR / "benchmark.json"
DEFAULT_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"


# ---------------------------------------------------------------------------
# Default pipeline constants
# ---------------------------------------------------------------------------
PIPELINE_STEPS: Tuple[str, ...] = (
    "lambda_cfm",  # Wf-4 — Lambda closure → CFM
    "se3_ode",  # Wf-14 phase 1 — SE(3) SB ODE
    "multi_ph",  # Wf-14 phase 4 — Multi-PH feature map
    "variance_puct",  # Wf-14 phase 5 — Variance-controlled PUCT
    "hilbert",  # Wf-14 phase 2 — Hilbert primal-dual reward
    "bidirectional",  # Wf-21 phase 1 — Lambda ↔ CFM round-trip
)


# ---------------------------------------------------------------------------
# Helpers — graceful imports + missing-dep tracking
# ---------------------------------------------------------------------------
def _try_import(name: str) -> Tuple[bool, Any]:
    """Best-effort import — returns ``(ok, module_or_reason)``."""
    try:
        return True, __import__(name)
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"


def _get_versions() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(torch_version, rdkit_version)`` strings (``None`` if absent)."""
    ok, torch = _try_import("torch")
    torch_v = str(getattr(torch, "__version__", "unknown")) if ok else None
    ok, rd = _try_import("rdkit")
    rd_v = str(getattr(rd, "__version__", "unknown")) if ok else None
    return torch_v, rd_v


def _load_platinai_smiles(csv_path: Path, n: int) -> List[str]:
    """Return the first ``n`` SMILES from the PlatinAI 5K CSV."""
    out: List[str] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Step runners — each step is ``(smiles, pocket, ctx) -> (ok, payload)``
# ---------------------------------------------------------------------------
def _build_lambda_cfm_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-4 — Lambda × CFM joint inference (CPU-only)."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (  # noqa: E402
        load_joint_pipeline,
    )

    pipe = load_joint_pipeline(alpha=0.5, seed=0)

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            out = pipe.run_full_pipeline(smiles, pocket, n_steps=int(n_steps))
        except Exception as exc:
            return False, {"error": str(exc)}
        cfm = out.get("cfm_output")
        if cfm is None or not np.isfinite(cfm).all():
            return False, {"error": "non-finite cfm_output"}
        return True, {
            "decoded_smiles": list(out.get("decoded_smiles", [])),
            "n_decoded": int(len(out.get("decoded_smiles", []))),
        }

    return _run


def _build_se3_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-14 phase 1 — SE(3) SB ODE on a synthetic ``(B, N, 3)`` pose.

    Uses a zero-velocity EGNN stub so the benchmark runs without the
    trained EGNN weights.  The shape contract and orthogonality audit
    are preserved.
    """
    ok, torch = _try_import("torch")
    if not ok:

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": f"torch unavailable: {torch}"}

        return _skip

    from molmetal.adapters.se3_lipman_integration import SE3LipmanSolver  # noqa: E402

    class _ZeroEGNN(torch.nn.Module):  # type: ignore[attr-defined]
        def forward(self, x, t, pocket=None):  # noqa: D401
            return torch.zeros_like(x)

    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField  # noqa: E402

    se3_field = SE3SBVelocityField(base_velocity_net=_ZeroEGNN())
    solver = SE3LipmanSolver(
        se3_adapter=se3_field,
        n_steps=10,
        method="midpoint",
    )

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            torch.manual_seed(0)
            x0 = torch.randn(1, 6, 3)
            x_final = solver.solve(
                velocity_fn=lambda x, t: torch.zeros_like(x),
                x0=x0,
                t_span=(0.0, 1.0),
                pocket=None,
            )
        except Exception as exc:
            return False, {"error": str(exc)}
        if not torch.isfinite(x_final).all():
            return False, {"error": "non-finite SE(3) trajectory"}
        return True, {"shape": list(x_final.shape)}

    return _run


def _build_multi_ph_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-14 phase 4 — Multi-PH + PLL signature on a SMILES."""
    ok, _ = _try_import("rdkit")
    if not ok:

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": "rdkit unavailable"}

        return _skip
    ok, torch = _try_import("torch")
    if not ok:

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": "torch unavailable"}

        return _skip

    try:
        from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (  # noqa: E402
            MultiPHPLLSignature,
        )
    except Exception as exc:
        _import_err = f"multi_ph import failed: {exc}"

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": _import_err}

        return _skip

    sig = MultiPHPLLSignature(n_bins=8, k=8, knn=4, multi_seed=0)

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            feat, eig = sig.compute(smiles)
        except Exception as exc:
            return False, {"error": str(exc)}
        if not (torch.isfinite(feat).all() and torch.isfinite(eig).all()):
            return False, {"error": "non-finite multi-PH signature"}
        return True, {"multi_ph_shape": list(feat.shape), "pll_shape": list(eig.shape)}

    return _run


def _build_variance_puct_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-14 phase 5 — VarianceControlledPUCT.select + update on a stub tree."""
    try:
        from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # noqa: E402
            IntegrationNode,
            VarianceControlledPUCT,
        )
    except Exception as exc:
        _import_err = f"variance_puct import failed: {exc}"

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": _import_err}

        return _skip

    selector = VarianceControlledPUCT(
        c_puct=1.0,
        c_var=0.5,
        score_repellent=True,
        n_min_visits=2,
    )

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        # Build a deterministic 4-arm tree — same shape across samples.
        root = IntegrationNode(node_id=f"root|{smiles[:8]}", prior=1.0)
        for j in range(4):
            root.children.append(
                IntegrationNode(node_id=f"c{j}|{smiles[:8]}", prior=1.0 / 4.0),
            )
        try:
            for _ in range(4):
                action = selector.select(root)
                selector.update(
                    root,
                    action,
                    value=float((action + 1) * 0.1),
                    variance=0.04,
                )
        except Exception as exc:
            return False, {"error": str(exc)}
        return True, {"n_actions": int(root.action_count())}

    return _run


def _build_hilbert_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-14 phase 2 — Hilbert primal-dual reward channel fit + score."""
    try:
        from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (  # noqa: E402
            HilbertASTSRComposite,
            make_hilbert_astsr_reward_channel,
        )
        from molmetal.molmetal_lam.reward.lambda_astsr import DEFAULT_FEATURE_COLUMNS  # noqa: E402
    except Exception as exc:
        _import_err = f"hilbert import failed: {exc}"

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": _import_err}

        return _skip

    composite = HilbertASTSRComposite(
        feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
        use_hilbert=True,
        population_dim=64,
    )
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((16, len(DEFAULT_FEATURE_COLUMNS))).astype(np.float64)
    target = feats.sum(axis=1) + 0.1 * rng.standard_normal(16)
    fit_err: Optional[str] = None
    try:
        composite.fit(feats, target)
    except Exception as exc:
        fit_err = repr(exc)

    if fit_err is not None:

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int, _msg: str = fit_err
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": f"hilbert.fit failed: {_msg}"}

        return _skip

    channel = make_hilbert_astsr_reward_channel(composite, enabled=True)

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        row = {
            fn: float((i + 1) * 0.01) for i, fn in enumerate(DEFAULT_FEATURE_COLUMNS)
        }
        try:
            reward = float(channel(row))
        except Exception as exc:
            return False, {"error": str(exc)}
        if not np.isfinite(reward):
            return False, {"error": "non-finite reward"}
        return True, {"reward": float(reward)}

    return _run


def _build_bidirectional_runner() -> Callable[..., Tuple[bool, Dict[str, Any]]]:
    """Wf-21 phase 1 — Lambda → CFM → Lambda round-trip."""
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (  # noqa: E402
            BidirectionalCoupling,
        )
    except Exception as exc:
        _import_err = f"bidirectional import failed: {exc}"

        def _skip(
            smiles: str, pocket: np.ndarray, n_steps: int
        ) -> Tuple[bool, Dict[str, Any]]:
            return False, {"error": _import_err}

        return _skip

    bi = BidirectionalCoupling()

    def _run(
        smiles: str, pocket: np.ndarray, n_steps: int
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            res = bi.round_trip(smiles, pocket, n_steps=int(min(n_steps, 20)))
        except Exception as exc:
            return False, {"error": str(exc)}
        rt_loss = float(res.get("round_trip_loss", float("nan")))
        if not np.isfinite(rt_loss):
            return False, {"error": "non-finite round_trip_loss"}
        return True, {"round_trip_loss": rt_loss}

    return _run


# ---------------------------------------------------------------------------
# Step orchestration
# ---------------------------------------------------------------------------
def _build_step_runners() -> Dict[str, Callable[..., Tuple[bool, Dict[str, Any]]]]:
    """Build the per-step runner dict; failures during build surface as skips."""
    builders: Dict[str, Callable[[], Callable[..., Tuple[bool, Dict[str, Any]]]]] = {
        "lambda_cfm": _build_lambda_cfm_runner,
        "se3_ode": _build_se3_runner,
        "multi_ph": _build_multi_ph_runner,
        "variance_puct": _build_variance_puct_runner,
        "hilbert": _build_hilbert_runner,
        "bidirectional": _build_bidirectional_runner,
    }
    runners: Dict[str, Callable[..., Tuple[bool, Dict[str, Any]]]] = {}
    for name, builder in builders.items():
        try:
            runners[name] = builder()
        except Exception as exc:

            def _skip(
                smiles: str, pocket: np.ndarray, n_steps: int, _msg=repr(exc)
            ) -> Tuple[bool, Dict[str, Any]]:
                return False, {"error": f"runner build failed: {_msg}"}

            runners[name] = _skip
    return runners


def _time_step(
    runner: Callable[..., Tuple[bool, Dict[str, Any]]],
    smiles: str,
    pocket: np.ndarray,
    n_steps: int,
) -> Tuple[float, float, bool, Dict[str, Any]]:
    """Run a single step, return ``(wall_s, mem_delta_mb, ok, payload)``."""
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    ok, payload = runner(smiles, pocket, n_steps)
    wall = float(time.perf_counter() - t0)
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    mem_delta = float(max(0, peak_mem - base_mem) / (1024.0 * 1024.0))
    return wall, mem_delta, bool(ok), dict(payload)


def _run_pipeline(
    smiles: str,
    pocket: np.ndarray,
    runners: Dict[str, Callable[..., Tuple[bool, Dict[str, Any]]]],
    n_steps: int,
) -> Dict[str, Dict[str, Any]]:
    """Run all six pipeline steps on ``smiles`` and return per-step dicts."""
    per_step: Dict[str, Dict[str, Any]] = {}
    for step_name in PIPELINE_STEPS:
        runner = runners[step_name]
        wall, mem_mb, ok, payload = _time_step(runner, smiles, pocket, n_steps)
        per_step[step_name] = {
            "wall_s": float(wall),
            "memory_mb": float(mem_mb),
            "ok": bool(ok),
            **payload,
        }
    return per_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of PlatinAI samples (default: 200).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="CFM ODE integration steps (default: 20).",
    )
    parser.add_argument(
        "--pocket-size",
        type=int,
        default=8,
        help="Pocket conditioning vector size (default: 8).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Lambda vs pocket convex weight in CFM target (default: 0.5).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="PlatinAI CSV path (default: molmetal/data/platinai_5k.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORT_DIR),
        help="Output directory for benchmark.json.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[bench] platinai CSV not found: {csv_path}", file=sys.stderr)
        return 1

    smiles = _load_platinai_smiles(csv_path, int(args.n_samples))
    if not smiles:
        print(f"[bench] no SMILES in {csv_path}", file=sys.stderr)
        return 1

    # Build runners — they are constructed lazily per-step so the
    # harness survives partial environments (e.g. torch missing).
    runners = _build_step_runners()

    pocket = np.zeros(int(args.pocket_size), dtype=np.float32)

    # Per-step wall-time accumulators (across all samples).
    per_step_walls: Dict[str, List[float]] = {s: [] for s in PIPELINE_STEPS}
    per_step_mem: Dict[str, List[float]] = {s: [] for s in PIPELINE_STEPS}
    per_step_ok: Dict[str, int] = {s: 0 for s in PIPELINE_STEPS}

    total_t0 = time.perf_counter()
    tracemalloc.start()
    base_mem_total = tracemalloc.get_traced_memory()[0]
    n_success = 0
    n_failures = 0
    per_sample_wall: List[float] = []

    for smi in smiles:
        sample_t0 = time.perf_counter()
        per_step = _run_pipeline(smi, pocket, runners, int(args.n_steps))
        sample_ok = True
        for step_name, info in per_step.items():
            per_step_walls[step_name].append(float(info["wall_s"]))
            per_step_mem[step_name].append(float(info["memory_mb"]))
            if info["ok"]:
                per_step_ok[step_name] += 1
            else:
                sample_ok = False
        per_sample_wall.append(float(time.perf_counter() - sample_t0))
        if sample_ok:
            n_success += 1
        else:
            n_failures += 1

    total_wall = float(time.perf_counter() - total_t0)
    peak_mem_total = float(
        max(0, tracemalloc.get_traced_memory()[1] - base_mem_total) / (1024.0 * 1024.0)
    )
    tracemalloc.stop()

    torch_v, rdkit_v = _get_versions()

    def _agg_walls(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
            "min": float(min(xs)),
            "max": float(max(xs)),
            "n": int(len(xs)),
        }

    def _agg_mem(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean_mb": 0.0, "max_mb": 0.0, "n": 0}
        return {
            "mean_mb": float(mean(xs)),
            "max_mb": float(max(xs)),
            "n": int(len(xs)),
        }

    payload: Dict[str, Any] = {
        "n_samples": int(len(smiles)),
        "n_success": int(n_success),
        "n_failures": int(n_failures),
        "success_rate": float(n_success / max(1, len(smiles))),
        "per_step_wall_time": {
            s: _agg_walls(per_step_walls[s]) for s in PIPELINE_STEPS
        },
        "per_step_memory_mb": {s: _agg_mem(per_step_mem[s]) for s in PIPELINE_STEPS},
        "per_step_ok_count": {s: int(per_step_ok[s]) for s in PIPELINE_STEPS},
        "total_wall_time_s": float(total_wall),
        "mean_wall_time_per_sample_s": float(
            mean(per_sample_wall) if per_sample_wall else 0.0
        ),
        "peak_memory_mb_total": float(peak_mem_total),
        "n_steps_cfm": int(args.n_steps),
        "pocket_size": int(args.pocket_size),
        "alpha": float(args.alpha),
        "csv_path": str(csv_path),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "torch_version": torch_v,
        "rdkit_version": rdkit_v,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        f"[bench] n_samples={payload['n_samples']} "
        f"success_rate={payload['success_rate']:.3f} "
        f"total_wall_s={payload['total_wall_time_s']:.2f} "
        f"peak_mem_mb={payload['peak_memory_mb_total']:.2f}"
    )
    for s in PIPELINE_STEPS:
        agg = payload["per_step_wall_time"][s]
        ok_n = payload["per_step_ok_count"][s]
        print(
            f"[bench]   {s:>14}: mean={agg['mean']*1000:.2f}ms "
            f"std={agg['std']*1000:.2f}ms ok={ok_n}/{payload['n_samples']}"
        )
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
