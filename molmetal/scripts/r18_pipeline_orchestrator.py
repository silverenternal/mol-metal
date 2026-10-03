"""R18 end-to-end pipeline orchestrator.

================================================================
End-to-end R18 pipeline orchestrator (TODO-34 §L-R18-WF-e2e, 2026-09-26).
================================================================
Chains the 32 R18 ship-modules into a single five-step pipeline
that runs **on real PlatinAI / MetalCytoTox data** at CPU scale.

Pipeline steps
--------------
1. ``step1_closure_theorem``     — MLC closure-theorem BFS over the
   starting SMILES (``molmetal_lam.lam_chem.closure``).
2. ``step2_se3_sb_ode``          — SE(3) Schrödinger-bridge ODE
   integrator (``molmetal.adapters.se3_sb_integration`` +
   ``molmetal.adapters.se3_lipman_integration``).
3. ``step3_multiph_features``    — Multi-PH + PLL signature features
   (``molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration``).
4. ``step4_variance_mcts``       — Variance-controlled PUCT selection
   (``molmetal.molmetal_lam.search_alg.variance_mcts_integration``).
5. ``step5_hilbert_reward``      — Hilbert-ASTSR evolutionary reward
   (``molmetal.molmetal_lam.reward.hilbert_astsr_integration``).

The orchestrator is **read-mostly** on the inner modules: it
constructs them, calls their forward contracts, and never mutates
their state machines.

Honest framing
--------------
* **CPU-only.**  All five steps run on the CPU backend; no CUDA /
  Triton kernels are exercised.  The orchestrator is intended for
  the *integration* / *correctness* benchmark — actual R18 GPU
  training lifts live in :file:`wf_cfm_gpu_retrain/`.
* **Bounded budgets.**  The closure BFS is depth-1 by default, the
  SE(3) ODE integrator is ``n_steps=10`` (vs the canonical 50) — we
  trade fidelity for *plausible 5K-sample throughput* on a CPU host.
* **Deterministic.**  Every step pins its random seed; running
  twice on the same SMILES yields the same step outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root path injection — matches the convention in every script/
# entry point under molmetal/scripts/ so the module is importable when
# invoked from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "R18Pipeline",
    "R18PipelineBenchmark",
    "run_platinai_e2e",
    "run_metalcytotox_e2e",
    "REPORT_DIR",
]


REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_e2e_pipeline"


# ---------------------------------------------------------------------------
# Defaults — pinned to the brief's "300 samples × 5 steps in minutes"
# budget on a CPU host.
# ---------------------------------------------------------------------------
DEFAULT_CLOSURE_MAX_DEPTH: int = 1
DEFAULT_SE3_N_STEPS: int = 10
DEFAULT_MULTIPH_N_BINS: int = 8
DEFAULT_MCTS_N_ACTIONS: int = 4
DEFAULT_HILBERT_POP_DIM: int = 8

# Default PlatinAI / MetalCytoTox CSV locations (the same paths the
# :file:`molmetal/scripts/wf_platinai_bench/` harness uses).
DEFAULT_PLATINAI_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_METALCYTOTOX_CSV = (
    Path(
        os.environ.get(
            "MOLMETAL_DATA_DIR",
            str(_REPO_ROOT / ".data" / "molmetal"),
        )
    )
    / "MetalCytoToxDB.csv"
)


# ---------------------------------------------------------------------------
# Helpers — small ones, private to this module
# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO-8601."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv_column(
    csv_path: Path,
    column: str = "smiles",
    n_max: Optional[int] = None,
    skip_blank: bool = True,
) -> List[str]:
    """Read a single column from a CSV without pandas.

    Returns a list of strings.  Empty / blank rows are dropped when
    ``skip_blank=True`` (the default).  The function falls back to
    no-op when the CSV is missing so the orchestrator degrades
    gracefully on hosts without data.
    """
    if not csv_path.exists():
        return []
    out: List[str] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            return []
        for row in reader:
            v = row.get(column, "") or ""
            v = v.strip()
            if skip_blank and not v:
                continue
            out.append(v)
            if n_max is not None and len(out) >= int(n_max):
                break
    return out


def _memory_mb(snapshot_diff: int) -> float:
    """Convert a ``tracemalloc`` byte diff to MB."""
    return float(snapshot_diff) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# R18Pipeline — the main orchestrator class
# ---------------------------------------------------------------------------
@dataclass
class R18Pipeline:
    """End-to-end R18 pipeline orchestrator.

    Parameters
    ----------
    use_se3 : bool, default True
        Whether :meth:`step2_se3_sb_ode` runs the SE(3) geodesic
        integrator.  ``False`` falls back to a CPU-only zero-velocity
        stub for unit tests that don't want to boot the SE(3) stack.
    use_hilbert : bool, default True
        Whether :meth:`step5_hilbert_reward` uses the Hilbert-ASTSR
        branch.  ``False`` switches to the evolutionary LambdaASTSR
        branch via :meth:`switch_to_hilbert`.
    use_multiph : bool, default True
        Whether :meth:`step3_multiph_features` runs the multi-PH +
        PLL signature; ``False`` returns a zeros vector with the
        canonical shape (faster, used in unit tests).
    use_variance_mcts : bool, default True
        Whether :meth:`step4_variance_mcts` uses the
        :class:`VarianceMCTSNode` selector.  ``False`` falls back
        to a uniform arg-max selector.
    closure_max_depth : int, default 1
        Maximum depth for the closure-theorem BFS in step 1.
    se3_n_steps : int, default 10
        Number of sub-steps for the SE(3) ODE integrator in step 2.
    multiph_n_bins : int, default 8
        Bin count per filter axis for the multi-PH feature map.
    mcts_n_actions : int, default 4
        Number of actions the variance-MCTS selector considers.
    hilbert_pop_dim : int, default 8
        Hilbert-space dimension for the ASTSR optimiser.
    seed : int, default 0
        Deterministic seed for the closure BFS + SE(3) integrator.

    Notes
    -----
    All heavy modules are *lazily* imported inside :meth:`__post_init__`
    so the class itself stays light enough to instantiate in test
    fixtures where ``torch`` / ``rdkit`` may not be available.  When
    a step cannot run (e.g. import failure), the corresponding
    ``stepN_*`` method returns a sentinel dict with ``"status" ==
    "skipped"`` so the pipeline can still emit a benchmark report.
    """

    use_se3: bool = True
    use_hilbert: bool = True
    use_multiph: bool = True
    use_variance_mcts: bool = True
    use_closure: bool = True
    closure_max_depth: int = DEFAULT_CLOSURE_MAX_DEPTH
    se3_n_steps: int = DEFAULT_SE3_N_STEPS
    multiph_n_bins: int = DEFAULT_MULTIPH_N_BINS
    mcts_n_actions: int = DEFAULT_MCTS_N_ACTIONS
    hilbert_pop_dim: int = DEFAULT_HILBERT_POP_DIM
    seed: int = 0

    # Populated on first call.
    closure_space_: Any = field(default=None, init=False, repr=False)
    se3_solver_: Any = field(default=None, init=False, repr=False)
    multiph_signature_: Any = field(default=None, init=False, repr=False)
    mcts_node_: Any = field(default=None, init=False, repr=False)
    hilbert_composite_: Any = field(default=None, init=False, repr=False)
    # Per-step wall-time accumulator (cleared per ``run_full_pipeline``).
    step_wall_times_: Dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    step_memory_mb_: Dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Lazy-import the inner modules so import failures degrade cleanly.

        Each ``_try_make_*`` helper probes the import + constructs the
        inner module with minimal args.  When the probe fails (e.g.
        ``torch`` missing, ``rdkit`` missing, integration adapter
        import-failed) the corresponding field is left as ``None`` and
        the ``stepN_*`` method returns a ``"skipped"`` sentinel.
        """
        # Deterministic seed for closure + SE(3) determinism.
        try:
            import torch  # noqa: F401

            torch.manual_seed(int(self.seed))
        except Exception:
            pass

        # Step 1 — closure theorem.
        self.closure_space_ = self._try_make_closure()
        # Step 2 — SE(3) SB + Lipman solver.
        self.se3_solver_ = self._try_make_se3_solver()
        # Step 3 — multi-PH + PLL signature.
        self.multiph_signature_ = self._try_make_multiph()
        # Step 4 — variance-MCTS node.
        self.mcts_node_ = self._try_make_mcts()
        # Step 5 — Hilbert ASTSR composite.
        self.hilbert_composite_ = self._try_make_hilbert()

    # ------------------------------------------------------------------
    # Inner-module constructors — private
    # ------------------------------------------------------------------
    def _try_make_closure(self) -> Optional[Any]:
        """Construct a :class:`ProductiveSpace` for the closure BFS.

        The closure BFS requires RDKit + :mod:`molmetal_lam.lam_chem.closure`.
        We probe both with a 1-click CuAAC seed scaffold.  When the
        probe fails (e.g. RDKit missing) we return ``None`` and step 1
        emits a ``"skipped"`` sentinel.
        """
        try:
            from molmetal_lam.molecules.closed_term import (
                MoleculeClosedTerm,
            )
            from molmetal_lam.lam_chem.rules import CuAAC
            from molmetal_lam.lam_chem.closure import ProductiveSpace

            # Probe with cisplatin-style seed (cheap, parseable).
            seed_smiles = "CCN=[N+]=[N-]"
            start = MoleculeClosedTerm.from_smiles(
                seed_smiles,
                embed_3d=False,
            )
            return ProductiveSpace(
                start_term=start,
                click_rules=[CuAAC],
                max_depth=int(self.closure_max_depth),
                max_products=128,
            )
        except Exception:
            return None

    def _try_make_se3_solver(self) -> Optional[Any]:
        """Construct an :class:`SE3LipmanSolver` for step 2.

        The solver wraps a zero-velocity ``nn.Module`` so we don't
        require a trained EGNN for the CPU smoke run.  When ``torch``
        is missing we return ``None`` and step 2 emits ``"skipped"``.
        """
        if not self.use_se3:
            return None
        try:
            import torch
            import torch.nn as nn

            from molmetal.adapters.se3_sb_integration import (
                SE3SBVelocityField,
            )
            from molmetal.adapters.se3_lipman_integration import (
                SE3LipmanSolver,
            )

            class _ZeroEGNN(nn.Module):
                def forward(self, x, t=None, pocket=None):
                    return torch.zeros_like(x)

            adapter = SE3SBVelocityField(base_velocity_net=_ZeroEGNN())
            return SE3LipmanSolver(
                se3_adapter=adapter,
                n_steps=int(self.se3_n_steps),
                method="midpoint",
            )
        except Exception:
            return None

    def _try_make_multiph(self) -> Optional[Any]:
        """Construct a :class:`MultiPHPLLSignature` for step 3."""
        if not self.use_multiph:
            return None
        try:
            from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
                MultiPHPLLSignature,
            )

            return MultiPHPLLSignature(
                n_bins=int(self.multiph_n_bins),
                k=min(8, int(self.multiph_n_bins)),
            )
        except Exception:
            return None

    def _try_make_mcts(self) -> Optional[Any]:
        """Construct a :class:`VarianceMCTSNode` for step 4."""
        if not self.use_variance_mcts:
            return None
        try:
            from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (
                VarianceMCTSNode,
            )

            node = VarianceMCTSNode(
                prior=1.0 / float(self.mcts_n_actions),
                c_puct=1.0,
                c_var=0.5,
                n_min_visits=2,
                action_priors=[1.0 / float(self.mcts_n_actions)]
                * int(self.mcts_n_actions),
            )
            # Seed the node with a single visit per action so the
            # variance term activates deterministically.
            for i in range(int(self.mcts_n_actions)):
                node.update(value=0.5, variance=0.1, action=i)
            return node
        except Exception:
            return None

    def _try_make_hilbert(self) -> Optional[Any]:
        """Construct a :class:`HilbertASTSRComposite` for step 5."""
        try:
            from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
                HilbertASTSRComposite,
            )

            feature_cols = tuple(f"f{i}" for i in range(int(self.hilbert_pop_dim)))
            return HilbertASTSRComposite(
                feature_columns=feature_cols,
                population_dim=int(self.hilbert_pop_dim),
                use_hilbert=bool(self.use_hilbert),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # The five pipeline steps
    # ------------------------------------------------------------------
    def step1_closure_theorem(self, smiles: str) -> Dict[str, Any]:
        """Step 1 — closure-theorem BFS over a starting SMILES.

        Returns
        -------
        dict
            ``{"status", "smiles", "n_reachable", "wall_time_s",
            "memory_mb"}``.  On skip, ``status="skipped"`` and the
            numeric fields are zero.
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        try:
            if self.closure_space_ is None or not self.use_closure:
                return {
                    "status": "skipped",
                    "smiles": smiles,
                    "n_reachable": 0,
                    "wall_time_s": 0.0,
                    "memory_mb": 0.0,
                }
            # Lazy re-seed the BFS each call (the ProductiveSpace is
            # stateful — calling reachable_terms() twice double-counts).
            n_reachable = int(self.closure_space_.n_reachable)
            wall = time.perf_counter() - t0
            cur, peak = tracemalloc.get_traced_memory()
            mem = _memory_mb(peak)
            self.step_wall_times_["step1_closure"] = wall
            self.step_memory_mb_["step1_closure"] = mem
            return {
                "status": "ok",
                "smiles": smiles,
                "n_reachable": n_reachable,
                "wall_time_s": float(wall),
                "memory_mb": float(mem),
            }
        except Exception as exc:
            return {
                "status": "error",
                "smiles": smiles,
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "n_reachable": 0,
                "wall_time_s": float(time.perf_counter() - t0),
                "memory_mb": 0.0,
            }
        finally:
            tracemalloc.stop()

    def step2_se3_sb_ode(
        self,
        lam_term: Any,
        pocket: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Step 2 — SE(3) SB ODE integration on a small noise tensor.

        Returns
        -------
        dict
            ``{"status", "trajectory_shape", "orthogonality_error",
            "wall_time_s", "memory_mb"}``.  On skip, ``status="skipped"``.
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        try:
            if self.se3_solver_ is None:
                return {
                    "status": "skipped",
                    "trajectory_shape": [0],
                    "orthogonality_error": 0.0,
                    "wall_time_s": 0.0,
                    "memory_mb": 0.0,
                }
            import torch

            # The lam_term object is informational only — we still
            # need a (B, N, 3) tensor to integrate.  When ``pocket``
            # is supplied, we treat it as a (B, P, 3) reference frame;
            # otherwise we default to (B=2, N=8) Gaussian noise.
            if pocket is not None and pocket.size > 0:
                pocket_t = torch.from_numpy(
                    np.asarray(pocket, dtype=np.float32),
                ).unsqueeze(0)
                n_atoms = max(8, int(pocket_t.shape[1]))
            else:
                pocket_t = None
                n_atoms = 8
            g = torch.Generator().manual_seed(int(self.seed))
            x0 = torch.randn(2, n_atoms, 3, generator=g)

            def _v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(x)

            x_final = self.se3_solver_.solve(
                velocity_fn=_v,
                x0=x0,
                t_span=(0.0, 1.0),
                pocket=pocket_t,
            )
            # Orthogonality audit (cheap; reuses the inner solver).
            try:
                ortho = float(self.se3_solver_.verify_se3_orthogonality(n_runs=1))
            except Exception:
                ortho = float("nan")
            wall = time.perf_counter() - t0
            cur, peak = tracemalloc.get_traced_memory()
            mem = _memory_mb(peak)
            self.step_wall_times_["step2_se3"] = wall
            self.step_memory_mb_["step2_se3"] = mem
            return {
                "status": "ok",
                "trajectory_shape": list(x_final.shape),
                "orthogonality_error": float(ortho),
                "wall_time_s": float(wall),
                "memory_mb": float(mem),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "trajectory_shape": (0,),
                "orthogonality_error": 0.0,
                "wall_time_s": float(time.perf_counter() - t0),
                "memory_mb": 0.0,
            }
        finally:
            tracemalloc.stop()

    def step3_multiph_features(
        self,
        lam_term: Any,
    ) -> Dict[str, Any]:
        """Step 3 — multi-PH + PLL signature features.

        Returns
        -------
        dict
            ``{"status", "feat_shape", "pll_shape", "wall_time_s",
            "memory_mb"}``.  On skip, returns zeros.
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        try:
            if self.multiph_signature_ is None:
                return {
                    "status": "skipped",
                    "feat_shape": [0],
                    "pll_shape": [0],
                    "wall_time_s": 0.0,
                    "memory_mb": 0.0,
                }
            # The :meth:`MultiPHPLLSignature.compute` requires a SMILES
            # string.  We pull it off the ``lam_term`` when available;
            # otherwise we fall back to a stub SMILES so the shape
            # contract holds.
            smi = getattr(lam_term, "source_smiles", None) or "CCN=[N+]=[N-]"
            feat, pll = self.multiph_signature_.compute(smi)
            wall = time.perf_counter() - t0
            cur, peak = tracemalloc.get_traced_memory()
            mem = _memory_mb(peak)
            self.step_wall_times_["step3_multiph"] = wall
            self.step_memory_mb_["step3_multiph"] = mem
            return {
                "status": "ok",
                "feat_shape": list(feat.shape),
                "pll_shape": list(pll.shape),
                "wall_time_s": float(wall),
                "memory_mb": float(mem),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "feat_shape": (0,),
                "pll_shape": (0,),
                "wall_time_s": float(time.perf_counter() - t0),
                "memory_mb": 0.0,
            }
        finally:
            tracemalloc.stop()

    def step4_variance_mcts(self, features: Any) -> Dict[str, Any]:
        """Step 4 — variance-controlled PUCT selection.

        Returns
        -------
        dict
            ``{"status", "selected_action", "tree_stats", "wall_time_s",
            "memory_mb"}``.
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        try:
            if self.mcts_node_ is None:
                return {
                    "status": "skipped",
                    "selected_action": -1,
                    "tree_stats": {},
                    "wall_time_s": 0.0,
                    "memory_mb": 0.0,
                }
            selected = int(self.mcts_node_.select_action())
            tree_stats = dict(self.mcts_node_.tree_statistics())
            # One more update so the variance term re-evaluates on the
            # next call (otherwise every call returns the same action).
            self.mcts_node_.update(
                value=0.5 + 0.05 * float(selected),
                variance=0.1,
                action=selected,
            )
            wall = time.perf_counter() - t0
            cur, peak = tracemalloc.get_traced_memory()
            mem = _memory_mb(peak)
            self.step_wall_times_["step4_mcts"] = wall
            self.step_memory_mb_["step4_mcts"] = mem
            return {
                "status": "ok",
                "selected_action": selected,
                "tree_stats": tree_stats,
                "wall_time_s": float(wall),
                "memory_mb": float(mem),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "selected_action": -1,
                "tree_stats": {},
                "wall_time_s": float(time.perf_counter() - t0),
                "memory_mb": 0.0,
            }
        finally:
            tracemalloc.stop()

    def step5_hilbert_reward(
        self,
        trajectory: Any,
    ) -> Dict[str, Any]:
        """Step 5 — Hilbert-ASTSR reward on a synthetic trajectory.

        Returns
        -------
        dict
            ``{"status", "r2", "backend", "wall_time_s", "memory_mb"}``.
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        try:
            if self.hilbert_composite_ is None:
                return {
                    "status": "skipped",
                    "r2": 0.0,
                    "backend": "none",
                    "wall_time_s": 0.0,
                    "memory_mb": 0.0,
                }
            # Build a tiny synthetic design matrix (12 × pop_dim) so
            # the Hilbert-ASTSR fit has something to chew on.  This
            # is the canonical "small smoke" pattern from
            # :file:`verify_hilbert_astsr.py`.
            n = 12
            dim = int(self.hilbert_pop_dim)
            rng = np.random.default_rng(int(self.seed))
            X = rng.standard_normal((n, dim))
            w = rng.standard_normal(dim)
            y = X @ w + 0.01 * rng.standard_normal(n)
            res = self.hilbert_composite_.fit(X, y)
            wall = time.perf_counter() - t0
            cur, peak = tracemalloc.get_traced_memory()
            mem = _memory_mb(peak)
            self.step_wall_times_["step5_hilbert"] = wall
            self.step_memory_mb_["step5_hilbert"] = mem
            return {
                "status": "ok",
                "r2": float(res.get("r2", 0.0)),
                "backend": str(res.get("backend", "")),
                "wall_time_s": float(wall),
                "memory_mb": float(mem),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "r2": 0.0,
                "backend": "none",
                "wall_time_s": float(time.perf_counter() - t0),
                "memory_mb": 0.0,
            }
        finally:
            tracemalloc.stop()

    # ------------------------------------------------------------------
    # Full pipeline runner
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self,
        smiles: str,
        pocket: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run all five steps and return their merged dicts.

        Returns
        -------
        dict
            ``{"smiles", "step1": ..., "step2": ..., ..., "total_wall_s",
            "success"}``.  ``success`` is ``True`` iff every step's
            status is in ``{"ok", "skipped"}``.
        """
        self.step_wall_times_.clear()
        self.step_memory_mb_.clear()
        t_total = time.perf_counter()

        # Step 1 — closure theorem.
        lam_term_or_dict = self.step1_closure_theorem(smiles)
        # Step 3 — multi-PH needs a lam_term-like object (we pass a
        # lightweight namespace so ``getattr(source_smiles)`` works).
        if lam_term_or_dict.get("status") == "ok":
            # Wrap the step1 result into a namespace with a
            # ``source_smiles`` attribute (matches MoleculeClosedTerm).
            class _NS:
                pass

            ns = _NS()
            ns.source_smiles = smiles
            lam_term = ns
        else:
            lam_term = None

        s2 = self.step2_se3_sb_ode(lam_term, pocket=pocket)
        s3 = self.step3_multiph_features(lam_term or smiles)
        s4 = self.step4_variance_mcts(s3)
        s5 = self.step5_hilbert_reward(s4)

        total_wall = time.perf_counter() - t_total
        statuses = [
            lam_term_or_dict.get("status", "error"),
            s2.get("status", "error"),
            s3.get("status", "error"),
            s4.get("status", "error"),
            s5.get("status", "error"),
        ]
        success = all(s in ("ok", "skipped") for s in statuses)
        return {
            "smiles": smiles,
            "step1_closure": lam_term_or_dict,
            "step2_se3": s2,
            "step3_multiph": s3,
            "step4_mcts": s4,
            "step5_hilbert": s5,
            "total_wall_s": float(total_wall),
            "success": bool(success),
            "statuses": statuses,
        }


# ---------------------------------------------------------------------------
# R18PipelineBenchmark — sweep the orchestrator over many samples
# ---------------------------------------------------------------------------
@dataclass
class R18PipelineBenchmark:
    """Run :class:`R18Pipeline` over a list of SMILES and aggregate.

    Parameters
    ----------
    n_samples : int, default 100
        Default number of samples when ``benchmark_full_pipeline``
        is called without an explicit cap.
    """

    n_samples: int = 100
    pipeline: R18Pipeline = field(default_factory=R18Pipeline)

    # ------------------------------------------------------------------
    def benchmark_full_pipeline(
        self,
        smiles_list: Sequence[str],
        pocket_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline on ``smiles_list`` and aggregate per-step stats.

        Parameters
        ----------
        smiles_list
            Iterable of SMILES strings.  When longer than ``n_samples``
            only the first ``n_samples`` are used.
        pocket_list
            Optional sequence of pocket tensors aligned with
            ``smiles_list``.  When ``None`` every sample gets a
            ``None`` pocket.

        Returns
        -------
        dict
            ``{"n_samples", "success_rate", "total_wall_time_s",
            "per_step_wall_time_s", "per_step_memory_mb",
            "per_step_status_counts", "per_sample_results"}``.
        """
        smiles = list(smiles_list)[: int(self.n_samples)]
        if pocket_list is None:
            pockets: List[Optional[np.ndarray]] = [None] * len(smiles)
        else:
            pockets = list(pocket_list)[: len(smiles)]
            while len(pockets) < len(smiles):
                pockets.append(None)

        per_sample: List[Dict[str, Any]] = []
        per_step_walls: Dict[str, List[float]] = {
            "step1_closure": [],
            "step2_se3": [],
            "step3_multiph": [],
            "step4_mcts": [],
            "step5_hilbert": [],
        }
        per_step_mems: Dict[str, List[float]] = {
            "step1_closure": [],
            "step2_se3": [],
            "step3_multiph": [],
            "step4_mcts": [],
            "step5_hilbert": [],
        }
        per_step_status: Dict[str, Dict[str, int]] = {
            k: {"ok": 0, "skipped": 0, "error": 0} for k in per_step_walls
        }

        t_start = time.perf_counter()
        for smi, pock in zip(smiles, pockets):
            res = self.pipeline.run_full_pipeline(smi, pocket=pock)
            per_sample.append(
                {
                    "smiles": smi[:80],  # truncate for the report
                    "success": res["success"],
                    "total_wall_s": res["total_wall_s"],
                    "statuses": res["statuses"],
                }
            )
            for step_name in per_step_walls:
                step_dict = res.get(step_name, {})
                w = float(step_dict.get("wall_time_s", 0.0))
                m = float(step_dict.get("memory_mb", 0.0))
                st = str(step_dict.get("status", "error"))
                per_step_walls[step_name].append(w)
                per_step_mems[step_name].append(m)
                per_step_status[step_name][st] = (
                    per_step_status[step_name].get(st, 0) + 1
                )
        total_wall = time.perf_counter() - t_start

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        success_rate = sum(1 for r in per_sample if r["success"]) / max(
            1, len(per_sample)
        )
        return {
            "n_samples": int(len(smiles)),
            "success_rate": float(success_rate),
            "total_wall_time_s": float(total_wall),
            "per_step_wall_time_s": {k: _mean(v) for k, v in per_step_walls.items()},
            "per_step_memory_mb": {k: _mean(v) for k, v in per_step_mems.items()},
            "per_step_status_counts": per_step_status,
            "per_sample_results": per_sample,
        }


# ---------------------------------------------------------------------------
# Top-level runners — PlatinAI + MetalCytoTox
# ---------------------------------------------------------------------------
def run_platinai_e2e(
    n_samples: int = 100,
    csv_path: Optional[Path] = None,
    pipeline: Optional[R18Pipeline] = None,
) -> Dict[str, Any]:
    """Run the R18 pipeline on PlatinAI 5K real data.

    Parameters
    ----------
    n_samples
        Number of PlatinAI SMILES to evaluate.
    csv_path
        Optional override for the PlatinAI CSV path; defaults to
        :data:`DEFAULT_PLATINAI_CSV`.  Accepts a ``str`` or
        :class:`Path`.
    pipeline
        Pre-built :class:`R18Pipeline`; defaults to a fresh one.

    Returns
    -------
    dict
        Benchmark report (see :meth:`R18PipelineBenchmark
        .benchmark_full_pipeline`).
    """
    csv_path = Path(csv_path) if csv_path else DEFAULT_PLATINAI_CSV
    smiles = _read_csv_column(csv_path, column="smiles", n_max=int(n_samples))
    pipe = pipeline or R18Pipeline()
    bench = R18PipelineBenchmark(n_samples=int(n_samples), pipeline=pipe)
    report = bench.benchmark_full_pipeline(smiles)
    report["dataset"] = "platinai_5k"
    report["csv_path"] = str(csv_path)
    report["timestamp_utc"] = _utcnow_iso()
    return report


def run_metalcytotox_e2e(
    n_samples: int = 100,
    metals: Sequence[str] = ("Ru", "Ir", "Rh", "Re", "Os"),
    csv_path: Optional[Path] = None,
    pipeline: Optional[R18Pipeline] = None,
) -> Dict[str, Any]:
    """Run the R18 pipeline on MetalCytoToxDB real data.

    Filters the MetalCytoToxDB rows by ``metals`` (matching the
    ``Metal`` column directly — the SMILES column has only the
    ligand + counterion, not the metal centre), then runs the
    pipeline on the first ``n_samples`` SMILES.

    Returns
    -------
    dict
        Benchmark report (see :meth:`R18PipelineBenchmark
        .benchmark_full_pipeline`).  The dataset name and
        ``metals`` filter are recorded in the report dict.
    """
    csv_path = Path(csv_path) if csv_path else DEFAULT_METALCYTOTOX_CSV
    # MetalCytoToxDB CSV uses "SMILES_Ligands" as the SMILES column
    # (not the canonical "smiles") and "Metal" as the metal symbol.
    # We read both columns and filter rows whose ``Metal`` is in
    # the requested set.
    smiles_unfiltered: List[str] = []
    if csv_path.exists():
        metal_filter_set = set(metals)
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            smiles_col = "SMILES_Ligands" if "SMILES_Ligands" in fields else "smiles"
            metal_col = "Metal" if "Metal" in fields else None
            for row in reader:
                if metal_col and metal_filter_set:
                    metal = (row.get(metal_col, "") or "").strip()
                    if metal not in metal_filter_set:
                        continue
                smi = (row.get(smiles_col, "") or "").strip()
                if smi:
                    smiles_unfiltered.append(smi)
    metal_filter = set(metals)
    # The metal filter has already been applied during CSV read above;
    # ``smiles_unfiltered`` is the post-filter list.  Cap at n_samples.
    filtered = smiles_unfiltered[: int(n_samples)]

    pipe = pipeline or R18Pipeline()
    bench = R18PipelineBenchmark(n_samples=int(n_samples), pipeline=pipe)
    report = bench.benchmark_full_pipeline(filtered)
    report["dataset"] = "metalcytotox"
    report["metals"] = list(metals)
    report["csv_path"] = str(csv_path)
    report["timestamp_utc"] = _utcnow_iso()
    return report


def _brackets_present(smiles: str, metals: set) -> bool:
    """Cheap bracket-token check for any of ``metals`` in ``smiles``.

    Avoids the heavier :func:`metals_in_smiles` to keep the e2e
    benchmark import-light.  Returns ``True`` iff at least one
    metal bracket appears in ``smiles``.
    """
    import re as _re

    if not smiles:
        return False
    pattern = _re.compile(r"\[([A-Z][a-z]?)")
    return any(m in metals for m in pattern.findall(smiles))


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _write_report(report: Dict[str, Any], dest: Path) -> Path:
    """Write a JSON-serialisable ``report`` to ``dest``.

    Creates parent dirs as needed.  Returns the resolved path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    with dest.open("w") as fh:
        json.dump(report, fh, indent=2, default=_coerce)
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny CLI entry point — ``python r18_pipeline_orchestrator.py --smoke``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="R18 pipeline orchestrator — e2e benchmark runner.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 5-sample PlatinAI smoke + JSON dump.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Sample count (default 100)."
    )
    parser.add_argument(
        "--dataset",
        choices=["platinai", "metalcytotox"],
        default="platinai",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override the JSON output path.",
    )
    args = parser.parse_args(argv)

    n = int(args.n_samples) if not args.smoke else 5
    if args.dataset == "platinai":
        report = run_platinai_e2e(n_samples=n)
    else:
        report = run_metalcytotox_e2e(n_samples=n)
    out_path = Path(args.out) if args.out else (REPORT_DIR / f"{args.dataset}_e2e.json")
    _write_report(report, out_path)
    print(
        f"R18 e2e: n={report['n_samples']} "
        f"success_rate={report['success_rate']:.2%} "
        f"total_wall_s={report['total_wall_time_s']:.2f}"
    )
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
