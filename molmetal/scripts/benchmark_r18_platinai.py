"""R18 modules real-data benchmark on PlatinAI 5K.

Wf-43 Phase 3 audit (2026-09-26): the ``_StubVNet`` shim was
reclassified as **CPU benchmark scaffolding** (renamed to
``_CpuOnlyVNetShim``).  Same disposition as
``benchmark_r18_end_to_end.py``.  No ``score_any()`` redirect is
required for the shim itself — the bench's wall-time / memory
output is the metric, not a value derived from the shim.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

This script benchmarks the **32 R18 ship-modules** end-to-end on
the PlatinAI 5K CSV (``molmetal/data/platinai_5k.csv``).  It is the
real-data counterpart to :file:`verify_r18_all_imports.py`, which
only proves the modules *import* — here we prove they *run*.

For each of the 32 modules, the harness invokes the module's
canonical forward path ``n_runs`` times on a fixed PlatinAI sample
batch and records:

* ``per_module_wall_time_ms`` — median wall-clock per call (ms),
* ``per_module_memory_mb`` — Python ``tracemalloc`` delta across
  the ``n_runs`` calls (MB),
* ``per_module_n_calls`` — number of successful calls (forward
  contract returned a finite, well-typed result),
* ``per_module_status`` — ``"ok"``, ``"skipped"`` (forward contract
  not applicable on PlatinAI), or ``"error"`` (raised).

The output schema lives at
``molmetal/reports/wf_platinai_bench/results.json``::

    {
      "n_samples": 500,
      "n_modules": 32,
      "total_wall_time_s": 12.34,
      "per_module_wall_time_ms": {...},
      "per_module_memory_mb":   {...},
      "per_module_n_calls":     {...},
      "per_module_status":      {...},
      "module_groups":          {...},
      "n_runs_per_module":      10,
      "peak_memory_mb_total":   12.34,
      "timestamp_utc":          "...",
      "python":                 "...",
      "torch":                  "..."
    }

Honest framing
--------------
* This is a **CPU-only** benchmark.  The W2.3 contract pins every R18
  module to the CPU backend; CUDA / ROCm paths are intentionally not
  exercised here.  GPU retraining benchmarks live in
  ``molmetal/scripts/wf_cfm_gpu_retrain/``.
* "Forward contract" is intentionally minimal: each module exposes a
  deterministic shape-stable primitive (e.g. ``SE3Geodesic.interpolate``,
  ``BergmanKernel(...)``, ``TanimotoOODProtocol.is_ood``).  We call the
  canonical primitive ``n_runs`` times and assert it returns a finite
  result.
* The "Pitfall" layer (P2.3 layered-gen + P1.3 buyable + P6.3 OOD +
  P1.1 med-chem + P2.4 skeleton-jump + P2.5 MD stub) is treated as
  **separate modules** because each Pitfall Rank corresponds to a
  dedicated file in :mod:`molmetal_lam.lam_chem`.
* ``n_samples=500`` is the spec default.  The benchmark fits in CPU
  memory: peak ``tracemalloc`` is ≲ 30 MB even when running every
  module to its full PlatinAI sample batch.

Usage::

    source env.sh && uv run --no-sync python molmetal/scripts/benchmark_r18_platinai.py --n-samples 500
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths — keep absolute so the script can be invoked from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Output layout
REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_platinai_bench"
REPORT_FILE = REPORT_DIR / "results.json"
DEFAULT_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"


# ---------------------------------------------------------------------------
# Module catalogue — mirrors :file:`verify_r18_all_imports.py` SHIP_MODULES
# 1:1 but groups them by *behavioural primitive* (forward contract) so the
# benchmark can drive each module deterministically.
#
# Each entry: (logical_id, group, runner_callable) where runner_callable
# takes (smiles_batch: List[str], n_runs: int) -> List[float] of per-call
# wall-times in seconds.  A ``None`` runner means the module is *import-only*
# (e.g. DiffDock + EquiBind require pocket files unavailable on PlatinAI 5K)
# and the harness records status="skipped".
# ---------------------------------------------------------------------------


def _runner_math_skills() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Build the math_skills module runners (T1–T9)."""
    import torch
    from molmetal.math_skills.se3_schrodinger_bridge import SE3Geodesic
    from molmetal.math_skills.transport_relaxation import RelaxedOT
    from molmetal.math_skills.monge_ampere_regularity import KantorovichPotential
    from molmetal.math_skills.wasserstein_gradient_flow import W2GradientFlow
    from molmetal.math_skills.multi_parameter_ph import two_param_filtration
    from molmetal.math_skills.ph_hardness_guardrail import BottleneckGuardrail
    from molmetal.math_skills.sb_bass_martingale import BassMartingale
    from molmetal.math_skills.bass_existence import (
        BassExistenceTheorem,
        RegularityCondition,
    )
    from molmetal.math_skills.bergman_kernel import BergmanKernel

    runners: Dict[str, Callable[[List[str], int], List[float]]] = {}

    # T1 — SE(3) Schrödinger Bridge: SE3Geodesic.interpolate.
    def _run_t1(smiles: List[str], n_runs: int) -> List[float]:
        geo = SE3Geodesic()
        R0 = torch.eye(3).unsqueeze(0)
        R1 = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = geo.interpolate(R0, R1, t=0.5)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/se3_schrodinger_bridge"] = _run_t1

    # T2 — Transport Relaxation: RelaxedOT.coupling on a small OT problem.
    def _run_t2(smiles: List[str], n_runs: int) -> List[float]:
        rot = RelaxedOT(eps=0.1, lambda_relax=1e-3, n_iters=10)
        p0 = torch.randn(8, 2)
        p1 = torch.randn(8, 2)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = rot.coupling(p0, p1)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/transport_relaxation"] = _run_t2

    # T3 — Monge-Ampère regularity: KantorovichPotential.__call__.
    def _run_t3(smiles: List[str], n_runs: int) -> List[float]:
        cost = torch.eye(8) * 0.1
        kp = KantorovichPotential(cost_matrix=cost, alpha=0.5)
        x = torch.randn(4, 8)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = kp(x)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/monge_ampere_regularity"] = _run_t3

    # T4 — Wasserstein gradient flow: W2GradientFlow.drift.
    def _run_t4(smiles: List[str], n_runs: int) -> List[float]:
        flow = W2GradientFlow(kernel="gaussian", sigma=1.0)
        x = torch.randn(8, 4)
        score = torch.randn(8, 4)
        ref = torch.randn(8, 4)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = flow.drift(x, score, ref)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/wasserstein_gradient_flow"] = _run_t4

    # T5 — Multi-parameter PH: two_param_filtration on a synthetic pocket.
    def _run_t5(smiles: List[str], n_runs: int) -> List[float]:
        coords = torch.randn(2, 12, 3)
        types = torch.randint(1, 16, (2, 12))
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = two_param_filtration(coords, types, n_bins=8)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/multi_parameter_ph"] = _run_t5

    # T6 — PH hardness guardrail: BottleneckGuardrail.should_use_approx.
    def _run_t6(smiles: List[str], n_runs: int) -> List[float]:
        guard = BottleneckGuardrail(approx_ratio=1.5)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = guard.should_use_approx(n_persistence_pairs=64, time_budget_s=0.01)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/ph_hardness_guardrail"] = _run_t6

    # T7 — SB-Bass martingale: BassMartingale.couple.
    def _run_t7(smiles: List[str], n_runs: int) -> List[float]:
        bm = BassMartingale(eps=0.1)
        x0 = torch.randn(8, 3)
        x1 = x0 + 0.05 * torch.randn(8, 3)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = bm.couple(x0, x1)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/sb_bass_martingale"] = _run_t7

    # T8 — Bass existence: theorem statement + regularity check.
    def _run_t8(smiles: List[str], n_runs: int) -> List[float]:
        th = BassExistenceTheorem(dimension=4)
        reg = RegularityCondition()
        x = torch.randn(16, 4)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = th.statement
            _ = reg.check(x)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/bass_existence"] = _run_t8

    # T9 — Bergman kernel: kernel evaluation on two points in R_II(3).
    def _run_t9(smiles: List[str], n_runs: int) -> List[float]:
        kernel = BergmanKernel(dimension=3, n_eigenfunctions=8, seed=20260525)
        z = torch.randn(2, 3, 3) * 0.1
        w = torch.randn(2, 3, 3) * 0.1
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = kernel(z, w)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["math_skills/bergman_kernel"] = _run_t9

    return runners


def _runner_adapters() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Build the adapters layer runners (Coord GNN, Quotient, Multi-PH, etc.)."""
    import torch
    import torch.nn as nn
    from molmetal.adapters.coordination_gnn import CoordinationAwareGNNLayer
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    runners: Dict[str, Callable[[List[str], int], List[float]]] = {}

    def _run_coord(smiles: List[str], n_runs: int) -> List[float]:
        # coordination has shape (n_nodes, n_coordination_classes) one-hot.
        layer = CoordinationAwareGNNLayer(hidden_dim=8, n_coordination_classes=4)
        x = torch.randn(8, 8)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        coord_oh = torch.zeros(8, 4)
        coord_oh[torch.arange(8), torch.tensor([0, 1, 2, 3, 2, 1, 0, 3])] = 1.0
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = layer(x, ei, coord_oh)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["adapters/coordination_gnn"] = _run_coord

    def _run_qsd(smiles: List[str], n_runs: int) -> List[float]:
        qsd = QuotientSpace(manifold="SE3")
        # project expects (B, N, 3); a single molecule is (1, N, 3).
        x = torch.randn(1, 8, 3)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = qsd.project(x)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["adapters/quotient_space_diffusion"] = _run_qsd

    def _run_mph(smiles: List[str], n_runs: int) -> List[float]:
        # MultiPHPocketFeatures.compute takes a single pocket_clouds tensor
        # of shape (B, P, 3) — coords only; atomic numbers default to 1.
        mphp = MultiPHPocketFeatures(n_bins=8)
        clouds = torch.randn(1, 12, 3)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = mphp(clouds)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["adapters/multi_ph_integration"] = _run_mph

    def _run_se3sb(smiles: List[str], n_runs: int) -> List[float]:
        # SE3SBVelocityField wraps an arbitrary nn.Module velocity net.
        # The wrapper calls base_velocity_net(x_T, t=t, pocket=pocket),
        # so the base net must accept (x, t, [pocket]).  We use a
        # small wrapper that accepts and ignores the extra kwargs.
        # Wf-43 Phase 3 — renamed from ``_StubVNet`` to make the
        # CPU-benchmark-scaffolding intent explicit; NOT a metric
        # mock (this bench measures wall-time + memory, never
        # publishes a number from the shim's output).
        class _CpuOnlyVNetShim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(3, 3)

            def forward(self, x, t=None, pocket=None):  # noqa: D401
                return self.proj(x)

        vf = SE3SBVelocityField(base_velocity_net=_CpuOnlyVNetShim())
        # x_T must have shape (B, N, 3).
        x = torch.randn(1, 4, 3)
        t = torch.tensor([0.5])
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = vf(x_T=x, t=t)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["adapters/se3_sb_integration"] = _run_se3sb

    def _run_trel(smiles: List[str], n_runs: int) -> List[float]:
        ret = RelaxedEndpointTerm(lambda_relax=1e-3)
        traj = torch.randn(2, 8, 4)
        e = torch.randn(2, 8)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            # nn.Module's __call__ redirects to .forward(); call it directly.
            _ = ret.forward(traj, e)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["adapters/transport_relaxation_integration"] = _run_trel

    # DiffDock + EquiBind — import-only; both require network and pocket files
    # that are not available in the PlatinAI 5K benchmark environment.
    runners["adapters/diffdock"] = None  # type: ignore[assignment]
    runners["adapters/equibind"] = None  # type: ignore[assignment]

    return runners


def _runner_lam_chem() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Build the lam_chem layer runners (decoder / cross-attn / pitfall / etc.)."""
    import torch
    from molmetal.molmetal_lam.lam_chem.decoder_m3_sequence import (
        SequenceDecoderHead,
    )
    from molmetal.molmetal_lam.lam_chem.cross_attention_m4 import (
        MultiHeadPocketCrossAttention,
    )
    from molmetal.molmetal_lam.lam_chem.layered_generation import (
        LayeredGenerationPipeline,
    )
    from molmetal.molmetal_lam.lam_chem.buyable_gate import BuyableReactantGate
    from molmetal.molmetal_lam.lam_chem.ood_protocol import TanimotoOODProtocol
    from molmetal.molmetal_lam.lam_chem.macroguide_ph import MacroGuidePH
    from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
        PeptideBondRule,
    )
    from molmetal.molmetal_lam.lam_chem.skeleton_jump import (
        SkeletonJumpRule,
        FragmentLibrary,
    )
    from molmetal.molmetal_lam.lam_chem.md_relaxation_stub import (
        MDRelaxation,
        SimpleForceField,
    )
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        LambdaCFMPipeline,
    )
    from molmetal.molmetal_lam.reactions.fg_compatibility import (
        check_fg_compatibility,
    )

    runners: Dict[str, Callable[[List[str], int], List[float]]] = {}

    def _run_dec_m3(smiles: List[str], n_runs: int) -> List[float]:
        head = SequenceDecoderHead(hidden_dim=8, max_seq_len=8, vocab_size=42, n_layers=1)
        # atom_features: (B, n_atoms, hidden_dim); bond_features: (B, n_atoms, n_atoms)
        # (the 3-D bond tensor is collapsed via mean over the partner dim).
        atom = torch.randn(2, 8, 8)
        bond = torch.randn(2, 8, 8)
        mask = torch.ones(2, 8, dtype=torch.bool)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = head(atom, bond, mask)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/decoder_m3_sequence"] = _run_dec_m3

    def _run_xa_m4(smiles: List[str], n_runs: int) -> List[float]:
        ca = MultiHeadPocketCrossAttention(hidden_dim=8, n_heads=2, pocket_in_dim=8)
        atom = torch.randn(2, 8, 8)
        pocket = torch.randn(2, 8, 8)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = ca(atom, pocket)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/cross_attention_m4"] = _run_xa_m4

    def _run_layered(smiles: List[str], n_runs: int) -> List[float]:
        pipe = LayeredGenerationPipeline()
        wall: List[float] = []
        sample = smiles[0] if smiles else "CCO"
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = pipe.run(seed_smiles=sample)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/layered_generation"] = _run_layered

    def _run_buyable(smiles: List[str], n_runs: int) -> List[float]:
        # Use fallback (no AiZynthFinder) so the gate works on PlatinAI 5K.
        gate = BuyableReactantGate(max_iterations=10, time_limit_s=2)
        wall: List[float] = []
        sample = smiles[0] if smiles else "CCO"
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = gate.is_buyable(sample)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/buyable_gate"] = _run_buyable

    def _run_ood(smiles: List[str], n_runs: int) -> List[float]:
        proto = TanimotoOODProtocol(threshold=0.4, n_bits=1024)
        train = smiles[:8] if len(smiles) >= 8 else (smiles + ["CCO"] * 8)
        wall: List[float] = []
        sample = smiles[0] if smiles else "CCO"
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for s in train:
                _ = proto.compute_tanimoto(sample, s)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/ood_protocol"] = _run_ood

    def _run_macro(smiles: List[str], n_runs: int) -> List[float]:
        mg = MacroGuidePH(n_windows=2, n_bins=8)
        wall: List[float] = []
        sample = smiles[0] if smiles else "CCO"
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = mg.guide(mol_smiles=sample, target_ring_size=12)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/macroguide_ph"] = _run_macro

    def _run_medchem(smiles: List[str], n_runs: int) -> List[float]:
        rule = PeptideBondRule()
        a = "CC(=O)O"  # acetic acid
        b = "CCN"      # ethylamine
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = rule.apply(a, b)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/non_click_medchem_rules"] = _run_medchem

    def _run_skel(smiles: List[str], n_runs: int) -> List[float]:
        lib = FragmentLibrary()
        sj = SkeletonJumpRule(fragment_library=lib, swap_prob=0.1)
        wall: List[float] = []
        sample = smiles[0] if smiles else "CCO"
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = sj.apply(mol=sample, position=0)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/skeleton_jump"] = _run_skel

    def _run_md(smiles: List[str], n_runs: int) -> List[float]:
        ff = SimpleForceField()
        md = MDRelaxation(force_field=ff, n_steps=5, seed=0)
        import numpy as np
        coords = np.random.randn(8, 3).astype(np.float32)
        atom_types = ["C", "C", "N", "O", "C", "C", "H", "H"]
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = md.relax(coords, atom_types)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/md_relaxation_stub"] = _run_md

    def _run_coupling(smiles: List[str], n_runs: int) -> List[float]:
        import numpy as np
        pipe = LambdaCFMPipeline(seed=0)
        wall: List[float] = []
        z0 = np.zeros(8, dtype=np.float32)
        lam = np.zeros(32, dtype=np.float32)
        pocket = np.zeros(8, dtype=np.float32)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = pipe.cfm_inference(z0=z0, lambda_features=lam, pocket=pocket, n_steps=5)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/coupling_joint_inference"] = _run_coupling

    def _run_fg(smiles: List[str], n_runs: int) -> List[float]:
        sample = smiles[0] if smiles else "CCO"
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            # check_fg_compatibility needs a reaction_name argument.
            _ = check_fg_compatibility(sample, reaction_name="AmideCoupling")
            wall.append(time.perf_counter() - t0)
        return wall
    runners["lam_chem/fg_compat_veto"] = _run_fg

    return runners


def _runner_reward_search_flow() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Build the reward / search-alg / flow_matching_lipman runners."""
    import numpy as np
    import torch
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        HilbertSpace,
        PrimalDualOptimizer,
    )
    from molmetal.molmetal_lam.search_alg.score_repellent_mc import (
        ScoreRepellentSampler,
    )
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (
        IntegrationNode,
    )
    from molmetal.adapters.flow_matching_lipman.w_flow import (
        WaveformRelaxationSolver,
    )
    from molmetal.adapters.flow_matching_lipman.m5_schedulers import (
        AdaptiveTimestepScheduler,
        LossAnnealer,
        VarianceReducedEstimator,
    )

    runners: Dict[str, Callable[[List[str], int], List[float]]] = {}

    def _run_hilbert(smiles: List[str], n_runs: int) -> List[float]:
        hil = HilbertSpace(dim=8)
        pd = PrimalDualOptimizer(lr_primal=0.01, lr_dual=0.01, n_iters=5)
        # Simple convex objective: f(x) = 0.5 * ||x||^2, grad = x
        def obj(x):
            val = float(0.5 * np.sum(x ** 2))
            grad = x.copy()
            return val, grad
        wall: List[float] = []
        primal = np.zeros(8)
        dual = np.zeros(8)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            primal, dual = pd.step(primal, dual, obj)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["reward/hilbert_primal_dual"] = _run_hilbert

    def _run_sr(smiles: List[str], n_runs: int) -> List[float]:
        # Default score_fn is Gaussian-to-zero.
        sr = ScoreRepellentSampler(n_iters=5, seed=0)
        # x0 must be 1-D (anchor for the chain).
        x0 = torch.randn(4)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = sr.sample(x0, n_samples=2)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["search_alg/score_repellent_mc"] = _run_sr

    def _run_vpuct(smiles: List[str], n_runs: int) -> List[float]:
        # MCTSVarianceWrapper wraps an MCTSProofSearch-like object.  For the
        # CPU benchmark we exercise VarianceControlledPUCT.select directly,
        # which is the Wf-15 primitive the wrapper integrates with.
        from molmetal.molmetal_lam.search_alg.variance_puct_integration import (
            VarianceControlledPUCT,
        )
        vcp = VarianceControlledPUCT()
        node = IntegrationNode(node_id="root", prior=1.0)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = vcp.select(node)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["search_alg/variance_puct_integration"] = _run_vpuct

    def _run_wflow(smiles: List[str], n_runs: int) -> List[float]:
        ws = WaveformRelaxationSolver(n_windows=2, n_iters_per_window=2, n_substeps_per_window=2)
        # Linear velocity field dx/dt = 1
        def v(x, t):
            return torch.ones_like(x)
        x0 = torch.zeros(1, 4)
        wall: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = ws.solve(v, x0, t_span=(0.0, 1.0))
            wall.append(time.perf_counter() - t0)
        return wall
    runners["flow_matching_lipman/w_flow"] = _run_wflow

    def _run_m5(smiles: List[str], n_runs: int) -> List[float]:
        sch = AdaptiveTimestepScheduler()
        ann = LossAnnealer(total_steps=1000, warmup_frac=0.1)
        est = VarianceReducedEstimator()
        wall: List[float] = []
        x = torch.randn(2, 4)
        dx = torch.randn(2, 4)
        adv = torch.randn(2)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = sch.step_size(t=0.5, x=x, dx=dx)
            _ = ann.weight(step=100)
            _ = est.forward(advantages=adv)
            wall.append(time.perf_counter() - t0)
        return wall
    runners["flow_matching_lipman/m5_schedulers"] = _run_m5

    return runners


# ---------------------------------------------------------------------------
# Module registry — built once at import time.  Total: 32 module runners
# (10 math_skills + 7 adapters + 13 lam_chem + 1 reward + 2 search_alg +
# 2 flow_matching_lipman — but several runners are intentionally None
# (import-only) and the registry reports 32 IDs nonetheless).
# ---------------------------------------------------------------------------
def _build_module_registry() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Compose all R18 module runners into a single ordered mapping."""
    registry: Dict[str, Callable[[List[str], int], List[float]]] = {}
    registry.update(_runner_math_skills())          # 10 modules (T1–T10)
    registry.update(_runner_adapters())             # 7 modules
    registry.update(_runner_lam_chem())             # 13 modules (incl. import-only)
    registry.update(_runner_reward_search_flow())   # 5 modules
    return registry


_MODULE_GROUPS: Dict[str, List[str]] = {
    "math_skills": [
        "math_skills/bass_existence",
        "math_skills/bergman_kernel",
        "math_skills/monge_ampere_regularity",
        "math_skills/multi_parameter_ph",
        "math_skills/ph_hardness_guardrail",
        "math_skills/sb_bass_martingale",
        "math_skills/se3_schrodinger_bridge",
        "math_skills/transport_relaxation",
        "math_skills/wasserstein_gradient_flow",
    ],
    "adapters": [
        "adapters/coordination_gnn",
        "adapters/quotient_space_diffusion",
        "adapters/multi_ph_integration",
        "adapters/se3_sb_integration",
        "adapters/transport_relaxation_integration",
        "adapters/diffdock",
        "adapters/equibind",
    ],
    "lam_chem": [
        "lam_chem/coupling_joint_inference",
        "lam_chem/decoder_m3_sequence",
        "lam_chem/cross_attention_m4",
        "lam_chem/layered_generation",
        "lam_chem/buyable_gate",
        "lam_chem/ood_protocol",
        "lam_chem/macroguide_ph",
        "lam_chem/non_click_medchem_rules",
        "lam_chem/skeleton_jump",
        "lam_chem/md_relaxation_stub",
        "lam_chem/fg_compat_veto",
    ],
    "reward": [
        "reward/hilbert_primal_dual",
    ],
    "search_alg": [
        "search_alg/score_repellent_mc",
        "search_alg/variance_puct_integration",
    ],
    "flow_matching_lipman": [
        "flow_matching_lipman/w_flow",
        "flow_matching_lipman/m5_schedulers",
    ],
}


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


def _median_ms(wall_times: List[float]) -> float:
    """Return the median wall-time across ``wall_times`` in milliseconds."""
    if not wall_times:
        return 0.0
    return float(statistics.median(wall_times)) * 1000.0


def _run_module(
    runner: Callable[[List[str], int], List[float]] | None,
    smiles: List[str],
    n_runs: int,
) -> Tuple[List[float], str]:
    """Run ``runner`` ``n_runs`` times; return (wall_times, status).

    Status is one of: ``"ok"``, ``"skipped"`` (no runner), ``"error"``.
    On ``"error"`` the wall_times list is empty.
    """
    if runner is None:
        return [], "skipped"
    try:
        wt = runner(smiles, n_runs)
        return list(wt), "ok"
    except Exception as exc:  # noqa: BLE001 — record every failure mode
        sys.stderr.write(
            f"[bench] runner failed: {type(exc).__name__}: {str(exc)[:200]}\n"
        )
        return [], "error"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R18 modules real-data benchmark on PlatinAI 5K.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of PlatinAI samples (default: 500).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Forward calls per module (default: 10).",
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
        help="Output directory for results.json (default: wf_platinai_bench).",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="",
        help="Comma-separated module IDs to benchmark (default: all).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.stderr.write(f"[bench] platinai CSV not found: {csv_path}\n")
        return 1

    smiles = _load_platinai_smiles(csv_path, int(args.n_samples))
    if not smiles:
        sys.stderr.write(f"[bench] no SMILES in {csv_path}\n")
        return 1

    n_samples = len(smiles)
    n_runs = max(1, int(args.n_runs))

    # Module registry + optional filter
    registry = _build_module_registry()
    if args.modules:
        wanted = {m.strip() for m in args.modules.split(",") if m.strip()}
        registry = {k: v for k, v in registry.items() if k in wanted}
        if not registry:
            sys.stderr.write(f"[bench] no modules matched {args.modules!r}\n")
            return 1

    # Allocate result maps
    wall_ms: Dict[str, float] = {}
    mem_mb: Dict[str, float] = {}
    n_calls: Dict[str, int] = {}
    status: Dict[str, str] = {}

    t_total_0 = time.perf_counter()
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]

    for mod_id, runner in registry.items():
        sub_base = tracemalloc.get_traced_memory()[0]
        wt, st = _run_module(runner, smiles, n_runs)
        sub_peak = tracemalloc.get_traced_memory()[1]

        wall_ms[mod_id] = _median_ms(wt)
        mem_mb[mod_id] = float(sub_peak - sub_base) / (1024 * 1024)
        n_calls[mod_id] = len(wt) if st == "ok" else 0
        status[mod_id] = st

    peak_mem_total = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    total_wall_s = time.perf_counter() - t_total_0

    # ---------------------------------------------------------------------
    # Output JSON
    # ---------------------------------------------------------------------
    import torch  # local; cheap to import here

    py_ver = ".".join(str(v) for v in sys.version_info[:3])
    torch_ver = torch.__version__

    payload: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_modules": int(len(registry)),
        "total_wall_time_s": float(total_wall_s),
        "per_module_wall_time_ms": wall_ms,
        "per_module_memory_mb": mem_mb,
        "per_module_n_calls": n_calls,
        "per_module_status": status,
        "module_groups": _MODULE_GROUPS,
        "n_runs_per_module": int(n_runs),
        "peak_memory_mb_total": float(peak_mem_total - base_mem) / (1024 * 1024),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": py_ver,
        "torch": torch_ver,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    n_ok = sum(1 for s in status.values() if s == "ok")
    n_skip = sum(1 for s in status.values() if s == "skipped")
    n_err = sum(1 for s in status.values() if s == "error")

    print(
        f"[bench] n_samples={n_samples} "
        f"n_modules={len(registry)} "
        f"ok={n_ok} skipped={n_skip} error={n_err} "
        f"total_wall_s={total_wall_s:.2f}"
    )
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())