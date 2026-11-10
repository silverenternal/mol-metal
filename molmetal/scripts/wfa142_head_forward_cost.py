#!/usr/bin/env python3
# Copyright 2026 The Mol-Metal Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A142 (2026-10-31) — CPU cost probe for the MolDiff-style bond-predictor
confidence guidance (row 3 of ``/tmp/wfa140/frontier-lit.md``).

WHAT THIS MEASURES (and nothing else)
-------------------------------------
Wall-clock cost, on CPU only, of the three pieces a
"bond-predictor confidence gradient" guidance term would add to the ODE
loop:

1. ``BondOrderHead`` (``molmetal/models/bond_head.py``) — forward only, and
   forward + ``autograd.grad`` w.r.t. the atom POSITIONS (the MolDiff term
   needs ``d/dx``, not ``d/dlogits``).
2. ``YuelBondDecoderHead`` (``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py``)
   — the CFM production bond predictor — same two numbers.
3. ``_BondLengthGuidance`` (``molmetal/adapters/flow_matching_lipman``) —
   the CURRENT table-based guidance — as the BASELINE the new term would be
   compared against.

No checkpoint is loaded (they are CUDA-serialised and unreadable CPU-only).
No GPU is used.  Every tensor is randomly generated on the CPU.

DEFAULT SHAPES (all MEASURED / read off disk, not guessed)
----------------------------------------------------------
* ``n_atoms = 39`` — the MEDIAN heavy-atom count of the CFM production
  cohort ``.data/molmetal/metalcytotox_5metal_multitask_v3.csv`` (26,632
  rows; median 39, mean 41.48, min 9, max 234).  See ``run_probe`` for the
  recomputation.
* ``cutoff = 3.5 Å`` — ``edge_cutoff`` in ``configs/cfm_5metal_default.yaml``
  (line 204) and ``configs/cfm_5metal_wfa75_bond_guidance.yaml`` (line 160);
  also ``YuelBondDecoderHead.cutoff``'s default.
* ``n_steps = 100`` — the trainer's argparse default
  (``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:12357``); 33 of the
  52 ``configs/cfm_*.yaml`` set ``n_steps: 100`` explicitly.  The
  ``ports.GenerationConfig`` default is 50 (``molmetal/ports/__init__.py:56``).
* model evaluations per ODE step: euler 1, midpoint 2, rk4 4
  (``molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py``).

USAGE
-----
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa142_head_forward_cost.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --------------------------------------------------------------------------
# Defaults — every one of these is traceable to a file on disk (see the
# module docstring).  They are DEFAULTS, not measurements.
# --------------------------------------------------------------------------
DEFAULT_N_ATOMS: int = 39
DEFAULT_CUTOFF: float = 3.5
DEFAULT_N_STEPS: int = 100
DEFAULT_REPEATS: int = 7
DEFAULT_SEED: int = 0
DEFAULT_OUT: Path = (
    _REPO_ROOT / "molmetal" / "reports" / "wfa142_head_forward_cost" / "cost.json"
)

#: Number of velocity-model evaluations per ODE step, per integrator.
#: ``_reference/solver/__init__.py``: ``euler_step`` evaluates once,
#: ``midpoint_step`` twice (Heun), ``rk4_step`` four times.
MODEL_EVALS_PER_STEP: Dict[str, int] = {
    "euler": 1,
    "midpoint": 2,
    "heun3": 2,
    "rk4": 4,
}


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def time_callable(
    fn: Callable[[], Any],
    repeats: int = DEFAULT_REPEATS,
    warmup: int = 2,
) -> Dict[str, float]:
    """Return median / min / max wall-time of ``fn`` in milliseconds.

    Uses :func:`time.perf_counter`.  ``warmup`` untimed calls absorb the
    first-call lazy-init cost (torch's first autograd node allocation,
    MKL/oneDNN kernel selection) so the reported numbers are steady-state.

    ``min_ms`` is the HEADLINE number, not ``median_ms``.  MEASURED
    2026-10-31 on this host: the load average was 53 (many concurrent
    agents), and the median of a 7-sample run was anywhere between 1.3 ms
    and 1610 ms for the SAME call, while the MINIMUM was stable at
    ~1.0-1.4 ms across shapes.  Under CPU contention the minimum is the
    only estimator that measures the work rather than the contention.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    for _ in range(max(0, int(warmup))):
        fn()
    samples: List[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "headline_ms": round(min(samples), 4),
        "repeats": int(repeats),
        "warmup": int(max(0, int(warmup))),
    }


def prewarm() -> None:
    """One untimed pass over every measured code path.

    MEASURED 2026-10-31: without it the FIRST shape measured in a fresh
    process reported a forward of 1426 ms where the same call, repeated
    immediately after, took 0.98 ms — a ~1400x one-off lazy-init cost
    (torch kernel selection + the dataclass ``.to()`` path) that would
    otherwise land in the artifact as a fake cost of the technique.
    """
    measure_bond_order_head(n_atoms=8, repeats=1)
    measure_yuelbond_head(n_atoms=8, repeats=1)
    measure_table_guidance_baseline(n_atoms=8, repeats=1)


def measure_pair_sets(
    n_atoms: int = DEFAULT_N_ATOMS,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """The TWO pair sets the guidance and the head would each work on.

    MEASURED on ONE geometry (the same synthetic cloud), so the two counts
    are directly comparable:

    * the shipped guidance builds a SYMMETRIC radius graph
      (``radius_edge_index``, adapter:1382) — every ordered pair
      ``src != dst`` inside the cutoff, so each unique pair appears TWICE;
    * the production bond head returns an UPPER-TRIANGLE candidate list
      (``YuelBondDecoderHead.forward``, yuelbond_decoder.py:891-903), so
      each unique pair appears ONCE.

    A MolDiff term must sum over the HEAD's list.  Summing over the
    guidance's symmetric list would double-count every pair (and break the
    Newton-3rd-law pairing), which is why this is measured rather than
    assumed.
    """
    from molmetal.adapters.flow_matching_lipman import radius_edge_index
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondDecoderHead

    cloud = synthetic_cloud(n_atoms, cutoff, seed)
    head = YuelBondDecoderHead(cutoff=float(cutoff)).eval()
    with torch.no_grad():
        out = head(
            AtomCloud(
                positions=cloud["positions"],
                atomic_numbers=cloud["atomic_numbers"].long(),
            )
        )
        head_i, head_j = out[1] if isinstance(out, tuple) else (None, None)
        n_head = int(head_i.shape[0])

    x = cloud["positions"].unsqueeze(0)
    node_mask = torch.ones((1, int(n_atoms)), dtype=torch.bool)
    edge_index, edge_mask = radius_edge_index(x, node_mask, float(cutoff))
    n_guidance = int(edge_mask[0].sum().item())

    # Unique unordered pairs behind the guidance's symmetric edge list.
    n_unique = n_guidance // 2 if n_guidance % 2 == 0 else n_guidance
    return {
        "n_atoms": int(n_atoms),
        "cutoff": float(cutoff),
        "seed": int(seed),
        "head_candidate_pairs": n_head,
        "guidance_radius_graph_edges": n_guidance,
        "guidance_unique_pairs": int(n_unique),
        "head_is_upper_triangle": bool(
            torch.all(head_i < head_j) if head_i is not None else False
        ),
        "ratio_guidance_edges_over_head_pairs": (
            float(n_guidance) / float(n_head) if n_head else float("nan")
        ),
        "note": (
            "same geometry for both counts; the guidance's graph is "
            "SYMMETRIC (each pair twice) and the head's list is "
            "UPPER-TRIANGLE (each pair once)."
        ),
    }


# --------------------------------------------------------------------------
# Synthetic geometry
# --------------------------------------------------------------------------
def synthetic_cloud(
    n_atoms: int = DEFAULT_N_ATOMS,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    sigma: float = 1.6,
) -> Dict[str, torch.Tensor]:
    """Random point cloud + its candidate-pair list under ``cutoff``.

    The candidate rule matches the production bond predictor's: the upper
    triangle ``i < j`` of the atom list, filtered to ``d < cutoff``
    (``YuelBondDecoderHead.forward``, ``mask = d_raw < self.cutoff``).

    Atomic numbers are drawn from the project's donor/metal vocabulary
    (the 12-element default in ``BondOrderHead.__init__`` plus the seven
    5-metal cohort centres) so the dative feature is not vacuous.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    positions = torch.randn(int(n_atoms), 3, generator=g) * float(sigma)
    z_pool = torch.tensor(
        [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78, 44, 45, 46, 75, 76, 77],
        dtype=torch.long,
    )
    idx = torch.randint(
        0, int(z_pool.shape[0]), (int(n_atoms),), generator=g, dtype=torch.long
    )
    atomic_numbers = z_pool[idx]
    i_idx, j_idx = torch.triu_indices(int(n_atoms), int(n_atoms), offset=1)
    d = (positions[j_idx] - positions[i_idx]).norm(dim=-1)
    keep = d < float(cutoff)
    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "i_idx": i_idx[keep],
        "j_idx": j_idx[keep],
        "distances": d[keep],
        "all_pairs": int(i_idx.shape[0]),
        "candidate_pairs": int(keep.sum().item()),
    }


def _bond_order_pair_features(
    n_atoms: int,
    n_pairs: int,
    in_dim: int,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    """A synthetic ``(E, in_dim)`` pair-feature tensor for ``BondOrderHead``.

    Columns 0 and 8 (distance, gap) are position-derived in the real
    pipeline; the rest are chemistry columns.  For a COST probe the exact
    values do not matter — only the shape and the dtype do — so the tensor
    is filled with a fixed-seed normal sample.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(int(n_pairs), int(in_dim), generator=g)


def _position_derived_features(
    positions: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    in_dim: int,
) -> torch.Tensor:
    """Build ``(E, in_dim)`` features whose column 0 is the pair distance.

    The distance column is the ONLY position-dependent channel in the
    ``BondOrderHead`` input layout (``PAIR_FEATURE_NAMES``, index 0
    ``"distance"`` and index 8 ``"gap"``).  Making column 0 a live function
    of ``positions`` is what lets the probe time the real
    ``d/dx log P_head`` chain rather than a ``d/dfeatures`` proxy.
    """
    e = int(i_idx.shape[0])
    feats = torch.zeros(e, int(in_dim), dtype=positions.dtype)
    feats[:, 0] = (positions[j_idx] - positions[i_idx]).norm(dim=-1)
    if int(in_dim) > 8:
        feats[:, 8] = feats[:, 0] - 1.5
    return feats


# --------------------------------------------------------------------------
# Head measurements
# --------------------------------------------------------------------------
def measure_bond_order_head(
    n_atoms: int = DEFAULT_N_ATOMS,
    in_dim: int = 11,
    hidden_dim: int = 64,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    bond_class: int = 1,
) -> Dict[str, Any]:
    """Forward and forward+position-gradient cost of ``BondOrderHead``.

    ``molmetal/models/bond_head.py::BondOrderHead.forward`` takes
    ``(E, in_dim)`` pair features and returns ``(E, num_classes)`` logits.
    It is a pure function of its arguments apart from ``nn.Dropout``
    (identity in ``eval()`` mode) — no mutable per-call state.
    """
    from molmetal.models.bond_head import BondOrderHead, PAIR_FEATURE_DIM

    cloud = synthetic_cloud(n_atoms, cutoff, seed)
    n_pairs = int(cloud["i_idx"].shape[0])
    head = BondOrderHead(in_dim=int(in_dim), hidden_dim=int(hidden_dim), num_classes=5)
    head.eval()
    params = int(sum(p.numel() for p in head.parameters()))

    positions = cloud["positions"].clone().requires_grad_(True)
    i_idx, j_idx = cloud["i_idx"], cloud["j_idx"]

    def _forward_only() -> torch.Tensor:
        with torch.no_grad():
            feats = _position_derived_features(positions.detach(), i_idx, j_idx, in_dim)
            return head(feats)

    def _forward_with_pos_grad() -> torch.Tensor:
        feats = _position_derived_features(positions, i_idx, j_idx, in_dim)
        logits = head(feats)
        logp = torch.log_softmax(logits, dim=-1)[:, int(bond_class)].sum()
        (grad,) = torch.autograd.grad(logp, positions, allow_unused=False)
        return grad

    fwd = time_callable(_forward_only, repeats)
    fwd_bwd = time_callable(_forward_with_pos_grad, repeats)
    return {
        "module": "molmetal.models.bond_head.BondOrderHead",
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "num_classes": 5,
        "params": params,
        "n_atoms": int(n_atoms),
        "n_candidate_pairs": n_pairs,
        "n_all_pairs": int(cloud["all_pairs"]),
        "forward_ms": fwd,
        "forward_posgrad_ms": fwd_bwd,
        "pair_feature_dim_layout": int(PAIR_FEATURE_DIM),
    }


def measure_yuelbond_head(
    n_atoms: int = DEFAULT_N_ATOMS,
    hidden_dim: int = 64,
    n_layers: int = 3,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    bond_class: int = 1,
) -> Dict[str, Any]:
    """Forward and forward+position-gradient cost of ``YuelBondDecoderHead``.

    This is the CFM production bond predictor
    (``train_cfm_yuelbond_wasserstein.py:3737``).  Its forward takes an
    ``AtomCloud`` and returns ``(logits, (i_idx, j_idx))``.
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondDecoderHead

    cloud = synthetic_cloud(n_atoms, cutoff, seed)
    head = YuelBondDecoderHead(
        hidden_dim=int(hidden_dim), n_layers=int(n_layers), cutoff=float(cutoff)
    )
    head.eval()
    params = int(sum(p.numel() for p in head.parameters()))

    positions = cloud["positions"].clone().requires_grad_(True)
    atomic_numbers = cloud["atomic_numbers"].long()

    def _make() -> "AtomCloud":
        return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)

    def _forward_only() -> torch.Tensor:
        with torch.no_grad():
            out = head(_make())
            logits = out[0] if isinstance(out, tuple) else out
            return logits

    def _forward_with_pos_grad() -> torch.Tensor:
        out = head(_make())
        logits = out[0] if isinstance(out, tuple) else out
        logp = torch.log_softmax(logits, dim=-1)[:, int(bond_class)].sum()
        (grad,) = torch.autograd.grad(logp, positions, allow_unused=False)
        return grad

    fwd = time_callable(_forward_only, repeats)
    fwd_bwd = time_callable(_forward_with_pos_grad, repeats)
    with torch.no_grad():
        out = head(_make())
        logits = out[0] if isinstance(out, tuple) else out
    return {
        "module": "molmetal.molmetal_lam.lam_chem.yuelbond_decoder."
        "YuelBondDecoderHead",
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "cutoff": float(cutoff),
        "params": params,
        "n_atoms": int(n_atoms),
        "n_candidate_pairs": int(logits.shape[0]),
        "forward_ms": fwd,
        "forward_posgrad_ms": fwd_bwd,
        "mutable_state_note": (
            "forward() writes self.last_clash_rejected_pairs on every call "
            "(yuelbond_decoder.py:862) — the only mutable per-call state."
        ),
    }


def measure_table_guidance_baseline(
    n_atoms: int = DEFAULT_N_ATOMS,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    guidance_weight: float = 0.1,
    upper_bound: bool = True,
) -> Dict[str, Any]:
    """Baseline: what the CURRENT table-based guidance costs per ODE step.

    ``_BondLengthGuidance`` is the code that ships today.  It is measured
    on the SAME geometry so the "what would the head-based term cost on
    top of this" comparison is apples-to-apples.
    """
    from molmetal.adapters.flow_matching_lipman import _BondLengthGuidance

    cloud = synthetic_cloud(n_atoms, cutoff, seed)
    guidance = _BondLengthGuidance(
        atom_types=cloud["atomic_numbers"].unsqueeze(0),
        cutoff=float(cutoff),
        guidance_weight=float(guidance_weight),
        upper_bound=bool(upper_bound),
    )
    x = cloud["positions"].unsqueeze(0)

    def _call() -> torch.Tensor:
        with torch.no_grad():
            return guidance(x)

    t = time_callable(_call, repeats)
    with torch.no_grad():
        force = guidance(x)
    return {
        "module": "molmetal.adapters.flow_matching_lipman._BondLengthGuidance",
        "guidance_weight": float(guidance_weight),
        "upper_bound": bool(upper_bound),
        "n_atoms": int(n_atoms),
        "force_shape": list(force.shape),
        "forward_ms": t,
        "note": (
            "single-cloud (B=1) call; the shipped guidance loops over the "
            "batch in Python (adapter __init__.py:1404)."
        ),
    }


def measure_batch_cost(
    batch_size: int = 8,
    n_atoms: int = DEFAULT_N_ATOMS,
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    guidance_weight: float = 0.1,
) -> Dict[str, Any]:
    """Cost of ONE model evaluation's worth of guidance on a BATCH.

    The two terms have DIFFERENT APIs, which is the whole cost story:

    * ``_BondLengthGuidance`` is BATCHED — one call handles (B, N, 3),
      because it builds the radius graph with the batched
      ``radius_edge_index`` and loops over the batch in Python.
    * ``YuelBondDecoderHead.forward`` is PER-CLOUD — it takes ONE
      ``AtomCloud`` and returns ONE edge list, so a batch of B must be
      driven as B sequential calls.

    ``batch_size = 8`` is the trainer's ``pad_group_bucket`` default
    (``train_cfm_yuelbond_wasserstein.py:9231``).
    """
    from molmetal.adapters.flow_matching_lipman import _BondLengthGuidance
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondDecoderHead

    clouds = [synthetic_cloud(n_atoms, cutoff, seed + i) for i in range(batch_size)]
    guidance = _BondLengthGuidance(
        atom_types=clouds[0]["atomic_numbers"].unsqueeze(0).repeat(int(batch_size), 1),
        cutoff=float(cutoff),
        guidance_weight=float(guidance_weight),
    )
    x = torch.stack([c["positions"] for c in clouds], dim=0)
    head = YuelBondDecoderHead(cutoff=float(cutoff)).eval()

    def _table_batched() -> torch.Tensor:
        with torch.no_grad():
            return guidance(x)

    def _head_per_cloud() -> int:
        total = 0
        for c in clouds:
            ac = AtomCloud(
                positions=c["positions"], atomic_numbers=c["atomic_numbers"].long()
            )
            out = head(ac)
            logits = out[0] if isinstance(out, tuple) else out
            total += int(logits.shape[0])
        return total

    t_table = time_callable(_table_batched, repeats)
    t_head = time_callable(_head_per_cloud, repeats)
    return {
        "batch_size": int(batch_size),
        "n_atoms": int(n_atoms),
        "table_guidance_batched_ms": t_table,
        "yuelbond_head_per_cloud_loop_ms": t_head,
        "note": (
            "the table guidance is ONE batched call; the head is B "
            "sequential per-cloud calls — that asymmetry is the cost "
            "difference, not the per-call cost."
        ),
    }


# --------------------------------------------------------------------------
# Cohort shape (measured, not assumed)
# --------------------------------------------------------------------------
def measure_cohort_heavy_atoms(
    csv_path: str = ".data/molmetal/metalcytotox_5metal_multitask_v3.csv",
) -> Dict[str, Any]:
    """Heavy-atom count distribution of the CFM production cohort.

    This is what makes ``DEFAULT_N_ATOMS`` a measurement rather than a
    guess.  Returns ``{}`` when the CSV is absent (the probe stays usable).
    """
    import csv as _csv

    p = _REPO_ROOT / csv_path
    if not p.exists():
        return {"available": False, "path": csv_path}
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception as exc:  # pragma: no cover — RDKit is a hard dep
        return {"available": False, "path": csv_path, "error": str(exc)}
    rows = list(_csv.DictReader(p.open()))
    counts: List[int] = []
    for r in rows:
        s = r.get("SMILES_Ligands") or r.get("SMILES") or r.get("smiles")
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        counts.append(int(mol.GetNumHeavyAtoms()))
    if not counts:
        return {"available": False, "path": csv_path, "rows": len(rows)}
    return {
        "available": True,
        "path": csv_path,
        "n_rows": len(rows),
        "n_parsed": len(counts),
        "heavy_atoms_min": min(counts),
        "heavy_atoms_median": statistics.median(counts),
        "heavy_atoms_mean": round(statistics.mean(counts), 4),
        "heavy_atoms_max": max(counts),
    }


# --------------------------------------------------------------------------
# ODE cost extrapolation
# --------------------------------------------------------------------------
def extrapolate_ode_cost(
    per_call_ms: float,
    n_steps: int = DEFAULT_N_STEPS,
    n_model_evals_per_step: int = 2,
) -> Dict[str, float]:
    """Total added wall-time if the term runs once per model evaluation.

    The shipped guidance is wired as a velocity wrapper
    (``train_cfm_yuelbond_wasserstein.py:9219``), so it runs on EVERY model
    evaluation — ``n_steps * evals_per_step`` calls, not ``n_steps``.
    """
    n_calls = int(n_steps) * int(n_model_evals_per_step)
    return {
        "per_call_ms": round(float(per_call_ms), 4),
        "n_steps": int(n_steps),
        "model_evals_per_step": int(n_model_evals_per_step),
        "n_calls": n_calls,
        "total_s": round(float(per_call_ms) * n_calls / 1000.0, 4),
    }


def run_probe(
    n_atoms_list: Sequence[int] = (DEFAULT_N_ATOMS,),
    cutoff: float = DEFAULT_CUTOFF,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    n_steps: int = DEFAULT_N_STEPS,
    bond_head_in_dims: Sequence[int] = (11, 265),
) -> Dict[str, Any]:
    """Full probe -> the JSON artifact as a plain dict."""
    import datetime as _dt

    cohort = measure_cohort_heavy_atoms()
    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover — non-Linux
        load1 = load5 = load15 = -1.0
    prewarm()
    shapes: List[Dict[str, Any]] = []
    for n_atoms in n_atoms_list:
        row: Dict[str, Any] = {"n_atoms": int(n_atoms)}
        row["table_guidance_baseline"] = measure_table_guidance_baseline(
            n_atoms=n_atoms, cutoff=cutoff, seed=seed, repeats=repeats
        )
        row["yuelbond_decoder_head"] = measure_yuelbond_head(
            n_atoms=n_atoms, cutoff=cutoff, seed=seed, repeats=repeats
        )
        row["bond_order_heads"] = {
            f"in_dim_{d}": measure_bond_order_head(
                n_atoms=n_atoms,
                in_dim=int(d),
                cutoff=cutoff,
                seed=seed,
                repeats=repeats,
            )
            for d in bond_head_in_dims
        }
        row["pair_sets"] = measure_pair_sets(n_atoms=n_atoms, cutoff=cutoff, seed=seed)
        shapes.append(row)

    # The ODE cost model is anchored on the COHORT-MEDIAN molecule (the
    # shape the production cohort actually has), not on shapes[0].
    target_n = int(cohort.get("heavy_atoms_median") or DEFAULT_N_ATOMS)
    primary = next((s for s in shapes if int(s["n_atoms"]) == target_n), shapes[0])
    batch = measure_batch_cost(
        batch_size=8,
        n_atoms=int(cohort.get("heavy_atoms_median") or DEFAULT_N_ATOMS),
        cutoff=cutoff,
        seed=seed,
        repeats=repeats,
    )
    ode = _build_ode_cost_model(shapes, batch, int(n_steps), int(primary["n_atoms"]))
    return {
        "schema": "wfa142.head_forward_cost.v1",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "wave": "Wf-A142",
        "cost_model_anchor_n_atoms": int(primary["n_atoms"]),
        "purpose": (
            "CPU cost probe for the MolDiff-style bond-predictor confidence "
            "guidance (frontier-lit row 3). READ-ONLY survey; no adapter "
            "change was made."
        ),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "loadavg_1_5_15": [round(load1, 2), round(load5, 2), round(load15, 2)],
        },
        "probe": {
            "n_atoms_list": [int(a) for a in n_atoms_list],
            "cutoff": float(cutoff),
            "seed": int(seed),
            "repeats": int(repeats),
            "n_steps": int(n_steps),
            "torch_num_threads": int(torch.get_num_threads()),
            "model_evals_per_step": dict(MODEL_EVALS_PER_STEP),
        },
        "cohort_shape": cohort,
        "batch_cost": batch,
        "shapes": shapes,
        "ode_cost_model": ode,
        "notes": [
            "All timings are CPU wall-clock (time.perf_counter).",
            "headline_ms == min over the repeats; under the MEASURED host "
            "contention (load average ~53, many concurrent agents) the "
            "MEDIAN of a 7-sample run varied from 1.3 ms to 1610 ms for the "
            "same call while the MINIMUM was stable, so the minimum is the "
            "headline estimator.",
            "forward_posgrad_ms includes autograd.grad of "
            "sum log_softmax(logits)[:, bond_class] w.r.t. atom positions.",
            "No checkpoint is loaded; all tensors are synthetic.",
            "n_atoms default 39 = MEDIAN heavy-atom count of "
            ".data/molmetal/metalcytotox_5metal_multitask_v3.csv.",
            "The shipped guidance is a velocity WRAPPER, so it runs once per "
            "MODEL EVALUATION, not once per ODE step.",
        ],
    }


def merge_cost_artifacts(
    paths: Sequence[Path],
    out_path: Path,
) -> Dict[str, Any]:
    """Merge N probe artifacts cell-by-cell, keeping the MINIMUM of each.

    MEASURED 2026-10-31 on this host: three runs of the SAME probe minutes
    apart produced per-call minima that differed by up to ~3x (n=39 table
    guidance: 0.644 ms in one run, 2.094 ms in another), with identical
    geometry (identical candidate-pair counts) and identical code.  The
    cause is host CPU contention, not the technique: the load average was
    45-53 with many concurrent agents.

    The minimum across independent runs is the least-contaminated estimate
    of the work itself; the maximum across runs is the contended upper
    bound.  Both are kept in the merged artifact, per cell, so a reader can
    see the spread instead of a single lucky number.
    """
    loaded = [json.loads(Path(p).read_text()) for p in paths]
    if not loaded:
        raise ValueError("no artifacts to merge")
    merged = json.loads(json.dumps(loaded[0]))
    merged["merged_from"] = [
        {
            "path": str(p),
            "created_utc": doc.get("created_utc"),
            "loadavg_1_5_15": doc.get("host", {}).get("loadavg_1_5_15"),
        }
        for p, doc in zip(paths, loaded)
    ]

    def _minmax(
        cell_docs: List[Dict[str, Any]], key_path: Sequence[str]
    ) -> Dict[str, float]:
        vals: List[float] = []
        for doc in cell_docs:
            node: Any = doc
            for k in key_path:
                node = node[k]
            vals.append(float(node))
        return {
            "min_ms": round(min(vals), 4),
            "max_ms": round(max(vals), 4),
            "n_sources": len(vals),
        }

    # Per-shape cells.
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for doc in loaded:
        for row in doc["shapes"]:
            by_n.setdefault(int(row["n_atoms"]), []).append(row)
    merged_shapes: List[Dict[str, Any]] = []
    for n_atoms in sorted(by_n):
        rows = by_n[n_atoms]
        out_row: Dict[str, Any] = {"n_atoms": n_atoms, "n_sources": len(rows)}
        out_row["table_guidance_baseline"] = {
            "forward_ms": _minmax(
                [r["table_guidance_baseline"] for r in rows],
                ("forward_ms", "headline_ms"),
            )
        }
        out_row["yuelbond_decoder_head"] = {
            "n_candidate_pairs": int(
                rows[0]["yuelbond_decoder_head"]["n_candidate_pairs"]
            ),
            "params": int(rows[0]["yuelbond_decoder_head"]["params"]),
            "forward_ms": _minmax(
                [r["yuelbond_decoder_head"] for r in rows],
                ("forward_ms", "headline_ms"),
            ),
            "forward_posgrad_ms": _minmax(
                [r["yuelbond_decoder_head"] for r in rows],
                ("forward_posgrad_ms", "headline_ms"),
            ),
        }
        out_row["bond_order_heads"] = {}
        for key in rows[0]["bond_order_heads"]:
            out_row["bond_order_heads"][key] = {
                "params": int(rows[0]["bond_order_heads"][key]["params"]),
                "forward_ms": _minmax(
                    [r["bond_order_heads"][key] for r in rows],
                    ("forward_ms", "headline_ms"),
                ),
                "forward_posgrad_ms": _minmax(
                    [r["bond_order_heads"][key] for r in rows],
                    ("forward_posgrad_ms", "headline_ms"),
                ),
            }
        merged_shapes.append(out_row)
    merged["shapes"] = merged_shapes

    # Pair-set counts (DETERMINISTIC, not timings): carry them across and
    # assert the runs agree — a disagreement would mean the geometry
    # differed, which would invalidate the timing comparison.
    for row in merged["shapes"]:
        n = int(row["n_atoms"])
        src_rows = by_n[n]
        pair_sets = [r["pair_sets"] for r in src_rows if "pair_sets" in r]
        if not pair_sets:
            continue
        keys = ("head_candidate_pairs", "guidance_radius_graph_edges")
        for k in keys:
            vals = {int(p[k]) for p in pair_sets}
            if len(vals) != 1:
                raise ValueError(f"pair-set counts differ across runs at N={n}: {vals}")
        row["pair_sets"] = pair_sets[0]

    # Batch cost.
    merged["batch_cost"] = {
        "batch_size": int(loaded[0]["batch_cost"]["batch_size"]),
        "n_atoms": int(loaded[0]["batch_cost"]["n_atoms"]),
        "table_guidance_batched_ms": _minmax(
            [d["batch_cost"] for d in loaded],
            ("table_guidance_batched_ms", "headline_ms"),
        ),
        "yuelbond_head_per_cloud_loop_ms": _minmax(
            [d["batch_cost"] for d in loaded],
            ("yuelbond_head_per_cloud_loop_ms", "headline_ms"),
        ),
        "note": loaded[0]["batch_cost"]["note"],
    }

    # ODE cost model — recompute from the merged minima.
    merged["ode_cost_model"] = _build_ode_cost_model(
        merged_shapes,
        merged["batch_cost"],
        int(merged["probe"]["n_steps"]),
        int(merged.get("cost_model_anchor_n_atoms", merged_shapes[0]["n_atoms"])),
    )
    merged["notes"] = list(loaded[0].get("notes", [])) + [
        "MERGED artifact: each cell carries min_ms (best run) and max_ms "
        "(worst run) across the N independent runs listed in merged_from.",
        "The min/max spread is host CPU contention, not measurement error: "
        "the geometry (candidate-pair counts) and the code are identical "
        "across runs.",
    ]
    _atomic_write_json(Path(out_path), merged)
    return merged


def _cell_ms(node: Dict[str, Any]) -> float:
    """Read the headline millisecond value from a timing cell.

    A freshly-measured cell carries ``headline_ms``; a MERGED cell carries
    ``min_ms``.  Both mean "the best estimate of the work itself".
    """
    for key in ("min_ms", "headline_ms"):
        if key in node:
            return float(node[key])
    raise KeyError(f"no millisecond value in cell: {sorted(node)}")


def _build_ode_cost_model(
    shapes: List[Dict[str, Any]],
    batch: Dict[str, Any],
    n_steps: int,
    anchor_n: int,
) -> Dict[str, Any]:
    """ODE cost extrapolation from the merged minima."""
    primary = next((s for s in shapes if int(s["n_atoms"]) == int(anchor_n)), shapes[0])
    base_ms = _cell_ms(primary["table_guidance_baseline"]["forward_ms"])
    out: Dict[str, Any] = {
        "table_guidance_baseline_per_cloud": {
            m: extrapolate_ode_cost(base_ms, n_steps, ev)
            for m, ev in MODEL_EVALS_PER_STEP.items()
        },
        "yuelbond_head_forward_per_cloud": {
            m: extrapolate_ode_cost(
                _cell_ms(primary["yuelbond_decoder_head"]["forward_ms"]),
                n_steps,
                ev,
            )
            for m, ev in MODEL_EVALS_PER_STEP.items()
        },
        "yuelbond_head_forward_posgrad_per_cloud": {
            m: extrapolate_ode_cost(
                _cell_ms(primary["yuelbond_decoder_head"]["forward_posgrad_ms"]),
                n_steps,
                ev,
            )
            for m, ev in MODEL_EVALS_PER_STEP.items()
        },
        "batch_of_8_table_guidance_batched": {
            m: extrapolate_ode_cost(
                _cell_ms(batch["table_guidance_batched_ms"]), n_steps, ev
            )
            for m, ev in MODEL_EVALS_PER_STEP.items()
        },
        "batch_of_8_yuelbond_head_per_cloud_loop": {
            m: extrapolate_ode_cost(
                _cell_ms(batch["yuelbond_head_per_cloud_loop_ms"]),
                n_steps,
                ev,
            )
            for m, ev in MODEL_EVALS_PER_STEP.items()
        },
    }
    return out


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` atomically (POSIX os.replace) per the repo protocol."""
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )

    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    reason = detect_truncation(text, expected_min_length=200)
    if reason is not None:
        raise RuntimeError(f"refusing to write truncated JSON: {reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_overwrite_check(path, text, raise_on_risk=True)
    atomic_write_file(path, text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-atoms", type=int, nargs="+", default=[DEFAULT_N_ATOMS])
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--no-cohort-scan", action="store_true")
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads. Default 1: the heads are tiny MLPs whose "
        "cost is dominated by thread-pool overhead, and a single thread is "
        "the only setting that survives the MEASURED host contention.",
    )
    ap.add_argument(
        "--merge",
        type=str,
        nargs="+",
        default=None,
        help="Merge these existing cost.json artifacts (cell-wise MIN) into "
        "--out. Use this to cancel out host CPU contention, which was "
        "MEASURED to move the per-call minimum by up to ~3x between runs.",
    )
    ap.add_argument(
        "--pairs-only",
        action="store_true",
        help="Emit ONLY the pair-set census (head candidates vs the "
        "guidance's symmetric radius graph) and skip every timing. The "
        "counts are deterministic, so they do not need the multi-run "
        "merge that the timings need.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.pairs_only:
        import datetime as _dt

        rows = [
            measure_pair_sets(n_atoms=int(n), cutoff=args.cutoff, seed=args.seed)
            for n in args.n_atoms
        ]
        payload = {
            "schema": "wfa142.pair_sets.v1",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "wave": "Wf-A142",
            "purpose": (
                "Pair-set census: the shipped guidance's SYMMETRIC radius "
                "graph vs the production bond head's UPPER-TRIANGLE "
                "candidate list, on the SAME geometry. Deterministic — no "
                "timing, no merge needed."
            ),
            "host": {
                "device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "torch_version": torch.__version__,
            },
            "cutoff": float(args.cutoff),
            "seed": int(args.seed),
            "pair_sets": rows,
        }
        _atomic_write_json(Path(args.out), payload)
        print(f"[wfa142] wrote {args.out}")
        for r in rows:
            print(
                f"[wfa142] n_atoms={r['n_atoms']} "
                f"head_pairs={r['head_candidate_pairs']} "
                f"guidance_edges={r['guidance_radius_graph_edges']} "
                f"(unique {r['guidance_unique_pairs']})"
            )
        return 0

    if args.merge:
        merged = merge_cost_artifacts([Path(p) for p in args.merge], Path(args.out))
        print(f"[wfa142] merged {len(args.merge)} artifacts -> {args.out}")
        for row in merged["shapes"]:
            yb = row["yuelbond_decoder_head"]
            print(
                f"[wfa142] n_atoms={row['n_atoms']} "
                f"table={row['table_guidance_baseline']['forward_ms']['min_ms']}"
                f"..{row['table_guidance_baseline']['forward_ms']['max_ms']} ms  "
                f"yb_fwd+grad={yb['forward_posgrad_ms']['min_ms']}"
                f"..{yb['forward_posgrad_ms']['max_ms']} ms"
            )
        return 0

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print(
            "WARNING: CUDA_VISIBLE_DEVICES is not empty — this probe is "
            "CPU-only by construction (Wf-A142 hard constraint 1).",
            file=sys.stderr,
        )
    if args.threads >= 1:
        torch.set_num_threads(int(args.threads))

    payload = run_probe(
        n_atoms_list=args.n_atoms,
        cutoff=args.cutoff,
        seed=args.seed,
        repeats=args.repeats,
        n_steps=args.n_steps,
    )
    if args.no_cohort_scan:
        payload["cohort_shape"] = {"available": False, "skipped": True}

    out = Path(args.out)
    _atomic_write_json(out, payload)
    print(f"[wfa142] wrote {out}")
    for row in payload["shapes"]:
        base = row["table_guidance_baseline"]["forward_ms"]["headline_ms"]
        yb = row["yuelbond_decoder_head"]
        print(
            f"[wfa142] n_atoms={row['n_atoms']} "
            f"table_guidance={base:.3f} ms  "
            f"yuelbond_fwd={yb['forward_ms']['headline_ms']:.3f} ms  "
            f"yuelbond_fwd+grad={yb['forward_posgrad_ms']['headline_ms']:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
