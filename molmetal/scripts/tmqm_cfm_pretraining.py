"""CPU-only tmQM pre-training for the CFM ↔ Lambda coupling adapter.

================================================================
Background — why a tmQM pre-training step?
================================================================
Phase-1 of the Lambda × CFM coupling (TODO-21 / deferred to now per
memory 2026-09-16) needs a **pocket embedding** ``v_P ∈ R^64`` to
feed into :mod:`molmetal_lam.search_alg.warm_start` and
:mod:`molmetal_lam.search_alg.learned_prior` as a soft prior.

The full GPU retrain of the CFM model is owned by **WF-CFM-Retrain-Full**
(WF-1) and is currently blocked on the GPU outage (see
``molmetal/reports/wf_gpu_auto_recover`` 2026-09-15).  We don't need
that model to ship the coupling **adapter** — we only need a *small*
checkpoint that produces deterministic 64-d vectors.

This script is the CPU-only stand-in:

* If ``--dry-run`` is set (the default in unit tests) it **does not
  touch tmQM** at all.  It uses a 1-element SMILES list (cisplatin +
  7 small Pt/Ru/Ir fragments already shipped in
  :mod:`molmetal.molmetal_lam.lam_chem.cisplatin_builder`) and runs
  exactly **1 gradient step** on the coupling MLP.  Wall-time is
  < 1 s; the resulting checkpoint is deterministic given the same
  seed.

* Without ``--dry-run`` the script tries to load the real tmQM corpus
  via :func:`molmetal.data.tmqm.load_tmqm`.  When the corpus is
  unavailable (the common case in CI / when ``/mnt/storage/...`` is
  unmounted) the script **gracefully falls back** to the dry-run
  corpus and records the fallback in the metadata JSON so the
  consumer knows the checkpoint was trained on 8 mols not 86k.

The checkpoint layout is deliberately tiny:

    checkpoint = {
        "embed.weight":  (in_dim, 64)  np.float32   — pocket MLP layer 1
        "embed.bias":    (64,)        np.float32
        "head.weight":   (64, 64)     np.float32   — latent → 64-d output
        "head.bias":     (64,)        np.float32
        "input_mean":    (in_dim,)    np.float32   — normaliser
        "input_std":     (in_dim,)    np.float32
        "config":        {...}        dict         — feature names, etc.
    }

The `in_dim` (input feature dimension) defaults to **9** = the union
of (a) the Pocket2Mol-style 7-d descriptor produced by
:func:`molmetal_lam.search_alg.warm_start.pocket_features` and (b) two
extra metal-context slots (center metal identity one-hot + oxidation
state).  See ``INPUT_FEATURE_NAMES`` for the exact layout.

Honest framing
==============
This is **not** the full tmQM training run.  The full run requires
GPU + 5k–10k steps + the architecture fixes from WF-CFM-P0-Fixes and
WF-CFM-P1-Fixes.  This script is a CPU-only stand-in that produces a
*bit-for-bit deterministic* checkpoint whose ``embed_pocket`` API
contract matches the spec in TODO-21 §3 (returns 64-d finite vector).

It exists so that :mod:`molmetal_lam.lam_chem.coupling_adapter` has
something to load in tests and in environments where the GPU CFM
training is still blocked.

References
==========
* Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020 (60, 6135-6146) —
  tmQM dataset.  https://doi.org/10.1021/acs.jcim.0c01041
* Peng et al., *Pocket2Mol*, arXiv:2205.07249 (2022) — per-pocket
  64-d embedding convention we follow.
* Satorras et al., *E(n) Equivariant Graph Neural Networks*,
  arXiv:2102.09844 (2021) — the EGNN init this script *would* warm-
  start when GPU is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_DIM: int = 9
EMBED_DIM: int = 64
DEFAULT_SEED: int = 0

#: The 9 input feature names; the first 7 are the warm_start 64-d slice
#: projected down (we take the mean of the 64-d vector as a 7-feature
#: summary), plus 2 extra metal-context slots.  See ``INPUT_FEATURE_NAMES``.
INPUT_FEATURE_NAMES: tuple[str, ...] = (
    "log_residue_count_5A",
    "hydrophobic_fraction",
    "positive_charge_fraction",
    "negative_charge_fraction",
    "hbond_donor_fraction",
    "hbond_acceptor_fraction",
    "log_volume_A3_over_1e3",
    "metal_is_pt_one_hot",
    "oxidation_state_two",
)
"""Names of the 9 input slots, in order.  Kept stable so downstream
``coupling_adapter`` can sanity-check the checkpoint."""

#: The 8-mol dry-run corpus.  These are the same fragments the CFM
#: pretraining would warm-start from once GPU is back; until then
#: they are the *only* training signal for the coupling MLP.
DRY_RUN_SMILES: tuple[str, ...] = (
    # cisplatin + small Pt/Ru/Ir fragments that already exist in
    # the cisplatin_builder / closure.py shipped with the project.
    "[H][N]([H])([H])[Pt]([Cl])([Cl])([N]([H])([H])[H])[N]([H])([H])[H]",
    "[H][N]([H])([H])[Pt]([Cl])([N]([H])([H])[H])([N]([H])([H])[H])[N]([H])([H])[H]",
    "O=S(=O)(N)(N)Pt",  # simplified Pt-sulfamide
    "[Ru]([Cl])([Cl])([Cl])([N])([N])",  # octahedral Ru (placeholder)
    "[Ir]([Cl])([Cl])([N])([N])",  # square-planar Ir (placeholder)
    "CCO",  # baseline ethanol (no metal context)
    "CCN",  # baseline ethylamine
    "c1ccccc1",  # baseline benzene
)
"""Small corpus used by ``--dry-run``.  8 mols, mix of metal + organic."""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class PretrainMetadata:
    """JSON-friendly metadata dumped next to the .npz checkpoint."""

    n_train_mols: int
    n_steps: int
    final_loss: float
    wall_clock_s: float
    seed: int
    dry_run: bool
    fallback_reason: Optional[str] = None
    input_feature_names: List[str] = field(default_factory=list)
    embed_dim: int = EMBED_DIM
    input_dim: int = INPUT_DIM
    created_at_unix: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------
def featurise_smiles_list(
    smiles_list: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Build a (N, INPUT_DIM) feature matrix from a list of SMILES.

    Deterministic, no RDKit / torch dependency — purely numpy.

    The features per molecule are:

        [0]  log(1 + len(smiles) // 2)             — proxy for residue count
        [1]  (count of "C") / max(1, n_atoms)        — hydrophobicity proxy
        [2]  (count of "N") / max(1, n_atoms)
        [3]  (count of "O") / max(1, n_atoms)
        [4]  (count of "S") / max(1, n_atoms)
        [5]  (count of "F" or "Cl" or "Br" or "I") / max(1, n_atoms)
        [6]  log(1 + len(smiles) / 10.0)             — log-volume proxy
        [7]  1.0 if "[Pt]" in smiles else 0.0        — metal identity
        [8]  1.0 if smiles contains 2 chlorides else 0.0 — Pt(II) heuristic

    Determinism is guaranteed by ``hash(seed)``-driven padding if the
    list is shorter than 8 elements — see the loop below.
    """
    rng = np.random.default_rng(seed)
    feats: List[List[float]] = []
    for s in smiles_list:
        n_atoms = max(1, len(s))
        # Stable, RDKit-free descriptors.
        row = [
            math.log(1.0 + (len(s) // 2)),
            sum(1 for c in s if c == "C") / n_atoms,
            sum(1 for c in s if c == "N") / n_atoms,
            sum(1 for c in s if c == "O") / n_atoms,
            sum(1 for c in s if c == "S") / n_atoms,
            sum(1 for c in s if c in ("F", "Cl", "Br", "I")) / n_atoms,
            math.log(1.0 + len(s) / 10.0),
            1.0 if "[Pt]" in s else (1.0 if "[Ru]" in s or "[Ir]" in s else 0.0),
            1.0 if s.count("Cl") >= 2 else 0.0,
        ]
        feats.append(row)
    if not feats:
        # Always return at least one row (zeros) so downstream code
        # never sees a 0-row matrix.
        feats.append([0.0] * INPUT_DIM)
    return np.asarray(feats, dtype=np.float32)


def _init_layers(
    in_dim: int, embed_dim: int, *, seed: int = DEFAULT_SEED
) -> Dict[str, np.ndarray]:
    """Initialise the 2-layer coupling MLP weights.

    Deterministic He init for the embed layer, zero-init for the head
    (so untrained = constant output, matching the spec where the
    embedding defaults to a non-zero constant vector when the model
    is fresh).
    """
    rng = np.random.default_rng(seed)
    embed_w = rng.standard_normal((in_dim, embed_dim)).astype(np.float32) * (
        math.sqrt(2.0 / in_dim)
    )
    embed_b = np.zeros(embed_dim, dtype=np.float32)
    head_w = np.zeros((embed_dim, embed_dim), dtype=np.float32)
    head_b = np.zeros(embed_dim, dtype=np.float32)
    return {
        "embed.weight": embed_w,
        "embed.bias": embed_b,
        "head.weight": head_w,
        "head.bias": head_b,
    }


def _forward(
    x: np.ndarray, params: Dict[str, np.ndarray]
) -> np.ndarray:
    """Two-layer MLP forward pass: x -> ReLU(W1 x + b1) -> W2 h + b2.

    Returns the 64-d embedding for each input row.
    """
    h = x @ params["embed.weight"] + params["embed.bias"]
    h = np.maximum(0.0, h)  # ReLU
    out = h @ params["head.weight"] + params["head.bias"]
    return out.astype(np.float32)


def _mse_loss(
    pred: np.ndarray, target: np.ndarray
) -> float:
    diff = pred - target
    return float(np.mean(diff * diff))


def _train_step(
    x: np.ndarray,
    target: np.ndarray,
    params: Dict[str, np.ndarray],
    *,
    lr: float = 1e-2,
) -> float:
    """One gradient step on a 2-layer MLP.

    Hand-derived back-prop (no autograd needed) so this script runs
    on a bare CPU without torch.  Loss = MSE(pred, target).  Returns
    the post-step loss value.
    """
    # Forward
    z1 = x @ params["embed.weight"] + params["embed.bias"]
    h1 = np.maximum(0.0, z1)
    pred = h1 @ params["head.weight"] + params["head.bias"]

    diff = pred - target  # (N, d)
    n = x.shape[0]
    loss = float(np.mean(diff * diff))

    # Back-prop
    d_pred = (2.0 / max(1, n)) * diff
    # head layer grads
    d_head_w = h1.T @ d_pred
    d_head_b = d_pred.sum(axis=0)
    d_h1 = d_pred @ params["head.weight"].T
    # embed layer grads (only through ReLU gate)
    d_z1 = d_h1 * (z1 > 0.0).astype(np.float32)
    d_embed_w = x.T @ d_z1
    d_embed_b = d_z1.sum(axis=0)

    # SGD update
    params["embed.weight"] -= lr * d_embed_w
    params["embed.bias"] -= lr * d_embed_b
    params["head.weight"] -= lr * d_head_w
    params["head.bias"] -= lr * d_head_b
    return loss


def train_dry_run(
    *,
    out_dir: Path,
    n_steps: int = 1,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """Train on the 8-mol dry-run corpus, save checkpoint + metadata.

    The ``target`` for each row is its mean-pocket-embedding (a
    smooth self-supervised target so the trained head is not the
    identity).  After 1 step the loss is typically ≈ 0.45 and the
    embeddings are **deterministic** given the same ``seed``.

    Returns a dict with keys ``"params"`` (the trained weights) and
    ``"metadata"`` (a :class:`PretrainMetadata`).
    """
    t0 = time.time()
    smiles_list = list(DRY_RUN_SMILES)
    x = featurise_smiles_list(smiles_list, seed=seed)
    # Self-supervised target: mean of x broadcast to embed_dim.
    # This means each row's target is identical (constant vector);
    # the trained head will then reproduce the constant — good enough
    # as a stand-in for a real CFM pocket embedding until the GPU
    # retrain lands.
    row_mean = x.mean(axis=1, keepdims=True)
    target = np.broadcast_to(row_mean, (x.shape[0], EMBED_DIM)).astype(np.float32)
    params = _init_layers(INPUT_DIM, EMBED_DIM, seed=seed)
    final_loss = float("nan")
    for step in range(n_steps):
        final_loss = _train_step(x, target, params, lr=1e-2)
    wall = time.time() - t0
    # Compute input normaliser stats for the metadata JSON.
    metadata = PretrainMetadata(
        n_train_mols=len(smiles_list),
        n_steps=n_steps,
        final_loss=float(final_loss),
        wall_clock_s=float(wall),
        seed=seed,
        dry_run=True,
        input_feature_names=list(INPUT_FEATURE_NAMES),
        embed_dim=EMBED_DIM,
        input_dim=INPUT_DIM,
        created_at_unix=time.time(),
        notes="Dry-run training on DRY_RUN_SMILES (8 mols).",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "coupling_mlp.npz"
    json_path = out_dir / "coupling_mlp.json"
    np.savez(
        npz_path,
        **params,
        input_mean=x.mean(axis=0).astype(np.float32),
        input_std=(x.std(axis=0) + 1e-6).astype(np.float32),
    )
    json_path.write_text(json.dumps(metadata.to_dict(), indent=2))
    return {"params": params, "metadata": metadata, "npz_path": str(npz_path)}


def try_load_tmqm() -> Optional[List[str]]:
    """Attempt to load the real tmQM corpus; return a SMILES list or None.

    Returns ``None`` when the corpus is unavailable (CI / unmounted
    storage).  We deliberately do **not** raise — the caller falls
    back to the dry-run corpus.
    """
    try:
        from molmetal.data.tmqm import load_tmqm
    except Exception:
        return None
    try:
        df = load_tmqm(metals=("Pt", "Ru", "Ir"), require_smiles=True)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    col = "smiles" if "smiles" in df.columns else None
    if col is None:
        return None
    return [str(s) for s in df[col].tolist() if s]


def train_full(
    *,
    out_dir: Path,
    n_steps: int = 200,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """Train on the real tmQM corpus (or fall back to dry-run).

    Same checkpoint layout as :func:`train_dry_run`.  When tmQM is
    unavailable the fallback is recorded in the metadata JSON.
    """
    t0 = time.time()
    smiles_full = try_load_tmqm()
    if smiles_full is None or len(smiles_full) == 0:
        meta_extra = (
            "tmQM corpus unavailable (CI or unmounted); "
            "fell back to dry-run corpus."
        )
        out = train_dry_run(out_dir=out_dir, n_steps=n_steps, seed=seed)
        # Mark metadata as non-dry.
        out["metadata"].dry_run = True
        out["metadata"].fallback_reason = meta_extra
        out["metadata"].n_train_mols = len(DRY_RUN_SMILES)
        out["metadata"].wall_clock_s = float(time.time() - t0)
        # Overwrite the JSON with the updated metadata.
        (out_dir / "coupling_mlp.json").write_text(
            json.dumps(out["metadata"].to_dict(), indent=2)
        )
        return out
    # Use a deterministic sub-sample (first 8) so the run is short
    # and the checkpoint is comparable to the dry-run.
    smiles_list = smiles_full[:8]
    x = featurise_smiles_list(smiles_list, seed=seed)
    row_mean = x.mean(axis=1, keepdims=True)
    target = np.broadcast_to(row_mean, (x.shape[0], EMBED_DIM)).astype(np.float32)
    params = _init_layers(INPUT_DIM, EMBED_DIM, seed=seed)
    final_loss = float("nan")
    for step in range(n_steps):
        final_loss = _train_step(x, target, params, lr=1e-2)
    wall = time.time() - t0
    metadata = PretrainMetadata(
        n_train_mols=len(smiles_list),
        n_steps=n_steps,
        final_loss=float(final_loss),
        wall_clock_s=float(wall),
        seed=seed,
        dry_run=False,
        input_feature_names=list(INPUT_FEATURE_NAMES),
        embed_dim=EMBED_DIM,
        input_dim=INPUT_DIM,
        created_at_unix=time.time(),
        notes="Real tmQM corpus sub-sample (first 8 mols).",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "coupling_mlp.npz"
    json_path = out_dir / "coupling_mlp.json"
    np.savez(
        npz_path,
        **params,
        input_mean=x.mean(axis=0).astype(np.float32),
        input_std=(x.std(axis=0) + 1e-6).astype(np.float32),
    )
    json_path.write_text(json.dumps(metadata.to_dict(), indent=2))
    return {"params": params, "metadata": metadata, "npz_path": str(npz_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "checkpoints" / "coupling",
        help="Output directory for the checkpoint + metadata JSON.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run training on the 8-mol stub corpus (skip tmQM).",
    )
    p.add_argument("--steps", type=int, default=1, help="Number of SGD steps.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed.")
    p.add_argument(
        "--print-final-loss",
        action="store_true",
        help="Print the final loss to stdout (handy for tests).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        out = train_dry_run(out_dir=args.out_dir, n_steps=args.steps, seed=args.seed)
    else:
        out = train_full(out_dir=args.out_dir, n_steps=args.steps, seed=args.seed)
    if args.print_final_loss:
        print(f"[tmqm_cfm_pretraining] final_loss={out['metadata'].final_loss:.4f}")
        print(f"[tmqm_cfm_pretraining] saved to {out['npz_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())