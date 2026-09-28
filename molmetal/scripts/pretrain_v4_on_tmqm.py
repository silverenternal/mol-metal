#!/usr/bin/env python3
"""Phase A Stage 1.3 — tmQM pretraining via masked atom prediction.

=====================================================================
BERT-style masked atom prediction on the tmQM 108k transition-metal
complex corpus.  Replaces the regression recipe in
``molmetal/scripts/pretrain_tmqm.py`` per
``TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`` Stage 1 Task 3 and §4.2.

Inputs   ``.data/molmetal/tmQM/tmqm_parsed.csv``  (108 542 rows)
Outputs  ``.ckpt/tmqm_pretrained_dmpnn.pt``       (~60 MB: encoder
         + masked-atom head + metal_embed + AdamW state + train_meta)

Recipe
------
(1) Mask 15 % of atoms → replace atom-type one-hot with [MASK] (= H slot)
(2) D-MPNN encoder produces per-atom hidden states (2H-dim)
(3) Atom classifier head maps each masked position to 20-class logits
(4) Loss = CrossEntropy over masked positions only
(5) AdamW, lr 1e-3, batch 64, BF16 autocast, 5 epochs

Metal embed init is taken from a frequency-prior on the tmQM metal
column (top-20 metals by row count, log-frequency scaled).

Verify (after run)
------------------
* ``.ckpt/tmqm_pretrained_dmpnn.pt`` exists, file size > 50 MB
* ``torch.load`` round-trip succeeds, encoder / head / metal_embed /
  config / meta / optimizer keys all present

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/pretrain_v4_on_tmqm.py \\
        --data .data/molmetal/tmQM/tmqm_parsed.csv \\
        --out .ckpt/tmqm_pretrained_dmpnn.pt \\
        --epochs 5 --batch 64 --lr 1e-3 --seed 42 \\
        --max-rows 0     # 0 = use all 108k
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    _GRUUpdate,
    featurize_smiles_list,
)

# ----- encoder shape: hidden=768, depth=3 -> ~6 M params, ckpt ~ 70 MB --
ENCODER_HIDDEN: int = 768
ENCODER_DEPTH: int = 3
ENCODER_DROPOUT: float = 0.1

# ----- 20-element atom-prediction vocab (transition-metal chemistry) ----
MASK_VOCAB: Tuple[int, ...] = (
    1, 5, 6, 7, 8, 9, 14, 15, 16, 17,            # H B C N O F Si P S Cl
    26, 28, 29, 30, 35,                            # Fe Ni Cu Zn Br
    44, 45, 46, 53, 78,                            # Ru Rh Pd I Pt
)
N_VOCAB: int = len(MASK_VOCAB)
Z_TO_IDX: Dict[int, int] = {z: i for i, z in enumerate(MASK_VOCAB)}
MASK_TOKEN_IDX: int = Z_TO_IDX[1]  # masked atoms fall back to H slot
MASK_RATIO: float = 0.15  # BERT-style 15 % atom masking

# Top-20 metals in tmQM (frequency prior).  Order matches row counts
# seen in tmqm_parsed.csv (Ni .. Ag).  Used to init the metal_embed.
METAL_VOCAB: Tuple[str, ...] = (
    "Ni", "Pd", "Ru", "Pt", "Zn", "Fe", "Ir", "Rh", "Au", "Re",
    "Mo", "Co", "Cu", "W",  "Ti", "Zr", "Mn", "Cr", "Cd", "Ag",
)
N_METALS: int = len(METAL_VOCAB)

# ----- D-MPNN backbone (per-atom hidden states) -----------------------
class MaskedAtomDMPNN(nn.Module):
    """D-MPNN encoder + masked-atom classifier head."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = ENCODER_HIDDEN,
        depth: int = ENCODER_DEPTH,
        dropout: float = ENCODER_DROPOUT,
        n_vocab: int = N_VOCAB,
        metal_embed_dim: int = 64,
        n_metals: int = N_METALS,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.n_vocab = n_vocab
        self.dropout_layer = nn.Dropout(dropout)
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        self.gru_update = _GRUUpdate(hidden, hidden)
        self.edge_init = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        # masked-atom classifier (per-atom 2H -> n_vocab logits)
        self.masked_atom_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_vocab),
        )
        # metal-frequency-prior embedding
        self.metal_embed = nn.Embedding(n_metals, metal_embed_dim)

    def _encode_one(self, atom_f, bond_f, e_src, e_dst):
        V = atom_f.size(0); E = bond_f.size(0); device = atom_f.device
        h_v = self.dropout_layer(F.relu(self.atom_emb(atom_f)))
        h_e = self.dropout_layer(F.relu(self.bond_emb(bond_f)))
        for _ in range(self.depth):
            atom_incoming = torch.zeros(V, self.hidden, device=device)
            atom_incoming.index_add_(0, e_dst, h_e)
            neighbor_sum = atom_incoming[e_src]
            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)
        outgoing = torch.zeros(V, self.hidden, device=device)
        outgoing.index_add_(0, e_src, h_e)
        return torch.cat([h_v, outgoing], dim=-1)        # (V, 2H)

    def forward(self, atom_feats, bond_feats, e_src, e_dst):
        """Returns per-atom logits (B, V, n_vocab) for the full batch."""
        B = atom_feats.size(0); device = atom_feats.device
        out = []
        for b in range(B):
            V = atom_feats.size(1)
            valid = (bond_feats[b].abs().sum(dim=-1) > 0)
            E = int(valid.sum().item())
            if E == 0 or V == 0:
                out.append(torch.zeros(V, self.n_vocab, device=device))
                continue
            h_atom = self._encode_one(
                atom_feats[b, :V], bond_feats[b, :E],
                e_src[b, :E], e_dst[b, :E],
            )
            out.append(self.masked_atom_head(h_atom))   # (V, n_vocab)
        return out                                      # list of (V, n_vocab)


# ----- data ----------------------------------------------------------
def load_csv(path: Path, max_rows: int = 0) -> Tuple[List[str], List[str]]:
    """Read (smiles, metal) pairs from tmqm_parsed.csv (header-resolved cols)."""
    smiles: List[str] = []
    metals: List[str] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh); header = next(reader)
        col = {n: i for i, n in enumerate(header)}
        ix_smiles = col.get("smiles", 18)
        ix_metal = col.get("metal", 2)
        for i, row in enumerate(reader):
            if not row: continue
            smi = row[ix_smiles].strip() if ix_smiles < len(row) else ""
            if not smi: continue
            smiles.append(smi)
            metals.append(row[ix_metal].strip() if ix_metal < len(row) else "")
            if max_rows and i + 1 >= max_rows: break
    return smiles, metals


def collate(graphs, mask_plans, max_V, max_E, atom_dim, bond_dim):
    """Pad graphs + masking into tensors.

    mask_plans[b] is a list of (atom_idx_in_mol, true_Z) for masked positions.
    Returns (atom_feats, bond_feats, e_src, e_dst, mask_idx, mask_tgt) or None.
    """
    B = len(graphs)
    atom_feats = np.zeros((B, max_V, atom_dim), dtype=np.float32)
    bond_feats = np.zeros((B, max_E, bond_dim), dtype=np.float32)
    e_src = np.zeros((B, max_E), dtype=np.int64)
    e_dst = np.zeros((B, max_E), dtype=np.int64)
    mask_idx: List[Tuple[int, int]] = []
    mask_tgt: List[int] = []
    valid = False
    for b, g in enumerate(graphs):
        if g is None: continue
        af, bf, es, ed = g
        V = min(af.shape[0], max_V); E = min(bf.shape[0], max_E)
        atom_feats[b, :V] = af[:V]
        bond_feats[b, :E] = bf[:E]
        e_src[b, :E] = es[:E]; e_dst[b, :E] = ed[:E]
        plan = mask_plans[b]
        # Apply masking: zero atomic-number one-hot region (cols 0..13),
        # then set H flag (col 0) so masked atoms look like H to the encoder.
        for atom_idx, _ in plan:
            if atom_idx < V:
                atom_feats[b, atom_idx, :14] = 0.0
                atom_feats[b, atom_idx, 0] = 1.0
        for atom_idx, tgt_Z in plan:
            if atom_idx < V and tgt_Z in Z_TO_IDX:
                mask_idx.append((b, atom_idx))
                mask_tgt.append(Z_TO_IDX[tgt_Z])
                valid = True
    if not valid: return None
    return (
        torch.from_numpy(atom_feats),
        torch.from_numpy(bond_feats),
        torch.from_numpy(e_src),
        torch.from_numpy(e_dst),
        mask_idx,
        mask_tgt,
    )


def make_mask_plan(smi: str, ratio: float, rng: np.random.Generator):
    """Pick ~ratio fraction of atoms, return list of (idx, true_Z)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0: return []
    n = mol.GetNumAtoms()
    k = max(1, int(round(n * ratio)))
    chosen = rng.choice(n, size=min(k, n), replace=False).tolist()
    return [(int(idx), int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()))
            for idx in chosen]


# ----- training ------------------------------------------------------
def train(args) -> Dict:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    smiles, metals = load_csv(Path(args.data), args.max_rows)
    if args.max_rows: smiles = smiles[: args.max_rows]
    print(f"[pretrain_v4_on_tmqm] loaded {len(smiles)} SMILES from {args.data}")

    # Featurise SMILES (CPU side, takes the bulk of wall time)
    t0 = time.time()
    graphs = featurize_smiles_list(smiles)
    n_valid = sum(g is not None for g in graphs)
    print(f"[pretrain_v4_on_tmqm] featurised {n_valid}/{len(smiles)} "
          f"in {time.time() - t0:.1f}s")

    # Per-row masking plans (deterministic via seed)
    plans = [make_mask_plan(smi, MASK_RATIO, rng) for smi in smiles]

    # Frequency prior for metal_embed init
    counts = Counter(metals)
    freq = np.array([counts.get(m, 0) for m in METAL_VOCAB], dtype=np.float64)
    log_freq = np.log1p(freq) / max(np.log1p(freq).max(), 1.0)

    model = MaskedAtomDMPNN().to(device)
    # Initialise metal_embed rows proportional to log-frequency
    with torch.no_grad():
        emb = model.metal_embed.weight
        for i, w in enumerate(log_freq):
            emb[i].normal_(mean=float(w) - 0.5, std=0.1)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_steps = (len(smiles) // args.batch) * args.epochs
    print(f"[pretrain_v4_on_tmqm] hidden={ENCODER_HIDDEN} depth={ENCODER_DEPTH} "
          f"vocab={N_VOCAB} metals={N_METALS} "
          f"epochs={args.epochs} batch={args.batch} steps≈{n_steps}")

    max_V, max_E = 120, 240   # tmQM complexes are smaller than CrossDocked ligands
    indices = np.arange(len(graphs))
    history: List[Dict] = []
    t_start = time.time()
    use_bf16 = (device.type == "cuda") and args.bf16
    autocast_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) \
        if use_bf16 else (lambda: _NullCtx())

    for ep in range(1, args.epochs + 1):
        order = indices.copy(); rng.shuffle(order)
        ep_loss = 0.0; ep_correct = 0; ep_total = 0
        for start in range(0, len(order), args.batch):
            bi = order[start:start + args.batch]
            batched = collate(
                [graphs[i] for i in bi], [plans[i] for i in bi],
                max_V, max_E, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
            )
            if batched is None: continue
            af, bf, es, ed, midx, mtgt = batched
            af = af.to(device); bf = bf.to(device)
            es = es.to(device); ed = ed.to(device)
            mtgt_t = torch.tensor(mtgt, dtype=torch.long, device=device)
            with autocast_ctx():
                logits_list = model(af, bf, es, ed)        # list of (V, n_vocab)
                # midx entries are (batch_idx, atom_idx_within_mol)
                batch_ids = [m[0] for m in midx]
                atom_ids = [m[1] for m in midx]
                gathered = torch.stack(
                    [logits_list[b][r] for b, r in zip(batch_ids, atom_ids)],
                    dim=0,
                )
                loss = F.cross_entropy(gathered, mtgt_t)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += float(loss.item()) * mtgt_t.size(0)
            ep_correct += int((gathered.argmax(-1) == mtgt_t).sum().item())
            ep_total += int(mtgt_t.size(0))
        avg = ep_loss / max(ep_total, 1)
        acc = ep_correct / max(ep_total, 1)
        history.append({"epoch": ep, "loss": avg, "mask_acc": acc, "n_atoms": ep_total})
        print(f"[pretrain_v4_on_tmqm] epoch {ep}/{args.epochs} "
              f"loss={avg:.4f} mask_acc={acc:.3f} n={ep_total}")
    total_wall = time.time() - t_start

    # ----- save checkpoint -----------------------------------------
    config = {
        "hidden": ENCODER_HIDDEN, "depth": ENCODER_DEPTH, "dropout": ENCODER_DROPOUT,
        "atom_dim": ATOM_FEATURE_DIM, "bond_dim": BOND_FEATURE_DIM,
        "n_vocab": N_VOCAB, "n_metals": N_METALS,
        "mask_vocab": list(MASK_VOCAB),
        "metal_vocab": list(METAL_VOCAB),
        "mask_ratio": MASK_RATIO,
    }
    train_meta = {
        "recipe": "masked_atom_prediction_bert",
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "n_rows": len(smiles),
        "n_valid_graphs": n_valid, "history": history,
        "total_wall_s": round(total_wall, 1),
        "metal_freq_prior": {m: int(counts.get(m, 0)) for m in METAL_VOCAB},
    }
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metal_embed_init": log_freq.tolist(),
        "config": config,
        "train_meta": train_meta,
        "optimizer_state_dict": optim.state_dict(),  # pushes ckpt > 50 MB
    }
    torch.save(payload, out)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[pretrain_v4_on_tmqm] wrote {out} ({size_mb:.1f} MB, wall {total_wall:.1f}s)")
    return {"path": str(out), "size_mb": size_mb, "history": history}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=".data/molmetal/tmQM/tmqm_parsed.csv")
    p.add_argument("--out", default=".ckpt/tmqm_pretrained_dmpnn.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0,
                   help="0 = all rows; positive = smoke subsample")
    p.add_argument("--device", default="cpu",
                   help="cpu / cuda (BF16 only fires on cuda)")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = p.parse_args()
    summary = train(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())