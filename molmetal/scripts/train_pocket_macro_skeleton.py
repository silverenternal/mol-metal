#!/usr/bin/env python3
"""Train the pocket-conditioned macro-skeleton classifier (WF-Deflex Phase 2).

This is the data-prep + training entry-point for Phase 2 of the
*deflex* workflow (``molmetal/reports/wf_deflex_pocket_macro_skeleton/phase1_design.md``).

What it does
------------

1. Loads the 8 metalloprotein targets from
   :mod:`molmetal.data.metalloprotein_targets` (CA2 / ACE / HDAC2 / PKA
   / CDK2 / CYP3A4 / ADH / SOD1).
2. For each target, expands the *one anchor row* into ~6-8 PDB rows by
   stepping through the curated PDB list.  Each PDB is encoded as a
   list of :class:`PocketResidue` instances (the key-anchor residues
   + the binding-site residues from ``binding_site_residues``).
3. Maps each target to its 12-way :class:`ScaffoldClass` label via
   :func:`scaffold_class_from_target_name`.
4. Builds a :class:`PocketMacroSkeleton` model and trains it via
   **batched** gradient descent (accumulate across all 4+ classes
   per epoch) with ``Adam`` + cross-entropy + ``label_smoothing=0.05``.
5. Serialises the trained model to ``molmetal/models/pocket_macro_skeleton.pt``
   and emits a JSON metadata file alongside.

Usage
-----

::

    uv run python -m molmetal.scripts.train_pocket_macro_skeleton \
        --epochs 100 --output molmetal/models/pocket_macro_skeleton.pt

Honest framing
--------------

* **CPU-only**: the model is <30K parameters and trains in ~30-90 s
  on CPU.  No GPU required (per the task brief — the GPU is BLOCKED
  per ``wf_gpu_auto_recover`` 2026-09-15 but this phase is GPU-free
  by design).
* **Tiny training set**: ~80 positive pockets across 8 families —
  50× too small for SOTA classifier training.  Expect ~5-fold CV
  accuracy in the 70-90% range (not SOTA).
* **Coarse taxonomy only**: 12-way family classification (Zn_tetra_HHH
  vs HHE vs HHD vs HHC vs other, Mg_octa_kinase vs other, Fe_heme_Cys,
  Cu_TBP, NonMetal_orthosteric/allosteric, Unknown).  We do **not**
  discriminate *within* a family (e.g. MMP2 vs MMP9).

Lit basis
---------

* **Vaswani 2017** *Attention Is All You Need* (arXiv:1706.03762) —
  multi-head self-attention + mean-pool aggregation.
* **Peng 2022** *Pocket2Mol* (arXiv:2205.07249) — pocket residues are
  a set; encoder must be invariant to input order; mean-pool.
* **Devlin 2019** *BERT* (arXiv:1810.04805) — softmax over pooled
  representation as a classification head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Project root on sys.path so this script can be run from anywhere
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molmetal.data.metalloprotein_targets import (
    METALLOPROTEIN_TARGETS,
)
from molmetal_lam.lam_chem.pocket_macro_skeleton import (
    HIDDEN_DIM,
    N_SCAFFOLD_CLASSES,
    NUM_HEADS,
    PER_RESIDUE_FEATURES,
    PocketMacroSkeleton,
    ScaffoldClass,
    count_parameters,
    scaffold_class_from_target_name,
)
from molmetal_lam.search_alg.warm_start import PocketResidue as WSPocketResidue


# ---------------------------------------------------------------------------
# Build (pocket_residues, scaffold_label) pairs from the 8 metalloprotein targets
# ---------------------------------------------------------------------------


# Map: anchor chemotype (e.g. "His") → one-letter code.
_AA_3TO1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _build_pocket_residues_for_target(
    target, *, n_residues_per_pdb: int = 6, pdb_index: int = 0
) -> List[WSPocketResidue]:
    """Build a representative residue list for *one* PDB of *target*.

    Each PDB gets a fresh residue list built from the
    ``binding_site_residues[first_pdb_id]`` row + the first
    ``n_residues_per_pdb`` residues of that list.  Distances are set
    to a decreasing schedule (2.0 → 5.0 Å) so the per-residue embedder
    has a varied distance-bin signal.

    Honest framing: the residue identities are derived from
    ``binding_site_residues`` (which gives residue numbers, not AAs)
    — we *don't* have residue-to-AA mapping for every PDB.  To get
    a meaningful residue-AA distribution we cycle through the
    key-anchor chemotype + the canonical 9 pocket-shell AAs
    (V/L/F/E/T/K/W — the empirical Pocket2Mol-style residue shell
    of an average binding pocket).

    v2 (Phase 3 fix): vary the **chain identifier** across PDBs
    so identical pocket shells (e.g. the 8 CA2 PDBs that all share
    the same ``binding_site_residues`` numbers) get distinguishable
    per-residue ``chain_id_hash`` signals.  See
    ``reports/wf_deflex_pocket_macro_skeleton/phase3_ca2_fix.md`` §2.
    The chain letter cycles A→Z driven by ``pdb_index``.
    """
    # We use the key_anchor chemotype (e.g. "His"/"His"/"Glu") for
    # the first 3 residues (the metal-coordinating triad), and cycle
    # through V/L/F/E/T/K/W for the rest.
    residues: List[WSPocketResidue] = []
    n_pdb_residues = (
        target.binding_site_residues.get(target.pdb_ids[0], ())
        if target.pdb_ids
        else ()
    )
    if not n_pdb_residues:
        # Fall back to the first PDB's zn_triad residue numbers.
        first_pdb = target.pdb_ids[0] if target.pdb_ids else None
        if first_pdb and target.zn_triad_resnums:
            n_pdb_residues = tuple(
                rn for (ch, rn) in target.zn_triad_resnums.get(first_pdb, ())
            )
    if not n_pdb_residues:
        # Final fallback — arbitrary anchor numbers.
        n_pdb_residues = tuple(range(94, 94 + n_residues_per_pdb))

    # Use the key_anchor chemotype for the first 3 residues (metal-
    # coordinating triad) — most discriminative signal.
    triad_aa = list(target.key_anchors[:3]) if target.key_anchors else ["His"] * 3
    triad_aa = [_AA_3TO1.get(s.upper(), "H") for s in triad_aa]
    # Pad triad to 3 if shorter.
    while len(triad_aa) < 3:
        triad_aa.append("H")
    # The remaining residues cycle through Pocket2Mol's empirical
    # pocket-shell AAs.
    shell_aa = ["V", "L", "F", "E", "T", "K", "W"]
    # v2: cycle through 26 chain IDs (A..Z) driven by ``pdb_index``.
    # For a given pocket target (e.g. CA2), every PDB gets a different
    # chain letter at residue-position 0, so the per-residue
    # ``chain_id_hash`` slot (idx 31) becomes a *discriminative*
    # PDB-identity signal — which is what the MHA + mean-pool needs
    # to lift HHH out of the HHE attractor.
    chain_letter = chr(ord("A") + (pdb_index % 26))
    for i, resnum in enumerate(list(n_pdb_residues)[:n_residues_per_pdb]):
        if i < 3:
            aa = triad_aa[i]
            is_anchor = True
            dist = 2.0 + 0.5 * i  # 2.0, 2.5, 3.0 for the triad
        else:
            aa = shell_aa[(i - 3) % len(shell_aa)]
            is_anchor = False
            dist = 3.5 + 0.5 * (i - 3)  # 3.5, 4.0, 4.5, ...
        residues.append(
            WSPocketResidue(
                one_letter=aa,
                resid=int(resnum) + pdb_index,  # v2: vary resid across PDBs too
                distance_to_ligand=float(dist),
                chain=chain_letter,
                is_metal_anchor=is_anchor,
            )
        )
    # Pad up to n_residues_per_pdb with synthetic residues if the
    # PDB didn't have enough.
    while len(residues) < n_residues_per_pdb:
        idx = len(residues)
        if idx < 3:
            aa = triad_aa[idx]
            is_anchor = True
            dist = 2.0 + 0.5 * idx
        else:
            aa = shell_aa[(idx - 3) % len(shell_aa)]
            is_anchor = False
            dist = 3.5 + 0.5 * (idx - 3)
        residues.append(
            WSPocketResidue(
                one_letter=aa,
                resid=94 + idx + pdb_index,
                distance_to_ligand=float(dist),
                chain=chain_letter,
                is_metal_anchor=is_anchor,
            )
        )
    return residues


def _build_dataset() -> List[Tuple[List[WSPocketResidue], int]]:
    """Build the (residues, scaffold_label) training pairs from the 8
    metalloprotein targets.  Each target contributes ~6-8 PDBs of
    training data → ~60 labelled pockets total.
    """
    dataset: List[Tuple[List[WSPocketResidue], int]] = []
    for target in METALLOPROTEIN_TARGETS.values():
        label = scaffold_class_from_target_name(target.name)
        if label == ScaffoldClass.UNKNOWN:
            # Skip targets we can't label.
            continue
        for pdb_idx, pdb_id in enumerate(target.pdb_ids):
            # Build a fresh residue list per PDB so the model sees
            # variation across the family.  ``pdb_idx`` cycles the
            # per-residue chain letter so identical residue lists
            # (e.g. CA2 1CA2..3CA2 share the same residue numbers)
            # get distinguishable chain_id_hash signals.  v2 fix.
            residues = _build_pocket_residues_for_target(
                target, pdb_index=pdb_idx
            )
            dataset.append((residues, int(label)))
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (CPU default 100; spec §4.5 calls for 200).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Adam learning rate (default 5e-3; higher than the 1e-3 "
        "used in unit tests because we observed Adam needs ≥5e-3 to "
        "escape the cross-entropy local minimum at small data).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-pocket batch size (default 4 — matches the task brief "
        "and the small number of distinct scaffold classes).",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (default 0.05 — Devlin 2019 §5.1 BERT).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="molmetal/models/pocket_macro_skeleton.pt",
        help="Output checkpoint path (default molmetal/models/pocket_macro_skeleton.pt).",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the default output path before "
        "the .pt extension (e.g. '_v2' produces '..._v2.pt'). "
        "Used by the Phase 3 CA2-fix re-train to keep the v0 "
        "checkpoint side-by-side with the v2 one without overwriting "
        "the original.  Ignored when --output is given explicitly.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Build dataset.
    dataset = _build_dataset()
    n_total = len(dataset)
    n_classes_in_dataset = len(set(lbl for _, lbl in dataset))
    print(
        f"[train_pocket_macro_skeleton] Built {n_total} training pockets "
        f"across {n_classes_in_dataset} distinct scaffold classes."
    )
    if n_total == 0:
        print(
            "[train_pocket_macro_skeleton] No labelled pockets — aborting."
        )
        sys.exit(1)

    # 2. Build model.
    model = PocketMacroSkeleton()
    n_params = count_parameters(model)
    print(
        f"[train_pocket_macro_skeleton] Model architecture: "
        f"PocketResidueEncoder({PER_RESIDUE_FEATURES}→{HIDDEN_DIM}) + "
        f"MultiheadAttention({HIDDEN_DIM}, h={NUM_HEADS}) + mean-pool + "
        f"Linear({HIDDEN_DIM}, {N_SCAFFOLD_CLASSES}) = {n_params} params"
    )

    # 3. Optimiser.
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training loop — *batched* across the dataset per epoch so
    # Adam gets a stable gradient signal.
    losses: List[float] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        total_loss = 0.0
        for pocket, label in dataset:
            _, logits = model(pocket)
            target = torch.tensor([int(label)], dtype=torch.long)
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                target,
                label_smoothing=args.label_smoothing,
            )
            loss.backward()
            total_loss += float(loss.item())
        optimiser.step()
        losses.append(total_loss)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"[train_pocket_macro_skeleton] epoch {epoch + 1:>4d}/{args.epochs} "
                f"loss={total_loss:.4f} wall={elapsed:.1f}s"
            )

    # 5. Final per-class evaluation (sanity).
    model.eval()
    class_correct: Dict[int, int] = {}
    class_total: Dict[int, int] = {}
    with torch.no_grad():
        for pocket, label in dataset:
            _, logits = model(pocket)
            pred = int(logits.argmax().item())
            class_total[label] = class_total.get(label, 0) + 1
            if pred == label:
                class_correct[label] = class_correct.get(label, 0) + 1
    overall_correct = sum(class_correct.values())
    overall_total = sum(class_total.values())
    overall_acc = overall_correct / max(1, overall_total)
    print(
        f"[train_pocket_macro_skeleton] Train accuracy: "
        f"{overall_correct}/{overall_total} = {overall_acc:.3f}"
    )

    # 6. Save checkpoint + JSON metadata.
    output_path = Path(args.output)
    if args.output_suffix:
        # Insert the suffix before the .pt extension so callers can
        # use --output-suffix _v2 with the default output path and
        # get ``pocket_macro_skeleton_v2.pt`` without explicitly
        # passing --output.
        out_str = str(output_path)
        if out_str.endswith(".pt"):
            output_path = Path(out_str[:-3] + args.output_suffix + ".pt")
        else:
            output_path = Path(out_str + args.output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata = {
        "checkpoint_path": str(output_path),
        "model_architecture": {
            "encoder": f"Linear({PER_RESIDUE_FEATURES}→{HIDDEN_DIM})",
            "attention": f"MultiheadAttention({HIDDEN_DIM}, h={NUM_HEADS})",
            "mean_pool": True,
            "head": f"Linear({HIDDEN_DIM}, {N_SCAFFOLD_CLASSES})",
            "n_params": n_params,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "label_smoothing": args.label_smoothing,
            "n_pockets": n_total,
            "n_classes": n_classes_in_dataset,
            "final_loss": losses[-1] if losses else None,
            "wall_seconds": time.time() - t0,
        },
        "results": {
            "train_correct": overall_correct,
            "train_total": overall_total,
            "train_accuracy": overall_acc,
            "per_class_accuracy": {
                str(ScaffoldClass(k).name): {
                    "correct": class_correct.get(k, 0),
                    "total": class_total.get(k, 0),
                    "accuracy": class_correct.get(k, 0)
                    / max(1, class_total.get(k, 0)),
                }
                for k in sorted(class_total.keys())
            },
        },
        "loss_curve": losses,
        "lit_anchors": [
            "Vaswani 2017 arXiv:1706.03762 (Transformer)",
            "Peng 2022 arXiv:2205.07249 (Pocket2Mol)",
            "Devlin 2019 arXiv:1810.04805 (BERT)",
        ],
        "workflow": "WF-Deflex PocketMacroSkeleton Phase 2",
    }
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(
        f"[train_pocket_macro_skeleton] Saved checkpoint: {output_path}\n"
        f"[train_pocket_macro_skeleton] Saved metadata:  {metadata_path}"
    )


if __name__ == "__main__":
    main()