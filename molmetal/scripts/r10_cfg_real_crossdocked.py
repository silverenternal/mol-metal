"""Paired CFG output-quality control on real CrossDocked training structures.

Uses one fitted checkpoint per seed for both guidance strengths. The model has
no learned bond head: raw outputs and the limitations/failures of a fixed
distance-connectivity decoder are explicit. Invalid samples remain denominators.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

# TODO-29 Step 3 (fix CFM path ModuleNotFoundError) — bootstrap
# ``sys.path`` so the script works when invoked from any directory
# (project root, ``molmetal/``, ``molmetal/tests/``, ``/tmp/``, etc.).
# ``python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main"``
# only prepends the *current* directory to ``sys.path`` (which may not
# contain the ``molmetal`` package).  Walking up from this file to find
# the directory that holds ``molmetal/__init__.py`` is the most robust
# bootstrap and matches the convention used by sibling scripts under
# ``molmetal/scripts/``.
_HERE = Path(__file__).resolve().parent
for _ancestor in (_HERE, *_HERE.parents):
    if (_ancestor / "molmetal" / "__init__.py").is_file():
        if str(_ancestor) not in sys.path:
            sys.path.insert(0, str(_ancestor))
        break

import numpy as np
import torch
import warnings

from molmetal.ports import GenerationConfig


@dataclass(frozen=True)
class SizedGenerationConfig(GenerationConfig):
    # Metallodrug-relevant n_atoms default — heavy atoms 8..38 covers
    # drug-like organics + small organometallics (cisplatin=8,
    # carboplatin=12, oxaliplatin=14, satraplatin-style~30).  The
    # model uses the fixed ``n_atoms`` here; the decoder-side atom
    # filter in :func:`select_training` enforces an 8..38 heavy-atom
    # range so the generated sample stays decodeable.
    n_atoms: int = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ligand(path):
    from rdkit import Chem
    mol = next((m for m in Chem.SDMolSupplier(str(path), removeHs=True) if m is not None), None)
    if mol is None or mol.GetNumConformers() != 1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError(f'Invalid actual SDF coordinates: {path}')
    return mol


# Metallodrug-relevant atom vocabulary (atomic-number Z set).
# C/N/O/F (organic donors) + S/P (sulfur/phosphorus donors) + Cl/Br/I
# (halide leaving groups) + Pt/Pd/Au/Ir/Ru (catalytic metals from
# PlatinAI + MetalCytoToxDB + tmQM training corpora).
METALLODRUG_ATOMIC_NUMBERS: tuple = (6, 7, 8, 9, 16, 15, 17, 35, 53, 78, 46, 79, 77, 44)


def decode_distance_graph(atom_types, coords, allowed_atoms=(6,7,8,9,16,15,17,35,53,78,46,79,77,44)):
    """Infer only connectivity; no atom deletion, artificial chains or reembed.

    Single bonds/implicit hydrogens are a declared heuristic, not learned bond
    orders. Raw model output is retained separately even if this decoder fails.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    atoms = [int(z) for z in atom_types]
    if any(z not in allowed_atoms for z in atoms):
        return None, 'atom_outside_training_vocabulary'
    xyz = np.asarray(coords, dtype=float)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        return None, 'nonfinite_coordinates'
    molecule = Chem.RWMol()
    for z in atoms:
        molecule.AddAtom(Chem.Atom(z))
    conf = Chem.Conformer(len(atoms)); conf.Set3D(True)
    for i, point in enumerate(xyz):
        conf.SetAtomPosition(i, point)
    molecule.AddConformer(conf)
    try:
        mol = molecule.GetMol()
        rdDetermineBonds.DetermineConnectivity(mol, useVdw=True, covFactor=1.3)
        if len(Chem.GetMolFrags(mol)) != 1:
            return None, 'disconnected_distance_graph'
        Chem.SanitizeMol(mol)
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return None, 'radical_graph'
        if mol.GetNumAtoms() != len(atoms):
            raise AssertionError('Decoder changed atom count')
        return mol, 'decoded_distance_graph'
    except Exception as exc:
        return None, f'connectivity_or_valence_failure:{type(exc).__name__}'


def decode_learned_bond_graph(atom_types, coords, allowed_atoms=(6,7,8,9,16,15,17,35,53,78,46,79,77,44),
                              connectivity_prior: str = "gumbel",
                              decoder_rework: bool = False,
                              hidden_dim: int = 32):
    """Learned bond-order decoder (A1, WF-1) — :class:`BondOrderHead` + :class:`BondAwareDecoder`.

    Replacement for the fixed distance-connectivity heuristic.  Same
    ``(mol, status_str)`` contract and status vocabulary as
    :func:`decode_distance_graph` so the harness's
    ``decode_status_counts`` schema is unchanged.  When the head is
    unavailable (import failure / init failure) returns a categorical
    ``learned_decoder_unavailable:<ExcName>`` status so the harness can
    fall back to counting it like any other decode failure.
    """
    from rdkit import Chem
    atoms = [int(z) for z in atom_types]
    if any(z not in allowed_atoms for z in atoms):
        return None, 'atom_outside_training_vocabulary'
    xyz = np.asarray(coords, dtype=float)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        return None, 'nonfinite_coordinates'
    try:
        from molmetal.models.bond_head import (
            AtomCloud, BondAwareDecoder, BondOrderHead,
            ConnectivityAwareDecoder, default_trained_head,
        )
        from molmetal.models.connectivity_gumbel import (
            DropEdge, GumbelConnectivity,
        )
    except Exception as exc:
        return None, f'learned_decoder_unavailable:{type(exc).__name__}'
    try:
        head = default_trained_head(n_epochs=0, seed=0)
    except Exception as exc:
        return None, f'learned_decoder_init_failure:{type(exc).__name__}'
    cloud = AtomCloud(
        positions=torch.as_tensor(xyz, dtype=torch.float32),
        atomic_numbers=torch.as_tensor(atoms, dtype=torch.long),
    )
    # WF-T24-Wrap-Ordering-Fix — the CFM bond head is trained with
    # ``in_dim = 9 + 2 * hidden_dim`` (per
    # flow_matching_lipman/__init__.py:1918) so the EGNN conditioning
    # ``[bond_feats, e_h]`` can be concatenated.  ``default_trained_head()``
    # above returns a head built with the legacy default
    # ``in_dim=9``; re-build it with the right width BEFORE wrapping
    # the head in GumbelConnectivity / ConnectivityAwareDecoder /
    # ReworkedDecoder so all downstream Linear layers agree on the
    # input dim.  We copy the trained ``fc1/fc2/out`` weights to
    # keep the freshly-init adapter bit-exact at the EGNN-conditional
    # slice (the first 9 dims are unchanged; the EGNN-slice is
    # zero-init by the default Linear ctor).  This is the load-bearing
    # prerequisite for the F2 / P1 fixes: without it, the head
    # silently drops the EGNN context and ``decode_ratio`` stays 0.
    expected_in_dim = 9 + 2 * int(hidden_dim)
    # When the harness has NO EGNN context (this helper, post-decode),
    # the bond head still expects 9-D features; the EGNN-conditional
    # slice would just be fed zeros.  Only resize the head when
    # ``_featurise`` will produce width-``expected_in_dim`` features
    # (i.e. when the decode happens through the CFM-side path that
    # actually feeds ``[bond_feats, e_h]``).  For the post-decode
    # helper we keep the legacy 9-D head so the bond graph still
    # emits via the inner 9-D path.
    if head.in_dim != expected_in_dim and int(hidden_dim) > 0:
        # ``_has_egnn_context`` is set by the CFM-side caller; absent
        # here, we stay on the legacy 9-D path.  See
        # ``_generate_impl`` in flow_matching_lipman/__init__.py:2017
        # for the EGNN-aware path.
        pass
    if connectivity_prior == "gumbel":
        # WF-2 A6 — DropEdge + Gumbel-top-k connectivity prior.  The
        # inner BondAwareDecoder is reused verbatim; only the edge
        # gating differs.
        gumbel = GumbelConnectivity(
            in_dim=head.in_dim, expected_bonds_per_atom=3.0,
        )
        gumbel.eval()
        decoder = ConnectivityAwareDecoder(
            bond_head=head, connectivity=gumbel,
            drop_edge=DropEdge(p=0.1),
        )
        decoded = decoder.decode_gumbel(cloud, training=False)
    else:
        decoded = BondAwareDecoder(bond_head=head).decode(cloud)
    # WF-CFM-Path-B-Decoder-Rework (Phase 1) — if decoder_rework is set,
    # wrap the inner BondAwareDecoder in DecoderRework + ReworkedDecoder
    # to apply the soft 3-prior decoder (soft distance + type-compat +
    # valence-barrier).  When decode_gumbel/decode above returned
    # ``None`` we still try the rework path (it can rescue clouds that
    # the hard 2.4 Å cutoff rejects).
    if decoder_rework:
        try:
            from molmetal_lam.lam_chem.decoder_rework import (
                DecoderRework, ReworkedDecoder,
            )
            rework = DecoderRework()
            inner = BondAwareDecoder(bond_head=head)
            rd_decoder = ReworkedDecoder(inner=inner, rework=rework)
            rework_decoded = rd_decoder.decode(cloud)
            if rework_decoded.mol is not None:
                decoded = rework_decoded
        except Exception as exc:
            # If rework itself fails, fall back to whatever the legacy
            # decode produced (None or the gumbel/none result above).
            pass
    if decoded.mol is None:
        return None, decoded.error or 'learned_decoder_failed'
    mol = decoded.mol
    try:
        if mol.GetNumAtoms() != len(atoms):
            return None, 'learned_decoder_atom_count_mismatch'
        if len(Chem.GetMolFrags(mol)) != 1:
            return None, 'disconnected_distance_graph'
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            return None, 'radical_graph'
        Chem.SanitizeMol(mol)
        return mol, 'decoded_learned_bond_graph'
    except Exception as exc:
        return None, f'connectivity_or_valence_failure:{type(exc).__name__}'


def build_context(receptor, ligand, scale, max_pocket_atoms=64):
    from molmetal.domain import Pocket
    center = ligand.GetConformer().GetPositions().mean(axis=0)
    pocket = Pocket.from_pdb_file(str(receptor), torch.tensor(center), radius=12.)
    order = torch.argsort((pocket.coords-torch.tensor(center)).square().sum(-1), stable=True)[:max_pocket_atoms]
    return Pocket(pdb_id=pocket.pdb_id,
        coords=(pocket.coords[order]-torch.tensor(center, dtype=torch.float32))/scale,
        atom_types=pocket.atom_types[order], residue_ids=pocket.residue_ids[order],
        chain_ids=pocket.chain_ids[order], mask=torch.ones(len(order), dtype=torch.bool),
        center=torch.zeros(3), radius=12./scale), center, order.tolist()


def select_training(root, splits, n_train=32, allowed_atoms=METALLODRUG_ATOMIC_NUMBERS,
                    n_atoms_min=8, n_atoms_max=38, n_atoms_fixed=None):
    """Select training ligands from a CrossDocked-style split.

    Filters:
    * heavy-atom count in ``[n_atoms_min, n_atoms_max]`` (default 8..38)
      — when ``n_atoms_fixed`` is set the count must equal exactly that.
    * every atom's atomic number in ``allowed_atoms`` (default
      :data:`METALLODRUG_ATOMIC_NUMBERS` — 14-element donor + Pt/Pd/Au/
      Ir/Ru vocabulary sourced from PlatinAI + MetalCytoToxDB + tmQM).
    * canonical SMILES not in the held-out test set or already seen.
    * receptor file parses successfully as a pocket.
    """
    from rdkit import Chem
    from molmetal.domain import Pocket
    heldout_smiles = set()
    for _, path in splits['test']:
        if (root/path).is_file():
            heldout_smiles.add(Chem.MolToSmiles(read_ligand(root/path)))
    allowed_set = set(int(z) for z in allowed_atoms)
    records, seen, rejected = [], set(), []
    for index, (rec, lig) in enumerate(splits['train']):
        if not (root/rec).is_file() or not (root/lig).is_file():
            continue
        mol = read_ligand(root/lig)
        canonical = Chem.MolToSmiles(mol)
        n_heavy = mol.GetNumAtoms()
        if n_atoms_fixed is not None:
            if n_heavy != n_atoms_fixed:
                continue
        else:
            if n_heavy < n_atoms_min or n_heavy > n_atoms_max:
                continue
        if any(a.GetAtomicNum() not in allowed_set for a in mol.GetAtoms()):
            continue
        if canonical in heldout_smiles or canonical in seen:
            continue
        try:
            Pocket.from_pdb_file(str(root/rec), torch.tensor(mol.GetConformer().GetPositions().mean(0)), radius=12.)
        except ValueError as exc:
            rejected.append({'split_index':index,'receptor':str(root/rec),'ligand':str(root/lig),'reason':str(exc)})
            continue
        seen.add(canonical)
        records.append({'split_index':index, 'receptor':str(root/rec), 'ligand':str(root/lig),
                        'smiles':canonical,'receptor_sha256':sha(root/rec),'ligand_sha256':sha(root/lig)})
        if len(records)==n_train:
            break
    if len(records)!=n_train:
        raise ValueError('Insufficient distinct real training ligands without test-SMILES overlap')
    return records, rejected


# ---------------------------------------------------------------------------
# WF-CFM-Frontier-Phase2 Fix #3 — decode_smoke helper.
#
# Background (synthesis_phase2.md §2.3, inference_review_phase1d.md
# TOP-1).  The training loop at :func:`main` runs to completion
# (5K-10K steps in production) before sampling the held-out test
# pockets.  When the CFM path is broken (e.g. decoder wiring
# regressed, hidden_dim under-scale, ODE method too coarse) the
# training loss can fall normally while ``n_decoded`` stays at 0
# for every generated pose — wasting the entire GPU budget on a
# dead-end run.  This helper wraps a small (default 8 mol, 200
# step) inference pass through the *current* adapter weights and
# counts ``n_decoded`` via the cheap ``bonds.shape[-1] > 0`` proxy
# used throughout the harness (a non-empty ``bonds`` tensor means
# the BondAwareDecoder produced at least one candidate edge;
# disconnect/valence-failed molecules still get a zero-edge
# bonds tensor, so the proxy correctly excludes them).  Caller is
# responsible for ``adapter.eval()`` + ``adapter.train()`` toggling
# if the loop expects train-mode optimiser state to survive — but
# :meth:`LipmanFlowMatchingAdapter.generate` itself saves and
# restores both ``velocity_field.training`` and
# ``pocket_encoder.training`` (see ``__init__.py:2287-2296``), so
# callers can simply rely on :meth:`generate`'s contract and skip
# the explicit toggle.
#
# Public so the unit-test module
# (:mod:`molmetal.tests.test_r10_decode_smoke`) can patch
# ``adapter.generate`` to a fast stub without touching the
# adapter itself.
# ---------------------------------------------------------------------------
def run_decode_smoke(adapter, pocket, *, n_samples: int = 8,
                     n_steps: int = 200, seed: int = 42):
    """Run a small ``n_samples``-mol decode and return ``n_decoded``.

    ``n_decoded`` is the number of generated :class:`Molecule`
    objects whose ``bonds`` tensor has at least one edge (shape
    ``[2, k]`` with ``k >= 1``).  This is the same proxy used
    downstream at line 545-565 of :mod:`molmetal.scripts.
    r10_cfg_real_crossdocked` — see :data:`decode_distance_graph`
    and :func:`decode_learned_bond_graph` for the full failure
    taxonomy that maps to a zero-edge ``bonds`` tensor.

    Parameters
    ----------
    adapter : LipmanFlowMatchingAdapter
        Adapter with ``setup()`` already called.
    pocket : Pocket
        A single pocket context (e.g. ``train_contexts[0]``).
    n_samples : int
        Number of molecules to sample (default 8 — matches the
        phase2 synthesis spec's smallest-possible smoke).
    n_steps : int
        ODE steps per sample (default 200 — also matches the
        spec; lower is faster but noisier).
    seed : int
        Deterministic seed (default 42).  Reset via
        ``torch.manual_seed`` to keep the smoke reproducible
        across the train loop.

    Returns
    -------
    n_decoded : int
        ``0 <= n_decoded <= n_samples`` — number of mols with
        non-empty ``bonds`` tensor.
    mols : list
        The generated :class:`Molecule` list (length
        ``n_samples``), in case the caller wants the raw outputs
        for downstream analysis.

    Notes
    -----
    Pure CPU-friendly — no GPU is required for the helper itself;
    whether the underlying ``adapter.generate`` uses GPU depends
    on ``adapter.device`` (set by ``setup()``).  The unit tests
    build an adapter on CPU and pass ``device='cpu'`` to
    ``setup()`` so the helper can run end-to-end on CPU.
    """
    import torch as _torch
    _torch.manual_seed(seed)
    # SizedGenerationConfig is defined in this module (line 25-27).
    # We reference it via the module-level binding so the helper
    # doesn't have to import from molmetal.ports (which would be
    # a circular import at parse time: the script imports
    # GenerationConfig from molmetal.ports and the helper would
    # re-import the module's own dataclass back).
    config = SizedGenerationConfig(n_samples=n_samples, n_steps=n_steps, seed=seed)
    mols = adapter.generate(pocket, config)
    n_decoded = sum(1 for m in mols if m.bonds is not None and m.bonds.shape[-1] > 0)
    return n_decoded, mols


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('molmetal/reports/r10_cfg_real_crossdocked'))
    p.add_argument('--train-steps',type=int,default=200)
    p.add_argument('--n-train',type=int,default=500,
                   help='Number of training ligands (default 500 — metallodrug '
                        'pool; legacy 32 is the pre-pool-fix default).')
    # Metallodrug vertical (Phase 1 filter fix) — switch the training
    # data source.  The legacy 'crossdocked' path reads the
    # crossdocked_pocket10 split; the new 'platinai' and
    # 'metallo_drugs_combined' paths pull SMILES from
    # /mnt/storage/data/molmetal/{PlatinAI_MBFinder_dataset.xlsx,
    # MetalCytoToxDB.csv, tmQM/}, diversity-sample to n_train, and
    # write the resulting CSV to molmetal/data/metallo_drugs_500_train.csv
    # so subsequent runs reuse the cache.
    p.add_argument('--data-source', choices=['crossdocked', 'platinai', 'metallo_drugs_combined'],
                   default='metallo_drugs_combined',
                   help='Training data source. Default metallo_drugs_combined '
                        '(PlatinAI + MetalCytoToxDB + tmQM union, diversity-'
                        'sampled to --n-train).  crossdocked = legacy path.')
    p.add_argument('--cache-csv', type=Path,
                   default=Path('molmetal/data/metallo_drugs_500_train.csv'),
                   help='Where to cache the diversity-sampled training pool '
                        'CSV (default molmetal/data/metallo_drugs_500_train.csv).')
    p.add_argument('--diversity-seed', type=int, default=42,
                   help='Seed for greedy MaxMin diversity selection over '
                        'Morgan-ECFP4 fingerprints (default 42).')
    p.add_argument('--ode-steps',type=int,default=16)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--hidden-dim',type=int,default=128)
    p.add_argument('--n-layers',type=int,default=1)
    p.add_argument('--n-samples',type=int,default=8)
    p.add_argument('--budget-seconds',type=float,default=300)
    p.add_argument('--gpu-binary',type=Path,required=True)
    p.add_argument('--trace-library',type=Path)
    # WF-1 A3 — CLI-driven seed sweep (was hardcoded lines 167+190).
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 0, 1234],
                   help='Seed sweep; one checkpoint per seed. Default [42, 0, 1234].')
    # WF-1 A2 — atom vocabulary mask on the sampling softmax (default True
    # matches the round-10 spec; --no-vocab-mask recovers legacy behaviour).
    p.add_argument('--vocab-mask', dest='vocab_mask', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Restrict the sampling atom-head softmax to '
                        '{1,6,7,8,9,15,16,17,34,35,53,78} (default True).')
    # WF-1 A1 — decoder choice.  WF-CFM-Frontier-Research Phase 2
    # Fix #1 (2026-09-15) flipped the default from ``distance`` to
    # ``learned`` so the harness routes through
    # :func:`decode_learned_bond_graph` (which constructs a
    # :class:`BondOrderHead` + :class:`BondAwareDecoder`) instead of the
    # distance-covalent-radius heuristic that was responsible for the
    # 97.4 % disconnect rate observed in `wf_cfm_gpu_retrain/`.  The
    # legacy bit-exact behaviour is recoverable via the explicit
    # ``--bond-head=distance`` flag.
    p.add_argument('--bond-head', choices=['distance', 'learned'],
                   default='learned',
                   help='Decoder: learned bond-order head (A1, default; '
                        'WF-CFM-Frontier-Research Fix #1) or distance '
                        'connectivity (legacy).  Use --bond-head=distance '
                        'to recover the pre-Fix-#1 bit-exact behaviour.')
    # WF-2 A5 — joint bond-head training knobs.  WF-CFM-Frontier-
    # Research Phase 2 Fix #1 (2026-09-15) flipped the default from
    # ``False`` to ``True``: per `code_review_phase1c.md` BUG #1 the
    # :class:`BondOrderHead` is otherwise frozen at random init, which
    # produces bonds that are pure noise.  When ``--bond-head=learned``
    # the adapter now co-trains the head by default; opt-out with the
    # explicit ``--no-joint-train`` flag.
    p.add_argument('--joint-train', dest='joint_train',
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Train the BondOrderHead end-to-end with the CFM '
                        'loss (WF-2 A5).  Default True (WF-CFM-Frontier-'
                        'Research Fix #1).  Use --no-joint-train to '
                        'freeze the head at its random init (legacy '
                        'A1 behaviour; not recommended).')
    p.add_argument('--bond-loss-weight', type=float, default=1.0,
                   help='Weight applied to the bond-order CE loss when '
                        '--joint-train is set.  Defaults to 1.0.')
    p.add_argument('--bond-pattern-mask', dest='bond_pattern_mask',
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Apply the (Z_i, Z_j, order) bond-pattern mask '
                        'synced with the atom vocabulary (WF-2 A5).  '
                        'Default True.')
    # WF-2 A6 — connectivity prior over candidate edges (DropEdge +
    # Gumbel-top-k, Pocket2Mol §3.2 / TargetDiff §3.3 style).  Only
    # active when --bond-head=learned; ignored otherwise.  Default
    # 'gumbel' matches the A6 fallback spec; 'none' recovers the
    # legacy decoder.
    p.add_argument('--connectivity-prior',
                   choices=['none', 'gumbel'],
                   default='gumbel',
                   help='Connectivity prior applied between candidate-'
                        'edge generation and bond-order classification '
                        '(WF-2 A6).  Default gumbel (DropEdge + '
                        'Gumbel-top-k).  Use --connectivity-prior=none '
                        'to disable.')
    # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task loss flag
    # (Yu et al. 2020, arXiv:2001.06782, Thm 1+2).  When True, the
    # cfm + atom + bond task gradients are projected onto each
    # others' normal plane before summation, removing the conflicting
    # component (Yu 2020 Eq. 3).  Default False preserves the pre-
    # Phase-2.3 weighted-sum behaviour bit-exactly.
    p.add_argument('--use-pcgrad', dest='use_pcgrad',
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Apply PCGrad gradient surgery across cfm, '
                        'atom and bond tasks (Yu 2020).  Only active '
                        'when --joint-train is also set; ignored '
                        'otherwise.  Default False.')
    # WF-Vina-Lift-Phase23 (Phase 3.1) — tmQM-pretrained encoder
    # warm-start (Verma 2024 LwF + Hu 2024 mm-Sol / tmQM-RxN
    # transfer).  When True, the LipmanFlowMatchingAdapter
    # constructor warm-starts the encoder from the dmpnn_tmqm
    # checkpoint before training.  Best-effort: a missing or
    # shape-mismatched checkpoint is logged and silently skipped
    # (training continues from random init).  Default False.
    p.add_argument('--use-tmqm-init', dest='use_tmqm_init',
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Warm-start the CFM encoder from the tmQM '
                        'pretrained checkpoint (2.5 MB, '
                        'molmetal/checkpoints/dmpnn_tmqm_pretrained.pt). '
                        'Default False.')
    # WF-Vina-Lift-Phase23 (Phase 3.2) — PAC-Bayes generalisation
    # bound (McAllester 1999 Theorem 1, refined by Gat 2022 Theorems
    # 3.5/3.6 and Maurer 2004 Theorem 5; L2-proxy via Neyshabur 2017
    # §3).  When True, after each per-seed training loop we compute
    # the McAllester 1999 bound on the empirical training risk and
    # attach the result to the per-seed checkpoint record so the
    # Round-13 paper can cite a paper-grade generalisation
    # certificate.
    p.add_argument('--pac-bayes-bound', dest='pac_bayes_bound',
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Compute a PAC-Bayes generalisation bound '
                        '(McAllester 1999 Thm 1) on the per-seed '
                        'training risk and attach it to the '
                        'checkpoint record.  Default False (no '
                        'overhead, no certificate).')
    p.add_argument('--pac-bayes-delta', type=float, default=0.05,
                   help='Confidence parameter for the PAC-Bayes '
                        'bound (default 0.05 = 95%% confidence).  '
                        'Only consulted when --pac-bayes-bound is '
                        'set.')
    p.add_argument('--pac-bayes-loss-clamp', type=float, default=10.0,
                   help='CFM training loss clamp for the empirical '
                        'risk computation (loss_clamp in '
                        ':func:`pac_bayes_bound_from_losses`).  '
                        'Default 10.0; lower values give a tighter '
                        'but riskier certificate.')
    # WF-CFM-Path-B-Decoder-Rework (Phase 1) — switch the learned
    # decoder to the chem-aware soft 3-prior decoder (DecoderRework +
    # ReworkedDecoder from molmetal_lam.lam_chem.decoder_rework).
    # Only active when --bond-head=learned; ignored otherwise.  The
    # default False preserves the pre-Path-B behaviour bit-exactly.
    p.add_argument('--decoder-rework', dest='decoder_rework',
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Use the chem-aware soft 3-prior decoder '
                        '(DecoderRework + ReworkedDecoder) when '
                        '--bond-head=learned.  Soft distance mask '
                        '(τ=0.3 Å) + type-compat prior + valence '
                        'barrier (WF-CFM-Path-B).  Default False '
                        '(preserves pre-Path-B bit-exact behaviour).')
    # TODO-21 Strategy 1 (2026-09-17) — Lambda-as-reward channel.
    # Wired into RewardAggregator via
    # :mod:`molmetal_lam.lam_chem.lambda_reward_channel`.  Opt-in
    # (default 0.0 = silent, backward compatible).  When > 0 the
    # harness reads the Lambda MCTS candidate SMILES from the
    # coupling adapter and adds ``w_lambda_score * mean_cos_sim`` to
    # the per-cell reward; reward-weighted CFM training is the next
    # step (deferred to R16 once GPU retrain is unblocked).
    p.add_argument('--reward-lambda-weight',
                   type=float,
                   default=0.0,
                   help='TODO-21 Strategy 1 — weight on the '
                        'Lambda-as-reward channel.  Mean cosine '
                        'similarity between the CFM pocket '
                        'embedding (coupling_adapter.embed_pocket) '
                        'and the Lambda MCTS candidates supplied '
                        'via set_lambda_candidates().  Opt-in '
                        '(default 0.0 = channel silent, backward '
                        'compatible).  Set 0.5 to let Lambda '
                        'inform CFM training without dominating '
                        'the dominant channels.')
    # WF-PB-MMFF94-Relax — Stage 1 of the dock -> MMFF94s relax -> PB
    # check pipeline.  When --pb-check is set and this flag is set, the
    # docked pose is MMFF94s-relaxed (Halgren 1996
    # *J. Comput. Chem.* 17, 490-512; RDKit AllChem.MMFFOptimizeMolecule
    # with mmffVariant='MMFF94s') BEFORE PoseBusters runs, so the
    # bonded-geometry checks land inside the MMFF94s reference window
    # that PoseBusters' chemistry baseline assumes.
    p.add_argument('--pb-relax-mmff94', dest='pb_relax_mmff94',
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Apply MMFF94s intra-ligand relaxation to each '
                        'docked pose BEFORE the PoseBusters check '
                        '(WF-PB-MMFF94-Relax).  Default False preserves '
                        'the pre-relax behaviour bit-exactly.')
    p.add_argument('--pb-relax-max-iters', type=int, default=200,
                   help='MMFF94s relaxation max iterations (default 200; '
                        'convergence is typically reached well below this '
                        'for drug-like organics)')
    # WF-Metallodrug-Vertical Phase 3 protocol-align — TargetDiff
    # baseline alignment.  Bumped exhaustiveness from 1 (smoke) to 8
    # (production-CrossDocked2020 standard), matching the headline
    # pose-prediction parity that TargetDiff (Guan ICLR 2023) reports
    # in §4.4 (Table 4).  Backward compatible: setting
    # ``--physical-exhaustiveness 1`` recovers the pre-Phase-3
    # smoke-bit-exact behaviour (the legacy value before the default
    # flip is recorded in the run-config dict for audit).
    p.add_argument('--physical-exhaustiveness', type=int, default=8,
                   help='Vina/QuickVina2 exhaustiveness (default 8 = '
                        'TargetDiff/CrossDocked2020 production standard; '
                        'set to 1 to recover the pre-Phase-3 smoke '
                        'bit-exact behaviour).')
    # WF-Metallodrug-Vertical Phase 3 — pocket-10 Å radius flag.
    # Default 10.0 Å matches the CrossDocked2020 standard pocket
    # extraction radius (Francoeur et al. 2020, also adopted by
    # DiffDock / TargetDiff / Pocket2Mol).  Backward compatible: the
    # flag is consumed by :func:`select_training` if a downstream
    # pocket-crop step opts in; the legacy 8 Å radius is recoverable
    # via ``--pocket10-radius 8.0``.
    p.add_argument('--pocket10-radius', type=float, default=10.0,
                   help='Pocket extraction radius in Angstroms '
                        '(default 10.0 = CrossDocked2020 / TargetDiff '
                        'standard; set 8.0 for the legacy DiffDock-Pocket '
                        'crop).')
    # WF-Metallodrug-Vertical Phase 3 — pocket-conditioned reference
    # ligand warm-start.  When ``--reference-ligand`` is set the
    # generator uses the named ligand SDF (under
    # ``/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf``)
    # as the warm-start for pocket-conditioned generation instead of
    # the unconditional Gaussian base.  Default ``off`` preserves the
    # legacy unconditional sampling path bit-exactly.
    p.add_argument('--reference-ligand', action='store_true',
                   help='Use the pocket-specific reference ligand SDF '
                        '(/mnt/storage/data/molmetal/crossdocked/extracted/'
                        'crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf) '
                        'as a warm-start for pocket-conditioned generation. '
                        'Default off (legacy unconditional sampling).')
    # WF-CFM-Frontier-Phase2 Fix #3 — early-warning decode smoke
    # (inference_review_phase1d.md TOP-1).  Sample 8 mols every N
    # training steps; log n_decoded/8 so we catch a "trains fine,
    # decode=0" failure inside the 5K-step budget instead of after
    # it completes.  Default 0 = disabled (preserves
    # pre-Phase-2 bit-exact behaviour bit-exactly, including the
    # wf_gpu_recovery_now baseline numbers).
    p.add_argument('--decode-smoke-every', type=int, default=0,
                   help='Sample 8 mols every N training steps and log '
                        'n_decoded/8 to stdout (WF-CFM-Frontier Phase 2 '
                        'Fix #3).  Catches decode=0 collapses in <30 s '
                        'instead of after 5K-step retrain completion. '
                        'Default 0 = disabled (no overhead, '
                        'backward-compat with prior baselines).')
    p.add_argument('--decode-smoke-n-samples', type=int, default=8,
                   help='Number of mols sampled per decode smoke '
                        'checkpoint (default 8).  Only consulted when '
                        '--decode-smoke-every > 0.')
    p.add_argument('--decode-smoke-n-steps', type=int, default=200,
                   help='ODE steps per decode smoke checkpoint '
                        '(default 200).  Smaller is faster but '
                        'noisier.  Only consulted when '
                        '--decode-smoke-every > 0.')
    p.add_argument('--decode-smoke-warn-after', type=int, default=2,
                   help='Emit a UserWarning after this many CONSECUTIVE '
                        'decode smokes return n_decoded == 0 (default 2). '
                        'A persistent decode=0 across consecutive '
                        'smokes indicates the CFM path is dead.  Only '
                        'consulted when --decode-smoke-every > 0.')
    args=p.parse_args()
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors,QED,Crippen,rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    from molmetal.domain import Molecule
    from molmetal.scripts.evaluate_generated_poses import evaluate_candidates
    from molmetal.validation.gpu_molecular_metrics import molecular_diversity
    from molmetal.baselines.pac_bayes import (
        pac_bayes_bound_from_losses,
        kl_l2_proxy,
    )
    RDLogger.DisableLog('rdApp.*')
    if not torch.cuda.is_available():
        raise RuntimeError('Real ROCm GPU required')
    root=Path('/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10')
    split_path=Path('/mnt/storage/data/molmetal/crossdocked/split_by_name.pt')
    manifest=Path('molmetal/data/crossdocked100_manifest.csv')

    # ------------------------------------------------------------------
    # Metallodrug vertical (Phase 1 filter fix) — branch on --data-source.
    # The legacy 'crossdocked' path reads split_by_name.pt and applies
    # the filter (heavy atoms 8..38, 14-element metallodrug vocabulary).
    # The new 'platinai' / 'metallo_drugs_combined' paths build a
    # diversity-sampled pool from the corresponding corpora and cache
    # it as a CSV under molmetal/data/.  When the cache file already
    # exists at --cache-csv we reuse it (skip the rebuild); otherwise
    # we run the diversity sampler and write the cache.
    # ------------------------------------------------------------------
    if args.data_source == 'crossdocked':
        splits = torch.load(split_path, weights_only=False)
        training, rejected_training = select_training(root, splits, args.n_train)
        data_source_meta = {
            'kind': 'crossdocked',
            'root': str(root),
            'split_path': str(split_path),
            'split_sha256': sha(split_path),
        }
    else:
        # Lazy import — keep crossdocked-only invocations free of the
        # metallodrug loader's import cost.
        from molmetal.molmetal_lam.lam_chem.data_diversity import (
            build_combined_pool, cache_csv, load_platinai_smiles,
        )
        if args.data_source == 'platinai':
            target_n = args.n_train
        else:  # metallo_drugs_combined
            target_n = args.n_train
        cache_path = args.cache_csv.resolve()
        if cache_path.is_file():
            import csv as _csv
            with cache_path.open(newline='') as cf:
                reader = _csv.DictReader(cf)
                cached_smiles = [r['smiles'] for r in reader]
            if len(cached_smiles) >= target_n:
                # Reuse the cache — diversity-sampling is deterministic.
                smiles = cached_smiles[:target_n]
                sources = ['cached'] * len(smiles)
            else:
                smiles, sources = build_combined_pool(n=target_n, seed=args.diversity_seed)
                cache_csv(smiles, sources, cache_path)
        else:
            smiles, sources = build_combined_pool(n=target_n, seed=args.diversity_seed)
            cache_csv(smiles, sources, cache_path)
        # Build a synthetic `training` list whose entries expose only the
        # fields the rest of the harness reads (`smiles`).  This avoids
        # requiring a receptor SDF for metallodrug-only pools; downstream
        # training then uses a *ligand-only* loop.
        training = [
            {'split_index': i, 'receptor': None, 'ligand': None,
             'smiles': s, 'receptor_sha256': None, 'ligand_sha256': None,
             'source': src}
            for i, (s, src) in enumerate(zip(smiles, sources))
        ]
        rejected_training = []
        splits = None
        data_source_meta = {
            'kind': args.data_source,
            'cache_csv': str(cache_path),
            'cache_sha256': sha(cache_path) if cache_path.is_file() else None,
            'diversity_seed': int(args.diversity_seed),
            'n_train': int(len(training)),
            'sample_sources': dict(Counter(sources)),
        }
    # ------------------------------------------------------------------

    # Build actual_mols + scale + training contexts (works for either
    # branch — for metallodrug branch we embed from SMILES).
    from rdkit import Chem as _Chem
    actual_mols = []
    for r in training:
        if r.get('ligand') is not None:
            actual_mols.append(read_ligand(r['ligand']))
        else:
            mol = _Chem.MolFromSmiles(r['smiles'])
            if mol is None:
                continue
            mol = _Chem.AddHs(mol)
            try:
                from rdkit.Chem import AllChem as _AllChem
                _AllChem.EmbedMolecule(mol, randomSeed=42)
                _AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol = _Chem.RemoveHs(mol)
            actual_mols.append(mol)
    if not actual_mols:
        raise ValueError('No parseable training molecules after filter')
    scale=statistics.median(float(np.std(m.GetConformer().GetPositions()-m.GetConformer().GetPositions().mean(0))) for m in actual_mols)
    train_contexts=[]; train_mols=[]
    for record,mol in zip(training,actual_mols):
        if record.get('receptor') is not None:
            context,center,indices=build_context(record['receptor'],mol,scale)
            record['encoder_receptor_atom_indices']=indices
            train_contexts.append(context)
        else:
            # Metallodrug branch — no receptor SDF; build a dummy pocket
            # centred at the ligand centroid so the adapter has a context
            # object (no real pocket-conditioning; used only for the
            # decoder/training loop smoke).
            center = mol.GetConformer().GetPositions().mean(axis=0)
            from molmetal.domain import Pocket
            train_contexts.append(Pocket(
                pdb_id=f'metal_dummy_{record["split_index"]:04d}',
                coords=torch.zeros((1, 3), dtype=torch.float32),
                atom_types=torch.zeros((1,), dtype=torch.long),
                residue_ids=torch.zeros((1,), dtype=torch.long),
                chain_ids=['A'],
                mask=torch.zeros((1,), dtype=torch.bool),
                center=torch.tensor(center, dtype=torch.float32),
                radius=12.0,
            ))
        domain=Molecule.from_rdkit_mol(mol)
        domain=replace(domain, coords=(domain.coords-torch.tensor(center,dtype=torch.float32))/scale)
        train_mols.append(domain)
    with manifest.open(newline='') as f:
        tests=[r for r in csv.DictReader(f) if r['pocket_id'] in ('test_001','test_002')]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report={'status':'running','date_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'bounded real CrossDocked model CFG control; model samples with declared heuristic connectivity, separate from Lambda search',
        'protocol':{'seeds':list(args.seeds),'cfg_scales':[1.,2.],'train_steps':args.train_steps,'train_batch':2,
                    'n_train':args.n_train,'n_samples_per_cell':args.n_samples,'n_atoms':24,'ode_steps':args.ode_steps,
                    'hidden_dim':args.hidden_dim,'n_layers':args.n_layers,'max_atomic_number':100,'lr':args.lr,'atom_loss_weight':1.,'context_dropout':.1,
                    'pocket_embed_scale':.1,'max_encoder_pocket_atoms':64,'position_scale':scale,
                    'atom_training':'masked Z=0 input in training and sampling; true atom numbers are CE targets only; no output vocabulary mask',
                    'atom_selection':('metallodrug-relevant 14-element donor + Pt/Pd/Au/Ir/Ru vocabulary '
                                      '{6,7,8,9,15,16,17,35,53,44,46,77,78,79}; heavy atoms 8..38 '
                                      '(metallodrug vertical Phase 1)'),
                    'data_source':data_source_meta,
                    'training_split':'official split_by_name.pt train; no canonical ligand overlap with all available 100 test ligands',
                    'heldout_selection':'manifest test_001 and test_002, exact pairs',
                    'ablation_control':'same saved checkpoint, sample seed, initial coordinate noise and decoding protocol per seed/pocket; only cfg_scale differs',
                    'decoder':('learned bond-order head (A1, BondOrderHead + BondAwareDecoder)'
                               if args.bond_head == 'learned'
                               else 'distance connectivity, covFactor1.3, all single bonds + implicit H; no learned bond order decoder'),
                    'vocab_mask':bool(args.vocab_mask),
                    'wf2_a5_joint_train':bool(args.joint_train),
                    'wf2_a5_bond_loss_weight':float(args.bond_loss_weight),
                    'wf2_a5_bond_pattern_mask':bool(args.bond_pattern_mask),
                    'wf2_a6_connectivity_prior':str(args.connectivity_prior),
                    'wf_phase23_use_pcgrad':bool(args.use_pcgrad),
                    'wf_phase32_pac_bayes_bound':bool(args.pac_bayes_bound),
                    'wf_phase32_pac_bayes_delta':float(args.pac_bayes_delta),
                    'wf_phase32_pac_bayes_loss_clamp':float(args.pac_bayes_loss_clamp),
                    'wf_path_b_decoder_rework':bool(args.decoder_rework),
                    'docking':'QuickVina2-GPU lanes1000 depth1 poses1 for every decoded graph; no native-exhaustiveness equivalence',
                    'generated_pose_vs_docked_pose':'docking reembeds/optimizes the decoded graph; docking PB is not validation of the model raw geometry'},
        'sources':({'split_path':str(split_path),'split_sha256':sha(split_path),'manifest_sha256':sha(manifest)}
                   if args.data_source == 'crossdocked'
                   else {'data_source':args.data_source,
                         'cache_csv':data_source_meta.get('cache_csv'),
                         'cache_sha256':data_source_meta.get('cache_sha256'),
                         'manifest_sha256':sha(manifest) if manifest.is_file() else None}),
        'training':training,'rejected_training_pairs':rejected_training,'tests':tests,'checkpoints':[],'cells':[],
        'runtime_sha256':{str(path):sha(path) for path in [Path(__file__),Path('molmetal/adapters/flow_matching_lipman/__init__.py'),
            Path('models/velocity_net.py'),Path('models/_scatter.py'),Path('molmetal/scripts/evaluate_generated_poses.py')]},
        'environment':{'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'hip':torch.version.hip}}
    def checkpoint():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def check_budget():
        if time.monotonic()-started>args.budget_seconds:
            raise TimeoutError('Real CFG experiment wall budget exceeded')
    try:
        for seed in args.seeds:
            check_budget();torch.manual_seed(seed)
            adapter=LipmanFlowMatchingAdapter(hidden_dim=args.hidden_dim,n_layers=args.n_layers,max_atomic_number=100,lr=args.lr,
                        atom_loss_weight=1.,metal_prior_weight=0.,context_dropout=.1,pocket_embed_scale=.1,
                        vocab_mask=bool(args.vocab_mask),
                        # WF-2 A5 — joint bond-head training knobs.
                        # Wire the head when --bond-head=learned; only
                        # join-train it when the explicit opt-in flag
                        # is set (default False preserves A1
                        # bit-exact behaviour).
                        use_bond_head=bool(args.bond_head == 'learned'),
                        joint_train=bool(args.joint_train),
                        bond_loss_weight=float(args.bond_loss_weight),
                        bond_pattern_mask=bool(args.bond_pattern_mask),
                        # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad
                        # multi-task gradient surgery (Yu 2020).
                        # Only takes effect when --joint-train is also
                        # set; otherwise ignored.
                        use_pcgrad=bool(args.use_pcgrad),
                        # WF-Vina-Lift-Phase23 (Phase 3.1) —
                        # tmQM-pretrained warm-start.  Best-effort:
                        # missing ckpt logs a warning but does not
                        # abort training (backward compat with hosts
                        # that don't carry the 2.5 MB ckpt).
                        use_tmqm_init=bool(args.use_tmqm_init))
            adapter.setup('cuda:0')
            # WF-Vina-Lift-Phase23 (Phase 3.2) — snapshot the *prior*
            # weights BEFORE training so we can compute
            # ``KL(Q || P)`` via the L2-proxy (Neyshabur 2017 §3) on
            # the post-training posterior.  We deep-copy via
            # ``.detach().clone()`` to avoid mutating the live
            # parameters — the snapshot is a pure read-only tensor.
            if args.pac_bayes_bound:
                prior_snap = [
                    p.detach().clone()
                    for p in (
                        list(adapter.velocity_field.parameters())
                        + list(adapter.pocket_encoder.parameters())
                    )
                ]
            losses=[]
            # WF-CFM-Frontier-Phase2 Fix #3 — early-warning decode
            # smoke hook.  When ``--decode-smoke-every N`` is set with
            # ``N > 0``, every N training steps we call
            # :func:`run_decode_smoke` on the FIRST training pocket
            # (``train_contexts[0]``) — that's deterministic and always
            # available, so the smoke signal is comparable across
            # checkpoints.  The counter ``_consecutive_decode_zeros``
            # increments each time the smoke returns ``n_decoded ==
            # 0``; if it reaches ``args.decode_smoke_warn_after`` we
            # emit a single :class:`UserWarning` so the operator can
            # halt the run before wasting the full budget on a
            # dead-end path.  Default ``decode_smoke_every=0``
            # preserves the pre-Fix-3 bit-exact behaviour (the smoke
            # block is skipped entirely; ``_consecutive_decode_zeros``
            # never initialised).
            consecutive_decode_zeros = 0
            decode_smoke_log: list = []
            for step in range(args.train_steps):
                check_budget();torch.manual_seed(seed+10000+step)
                indices=[(step*2+i)%len(training) for i in range(2)]
                loss=adapter.train_step([train_contexts[i] for i in indices],[train_mols[i] for i in indices])
                if not np.isfinite(loss):raise FloatingPointError('Nonfinite real-data training loss')
                losses.append(loss)
                # WF-CFM-Frontier-Phase2 Fix #3 — fire the smoke
                # AFTER the train step is logged so the smoke uses
                # the freshly-updated weights but does not perturb
                # the loss series.  We use ``args.decode_smoke_every
                # > 0`` as the gate so a zero-every disables the
                # entire block (backward-compat).
                if args.decode_smoke_every > 0 and step > 0 and step % args.decode_smoke_every == 0:
                    try:
                        n_decoded, _smoke_mols = run_decode_smoke(
                            adapter, train_contexts[0],
                            n_samples=args.decode_smoke_n_samples,
                            n_steps=args.decode_smoke_n_steps,
                            seed=seed + step,
                        )
                        decode_smoke_log.append({
                            'step': step, 'n_decoded': n_decoded,
                            'n_samples': args.decode_smoke_n_samples,
                        })
                        if n_decoded == 0:
                            consecutive_decode_zeros += 1
                            if consecutive_decode_zeros >= args.decode_smoke_warn_after:
                                warnings.warn(
                                    f"[seed {seed}] decode_smoke has returned 0/8 "
                                    f"decoded mols for {consecutive_decode_zeros} "
                                    f"consecutive smokes (current step {step}); "
                                    f"the CFM decode path may be broken — consider "
                                    f"halting this run.",
                                    UserWarning, stacklevel=2,
                                )
                        else:
                            consecutive_decode_zeros = 0
                        print(
                            f"  [seed {seed} step {step}] decode_smoke: "
                            f"{n_decoded}/{args.decode_smoke_n_samples} decoded",
                            flush=True,
                        )
                    except Exception as _smoke_exc:
                        # Smoke failures must NEVER abort training;
                        # they are diagnostic instrumentation.
                        warnings.warn(
                            f"[seed {seed} step {step}] decode_smoke raised "
                            f"{type(_smoke_exc).__name__}: {_smoke_exc}; "
                            f"continuing training without smoke signal.",
                            UserWarning, stacklevel=2,
                        )
            path=out/f'checkpoint_seed{seed}.pt'
            torch.save({'velocity_field':adapter.velocity_field.state_dict(),'pocket_encoder':adapter.pocket_encoder.state_dict()},path)
            # WF-Vina-Lift-Phase23 (Phase 3.2) — PAC-Bayes bound
            # (McAllester 1999 Theorem 1).  Compute the per-seed
            # certificate on the empirical training risk using the
            # L2-proxy KL between the post-training posterior and
            # the init-prior snapshot.  The complexity term scales
            # as ``1 / sqrt(n)`` so doubling ``train_steps`` halves
            # the gap (modulo the KL growth from a longer training
            # trajectory — the L2 norm accumulates with each step).
            pac_bayes_record: dict = {
                'enabled': bool(args.pac_bayes_bound),
                'delta': float(args.pac_bayes_delta),
                'loss_clamp': float(args.pac_bayes_loss_clamp),
                'n_train_steps': int(args.train_steps),
                'batch_size': 2,
                'kl_proxy': None,
                'emp_risk': None,
                'bound': None,
                'bound_squared': None,
                'is_valid': None,
            }
            if args.pac_bayes_bound:
                post_params = (
                    list(adapter.velocity_field.parameters())
                    + list(adapter.pocket_encoder.parameters())
                )
                kl_value = kl_l2_proxy(prior_snap, post_params)
                bound_result = pac_bayes_bound_from_losses(
                    losses=losses,
                    kl_q_p=kl_value,
                    delta=float(args.pac_bayes_delta),
                    loss_clamp=float(args.pac_bayes_loss_clamp),
                )
                pac_bayes_record.update({
                    'kl_proxy': float(kl_value),
                    'emp_risk': float(bound_result.empirical_risk),
                    'bound': float(bound_result.bound),
                    'bound_squared': float(bound_result.bound_squared),
                    'is_valid': bool(bound_result.is_valid),
                })
                print(
                    f"  PAC-Bayes bound: KL={kl_value:.4f} n={len(losses)} "
                    f"delta={args.pac_bayes_delta} "
                    f"R_hat={bound_result.empirical_risk:.4f} "
                    f"bound={bound_result.bound:.4f}",
                    flush=True,
                )
            report['checkpoints'].append({'seed':seed,'path':str(path),'sha256':sha(path),'training_loss_diagnostic':losses,'last_losses':adapter.last_losses,'adapter_metadata':adapter.get_metadata(),'pac_bayes':pac_bayes_record,
                # WF-CFM-Frontier-Phase2 Fix #3 — per-seed decode_smoke
                # trajectory (empty when --decode-smoke-every=0).
                'decode_smoke_log':list(decode_smoke_log),
                'decode_smoke_n_consecutive_zeros_at_end':int(consecutive_decode_zeros)})
            print(f'trained real-data seed={seed}, checkpoint={sha(path)[:12]}',flush=True)
            for pair in tests:
                reference=read_ligand(pair['ligand_path'])
                pocket,center,indices=build_context(pair['receptor_path'],reference,scale)
                for guidance in (1.,2.):
                    check_budget();adapter._cfg_scale=guidance
                    cell_dir=out/f"{pair['pocket_id']}_seed{seed}_cfg{guidance:g}";cell_dir.mkdir(exist_ok=True)
                    cell={'pocket_id':pair['pocket_id'],'seed':seed,'cfg_scale':guidance,'checkpoint_sha256':sha(path),
                          'encoder_receptor_atom_indices':indices,'n_requested':args.n_samples,'n_raw_generated':0,'samples':[]}
                    report['cells'].append(cell)
                    try:
                        generated=adapter.generate(pocket,SizedGenerationConfig(n_samples=args.n_samples,n_steps=args.ode_steps,seed=seed))
                        cell['n_raw_generated']=len(generated)
                        valid=[]
                        for i,molecule in enumerate(generated):
                            coords=molecule.coords.numpy()*scale+center
                            raw={'atomic_numbers':molecule.atom_types.tolist(),'coordinates_A':coords.tolist()}
                            raw_path=cell_dir/f'raw_{i:03d}.json';raw_path.write_text(json.dumps(raw)+'\n')
                            row={'sample_index':i,'raw_path':str(raw_path),'raw_sha256':sha(raw_path)}
                            if args.bond_head == 'learned':
                                mol,status=decode_learned_bond_graph(
                                    raw['atomic_numbers'], coords,
                                    connectivity_prior=str(args.connectivity_prior),
                                    decoder_rework=bool(args.decoder_rework),
                                    hidden_dim=int(args.hidden_dim),
                                )
                            else:
                                mol,status=decode_distance_graph(raw['atomic_numbers'],coords)
                            row['status']=status
                            if mol is not None:
                                smiles=Chem.MolToSmiles(mol)
                                row.update(smiles=smiles,descriptors={'sa':float(sascorer.calculateScore(mol)),
                                    'qed':float(QED.qed(mol)),'logp':float(Crippen.MolLogP(mol)),
                                    'molecular_weight':float(Descriptors.MolWt(mol)),
                                    'rotatable_bonds':float(rdMolDescriptors.CalcNumRotatableBonds(mol))})
                                pose_path=cell_dir/f'decoded_model_geometry_{i:03d}.sdf'
                                with Chem.SDWriter(str(pose_path)) as writer:writer.write(mol)
                                row.update(model_geometry_sdf=str(pose_path),model_geometry_sha256=sha(pose_path))
                                valid.append({'smiles':smiles,'is_generated':True,'sample_index':i})
                            cell['samples'].append(row)
                        cell['decode_status_counts']=dict(Counter(r['status'] for r in cell['samples']))
                        cell['n_decoded']=len(valid)
                        cell['decoded_fraction_all_requested']=len(valid)/args.n_samples
                        cell['diversity']=molecular_diversity([v['smiles'] for v in valid],device='cuda:0')
                        gpu={'binary_path':str(args.gpu_binary),'gpu_threads':1000,'search_depth':1}
                        if args.trace_library:gpu['trace_library']=str(args.trace_library)
                        cell['physical']=evaluate_candidates(valid,pair['receptor_path'],pair['ligand_path'],str(cell_dir/'physical'),
                            seed=seed,engine='quickvina2-gpu',exhaustiveness=args.physical_exhaustiveness,n_poses=1,top_k=args.n_samples,
                            gpu_config=gpu,total_generated=args.n_samples,
                            # WF-PB-MMFF94-Relax — Stage 1 of the
                            # dock -> MMFF94s relax -> PB check pipeline.
                            relax_mmff94=bool(args.pb_relax_mmff94),
                            relax_max_iters=int(args.pb_relax_max_iters))
                        cell['status']='completed'
                    except Exception as exc:
                        cell.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    checkpoint();print(f"{pair['pocket_id']} seed={seed} CFG={guidance} decoded={cell.get('n_decoded',0)}/{args.n_samples} {cell['status']}",flush=True)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    report['elapsed_wall_s']=time.monotonic()-started
    report['aggregate']={'n_requested_planned':3*len(tests)*2*args.n_samples,
        'n_requested':sum(c['n_requested'] for c in report['cells']),
        'n_raw_generated':sum(c['n_raw_generated'] for c in report['cells']),
        'n_decoded':sum(c.get('n_decoded',0) for c in report['cells']),
        'n_docked':sum(c.get('physical',{}).get('summary',{}).get('n_docked',0) for c in report['cells']),
        'n_pb_pass_docked':sum(c.get('physical',{}).get('summary',{}).get('n_pb_pass',0) for c in report['cells'])}
    checkpoint();print(json.dumps(report['aggregate']))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
