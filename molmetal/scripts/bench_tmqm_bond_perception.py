#!/usr/bin/env python3
"""tmQM bond-perception reference table.

==============================================================================
What this measures, and why it exists
==============================================================================

The task: given a metal complex's 3D geometry (coordinates + element types),
recover its molecular graph — including metal-ligand bonds.

This is what the project's CFM does (`decode_ratio`).  A 2026-10-04 literature
sweep established that **no published number exists for this task on tmQM or on
any transition-metal corpus**:

  * Published bond-perception baselines are all on ORGANIC molecules —
    RDKit `DetermineBonds` 95.3% exact-type / macro-F1 0.9886 and OpenBabel
    64.4% / 0.8935 on a fixed 26,940-molecule GEOM-DRUGS cohort
    (BondNet protocol, ChemRxiv preprint 10.26434/chemrxiv.15008897/v1;
    note that preprint is unrefereed, so its own numbers are NOT quoted here).
  * RDKit `DetermineBonds` cannot handle transition metals at all — the
    published organic pipelines strip the metal first.
  * On metals, the only published works report interpretation success
    (cell2mol: 77% on an 8-metal mononuclear CSD set, 95% of those correct;
    npj Comput. Mater. 2022, 8:188) or inter-route AGREEMENT, not accuracy
    (xyz2mol_tm, "SMILES all around", J. Cheminform. 2025, 17:63).
  * The tmQM paper itself (Balcells & Skjelstad, JCIM 2020, 60:6135,
    doi 10.1021/acs.jcim.0c01041) benchmarks DFT level-of-theory agreement
    only — no structure or topology task — so it shares no metric with this.

So we are not reproducing anyone's number.  We are producing the reference
table that does not yet exist, by running the field's deterministic baselines
on tmQM under one stated protocol.

==============================================================================
Protocol
==============================================================================

GROUND TRUTH comes from tmQM's own ``tmQM_X{1,2,3}.BO.gz`` Wiberg bond-order
blocks, NOT from the SMILES column.  The reason is atom ordering: the SMILES
in ``tmQM_y.csv`` carries its own (different) atom order, so scoring would
first need a graph-matching step whose failures would be indistinguishable
from perception failures.  The BO blocks are emitted in the SAME atom order as
the ``.xyz.gz`` geometries, so the comparison is index-for-index.

Per BO-block line::

    2  Se  1.967        P    8 1.313    La   1 0.285    Se   3 0.127
    ^  ^   ^            ^    ^ ^        ^    ^ ^
    idx sym total_bo        nbr_sym nbr_idx bo

A bond exists when the Wiberg bond order is >= ``--bo-threshold`` (default
0.3; the shared H...H contacts in tmQM sit near 0.1 and are not bonds).  Its
type is the nearest of {single 1.0, double 2.0, triple 3.0} by BO magnitude.

METRICS (three separate columns, deliberately not collapsed into one number):

  * ``success_rate``  — the baseline produced a parseable molecule at all.
    This is the metric class our CFM's ``decode_ratio`` belongs to.
  * ``exact_type``    — fraction of successes whose bond set matches the
    ground truth EXACTLY, bond orders included.  This is what the organic
    literature reports as "exact-type rate".
  * ``macro_f1``      — macro-averaged F1 over bond types.  Reported over bond
    EXISTENCE and, separately, over bond TYPE, because the two differ sharply:
    a method can find every bond and get most orders wrong.

BASELINES:

  * ``rdkit_stripped`` — remove the metal, run ``rdDetermineBonds`` on the
    remaining organic fragment, score only bonds among non-metal atoms.
    This is the published organic pipeline; it is included precisely to show
    what happens to it on metal complexes.
  * ``openbabel`` — ``ConnectTheDots`` + ``PerceiveBondOrders`` on the FULL
    complex, metal included.  Scored on all bonds.

Each baseline is scored on the bond subset it can in principle address, and
that subset is named in the output — comparing a metal-free method against
metal bonds would be a rigged comparison, and hiding the subsetting would be
a silent one.

Usage::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    python -m molmetal.scripts.bench_tmqm_bond_perception \\
        --tmqm-dir .data/molmetal/tmQM \\
        --metals Pt,Ru,Ir,Rh,Os,Re --n-samples 150 --out-dir /tmp/bp_ref
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: BO blocks live next to the geometries and share their atom indexing.
_BO_FILES = ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")

#: A Wiberg bond order below this is treated as "no bond".  tmQM's own format
#: lists every neighbour pair including sub-0.15 contacts between ligand
#: hydrogens; 0.3 is the conventional single/dative separation used in the
#: BO-based perception literature.
DEFAULT_BO_THRESHOLD = 0.3

#: Bond-order class centres a BO magnitude is rounded to.
_BO_TYPE_CENTRES = (("single", 1.0), ("double", 2.0), ("triple", 3.0))


# ---------------------------------------------------------------------------
# Ground truth — tmQM BO blocks
# ---------------------------------------------------------------------------
def iter_bo_blocks(
    bo_paths: Sequence[Path],
) -> Iterator[Tuple[str, List[Tuple[int, str, List[Tuple[int, float]]]]]]:
    """Yield ``(csd_code, atoms)`` from the gzipped BO shards.

    ``atoms`` is a list of ``(atom_index, symbol, neighbours)`` in file order,
    where ``neighbours`` is ``[(neighbour_index, bond_order), ...]``.  Indexing
    is 1-based in the file and kept that way here so it lines up with the XYZ
    block, which is also 1-based when read the same way.
    """
    for path in bo_paths:
        if not Path(path).is_file():
            continue
        code: Optional[str] = None
        atoms: List[Tuple[int, str, List[Tuple[int, float]]]] = []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, atoms
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    # A trailing fragment or a blank separator.
                    if code is not None and atoms:
                        yield code, atoms
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                sym = parts[1]
                rest = parts[3:]
                nbrs: List[Tuple[int, float]] = []
                # Each neighbour is three tokens: symbol, index, bond order.
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, sym, nbrs))
        if code is not None and atoms:
            yield code, atoms


def ground_truth_bonds(
    atoms: Sequence[Tuple[int, str, List[Tuple[int, float]]]],
    threshold: float = DEFAULT_BO_THRESHOLD,
) -> Dict[Tuple[int, int], float]:
    """``{(i, j): bond_order}`` with ``i < j``, keeping BO >= ``threshold``."""
    bonds: Dict[Tuple[int, int], float] = {}
    for idx, _sym, nbrs in atoms:
        for nbr_idx, bo in nbrs:
            if bo < threshold or nbr_idx == idx:
                continue
            key = (min(idx, nbr_idx), max(idx, nbr_idx))
            # Both directions are listed; keep the larger reading.
            if key not in bonds or bo > bonds[key]:
                bonds[key] = bo
    return bonds


def bo_to_type(bo: float) -> str:
    """Nearest bond-order class for a Wiberg BO magnitude."""
    return min(_BO_TYPE_CENTRES, key=lambda kv: abs(kv[1] - float(bo)))[0]


# ---------------------------------------------------------------------------
# Geometry — tmQM XYZ blocks
# ---------------------------------------------------------------------------
def iter_xyz_blocks(xyz_paths: Sequence[Path]) -> Iterator[Tuple[str, List[str], np.ndarray]]:
    """Yield ``(csd_code, symbols, coords)`` from the gzipped XYZ shards.

    NOTE the two shard families order their headers DIFFERENTLY, which is easy
    to get wrong (it cost a silent "0 geometries loaded" here):

        .xyz.gz :  <n_atoms>            <- count FIRST
                   CSD_code = ... | ...
                   <sym> <x> <y> <z>
                   ...

        .BO.gz  :  CSD_code = ... | ... <- header FIRST
                   <idx> <sym> <total_bo> <nbr_sym> <nbr_idx> <bo> ...

    This follows the standard XYZ convention (count, comment, atoms).
    """
    for path in xyz_paths:
        if not Path(path).is_file():
            continue
        with gzip.open(path, "rt") as fh:
            lines = fh.read().split("\n")
        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            if i + 1 >= len(lines):
                break
            head = lines[i + 1].strip()
            if not head.startswith("CSD_code"):
                i += 1
                continue
            code = head.split("=", 1)[1].split("|", 1)[0].strip()
            i += 2
            symbols: List[str] = []
            coords: List[List[float]] = []
            for _ in range(n_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) < 4:
                    continue
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if symbols:
                yield code, symbols, np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _bond_type_str(bond) -> str:
    name = str(bond.GetBondType())
    return {
        "SINGLE": "single",
        "DOUBLE": "double",
        "TRIPLE": "triple",
        "AROMATIC": "aromatic",
        "DATIVE": "dative",
    }.get(name, name.lower())


def _mol_bonds(mol, exclude_metals: bool, metal_z: Optional[set[int]]) -> Dict[Tuple[int, int], str]:
    """``{(i, j): type}`` for a molecule, optionally dropping metal bonds.

    Atom indices are 1-based to match the BO convention.
    """
    out: Dict[Tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if exclude_metals and metal_z is not None:
            if a.GetAtomicNum() in metal_z or b.GetAtomicNum() in metal_z:
                continue
        key = (min(a.GetIdx(), b.GetIdx()) + 1, max(a.GetIdx(), b.GetIdx()) + 1)
        out[key] = _bond_type_str(bond)
    return out


def baseline_rdkit_stripped(
    symbols: Sequence[str],
    coords: np.ndarray,
    metal_symbols: Sequence[str],
) -> Tuple[Optional[object], Dict[Tuple[int, int], str]]:
    """Metal-stripped RDKit ``DetermineBonds`` — the published organic pipeline.

    Returns ``(mol_or_None, bonds)``.  ``bonds`` covers only non-metal atoms,
    which is the only subset this method can address.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    z_set = {_SYM_TO_Z[s] for s in metal_symbols if s in _SYM_TO_Z}
    keep = [i for i, s in enumerate(symbols) if _SYM_TO_Z.get(s, 0) not in z_set]
    if len(keep) < 2:
        return None, {}
    # Remap to a compact block; record the map so indices can be restored.
    lines = [str(len(keep)), "metal-stripped fragment"]
    remap = {}
    for new_i, old_i in enumerate(keep):
        remap[new_i] = old_i
        lines.append(f"{symbols[old_i]} {coords[old_i][0]:.6f} {coords[old_i][1]:.6f} {coords[old_i][2]:.6f}")
    block = "\n".join(lines) + "\n"
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        return None, {}
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None, {}
    # Restore original (1-based) indices in the bond keys.
    out: Dict[Tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        a = remap[bond.GetBeginAtomIdx()] + 1
        b = remap[bond.GetEndAtomIdx()] + 1
        out[(min(a, b), max(a, b))] = _bond_type_str(bond)
    return mol, out


def baseline_openbabel(
    symbols: Sequence[str],
    coords: np.ndarray,
) -> Tuple[Optional[object], Dict[Tuple[int, int], str]]:
    """OpenBabel ``ConnectTheDots`` + ``PerceiveBondOrders`` on the full complex."""
    try:
        from openbabel import openbabel as ob
    except Exception:
        return None, {}
    conv = ob.OBConversion()
    if not conv.SetInFormat("xyz"):
        return None, {}
    lines = [str(len(symbols)), "tmqm"]
    for s, xyz in zip(symbols, coords):
        lines.append(f"{s} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    mol = ob.OBMol()
    if not conv.ReadString(mol, "\n".join(lines) + "\n"):
        return None, {}
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    out: Dict[Tuple[int, int], str] = {}
    for bond in ob.OBMolBondIter(mol):
        a = bond.GetBeginAtomIdx()  # 1-based in OpenBabel
        b = bond.GetEndAtomIdx()
        order = bond.GetBondOrder()
        out[(min(a, b), max(a, b))] = {1: "single", 2: "double", 3: "triple"}.get(
            int(order), "aromatic"
        )
    return mol, out


#: The learned model, scored under this same harness.
#:
#: 2026-10-04 — why this belongs here.  The table previously held only
#: classical perception, and the project had no number that could answer "does
#: the learned bond head earn its place?".  MEASURED on the same task,
#: OpenBabel recovers bond EXISTENCE at F1 0.985 while the CFM sat at 0.481
#: (4 epochs, probe scale) — but OpenBabel's weak axis is bond TYPE
#: (macro-F1 0.539: it has no metal coordination chemistry), and THAT is the
#: axis the CFM has to win.  Scoring all three through one `score_bonds` on
#: one scope is the only way to say anything about it.
#:
#: The CFM is trained on hydrogen-free clouds, so a comparison against
#: baselines scored over all atoms would be scoring the CFM out of
#: distribution.  `--heavy-atom-scope` restricts every method to the same
#: non-hydrogen pair set, which is what makes the rows comparable; the
#: unrestricted rows are still reported for the classical methods.
_CFM_CLASS_TO_TYPE = {
    1: "single",   # trainers' BOND_SINGLE
    2: "double",   # BOND_DOUBLE
    3: "triple",   # BOND_TRIPLE
    4: "aromatic",  # BOND_AROMATIC
}


def _scope_label(name: str, heavy_atom_scope: bool) -> str:
    """Human-readable bond scope for one row of the table.

    The scope is the whole reason these rows can or cannot be compared, so it
    is spelled out rather than left implied.
    """
    if heavy_atom_scope:
        if name == "rdkit_stripped":
            return "non-metal AND non-hydrogen pairs"
        return "non-hydrogen pairs"
    return "non-metal bonds only" if name == "rdkit_stripped" else "all bonds"


def torch_load_meta(ckpt_path: Path) -> Dict:
    """Provenance of a CFM checkpoint, for the output record.

    A bond-perception number is only meaningful alongside how trained the model
    was, so the epoch / cohort size travel with the row rather than being
    remembered separately.
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return {}
    saved = state.get("args", {}) or {}
    return {
        "path": str(ckpt_path),
        "epoch": state.get("epoch"),
        "best_epoch": state.get("best_epoch"),
        "n_train": saved.get("n_train"),
        "n_val": saved.get("n_val"),
        "epochs": saved.get("epochs"),
        "hidden_dim": saved.get("hidden_dim"),
        "use_egnn_velocity": saved.get("use_egnn_velocity"),
        "bond_class_weight_power": saved.get("bond_class_weight_power"),
        "bond_decision": saved.get("bond_decision"),
        "dataset": saved.get("dataset"),
    }


def load_cfm_bond_scorer(
    ckpt_path: Path,
    *,
    decision: str = "argmax",
    threshold: float = 0.5,
    device: str = "cpu",
):
    """Return ``score(symbols, coords) -> {(i, j): type}`` backed by a CFM head.

    Rebuilds the architecture from the checkpoint's OWN saved args, because
    the atom-encoder width is set by the YuelBond vocab preset (13 entries for
    ``pt``, 18 for ``all6``) and guessing it is a shape mismatch.  Atom indices
    are 1-based and line up 1:1 with the XYZ block — the cloud is built in file
    order with every atom kept, so no remapping is needed.
    """
    import torch

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table
    from molmetal.models.bond_head import AtomCloud

    import importlib.util as _ilu

    trainer_path = Path(__file__).resolve().parent / "train_cfm_yuelbond_wasserstein.py"
    _spec = _ilu.spec_from_file_location("_cfm_trainer_for_bench", str(trainer_path))
    assert _spec and _spec.loader
    trainer = _ilu.module_from_spec(_spec)
    sys.modules["_cfm_trainer_for_bench"] = trainer
    _spec.loader.exec_module(trainer)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = state.get("args", {}) if isinstance(state, dict) else {}
    model = trainer.CFMJointModel(
        hidden_dim=int(saved.get("hidden_dim", trainer.DEFAULT_HIDDEN_DIM)),
        n_layers=int(saved.get("n_layers", trainer.DEFAULT_N_LAYERS)),
        cutoff=float(saved.get("edge_cutoff", trainer.DEFAULT_EDGE_CUTOFF)),
        fm_objective=bool(saved.get("use_fm_objective", False)),
        z_table=get_z_table(str(saved.get("yuelbond_vocab", "pt") or "pt")),
        use_egnn_velocity=bool(saved.get("use_egnn_velocity", False)),
        max_atomic_number=int(saved.get("egnn_max_atomic_number", 100)),
    )
    missing, unexpected = model.load_state_dict(
        state.get("model_state_dict", state), strict=False
    )
    if missing or unexpected:
        raise SystemExit(
            f"architecture mismatch loading {ckpt_path}: {len(missing)} missing "
            f"/ {len(unexpected)} unexpected (e.g. {list(missing)[:3]})"
        )
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    featuriser = YuelBondDecoder(head=model.head)

    def score(symbols: Sequence[str], coords: np.ndarray):
        zs = [_SYM_TO_Z.get(s, 0) for s in symbols]
        unknown = [s for s, z in zip(symbols, zs) if z == 0]
        if unknown:
            raise SystemExit(f"unknown element(s) {sorted(set(unknown))} in complex")
        # AtomCloud is a plain dataclass with no `.to()`; build the tensors on
        # the target device (same shape as the trainer's _to_device_cloud).
        cloud = AtomCloud(
            positions=torch.as_tensor(
                np.asarray(coords), dtype=torch.float32, device=dev
            ),
            atomic_numbers=torch.as_tensor(zs, dtype=torch.long, device=dev),
        )
        with torch.no_grad():
            res = featuriser.featurise(cloud)
        if not res.decode_succeeded or res.logits.shape[0] == 0:
            return None, {}
        pred_class = trainer._bond_decision(res.logits, decision, threshold)
        out: Dict[Tuple[int, int], str] = {}
        for k, (s, d) in enumerate(
            zip(res.edge_index[0].tolist(), res.edge_index[1].tolist())
        ):
            cls = int(pred_class[k])
            if cls == trainer.BOND_NO_BOND:
                continue
            out[(min(s, d) + 1, max(s, d) + 1)] = _CFM_CLASS_TO_TYPE.get(
                cls, "unknown"
            )
        return model, out

    return score


#: Symbols → atomic number, for the metal set and the strip step.
_SYM_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
    "Br": 35, "I": 53, "Pt": 78, "Ru": 44, "Ir": 77, "Rh": 45, "Os": 76,
    "Re": 75, "Pd": 46, "Au": 79, "Se": 34, "B": 5, "Si": 14, "As": 33,
    "Te": 52, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Mn": 25,
    "Cr": 24, "V": 23, "Ti": 22, "Sc": 21, "Y": 39, "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ag": 47, "Cd": 48, "W": 74, "Ta": 73, "Hf": 72,
    "La": 57, "Ce": 58, "Lu": 71,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_bonds(
    pred: Dict[Tuple[int, int], str],
    truth_bo: Dict[Tuple[int, int], float],
    restrict_to: Optional[set] = None,
) -> Dict[str, float]:
    """Exact-type rate + F1 (existence and type) for one complex.

    ``restrict_to`` names the SCOPE — the atom pairs a baseline can in
    principle address (for a metal-stripped method, the non-metal pairs).
    It filters BOTH sides down to that scope and nothing more.

    It must not filter the prediction down to the ground truth: an earlier
    version here wrote ``pred_keys = set(pred) & set(truth)``, which forces
    precision to 1.0 by construction and reported a perfect F1(exist) for a
    method that was in fact missing bonds.  Filter the scope, never the truth.
    """
    scope = None if restrict_to is None else set(restrict_to)
    truth = {
        k: bo_to_type(v)
        for k, v in truth_bo.items()
        if scope is None or k in scope
    }
    pred_keys = {k for k in pred if scope is None or k in scope}

    truth_keys = set(truth)
    exact = 1.0 if pred_keys == truth_keys else 0.0
    # `exact_type` additionally requires every shared bond's order to match.
    exact_type = 1.0 if (exact and all(pred[k] == truth[k] for k in truth_keys)) else 0.0

    # --- F1 over bond EXISTENCE -------------------------------------------
    # Binary: a pair is in the ground-truth bond set or it is not.
    tp_e = len(pred_keys & truth_keys)
    f1_exist = (
        0.0
        if not (pred_keys or truth_keys)
        else 2 * tp_e / (len(pred_keys) + len(truth_keys))
    )

    # --- macro-F1 over bond TYPE ------------------------------------------
    # Per type class c:
    #   TP = pairs whose truth is c AND whose prediction is c
    #   FP = pairs predicted c but whose truth is a DIFFERENT class (or none)
    #   FN = pairs whose truth is c but predicted as a different class (or none)
    #
    # An earlier version of this scored `F1(exist)` and `F1(type)` with the
    # same expression, because the predicted-class set had already been
    # filtered by class before both loops — so the two columns were equal by
    # construction and the "type" column measured nothing.  Keep them distinct.
    classes = sorted({truth[k] for k in truth_keys} | {pred[k] for k in pred_keys})
    f1s_type = []
    for cls in classes:
        t_c = {k for k in truth_keys if truth[k] == cls}
        p_c = {k for k in pred_keys if pred[k] == cls}
        tp_t = len(t_c & p_c)
        f1s_type.append(
            0.0 if not (t_c or p_c) else 2 * tp_t / (len(t_c) + len(p_c))
        )
    return {
        "exact": exact,
        "exact_type": exact_type,
        "f1_exist": f1_exist,
        "macro_f1_type": float(np.mean(f1s_type)) if f1s_type else 0.0,
        "n_pred": len(pred_keys),
        "n_true": len(truth_keys),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _resolve(root: str) -> Path:
    p = Path(root)
    if p.is_dir():
        return p
    alt = PROJECT_ROOT / root
    if alt.is_dir():
        return alt
    raise SystemExit(f"tmQM directory not found: {root!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tmqm-dir", default=".data/molmetal/tmQM")
    ap.add_argument("--metals", default="Pt,Ru,Ir,Rh,Os,Re")
    ap.add_argument("--n-samples", type=int, default=150, help="per metal")
    ap.add_argument("--bo-threshold", type=float, default=DEFAULT_BO_THRESHOLD)
    ap.add_argument(
        "--baselines",
        default="rdkit_stripped,openbabel",
        help="comma-separated subset of {rdkit_stripped, openbabel, cfm}",
    )
    ap.add_argument(
        "--cfm-ckpt",
        default=None,
        help="CFM checkpoint for --baselines cfm. Records its provenance "
             "(epoch / n_train) in the output so a probe-scale model cannot "
             "be mistaken for a trained one.",
    )
    ap.add_argument(
        "--cfm-decision", default="argmax", choices=("argmax", "threshold"),
        help="Bond-existence rule, matching the trainer's --bond-decision.",
    )
    ap.add_argument(
        "--cfm-decision-threshold", type=float, default=0.5,
        help="P(bonded) cut when --cfm-decision threshold.",
    )
    ap.add_argument(
        "--heavy-atom-scope",
        action="store_true",
        help=(
            "Restrict EVERY method to the non-hydrogen pair set. This is what "
            "makes the rows comparable: the CFM trains on hydrogen-free "
            "clouds, so scoring it over all atoms would measure it out of "
            "distribution while OpenBabel/RDKit keep the hydrogens."
        ),
    )
    ap.add_argument("--out-dir", default="molmetal/reports/tmqm_bond_perception")
    args = ap.parse_args(argv)

    root = _resolve(args.tmqm_dir)
    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    wanted = set(metals)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    print(f"[bp] tmQM={root}  metals={metals}  n/samples={args.n_samples} "
          f"BO threshold={args.bo_threshold}  baselines={baselines}", flush=True)

    # Ground truth: BO blocks, filtered to complexes whose metal is in scope.
    bo_paths = [root / name for name in _BO_FILES]
    xyz_paths = [Path(str(p).replace(".BO.gz", ".xyz.gz")) for p in bo_paths]

    truth_by_code: Dict[str, Dict[Tuple[int, int], float]] = {}
    syms_by_code: Dict[str, List[str]] = {}
    for code, atoms in iter_bo_blocks(bo_paths):
        syms = [a[1] for a in atoms]
        if not (set(syms) & wanted):
            continue
        truth_by_code[code] = ground_truth_bonds(atoms, args.bo_threshold)
        syms_by_code[code] = syms
    print(f"[bp] ground truth: {len(truth_by_code)} complexes with a "
          f"{'/'.join(metals)} centre", flush=True)

    # Balanced sample across metals.
    import random

    rng = random.Random(0)
    per_metal: Dict[str, List[str]] = defaultdict(list)
    for code, syms in syms_by_code.items():
        for m in set(syms) & wanted:
            per_metal[m].append(code)
    chosen: List[str] = []
    for m in metals:
        pool = sorted(set(per_metal.get(m, [])))
        rng.shuffle(pool)
        chosen.extend(pool[: args.n_samples])
    chosen = sorted(set(chosen))
    print(f"[bp] sampled {len(chosen)} complexes "
          f"({dict((m, min(len(set(per_metal.get(m, []))), args.n_samples)) for m in metals)})",
          flush=True)

    # Geometries for the sampled codes.
    coords_by_code: Dict[str, Tuple[List[str], np.ndarray]] = {}
    want = set(chosen)
    for code, symbols, coords in iter_xyz_blocks(xyz_paths):
        if code in want:
            coords_by_code[code] = (symbols, coords)
            if len(coords_by_code) == len(want):
                break
    print(f"[bp] geometries loaded for {len(coords_by_code)}/{len(want)}", flush=True)

    cfm_score = None
    if "cfm" in baselines:
        if not args.cfm_ckpt:
            raise SystemExit("--baselines cfm requires --cfm-ckpt")
        cfm_score = load_cfm_bond_scorer(
            Path(args.cfm_ckpt),
            decision=args.cfm_decision,
            threshold=args.cfm_decision_threshold,
        )
        _ck = torch_load_meta(Path(args.cfm_ckpt))
        print(f"[bp] CFM checkpoint {args.cfm_ckpt} — epoch "
              f"{_ck.get('epoch')}, n_train {_ck.get('n_train')}, "
              f"epochs budget {_ck.get('epochs')}", flush=True)

    results: Dict[str, Dict] = {}
    for name in baselines:
        rows = []
        for code in chosen:
            if code not in coords_by_code:
                continue
            symbols, coords = coords_by_code[code]
            truth_bo = truth_by_code[code]
            z_set = {_SYM_TO_Z[m] for m in metals if m in _SYM_TO_Z}
            non_metal_pairs = {
                k for k, _ in truth_bo.items()
                if _SYM_TO_Z.get(symbols[k[0] - 1], 0) not in z_set
                and _SYM_TO_Z.get(symbols[k[1] - 1], 0) not in z_set
            }
            # Pairs with no hydrogen on either end.  Applied to BOTH sides by
            # `score_bonds` (it filters the scope, never the truth).
            non_h_pairs = {
                k for k, _ in truth_bo.items()
                if symbols[k[0] - 1] != "H" and symbols[k[1] - 1] != "H"
            }
            if name == "rdkit_stripped":
                mol, pred = baseline_rdkit_stripped(symbols, coords, metals)
                scope = non_metal_pairs
            elif name == "openbabel":
                mol, pred = baseline_openbabel(symbols, coords)
                scope = None
            elif name == "cfm":
                mol, pred = cfm_score(symbols, coords)
                scope = None
            else:
                print(f"[bp][WARNING] unknown baseline {name!r}; skipped", file=sys.stderr)
                break
            if args.heavy_atom_scope:
                scope = non_h_pairs if scope is None else (scope & non_h_pairs)
            if mol is None:
                rows.append({"code": code, "success": 0})
                continue
            s = score_bonds(pred, truth_bo, restrict_to=scope)
            s.update({"code": code, "success": 1})
            rows.append(s)
        if not rows:
            continue
        n = len(rows)
        n_ok = sum(r["success"] for r in rows)
        ok_rows = [r for r in rows if r["success"]]
        results[name] = {
            "n_complexes": n,
            "n_success": n_ok,
            "success_rate": n_ok / n,
            "exact_type_rate_of_successes": (
                float(np.mean([r["exact_type"] for r in ok_rows])) if ok_rows else 0.0
            ),
            "f1_exist_of_successes": (
                float(np.mean([r["f1_exist"] for r in ok_rows])) if ok_rows else 0.0
            ),
            "macro_f1_type_of_successes": (
                float(np.mean([r["macro_f1_type"] for r in ok_rows])) if ok_rows else 0.0
            ),
            "scope": _scope_label(name, args.heavy_atom_scope),
        }
        r = results[name]
        print(
            f"[bp] {name:16s} n={r['n_complexes']:4d}  "
            f"success={r['success_rate']:.3f}  "
            f"exact_type={r['exact_type_rate_of_successes']:.3f}  "
            f"F1(exist)={r['f1_exist_of_successes']:.3f}  "
            f"F1(type)={r['macro_f1_type_of_successes']:.3f}  "
            f"[{r['scope']}]",
            flush=True,
        )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "tmQM bond perception (3D geometry -> molecular graph)",
        "protocol": {
            "ground_truth": "tmQM_X{1,2,3}.BO.gz Wiberg bond orders (same atom "
                             "indexing as the .xyz.gz geometries)",
            "bo_threshold": args.bo_threshold,
            "bond_type": "nearest of single(1.0)/double(2.0)/triple(3.0) by BO",
            "metrics": ["success_rate", "exact_type_rate_of_successes",
                        "f1_exist_of_successes", "macro_f1_type_of_successes"],
            "note": "each baseline is scored on the bond subset it can address; "
                    "the subset is named per row. No published tmQM bond-perception "
                    "number exists to compare against (2026-10-04 literature sweep).",
        },
        "metals": metals,
        "n_samples_per_metal": args.n_samples,
        "heavy_atom_scope": bool(args.heavy_atom_scope),
        "cfm": (
            {
                **torch_load_meta(Path(args.cfm_ckpt)),
                "decision": args.cfm_decision,
                "decision_threshold": args.cfm_decision_threshold,
            }
            if args.cfm_ckpt
            else None
        ),
        "results": results,
    }
    (out_dir / "tmqm_bond_perception.json").write_text(json.dumps(payload, indent=2))
    print(f"[bp] wrote {out_dir / 'tmqm_bond_perception.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
