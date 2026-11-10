#!/usr/bin/env python
"""Wf-A142 — decode-side DEMONSTRATION that bond-order DEMOTION (not edge
deletion) is what frees the valence budget a FORCED bridge needs.

Why this script exists
----------------------
Frontier row 5 of the Wf-A140 audit
(``/tmp/wfa140/frontier-lit.md`` §1, row 5) is the published repair of
Yao, Gruich, Goldsmith, Wang, *"Fixed-Dimensional Latent Flow for
Generating Variable-Size 3D Molecules"*, arXiv:2609.08333, §C.2 / Fig. 7:
the two verified quotes are

    "Their bond orders are lowered one level at a time"
    "only when the resulting graph passes RDKit sanitization; no edge is
     deleted."

The repo already IMPLEMENTS that rule — the demotion helper is
:func:`YuelBondBondDecoder._demote_one_order`
(``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1959``), called
from :meth:`YuelBondBondDecoder._force_connectivity`
(``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1701``) at the two
valence-budget gates (``cfm_decoder_yuelbond.py:1916`` and ``:1928``).
What the repo does NOT have is a MEASUREMENT of how often the demotion
path is the thing that actually rescues a forced bridge.  This script
produces one, on CPU, with no checkpoint and no GPU.

The situation being measured
----------------------------
The c7 epoch-17 pool of record is FRAGMENTED: the generated molecules come
out as several components (``metrics/by_metric/c7_fullgen_pool_quality.json``
-> ``overall.single_fragment_rate`` 0.0851).  When two of those components
sit close enough to be inside the bond head's own 3.5 A candidate window,
the decoder's forced stage can bridge them.  The bridge costs ONE unit of
valence budget at each endpoint.  If the endpoint is already saturated by
the bonds the head emitted, the bridge cannot be paid for — and the two
published ways to pay for it are:

  (a) LOWER a bond order (double -> single, triple -> double): frees
      1.0 of budget, removes no edge, removes no atom.  This is the
      published repair (arXiv:2609.08333 §C.2).
  (b) DELETE an edge: frees 1.0 of budget, but removes a bond and splits
      the molecule further.  The published repair forbids this.

This script counts how often (a) is the thing that makes the bridge
possible, and how often (b) would have been the only alternative.

Scenario construction
---------------------
Each scenario is built from a REAL row of the c7 pool
(``molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv``):

  * the row's two largest heavy-atom fragments are extracted (they are
    pieces of one generated molecule, with real bond orders and real
    chemistry);
  * each is embedded in 3-D with RDKit ETKDG (CPU, no GPU);
  * fragment B is translated so that the CLOSEST heavy-atom pair between
    the two fragments sits at exactly ``--bridge-distance`` (default
    2.9 A — inside the head's 3.5 A candidate window, outside the spring's
    reach, i.e. the "2.5-3.5 A dead zone" the frontier audit names);
  * the two fragments' TRUE bonds form the head's emitted bond list, so
    every atom carries its REAL valence budget;
  * the head's candidate pair set is rebuilt exactly as the head builds
    it (all heavy pairs strictly inside 3.5 A, minus the steric-collapse
    veto), and the logits are set so that only the intended cross-fragment
    pair clears the forced stage's floor.

Two experiments are run:

  * **EXP-A (natural placement)** — fragment B is placed along the
    centroid-to-centroid axis, so which atom ends up closest is whatever
    the chemistry says.  This is the UNBIASED rate.
  * **EXP-B (saturated endpoint)** — fragment B is placed so that the
    closest pair lands on a heavy atom of fragment A whose valence budget
    is already fully consumed.  This is the stress case for the demotion
    path.

Each scenario is run TWICE: with the production `_demote_one_order`, and
counterfactually with it stubbed to return ``None`` (the published repair
disabled).  The difference between the two runs is the number of forced
bridges that exist ONLY because a bond order was lowered.

A third, CONTRAST experiment is reported:

  * **EXP-C (cut a bond)** — the same molecule with one of its own bonds
    cut, and the cut bond itself proposed as the bridge.  This shows the
    opposite regime (the cut has already freed budget), and it is the
    reason EXP-A/EXP-B are built from TWO fragments rather than from a
    cut.

MEASURED-vs-PROJECTED
---------------------
Every number this script writes into ``demo.json`` is MEASURED on this
machine from a file on disk.  Nothing is projected.  The published
literature numbers (98.4 % / 92.7 % F1, 3 A cutoff, 217/1000) are quoted
ONLY as citations and are never an input to any measurement here.

CPU-only
--------
No CUDA call is made anywhere: no checkpoint is loaded, no model forward
is run, no ODE is integrated.  The decoder is constructed with a
randomly-initialised head because the head's WEIGHTS are irrelevant here —
the candidate pair set is rebuilt from the geometry (exactly what the
head's own cutoff does) and the logits are set by the scenario.  Run with
``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from molmetal.models.bond_head import (  # noqa: E402
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    AtomCloud,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (  # noqa: E402
    YuelBondResult,
    pair_collapse_mask,
)

#: The head's own candidate-edge radius (``YuelBondDecoderHead.cutoff``).
#: Wf-A140 row 6 records that the PUBLISHED method uses 3 A and ours is
#: 3.5 A, i.e. ours is already wider; this script must use OURS, because
#: the question is what OUR decoder does.
HEAD_CUTOFF_A = 3.5

#: The forced stage's logit floor (``connectivity_force_min_logit``).
#: A candidate pair whose best non-NO_BOND logit is below this is skipped.
FORCE_FLOOR = -2.0

#: Logit assigned to the intended bridge pair's SINGLE class.  The head
#: "declined" the bond (NO_BOND is the argmax) but its SINGLE logit is
#: still above the forced floor, which is exactly the case
#: `_force_connectivity` exists for.
BRIDGE_SINGLE_LOGIT = -1.0

#: Logit assigned to a pair that IS a real bond (the head is confident).
TRUE_BOND_LOGIT = 5.0

#: Logit assigned to a candidate pair that is NOT a bond.  Its best
#: non-NO_BOND logit is below the forced floor, so the forced stage
#: ignores it — the same as a head that says NO_BOND.
FALSE_BOND_LOGIT = -5.0

_POOL_CSV = (
    _REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c7_fullgen_2026_10_20"
    / "generated_smiles.csv"
)

_RDKIT_BOND_TO_CLASS = {
    Chem.BondType.SINGLE: BOND_SINGLE,
    Chem.BondType.DOUBLE: BOND_DOUBLE,
    Chem.BondType.TRIPLE: BOND_TRIPLE,
    Chem.BondType.AROMATIC: BOND_AROMATIC,
}

_CLASS_NAME = {
    BOND_SINGLE: "single",
    BOND_DOUBLE: "double",
    BOND_TRIPLE: "triple",
    BOND_AROMATIC: "aromatic",
}


# ----------------------------------------------------------------------
# corpus
# ----------------------------------------------------------------------
def load_pool_rows(csv_path: Path, epoch: Optional[int]) -> List[Dict[str, str]]:
    """Read the c7 pool CSV, optionally filtered to one epoch."""
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if epoch is not None:
                try:
                    if int(row["epoch"]) != int(epoch):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def fragments_by_heavy_atoms(mol: Chem.Mol) -> List[Chem.Mol]:
    """The molecule's fragments, sorted by heavy-atom count (descending)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    out = []
    for frag in frags:
        heavy = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() != 1)
        out.append((heavy, frag))
    out.sort(key=lambda t: -t[0])
    return [f for _h, f in out]


def embed_3d(mol: Chem.Mol, seed: int) -> Optional[Chem.Mol]:
    """Embed ``mol`` in 3-D with ETKDG (CPU).  ``None`` on failure."""
    work = Chem.Mol(mol)
    work = Chem.AddHs(work)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % (2**31 - 1)
    params.useRandomCoords = True
    params.maxIterations = 200
    if AllChem.EmbedMolecule(work, params) != 0:
        params.useBasicKnowledge = False
        if AllChem.EmbedMolecule(work, params) != 0:
            return None
    work = Chem.RemoveHs(work)
    return work


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def heavy_arrays(
    mol3d: Chem.Mol,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]]:
    """(z array, positions array, heavy bond list) for a 3-D molecule."""
    if mol3d.GetNumConformers() == 0:
        return None
    conf = mol3d.GetConformer()
    keep = [a.GetIdx() for a in mol3d.GetAtoms() if a.GetAtomicNum() != 1]
    if len(keep) < 1:
        return None
    index = {rdx: k for k, rdx in enumerate(keep)}
    z = np.zeros(len(keep), dtype=np.int64)
    pos = np.zeros((len(keep), 3), dtype=np.float32)
    for rdx, k in index.items():
        atom = mol3d.GetAtomWithIdx(rdx)
        z[k] = int(atom.GetAtomicNum())
        p = conf.GetAtomPosition(rdx)
        pos[k] = (p.x, p.y, p.z)
    bonds: List[Tuple[int, int, int]] = []
    for bond in mol3d.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        bonds.append((i, j, _RDKIT_BOND_TO_CLASS.get(bond.GetBondType(), BOND_SINGLE)))
    return z, pos, bonds


def _min_cross_distance(pos_a: np.ndarray, pos_b: np.ndarray, shift: np.ndarray) -> float:
    d = pos_a[:, None, :] - (pos_b[None, :, :] + shift[None, None, :])
    return float(np.sqrt((d**2).sum(-1)).min())


def place_b_at_distance(
    pos_a: np.ndarray, pos_b: np.ndarray, target: float
) -> np.ndarray:
    """Translation of fragment B so the closest A-B heavy pair is ``target`` away.

    Fragment B is moved along the centroid-to-centroid axis, so the
    direction is chemistry-neutral and the result is deterministic.
    """
    ca = pos_a.mean(axis=0)
    cb = pos_b.mean(axis=0)
    axis = ca - cb
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    # Grow t until the closest pair is at least `target` apart.
    hi = 1.0
    while _min_cross_distance(pos_a, pos_b, hi * axis) < target:
        hi *= 2.0
        if hi > 500.0:
            break
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _min_cross_distance(pos_a, pos_b, mid * axis) < target:
            lo = mid
        else:
            hi = mid
    return hi * axis


def place_b_next_to_atom(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    atom_a: int,
    target: float,
) -> Tuple[np.ndarray, int]:
    """Translation of fragment B so ``atom_a`` is ``target`` away from one B atom.

    B is pushed OUTWARD from A's centroid along the ray through
    ``atom_a``, and the B atom that ends up closest to ``atom_a`` is
    placed exactly ``target`` away from it.  Returns
    ``(shift, atom_b_index)``.

    The caller then names the pair ``(atom_a, atom_b)`` as THE bridge
    candidate and gives it the only logit above the forced floor, so the
    forced stage's choice is the intended one regardless of whether some
    other A-B pair happens to be closer.
    """
    ca = pos_a.mean(axis=0)
    axis = pos_a[atom_a] - ca
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    target_point = pos_a[atom_a].astype(np.float64) + target * axis
    d = pos_b.astype(np.float64) - target_point[None, :]
    j_star = int(np.argmin((d**2).sum(-1)))
    shift = target_point - pos_b[j_star].astype(np.float64)
    return shift, j_star


def closest_pair(
    pos_a: np.ndarray, pos_b: np.ndarray, shift: np.ndarray
) -> Tuple[int, int, float]:
    """(i in A, j in B, distance) of the closest cross pair."""
    d = pos_a[:, None, :] - (pos_b[None, :, :] + shift[None, None, :])
    dist = np.sqrt((d**2).sum(-1))
    k = int(np.argmin(dist))
    i, j = divmod(k, dist.shape[1])
    return int(i), int(j), float(dist[i, j])


# ----------------------------------------------------------------------
# the head's own candidate rule
# ----------------------------------------------------------------------
def candidate_pairs(
    z: np.ndarray, pos: np.ndarray, decoder: Any
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """The head's post-veto candidate pair set, rebuilt from the geometry.

    Reproduces ``YuelBondDecoderHead.forward``'s candidate rule: every
    heavy-atom pair strictly inside ``HEAD_CUTOFF_A``, minus the
    steric-collapse veto (``pair_collapse_mask`` with the head's own
    ``clash_ratio``).  The forced stage draws its candidates from exactly
    this set, so the scenario must use exactly this set.
    """
    z = np.asarray(z, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.float32)
    n = len(z)
    ii, jj, dd = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[j] - pos[i]))
            if d < HEAD_CUTOFF_A:
                ii.append(i)
                jj.append(j)
                dd.append(d)
    if not ii:
        return [], np.zeros((0,), dtype=np.float32)
    z_t = torch.from_numpy(z)
    i_t = torch.tensor(ii, dtype=torch.long)
    j_t = torch.tensor(jj, dtype=torch.long)
    d_t = torch.tensor(dd, dtype=torch.float32)
    veto = pair_collapse_mask(
        z_t, i_t, j_t, d_t, decoder._decoder.head.clash_ratio
    ).bool().numpy()
    pairs = [
        (int(a), int(b)) for a, b, keep in zip(ii, jj, ~veto) if keep
    ]
    d_all = np.asarray(dd, dtype=np.float32)[~veto]
    return pairs, d_all


def build_logits(
    pairs: Sequence[Tuple[int, int]],
    bonds: Sequence[Tuple[int, int, int]],
    bridge_pair: Tuple[int, int],
) -> torch.Tensor:
    """The head's logits for every candidate pair.

    * a real bond     -> its true class gets ``TRUE_BOND_LOGIT``;
    * the bridge pair -> NO_BOND wins, but SINGLE sits above the floor;
    * anything else   -> NO_BOND wins by a wide margin, so the forced
      stage ignores it.
    """
    bond_class_of = {
        ((i, j) if i < j else (j, i)): order for i, j, order in bonds
    }
    logits = torch.full((len(pairs), 5), FALSE_BOND_LOGIT, dtype=torch.float32)
    for k, (i, j) in enumerate(pairs):
        key = (i, j) if i < j else (j, i)
        logits[k, BOND_NO_BOND] = -TRUE_BOND_LOGIT
        if key == bridge_pair:
            logits[k, BOND_NO_BOND] = 0.0
            logits[k, BOND_SINGLE] = BRIDGE_SINGLE_LOGIT
        elif key in bond_class_of:
            logits[k, bond_class_of[key]] = TRUE_BOND_LOGIT
    return logits


def make_yres(edge_index: torch.Tensor, logits: torch.Tensor) -> YuelBondResult:
    """A ``YuelBondResult`` carrying only what `_force_connectivity` reads."""
    return YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
        error=None,
        clash_rejected_pairs=0,
    )


# ----------------------------------------------------------------------
# running the forced stage
# ----------------------------------------------------------------------
class _NoDemotion:
    """Context manager: disable the published demotion repair."""

    def __init__(self, decoder: YuelBondBondDecoder) -> None:
        self._decoder = decoder
        self._saved = None

    def __enter__(self):
        self._saved = self._decoder._demote_one_order
        self._decoder._demote_one_order = lambda *a, **k: None  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        self._decoder._demote_one_order = self._saved  # type: ignore[assignment]
        return False


def run_forced(
    decoder: YuelBondBondDecoder,
    cloud: AtomCloud,
    yres: YuelBondResult,
    bond_orders: List[Tuple[int, int, int]],
) -> Tuple[Any, List[Tuple[int, int, int]]]:
    """Run `_force_connectivity` and return (stats, resulting bond list)."""
    bonds = list(bond_orders)
    dative: List[Tuple[int, int]] = []
    stats = decoder._force_connectivity(cloud, yres, bonds, dative)
    return stats, bonds


def sanitize_check(
    z: Sequence[int], bonds: Sequence[Tuple[int, int, int]]
) -> str:
    """Can RDKit sanitize the graph after the forced stage?

    One of ``ok`` / ``fail_valence`` / ``fail_kekulize`` / ``fail_other``.
    """
    rw = Chem.RWMol()
    for zi in z:
        rw.AddAtom(Chem.Atom(int(zi)))
    bt = {
        BOND_SINGLE: Chem.BondType.SINGLE,
        BOND_DOUBLE: Chem.BondType.DOUBLE,
        BOND_TRIPLE: Chem.BondType.TRIPLE,
        BOND_AROMATIC: Chem.BondType.AROMATIC,
    }
    for i, j, order in bonds:
        try:
            rw.AddBond(int(i), int(j), bt[int(order)])
        except Exception:
            return "fail_other"
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        text = str(exc)
        if "ekuliz" in text:
            return "fail_kekulize"
        if "valence" in text.lower():
            return "fail_valence"
        return "fail_other"
    return "ok"


# ----------------------------------------------------------------------
# one scenario
# ----------------------------------------------------------------------
def evaluate_scenario(
    decoder: YuelBondBondDecoder,
    z: np.ndarray,
    pos: np.ndarray,
    bonds: List[Tuple[int, int, int]],
    pairs: List[Tuple[int, int]],
    bridge_pair: Tuple[int, int],
) -> Dict[str, Any]:
    """Run the forced stage twice (with / without the demotion repair)."""
    logits = build_logits(pairs, bonds, bridge_pair)
    edge_index = torch.tensor(
        [[p[0] for p in pairs], [p[1] for p in pairs]], dtype=torch.long
    ).reshape(2, len(pairs))
    cloud = AtomCloud(
        positions=torch.from_numpy(np.asarray(pos, dtype=np.float32)),
        atomic_numbers=torch.from_numpy(np.asarray(z, dtype=np.int64)),
    )
    yres = make_yres(edge_index, logits)

    stats_on, bonds_on = run_forced(decoder, cloud, yres, list(bonds))
    with _NoDemotion(decoder):
        stats_off, bonds_off = run_forced(decoder, cloud, yres, list(bonds))

    added_on = int(stats_on.bonds_added)
    added_off = int(stats_off.bonds_added)
    demoted = int(stats_on.bonds_demoted)

    base_pairs = {(min(i, j), max(i, j)) for i, j, _o in bonds}
    after_pairs = {(min(i, j), max(i, j)) for i, j, _o in bonds_on}
    base_orders = {(min(i, j), max(i, j)): o for i, j, o in bonds}
    after_orders = {(min(i, j), max(i, j)): o for i, j, o in bonds_on}
    edges_deleted = len(base_pairs - after_pairs)
    orders_lowered = sum(
        1
        for key, o0 in base_orders.items()
        if key in after_orders and after_orders[key] < o0
    )

    if added_on == 1 and added_off == 0:
        outcome = "rescued_by_demotion"
    elif added_on == 1 and added_off == 1:
        outcome = "bridge_without_demotion"
    elif added_on == 0 and added_off == 0:
        outcome = "bridge_refused"
    else:  # pragma: no cover - defensive
        outcome = "unexpected"

    return {
        "outcome": outcome,
        "bonds_added_with_demotion": added_on,
        "bonds_added_without_demotion": added_off,
        "bonds_demoted": demoted,
        "orders_lowered": orders_lowered,
        "edges_deleted": edges_deleted,
        "reasons": {str(k): int(v) for k, v in (getattr(stats_on, "reasons", {}) or {}).items()},
        "sanitize_after_forced_stage": sanitize_check(z, bonds_on),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pool-csv", type=Path, default=_POOL_CSV)
    p.add_argument("--epoch", type=int, default=17)
    p.add_argument("--max-molecules", type=int, default=200)
    p.add_argument("--bridge-distance", type=float, default=2.9)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument(
        "--reference-csv",
        type=Path,
        default=_REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
    )
    p.add_argument("--max-reference", type=int, default=150)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT
        / "molmetal"
        / "reports"
        / "wfa142_demotion_demo"
        / "demo.json",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    if not args.pool_csv.exists():
        print(f"FAIL: pool CSV not found: {args.pool_csv}")
        return 1

    rows = load_pool_rows(args.pool_csv, args.epoch)
    pool_total_rows = len(load_pool_rows(args.pool_csv, None))
    print(
        f"[wfa142] pool rows (epoch={args.epoch}): {len(rows)} "
        f"of {pool_total_rows} total"
    )

    decoder = YuelBondBondDecoder()
    head_clash_ratio = float(decoder._decoder.head.clash_ratio)

    skip = Counter()
    exp_a = Counter()
    exp_b1 = Counter()
    exp_b2 = Counter()
    exp_c = Counter()
    sanitize = Counter()
    reasons_a = Counter()
    reasons_b1 = Counter()
    reasons_b2 = Counter()
    endpoint_stats = Counter()
    samples: List[Dict[str, Any]] = []

    n_seen = 0
    n_used = 0
    for row in rows:
        if n_used >= args.max_molecules:
            break
        n_seen += 1
        smi = (row.get("smiles") or "").strip()
        if not smi:
            skip["row_empty_smiles"] += 1
            continue
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            skip["row_unparseable"] += 1
            continue
        frags = fragments_by_heavy_atoms(mol)
        if len(frags) < 2:
            skip["row_single_fragment"] += 1
            continue
        big, second = frags[0], frags[1]
        if big.GetNumAtoms() < 3:
            skip["row_largest_fragment_too_small"] += 1
            continue
        a3 = embed_3d(big, args.seed + n_seen)
        b3 = embed_3d(second, args.seed + 100000 + n_seen)
        if a3 is None or b3 is None:
            skip["row_embed_failed"] += 1
            continue
        arr_a = heavy_arrays(a3)
        arr_b = heavy_arrays(b3)
        if arr_a is None or arr_b is None:
            skip["row_no_heavy_arrays"] += 1
            continue
        z_a, pos_a, bonds_a = arr_a
        z_b, pos_b, bonds_b = arr_b
        if not bonds_a:
            skip["row_no_bonds"] += 1
            continue
        n_used += 1
        skip["row_used"] += 1

        # --- combined system: fragment A then fragment B -----------------
        offset = len(z_a)
        z = np.concatenate([z_a, z_b])
        bonds = list(bonds_a) + [(i + offset, j + offset, o) for i, j, o in bonds_b]

        # --- EXP-A: natural placement along the centroid axis -------------
        shift = place_b_at_distance(pos_a, pos_b, args.bridge_distance)
        pos = np.concatenate([pos_a, pos_b + shift.astype(np.float32)])
        ia, jb, dist = closest_pair(pos_a, pos_b, shift)
        bridge = (ia, jb + offset)
        pairs, _d = candidate_pairs(z, pos, decoder)
        if bridge not in pairs:
            skip["expA_bridge_not_in_candidate_set"] += 1
        else:
            res = evaluate_scenario(decoder, z, pos, bonds, pairs, bridge)
            exp_a[res["outcome"]] += 1
            exp_a["scenarios_total"] += 1
            sanitize["A:" + res["sanitize_after_forced_stage"]] += 1
            for k, v in res["reasons"].items():
                reasons_a[k] += int(v)
            endpoint_stats["A_total"] += 1
            if res["outcome"] == "rescued_by_demotion":
                endpoint_stats["A_rescued"] += 1
                samples.append({"exp": "A", "bridge_distance_A": round(dist, 4), **res})
            elif res["outcome"] == "bridge_refused":
                n_del = _edges_to_delete_for_budget(bonds, bridge[0])
                endpoint_stats[
                    "A_refused_edge_deletion_would_free" if n_del is not None
                    else "A_refused_no_edge_deletion_would_free"
                ] += 1
                if n_del is not None:
                    endpoint_stats["A_refused_edge_deletion_count_total"] += n_del
                samples.append({"exp": "A", "bridge_distance_A": round(dist, 4), **res})

        # --- EXP-B: place B next to a SATURATED atom of A ------------------
        used_b = place_b_at_distance(pos_a, pos_b, args.bridge_distance)
        saturated_demotable, saturated_plain = _saturated_atoms(
            decoder, z_a, bonds_a
        )
        endpoint_stats["A_saturated_with_demotable_bond"] += len(saturated_demotable)
        endpoint_stats["A_saturated_without_demotable_bond"] += len(saturated_plain)
        # B1 = the endpoint the published repair CAN serve (it carries a
        # DOUBLE/TRIPLE bond whose order can be lowered).  B2 = the
        # endpoint it CANNOT serve (every incident bond is SINGLE or
        # AROMATIC, so nothing can be lowered and the only way to pay for
        # the bridge would be to delete an edge — which the same published
        # rule forbids).
        for variant, atom_list, counter, reason_counter in (
            ("B1", saturated_demotable, exp_b1, reasons_b1),
            ("B2", saturated_plain, exp_b2, reasons_b2),
        ):
            if not atom_list:
                skip[f"exp{variant}_no_saturated_atom_of_this_kind_in_A"] += 1
                continue
            atom_a = int(atom_list[0])
            endpoint_stats[f"{variant}_total"] += 1
            shift_b, atom_b = place_b_next_to_atom(
                pos_a, pos_b, atom_a, args.bridge_distance
            )
            pos_b2 = np.asarray(pos_b + shift_b.astype(np.float32), dtype=np.float32)
            pos_b3 = np.concatenate([pos_a, pos_b2])
            z_ab = np.concatenate([z_a, z_b])
            bridge_b = (atom_a, atom_b + offset)
            pairs_b, _d = candidate_pairs(z_ab, pos_b3, decoder)
            if bridge_b not in pairs_b:
                skip[f"exp{variant}_bridge_not_in_candidate_set"] += 1
                continue
            res_b = evaluate_scenario(decoder, z_ab, pos_b3, bonds, pairs_b, bridge_b)
            counter[res_b["outcome"]] += 1
            counter["scenarios_total"] += 1
            sanitize[f"{variant}:" + res_b["sanitize_after_forced_stage"]] += 1
            for k, v in res_b["reasons"].items():
                reason_counter[k] += int(v)
            if res_b["outcome"] == "rescued_by_demotion":
                endpoint_stats[f"{variant}_rescued"] += 1
                samples.append(
                    {"exp": variant, "bridge_pair": list(bridge_b), **res_b}
                )
            elif res_b["outcome"] == "bridge_refused":
                n_del = _edges_to_delete_for_budget(bonds, atom_a)
                endpoint_stats[
                    f"{variant}_refused_edge_deletion_would_free"
                    if n_del is not None
                    else f"{variant}_refused_no_edge_deletion_would_free"
                ] += 1
                if n_del is not None:
                    endpoint_stats[f"{variant}_refused_edge_deletion_count_total"] += n_del

        # --- EXP-C (contrast): cut one of A's own bonds, bridge the cut ----
        if bonds_a:
            ci, cj, corder = bonds_a[len(bonds_a) // 2]
            cut = (ci, cj)
            kept = [(i, j, o) for (i, j, o) in bonds if (i, j) != cut]
            pos_c = np.concatenate([pos_a, pos_b + used_b.astype(np.float32)])
            pairs_c, _d = candidate_pairs(z, pos_c, decoder)
            res_c = evaluate_scenario(
                decoder, z, pos_c, kept, pairs_c, (min(cut), max(cut))
            )
            exp_c[res_c["outcome"]] += 1
            sanitize["C:" + res_c["sanitize_after_forced_stage"]] += 1

    def _rates(counter: Counter) -> Dict[str, Any]:
        total = max(1, int(counter.get("scenarios_total", 0)) or sum(counter.values()))
        return {
            "total": int(total),
            "rescued_by_demotion": int(counter["rescued_by_demotion"]),
            "rescued_by_demotion_rate": round(
                counter["rescued_by_demotion"] / total, 6
            ),
            "bridge_without_demotion": int(counter["bridge_without_demotion"]),
            "bridge_without_demotion_rate": round(
                counter["bridge_without_demotion"] / total, 6
            ),
            "bridge_refused": int(counter["bridge_refused"]),
            "bridge_refused_rate": round(counter["bridge_refused"] / total, 6),
        }

    # --- EXP-0: direct chemistry census over the epoch-17 pool ----------
    epoch_smiles = [
        (r.get("smiles") or "").strip()
        for r in rows
        if (r.get("smiles") or "").strip()
    ]
    census = census_chemistry(decoder, epoch_smiles)

    # --- EXP-D: positive control on real (non-generated) chemistry --------
    ref_smiles: List[str] = []
    ref_path = args.reference_csv
    if ref_path.exists():
        with ref_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s = (row.get("smiles") or "").strip()
                if s:
                    ref_smiles.append(s)
    exp_d, reasons_d, samples_d = run_reference_control(
        decoder,
        ref_smiles,
        args.bridge_distance,
        args.seed,
        args.max_reference,
    )

    report: Dict[str, Any] = {
        "wave": "Wf-A142",
        "date": "2026-10-31",
        "mode": "CPU-only; no pre-existing repo file was modified",
        "gpu_used": False,
        "question": (
            "How often does bond-order DEMOTION (rather than edge deletion) "
            "free the valence budget a forced bridge needs?"
        ),
        "published_rule_under_test": {
            "source": (
                "Yao, Gruich, Goldsmith, Wang, 'Fixed-Dimensional Latent Flow "
                "for Generating Variable-Size 3D Molecules', arXiv:2609.08333, "
                "§C.2 / Fig. 7"
            ),
            "quotes": [
                "Their bond orders are lowered one level at a time",
                "only when the resulting graph passes RDKit sanitization; "
                "no edge is deleted.",
            ],
            "verification": (
                "PUBLISHED — quotes verified in the Wf-A140 frontier audit "
                "(/tmp/wfa140/frontier-lit.md §1 row 5, §4 last row)"
            ),
        },
        "implementation_under_test": {
            "demotion_helper": (
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1959 "
                "YuelBondBondDecoder._demote_one_order"
            ),
            "caller": (
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1701 "
                "YuelBondBondDecoder._force_connectivity"
            ),
            "call_sites": [
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1916 "
                "(endpoint i budget gate)",
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1928 "
                "(endpoint j budget gate)",
            ],
            "head_candidate_cutoff_A": HEAD_CUTOFF_A,
            "forced_floor_logit": FORCE_FLOOR,
            "head_clash_ratio": head_clash_ratio,
            "bridge_distance_A": args.bridge_distance,
        },
        "corpus": {
            "path": str(args.pool_csv.relative_to(_REPO_ROOT)),
            "epoch_filter": args.epoch,
            "pool_rows_total": pool_total_rows,
            "pool_rows_in_epoch": len(rows),
            "rows_examined": n_seen,
            "rows_used": n_used,
            "skip_counts": {k: int(v) for k, v in sorted(skip.items())},
        },
        "scenario_definition": (
            "For each pool row: the two largest heavy-atom fragments are "
            "ETKDG-embedded (CPU) and re-assembled as ONE disconnected "
            "system, so the head's bond list is the fragments' TRUE bonds and "
            "every atom carries its real valence budget. Fragment B is then "
            "translated so the closest A-B heavy pair sits at "
            f"{args.bridge_distance} A — inside the head's 3.5 A candidate "
            "window. The forced stage is asked to bridge it, twice: with the "
            "production _demote_one_order, and with it stubbed to return "
            "None (the published repair disabled)."
        ),
        "experiments": {
            "A_natural_placement": {
                "what": (
                    "Fragment B is placed along the centroid-to-centroid "
                    "axis; which atom ends up closest is whatever the "
                    "chemistry says. Unbiased."
                ),
                **_rates(exp_a),
                "reasons": {k: int(v) for k, v in sorted(reasons_a.items())},
            },
            "B1_saturated_endpoint_with_demotable_bond": {
                "what": (
                    "Fragment B is placed so the bridge pair lands on a heavy "
                    "atom of fragment A whose valence budget is fully "
                    "consumed AND which carries a DOUBLE or TRIPLE bond — "
                    "the only case the published demotion repair can serve."
                ),
                **_rates(exp_b1),
                "reasons": {k: int(v) for k, v in sorted(reasons_b1.items())},
            },
            "B2_saturated_endpoint_without_demotable_bond": {
                "what": (
                    "The same placement, but on a saturated endpoint whose "
                    "incident bonds are all SINGLE or AROMATIC. Nothing can "
                    "be lowered there, so the published repair cannot pay "
                    "for the bridge and the only alternative would be to "
                    "DELETE an edge — which the same rule forbids."
                ),
                **_rates(exp_b2),
                "reasons": {k: int(v) for k, v in sorted(reasons_b2.items())},
            },
            "C_contrast_cut_own_bond": {
                "what": (
                    "One of fragment A's OWN bonds is cut and the cut bond "
                    "is proposed as the bridge. The cut has already freed "
                    "budget at that endpoint, so this is the regime where "
                    "demotion is NOT needed. Reported as the contrast that "
                    "explains why EXP-A/EXP-B are built from two fragments."
                ),
                **_rates(exp_c),
            },
            "D_positive_control_real_chemistry": {
                "what": (
                    "The SAME forced-bridge scenario applied to REAL "
                    "drug-like molecules from the reference corpus "
                    "(.data/molmetal/metalcytotox_5metal_multitask_v4.csv), "
                    "where saturated heavy atoms carrying a DOUBLE/TRIPLE "
                    "bond DO exist. If the demotion path fires here, the "
                    "harness can see it, and a zero on the c7 pool is a "
                    "property of the pool's chemistry rather than a "
                    "broken measurement."
                ),
                **_rates(exp_d),
                "rows_used": int(exp_d.get("rows_used", 0)),
                "saturated_with_demotable_bond": int(
                    exp_d.get("saturated_with_demotable_bond", 0)
                ),
                "saturated_without_demotable_bond": int(
                    exp_d.get("saturated_without_demotable_bond", 0)
                ),
                "reasons": {k: int(v) for k, v in sorted(reasons_d.items())},
            },
        },
        "EXP0_pool_chemistry_census": {
            "what": (
                "Direct census of the epoch-17 pool's chemistry (no "
                "scenario machinery): how many heavy atoms are "
                "valence-saturated, and how many of those carry a DOUBLE or "
                "TRIPLE bond — the only orders the published demotion "
                "repair can lower."
            ),
            "corpus": "c7 epoch-17 pool (all rows)",
            **census,
        },
        "sanitize_after_forced_stage": {k: int(v) for k, v in sorted(sanitize.items())},
        "endpoint_and_alternative_counts": {
            k: int(v) for k, v in sorted(endpoint_stats.items())
        },
        "wall_seconds": round(time.time() - t0, 2),
        "example_scenarios": samples[:40],
        "example_scenarios_reference_control": samples_d[:20],
        "tag": (
            "MEASURED — every count above was produced by this run on this "
            "machine from files on disk"
        ),
    }

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=False) + "\n"
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out, payload)
    for name, counter in (("A", exp_a), ("B1", exp_b1), ("B2", exp_b2), ("C", exp_c), ("D", exp_d)):
        print(
            f"[wfa142] EXP-{name}: total={counter.get('scenarios_total', sum(counter.values()))} "
            f"rescued_by_demotion={counter['rescued_by_demotion']} "
            f"bridge_without_demotion={counter['bridge_without_demotion']} "
            f"bridge_refused={counter['bridge_refused']}"
        )
    print(
        "[wfa142] EXP-0 census: saturated_heavy_atoms={s} "
        "with_demotable={d} without_demotable={n}".format(
            s=census.get("saturated_heavy_atoms", 0),
            d=census.get("saturated_with_demotable_bond", 0),
            n=census.get("saturated_without_demotable_bond", 0),
        )
    )
    print(f"[wfa142] wrote {out}")
    return 0


def _saturated_atoms(
    decoder: YuelBondBondDecoder, z: np.ndarray, bonds: Sequence[Tuple[int, int, int]]
) -> Tuple[List[int], List[int]]:
    """Saturated heavy atoms of a fragment, split by DEMOTABILITY.

    Returns ``(with_demotable_bond, without_demotable_bond)``.

    A saturated atom is one whose valence budget is fully consumed.  The
    published repair (arXiv:2609.08333 §C.2) can only help if the atom
    carries a bond whose ORDER can be lowered — i.e. a DOUBLE or a TRIPLE.
    A saturated atom whose bonds are all SINGLE has nothing to lower, and
    for it the only way to pay for a bridge would be to DELETE an edge,
    which the same published rule forbids.  Splitting the two groups is
    the whole point of the measurement.
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _replay_remaining_budget,
    )

    caps = [decoder._cap_for_z(int(zi)) for zi in z]
    remaining, _arom = _replay_remaining_budget(
        list(bonds), [int(zi) for zi in z], caps, decoder._is_metal_z
    )
    with_demotable: List[int] = []
    without: List[int] = []
    for k, rem in enumerate(remaining):
        if rem > 0.0:
            continue
        incident = [o for i, j, o in bonds if k in (i, j)]
        if not incident:
            continue
        if any(o in (BOND_DOUBLE, BOND_TRIPLE) for o in incident):
            with_demotable.append(k)
        else:
            without.append(k)
    return with_demotable, without


def _edges_to_delete_for_budget(
    bonds: Sequence[Tuple[int, int, int]], endpoint: int, need: float = 1.0
) -> Optional[int]:
    """Minimum edges to DELETE at ``endpoint`` to free ``need`` of budget.

    This is the alternative the published repair forbids; it is counted
    so the report can say how big the forbidden alternative would have
    been.  ``None`` means no edge at that endpoint could free the budget.
    """
    freed = 0.0
    n = 0
    for i, j, order in bonds:
        if endpoint not in (i, j):
            continue
        freed += float(order)
        n += 1
        if freed >= need - 1e-9:
            return n
    return None


# ----------------------------------------------------------------------
# EXP-0 — direct chemistry census (no scenario machinery)
# ----------------------------------------------------------------------
def census_chemistry(
    decoder: YuelBondBondDecoder, smiles_iter: Sequence[str]
) -> Dict[str, Any]:
    """How many heavy atoms are valence-saturated, and can they be demoted?

    For every molecule, every fragment, every heavy atom: is the atom's
    valence budget fully consumed, and does it carry a DOUBLE or TRIPLE
    bond (the only orders the published demotion repair can lower)?
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _replay_remaining_budget,
    )

    counts = Counter()
    for smi in smiles_iter:
        counts["molecules"] += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            counts["unparseable"] += 1
            continue
        for frag in fragments_by_heavy_atoms(mol):
            z = [int(a.GetAtomicNum()) for a in frag.GetAtoms()]
            if not z:
                continue
            bonds: List[Tuple[int, int, int]] = []
            for bond in frag.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bonds.append(
                    (a, b, _RDKIT_BOND_TO_CLASS.get(bond.GetBondType(), BOND_SINGLE))
                )
            caps = [decoder._cap_for_z(zi) for zi in z]
            remaining, _arom = _replay_remaining_budget(
                bonds, z, caps, decoder._is_metal_z
            )
            for k, rem in enumerate(remaining):
                if rem > 0.0:
                    continue
                incident = [o for i, j, o in bonds if k in (i, j)]
                if not incident:
                    continue
                counts["saturated_heavy_atoms"] += 1
                if any(o in (BOND_DOUBLE, BOND_TRIPLE) for o in incident):
                    counts["saturated_with_demotable_bond"] += 1
                else:
                    counts["saturated_without_demotable_bond"] += 1
            for _i, _j, o in bonds:
                counts["bond_order_" + _CLASS_NAME.get(o, str(o))] += 1
    counts["heavy_atoms_total"] = sum(
        v for k, v in counts.items() if k.startswith("bond_order_")
    ) // 2
    return {k: int(v) for k, v in sorted(counts.items())}


# ----------------------------------------------------------------------
# EXP-D — positive control on REAL (non-generated) chemistry
# ----------------------------------------------------------------------
def run_reference_control(
    decoder: YuelBondBondDecoder,
    smiles_list: Sequence[str],
    bridge_distance: float,
    seed: int,
    max_mols: int,
) -> Tuple[Counter, Counter, List[Dict[str, Any]]]:
    """The SAME forced-bridge scenario, on real drug-like molecules.

    The c7 pool is GENERATED chemistry; the reference corpus is real.
    This control exists so that a zero on the pool can be read correctly:
    if the demotion path fires here, the harness can SEE it, and a zero
    on the pool is a property of the pool's chemistry rather than a
    broken measurement.
    """
    counts = Counter()
    reasons = Counter()
    samples: List[Dict[str, Any]] = []
    used = 0
    for smi in smiles_list:
        if used >= max_mols:
            break
        counts["rows_seen"] += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            counts["unparseable"] += 1
            continue
        frags = fragments_by_heavy_atoms(mol)
        if not frags:
            counts["no_fragment"] += 1
            continue
        big = frags[0]
        if big.GetNumAtoms() < 3:
            counts["fragment_too_small"] += 1
            continue
        a3 = embed_3d(big, seed + used)
        b3 = embed_3d(big, seed + 500000 + used)
        if a3 is None or b3 is None:
            counts["embed_failed"] += 1
            continue
        arr_a = heavy_arrays(a3)
        arr_b = heavy_arrays(b3)
        if arr_a is None or arr_b is None:
            counts["no_heavy_arrays"] += 1
            continue
        z_a, pos_a, bonds_a = arr_a
        z_b, pos_b, bonds_b = arr_b
        used += 1
        counts["rows_used"] += 1
        offset = len(z_a)
        z = np.concatenate([z_a, z_b])
        bonds = list(bonds_a) + [(i + offset, j + offset, o) for i, j, o in bonds_b]
        sat_dem, sat_plain = _saturated_atoms(decoder, z_a, bonds_a)
        counts["saturated_with_demotable_bond"] += len(sat_dem)
        counts["saturated_without_demotable_bond"] += len(sat_plain)
        if not sat_dem:
            counts["no_saturated_demotable_atom"] += 1
            continue
        atom_a = int(sat_dem[0])
        shift_b, atom_b = place_b_next_to_atom(
            pos_a, pos_b, atom_a, bridge_distance
        )
        pos_b2 = np.asarray(pos_b + shift_b.astype(np.float32), dtype=np.float32)
        pos2 = np.concatenate([pos_a, pos_b2])
        bridge = (atom_a, atom_b + offset)
        pairs, _d = candidate_pairs(z, pos2, decoder)
        if bridge not in pairs:
            counts["bridge_not_in_candidate_set"] += 1
            continue
        res = evaluate_scenario(decoder, z, pos2, bonds, pairs, bridge)
        counts[res["outcome"]] += 1
        counts["scenarios_total"] += 1
        for k, v in res["reasons"].items():
            reasons[k] += int(v)
        if res["outcome"] == "rescued_by_demotion":
            samples.append({"exp": "D", "bridge_pair": list(bridge), **res})
    return counts, reasons, samples


if __name__ == "__main__":
    raise SystemExit(main())
