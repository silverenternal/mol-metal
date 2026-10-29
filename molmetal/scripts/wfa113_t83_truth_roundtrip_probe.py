#!/usr/bin/env python
"""Wf-A113 (T83) — does the corpus's own bond graph survive the decoder's assembly?

The decoder never sees hydrogens: ``load_tmqm_clouds`` drops them, and
``_assemble_mol`` adds bare ``Chem.Atom(z)`` with no explicit H count.  RDKit
then computes an aromatic atom's implicit H from the *aromatic* bond
contribution (1.5 per bond), which is the wrong basis for a pyrrole-type
donor: an aromatic N with two aromatic ring bonds is charged 3.0, so RDKit
gives it zero H, and the kekulizer — which needs that N to hold two SINGLE
bonds plus one H — finds no assignment and refuses the WHOLE molecule.
Furan and thiophene survive because O and S are divalent, not because the
construction is right.

That is measurable without a model.  Take the corpus's true bond classes,
hand them to the decoder's own ``_assemble_mol``, and count how many
complexes come out sanitized.  This is the decoder's CEILING on the truth
graph: any complex that fails here can never be decoded correctly by any
checkpoint, because the head would have to predict something other than the
truth to survive.

Part 2 measures the second defect in the same repair chain:
``_detect_and_demote_anti_aromatic`` treats ``ring_size % 4 == 0`` as the
criterion for a ring RDKit cannot kekulize.  Measured on RDKit 2026.03.6, the
criterion is inverted for the rings it targets: 4-, 8-, 10- and 12-membered
all-aromatic monocycles all kekulize, while 3-, 5-, 7- and 9-membered ones
all fail.  The rule therefore demotes rings that need no repair, and misses
the odd rings that do.

Writes ``metrics/by_metric/wfa113_t83_truth_roundtrip.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t83_truth_roundtrip_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t83_truth_roundtrip.json"

BO_PATHS = [
    ROOT / ".data/molmetal/tmQM/tmQM_X1.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X2.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X3.BO.gz",
]

_Z_OF = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53,
    "Fe": 26, "Ni": 28, "Cu": 29, "Zn": 30, "Ru": 44, "Rh": 45, "Pd": 46,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
}
_SIX = {"Ru", "Rh", "Ir", "Os", "Re", "Pt"}


def _ring_sizes_all_aromatic(
    zs: List[int], bonds: List[Tuple[int, int, int]]
) -> Counter:
    """Ring sizes (SSSR) whose every bond is BOND_AROMATIC."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rw = Chem.RWMol()
    for z in zs:
        rw.AddAtom(Chem.Atom(int(z)))
    for i, j, o in bonds:
        bt = Chem.BondType.AROMATIC if int(o) == 4 else Chem.BondType.SINGLE
        try:
            rw.AddBond(int(i), int(j), bt)
        except Exception:
            pass
    m = rw.GetMol()
    try:
        Chem.GetSSSR(m)
    except Exception:
        return Counter()
    arom = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in m.GetBonds()
        if b.GetBondType() == Chem.BondType.AROMATIC
    }
    out: Counter = Counter()
    for ring in m.GetRingInfo().AtomRings():
        edges = [
            tuple(sorted((int(ring[k]), int(ring[(k + 1) % len(ring)]))))
            for k in range(len(ring))
        ]
        if all(e in arom for e in edges):
            out[len(ring)] += 1
    return out


def _has_pyrrole_n(zs: List[int], bonds: List[Tuple[int, int, int]]) -> bool:
    """An aromatic N with exactly two aromatic ring bonds and no other bond.

    This is the atom RDKit cannot kekulize once the H has been stripped: it
    needs ``[nH]`` (two single bonds plus one H) but the aromatic-valence
    calculation leaves no room for the H.
    """
    deg: Counter = Counter()
    arom_deg: Counter = Counter()
    for i, j, o in bonds:
        deg[i] += 1
        deg[j] += 1
        if int(o) == 4:
            arom_deg[i] += 1
            arom_deg[j] += 1
    for idx, z in enumerate(zs):
        if int(z) == 7 and deg[idx] == 2 and arom_deg[idx] == 2:
            return True
    return False


def _apply_pyrrole_h_fix(mol) -> int:
    """Give every aromatic N/P/As with two aromatic bonds and one H-slot a
    real explicit H before sanitisation.  Returns the number of atoms fixed.
    """
    from rdkit import Chem

    n_fixed = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (7, 15, 33):
            continue
        if not atom.GetIsAromatic():
            continue
        bonds = atom.GetBonds()
        if len(bonds) != 2:
            continue
        if not all(b.GetBondType() == Chem.BondType.AROMATIC for b in bonds):
            continue
        if atom.GetNumExplicitHs() or atom.GetFormalCharge():
            continue
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        n_fixed += 1
    return n_fixed


def _n_aromatic_rings(mol) -> int:
    from rdkit import Chem

    try:
        Chem.GetSSSR(mol)
    except Exception:
        return -1
    n = 0
    for ring in mol.GetRingInfo().AtomRings():
        try:
            if all(
                mol.GetBondBetweenAtoms(
                    int(ring[k]), int(ring[(k + 1) % len(ring)])
                ).GetIsAromatic()
                for k in range(len(ring))
            ):
                n += 1
        except Exception:
            continue
    return n


def _legacy_assemble(zs, bond_orders):
    """Reproduce the PRE-T83 repair chain exactly, for the before/after number.

    The old ``_assemble_mol`` ran ``_demote_aromatic_outside_rings`` and
    ``_demote_small_ring_aromatic``, then ``Chem.SanitizeMol(mol)`` and, in the
    ``except`` branch, handed the SAME ``mol`` to ``_demote_all_aromatic``.
    RDKit has already rewritten every aromatic bond to SINGLE by then, so the
    demotion is a no-op and the re-sanitise only succeeds on the stripped
    graph.  Returns ``(sanitized, repair)``.
    """
    from rdkit import Chem

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _demote_all_aromatic,
        _demote_aromatic_outside_rings,
        _demote_small_ring_aromatic,
    )

    bt = {
        1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for z in zs:
        rw.AddAtom(Chem.Atom(int(z)))
    for i, j, o in bond_orders:
        try:
            rw.AddBond(int(i), int(j), bt.get(int(o), Chem.BondType.SINGLE))
        except Exception:
            pass
    mol = _demote_aromatic_outside_rings(rw.GetMol())
    mol = _demote_small_ring_aromatic(mol)
    try:
        Chem.SanitizeMol(mol)
        return True, ""
    except Exception as exc:
        if "Kekulize" not in type(exc).__name__:
            return False, type(exc).__name__
        mol = _demote_all_aromatic(mol)
        try:
            Chem.SanitizeMol(mol)
            return True, "kekulize_demoted_all"
        except Exception as exc2:
            return False, type(exc2).__name__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0,
                    help="per shard; 0 = all")
    args = ap.parse_args(argv)

    from rdkit import Chem, RDLogger

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        BOND_AROMATIC,
        YuelBondBondDecoder,
    )
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks
    import torch

    RDLogger.DisableLog("rdApp.*")
    dec = YuelBondBondDecoder()

    n_cx = 0
    n_clean = 0
    n_repaired = 0
    n_failed = 0
    legacy_clean = 0
    legacy_repaired = 0
    legacy_failed = 0
    fail_reason: Counter = Counter()
    n_with_pyrrole_n = 0
    fail_with_pyrrole_n = 0
    n_clean_after_h_fix = 0
    n_fail_after_h_fix = 0
    h_fix_atoms = 0
    ring_hist: Counter = Counter()
    n_cx_has_4n_ring = 0
    n_4n_ring_total = 0
    examples: Dict[str, List[dict]] = defaultdict(list)
    # Reference-SMILES agreement: aromatic rings kept by the repair versus the
    # number the corpus's own SMILES column reports for the same complex.
    ref_delta: Counter = Counter()
    ref_available = 0
    ref_kept_any = 0
    ref_before: Counter = Counter()
    import pandas as pd

    meta_path = ROOT / ".data/molmetal/tmQM/tmqm_parsed.csv"
    meta = (
        pd.read_csv(meta_path).set_index("csd_code")
        if meta_path.is_file()
        else None
    )

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if not (set(syms.values()) & _SIX):
                continue
            # Heavy atoms only, matching load_tmqm_clouds.
            syms = {i: s for i, s in syms.items() if s != "H"}
            order = sorted(syms)
            remap = {old: new for new, old in enumerate(order)}
            zs = [_Z_OF[syms[i]] for i in order if syms[i] in _Z_OF]
            if len(zs) != len(order):
                continue
            bonds: Dict[Tuple[int, int], float] = {}
            for idx, _sym, nbrs in atoms:
                if idx not in remap:
                    continue
                for nbr_idx, bo in nbrs:
                    if nbr_idx == idx or bo < 0.3 or nbr_idx not in remap:
                        continue
                    key = (min(remap[idx], remap[nbr_idx]),
                           max(remap[idx], remap[nbr_idx]))
                    if key not in bonds or bo > bonds[key]:
                        bonds[key] = bo
            bond_orders = [
                (i, j, int(bo_to_bond_class(bo))) for (i, j), bo in bonds.items()
            ]
            n_cx += 1
            n += 1

            rs = _ring_sizes_all_aromatic(zs, bond_orders)
            for size, cnt in rs.items():
                ring_hist[size] += cnt
            four_n = sum(c for s, c in rs.items() if s % 4 == 0)
            n_4n_ring_total += four_n
            if four_n:
                n_cx_has_4n_ring += 1

            has_pyr = _has_pyrrole_n(zs, bond_orders)
            if has_pyr:
                n_with_pyrrole_n += 1

            cloud = AtomCloud(
                positions=torch.zeros((len(zs), 3), dtype=torch.float32),
                atomic_numbers=torch.tensor(zs, dtype=torch.long),
            )
            res = dec._assemble_mol(cloud, bond_orders, dative_bonds=[])
            if res.sanitized and not res.repair:
                n_clean += 1
            elif res.sanitized:
                n_repaired += 1
                fail_reason[f"repaired:{res.repair}"] += 1
            else:
                n_failed += 1
                reason = (res.error or "").split(":")[0] or "unknown"
                fail_reason[reason] += 1
                if has_pyr:
                    fail_with_pyrrole_n += 1
                if len(examples[reason]) < 3:
                    examples[reason].append({
                        "csd": code,
                        "n_atoms": len(zs),
                        "has_pyrrole_n": has_pyr,
                        "error": (res.error or "")[:90],
                    })

            if meta is not None and res.sanitized:
                smi = meta["smiles"].get(code)
                if isinstance(smi, str) and smi:
                    ref = Chem.MolFromSmiles(smi)
                    if ref is not None:
                        ref_available += 1
                        ref_before[_n_aromatic_rings(ref)] += 1
                        got = _n_aromatic_rings(res.mol)
                        ref_delta[got - _n_aromatic_rings(ref)] += 1
                        if got > 0:
                            ref_kept_any += 1

            # Before/after: run the pre-T83 repair chain on the same bonds.
            legacy_ok, legacy_repair = _legacy_assemble(zs, bond_orders)
            if not legacy_ok:
                legacy_failed += 1
            elif legacy_repair:
                legacy_repaired += 1
            else:
                legacy_clean += 1

            # Candidate fix: supply the missing donor H before sanitising.
            if has_pyr:
                rw = Chem.RWMol()
                for z in zs:
                    rw.AddAtom(Chem.Atom(int(z)))
                bt = {
                    1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
                }
                for i, j, o in bond_orders:
                    try:
                        rw.AddBond(int(i), int(j), bt.get(int(o), Chem.BondType.SINGLE))
                    except Exception:
                        pass
                m = rw.GetMol()
                h_fix_atoms += _apply_pyrrole_h_fix(m)
                try:
                    Chem.SanitizeMol(m)
                    n_clean_after_h_fix += 1
                except Exception:
                    n_fail_after_h_fix += 1

            if args.max_complexes and n >= args.max_complexes:
                break

    payload = {
        "wave": "Wf-A113",
        "task": "T83",
        "date": "2026-10-29",
        "question": (
            "given the corpus's TRUE bond classes, how many complexes can the "
            "decoder's _assemble_mol turn into a sanitized RDKit molecule?  "
            "This is the ceiling on any checkpoint."
        ),
        "method": (
            "tmQM BO shards -> heavy-atom graph -> bo_to_bond_class -> "
            "YuelBondBondDecoder._assemble_mol(cloud, true_bonds).  No model, "
            "no GPU, no sampling: the head is replaced by the answer."
        ),
        "n_complexes": n_cx,
        "assemble_truth": {
            "clean": n_clean,
            "repaired": n_repaired,
            "failed": n_failed,
            "clean_frac": n_clean / n_cx if n_cx else 0.0,
            "clean_or_repaired_frac": (n_clean + n_repaired) / n_cx if n_cx else 0.0,
            "outcomes": dict(fail_reason.most_common()),
            "examples": {k: v for k, v in examples.items()},
        },
        "before_the_fix": {
            "clean": legacy_clean,
            "repaired_all_aromaticity_destroyed": legacy_repaired,
            "failed": legacy_failed,
            "note": (
                "the pre-T83 chain, re-run on the same bond classes.  Every "
                "one of its repairs was `kekulize_demoted_all`, and that "
                "demotion was a no-op on an already-stripped molecule, so the "
                "'repair' was the aromaticity loss itself."
            ),
        },
        "reference_smiles_agreement": {
            "n_with_a_parsable_reference": ref_available,
            "reference_aromatic_ring_counts": {
                str(k): v for k, v in sorted(ref_before.items())
            },
            "delta_repaired_minus_reference": {
                f"{k:+d}": v for k, v in sorted(ref_delta.items())
            },
            "exact_aromatic_ring_match": ref_delta.get(0, 0),
            "exact_frac": (
                ref_delta.get(0, 0) / ref_available if ref_available else 0.0
            ),
            "repaired_molecules_keeping_any_aromatic_ring": ref_kept_any,
            "note": (
                "the pre-fix chain demoted EVERY aromatic bond on the same "
                "molecules, so its delta was -reference for all of them; the "
                "delta here is what the ring-local repair recovers"
            ),
        },
        "pyrrole_nitrogen": {
            "complexes_with_an_aromatic_N_of_degree_2": n_with_pyrrole_n,
            "frac_of_cohort": n_with_pyrrole_n / n_cx if n_cx else 0.0,
            "failed_complexes_that_have_one": fail_with_pyrrole_n,
            "h_atoms_supplied": h_fix_atoms,
            "clean_after_supplying_the_donor_H": n_clean_after_h_fix,
            "still_failing_after": n_fail_after_h_fix,
            "note": (
                "the fix is applied only to the subset that has such an N, so "
                "'clean_after' is comparable to the failed count of the same "
                "subset, not to n_complexes"
            ),
        },
        "all_aromatic_ring_sizes_sssr": {
            str(k): v for k, v in sorted(ring_hist.items())
        },
        "anti_aromatic_demotion": {
            "complexes_with_a_4n_all_aromatic_ring": n_cx_has_4n_ring,
            "frac_of_cohort": n_cx_has_4n_ring / n_cx if n_cx else 0.0,
            "n_4n_rings_total": n_4n_ring_total,
            "measured_rdkit_kekulize_behaviour": {
                "succeeds": [4, 6, 8, 10, 12],
                "fails": [3, 5, 7, 9],
                "rule": "even ring size kekulizes, odd ring size does not",
                "code_rule": "_is_anti_aromatic returns ring_size % 4 == 0",
                "consequence": (
                    "the demotion targets exactly the rings RDKit handles and "
                    "never touches the odd rings that fail"
                ),
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
