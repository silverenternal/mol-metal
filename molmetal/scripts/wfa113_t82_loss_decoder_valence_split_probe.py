#!/usr/bin/env python
"""Wf-A113 (T82) — does the soft-valence LOSS agree with the decoder?

The training loop's ``gen_validity_weight`` term is a squared hinge on
per-atom valence (``valence_log_barrier``).  It is the only place the
objective says anything about valence, and it is fed a *different* table from
the one the decoder uses:

    loss      ``{**DEFAULT_VALENCES, **TMQM_MEASURED_METAL_CAPS}``
              ligand caps C 4 N 3 O 2 P 3 S 2 Se 2, unknown Z -> 4
              aromatic bond charged 1.5 at EVERY atom

    decoder   ``_COVALENT_VALENCE``
              ligand caps C 4 N 3 O 2 P 5 S 6 Se 6 As 5 B 3 Si 4
              aromatic bond charged 1.0 at a heteroatom (T78) and 1.0 for
              the third and later at a carbon (T81)

So on the corpus's own bond graph the two disagree in both directions: the
loss penalises atoms the decoder accepts (it pushes the model away from the
truth) and permits atoms the decoder refuses (it pays for a bond the decode
then drops).

This probe counts both, per element, on the six-metal cohort.

Writes ``metrics/by_metric/wfa113_t82_loss_decoder_valence_split.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t82_loss_decoder_valence_split_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t82_loss_decoder_valence_split.json"

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
_METAL_ZS = frozenset({26, 28, 29, 30, 44, 45, 46, 75, 76, 77, 78, 79})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0)
    args = ap.parse_args(argv)

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _COVALENT_VALENCE as DEC_CAP,
        _endpoint_valence_cost,
    )
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        DEFAULT_VALENCES as LOSS_CAP,
        TMQM_MEASURED_METAL_CAPS,
    )
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    def is_metal_z(z: int) -> bool:
        return z in _METAL_ZS

    loss_valences = {**LOSS_CAP, **TMQM_MEASURED_METAL_CAPS}
    #: The cap table the fix proposes: the decoder's own ligand caps, which are
    #: RDKit's valence model, with the measured metal caps (bond-order-sum
    #: units) on top.
    fixed_valences = {**DEC_CAP, **TMQM_MEASURED_METAL_CAPS}
    #: The barrier's own order lookup: index 4 (aromatic) -> 1.5.
    loss_order = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}

    n_atoms: Counter = Counter()
    loss_only: Counter = Counter()   # loss penalises, decoder accepts
    dec_only: Counter = Counter()    # decoder refuses, loss accepts
    both: Counter = Counter()
    neither: Counter = Counter()
    pat_loss_only: Dict[str, Counter] = defaultdict(Counter)
    #: Same split with the CAPS fixed but the flat 1.5 aromatic charge kept, so
    #: the residual is attributable to the aromatic rule alone.
    cap_only_loss_only: Counter = Counter()
    cap_only_dec_only: Counter = Counter()
    #: Third state: caps fixed AND the aromatic contribution charged the way
    #: the decoder charges it.  This must agree with the decoder everywhere;
    #: ``mismatch_final`` is the self-check.
    final_mismatch: Counter = Counter()
    n_complexes = 0
    cx_split = 0
    cx_cap_only = 0

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for _code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if not (set(syms.values()) & _SIX):
                continue
            syms = {i: s for i, s in syms.items() if s != "H"}
            bonds: Dict[Tuple[int, int], float] = {}
            for idx, _sym, nbrs in atoms:
                if idx not in syms:
                    continue
                for nbr_idx, bo in nbrs:
                    if nbr_idx == idx or bo < 0.3 or nbr_idx not in syms:
                        continue
                    key = (min(idx, nbr_idx), max(idx, nbr_idx))
                    if key not in bonds or bo > bonds[key]:
                        bonds[key] = bo

            loss_v: Dict[int, float] = defaultdict(float)
            dec_v: Dict[int, float] = defaultdict(float)
            final_v: Dict[int, float] = defaultdict(float)
            arom_seen: Counter = Counter()
            arom_final: Counter = Counter()
            pat: Dict[int, List[str]] = defaultdict(list)
            for (i, j), bo in sorted(bonds.items()):
                cls = bo_to_bond_class(bo)
                zi, zj = _Z_OF.get(syms.get(i, "")), _Z_OF.get(syms.get(j, ""))
                if zi is None or zj is None:
                    continue
                loss_v[i] += loss_order[cls]
                loss_v[j] += loss_order[cls]
                dec_v[i] += _endpoint_valence_cost(
                    cls, zi, zj, is_metal_z, arom_seen[i]
                )
                dec_v[j] += _endpoint_valence_cost(
                    cls, zj, zi, is_metal_z, arom_seen[j]
                )
                if cls == 4:
                    arom_seen[i] += 1
                    arom_seen[j] += 1
                # The barrier's own formula: non-aromatic part at its order,
                # aromatic part 1.0 at a heteroatom and 1.5/1.5/1.0/1.0...
                # at a carbon.
                for idx, zz, other in ((i, zi, zj), (j, zj, zi)):
                    if is_metal_z(other) and not is_metal_z(zz):
                        pass  # dative at the donor: costs the ligand nothing
                    elif is_metal_z(zz):
                        final_v[idx] += 1.0
                    elif cls == 4:
                        n = arom_final[idx]
                        if zz in {5, 7, 8, 15, 16, 33, 34}:
                            final_v[idx] += 1.0
                        else:
                            final_v[idx] += 1.5 if n < 2 else 1.0
                        arom_final[idx] = n + 1
                    else:
                        final_v[idx] += loss_order[cls]
                pat[i].append({1: "S", 2: "D", 3: "T", 4: "A"}[cls])
                pat[j].append({1: "S", 2: "D", 3: "T", 4: "A"}[cls])

            split = False
            cap_split = False
            for idx, sym in syms.items():
                z = _Z_OF.get(sym)
                if z is None or is_metal_z(z):
                    continue
                if z not in DEC_CAP:
                    continue
                n_atoms[sym] += 1
                lp = loss_v[idx] > float(loss_valences.get(z, 4)) + 1e-9
                dp = dec_v[idx] > float(DEC_CAP[z]) + 1e-9
                cp = loss_v[idx] > float(fixed_valences.get(z, 4)) + 1e-9
                fp = final_v[idx] > float(fixed_valences.get(z, 4)) + 1e-9
                key = tuple(sorted(pat[idx]))
                if lp and not dp:
                    loss_only[sym] += 1
                    pat_loss_only[sym][key] += 1
                    split = True
                elif dp and not lp:
                    dec_only[sym] += 1
                    split = True
                elif lp and dp:
                    both[sym] += 1
                else:
                    neither[sym] += 1
                if cp and not dp:
                    cap_only_loss_only[sym] += 1
                    cap_split = True
                if dp and not cp:
                    cap_only_dec_only[sym] += 1
                    cap_split = True
                if fp != dp:
                    final_mismatch[sym] += 1
            n_complexes += 1
            cx_split += int(split)
            cx_cap_only += int(cap_split)
            n += 1
            if args.max_complexes and n >= args.max_complexes:
                break

    per_element = {}
    for sym in sorted(n_atoms, key=lambda s: -loss_only[s]):
        c = n_atoms[sym]
        per_element[sym] = {
            "loss_cap": int(loss_valences.get(_Z_OF[sym], 4)),
            "decoder_cap": int(DEC_CAP.get(_Z_OF[sym], 0)),
            "n_atoms": c,
            "frac_loss_penalises_decoder_accepts": loss_only[sym] / c if c else 0.0,
            "frac_decoder_refuses_loss_accepts": dec_only[sym] / c if c else 0.0,
            "frac_loss_only_with_caps_fixed": (
                cap_only_loss_only[sym] / c if c else 0.0
            ),
            "frac_dec_only_with_caps_fixed": (
                cap_only_dec_only[sym] / c if c else 0.0
            ),
            "frac_both_over": both[sym] / c if c else 0.0,
            "top_loss_only_patterns": [
                {"pattern": "".join(k), "n": v}
                for k, v in pat_loss_only[sym].most_common(3)
            ],
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T82",
        "date": "2026-10-28",
        "question": (
            "does the soft-valence loss penalise atoms the decoder accepts, or "
            "accept atoms the decoder refuses, on the corpus's own bond graph?"
        ),
        "tables": {
            "loss": "DEFAULT_VALENCES + TMQM_MEASURED_METAL_CAPS, aromatic=1.5",
            "decoder": "_COVALENT_VALENCE, aromatic=1.0 at heteroatoms and at "
                       "a carbon's third and later aromatic bond",
        },
        "measured": {
            "cohort": "six project metals",
            "max_complexes_per_shard": args.max_complexes,
            "n_complexes": n_complexes,
            "n_atoms_total": sum(n_atoms.values()),
            "n_complexes_where_the_two_disagree": cx_split,
            "frac_complexes_where_the_two_disagree": (
                cx_split / n_complexes if n_complexes else 0.0
            ),
            "n_atoms_mismatched_after_both_fixes": sum(final_mismatch.values()),
            "n_complexes_where_the_two_disagree_caps_fixed": cx_cap_only,
            "frac_complexes_disagree_caps_fixed": (
                cx_cap_only / n_complexes if n_complexes else 0.0
            ),
            "per_element": per_element,
        },
        "script": (
            "molmetal/scripts/wfa113_t82_loss_decoder_valence_split_probe.py"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    m = payload["measured"]
    print(
        f"complexes {n_complexes}  disagreement in {cx_split} "
        f"({m['frac_complexes_where_the_two_disagree']:.4f})"
    )
    print(
        f"caps fixed only: {cx_cap_only} "
        f"({m['frac_complexes_disagree_caps_fixed']:.4f})"
    )
    print(
        f"{'el':3s} {'loss':>5s} {'dec':>4s} {'atoms':>8s} "
        f"{'loss>dec':>9s} {'dec>loss':>9s} {'capfix':>7s} {'capfix':>7s}"
    )
    for s, v in per_element.items():
        if v["n_atoms"] < 200:
            continue
        print(
            f"{s:3s} {v['loss_cap']:5d} {v['decoder_cap']:4d} {v['n_atoms']:8d} "
            f"{v['frac_loss_penalises_decoder_accepts']:9.4f} "
            f"{v['frac_decoder_refuses_loss_accepts']:9.4f} "
            f"{v['frac_loss_only_with_caps_fixed']:7.4f} "
            f"{v['frac_dec_only_with_caps_fixed']:7.4f}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
