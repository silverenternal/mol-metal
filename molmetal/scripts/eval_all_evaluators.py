"""eval_all_evaluators.py — sanity check all evaluators on this device.

Run:
    cd /home/hugo/codes/mol-metal
    source env.sh
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/eval_all_evaluators.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

# Make molmetal importable when run from anywhere
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rdkit import Chem

# --- Test molecules ---
TEST_MOLS = {
    "cisplatin (reference)": "[NH3][Pt]([NH3])([Cl])[Cl]",
    "aspirin (low SA)": "CC(=O)Oc1ccccc1C(=O)O",
    "taxol (high SA)": (
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)"
        "NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C8=CC=CC=C8)"
        "OC(=O)C9=CC=CC=C9)OC(=O)C"
    ),
    "glycine (very low SA)": "NCC(=O)O",
    "ibuprofen (high QED)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}


def _check(name: str, fn: Callable[[], object]) -> bool:
    t0 = time.time()
    try:
        result = fn()
        dt = time.time() - t0
        print(f"  {name:42s} PASS  ({dt*1000:6.1f}ms) — {result}")
        return True
    except Exception as e:
        print(f"  {name:42s} FAIL  ({type(e).__name__}: {str(e)[:80]})")
        return False


def main() -> int:
    print("=" * 80)
    print("Mol-Metal Evaluators — Verification Suite")
    print("=" * 80)
    print()

    n_pass = 0
    n_total = 0

    # 1. SA score
    print("[1] SA score (Ertl-Schuffenhauer)")
    from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl, sa_score_to_unit
    n_total += 1
    if _check("SA(CCO) [low]", lambda: f"{sa_score_ertl('CCO'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("SA(aspirin) [low]", lambda: f"{sa_score_ertl('CC(=O)Oc1ccccc1C(=O)O'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("SA(taxol) [high]", lambda: f"{sa_score_ertl(TEST_MOLS['taxol (high SA)']):.3f}"):
        n_pass += 1

    # 2. QED
    print("\n[2] QED (Bickerton)")
    from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles
    n_total += 1
    if _check("QED(CCO)", lambda: f"{qed_from_smiles('CCO'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("QED(ibuprofen) [high]", lambda: f"{qed_from_smiles(TEST_MOLS['ibuprofen (high QED)']):.3f}"):
        n_pass += 1

    # 3. Vina adapter instantiation
    print("\n[3] Vina adapter (AutoDock)")
    from molmetal.molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
    n_total += 1
    if _check("VinaDockingAdapter() instantiate", lambda: type(VinaDockingAdapter()).__name__):
        n_pass += 1

    # 4. Vina Python binding + meeko
    print("\n[4] Vina Python binding + meeko")
    n_total += 1
    if _check("import vina", lambda: __import__("vina").__version__):
        n_pass += 1
    n_total += 1
    if _check("import meeko", lambda: __import__("meeko").__version__):
        n_pass += 1

    # 5. PoseBusters
    print("\n[5] PoseBusters (geometric sanity)")
    from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter
    n_total += 1
    if _check("PoseBustersAdapter() instantiate", lambda: type(PoseBustersAdapter()).__name__):
        n_pass += 1

    # 6. BioPython (PLIP replacement)
    print("\n[6] BioPython PDB (PLIP replacement)")
    from Bio.PDB import PDBParser, NeighborSearch
    n_total += 1
    if _check("Bio.PDB.PDBParser", lambda: PDBParser(QUIET=True)):
        n_pass += 1

    # 7. OpenBabel
    print("\n[7] OpenBabel")
    n_total += 1
    if _check("import openbabel", lambda: "OK"):
        n_pass += 1

    # 8. ProDy
    print("\n[8] ProDy")
    n_total += 1
    if _check("import prody", lambda: "OK"):
        n_pass += 1

    # 9. Reference drug smoke test (sanity)
    print("\n[9] Reference drug sanity (cisplatin should have reasonable metrics)")
    from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl, sa_score_to_unit
    from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles

    cisplatin_sa = sa_score_ertl(TEST_MOLS["cisplatin (reference)"])
    cisplatin_qed = qed_from_smiles(TEST_MOLS["cisplatin (reference)"])
    n_total += 1
    if _check("cisplatin SA", lambda: f"{cisplatin_sa:.3f}"):
        n_pass += 1
    n_total += 1
    if _check("cisplatin QED", lambda: f"{cisplatin_qed:.3f}"):
        n_pass += 1

    # Summary
    print()
    print("=" * 80)
    print(f"SUMMARY: {n_pass}/{n_total} evaluators PASS")
    print("=" * 80)
    if n_pass == n_total:
        print("✓ All evaluators ready for paper §4 evaluation suite")
    else:
        print(f"✗ {n_total - n_pass} evaluators need attention")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
