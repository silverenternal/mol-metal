"""CLI driver for the Phase-0 DesignLoop pipeline (TODO/11 W1).

Wires the standard adapter stack for the demo run:

* :class:`MockGenerator`            — random 8-atom molecules
* :class:`DiffDockAdapter` (STUB)   — random SE(3) poses, realistic Vina range
* :class:`RDKitPropertyPredictor`   — real QED / logP / MW / TPSA / SA-score
* :class:`WeightedSumScorer`        — multi-objective weighted sum

The :class:`SMILESInjectingGenerator` wrapper around ``MockGenerator``
synthesises a plausible canonical SMILES for each generated molecule
so the RDKit predictor can compute real 2D descriptors.  This is a
*Phase-0* convenience; the real generator (Lipman flow-matching) will
emit SMILES directly.

Usage
-----

    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.run_design_loop --metal Ru --n-generate 100 \\
        --n-dock 20 --n-top 5

Notes
-----
* No real docking happens here — DiffDockAdapter runs in STUB mode until
  Phase 2 wires up the actual model.
* The ``--pdb-id`` argument is recorded in the report but the loop uses
  a synthetic 64-atom pocket unless the user passes ``--use-real-pocket``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.diffdock import DiffDockAdapter
from molmetal.adapters.mock import MockGenerator
from molmetal.adapters.rdkit_predictor import RDKitPropertyPredictor
from molmetal.domain import Molecule, Pocket
from molmetal.orchestration.design_loop import (
    DesignLoop,
    LoopResult,
)
from molmetal.orchestration.closed_loop import WeightedSumScorer
from molmetal.ports import GenerationConfig


REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


# ---------------------------------------------------------------------------
# Synthetic pocket (no real PDB loader needed for the smoke run)
# ---------------------------------------------------------------------------
def _make_synthetic_pocket(pdb_id: str = "SYNTH", n_atoms: int = 64) -> Pocket:
    """Build a small random pocket with realistic-ish atom-type distribution.

    The Phase-0 demo does not need a real PDB parse; we only need a
    ``Pocket`` with the right shape so generator + docker can run end
    to end.
    """
    coords = torch.randn(n_atoms, 3, dtype=torch.float32) * 8.0
    # C/N/O/S — typical protein atom distribution
    pool = torch.tensor([6, 7, 8, 16], dtype=torch.long)
    idx = torch.randint(0, len(pool), (n_atoms,))
    atom_types = pool[idx]
    return Pocket(
        pdb_id=pdb_id,
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.arange(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=coords.mean(dim=0),
        radius=10.0,
    )


# ---------------------------------------------------------------------------
# SMILES post-processing for the mock generator
# ---------------------------------------------------------------------------
def _mock_smiles_for(mol: Molecule) -> str:
    """Heuristic SMILES from an 8-atom mock molecule.

    The :class:`MockGenerator` produces 8 atoms with types in {C, N, O}
    linked as a single chain.  We synthesise a plausible SMILES string
    so the RDKit predictor can compute real 2D descriptors.

    Returns ``""`` if reconstruction fails — the predictor will then
    emit a default ``PropertyPrediction`` and the candidate will
    naturally rank low.
    """
    types = mol.atom_types.tolist()
    elem_map = {6: "C", 7: "N", 8: "O"}
    parts: List[str] = []
    for i, t in enumerate(types):
        e = elem_map.get(int(t), "C")
        if e == "C":
            parts.append("C")
        elif e == "N":
            parts.append("N")
        elif e == "O":
            parts.append("O")
    # Insert branch markers at every O to add a bit of variety.
    smiles_chars: List[str] = []
    for i, p in enumerate(parts):
        if p == "O" and 0 < i < len(parts) - 1:
            smiles_chars.append("(")
            smiles_chars.append("O")
            smiles_chars.append(")")
            continue
        smiles_chars.append(p)
    smi = "".join(smiles_chars)
    # Validate via RDKit; if it doesn't parse, return "" (predictor
    # falls back to defaults).
    try:
        from rdkit import Chem  # noqa: WPS433

        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return ""


def _attach_smiles(molecules: List[Molecule]) -> List[Molecule]:
    """Return new ``Molecule`` objects with ``smiles`` set (frozen dataclass)."""
    out: List[Molecule] = []
    for mol in molecules:
        smi = _mock_smiles_for(mol)
        out.append(
            Molecule(
                coords=mol.coords,
                atom_types=mol.atom_types,
                bonds=mol.bonds,
                bond_types=mol.bond_types,
                formal_charges=mol.formal_charges,
                smiles=smi,
                qed=mol.qed,
                sa_score=mol.sa_score,
                logp=mol.logp,
            )
        )
    return out


# ---------------------------------------------------------------------------
# SMILESInjectingGenerator — MoleculeGenerator wrapper
# ---------------------------------------------------------------------------
class SMILESInjectingGenerator:
    """Wraps a generator and attaches plausible SMILES to its output.

    The :class:`MockGenerator` produces molecules with empty SMILES,
    which the RDKit predictor cannot use.  This wrapper post-processes
    each output batch with :func:`_attach_smiles` so the predictor can
    compute real 2D descriptors.

    It duck-types the :class:`MoleculeGenerator` protocol — same name,
    setup, generate, train_step, get_metadata attributes.
    """

    @property
    def name(self) -> str:
        return f"SMILESInjectingGenerator({self._inner.name})"

    def __init__(self, inner) -> None:
        self._inner = inner

    def setup(self, device: str = "cpu") -> None:
        setup = getattr(self._inner, "setup", None)
        if setup is not None:
            setup(device=device)

    def generate(self, pocket: Pocket, config: GenerationConfig) -> List[Molecule]:
        mols = list(self._inner.generate(pocket, config))
        return _attach_smiles(mols)

    def train_step(self, pocket: Pocket, mols: List[Molecule]) -> float:
        ts = getattr(self._inner, "train_step", None)
        if ts is None:
            return 0.0
        return float(ts(pocket, mols))

    def get_metadata(self) -> dict:
        meta = self._inner.get_metadata() if hasattr(self._inner, "get_metadata") else {}
        meta = dict(meta)
        meta["wrapper"] = "SMILESInjectingGenerator_v1"
        meta["smiles_source"] = "heuristic atom-type chain reconstruction"
        return meta


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def _print_top_table(result: LoopResult, metal: str) -> None:
    print()
    print(f"=== DesignLoop Phase-0 result ({metal}) ===")
    print(f"pocket_id        : {result.pocket_id}")
    print(f"n_generate       : {result.n_generate}")
    print(f"n_dock           : {result.n_dock}")
    print(f"n_top            : {result.n_top}")
    print(f"refinement_iters : {result.refinement_iters}")
    print(f"wall_clock_s     : {result.wall_clock_s:.2f}")
    print(f"n_candidates     : {len(result.candidates)}")
    print()
    print(
        f"{'rank':>4}  {'score':>8}  {'qed':>6}  {'mw':>7}  "
        f"{'sa':>6}  {'vina':>8}  {'smiles'}"
    )
    print("-" * 96)
    for c in result.candidates:
        vina = f"{c.vina_score:8.3f}" if c.vina_score is not None else "    n/a"
        print(
            f"{c.rank:>4}  {c.combined_score:>8.4f}  {c.qed:>6.3f}  "
            f"{c.mol_weight:>7.2f}  {c.sa_score:>6.3f}  {vina}  {c.smiles}"
        )
    print()
    # Per-iteration summary
    print("Per-iteration summary:")
    for r in result.per_iteration_summary:
        print(
            f"  iter={r.get('iteration'):>2} "
            f"gen={r.get('n_generated', 0):>4} "
            f"dock={r.get('n_docked', 0):>3} "
            f"best={r.get('best_score', float('nan')):.4f} "
            f"mean={r.get('mean_score', float('nan')):.4f} "
            f"({r.get('elapsed_s', 0.0):.2f}s)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Phase-0 DesignLoop pipeline (mock + DiffDock STUB + RDKit)."
    )
    p.add_argument("--metal", default="Ru", help="Target metal (recorded in metadata).")
    p.add_argument("--pdb-id", default="SYNTH", help="Pocket PDB ID (synthetic by default).")
    p.add_argument("--n-generate", type=int, default=100, help="Number of molecules to generate.")
    p.add_argument("--n-dock", type=int, default=20, help="Top-K to dock after pre-filter.")
    p.add_argument("--n-top", type=int, default=5, help="Top-N to return.")
    p.add_argument("--refinement-iters", type=int, default=0, help="Extra refinement rounds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None, help="Optional path to dump the LoopResult as JSON.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    print(f"[design_loop] Starting Phase-0 run (metal={args.metal}, pocket={args.pdb_id})")
    t0 = time.time()

    # --- 1. Build adapters ---------------------------------------------------
    base_generator = MockGenerator(seed=args.seed)
    # Wrap so generated molecules carry a synthesised SMILES — required
    # for RDKit descriptors.
    generator = SMILESInjectingGenerator(base_generator)
    generator.setup(device="cpu")

    docker = DiffDockAdapter(checkpoint_path=None)  # STUB mode
    docker.setup(device="cpu")

    predictor = RDKitPropertyPredictor()
    predictor.setup(device="cpu")

    scorer = WeightedSumScorer()

    loop = DesignLoop(
        generator=generator,
        docker=docker,
        predictor=predictor,
        scorer=scorer,
    )

    # --- 2. Pocket -----------------------------------------------------------
    pocket = _make_synthetic_pocket(pdb_id=args.pdb_id)

    # --- 3. Run --------------------------------------------------------------
    # For the demo run we dock every generated molecule (n_dock = n_generate)
    # so the Vina score is always present in the candidates table.  In
    # production the n_dock cap is what keeps the bill under control.
    n_dock_eff = min(args.n_dock, args.n_generate)
    result: LoopResult = loop.run(
        pocket=pocket,
        n_generate=args.n_generate,
        n_dock=n_dock_eff,
        n_top=args.n_top,
        refinement_iters=args.refinement_iters,
        seed=args.seed,
    )

    print(f"[design_loop] Done in {time.time() - t0:.2f}s")
    _print_top_table(result, args.metal)

    # --- 4. Optional JSON dump ---------------------------------------------
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop non-serialisable pose tensors.
        serialisable = {
            "pocket_id": result.pocket_id,
            "n_generate": result.n_generate,
            "n_dock": result.n_dock,
            "n_top": result.n_top,
            "refinement_iters": result.refinement_iters,
            "wall_clock_s": result.wall_clock_s,
            "per_iteration_summary": result.per_iteration_summary,
            "loop_metadata": result.loop_metadata,
            "candidates": [
                {k: v for k, v in asdict(c).items() if k != "pose_coords"}
                for c in result.candidates
            ],
        }
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(f"[design_loop] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
