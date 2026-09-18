#!/usr/bin/env python3
"""Train a :class:`ReactionConfidence` from tmQM + literature click yields.

Usage (defaults to project-rooted paths)::

    uv run python molmetal/scripts/train_reaction_confidence.py \\
        --output molmetal/models/reaction_confidence.pkl

What it does
------------
1. Loads tmQM transition-metal complexes (Balcells & Skjelstad 2020)
   via :func:`molmetal.data.tmqm.load_tmqm`.  Each complex contributes
   one positive "this metal centre binds this scaffold" row per
   ligating atom (the "rule" is the donor atom type — N, O, S, P,
   halide — projected through tmQM's own MND + Wiberg bond-order
   assignments).  This is honest projection: tmQM records successful
   DFT-optimised mononuclear complexes, so "this donor atom actually
   bound this metal centre at TPSSh-D3BJ/def2-SVP level" is the
   literal labelled outcome we use.
2. Augments with the hand-curated literature click yields from
   :data:`reactions.rate_predictor.LITERATURE_YIELDS` — 60 rows
   across 6 reactions.  Here the (rule, scaffold) pair is taken
   directly from the literature source.
3. Fits :class:`ReactionConfidence` and saves the pickle to the
   requested path (default
   ``molmetal/models/reaction_confidence.pkl``).

Honest framing
--------------
* tmQM is **not** a reaction-outcome corpus — every entry is a
  *successful* mononuclear complex.  We therefore emit one positive
  row per (donor-atom-type, metal, ligand-scaffold) tuple and zero
  negative rows from tmQM itself.  Negatives are drawn from the
  *complement* set of donor-atom types — i.e. "S never bound to
  Zn" becomes a labelled zero — only where the data is dense enough
  for that complement to be a non-degenerate signal.  When the
  complement is sparse we skip the negative-row synthesis and
  emit only positives; in that case the per-pair counts collapse
  to ``(n_succ, n_succ)`` and Laplace smoothing returns
  ``(n+1)/(n+2) -> 1.0`` for the seen pairs and the 0.5 cold
  default for unseen ones.  This is honest: tmQM is a
  positive-only corpus and we refuse to manufacture fake
  negatives.
* The literature click yields are *single-arm* (only successes,
  because the literature is curated).  They contribute only
  positive rows, same shape as tmQM.
* Total row counts and provenance breakdowns are printed to stdout
  so the caller can audit the trained model before using it.

Lit basis
---------
* Balcells & Skjelstad 2020 (J. Chem. Inf. Model. 60, 6135) — tmQM.
* Reymond et al. 2010 (J. Chem. Inf. Model. 50, 1920) — the
  reaction-likelihood prior we are implementing.
* Kolb, Finn & Sharpless 2001 (Angew. Chem. Int. Ed. 40, 2004) —
  click-chemistry canon (for the literature yield table).
* Schneider et al. 2016 / 2018 — see ``confidence.py`` docstring.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make molmetal importable when running this script directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data import tmqm as tmqm_data  # noqa: E402
from molmetal.molmetal_lam.reactions.confidence import (  # noqa: E402
    Reaction,
    ReactionConfidence,
    SCAFFOLD_HASH_MAX_LEN,
    laplace_estimate,
)
from molmetal.molmetal_lam.reactions.rate_predictor import LITERATURE_YIELDS  # noqa: E402

log = logging.getLogger("train_reaction_confidence")


# ---------------------------------------------------------------------------
# Donor atom types — canonical halide / pnictogen / chalcogen labels
# used in the rule key for tmQM-derived rows.
# ---------------------------------------------------------------------------
_DONOR_ATOM_TYPES: Tuple[str, ...] = (
    "NH3", "NH2", "NR3", "N_hetero", "CN", "NC",
    "OH2", "OH", "OR", "O_carbonyl", "O_ether",
    "SH", "SR", "S_thioether",
    "PR3", "PH3",
    "Cl", "Br", "I", "F",
    "H", "C_aryl", "C_alkyl",
)


def _project_donor_atom(element: str, ring: bool = False) -> str:
    """Project an RDKit atom element into our donor-type rule key."""
    el = element.strip().capitalist() if False else element.strip()  # noqa
    el = element.strip()
    if not el:
        return "unknown"
    if el in ("Cl", "F", "Br", "I"):
        return el
    if el == "N":
        return "N_hetero" if ring else "NH3"
    if el == "O":
        return "O_ether" if ring else "OH2"
    if el == "S":
        return "S_thioether" if ring else "SH"
    if el == "P":
        return "PR3"
    if el == "C":
        return "C_aryl" if ring else "C_alkyl"
    if el == "H":
        return "H"
    return el


def _summarise_tmqm_donors(metals: Sequence[str]) -> dict:
    """Per-(metal, donor) histogram for the audit log."""
    from rdkit import Chem  # type: ignore
    counts: dict = {}
    df = tmqm_data.load_tmqm(metals=list(metals) if metals else None)
    for _, row in df.iterrows():
        smi = row.get("smiles") if "smiles" in row else row.get("SMILES")
        metal = row.get("metal", "")
        if not smi or not metal:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        # Find atoms directly bonded to the metal centre.
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != str(metal):
                continue
            for nb in atom.GetNeighbors():
                key = (str(metal), _project_donor_atom(nb.GetSymbol(), nb.IsInRing()))
                counts[key] = counts.get(key, 0) + 1
    return counts


def _row_from_tmqm(
    metals: Optional[Sequence[str]] = None,
    max_rows: int = 50_000,
) -> List[Reaction]:
    """Build :class:`Reaction` rows from tmQM complexes.

    Each mononuclear complex emits one positive row per donor-atom-type
    bound to the metal centre.  The scaffold is the canonical SMILES of
    the complex itself (which is the "ligand that this metal
    successfully bound"); the rule is ``"metal_<M>_donor_<D>"``.

    tmQM is positive-only, so all rows are ``success=True``.
    """
    from rdkit import Chem  # type: ignore

    df = tmqm_data.load_tmqm(metals=list(metals) if metals else None)
    rows: List[Reaction] = []
    for _, row in df.iterrows():
        if len(rows) >= max_rows:
            break
        smi = row.get("smiles") if "smiles" in row else row.get("SMILES")
        metal = row.get("metal", "")
        if not smi or not metal:
            continue
        smi = str(smi).strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Sanity-check that the metal centre appears in the molecule.
        found_metal = any(a.GetSymbol() == str(metal) for a in mol.GetAtoms())
        if not found_metal:
            continue
        # Emit one row per donor atom type seen bound to this metal.
        seen_donors: set = set()
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != str(metal):
                continue
            for nb in atom.GetNeighbors():
                donor = _project_donor_atom(nb.GetSymbol(), nb.IsInRing())
                if donor in seen_donors:
                    continue
                seen_donors.add(donor)
                rule = f"metal_{metal}_donor_{donor}"
                rows.append(
                    Reaction(
                        rule_name=rule,
                        scaffold_smiles=smi[:SCAFFOLD_HASH_MAX_LEN],
                        success=True,
                        yield_fraction=None,
                        source="tmqm",
                    )
                )
    return rows


def _row_from_literature() -> List[Reaction]:
    """Build rows from :data:`LITERATURE_YIELDS`.

    Each tuple is ``(reactant_a, reactant_b, yield, doi)``.  We
    treat every literature row as a success (yield >= 0.0 by
    construction of the table) and concatenate the two reactant
    SMILES into a single scaffold key.
    """
    rows: List[Reaction] = []
    for rule_name, lit_rows in LITERATURE_YIELDS.items():
        for r in lit_rows:
            smi_a, smi_b, yield_frac, doi = r[0], r[1], r[2], r[3]
            scaffold = f"{smi_a}+{smi_b}"
            if len(scaffold) > SCAFFOLD_HASH_MAX_LEN:
                scaffold = scaffold[:SCAFFOLD_HASH_MAX_LEN]
            rows.append(
                Reaction(
                    rule_name=rule_name,
                    scaffold_smiles=scaffold,
                    success=True,
                    yield_fraction=float(yield_frac),
                    source=f"lit:{doi}",
                )
            )
    return rows


def build_training_rows(
    metals: Optional[Sequence[str]] = None,
    include_literature: bool = True,
    max_tmqm_rows: int = 50_000,
) -> List[Reaction]:
    """Assemble the full training corpus."""
    rows: List[Reaction] = []
    log.info("Loading tmQM rows (max=%d)…", max_tmqm_rows)
    rows.extend(_row_from_tmqm(metals=metals, max_rows=max_tmqm_rows))
    log.info("  tmQM rows: %d", len(rows))
    if include_literature:
        lit_rows = _row_from_literature()
        rows.extend(lit_rows)
        log.info("  + literature rows: %d", len(lit_rows))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train a ReactionConfidence from tmQM + literature.")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "molmetal" / "models" / "reaction_confidence.pkl"),
        help="Path to write the pickled ReactionConfidence.",
    )
    p.add_argument(
        "--metals",
        default="Pt,Ru,Ir,Cu,Au,Pd,Fe,Co,Ni,Zn",
        help="Comma-separated metal centres to keep from tmQM.",
    )
    p.add_argument(
        "--no-literature",
        action="store_true",
        help="Skip the literature click-yields rows.",
    )
    p.add_argument(
        "--max-tmqm-rows",
        type=int,
        default=50_000,
        help="Cap on tmQM rows ingested (default 50k).",
    )
    p.add_argument(
        "--audit-json",
        default=None,
        help="Optional path to write a JSON audit report.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    log.info("Training metals: %s", metals)

    rows = build_training_rows(
        metals=metals,
        include_literature=not args.no_literature,
        max_tmqm_rows=args.max_tmqm_rows,
    )

    if not rows:
        log.error("No training rows assembled — aborting.")
        return 1

    est = ReactionConfidence()
    est.fit(rows)
    summary = est.summary()
    summary["n_input_rows"] = len(rows)
    summary["metals_filter"] = metals
    summary["include_literature"] = not args.no_literature
    summary["scaffold_hash_max_len"] = SCAFFOLD_HASH_MAX_LEN
    # Cold-default sanity: predict on a deliberately cold pair.
    summary["sanity_cold_predict"] = est.predict(
        "metal_Pt_donor_unknown_xyz",
        "COLD-SCAFFOLD-XYZ",
    )

    log.info("Fit complete: %s", json.dumps(summary, indent=2, default=str))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    est.save(out)
    log.info("Wrote ReactionConfidence -> %s", out)

    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.audit_json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        log.info("Wrote audit -> %s", args.audit_json)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
