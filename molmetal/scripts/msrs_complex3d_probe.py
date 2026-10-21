#!/usr/bin/env python3
"""UP-3 step 1 — is a 3D coordination graph even constructible from this corpus?

The MSRS table stores ``ligand_canonical`` as a **disconnected SMILES**: the
organic ligand(s), the ancillary anions (``[Cl-]``, ``[c-]1ccccc1``, ``[C-]#[O+]``)
and the counter-ions, with **the metal atom omitted entirely** (0.000 explicit
metal tokens in a 200-complex sample).  So before spending a day of GPU on a 3D
graph we have to answer a data question, not a modelling question:

    can a metal centre and its donor bonds be recovered from these strings at
    all, and does distance-geometry embedding survive the reconstruction?

Three constructions are measured, because they answer different questions:

``organic``   ETKDG on the fragments as written (no metal).  Cheap, and almost
              always succeeds — but the fragments are disconnected, so their
              relative geometry is unconstrained.  Success here is *not*
              evidence that the complex has a geometry.
``complete``  metal + DATIVE bonds to every plausible donor, including aromatic
              N and aromatic carbanions (Cp / aryl anions).  This is the
              chemically correct donor set for this corpus.
``legal``     metal + DATIVE bonds to the subset RDKit can represent without
              breaking kekulisation: neutral non-aromatic N/P/S/O and
              charge-neutralised halides.

The gate is scored on ``complete``.  A construction that silently drops the Cp
and pyridyl donors does not describe the species whose pIC50 is in the table;
it describes a chloride complex with a floating organic fragment.

Gate (frozen before the run, 2026-10-21):

    PASS  if organic_embed_success  >= 0.95
         and complete_embed_success >= 0.80
         and donor_count_ge_2       >= 0.90
    FAIL  otherwise -> UP-3 falls back to "2D graph + explicit metal node +
         coordination bonds", which needs no geometry at all.

Writes ``metrics/by_metric/msrs_complex3d_probe.json``.  Trains nothing.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
DEDUP = REPO / "data" / "msrs" / "measurements_dedup.parquet"
OUT = REPO / "metrics" / "by_metric" / "msrs_complex3d_probe.json"

SEED = 20261021
N_PER_METAL = 40
METALS = ("Ru", "Ir", "Rh", "Os", "Re")
CN = {m: 6 for m in METALS}

HALIDES = (9, 17, 35, 53)
HETERO = (7, 15, 16, 8)
METAL_TOKEN = {
    "Ru", "Ir", "Rh", "Os", "Re", "Pt", "Pd", "Cu", "Zn", "Au", "Ag",
    "Fe", "Co", "Ni", "Mn", "Ti", "V", "Cr", "Mo", "W",
}

GATE = {
    "organic_embed_success": 0.95,
    "complete_embed_success": 0.80,
    "donor_count_ge_2": 0.90,
    "frozen_at": "2026-10-21T15:10+08:00",
}


def donor_classes(mol: Chem.Mol) -> dict[str, list[int]]:
    """Split candidate donors into the ones RDKit can and cannot represent."""
    out = {"neutral_hetero": [], "anionic_hetero": [], "halide": [],
           "aromatic_n": [], "aromatic_carbanion": []}
    for atom in mol.GetAtoms():
        z, q = atom.GetAtomicNum(), atom.GetFormalCharge()
        if z == 6:
            if q == -1 and atom.GetIsAromatic():
                out["aromatic_carbanion"].append(atom.GetIdx())
            continue
        if z in HALIDES:
            if q == -1:
                out["halide"].append(atom.GetIdx())
            continue
        if z not in HETERO:
            continue
        if atom.GetIsAromatic() and z == 7 and q == 0:
            out["aromatic_n"].append(atom.GetIdx())
        elif q == 0:
            out["neutral_hetero"].append(atom.GetIdx())
        elif q < 0:
            out["anionic_hetero"].append(atom.GetIdx())
    return out


def embed(mol: Chem.Mol, tries: int = 3) -> bool:
    """ETKDGv3 with a deterministic seed; random coords on retry."""
    molh = Chem.AddHs(mol)
    for attempt in range(tries):
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED + attempt
        params.useRandomCoords = attempt > 0
        params.maxIterations = 400
        if AllChem.EmbedMolecule(molh, params) == 0:
            return True
    return False


def attach(mol: Chem.Mol, metal: str, idxs: list[int]) -> Chem.Mol | None:
    """Metal + dative bonds to ``idxs``.  Anionic donors are charge-neutralised.

    Returns None when RDKit refuses the graph (kekulisation or valence).
    """
    if not idxs:
        return None
    rw = Chem.RWMol(mol)
    midx = rw.AddAtom(Chem.Atom(metal))
    for i in idxs:
        atom = rw.GetAtomWithIdx(i)
        if atom.GetFormalCharge() < 0:
            atom.SetFormalCharge(0)
        rw.AddBond(midx, i, Chem.BondType.DATIVE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def main() -> int:
    frame = pd.read_parquet(DEDUP, columns=["ligand_canonical", "metal"])
    frame = frame.drop_duplicates(subset=["ligand_canonical", "metal"])

    rng = np.random.default_rng(SEED)
    sampled = []
    for metal in METALS:
        pool = frame.loc[frame["metal"] == metal, "ligand_canonical"].to_numpy()
        if len(pool) == 0:
            continue
        take = min(N_PER_METAL, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sampled.extend((metal, s) for s in pool[idx])

    rows = []
    for metal, smiles in sampled:
        rec = {
            "metal": metal, "parse": False, "n_fragments": 0, "explicit_metal": False,
            "n_heavy": 0, "n_donors": 0, "n_ancillary_fragments": 0,
            "n_aromatic_n": 0, "n_aromatic_carbanion": 0, "n_halide": 0,
            "organic_embed": False,
            "legal_built": False, "legal_embed": False,
            "complete_built": False, "complete_embed": False,
            "complete_fail_reason": "",
        }
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append(rec)
            continue
        rec["parse"] = True
        rec["n_heavy"] = mol.GetNumHeavyAtoms()
        frags = Chem.GetMolFrags(mol)
        rec["n_fragments"] = len(frags)
        rec["explicit_metal"] = any(a.GetSymbol() in METAL_TOKEN for a in mol.GetAtoms())

        cls = donor_classes(mol)
        rec["n_aromatic_n"] = len(cls["aromatic_n"])
        rec["n_aromatic_carbanion"] = len(cls["aromatic_carbanion"])
        rec["n_halide"] = len(cls["halide"])
        rec["n_donors"] = sum(len(v) for v in cls.values())
        rec["n_ancillary_fragments"] = sum(
            1 for f in frags
            if len(f) == 1 or any(mol.GetAtomWithIdx(i).GetFormalCharge() != 0 for i in f)
        )
        rec["organic_embed"] = embed(mol)

        # legal subset: what RDKit can represent without breaking aromaticity
        legal_idx = (cls["neutral_hetero"] + cls["halide"] + cls["anionic_hetero"])[: CN[metal]]
        legal = attach(mol, metal, legal_idx)
        rec["legal_built"] = legal is not None
        rec["legal_embed"] = bool(legal is not None and embed(legal))

        # complete donor set: legal + aromatic N + aromatic carbanion
        complete_idx = (cls["neutral_hetero"] + cls["halide"] + cls["anionic_hetero"]
                        + cls["aromatic_n"] + cls["aromatic_carbanion"])[: CN[metal]]
        complete = attach(mol, metal, complete_idx)
        rec["complete_built"] = complete is not None
        if complete is None:
            rec["complete_fail_reason"] = "graph_rejected"
        else:
            rec["complete_embed"] = embed(complete)
            if not rec["complete_embed"]:
                rec["complete_fail_reason"] = "embed_failed"
        rows.append(rec)

    df = pd.DataFrame(rows)
    n = len(df)

    def mean(col: str) -> float:
        return float(df[col].mean()) if n else 0.0

    summary = {
        "n_sampled": int(n),
        "parse_success": mean("parse"),
        "organic_embed_success": mean("organic_embed"),
        "legal_built": mean("legal_built"),
        "legal_embed_success": mean("legal_embed"),
        "complete_built": mean("complete_built"),
        "complete_embed_success": mean("complete_embed"),
        "complete_fail_graph_rejected": float((df["complete_fail_reason"] == "graph_rejected").mean()) if n else 0.0,
        "complete_fail_embed": float((df["complete_fail_reason"] == "embed_failed").mean()) if n else 0.0,
        "donor_count_ge_2": float((df["n_donors"] >= 2).mean()) if n else 0.0,
        "donor_count_ge_cn": float(np.mean([r.n_donors >= CN.get(r.metal, 6) for r in df.itertuples()])) if n else 0.0,
        "needs_aromatic_n": float((df["n_aromatic_n"] > 0).mean()) if n else 0.0,
        "needs_aromatic_carbanion": float((df["n_aromatic_carbanion"] > 0).mean()) if n else 0.0,
        "explicit_metal_share": mean("explicit_metal"),
        "multi_fragment_share": float((df["n_fragments"] > 1).mean()) if n else 0.0,
        "ancillary_fragment_mean": float(df["n_ancillary_fragments"].mean()) if n else 0.0,
        "n_fragments_median": float(df["n_fragments"].median()) if n else 0.0,
        "heavy_atoms_median": float(df["n_heavy"].median()) if n else 0.0,
        "n_donors_median": float(df["n_donors"].median()) if n else 0.0,
    }

    per_metal = {}
    for metal in METALS:
        sub = df[df["metal"] == metal]
        if len(sub) == 0:
            continue
        per_metal[metal] = {
            "n": int(len(sub)),
            "organic_embed_success": float(sub["organic_embed"].mean()),
            "legal_embed_success": float(sub["legal_embed"].mean()),
            "complete_embed_success": float(sub["complete_embed"].mean()),
            "n_donors_median": float(sub["n_donors"].median()),
            "n_fragments_median": float(sub["n_fragments"].median()),
            "needs_aromatic_carbanion": float((sub["n_aromatic_carbanion"] > 0).mean()),
        }

    passes = (
        summary["organic_embed_success"] >= GATE["organic_embed_success"]
        and summary["complete_embed_success"] >= GATE["complete_embed_success"]
        and summary["donor_count_ge_2"] >= GATE["donor_count_ge_2"]
    )
    if passes:
        verdict = "GEOMETRY_CONSTRUCTIBLE"
    elif summary["organic_embed_success"] >= GATE["organic_embed_success"]:
        verdict = "ORGANIC_ONLY_FALLBACK"
    else:
        verdict = "NO_GEOMETRY"

    doc = {
        "artifact": "msrs_complex3d_probe",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 representation / UP-3 step 1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(DEDUP),
            "script": str(Path(__file__).relative_to(REPO)),
            "seed": SEED,
            "n_per_metal": N_PER_METAL,
            "rdkit": Chem.rdBase.rdkitVersion,
            "python": platform.python_version(),
        },
        "design": {
            "sample": "metal-stratified unique (ligand_canonical, metal) pairs",
            "embed": "ETKDGv3, 3 attempts (deterministic seed, random coords on retry)",
            "metal_construction": "bare metal atom + DATIVE bonds to the top-CN donors",
            "complete_donor_set": "neutral hetero N/P/S/O + halides + anionic hetero "
                                  "+ aromatic N + aromatic carbanion",
            "legal_donor_set": "neutral hetero N/P/S/O + halides + anionic hetero "
                               "(the subset RDKit sanitises without breaking kekulisation)",
            "coordination_number": CN,
            "gate": GATE,
            "gate_uses": "complete_embed_success",
        },
        "summary": summary,
        "per_metal": per_metal,
        "verdict": verdict,
        "verdict_rule": "PASS iff organic_embed_success >= 0.95 AND complete_embed_success >= 0.80 "
                        "AND donor_count_ge_2 >= 0.90",
        "caveats": [
            "The corpus stores the complex as DISCONNECTED fragments with the metal omitted; "
            "explicit_metal_share measures exactly that.",
            "organic_embed_success is a weak signal: disconnected fragments are placed "
            "independently, so their relative geometry is unconstrained.",
            "The gate is scored on the COMPLETE donor set. A construction that drops the Cp / "
            "pyridyl donors embeds fine but no longer describes the species in the table; "
            "legal_embed_success is reported only to show where the failure comes from.",
            "The binding mode is not in the data. donor_classes() is a heuristic; a Cp ring is "
            "counted as one donor and a chelating N,N ligand is not distinguished from two "
            "monodentate donors.",
            "A successful embed is not a correct geometry: no force field was minimised and no "
            "reference structure was compared against.",
        ],
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[UP-3 probe] n={n} organic={summary['organic_embed_success']:.3f} "
        f"legal={summary['legal_embed_success']:.3f} "
        f"complete={summary['complete_embed_success']:.3f} "
        f"(graph_rejected={summary['complete_fail_graph_rejected']:.3f} "
        f"embed_failed={summary['complete_fail_embed']:.3f}) "
        f"needs_Cp={summary['needs_aromatic_carbanion']:.3f} -> {verdict}"
    )
    print(f"json: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
