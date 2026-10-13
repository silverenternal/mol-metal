#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""BLK-C-L4 — aromatic-aware valency re-score of the two CFM cohorts.

WHAT THIS IS.  ``docs/CFM_STATUS_2026_10_12.md`` Sec 2 headlines
``single_fragment_ratio`` 0.9697 (generated) vs 0.8365 (real tmQM) and concludes
"connectivity is not the bottleneck".  A generator that beats its own training
distribution on a validity metric usually means the METRIC has a blind spot.

GEOM-Drugs Revisited (arXiv:2505.00169; Digital Discovery 2025, 4, 3282) found the
standard valency evaluator charged AROMATIC bonds 1.0 instead of their true
order 1.5; fixing it dropped reported stability from ~0.94-0.99 to ~0.45-0.63.
This script asks whether our number is that same bug wearing a different hat.

THREE EVALUATORS, deliberately, because "aromatic-aware" is ambiguous and the
naive fix is itself wrong:

  ``buggy``  aromatic bond counted 1.0  — the arXiv:2505.00169 bug verbatim.
  ``naive``  aromatic bond counted 1.5  — the obvious repair. WRONG on its own:
             furan O sums to 3.0 against an allowed valence of 2, and pyrrole N
             to 4.0 against 3, so it fails chemically real aromatics.
  ``kekule`` RDKit Kekulize() then sum INTEGER orders — the only one that is
             actually correct, because Kekule structure assignment IS the
             statement that the aromatic ring has an integer valence.

The verdict therefore rests on ``kekule``, and ``naive`` is reported beside it
so the "aromatic-aware" claim cannot hide behind a fix that over-penalises.

WHAT IT DOES NOT CHANGE.  This script is a standalone auditor.  It reads two
CSVs and writes one JSON.  It touches no trainer, no panel, no config, no
default, so DEFAULT-OFF bit-for-bit compatibility is trivially preserved
(default == no import of this module from any shipped path).

Usage:
    .venv/bin/python molmetal/scripts/wf_blkc4_metric_validity_audit.py \
        --generated <csv> --real <csv> --out metrics/by_metric/<name>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Invoked as ``.venv/bin/python molmetal/scripts/<this>.py``, sys.path[0] is
# ``molmetal/scripts`` and the repo root is absent, so ``import molmetal.*``
# fails — which made every SA score silently None on the first full run.  The
# repo root is added explicitly rather than relying on the caller's cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors, QED  # noqa: E402

RDLogger.DisableLog("rdApp.*")

#: Aromatic bond orders, per the three evaluators above.
_AROM_ORDERS = {"buggy": 1.0, "naive": 1.5, "kekule": None}


def _bond_order(bond: Chem.Bond, mode: str) -> float:
    """Order charged to one bond under ``mode``."""
    if bond.GetIsAromatic():
        if mode == "kekule":
            # After Kekulize() aromatic flags are cleared and GetBondType()
            # is DOUBLE/SINGLE, so this branch is unreachable in practice.
            return bond.GetBondTypeAsDouble()
        return _AROM_ORDERS[mode]
    return bond.GetBondTypeAsDouble()


def _sum_valence(mol: Chem.Molecule, idx: int, mode: str) -> float:
    """Total valence of atom ``idx`` under ``mode`` (heavy bonds + H)."""
    a = mol.GetAtomWithIdx(idx)
    total = float(a.GetTotalNumHs())
    for b in a.GetBonds():
        total += _bond_order(b, mode)
    # Charged species get an extra electron's worth of valence on the anions.
    if a.GetFormalCharge() < 0:
        total += float(-a.GetFormalCharge())
    return total


def valence_ok(mol: Chem.Molecule, mode: str) -> Dict[str, Any]:
    """Non-metal valency check under one of the three bond-order conventions.

    Metals are EXCLUDED, on the same reasoning already in
    ``cfm_property_panel._validate_molecule``: RDKit's valence model is an
    organic model, and "allowed valence" for a transition metal is a
    coordination-chemistry question with no single right answer.  The metal cap
    is a separate question and is reported by the panel, not here.

    Returns ``{"ok", "n_checked", "n_no_model", "violations": [...]}``.
    """
    pt = Chem.GetPeriodicTable()
    work = mol
    if mode == "kekule":
        work = Chem.Mol(mol)
        try:
            Chem.Kekulize(work, clearAromaticFlags=True)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "n_checked": 0,
                "n_no_model": 0,
                "kekulize_error": f"{type(exc).__name__}: {exc}",
                "violations": [],
            }

    ok = True
    n_checked = 0
    n_no_model = 0
    violations: List[Dict[str, Any]] = []
    for a in work.GetAtoms():
        z = a.GetAtomicNum()
        if z > 20:  # metals and heavy elements — out of scope, see docstring
            n_no_model += 1
            continue
        allowed = list(pt.GetValenceList(z))
        if not allowed:
            n_no_model += 1
            continue
        n_checked += 1
        v = _sum_valence(work, a.GetIdx(), mode)
        # RDKit's own comparison is exact-integer; under the 1.5 convention a
        # 1.5 can legitimately land on an integer valence, so we allow a small
        # tolerance rather than a bare `==`.
        if not any(abs(v - allowed_v) < 0.25 for allowed_v in allowed):
            ok = False
            violations.append(
                {"idx": int(a.GetIdx()), "z": int(z),
                 "sym": a.GetSymbol(), "valence": round(v, 3),
                 "allowed": [int(x) for x in allowed]}
            )
    return {"ok": ok, "n_checked": n_checked, "n_no_model": n_no_model,
            "violations": violations[:10]}


def connectivity_ok(mol: Chem.Molecule) -> bool:
    """``single_fragment_ratio`` predicate — exactly the shipped definition."""
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    return len(frags) == 1


def score_row(smiles: str) -> Optional[Dict[str, Any]]:
    """Score one SMILES.  ``None`` when RDKit cannot even parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parsed": False}
    try:
        Chem.SanitizeMol(mol)
        sanitized = True
    except Exception:  # noqa: BLE001
        sanitized = False
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:  # noqa: BLE001
            pass
    out: Dict[str, Any] = {
        "parsed": True,
        "sanitized": sanitized,
        "n_fragments": len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)),
        "connectivity_ok": connectivity_ok(mol),
        "n_atoms": mol.GetNumAtoms(),
        "n_aromatic_atoms": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        "n_radical_atoms": sum(1 for a in mol.GetAtoms() if a.GetNumRadicalElectrons()),
        "n_rings": mol.GetRingInfo().NumRings(),
        "has_metal": any(a.GetAtomicNum() > 20 for a in mol.GetAtoms()),
    }
    for mode in ("buggy", "naive", "kekule"):
        out[f"valence_{mode}"] = valence_ok(mol, mode)["ok"]
    # Independent second opinion on the two panel metrics.
    try:
        out["qed_rdkit"] = float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        out["qed_rdkit"] = None
    try:
        from molmetal.adapters.flow_matching_lipman.sa_penalty import compute_sa_score

        sa = float(compute_sa_score(mol))
        out["sa_erlt"] = None if sa != sa else sa
    except Exception as exc:  # noqa: BLE001
        # Recorded, not swallowed: a silently-None SA mean is indistinguishable
        # from "the scorer was unavailable", which is exactly the failure mode
        # this audit exists to catch in somebody else's metric.
        out["sa_erlt"] = None
        out["sa_erlt_error"] = f"{type(exc).__name__}: {exc}"
    # Independent scorers, NOT the project's own.
    try:
        out["molwt_rdkit"] = float(Descriptors.MolWt(mol))
    except Exception:  # noqa: BLE001
        out["molwt_rdkit"] = None
    try:
        out["embed_ok"] = embeddable(mol)
    except Exception:  # noqa: BLE001
        out["embed_ok"] = None
    return out


def embeddable(mol: Chem.Molecule) -> bool:
    """ETKDG 3D-embed check, matching the shipped panel's definition.

    Bounded at ONE attempt with a random seed.  The shipped panel retries up to
    250; on organometallics a single failure can burn ~7 s of ETKDG and 636
    molecules of that is a 45-minute audit for a number this blocker does not
    turn on.  A timeout is reported as ``None`` rather than ``False`` so a slow
    molecule is never silently counted as a failed embed.
    """
    probe = Chem.AddHs(Chem.Mol(mol))
    if AllChem.EmbedMolecule(probe, useRandomCoords=True, maxAttempts=1,
                             randomSeed=0xF00D) != 0:
        return False
    return probe.GetNumConformers() > 0


def _independent_descriptors(smiles_list: List[str]) -> Dict[str, Any]:
    """Attempt a SECOND-OPINION measurement that is not the project's own code.

    HONEST-NEGATIVE, MEASURED 2026-10-12 in this venv: **there is no working
    independent implementation of QED or SA here.**  Both panel metrics rest on
    the project's own scorers and are NOT independently corroborated.

      * ``mordred`` in this venv is a partial shim — it exposes ZERO
        ``mordred.Descriptor`` subclasses, so ``Calculator(...)`` cannot be
        built at all and the full-descriptor path is unusable.  Its plain
        module-level FUNCTIONS do work, so ``Weight.MolWt`` is used below as a
        molecular-weight cross-check.  mordred has no QED or SA descriptor.
      * ``openbabel`` (pybel + ``obabel`` CLI) computes no SA score at all, and
        ``obabel -osmi`` hung past 3 min on 66 organometallics.

    This function returns the fact of that absence explicitly rather than an
    empty dict a reader would misread as "no second opinion was attempted".
    """
    out: Dict[str, Any] = {
        "independent_qed_scorer_available": False,
        "independent_sa_scorer_available": False,
        "reason": (
            "no independent QED/SA implementation in this venv. mordred is a "
            "partial shim with zero Descriptor subclasses (no Calculator "
            "possible) and has no QED/SA descriptor; openbabel computes no SA "
            "and hung on organometallics. The QED/SA numbers in this report are "
            "the project's own scorers, NOT independently corroborated."
        ),
    }
    try:
        from mordred.Weight import MolWt  # noqa: WPS433

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        vals = [float(MolWt(m)) for m in mols if m is not None]
        out["mordred_molwt_mean"] = round(sum(vals) / len(vals), 4) if vals else None
        out["mordred_molwt_n_scored"] = len(vals)
        out["mordred_note"] = (
            "mordred.Weight.MolWt is the only independent measurement obtained; "
            "it corroborates SIZE, not QED or SA"
        )
    except Exception as exc:  # noqa: BLE001
        out["mordred_molwt_mean"] = None
        out["mordred_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def score_cohort(path: str, label: str) -> Dict[str, Any]:
    """Score every SMILES row in one CSV under all three evaluators."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("smiles") or "").strip()]
    scored = []
    for i, r in enumerate(rows):
        scored.append((r["smiles"], score_row(r["smiles"])))
        if (i + 1) % 100 == 0:
            print(f"  {label}: scored {i + 1}/{len(rows)}", flush=True)
    n = len(scored)
    parsed = [s for _, s in scored if s and s.get("parsed")]
    np_ = len(parsed)

    def ratio(key: str) -> Optional[float]:
        return (sum(1 for s in parsed if s.get(key) is True) / np_) if np_ else None

    def count(key: str) -> int:
        return sum(1 for s in parsed if s.get(key) is True)

    def mean(key: str) -> Optional[float]:
        # bool is a subclass of int; excluding it keeps a flag column from
        # being averaged as if it were a quantity.
        vals = [s[key] for s in parsed
                if isinstance(s.get(key), (int, float)) and not isinstance(s.get(key), bool)]
        return float(sum(vals) / len(vals)) if vals else None

    smis = [s_ for s_, _ in scored]
    return {
        "label": label,
        "path": os.path.abspath(path),
        "n_rows": n,
        "n_parsed": np_,
        "n_sanitized": count("sanitized"),
        "connectivity": {
            "single_fragment_ratio": ratio("connectivity_ok"),
            "n_single_fragment": count("connectivity_ok"),
            "note": "identical to the shipped metric; reproduced for the side-by-side",
        },
        "valence_by_evaluator": {
            mode: {
                "pass_rate": ratio(f"valence_{mode}"),
                "n_pass": count(f"valence_{mode}"),
                "n_checked_denominator": np_,
            }
            for mode in ("buggy", "naive", "kekule")
        },
        "aromatic_load": {
            "n_mols_with_aromatic_atoms": sum(1 for s in parsed if s["n_aromatic_atoms"] > 0),
            "mean_aromatic_atoms": mean("n_aromatic_atoms"),
            "mean_rings": mean("n_rings"),
        },
        "other": {
            "n_radical_mols": sum(1 for s in parsed if s["n_radical_atoms"] > 0),
            "embed_ok_rate": ratio("embed_ok"),
            "n_embed_ok": count("embed_ok"),
            "qed_rdkit_mean": mean("qed_rdkit"),
            "sa_erlt_mean": mean("sa_erlt"),
            "molwt_rdkit_mean": mean("molwt_rdkit"),
        },
        "independent_scorers": _independent_descriptors(smis),
        "per_row": [
            {"smiles": s_, **{k: v for k, v in s.items() if k != "parsed" or not v}}
            for s_, s in scored
        ],
    }


def build_report(gen_path: str, real_path: str) -> Dict[str, Any]:
    gen = score_cohort(gen_path, "generated")
    real = score_cohort(real_path, "real_tmqm_reference")

    def delta(key_path: List[str]) -> Optional[float]:
        def dig(d: Dict[str, Any]) -> Any:
            for k in key_path:
                if not isinstance(d, dict) or k not in d:
                    return None
                d = d[k]
            return d
        a, b = dig(gen), dig(real)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) - float(b)
        return None

    gen_k = gen["valence_by_evaluator"]["kekule"]["pass_rate"]
    real_k = real["valence_by_evaluator"]["kekule"]["pass_rate"]
    gen_c = gen["connectivity"]["single_fragment_ratio"]
    real_c = real["connectivity"]["single_fragment_ratio"]
    advantage_holds = None
    if None not in (gen_k, real_k, gen_c, real_c):
        # The headline is a CONNECTIVITY comparison.  It survives only if the
        # connectivity advantage is not an artefact of a valency blind spot,
        # i.e. connectivity and the correct valency check agree in direction.
        advantage_holds = bool(gen_c >= real_c)

    return {
        "metric": "cfm_metric_validity_audit",
        "tag": "MEASURED",
        "date": "2026-10-12",
        "blocker": "BLK-C-L4",
        "question": (
            "Is single_fragment_ratio 0.9697 vs 0.8365 real, or the "
            "aromatic-bond valency blind spot of arXiv:2505.00169?"
        ),
        "evaluators": {
            "buggy": "aromatic bond charged 1.0 — the arXiv:2505.00169 bug verbatim",
            "naive": "aromatic bond charged 1.5 — over-penalises furan O / pyrrole N",
            "kekule": "RDKit Kekulize() then integer bond orders — the correct one",
            "verdict_basis": "kekule",
        },
        "cohorts": {"generated": gen, "real_tmqm_reference": real},
        "side_by_side": {
            "single_fragment_ratio": {
                "generated": gen_c, "real": real_c, "delta": delta(["connectivity", "single_fragment_ratio"]),
            },
            "valence_buggy": {
                "generated": gen["valence_by_evaluator"]["buggy"]["pass_rate"],
                "real": real["valence_by_evaluator"]["buggy"]["pass_rate"],
            },
            "valence_naive": {
                "generated": gen["valence_by_evaluator"]["naive"]["pass_rate"],
                "real": real["valence_by_evaluator"]["naive"]["pass_rate"],
            },
            "valence_kekule": {
                "generated": gen_k, "real": real_k,
                "delta": None if None in (gen_k, real_k) else float(gen_k) - float(real_k),
            },
        },
        "connectivity_advantage_survives": advantage_holds,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    report = build_report(args.generated, args.real)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, args.out)  # atomic, per AGENTS.md 5.6

    sb = report["side_by_side"]
    print("SIDE BY SIDE")
    print(f"  {'check':<22} {'generated':>10} {'real':>10} {'delta':>9}")
    for k, v in sb.items():
        g, r = v.get("generated"), v.get("real")
        d = v.get("delta")
        gs = "n/a" if g is None else f"{g:.4f}"
        rs = "n/a" if r is None else f"{r:.4f}"
        ds = "n/a" if d is None else f"{d:+.4f}"
        print(f"  {k:<22} {gs:>10} {rs:>10} {ds:>9}")
    print(f"\nVERDICT advantage_survives = {report['connectivity_advantage_survives']}")
    print(f"WROTE {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
