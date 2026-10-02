# Wf-299B — Chemistry-aware metric on PlatinAI mols

## TL;DR

SA + QED panel **succeeded** (97/100 parsed, real values).
Vina docking panel **failed honestly**: 0/100 docked because RDKit
UFF cannot parameterise Pt+2 in MMFF/UFF — PlatinAI mols are Pt(II)
coordination complexes, not standard organics.

## Input

* CSV: `.data/molmetal/platinai_4cell_mbfinder.csv` (n=100)
* Column: `smiles` (lowercase; auto-detected)
* Cohor: PlatinAI 4-cell mbfinder

## SA + QED (RDKit, n_parsed=97/100)

| metric | n | mean | std | p50 |
| --- | --- | --- | --- | --- |
| SA  | 97 | 5.0539 | 0.9069 | 4.8469 |
| QED | 97 | 0.3598 | 0.1602 | 0.3745 |

* SA mean 5.05 (lower = more synth-friendly; GuacaMol typical drug-like
  2-5). PlatinAI Pt complexes are moderately synth-heavy.
* QED mean 0.36 — below the 0.5 drug-like threshold. Pt coordination
  complexes are generally drug-like in *cytotoxicity* space, not
  Lipinski-Ro5 QED space.

## Vina docking (pocket 1h36, vina_binary_ok=True, n_docked=0/100)

| status | n |
| --- | --- |
| REAL_VINA        | 0 |
| STUB_FALLBACK    | 0 |
| NO_RECEPTOR      | 0 |
| DOCK_FAIL        | 100 (all RDKit embed → UFF Pt+2 fail) |

* Vina binary probed OK (`AutoDock Vina f458505-mod`).
* Receptor `receptor.pdbqt` staged under `wf30_pocket_prep/1h36/`.
* Failure mode: `RDKit UFFTYPER: Unrecognized atom type: Pt+2 (1)` —
  every PlatinAI mol contains an explicit Pt atom. RDKit MMFF/UFF
  has no Pt parameter → 3D conformer generation fails →
  PDBQT never written → 0 docks.

This is an **honest-negative result**, not a stub floor.

## Follow-up (P1.3 → P1.4 candidates)

1. **Pt-stripped Vina**: drop the `[Pt](...)(...)` coordination shell,
   dock the *ligand only* (scaffold after Pt removal) — would
   rescue ~100 mols and let us compare CFM-generated scaffolds to
   PlatinAI reference ligands.
2. **Morgan / 2D-only metric panel**: for chem-aware eval that does
   not need 3D, use SA + QED + MolWt + LogP + Lipinski-violations
   directly on the SMILES (skip Vina for the coord complexes).
3. **OpenBabel / Meeko direct path**: skip RDKit MMFF, feed the
   molecule to Meeko via SMILES → 3D with `obabel` (handles metals
   via MMFF94s+metal patches).

## Output files

* `molmetal/reports/wf299b_sa_qed/sa_qed.json`
* `molmetal/reports/wf299b_vina/vina.json`
* `molmetal/reports/wf299b_chem_metric/chem_metric_report.md` (this file)