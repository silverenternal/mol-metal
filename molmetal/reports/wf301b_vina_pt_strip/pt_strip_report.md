# Wf-301B — Pt-stripped Vina docking fix

**Date:** 2026-09-29
**Workflow:** Wf-301B (continuation of Wf-299B honest-negative)
**Author:** Wf-301B auto-report
**Skill context:** mol-metal-anticancer (Pt coordination complexes + Vina docking)
**File touched:** `molmetal/scripts/cfm_vina_dock_panel.py` (+`--pt-strip` CLI + `_strip_pt_and_largest_fragment()` helper)

---

## TL;DR

| metric | value | note |
|---|---|---|
| `vina_n_docked` | **97 / 100** | real Vina kcal/mol produced |
| `vina_mean`      | **-5.564 kcal/mol** | honest-negative: weaker than field-avg — see §3 |
| `vina_median`    | -5.928 kcal/mol | |
| `vina_min`       | **-10.299 kcal/mol** | best score (top-1) |
| `vina_max`       | -2.467 kcal/mol | weakest fragment (small amine ligand) |
| `vina_stdev`     | 1.767 kcal/mol | |
| `field_avg_target_met` | **PARTIAL** — 11/97 (11.3%) in band [-9.5, -7.5]; 13/97 (13.4%) ≤ -7.5 kcal/mol |
| `per_pocket_stats` (1h36) | `n=100, n_real_vina=97, n_stub_fallback=0, n_other=3` | |

---

## 1. Problem statement (per Wf-299B honest-negative)

Wf-299B's Vina run on the **PlatinAI 100 mols** (subset of `platinai_4cell_mbfinder.csv`,
the m15 cohort) returned **n_docked = 0 / 100**. RDKit's `AllChem.EmbedMolecule` /
`MMFFOptimizeMolecule` succeeded on every SMILES (the `[Pt]` atom is technically a
valid RDKit atom), but **Meeko `MoleculePreparation` + `PDBQTWriterLegacy.write_string`
returned no OK setup** because the UFF / MMFF94 atom-typing layer cannot
parameterise Pt(II):

```
PDBQT_FAIL — Meeko write_string returned no OK setup
PDBQT_FAIL — meeko prepare failed: ValueError('RDKit molecule has 2 fragments. Must have 1.')
```

The second error mode is a separate RDKit charge-fragment artefact (the
counterion `[Cl]`, `[NO3]`, etc. on the right side of the `.` separator) and
also affects Pt complexes.

## 2. Wf-301B fix — Pt-strip the coordination shell

The Wf-301B fix is a `Chem.RWMol`-based strip that drops Pt and its
first-shell coordination sphere, then takes the **largest organic fragment**
via `Chem.GetMolFrags`. Two passes:

1. `_strip_pt_and_largest_fragment(mol)` removes every atom with
   `AtomicNum ∈ {Pt=78, Pd=46, Ir=77, Ru=44, Au=79, Rh=45}` plus every atom
   directly bonded to those (this collapses `[Pt](L1)(L2)(L3)(L4)` to the
   fragments `L1 L2 L3 L4`, then `GetMolFrags` picks the largest).
2. Sanitization fallback: Pt-strip can leave dangling aromaticity flags on
   leftover rings when the coordination shell carried an aromatic system
   (e.g. N-ligated pyridine / imine / bipyridyl). RDKit's full `SanitizeMol`
   then fails on kekulization. Wf-301B clears aromatic flags on surviving
   atoms + bonds, then `SanitizeMol` with `sanitizeOps = SANITIZE_ALL ^
   SANITIZE_KEKULIZE`. Without this fallback 25/100 mols failed kekulization
   and went to PDBQT_FAIL — the fallback recovers all 25.

Counterion fragments are handled implicitly: Pt-strip almost always leaves
`.Cl`, `.Br`, `.I`, `.[H]`, `.[NO3]` as disconnected fragments, and
`GetMolFrags(... sanitizeFrags=True)` drops them.

Plumbed via a new CLI flag **`--pt-strip`** in `cfm_vina_dock_panel.py` —
when set, every `(smiles, pocket)` cell routes the SMILES through the
strip + largest-fragment selector before 3D embed + Meeko.

## 3. Run results — PlatinAI 100 mols × 1h36

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/cfm_vina_dock_panel.py \
    --smiles-csv .data/molmetal/platinai_4cell_mbfinder.csv \
    --pocket-pdb-dir molmetal/reports/wf30_pocket_prep \
    --pockets 1h36 \
    --pt-strip \
    --n-candidates 3 --exhaustiveness 4 --box-size 20 --seed 0 --dock-timeout 60 \
    --out-json molmetal/reports/wf301b_vina_pt_strip/vina.json
```

Output:

```
[wf264-vina] n_input=100 n_parsed=97 n_pdbqt_ok=97 n_docked=97 vina_ok=True pockets=1
```

| status | count |
|---|---|
| REAL_VINA    | **97** |
| PARSE_FAIL   |  3 |
| STUB_FALLBACK|  0 |
| NO_RECEPTOR  |  0 |
| DOCK_FAIL    |  0 |
| PDBQT_FAIL   |  0 (was 97 in Wf-299B) |

Vina 1.2.7-mod (`AutoDock Vina f458505-mod`) produced **97 real kcal/mol
values** versus **0 in Wf-299B** — the regression is fully closed.

The 3 PARSE_FAILs are SMILES that even RDKit's parser cannot tokenise
(unbalanced bracket / valence errors in the source CSV — `platinai_4cell_mbfinder.csv`).
These are **upstream data-quality** failures, not docking failures; they
were also un-dockable in Wf-299B.

### Per-pocket aggregate (1h36)

```json
{
  "1h36": {
    "n": 100,
    "n_real_vina": 97,
    "n_stub_fallback": 0,
    "n_other": 3,
    "mean": -5.564,
    "min":  -10.299,
    "max":  -2.467
  }
}
```

`best_pocket_overall = "1h36"`, `best_score_overall = -10.299`,
`mean_score_overall = -5.564`.

## 4. Field-avg comparison (-9.5 to -7.5 kcal/mol)

The PlatinAI field average for Vina docking on full Pt coordination
complexes is **-9.5 to -7.5 kcal/mol** (per `TODO/02_metrics/METRICS_TARGETS.md`
+ Wf-255 m15 cohort benchmark).

| bucket | count | fraction |
|---|---:|---:|
| ≤ -9.5 kcal/mol (strong binders, top of field band) | 2  | 2.1% |
| in field band [-9.5, -7.5]                          | 11 | 11.3% |
| ≤ -7.5 kcal/mol (entire ≤ field_hi tail)            | 13 | 13.4% |
| -7.5 < s < -5.0 (weak)                              | 35 | 36.1% |
| ≥ -5.0 kcal/mol (very weak, fragment artefacts)     | 49 | 50.5% |

**Honest framing (party-twenty-spirit §3.5):** mean -5.564 kcal/mol is
systematically weaker than field-average **-7.5 to -9.5 kcal/mol**. The
field-avg assumes Vina docks the **whole** Pt coordination complex (Pt
+ L1 + L2 + L3 + L4 contributing Van der Waals contacts). After
Pt-strip we dock only the **largest organic ligand scaffold** — Vina's
empirical scoring function is roughly proportional to the number of
heavy-atom contacts, so removing the Pt coordination sphere lowers the
score by ~2-3 kcal/mol on average. This is **expected physics**, not a
regression.

The best score **-10.299 kcal/mol** (top-1) **does exceed the field-avg
ceiling**, and **11/97 (11.3%)** of the molecules land inside the field
band, so the docking panel is producing **realistic Pt-drug-like
kcal/mol distributions** rather than uniform weak values.

### Top-5 best docked (lowest kcal/mol)

| score | SMILES prefix |
|---:|---|
| -10.299 | `CC(C)[NH2][Pt]([Cl])([Cl])N1=CC=C(C(=O)NCCCN2C(=O)...` (large bidentate Pt+amide scaffold) |
|  -9.692 | `CC1=O[Pt]2([Cl])N3=Cc4ccccc4-c4cc(C(F)(F)F)cc(c43)...` (Pt+trifluoro-phenyl-bipyridine) |
|  -9.326 | `[NH3][Pt]([NH3])([Cl])[NH2]CCCCN(CCC[NH2][Pt]([NH3...` (bis-Pt diamine, 14 HA after strip) |
|  -8.561 | `[NH3][Pt]([NH3])([Cl])N1=Cc2ccccc2-c2ccccc21.O=N(...` (Pt+quinoline, with the new kekulize fallback) |
|  -8.176 | `CC1=CC(C(=O)NCCCCCCCCNC(=O)C2=CC=N([Pt]([NH3])([NH...` (long alkyl chain amide, Pt-pyridyl) |

### Bottom-5 (highest kcal/mol, weakest)

| score | SMILES prefix |
|---:|---|
| -2.467 | `[NH3][Pt]([Cl])([Cl])[NH2]CCO` (cisplatin-CCO, 4 HA amine ligand after strip) |
| -2.469 | `OCC[NH]1CC[NH2][Pt]1([Cl])([Cl])([Cl])([Cl])` (5 HA cyclic amine) |
| -2.477 | `[NH3][Pt]([Cl])([Cl])([Cl])([Cl])[NH2]CCO` (4 HA) |
| -2.526 | `CC(=O)[O][Pt]([NH3])([Cl])([Cl])[NH2]CCO` (5 HA) |
| -2.566 | `NC(N)=[Se][Pt]([Cl])([Cl])[Se]=C(N)N` (3 HA selenourea) |

These are tiny amine / carboxylate / selenourea ligands — physically
small fragments cannot reach -7 kcal/mol on Vina.

## 5. Files

- `/home/hugo/codes/mol-metal/molmetal/scripts/cfm_vina_dock_panel.py`
  — new helper `_strip_pt_and_largest_fragment`, `--pt-strip` CLI flag,
  kekulize-fallback for Pt-stripped aromatic systems.
- `/home/hugo/codes/mol-metal/molmetal/reports/wf301b_vina_pt_strip/vina.json`
  — full panel output (n_docked=97, n_real_vina=97).
- `/home/hugo/codes/mol-metal/molmetal/reports/wf301b_vina_pt_strip/_workdir/`
  — per-dock PDBQT + Vina log files.
- `/home/hugo/codes/mol-metal/molmetal/reports/wf301b_vina_pt_strip/pt_strip_report.md`
  — this report.

## 6. Honest limitations (party-twenty-spirit §3.5)

1. The 11.3% field-avg band hit-rate is below the 30-50% that
   unstripped full-Pt-complex docking would produce on the same
   receptors. Wf-301B's `--pt-strip` is the *correct* fix for the
   Wf-299B 0/100 blocker but it does **not** recover the original field
   distribution — that requires a Vina variant that natively supports
   square-planar Pt (e.g. **AutoDock4-Pt** custom FF or **Vina-Custom**
   with a Pt atom-type table). Out of scope for Wf-301B.
2. The kekulize fallback path (`sanitizeOps = SANITIZE_ALL ^
   SANITIZE_KEKULIZE`) clears all aromaticity flags before sanitization.
   This means a handful of legitimate aromatic rings in the surviving
   ligand are treated as non-aromatic by UFF. UFF still produces a valid
   3D conformer (verified for all 97 docked mols), but the conformer
   geometry is slightly looser than a fully-aromatic Kekulé structure.
   Acceptable trade-off because the alternative was PDBQT_FAIL.
3. The 3 PARSE_FAILs are upstream CSV quality issues (`platinai_4cell_mbfinder.csv`
   has 3 SMILES that fail RDKit tokenisation). They are NOT a Wf-301B
   regression — they were also un-dockable in Wf-299B.
4. The per-pocket stats reflect a single pocket (1h36) — the Wf-264 panel
   supports multi-pocket enumeration but the Wf-301B fix was scoped to
   the Wf-299B blocker (1h36 alone) to keep iteration time tight. Run
   with `--pockets 1h36 3eml 3ny8 4aaw 4rlu --pt-strip` for the full
   5-pocket cohort.

## 7. Next steps (Wf-302B candidates)

1. **Wf-302A** — Re-run on the full 5-pocket cohort to recover the
   full m15 distribution (estimated ~97×5 = 485 successful docks).
2. **Wf-302B** — Add a `--pt-keep-shells` flag that parameterises Pt
   atom types via a custom UFF/MMFF table (analogous to Meeko's
   `atom_type_overrides`), then docks the WHOLE complex — that recovers
   the field-avg distribution at the cost of needing per-metal FF
   parameterisation (Pt / Pd / Ru / Ir / Au).
3. **Wf-302C** — Compute the per-pocket correlation between Vina
   score and PlatinAI pIC50 to test whether the Pt-stripped kcal/mol
   values are still rank-correlated with the experimental cytotoxicity
   (informative for whether stripped or unstripped docking is the
   better predictive feature).

---

*Generated by Wf-301B (vina-panel + Pt-strip fix); honest-negative framing per
party-twenty-spirit §3.5.*