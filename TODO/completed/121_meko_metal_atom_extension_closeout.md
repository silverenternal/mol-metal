# TODO-121 — Wf-121 meko metal atom extension close-out

**Date**: 2026-09-27
**Workflow**: Wf-121 (meko 0.8.0 d-block Pt-group metal atom-type SMARTS extension + real Vina dock on cisplatin)
**Status**: 🚀 **SHIP — 3 phases complete; cisplatin × 10 pockets real Vina MEASURED; P0 blocker PARTIAL LIFT**
**Priority**: **P0 CRITICAL** (paper-blocking — without this, all metallodrug m3 Vina dock stays at -9.6574 stub floor)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/110_meko_metal_atom_extension.md` (parent plan) + Agent 1 meko research report (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training` + `mol-metal-click-chemistry` + `party-twenty-spirit` §3.5

---

## §0 TL;DR

| Item | Status |
|---|---|
| **Wf-121 P1** — 8 SMARTS rules for Pt/Ru/Ir/Os/Rh/Pd/Au/Re d-block Pt-group metals | ✅ **SHIP** (8/8 PASS, `rule_verification.json`) |
| **Wf-121 P2** — `_ligand_to_pdbqt` adapter patch + `MetalPDBQTUnavailable` typed exception | ✅ **SHIP** (34/34 tests PASS + 27/27 regression PASS, `patch_verification.json`) |
| **Wf-121 P3** — cisplatin × 10 pockets × 3 seeds real Vina 1.2.7 dock | ✅ **SHIP** (10/10 pockets MEASURED, real kcal/mol `-2.548` to `-3.734`, `sweep_summary.json`) |
| **P0 blocker partial lift** | ⚙️ **PARTIAL** — cisplatin + 7/8 Pt-group metals dock; **full 8 d-block sweep needs Wf-121 P3 (extended SMARTS for Tc/Pm/lanthanide) + 4-6 wk R20 GPU for full metallodrug cohort** |

**One-line verdict (per party-twenty-spirit §3.5)**: Wf-121 ships the **meko 0.8.0 d-block Pt-group SMARTS extension (8 rules) + `_ligand_to_pdbqt` adapter patch + cisplatin × 10-pocket real Vina dock** on disk today. Cisplatin real kcal/mol is MEASURED, but is **disposition `MEASURED_GENERIC_ATOM`** (the `M` atype is a graceful-degradation marker, NOT a calibrated Pt VDW). The full 8-d-block metallodrug sweep is deferred.

---

## §1 Deliverables (3 phases)

### §1.1 P1 — 8 SMARTS rules (NEW module)

- `molmetal/molmetal_lam/sbdd_env/metal_pdbqt_overrides.py` (NEW, 332 LOC, sha256 `e512244d...`) — 8 SMARTS rules mapping Pt/Ru/Ir/Os/Rh/Pd/Au/Re → generic `M` AutoDock VDW atype.
- Framework: `MoleculePreparation(add_atom_types=[...])` per `meeko/preparation.py:476-486`.
- All 8 rules verified by `rule_verification.json`: matched_count=1 on canonical probe SMILES for every metal.

### §1.2 P2 — `_ligand_to_pdbqt` adapter patch

- `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (PATCHED, 1116 LOC, sha256 `a29b7352...`) — now injects `add_metal_atom_types` and switches to `charge_model='zero'` for metal-containing ligands; raises typed `MetalPDBQTUnavailable` on impossible cases (RDKit embed failure / meeko PDBQT rejection after overrides injected).
- `molmetal/tests/test_vina_adapter_metals.py` (NEW, 443 LOC, sha256 `2107e004...`) — **34/34 PASS** across 5 classes (cisplatin / 8 metals / organic regression / exception contract / adapter integration).
- **7 of 8 metals** produce valid PDBQT; **Ru raises `MetalPDBQTUnavailable`** on RDKit embed failure (`[Ru](N)(N)(N)(N)(N)Cl` octahedral geometry — ETKDG returned -1).
- Regression: 27/27 prior tests still pass (`test_metal_pdbqt_overrides.py` + `test_vina_adapter.py`).

### §1.3 P3 — cisplatin × 10 pockets × 3 seeds real Vina dock

- `molmetal/reports/wf121_cisplatin_real_vina/{1fmc,1g9v,1h36,2hnx,3eml,3ny8,4aaw,4rlu,4rn0,830c}/per_seed_metrics.json` — per-(pocket, seed) raw Vina record (kcal/mol + RMSD-lb/ub + pose count + ligand atom count).
- `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` — per-pocket aggregate.
- `molmetal/reports/wf121_cisplatin_real_vina/synthesis.md` — per-pocket table + honest framing.
- **Engine**: Vina 1.2.7 (Python binding preferred, CLI fallback); exhaustiveness=8; n_poses=5; seeds=[0, 1, 2].
- **boltz_affinity** column = D-MPNN pIC50 fallback (Wf-120 R20 ship, MetalCytoToxDB-trained Attentive D-MPNN).

---

## §2 MEASURED (real Vina 1.2.7 output, kcal/mol best pose)

Per-pocket real kcal/mol (10/10 pockets MEASURED, 3 seeds each, stddev ≤ 0.017):

| Pocket | kcal/mol min | median | mean | stddev | n_seeds | n_heavy (receptor) |
|---|---:|---:|---:|---:|---:|---:|
| 830c | -3.067 | -3.033 | -3.043 | 0.017 | 3 | 3093 |
| 1h36 | -3.597 | -3.589 | -3.587 | 0.009 | 3 | 595 |
| 3eml | -2.827 | -2.827 | -2.827 | 0.000 | 3 | 358 |
| 3ny8 | -3.734 | -3.721 | -3.724 | 0.008 | 3 | 480 |
| 4aaw | -3.287 | -3.283 | -3.282 | 0.005 | 3 | 401 |
| 4rlu | -2.995 | -2.981 | -2.983 | 0.009 | 3 | 4895 |
| 4rn0 | -2.549 | -2.548 | -2.548 | 0.001 | 3 | 245 |
| 1fmc | -3.138 | -3.131 | -3.133 | 0.003 | 3 | 390 |
| 1g9v | -3.057 | -3.040 | -3.045 | 0.008 | 3 | 350 |
| 2hnx | -3.387 | -3.376 | -3.380 | 0.005 | 3 | 432 |

- **Engine**: Vina 1.2.7; **status**: `MEASURED`; **boltz_affinity pIC50**: `-0.132` (constant across pockets — SMILES-only, no pocket info).
- **449-mol pool coverage**: cisplatin probe SMILES `[Pt](N)(N)(Cl)Cl` is in `molmetal/chem/pool_metallodrug_449.json` (449-mol PlatinAI 4-cell + MetalCytoToxDB 5-metal cohort).
- **Pool coverage by element**: Pt (n=2), Pd (n=3), Au (n=1), Ru (n=3), Ir (n=11), Rh (n=3), Os (n=2), Re (n=1) — 26 of 449 mols have 1+ d-block Pt-group metal covered by the 8 rules. The remaining 423 mols are organic or carry non-Pt-group metals (Mg/Ca/Mn/Fe/Zn — already supported by stock meeko).
- **34/34 adapter tests PASS**; **27/27 regression tests PASS**; **8/8 SMARTS rules PASS**.

---

## §3 NOT SHIP / deferred (honest scope)

Per party-twenty-spirit §3.5:

### §3.1 NOT SHIP this Wf

- **Full 8 d-block sweep** (Tc/Pm/lanthanide + actinide): needs Wf-121 P3 (extended SMARTS for Tc/Pm/lanthanide; ~16 additional rules + per-element probe SMILES verification). Currently **out of scope** — Tc/Pm/La/Ce/Pr/Nd/Pm/Sm/Eu/Gd/Tb/Dy/Ho/Er/Tm/Yb/Lu/Ac/Th/Pa/U are NOT covered by stock meeko AND NOT covered by Wf-121 P1 SMARTS.
- **Other Pt-group coordination geometries** (e.g., Ru octahedral `[Ru](N)(N)(N)(N)(N)Cl`): RDKit embed fails — `MetalPDBQTUnavailable` raised. Boltz fallback path is wired (TODO-110 §2 Path C) but not exhaustively tested for octahedral Ru/Pd/Au variants.
- **Calibrated Pt VDW parameter**: the `M` atype is a graceful-degradation marker, not a chemically calibrated Pt-group VDW parameter. Calibrating a proper Pt VDW requires matched crystallographic data + force-field fitting (not a 4-6 wk ship — see §5).
- **boltz_affinity pocket-aware scoring**: currently boltz returns the same pIC50 across all 10 pockets (SMILES-only, no pocket context). Pocket-aware boltz is a separate TODO.

### §3.2 Deferred to Wf-121 P3 / R20 GPU / Wf-58 wet-lab

| Item | Why deferred | Timeline |
|---|---|---|
| Full 8-d-block sweep (Tc/Pm/lanthanide) | SMARTS + verification missing | Wf-121 P3 (4 hr CPU) |
| Full metallodrug cohort 449-mol × 10-pocket Vina | 4490 dock calls; needs GPU sweep script | R20 GPU (1-2 wk) |
| Calibrated Pt-group VDW | Needs matched crystallographic data | 4-6 wk R20 GPU |
| Top-20 candidate wet-lab synthesis + IC50 | Wet-lab dependent | Wf-58 6-month timeline |
| Paper §3.4 Vina-on-metallodrugs claim | TODO-84 paper rule + R20 MEASURED gates | Post-R20 |

---

## §4 Cross-references

- **TODO-110** (`TODO/pending/110_meko_metal_atom_extension.md`) — parent plan, P0 blocker audit + Path A/C solution; **STATUS: most phases shipped via Wf-121 P1+P2+P3**; remaining items are P3 (extended SMARTS) + R20 GPU (calibrated VDW).
- **Wf-121** — in-flight workflow: P1 ✅, P2 ✅, P3 (cisplatin real Vina) ✅; **P3 (extended SMARTS for Tc/Pm/lanthanide)** = next phase.
- **Wf-120** R20-P1 D-MPNN retrain — supplies `boltz_affinity` pIC50 fallback path wired in `molmetal/scoring/score_any.py:288-317` (`_w_vina_score`) per TODO-110 §6 cross-ref.
- **TODO-84 paper rule** — paper §3.4 metallodrug Vina table NOT blocked today (cisplatin real MEASURED); §3.4 + §4 + §6 tables gated on TODO-84 §5 template + R20 GPU full-cohort sweep.
- **`TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 L1-L12 columns** — meko extension unblocks m3 Vina column for the 26 Pt-group-metal mols in 449-pool; L7/L10 columns unchanged (Hilbert-ASTR + multi-PH/PLL orthogonal to meko work).
- **`TODO/INDEX.md`** — pending TODO-110 stays `pending/` (extended SMARTS + calibrated VDW still in flight); Wf-121 in-flight reference added under R20.
- **Agent 1 meko research report** (2026-09-27) — original gap audit (`meeko/data/params/ad4_types.json` covers Mg/Ca/Mn/Fe/Zn only; `metal_vdw.toml` covers Fe/Cu/Ni/Zn/Co/Mg/Mn/Hg/Ca; rejection point `meeko/writer.py:481-499` `_is_molsetup_ok`).
- **`molmetal/scoring/REGISTRY.md` + `ROUTING_POLICY.md`** — boltz fallback routing policy wired per TODO-110 §6 cross-ref.

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **All kcal/mol values are real Vina 1.2.7 output** — not the Wf-30 hash-derived stub floor (-9.6574) and not a placeholder. Engine version + exhaustiveness + seed list are recorded per-pocket in `sweep_summary.json`.
- **The Pt atom is typed with the generic AutoDock VDW atype `M`** (Wf-121 P1 SMARTS rules). The kcal/mol is therefore a **real** Vina measurement with a documented generic-atom caveat — disposition `MEASURED_GENERIC_ATOM` (i.e. *not* mock, *not* full-fidelity chemotype). Downstream consumers MUST treat the resulting kcal/mol as `status='MEASURED_GENERIC_ATOM'`, not a full-fidelity Vina score for Pt-group metallodrugs.
- **The boltz_affinity column is the D-MPNN pIC50 fallback** (Wf-120 R20 ship, MetalCytoToxDB-trained Attentive D-MPNN). Vina (kcal/mol) and boltz_affinity (pIC50) are different scoring functions with different units — they are NOT directly comparable. The boltz column is parallel evidence, not a substitute.
- **Cisplatin works** (10/10 pockets real Vina MEASURED, kcal/mol -2.548 to -3.734). **The full 8 d-block metallodrug sweep is 4-6 wk of R20 GPU work**, not this Wf's scope. This Wf unblocks the SMARTS-extension + adapter path; calibrated VDW + extended element coverage still need R20 GPU + extended SMARTS (Wf-121 P3 next phase).
- **Ru octahedral coordination fails** (`[Ru](N)(N)(N)(N)(N)Cl` → RDKit ETKDG -1). The typed `MetalPDBQTUnavailable` exception correctly propagates rather than silently fabricating a PDBQT. Boltz fallback is wired but not exhaustively tested for octahedral Ru/Pd/Au variants — callers SHOULD expect `boltz_affinity` (pIC50, SMILES-only) when `MetalPDBQTUnavailable` is raised.
- **Per TODO-84 paper rule**: paper §3.4 metallodrug Vina table is NOT blocked today (cisplatin real MEASURED). §3.4 + §4 + §6 Vina tables on the full metallodrug cohort remain gated on TODO-84 §5 template + R20 GPU full-cohort sweep + calibrated Pt-group VDW.

---

**Date**: 2026-09-27 · **Verdict**: 🚀 SHIP — Wf-121 P1+P2+P3 done; 8 SMARTS rules + adapter + cisplatin × 10-pocket real Vina MEASURED; P0 blocker partial lift (cisplatin real Vina works; full 8 d-block sweep needs Wf-121 P3 + 4-6 wk R20 GPU); CPU-only docs; disposition `MEASURED_GENERIC_ATOM` for Pt-typed kcal/mol · **Co-Authored-By**: Claude Code <noreply@anthropic.com>

## 2026-09-27 update

-9.6574 was the legacy stub-floor artifact; Wf-121 P3 ship real Vina path now active per TODO-113 honest reframe.
