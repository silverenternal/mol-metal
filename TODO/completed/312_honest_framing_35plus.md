# Wf-312D — §9 honest framing catalog extended 30 → 36 entries (≥35 floor met)

**Date**: 2026-09-29
**Authoring**: Wf-312D subagent (post-Wf-311A PART 2 + post-Wf-308 synthesis)
**Status**: ✅ SHIPPED — catalog 30 → 36 entries; pre-flight `p3_section9_honest_framing_min30` gate floor updated 30 → 36

---

## 1. Task

Per Wf-306C: §9 honest framing catalog 25 entries. Per Wf-311A PART 2: 25 → 30 entries (added §9.AR + §9.AS + §9.AT + §9.AU + §9.AV). Wf-312D further extended the catalog to **≥35 entries** by adding 6 NEW honest-negatives from Wf-308 + Wf-303C T2 + Wf-252 + Wf-308C Part 2.

## 2. NEW entries added (6 total)

| § | Title | Source | Honest-negative finding |
|---|---|---|---|
| **§9.AW** | Wf-308C Re temporal-split degenerate n_test=1 | `molmetal/reports/wf308c_temporal_split/sweep_summary.json` + `molmetal/reports/wf308c_todo220_launch/launch_plan.md` §3.1 | Re n_test=1 (strict Krasnov 2026 cutoff), random-chance mean_auc=0.5175; structural honest-negative for smallest-cohort temporal-split fold |
| **§9.AX** | Wf-308C NCI-60 PC3 0.2153 BELOW_RANDOM (n_test=55) | `molmetal/reports/wf308c_todo220_launch/launch_plan.md` §3.2 + `metrics/by_metric/nci60_precious_9cell_auc.json` per_cell_line.PC3 | 1-epoch CPU smoke BELOW_RANDOM by Δ=−0.2847; 50-ep GPU protocol-validated PC3=0.87 (3rd-highest per-cell AUC); canonical epoch-floor sensitivity on small NCI-60 cohorts |
| **§9.AY** | Wf-308B ADMET aqueous_solubility_logS -7.612 FAIL | `metrics/by_metric/aqueous_solubility_logS.json` (Wf-303C T1 + Wf-308B R03) | Mean=−7.612 ± 3.936 FAIL vs target band > −4.0 (Δ=−3.612 log S units = ~4000× solubility deficit); HONEST_NEGATIVE-tagged per Wf-308B |
| **§9.AZ** | Wf-303C T2 hepatotox_index 0.548 slightly above 0.5 | `metrics/by_metric/hepatotox_index.json` | Value=0.5477 slightly above 0.5 chance line (Δ=+0.0477); structural-alert proxy (Hughes 2008 + Stepan 2011); consistent with clinical Pt/Ru/Ir hepatotoxicity literature |
| **§9.BA** | Wf-303C T2 logp7_4 6.413 out of typical 0-5 range | `metrics/by_metric/logp7_4.json` | Value=6.4128 above Lipinski Rule-of-5 ceiling of 5 (Δ=+1.413 logP units = ~28% above); structural cost of metal-coordination aromatic ligand systems; consistent with parenteral-only administration route |
| **§9.BB** | Wf-252 m6_kappa 0.472 BELOW 0.6 | `metrics/by_metric/m6_kappa.json` (Wf-252-Apply Fix #2 + Wf-253D m6_kappa_validation) | Shape Kappa Index=0.4719 BELOW 0.6 threshold by Δ=−0.128; MetalCytoToxDB labelled pIC50 validation NEGATIVE_WEAK (Pearson r=0.081 / Spearman ρ=0.111); canonical shape-kappa is NOT a learned metallodrug score |

## 3. Task interpretation notes

### 3.1 Letter designation shift

The original task spec listed 7 entries under letter designations §9.AV through §9.BB. However, §9.AV was already taken by Wf-311A PART 2 (DrugOOD v3 BELOW_RANDOM 0.4574) and §9.AS was already taken by the Vina covalent mechanism architectural ceiling (Wf-143 + Wf-301B). To avoid duplicate / conflicting letter assignments, Wf-312D used the next available letters: §9.AW + §9.AX + §9.AY + §9.AZ + §9.BA + §9.BB.

The Vina covalent mechanism architectural ceiling entry requested by the task spec is already covered at §9.AS (added by Wf-311A PART 2); adding it again as §9.BC would be redundant. The Wf-312D catalog therefore adds 6 new entries (§9.AW-§9.BB) rather than 7, bringing the total from 30 → 36 (still ≥35 floor met).

### 3.2 4 ADMET honest-negatives cluster around 449-mol metallo-drug subset

§9.AY + §9.AZ + §9.BA + §9.BB form a coherent cluster of 4 ADMET honest-negatives, all measured on the same 449-SMILES metallo_drugs_500_train cohort (per Wf-303C T1 + Wf-308B R03 ADMET pass-through integration):

- **§9.AY aqueous_solubility_logS −7.612** (descriptive proxy, FAIL vs > −4.0 target)
- **§9.AZ hepatotox_index 0.548** (structural-alert proxy, slightly above 0.5 chance line)
- **§9.BA logp7_4 6.413** (Crippen logP, above Lipinski Rule-of-5 ceiling)
- **§9.BB m6_kappa 0.472** (Shape Kappa Index, BELOW 0.6 drug-likeness threshold; MetalCytoToxDB validation NEGATIVE_WEAK)

All 4 are **descriptor proxies**, NOT learned metallodrug scores (per the `description` + `qed_comparison.honest_framing` blocks in the JSON contracts). The cluster captures the metallo-drug cohort's structural distance from the typical drug-like organic-only chemotype — the honest-negative is the structural distance itself, NOT a defect.

### 3.3 Vina covalent ceiling duplicate

Per Wf-311A PART 2, §9.AS was added with title "Vina covalent mechanism architectural ceiling — 0/10 pockets reach industry ≥−7 kcal/mol (Wf-143 + Wf-301B honest-negative, 2026-09-29)". The task spec's §9.BB entry ("Wf-143 + Wf-301B Vina covalent ceiling 0/10 ≥-7") is the SAME finding catalogued at §9.AS. Wf-312D does not duplicate this finding; the §9.AS entry is the canonical record.

## 4. Files modified

1. **`TODO/completed/207_unified_pre_flight_checklist_v2.md`**:
   - Added 6 new §9 entries (§9.AW through §9.BB) at end of §9 catalog (between §9.AV and §10 sign-off)
   - Updated §11 cross-references header line to record Wf-312D expansion (30 → 36 entries, ≥35 floor)
   - Added 6 new bullet entries to §11 cross-references list
2. **`molmetal/scripts/pre_flight_check.py`**:
   - Updated gate `p3_section9_honest_framing_min30` `min_entries` floor from 30 → 36 (both occurrences at lines 1509 + 1651)

## 5. Verification

### 5.1 §9 entry count

- **Pre-Wf-312D**: 30 entries (per Wf-311A PART 2)
- **Post-Wf-312D**: **36 entries** (added §9.AW + §9.AX + §9.AY + §9.AZ + §9.BA + §9.BB)
- **Net delta**: +6 entries; **≥35 floor met**

### 5.2 Per-entry honesty

- **§9.AW Re temporal-split degenerate n_test=1**: real MEASURED finding from `molmetal/reports/wf308c_temporal_split/sweep_summary.json`; not a fabrication
- **§9.AX NCI-60 PC3 0.2153 BELOW_RANDOM**: real MEASURED finding from Wf-308C Part 2 smoke (`molmetal/reports/wf308c_todo220_launch/launch_plan.md` §3.2); paired with `metrics/by_metric/nci60_precious_9cell_auc.json` per_cell_line.PC3 50-ep GPU 0.87 for honest-positive context
- **§9.AY aqueous_solubility_logS -7.612**: real MEASURED finding from `metrics/by_metric/aqueous_solubility_logS.json` `value` field; HONEST_NEGATIVE-tagged per Wf-308B R03
- **§9.AZ hepatotox_index 0.548**: real MEASURED finding from `metrics/by_metric/hepatotox_index.json` `value` field
- **§9.BA logp7_4 6.413**: real MEASURED finding from `metrics/by_metric/logp7_4.json` `value` field
- **§9.BB m6_kappa 0.472**: real MEASURED finding from `metrics/by_metric/m6_kappa.json` `value` field + Wf-253D m6_kappa_validation NEGATIVE_WEAK cross-check

All 6 entries cite real MEASURED metrics or report JSON paths; no fabricated findings.

### 5.3 Pre-flight gate update

The `p3_section9_honest_framing_min30` gate (in `molmetal/scripts/pre_flight_check.py` lines 1504-1511 + 1646-1653) now requires `min_entries=36` (updated from 30). The gate will PASS with the current 36-entry catalog; future workflow batches that drop below 36 will FAIL the gate.

## 6. Cross-references

- **`TODO/completed/207_unified_pre_flight_checklist_v2.md`** §9.AW + §9.AX + §9.AY + §9.AZ + §9.BA + §9.BB — new entries
- **`TODO/completed/207_unified_pre_flight_checklist_v2.md`** §11 — updated cross-references header + 6 new bullet entries
- **`molmetal/scripts/pre_flight_check.py`** lines 1504-1511 + 1646-1653 — `min_entries=36` floor
- **`metrics/by_metric/aqueous_solubility_logS.json`** — §9.AY source
- **`metrics/by_metric/hepatotox_index.json`** — §9.AZ source
- **`metrics/by_metric/logp7_4.json`** — §9.BA source
- **`metrics/by_metric/m6_kappa.json`** — §9.BB source (+ `metalcytotox_validation` block for NEGATIVE_WEAK cross-check)
- **`metrics/by_metric/nci60_precious_9cell_auc.json`** per_cell_line.PC3 — §9.AX 50-ep GPU 0.87 honest-positive context
- **`molmetal/reports/wf308c_temporal_split/sweep_summary.json`** — §9.AW source
- **`molmetal/reports/wf308c_todo220_launch/launch_plan.md`** §3.1 + §3.2 — §9.AW + §9.AX sources
- **`molmetal/reports/wf308_synthesis/final_state.md`** §Wf-308B R03 — §9.AY context

---

**Verdict (party-twenty-spirit §3.5 honest ledger)**: Wf-312D shipped 6 NEW §9 entries (§9.AW-§9.BB) extending the catalog from 30 → 36 entries. All 6 entries cite real MEASURED metrics or report JSON paths; no fabricated findings. Pre-flight `p3_section9_honest_framing_min30` gate floor updated 30 → 36 to enforce the new ≥35 floor.