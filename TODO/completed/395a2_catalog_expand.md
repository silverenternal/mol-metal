# Wf-395-A2 — docs/HONEST_NEGATIVE_CATALOG.md expansion 73 → 81 (8 NEW §6.18-§6.25 entries from Wf-388 + Wf-390 + Wf-392 wave)

**Last updated:** 2026-09-30 (Wf-395-A2 close)
**Status:** ✅ SHIPPED (CPU-only docs+test surface; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning)
**Workflow:** Wf-395-A2 (ultracode CPU-only docs-only wave, 2026-09-30)
**Author:** Wf-395-A2 agent (text-only brief per Wf-291C workaround + Wf-302C plain-text template)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.2 honest framing mandate

---

## §0 TL;DR

Wf-395-A2 expands `docs/HONEST_NEGATIVE_CATALOG.md` from 73 → **81 entries** via 8 NEW §6.18-§6.25 entries that aggregate honest-negative findings from Wf-388 + Wf-390 + Wf-392 wave CPU-only docs aggregation. Each NEW entry cites its MEASURED source path on disk + WORKFLOW attribution. PROJECTED vs MEASURED tag discipline preserved per `party-twenty-spirit` §3.5. 0 GPU launches performed; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §5.4 + §10.

**Cumulative expansion**: 9 (Wf-263B legacy) → 81 = **9.0× expansion** vs the legacy 9-entry catalog. CPU-only docs+test surface; 1 NEW test PASS via `molmetal/tests/test_wf395a2_catalog_size.py` (`test_catalog_size`).

---

## §1 NEW §6.18-§6.25 entries (8 entries)

Each NEW entry is sourced from a Wf-388 / Wf-390 / Wf-392 closeout. The catalog entry type field uses the new type tags introduced by Wf-395-A2 (PROJECTED_PENDING_GPU_RETRAIN, SCAFFOLD_TO_WIRED, PROJECTED_USER_OK_GATED, HONEST_NEGATIVE_TRAINER_EXTENSION) to distinguish scaffold vs wired vs pending-GPU states.

### §6.18 — wf388_m2_real_cohort_r2_lift (PROJECTED_PENDING_GPU_RETRAIN, m2_r2)

- **Source**: Wf-388 (m2 real cohort lift via chemprop-v2 + Hilbert joint retrain)
- **Current MEASURED**: platinai_4cell R² **-0.1251 ± 0.1515** + hilbert R² **-0.1268 ± 0.1709** (12-d baseline per Wf-335-M2 + Wf-367-B1)
- **PROJECTED**: platinai_4cell R² **+0.30 to +0.60** if Wf-388 chemprop-v2 + Hilbert joint retrain passes Wf-343C4 gates + per-wave USER_OK per AGENTS.md §5.1

### §6.19 — wf388_m7_diversity_wire_in (SCAFFOLD_TO_WIRED, m7_diversity)

- **Source**: Wf-388 (m7 Tanimoto diversity reward wire-in to D-MPNN multitask trainer)
- **Current MEASURED**: CFM Tanimoto diversity **0.1065** (post Tanimoto reward +22.5% lift = **0.0397** NEGATIVE; per §1.6)
- **Status**: SCAFFOLD SHIPPED (per Wf-388 wire-in); full D-MPNN WIRING + multi-seed × 5-fold CV R² lift DEFERRED pending USER_OK

### §6.20 — wf388_m4_sa_penalty_wire_in (SCAFFOLD_TO_WIRED, m4_sa)

- **Source**: Wf-388 (m4 SA penalty wire-in to chemprop_v2_v3 + D-MPNN multitask trainers)
- **Current MEASURED**: SA global 3.289 ± 0.851 (n=26,801 Wf-264 panel; per §2.8); no SA penalty in trainer loss yet
- **Status**: SCAFFOLD SHIPPED (per Wf-388 wire-in); chemprop_v2_v3 trainer integration + D-MPNN multitask integration DEFERRED to Wf-388-GPU retrain phase

### §6.21 — wf390_todo376_5metal_subset_path_a (PROJECTED_USER_OK_GATED, m2_r2)

- **Source**: Wf-390 + TODO-376 (5-metal subset Path A re-scope to avoid SCAFFOLD_NOT_FUNCTIONAL trainer constraint per Wf-379-A2)
- **Current MEASURED**: Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL on `train_dmpnn_multitask_wasserstein.py` `--metals` restricted to 5 metals (line 247 `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`); 11-metal wrapper wants (Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au) — VIOLATION
- **PROJECTED**: 5-metal subset (Ru/Ir/Rh/Os/Re) R² CEILING_BREAK sustained at Wf-335-M2 levels; LIFT to ≥ +0.30 on metalcytotox_combined subset PROJECTED if Wf-390-A USER_OK granted

### §6.22 — wf390_todo179_scaffold_not_functional (HONEST_NEGATIVE_TRAINER_EXTENSION, m1_auc)

- **Source**: Wf-390 + TODO-179 (W₂ pIC50 alignment + m1_auc 5-metal × 5-seed extension)
- **Current MEASURED**: TODO-179 W₂ pIC50 alignment SCAFFOLD_NOT_FUNCTIONAL per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` §2 (STATUS_HOLD); m1_auc Krasnov 0.86 gate target NOT YET HIT (Ru 0.7613 1-seed)
- **Verdict**: closure requires trainer extension OR alternative encoder bootstrap

### §6.23 — wf390_todo383_se3_sb_wire_in (SCAFFOLD_TO_WIRED, m15_rmsd)

- **Source**: Wf-390 + TODO-383 (SE(3) Schrödinger Bridge wire-in to docking panel trainer)
- **Current MEASURED**: SE(3) SB 10000-step baseline per Wf-383-A1 §6.16 (REVISED PROJECTED [1500, 2500] steps post-EDM+OT-CFM+log-normal+EMA+lr-warmup)
- **Status**: SCAFFOLD SHIPPED; full DOCKING panel integration MEASURED-correctness DEFERRED

### §6.24 — wf392_todo94_multiph_pll_wire_in (SCAFFOLD_TO_WIRED, m22_se3sb)

- **Source**: Wf-392 + TODO-94 (Multi-PH 32-bin Fourier + PLL 128-d signature joint feature extraction wire-in to SE(3) SB trainer)
- **Current MEASURED**: m22 SE(3) SB 10000-step GPU retrain DEFERRED per §1.10 + Wf-339-B2; Multi-PH + PLL scaffold SHIPPED
- **Status**: SCAFFOLD SHIPPED; full SE(3) SB trainer integration + R² ≥ 0.70 honest-neg gate MEASUREMENT DEFERRED

### §6.25 — wf392_todo155_chemprop_v3_cache_wire_in (SCAFFOLD_TO_WIRED, chemprop_v3)

- **Source**: Wf-392 + TODO-155 (chemprop_v3 featurization cache wire-in to chemprop_v2_v3 trainer + D-MPNN multitask trainer)
- **Current MEASURED**: chemprop_v3 featurization CPU scaffold SHIPPED; Wf-326-W10 16.83× featurization speedup MEASURED on D-MPNN path
- **Status**: SCAFFOLD SHIPPED; full chemprop_v2_v3 trainer integration MEASURED-correctness DEFERRED

---

## §2 Catalog count update (73 → 81)

The catalog was updated as follows:

- **Header (§0 + status line)**: total entries **73 → 81** (8 NEW §6.18-§6.25 entries)
- **§6 header**: 17 → 25 entries (Wf-369-A3 15 + Wf-383-A1 §6.16 + Wf-386 §6.17 + Wf-395-A2 §6.18-§6.25 = 25 cumulative)
- **§8 Closing**: total count `72 → 81` (reconciled against actual on-disk entry count); Wf-395-A2 expansion ratio 73 → 81 = +8 (+11.0%); cumulative 9 → 81 = 9.0× expansion vs legacy 9-entry Wf-263B catalog
- **END trailer**: cumulative ratio 9 → 71 → **9 → 81 = 9.0×**

Per honest framing per `party-twenty-spirit` §3.5: the actual catalog count IS 81 (verified by `grep -c "^### §[1-7]\." docs/HONEST_NEGATIVE_CATALOG.md` = 81). The task brief PROJECTED `65 → 73` (8 NEW entries from a 65 baseline); the actual baseline was 73 (after Wf-375-A2 + Wf-383-A1 + Wf-386-A3+B1 cumulative expansions), so the actual ratio is `73 → 81 = +8 entries` with the same +11.0% expansion rate.

---

## §3 Test + verification

**1 NEW test PASS** at `molmetal/tests/test_wf395a2_catalog_size.py`:

- `test_catalog_size`: catalog has ≥ 80 entries + 7 main sections (§0..§7) + §6 ≥ 25 entries + §7 = 6 entries (WETLAB_DEPRECATED preserved per AGENTS.md §5.4); all 8 NEW entry IDs from `EXPECTED_NEW_ENTRIES` table appear in §6; each NEW entry is attributed to its source workflow (Wf-388 / Wf-390 / Wf-392); every NEW entry cites a MEASURED source path on disk (`TODO/completed/*.md` or `molmetal/reports/*.json`); standing directives §5.1 + §5.2 + §5.3 + §5.4 references preserved; MEASURED/PROJECTED tag discipline preserved

CPU-only contract: pure file-system read + regex parse; no GPU launch; no external network; wall-time ~1.5 s.

---

## §4 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED-vs-PROJECTED tag discipline preserved**: every NEW entry uses the appropriate tag (MEASURED for current state, PROJECTED for forward-looking claims). Example: §6.18 explicitly states `MEASURED (current)` vs `PROJECTED (post-GPU retrain)` with the gate condition (Wf-343C4 + per-wave USER_OK per AGENTS.md §5.1).
- **Source-citation discipline preserved**: every NEW entry cites its MEASURED source path on disk (e.g. `TODO/completed/388_m2_real_cohort_r2_lift.md` + `molmetal/reports/wf335m2_w2_chemprop/result.json` + `molmetal/reports/wf367b1_m2_r2/result.json` for §6.18).
- **Workflow attribution preserved**: every NEW entry is attributed to its source workflow (Wf-388 / Wf-390 / Wf-392) + the upstream Wf closeout file path.
- **No aspirational / fabricated numbers**: all numeric values cited are MEASURED (e.g. platinai_4cell R² -0.1251 ± 0.1515 is from `molmetal/reports/wf367b1_m2_r2/result.json`); PROJECTED ranges (e.g. R² +0.30 to +0.60) are explicitly labeled as PROJECTED with gate conditions.
- **CPU-only contract preserved**: 0 GPU launches; 0 `paper/*` touches per AGENTS.md §5.3 TODO-84; 0 wet-lab planning per AGENTS.md §5.4 + §10.

---

## §5 Standing directives re-confirmed (per AGENTS.md §5)

- **§5.1 Standing GPU authorization** — preserved; Wf-395-A2 performed 0 GPU launches per the CPU-only docs+test scope.
- **§5.2 Honest framing mandate** — preserved; MEASURED/PROJECTED tag discipline maintained for all 8 NEW entries.
- **§5.3 No paper work without authorization** — preserved; 0 `paper/*` touches.
- **§5.4 Tier-2 journal target + NO wet-lab planning** — preserved; 0 wet-lab planning; WETLAB_DEPRECATED §7 preserved with exactly 6 entries.
- **§10 Wetlab deprecation FAQ** — preserved; WETLAB_DEPRECATED entries preserved unchanged.

---

## §6 Cross-references

- `docs/HONEST_NEGATIVE_CATALOG.md` (modified; 73 → 81 entries)
- `molmetal/tests/test_wf395a2_catalog_size.py` (NEW; 1 test PASS in ~1.5 s)
- `TODO/completed/{388_*, 390_*, 392_*}_*.md` (NEW upstream closeouts per Wf-388 + Wf-390 + Wf-392 wave)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (legacy 9-entry catalog)
- `TODO/completed/{369a3, 375a2, 383a1, 386a1}_*.md` (prior expansion waves — baseline 73 entries)
- AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10 (standing directives)
- `party-twenty-spirit` skill §3.5 (honest framing MEASURED-vs-PROJECTED tag discipline)

---

**END OF Wf-395-A2 CLOSEOUT** — Owner Wf-395-A2 agent; cross-ref `party-twenty-spirit` §3.5; catalog expansion 73 → 81 entries (cumulative 9 → 81 = 9.0× vs legacy Wf-263B catalog); 8 NEW §6.18-§6.25 entries from Wf-388 + Wf-390 + Wf-392 wave CPU-only docs aggregation; 1 NEW test PASS; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.
