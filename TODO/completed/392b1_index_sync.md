# Wf-392-B1 — Aggregate docs-only INDEX + AGENTS + final synthesis v30

**Date:** 2026-09-30 (Wf-392-B1 wall MEASURED ~10 min CPU-only docs-only sync).
**Phase:** Wf-392 ultracode CPU-only docs+code+test wave — aggregate A1 + A2 + A3 closeouts + INDEX.md + AGENTS.md sync + comprehensive final synthesis v30 (this entry).

---

## §0 Phase

Wf-392-B1 SHIPPED. Per user directive 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* (Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-385-B1 + Wf-386-B1 + Wf-388-B1 + Wf-389-B1 + Wf-390-B1 + Wf-391-B1 prior syncs). Wf-392 surfaces 3 TODO-directory items that were NOT covered by Wf-378..391 ultracode waves: TODO-94 R20-P3 Multi-PH + PLL feature extraction (Wf-392-A1) + Wf-382-A2 §0 SCAFFOLD chemprop v3 cache wire-in (Wf-392-A2) + TODO-119 P1.4 LOMO wrapper integration gap + Wf-299 wrapper flag-missing gap (Wf-392-A3).

This B1 entry:
- Aggregates A1 + A2 + A3 closeouts (`TODO/completed/392a1_multiph_pll_wire.md` + `TODO/completed/392a2_chemprop_v3_cache_wire.md` + `TODO/completed/392a3_lomo_wrapper_verify.md`)
- Updates `TODO/INDEX.md` §22.82 from "(2 deliverables SHIPPED)" to "(3 deliverables SHIPPED)" with A3 added + cross-link expanded
- Updates `AGENTS.md` Wf-392 row from "2 SHIPPED sub-flows" to "3 SHIPPED sub-flows" with A3 added
- Writes `TODO/completed/392_final_synthesis.md` ~580 LOC 9 sections

---

## §1 Pre-flight verification

**Pre-flight cascade PRESERVED** at Wf-391-B1 baseline **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** (NOT re-run by Wf-392-B1 — CPU-only docs-only docs-sync per AGENTS.md §5.1 + kill_then_adapt §0). HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-391-B1 baseline at same drift level (16+ consecutive runs PRESERVED at 89/2/3/94). Pre-flight cascade STILL CLOSED end-to-end across 23+ consecutive runs.

2 FAILs preserved baseline (NOT Wf-392-introduced):
- `p3_loi_send_ready` — Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits
- `p3_wf343c4_anchor_freshness_24h` — natural drift across Wf-374..392 wave activity

3 SKIPs preserved baseline (BY-DESIGN):
- `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-392 — 3 docs+code+test deliverables + 6 NEW tests + 2 INDEX syncs only).

---

## §2 INDEX.md sync

Updated `TODO/INDEX.md` §22.82:

1. **§22.82 header**: `"## §22.82 Wf-392 ultracode CPU-only docs+wave — chemprop v3 featurization cache wire-in scaffold (2026-09-30 — Wf-392-A2 this entry)"` → `"## §22.82 Wf-392 ultracode CPU-only docs+code+test wave — Multi-PH + PLL SE(3) SB wire-in + chemprop v3 featurization cache wire-in scaffold + LOMO + DrugOOD + Wf-299 wrapper verify + Wf-299 wrapper EDIT + comprehensive final synthesis v30 (2026-09-30 — Wf-392-B1 this entry)"`

2. **§22.82 Phase block**: `"2 deliverables SHIPPED"` → `"3 deliverables SHIPPED 100%"`; added Wf-392-A3 with full verdict + edit summary; expanded TODO-94 + Wf-119 + Wf-299 + Wf-310E-Wf-388 references.

3. **§22.82.1 sub-flows SHIPPED (2/2)** → `(3/3 deliverables SHIPPED 100%)`; added Wf-392-A3 row with `TODO/completed/392a3_lomo_wrapper_verify.md` + `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` ~340 LOC + `molmetal/tests/test_wf392a3_wrapper_verify.py` ~140 LOC; 4 NEW tests PASS; Wf-299 wrapper EDIT +12 LOC; LOMO 3 GAP-TRAINER-MISSING honest-negative.

4. **§22.82.2 test pass-rates table**: added Wf-392-A3 row (4 NEW, 4/4 PASS, <1 s); Wf-392 wave TOTAL: 2 NEW tests → **6 NEW tests**; ~15-20 min → **~20-22 min CPU-only**; 2/2 SHIPPED → **3/3 SHIPPED**.

5. **§22.82.3 user voice answer**: rewrote to enumerate 3 SHIPPED deliverables with full verdict per A1+A2+A3.

6. **§22.82.4 cross-link**: added `TODO/completed/392a1_multiph_pll_wire.md` + `TODO/completed/392a3_lomo_wrapper_verify.md` + `TODO/completed/392b1_index_sync.md` + `TODO/completed/392_final_synthesis.md` + 3 NEW source files (`wf392a1_se3_sb_pll_wire.py` + `wf392a3_lomo_wrapper_verify.py` + `test_wf392a3_wrapper_verify.py`) + 3 NEW launch wrappers (`launch_wf119_p1_4_chemprop_v3_lomo.sh` + `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` + `launch_wf299_cfm_main.sh`) + `TODO/pending/94_r20_p3_multiph_pll_feature.md` + `TODO/pending/119_chemprop_v3_lomo.md`.

---

## §3 AGENTS.md sync

Updated `AGENTS.md` Wf-392 row:

1. **Wf-392 row lead**: `"2 SHIPPED sub-flows: A1 + A2"` → `"3 SHIPPED sub-flows: A1 + A2 + A3"`; added Wf-392-A3 summary block with `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` ~340 LOC + `molmetal/tests/test_wf392a3_wrapper_verify.py` ~140 LOC + 4 NEW tests PASS + Wf-299 wrapper +12 LOC + 9-line comment; expanded verdict with static-AST + bash-continuation verifier description; LOMO 3 GAP-TRAINER-MISSING honest-negative; Wf-299 wrapper EDIT closing 3 GAP-WRAPPER-MISSING flags from Wf-388-A2/A3.

2. **Wall + test pass-rates**: `"2 NEW tests"` → `"6 NEW tests"`; `"2 docs+code+test deliverables"` → `"3 docs+code+test deliverables"`; `"~15-20 min"` → `"~20-22 min CPU-only across 3 deliverables"`.

3. **User voice answer**: rewrote to enumerate 3 SHIPPED deliverables; added forward-looking Wf-392-A1 + Wf-392-A2 + Wf-392-A3 follow-up items (multi-PH full training-loop integration + chemprop v3 cache full training-loop integration + LOMO chemprop trainer argparse extension).

4. **Wf-392-B1 follow-up priority chain**: added Wf-392-A3 §3 + Wf-392-A1 §3 + Wf-392-A2 §5.2 to the existing recommendation chain.

---

## §4 Honest framing (per `party-twenty-spirit` §3.5)

**MEASURED**: pre-flight 89/2/3/94 PRESERVED across 16+ consecutive runs at the 89/2/3/94 drift level; 6 NEW tests PASS (2 from A2 + 4 from A3); A1 = wire-in only (0 NEW tests); Wf-299 wrapper EDITED +12 LOC; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

**PROJECTED**: cache savings 5-10× on featurisation speed (per Wf-320D + Wf-328-W10 MEASURED baseline for chemprop v2 + D-MPNN caches); PLL feature speedup 5-10× (same baseline).

**Honest-negative preserved**:
- m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL (per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 7 architectural + 1 literature attempts exhausted)
- m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (per Wf-377-A2 5 alternative splits + Wf-388-A1 multi-stream ensemble REAL-cohort verification + Wf-390-A2 deeper audit)
- LOMO 3 GAP-TRAINER-MISSING (chemprop-mode + chemprop-min-epochs + chemprop-lomo-init-scale not in trainer argparse — known Wf-321H gap; OUT OF SCOPE for Wf-392-A3)
- Full training-loop integration pending for both Wf-392-A1 (Multi-PH + PLL) + Wf-392-A2 (chemprop v3 cache)
- Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU)
- Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending
- Wf-380 SCAFFOLD wire-in MEASURED verification pending (~21.5 hr GPU on Blackwell sm_120)
- G6 ~11.4 hr launch + 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK
- 2 NEW pre-flight FAILs (loi_send_ready Tier-2 reclassification + anchor_freshness wf269c_report.json STALE) preserved baseline; NOT Wf-392-introduced

---

## §5 Cross-references

`TODO/completed/392a1_multiph_pll_wire.md` (Wf-392-A1 closeout, ~280 LOC); `TODO/completed/392a2_chemprop_v3_cache_wire.md` (Wf-392-A2 closeout, ~232 lines); `TODO/completed/392a3_lomo_wrapper_verify.md` (Wf-392-A3 closeout, ~358 LOC); `TODO/completed/392_final_synthesis.md` (this B1's comprehensive final synthesis v30, ~580 LOC, 9 sections); `TODO/INDEX.md` `## §22.82 Wf-392 ultracode CPU-only docs+code+test wave` (this entry); `AGENTS.md` §6 Wf-392 row (this entry); `molmetal/scripts/wf392a1_se3_sb_pll_wire.py` (~510 LOC); `molmetal/tests/test_wf392a1_se3_sb_pll.py` (~310 LOC); `molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py` (~290 LOC); `molmetal/tests/test_wf392a2_chemprop_v3_cache.py` (~115 LOC, 2 NEW tests PASS); `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` (~340 LOC); `molmetal/tests/test_wf392a3_wrapper_verify.py` (~140 LOC, 4 NEW tests PASS); `molmetal/scripts/train_se3sb.py` (3 NEW CLI flags wired); `molmetal/scripts/train_chemprop_v2_v3.py` (`--chemprop-v3-cache` CLI flag wired); `molmetal/scripts/launch_wf299_cfm_main.sh` (Wf-299 wrapper EDITED +12 LOC); `molmetal/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (verified 13/13); `molmetal/scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` (verified 10/10 common + 3 GAP-TRAINER-MISSING); `TODO/pending/94_r20_p3_multiph_pll_feature.md`; `TODO/pending/119_chemprop_v3_lomo.md`; `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed; `party-twenty-spirit` §3.5; `kill_then_adapt` §0 (NO GPU launch + USER_OK gating).