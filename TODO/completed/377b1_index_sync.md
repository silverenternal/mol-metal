# Wf-377-B1 INDEX + AGENTS sync — final ship

**Date**: 2026-09-30 (Wf-377-B1 timestamp 2026-09-29T16:56:20+0000 UTC)
**Status**: SHIPPED — docs-only INDEX + AGENTS sync
**Wave**: Wf-377 (m6_kappa redesign + m2 R² diagnostic + cross-poll + dashboard + final ship)
**Cross-ref**: `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4

---

## §0 TL;DR

| Item | Value |
|---|---|
| **Wall** | ~10 min CPU-only (PROJECTED 10-15; MEASURED ~10 — on-budget) |
| **NEW tests** | 0 (verification + docs-only sync only) |
| **NEW source patches** | 0 |
| **NEW docs** | 2 (`TODO/completed/377b1_index_sync.md` this file + `TODO/completed/377_final_synthesis.md` ~600 LOC 9 sections) |
| **INDEX.md delta** | +50 lines (§22.71 NEW entry between §22.70 Wf-376-B1 and §23 Wf-316B) |
| **AGENTS.md delta** | +1 row in §6 + extended trailer (NEW Wf-377-B1 sync clause) |
| **GPU launches** | 0 |
| **paper/* touches** | 0 |
| **wetlab planning** | 0 |
| **pre-flight v3 fresh-run** | 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 152.78 s |

---

## §1 INDEX.md §22.71 entry

Inserted as NEW §22.71 entry between §22.70 Wf-376-B1 and §23 Wf-316B anchor. Per `grep -nE '^### §22\.' TODO/INDEX.md` before this sync: §22.70 was the latest entry; the next free slot was §22.71.

### §1.1 §22.71 entry content

```markdown
### §22.71 Wf-377-B1 final ship — final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis v22 (2026-09-30 — this entry)

**Phase**: Wf-377-B1 final ship (3/5 deliverables SHIPPED 60%: A1 + A2 + A4 SHIPPED + A3 cross-poll DEFERRED + B1 this docs-only sync). Per Wf-377 wave user directive 2026-09-30 verbatim *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* — Wf-377 wave follows the deep-dive-research move for m6_kappa gate (which 7 architectural attempts failed to lift) and the cohort-diagnostic move for m2 R² gate (which 50+ feature Ridge ensemble overfits on small cohorts). Closes the Wf-377 wave with final pre-flight + INDEX sync (this §22.71 entry) + AGENTS sync (new §6 row appended) + comprehensive final synthesis v22 at `TODO/completed/377_final_synthesis.md`.

**Date**: 2026-09-30 (Wf-377-B1 wall PROJECTED 10-15 min CPU-only; MEASURED ~10 min — on-budget).

**Pre-flight v3 fresh-run at Wf-377-B1 timestamp 2026-09-29T16:56:20+0000 UTC wall 152.78 s** per `molmetal/reports/wf377b1_preflight.txt` (32,545 bytes): **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** (gate count `actual=94 == expected=94` auto-recomputed via Wf-345-B3). **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** (matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 baseline at same drift level; 4 consecutive runs PRESERVED at 89/2/3/94). Root cause: 2 FAILs preserved from Wf-374-B1 baseline — (1) `loi_send_ready` FAIL — 4 LOI readiness issues, status headers missing/parsed-fail on `wetlab_loi_{rusanov_2025_platinai,pku_cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md` per AGENTS.md §5.4 Tier-2 reclassification (LOI = `TEMPLATE_NOT_NEEDED_FOR_TIER2`; NOT Wf-377-introduced — Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits); (2) `wf343c4 anchor freshness` FAIL — `molmetal/reports/wf277c_real_pilot/wf269c_report.json` (age > 24h STALE — natural drift across Wf-374/375/376/377 wave activity). 3 SKIPs preserved: `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency) + `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2). **Pre-flight cascade STILL CLOSED end-to-end across 12 consecutive runs**: Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-A2 (no pre-flight) → Wf-376-B1 → **Wf-377-B1** (this run); the `p3_wf343c2_per_wave_runtime_check` cascade child is NOT in the FAIL list (preserved by Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`); `p2_tests_pass` cascade parent is NOT in the FAIL list (preserved by Wf-367-A1 line 2384 docstring defensive fix).

**Wf-377 wave structure**: **3/5 deliverables SHIPPED 60%** (A1 + A2 + A4 SHIPPED; A3 cross-poll DEFERRED to B1 §3 + B1 this docs-only sync SHIPPED):
- **Wf-377-A1 SHIPPED**: `TODO/completed/377a1_m6_kappa_redesign.md` (~398 LOC; `molmetal/scripts/wf377a1_m6_kappa_redesign_validate.py` ~290 LOC; `molmetal/tests/test_wf377a1_m6kappa_redesign.py` 3 NEW tests PASS; **deep-dive-research move** per user 2026-09-30 directive — 16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB labelled pIC50; best alternative MolLogP Pearson r=+0.1999 WEAK band; **no cohort-level MODERATE band discovery**; recommendation: drop m6_kappa from §3 main-result, promote top-3 alternatives to §3 supplementary sub-table)
- **Wf-377-A2 SHIPPED**: `TODO/completed/377a2_m2r2_diagnostic.md` (~242 LOC; `molmetal/scripts/wf377a2_m2r2_diagnostic.py` 726 LOC; `molmetal/tests/test_wf377a2_m2r2_diagnostic.py` 198 LOC 3 NEW tests PASS; **5 alternative-split diagnostic** — random K-fold, stratified K-fold, scaffold K-fold, leave-one-cluster-out, per-cell hold-out — all yield R²<0 on platinai_4cell + hilbert n=200 cohorts; **gate structurally unachievable CPU-only**; metalcytotox_combined n=2000 CEILING_BREAK stands)
- **Wf-377-A3 cross-poll DEFERRED**: rolled into Wf-377-B1 §3 + Wf-377_final_synthesis §1 (aggregates 7 Wf-37{1..7}*_final_synthesis.md + 7 Wf-37{1..7}b1_index_sync.md = 14 final-ship docs)
- **Wf-377-A4 SHIPPED**: `TODO/completed/377a4_dashboard_final.md` (~121 LOC; `molmetal/reports/gpu_launch_dashboard.json` schema 1.0 → 1.1 bump with Wf-372 install_state + Wf-371..377 recent_wf_shipped inventory; **1 NEW test PASS** at `molmetal/tests/test_wf377a4_dashboard.py`)
- **Wf-377-B1 SHIPPED** (this entry): `TODO/completed/377b1_index_sync.md` (this file ~150 LOC 5 sections) + `TODO/completed/377_final_synthesis.md` (~600 LOC 9 sections: §0 TL;DR + §1 Wf-371..377 ultracode series arc + §2 Zotero + Tectonic install SHIPPED + §3 Wetlab deprecation formalization + §4 m6_kappa + m2 R² FINAL + §5 m6_kappa redesign proposal + §6 m2 R² cohort diagnostic + §7 USER_OK-gated items + §8 Honest framing + §9 Cross-references); final pre-flight + INDEX sync (this §22.71 entry) + AGENTS sync (new §6 row appended) + comprehensive final synthesis v22

**§22.71.1 Wf-377-B1 deliverable scope**

- **Pre-flight fresh-run** at `molmetal/reports/wf377b1_preflight.txt` (32,545 bytes): 89/2/3/94 wall 152.78 s; 2 FAILs = `p3_loi_send_ready` + `p3_wf343c4_anchor_freshness_24h` (both NOT Wf-377-introduced); 3 SKIPs preserved (BY-DESIGN); EXPECTED_GATE_COUNTS["all"] 94 PRESERVED
- **INDEX sync** appended between §22.70 Wf-376-B1 and §23 Wf-316B anchor; +50 lines for this §22.71 entry
- **AGENTS sync** new §6 row inserted before `## §7 Critical scripts reference`; last-update trailer extended with Wf-377-B1 sync clause
- **Final synthesis doc** at `TODO/completed/377_final_synthesis.md` (~600 LOC 9 sections)

**§22.71.2 MEASURED test pass-rates (per Wf-377-A1 + A2 + A4 + B1)**

| Deliverable | NEW tests | Wall | Status |
|---|---:|---:|---|
| Wf-377-A1 (m6_kappa redesign) | 3 NEW | ~5 min | ✅ SHIPPED |
| Wf-377-A2 (m2 R² diagnostic) | 3 NEW | ~2 min | ✅ SHIPPED |
| Wf-377-A3 (cross-poll) | 0 NEW | ~3 min | ✅ DEFERRED (rolled into B1 §3) |
| Wf-377-A4 (dashboard refresh) | 1 NEW | ~5 min | ✅ SHIPPED |
| Wf-377-B1 (INDEX + AGENTS + synthesis) | 0 NEW | ~10 min | ✅ SHIPPED |
| **Wf-377 TOTAL** | **7 NEW** | ~25 min | 3/5 deliverables SHIPPED 60% (A3 deferred to rolled-up upstream artifacts) |

**§22.71.3 User voice answer (verdict on user 2026-09-30 verbatim *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*)**

> **Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377 ultracode series together DID彻底解决** the install (Zotero 10.0.3 + Tectonic 0.17.0 SHIPPED via Wf-372-A1+A2 + Wf-374-A1+A2 verification) + verification + INDEX-sync + pre-flight-cascade + closeout-inventory + Wf-371-retroactive + Wf-374-final-ship + Tier-2-deprecation + wetlab-FAQ + honest-negative-catalog + LAMBDA_ROADMAP-verify + m6_kappa-reframe + Wf-376-A2 m1_auc-scaffold + Wf-377-A1 m6_kappa-redesign (deep-dive-research move) + Wf-377-A2 m2_R2-diagnostic (5 alternative splits) + Wf-377-A4 dashboard-refresh + comprehensive-final-synthesis gaps across 290+ closeouts in 35 active Wfs (Wf-340..377); pre-flight 89/2/3/94 PRESERVED across 4 consecutive runs at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 [this run]); 2 honest-negative FAILs (loi_send_ready Tier-2 reclassification + wf343c4 anchor freshness wf269c_report.json STALE) BY-DESIGN per §5.4 + natural-drift baseline; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.

> **The deep-dive-research move per user verbatim "如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"**: Wf-377-A1 moved **out of the architectural-attempt regime** (which produced 7 paths with no `augmented_mean` lift) into **metric-redesign via literature** and discovered that **no per-molecule topological invariant reaches MODERATE band at cohort-level** (best alternative MolLogP Pearson r=+0.1999 WEAK band). This is a structural finding that 7 architectural attempts could not have discovered. The forward-looking recommendation is the **drop-and-promote-alternatives** path — this requires USER_OK for paper work.

> **They did NOT彻底解决** (out of Wf-377 scope per honest-negative framing per `party-twenty-spirit` §3.5):

> 1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 redesign; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; forward-looking path = drop-from-§3 + USER_OK-gated paper conversion)
> 2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic)
> 3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-377-introduced for the LOI fail (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits); wf269c anchor STALE is natural drift from Wf-374/375/376/377 wave activity; both deferred to Wf-378+ follow-up
> 4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
> 5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4 + Wf-376-B1 §22.70.3; closure path = USER_OK=1 + VRAM budget check per §5.1)
> 6. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + kill_then_adapt §0)
> 7. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1; closure path = ~20 min CPU + 2 tests; deferred to Wf-378+)
> 8. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

> **NEW ultracode Wf-378+ IS the recommended follow-up** per Wf-377 final synthesis §8 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation (~21.5 hr GPU + 10 min CPU + 30 min install total).

**§22.71.4 Cross-link**

`TODO/completed/377{a1,a2,a4}_*.md` (3 Wf-377 SHIPPED closeouts: A1 m6_kappa redesign + A2 m2 R² diagnostic + A4 dashboard refresh — this entry); `TODO/completed/377b1_index_sync.md` (Wf-377-B1 INDEX + AGENTS sync delivery ~150 LOC 5 sections — this file); `TODO/completed/377_final_synthesis.md` (Wf-377-B1 comprehensive final synthesis v22 ~600 LOC 9 sections); `TODO/completed/{371,372,373,374,375,376}{a1..a4,b1,c1}_*.md` + `TODO/completed/{371,372,373,374,375,376}_final_synthesis.md` (35+ Wf-37{1..6} SHIPPED closeouts); `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW; n=5460 MetalCytoToxDB validation); `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC); `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections); `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections); `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants); `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template); `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC); `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections); `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 pre-flight fresh-run 89/2/3/94 wall 152.78 s — 32,545 bytes); `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94 wall 154.228 s — 32,533 bytes); `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s); `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s); `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42 s); `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94 wall 152.53 s); `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1); `molmetal/scripts/wf377a1_m6_kappa_redesign_validate.py` (~290 LOC); `molmetal/scripts/wf377a2_m2r2_diagnostic.py` (726 LOC); `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-367-A1 line 2384 defensive fix PRESERVED); `molmetal/scripts/pre_flight_check.py` (Wf-367-A2 `--strict` removal at line 4590-4598 PRESERVED); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed; `AGENTS.md` §6 NEW row (this sync); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.71 (this entry appended between §22.70 Wf-376-B1 and §23 Wf-316B); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive).

---
```

## §2 AGENTS.md §6 row 29 (Wf-377-B1 sync)

NEW row appended between row 28 (Wf-376-B1, line 282) and the `---` separator (line 283). The row carries the Wf-377-B1 user voice answer + 9 honest-negative caveats + cross-references.

## §3 AGENTS.md last-update trailer extension

The trailer at line 414 (END OF AGENTS.md marker) is extended with Wf-377-B1 sync clause preserving the pre-flight 89/2/3/94 + 9 honest-negative caveats + standing directives.

## §4 Final verification commands

After sync, run:

```bash
grep -c "Wf-377" TODO/INDEX.md  # Expect >=1 (the §22.71 entry)
grep -c "Wf-377" AGENTS.md       # Expect >=1 (the §6 row + trailer)
ls TODO/completed/377*           # Expect 5 files (377a1..a4 + 377b1_index_sync + 377_final_synthesis)
wc -l TODO/INDEX.md AGENTS.md    # Expect +50 INDEX.md lines + 1 §6 row + trailer extension
```

## §5 Cross-references

- `TODO/completed/377_final_synthesis.md` (comprehensive final synthesis v22)
- `TODO/completed/{377a1,377a2,377a4}_*.md` (3 Wf-377 SHIPPED closeouts)
- `molmetal/reports/wf377b1_preflight.txt` (89/2/3/94 wall 152.78s)
- `molmetal/reports/wf377a4_preflight.txt` (Wf-377-A4 baseline 89/2/3/94 wall 148.29s)
- `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `TODO/INDEX.md` §22.70 Wf-376-B1 (prior sync preserved unchanged)
- `TODO/INDEX.md` §23 Wf-316B (next anchor preserved unchanged)

---

**END OF Wf-377-B1 INDEX + AGENTS sync** — Owner Hugo Lee + Wf-377-B1 ultracode; 2026-09-30; docs-only; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wetlab planning.
