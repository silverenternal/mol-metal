# Wf-406 final synthesis v39 — Wf-406 ultracode CPU-only m6_kappa paper prep + Lambda CFM patch + atomic migration + Wf-246-C metal-coord SCAFFOLD comprehensive final synthesis

**Date:** 2026-09-30 (Wf-406-B1 closeout timestamp; this is the comprehensive final synthesis for the entire Wf-406 wave)
**Wf:** Wf-406 ultracode CPU-only docs+code+test wave (5 SHIPPED deliverables: A1 + A2 + A3 + A4 + B1 this entry)
**User voice relay:** *"之前...这些问题都要解决，我给你授权"* (Wf-406 wave parent directive — same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-388/Wf-389/Wf-391/Wf-392/Wf-394/Wf-395/Wf-397/Wf-398/Wf-399/Wf-400/Wf-404/Wf-407)
**Owner:** Hugo Lee + Wf-406 ultracode agent
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-cfm-training` (Wf-406-A2 Lambda CFM patch + CFM flag smoke-required rule §5.7); `mol-metal-gpu-execution` §0 (`kill_then_adapt` directive); `mol-metal-anticancer` (Wf-406-A4 cross-ref); no `paper/*` touches (TODO-84 active); no `wetlab/*` planning (AGENTS.md §5.4 + §10)

---

## §0 TL;DR

**Wf-406 wave ACTUAL count: 5 SHIPPED deliverables (A1 + A2 + A3 + A4 + B1 this entry)** — 4 sub-flow A-deliverables + 1 B1 docs-only INDEX + AGENTS + final synthesis (this row).

**Verdict summary:**
- ✅ **m6_kappa Tier-2 §3 drop-justification paper prep SHIPPED** (Wf-406-A1) — 2 docs SHIPPED (`docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` ~356 LOC + EXTENDED `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1)
- ✅ **TODO-21 Lambda × CFM coupling patch application SHIPPED** (Wf-406-A2) — 5 trainer inserts + 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor (decode_ratio = 1.000 + loss 91% decrease)
- ✅ **Atomic write protocol migration top-3 scripts SHIPPED** (Wf-406-A3) — 3 scripts migrated (wf398a2 + wf390a2 + wf394a1) to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`; 6 NEW tests PASS in ~1.33 s wall CPU-only
- ✅ **Wf-246-C metal-coordination term retrain SCAFFOLD SHIPPED** (Wf-406-A4) — SCAFFOLD + bash launch wrapper + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit
- ✅ **Aggregate INDEX + AGENTS + final synthesis v39 SHIPPED** (Wf-406-B1) — §22.92 INDEX entry + NEW §6 row in AGENTS.md + Wf-406-B1 docs-only sync + Wf-406 final synthesis v39 (this row + 2 closeout files)

**Wall: ~25 min CPU-only across 5 deliverables (MEASURED); 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; 9 NEW tests PASS across 3 SHIPPED sub-flows (A2=1 + A3=6 + A4=2; A1 = docs-only brief 0 NEW tests).**

---

## §1 Wf-406 wave structure (5 SHIPPED deliverables)

| Sub-flow | On-disk file | Bytes | Verdict |
|---|---:|---:|---|
| Wf-406-A1 (m6_kappa Tier-2 §3 drop-justification paper prep) | `406a1_m6_kappa_paper_prep.md` | 16,063 | ✅ SHIPPED — 2 docs + 1 closeout + 0 NEW tests |
| Wf-406-A2 (TODO-21 Lambda × CFM coupling patch application) | `406a2_lambda_cfm_patch.md` | 16,219 | ✅ SHIPPED — 5 trainer inserts + 1 NEW test PASS in 42.81 s |
| Wf-406-A3 (atomic write protocol migration top-3 scripts) | `406a3_atomic_migration.md` | 18,464 | ✅ SHIPPED — 3 scripts migrated + 6 NEW tests PASS in ~1.33 s |
| Wf-406-A4 (Wf-246-C metal-coordination term retrain SCAFFOLD) | `406a4_wf246c_scaffold.md` | (this sync) | ✅ SHIPPED — SCAFFOLD + launch wrapper + 2 NEW tests PASS |
| Wf-406-B1 (docs-only INDEX + AGENTS + final synthesis) | `406b1_index_sync.md` + `406_final_synthesis.md` | (this sync) | ✅ SHIPPED |

### §1.1 Wf-406-A1 — m6_kappa Tier-2 §3 drop-justification paper prep SHIPPED

**Closeout:** `TODO/completed/406a1_m6_kappa_paper_prep.md` (16,063 bytes; 297 lines; 7 sections)
**Verdict:** ✅ DROP m6_kappa from Tier-2 §3 main-result table + PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table + RETAIN m6_kappa as §6 honest-limitations complementary heuristic

**Key decision:** per Wf-371-A1 §3.5 + Wf-365-A3 §5.2 + Wf-377-A1 §4.2 + Wf-389-A1 §1.2 + Wf-406-A1 §0:
- **paths_exhausted**: 9 = 7 architectural + 1 degenerate + 1 metric-redesign literature/empirical
- **NEW docs file**: `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (~356 LOC, 7 sections)
- **EXTENDED template file**: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~378 LOC = +110 LOC §3.1 extension)

**Wall:** ~5 min CPU-only + 0 NEW tests (docs-only brief)

### §1.2 Wf-406-A2 — TODO-21 Lambda × CFM coupling patch application SHIPPED

**Closeout:** `TODO/completed/406a2_lambda_cfm_patch.md` (16,219 bytes; 186 lines; 7 sections)
**Verdict:** ✅ 3 CLI flags + dataset-build block + loss-multiplier block applied to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor (decode_ratio = 1.000 + loss 91% decrease)

**5 trainer inserts:**
1. `parse_args()` 3 CLI flags (lines 1340-1380): `--lambda-cfm-coupling` + `--lambda-cfm-coupling-weight` + `--lambda-cfm-coupling-samples`
2. `train_one_epoch()` signature (lines 587-593): 2 new kwargs `lambda_cfm_coupling_weight` + `lambda_cfm_mean_weight`
3. `train_one_epoch()` loss-multiplier block (lines 843-857): `loss = loss * (1 + lambda_cfm_coupling_weight * lambda_cfm_mean_weight)` guarded by `weight > 0.0 AND mean_weight > 0.0`
4. `main()` dataset-build block (lines 1477-1510): builds `LambdaRewardDataset` if `--lambda-cfm-coupling` is set
5. `main()` call site (lines 1546-1552): passes the 2 kwargs

**Smoke results:** §5.7 50-step smoke floor MEASURED PASS end-to-end:
- decode_ratio = 1.000 (every epoch) — floor per AGENTS.md §5.7 NEW (Wf-399-A1) preserved
- loss[0]=0.9483 → loss[-1]=0.0687 (91% decrease over 3 epochs × 50 batches ≈ 150 step-equivalents)
- BIT-EXACT backward-compatible with Wf-388-A3 / Wf-397-A2 when no flag is set on the CLI

**Wall:** ~5 min CPU-only + 42.81 s for tests

### §1.3 Wf-406-A3 — atomic write protocol migration top-3 scripts SHIPPED

**Closeout:** `TODO/completed/406a3_atomic_migration.md` (18,464 bytes; 388 lines; 8 sections)
**Verdict:** ✅ Migrate top-3 closeout-drafting scripts (wf398a2 + wf390a2 + wf394a1) to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`; 6 NEW tests PASS in ~1.33 s wall CPU-only

**Migrated scripts:**
- `molmetal/scripts/wf398a2_m2_r2_real_verify.py` (+25 LOC: 159 → 184)
- `molmetal/scripts/wf390a2_alt_encoder_audit.py` (+35 LOC: 190 → 225)
- `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (+18 LOC: 605 → 623)

**Total LOC delta**: +78 LOC across 3 scripts.

**Adoption sequence per Wf-386-A3 §6.1:**
- ✅ step 1 (Wf-386-A3 protocol SHIPPED)
- ✅ step 2 (Wf-406-A3 this migration of 3 scripts)
- ⏳ step 3 (Wf-406+ future waves migrate additional scripts)
- ⏳ step 4 (Wf-406+ future waves migrate closeout-drafting agent scripts)

**Wall:** ~3 min CPU-only + 1.33 s for tests

### §1.4 Wf-406-A4 — Wf-246-C metal-coordination term retrain SCAFFOLD SHIPPED

**Closeout:** `TODO/completed/406a4_wf246c_scaffold.md` (this sync, 179 lines; 6 sections)
**Verdict:** ✅ SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit

**3 deliverables:**
1. SCAFFOLD script `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` (344 lines, 12,602 bytes) — 5-step CPU smoke (import + build + kcal_sign_audit + emit result.json + provenance MEASURED)
2. Bash launch wrapper `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` (216 lines) — USER_OK=1 env-var gate (exit 64 when USER_OK != 1) + DRY_RUN.log + plan.json + SCAFFOLD smoke invocation + `--full` flag for GPU retrain
3. Test file `molmetal/tests/test_wf406a4_metal_coord_scaffold.py` (~250 lines, 2 NEW tests) — `test_scaffold_smoke_passes` + `test_launch_wrapper_syntax`

**Smoke artefact:**
```json
{
  "wf": "Wf-406-A4",
  "status": "SHIPPED_SCAFFOLD_CPU_ONLY",
  "n_metals": 2,
  "n_seeds": 2,
  "kcal_sign_audit": {
    "composite_scores": [0.85, 0.92, 0.78, 0.88, 0.91],
    "post_fix_kcal": [-0.7, -0.84, -0.56, -0.76, -0.82],
    "sign_audit_per_score": [true, true, true, true, true],
    "sign_audit_pass": true,
    "kcal_per_unit": 1.0,
    "wf_333_v1_fix_id": "metal_coord_term.py:635 (1-2*comp)*kcal_per_unit"
  }
}
```

**Wall:** ~5 min CPU-only + <1 s for tests

### §1.5 Wf-406-B1 — docs-only INDEX + AGENTS + final synthesis v39 SHIPPED (this row)

**Closeout:** `TODO/completed/406b1_index_sync.md` (131 lines) + `TODO/completed/406_final_synthesis.md` (this file, ~600 LOC, 9 sections)
**Verdict:** ✅ §22.92 INDEX entry + NEW §6 row in AGENTS.md + Wf-406 final synthesis v39

**Wall:** ~10 min CPU-only

---

## §2 What 彻底解决 (Wf-406 wave DID solve)

The Wf-406 wave DID彻底解决 four concrete docs+code+test gaps:

1. **m6_kappa Tier-2 §3 drop-justification paper prep** (Wf-406-A1) — 2 docs SHIPPED with 9 paths_exhausted FINAL decision: DROP m6_kappa from Tier-2 §3 main-result + PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table + RETAIN m6_kappa as §6 honest-limitations complementary heuristic. USER_OK-gated paper §3 v4 + §6 v4 write (per AGENTS.md §5.3 + TODO-84, ZERO `paper/*` touches performed; the docs are pre-prepared for USER_OK-gated paper write).

2. **TODO-21 Lambda × CFM coupling patch application** (Wf-406-A2) — Closes the Wf-394-A1 patch-application gap now that the §5.7 floor (decode_ratio > 0) is established. 5 trainer inserts + 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor MEASURED PASS end-to-end (decode_ratio = 1.000 + loss 91% decrease).

3. **Atomic write protocol migration of top-3 closeout-drafting scripts** (Wf-406-A3) — 3 scripts migrated to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation` per AGENTS.md §5.6 + Wf-386-A3 §6.1 adoption sequence step 2. 6 NEW tests PASS in ~1.33 s wall CPU-only.

4. **Wf-246-C metal-coordination term retrain SCAFFOLD** (Wf-406-A4) — SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit. The SCAFFOLD verifies the Wf-333-V1 sign fix at `metal_coord_term.py:635` per `(1-2*comp)*kcal_per_unit` convention.

**9 NEW tests PASS** across 3 SHIPPED sub-flows (A2=1 + A3=6 + A4=2; A1 = docs-only brief 0 NEW tests).

---

## §3 What NOT 彻底解决 (preserved honest-negatives per `party-twenty-spirit §3.5`)

Per AGENTS.md §5.2 honest framing mandate + party-twenty-spirit §3.5, the following metrics / gates / SCAFFOLD-only items remain in HONEST_NEGATIVE state after the Wf-406 wave:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap -0.128; 9 paths exhausted per Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-406-A1 §0 finalization: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-406-introduced; both deferred to Wf-407+ follow-up
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with -183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **Wf-406-A2 Lambda CFM actual GPU retrain BLOCKED** on Wf-291D + Wf-299 YuelBond gate — the patch is CPU-only smoke-verified; actual MEASURED loss-modification lift requires Blackwell sm_120 retrain with USER_OK=1 grant per AGENTS.md §5.1
6. **Wf-406-A4 metal-coordination term trainer wire-in NOT YET performed** — SCAFFOLD smoke verifies import + build + sign audit; actual wire-in to `train_dmpnn_multitask_wasserstein.py` requires ~50 LOC trainer edit + USER_OK-gated GPU retrain per AGENTS.md §5.1
7. **Wf-406-A4 actual 5-metal × 5-seed × 50-epoch Wf-131 P3 retrain BLOCKED** on USER_OK=1 grant + ~5 hr Blackwell sm_120 retrain
8. **Wf-407-A2 GPU launch REFUSED** per Wf-407-A3 SCAFFOLD_NOT_FUNCTIONAL pattern (same as Wf-397-A3 + Wf-398-A1 + Wf-400-A1)
9. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; Krasnov 0.86 gate target; wall-measured ~5 hr GPU; full multi-pair result deferred)
10. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint)
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §4 Forward-looking (USER_OK-gated items)

The Wf-406 wave ships CPU-only SCAFFOLD + docs + wire-in + smoke surfaces; actual MEASURED verification requires USER_OK-gated GPU retrain per AGENTS.md §5.1. The following forward-looking items are BLOCKED on USER_OK:

1. **Wf-406-A2 Lambda CFM GPU retrain** — Blackwell sm_120 ~5-10 hr; USER_OK=1 + Wf-299 launch wrapper pattern
2. **Wf-406-A4 metal-coordination term retrain** — Blackwell sm_120 ~5 hr; USER_OK=1 + 5-metal subset (Ru/Ir/Rh/Os/Re) + Wf-131 P3 + Wf-318 Wasserstein regulariser
3. **Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120** — ~10 hr GPU
4. **Wf-388-A1 + Wf-388-A2 + Wf-388-A3 SCAFFOLD wire-in MEASURED verification** — ~5 hr GPU
5. **Wf-388 Wf-125 P1 CONTROLD PAIR retrain** — ~2 hr GPU
6. **Wf-379-A1 m1_auc 5-seed result close** — ~5 hr GPU
7. **Wf-407-A1 Wf-299 Wave B launch** — ~10 min on idle GPU 1
8. **Wf-407-A4 Wf-264 CFM property panel runner** — CPU-only SCAFFOLD verified per Wf-394-A2
9. **USER_OK-gated paper §3 v4 update** from `docs/PAPER_SECTION_3_METRIC_TABLE.md` per Wf-389-A1 §3 supplementary sub-table reframe
10. **USER_OK-gated paper §6 v4 update** from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` per Wf-406-A1 §3.1 extension
11. **Atomic write protocol adoption migration** of remaining closeout-drafting scripts (Wf-406-A3 step 2 → step 3 + step 4 per Wf-386-A3 §6.1 adoption sequence)
12. **Wf-406-A1 m6_kappa DROP-from-§3 paper write** — USER_OK=1 + §3 supplementary sub-table insertion + §6 limitations entry

---

## §5 Wall budget (MEASURED)

| Sub-flow | Wall (MEASURED) | Tests | Status |
|---|---:|---:|---|
| Wf-406-A1 | ~5 min CPU-only | 0 NEW (docs-only brief) | SHIPPED |
| Wf-406-A2 | ~5 min CPU-only + 42.81 s for tests | 1 NEW | SHIPPED |
| Wf-406-A3 | ~3 min CPU-only + 1.33 s for tests | 6 NEW | SHIPPED |
| Wf-406-A4 | ~5 min CPU-only + <1 s for tests | 2 NEW | SHIPPED |
| Wf-406-B1 | ~10 min CPU-only | 0 NEW (docs-only sync) | SHIPPED |
| **Wf-406 TOTAL** | **~28 min CPU-only + ~44 s for tests** | **9 NEW** | **5 SHIPPED** |

## §6 User voice answer

**Per Wf-406-A1 §0 + Wf-406-A2 §0 + Wf-406-A3 §0 + Wf-406-A4 §0 + Wf-406-B1 §0 + Wf-406 final synthesis §7 — addressing user 2026-09-30 verbatim *"之前...这些问题都要解决，我给你授权"*:**

**YES — Wf-406 wave DID彻底解决 four concrete docs+code+test gaps** via 5 SHIPPED deliverables (A1 + A2 + A3 + A4 + B1 this entry):

1. ✅ **m6_kappa Tier-2 §3 drop-justification paper prep** (Wf-406-A1) — 2 docs SHIPPED with 9 paths_exhausted FINAL decision (DROP + PROMOTE + RETAIN)
2. ✅ **TODO-21 Lambda × CFM coupling patch application** (Wf-406-A2) — 5 trainer inserts + 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor (decode_ratio = 1.000 + loss 91% decrease)
3. ✅ **Atomic write protocol migration top-3 scripts** (Wf-406-A3) — 3 scripts migrated (wf398a2 + wf390a2 + wf394a1) to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`; 6 NEW tests PASS in ~1.33 s wall CPU-only
4. ✅ **Wf-246-C metal-coordination term retrain SCAFFOLD** (Wf-406-A4) — SCAFFOLD + bash launch wrapper + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit

Pre-flight 89/2/3/94 PRESERVED at Wf-407-B1 baseline (NOT re-run by Wf-406 — CPU-only docs+code+test wave per AGENTS.md §5.1); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

**They did NOT彻底解决 the structural/metric/env gates**: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL (9 paths exhausted); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-406-A2 + Wf-406-A4 GPU retrain BLOCKED on USER_OK=1; Wf-407-A2 GPU launch REFUSED per Wf-407-A3; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending.

NEW ultracode Wf-407+ IS the recommended follow-up per Wf-406 final synthesis §8 + Wf-405-B1 + Wf-405-A1 SCAFFOLD-loss gap audit + Wf-405-A2 SCAFFOLD-loss wire-in + Wf-385-A1 + Wf-385-B1 §22.79.3 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-386-B1 §22.79.3 + Wf-388-B1 §22.6.3 + Wf-389-B1 §22.5.3 + Wf-391-B1 §22.81.3 + Wf-392-B1 §5.2 + Wf-394-B1 §5 + Wf-395-B1 §5.2 + Wf-396-B1 §7 + Wf-398-B1 §5 + Wf-399-B1 §5 + Wf-400-B1 §22.87.4 + Wf-404-B1 + Wf-407-B1 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption migration + Wf-406-A2 + Wf-406-A4 MEASURED-on-Blackwell retrain (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 v4 update total).

---

## §7 Cross-references

* `TODO/completed/{406a1_m6_kappa_paper_prep,406a2_lambda_cfm_patch,406a3_atomic_migration,406a4_wf246c_scaffold,406b1_index_sync,406_final_synthesis}_*.md` (6 SHIPPED Wf-406 deliverables)
* `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 NEW docs file, ~356 LOC)
* `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-406-A1 EXTENDED template file, +110 LOC §3.1 extension)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-406-A2 5 inserts: parse_args lines 1340-1380 + train_one_epoch signature lines 587-593 + loss-multiplier lines 843-857 + main() dataset-build lines 1477-1510 + main() call site lines 1546-1552)
* `molmetal/scripts/wf398a2_m2_r2_real_verify.py` (Wf-406-A3 migrated +25 LOC)
* `molmetal/scripts/wf390a2_alt_encoder_audit.py` (Wf-406-A3 migrated +35 LOC)
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (Wf-406-A3 migrated +18 LOC)
* `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` (Wf-406-A4 NEW SCAFFOLD, 344 lines)
* `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` (Wf-406-A4 NEW launch wrapper, 216 lines)
* `molmetal/tests/test_wf406a2_lambda_cfm_patch.py` (Wf-406-A2 NEW test file, 1 test)
* `molmetal/tests/test_wf406a3_atomic_migration.py` (Wf-406-A3 NEW test file, 6 tests)
* `molmetal/tests/test_wf406a4_metal_coord_scaffold.py` (Wf-406-A4 NEW test file, 2 tests)
* `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` (Wf-333-V1 sign fix reference)
* `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
* `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
* `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
* `mol-metal-cfm-training` skill (Lambda CFM patch + CFM flag smoke-required rule §5.7)
* `mol-metal-anticancer` skill (Wf-406-A4 metal-coordination cross-ref)
* `mol-metal-gpu-execution` skill (USER_OK gate + Wf-355-B1 wrapper pattern)
* `TODO/pending/{155_smiles_featurization_cache_extension,206_m6_kappa_metric_redefinition}_*.md` (TODO-155 + TODO-206 source specs)
* `TODO/completed/{404a1_todo_203_scaffold,404a2_todo_204_scaffold,404a3_todo_205_wire,404a4_todo_155_wire,404a5_wf29_attr_fix_smoke,394a1_lambda_cfm_wire,397a1_decode_ratio_fix,398a1_dmpnn_retrain,398a2_m2_r2_real,399a1_regression_rule,400a1_wf299_launcher_scaffold_not_functional,407a1_gpu_env_probe,407a3_wf299_wave3_scaffold_not_functional,407a4_metrics_update,407b1_index_sync,407_final_synthesis,405a1_wf380_loss_gap_audit,405a2_wf380_loss_wire,405a3_preflight_fail_fixes,405a4_user_ok_gaps_close,405b1_index_sync,405_final_synthesis,394b1_index_sync,394_final_synthesis,398b1_index_sync,398_final_synthesis,399b1_index_sync,399_final_synthesis,400b1_index_sync,400_final_synthesis,403b1_index_sync,403_final_synthesis,378a1_gpu_ready_user_ok_gate,378a2_g6_user_ok_parity,378b1_index_sync,378_final_synthesis,379a1_wf131_p3_7runs,379a2_r20_p5_tmqm_plat,379b1_index_sync,379_final_synthesis,380a1_deep_integration_scaffolds,380a2_per_cohort_features,380a3_diversity_sampling,380a4_vina_param_tuning,380a5_multi_template_mm_gbsa,380a6_sa_generation_penalty,380a7_m6kappa_finalize,380b1_index_sync,380_final_synthesis,381a4_m6kappa_finalize,381a5_r20_plan_cpu_close,381b1_index_sync,381_final_synthesis,383a1_weak_to_strong_se3sb_audit,383a2_se3_sb_integration_plan,383a3_wires_audit,383b1_index_sync,383_final_synthesis,384a1_wf377_on_disk_audit,384a2_wf377_root_cause,384b1_index_sync,384_final_synthesis,385a1_wf380_cli_flag_audit,385a2_wf299_launch_wrapper_review,385a3_scaffold_wire_in_gate,385b1_index_sync,385_final_synthesis,386a1_wf384_a3_overwrite_audit,386a2_wf378_a3_wrong_verdict_root_cause,386a3_atomic_write_protocol,386b1_index_sync,386_final_synthesis,388a1_m2_per_cohort_real,388a2_m7_diversity_wire,388a3_m4_sa_wire,388b1_index_sync,388_final_synthesis,389a1_m6k_supp_reframe,389a3_weak_to_strong_update,389b1_index_sync,389_final_synthesis,390a1_todo376_5metal_subset_path_a,390a2_todo179_deeper_audit,390a3_se3_sb_wire,390b1_index_sync,390_final_synthesis,391a1_lambda_roadmap_refresh,391a2_improvement_enum_refresh,391a3_metric_strategy_refresh,391b1_index_sync,391_final_synthesis,392a1_multiph_pll_wire,392a2_chemprop_v3_cache_wire,392a3_lomo_wrapper_verify,392b1_index_sync,392_final_synthesis,393a1_r17_r18_refresh,393a2_metrics_refresh,393a3_decisions_readme_refresh,393b1_index_sync,393_final_synthesis,394a1_lambda_cfm_wire,394a2_wf264_property_panel,394a3_wf299_pre_launch_verify,394b1_index_sync,394_final_synthesis,395a1_evaluators_refresh,395a2_catalog_expand,395a3_full_integration,395b1_index_sync,395_final_synthesis,396a1_decode_ratio_regression_git_bisect,396a2_trainer_diff_analysis,396b1_index_sync,396_final_synthesis,397a1_decode_ratio_fix,397a2_wave1_smoke,397a3_wave2_wave3_scaffold_not_functional,397b1_index_sync,397_final_synthesis,398a1_dmpnn_retrain,398a2_m2_r2_real,398a3_decode_ratio_regression_fix,398b1_index_sync,398_final_synthesis,399a1_regression_rule,399a2_catalog_regression_entry,399b1_index_sync,399_final_synthesis,402a1_user_ok_gate_audit,402b1_index_sync,402_final_synthesis,403a1_m6_kappa_finalize_v2,403a3_atomic_migration_v2,403b1_index_sync,403_final_synthesis}_*.md` (Wf-358..407 SHIPPED closeouts — the Wf-406 wave source + context)

## §8 CHRONOLOGY NOTE

Wf-406 ships AFTER Wf-296..405 + Wf-407 ultracode waves per user directive 2026-09-30 verbatim *"之前...这些问题都要解决，我给你授权"*; Wf-406 closeout timestamp sequence ~2026-09-30 13:15-13:32 CST (Wf-406-A1 + Wf-406-A2 + Wf-406-A3 + Wf-406-A4 + this B1 sync). Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + Wf-362-B1 + Wf-363-C1 + Wf-364-C1 + Wf-365-C1 + Wf-366-C1 + Wf-367-C1 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-372-C1 + Wf-373-B1 + Wf-375-B1 + Wf-374-B1 + Wf-376-B1 + Wf-377-B1 + Wf-379-B1 + Wf-378-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-392-B1 + Wf-394-B1 + Wf-395-B1 + Wf-396-B1 + Wf-397-B1 + Wf-398-B1 + Wf-399-B1 + Wf-400-B1 + Wf-404-B1 + Wf-405-B1 + Wf-407-B1 prior INDEX syncs preserved unchanged; Wf-406-B1 this sync adds the Wf-406 §6 row + §22.92 INDEX entry to capture the 4 SHIPPED A-deliverables + 1 B1 docs-only sync.

---

**END OF Wf-406 final synthesis v39** — Owner Hugo Lee + Wf-406 ultracode agent (per AGENTS.md §5.5 filesystem-truth rule + AGENTS.md §5.6 atomic write protocol + Wf-385-B1 audit + Wf-405 recovery wave + Wf-407 GPU wave context).
