# Wf-406-B1 — docs-only INDEX + AGENTS + final synthesis sync — Wf-406 ultracode CPU-only m6_kappa paper prep + Lambda CFM patch + atomic migration + Wf-246-C metal-coord SCAFFOLD (per user directive 2026-09-30 + Wf-394-A1 patch + Wf-385-A3 prevention gate + AGENTS.md §5.6 atomic write protocol + Wf-405 recovery)

**Date:** 2026-09-30 (Wf-406-B1 closeout timestamp; this is the docs-only sync for the Wf-406 wave)
**Wf:** Wf-406 ultracode CPU-only docs+code+test wave (4 SHIPPED A1..A4 sub-flow deliverables + this B1 sync)
**User voice relay:** *"之前...这些问题都要解决，我给你授权"* (Wf-406 wave parent directive — same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-388/Wf-389/Wf-391/Wf-392/Wf-394/Wf-395/Wf-397/Wf-398/Wf-399/Wf-400/Wf-404/Wf-407)
**Owner:** Hugo Lee + Wf-406 ultracode agent
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-cfm-training` (Wf-406-A2 Lambda CFM patch); `mol-metal-gpu-execution` §0 (`kill_then_adapt` directive); `mol-metal-anticancer` (Wf-406-A4 cross-ref); no `paper/*` touches (TODO-84 active); no `wetlab/*` planning (AGENTS.md §5.4 + §10)

---

## §0 TL;DR

**Wf-406 wave ACTUAL count: 5 SHIPPED deliverables (A1 + A2 + A3 + A4 + B1 this entry)** — 4 sub-flow A-deliverables + 1 B1 docs-only INDEX + AGENTS + final synthesis (this row).

The Wf-406 wave DID彻底解决 four concrete docs+code+test gaps:

1. ✅ **m6_kappa Tier-2 §3 drop-justification paper prep** (Wf-406-A1) — 2 docs SHIPPED (`docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` ~356 LOC + EXTENDED `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1)
2. ✅ **TODO-21 Lambda × CFM coupling patch application** (Wf-406-A2) — 3 CLI flags + dataset-build block + loss-multiplier block applied to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor (decode_ratio = 1.000 + loss 91% decrease)
3. ✅ **Atomic write protocol migration of top-3 closeout-drafting scripts** (Wf-406-A3) — 3 scripts migrated (wf398a2 + wf390a2 + wf394a1) to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`; 6 NEW tests PASS in ~1.33 s wall CPU-only
4. ✅ **Wf-246-C metal-coordination term retrain SCAFFOLD** (Wf-406-A4) — SCAFFOLD script (`molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` 344 lines) + bash launch wrapper (`scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` 216 lines) + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit

**Wall: ~25 min CPU-only across 5 deliverables (MEASURED); 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; 9 NEW tests PASS across 3 SHIPPED sub-flows (A2=1 + A3=6 + A4=2; A1 = docs-only brief 0 NEW tests).**

---

## §1 Wf-406 wave structure (4 SHIPPED A-deliverables + B1 this sync)

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

### §1.2 Wf-406-A2 — TODO-21 Lambda × CFM coupling patch application SHIPPED

**Closeout:** `TODO/completed/406a2_lambda_cfm_patch.md` (16,219 bytes; 186 lines; 7 sections)
**Verdict:** ✅ 3 CLI flags + dataset-build block + loss-multiplier block applied to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; 1 NEW test PASS in 42.81 s wall CPU-only with §5.7 50-step smoke floor (decode_ratio = 1.000 + loss 91% decrease)

**5 trainer inserts:**
1. `parse_args()` 3 CLI flags (lines 1340-1380): `--lambda-cfm-coupling` + `--lambda-cfm-coupling-weight` + `--lambda-cfm-coupling-samples`
2. `train_one_epoch()` signature (lines 587-593): 2 new kwargs `lambda_cfm_coupling_weight` + `lambda_cfm_mean_weight`
3. `train_one_epoch()` loss-multiplier block (lines 843-857): `loss = loss * (1 + lambda_cfm_coupling_weight * lambda_cfm_mean_weight)` guarded by `weight > 0.0 AND mean_weight > 0.0`
4. `main()` dataset-build block (lines 1477-1510): builds `LambdaRewardDataset` if `--lambda-cfm-coupling` is set
5. `main()` call site (lines 1546-1552): passes the 2 kwargs

### §1.3 Wf-406-A3 — atomic write protocol migration top-3 scripts SHIPPED

**Closeout:** `TODO/completed/406a3_atomic_migration.md` (18,464 bytes; 388 lines; 8 sections)
**Verdict:** ✅ Migrate top-3 closeout-drafting scripts (wf398a2 + wf390a2 + wf394a1) to `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`; 6 NEW tests PASS in ~1.33 s wall CPU-only

**Migrated scripts:**
- `molmetal/scripts/wf398a2_m2_r2_real_verify.py` (+25 LOC: 159 → 184)
- `molmetal/scripts/wf390a2_alt_encoder_audit.py` (+35 LOC: 190 → 225)
- `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (+18 LOC: 605 → 623)

**Total LOC delta**: +78 LOC across 3 scripts.

### §1.4 Wf-406-A4 — Wf-246-C metal-coordination term retrain SCAFFOLD SHIPPED

**Closeout:** `TODO/completed/406a4_wf246c_scaffold.md` (this sync, 179 lines; 6 sections)
**Verdict:** ✅ SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + 5-step CPU smoke + kcal sign audit

**3 deliverables:**
1. SCAFFOLD script `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` (344 lines, 12,602 bytes)
2. Bash launch wrapper `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` (216 lines)
3. Test file `molmetal/tests/test_wf406a4_metal_coord_scaffold.py` (~250 lines, 2 NEW tests)

---

## §2 What Wf-406-B1 sync added

### §2.1 `TODO/INDEX.md §22.92 Wf-406 ultracode CPU-only m6_kappa paper prep + Lambda CFM patch + atomic migration + Wf-246-C metal-coord SCAFFOLD`

Appended between §22.91 Wf-405 and §22.93 Wf-407 (~85 lines). 7 subsections:
- §22.92.0 — ACTUAL on-disk state per AGENTS.md §5.5 filesystem-truth rule: 5 SHIPPED files (A1 + A2 + A3 + A4 + B1)
- §22.92.1 — Wf-406 sub-flows SHIPPED: A1 + A2 + A3 + A4 + B1 all 5 deliverables
- §22.92.2 — What 彻底解决 (Wf-406 wave DID solve)
- §22.92.3 — What NOT 彻底解决 (preserved honest-negatives per `party-twenty-spirit §3.5`)
- §22.92.4 — MEASURED test pass-rates (per Wf-406 wave): 9 NEW tests
- §22.92.5 — User voice answer (per Wf-406 final synthesis §7)
- §22.92.6 — Cross-link + chronology note

### §2.2 NEW §6 row in `AGENTS.md` for Wf-406 (last-update trailer extended)

The `AGENTS.md` §6 Recent Wf history pointer table is extended with a NEW row for Wf-406-B1 sync. The row text matches Wf-406 final synthesis §0 TL;DR.

### §2.3 `TODO/completed/406_final_synthesis.md` comprehensive final synthesis v39

`TODO/completed/406_final_synthesis.md` (~600 LOC, 9 sections: §0 TL;DR + §1 wave structure + §2 what 彻底解决 + §3 what NOT 彻底解决 + §4 forward-looking + §5 honest framing + §6 cross-references + §7 wall budget + §8 USER_OK answer + §9 chronology note).

---

## §3 Cross-references

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
* `TODO/completed/{404a1_todo_203_scaffold,404a2_todo_204_scaffold,404a3_todo_205_wire,404a4_todo_155_wire,404a5_wf29_attr_fix_smoke,394a1_lambda_cfm_wire,397a1_decode_ratio_fix,398a1_dmpnn_retrain,398a2_m2_r2_real,399a1_regression_rule,400a1_wf299_launcher_scaffold_not_functional,407a1_gpu_env_probe,407a3_wf299_wave3_scaffold_not_functional,407a4_metrics_update,407b1_index_sync,407_final_synthesis,405a1_wf380_loss_gap_audit,405a2_wf380_loss_wire,405a3_preflight_fail_fixes,405a4_user_ok_gaps_close,405b1_index_sync,405_final_synthesis,394b1_index_sync,394_final_synthesis}_*.md` (Wf-394..407 SHIPPED closeouts — the Wf-406 wave source + context)

---

**Wf-406-B1 closeout** — 2026-09-30 — Owner Hugo Lee + Wf-406 ultracode agent (per AGENTS.md §5.5 filesystem-truth rule + AGENTS.md §5.6 atomic write protocol + Wf-405-B1 recovery wave + Wf-407-B1 GPU launch context).
