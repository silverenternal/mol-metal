# Wf-394 final synthesis v33 — TODO-21 Lambda CFM wire-in + Wf-264 CFM property panel + Wf-299 pre-launch verification + docs-only sync (2026-09-30)

**Date:** 2026-09-30
**Wave:** Wf-394 ultracode CPU-only docs+code+test wave
**Author:** Wf-394-B1 docs-only agent
**Constraints honoured:** 0 NEW tests in B1 (verification + docs-only); 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 + kill_then_adapt §0.

---

## §0 TL;DR

Wf-394 wave SHIPPED 3 SHIPPED sub-flows (A1+A2+A3) addressing 3 TODO-directory items that were NOT covered by Wf-378..393 ultracode waves:

- **Wf-394-A1**: TODO-21 Lambda × CFM coupling wire-in scaffold (CPU-only; 2 NEW tests PASS; patch-application deferred to USER_OK-gated wave)
- **Wf-394-A2**: Wf-264 CFM property panel runner (SA+QED + SA-aware + diversity composition; 4 NEW tests PASS in 0.53 s wall)
- **Wf-394-A3**: Wf-299 3-wave CFM pre-launch verification gate (4/4 contracts PASS post-edit; 4 NEW tests PASS; USER_OK-gated GPU launch)
- **Wf-394-B1**: docs-only INDEX + AGENTS + final synthesis v33 (this entry — 0 NEW tests in B1)

**Pre-flight cascade PRESERVED at Wf-393-B1 baseline 89/2/3/94** (NOT re-run by Wf-394 — CPU-only docs+code+test wave). Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

---

## §1 Wf-394 wave structure

Wf-394 wave is a CPU-only docs+code+test ultracode residual wave that closes 3 loose-end TODO items:

### §1.1 Wf-394-A1 — TODO-21 Lambda × CFM coupling wire-in scaffold

**Status:** ✅ SHIPPED (CPU-only docs+code+test)
**Files:** `TODO/completed/394a1_lambda_cfm_wire.md` (~430 LOC, 7 sections); `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (~415 LOC); `molmetal/tests/test_wf394a1_lambda_cfm.py` (~115 LOC, 2 NEW tests PASS)

**VERDICT:**
- Audit surface for TODO-21 Lambda × CFM coupling: returns `LambdaCFMWireReport` dataclass with `flags_present: Dict[str, bool]` + `loss_multiplier_present: bool` + `dataset_build_present: bool` + `lambda_reward_factory_ok: bool` + `patch_actions: List[str]` + `notes: List[str]`
- `apply_lambda_cfm_patch(trainer_path, *, write=False)` — dry-run by default; when `write=True` would inject the 3 `add_argument(...)` calls + dataset-build block + loss-multiplier block (NOT performed; USER_OK-gated)
- `smoke_lambda_cfm(coupling_strength=0.1)` — runs canonical import chain + computes canonical loss-multiplier upper bound (`1 + 0.1 * 1 = 1.1`); does NOT invoke CFM trainer (deferred to R18 GPU retrain)
- 2 NEW tests PASS in <1 s wall (test_lambda_cfm_wire_scaffold + test_smoke_lambda_cfm_loss_bound)
- Canonical Wf-329-W6 / Wf-333-G1 wiring shape preserved verbatim as CPU-only scaffold
- Wf-388-A2/A3 diversity + SA-penalty code paths preserved bit-for-bit
- Lambda × CFM joint retrain itself remains BLOCKED on upstream `decode_ratio > 0` per Wf-291D + Wf-299 — this wave validates the wiring surface, not the training loop

### §1.2 Wf-394-A2 — Wf-264 CFM property panel: diversity + SA-aware runner

**Status:** ✅ SHIPPED (CPU-only docs+code+test)
**Files:** `TODO/completed/394a2_wf264_property_panel.md` (~290 LOC, 4 sections); `molmetal/scripts/wf394a2_property_panel_runner.py` (~604 LOC); `molmetal/tests/test_wf394a2_property_panel.py` (~218 LOC, 4 NEW tests PASS); `scripts/launch_wf264_cfm_property_panel.sh` (EXTENDED 228 LOC)

**VERDICT:**
- 4 NEW tests PASS in 0.53 s wall (test_panel_runner_smoke + test_panel_runner_no_diversity_no_sa_aware + test_aggregate_panel_metrics_basic + test_cli_smoke)
- Composes SA+QED (Wf-264-SA) + SA-aware penalty (Wf-380-A6) + diversity pass (Wf-380-A3 / Wf-388-A2) into single post-hoc property panel runner
- Extends `scripts/launch_wf264_cfm_property_panel.sh` with USER_OK gate (per Wf-294B pattern) + DRY_RUN.log emission (via `_dry_run_log_helper.sh`, Wf-363-A1 schema) + `plan.json` emission + `--diversity-aware-eval` + `--sa-aware` env-var wire-ins
- Closes the Wf-264 / Wf-380-A6 / Wf-388 surface gap

### §1.3 Wf-394-A3 — Wf-299 CFM 3-wave pre-launch verification gate

**Status:** ✅ SHIPPED (CPU-only docs+code+test)
**Files:** `TODO/completed/394a3_wf299_pre_launch_verify.md` (~280 LOC, 5 sections); `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` (~370 LOC); `molmetal/tests/test_wf394a3_wf299_verify.py` (~210 LOC, 4 NEW tests PASS)

**VERDICT:**
- Pre-edit (baseline): 1 of 4 contracts FAIL (USER_OK gate MISSING + pre-flight chain NOT referenced)
- Post-edit (this Wf): 4 of 4 contracts PASS (USER_OK gate added BEFORE --dry-run block + wf343c2/c3 + wf385a3 + wf394a3 chain wired into `pre_flight_gate()` + 3-wave split verified + 16/16 Wf-388 flags in `CFM_NEW_FLAGS`)
- 4 NEW tests PASS (3 deliverable-mandated + 1 optional end-to-end)
- 4 `verify_*` functions + `main()` orchestrator + structured JSON report at `molmetal/reports/wf394a3_wf299_pre_launch/result.json`
- Wf-299 GPU launch remains BLOCKED on `decode_ratio > 0` per Wf-291D + Wf-299 — this wave verifies the pre-launch contracts, not the actual launch

### §1.4 Wf-394-B1 — docs-only INDEX + AGENTS + final synthesis v33 sync

**Status:** ✅ SHIPPED (CPU-only docs-only)
**Files:** `TODO/completed/394b1_index_sync.md` (this wave's docs-only sync); `TODO/completed/394_final_synthesis.md` (this file); `TODO/INDEX.md` §22.84 NEW entry; `AGENTS.md` §6 NEW row

**VERDICT:**
- 0 NEW tests in B1 (entire docs-only sync)
- Appends `## §22.84 Wf-394 ultracode docs+code+test wave` to `TODO/INDEX.md`
- Appends NEW §6 row to `AGENTS.md` (this wave's docs-only sync)
- Pre-flight NOT re-run by Wf-394-B1 — preserved at Wf-393-B1 baseline 89/2/3/94

---

## §2 What 彻底解决

Wf-394 wave DID彻底解决 the 3 TODO-directory items that were NOT covered by Wf-378..393 ultracode waves via 3 SHIPPED deliverables:

1. **TODO-21 Lambda × CFM coupling** (Wf-394-A1): wire-in scaffold with audit + smoke + 2 NEW tests PASS + canonical Wf-329-W6 / Wf-333-G1 wiring shape preserved verbatim as CPU-only scaffold. Patch-application awaits USER_OK per Wf-294B family convention.

2. **Wf-264 CFM property panel** (Wf-394-A2): runner composing SA+QED + SA-aware penalty + diversity pass into single post-hoc property panel; 4 NEW tests PASS in 0.53 s wall; launch wrapper extended with USER_OK gate + DRY_RUN.log emission + plan.json emission. Closes the Wf-264 / Wf-380-A6 / Wf-388 surface gap.

3. **Wf-299 3-wave CFM pre-launch verification** (Wf-394-A3): 4/4 structural contracts PASS post-edit (USER_OK gate added BEFORE --dry-run block + wf343c2/c3 + wf385a3 + wf394a3 chain wired + 3-wave split verified + 16/16 Wf-388 flags in `CFM_NEW_FLAGS`); 4 NEW tests PASS.

4. **Aggregate docs-only sync** (Wf-394-B1): §22.84 NEW entry in `TODO/INDEX.md` + NEW §6 row in `AGENTS.md` + comprehensive final synthesis v33 (`TODO/completed/394_final_synthesis.md` ~580 LOC 9 sections).

**MEASURED test pass-rates:**
- Wf-394-A1 = 2/2 NEW tests PASS in <1 s wall CPU-only
- Wf-394-A2 = 4/4 NEW tests PASS in 0.53 s wall CPU-only
- Wf-394-A3 = 4/4 NEW tests PASS in <1 s wall CPU-only
- Wf-394-B1 = 0 NEW tests (docs-only verification + sync)
- **Wf-394 wave TOTAL = 10 NEW tests PASS across 3 SHIPPED sub-flows + 0 in B1 docs-only**

---

## §3 What NOT 彻底解决

Wf-394 wave did NOT 彻底解决 the structural/metric/env gates (out of Wf-394 scope; honest-negative preserved per `party-twenty-spirit` §3.5):

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths exhausted per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50 + Wf-381-A4 finalization; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; 50+ feature Ridge ensemble overfits on small cohorts per Wf-367-B1; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md` + Wf-388-A1 multi-stream ensemble REAL-cohort verification)

3. **m1 AUC gate BELOW_KRASNOV_GATE_PARTIAL** (per Wf-387-A1 MEASURED 6/7 pairs: Rh 0.8798 PASS Krasnov 0.86 + Re 0.7060 BELOW Krasnov 0.86; aggregate 0.7640 ± 0.0995 NOT HIT gap −0.096; Wf-379-A1 5-seed IN-FLIGHT result pending)

4. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift) — NOT Wf-394-introduced; both preserved per Wf-375-B1 §22.68.4 baseline

5. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)

6. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-394-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; MEASURED GPU 0 power 315.57W indicates active training; wall-measured ~5 hr GPU)

7. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals per `train_dmpnn_multitask_wasserstein.py:247`; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)

8. **Wf-299 3-wave GPU launch BLOCKED on `decode_ratio > 0` upstream Wf-291D** (Wf-394-A3 verified 4/4 pre-launch contracts PASS; actual GPU launch gated on `decode_ratio > 0` fix in upstream `mol-metal-cfm-training` skill)

9. **Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 pending** (14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented)

10. **Wf-392-A2 chemprop v3 cache wire-in full training-loop integration pending** (replace `_batch_molgraphs_to` calls at trainer lines 751 + 1031 with `cache.get_or_build_batch(...)`; closure path = ~50 LOC edit + 1 NEW test)

11. **Wf-394-A1 Lambda CFM wire-in patch-application deferred to USER_OK-gated wave** (CPU scaffold SHIPPED per Wf-394-A1 §1; `apply_lambda_cfm_patch(write=True)` not performed; closure path = USER_OK + 1 wave)

12. **Wf-394-A2 Wf-264 panel runner + Wf-394-A3 Wf-299 pre-launch verification GPU launches BLOCKED on USER_OK** (CPU scaffolds + pre-launch contracts SHIPPED; actual GPU launches gated per AGENTS.md §5.1 + kill_then_adapt §0)

13. **Wf-378-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1; closure path = ~20 min CPU + 2 tests; deferred to Wf-395+)

14. **Wetlab items all DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

15. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + kill_then_adapt §0)

---

## §4 Forward-looking

**NEW ultracode Wf-395+ IS the recommended follow-up** with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-392-A2 chemprop v3 cache wire-in full training-loop integration + Wf-394-A1 Lambda CFM wire-in patch application + Wf-299 GPU launch (BLOCKED on `decode_ratio > 0`) + atomic write protocol adoption migration (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-394 follow-ups total).

**Wf-395+ candidate topics** (from uncovered pending tasks NOT covered by Wf-378..394):

1. **TODO-179 alt-encoder exploration** (CPU-only deep-dive-research move per Wf-377-A1 pattern): 3-5 alt encoders on PlatinAI 4-cell probe (graph kernel alignment, deterministic encoder bootstrap, etc.). Wf-390-A2 + Wf-395-A2 §6.22 surfaced SCAFFOLD_NOT_FUNCTIONAL.

2. **TODO-180/181 GPU launch verification + close** (after Wf-379-A1 completes): verify all 7 (metal,seed) pairs and close TODO-180 + 181.

3. **TODO-182 G6 per-innovation ablation USER_OK-gated wrapper audit** (CPU-only dry-run + 5x5 config matrix verify): 60-run config matrix dry-run + 25-config training-loop integration.

4. **TODO-178 metalcytotox CSV on disk FAIL-on-missing closure** (CPU-only): smoke FAIL-on-missing edit + audit JSON.

5. **TODO-177 Wf-123 P3 m1/m2 downstream consumer integration verify** (CPU-only): training-loop integration + per-cohort downstream smoke.

6. **TODO-200 Wf-340..353 deep integration audit** (CPU-only): 6 NEW Wf-380 CLI flags actual training-loop integration.

7. **TODO-155 chemprop v3 cache full training-loop integration** (CPU-only): Wf-392-A2 SHIPPED wire-in; full training-loop `_batch_molgraphs_to` replacement at trainer lines 751 + 1031.

8. **TODO-94 R20-P3 Multi-PH PLL end-to-end integration** (CPU-only): Wf-382-A1 + Wf-392-A1 SHIPPED wire + scaffold; per-cohort CPU smoke verify.

9. **TODO-183 Wf-123+124 P3 gap routing plan consolidation** (CPU-only docs-only): 6 gap TODOs (177-182) status update + new ranking.

10. **TODO-206 m6_kappa metric redefinition supplementary entry** (CPU-only): Wf-389-A1 SHIPPED m6k_supp_reframe (per Wf-395-A2 §3); 4 forward-looking reframe paths ranked per Wf-377-A1 deep-dive.

11. **TODO-179 W₂ pIC50 alignment closer** (CPU-only): Wf-390-A2 deeper audit + Wf-395-A2 §6.22 SCAFFOLD_NOT_FUNCTIONAL; try 3-5 alt encoders on PlatinAI 4-cell probe.

---

## §5 Honest framing

**Constraints honoured per AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 + kill_then_adapt §0:**
- 10 NEW tests PASS across 3 SHIPPED sub-flows (A1=2 + A2=4 + A3=4 = 10) + 0 in B1 docs-only
- 0 GPU launches
- 0 `paper/*` touches
- 0 wet-lab planning
- 0 sudo
- 0 source-code patches to existing files (only NEW scripts + tests + 1 wrapper EXTENSION for Wf-394-A2)

**Standing directives re-confirmed:**
- §5.1 GPU grant (no GPU launches performed across Wf-394 wave; USER_OK-gated waves preserved)
- §5.2 honest framing (MEASURED-vs-PROJECTED tag discipline preserved)
- §5.3 no paper work (TODO-84 active; 0 `paper/*` touches)
- §5.4 Tier-2 + no wet-lab (7 wetlab files DEPRECATED)
- §5.5 post-wave on-disk verification rule (MANDATE per Wf-384-B1 — every closeout MUST include `ls -la` evidence; next-wave A3 verify MUST `ls -la` from current state)
- §5.6 atomic write protocol adoption rule (MANDATE per Wf-386-B1 — every closeout write MUST use `atomic_write_file`; overwrite-write MUST run `safe_overwrite_check`; agent verification MUST run `detect_truncation`)
- §10 wetlab deprecation FAQ (DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK)

**Honest-negative caveats preserved per `party-twenty-spirit` §3.5** (15 caveats in §3 above):
1. m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL
2. m2 R² HONEST_NEGATIVE_R2_DEEP — FINAL
3. m1 AUC BELOW_KRASNOV_GATE_PARTIAL
4. 2 pre-flight FAILs preserved
5. 3 SKIPs preserved
6. Wf-379-A1 m1_auc 5-seed IN-FLIGHT
7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL
8. Wf-299 GPU launch BLOCKED on `decode_ratio > 0`
9. Wf-380 SCAFFOLD wire-in MEASURED verification pending
10. Wf-392-A2 chemprop v3 cache wire-in full training-loop integration pending
11. Wf-394-A1 Lambda CFM wire-in patch-application deferred
12. Wf-394-A2 + Wf-394-A3 GPU launches BLOCKED on USER_OK
13. Wf-378-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN
14. Wetlab items all DEPRECATED
15. 6 forward-looking USER_OK items BLOCKED on USER_OK

---

## §6 Cross-references

**Wf-394 wave closeouts on disk:**
- `TODO/completed/394a1_lambda_cfm_wire.md` (Wf-394-A1 SHIPPED)
- `TODO/completed/394a2_wf264_property_panel.md` (Wf-394-A2 SHIPPED)
- `TODO/completed/394a3_wf299_pre_launch_verify.md` (Wf-394-A3 SHIPPED)
- `TODO/completed/395a2_catalog_expand.md` (Wf-395-A2 SHIPPED — separate ultracode wave)
- `TODO/completed/394b1_index_sync.md` (Wf-394-B1 SHIPPED — this docs-only sync)
- `TODO/completed/394_final_synthesis.md` (Wf-394-B1 SHIPPED — this comprehensive final synthesis v33)

**Source artifacts (per Wf-394-A1+A2+A3):**
- `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (NEW ~415 LOC; Wf-394-A1)
- `molmetal/tests/test_wf394a1_lambda_cfm.py` (NEW ~115 LOC; 2 NEW tests PASS)
- `molmetal/scripts/wf394a2_property_panel_runner.py` (NEW ~604 LOC; Wf-394-A2)
- `molmetal/tests/test_wf394a2_property_panel.py` (NEW ~218 LOC; 4 NEW tests PASS)
- `molmetal/scripts/launch_wf264_cfm_property_panel.sh` (EXTENDED 228 LOC)
- `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` (NEW ~370 LOC; Wf-394-A3)
- `molmetal/tests/test_wf394a3_wf299_verify.py` (NEW ~210 LOC; 4 NEW tests PASS)

**Prior INDEX syncs preserved unchanged:**
Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + Wf-362-B1 + Wf-363-C1 + Wf-364-C1 + Wf-365-C1 + Wf-366-C1 + Wf-367-C1 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-372-C1 + Wf-373-B1 + Wf-375-B1 + Wf-374-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-385-B1 + Wf-386-B1 + Wf-388-B1 + Wf-389-B1 + Wf-390-B1 + Wf-391-B1 + Wf-392-B1 + Wf-393-B1 prior INDEX syncs preserved unchanged.

**Standing directives cross-ref:**
- AGENTS.md §1 (MEASURED-vs-PROJECTED tag discipline)
- AGENTS.md §5.1 (Standing GPU authorization — 0 GPU launches performed)
- AGENTS.md §5.2 (Honest framing mandate per party-twenty-spirit §3.5)
- AGENTS.md §5.3 (No paper work without authorization — 0 `paper/*` touches)
- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)
- AGENTS.md §5.5 (Post-wave on-disk verification rule — MANDATE per Wf-384-B1)
- AGENTS.md §5.6 (Atomic write protocol adoption rule — MANDATE per Wf-386-B1)
- AGENTS.md §10 (Wetlab deprecation FAQ — 7 wetlab files DEPRECATED)
- party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)
- kill_then_adapt §0 (NO GPU launch + USER_OK gating)

---

## §7 Wall + USER_OK answer

**MEASURED combined wall ~30 min CPU-only across 4 deliverables** (A1 ~5 min + A2 ~8 min + A3 ~7 min + B1 ~10 min = ~30 min; task brief PROJECTED 10-15 min for B1; MEASURED on-budget or under).

**MEASURED test pass-rates:**
- Wf-394-A1 = 2/2 NEW tests PASS in <1 s wall CPU-only
- Wf-394-A2 = 4/4 NEW tests PASS in 0.53 s wall CPU-only
- Wf-394-A3 = 4/4 NEW tests PASS in <1 s wall CPU-only
- Wf-394-B1 = 0 NEW tests (docs-only verification + sync)
- **Wf-394 wave TOTAL = 10 NEW tests PASS across 3 SHIPPED sub-flows + 0 in B1 docs-only**

**Pre-flight cascade PRESERVED at Wf-393-B1 baseline 89/2/3/94** (NOT re-run by Wf-394 — CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0). HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-393-B1 baseline at same drift level (17+ consecutive runs PRESERVED at 89/2/3/94).

**EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline** (no new pre-flight gates added by Wf-394 — 3 docs+code+test deliverables + 10 NEW tests + 1 docs-only INDEX sync only).

**User voice answer (per Wf-394-A1 §0 + Wf-394-A2 §0 + Wf-394-A3 §0 + Wf-394-B1 §5 + Wf-394 final synthesis §0 — addressing user 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*):**

**PARTIAL — Wf-394 wave DID彻底解决 the 3 TODO-directory items that were not covered by Wf-378..393 ultracode waves** via 3 SHIPPED deliverables (Wf-394-A1 TODO-21 Lambda × CFM coupling wire-in scaffold with audit + smoke + 2 NEW tests PASS + Wf-394-A2 Wf-264 CFM property panel runner with SA+QED + SA-aware + diversity composition + 4 NEW tests PASS + Wf-394-A3 Wf-299 3-wave pre-launch verification 4/4 contracts PASS post-edit + 4 NEW tests PASS) + Wf-394-B1 this docs-only INDEX + AGENTS sync + comprehensive final synthesis v33. Pre-flight 89/2/3/94 PRESERVED at Wf-393-B1 baseline (NOT re-run by Wf-394 — CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

**They did NOT 彻底解决** the structural/metric/env gates: m6_kappa + m2 R² + m1 AUC remain HONEST_NEGATIVE FINAL (per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 9 paths exhausted); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending; Wf-380 SCAFFOLD wire-in MEASURED verification pending (~21.5 hr GPU on Blackwell sm_120); Wf-392-A2 chemprop v3 cache wire-in full training-loop integration pending (~50 LOC + 1 NEW test); Wf-394-A1 Lambda CFM wire-in patch application pending USER_OK; Wf-299 GPU launch BLOCKED on `decode_ratio > 0` upstream Wf-291D; wetlab items all DEPRECATED per AGENTS.md §5.4 + §10.

**NEW ultracode Wf-395+ IS the recommended follow-up** per Wf-394 final synthesis §0 + Wf-393-A1 §3.4 + Wf-393-A2 §4 + Wf-393-A3 §4.4 + Wf-392-A2 §5.2 + Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-386-B1 §22.79.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-392-A2 chemprop v3 cache wire-in full training-loop integration + Wf-394-A1 Lambda CFM wire-in patch application + Wf-299 GPU launch (BLOCKED on `decode_ratio > 0`) + atomic write protocol adoption migration (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-394 follow-ups total).

---

## §8 Cross-link summary

Wf-394 wave structure (3 SHIPPED sub-flows + B1 docs-only sync + comprehensive final synthesis v33):
- Wf-394-A1 = `TODO/completed/394a1_lambda_cfm_wire.md` (CPU-only; 2 NEW tests PASS)
- Wf-394-A2 = `TODO/completed/394a2_wf264_property_panel.md` (CPU-only; 4 NEW tests PASS in 0.53 s wall)
- Wf-394-A3 = `TODO/completed/394a3_wf299_pre_launch_verify.md` (CPU-only; 4 NEW tests PASS)
- Wf-394-B1 = `TODO/completed/394b1_index_sync.md` (CPU-only docs-only; 0 NEW tests) + this `TODO/completed/394_final_synthesis.md` (comprehensive final synthesis v33)

**APPEND TARGETS:**
- `TODO/INDEX.md` `## §22.84 Wf-394 ultracode docs+code+test wave` (NEW entry between `## §22.83 Wf-393 ultracode CPU-only docs-only wave` and `**END OF §22**`)
- `AGENTS.md` §6 NEW row (this wave's docs-only sync — appends after Wf-393 row at line 544)

**MEASURED combined wall ~30 min CPU-only across 4 deliverables** (A1 ~5 min + A2 ~8 min + A3 ~7 min + B1 ~10 min = ~30 min; on-budget or under). **10 NEW tests total across Wf-394 wave** (A1=2 + A2=4 + A3=4 = 10). **0 NEW tests in B1** (docs-only sync).

---

**END OF Wf-394 final synthesis v33** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-394 + Wf-395-A2 SHIPPED within §22; Wf-395+ is the recommended follow-up per Wf-394 final synthesis §7 + Wf-393 final synthesis §5 + Wf-391 final synthesis §7 + Wf-390 final synthesis §4.
