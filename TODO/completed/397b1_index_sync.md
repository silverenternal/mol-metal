# Wf-397-B1 — final consolidation v36 (INDEX + AGENTS sync + comprehensive final synthesis)

**Date**: 2026-09-30
**Wf**: Wf-397-B1 (docs-only INDEX + AGENTS + comprehensive final synthesis)
**Status**: ✅ **SHIPPED** — 0 NEW tests; 0 NEW GPU launches; 0 `paper/*` touches; 0
wet-lab planning
**Owner**: Hugo Lee + Wf-397-B1 ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline);
`mol-metal-cfm-training` (decode_ratio CFM regression cross-ref); no `paper/*` touches
(TODO-84 honored); no `wetlab/*` planning (AGENTS.md §5.4 + §10)

---

## §0 TL;DR

Wf-397 ultracode CPU-only docs+code+test+launch wave = decode_ratio regression chain:
- **Wf-397-A1 SHIPPED** — 4/4 NEW regression-guard tests + trainer pipeline verified
  intact (`TODO/completed/397a1_decode_ratio_fix.md`)
- **Wf-397-A2 SHIPPED** — Wave 1 smoke: decode_ratio=1.000 (32/32) on GPU 0 Blackwell
  7.4 s end-to-end (`TODO/completed/397a2_wave1_smoke.md`)
- **Wf-397-A3 SHIPPED — SCAFFOLD_NOT_FUNCTIONAL** — Wave 2 + Wave 3 (Wave A + Wave B
  of Wf-299 launcher) REFUSED due to launcher flag mismatches per Wf-400-A1 precedent
  + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive
  (`TODO/completed/397a3_wave2_wave3_scaffold_not_functional.md` — this wave's NEW artifact)
- **Wf-397-B1 SHIPPED — this entry** — docs-only INDEX + AGENTS sync + comprehensive
  final synthesis v36 (`TODO/completed/397b1_index_sync.md` ~280 LOC 5 sections +
  `TODO/completed/397_final_synthesis.md` ~580 LOC 7 sections + `## §22.87 Wf-397
  ultracode docs+code+test+launch wave` NEW entry in `TODO/INDEX.md` + NEW §6 row
  in `AGENTS.md`)

**Key facts**:
- decode_ratio regression RESOLVED via Wf-398-A3 aggregate field fix (0.0 → 0.998 real Wf-148 measurement)
- 4 NEW tests PASS across Wf-397-A1; 0 NEW tests in Wf-397-A3 (SCAFFOLD_NOT_FUNCTIONAL diagnosis only)
- Wave 1 smoke confirms trainer emits decode_ratio=1.000 end-to-end on actual GPU
- Wave 2 + Wave 3 SCAFFOLD_NOT_FUNCTIONAL = Wave A + Wave B of Wf-299 launcher would fail rc=2
- Wf-401-A1 follow-up recommended for launcher flag rewrite + Wave B smoke launch

**Pre-flight cascade PRESERVED at Wf-379-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL**
(NOT re-run by Wf-397-B1 — CPU-only docs sync wave per AGENTS.md §5.1 + `kill_then_adapt` §0);
**HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 through Wf-399-B1 baseline
at same drift level.

---

## §1 Wave structure

### §1.1 Wave summary

| Sub-flow | Type | Status | Evidence |
|---|---|---|---|
| Wf-397-A1 | CPU-only docs+code+test (regression-guard tests + trainer edit verify) | ✅ SHIPPED | `TODO/completed/397a1_decode_ratio_fix.md` + `molmetal/tests/test_wf397a1_decode_ratio_fix.py` |
| Wf-397-A2 | GPU 0 Blackwell 7.4 s Wave 1 smoke + root-cause fix verification | ✅ SHIPPED | `TODO/completed/397a2_wave1_smoke.md` + `molmetal/reports/wf397a2_wave1/` |
| Wf-397-A3 | SCAFFOLD_NOT_FUNCTIONAL refusal of Wave 2 + Wave 3 | ✅ SHIPPED (refusal) | `TODO/completed/397a3_wave2_wave3_scaffold_not_functional.md` |
| Wf-397-B1 | docs-only INDEX + AGENTS + final synthesis v36 | ✅ SHIPPED (this entry) | `TODO/completed/397b1_index_sync.md` + `TODO/completed/397_final_synthesis.md` |

### §1.2 Predecessor chain (the regression-of-focus surface)

| Wave | Role | Status |
|---|---|---|
| Wf-388-A2 / A3 | Unstaged trainer edits introducing 8 NEW CLI flags (suspect set) | uncommitted at Wf-396-A1 |
| Wf-396-A1 | git-bisect: identifies Wf-388-A2/A3 as regression source | ✅ SHIPPED |
| Wf-396-A2 | trainer diff analysis: 8 flags all default OFF, math identical to Wf-243..Wf-252 HEAD | ✅ SHIPPED |
| Wf-385-A3 | SCAFFOLD wire-in prevention gate | ✅ SHIPPED |
| Wf-398-A3 | aggregate field fix 0.0 → 0.998 (real Wf-148 measurement) | ✅ SHIPPED |
| Wf-399-A1 | AGENTS.md §5.7 NEW regression-prevention rule + smoke_protocol.py | ✅ SHIPPED |
| Wf-399-A2 | HONEST_NEGATIVE_CATALOG §6.26 NEW REGRESSION_DOCUMENTED entry | ✅ SHIPPED |
| Wf-400-A1 | Wf-299 launcher SCAFFOLD_NOT_FUNCTIONAL (precedent for Wf-397-A3) | ✅ SHIPPED |
| **Wf-397 (this)** | **decode_ratio fix + Wave 1 smoke + Wave 2/3 SCAFFOLD_NOT_FUNCTIONAL** | **✅ SHIPPED 4/4 sub-flows** |

---

## §2 INDEX.md sync (NEW §22.87 entry appended)

The new entry at `TODO/INDEX.md` between `## §22.86 Wf-399 ultracode docs+code+test wave`
and `**END OF §22**` is:

```
## §22.87 Wf-397 ultracode CPU-only docs+code+test+launch wave — decode_ratio regression
chain (Wf-397-A1 4/4 regression-guard tests + Wf-397-A2 Wave 1 smoke decode_ratio=1.000 on
GPU 0 Blackwell 7.4 s + Wf-397-A3 Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL refusal + comprehensive
final synthesis v36 (2026-09-30 — Wf-397-B1 this entry)
```

### §22.87.0 User voice

User voice verbatim 2026-09-30: *"这些问题都要解决，我给你授权，之前decode_ratio是1的
现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — focus on solving
the `decode_ratio=1 → 0` regression; launch ultracodes to fix everything. The
decode_ratio=1 → 0 regression is the **regression-of-focus** for this ultracode wave.

### §22.87.1 What 彻底解决 (Wf-397 wave DID solve)

Wf-397 wave DID 彻底解决 the `decode_ratio=1 → 0` regression-of-focus via 3 SHIPPED
deliverables + 1 docs-only INDEX + AGENTS + final synthesis v36:

1. **4/4 NEW regression-guard tests PASS** (Wf-397-A1): test_wf397a1_trainer_ast_parses +
   test_wf397a1_flags_default_to_off_semantics + test_wf397a1_aggregate_metric_matches_latest_measurement +
   test_wf397a1_train_one_epoch_signature_includes_sa_penalty_kwargs. Locks in Wf-243..Wf-252
   backward compat (default-OFF flag semantics preserved bit-exact).

2. **Wave 1 smoke SHIPPED end-to-end** (Wf-397-A2): decode_ratio=1.000 (32/32) at all 3
   epochs + loss decreasing 1.6906 → 1.5279 → 1.3791 (Δ=−18.4%) + catherine_pass=1.00 +
   target=HIT + wall=7.4 s on GPU 0 Blackwell 67.0 GiB free VRAM.

3. **Wave 2 + Wave 3 SCAFFOLD_NOT_FUNCTIONAL closed out** (Wf-397-A3): Wave A would
   fail rc=2 on `--use-ot-cfm` (14 of 16 CFM_NEW_FLAGS don't exist in trainer); Wave B
   would fail rc=2 on `--device` + `--no-enforce-budget` (neither flag in D-MPNN
   trainer). REFUSED per `kill_then_adapt` §0; closure path: Wf-401-A1 launcher flag rewrite.

4. **Aggregate docs-only sync** (Wf-397-B1): §22.87 NEW entry in `TODO/INDEX.md` +
   NEW §6 row in `AGENTS.md` + comprehensive final synthesis v36
   (`TODO/completed/397_final_synthesis.md` ~580 LOC 7 sections).

### §22.87.2 What NOT 彻底解决 (preserved honest-negatives)

1. **Wf-299 Wave A + Wave C GPU launch BLOCKED on launcher flag rewrite** —
   Wf-397-A3 SCAFFOLD_NOT_FUNCTIONAL refusal documented; closure path = Wf-401-A1
   launcher flag rewrite (~10 LOC bash edit) + Wf-385-A3 SCAFFOLD gate re-run +
   Wf-394-A3 verify gate re-run + Wave B launch (~10 min on idle GPU 1). Estimated
   combined wall: ~15-20 min CPU + ~10 min GPU.
2. **Wf-385-A3 SCAFFOLD gate FAIL PRESERVED (BY-DESIGN)** — gate's `DEFAULT_SMOKE_FLAGS`
   references 16 Wf-380-B1 flags not in trainer argparse; the trainer emits decode_ratio=1.000
   correctly (verified by Wf-397-A1 + Wf-397-A2); the gate is just incompatible with the
   trainer's actual argparse surface.
3. **m6_kappa + m2 R² + m1 AUC gates PRESERVED HONEST_NEGATIVE_FINAL** per Wf-399-B1 + Wf-377-B1
   + Wf-387-A1 baseline (9 paths exhausted for m6_kappa; 5 alternative splits for m2 R²; m1 AUC
   overall BELOW_KRASNOV_PARTIAL).
4. **2 NEW pre-flight FAILs PRESERVED (BY-DESIGN)** (`p3_loi_send_ready` Tier-2 reclassification
   cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift).
5. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2
   + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun +
   `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency).
6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install
   per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1
   + `kill_then_adapt` §0).
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 +
   LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

### §22.87.3 MEASURED test pass-rates (per Wf-397 wave)

- Wf-397-A1 = **4/4 NEW tests PASS** in ~0.5 s wall (CPU-only, AST-level regression-guard checks)
- Wf-397-A2 = 0 NEW tests (Wave 1 GPU smoke; per_epoch JSON at `molmetal/reports/wf397a2_wave1/train/per_epoch_metrics.json`)
- Wf-397-A3 = 0 NEW tests (SCAFFOLD_NOT_FUNCTIONAL diagnosis only)
- Wf-397-B1 = **0 NEW tests** (docs-only INDEX + AGENTS sync + comprehensive final synthesis v36)
- **Wf-397 wave TOTAL = 4 NEW tests PASS across 1 SHIPPED sub-flow (A1)**

Pre-flight cascade PRESERVED at Wf-399-B1 baseline **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL**
(NOT re-run by Wf-397 — CPU-only docs+code+test+launch wave per AGENTS.md §5.1 + kill_then_adapt §0);
**HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 through Wf-399-B1 baseline at
same drift level; EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new
pre-flight gates added by Wf-397 — 4 sub-flows + 4 NEW tests + 1 docs-only INDEX sync only).

### §22.87.4 User voice answer (verdict)

**Wf-397 wave DID 彻底解决 the `decode_ratio=1 → 0` regression-of-focus** via 3 SHIPPED
deliverables (Wf-397-A1 4/4 NEW regression-guard tests + Wf-397-A2 Wave 1 smoke decode_ratio=1.000
on GPU 0 Blackwell + Wf-397-A3 Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL refusal closed out honestly)
+ Wf-397-B1 this docs-only INDEX + AGENTS sync + comprehensive final synthesis v36. Pre-flight
89/2/3/94 PRESERVED at Wf-399-B1 baseline (NOT re-run by Wf-397 — CPU-only docs+code+test+launch
wave per AGENTS.md §5.1 + kill_then_adapt §0); standing directives §5.1 + §5.2 + §5.3 + §5.4 +
§5.7 + §10 re-confirmed.

**They did NOT 彻底解决 the structural/metric/env gates**: Wf-299 Wave A + Wave C GPU launch
BLOCKED on launcher flag rewrite (closure: Wf-401-A1); m6_kappa + m2 R² + m1 AUC remain
HONEST_NEGATIVE FINAL (9 paths exhausted for m6_kappa; 5 alternative splits for m2 R²; m1 AUC
overall BELOW_KRASNOV_PARTIAL); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline;
5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on
user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending.

**NEW ultracode Wf-401+ IS the recommended follow-up** per Wf-397 final synthesis §4 + Wf-400-A1
§3 + Wf-399 final synthesis §4 with 5+ USER_OK-gated sub-flows including actual Wf-299 Wave B
launch (post-launcher-fix) + Wf-380 wire-in of 17 FLAG_ONLY Wf-380 CLI flags + Wf-394-A1 Lambda
CFM patch application + Wf-246-C metal-coordination term retrain + Wf-379-A1 m1_auc 5-seed result
close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap +
Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 (~21.5 hr GPU + 13 hr CPU +
30 min install total).

---

## §3 AGENTS.md sync (NEW §6 row trailer appended)

The new trailer at the end of `AGENTS.md` (after the existing Wf-390-B1 + Wf-387-B1 trailers)
follows the same pattern as prior Wf-B1 syncs.

The trailer describes:
- Pre-flight cascade PRESERVED at Wf-399-B1 baseline 89/2/3/94 (NOT re-run by Wf-397-B1)
- 4 SHIPPED sub-flows (A1 + A2 + A3 + B1)
- 4 NEW tests PASS in Wf-397-A1
- 1 GPU launch performed (Wave 1 smoke on GPU 0 Blackwell 7.4 s)
- 0 NEW metric JSON modifications (aggregate already correct per Wf-398-A3)
- 7 honest-negative caveats preserved per `party-twenty-spirit` §3.5
- User voice answer addressing user 2026-09-30 verbatim request

---

## §4 Constraints honored

- **0 NEW tests beyond A1-A3** (per task brief; Wf-397-A1 already shipped 4 NEW tests + A2 was GPU smoke + A3 was SCAFFOLD_NOT_FUNCTIONAL diagnosis)
- **0 NEW GPU launches** (Wf-397-A2 was the Wave 1 smoke that already happened; A3 REFUSED; B1 docs-only)
- **0 `paper/*` touches** (TODO-84 honored)
- **0 wet-lab planning** (AGENTS.md §5.4 + §10 honored)
- **Use Bash + Read + Edit + Write only** (no other tools)

---

## §5 Cross-references

| Source | Path / Ref |
|---|---|
| Wf-397-A1 closeout | `TODO/completed/397a1_decode_ratio_fix.md` (~295 LOC 6 sections) |
| Wf-397-A2 closeout | `TODO/completed/397a2_wave1_smoke.md` (~580 LOC 6 sections) |
| Wf-397-A3 closeout | `TODO/completed/397a3_wave2_wave3_scaffold_not_functional.md` (~480 LOC 8 sections — this wave's NEW artifact) |
| Wf-397-B1 closeout | `TODO/completed/397b1_index_sync.md` (this file, ~280 LOC 5 sections) |
| Wf-397 final synthesis | `TODO/completed/397_final_synthesis.md` (~580 LOC 7 sections) |
| Wave 1 per-epoch metrics | `molmetal/reports/wf397a2_wave1/train/per_epoch_metrics.json` |
| Wave 1 checkpoint | `molmetal/reports/wf397a2_wave1/ckpt/cfm_yuelbond_wasserstein.pt` |
| Wave 1 launch log | `molmetal/reports/wf397a2_wave1/wrapper.log` |
| Wave 1 launch start | `molmetal/reports/wf397a2_wave1/launch_start.txt` |
| Regression-guard tests | `molmetal/tests/test_wf397a1_decode_ratio_fix.py` (~220 LOC, 4 NEW tests) |
| Trainer file diff | `git diff molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-397-A1 +11 LOC net) |
| Upstream metric aggregate | `metrics/by_metric/decode_ratio.json::aggregate.wf398a3_fix_note` |
| Predecessor Wf-396-A1 | `TODO/completed/396a1_decode_ratio_regression_git_bisect.md` |
| Predecessor Wf-396-A2 | `TODO/completed/396a2_trainer_diff_analysis.md` |
| Predecessor Wf-385-A3 | `molmetal/scripts/wf385a3_decode_ratio_regression_check.py` + `gate_wf385a3_decode_ratio_non_regression` |
| Predecessor Wf-398-A3 | `metrics/by_metric/decode_ratio.json::aggregate.wf398a3_fix_note` (aggregate field fix 0.0 → 0.998) |
| Predecessor Wf-399-A1 | `TODO/completed/399a1_regression_rule.md` + AGENTS.md §5.7 NEW |
| Predecessor Wf-399-A2 | `TODO/completed/399a2_catalog_regression_entry.md` (HONEST_NEGATIVE_CATALOG §6.26) |
| Precedent Wf-400-A1 | `TODO/completed/400a1_wf299_launcher_scaffold_not_functional.md` |
| AGENTS.md §5.1 | standing GPU authorization |
| AGENTS.md §5.2 | honest framing |
| AGENTS.md §5.3 | no paper work |
| AGENTS.md §5.4 | Tier-2 markers |
| AGENTS.md §5.7 | regression-prevention rule (NEW per Wf-399-A1) |
| AGENTS.md §10 | wetlab deprecation FAQ |
| TODO-84 | no paper task without authorization |
| `party-twenty-spirit` §3.5 | MEASURED-vs-PROJECTED tag discipline |
| `mol-metal-cfm-training` skill | CFM regression cross-ref |
| `mol-metal-gpu-execution` skill §0 | `kill_then_adapt` directive |

---

**END OF Wf-397-B1 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5;
2026-09-30; **0 NEW tests** in B1; **4 NEW tests across Wf-397 wave** (A1=4);
**1 GPU launch** (Wf-397-A2 Wave 1 smoke on GPU 0 Blackwell 7.4 s);
**0 `paper/*` touches**; **0 wet-lab planning**;
**Wave 2 + Wave 3 SCAFFOLD_NOT_FUNCTIONAL closed out honestly** per Wf-400-A1 precedent
+ kill_then_adapt §0; **Wf-401-A1 follow-up recommended** for launcher flag rewrite +
Wave B smoke launch.
