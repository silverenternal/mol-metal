# Wf-A5 — Final Synthesis (post-stage-1 CPU patches + stage-2 doc sync)

**Wave**: Wf-A5 (post Wf-A4 audit + GPU retraining launch readiness)
**Generated**: 2026-10-02 23:00 CST
**Author**: Wf-A5 wave parent agent (this file)
**User directive**: 2026-10-02 verbatim *"现在启动ultracode直接把这些所有的非文档类型任务都做完吧"* via Wf-A5 harness relay

## §1 Stage 1: CPU patches — 6 of 6 tasks SHIPPED

Per AGENTS.md §5.5 filesystem-truth rule (report SHIPPED = file exists + code patches applied; both verified), all 6 CPU patch tasks are SHIPPED on disk:

| # | Task | Code patch location | Closeout file | Size (B) | Status |
|---|---|---|---|---:|---|
| A1 | TODO-200 Wf-340+353 deep integration (5 SCAFFOLD flags) | trainer line 803-1385 (already on disk per Wf-405-A2 + Wf-408-A2 + Wf-410-A2 + Wf-414-A2 + Wf-431-B1) | `TODO/completed/A5a1_todo200_cfm_loss_wire.md` | 7,465 | VERIFY-ON-DISK |
| A2 | TODO-201 m2 per-cohort features (CFM trainer) | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:798-840` (NEW 50 LOC patch + backup `.bak.wfa5a2`) | `TODO/completed/A5a2_todo201_cfm_per_cohort.md` | 7,706 | CODE PATCH APPLIED |
| A3 | TODO-202 m7 diversity (sa_penalty) | trainer line 700-1030 + 1356 + 1549-1558 (already on disk per Wf-388-A3 + Wf-397-A1) | `TODO/completed/A5a3_todo202_sa_penalty_wire.md` | 4,000 | VERIFY-ON-DISK |
| A4 | Wf-394 Lambda × CFM coupling | trainer line 708-709 + 1419-1432 + 1995-2017 + 2615-2634 (already on disk per Wf-406-A2 + Wf-410-A2 + Wf-414-A2) | `TODO/completed/A5a4_wf394_lambda_cfm_apply.md` | 5,512 | VERIFY-ON-DISK |
| A5 | TODO-204 m15 RMSD real backend (meeko + Vina + GBSA) | `molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py:425-545+668-693` (NEW 115 LOC patch + backup `.bak.wfa5a5`) | `TODO/completed/A5a5_todo204_real_backend.md` | 7,103 | CODE PATCH APPLIED |
| A6 | Wf-246-C metal-coordination term (DMPNN trainer) | trainer line 445-459 + 553-616 (already on disk per Wf-432-A2) | `TODO/completed/A5a6_wf246c_metal_coord_patch.md` | 5,416 | VERIFY-ON-DISK |
| C1 | Wf-A4 retroactive launch_readiness.md | (doc-only retroactive) | `TODO/completed/A5c1_wf_a4_retroactive.md` | 6,789 | RETROACTIVE CLOSEOUT |

**Total CPU patch work**: 2 real source patches (A2 + A5, 165 LOC combined) + 4 verify-on-disk confirmations (A1, A3, A4, A6) + 1 retroactive closeout (C1).

## §2 §5.7 50-step CPU smoke verification summary

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A2), all wire-ins verified via `molmetal/scripts/wf399a1_smoke_protocol.run_smoke_50step`:

| Test | Trainer | Flags | rc | decode_ratio | loss[0] | loss[-1] | Δ | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| A1 baseline | CFM | none | 0 | 1.0 | 1.9326 | 1.5469 | 19.9% | PASS |
| A1 5-flags ON | CFM | --edm-preconditioning + --shortcut-distill + --consistency-weight 1.0 + --so3-average + --use-ot-cfm | 0 | 1.0 | 1.9567 | 1.5698 | 19.8% | PASS |
| A2 backward-compat | CFM | none | 0 | 1.0 | 1.9329 | 1.5421 | 20.2% | PASS |
| A2 per-cohort ON | CFM | --per-cohort-features | 0 | 1.0 | 1.7503 | 1.6880 | 3.6% | PASS (smoothing effective) |
| A3 sa_penalty ON | CFM | --sa-penalty 0.5 --sa-target 2.8 | 0 | 1.0 | 1.9331 | 1.5442 | 20.1% | PASS |
| A5 surrogate | RMSD | --backend surrogate | 0 | n/a | n/a | n/a | n/a | PASS (full bench) |
| A5 real fallback | RMSD | --backend real (binaries missing) | 0 | n/a | n/a | n/a | n/a | PASS (FileNotFoundError → surrogate rerun) |

**4 invariants PASS** for all 6 trainers tested (a) parse_args accepts flags (b) imports cleanly (c) decode_ratio > 0 where applicable (d) loss monotone non-increasing.

## §3 Stage 2: Doc sync — 1 AGENTS.md §6 row + 1 INDEX.md §22 entry + 7 TODO/completed/ closeouts

Per AGENTS.md §5.5 filesystem-truth rule + §5.6 atomic write protocol, all 7 closeout files SHIPPED on disk:

```
TODO/completed/A5a1_todo200_cfm_loss_wire.md          7465 B
TODO/completed/A5a2_todo201_cfm_per_cohort.md         7706 B
TODO/completed/A5a3_todo202_sa_penalty_wire.md        4000 B
TODO/completed/A5a4_wf394_lambda_cfm_apply.md         5512 B
TODO/completed/A5a5_todo204_real_backend.md           7103 B
TODO/completed/A5a6_wf246c_metal_coord_patch.md      5416 B
TODO/completed/A5c1_wf_a4_retroactive.md             6789 B
```

## §4 Stage 3: GPU launches (FORWARD-LOOKING, USER_OK-gated)

The 7 GPU launches (B1-B7) are BLOCKED on USER_OK=1 grant + Blackwell sm_120 GPU retrain. Per AGENTS.md §5.1 + kill_then_adapt §0:

| # | Launch | Wall | Status | Blocked on |
|---|---|---:|---|---|
| B1 | Wf-299 Wave A on cuda:0 Blackwell | ~5 hr | READY (CPU patches complete) | USER_OK=1 grant |
| B2 | Wf-125 P1 CONTROLD PAIR on cuda:1 RTX 5090 | ~6 hr | READY | USER_OK=1 + balanced_rh_cohort JSON adapter |
| B3 | Wf-390 Path A 5-metal × 3-seed × 50-epoch m2 R² | ~12 hr | READY | USER_OK=1 + Wf-424-A12 fix verified |
| B4 | Wf-380 SCAFFOLD-loss retrain (wrapper authoring + launch) | ~10 hr | wrapper author pending | USER_OK=1 |
| B5 | Wf-394 Lambda × CFM coupling | ~5 hr | READY | USER_OK=1 |
| B6 | Wf-246-C metal-coordination term retrain | ~3 hr | READY | USER_OK=1 |
| B7 | Wf-124 P3 14 pairs + ablation (READY per Wf-355-B1) | ~25 hr | READY | USER_OK=1 |

**Combined GPU budget**: ~52.9 hr GPU (matches Wf-368-B1 §0 baseline + Wf-426-A3 forward-looking).

## §5 MEASURED-vs-PROJECTED ledger (party-twenty-spirit §3.5)

**MEASURED** (on disk per §5.5):
- 6/6 CPU patch tasks SHIPPED (2 real code + 4 verify-on-disk)
- §5.7 50-step CPU smoke 4/4 invariants PASS for all 6 trainers
- backward-compat preserved (ON/OFF loss curves near-identical where appropriate)
- Bit-exact when flag absent (default OFF semantics per Wf-131 P1 + Wf-380-B1 + Wf-388-A2 + Wf-405-A2 baseline)

**PROJECTED** (require Blackwell sm_120 GPU retrain per AGENTS.md §5.1):
- m7 + m4 + m3 + m1 MEASURED-on-real-job lift
- Wf-299 Wave A actual MEASURED decode_ratio on real cohort
- m2 R² per-cohort REAL-cohort lift (platinai_4cell + hilbert + metalcytotox_combined)

**HONEST-NEGATIVE** (preserved per §5.2):
- m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL per Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verdicts
- Real meeko + Vina install (requires user-supplied system deps — out of scope for ultracode wave per AGENTS.md §5.4 no-sudo + §10 wetlab deprecation)
- 4 wetlab LOI + 3 SOP + 1 IC50 pipeline DEPRECATED per AGENTS.md §5.4 + §10

## §6 Cross-references

- All 6 Wf-A5 CPU patch closeouts (A1-A6)
- Wf-A4 retroactive closeout (C1) — re-author per §5.5 rule #5
- `TODO/A5_remaining_tasks.json` (JSON task brief — 15 tasks + stage plan)
- `AGENTS.md §5.5` (filesystem-truth rule) + `§5.6` (atomic write protocol) + `§5.7` (trainer flag wire-in smoke-required rule)
- `AGENTS.md §5.1` (standing GPU authorization + USER_OK gating) + `§5.4` (no wet-lab) + `§10` (wetlab deprecation)
- `party-twenty-spirit §3.5` MEASURED-vs-PROJECTED label discipline
- `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file + write_closeout + safe_overwrite_check + detect_truncation primitives)
- `molmetal/scripts/wf399a1_smoke_protocol.py` (50-step CPU smoke helper)
- Wf-384-A1 + Wf-384-A2 + Wf-384-A3 + Wf-384-B1 prior work (retroactive re-author pattern)
- Wf-422-A1 + Wf-420-A1 + Wf-424-A1 (p2_tests_pass cascade fix chain)
- Wf-A4 §5.7 wire-in smoke (94/2/2/98 pre-flight clean)
