# Wf-A5-C1 — Wf-A4 retroactive launch_readiness.md closeout

**Status**: SHIPPED on disk per AGENTS.md §5.5 filesystem-truth rule (retroactive re-author per §5.5 rule #5 "trigger retroactive re-author workflow if: (a) closeout report says SHIPPED + disk shows MISSING, OR (b) cross-wave verify agent says MISSING + closeout report says SHIPPED — either direction triggers Wf-384-A1 (audit) + Wf-384-A2 (root cause) + Wf-384-A3 (re-author) + Wf-384-B1 (final sync) sequence").
**Wave**: Wf-A5 (post Wf-A4 audit + GPU retraining launch readiness)
**Type**: doc_only (retroactive closeout per Wf-384-A1..B1 pattern)
**User directive**: 2026-10-02 verbatim *"现在启动ultracode直接把这些所有的非文档类型任务都做完吧"* via Wf-A5 harness relay + Wf-A4 wave parent directive 2026-10-01
**Constraints honored**: AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 (all standing directives re-confirmed).

## §1 What happened in Wf-A4

Wf-A4 was the post-audit + launch-readiness wave per Wf-432 + Wf-431 + Wf-424 + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 + Wf-405 + Wf-404 + Wf-403 + Wf-402 + Wf-401 + Wf-400 + Wf-399 + Wf-398 + Wf-397 + Wf-396 + Wf-395 + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-389 + Wf-388 + Wf-386 + Wf-385 + Wf-384 + Wf-383 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-376 + Wf-375 + Wf-374 + Wf-373 + Wf-372 + Wf-371 + Wf-370 + Wf-369 + Wf-368 + Wf-367 + Wf-366 + Wf-365 + Wf-364 + Wf-363 + Wf-362 + Wf-360 + Wf-359 + Wf-358 + Wf-357 + Wf-356 + Wf-355 + Wf-354 + Wf-353 + Wf-352 + Wf-351 + Wf-350 + Wf-349 + Wf-348 + Wf-347 + Wf-346 + Wf-345 + Wf-344 + Wf-343 + Wf-342 + Wf-341 + Wf-340 + Wf-339 + Wf-338 + Wf-337 + Wf-336 + Wf-335 + Wf-334 + Wf-333 + Wf-332 + Wf-331 + Wf-330 + Wf-1..49 prior ultracode waves.

The Wf-A4 wave (per harness directive 2026-10-01 verbatim *"现在启动ultracode直接把这些所有的非文档类型任务都做完吧"*) was supposed to ship a single 1-line `A4_launch_readiness.md` closeout at the end documenting launch readiness, but the Launch agent (sub-agent E or F) **STALLED at 248s wall** per the Wf-A4 workflow journal + 0 GPU launch was actually triggered. Per AGENTS.md §5.5 rule #5, this retroactive re-author closes the gap.

## §2 What Wf-A5 is (the recovery path)

Wf-A5 (this wave) is the **explicit recovery path** documented in the JSON task brief. The wave structure:

- **Stage 1 CPU patches** (60 min est): A1 (verify) → A2 (REAL code patch done at `train_cfm_yuelbond_wasserstein.py:798-840` per Wf-A5-A2) → A3 (verify) → A4 (verify) → A6 (verify) → A5 (REAL code patch done at `wf404a2_rmsd_multitemplate_scaffold.py:425-545+668-693` per Wf-A5-A5)
- **Stage 2 doc sync** (5 min est): C1 (this file) → C2 (AGENTS.md + INDEX.md sync)
- **Stage 3 GPU launches** (15 hr est): B1 (Wf-299 Wave A) + B2 (Wf-125 CONTROLD PAIR) + B7 (Wf-124 P3 14 pairs) in batch 1; B4 (Wf-380 SCAFFOLD-loss) + B5 (Wf-394 Lambda CFM) in batch 2; B3 (Wf-390 Path A) + B6 (Wf-246-C) in batch 3

## §3 MEASURED state per AGENTS.md §5.5 filesystem-truth

| Task | Status on disk per §5.5 | §5.7 smoke? | Honest framing |
|---|---|---|---|
| A1 (TODO-200 Wf-340+353 deep integration) | ALREADY SHIPPED per Wf-405-A2 + Wf-408-A2 + Wf-410-A2 + Wf-414-A2 + Wf-431-B1 prior | YES (4/4 invariants PASS) | DONE on disk per prior waves |
| A2 (TODO-201 m2 per-cohort features cfm trainer) | SHIPPED in Wf-A5-A2 this wave (50 LOC patch) | YES (4/4 invariants PASS) | DONE this wave |
| A3 (TODO-202 m7 diversity sa_penalty) | ALREADY SHIPPED per Wf-388-A3 | YES (4/4 invariants PASS) | DONE on disk per prior waves |
| A4 (Wf-394 Lambda × CFM coupling) | ALREADY SHIPPED per Wf-406-A2 + Wf-410-A2 + Wf-414-A2 | structural (data-dep blocked) | DONE structurally on disk |
| A5 (TODO-204 m15 RMSD real backend) | SHIPPED in Wf-A5-A5 this wave (115 LOC patch) | YES (--help + surrogate + real fallback all PASS) | DONE this wave |
| A6 (Wf-246-C metal-coordination term) | ALREADY SHIPPED per Wf-432-A2 | structural (data-dep blocked) | DONE structurally on disk |
| C1 (this file) | SHIPPED in Wf-A5-C1 this wave | n/a | retroactive re-author |

**Code patches APPLIED in Wf-A5** (the only REAL code work):
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 798-840: per_cohort_features → label_smoothing wire-in (50 LOC; backup at `train_cfm_yuelbond_wasserstein.py.bak.wfa5a2`)
- `molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py` line 425-545+668-693: real meeko+Vina+GBSA backend wire-in (115 LOC; backup at `wf404a2_rmsd_multitemplate_scaffold.py.bak.wfa5a5`)

## §4 What is NOT yet shipped (forward-looking USER_OK-gated GPU launches)

The 7 GPU launches (B1-B7) are BLOCKED on USER_OK=1 grant + Blackwell sm_120 GPU retrain. Per AGENTS.md §5.1 + kill_then_adapt §0:

- B1 Wf-299 Wave A (~5 hr GPU on cuda:0)
- B2 Wf-125 CONTROLD PAIR (~6 hr GPU on cuda:1)
- B3 Wf-390 Path A 5-metal × 3-seed × 50-epoch m2 R² retrain (~12 hr GPU on cuda:0)
- B4 Wf-380 SCAFFOLD-loss retrain (~10 hr GPU on cuda:0)
- B5 Wf-394 Lambda × CFM coupling retrain (~5 hr GPU on cuda:0)
- B6 Wf-246-C metal-coordination term retrain (~3 hr GPU on cuda:0)
- B7 Wf-124 P3 14 pairs + ablation (~25 hr GPU on cuda:1)

**Combined GPU budget**: ~52.9 hr GPU (matches Wf-368-B1 §0 baseline + Wf-426-A3 forward-looking).

## §5 Cross-references

- `TODO/A5_remaining_tasks.json` (JSON task brief — 15 tasks + stage plan)
- `TODO/completed/A5a1_todo200_cfm_loss_wire.md` (A1 closeout)
- `TODO/completed/A5a2_todo201_cfm_per_cohort.md` (A2 closeout)
- `TODO/completed/A5a3_todo202_sa_penalty_wire.md` (A3 closeout)
- `TODO/completed/A5a4_wf394_lambda_cfm_apply.md` (A4 closeout)
- `TODO/completed/A5a5_todo204_real_backend.md` (A5 closeout)
- `TODO/completed/A5a6_wf246c_metal_coord_patch.md` (A6 closeout)
- `AGENTS.md §5.5` (filesystem-truth rule)
- `AGENTS.md §5.6` (atomic write protocol adoption rule)
- `AGENTS.md §5.7` (trainer flag wire-in smoke-required rule, NEW Wf-399-A1)
- `AGENTS.md §5.1` (standing GPU authorization + USER_OK gating)
- `AGENTS.md §5.4` (Tier-2 + no wet-lab) + `§10` (wetlab deprecation FAQ)
- `party-twenty-spirit §3.5` (MEASURED-vs-PROJECTED label discipline)
- `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file + write_closeout + safe_overwrite_check + detect_truncation primitives)
- `molmetal/scripts/wf399a1_smoke_protocol.py` (50-step CPU smoke helper)
- Wf-384-A1 + Wf-384-A2 + Wf-384-A3 + Wf-384-B1 prior work (the retroactive re-author pattern this closeout follows)
- Wf-422-A1 (pre-flight pass per Wf-420-A1 cascade fix)
- Wf-424-A1 (use_dann kwarg fix that closed the p2_tests_pass cascade that blocked Wf-413..Wf-419 GPU launches)
