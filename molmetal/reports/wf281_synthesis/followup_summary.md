# Wf-281E — Follow-up Synthesis (Aggregation of Wf-281A/B/C/D)

**Date**: 2026-09-29
**Wf**: Wf-281E (aggregate Wf-281A-D follow-up outcomes)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative + `mol-metal-paper-writing`
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 (Wf-281A / B / C / D) | — |
| **G2_n_completed_with_on_disk_output** | **3** (Wf-281A ✅ + Wf-281C ✅ + Wf-281D ✅) | ⚠️ 3/4 — Wf-281B DID NOT LAUNCH this session |
| **G3_n_failed_or_undefined** | **1** (Wf-281B TODO-207 §10 cross-ref audit NOT LAUNCHED) | ⚠️ NOT MET — 1 wave missing on disk |
| **G4_critical_honest_negatives_remaining** | **6 items** (see §3) | ⚠️ NOT MET — 6 critical honest-negatives inherited from upstream Wf-278 + Wf-280 |
| **G5_next_priorities** | 6 items (see §4) — **4 are parallelizable CPU-only ultracode repair tasks** | — |
| **G6_report_path** | `molmetal/reports/wf281_synthesis/followup_summary.md` | PASS — this file |

**Honest framing note (party-twenty-spirit §3.5)**: Wf-281 produced **3 MEASURED deliverables** (Wf-281A recovery report, Wf-281C G6 readiness, Wf-281D 188 paper-edit awareness) + **1 missing wave** (Wf-281B §10 cross-ref audit did not launch). All 3 delivered waves are honest-negative OR awareness-only — **zero paper edits performed**, **zero GPU hours consumed**, **zero fabrication**. The Wf-280 4-of-4 streak is BROKEN at 3-of-4 by Wf-281B no-launch; the underlying honest-negative ledger (TODO-180 n=14, G6 30 UNVERIFIED, 188 paper edits BLOCKED on TODO-84, 15/17 Cf-258 waves PENDING) is unchanged.

---

## §1 Per-Agent Summary

### §1.1 Wf-281A — TODO-180 G4 Ru seed=0 Recovery — **MEASURED HONEST-NEGATIVE**

**File on disk**: `molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md`

**Findings**:
- `pairs_before_recovery`: 14 (Ru seed=0 missing from `results/wf124_p3_remaining_metals_g4/`)
- `pairs_after_recovery`: **14 — RECOVERY REFUSED on data-integrity grounds**
- 2 candidate Wf-124 P3 Ru seed=0 files inspected, both rejected:
  - `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` — n_train=2509 (single-metal early run), INCOMPATIBLE with TODO-180 G4 5-metal cohort (n_train=12471)
  - `results/wf270a_smoke/ru/per_seed_metrics_seed0.json` — smoke config (h=64, d=2), not the Wf-124 P3 5-metal full config (h=300, d=3)
- Valid source: `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/` (G5 live, epoch 47/50 at write time, val_auc=0.8177)
- **Action**: NO copy/link performed. Aggregate test_auc remains 0.7209 ± 0.0458 (n=14, unchanged). Gap preserved as honest-negative per `party-twenty-spirit` §3.5.

**Verdict**: ✅ MEASURED HONEST-NEGATIVE — gap preserved, no fabrication.

### §1.2 Wf-281B — TODO-207 §10 Cross-Ref Audit — **NOT LAUNCHED**

**File on disk**: **NONE** (no `wf281b_*` directory exists)

**Findings**:
- Wf-281B was dispatched as the TODO-207 §10 cross-ref audit task per the harness relayed request. Did not launch in this session.
- Per Wf-281D awareness doc (`TODO/completed/188_paper_edit_awareness_2026-09-29.md` §0): "Wf-281B did not launch in this session" — explicit acknowledgment by the Wf-281D owner.
- The §10 cross-ref audit scope (extending the Wf-277D-style cite-target audit to TODO-207 §10 sign-off block) was first recommended in Wf-278E §4 item 5 and remains un-launched.

**Verdict**: ❌ NOT LAUNCHED — gap preserved. Re-launch as Wf-281B-retry (or fold into Wf-283 §10 audit extension).

### §1.3 Wf-281C — G6 Launch Readiness — **MEASURED SUCCESS (LAUNCH DEFERRED)**

**File on disk**: `molmetal/reports/wf281c_g6_readiness/readiness_report.md`

**Findings**:
- Wf-280A `--no-enforce-budget` fix VERIFIED: G6 wrapper matches G5 pattern (policy-level directive, not python CLI flag; `train_dmpnn_multitask_wasserstein.py` lacks the flag entirely).
- Dry-run PASS: 60 (config, metal, seed) tuples enumerated, no GPU work executed.
- G6 wrapper has 30 CONFIRMED (--single-task + --wasserstein-weight 0.0) + 30 UNVERIFIED (--no-kendall + --no-dann) tuples.
- **Launch verdict: NOT YET — wait for G5 to free GPU-1** (~4.5 hr remaining after Wave A Ru seed=0 epoch 50 completes).
- Recommendation: sequential launch after G5; OR interleave CONFIRMED-only (30 runs) on GPU-0 if idle.

**Verdict**: ✅ MEASURED — wrapper ready, launch deferred until G5 completes.

### §1.4 Wf-281D — 188 Paper-Edit Awareness Doc — **MEASURED SUCCESS (AWARENESS ONLY)**

**File on disk**: `TODO/completed/188_paper_edit_awareness_2026-09-29.md`

**Findings**:
- 5 + 1 (Wf-284 umbrella) PROPOSED `paper/sections/04_results.tex` + `05_ablation.tex` edits documented:
  - C1: 188 §4.1 (m14 CFM decode_ratio) reverse-cite → 207 §9.6.3 (from Wf-277D)
  - C2: 188 §4.4 (m7 Tanimoto diversity) reverse-cite → 207 §9.6.2 (from Wf-277D)
  - C3: 188 §5.2 m17 DO NOT claim wording tighten (from Wf-278D)
  - C4: 188 §4.5.1 G4 n=14→n=15 update (contingent on G5 Ru seed=0 completion)
  - C5: 188 §5 G6 per-innovation ablation paragraph (contingent on G6 launch)
  - C6: Wf-284 batch-apply all 207 §9.x → 188 §x.y reverse-cites (umbrella)
- **NO `paper/*.tex` writes performed**. ALL 6 candidates BLOCKED per TODO-84 ACTIVE rule (user instruction 2026-09-26 13:00 CST) + wet-lab LOI gate (all 3 LOIs DRAFT, not sent).

**Verdict**: ✅ MEASURED AWARENESS — 6 candidates documented, 0 paper edits applied. Gate = TODO-84 + wet-lab LOI.

---

## §2 Aggregate Counts

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **3** (Wf-281A ✅ + Wf-281C ✅ + Wf-281D ✅) |
| Upstream waves MISSING (no on-disk report) | **1** (Wf-281B TODO-207 §10 cross-ref audit) |
| MEASURED HONEST-NEGATIVES delivered | **2** (Wf-281A recovery refused; Wf-281D awareness-only) |
| MEASURED SHIPPED deliverables | **1** (Wf-281C G6 readiness — launch deferred) |
| Total work MEASURED-completed | **3 of 4 attempted** = 75% (down from Wf-280's 100%) |
| Total work NOT LAUNCHED | **1** (Wf-281B) |
| Total work failed (hard) | **0** |
| Pipeline regression class occurrence count | **1** (Wf-281B no-launch) — Wf-280's 4-of-4 streak BROKEN |

---

## §3 Critical Honest-Negatives Remaining

| # | Item | Severity | Evidence | Owner |
|---|---|---|---|---|
| **HN-1** | **TODO-180 G4 still n=14, Ru seed=0 missing** (carry-over from Wf-278A) | **HIGH** (TODO-180 ship) | `molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md` §TL;DR + `results/wf124_p3_remaining_metals_g4/training_summary.json` | ultracode (wait for G5 Ru seed=0 epoch 50) |
| **HN-2** | **G6 wrapper never launched + `--no-kendall` / `--no-dann` flags UNVERIFIED** (carry-over from Wf-280D §HN-2) | **HIGH** (TODO-181 ship) | `molmetal/reports/wf281c_g6_readiness/readiness_report.md` §3 + `train_dmpnn_multitask_wasserstein.py --help` lacks the 2 flags | ultracode (Python CLI author) |
| **HN-3** | **Wf-281B §10 cross-ref audit NOT LAUNCHED** (NEW from Wf-281) | **MEDIUM** (207 §10 doc gap) | No `wf281b_*` directory on disk; Wf-281D §0 acknowledges | Hugo Lee (re-dispatch Wf-281B-retry or fold into Wf-283) |
| **HN-4** | **188 paper edits BLOCKED on TODO-84 + wet-lab LOI gate** (NEW from Wf-281D) | **HIGH** (paper §3/§4/§5 ship) | `TODO/completed/188_paper_edit_awareness_2026-09-29.md` §6 + all 3 wet-lab LOIs DRAFT not sent | Hugo Lee (user-coordinated; not autonomous) |
| **HN-5** | **15/17 Cf-258 waves PENDING** (carry-over from Wf-280D) | **HIGH** (Cf-258 ship) | `molmetal/reports/wf280d_c258_waves/audit_report.md` §2 (11.76% complete) | ultracode (GPU runner) |
| **HN-6** | **PlatinAI pIC50 labels SYNTHETIC, not real** (carry-over from Wf-278B) | **HIGH** (TODO-119 P1.3 / TODO-98 L12) | Cloudflare JS challenge on Rusanov 2025 S4-S5 | Hugo Lee (browser fallback: Selenium or manual PDF harvest) |

**Note**: HN-1, HN-2, HN-5, HN-6 are carried forward from Wf-280E §3. HN-3 is NEW from Wf-281B no-launch. HN-4 is NEW from Wf-281D awareness doc (was implicit in Wf-280E as part of "Wf-284 batch-apply" but is now explicitly documented with 6 candidate edits).

---

## §4 Recommended Next Follow-Up Priorities (ordered by ROI × urgency × parallelizability)

### P1.1 — Add `--no-kendall` and `--no-dann` flags to `train_dmpnn_multitask_wasserstein.py` (~2 hr CPU + smoke)

**Evidence**: HN-2 + Wf-281C §3 (UNVERIFIED tuples).
**Action**: Add 2 argparse options + 2 unit tests. CLOSES Wf-281C honest caveat. UNBLOCKS 30 G6 UNVERIFIED tuples.
**Owner**: ultracode (Python CLI author). **PARALLELIZABLE**: CPU-only.

### P1.2 — Wait for G5 Ru seed=0 epoch 50 → re-run `wf278a_g4_aggregate.py` (~3 min wait + 30s CPU)

**Evidence**: HN-1 + Wf-281A §recommended next-step.
**Action**: Poll `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` for epoch 50 completion; copy per_seed_metrics_seed0.json; re-aggregate.
**Owner**: ultracode (read + copy + rerun aggregator). **NOT parallelizable**: depends on G5 runtime.

### P1.3 — Wire Wf-262 trio into Wf-124 P3 launchers (~2 hr CPU + smoke verify)

**Evidence**: Wf-279D §6 + carry-over from Wf-280E P1.1.
**Action**: Edit `scripts/launch_wf124_p3_g4.sh`, `g5.sh`, `g6.sh` to insert pre_flight_check.py + per_wave_checklist.py + wave_budget_tracker.py hooks.
**Owner**: ultracode (Bash). **PARALLELIZABLE**: CPU-only.

### P1.4 — Re-dispatch Wf-281B §10 cross-ref audit (~1 hr CPU + read)

**Evidence**: HN-3 + Wf-278E §4 item 5.
**Action**: Walk TODO-207 §10 (sign-off block + §10.1 per-wave sub-checklist + §10.2 P0 BLOCKER user OK gate + §10.3 auto-verify JSON hash) and identify which sub-sections need cross-refs to 188 §x.y / TODO-180 / TODO-181. Emit `wf281b_xref_audit/audit_report.md`.
**Owner**: ultracode (read + write). **PARALLELIZABLE**: CPU-only.

### P2.1 — PlatinAI real-label loader (~4 hr CPU; Selenium or manual harvest)

**Evidence**: HN-6 + Wf-278B §honest-negative.
**Action**: Bypass Cloudflare JS challenge on Rusanov 2025 S4-S5; produce `platinai_real.csv` for 4-cell pipeline.
**Owner**: Hugo Lee (browser fallback) + ultracode (CSV builder). **NOT parallelizable** for Selenium part.

### P2.2 — Apply 188 paper edits C1..C6 ONLY after TODO-84 lifted + wet-lab LOI signed (gated)

**Evidence**: HN-4 + Wf-281D §8 path.
**Action**: User-coordinated. NO autonomous execution. The path is: (a) send 3 LOI cold-emails, (b) PI counter-signs, (c) wet-lab executes, (d) user lifts TODO-84, (e) apply edits per `mol-metal-paper-writing` §4 verbatim template.
**Owner**: Hugo Lee (gate-keeper). **NOT parallelizable**: depends on user authorization.

### P3.1 — After G5 + G6 launch: re-run Wf-264 panel on Wf-269C CFM-generated SMILES (~6 hr GPU + CPU)

**Evidence**: Carry-over from Wf-280E P3.1 + Wf-264 §4 honest caveat.
**Action**: CFM decode on Wf-269C stratified cohort; pass actual generated SMILES through SA+QED+Vina panel.
**Owner**: ultracode (GPU + Vina). **NOT parallelizable**: needs GPU-0 Blackwell + Vina CPU.

---

## §5 Parallelizability Matrix

| Priority | Wall | GPU? | Parallelizable? | Depends on |
|---|---|---|---|---|
| **P1.1** (--no-kendall/--no-dann flags) | ~2 hr | NO | **YES** — CPU only, can run with P1.3 + P1.4 | train_dmpnn_multitask_wasserstein.py source |
| **P1.2** (wait for G5 + re-aggregate) | ~3 min wait + 30s | NO (after wait) | **YES** — CPU only post-wait | G5 Ru seed=0 epoch 50 |
| **P1.3** (wire Wf-262 trio) | ~2 hr | NO | **YES** — CPU only, can run with P1.1 + P1.4 | Wf-262 trio (already shipped) |
| **P1.4** (Wf-281B retry §10 audit) | ~1 hr | NO | **YES** — CPU only, can run with P1.1 + P1.3 | TODO-207 §10 + 188 §x.y + Wf-277D |
| **P2.1** (PlatinAI real loader) | ~4 hr | NO | NO — browser fallback not parallel | Rusanov 2025 S4-S5 + Cloudflare bypass |
| **P2.2** (188 paper edits) | gated | NO | NO — user-coordinated | TODO-84 lift + 3 wet-lab LOI signed |
| **P3.1** (Wf-264 panel rerun) | ~6 hr | YES | NO — needs GPU-0 | Wf-269C CFM checkpoint + G6 completion |

**Recommended ultracode parallelization** (4 agents in parallel, all CPU-only):
1. **ultracode-281E-A**: P1.1 (--no-kendall/--no-dann flags, ~2 hr)
2. **ultracode-281E-B**: P1.3 (wire Wf-262 trio, ~2 hr)
3. **ultracode-281E-C**: P1.4 (Wf-281B retry §10 cross-ref audit, ~1 hr)
4. **ultracode-281E-D**: P1.2 (poll G5 + re-aggregate, ~3 min wait + 30s)

**Total wall** (4 agents parallel): **~2 hr** (limited by P1.1 + P1.3)
**Total wall** (sequential): **~5 hr** (P1.1 + P1.3 + P1.4 + P1.2)
**Speedup**: 2.5× via parallel dispatch.

**GPU-only follow-ups** (post-CPU, sequential after G5 + G6):
- G6 launch (60 runs, ~7.5 hr on GPU-1 after G5 completes)
- P3.1 (Wf-264 panel rerun, ~6 hr on GPU-0 Blackwell)

---

## §6 Files Referenced

- `/home/hugo/codes/mol-metal/molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md` (Wf-281A, MEASURED HONEST-NEGATIVE)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf281c_g6_readiness/readiness_report.md` (Wf-281C, MEASURED — launch deferred)
- `/home/hugo/codes/mol-metal/TODO/completed/188_paper_edit_awareness_2026-09-29.md` (Wf-281D, MEASURED AWARENESS — 0 paper edits)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280_synthesis/followup_summary.md` (Wf-280E, prior synthesis)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf278_synthesis/followup_summary.md` (Wf-278E, prior synthesis)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280d_c258_waves/audit_report.md` (Cf-258 17-wave audit, 15/17 PENDING)
- `/home/hugo/codes/mol-metal/TODO/completed/207_unified_pre_flight_checklist_v2.md` §10 (sign-off block, §10.1 per-wave sub-checklist)
- `/home/hugo/codes/mol-metal/TODO/pending/84_no_paper_task_without_authorization.md` (TODO-84 ACTIVE rule)
- `/home/hugo/codes/mol-metal/TODO/pending/wetlab_loi_ornl_protein_metal.md` + `wetlab_loi_pku_cisplatin_ic50.md` + `wetlab_loi_xiehe_nci60.md` (3 wet-lab LOIs, all DRAFT not sent)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (G5 in-flight, epoch 47/50)
- `/home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json` (n=14, gap preserved)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wf278a_g4_aggregate.py` (n=14 → n=15 re-aggregator post-G5)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` + `launch_wf124_p3_g6.sh` (G5/G6 wrappers, --no-enforce-budget at policy layer)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (missing --no-kendall/--no-dann CLI flags)

---

**End of Wf-281E synthesis.**
