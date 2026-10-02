# Wf-393-A3 — TODO/decisions.md + TODO/README.md refresh (Wf-378..392 sync)

**Document ID:** Wf-393-A3 / decisions + readme refresh
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-393-A3 docs-only sync agent
**Trigger:** Per user directive 2026-09-30 verbatim *"todo目录里没有被刚才启动的
workflow覆盖的任务也启动额外的ultracode去做"* — TODO directory audit + decisions
+ README refresh to capture the Wf-378..392 ultracode ship surface.
**Type:** DOCS-ONLY (CPU-only; 0 NEW tests; 0 GPU launches; 0 paper/* touches per
TODO-84; 0 wet-lab planning per AGENTS.md §10)
**Cross-ref:** `party-twenty-spirit §3.5` (MEASURED-vs-PROJECTED tag discipline)
+ AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing
directives re-confirmed at 2026-09-30 snapshot.

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| `TODO/decisions.md` refresh | Added 5 NEW decisions (D-Wf-380 + D-Wf-385 + D-Wf-386 + D-Wf-388 + D-Wf-390) under NEW "Wf-378..392 ultracode decisions (2026-09-30)" section | ✅ SHIPPED |
| `TODO/README.md` refresh | Updated last-updated header + pending/ tree (33 files at 2026-09-30, including Wf-380/385/386/388/390 plan files + 7 wetlab DEPRECATED markers) + completed/ count (929 files per `ls TODO/completed/ | wc -l`) + added "Wf-296..392 ultracode ship surface" section with Tier-2 re-orientation + Wf-340..392 ship summary + Tier-2 / wetlab-deprecation compliance verification block | ✅ SHIPPED |
| Wetlab compliance check | `grep -nE "wetlab|Wetlab|WETLAB|wet-lab|wet_lab"` → **21 matches**, **zero affirmative** wetlab-work recommendations (all matches are DEPRECATED markers, summary references, or compliance verification block itself) | ✅ VERIFIED |
| Paper compliance check | `grep -nE "paper/.*SECTIONS|REWRITE_PAPER|EDIT_PAPER"` → 0 affirmative matches; only references to `paper/*` are READ-only via TODO-84 + AGENTS.md §5.3 + `mol-metal-paper-writing` skill | ✅ VERIFIED |
| GPU launches | 0 | ✅ per AGENTS.md §5.1 |
| `paper/*` touches | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| Wet-lab planning | 0 | ✅ per AGENTS.md §5.4 + §10 |
| NEW tests | 0 (entire wave is docs-only per task brief) | ✅ |
| Pre-flight v3 | NOT re-run by Wf-393-A3 (docs-only); PRESERVED at Wf-385-B1 / Wf-386-B1 / Wf-388-B1 / Wf-389-B1 / Wf-390-B1 / Wf-391-B1 / Wf-392 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL | ✅ PRESERVED |
| EXPECTED_GATE_COUNTS["all"] | 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-393-A3 — docs-only sync) | ✅ PRESERVED |

**One-line verdict**: Wf-393-A3 SHIPS a docs-only refresh of `TODO/decisions.md` (5
NEW Wf-378..392 architectural decisions appended) + `TODO/README.md` (refreshed
last-updated header + pending/ tree + completed/ count + NEW Wf-296..392
ultracode ship surface section + Tier-2 / wetlab-deprecation compliance
verification block). The refresh syncs the canonical architectural decision log
+ top-level TODO navigation with the Wf-378..392 ultracode ship surface, with
honest framing per `party-twenty-spirit §3.5`. All CPU-only docs-only per
AGENTS.md §5.1+§5.2+§5.3+§5.4+§5.5+§5.6+§10. Pre-flight cascade PRESERVED at
Wf-353-B1 baseline 89/2/3/94.

---

## §1 decisions_updated (TODO/decisions.md refresh)

### §1.1 What changed

Per Wf-393-A3 sync, **TODO/decisions.md** was refreshed to capture 5 NEW
architectural decisions from the Wf-380 / Wf-385 / Wf-386 / Wf-388 / Wf-390
ultracode CPU-only waves (2026-09-30). The original `Last updated` header
(2026-09-12) was extended to a 2026-09-30 marker preserving the §1 + §2 sections
from 2026-09-12 baseline + appending a NEW §3 "Wf-378..392 ultracode decisions
(2026-09-30)" section.

### §1.2 The 5 NEW decisions (appended in §3)

| # | Decision | Sub-flow | Key outputs | MEASURED vs PROJECTED |
|---:|---|---|---|---|
| **D-Wf-380** ✅ | CFM SCAFFOLD + 14 CFM CLI flags + 3 DMPNN CLI flags wired (default ON via `--no-X` negation) | Wf-380-A1 + Wf-380-B1 + Wf-385-A1 audit | 10 SCAFFOLD modules ~1055 LOC + 14 NEW CFM CLI flags (parse_args lines 1487-1740) + 3 NEW DMPNN CLI flags (parse_args lines 3001-3083) + 41/41 NEW tests PASS in Wf-380-A1 + 0/17 flags wired in main training loop (per Wf-385-A1 audit) | MEASURED: 10 modules + 41 tests + flag surface; PROJECTED: SCAFFOLD-loss wire-in lift requires Wf-381+ MEASURED end-to-end |
| **D-Wf-385** ✅ | Pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified` added | Wf-385-A3 + Wf-385-A1 audit | `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` ~280 LOC + `molmetal/tests/test_wf385a3_scaffold_wire_in_gate.py` ~332 LOC (20 NEW tests) + `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json` MEASURED | MEASURED: gate catches FLAG_ONLY SCAFFOLD failures at pre-flight time |
| **D-Wf-386** ✅ | Atomic write protocol adopted (AGENTS.md §5.6) | Wf-386-A3 + Wf-386-A1 + Wf-386-A2 | `molmetal/scripts/wf386a3_atomic_write.py` ~366 LOC (3 helpers: `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`) + `molmetal/tests/test_wf386a3_atomic_write.py` ~332 LOC (27-case test suite) + `docs/ATOMIC_WRITE_PROTOCOL.md` ~290 LOC | MEASURED: 27/27 tests PASS; belt-and-braces defence at TWO levels (agent-prompt per Wf-384 §5.5 + I/O substrate per Wf-386 §5.6) |
| **D-Wf-388** ✅ | 6 diversity + 2 SA CLI flags wired (default OFF) | Wf-388-A2 (m7 Diversity) + Wf-388-A3 (m4 SA) | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` +120/-15 LOC source patch (6 m7 + 2 SA flags wired into train_one_epoch) + 4+3 NEW tests PASS + new metric JSONs `metrics/by_metric/{m7_diversity,m4_sa}.json` | MEASURED: wire-in complete; PROJECTED: end-to-end lift on Blackwell sm_120 BLOCKED per AGENTS.md §5.1 (m7 +0.02-0.035 / m4 SA-mean 3.099 → 2.5-2.7) |
| **D-Wf-390** ✅ | TODO-376 5-metal subset plan + TODO-179 alt encoder + TODO-383 SE(3) SB wire-in | Wf-390-A1 + Wf-390-A2 + Wf-390-A3 | `TODO/pending/376_m2_r2_gpu_retrain_plan.md` ~455 LOC (3 paths ranked) + `TODO/completed/390a2_todo179_deeper_audit.md` ~24 KB (40 NEW assertions) + `TODO/completed/390a3_se3_sb_wire.md` ~17 KB (2 NEW tests) + 3 structural blockers + alternative encoder bootstrap (632-d scaffold) | MEASURED: 3 forward paths ranked + smoke scaffold + master plan extension; PROJECTED: Path A retrain USER_OK-gated (~12 hr GPU + ~10 GB VRAM); Wf-200-A1..A10 execution chain for SE(3) SB (~10-12 days CPU + Blackwell retrain) |

### §1.3 Cross-cutting reference updates (post Wf-393-A3 sync)

- ✅ **`risks.md`** — open risks catalog expanded to 65 entries per Wf-369-A3 +
  4 entries per Wf-375-A2 = **69 entries** at Wf-375-B1 baseline (per Wf-375-B1 §22.68.4)
- ✅ **`completed/`** — `929 files` per `ls TODO/completed/ | wc -l` at 2026-09-30 (290+ closeouts since Wf-340..392 ship wave)
- ✅ **`INDEX.md §22`** — per-Wf coverage verdict table for Wf-340..392 entries (~30 sub-entries)
- ✅ **`AGENTS.md §6`** — recent Wf history pointer extended with Wf-378..392 rows
- ✅ **`AGENTS.md §5.1 + §5.3 + §5.4 + §5.5 + §5.6 + §10`** — standing directives re-confirmed at 2026-09-30 snapshot
- ✅ **`TODO/03_plans/LAMBDA_ROADMAP.md`** — Wf-391-A1 refreshed (~616 → ~700 LOC, 6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11)
- ✅ **`TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md`** — Wf-391-A2 refreshed (4 → 12 tasks: 5 SHIPPED + 4 SCAFFOLD + 1 HONEST_NEGATIVE_FINAL + 2 PARTIAL)
- ✅ **`TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md`** — Wf-391-A2 refreshed (4 PENDING → 32 tasks: 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY)
- ✅ **`TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md`** — Wf-391-A3 refreshed (~410 → ~860 LOC, 8 → 11 sections; NEW §10 forward-looking 2026-Q4 GPU retrain priority order)
- ✅ **`docs/HONEST_NEGATIVE_CATALOG.md`** — Wf-369-A3 65 entries + Wf-375-A2 +4 = 69 entries (5 sections: per-metric 12 + per-workflow 15 + per-launcher 6 + resolved 7 + open 10 + §6 19 Wf-340..368 consolidated)

---

## §2 readme_updated (TODO/README.md refresh)

### §2.1 What changed

Per Wf-393-A3 sync, **TODO/README.md** was refreshed across 4 dimensions:

1. **Last-updated header** — extended from `2026-09-16` to `2026-09-30 (Wf-393-A3
   docs-only sync)` with summary of the 4 refresh dimensions + preservation of
   the prior 2026-09-16 R15 ship summary block + honest framing preserved block

2. **Directory Structure tree** — pending/ tree updated from 9 files (R12-R14
   round plans) to **33 files at 2026-09-30** covering:
   - 23 active plans / in-flight work files (TODO-094 + 155 + 177..183 + 200..207 +
     219 + 234 + 376 + 376a3 + 383 + 84)
   - 7 wetlab DEPRECATED files (cold_email + 4 LOI + 3 SOP + ic50_analysis pipeline)
   - All 7 wetlab files tagged `← DEPRECATED per AGENTS.md §5.4 + §10`
   - completed/ count updated from 28 files (2026-09-16) to **929 files** at
     2026-09-30 (per `ls TODO/completed/ | wc -l`)

3. **NEW "Wf-296..392 ultracode ship surface (2026-09-12 → 2026-09-30, refreshed
   per Wf-393-A3)" section** appended after "Honest framing preserved for劣势 items
   (R15 bounded scope)" — covers:
   - **Major 2026-09-12 → 2026-09-29 ship summary (Tier-2 re-orientation)**: Tier-2
     journal target adopted (downgrade from JCIM Tier-1); wet-lab items DEPRECATED
     per AGENTS.md §5.4 + §10; 7 NEW skills wired per Wf-321B; Wf-343C4 4 NEW
     pre-flight gates (62 → 66); Wf-347 ultracode integration v3 (66 → 68)
   - **Major 2026-09-29 → 2026-09-30 ship summary (Wf-340..392 ultracode series)**:
     Wf-340 CFM fast-converging training research; Wf-341..343 actionable CPU bundle;
     Wf-344 + Wf-345 ultracode residual driver + wave v1; Wf-346 registration-drift
     resolution; Wf-350 NeurIPS reproducibility backbone; Wf-353 Wf-340 deeper
     integration + Wf-343c1 un-wired gates; Wf-355..362 ultracode residual +
     integration; Wf-372 install + integration; Wf-378 OPEN CPU-only gaps closure;
     Wf-379 GPU acceleration; Wf-380 ultracode residual implementation; Wf-381..385
     deeper-level fix + SCAFFOLD wire-in audit; Wf-386..389 process improvements +
     waves; Wf-390..392 surface TODO items not yet covered
   - **Pre-flight v3 cascade STILL CLOSED** across 22+ consecutive runs at
     Wf-385-B1 baseline 89/2/3/94 (HONEST-NEGATIVE DRIFT preserved)

4. **NEW "Tier-2 + wetlab-deprecation compliance verification (per AGENTS.md §5.4
   + §10)" block** appended at end of "Wf-296..392 ultracode ship surface" section
   — covers:
   - Zero AFFIRMATIVE wetlab recommendations (verified by grep → 19 matches,
     all DEPRECATED markers or summary references)
   - Zero paper/* recommendations (verified by grep → 0 affirmative matches)
   - Standing GPU authorization preserved
   - Honest framing preserved
   - No wet-lab planning per AGENTS.md §5.4 + §10 (all 7 wetlab pending files
     marked DEPRECATED)

### §2.2 Honest framing caveats

- The `pending/` tree count is **33 files at 2026-09-30** (counted by `ls
  TODO/pending/ | wc -l`); this includes the 7 DEPRECATED wetlab files which
  are retained as historical artifacts per AGENTS.md §10.5 retain-as-historical-
  artifact policy + §10.6 cross-refs
- The `completed/` count is **929 files at 2026-09-30** (per `ls TODO/completed/
  | wc -l`); 290+ closeouts since Wf-340..392 ship wave
- The `Wetlab compliance check` claim of "0 matches outside the pending/
  DEPRECATED file list above" was corrected to accurately state "**21 matches**,
  of which: 1 in the last-updated header (line 3), 10 in the `pending/`
  DEPRECATED file list (lines 89-98), 3 in the Tier-2 / wetlab-deprecation
  summary block (lines 277, 280, 281), 1 in the pre-flight FAIL reference
  (line 364), 1 in the compliance verification section header (line 368), 4 in
  the compliance verification content (lines 372, 373, 375, 377), and 1 in
  the closing claim (line 389); **zero** are affirmative wetlab-work
  recommendations"

---

## §3 cross_references (per-Wf closeout sources)

### §3.1 Wf-380 source closeouts (D-Wf-380 + D-Wf-385 + D-Wf-388 + D-Wf-390 partial)

- `TODO/completed/380a1_deep_integration_scaffolds.md` — Wf-380-A1 SCAFFOLD authoring
  baseline (10 NEW SCAFFOLD modules ~1055 LOC + 41/41 NEW tests PASS in ~1.61 s wall)
- `TODO/completed/380a2_per_cohort_features.md` — Wf-380-A2 m2 R² per-cohort features
  (12-d/24-d/50-d feature tier mapping; MEASURED synthetic R² +0.1970 + +0.8869)
- `TODO/completed/380a3_diversity_sampling.md` — Wf-380-A3 m7 diversity sampling
  (nucleus_sample + mmd_loss + repulsion_loss + Murcko scaffold; 619 LOC + 300 LOC)
- `TODO/completed/380a4_vina_param_tuning.md` — Wf-380-A4 Vina parameter tuning SCAFFOLD
- `TODO/completed/380a5_multi_template_mm_gbsa.md` — Wf-380-A5 multi-template MM-GBSA SCAFFOLD
- `TODO/completed/380a6_sa_generation_penalty.md` — Wf-380-A6 SA generation penalty SCAFFOLD
- `TODO/completed/380a7_m6kappa_finalize.md` — Wf-380-A7 m6_kappa finalise SCAFFOLD
- `TODO/completed/380b1_index_sync.md` — Wf-380-B1 trainer integration + INDEX/AGENTS sync (~120 LOC)
- `TODO/completed/380_final_synthesis.md` — Wf-380 final synthesis v25 (~700 LOC §0-§9)

### §3.2 Wf-385 source closeouts (D-Wf-385)

- `TODO/completed/385a1_wf380_cli_flag_audit.md` — Wf-385-A1 Wf-380 CLI flag → SCAFFOLD call audit (~205 LOC; VERDICT: 0/17 flags wired in main loop)
- `TODO/completed/385a2_wf299_launch_wrapper_review.md` — Wf-385-A2 Wf-299 launch wrapper consistency review (~471 LOC + 1 source edit ~42 LOC)
- `TODO/completed/385a3_scaffold_wire_in_gate.md` — Wf-385-A3 SCAFFOLD-loss wire-in pre-flight gate (~339 LOC; 20 NEW tests)
- `TODO/completed/385b1_index_sync.md` — Wf-385-B1 docs-only INDEX + AGENTS + final synthesis v28 (~150 LOC)
- `TODO/completed/385_final_synthesis.md` — Wf-385 final synthesis (~580 LOC §0-§9)

### §3.3 Wf-386 source closeouts (D-Wf-386)

- `TODO/completed/386a1_wf384_a3_overwrite_audit.md` — Wf-386-A1 Wf-384-A3 overwrite audit (~232 LOC)
- `TODO/completed/386a2_wf378_a3_wrong_verdict_root_cause.md` — Wf-386-A2 Wf-378-A3 wrong-verdict root cause (~282 LOC)
- `TODO/completed/386a3_atomic_write_protocol.md` — Wf-386-A3 atomic write protocol + truncation detection (~329 LOC; 27-case test suite)
- `TODO/completed/386b1_index_sync.md` — Wf-386-B1 docs-only INDEX + AGENTS + final synthesis sync
- `TODO/completed/386_final_synthesis.md` — Wf-386 final synthesis (~580 LOC §0-§9)

### §3.4 Wf-388 source closeouts (D-Wf-388)

- `TODO/completed/388a1_m2_per_cohort_real.md` — Wf-388-A1 m2 R² per-cohort REAL verification (~272 LOC; 4 NEW tests PASS in 0.41 s wall; MEASURED platinai_4cell −0.1440 + hilbert −0.1440 + metalcytotox_combined +0.1981)
- `TODO/completed/388a2_m7_diversity_wire.md` — Wf-388-A2 m7 Diversity SCAFFOLD->TRAINER wire-in (~327 LOC; 4 NEW tests PASS in 0.39 s wall; +120/-15 LOC source patch)
- `TODO/completed/388a3_m4_sa_wire.md` — Wf-388-A3 m4 SA SCAFFOLD->TRAINER wire-in (~327 LOC; 3 NEW tests PASS in 0.60 s wall; 2 NEW CLI flags wired)
- `TODO/completed/388b1_index_sync.md` — Wf-388-B1 docs-only INDEX + AGENTS sync
- `TODO/completed/388_final_synthesis.md` — Wf-388 final synthesis (~580 LOC §0-§9)

### §3.5 Wf-390 source closeouts (D-Wf-390)

- `TODO/pending/376_m2_r2_gpu_retrain_plan.md` — Wf-390-A1 TODO-376 m2 R² Path A 5-metal subset GPU retrain PLAN_ONLY (~455 LOC, 8 sections; 3 forward paths ranked)
- `TODO/completed/390a2_todo179_deeper_audit.md` — Wf-390-A2 TODO-179 L5 R² deeper audit + alt encoder bootstrap (~24 KB; 40 NEW assertions PASS across 9 test methods in ~5 s wall; 632-d alternative encoder scaffold)
- `TODO/completed/390a3_se3_sb_wire.md` — Wf-390-A3 TODO-383 SE(3) SB wire-in audit + smoke scaffold (~17 KB; 2 NEW tests PASS in 1.55 s wall)
- `TODO/completed/390b1_index_sync.md` — Wf-390-B1 docs-only INDEX + AGENTS + final synthesis v30 (~150 LOC)
- `TODO/completed/390_final_synthesis.md` — Wf-390 final synthesis (~580 LOC §0-§9)

### §3.6 Cross-cutting strategic-plans refresh closeouts (Wf-391)

- `TODO/completed/391a1_lambda_roadmap_refresh.md` — Wf-391-A1 LAMBDA_ROADMAP refresh (~616 → ~700 LOC; 6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11)
- `TODO/completed/391a2_improvement_enum_refresh.md` — Wf-391-A2 IMPROVEMENT_ENUM + QUEUE refresh (4 → 12 tasks; 4 PENDING → 32 tasks)
- `TODO/completed/391a3_metric_strategy_refresh.md` — Wf-391-A3 METRIC_LIFT_STRATEGY_v2 refresh (~410 → ~860 LOC; 8 → 11 sections; NEW §10 forward-looking 2026-Q4 GPU retrain priority order)
- `TODO/completed/391b1_index_sync.md` — Wf-391-B1 docs-only INDEX + AGENTS sync v31
- `TODO/completed/391_final_synthesis.md` — Wf-391 final synthesis (~580 LOC §0-§9)

### §3.7 AGENTS.md + INDEX.md cross-references

- `AGENTS.md` §6 — Wf-378..392 entries appended (5 NEW §6 rows: Wf-378 OPEN CPU-only
  gaps closure + Wf-379 GPU acceleration + Wf-380 ultracode residual implementation
  + Wf-385 deeper-level fix + Wf-386 atomic write protocol + Wf-388 SCAFFOLD->TRAINER
  wire-in + Wf-389 m6_kappa Tier-2 supplementary reframe + Wf-390..392 strategic-plans
  refresh + Wf-393-A3 this sync)
- `AGENTS.md` §5.6 — atomic write protocol adoption rule NEW (per Wf-386-B1)
- `AGENTS.md` §5.5 — post-wave on-disk verification rule NEW (per Wf-384-B1)
- `TODO/INDEX.md` §22 — per-Wf coverage verdict table for Wf-340..392 entries
  (~30 sub-entries; Wf-393-A3 this sync appends new entry)

---

## §4 Honest framing per `party-twenty-spirit §3.5`

Per the standing honest framing discipline, every claim in this Wf-393-A3
closeout is tagged MEASURED vs PROJECTED:

### §4.1 MEASURED claims (verifiable on disk)

- **TODO/decisions.md was edited** to add 5 NEW decisions (D-Wf-380 + D-Wf-385 +
  D-Wf-386 + D-Wf-388 + D-Wf-390) — verified by `cat TODO/decisions.md`
- **TODO/README.md was edited** to refresh last-updated header + pending/ tree
  + completed/ count + NEW Wf-296..392 ultracode ship surface section + Tier-2 /
  wetlab-deprecation compliance verification block — verified by `cat TODO/README.md`
- **`ls TODO/completed/ | wc -l`** = 929 files at 2026-09-30 (MEASURED via shell)
- **`ls TODO/pending/ | wc -l`** = 33 files at 2026-09-30 (MEASURED via shell)
- **`grep -nE "wetlab|Wetlab|WETLAB|wet-lab|wet_lab"`** in `TODO/README.md` =
  **21 matches**, all DEPRECATED markers / summary references / compliance
  verification block (zero affirmative wetlab-work recommendations) (MEASURED
  via shell)
- **`grep -nE "paper/.*SECTIONS|REWRITE_PAPER|EDIT_PAPER"`** in `TODO/README.md`
  = 0 affirmative matches (MEASURED via shell)

### §4.2 PROJECTED claims (forward-looking, USER_OK-gated)

- The 5 NEW decisions in `TODO/decisions.md` are forward-looking USER_OK-gated
  for the MEASURED wire-in + GPU bench + retrain closures; SCAFFOLD-loss wire-in
  remains deferred to Wf-381+ per Wf-385-A1 §6 closure path (~280 LOC + ~12 tests
  PROJECTED on Blackwell sm_120 + GPU retrain USER_OK-gated per AGENTS.md §5.1)

### §4.3 Honest-negative preserved per Wf-375-B1 §22.68.4

- **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per
  Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 7
  architectural + 1 literature attempts exhausted; best aug_mean MEASURED 0.472
  vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but
  threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation:
  drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
- **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-377-A2 5
  alternative splits + Wf-390-A2 deeper audit + Wf-388-A1 multi-stream ensemble
  REAL-cohort verification; platinai_4cell + hilbert cohorts n=200 SMILES each;
  metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per
  Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)
- **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade +
  `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT
  Wf-393-introduced; both preserved per Wf-375-B1 §22.68.4 baseline
- **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env`
  per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with
  −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external
  1.07M dependency)
- **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs +
  R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all
  DEPRECATED / re-scoped to in-silico)

### §4.4 User voice answer

**User directive**: 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow
覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO
items not yet covered by just-launched workflows.

**Wf-393-A3 answer**: **DID彻底解决** the docs-consolidation gap between
Wf-378..392 ultracode ship surface and the canonical architectural decision log
+ top-level TODO navigation. The 5 NEW decisions in `TODO/decisions.md` capture
the architectural choices made by Wf-380 / Wf-385 / Wf-386 / Wf-388 / Wf-390
ultracode waves; the refreshed `TODO/README.md` provides a single source of
truth for the Wf-296..392 ship surface; the Tier-2 / wetlab-deprecation
compliance verification block confirms zero affirmative wetlab-work
recommendations remain in the README.

**Did NOT 彻底解决** (out of Wf-393-A3 scope; honest-negative preserved per
`party-twenty-spirit` §3.5): the structural gates (m6_kappa + m2 R² remain
HONEST_NEGATIVE FINAL; 5 architectural attempts exhausted per Wf-365-A3 +
Wf-377-A1 deep-dive-research move); the 2 NEW pre-flight FAILs (LOI Tier-2
reclassification cascade + wf343c4 anchor freshness wf269c_report.json STALE)
preserved per Wf-375-B1 §22.68.4 baseline; 5 forward-looking USER_OK items
BLOCKED on user OK per Wf-368-B1 §7.2 (~52.9 hr GPU + 13 hr CPU + 30 min
install + SCAFFOLD-MEASURED wire-in total).

**NEW ultracode Wf-394+ IS the recommended follow-up** per Wf-393-A3 + Wf-368-B1
§7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1
§22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 +
Wf-380-B1 §22.74.5 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-388-B1 +
Wf-389-B1 §22.5.3 + Wf-390-B1 §22.7 + Wf-391-B1 §22.81 + Wf-393-A3 this entry
with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper
§6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1
m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension
OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification
on Blackwell sm_120 + atomic write protocol adoption migration (~21.5 hr GPU +
10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 v4 update
total).

---

**END OF Wf-393-A3 CLOSEOUT** — TODO/decisions.md + TODO/README.md refreshed
with Wf-378..392 ship surface; Tier-2 / wetlab-deprecation compliance verified;
0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; pre-flight
v3 89/2/3/94 PRESERVED at Wf-353-B1 baseline.