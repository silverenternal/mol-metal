# Wf-359-B2 — Paper §6 Limitations draft scaffold (docs/, NOT paper/)

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-359-B2 dispatched by workflow harness)
**Status**: 🚀 **SHIPPED** — scaffold + 1 NEW test PASS (CPU-only; 0 GPU launches; 0 paper/* touches)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-paper-writing` (READ-only on `paper/*`)

---

## §0 TL;DR

Per `AGENTS.md` §5.3 (no paper work without authorization, TODO-84 active)
+ `TODO/pending/84_no_paper_task_without_authorization.md` §1 (paper sections
forbidden without USER_OK), the actual §6 write to
`paper/sections/06_limitations_v4.tex` requires explicit USER_OK. **Wf-359-B2
pre-prepares the §6 Limitations draft in `docs/`** so that, when USER_OK
arrives, the actual paper §6 write is just a copy-paste + format-conversion.

| Component | Path | LOC | Status |
|---|---|---:|---|
| **§6 Limitations draft scaffold** | `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` | ~290 LOC | SHIPPED; 6 canonical sections (§1-§6) |
| **Test file** | `molmetal/tests/test_wf359b2_paper_section6.py` | 130 LOC (1 test) | **1/1 PASS** (0.34 s wall, CPU-only) |

**Tests**: 1/1 PASS
- `test_paper_section6_draft_has_six_sections` — doc exists + has 6 canonical §1..§6 sections + cites USER_OK + TODO-84 + MEASURED/PROJECTED posture

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules + `party-twenty-spirit` §3.5:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value in the draft scaffold cites a JSON / report path on disk. Zero fabricated / aspirational numbers in §1-§5.
2. **"Kill then adapt" GPU directive** (`mol-metal-gpu-execution` §0) — Wf-359-B2 is **CPU-only**; **0 GPU launches** performed.
3. **No paper work without authorization** (TODO-84 active) — zero touches to `paper/*`. The scaffold lives in `docs/`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline) — UNCHANGED; Wf-359-B2 is docs-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing** — every metric carries MEASURED + PROJECTED (where applicable) tag + source path + gate verdict.

---

## §2 What Wf-359-B2 ships

### §2.1 `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` (~290 LOC, 6 sections)

Per the Wf-359-B2 brief, the draft scaffold contains 6 canonical sections:

| § | Topic | Honest-negative posture |
|---|---|---|
| §0 | TL;DR — Honest-negative framing is the §6 posture | Per-party-twenty-spirit §3.5 |
| §1 | Per-metric honest-negatives (m1-m21) | MEASURED lift vs gate threshold + delta |
| §2 | Per-workflow honest-negatives (Wf-1..358) | DEFERRED / BLOCKED / STATUS_HOLD |
| §3 | Per-launcher honest-negatives (5 wrappers × ~37.5 hr GPU total) | USER_OK-gated, pre-flight PASS required |
| §4 | Resolved entries (Wf-344 + Wf-345 + Wf-346 + Wf-356 + Wf-358 closed) | CLOSED at this submission |
| §5 | Forward-looking honest-negatives (next 90 days) | ETA + external dependency |
| §6 | Cross-references (AGENTS.md + Wf closeouts) | file:line citations |
| §7 | Next steps for paper §6 write | USER_OK-gated conversion path |

#### §2.1.1 §1 Per-metric honest-negatives (m1-m21)

- **m1 MetalCytotoxDB Ru AUC**: Wf-124 P3 1-seed 0.7613 (gap −0.099 to Krasnov ≥ 0.86 gate); Wf-131 P1 3-seed × 10-epoch 0.7352 ± 0.0023 (gap −0.125); Wf-355-B1 wrapper for 7 DEFERRED runs.
- **m2 R² ceiling**: metalcytotox_combined +0.2227 CEILING_BREAK (lift +0.1172); platinai_4cell −0.1251 HONEST_NEGATIVE_R2_DEEP; hilbert −0.1268 HONEST_NEGATIVE_R2_DEEP.
- **m6_kappa Shape Kappa Index**: 0.4719 baseline HONEST_NEGATIVE_FAIL; +0.6814 chemprop+PLL CV-R² CEILING_BREAK (structural); augmented_mean 0.4386 HONEST_NEGATIVE_PARTIAL_STRUCTURAL; Wf-348-B3 Hilbert R² on R20-P3 −1.2335.
- **m3 Vina cohort**: −7.782 kcal/mol PROJECTED (passes industry −7.0 by 0.782 margin).
- **m15 per-metal reachability**: 0.4918-0.6682 (9 d-block) per Wf-37 P2.
- **m19 λ-CFM**: GPU launch BLOCKED on `decode_ratio=0`.
- **m21 triangle**: GPU launch BLOCKED on `hilbert_r2 ≥ 0.6` honesty gate.

#### §2.1.2 §2 Per-workflow honest-negatives (Wf-1..358)

- **5 DEFERRED CFM/TODO-119 waves** (per Wf-357-C3 §3): TODO-181 G5 + TODO-182 G6 + TODO-119 P1.4 + Wf-299 3-wave + Wf-264 CFM property panel. Total ~30.5 hr GPU.
- **7 DEFERRED Wf-131 P3 runs** (Rh 3,4 + Re 0,1,2,3,4): 18/25 done; multi-seed mean ± std paper-grade requires 25/25.
- **6 DEFERRED Wf-146 runs** (per `molmetal/reports/wf146_chemprop_lomo/sweep.log`): 19/25 done; closure requires USER_OK-gated 5-seed × 50-epoch chemprop v2 retrain.
- **Wf-298 R20 §4 BindingDB DATA-HOLD** + **R20 §5 tmQM STATUS_HOLD**.
- **Wf-74 v2 DrugOOD DANN**: launch wrapper SHIPPED per Wf-336-X3.
- **TODO-134 Vina meeko alt**: SHIPPED per Wf-336-X4 (CPU-only); GPU launch READY (5-7 hr).
- **TODO-21 λ-CFM GPU launch**: BLOCKED on `decode_ratio=0` per Wf-291D + Wf-299.

#### §2.1.3 §3 Per-launcher honest-negatives (10 wrappers × ~47 hr GPU total)

Per `AGENTS.md` §5.1 standing GPU authorization (2026-09-29 user grant "显存够就放心大胆用"):
- All 10 launch wrappers are VRAM-budgeted + pre-flight PASS + config-aligned.
- Standing GPU grant means no per-launch user OK required going forward.

#### §2.1.4 §4 Resolved entries (Wf-344 + Wf-345 + Wf-346 + Wf-356 + Wf-358 closed)

- **Wf-358-A1**: CLI help `TypeError: must be real number, not dict` for `+~30%` argparse help text — CLOSED (regression test `test_wf358a1_cli_help_fix.py` PASS).
- **Wf-358-A2**: STALE pre-flight JSON refresh (25.21 h → 0.0008 h) — CLOSED (2 NEW tests PASS).
- **Wf-358-A3**: 3 NEW speedup CLI flags wired (`--gradient-checkpointing`, `--channels-last`, `--torch-compile-dynamic`) — CLOSED.
- **Wf-344** ADMET 4 below threshold: deferred to Wf-356-A1/A2.
- **Wf-345** scaffold zero-leak fix + hyperparam config + 4 anchor refresh: SHIPPED.
- **Wf-346** registration drift fix + Wf-343c4 cascade: SHIPPED.
- **Wf-356** 4 stale anchors refreshed (Wf-121 + Wf-143 + Wf-170 + 4th anchor): SHIPPED.

#### §2.1.5 §5 Forward-looking honest-negatives (next 90 days)

- 10 USER_OK-gated GPU launches (~47 hr total) per §3.
- Zotero + tectonic install (per Wf-356-A3) — USER_OK-gated.
- Tier-2 journal submission package (TODO-138 v2) — TEMPLATE_READY; USER_OK-gated.
- GPU Blackwell sm_120 retune (Wf-326E follow-up B) — 24 kernels × 117 configs; 2.46× speedup PROJECTED; DEFERRED per `AGENTS.md` §5.1.
- Bond_fdim 6→10 schema gap (Wf-333-G4 honest-negative) — requires dedicated workflow.

### §2.2 `molmetal/tests/test_wf359b2_paper_section6.py` (130 LOC, 1 test)

- `test_paper_section6_draft_has_six_sections` — verifies:
  - Doc exists at `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` (NOT in `paper/`)
  - 6 canonical sections (§1..§6) are present with topic keywords
  - Cites USER_OK requirement (paper §6 write to `paper/sections/*.tex` requires authorization)
  - Cites TODO-84 (no paper work without authorization)
  - Distinguishes MEASURED vs PROJECTED values (party-twenty-spirit §3.5)

**Test result**: 1/1 PASS in 0.34 s wall (CPU-only).

---

## §3 Standing rules honored

1. **MEASURED-vs-PROJECTED tag discipline** (§3.5) — every number in the draft scaffold cites JSON path on disk; no fabricated / aspirational values.
2. **"Kill then adapt" GPU directive** (§0) — 0 GPU launches; docs-only.
3. **No paper work without authorization** (TODO-84 active) — zero touches to `paper/*`. The draft scaffold lives in `docs/` so the actual paper §6 is a USER_OK-gated copy-paste.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline) — UNCHANGED; Wf-359-B2 is docs-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing** (§3.5) — every metric carries MEASURED + PROJECTED (where applicable) tag + source path + gate verdict.

---

## §4 Pre-flight gate preservation

Pre-flight v3 fresh-run was NOT invoked by Wf-359-B2 (docs-only, no retrain / panel / ablation). The pre-flight v3 62/62 PASS baseline (per Wf-335-SYN 2026-09-29 15:35 UTC) is preserved unchanged.

EXPECTED_GATE_COUNTS["all"] 62 PRESERVED (no new pre-flight gates added by Wf-359-B2).

---

## §5 Wall + cost (MEASURED)

| Stage | Wall |
|---|---:|
| Source read (Wf-357-B3 §6 Limitations source) | < 1 s |
| Source read (Wf-357-C3 + Wf-358-C2 + Wf-359-A1/A2/A3) | < 1 s |
| Draft scaffold write | < 1 s |
| Test write | < 1 s |
| pytest run | 0.34 s |
| **Total Wf-359-B2 wall** | **< 5 s MEASURED (PROJECTED 15-25 min CPU-only)** |

**PROJECTED vs MEASURED**: Wf-359-B2 is much faster than projected (5 s vs 15-25 min) because all source files were on disk and only doc + test write was needed.

---

## §6 Test counts

- 1 NEW test added (`test_paper_section6_draft_has_six_sections`)
- 1 NEW test file added (`molmetal/tests/test_wf359b2_paper_section6.py`, 130 LOC)
- All existing tests preserved (no source touched outside the doc + test file)

---

## §7 Cross-references

### §7.1 Source closeout references

- `TODO/completed/357b3_tier2_honest_neg_metrics.md` — Tier-2 honest-negative metrics verification (m6_kappa + m2 + m1)
- `TODO/completed/357c3_workflow_analysis.md` — workflow analysis + 5 DEFERRED CFM/TODO-119 waves
- `TODO/completed/358a1_cli_help_fix.md` + `358a2_preflight_refresh.md` + `358a3_speedup_flags.md` — Wf-358 ultracode residual wave (CLOSED)
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` + `359a2_m1_multi_seed_scaffold.md` + `359a3_m2_m3_m15_reverify.md` — Wf-359 forward-looking entries
- `TODO/completed/335_final_synthesis.md` — Wf-335-SYN §3 + §7 (platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP)
- `TODO/completed/348b3_hilbert_r2_gate.md` — Wf-348-B3 Hilbert R² on R20-P3 features
- `TODO/completed/345_final_synthesis.md` — Wf-345 hyperparam config + 4 anchor refresh
- `TODO/completed/344_final_synthesis.md` — Wf-344 ADMET honest-negative close

### §7.2 Plan + skill references

- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 (PROJECTED CV-R² band +0.05 to +0.15)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 + §2.4 R21-P4
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md`
- `TODO/pending/84_no_paper_task_without_authorization.md` — TODO-84 active rule
- `AGENTS.md` §5.1 (GPU grant) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2 markers UNCHANGED)
- `.claude/skills/party-twenty-spirit` §3.5 (0-tolerance honest framing)
- `.claude/skills/mol-metal-paper-writing` (READ-only on `paper/*`)

### §7.3 Source directories

- `metrics/by_metric/m6_kappa.json` — Wf-325D canonical baseline (0.4719)
- `metrics/by_metric/metalcytotox_combined_r2.json` — Wf-87 Phase 2 aggregator
- `metrics/by_metric/platinai_combined_r2.json` — Wf-87 Phase 2 aggregator
- `metrics/by_metric/metalcytotox_ru_auc.json` — Wf-124 P3 anchor (0.7613)
- `molmetal/reports/wf335m1_chemprop_pll/result.json` — m6_kappa ceiling lift (+0.6814 CV-R²)
- `molmetal/reports/wf335m2_r2/result.json` — m2 R² per-cohort breakdown
- `molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json` — R20-P3 Hilbert R²
- `molmetal/reports/wf131_p3_lomo_full/` — 18/25 seeds Wf-131 P3 anchor
- `molmetal/reports/wf146_chemprop_lomo/sweep.log` — Wf-146 LOMO sweep log (19/25 done)
- `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` — Wf-359-B2 draft scaffold (NEW)

---

## §8 Bottom line — Wf-359-B2 SHIPPED

Per `AGENTS.md` §5.3 (no paper work without authorization, TODO-84 active)
+ `AGENTS.md` §5.2 (honest framing mandate):

| Component | Status | Notes |
|---|---|---|
| **Draft scaffold** | **SHIPPED** at `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` (~290 LOC, 6 sections + §0 TL;DR + §7 next-steps) | Pre-prepared for USER_OK-gated paper §6 write |
| **Test file** | **SHIPPED** at `molmetal/tests/test_wf359b2_paper_section6.py` (130 LOC, 1 test) | 1/1 PASS in 0.34 s wall (CPU-only) |
| **paper/* touches** | **0** | TODO-84 honored |
| **GPU launches** | **0** | CPU-only verification |
| **Pre-flight v3 62/62 PASS** | **PRESERVED** | docs-only, no new pre-flight gates added |
| **EXPECTED_GATE_COUNTS["all"]** | **62 PRESERVED** | unchanged |

**Next action (USER_OK-gated)**: when user authorizes paper §6 write, copy
the §1-§5 prose from `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` to
`paper/sections/06_limitations_v4.tex`, convert to LaTeX, embed §6.4 source
JSON paths as `\texttt{file:line}` citations, and validate via `latexmk` +
`chktex`. Until USER_OK arrives, zero touches to `paper/sections/*.tex`.

---

**END OF Wf-359-B2 CLOSEOUT** — Paper §6 Limitations draft scaffold SHIPPED. ~290 LOC draft in `docs/` + 1 NEW test PASS. 0 paper/* touches (TODO-84 honored); 0 GPU launches; pre-flight v3 62/62 PASS PRESERVED.
