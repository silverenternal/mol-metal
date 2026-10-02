# Wf-368-C1 final_synthesis — standalone closeout (reconstruction, Wf-371-A2)

**Date:** 2026-09-29 (Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC) — reconstructed 2026-09-30 per Wf-371-A2
**Owner:** Hugo Lee (autonomous Wf-371-A2 docs-only sub-flow)
**Workflow:** Wf-371-A2 (this entry — standalone closeout authored to fill the Wf-368-C1 final_synthesis
gap surfaced by Wf-369-A1 §1.1)
**Predecessors:** `TODO/completed/368{a1,a2,a3,a4,b1,b2,c1_index_sync}_*.md` (7 Wf-368 closeouts;
the original `368c1_index_sync.md` served as the Wf-368-C1 closeout but was named "index_sync"
without the "final_synthesis" tag the task brief asked for; this entry is the standalone
final_synthesis counterpart per Wf-369-A1 §1.1 HONEST-NEGATIVE_FLAG)
**Status:** SHIPPED — 0 NEW source-side fixes (docs-only reconstruction); 1 NEW regression-guard
test PASS at `molmetal/tests/test_wf371a2_368c1_standalone.py`; 0 GPU launches; 0 `paper/*` touches.
**Cross-ref:** `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.2 + §5.3 + §5.4` + Wf-368-B1
§0 + Wf-368-B2 §0 + Wf-369-A1 §1.1 + Wf-370-B1 §22.64.

---

## §0 TL;DR

**One-line answer (per `party-twenty-spirit` §3.5 honest framing)**: **Wf-368-C1 docs-only
synthesis reconstructed as standalone closeout — pre-flight v3 cascade STILL FULLY CLOSED
(92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL); Wf-368 wave 6/7 SHIPPED (86%); 7 USER_OK-gated
launch wrappers + 2 install + 7 paper + 2 metric gates catalogued; m6_kappa + m2 R²
HONEST-NEGATIVE FINAL preserved; 6 forward-looking items BLOCKED on user OK.**

| Metric | Value |
|---|---|
| **preflight_state** | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC wall 145.75 s per `molmetal/reports/wf368c1_preflight.txt` (PRESERVED at Wf-367-C1 baseline 92/0/2/94 wall 155.44 s — CLEAN STATE PRESERVED, NO REGRESSION) |
| **wf340_to_368_index_sync** | YES — all 29 Wf in [340, 368] have at least one §22 entry (per Wf-368-B2 §22.58..§22.61 gap-fill + Wf-368-C1 §22.62 entry) |
| **agmd_wf_recent_waves** | YES — Wf-368 §6 row appended between Wf-367 and Wf-369 §6 rows (per Wf-368-C1 + Wf-369-B1 §6 sync); AGENTS.md 332 → 337 → 338 lines |
| **honest_framing** | m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL + m2 R² HONEST_NEGATIVE_R2_DEEP FINAL; 7 USER_OK-gated launch wrappers AWAIT user OK |
| **cross_refs** | `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + Wf-340..368 closeouts + Wf-369-A1 §1.1 + Wf-370-B1 §22.64 |
| **final_verdict** | **SHIPPED — standalone closeout authored; pre-flight cascade STILL FULLY CLOSED; 1 NEW test PASS; 0 source patches; 0 GPU launches; 0 paper touches** |

---

## §1 Final pre-flight state (Wf-368-C1 fresh-run)

Per `molmetal/reports/wf368c1_preflight.txt`:

```json
{
  "timestamp": "2026-09-29T15:54:19+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 92,
    "fail": 0,
    "skip": 2,
    "total": 94,
    "total_wall_sec": 145.7535
  }
}
```

**Status counts** (per `grep -cE "\[PASS\]|\[FAIL\]|\[SKIP\]"` mapping):
- **92 PASS** (`p1_*` chemprop_v2_v3 gates + `p2_*` tests + `p3_*` chemprop + per-wave +
  chemprop_v3 + integration + GPU + cascade + LOMO + Catherine + closure + reachability +
  triangle closure + Vina + chemprop R20 + drugood + bindingdb + tmqm + platinai + Vina +
  per-metal reachability + per-domain + dry-run + per-metal-pass)
- **0 FAIL** (pre-flight cascade STILL FULLY CLOSED end-to-end per Wf-367-A2 +
  Wf-367-A1 + Wf-368-A1 verification)
- **2 SKIP** (BY-DESIGN: `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with
  −183.04 h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07 M
  dependency)
- **94 TOTAL**

**Pre-flight delta vs prior baselines**:
- vs Wf-367-C1 baseline (92/0/2/94): **0 / 0 / 0 / 0** (PRESERVED exactly)
- vs Wf-366-C1 baseline (91/1/2/94): **+1 / −1 / 0 / 0** (cascade FULLY CLOSED)

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)`.
EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates
added by Wf-368).

---

## §2 INDEX.md §22 entry — Wf-340..368 sync verified

Per Wf-368-B2 §0 (the dedicated INDEX-sync sub-flow that surfaced the Wf-340..368 gap):

| §22.NN | Wf | Title | Status |
|---:|---|---|---|
| §22.49 | Wf-360 | ultracode residual + retrospectives wave | ✅ present |
| §22.50 | Wf-358 | ultracode residual CPU-only fixes + improvements wave | ✅ present |
| §22.51 | Wf-362 | documentation gaps + missing data fixes wave | ✅ present |
| §22.52 | Wf-363 | GPU wrapper polish + dashboard enhancements wave | ✅ present |
| §22.53 | Wf-365 | m6_kappa FINAL verdict + cross-verify wave | ✅ present |
| §22.54 | Wf-364 | ultracode residual close-out wave | ✅ present |
| §22.55 | Wf-366 | USER_OK gate gaps closure + Wf-363/364/365 cross-verification wave | ✅ present |
| §22.56 | Wf-361 | m6_kappa Neural ODE + MAML architectural paths | ✅ present |
| §22.57 | Wf-367 | pre-flight 2 FAIL fixes + m2 R² recovery + truncated verify | ✅ present |
| §22.58 | Wf-368 | ultracode follow-up verification wave — 4 closeouts | ✅ present (Wf-368-B2) |
| §22.59 | Wf-342 | WEAK_TO_STRONG + DrugOOD loader ultracode residual wave | ✅ present (Wf-368-B2 fill) |
| §22.60 | Wf-343 | pre-retrain internal-check + audit + integration ultracode wave | ✅ present (Wf-368-B2 fill) |
| §22.61 | Wf-344 | ultracode residual ultracode wave — metal_cytotox + chemprop CLI + CFM configure_reference fix | ✅ present (Wf-368-B2 fill) |
| §22.62 | Wf-368 | ultracode follow-up verification wave — 4 closeouts + INDEX sync + final synthesis | ✅ present (Wf-368-C1) |

**Wf-340..368 INDEX sync verdict**: **29/29 Wf in [340, 368] inclusive have at least one §22
entry** (per Wf-368-B2 regression-guard test PASS at
`molmetal/tests/test_wf368b2_index_sync.py`).

INDEX.md line count delta since Wf-368-B2: 6679 → 6747 lines (+68 lines for §22.62 +
§22.63 + §22.64 added by Wf-368-C1 + Wf-369-B1 + Wf-370-B1).

---

## §3 AGENTS.md §6 entry — recent Wf waves verified

Per Wf-368-C1 §3 + Wf-369-B1 + Wf-370-B1 sync waves:

| §6 row | Wf | Verdict |
|---|---|---|
| Wf-367 §6 | pre-flight 2 FAIL fixes + m2 R² recovery | ✅ present (Wf-367-C1) |
| **Wf-368 §6** | ultracode follow-up verification wave — 4 closeouts + USER_OK + INDEX + final synthesis | **✅ present (Wf-368-C1 this entry)** |
| Wf-369 §6 | final consolidation wave — 3 cross-verify closeouts + INDEX + AGENTS + final synthesis | ✅ present (Wf-369-B1) |
| Wf-370 §6 | final consolidation v18 — final pre-flight + INDEX + AGENTS sync + final ship | ✅ present (Wf-370-B1) |

**AGENTS.md §6 sync verdict**: **Wf-368 row appended between Wf-367 and Wf-369 rows;
Wf-369 + Wf-370 rows appended after Wf-368; chronological order preserved**.

AGENTS.md line count delta since Wf-368-C1: 337 → 338 lines (current Wf-371-A2 read).

---

## §4 Honest framing — m6_kappa + m2 R² FINAL + 7 USER_OK-gated launches

Per `party-twenty-spirit` §3.5:

### §4.1 m6_kappa gate — HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL

- **MEASURED**: augmented_mean 0.4386 (best 5 attempts) vs threshold 0.6 — gap −0.1614
- **MEASURED**: structural CV-R² +0.6814 (chemprop + PLL) CEILING_BREAK at Wf-333-V4
  + Wf-335-M1 bit-identical on 60-SMILES smoke cohort
- **5 architectural attempts exhausted** per Wf-365-A3 §2 + Wf-366-B2 §4 + Wf-367-B2
  cross-poll + Wf-368-A4 cross-verify + Wf-369-A1 §2.2
- **FINAL**: gate NOT closed; structural ceiling LIFTED but threshold-reaching axis NOT lifted

### §4.2 m2 R² recovery — HONEST_NEGATIVE_R2_DEEP FINAL

- **MEASURED**: platinai_4cell + hilbert cohorts n=200 SMILES each — 50+ feature Ridge
  ensemble OVERFITS on small cohorts (R² ~ −0.1251)
- **MEASURED**: metalcytotox_combined n=2000 — CEILING_BREAK at R²=+0.3196 per Wf-335-M2
  + Wf-367-B1 multi-stream ensemble
- **FINAL per Wf-369-A2 verdict doc**: per-cohort verdict table preserved

### §4.3 7 USER_OK-gated launch wrappers (per Wf-368-B1 §2)

| # | Wrapper | Coarse scope | PROJECTED wall | USER_OK gate |
|---|---|---|---:|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | DrugOOD 1.58 M × chemprop v2 (R20 §3) | ~10 hr | `USER_OK=YES` |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | BindingDB 1.07 M × chemprop v2 (R20 §4) | ~7 hr | `USER_OK=YES` |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | tmQM × PlatinAI 4-cell joint W₂ (R20 §5) | ~12 hr | `USER_OK=1` |
| 4 | `launch_wf131_p3_deferred_runs.sh` | Wf-131 P3 7 DEFERRED runs | ~5 hr | `USER_OK=1` |
| 5 | `launch_wf146_deferred_runs.sh` | Wf-146 6 DEFERRED runs | ~3 hr | `USER_OK=YES` |
| 6 | `launch_wf124_p3_g5.sh` | G5 50-epoch full sweep × 5 metals × 3 seeds | ~5 hr | `Wf-366-A1 STRICT` |
| 7 | `recover_g6_failed_tuples.sh` | G6 57 failed + pending tuples recovery | ~11.4 hr | `Wf-366-A2 STRICT` |

**TOTAL PROJECTED wall**: **~52.9 hr GPU** sequential (best case) — all 7 BLOCKED on USER_OK
per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 + `kill_then_adapt` standing directive.

---

## §5 Cross-references (Wf-340..368 + standing directives)

### §5.1 Per-Wf closeout pointers (Wf-340..368 closeouts on disk)

| Wf | Closeout files | Key verdict |
|---|---|---|
| Wf-340 | `TODO/completed/340_*.md` + `340_final_synthesis.md` | CFM training framework survey + 6 NEW wires (~820 LOC) |
| Wf-341 | `TODO/completed/341_*.md` | CfM joint research + improvement ultracode |
| Wf-342 | `TODO/completed/342_final_closeout.md` + §22.59 | WEAK_TO_STRONG + DrugOOD loader ultracode residual |
| Wf-343 | `TODO/completed/343d_final.md` + §22.60 | pre-retrain internal-check + 4 NEW pre-flight gates |
| Wf-344 | `TODO/completed/344_final_synthesis.md` + §22.61 | metal_cytotox + chemprop CLI + CFM configure_reference fix |
| Wf-345 | `TODO/completed/345_*.md` | 3 Wf-343C4 LOGIC-fix (subset) |
| Wf-346 | `TODO/completed/346_final_synthesis.md` | Pre-flight v3 fresh-run 65/2/1/68 |
| Wf-347 | `TODO/completed/347_*.md` | 2 NEW pre-flight gates + 65/2/1/68 |
| Wf-348 | `TODO/completed/348_*.md` | Ultracode residual wave |
| Wf-349 | `TODO/completed/349_*.md` | Ultracode residual wave |
| Wf-350 | `TODO/completed/350_*.md` | 3 NEW pre-flight gates + 70/0/1/71 |
| Wf-351 | `TODO/completed/351_*.md` | SCI paper setup (Zotero + tectonic + Vale) |
| Wf-352 | `TODO/completed/352_*.md` | Per-pipeline audit + expected_gate_counts bump |
| Wf-353 | `TODO/completed/353_*.md` | Final consolidation v14 |
| Wf-354 | `TODO/completed/354_*.md` | Ultracode residual wave |
| Wf-355 | `TODO/completed/355_*.md` | R20 §3-§5 launch wrappers |
| Wf-356 | `TODO/completed/356_*.md` | Ultracode residual wave |
| Wf-357 | `TODO/completed/357_*.md` | Final synthesis wave |
| Wf-358 | `TODO/completed/358_*.md` + §22.50 | ultracode residual CPU-only fixes |
| Wf-359 | `TODO/completed/359_*.md` | Final synthesis wave |
| Wf-360 | `TODO/completed/360_*.md` + §22.49 | ultracode residual + retrospectives |
| Wf-361 | `TODO/completed/361_*.md` + §22.56 | m6_kappa Neural ODE + MAML |
| Wf-362 | `TODO/completed/362_*.md` + §22.51 | Documentation gaps + missing data fixes |
| Wf-363 | `TODO/completed/363_*.md` + §22.52 | GPU wrapper polish + dashboard enhancements |
| Wf-364 | `TODO/completed/364_*.md` + §22.54 | ultracode residual close-out |
| Wf-365 | `TODO/completed/365_*.md` + §22.53 | m6_kappa FINAL verdict + cross-verify |
| Wf-366 | `TODO/completed/366_*.md` + §22.55 | USER_OK gate gaps closure + cross-verification |
| Wf-367 | `TODO/completed/367_*.md` + §22.57 | pre-flight 2 FAIL fixes + m2 R² recovery |
| Wf-368 | `TODO/completed/368_*.md` + §22.58 + §22.62 (this entry) | ultracode follow-up verification |
| Wf-369 | `TODO/completed/369_*.md` + §22.63 | final consolidation wave |
| Wf-370 | `TODO/completed/370_*.md` + §22.64 | final consolidation v18 + final ship |

### §5.2 Standing directives (re-confirmed)

- **AGENTS.md §5.1** Standing GPU authorization — preserved; per-wrapper USER_OK gating
  intact (no auto-trigger)
- **AGENTS.md §5.2** Honest framing — every §1-§4 numeric tagged MEASURED / PROJECTED with
  source path on disk
- **AGENTS.md §5.3** No paper work without authorization — 0 `paper/*` touches across Wf-368
  wave
- **AGENTS.md §5.4** Tier-2 journal target + NO wet-lab planning — Tier-2 acceptance bar is
  computational-only; wet-lab items DEPRECATED per 2026-09-29 user directive
- **`party-twenty-spirit` §3.5** MEASURED-vs-PROJECTED tag discipline — every claim in this
  entry traces to a JSON / report path on disk

### §5.3 Per-section source pointers

| Section | Sources |
|---|---|
| §1 pre-flight state | `molmetal/reports/wf368c1_preflight.txt` (Wf-368-C1 fresh-run 92/0/2/94 wall 145.75 s) |
| §2 INDEX.md §22 entry | `TODO/INDEX.md` lines 6747 (current count) + Wf-368-B2 §22.58..§22.61 + Wf-368-C1 §22.62 + Wf-369-B1 + Wf-370-B1 §22.63 + §22.64 |
| §3 AGENTS.md §6 entry | `AGENTS.md` lines 338 (current count) + Wf-368 §6 row + Wf-369 + Wf-370 §6 rows |
| §4 honest framing | `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 NEW 320 LOC) + Wf-368-B1 §0 + Wf-368-B1 §2 (7 wrappers) |
| §5 cross-refs | Wf-340..370 closeouts on disk + `party-twenty-spirit` §3.5 + `AGENTS.md` §5.1..§5.4 |

### §5.4 Skill cross-refs

- **`party-twenty-spirit`** skill — MEASURED-vs-PROJECTED tag discipline + 5-step decision
  checklist (mandatory pre-flight for TODO work)
- **`mol-metal-gpu-execution`** skill — kill_then_adapt + Blackwell sm_120 + multi-GPU
  scheduling + USER_OK gates
- **`mol-metal-paper-writing`** skill — READ-only on `paper/*`; loads `paper/sections/*.tex`
  as data
- **`mol-metal-sci-paper-submission`** skill — Zotero + tectonic + Vale + Jane + Manubot +
  Quarto infra (Wf-351-B3)

### §5.5 Wall + cost

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Read Wf-368-B1 + Wf-368-B2 + Wf-368-C1 INDEX sync closeouts for context | ~2 min (Read) |
| Read `molmetal/reports/wf368c1_preflight.txt` for fresh-run output | ~1 min (Read) |
| Verify Wf-340..368 INDEX sync status (29/29 §22 entries present) | ~1 min (Bash grep) |
| Verify AGENTS.md §6 recent Wf waves (Wf-368 + Wf-369 + Wf-370 rows present) | ~1 min (Bash grep) |
| Write `TODO/completed/368c1_final_synthesis.md` (this closeout ~280 LOC, 5 sections) | ~5 min |
| Write `molmetal/tests/test_wf371a2_368c1_standalone.py` (1 NEW test ~70 LOC) | ~3 min |
| Run pytest | 0.34 s (MEASURED) |
| Write `TODO/completed/371a2_368c1_standalone_authored.md` (delivery doc ~140 LOC) | ~3 min |
| **Total Wf-371-A2 CPU wall** | **~16 min CPU-only** (within PROJECTED 5-10 min per task brief — adjusted to 16 min actual) |

**PROJECTED vs MEASURED**:
- PROJECTED 5-10 min CPU-only at task brief
- MEASURED ~16 min (slightly over brief; due to comprehensive 5-section closeout with
  full cross-references)

**GPU launches performed**: 0 (per AGENTS.md §5.1 kill_then_adapt standing directive;
docs-only sub-flow).
**Paper touches**: 0 (per AGENTS.md §5.3 + TODO-84; this closeout lives under `TODO/`
ONLY).

---

**END OF Wf-368-C1 final_synthesis standalone closeout (Wf-371-A2 reconstruction)** — Owner
Hugo Lee; cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.2 + §5.3 + §5.4`;
last update 2026-09-29 (Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC wall 145.75 s) —
reconstructed 2026-09-30 per Wf-371-A2; CPU-only docs surface; pre-flight v3 **92 PASS / 0
FAIL / 2 SKIP / 94 TOTAL** PRESERVED; INDEX.md Wf-340..368 §22 sync 29/29 verified; AGENTS.md
§6 Wf-368..370 rows verified; m6_kappa + m2 R² HONEST-NEGATIVE FINAL preserved; 7
USER_OK-gated launches + 6 forward-looking items BLOCKED on user OK; 0 GPU launches; 0
`paper/*` touches; 1 NEW test PASS.