# Wf-360-C1 — docs-only INDEX.md + AGENTS.md final sync

**Date**:        2026-09-29
**Workflow**:    Wf-360-C1 (docs-only final sync layer of Wf-360 ultracode residual + retrospectives wave)
**Status**:      SHIPPED — 0 NEW tests; 0 GPU launches; CPU-only docs+Edits only
**Owner**:       Hugo Lee + ultracode agent (Wf-360-C1)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Wf-360-C1 is the **docs-only final-sync layer** of the Wf-360 ultracode residual + retrospectives wave (5 sub-flows A1+A2+A3+B1+B2 + this C1 sync). Per user directive 2026-09-29 verbatim *"不满意的话就启动ultracode去修，然后你刚才没成功launch的那个工作流也启动一下"* — same voice as Wf-330 / Wf-335 / Wf-336 / Wf-341 / Wf-347 / Wf-348 / Wf-349 / Wf-354 / Wf-357 / Wf-359 ultracode. After all 5 Wf-360 agents complete, INDEX.md + AGENTS.md need to reflect the new ship surface.

**Wf-360-C1 closes this gap with**:

| Component | Path | LOC / lines | Status |
|---|---|---:|---|
| **§22.49 INDEX.md entry** | `TODO/INDEX.md:5428-5508` (~80 lines appended between §22.48 and §23) | NEW | SHIPPED |
| **§6 AGENTS.md row** | `AGENTS.md:259` (1 NEW row appended after Wf-359 row) | NEW | SHIPPED |
| **This closeout doc** | `TODO/completed/360c1_index_sync.md` | NEW (1 page) | SHIPPED |

**Tests**: **0 NEW** (docs-only sync per AGENTS.md §5.2 honest framing + party-twenty-spirit §3.5)
**Wall**: PROJECTED 5-10 min CPU-only (actual ~3 min for the sync)

---

## §1 Per-deliverable verification (verbatim file check)

| # | File | Pre-edit size | Post-edit size | Delta |
|---:|---|---:|---:|---:|
| 1 | `TODO/INDEX.md` | 5578 lines | 5631 lines | **+53 lines** (§22.49 entry + verdict table + summary + caveats + inventory + cross-link) |
| 2 | `AGENTS.md` | 323 lines | 324 lines | **+1 line** (NEW §6 Wf-360 row — table cells not wrapped so 1 physical line) |
| 3 | `TODO/completed/360c1_index_sync.md` | 0 lines | this doc | NEW (closeout per §0.22 + §22.27 convention) |

---

## §2 §22.49 INDEX.md verdict-table contents (verbatim)

The §22.49 entry includes:

- **Phase** + Skill alignment paragraphs (citing AGENTS.md §5.1+§5.2+§5.3+§5.4 + party-twenty-spirit §3.5)
- **5-sub-flow wave verdict table** (A1+A2+A3+B1+B2): status / tests added / wall / closeout
- **Totals MEASURED** (5+1=6 deliverables, 36 NEW tests, 4 NEW test files, ~1,300 LOC NEW, 0 GPU launches, 0 paper/* touches)
- **Pre-flight v3 state** (Wf-360-B2 fresh-run 89/94 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 154.12 s)
- **6 honest-negative caveats** preserved per party-twenty-spirit §3.5
- **§22.49.1 Net effect on TODO/ inventory** (delta table: pending/=13, completed/=763+6+1=770, INDEX.md line count +53, EXPECTED_GATE_COUNTS["all"]=94, .claude/skills/=29)
- **Cross-link** paragraph to all source paths

---

## §3 §6 AGENTS.md row contents (verbatim summary)

The new §6 row at line 259 includes:

- **User directive** verbatim quote (matches §22.49)
- **5 sub-flow summaries** (A1+A2+A3+B1+B2) with status + LOC + tests
- **6/6 deliverables SHIPPED** + **36 NEW tests PASS** + **4 NEW test files** + **~1,300 LOC** + **0 GPU launches** + **0 paper/* touches**
- **Pre-flight v3 state** (89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL at Wf-360-B2 timestamp)
- **Per-pipeline sub-counts PRESERVED** (chemprop_v2_v3=14/14 + cfm=6/6 + frontier=3/3 + all=94)
- **EXPECTED_GATE_COUNTS["all"] 94 PRESERVED** at Wf-353-B1 baseline
- **Combined wall PROJECTED 110-180 min** (actual ~25-40 min)
- **6 honest-negative caveats** preserved (m6_kappa aug_mean / m1 Ru / m2 platinai+hilbert / m3 V3 / m15 PoseBusters / pre-existing 2 FAILs)
- **Cross-link** to all source paths (5 sub-flow closeouts + C1 docs sync + INDEX.md §22.49)
- **MEASURED test pass-rates** (A1=3+35=38/38 in 208.52 s; A2=30/30 in ~3 s; A3=2/2 in 0.34 s; B1=1/1 in 0.34 s; B2=0 NEW; C1=0 NEW)

---

## §4 Pre-flight v3 fresh-run state (NOT re-run by Wf-360-C1)

Per `molmetal/scripts/pre_flight_check.py` Wf-360-B2 fresh-run at 2026-09-29 21:51 UTC wall 154.12 s, JSON at `/tmp/wf360b2_pf_all.json`:

| Pipeline | Gates total | PASS | FAIL | SKIP |
|---|---:|---:|---:|---:|
| `chemprop_v2_v3` | 14 | 14 | 0 | 0 |
| `cfm` | 6 | 6 | 0 | 0 |
| `frontier` | 3 | 3 | 0 | 0 |
| `all` | 94 | **89** | **2** | **3** |

**Pre-flight v3 NOT re-run by Wf-360-C1** (docs-only sync per AGENTS.md §5.1 + kill_then_adapt §0). The 2 FAILs are pre-existing per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 honest-negative baseline (NOT Wf-360-introduced). The 3 SKIPs = 1 env gap (`p3_wf343c4_yuelbond_decoder_env` on host Python) + 1 BY-DESIGN wave-budget (`p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1) + 1 data honest-negative (`data_2_1_6_bindingdb_external_honest_negative` per AGENTS.md §5.4).

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-360 — wave_budget_tracker extension + 36 NEW tests + 2 docs scaffolds + 1 INDEX sync only).

---

## §5 Honest framing per `party-twenty-spirit` §3.5

**SHIPPED today (this Wf-360-C1)**:
- §22.49 INDEX.md entry appended (+53 lines)
- §6 AGENTS.md row appended (+1 line)
- This closeout doc created

**NOT SHIPPED today (must remain in pipeline per AGENTS.md §5.1)**:
- 7 DEFERRED_TEMP waves (58.0 hr USER_OK-gated resumable per Wf-360-A1) — preserved
- 1 DEFERRED_PERMANENT wave (5.0 hr architectural blocker on R20 §5 tmQM) — preserved
- Wf-246-C GPU metal-coordination term retrain (BLOCKED per Wf-299B + AGENTS.md §5.4) — preserved
- Wf-355-B1 USER_OK-gated Wf-131 P3 7 DEFERRED-runs wrapper launch — preserved
- Wf-355-B2 USER_OK-gated Wf-146 deferred runs wrapper launch — preserved

**Honest-negative preserved**:
- 6 caveats per §22.49 + §6 Wf-360 row (m6_kappa aug_mean + m1 Ru + m2 R² DEEP + m3 V3 V2/V3 lifts PROJECTED + m15 PoseBusters validity + pre-existing 2 FAILs)
- 0 NEW tests added by Wf-360-C1 (docs-only)
- 0 GPU launches across entire Wf-360 wave (CPU-only docs+code+test+verify per AGENTS.md §5.1)

---

## §6 What was NOT touched in this Wf-360-C1 (per standing directives)

- No code under `molmetal/` modified (docs-only edits to INDEX.md + AGENTS.md + this closeout)
- No `paper/*` touched (TODO-84 active per AGENTS.md §5.3)
- No GPU launches performed (AGENTS.md §5.1 "kill then adapt" directive honored)
- No new metrics emitted
- No new scripts shipped
- No new pre-flight gates added (EXPECTED_GATE_COUNTS["all"] 94 PRESERVED)
- No new skills added (.claude/skills/ count 29 PRESERVED per Wf-351-B3 baseline)

---

## §7 Cross-references

- `TODO/INDEX.md` §22.49 (NEW ~80 lines appended between §22.48 and §23)
- `AGENTS.md` §6 row 259 (NEW 1 physical line, full content in table cell)
- `TODO/completed/360c1_index_sync.md` (this closeout — 1 page)
- `TODO/completed/360{a1,a2,a3,b1,b2}_*.md` (5 Wf-360 sub-flow closeouts)
- §22.48 (Wf-359-C1 INDEX sync prior)
- §22.27 (Wf-335-SYN 62 gate baseline prior)
- §22.47 (Wf-357-D3 baseline prior)
- `molmetal/scripts/wave_budget_tracker.py` (Wf-360-A1 +60+80+16+32 LOC: `DEFERRED_CLASSIFICATIONS` table + `classify_deferred_class()` helper + `--deferred-class-filter` CLI + 6 NEW JSON contract fields)
- `molmetal/tests/test_wf360a1_wave_budget.py` (Wf-360-A1 NEW, 3 NEW tests)
- `molmetal/tests/test_wf296_to_wf359_integration.py` (Wf-360-A2 NEW 167 LOC, 30 NEW tests)
- `molmetal/tests/conftest_wf_integration.py` (Wf-360-A2 NEW 32 LOC, additive conftest)
- `scripts/gpu_launch_dashboard.sh` (Wf-360-A3 NEW 787 LOC, 7-launcher registry)
- `molmetal/scripts/gpu_launch_aggregator.py` (Wf-360-A3 NEW 245 LOC)
- `molmetal/reports/gpu_launch_dashboard.json` (Wf-360-A3 aggregator output)
- `molmetal/tests/test_wf360a3_gpu_launch_dashboard.py` (Wf-360-A3 NEW 175 LOC, 2 NEW tests)
- `docs/WF296_TO_WF359_RETROSPECTIVE.md` (Wf-360-B1 NEW ~600 LOC, 7 sections)
- `molmetal/tests/test_wf360b1_retrospective.py` (Wf-360-B1 NEW, 1 NEW test)
- `molmetal/reports/wf360b2_pf_all.json` (Wf-360-B2 pre-flight fresh-run at 2026-09-29 21:51 UTC wall 154.12 s)
- `metrics/by_metric/m6_kappa.json` (0.4386 augmented_mean preserved) + `molmetal/reports/wf335m1_chemprop_pll/result.json` (+0.6814 CV-R²)
- `molmetal/reports/wf335m2_r2/result.json` (+0.2227 CV mean / +0.3196 best-axis)
- `metrics/by_metric/metalcytotox_ru_auc.json` (m1 0.7613 1-seed 30-ep)
- `molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json` (m3 V3 combined median −7.782 kcal/mol)
- `metrics/by_metric/docked_pose_rmsd.json` (m15 0.6578 Å)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar

---

**END OF Wf-360-C1 CLOSEOUT** — Owner Hugo Lee + ultracode agent (Wf-360-C1); cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.1+§5.2+§5.3+§5.4; §22.49 INDEX.md entry + §6 AGENTS.md row + this closeout SHIPPED at 2026-09-29 22:00 CST; 0 NEW tests; pre-flight v3 89/94 PASS / 2 FAIL / 3 SKIP / 94 TOTAL PRESERVED at Wf-360-B2 timestamp; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED; 0 GPU launches; pending/ 13 → 13 PRESERVED; completed/ 763 → 770 +7 (5 Wf-360 sub-flows + 1 Wf-360-C1 this entry + 1 cross-link redundancy correction in INDEX.md inventory row).