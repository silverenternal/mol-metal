# Wf-362-B1 — docs-only INDEX + AGENTS sync

**Workflow**: Wf-362 (documentation gaps + missing data fixes wave) — B1 docs-only sync phase.
**Date UTC**: 2026-09-29 ~14:30 UTC (post A1+A2+A3 closeouts)
**Wall**: ~10 min CPU-only MEASURED (PROJECTED 10-15 min)
**GPU launches**: 0 (CPU-only docs sync per AGENTS.md §5.1)
**Author**: Wf-362-B1 docs-only sync agent (party-twenty-spirit §3.5 honest framing)
**Status**: SHIPPED

---

## §0 What this sync ships

This is a docs-only INDEX + AGENTS + final-synthesis sync. No `molmetal/` source code modifications, no `paper/*` touches (per AGENTS.md §5.3 TODO-84 active), no pre-flight schema-breaking change, no GPU launches.

| Action | Detail |
|---|---|
| `TODO/INDEX.md` | Appended **§22.51** (NEW ~96 lines) between §22.50 (Wf-358-D2) and §23 (Wf-316B archival audit) |
| `AGENTS.md` | Appended **§6 +1 NEW row** for Wf-362 between the Wf-360 row and §7 Critical scripts reference |
| `TODO/completed/362_final_synthesis.md` | NEW — 5-section final synthesis with MEASURED-vs-PROJECTED discipline |
| `TODO/completed/362b1_index_sync.md` | NEW — this file |
| **Tests added** | **0 NEW** (docs-only sync; 19 NEW tests already shipped by Wf-362-A1+A2+A3) |
| **Pre-flight re-run** | YES — `pre_flight_check.py --pipeline all` at Wf-362-B1 timestamp; wall 154.55 s; **90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-353-B1 baseline |

---

## §1 Pre-flight v3 fresh-run state (post-Wf-362-A1+A2+A3)

Per `molmetal/reports/wf362b1_preflight.txt` at Wf-362-B1 timestamp 2026-09-29 14:25:00+0000 UTC, wall **154.55 s**:

| Metric | Value | Delta vs Wf-358-D1 baseline |
|---|---:|---|
| Total gates | **94** | PRESERVED (no new gates added by Wf-362) |
| PASS | **90** | PRESERVED |
| FAIL | **2** | PRESERVED (pre-existing) |
| SKIP | **2** | PRESERVED (pre-existing) |
| Wall | **154.55 s** | ~3 s longer than Wf-358-D1 149.99 s |
| `gate_count_consistency_audit` | **PASS** | `actual=94 == expected=94 (auto-computed via Wf-345-B3)` |
| EXPECTED_GATE_COUNTS["all"] | **94 PRESERVED** | No new pre-flight gates added by Wf-362 |

The 2 FAILs are PRE-EXISTING per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline. NOT Wf-362-introduced. Would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`.

The 2 SKIPs are PRE-EXISTING per Wf-350-A2 §5 honest-negative baseline:
1. `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1; −183.04h overrun)
2. `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

---

## §2 INDEX.md §22.51 — appended

**Location**: Between §22.50 (Wf-358-D2) and §23 (Wf-316B archival audit).

**Section heading**: `### §22.51 Wf-362 documentation gaps + missing data fixes wave (2026-09-29 — Wf-362-B1 docs-only INDEX + AGENTS + synthesis, this entry)`

**Contents** (6 subsections):
- **Top**: Skill alignment (party-twenty-spirit §3 + AGENTS.md §1 standing rules) + user directive verbatim + Wf-362 wave structure (3 SHIPPED sub-flows + 1 B1 docs sync = 4 deliverables table)
- **§22.51.1**: Net effect on TODO/ inventory (post-Wf-362-B1 INDEX sync)
- **§22.51.2**: Wall + cost accounting
- **§22.51.3**: Honest framing per party-twenty-spirit §3.5 (5 honest-negative caveats preserved)
- **§22.51.4**: Per-agent close-out summary (table: A1+A2+A3+B1 with metric JSON + action + NEW tests + wall)
- **Cross-link**: Sources + prior §22 syncs (PRESERVED) + AGENTS.md §6 +1 row

**Line count delta**: `TODO/INDEX.md` 5721 → **5817** (+96 lines).

---

## §3 AGENTS.md §6 — +1 NEW row

**Location**: Between the Wf-360 row and §7 Critical scripts reference.

**Row contents** (single-row table entry preserving the Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360 ultracode-family row format):

- 2026-09-29 date stamp
- Wf-362 documentation gaps + missing data fixes wave header
- User directive verbatim *"这些问题都要解决，你可以启动多个ultracode去修"*
- 3 SHIPPED sub-flows A1+A2+A3 (with metric JSON + action + NEW tests + wall) + Wf-362-B1 docs sync (this row)
- 4/4 deliverables SHIPPED + 19 NEW tests PASS (2+15+2=19) + 3 NEW test files + 3 metric JSONs refreshed + ~250 LOC NEW docs + tests + 0 GPU launches + 0 `paper/*` touches
- Pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL at Wf-362-B1 timestamp 2026-09-29 14:25:00+0000 UTC wall 154.55 s
- `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)`
- EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-362)
- Combined wall PROJECTED 25-45 min CPU-only across 4 deliverables per task brief (actual ~15-20 min, ~50% under PROJECTED)
- Honest framing per party-twenty-spirit §3.5: 5 honest-negative caveats preserved
- Cross-link to Wf-362 closeouts + §22.51 INDEX sync + final synthesis

**Line count delta**: `AGENTS.md` 325 → **326** (+1 line — the Wf-362 row is a single-line table entry preserving the row format).

---

## §4 Final synthesis — `TODO/completed/362_final_synthesis.md`

**Location**: `TODO/completed/362_final_synthesis.md` (NEW, 5 sections + 0 TL;DR + 1 Deliverables + 2 MEASURED + 3 NOT SHIP / deferred + 4 Cross-references + 5 Honest framing).

**Format**: Per task brief template + Wf-358-D2 / Wf-360-C1 / Wf-359-C1 / Wf-357-D3 prior synthesis row format.

**Key sections**:
- **§0 TL;DR**: Per Wf-359-A3 metric-ledger audit, three metric JSON files were flagged STALE / carrying documentation gaps. Wf-362 launched three ultracode agents to close each gap. 3 metric JSONs refreshed, 19 NEW tests PASS, 0 GPU launches, pre-flight v3 90/2/2/94 PRESERVED.
- **§1 Deliverables**: 4-row table (A1+A2+A3+B1) with closeout file paths + metric JSONs + actions + NEW tests + wall.
- **§2 MEASURED**: Pre-flight state + per-agent test pass-rates + per-agent JSON delta + inventory delta.
- **§3 NOT SHIP / deferred**: m3 Vina V3 combined fully-MEASURED + other deferred items (preserved honest-negatives).
- **§4 Cross-references**: Wf-362 closeouts + pre-flight + report artifacts + source artifacts + test files + INDEX + AGENTS + prior INDEX syncs + standing directives.
- **§5 Honest framing**: 4 subsections (SHIPPED today / NOT SHIPPED today / MEASURED-vs-PROJECTED discipline / Wall + cost accounting).

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 SHIPPED today (Wf-362-B1 sync)

- 1 docs-only INDEX entry appended (§22.51 — 96 lines)
- 1 docs-only AGENTS.md §6 row appended (1 line)
- 1 final synthesis written (`TODO/completed/362_final_synthesis.md`)
- 1 INDEX + AGENTS sync written (`TODO/completed/362b1_index_sync.md` — this file)
- 0 NEW tests (docs-only sync; 19 NEW tests already shipped by Wf-362-A1+A2+A3)
- 0 GPU launches performed
- 0 `paper/*` touches performed
- 0 `molmetal/` source code modifications (only docs + report file + this closeout file)

### §5.2 NOT SHIPPED today

- 2 pre-existing FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) NOT Wf-362-introduced (preserved per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline)
- 2 pre-existing SKIPs (`p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`) NOT Wf-362-introduced (preserved per Wf-350-A2 §5 honest-negative baseline)
- m6_kappa augmented_mean 0.4386 < 0.6 PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL (gate NOT lifted; Wf-362-A1 records only the metalcytotox_combined CEILING_BREAK, not the aug_mean gate)
- m2 platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP PRESERVED (Wf-362-A1 records only 1/3 cohorts CEILING_BREAK)
- m3 V3 combined −7.782 kcal/mol HONEST_NEGATIVE_VINA_RESOLVED but per-pocket delta constant −4.5 (path to fully-MEASURED requires GPU bench; BLOCKED)
- m15 PoseBusters validity SKIPPED graceful (preserved as honest-negative on validity NOT on accuracy)
- m1 Ru AUC 0.7613 (1-seed) Krasnov 0.86 gate NOT YET HIT

### §5.3 MEASURED-vs-PROJECTED discipline

Every numeric value in this sync document traces to a JSON / report path on disk:

- 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL → `molmetal/reports/wf362b1_preflight.txt` (Wf-362-B1 timestamp)
- 19/19 NEW tests PASS → combined `pytest` output above in `TODO/completed/362_final_synthesis.md` §2.2
- 3 metric JSONs refreshed → `metrics/by_metric/metalcytotox_combined_r2.json` + `metrics/by_metric/vina_kcal_per_mol.json` + `metrics/by_metric/docked_pose_rmsd.json`
- Wall ~10 min MEASURED → elapsed wall from Wf-362-B1 start to this file write
- +96 INDEX lines + +1 AGENTS line → `wc -l TODO/INDEX.md` (5721 → 5817) + `wc -l AGENTS.md` (325 → 326)

Zero tolerance for fabricated / aspirational numbers in main results.

### §5.4 Wall + cost accounting

| Item | Value |
|---|---|
| MEASURED wall (Wf-362-B1 only) | ~10 min CPU-only |
| PROJECTED wall (task brief) | 10-15 min CPU-only |
| MEASURED vs PROJECTED | on-target |
| GPU launches | 0 (MEASURED) |
| `paper/*` touches | 0 (MEASURED) |
| Net docs change | +96 lines INDEX.md + +1 line AGENTS.md + 2 NEW docs (this file + final synthesis) |

---

## §6 Cross-references

- `TODO/completed/362_final_synthesis.md` (Wf-362-B1 final synthesis — 5 sections, MEASURED-vs-PROJECTED discipline preserved)
- `TODO/completed/362{a1,a2,a3}_*.md` (3 Wf-362 sub-flow closeouts: A1 R² JSON update + A2 Vina JSON refresh + A3 RMSD JSON refresh)
- `TODO/completed/362b1_index_sync.md` (this file — Wf-362-B1 INDEX + AGENTS + final synthesis sync)
- `TODO/INDEX.md` §22.51 (NEW ~96 lines appended between §22.50 and §23)
- `AGENTS.md` §6 +1 NEW row (Wf-362 entry — this B1 sync)
- `molmetal/reports/wf362b1_preflight.txt` (Wf-362-B1 pre-flight fresh-run wall 154.55 s; 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL)
- `metrics/by_metric/metalcytotox_combined_r2.json` (Wf-362-A1 NEW `wf335m2_ceiling_break` block)
- `metrics/by_metric/vina_kcal_per_mol.json` (Wf-362-A2 NEW `wf333_v3_vina_combined_smoke` entry + STALE→FRESH)
- `metrics/by_metric/docked_pose_rmsd.json` (Wf-362-A3 NEW `redesign_wf341t1` block + STALE→FRESH)
- `molmetal/tests/test_wf362a1_r2_json_update.py` (Wf-362-A1 NEW 2 tests, 0.34 s wall)
- `molmetal/tests/test_wf362a2_vina_json_refresh.py` (Wf-362-A2 NEW 15 tests, 0.34 s wall)
- `molmetal/tests/test_wf362a3_rmsd_json_refresh.py` (Wf-362-A3 NEW 2 tests, 0.31 s wall)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar
- §22.27 (Wf-335-SYN 62 gate baseline) + §22.41 (Wf-353-C2 INDEX sync prior) + §22.46 (Wf-356-C2 INDEX sync prior) + §22.47 (Wf-357-D3 INDEX sync prior) + §22.48 (Wf-359-C1 INDEX sync prior) + §22.49 (Wf-360-C1 INDEX sync prior) + §22.50 (Wf-358-D2 INDEX sync prior)

---

**END OF Wf-362-B1 SYNC** — INDEX §22.51 appended (+96 lines) + AGENTS.md §6 +1 row (+1 line) + final synthesis + this sync file; 0 NEW tests (docs-only); 0 GPU launches; 0 `paper/*` touches; pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-353-B1 baseline; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.
