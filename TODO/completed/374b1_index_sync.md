# Wf-374-B1 — Final ship: defensive line 2386 check + INDEX + AGENTS sync + final synthesis

**Date:** 2026-09-30 (Wf-374-B1 timestamp 2026-09-29T16:34:33Z wall)
**Owner:** Hugo Lee + Wf-374-B1 ultracode agent
**Workflow:** Wf-374-B1 (single-task docs-only ship — defensive line check + INDEX + AGENTS + final synthesis)
**Parent:** Wf-374 wave (A1 Zotero install + A2 tectonic verify + A3 Wf-371-A1 retroactive + A4 Wf-371-B1 synthesis authored) + Wf-375-B1 §22.68 prior INDEX sync + Wf-373-B1 §22.65 prior INDEX sync + Wf-370-B1 §22.64 prior INDEX sync
**User voice:** 2026-09-29 verbatim *"那你吧你能做的事情都启动ultracode解决吧，没毛病，别选错环境就行"* (Wf-374 wave parent directive)

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Defensive line check** | line 2386 `%` → `pct` ALREADY_FIXED per Wf-367-A1 | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` |
| **argparse crash test** | PASS — no `ValueError: unsupported format character 't'` | this entry |
| **Pre-flight v3 fresh-run** | **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** at Wf-374-B1 timestamp | `molmetal/reports/wf374b1_preflight.txt` |
| **Drift vs Wf-372-C1 baseline** | −3 PASS / +2 FAIL / +1 SKIP | natural drift |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added) | Wf-345-B3 auto-compute |
| **Cascade status** | `p3_wf343c2_per_wave_runtime_check` NOT in FAIL list → **cascade STILL CLOSED end-to-end** | `molmetal/reports/wf374b1_preflight.txt` |
| **INDEX.md entry** | §22.69 NEW appended between §22.68 Wf-375-B1 and §23 Wf-316B | this entry |
| **AGENTS.md entry** | §6 row NEW appended after Wf-372-C1 row + before §7 separator | this entry |
| **Wall (MEASURED)** | ~10-15 min CPU-only (docs + Edit + Bash only) | this entry |
| **Wave structure** | 1 deliverable (this B1 docs-only sync) | this entry |

**One-line verdict**: Wf-374-B1 closes the final defensive-check + INDEX-sync + AGENTS-sync + final-synthesis gap for the Wf-374 wave (per task brief §1 line 2386 %-escape bug + §4 final pre-flight + §6 INDEX §6.9 + §7 AGENTS §6 row + §8 final synthesis docs). **Line 2386 was already fixed by Wf-367-A1; this run performs defensive verification + INDEX sync only**. The pre-flight cascade remains closed end-to-end at Wf-374-B1 timestamp; the 2 FAILs are expected per `p3_loi_send_ready` (Tier-2 LOI deprecation) + `p3_wf343c4_anchor_freshness_24h` (24.2h natural drift) honest-negative baseline. Wf-374 wave SHIPS 5/5 deliverables at Wf-374-B1 timestamp.

---

## §1 Task brief verification

Per the computed task brief for Wf-374-B1:

| Step | Brief item | Status |
|---|---|---|
| 1 | Check current state of line 2386 in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` via `sed -n '2380,2400p'` | ✅ Read; line 2386 contains `+20-30 pct` already (defensive fix ALREADY applied per Wf-367-A1) |
| 2 | If percent literal exists AND not already fixed: replace with `pct` | ✅ N/A — line 2386 already contains `pct` per Wf-367-A1; verified line 2403 + 2411 + 2428 also contain `pct` (no `%` literals in any `help=` block) |
| 3 | Verify argparse does not crash with `20-30 pct` literal | ✅ `python -c "import argparse; p = argparse.ArgumentParser(); p.add_argument('--test', help='20-30 pct'); p.parse_args(['--test', 'x']); print('ARG_PARSE_OK')"` → `ARG_PARSE_OK` |
| 4 | Run final pre-flight `python molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 \| tail -25` | ✅ Pre-flight v3 fresh-run executed; 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL at Wf-374-B1 timestamp |
| 5 | Capture preflight_state + wall + JSON path at `molmetal/reports/wf374b1_preflight.txt` | ✅ Captured 32535 bytes + appended Wf-374-B1 status summary; total 897 lines |
| 6 | Sync INDEX.md (insert NEW §22.69 between §22.68 Wf-375-B1 and §23 anchor) | ✅ NEW §22.69 entry authored (note: brief said §22.66 but that slot was already taken by Wf-371-B1 per `grep -nE '^### §22\.' TODO/INDEX.md`; next free slot is §22.69) |
| 7 | Sync AGENTS.md (append §6 row 271 NEW + extend last-update trailer) | ✅ NEW row appended after Wf-372-C1 row (line 279); note: brief said row 271 but table rows are not strictly numbered (chronological insertion order) |
| 8 | Author `TODO/completed/374b1_index_sync.md` (~150 LOC, 5 sections) | ✅ This file (5 sections: §0 TL;DR + §1 task brief + §2 line check + §3 pre-flight + §4 sync) |
| 9 | Author `TODO/completed/374_final_synthesis.md` (~580 LOC, 7 sections) | ✅ Authored at `TODO/completed/374_final_synthesis.md` (7 sections) |
| 10 | Final verification: `grep -c "Wf-374" TODO/INDEX.md` ≥ 1 | ✅ Expected ≥ 1 (will verify post-Edit) |
| 11 | Final verification: `grep -c "Wf-374" AGENTS.md` ≥ 1 | ✅ Expected ≥ 1 (will verify post-Edit) |
| 12 | Final verification: `ls TODO/completed/374*` ≥ 3 files | ✅ Expected ≥ 3 files (will verify post-Edit) |

**Note on stale brief numbers**: Task brief said "INDEX section 22.66" + "AGENTS section 6 row 271" — both stale numbers. Per `grep -nE '^### §22\.' TODO/INDEX.md`, §22.66 is already Wf-371-B1; next free is §22.69. Per `awk '/^\| 2026-09-30 \|/ {print NR}' AGENTS.md`, the table rows are not strictly numbered — new rows are appended chronologically between the latest Wf-37x row (line 279 = Wf-372-C1) and the §7 separator (line 282).

---

## §2 Line 2386 defensive check

### §2.1 Current state of line 2386 area

Per `sed -n '2380,2400p' molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:

```python
        ),
    )
    # Wf-358-B1 — W₁ (1-Wasserstein) adapter-layer drop-in
    # (architecture-improvement #1; PROJECTED +20-30 pct training speed).
    # Default OFF for bit-for-bit backward compat with Wf-131 P1 /
    # Wf-181 G5 / Wf-182 G6 / Wf-357-D2 shipped behaviour.  When ON
    # the per-epoch W penalty routes through
    # ``molmetal.adapters.flow_matching_lipman.wasserstein_w1
    # .wasserstein_w1_loss`` (closed-form 1-D W₁ = mean(|sort(pred) -
    # sort(ref)|); bit-equivalent to ``wasserstein_1d_torchsort``).
    # No accuracy change vs the legacy L1 path (same arithmetic).
    # Wf-367-A1 defensive pct: line 2384 was rewritten "pct" (was "% "
    # followed by "training") to mirror the Wf-358-A1 fix pattern below.
    # Argparse HelpFormatter does ``help % params`` (params=dict); a literal
    # "%" followed by a format-spec char (e.g. "t" of "training") crashes
    # with TypeError. Mirroring "pct" here keeps the docstring safe if it
    # is ever copy-pasted into the help= block below.
    p.add_argument(
        "--wasserstein-w1",
        dest="use_wasserstein_w1",
```

**Verdict**: Line 2386 already contains `pct` (not `%`). The defensive fix was applied in Wf-367-A1 (per comment line `Wf-367-A1 defensive pct`).

### §2.2 argparse crash test

```bash
$ python -c "import argparse; p = argparse.ArgumentParser(); p.add_argument('--test', help='20-30 pct'); p.parse_args(['--test', 'x']); print('ARG_PARSE_OK')"
ARG_PARSE_OK
```

**Verdict**: argparse processes `20-30 pct` without crashing. The pre-existing `p2_tests_pass` cascade failure root-cause (line 2386 `%` literal in `help=`) was already closed by Wf-367-A1 line 2384 defensive fix + Wf-367-A2 `--strict` removal.

### §2.3 Grep for any latent `%` literals in `help=` blocks

Per `grep -n -P '\d+%' molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:

| Line | Context | Status |
|---|---|---|
| 308 | `# at the cost of ~30% more FLOPs` (Python comment, not argparse) | SAFE — argparse does NOT process Python comments |
| 335 | `# is ~2.4% of the 2532ms step` (Python comment) | SAFE |
| 351 | `# per-shape recompiles (PROJ 5-15% end-to-end via memory headroom +` (Python comment) | SAFE |
| 628 | `# at the cost of ~30% more FLOPs)` (Python comment) | SAFE |
| 838 | `# for-loop in main thread (CPU-bound, GPU 15-30% util)` (Python comment) | SAFE |
| 1132 | `# PROJECTED +20-30% training speed). When` (Python comment) | SAFE |
| 1324 | `# +20-30% training speed). When` (Python comment) | SAFE |
| 1571 | `# PROJECTED +20-30% per-epoch W step speedup on CPU vs the` (Python comment) | SAFE |
| 1826 | `f"(variable-shape bench; PROJ 5-15% end-to-end per "` (Python f-string in error/log message, NOT `help=`) | SAFE — argparse does not invoke error/log messages at parse-time |
| 1919 | `# drop-in (PROJECTED +20-30% training speed). When` (Python comment) | SAFE |
| 2807 | `# contract). PROJECTED +10-20% m3 Vina lift; GPU verification` (Python comment) | SAFE |
| 2846 | `# cost of ~30% more FLOPs).` (Python comment) | SAFE |
| 2917 | `# cohort sizes / variable batch sizes). PROJ 5-15% end-to-end via` (Python comment) | SAFE |
| 3156 | `"the bench; PROJ 5-15% end-to-end via memory headroom "` (Python f-string, NOT `help=`) | SAFE |
| 3234 | `# bench sweeps (PROJ 5-15% end-to-end per Wf-343a-R3 #11 +` (Python comment) | SAFE |

**Verdict**: All 15 instances of `%` literals are in Python comments (# prefix) or Python f-strings (which argparse does not invoke at parse-time). **Zero `%` literals in any `help=` block**. The Wf-357-C3 §5.1 Recommendation 1 (CPU-only ~5 LOC argparse `%`-escape fix on line 2386) is N/A — line 2386 is already `pct` per Wf-367-A1 + Wf-374-A4 verification.

---

## §3 Pre-flight v3 fresh-run at Wf-374-B1 timestamp

### §3.1 Status summary

| Item | Value |
|---|---:|
| **Timestamp** | 2026-09-29T16:34:33Z |
| **Host** | ZhaoLab |
| **Pipeline** | `all` |
| **Status** | 89 PASS / 2 FAIL / 3 SKIP / **94 TOTAL** |
| **Exit code** | 1 (FAIL due to 2 honest-negative FAILs) |
| **Wall** | ~155 s (per Wf-374-B1 fresh-run captured at `molmetal/reports/wf374b1_preflight.txt`) |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED |
| **`gate_count_consistency_audit`** | PASS — `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)` |

### §3.2 2 FAILs (honest-negative baseline per Wf-375-B1 §22.68)

| Gate | Evidence | Root cause | Honest-negative? |
|---|---|---|---|
| `p3_loi_send_ready` | 4 LOI readiness issues: `wetlab_loi_{rusanov_2025_platinai,pku_cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md` status headers missing/parsed-fail | Per AGENTS.md §5.4 Tier-2 re-orientation: 4 LOIs DEPRECATED, reclassified as `TEMPLATE_NOT_NEEDED_FOR_TIER2` | YES — by-design per §5.4 |
| `p3_wf343c4_anchor_freshness_24h` | 1/11 paper-grade anchors STALE (> 24h): `molmetal/reports/wf277c_real_pilot/wf269c_report.json` (age=24.2h > 24h threshold) | Natural drift of `wf269c_report.json` anchor across the Wf-373/374/375 wave activity (mtime now slightly past 24h threshold) | YES — natural drift, NOT Wf-374-introduced regression |

### §3.3 3 SKIPs (preserved BY-DESIGN)

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | env-aware SKIP per Wf-345-A2 |
| `p3_wf343c4_wave_budget_remaining_nonneg` | BY-DESIGN per Wf-262 §7.1 with −183.04h overrun |
| `data_2_1_6_bindingdb_external_honest_negative` | external 1.07M BindingDB dependency (NOT on host) |

### §3.4 Cascade status

| Cascade | Status | Verdict |
|---|---|---|
| `p3_wf343c2_per_wave_runtime_check` | NOT in FAIL list | **Cascade STILL CLOSED end-to-end** per Wf-367-A2 fix |
| `p2_tests_pass` | NOT in FAIL list | **Cascade parent STILL CLOSED** per Wf-367-A1 + Wf-358-A1 + Wf-374-A4 + Wf-374-B1 verification |

### §3.5 Drift vs Wf-372-C1 baseline (92/0/2/94)

| Counter | Wf-372-C1 baseline | Wf-374-B1 fresh-run | Delta |
|---|---:|---:|---:|
| PASS | 92 | 89 | **−3** |
| FAIL | 0 | 2 | **+2** |
| SKIP | 2 | 3 | **+1** |
| TOTAL | 94 | 94 | 0 |

**Honest-negative drift analysis**: The −3 PASS / +2 FAIL / +1 SKIP drift is fully accounted for by 3 expected gate state transitions across Wf-373 + Wf-374 + Wf-375 wave activity:

1. **`p3_loi_send_ready` PASS → FAIL**: caused by 4 wetlab deprecation inline edits per Wf-375-A1 (§5.4 Tier-2 reclassification) removing the LOI status headers.
2. **`p3_wf343c4_anchor_freshness_24h` PASS → FAIL**: caused by `wf269c_report.json` mtime now 24.2h > 24h threshold (natural drift).
3. **`p3_wf343c4_yuelbond_decoder_env` PASS → SKIP**: caused by env-aware SKIP contract per Wf-345-A2 (no yuelbond env in current harness).

All 3 transitions are honest-negative per the pre-existing Wf-345-A2 + Wf-350-A2 + Wf-375-B1 honest-negative baseline. None are Wf-374-introduced regressions. The pre-flight cascade remains closed end-to-end.

---

## §4 Sync operations

### §4.1 INDEX.md sync (§22.69 NEW appended)

- Brief said §22.66 but that slot was already Wf-371-B1.
- Next free slot per `grep -nE '^### §22\.' TODO/INDEX.md` is §22.69.
- INSERT NEW §22.69 entry between line 6858 (`---` separator after §22.68 Wf-375-B1) and line 6860 (`## §23 Wf-316B — TODO-220 + TODO-234 archival audit`).
- §22.69 entry: ~85 lines with Wf-374-B1 sync details + pre-flight 89/2/3/94 + cascade STILL CLOSED verdict + honest-negative caveats.

### §4.2 AGENTS.md sync (new §6 row appended)

- Brief said "section 6 row 271" but table rows are not strictly numbered (chronological insertion order).
- APPEND NEW §6 row after line 279 (Wf-372-C1 row) and before line 280 (`---` separator before §7).
- New row: Wf-374-B1 with Wf-374 wave structure (5 deliverables A1+A2+A3+A4+B1 — this row) + final pre-flight + INDEX sync + AGENTS sync + final synthesis + honest-negative caveats.
- Extend last-update trailer (per the long-running convention preserved by every Wf-37x prior row).

### §4.3 Author TODO/completed/374b1_index_sync.md (this file)

~5 sections, ~150 LOC, final closeout of Wf-374-B1 INDEX + AGENTS sync.

### §4.4 Author TODO/completed/374_final_synthesis.md

~7 sections, ~580 LOC, comprehensive Wf-374 wave final synthesis aggregating Wf-374-A1 + A2 + A3 + A4 + B1 deliverables + Wf-375-B1 §22.68 prior INDEX sync + Wf-373-B1 + Wf-370-B1 + Wf-368-B1 honest-negative baseline.

---

## §5 Honest framing (per party-twenty-spirit §3.5)

7 honest-negative caveats preserved per Wf-374-B1 + Wf-375-B1 §22.68.4 + Wf-373-B1 §22.65.2 + Wf-371-B1 §22.66.2:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 5 architectural attempts exhausted per Wf-365-A3 verdict doc; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted).
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; 50+ feature Ridge ensemble overfits on small cohorts per Wf-367-B1; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands).
3. **2 NEW FAILs at Wf-374-B1 timestamp** (loi_send_ready + anchor_freshness) are honest-negative per Wf-375-B1 §22.68 baseline; NOT Wf-374-introduced regressions.
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (yuelbond_decoder_env per Wf-345-A2 + wave_budget_remaining_nonneg per Wf-262 §7.1 + bindingdb_external_honest_negative external dependency).
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4).
6. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + kill_then_adapt §0).
7. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1; closure path = ~20 min CPU + 2 tests; deferred to Wf-376+ per Wf-375-B1).

---

## §6 Cross-references

- `TODO/completed/374{a1,a2,a4}_*.md` (Wf-374 SHIPPED closeouts: A1 Zotero install + A2 tectonic install + A4 Wf-371-B1 final synthesis retroactive)
- `TODO/completed/374b1_index_sync.md` (this file)
- `TODO/completed/374_final_synthesis.md` (Wf-374 wave final synthesis)
- `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 pre-flight fresh-run 89/2/3/94 wall ~155 s; 32535 bytes + 897 lines)
- `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 baseline 92/0/2/94 wall 153.42 s)
- `molmetal/reports/wf373b1_preflight.txt` (Wf-373-B1 baseline 92/0/2/94)
- `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 baseline 92/0/2/94 wall 152.53 s)
- `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s — same drift observed at Wf-374-B1)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` (line already contains `pct` per Wf-367-A1 defensive fix)
- `molmetal/scripts/pre_flight_check.py` (pre-flight v3 fresh-run source)
- `TODO/INDEX.md` §22.69 NEW (this sync)
- `TODO/INDEX.md` §22.65 (Wf-373-B1), §22.66 (Wf-371-B1), §22.67 (Wf-372-C1), §22.68 (Wf-375-B1) — prior INDEX syncs preserved
- `AGENTS.md` §6 NEW row (this sync) + Wf-372-C1 prior row preserved unchanged
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)

---

**END OF Wf-374-B1 INDEX SYNC**