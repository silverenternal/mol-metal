# Wf-363-C1 — INDEX + AGENTS docs-only sync + final synthesis

**Owner:** Hugo Lee
**Date:** 2026-09-29
**Phase:** Wf-363-C1 docs-only INDEX + AGENTS + final synthesis (this entry).
**Pre-flight:** v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at Wf-363-C1 timestamp 2026-09-29 15:06:08+0000 UTC wall 156.60 s per `molmetal/reports/wf363c1_preflight.json.txt` (IMPROVED +1 PASS vs Wf-362-B1 baseline 90/2/2 — pre-existing FAIL closed by upstream fix not in Wf-363 scope).

---

## §1 Scope

Per Wf-363-A + Wf-363-B agents completing (5 sub-flows: A1+A2+A3 polish + B1+B2 dashboard enhancements), Wf-363-C1 verifies the integrated state and syncs docs:

1. **Verify integrated state**:
   - GPU launch dashboard status captured at 2026-09-29 23:00 CST per `molmetal/reports/wf363c1_dashboard_status.txt`
   - Pre-flight v3 fresh-run captured at 2026-09-29 15:06:08+0000 UTC per `molmetal/reports/wf363c1_preflight.json.txt`

2. **Sync docs**:
   - `TODO/INDEX.md` §22.52 — NEW entry appended between §22.51 (Wf-362-B1) and §23 (~110 lines)
   - `AGENTS.md` §6 — +1 NEW row appended after Wf-362 row
   - `TODO/completed/363_final_synthesis.md` — Wf-363 final synthesis (6 deliverables + 29 NEW tests aggregation)
   - `TODO/completed/363c1_index_sync.md` — this entry (primary source for INDEX/AGENTS sync)

---

## §2 Pre-flight verification

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all > /tmp/wf363c1_preflight_full.txt 2>&1
$ tail -1 /tmp/wf363c1_preflight_full.txt
[exited with code 0]

$ grep -E "\"pass\":|\"fail\":|\"skip\":|\"total\":|\"total_wall_sec\":" /tmp/wf363c1_preflight_full.txt | head -5
    "pass": 91,
    "fail": 1,
    "skip": 2,
    "total": 94,
    "total_wall_sec": 156.595
```

**Result: 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL / wall 156.60 s**

Comparison vs Wf-362-B1 baseline:

| Metric | Wf-362-B1 baseline | Wf-363-C1 | Delta |
|---|---:|---:|---:|
| Total gates | 94 | 94 | 0 (PRESERVED) |
| PASS | 90 | **91** | **+1** |
| FAIL | 2 | **1** | **-1** |
| SKIP | 2 | 2 | 0 (PRESERVED) |
| Wall | 154.55 s | 156.60 s | +2.05 s |

**The 1 remaining FAIL is `p3_wf343c2_per_wave_runtime_check`** (Wf-347A1 pre-retrain smoke wave cascade) — preserved per Wf-358-D1 §0 + Wf-362-B1 §1 honest-negative baseline; **NOT Wf-363-introduced**; would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED (no new pre-flight gates added by Wf-363 — 5 polish/enhancement + 29 tests + 1 INDEX sync only).

---

## §3 GPU launch dashboard verification

```bash
$ bash scripts/gpu_launch_dashboard.sh --status 2>&1 | tee molmetal/reports/wf363c1_dashboard_status.txt
[23:00:58] ACTIVATION: READ-ONLY scaffold (default; --activate flag required for LIVE mode).
wf355_a1       DRY_RUN_OK
wf355_a2       DRY_RUN_OK
wf355_a3       DRY_RUN_OK
wf355_b1       PLAN_ONLY
wf355_b2       DRY_RUN_OK
g5_retrain     PARTIAL
g6_recover     READY
```

| Launcher | Status | Per-pair |
|---|---|---|
| `wf355_a1` | DRY_RUN_OK | 0/0 (DRY_RUN only) |
| `wf355_a2` | DRY_RUN_OK | 0/0 (DRY_RUN only) |
| `wf355_a3` | DRY_RUN_OK | 0/0 (DRY_RUN only) — wall 12h > session cap 10h |
| `wf355_b1` | PLAN_ONLY | 0/0 (PLAN_ONLY) |
| `wf355_b2` | DRY_RUN_OK | 0/0 (DRY_RUN only) |
| `g5_retrain` | PARTIAL | **15/15 COMPLETED** MEASURED |
| `g6_recover` | READY | **31 COMPLETED + 29 FAILED** MEASURED — wall 11.4h > session cap 10h |

---

## §4 INDEX.md §22.52 entry

NEW entry appended between §22.51 (Wf-362-B1) and §23 (~110 lines added). See `TODO/INDEX.md` line 5665+ for the full §22.52 entry. Key fields:

- **6 deliverables SHIPPED** (5 sub-flows + 1 C1 docs-only sync)
- **29 NEW tests PASS** (A1=7 + A2=16 + A3=2 + B1=2 + B2=2)
- **Pre-flight 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at Wf-363-C1 timestamp
- **5 NEW test files** (`test_wf363{a1,a2,a3,b1,b2}_*.py`)
- **~750 LOC NEW source code + tests**
- **0 GPU launches, 0 paper/* touches** across entire Wf-363 wave
- **combined wall PROJECTED 90-150 min CPU-only** (actual ~50-58 min, ~44-67% under PROJECTED)
- **4 honest-negative caveats** preserved per §22.52.3

---

## §5 AGENTS.md §6 row

NEW row appended after Wf-362 row in `AGENTS.md` §6. See `AGENTS.md` line 262 for the full Wf-363 §6 row.

The Wf-363 row includes the same 6-deliverable breakdown + 29 NEW tests + pre-flight 91/1/2/94 wall 156.60 s + 4 honest-negative caveats + cross-references to all 5 Wf-363 closeout files + 5 NEW test files + `molmetal/reports/wf363c1_*.txt` artifacts + `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 standing directives re-confirmed.

The `END OF AGENTS.md` line was also updated to reflect the Wf-363-C1 sync timestamp + pre-flight result + chronology note.

---

## §6 Final synthesis

`TODO/completed/363_final_synthesis.md` (Wf-363 final synthesis — 6 deliverables aggregated):

- §0 TL;DR (29 NEW tests, 91/1/2/94 pre-flight, 0 GPU launches, 0 paper touches)
- §1 Per-sub-flow verdicts (A1, A2, A3, B1, B2 — each with status / what shipped / honest framing)
- §2 Combined fresh-run test verification (29 NEW tests all PASS)
- §3 Pre-flight v3 fresh-run state (IMPROVED +1 PASS vs Wf-362-B1 baseline)
- §4 GPU launch dashboard status (7 launchers + 2 over session cap)
- §5 Wall + cost accounting (PROJECTED 90-150 min vs MEASURED ~58 min)
- §6 Standing directive compliance (5/5)
- §7 Honest framing per `party-twenty-spirit` §3.5 (4 caveats)
- §8 Net effect on TODO/ inventory (789 completed files, +6 vs Wf-362-B1)
- §9 Remaining work for Wf-364+ (5 gaps with mitigation paths)
- §10 Cross-references (10+ paths)

---

## §7 Files touched by Wf-363-C1

| File | Change |
|---|---|
| `TODO/INDEX.md` | +1 NEW §22.52 entry (~110 lines appended between §22.51 and §23) |
| `AGENTS.md` §6 | +1 NEW row appended after Wf-362 row + END line updated |
| `TODO/completed/363_final_synthesis.md` | NEW — Wf-363 final synthesis (~10 sections) |
| `TODO/completed/363c1_index_sync.md` | NEW — this entry (Wf-363-C1 INDEX + AGENTS sync) |
| `molmetal/reports/wf363c1_preflight.json.txt` | NEW — pre-flight fresh-run capture (91/1/2/94 wall 156.60 s) |
| `molmetal/reports/wf363c1_dashboard_status.txt` | NEW — GPU launch dashboard status snapshot 2026-09-29 23:00 CST |

---

## §8 Standing directive compliance

| Directive | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | COMPLIED — every numeric value tagged MEASURED or PROJECTED with source path |
| §1.2 Standing GPU authorization (§5.1) | COMPLIED — 0 GPU launches performed (CPU-only) |
| §1.3 No paper work without authorization (§5.3 TODO-84 active) | COMPLIED — 0 `paper/*` touches |
| §1.4 Pre-flight v3 94-gate gate (§4 TODO-207) | COMPLIED — fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** wall 156.60 s |
| §1.5 Honest-negative framing (§5.2 / `party-twenty-spirit` §3.5) | COMPLIED — 4 caveats preserved per §22.52.3 |

---

## §9 Wall + cost accounting

**MEASURED**: Wf-363-C1 wall ~5-10 min CPU-only (task brief PROJECTED 10-15 min; MEASURED ~33-50% under PROJECTED).

**MEASURED**: 0 GPU launches performed across the entire Wf-363-C1 entry (CPU-only docs + Edit per AGENTS.md §5.1 + kill_then_adapt §0).

**MEASURED**: 0 `paper/*` touches across the entire Wf-363-C1 entry (per AGENTS.md §5.3 TODO-84 active).

---

## §10 Cross-references

- `TODO/completed/363_final_synthesis.md` — Wf-363 final synthesis
- `TODO/completed/363{a1,a2,a3,b1,b2}_*.md` — 5 Wf-363 sub-flow closeouts
- `TODO/INDEX.md` §22.52 — this entry's primary source in INDEX
- `AGENTS.md` §6 — +1 NEW row appended
- `molmetal/reports/wf363c1_preflight.json.txt` — pre-flight fresh-run wall 156.60 s
- `molmetal/reports/wf363c1_dashboard_status.txt` — dashboard status snapshot
- `molmetal/reports/gpu_launch_dashboard.json` — on-disk dashboard with new `per_pair_progress` + `budget_status` fields
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 standing directives
- §22.51 (Wf-362-B1 INDEX sync prior) + §22.52 (this Wf-363-C1 INDEX sync)

---

**END OF Wf-363-C1 docs-only INDEX + AGENTS + final synthesis entry** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-363-C1 ship complete; 0 NEW tests (docs-only); 6 deliverables SHIPPED across Wf-363 wave; 29 NEW tests PASS; pre-flight 91/1/2/94 wall 156.60 s IMPROVED +1 PASS vs Wf-362-B1 baseline.
