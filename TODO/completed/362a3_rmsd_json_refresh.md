# Wf-362-A3 — docked_pose_rmsd.json STALE → FRESH refresh

**Date:** 2026-09-29
**Author:** Wf-362-A3 subagent (party-twenty-spirit §3.5 honest framing)
**Status:** SHIPPED (2/2 NEW tests PASS, CPU-only)
**Wall budget:** PROJECTED 5-10 min CPU-only; actual ~0.5 min (JSON edit + 2 tests + doc write)
**GPU budget:** 0 (no GPU launch — refresh is JSON-edit + test only)
**Parent:** Wf-359-A3 finding (docked_pose_rmsd.json STALE ~29h)

---

## §0 TL;DR

Wf-362-A3 closes the Wf-359-A3 honest-negative finding by refreshing
`metrics/by_metric/docked_pose_rmsd.json` from STALE (~29h, >24h threshold)
to FRESH (~0.003h). The refresh:

| Action | Detail |
|---|---|
| Added `redesign_wf341t1` field | Wf-341-T1 integrated m15 redesign (Kabsch+Hungarian + 2.0 Å SOTA gate + PoseBusters graceful skip + meeko 0.8.0 SMILES→PDBQT) |
| Set `updated_at: 2026-09-29` | Top-level timestamp for STALE gate refresh |
| Updated `note` field | Reflects Wf-341-T1 redesign + predecessor Wf-246-B |
| Wrote 2 NEW tests | `molmetal/tests/test_wf362a3_rmsd_json_refresh.py` |
| Documented closeout | This file |

**Refresh headline (MEASURED, per `party-twenty-spirit` §3.5):**

| Metric | Before | After |
|---|---|---|
| File age | **29.99 h (STALE)** | **~0.003 h (FRESH)** |
| Has `redesign_wf341t1` | **No** | **Yes** |
| `updated_at` (top-level) | absent | **2026-09-29** |
| Tests added | — | **2/2 PASS** |

**Wf-341-T1 MEASURED values (preserved verbatim from Wf-341-T1 doc):**

| Metric | Value | Source |
|---|---|---|
| `aggregate_median_best_rmsd_angstrom` | **0.6578 Å** | `TODO/completed/341t1_m15_rmsd.md` §0 |
| `n_pockets_passed_2_0a` | **5/5** | `TODO/completed/341t1_m15_rmsd.md` §0 |
| `aggregate_success_rate_2_0a` | **1.0** | `TODO/completed/341t1_m15_rmsd.md` §0 |
| `difficulty_counts` | EASY=1, MEDIUM=3, HARD=1 | `TODO/completed/341t1_m15_rmsd.md` §0 |
| `posebusters_validity` | **SKIPPED** (graceful, not installed) | `TODO/completed/341t1_m15_rmsd.md` §0 |
| `m15_status` | **MEASURED** | `TODO/completed/341t1_m15_rmsd.md` §0 |

---

## §1 What ships in this Wf

### §1.1 Refreshed JSON: `metrics/by_metric/docked_pose_rmsd.json`

Added the `redesign_wf341t1` block (modeled on the existing `redesign_wf246b` block
to preserve backward-compatible schema):

```json
"redesign_wf341t1": {
  "workflow": "Wf-341-T1 TODO-134 m15 Docking RMSD redesign (integrated)",
  "date": "2026-09-29",
  "purpose": "Integrated m15 redesign module: Kabsch+Hungarian + 2.0A SOTA gate + PoseBusters validity (graceful skip) + per-pocket bench aggregation + meeko 0.8.0 SMILES->PDBQT",
  "aggregate_median_best_rmsd_angstrom": 0.6578,
  "n_pockets_passed_2_0a": 5,
  "n_pockets_measured": 5,
  "aggregate_success_rate_2_0a": 1.0,
  "difficulty_counts": {"EASY": 1, "MEDIUM": 3, "HARD": 1},
  "posebusters_validity": "SKIPPED",
  "posebusters_validity_note": "graceful skip — posebusters not installed; preserved as honest-negative",
  "source": "TODO/completed/341t1_m15_rmsd.md",
  "updated_at": "2026-09-29",
  "wall_budget": "< 10 min CPU-only (probe + integration + test run + doc write)",
  "gpu_budget": 0,
  "tests_passed": "4/4 NEW (molmetal/tests/test_wf341t1_m15_rmsd.py)",
  "status": "MEASURED",
  "predecessor_redesign": "redesign_wf246b"
}
```

Also updated:
- `updated_at` (top-level): absent → `"2026-09-29"`
- `note` (top-level): updated to reflect Wf-341-T1 + predecessor Wf-246-B

### §1.2 New tests: `molmetal/tests/test_wf362a3_rmsd_json_refresh.py`

| # | Test | What it validates |
|---:|---|---|
| 1 | `test_docked_pose_rmsd_json_includes_wf341t1_redesign_field` | JSON contains `redesign_wf341t1` block with workflow/source/updated_at/status fields |
| 2 | `test_docked_pose_rmsd_json_median_5_of_5_matches_wf341t1_source` | Median 0.6578 + 5/5 pockets pass + difficulty counts + PoseBusters SKIPPED + cross-check against top-level aggregate_median_across_pockets + Wf-341-T1 doc exists on disk |

**Test execution output (verbatim, 2026-09-29):**

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 2 items

molmetal/tests/test_wf362a3_rmsd_json_refresh.py::test_docked_pose_rmsd_json_includes_wf341t1_redesign_field PASSED [ 50%]
molmetal/tests/test_wf362a3_rmsd_json_refresh.py::test_docked_pose_rmsd_json_median_5_of_5_matches_wf341t1_source PASSED [100%]

========================= 2 passed, 1 warning in 0.31s =========================
```

---

## §2 Honest framing per `party-twenty-spirit` §3.5

- **STALE → FRESH**: file age went from 29.99h (STALE per >24h threshold) to
  ~0.003h (FRESH). The `updated_at: 2026-09-29` top-level marker makes the
  refresh explicit for downstream STALE gates.
- **MEASURED values**: all numeric values in this doc trace to either
  - `metrics/by_metric/docked_pose_rmsd.json` (the refreshed file itself)
  - `TODO/completed/341t1_m15_rmsd.md` (the Wf-341-T1 source doc)

  No fabricated / aspirational numbers. PoseBusters SKIPPED preserved as
  honest-negative (graceful skip, not installed).
- **PROJECTED vs MEASURED**:
  - Wall budget: **PROJECTED 5-10 min** (CPU-only); actual ~0.5 min
    (JSON edit + 2 tests + doc write)
  - GPU budget: **0** (MEASURED — no GPU launch performed)

---

## §3 Predecessor / successor chain

| Workflow | Role | Output |
|---|---|---|
| **Wf-179 BD-9** | Original m15 RMSD Phase 2 (5 pockets × 3 seeds × exhaustiveness=16) | `docked_pose_rmsd.json` first MEASURED ship |
| **Wf-246-B** | Per-pocket std + cross-pocket aggregate + difficulty flag | `redesign_wf246b` block |
| **Wf-336-X4** | Vina meeko alternative integration | PoseBusters graceful skip path |
| **Wf-341-T1** | Integrated m15 redesign (Kabsch+Hungarian + 2.0 Å SOTA gate + PoseBusters + per-pocket bench + meeko) | 4/4 NEW tests PASS at `molmetal/tests/test_wf341t1_m15_rmsd.py` |
| **Wf-359-A3** | Identified STALE ~29h finding | HONEST_NEGATIVE_STALE_FINDING |
| **Wf-362-A3 (this)** | Refresh `docked_pose_rmsd.json` from STALE → FRESH | `redesign_wf341t1` block + 2/2 NEW tests PASS |

---

## §4 Backward compatibility

The refresh adds new fields (`redesign_wf341t1`, `updated_at`) without
removing or renaming any existing fields. All previous consumers that read
`measurements`, `aggregate_median_across_pockets`, `redesign_wf246b`,
`per_pocket_with_difficulty`, `cross_pocket_aggregate`, or
`difficulty_counts` continue to work unchanged.

The `redesign_wf341t1` block is a sibling to `redesign_wf246b` — both are
preserved side-by-side for full provenance trail.

---

## §5 Remaining work for Wf-362+

| Item | Status | Notes |
|---|---|---|
| PoseBusters install | **DEFERRED** | not in scope; graceful skip preserved as honest-negative per `TODO/completed/336x4_vina_meeko_alt.md` |
| m15 redesign GPU re-bench | **DEFERRED** | bench is CPU-only; Wf-341-T1 already confirmed 5/5 PASS on CPU |
| Wf-362 other A* agents | **in flight** | per user directive "这些问题都要解决，你可以启动多个ultracode去修" |

---

**END OF Wf-362-A3 CLOSEOUT** — JSON refreshed, 2/2 tests PASS, CPU-only, ~0.5 min actual wall.
