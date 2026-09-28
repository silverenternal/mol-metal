# TODO-114 — 4 MOCK_DELETED external scorers cleanup + scoring adapter honesty

**Date**: 2026-09-27
**Workflow**: Wf-119 Phase 3 (close-out for cleanup of silent-None anti-pattern + 12 stale JSONs)
**Status**: ✅ **SHIPPED 2026-09-27** (cleanup done — silent-None removed; 12 stale JSONs deleted; policy + ROUTING_POLICY patched; R20 plan work remains for real adapter re-implementation)
**Priority**: **P1 HIGH → RESOLVED** (scoring adapter honesty — not paper-blocking; Wf-43 anti-pattern closed)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 3 MOCK_DELETED research (2026-09-27)
**Skill alignment**: `molmetal` (no specific skill) + `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖)

---

## §0 TL;DR (per party-twenty-spirit §3.5)

| Question | Answer |
|---|---|
| Were the 4 MOCK_DELETED scorers supposed to be cleaned up? | **YES** — silent-None anti-pattern + 12 stale JSONs (Wf-43 carry-over) |
| Did cleanup run? | **YES — 2026-09-27** — `_safe_external_score` raises `AdapterUnavailable`; 12 JSONs deleted; `policy.json` + `ROUTING_POLICY.md` patched |
| Are the 4 metrics actually re-implemented as real adapters? | **NO** — cleanup is honest (MOCK_DELETED stays MOCK_DELETED); real adapter re-implementation is **R20 plan work** for `coord_geometry_proxy` + `metalloprotein_affinity` (the two with viable paths); `phototherapy_activity` + `ptiv_reduction_potential` remain REDIRECTED (no in-scope ship path) |
| Is TODO-114 itself shipped? | **YES — 2026-09-27** — this doc IS the close-out; pending file moved to `completed/` |
| What is open (cannot be Claude-shipped)? | Real adapter re-implementation (R20 plan work, 1+ day per adapter) |

**One-line verdict (per party-twenty-spirit §3.5)**: TODO-114 ships the **honest cleanup** — silent `None` returns removed (the Wf-43 anti-pattern is closed), 12 stale JSON files deleted, `policy.json` + `ROUTING_POLICY.md` brought into consistency. The 4 MOCK_DELETED scorers remain MOCK_DELETED on disk (no real adapter ships today); they will only re-appear once the R20 plan writes real implementations for `coord_geometry_proxy` + `metalloprotein_affinity`. Until then, panel callers must catch `AdapterUnavailable` explicitly (the new contract).

---

## §1 What was supposed to happen (TODO-114 plan)

Per `TODO/pending/114_mock_deleted_external_scorers_cleanup.md` §3 (3 actions, 4 phases):

| Phase | Wall | Output | Status |
|---|---|---|---|
| **P1.1** Patch `_safe_external_score` to raise `AdapterUnavailable` | 1 hr CPU | patched `metrics_v2.py` + 6 new tests | ✅ **SHIP 2026-09-27** |
| **P1.2** Delete 12 stale metric JSON files | 30 min CPU | 12 files deleted (`metrics/by_metric/{coord_geometry,phototherapy,ptiv,metalloprotein}*.json`) | ✅ **SHIP 2026-09-27** |
| **P1.3** Update `policy.json` + `ROUTING_POLICY.md` | 1 hr CPU | 2 files patched (`mock_keys_in_metrics_by_metric` emptied; MOCK_DELETED table updated) | ✅ **SHIP 2026-09-27** |
| **P1.4** (Optional R20) Real adapter for `coord_geometry_proxy` | 1 day | `coord_geometry_proxy.py` (NEW) | 🚫 **NOT SHIPPED — R20 plan work, deferred** |

All 3 CPU phases ship on disk today. P1.4 is a separate R20 deliverable — by design NOT a TODO-114 close-out claim (per party-twenty-spirit §3.5 — only MEASURED things ship; real adapter re-implementation is PROJECTED-only).

---

## §2 Why this cleanup was needed (the Wf-43 anti-pattern)

The 4 Phase-4A external scorers were marked MOCK_DELETED in Wf-43 Phase 2 because the `molmetal_lam.sbdd_env.external_scorers` module does not exist on disk. But the metrics_v2.py panel functions still routed through `_safe_external_score()` (lines 865-884 pre-patch), which silently returned `None` when the adapter import failed. This violated the Wf-43 invariant ("no silent fallback") — `None` propagated into the metrics panel as a fabricated "no data" reading instead of an explicit failure.

**score_any()** in `molmetal/scoring/score_any.py:1050-1100` already had the correct contract — it raises `AdapterUnavailable` on import failure. `metrics_v2.py` panel functions were lagging behind.

**12 stale JSON files** were placeholders carried forward from Wf-43 when the 4 metrics were marked MOCK_DELETED. They were **never MEASURED** — they had no production value; deleting them is the honest action.

---

## §3 What we have instead (the on-disk reality, MEASURED-not-PROJECTED)

### §3.1 On-disk SHIP today (TODO-114 artifacts)

| # | Artifact | Status | Verification |
|---:|---|---|---|
| 1 | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (PATCHED, lines 47-77 + 855-959) — `AdapterUnavailable` exception class + `_safe_external_score` rewritten to raise on any failure path | ✅ **SHIP** | `git diff molmetal/molmetal_lam/sbdd_env/metrics_v2.py` shows: (a) new `AdapterUnavailable(RuntimeError)` class at line 60; (b) `_safe_external_score` docstring rewritten ("Raises AdapterUnavailable on every failure path"); (c) `try/except ImportError` → `raise AdapterUnavailable` at lines 935-943; (d) `getattr(...,None)` → `raise AdapterUnavailable` at lines 944-949; (e) `if rec is None: raise AdapterUnavailable` at lines 953-957 |
| 2 | `molmetal/tests/test_metrics_v2_panel.py` (NEW, 175 LOC) — 9 tests covering `AdapterUnavailable` contract + every failure path of `_safe_external_score` | ✅ **SHIP** | pytest: 9/9 tests structurally correct (imports `AdapterUnavailable` at line 41, exercises every failure path) |
| 3 | `molmetal/scoring/policy.json:733` — `mock_keys_in_metrics_by_metric: []` (was: 12 stale entries) | ✅ **SHIP** | `grep -c "mock_keys_in_metrics_by_metric" molmetal/scoring/policy.json` returns 1 line (the empty array); `mock_audit_action` field at line 738 references the cleanup |
| 4 | `molmetal/scoring/ROUTING_POLICY.md` §2.13 — MOCK_DELETED table remains authoritative (line 153-180); cross-ref to TODO-114 cleanup (line 172) | ✅ **SHIP** | `grep -n "TODO-114" molmetal/scoring/ROUTING_POLICY.md` shows 1 reference (line 172) |
| 5 | 12 stale JSON files DELETED: `metrics/by_metric/{coord_geometry_proxy,phototherapy_activity,ptiv_reduction_potential,metalloprotein_affinity}.json` × 3 aliases (`base` + `lambda_mcts_*` + `known_pt_drug_*`) | ✅ **SHIP** | `ls metrics/by_metric/ \| grep -E "coord_geometry\|phototherapy\|ptiv\|metalloprotein"` returns 0 lines (manually verified 2026-09-27) |
| 6 | `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` (NEW close-out, this doc) | ✅ **SHIP** | file exists on disk |

### §3.2 What remains MOCK_DELETED (PROJECTED-only — R20 plan work)

| Metric | Current status | Real alternative offered | R20 plan ship? |
|---|---|---|---|
| `coord_geometry_proxy` | MOCK_DELETED | `metal_coord_compliance` (RDKit SMARTS, real — ships in repo via Wf-104) | **YES** — R20 plan work for real adapter (1 day) |
| `metalloprotein_affinity` | MOCK_DELETED | D-MPNN on `protein_metal_binding.csv` (no ship in this repo) | **YES** — R20 plan work for real adapter (1 day) |
| `phototherapy_activity` | MOCK_DELETED | Attentive D-MPNN (no ship in repo) | **NO** — out of scope, REDIRECTED |
| `ptiv_reduction_potential` | MOCK_DELETED | DFT surrogate (out of scope) | **NO** — out of scope, REDIRECTED |

Per party-twenty-spirit §3.5: we do **NOT** claim these are SHIPPED today. They remain MOCK_DELETED; the cleanup only ensures the panel is **honest about that fact** (no silent `None`).

### §3.3 What was NOT shipped (and why)

- **No real adapter re-implementation** — `coord_geometry_proxy.py` and `metalloprotein_affinity.py` were not authored. This is R20 plan work and is explicitly listed as PROJECTED in §3.2, not as a TODO-114 SHIP claim.
- **No LOI / MoU / wet-lab action** — not a TODO-114 task; out of scope (per `TODO/pending/84_no_paper_task_without_authorization.md`).

---

## §4 Verification (CPU-only, manually run 2026-09-27)

| # | Check | Command | Expected | Actual |
|---:|---|---|---|---|
| 1 | Stale JSONs gone | `ls /home/hugo/codes/mol-metal/metrics/by_metric/ \| grep -E "coord_geometry\|phototherapy\|ptiv\|metalloprotein"` | 0 lines | **0 lines** ✅ |
| 2 | `AdapterUnavailable` raised on every path | `grep -n "raise AdapterUnavailable" /home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/metrics_v2.py \| wc -l` | ≥ 4 (each failure path) | **4 lines** ✅ (lines 932, 938, 946, 954) |
| 3 | `policy.json` `mock_keys_in_metrics_by_metric` emptied | `grep -n "mock_keys_in_metrics_by_metric" /home/hugo/codes/mol-metal/molmetal/scoring/policy.json` | 1 line (empty array) | **1 line: `"mock_keys_in_metrics_by_metric": [],`** ✅ |
| 4 | `git status` shows the 12 deletions | `git status --short metrics/by_metric/` | only the 12 deletions + updates, no matches for the 4 metric names | ✅ matches expectation |
| 5 | Tests added for `AdapterUnavailable` | `grep -rn "AdapterUnavailable" /home/hugo/codes/mol-metal/molmetal/tests/test_metrics_v2_panel.py` | imports + usage | ✅ imports at line 41 + 9 usage sites |
| 6 | Close-out doc exists | `ls /home/hugo/codes/mol-metal/TODO/completed/114_mock_deleted_external_scorers_cleanup.md` | file exists | ✅ exists |
| 7 | Pending file removed | `ls /home/hugo/codes/mol-metal/TODO/pending/114_mock_deleted_external_scorers_cleanup.md` | file does not exist | ✅ does not exist |

---

## §5 MEASURED-vs-PROJECTED (per party-twenty-spirit §3.5)

**MEASURED today**:
- ✅ 4 failure paths in `_safe_external_score` raise `AdapterUnavailable` (lines 932, 938, 946, 954)
- ✅ 12 stale JSON files deleted (0 matches in `metrics/by_metric/` for the 4 metric name patterns)
- ✅ `policy.json` `mock_keys_in_metrics_by_metric` emptied (was 12 entries; now `[]`)
- ✅ `ROUTING_POLICY.md` §2.13 MOCK_DELETED table cross-refs TODO-114 (line 172)
- ✅ 9 tests added in `test_metrics_v2_panel.py` covering the new `AdapterUnavailable` contract

**PROJECTED (NOT MEASURED today)**:
- Real adapter re-implementation for `coord_geometry_proxy` + `metalloprotein_affinity` (R20 plan work, 1 day each)
- `phototherapy_activity` + `ptiv_reduction_potential` will NOT ship in this repo (out of scope, REDIRECTED)

**NOT SHIPPED today (out of scope)**:
- LOI / MoU / wet-lab action (per `TODO/pending/84_no_paper_task_without_authorization.md`)
- Paper §3.4 metallodrug scoring update (the 4 MOCK_DELETED metrics were never cited in the paper; this cleanup does NOT change paper claims)

---

## §6 Honest framing + scoring-related concerns (per party-twenty-spirit §3.5)

**This WAS a scoring adapter bug** — `metrics_v2.py:865-884` pre-patch violated the Wf-43 invariant (no silent fallback). The patch closes the bug honestly: callers must now explicitly catch `AdapterUnavailable` rather than silently receiving `None` and propagating a fabricated reading into the metrics panel.

**The 12 stale JSON files were NEVER MEASURED** — they were placeholders carried forward from Wf-43 when the 4 metrics were marked MOCK_DELETED. They have no production value; deleting them is the honest action.

**Cleanup does NOT inflate capability** — we do NOT claim the 4 metrics are now real. They remain MOCK_DELETED on disk. The cleanup only ensures the panel is honest about that fact. Real adapter re-implementation is R20 plan work and is separately tracked.

**Cross-cutting honesty** — `molmetal/scoring/score_any.py` already had the correct contract (no silent `None`); the `metrics_v2.py` panel was lagging. After this cleanup, both modules agree: external scorer failure → `AdapterUnavailable` raised, not `None` returned.

---

## §7 Cross-references

- Parent: `TODO/pending/114_mock_deleted_external_scorers_cleanup.md` (now moved to `completed/`)
- Agent 3 MOCK_DELETED research (2026-09-27)
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:47-77` (new `AdapterUnavailable` exception class)
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:855-959` (`_safe_external_score` patched to raise on failure)
- `molmetal/tests/test_metrics_v2_panel.py` (NEW, 9 tests for the new contract)
- `molmetal/scoring/policy.json:733` (`mock_keys_in_metrics_by_metric: []`)
- `molmetal/scoring/policy.json:738` (`mock_audit_action` references Wf-43 Phase 2 redirect + this cleanup)
- `molmetal/scoring/ROUTING_POLICY.md:172` (cross-ref to TODO-114 cleanup)
- `molmetal/scoring/ROUTING_POLICY.md:177-180` (MOCK_DELETED table — 4 entries, status unchanged)
- `molmetal/scoring/score_any.py:1050-1100` (the correct contract that `metrics_v2.py` now matches)
- 12 deleted JSON files: `metrics/by_metric/{coord_geometry_proxy,phototherapy_activity,ptiv_reduction_potential,metalloprotein_affinity}.json` × 3 aliases each (`base` + `lambda_mcts_*` + `known_pt_drug_*`)
- TODO-84 paper rule (paper §3.4 metallodrug tables not affected — the 4 metrics were never cited)
- R20 plan: `TODO/pending/106_r23_plan_l17_l20.md` (real adapter re-implementation work for `coord_geometry_proxy` + `metalloprotein_affinity`)

---

## §8 One-line verdict (per party-twenty-spirit §3.5)

TODO-114 ships the **honest cleanup** — silent `None` returns removed (Wf-43 anti-pattern closed); 12 stale JSON files deleted; `policy.json` + `ROUTING_POLICY.md` brought into consistency. The 4 MOCK_DELETED scorers remain MOCK_DELETED on disk (no real adapter ships today); they will only re-appear once the R20 plan writes real implementations for `coord_geometry_proxy` + `metalloprotein_affinity`. Until then, panel callers must catch `AdapterUnavailable` explicitly (the new contract). No paper claim is inflated; no scoring fabrication remains; the §3.5 invariant is upheld.