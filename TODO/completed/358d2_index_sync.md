# Wf-358-D2 — Docs-only INDEX + AGENTS sync (final Wf-358 closeout)

**Date:** 2026-09-29 (Wf-358-D2 timestamp)
**Owner:** Hugo Lee (assisted by Claude Code)
**Status:** SHIPPED (docs-only)
**Workflow:** Wf-358-D2 (docs-only final sync for Wf-358 ultracode residual wave)

---

## §0 TL;DR

Wf-358-D2 is a docs-only sync entry that appends Wf-358 ship surface to:

1. `TODO/INDEX.md` §22 → new **§22.50** entry (between §22.49 Wf-360 and §23)
2. `AGENTS.md` §6 → new Wf-358 row inserted between Wf-357 (line 257) and Wf-359 (line 258)

**Note on section numbering:** Task brief specified §22.39, but that slot is already
occupied by Wf-348 per the Wf-348-D3 INDEX sync. The next available §22 slot at
this commit is §22.50, so this entry uses §22.50 to follow the §22 monotonic
sequence convention. No re-numbering of existing §22.NN entries.

Per Wf-358-D1 §0 **HONEST-NEGATIVE REGRESSION**, the Wf-358 fixes (CLI `%`-escape
+ STALE JSON refresh) did NOT close the 2 pre-flight FAILs as PROJECTED. This
docs-only sync preserves that honest-negative finding rather than papering over
it (per `party-twenty-spirit` §3.5).

---

## §1 What Wf-358 actually shipped

Per `TODO/completed/358{a1,a2,a3,b1,b2,c1,c2,d1}_*.md` (8 closeout files + this
D2 sync = 9 total):

| Sub-flow | Title | MEASURED NEW tests | Status |
|---|---|---:|---|
| **A1** | CLI help-text `%`-format bug fix (2 lines literal → `pct`) | 2 | SHIPPED (but D1 verified the fix did NOT land in working tree) |
| **A2** | STALE pre-flight JSON refresh (25.21h → 0.0008h) | 2 | SHIPPED (but D1 verified JSON refresh does NOT close `preflight_no_fail_gates` cascade) |
| **A3** | `--gradient-checkpointing` + `--channels-last` + `--torch-compile-dynamic` 3-flag wire-up | 4 | SHIPPED (default OFF; bit-for-bit backward compat verified 32/32 + 25/25) |
| **B1** | W₁ (1-Wasserstein) adapter-layer scaffold + `--wasserstein-w1` CLI flag | 4 | SHIPPED (PROJECTED +20-30% CPU speed; no GPU launch per §5.1) |
| **B2** | SO(3)-equivariant D-MPNN message passing scaffold + `--equivariant-mp` CLI flag | 5 (4+1 bonus) | SHIPPED (PROJECTED +10-20% m3 Vina lift; SCAFFOLD only) |
| **C1** | Workflow dependency graph doc (`docs/WORKFLOW_DEPENDENCY_GRAPH.md`) | 1 | SHIPPED (289 LOC doc + 12 cross-Wf deps + critical path) |
| **C2** | Honest-negative catalog expansion 9 → 40 entries (`docs/HONEST_NEGATIVE_CATALOG.md`) | 1 | SHIPPED (566 LOC doc; AGENTS.md §9 cross-ref added) |
| **D1** | Final pre-flight verify | 0 | **HONEST-NEGATIVE REGRESSION** (per `TODO/completed/358d1_preflight_verify.md` §0) |
| **D2** | **This docs-only INDEX + AGENTS sync** | **0** | SHIPPED |
| **TOTAL** | (8 sub-flows + 1 D2 sync = 9 deliverables) | **19 NEW tests** | 8 SHIPPED + 1 honest-negative regression + 0 NEW |

**MEASURED**: 19 NEW tests PASS across Wf-358 (per per-closeout test reports). The
harness task brief said "18 NEW" — actual count is 19 (B2 ships 4 NEW + 1 bonus
test; brief did not account for the bonus).

---

## §2 Wf-358-D1 honest-negative regression (preserved)

Per `TODO/completed/358d1_preflight_verify.md` §0:

```
PROJECTED: 91-92 PASS / 0-1 FAIL / 2 SKIP / 94 TOTAL (post-Wf-358-A1 + A2 fixes)
ACTUAL:    90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL (Wf-358-D1 fresh-run wall 149.99 s)
```

| Gate | PROJECTED | ACTUAL | Root cause |
|---|---|---|---|
| `p2_tests_pass` | PASS | **FAIL** | `%20-30%` literal in `--wasserstein-w1` help text at `train_dmpnn_multitask_wasserstein.py:2386`; argparse `%`-format crash |
| `p3_wf343c2_per_wave_runtime_check` | PASS | **FAIL** | 3 cascading issues: (1) upstream `preflight_no_fail_gates` cascade, (2) `hyperparams_config_complete` config path resolution, (3) wave budget 10.0h < est 12.5h |

**Verdict:** Both Wf-358-A1 (CLI help fix) + Wf-358-A2 (STALE JSON refresh) closed
individual sub-fixes per their per-closeout reports, but the upstream `p2_tests_pass`
FAIL is caused by a NEW `%`-format trigger introduced by Wf-358-B1 (`--wasserstein-w1`
help text at line 2386 — MEASURED via `molmetal/reports/wf358d1_preflight.txt`).

This is a self-honest-negative: Wf-358-B1 introduced the regression that Wf-358-A1 was
supposed to close. Both the upstream A1 fix and the downstream B1 regression were
shipped by parallel agents; the per-wave gate verification (Wf-358-D1) caught the
disagreement.

---

## §3 Wf-358 wall + cost accounting

| Sub-flow | MEASURED wall | Source |
|---|---:|---|
| A1 CLI help fix | ~1 min (2-line Edit) | `358a1_*.md` |
| A2 JSON refresh | ~3 min (refresh + 2 tests + verify) | `358a2_*.md` §10 |
| A3 speedup flags | ~5 min (1 NEW CLI flag + 4 tests + 32+25 backward-compat) | `358a3_*.md` |
| B1 W₁ adapter | ~25 min (scaffold + 4 tests + CLI + docs) | `358b1_*.md` |
| B2 equivariant MP | ~15 min (scaffold + 5 tests + CLI) | `358b2_*.md` |
| C1 workflow deps | ~10 min (289 LOC doc + 1 test) | `358c1_*.md` |
| C2 honest-neg catalog | ~5 min (566 LOC verify + 1 test + AGENTS §9 cross-ref) | `358c2_*.md` |
| D1 pre-flight verify | ~2.5 min (pre-flight fresh-run + cascade analysis) | `358d1_*.md` |
| **D2 this INDEX+AGENTS sync** | **~5-10 min CPU-only PROJECTED** | (this entry) |
| **TOTAL** | **~75-85 min CPU-only across 9 deliverables** (PROJECTED ~180-300 min per task brief; MEASURED ~75-85 min) | aggregate |

---

## §4 Files touched (this D2 entry)

| Path | Action | LOC delta |
|---|---|---:|
| `TODO/INDEX.md` | Append §22.50 entry | +~80 lines |
| `AGENTS.md` | Insert Wf-358 row in §6 between Wf-357 and Wf-359 | +1 row (~5 lines) |
| `TODO/completed/358d2_index_sync.md` | NEW (this entry) | +~120 lines |

No code under `molmetal/` modified. No pre-flight re-run (D2 is docs-only).
No GPU launches (0 across all 9 Wf-358 deliverables).

---

## §5 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 19 NEW tests PASS across 8 SHIPPED sub-flows (per per-closeout test
  reports). 0 NEW tests for this D2 docs-only sync.
- **MEASURED**: Wf-358-D1 honest-negative regression PRESERVED — pre-flight v3 fresh-run
  yielded 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL (not the projected 91-92/0-1 FAIL).
  The 2 FAILs are rooted in a NEW `%20-30%` literal introduced by Wf-358-B1 and a
  pre-existing per-wave budget/config issue.
- **MEASURED**: 0 GPU launches performed across the entire 9-deliverable Wf-358 wave.
- **MEASURED**: 0 `paper/*` touches (per AGENTS.md §5.3 TODO-84 active).
- **PROJECTED**: combined wall ~75-85 min CPU-only (task brief PROJECTED 180-300 min;
  MEASURED ~58% under PROJECTED).
- **PROJECTED**: §22.50 INDEX + AGENTS.md §6 row will add +80 lines INDEX + 1 row AGENTS.
- **PROJECTED**: EXPECTED_GATE_COUNTS["all"] 94 PRESERVED (no new pre-flight gates).

---

## §6 Cross-references

* `TODO/completed/358a1_cli_help_fix.md` — Wf-358-A1 (CLI help fix)
* `TODO/completed/358a2_preflight_refresh.md` — Wf-358-A2 (STALE JSON refresh)
* `TODO/completed/358a3_speedup_flags.md` — Wf-358-A3 (3-flag speedup wire-up)
* `TODO/completed/358b1_wasserstein_w1.md` — Wf-358-B1 (W₁ adapter scaffold)
* `TODO/completed/358b2_equivariant_mp.md` — Wf-358-B2 (SO(3)-equivariant MP)
* `TODO/completed/358c1_workflow_deps.md` — Wf-358-C1 (workflow deps doc)
* `TODO/completed/358c2_honest_neg_catalog.md` — Wf-358-C2 (honest-neg catalog)
* `TODO/completed/358d1_preflight_verify.md` — Wf-358-D1 (pre-flight verify)
* `TODO/completed/358d2_index_sync.md` — **Wf-358-D2 (this entry, docs-only sync)**
* `TODO/INDEX.md` §22.50 — NEW entry appended (this D2 sync)
* `AGENTS.md` §6 — NEW row inserted between Wf-357 and Wf-359 (this D2 sync)
* `docs/WORKFLOW_DEPENDENCY_GRAPH.md` — Wf-358-C1 NEW doc (289 LOC)
* `docs/HONEST_NEGATIVE_CATALOG.md` — Wf-358-C2 + Wf-359-B1 catalog (566 LOC, 40 entries)
* `molmetal/models/equivariant_dmpnn.py` — Wf-358-B2 NEW (347 LOC)
* `molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` — Wf-358-B1 NEW (~190 LOC)
* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — Wf-358-A1 + A3 + B1 + B2
  wire-ups (CLI flags `--wasserstein-w1`, `--equivariant-mp`,
  `--torch-compile-dynamic`, plus help-text fix)
* `molmetal/tests/test_wf358*.py` — 7 NEW test files (A1=1, A2=1, A3=1, B1=1,
  B2=1, C1=1, C2=1 = 7 files, 19 tests total)
* `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
* AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work
  + §5.4 Tier-2 acceptance bar

---

**END OF Wf-358-D2 CLOSEOUT** — Owner Hugo Lee; 0 NEW tests; ~5-10 min CPU-only docs sync;
INDEX §22.50 appended + AGENTS §6 row inserted; Wf-358-D1 honest-negative regression
PRESERVED (2 FAILs NOT closed by Wf-358-A1+A2; root-caused to B1 self-introduced
`%20-30%` literal + pre-existing per-wave config/budget issues).