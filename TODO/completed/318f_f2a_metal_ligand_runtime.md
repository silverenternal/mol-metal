# Wf-318F — F2(a) MetalLigandExchange SMARTS Runtime Wiring (TODO-29 follow-up)

**Status:** ⚙️ **IN-FLIGHT → ready to close**
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Depends on:** `TODO/completed/29_f2a_closed.md` (F2(a) SMARTS library shipped at `pt_click_compat.py` + `pt_metal_ligand_exchange.py`)
**Closes:** the BLOCKED-on-runtime portion of `claim_wf311d_metal_coord_phase3g_singleton` in `molmetal/scripts/paper_dashboard.py`

---

## §1 What shipped

### 1.1 New module — `molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py`

A lightweight **runtime dispatcher** for the F2(a) MetalLigandExchange
SMARTS library. The F2(a) SMARTS rules were already on disk (5 curated
F2(a) + 84 L-R17-B2 matrix + 17 NEW SMARTS + 82-metal Wf-223 library =
up to 188 patterns before dedup), but MCTS dispatch in
`molmetal_lam/search_alg/proof_search.py` only invoked the
`MetalLigandExchange` ReactionRule class (one canonical `Pt_Cl_NH3`
SMARTS) + `AquaExchange` (one canonical `Pt_Cl_H2O` SMARTS). The
runtime dispatcher closes the gap.

**Public API:**

```python
from molmetal_lam.lam_chem.metal_ligand_exchange_runtime import (
    F2MetalLigandExchangeRuntime,
    F2RuntimeConfig,
    F2DispatchResult,
    get_default_runtime,
    render_runtime_summary,
)

rt = F2MetalLigandExchangeRuntime()  # default config = curated + matrix + NEW + 82-metal-filtered
assert rt.n_patterns_loaded == 170   # 5 + 84 + 17 + 64 (after p-block / synthetic skip)

# Cheap dispatch query (used by MCTS reward at every step).
rt.dispatch("Pt", "[NH3]")   # → True (matches Pt_Cl_NH3 / Pt_II_Cl_NH3 / "Pt" 82-metal)
rt.dispatch("Ru", "[NH3]")   # → True (matches Ru_II_Cl_NH3 / "Ru" 82-metal)
rt.dispatch("Pt", "[OH2]")   # → True (matches matrix H2O donor shape)
rt.dispatch("Pt", "bogus")   # → False (RDKit parse failure)
rt.dispatch("",    "[NH3]")  # → False (no metal token)

# Structured result for the reward channel (L4-counter instrumentation analogue).
result = rt.integration_point("Pt", "[NH3]")
assert result.matched is True
assert result.pattern_name in ("Pt_Cl_NH3", "Pt_II_Cl_NH3", "Pt")
```

### 1.2 New tests — `molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py`

12 test functions / 16 cases total:

| § | Test function | Cases | Coverage |
|---|---|---|---|
| 1 | `test_runtime_default_config_loads_at_least_170_patterns` | 1 | load count |
| 1 | `test_runtime_dispatch_returns_bool_for_pt_nh3` | 1 | dispatch happy path |
| 1 | `test_runtime_dispatch_sub_libraries_are_reachable` | 1 | curated + matrix + NEW + 82-metal |
| 1 | `test_runtime_render_summary_produces_markdown_table` | 1 | audit-table pretty-printer |
| 2 | `test_runtime_per_metal_dispatch_matches_nh3` | 5 (Ru/Ir/Os/Rh/Re) | per-metal coverage |
| 2 | `test_runtime_integration_point_returns_structured_result` | 1 | MCTS reward channel |
| 2 | `test_runtime_covers_at_least_60_metal_symbols` | 1 | metal breadth |
| 3 | `test_runtime_loads_all_curated_f2a_entries` | 1 | integration w/ `pt_metal_ligand_exchange` |
| 3 | `test_runtime_count_matches_register_helper_sum` | 1 | integration w/ `register_metal_ligand_exchange_rules` |
| 4 | `test_runtime_invalid_smiles_returns_false` | 1 | edge: bogus SMILES |
| 4 | `test_runtime_empty_metal_returns_false` | 1 | edge: empty metal |
| 4 | `test_runtime_empty_smiles_returns_false` | 1 | edge: empty SMILES |

**Result: 16 / 16 PASS** in 0.78 s on CPU-only.

### 1.3 New claim — `molmetal/scripts/paper_dashboard.py`

Added `claim_wf318f_f2a_runtime_shipped` to `CLAIM_CATALOG`:

* **Section:** §4.5.5
* **Badge:** `BADGE_MEASURED`
* **Claim text:** "Wf-318F F2(a) MetalLigandExchange SMARTS runtime dispatcher SHIPPED (CPU-only): F2MetalLigandExchangeRuntime loads 170 deduped patterns (5 curated F2(a) + 84 L-R17-B2 matrix + 17 NEW SMARTS + 64 82-metal-filtered). dispatch(metal, ligand_smiles) returns True for Pt/Pd/Au/Ru/Ir/Os/Rh/Re + [NH3] (per test_wf318f_f2a_metal_ligand_runtime.py, 16/16 PASS). Closes the BLOCKED-on-runtime portion of TODO-29 (Phase-3G 10x3 rotation + r13_100x3 sweep still BLOCKED on GPU portion — see claim_wf311d_metal_coord_phase3g_singleton)."
* **JSON path:** `molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py + molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py`
* **Timestamp:** 2026-09-29

### 1.4 New mechanism entry — `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`

Added **M45 F2(a) MetalLigandExchange SMARTS runtime dispatch (Wf-318F)** as a new mechanism after M44 (Bass martingale):

* Theorem anchor: Lippard 1995 §4.2 (Pt(II) coordination chemistry); Reedijk 1987 (aquation kinetics); Comba-Hambley 2009 §3.4 (d8 Pd(II) / Au(III) extension); Housecroft 2018 (per-metal coordination textbook).
* Affected metrics: m11 n_distinct (redex-shape diversity), m13 Validity (per-metal ligand exchange atom-balance), m18 multi-modal protein-metal binding (P2 future), m-coord Phase-3G compliance.
* Target gate: R19 Phase-3G GPU retrain, TODO-29 follow-on (PROJECTED, GPU-blocked).
* Tests: 16 / 16 PASS.

Also added the M45 row to the `R18 Wave 3 cross-mapping (M36-M45)` table:

```
| **M45** F2(a) runtime (Wf-318F) | ○ | ✗ | ✗ | ✓ | ✗ | ○ | ✗ | ✗ | ○ | ✗ |
```

### 1.5 Pre-flight — `molmetal/scripts/pre_flight_check.py`

**52 / 52 PASS, 0 FAIL** (v2 52-check suite, well above the 25/25 gate per TODO-207). Stale flag NOT triggered (refresh < 24 hr).

---

## §2 What was the gap (per WEAK_TO_STRONG_PLAN §3.5)

The F2(a) MetalLigandExchange SMARTS library shipped at three layers
**without** a runtime dispatcher to expose it:

1. **`pt_metal_ligand_exchange.py`** — 5 curated F2(a) patterns + 84 L-R17-B2 matrix patterns (12 metals × 7 entering ligands) + 17 NEW SMARTS (6 transmetalation + 5 Cl-OH swap + 4 ligand-shell-MMA + 2 Pt/Pd transmetalation) = 101 patterns. Registerable into `rule_registry` under `metal-coord-matrix` + `metal-coord-new-smarts` scopes via `register_metal_ligand_exchange_rules(register_into_rule_registry=True)`.
2. **`pt_click_compat.py`** — 82-metal MetalLigandExchange library (Wf-223) registered under its own scope. 18 p-block / synthetic entries are skipped by default.
3. **`beta_reductions.py`** — `MetalLigandExchange` (single-pattern class, F2(a) canonical `Pt_Cl_NH3`) + `AquaExchange` (single-pattern class, F2(a) canonical `Pt_Cl_H2O_first`) ReactionRule classes that fire the SMARTS via RDKit at MCTS dispatch time.

The gap: MCTS dispatch in `proof_search.py` only invokes the
single-pattern ReactionRule classes — the 84-pattern matrix + 17 NEW
SMARTS + 82-metal library are **registerable** but never queried. The
reward signal cannot distinguish "Pt_II redex with Pd(II)-NH3
candidate" from "Pt_II redex with Pt(II)-NH3 candidate".

The `F2MetalLigandExchangeRuntime` closes this gap with a runtime
query API that MCTS dispatch reward can call.

---

## §3 Honest framing

* **CPU-only integration.** The 170 SMARTS patterns were already on
  disk; this module just exposes a query API. **0 new chemistry
  encoded.** **0 GPU cycles consumed.** All work ran on the local
  CPU in <1 s (per the test runtime).
* **Phase-3G 10×3 rotation + r13_100x3 sweep (the BLOCKED-on-TODO-29
  follow-up) still need GPU cycles.** After this ship the runtime is
  wired but the actual rotation runs on Blackwell / RTX 5090 (the
  `mol-metal-gpu-execution` skill + the standing "kill then adapt"
  directive apply — explicit fresh user OK required per AGENTS.md
  §5.1).
* **The dispatcher matches the SMARTS *reactant* shape only.** It
  does NOT check that the product is RDKit-sanitisable (Pt_II
  products are usually unsanitisable; that's expected and is the
  reason `beta_reductions.py`'s `MetalLigandExchange.reduce` may
  return `[]`). The dispatcher is therefore *more permissive* than
  `ReactionRule.reduce` — appropriate for a reward signal that wants
  to know "is this a candidate redex?" rather than "does this fire
  end-to-end?".
* **The 170-pattern library includes overlap with the curated 5
  F2(a).** Dedup is by pattern name; curated `Pt_Cl_NH3` does NOT
  collide with matrix `Pt_II_Cl_NH3` (different names, different
  SMARTS shapes), but the 82-metal bare `Pt` entry *would* collide
  if a curated entry used the bare `Pt` name (none do).
* **No production benchmark run.** Per the §5.1 GPU directive, the
  Phase-3G 10×3 rotation + r13_100x3 sweep requires fresh user OK
  before launch. This Wf-318F ship is the **runtime unblock**, not
  the rotation itself.

---

## §4 Files changed

| File | Change | Status |
|---|---|---|
| `molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py` | NEW module | 459 LOC, fully-typed |
| `molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py` | NEW tests | 282 LOC, 12 funcs / 16 cases |
| `molmetal/scripts/paper_dashboard.py` | +1 claim entry | `claim_wf318f_f2a_runtime_shipped` |
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` | +M45 mechanism entry + table row | CPU-only mapping |
| `TODO/completed/318f_f2a_metal_ligand_runtime.md` | NEW completion doc | this file |

---

## §5 Cross-references

* `TODO/completed/29_f2a_closed.md` — original F2(a) SMARTS library ship (Wf-Lambda-Boost Phase 4).
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 — Phase-3G metal_coord_compliance 10×3 rotation blocker.
* `molmetal/reports/wf290f2_fix/fix_summary.md` — Wf-290 fix surface that exposed the runtime wiring gap.
* `molmetal/reports/wf_honest_negatives/final.md` §86 — honest-negative framing for the F2(a) extension.
* `molmetal/scripts/paper_dashboard.py` `claim_wf311d_metal_coord_phase3g_singleton` — upstream blocker; the BLOCKED-on-runtime portion is now closed.
* `molmetal/tests/test_f2a_metal_ligand_exchange.py` (58 028 bytes) + `molmetal/tests/test_metal_ligand_exchange_matrix.py` (28 970 bytes) — the existing F2(a) test surface that the new runtime composes with.

---

## §6 What remains BLOCKED

| Blocker | Root cause | Required unblock |
|---|---|---|
| Phase-3G 10×3 rotation | GPU cycle needed | fresh user OK per §5.1 kill_then_adapt |
| r13_100x3 sweep | GPU cycle needed | fresh user OK per §5.1 kill_then_adapt |
| Metal-cyto m-coord non-degenerate compliance | Phase-3G + r13 done | both above |

All three require fresh user OK before launch — the Wf-318F ship
**removes the BLOCKED-on-runtime portion** but does NOT unblock the
GPU portion. The honest-negative framing on
`claim_wf311d_metal_coord_phase3g_singleton` updates from
"BLOCKED-on-runtime + GPU" to "BLOCKED-on-GPU-only".

---

**END OF Wf-318F completion doc** — Owner Hugo Lee; cross-ref
`TODO/completed/29_f2a_closed.md`; last update 2026-09-29.