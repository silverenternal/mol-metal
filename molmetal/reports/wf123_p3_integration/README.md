# Wf-123 Phase 3 — M1 + M2 cross-module integration verification

**Date**: 2026-09-27
**Status**: **SHIP** (CPU-only, 25 / 25 integration tests PASS, signal-floor
probe written, verification driver ready)
**Owner**: Hugo Lee + ultracode agents
**Parent**: TODO-126 §3 (master M1+M2 plan) + Wf-123 P1 (M1) + Wf-123 P2 (M2)

---

## §1 TL;DR

Wf-123 Phase 3 ships the **integration layer** between the two SHIPPED
shared modules from P1 and P2:

| Module | Path | Phase | Tests |
|---|---|---|---|
| M1 — Wasserstein regularization | `molmetal/molmetal_lam/reward/wasserstein_loss.py` | P1 SHIP | 43 PASS |
| M2 — Pocket closure bias | `molmetal/molmetal_lam/reward/pocket_closure_bias.py` | P2 SHIP | 55 PASS |
| **M1+M2 integration** | `molmetal/tests/test_wasserstein_closure_integration.py` | **P3 SHIP** | **25 PASS** |

The integration layer verifies that:

1. `pocket_specific_closure_bias("3eml")` returns a bias dict whose
   dominant rules (AmideCoupling + SPAAC) are reflected in the
   closure-bounded candidate distribution — the per-pocket bias
   **shifts** the cohort toward the pocket's chemistry.
2. Every combination of Wasserstein loss function (5 wrappers) and
   per-pocket closure bias verdict (5 pockets) is **finite** — the
   modules compose without raising.
3. The Wasserstein pIC50 penalty on the **PlatinAI 4-cell probe**
   yields a valid signal-floor R² — the integration wiring is
   correct, even though R² < 0 here (a constant mean predictor
   against a different cell line cohort yields R² ≤ 0 by definition).
4. The Wasserstein decay check (Phase 1 §6) passes on a synthetic
   strictly-decreasing trainer history — the integration layer is
   ready for in-trainer use.
5. The cross-metal transfer loss (TODO-121 m19 LOMO seam) is finite
   and the gradient flows back through the per-metal embedding
   tensors.
6. The M1 and M2 engine tags (`wasserstein_reg` and
   `closure_catherine_bounded`) are **disjoint** — no silent
   substitution between the two modules per TODO-126 §7.

The verification driver (`molmetal/scripts/verify_shared_modules.py`)
runs the full integration suite via pytest and writes the honest
signal-floor JSON to
`molmetal/reports/wf123_p3_integration/cross_module_verification.json`.

---

## §2 What changed

| File | Change |
|---|---|
| `molmetal/tests/test_wasserstein_closure_integration.py` | NEW — 25 integration tests (~750 LOC). |
| `molmetal/scripts/verify_shared_modules.py` | NEW — verification driver (~250 LOC), executable. |
| `molmetal/reports/wf123_p3_integration/cross_module_verification.json` | NEW — signal-floor report (CPU-only). |
| `molmetal/reports/wf123_p3_integration/README.md` | NEW — this file. |

No existing module is modified — the integration tests import the
public surface of both P1 and P2 modules only.

---

## §3 Test surface (25 tests)

| Section | Tests | Purpose |
|---|---:|---|
| `TestPerPocketBiasShiftsCandidateDistribution` | 7 | 3eml bias dominates in candidate distribution; per-pocket engine tag consistent across all 5 pockets; pocket_specific_closure_bias is invariant across calls. |
| `TestCrossModuleFiniteness` | 5 | Every M1 loss (W₂ generic, transfer, pIC50, pose, pocket) + M2 verdict is finite. |
| `TestPlatinai4CellProbeR2` | 4 | R² signal floor on PlatinAI 4-cell probe; W₂ cross-cell-line finite; pairwise W₂ symmetry. |
| `TestDecayCheckOnClosureCohortHistory` | 3 | JKO CFL gate passes on strictly-decreasing + realistic trainer curves; fails on diverging W₂. |
| `TestCrossMetalTransfer` | 2 | 5-metal transfer loss finite + gradient flows; metalcytotox 5-metal CSV integration smoke. |
| `TestCombinedEngineTags` | 3 | Canonical M1 / M2 engine tags + disjoint set (no silent substitution). |
| `TestHonestSignalFloorReport` | 1 | Writes the JSON report with honest signal-floor R². |

All 25 tests pass on a single CPU in < 5 seconds.

```
$ python -m pytest molmetal/tests/test_wasserstein_closure_integration.py -v
25 passed in 1.75s
```

---

## §4 Honest signal-floor semantics

The integration test
`TestPlatinai4CellProbeR2::test_wasserstein_aligned_predictor_yields_zero_r2_floor`
asserts that a constant mean predictor against the *same* cohort
yields R² = 0 (the floor), and that the Wasserstein penalty between
MCF7 and A2780 (a different cell line) is finite and positive
(W₂ ≈ 0.27 pIC50 units — the cell-line shift magnitude).

The verification driver also computes the **cross-cohort R²** of a
mean predictor against a held-out cell line, which yields R² ≈ -0.41
on the 256-row PlatinAI probe.  This is the honest signal floor — a
constant mean predictor against a different cell line cohort
**must** yield R² ≤ 0 by definition, because the cell-line means
differ by ~0.3 pIC50 units (MCF7 ≈ 5.05, A2780 ≈ 5.35).

**What would lift R² above the floor:**

A trained W₂ alignment (TODO-122 P1.3 deliverable) would close the
MCF7 → A2780 distribution shift by tightening the predicted cohort
toward the A2780 manifold.  The integration layer's job is to ensure
the modules **compose correctly**; the lift is the per-strengthen-TODO
deliverable.

---

## §5 Verification driver

`molmetal/scripts/verify_shared_modules.py` is the end-to-end driver
that:

1. Computes a content fingerprint for both shared modules
   (size + head/tail byte hash — NOT cryptographic, just a content
   marker for cross-referencing).
2. Runs `pytest` on the integration suite and parses the summary
   line (PASS / FAIL / SKIPPED counts).
3. Computes the honest signal-floor R² on the PlatinAI 4-cell probe
   (default 256 rows, configurable via `--probe-rows`).
4. Writes the JSON report to
   `molmetal/reports/wf123_p3_integration/cross_module_verification.json`.
5. Exits 0 on success; non-zero on pytest failure (with diagnostic
   details in the report).

Usage:

```bash
source env.sh
uv run --no-sync python -u molmetal/scripts/verify_shared_modules.py \
    --probe-rows 256 \
    --out-dir molmetal/reports/wf123_p3_integration
```

The driver accepts `--skip-pytest` for signal-floor-probe-only runs
(useful for CI smoke tests that don't have RDKit available).

---

## §6 Engine-tag audit (TODO-126 §7)

Per TODO-126 §7, every per-sample output that uses a NEW scoring
component MUST carry an explicit engine tag so downstream reporting
can attribute contributions honestly.

| Module | Engine tag | Used by |
|---|---|---|
| M1 (Wasserstein) | `wasserstein_reg` | TODO-119 / 120 / 121 / 122 / 123 / 125 |
| M2 (Pocket closure bias) | `closure_catherine_bounded` (cohort) / `pocket_bias_prior` (verdict) / `closure_catherine_oversample` (oversample) | TODO-119 / 121 / 124 / 125 |

The two module-level engine tags are **disjoint** — there is no
silent substitution between Wasserstein regularization and pocket
closure bias.  The integration test
`TestCombinedEngineTags::test_combined_engine_set_is_disjoint`
asserts this invariant.

---

## §7 Honest framing

* The 25 integration tests verify the **wiring** between M1 and M2
  on a real probe (PlatinAI 4-cell, 4078 Pt complexes × 4 cell
  lines) and a synthetic probe (5-metal transfer cohort).  They do
  **not** verify that the W₂ penalty + closure bias actually moves
  any downstream metric — that is the per-strengthen-TODO
  deliverable (TODO-119 P1.2, TODO-120 P1.2, TODO-121 P1.1,
  TODO-122 P1.3, TODO-123 P1.3, TODO-125 P1.2).
* The signal-floor R² in the verification report is the **honest
  measure** of the integration layer — it is ≤ 0 by construction for
  a constant mean predictor against a different cohort.  The W₂
  alignment (TODO-122 P1.3) is what would lift R² above the floor.
* Per TODO-126 §7: every per-sample output that flows through the
  integration layer carries an explicit engine tag
  (`wasserstein_reg` for M1, `closure_catherine_bounded` for M2
  cohort emissions).  The integration tests assert the engine-tag
  invariant explicitly.
* The decay check (Phase 1 §6) is a **necessary-but-not-sufficient**
  convergence indicator.  A non-increasing loss does not guarantee
  convergence to the optimum; it only guarantees the trainer is not
  actively diverging.  Trainer code should still snapshot best-val
  iterates (mirroring Wf-122 P1 L2 fix pattern).
* The metalcytotox 5-metal CSV integration smoke test
  (`TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke`)
  is **conditional on the CSV existing** in the project tree
  (`.data/molmetal/metalcytotox_5metal_multitask.csv`).  It skips
  gracefully if the CSV is absent — no hard dependency on the
  cross-metal test data.

---

## §8 Cross-references

* `molmetal/molmetal_lam/reward/wasserstein_loss.py` — M1 module (Wf-123 P1 SHIP).
* `molmetal/molmetal_lam/reward/pocket_closure_bias.py` — M2 module (Wf-123 P2 SHIP).
* `molmetal/tests/test_wasserstein_loss.py` — M1 43 unit tests.
* `molmetal/tests/test_pocket_closure_bias.py` — M2 55 unit tests.
* `molmetal/tests/test_wasserstein_closure_integration.py` — P3 25 integration tests (NEW).
* `molmetal/scripts/verify_shared_modules.py` — P3 verification driver (NEW, executable).
* `molmetal/reports/wf123_p1_wasserstein/README.md` — P1 sibling README template.
* `molmetal/reports/wf123_p2_closure_bias/README.md` — P2 sibling README template.
* `TODO/pending/126_innovation_strengthen_roadmap.md` §3 M1 + M2 — master plan.
* `TODO/pending/119_decode_ratio_strengthen.md` §3 P1.2 — M1 consumer.
* `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 — M1 consumer.
* `TODO/pending/121_m19_lomo_strengthen.md` §3 P1.1 — M1 + M2 consumer.
* `TODO/pending/122_platinai_4cell_r2_strengthen.md` §3 P1.3 — M1 consumer.
* `TODO/pending/123_m15_rmsd_strengthen.md` §3 P1.3 — M1 consumer.
* `TODO/pending/125_vina_per_pocket_strengthen.md` §3 P1.2 — M1 + M2 consumer.