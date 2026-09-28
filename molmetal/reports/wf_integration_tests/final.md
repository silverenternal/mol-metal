# WF-INTEGRATION-TESTS — Final Report

**Status**: ✅ **DONE** 2026-09-25
**Master plan**: `TODO/IMPROVEMENT_BACKLOG.md` extension item I1
**Outcome**: 5 / 5 cross-module integration tests pass on CPU in < 90 s
total.

---

## 1. TL;DR

The IMPROVEMENT_BACKLOG extension item **I1** ("5 cross-module
integration tests") is shipped.  All 5 tests live in a single pytest
module — `molmetal/tests/test_integration.py` — and verify that
modules work together end-to-end (not just in isolation).  The suite
catches the recurring "individual unit tests pass but the pipeline
crashes at runtime" failure mode that has bitten R17 close-out
multiple times (mismatched dataclass fields, drift in default kwargs,
import-path collisions, contract drift between paired modules).

| # | Test | Modules wired | Pass criterion |
|---|---|---|---|
| **1** | `test_001_lipman_to_vina_e2e` | `LipmanFlowMatchingAdapter` + `VinaDockingAdapter` | 10 candidates generated and docked, all 10 produce valid (finite) vina_score |
| **2** | `test_002_mcts_sa_qed_pipeline` | `MCTSProofSearch` + `sa_score` + `QED` scorer | 100 MCTS iterations → 5 candidates, ≥3 with SA<5 AND QED>0.4 |
| **3** | `test_003_lambda_astsr_coupling_adapter` | `LambdaASTSR` + `CouplingAdapter` (Lambda×CFM bridge) | Fit on 50-row synthetic data, R² > 0.5 |
| **4** | `test_004_mle_matrix_click_compat` | 84-rule MLE matrix + `pt_click_compat` | 84 rules registered in `metal-coord-matrix` scope; 5 Pt complexes wired through compat verdicts, ≥3 yield valid (scaffold, click_rule) matrices |
| **5** | `test_005_end_to_end_smoke` | `metallo_drug_smoke_retrain.py` (whole pipeline as a module) | Run with `--steps 100 --batch-size 4 --seed 0 --hidden-dim 64 --n-layers 2`; assert `ckpt.pt` saved with finite tensors |

**Wall-clock budget (CPU)**: ≤ 90 s for the full suite on the RTX PRO
6000 Blackwell host.  The original task budget was 5 min; we landed
well under at ~71 s.

---

## 2. Test design

Each test follows a **glue verification** recipe:

1. **Two modules** from different layers (e.g. Lipman / Vina) are
   exercised on the *same* inputs through their canonical APIs.
2. The test asserts on a **downstream observable** (finite score,
   registered-rule count, R² value, ckpt presence) rather than on
   any one module's internal state.  This catches contract drift
   even when both modules individually look correct in isolation.
3. **Deterministic seed** — every test calls `_seed_everything(0)`
   (or its local equivalent) so failures are reproducible.
4. **CPU-only** — every test runs on CPU so the suite can be used as
   a pre-merge CI guard without a GPU.  The Lipman adapter is built
   with `hidden_dim=16, n_layers=1` so the `generate()` smoke stays
   under 5 s.

### 2.1 Test-by-test design rationale

#### test_001 — Lipman → Vina
The Lipman adapter (without training) does **not** produce valid
SMILES; its `generate()` returns 3D Molecules with empty `smiles`.
Vina's `dock()` requires a SMILES.  To verify the integration
contract we therefore:
1. Exercise Lipman's `generate()` on a trivial pocket (proves the
   adapter's import chain + dataclass wiring works end-to-end).
2. Build 10 candidate Molecules from a curated
   `SAMPLE_TEN_SMILES` list (10 known-dockable drug-like SMILES)
   via `Molecule.from_smiles(smi, embed_3d=True)` — RDKit does
   the 3D embed so the Molecule dataclass is fully populated.
3. Dock each via `VinaDockingAdapter.dock()` against a real PDB
   pocket (loaded from a TargetDiff example file via BioPython).
4. Assert every Complex returned has a finite
   `vina_score ∈ [-30, +5]` kcal/mol.

The integration we verify is the **import chain + dataclass
contract + finite-output guarantee**, not a learned handoff.

#### test_002 — MCTS → SA + QED
The MCTS layer (proof_search) is fed an "identity" reaction rule so
expansion does not require real chemistry — the rule returns the
educt unchanged.  We then score every leaf with the canonical Ertl
SA scorer and the RDKit QED scorer.  The pipeline test exercises
the dataclass round-trip `MoleculeClosedTerm → SMILES → SA/QED`,
not the search algorithm's exploration quality.

#### test_003 — LambdaASTSR + CouplingAdapter
LambdaASTSR is fit on a 50-row sparse-linear synthetic dataset and
the CouplingAdapter is used to verify the **bridge** is wired:
`embed_features(7-d-pocket-vector)` returns a finite 64-d vector
for every row.  We do NOT claim the adapter improves ASTSR
predictions — the adapter's job is the Lambda × CFM *bridge*, not
a quality lift on ASTSR.  The `R² > 0.5` gate measures ASTSR; the
finite-embedding check measures the bridge.

#### test_004 — 84-rule MLE + pt_click_compat
This is the **trickiest** test because of a Python module-identity
quirk:

* `pt_metal_ligand_exchange.py` imports
  `from molmetal_lam.reactions.rule_registry import register_rule`
  (a top-level path with **no** `molmetal.` prefix).
* Our test imports `from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import ...`
  (the canonical nested path).

These two paths point to the SAME `.py` files but Python creates
TWO separate module instances — each with its own `_REGISTRY`
dict.  If we used the nested path for `list_rules()` introspection,
the count would silently disagree with the number of rules
`register_metal_ligand_exchange_rules()` actually inserted.

The fix is to import `list_rules` via the **top-level shim**
(`molmetal_lam.reactions.rule_registry`) so both registration and
introspection land in the same module instance.  This invariant is
documented inline in the test (see Notes block).

#### test_005 — End-to-end pipeline smoke
Invokes `metallo_drug_smoke_retrain.py` as a **subprocess** with
the smallest CPU-friendly config
(`--steps 100 --batch-size 4 --seed 0 --hidden-dim 64 --n-layers 2`).
The test asserts:
* exit code 0
* `final.json` written
* `ckpt.pt` saved with at least one finite tensor in
  `velocity_field` (proves the adapter trained to finite weights)

Quality assertions (decode_ratio, bond_loss) intentionally belong
to the canonical R17 script tests; this test is a regression guard
for the script's **invocation contract**.

---

## 3. Run output (2026-09-25)

```
$ cd /home/hugo/codes/mol-metal && source env.sh && \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_integration.py -v

molmetal/tests/test_integration.py::test_001_lipman_to_vina_e2e PASSED [ 20%]
molmetal/tests/test_integration.py::test_002_mcts_sa_qed_pipeline PASSED [ 40%]
molmetal/tests/test_integration.py::test_003_lambda_astsr_coupling_adapter PASSED [ 60%]
molmetal/tests/test_integration.py::test_004_mle_matrix_click_compat PASSED [ 80%]
molmetal/tests/test_integration.py::test_005_end_to_end_smoke PASSED     [100%]

=================== 5 passed, 3 warnings in 70.99s (0:01:10) ===================
```

Per-test wall-clock:

| Test | Time | Notes |
|---|---|---|
| `test_001` | ~17 s | Includes 10 × ~3 s Vina dockings |
| `test_002` | ~4 s | MCTS 5 calls × 20 simulations |
| `test_003` | ~2 s | ASTSR fit + 50-row embedding loop |
| `test_004` | ~2 s | MLE registration + scaffold detection |
| `test_005` | ~53 s | Full smoke retrain subprocess (100 steps) |

---

## 4. Files

| Path | Change | Description |
|---|---|---|
| `molmetal/tests/test_integration.py` | NEW | The 5 cross-module integration tests. |
| `molmetal/reports/wf_integration_tests/final.md` | NEW | This report. |

No production code was modified — the tests exercise the **existing**
modules through their public APIs.

---

## 5. Honest framing

* **Lipman → SMILES hand-off (test_001)**: the curated 10-SMILES
  list is a *proxy* for the Lipman→Vina handoff.  Without a
  trained Lipman checkpoint we cannot test a fully learned handoff;
  we therefore test the *import chain + dataclass contract*.  A
  learned-handoff integration test would need a separately-trained
  checkpoint and would belong in the canonical R17 pipeline tests,
  not the small-scale I1 extension.

* **Identity rule (test_002)**: the MCTS search uses a stub
  identity rule because we are testing the **pipeline**, not the
  rule's chemistry.  Real-chemistry MCTS integration is already
  covered by `molmetal/tests/test_proof_search_strengthened.py`.

* **Compiling-to-real-PDB (test_001 / test_005)**: Vina + meeko +
  `mk_prepare_receptor.py` must be installed for tests 1 and 5 to
  pass.  These are standard deps in the project's
  `env.sh`-provisioned venv; if they are absent the suite will fail
  loudly (no silent stub fallback).

* **mk_prepare_receptor.py PATH**: the test module patches
  `os.environ["PATH"]` at import time to include
  `/home/hugo/codes/CadButEaas/.venv-vlm/bin` so the
  `mk_prepare_receptor.py` subprocess call inside Vina's
  `_prepare_receptor` succeeds regardless of the host shell's
  default PATH.  This is a robustness layer; the canonical
  `env.sh` should also include this directory in PATH (currently
  it only sets `VIRTUAL_ENV`).

* **Module-identity invariant (test_004)**: see §2.1 test_004
  rationale.  An accidental `import molmetal.molmetal_lam...` would
  silently materialise a second `_REGISTRY` and break the count
  assertion.  The test deliberately uses the top-level shim
  (`import molmetal_lam` + `from molmetal_lam.reactions...`) and
  documents the invariant in a "Notes" block at module top.

---

## 6. Related / next steps

* The 5 tests are drop-in additions to the canonical pytest
  collection (`molmetal/tests/`).  No `conftest.py` or pytest
  plugin changes were needed.
* `test_005`'s wall-clock (53 s) is the largest in the suite.  If
  the budget tightens, drop `--steps 100` to `--steps 50` (~25 s)
  and the assertion still holds (training 50 steps is enough to
  emit a finite-tensor checkpoint).
* For the next integration round, the natural extensions are:
  * Lipman adapter + posebusters (post-docking filter)
  * Coupling adapter + warm_start (pocket-feature→MCTS prior)
  * SynFlowNet + pt_click_compat (reaction-rule wiring through
    click-compat gating)