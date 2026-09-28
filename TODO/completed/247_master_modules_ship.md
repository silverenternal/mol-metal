# TODO-247 — Master modules M1 + M2 ship (Wf-247-D, 2026-09-28)

**Date**: 2026-09-28
**Status**: SHIPPED — both master modules + tests + documentation.
**Priority**: P1 (innovation DRY refactor — feeds TODO-119 / 120 / 121 / 122 / 123 / 124 / 125).
**Parent**: TODO-126 §3 — single-master Wasserstein module + single-master Catherine bias module = 2 NEW modules + reuse = DRY across 6 strengthen TODOs.
**Constraint source**: TODO-126 §3 (shared machinery table) + TODO-129 §4 (R21-P2 ship surface).

---

## §0 TL;DR

This TODO ships **two master facade modules** that expose a single
`compute` / `apply` API on top of the existing canonical W₂ + Catherine
machinery.  The master modules are **additive** — they do NOT modify
or deprecate any existing public symbol.  New `feature_flag` selector
(ASTSR-side, TODO-122) added on the Catherine side; new
`loss_type="gradient_flow"` selector (`wasserstein_loss` alias) added
on the Wasserstein side.

| Module | Path | Lines | Tests | Backward compat |
|---|---|---:|---:|---|
| **M1** Master Wasserstein gradient flow | `molmetal/molmetal_lam/reward/master_wasserstein_gradient_flow.py` | ~340 | 28 PASS | canonical primitives untouched |
| **M2** Master Catherine closure bias | `molmetal/molmetal_lam/reward/master_catherine_closure_bias.py` | ~480 | 28 PASS | canonical sub-modules untouched |
| **M1 tests** | `molmetal/tests/test_master_wasserstein_gradient_flow.py` | 28 tests | 28 PASS | NEW |
| **M2 tests** | `molmetal/tests/test_master_catherine_closure_bias.py` | 28 tests | 28 PASS | NEW |

**Constraint check (FLAT)**:

| Constraint | Status |
|---|---|
| CPU-only | YES (no CUDA / ROCm / Triton) |
| Backward compat — `chemprop_v2_v3` + `hilbert_astsr_end_to_end` still work | YES (compile-check + 87 existing tests pass) |
| `Edit` tool minimal changes | YES (only NEW files — no edits to canonical modules) |
| Every NEW scoring component has explicit `engine` tag | YES (4 engine tags shipped: `wasserstein_reg`, `closure_catherine_oversample`, `closure_catherine_bounded`, `closure_catherine_feature_flag`) |

---

## §1 M1 — Master Wasserstein gradient flow

### §1.1 Public API

```python
from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
    MasterWassersteinGradientFlow,
    MasterWassersteinConfig,
    MasterWassersteinResult,
    SUPPORTED_LOSS_TYPES,
    ENGINE_TAG,                # = "wasserstein_reg"
    build_master_verification,
)

master = MasterWassersteinGradientFlow(
    config=MasterWassersteinConfig(n_mc=1000, seed=0),
)

# Selector 1: cross-metal transfer (TODO-121 / TODO-125)
per_metal = {"Ru": emb_ru, "Ir": emb_ir, "Os": emb_os, "Rh": emb_rh}
res = master.compute(per_metal, loss_type="transfer")
# → MasterWassersteinResult(loss_value=0.123, extras={"n_metals": 4, ...}, engine="wasserstein_reg")

# Selector 2: pIC50 manifold (TODO-120 / TODO-122)
res = master.compute(pred_pic50, ref_pic50_dist, loss_type="pic50")

# Selector 3: generic W₂ gradient flow (TODO-119 / TODO-123)
res = master.compute(pred_dist, ref_dist, loss_type="gradient_flow")

# Decay gate (JKO-style CFL)
master.append_history(res.loss_value)
master.decay_ok(window=10, tol=1e-3)        # True / False
```

### §1.2 Selector semantics

| `loss_type` | Input | Delegates to | Returns |
|---|---|---|---|
| `"transfer"` | `Mapping[str, Tensor]` of per-metal embeddings | `wasserstein_loss.wasserstein_transfer_loss` | scalar mean pairwise W₂ + `extras={"n_metals", "per_metal_n"}` |
| `"pic50"` | 1-D (or `(N, 1)`) predicted pIC50 + 1-D reference pIC50 | `wasserstein_loss.wasserstein_pic50_loss` | scalar 1-D W₂ + `extras={"pred_n", "ref_n"}` |
| `"gradient_flow"` | 2-D `(N, d)` pred + 2-D `(M, d)` ref | `wasserstein_loss.wasserstein_loss` | scalar sliced-W₂ + `extras={"pred_n", "ref_n"}` |

Each selector is a **thin façade** — the underlying primitive is called
verbatim, so a caller that switches from `wasserstein_loss(...)` to
`MasterWassersteinGradientFlow().compute(..., loss_type="gradient_flow")`
gets the *same* scalar value (validated in
`TestTransferSelector.test_matches_canonical_primitive` +
`TestPic50Selector.test_matches_canonical_primitive` +
`TestGradientFlowSelector.test_matches_canonical_primitive`).

### §1.3 Backward compat (canonical primitives untouched)

* `molmetal.molmetal_lam.reward.wasserstein_loss` — every public symbol
  re-exported and unchanged.
* `molmetal.molmetal_lam.reward.wasserstein_transfer` — every public
  symbol (`WassersteinTransferLoss`, `WassersteinPairResult`,
  `WassersteinTransferResult`, `pairwise_w2_matrix`) unchanged.

The master adds a **new top-level entry point** with a uniform API;
it does not deprecate any existing import path.  All callers of the
canonical APIs (train_chemprop_v2_v3.py, train_hilbert_astsr_end_to_end.py,
bench_deflexpressor_end_to_end.py, triangle_closure.py, ...) continue
to work without edits.

### §1.4 Test surface (28 tests in test_master_wasserstein_gradient_flow.py)

| Test class | Coverage |
|---|---|
| `test_canonical_primitives_unchanged` (1) | Backward-compat import sanity |
| `TestConstants` (2) | `SUPPORTED_LOSS_TYPES`, `ENGINE_TAG` values |
| `TestMasterWassersteinConfig` (3) | Defaults + validation (n_mc ≥ 1, int coercion) |
| `TestMasterWassersteinResult` (2) | Dataclass shape + default engine tag |
| `TestTransferSelector` (3) | Scalar return + extras + type error + **canonical-value match** |
| `TestPic50Selector` (3) | Scalar return + extras + ref-required + **canonical-value match** |
| `TestGradientFlowSelector` (5) | Scalar return + extras + type errors + **canonical-value match** + zero-on-identical |
| `TestSelectorValidation` (2) | Unknown selector raises; default = `transfer` |
| `TestDecayGate` (4) | Decreasing → True; growing → False; empty → True; history copy |
| `TestBuildMasterVerification` (2) | Round-trip shape + CPU-only marker |
| `TestToDict` (1) | Diagnostic snapshot |

Run: `/home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_master_wasserstein_gradient_flow.py -q` → **28 passed in 0.19s**.

---

## §2 M2 — Master Catherine closure bias

### §2.1 Public API

```python
from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
    MasterCatherineClosureBias,
    MasterCatherineBiasResult,
    CatherineClosureFeatureFlag,
    SUPPORTED_BIAS_TYPES,    # = ("oversample", "generation_bias", "feature_flag")
    ENGINE_OVERSAMPLE,        # = "closure_catherine_oversample"
    ENGINE_GENERATION_BIAS,   # = "closure_catherine_bounded"
    ENGINE_FEATURE_FLAG,      # = "closure_catherine_feature_flag"
    build_master_verification,
)

master = MasterCatherineClosureBias(
    bias_on_pass=1.0,
    bias_on_fail=0.0,
)

# Selector A: oversample (TODO-121 — synthetic Rh cohort)
res = master.apply(
    {"target_metal": "Rh", "n": 500, "per_scaffold_cap": 50, "max_depth": 2},
    bias_type="oversample",
)
# → engine="closure_catherine_oversample"

# Selector B: generation_bias (TODO-119 / 124 / 125 — per-pocket candidates)
res = master.apply(
    {"pocket_id": "1h36", "n": 100},
    bias_type="generation_bias",
)
# → engine="closure_catherine_bounded"

# Selector C: feature_flag (NEW — TODO-122 / ASTSR / Deflexpressor)
res = master.apply(
    {
        "term_id": "astsr_term_001",
        "metal": "Pt",
        "witness_path": ["CuAAC"],
        "smiles": "Nc1nc(N)nc(N)n1",
    },
    bias_type="feature_flag",
)
# → engine="closure_catherine_feature_flag"
# → payload = {"term_id": ..., "six_of_six_pass": True/False,
#              "per_property": {P1..P6: bool}, "bias_used": 0.0|1.0}

# Batch the ASTSR gate (Deflexpressor runs 1000s of rows per epoch)
res = master.apply([{"term_id": f"t_{i}", ...} for i in range(1024)],
                    bias_type="feature_flag")
# → res.feature_flags carries the batched per-row flags
```

### §2.2 Selector semantics

| `bias_type` | Payload | Delegates to | Returns |
|---|---|---|---|
| `"oversample"` | `{"target_metal": str, "n": int, [opts]}` | `pocket_closure_bias.closure_oversample` → `closure_oversample.ClosureOversampler.run()` | full cohort dict + 6/6 pass-rate + engine tag |
| `"generation_bias"` | `{"pocket_id": str, "n": int, [opts]}` | `pocket_closure_bias.closure_bounded_smiles_generation` | per-pocket candidates + Catherine 6/6 filter + engine tag |
| `"feature_flag"` *(NEW)* | `dict` or `list[dict]` with `term_id`, `metal`, `witness_path`, `smiles`, optional `n_visited` / `max_depth_seen` / `coord_violations` | inline proxy via `math_skills.catherine_properties.synthetic_closure_summary` + `verify_all_catherine_properties` | per-row binary gate + per-property dict + `bias_used` (1.0 on PASS, 0.0 on FAIL by default) |

### §2.3 The new `feature_flag` selector (TODO-122 / ASTSR)

This selector is the **only** new selector shipped by Wf-247-D and is
**not** a façade — it is a new path that the canonical
`closure_oversample` and `pocket_closure_bias` modules do **not**
expose.  It is needed because:

* ASTSR (Deflexpressor symbolic regression on λ-terms, TODO-122)
  runs **thousands of fits per epoch** across heterogeneous term
  populations.  A per-row full Catherine 6/6 sweep would be
  cost-prohibitive.
* The lightweight proxy
  (`synthetic_closure_summary` + `verify_all_catherine_properties`)
  is sufficient for the binary gate role: it consumes the
  witness_path + a SMILES coord-violation heuristic + a
  pre-computed `n_visited` / `max_depth_seen` and returns
  `six_of_six_pass` + per-property dict in O(1) per row.
* The result exposes `bias_used` directly so the caller can:
  * filter the ASTSR regression cohort (drop FAIL rows);
  * use it as a continuous ASTSR feature column;
  * weight the ASTSR loss via `bias_used` as a per-row weight.

### §2.4 Backward compat (canonical sub-modules untouched)

* `molmetal.molmetal_lam.reward.closure_oversample` — every public
  symbol (`ClosureOversampler`, `ClosureOversampleResult`,
  `RH_SEED_SCAFFOLDS`, `RICH_PARTNERS`, `CLOSURE_PARTNERS`,
  `RhSyntheticCompound`) unchanged.
* `molmetal.molmetal_lam.reward.pocket_closure_bias` — every public
  symbol (`PocketBiasVerdict`, `BoundedGenerationResult`,
  `pocket_specific_closure_bias`, `closure_bounded_smiles_generation`,
  `closure_oversample`, `per_pocket_pharmacophore_lookup`,
  `build_per_function_verification`, all `SUPPORTED_POCKETS` /
  `FIVE_CLICK_RULES` / `METAL_COORD_RULES` tables) unchanged.

The master adds a **new top-level entry point** with a uniform API;
it does not deprecate any existing import path.  All callers of the
canonical sub-modules continue to work without edits.

### §2.5 Test surface (28 tests in test_master_catherine_closure_bias.py)

| Test class | Coverage |
|---|---|
| `test_canonical_submodules_unchanged` (1) | Backward-compat import sanity |
| `TestConstants` (2) | `SUPPORTED_BIAS_TYPES`, `ENGINE_*` distinct values |
| `TestInitValidation` (7) | Defaults + 3+3 parametrized bias_on_pass/fail out-of-range |
| `TestFeatureFlag` (7) | Single + batch + bias_used + custom bias + empty + missing term_id + type error |
| `TestGenerationBias` (3) | Known pocket → candidates; unknown pocket → empty + notes; type error |
| `TestOversample` (2) | Rh cohort minimal (n=12) + type error |
| `TestSelectorValidation` (2) | Unknown selector raises; default = `feature_flag` |
| `TestToDict` (1) | Diagnostic snapshot |
| `TestBuildMasterVerification` (1) | Round-trip shape + CPU-only marker |
| `TestCatherineClosureFeatureFlag` (2) | Default engine + tag constant |

Run: `/home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_master_catherine_closure_bias.py -q` → **28 passed in 0.39s**.

---

## §3 Backward-compat suite (canonical tests, no regressions)

Combined run including the new + canonical tests (4 suites):

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_master_wasserstein_gradient_flow.py \
    molmetal/tests/test_master_catherine_closure_bias.py \
    molmetal/tests/test_wasserstein_loss.py \
    molmetal/tests/test_sa_closure_bias.py -q
144 passed, 1 warning in 4.17s
```

Plus the regression-heavy suites:

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_lomo_wasserstein.py \
    molmetal/tests/test_wasserstein_transfer_rh.py \
    molmetal/tests/test_pocket_closure_bias.py -q
87 passed, 1 warning in 108.21s (0:01:48)
```

Constraint scripts (`chemprop_v2_v3`, `hilbert_astsr_end_to_end`):

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -c "import py_compile; \
    py_compile.compile('molmetal/scripts/train_chemprop_v2_v3.py', doraise=True); \
    py_compile.compile('molmetal/scripts/train_hilbert_astsr_end_to_end.py', doraise=True); \
    print('OK')"
chemprop_v2_v3.py + hilbert_astsr_end_to_end.py compile OK
```

**No breaking changes** to any canonical public symbol.

---

## §4 Honest framing + scope notes

### §4.1 What this TODO ships

* **Two master façades** (M1 + M2) that expose a single uniform
  `compute(...)` / `apply(...)` API on top of the existing canonical
  modules.
* **One new selector** on the Catherine side (`"feature_flag"` —
  ASTSR/Deflexpressor Boolean gate).
* **One new selector alias** on the Wasserstein side
  (`"gradient_flow"` — delegates to `wasserstein_loss`).
* **56 unit tests** (28 + 28) covering the master APIs, the
  backward-compat requirements, the decay gate, the build verification,
  and the new ASTSR feature_flag batched path.
* **Full backward compatibility** — canonical primitives untouched,
  existing scripts (`train_chemprop_v2_v3.py`,
  `train_hilbert_astsr_end_to_end.py`, etc.) still compile + their
  test suites still pass.

### §4.2 What this TODO does NOT ship

* **No new scoring channel** — the master modules are pure facades;
  every result delegates to a canonical primitive whose value is
  propagated unchanged.  `engine` tags are added but the loss values
  themselves are NOT new scoring components.
* **No metric lift** — TODO-119 / 120 / 121 / 122 / 123 / 124 / 125
  remain the responsibility of their existing strengthen TODOs.  This
  TODO closes the DRY refactor (TODO-126 §3) but does NOT close the
  underlying strengthen TODOs.
* **No GPU work** — both modules are CPU-only.  GPU ports (e.g.
  batched sliced-W₂ on `B×N_per_metal × d` hidden states) are the
  responsibility of TODO-119 P1.3 / TODO-120 P1.3 etc.

### §4.3 Per-party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED)

* **MEASURED by this TODO**: master API surface, selector dispatch,
  canonical-value match (3 separate assertions), backward-compat
  import sanity (4 canonical modules + 2 scripts), 56 unit tests
  PASS, 87 regression tests PASS.
* **PROJECTED but NOT MEASURED**: end-to-end downstream metric lift
  on m1 / m5 / m14 / m15 / m19 / PlatinAI R² — those are downstream
  TODO-119 / 120 / 121 / 122 / 123 / 124 / 125 deliverables.

---

## §5 TODO-126 §3 closing entry

This TODO closes the **M1 + M2 master module** row of the TODO-126
§3 shared-machinery table.  Subsequent strengthen-TODO work
(TODO-119 P1.3 / TODO-120 P1.3 / TODO-121 P2 etc.) can wire through
the master facade with a single `compute` / `apply` call instead of
importing the canonical primitives directly.

| TODO-126 row | Status before this TODO | Status after this TODO |
|---|---|---|
| **M1** Wasserstein gradient flow master module | NOT built (canonical `wasserstein_loss.py` + `wasserstein_transfer.py` only) | **BUILT** — `master_wasserstein_gradient_flow.py` + 28 tests |
| **M2** Catherine closure bias master module | NOT built (`closure_oversample.py` + `pocket_closure_bias.py` only, application-specific) | **BUILT** — `master_catherine_closure_bias.py` + 28 tests + **new `feature_flag` selector for ASTSR** |

---

## §6 Cross-references

* TODO-126 §3 (shared machinery table — M1 + M2 master modules).
* TODO-129 §4 R21-P2 (Catherine feature_flag for ASTSR / TODO-122).
* molmetal/scripts/train_chemprop_v2_v3.py (canonical import of
  `wasserstein_transfer_loss` unchanged).
* molmetal/scripts/train_hilbert_astsr_end_to_end.py (canonical
  import of `wasserstein_pic50_loss` unchanged).
* molmetal/scripts/triangle_closure.py (canonical closures unchanged).
* molmetal/scripts/bench_deflexpressor_end_to_end.py
  (`LambdaASTSR` import unchanged).
* molmetal/molmetal_lam/reward/wasserstein_loss.py (canonical W₂
  primitives — unchanged).
* molmetal/molmetal_lam/reward/wasserstein_transfer.py (canonical
  W₂ transfer wrapper — unchanged).
* molmetal/molmetal_lam/reward/closure_oversample.py (canonical
  closure oversampler — unchanged).
* molmetal/molmetal_lam/reward/pocket_closure_bias.py (canonical
  per-pocket closure-bias — unchanged).
* molmetal/molmetal_lam/reward/README.md (catalog entry — TBD by
  Wf-247 follow-up).
