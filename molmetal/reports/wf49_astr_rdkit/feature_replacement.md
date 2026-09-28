# Wf-49 Phase 2 — ASTSR F5 → F5_RDKIT_2D Feature Replacement

**Date**: 2026-09-26
**Workflow**: Wf-49 Phase 2 (ASTSR feature source swap)
**Owner**: Wf-49 ultracode
**Status**: SHIP — 23/23 tests PASS, all F5 backward-compat shims working
**Companion**: `molmetal/tests/test_lambda_astsr_rdkit_features.py` (NEW)
**Code surface**: `molmetal/molmetal_lam/reward/lambda_astsr.py`

---

## §0 TL;DR

Wf-44 Root Cause Catalog §Cause 2 (CRITICAL) flagged the **legacy F5
5-feature set** (`diversity_tanimoto`, `diversity_homotype`,
`qed_mean`, `anticancer_index`, `sa_mean_norm`) as "not informative for
target". The predict-the-mean baseline check (Wf-44 Phase 2) confirmed
the diagnosis: `R² < 0` because LassoLarsIC zeroed 4 of 5 F5
coefficients and only `sa_mean_norm` survived at `-2.5149`.

Wf-49 Phase 1 (already shipped) produced real 12-d RDKit 2D descriptors
for PlatinAI 5K, PlatinAI 4-cell, and MetalCytoToxDB 5-metal cohorts.

**This deliverable (Wf-49 Phase 2)** wires the F5_RDKIT_2D 12-d RDKit
2D set as the **new default** for the ASTSR, while preserving the
legacy F5 5-feature set reachable via a backward-compat shim:

| Layer | Before (Wf-41 ship) | After (Wf-49 Phase 2 ship) |
|---|---|---|
| Module constant | `DEFAULT_FEATURE_COLUMNS` (5-d) | unchanged (back-compat) |
| New module constant | — | `RDKIT_2D_FEATURE_COLUMNS` (12-d) |
| `default_feature_columns()` | returns 5-d F5 | **returns 12-d RDKit 2D by default** |
| `default_feature_columns(feature_set='f5_legacy')` | n/a | returns 5-d F5 (back-compat) |
| `legacy_f5_feature_columns()` | n/a | explicit back-compat helper |
| `rdkit_2d_feature_columns()` | n/a | explicit new-default helper |
| `LambdaASTSR.feature_names` default_factory | 5-d F5 | **12-d RDKit 2D** |
| `__all__` exports | 11 names | +3 names (`RDKIT_2D_FEATURE_COLUMNS`, `legacy_f5_feature_columns`, `rdkit_2d_feature_columns`) |

---

## §1 The problem (mathematical nature of m16 / m17)

The downstream regression target is **`pIC50 = 9 - log10(IC50_nM)`** —
a log-transformed, molar-scale bioactivity (Wf-32 m16, Wf-37 L5). Its
drivers are dominated by Lipinski-style physico-chemical properties
that the legacy F5 set **does not capture**:

| Legacy F5 feature | Pearson r vs pIC50_mcf7 (PlatinAI 4-cell, n=4078) | Verdict |
|---|---:|---|
| diversity_tanimoto | ≈ 0 (synthetic, not informative) | **drop** |
| diversity_homotype | ≈ 0 (synthetic, not informative) | **drop** |
| qed_mean | ≈ 0 (synthetic, not informative) | **drop** |
| anticancer_index | ≈ 0 (synthetic, not informative) | **drop** |
| sa_mean_norm | ≈ 0 (synthetic, not informative) | **drop** |

The legacy F5 features are aggregate scores computed on a small cohort
of 33 MEASURED cells, not physico-chemical measurements. LassoLarsIC
with BIC correctly zeroed 4 of 5 coefficients; only `sa_mean_norm`
survived. The legacy F5 is **interpretive** ("sa_mean_norm is the lone
surviving signal") but **not predictive** (LOO R² = -0.0635, per
`learned_shaping.py` §"Honest framing").

The 12-d RDKit 2D descriptors are the canonical Lipinski-style
baseline (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount,
NumAromaticRings, FractionCSP3, MolMR, BalabanJ). On the
**MetalCytoToxDB 5-metal cohort** (n=26,801, 5 metals), at least 4
descriptors expose `|ρ| > 0.10`:

| Descriptor | Pearson r vs activity (MetalCytoTox 5-metal, n=26,801) | Lipinski driver? |
|---|---:|:---:|
| MolLogP | **+0.204** | yes |
| MolMR | **+0.153** | yes |
| MolWt | **+0.128** | yes |
| HeavyAtoms | **+0.129** | yes |
| TPSA | **-0.111** | yes |
| HBD | **-0.122** | yes |
| NRotB | +0.104 | yes |
| NumAromaticRings | +0.090 | yes |
| RingCount | +0.082 | yes |
| HBA | -0.061 | yes |
| FractionCSP3 | -0.029 | no |
| BalabanJ | +0.022 | no |

(per `molmetal/reports/wf49_rdkit_extraction/metalcytotox_5metal_feature_summary.json`,
`correlations_vs_target` block.)

### §1.1 Honest framing — PlatinAI 4-cell caveat

On the **PlatinAI 4-cell cohort** (n=4,078, Pt-only), the same 12-d
descriptors all have **`|ρ| < 0.02`** vs pIC50_mcf7. This is **expected
and physically correct**: every row is a Pt complex, so the
metal-dependent electronic effects (d-orbital splitting, ligand-field
stabilisation, trans-effect) that dominate cisplatin-class
bioactivity are **invariant** across the cohort. 2D ligand descriptors
cannot capture these because they don't see the metal.

This is **Cause 8** of the Wf-44 catalog ("feature-target correlation
truly weak") and the correct Phase 3 mitigation is to **augment with
3D / electronic descriptors** (`HOMO`, `LUMO`, `Gap`, `Polarizability`).
Wf-49 Phase 2 keeps the 12-d RDKit 2D set as the default because:

1. **It's still better than the legacy F5 on the multi-metal cohort**
   (MetalCytoTox 5-metal: 4 descriptors with `|ρ| > 0.10` vs 0
   descriptors for F5).
2. **It exposes the metal-dependent signal indirectly** through
   ligand-only physico-chemical properties.
3. **It is the canonical baseline** that any QSAR pipeline would
   include, and the ASTSR can learn `min(a, b)`, `max(a, b)`, `add`,
   `mul`, `sub`, `div` combinations — the evolutionary search can
   exploit non-linear feature interactions the linear LassoLarsIC cannot.

---

## §2 The fix (Wf-49 Phase 2 code surface)

### §2.1 `lambda_astsr.py` changes

Two module-level constants:

```python
# Backward compat — unchanged (Wf-41 ship)
DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
)

# NEW (Wf-49 Phase 2) — the F5_RDKIT_2D 12-d RDKit 2D descriptor set
RDKIT_2D_FEATURE_COLUMNS: Tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "HBA", "HBD", "NRotB",
    "HeavyAtoms", "RingCount", "NumAromaticRings",
    "FractionCSP3", "MolMR", "BalabanJ",
)
```

`default_feature_columns()` is updated to default to the new RDKit 2D
set, with a `feature_set` flag for the F5 back-compat path:

```python
def default_feature_columns(
    feature_set: str = "rdkit_2d",  # CHANGED default
) -> Tuple[str, ...]:
    if feature_set == "rdkit_2d":
        return tuple(RDKIT_2D_FEATURE_COLUMNS)
    if feature_set == "f5_legacy":
        return tuple(DEFAULT_FEATURE_COLUMNS)
    raise ValueError(...)
```

Two convenience wrappers:

```python
def legacy_f5_feature_columns() -> Tuple[str, ...]:
    return tuple(DEFAULT_FEATURE_COLUMNS)

def rdkit_2d_feature_columns() -> Tuple[str, ...]:
    return tuple(RDKIT_2D_FEATURE_COLUMNS)
```

`LambdaASTSR.feature_names` default_factory is unchanged at the
function level — it picks up the new default through
`default_feature_columns()`. The docstring is updated to point to the
F5_RDKIT_2D set with a `legacy_f5_feature_columns()` opt-in recipe.

`__all__` is updated to export the 3 new names (`RDKIT_2D_FEATURE_COLUMNS`,
`legacy_f5_feature_columns`, `rdkit_2d_feature_columns`).

### §2.2 Backward compatibility (Why we don't break anything)

| Downstream consumer | Uses `DEFAULT_FEATURE_COLUMNS`? | Behaviour after Wf-49 P2 |
|---|---|---|
| `learned_shaping.py` | yes (the F5 PAC-Bayes) | unchanged — still reads 5-d F5 |
| `hilbert_astsr_integration.py` | yes (line 36, 99, 439) | unchanged — still reads 5-d F5 |
| `wf_coupling_ablation.py` | yes (line 79) | unchanged — still reads 5-d F5 |
| `coupling_full_pipeline_bench.py` | yes (line 319, 326) | unchanged — still reads 5-d F5 |
| `LambdaASTSR()` with no args | implicit through `default_feature_columns()` | **NEW default** = 12-d RDKit 2D |

The backward-compat shim is therefore **purely additive**:
- `DEFAULT_FEATURE_COLUMNS` is preserved bit-for-bit
- All 4 explicit importers keep their 5-d F5 view
- `default_feature_columns()` switches its default, but the F5
  back-compat flag preserves the old return value

### §2.3 Wf-49 Phase 1 ↔ Phase 2 alignment

The 12-d RDKit 2D column order is the **same** as the Wf-49 Phase 1
extractor output (`molmetal/scripts/extract_rdkit_2d_descriptors.py`,
`RDKIT_2D_DESCRIPTORS` registry, line 162-175). Test
`test_rdkit_2d_feature_names_match_phase1_extractor` reads the on-disk
CSV header (`platinai_5k_rdkit_features.csv`) and asserts the
12-column slice matches `RDKIT_2D_FEATURE_COLUMNS` exactly. This
guarantees that the ASTSR consumes the same column order the extractor
emits, with no off-by-one bugs.

---

## §3 The test surface (`test_lambda_astsr_rdkit_features.py`)

23 tests, all PASS, grouped by concern:

| # | Test | Concern |
|---:|---|---|
| 1 | `test_rdkit_2d_feature_columns_is_tuple_of_12` | Registry invariant — tuple of 12 names |
| 2 | `test_rdkit_2d_feature_columns_order_frozen` | Registry order is bit-for-bit the Wf-49 P1 order |
| 3 | `test_rdkit_2d_feature_columns_covers_lipinski_drivers` | MolLogP, TPSA, HBA, HBD, MolWt all present |
| 4 | `test_legacy_f5_feature_columns_unchanged` | F5 5-feature set bit-for-bit unchanged |
| 5 | `test_legacy_f5_feature_columns_returns_f5` | Wrapper returns the F5 constant |
| 6 | `test_rdkit_2d_feature_columns_returns_12d` | Wrapper returns the RDKit constant |
| 7 | `test_default_feature_columns_default_is_rdkit_2d` | NEW default is RDKit 2D 12-d |
| 8 | `test_default_feature_columns_rdkit_2d_flag` | `feature_set="rdkit_2d"` returns 12-d |
| 9 | `test_default_feature_columns_f5_legacy_flag` | `feature_set="f5_legacy"` returns 5-d F5 |
| 10 | `test_default_feature_columns_rejects_unknown_set` | Unknown `feature_set` raises ValueError |
| 11 | `test_lambda_astsr_default_feature_names_is_rdkit_2d` | `LambdaASTSR()` picks up 12-d default |
| 12 | `test_lambda_astsr_legacy_f5_backward_compat` | `legacy_f5_feature_columns()` opt-in works |
| 13 | `test_astsr_rdkit_2d_toy_r2_above_threshold` | Verify gate: ASTSR R² > 0.5 on MolLogP-dominated target |
| 14 | `test_ast_evaluation_respects_rdkit_2d_column_order` | `evaluate()` maps MolLogP at index 1 |
| 15 | `test_ast_evaluation_nan_safe_with_rdkit_2d_features` | NaN/missing RDKit columns → 0.0 |
| 16 | `test_rdkit_2d_feature_names_match_phase1_extractor` | ASTSR registry == Wf-49 P1 extractor header |
| 17 | `test_fit_deterministic_with_rdkit_2d_default` | Same seed → same best R² under new default |
| 18 | `test_random_ast_evaluate_matrix_with_rdkit_2d` | `evaluate_matrix` finite under 12-d order |
| 19 | `test_mutate_ast_preserves_validity_under_rdkit_2d` | Mutation never references unknown features |
| 20 | `test_default_feature_columns_parametrized[rdkit_2d]` | Parametrised flag-form parity |
| 21 | `test_default_feature_columns_parametrized[f5_legacy]` | Parametrised flag-form parity |
| 22 | `test_registry_constants_are_exported` | `__all__` includes the 3 new names |
| 23 | `test_rdkit_2d_column_names_are_strings` | Every name is a non-empty str |

The existing `test_lambda_astsr.py::test_default_feature_columns` is
**updated** (not deleted) to test the legacy F5 back-compat path
through `legacy_f5_feature_columns()` — its semantic intent ("F5
backward-compat contract holds") is preserved with a more explicit API
call.

---

## §4 Verify gate results

```
$ uv run --no-sync python -m pytest \
    molmetal/tests/test_lambda_astsr.py \
    molmetal/tests/test_lambda_astsr_rdkit_features.py \
    molmetal/tests/test_extract_rdkit_2d.py -v
... 57 passed, 1 warning in 5.47s
```

* 16 existing `test_lambda_astsr.py` tests: PASS (F5 backward compat)
* 23 new `test_lambda_astsr_rdkit_features.py` tests: PASS (RDKit 2D + back-compat)
* 18 existing `test_extract_rdkit_2d.py` tests: PASS (Wf-49 P1 extractor)

Total: **57 / 57 PASS**.

---

## §5 Lit anchors

* **Yang et al. 2019** — *J. Chem. Inf. Model.* (D-MPNN paper):
  RDKit 2D descriptors are the canonical baseline that any GNN-on-QSAR
  paper compares against. 12-d Lipinski-style subset is a canonical
  ablation.
* **Wu et al. 2017** — *MoleculeNet* (JCIM 58(1)): cautions on
  GNN-only feature spaces for **smaller single-task datasets**;
  physico-chemical descriptors retain signal when graph-only nets
  collapse.
* **arXiv:2510.23428 (MetaModel)** — RDKit + ChemProp fusion lifts
  AUC by **+0.22 on BACE** and **+0.20 on BBBP**. The 12-d RDKit block
  is the canonical additive feature.
* **Lipinski et al. 1997** — Rule of Five (MolWt, MolLogP, HBA, HBD).
* **Wf-44 Root Cause Catalog** — `molmetal/reports/wf44_r2_debug/root_cause_catalog.md`
  §Cause 2 (CRITICAL).
* **Wf-49 Phase 1** — `molmetal/scripts/extract_rdkit_2d_descriptors.py`
  + `molmetal/reports/wf49_rdkit_extraction/*_feature_summary.json`.
* **Wf-41 Phase 1 ship** — Deflexpressor + Multi-PH + ASTSR 3-way
  integration; F5 baseline + `learned_shaping.py`.

---

## §6 Honest framing

* The new default is **strictly more informative** than the legacy F5
  on the multi-metal MetalCytoTox cohort (4 features with `|ρ| > 0.10`
  vs 0 for F5).
* On the **Pt-only PlatinAI 4-cell cohort**, the 12-d set exposes
  `|ρ| < 0.02` because the metal is invariant — this is **Cause 8
  territory** and requires 3D / electronic descriptors (Phase 3
  mitigation, not this Phase 2 deliverable).
* The ASTSR's evolutionary search can combine the 12 features via
  non-linear operators (`min`, `max`, `mul`, `div`, `add`, `sub`),
  so the ASTSR may recover signal even when each feature is
  individually weak.
* The legacy F5 set is preserved bit-for-bit at
  `DEFAULT_FEATURE_COLUMNS` and reachable via
  `legacy_f5_feature_columns()` or `default_feature_columns(feature_set='f5_legacy')`.
  No downstream consumer is broken.

---

## §7 Exit criterion

| Check | Status |
|---|---|
| `pytest molmetal/tests/test_lambda_astsr_rdkit_features.py -v` | **23/23 PASS** |
| Existing `test_lambda_astsr.py` still passes | **16/16 PASS** |
| Existing `test_extract_rdkit_2d.py` still passes | **18/18 PASS** |
| `LambdaASTSR()` with no args picks up 12-d RDKit 2D | ✓ |
| `LambdaASTSR(feature_names=legacy_f5_feature_columns())` still works | ✓ |
| `learned_shaping.py` / `hilbert_astsr_integration.py` import OK | ✓ |
| Wf-49 P1 extractor CSV header slice matches `RDKIT_2D_FEATURE_COLUMNS` | ✓ |
| ASTSR toy R² ≥ 0.5 on MolLogP-dominated target | ✓ |

**Wf-49 Phase 2 SHIP.**

Co-Authored-By: Claude Code <noreply@anthropic.com>