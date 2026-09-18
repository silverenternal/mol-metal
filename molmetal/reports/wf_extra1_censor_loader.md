# WF-Extra-1 — Censor-aware pIC50 training data loader

**Date:** 2026-09-14
**Workflow:** WF-Extra-1 (TODO-18: neural pIC50 predictor retrain, Attentive D-MPNN)
**Scope:** Sub-step A — build the dataset class that hands censored rows to the
training loop as **bounds**, not exact regression labels.
**Status:** SHIPPED (code + tests green). Retraining script is the next sub-step.

---

## 1. Background (MEASURED)

The legacy first-row-per-SMILES targets collapsed because they (a) silently
re-coded censored `IC50 > X uM` rows as if they were exact measurements at `X`
and (b) ignored counterion / oxidation / assay context. The audit script
`molmetal/scripts/audit_pic50_assay_data.py` already documents this and
exports a conditioned cohort CSV; the D-MPNN retrain (sub-step B, next)
needs a Dataset that preserves the censor flag for the loss function.

## 2. Column contract (verbatim from `audit_pic50_assay_data.py`)

| Logical field | Source column | Type | Notes |
|---|---|---|---|
| SMILES | `SMILES_Ligands` | str | canonicalised via RDKit |
| Raw IC50 cell | `IC50_Dark(M*10^-6)` | str | contains `<`, `>`, `≤`, `≥` for censored rows |
| Numeric IC50 | `IC50_Dark_value` | float (uM) | already-parsed numeric |
| Formulation meta | `Metal`, `Cell_line`, `Time(h)`, `DOI`, `Counterion`, `Oxidation_state`, `Charge_complex` | mixed | preserved verbatim |

The censoring rule is "raw cell contains any of `< > ≤ ≥`" (matches the audit
script's `dark_censored` derivation on line 18). The bound is extracted by the
first float in the raw cell — `>10` → `10.0`, `<0.1` → `0.1`, `≥5.0` → `5.0`.

## 3. Spec — `molmetal/molmetal_lam/training/pic50_censored_loader.py`

```text
class CensoredPIC50Dataset(torch.utils.data.Dataset):
    __init__(frame, *, smiles_col, raw_col, value_col, formulation_meta_cols, drop_unparseable)
    __getitem__(idx) -> (smiles: str, target: float, is_censored: bool, bound_value: float)

    # Convenience / diagnostics
    from_csv(path, *, filters=None)
    get_row(idx)            -> CensoredPIC50Row (incl. formulation_id + meta)
    formulation_meta(idx)   -> dict
    formulation_ids()       -> list[str]
    to_frame()              -> pd.DataFrame (audit/debug)
    n_censored_total        -> int  (rows surviving construction that carry the flag)
```

Guarantees:

1. **Censored rows are never silently re-coded.** For `is_censored=True`,
   `bound_value` is the parsed bound (e.g. `10.0` for `>10 uM`); the `target`
   field is set to the same value for tensor-shape convenience, but a downstream
   right-censored margin loss must branch on `is_censored` and use `bound_value`
   as the constraint, not `target`.
2. **Invalid SMILES are dropped with a `warnings.warn(...)` call**, not raised
   — so a single bad ligand cannot kill the sweep.
3. **Formulation meta is preserved verbatim** (column name → value), plus a
   stable `formulation_id` string `Metal|Cell_line|Time(h)|Oxidation|Charge|Counterion|smiles`
   for traceability into the conditioned-baseline protocol.

## 4. Tests — `molmetal/molmetal_lam/tests/test_pic50_censored_loader.py`

All 6 tests pass under `uv run pytest -q molmetal/molmetal_lam/tests/test_pic50_censored_loader.py --tb=short`:

```text
......                                                                   [100%]
6 passed, 6 warnings in 1.34s
```

| Test | Verifies |
|---|---|
| `test_censored_loader_returns_bounds` | `IC50 > 10 uM` row returns `(10.0, True, 10.0)` |
| `test_loader_skips_unparseable_smiles` | Invalid SMILES triggers a `UserWarning` and is dropped (not raised) |
| `test_loader_preserves_formulation_meta` | All 7 meta columns round-trip; `formulation_id` carries Metal + Cell_line |
| `test_exact_row_is_not_censored` | Uncensored rows have `is_censored=False` and `bound == target` |
| `test_left_censored_row_keeps_bound` | `IC50 < 0.1 uM` is preserved as `bound=0.1, is_censored=True` |
| `test_n_censored_total_excludes_dropped` | The diagnostic counter excludes rows that were dropped during construction |

The 6 warnings are the expected `dropping unparseable SMILES: 'this-is-not-a-smiles@@'`
emitted by the loader for the deliberately-invalid row in the fixture.

## 5. Metrics

| Metric | Value |
|---|---:|
| `n_tests` | **6** |
| `n_passed` | **6** |
| `n_failed` | 0 |
| `n_censored_rows_total` (fixture) | **2** (right-bound `>10`, left-bound `<0.1`) |
| Censored rows dropped by loader (fixture) | 0 |
| Unparseable SMILES dropped (fixture) | 1 (`this-is-not-a-smiles@@`) |

These are **fixture numbers**, not the HeLa48h cohort — running the loader on
the audit export will produce the real `n_censored_rows_total` (sub-step B).
The historical audit reported `ru_numeric_but_censored_rows = 3864 / 19135`,
i.e. ~20 % of Ru rows carry a censor symbol; expect the HeLa48h cohort
(702 formulations) to land in the low hundreds of censored rows.

## 6. Limitations / honest framing

- **MEASURED:** 6/6 unit tests pass under `uv run pytest`. Column contract
  derived by reading `audit_pic50_assay_data.py` lines 17–19 verbatim.
- **PROJECTED (next sub-step):** D-MPNN retrain script
  `molmetal/scripts/retrain_dmpnn_pic50_hela48.py` will consume this dataset
  with a right-censored margin loss; Pearson r on test target is **0.5–0.7**,
  with the 0.407 historical number as the floor to beat. Until sub-step B
  ships, no retrained weights exist — the historical
  `dmpnn_attn_ru_pic50.pt` checkpoint is **frozen and untouched**.
- This loader only touches the D-MPNN code path. Lambda search chemistry,
  CFM geometric fix, and round-trip / metal-seed / BNF fixes are out of scope
  and untouched.

## 7. Files added

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/training/__init__.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/training/pic50_censored_loader.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pic50_censored_loader.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_censor_loader.md` (this file)
