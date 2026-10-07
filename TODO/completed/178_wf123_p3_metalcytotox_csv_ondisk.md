# TODO 178 — Wf-123 P3 Gap: metalcytotox 5-metal CSV on disk (integration smoke hard dependency)

**Date**: 2026-09-27 (Wf-123 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 P1 — IN-FLIGHT (Wf-123 P3 §7 explicitly flags this as conditional skip)
**Parent**: `TODO/completed/123_master_plan_wasserstein_closure_closeout.md` + `molmetal/reports/wf123_p3_integration/README.md` §7
**Sibling**: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1) + `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (G3)

---

## §1 Gap description

**Wf-123 P3 §7 honest framing** explicitly states:

> "The metalcytotox 5-metal CSV integration smoke test (`TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke`) is **conditional on the CSV existing** in the project tree (`.data/molmetal/metalcytotox_5metal_multitask_v4.csv`). It skips gracefully if the CSV is absent — no hard dependency on the cross-metal test data."

The CSV is referenced but not guaranteed on disk. The integration test currently **silently skips** rather than fail, which is a hygiene problem (per `party-twenty-spirit` §3.5 self-revolution — silent skip ≠ measured). The CSV is the cross-metal transfer cohort needed for the M1 `wasserstein_transfer_loss` smoke test on real cross-metal data (not synthetic).

**Root cause**:
- Wf-120 P2 (TODO-120, m1 Ru strengthen) built the 5-metal joint cohort, but the on-disk CSV is keyed on `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` and depends on the CytotoxFilter pipeline being run end-to-end
- Wf-124 P3 (5-metal multi-task D-MPNN) uses the same cohort for training but the on-disk CSV path is **not** verified by `verify_shared_modules.py`
- The integration smoke test `test_metalcytotox_csv_integration_smoke` is the only verification

---

## §2 Current MEASURED state

| Item | Status | Source |
|---|---|---|
| metalcytotox 5-metal CSV on disk | UNKNOWN (not verified at Wf-123 P3 ship time) | `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` |
| `TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke` | SKIP (graceful) | `molmetal/tests/test_wasserstein_closure_integration.py` |
| CytotoxFilter pipeline execution | PROBABLY executed (Wf-120 P2 ran end-to-end on RTX 5090) | `molmetal/reports/wf120_p2_5seed_dmpnn/` |
| n_metal rows | 5 metals × ~1.2-2.5k each = ~5-8k total PROJECTED | TODO-120 P1.2 |

**MEASURED gaps**:
- The CSV's actual presence + schema on disk is not audited at Wf-123 P3 close-out
- If CSV is missing or schema is stale, `test_metalcytotox_csv_integration_smoke` silently skips → integration layer's cross-metal transfer guarantee is **unverified**
- Wf-124 P3 multi-task training used the same data path but in-memory (not via the on-disk CSV)

---

## §3 What's needed

### §3.1 CSV on-disk verification

1. **Check existence**: `ls -la .data/molmetal/metalcytotox_5metal_multitask_v4.csv` — if absent, run the CytotoxFilter pipeline to materialize it
2. **Schema validation**: at minimum the columns are `smiles`, `metal`, `pic50`, `active`, `cell_line`, `split` (train/val/test), `source` (cytotox/platinai/etc.)
3. **Row-count audit**: 5 metals × ≥ 1.0k rows each (≥ 5k total); if any metal < 1k, the per-metal Wasserstein transfer loss is degenerate (per Wf-124 P3 §5 honest caveat about smaller metals)

### §3.2 Integration smoke test fix

Update `molmetal/tests/test_wasserstein_closure_integration.py::TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke` to:

- **FAIL** (not SKIP) if CSV is missing (per `party-twenty-spirit` §3.5 self-revolution — silent skip is not acceptable for ship-cohort tests)
- **Validate schema**: column presence + dtype check
- **Row-count gate**: at least 1 metal ≥ 1.5k rows AND total ≥ 5k
- **W₂ transfer smoke**: compute `wasserstein_transfer_loss` on a stratified sample (50 rows per metal) and assert finite + non-zero gradient

### §3.3 Audit report

Write a JSON report at `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json` with:

```json
{
  "csv_path": ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
  "exists": true|false,
  "schema_valid": true|false,
  "row_counts_per_metal": {"Ru": ..., "Ir": ..., "Os": ..., "Rh": ..., "Re": ...},
  "total_rows": ...,
  "smoke_test_outcome": "PASS|FAIL",
  "smoke_w2_loss_value": ...,
  "audit_date": "2026-09-27",
  "owner": "Hugo Lee + ultracode agent"
}
```

---

## §4 Acceptance gate

- [G2.1] CSV exists on disk at `.data/molmetal/metalcytotox_5metal_multitask_v4.csv`
- [G2.2] Schema is valid (all required columns present, dtypes correct)
- [G2.3] At least 1 metal ≥ 1.5k rows AND total ≥ 5k rows
- [G2.4] `TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke` updated to FAIL (not SKIP) when CSV is missing or schema invalid
- [G2.5] `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json` written with full audit info
- [G2.6] Integration smoke test PASSES on real cross-metal data (not synthetic)

---

## §5 ETA + dependencies

**ETA**: 0.5 - 1 day (CSV materialization is fast; CytotoxFilter is the Wf-120 P2 pipeline)

**Dependencies**:
- Wf-120 P2 CytotoxFilter pipeline (already executed; outputs at `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` if materialization completed)
- `molmetal/scripts/cytotox_filter.py` (CytotoxFilter — Wf-120 P2 ship)
- `pytest` for integration smoke test
- `molmetal/reports/wf123_p3_integration/` directory

**No GPU required** — pure CPU smoke + CSV materialization.

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Wf-123 P3 SHIP with `test_metalcytotox_csv_integration_smoke` SKIP (graceful) — UNVERIFIED for real cross-metal data
- **PROJECTED**: CSV materialization will succeed (Wf-120 P2 ran end-to-end) and schema will be valid (standard CytotoxFilter output)
- **Silent skip hygiene problem**: the current behavior is a `party-twenty-spirit` §3.5 violation — silent skip is not acceptable for ship-cohort tests; must FAIL or have a `--strict` flag
- **No fabrication**: do NOT claim M1+M2 cross-metal transfer works on real data until the smoke test actually runs and PASSES

---

## §7 Cross-references

- `molmetal/tests/test_wasserstein_closure_integration.py::TestCrossMetalTransfer::test_metalcytotox_csv_integration_smoke`
- `molmetal/reports/wf123_p3_integration/README.md` §7 (honest framing cite)
- `molmetal/scripts/cytotox_filter.py` (CytotoxFilter — Wf-120 P2 ship)
- `molmetal/reports/wf120_p2_5seed_dmpnn/` (CytotoxFilter outputs)
- `TODO/completed/120_dmpnn_seed_integrity.md` (Wf-120 P2 close-out)
- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (parent strengthen TODO)
- `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1 sibling — depends on this gap closed first)

---

## §8 Follow-ups

- Once G2 closed: G1 (TODO-177) consumer metric lifts can proceed with full cross-metal transfer guarantee
- If CSV materialization fails (CytotoxFilter pipeline broken): escalate to Wf-120 P3 fix
- If schema is invalid: update CytotoxFilter to emit required columns, re-run end-to-end

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: silent skip ≠ measured. Until `test_metalcytotox_csv_integration_smoke` actually PASSES on real cross-metal data, M1+M2 cross-metal transfer is UNVERIFIED — this gap MUST be closed before paper §4 can cite cross-metal Wasserstein transfer.
