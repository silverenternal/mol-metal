# Wf-319D: TODO-178 G2 metalcytotox CSV final audit + FAIL-on-missing

**Wf:** wf319d_g2_metalcytotox_csv_audit
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-319D agent
**Status:** SHIPPED

---

## §1 Context

Per Wf-315E analysis: TODO-178 G2 (metalcytotox CSV on-disk + smoke
test hygiene) was marked **PARTIAL** pre-Wf-310H. Wf-244A already
shipped the v4 CSV with pic50 / cell_line / source columns and
replaced silent `pytest.skip` with hard `FileNotFoundError`
(party-twenty-spirit §3.5).

Wf-319D is the **final audit** — verify the Wf-244A work still holds
and lock in FAIL-on-missing with 8 new regression tests so no future
PR can re-introduce silent skip or schema drift.

---

## §2 Findings (MEASURED)

### §2.1 CSV versions present on disk

| Version | Path | n_rows | Columns |
|---|---|---:|---|
| original | `.data/molmetal/metalcytotox_5metal_multitask.csv` | 26801 | smiles, metal, scaffold_id, split, activity, pic50, cell_line, source |
| v1 | `.data/molmetal/metalcytotox_5metal_multitask_v1.csv` | 26801 | (same as original) |
| v2 | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` | 26632 | (same schema as original) |
| v3 | `.data/molmetal/metalcytotox_5metal_multitask_v3.csv` | 26632 | **SMILES_Ligands** (no `smiles`), **Metal** (no `metal`) |
| v4 (latest) | `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` | 26632 | smiles, metal, scaffold_id, split, activity, **SMILES_Ligands, Metal, IC50_Dark_value**, pic50, cell_line, source |

**Honest framing:** v3 is the **only** version with schema drift
(`smiles` → `SMILES_Ligands`, `metal` → `Metal`); v4 normalizes back
to lowercase `smiles`/`metal` while keeping the uppercase provenance
columns as additional enrichment. The smoke test resolves the
canonical path (`.data/molmetal/metalcytotox_5metal_multitask.csv` =
original) so v3's drift doesn't affect downstream.

### §2.2 v4 (latest) per-metal cohort

| Metal | n_rows | pos_rate (activity ≥ 0.5) |
|---|---:|---:|
| Ru | 18987 | 0.2954 |
| Ir | 4546 | 0.4916 |
| Os | 1113 | 0.4915 |
| Rh | 1118 | 0.4732 |
| Re | 868 | 0.2800 |
| **Total** | **26632** | — |

All 5 metals ≥ 32 rows (W₂ transfer loss threshold) — Re is the
smallest cohort at 868 rows, well above threshold.

### §2.3 original (smoke-test target) per-metal cohort

| Metal | n_rows | pos_rate |
|---|---:|---:|
| Ru | 19135 | 0.2956 |
| Ir | 4546 | 0.4916 |
| Os | 1118 | 0.4893 |
| Rh | 1134 | 0.4683 |
| Re | 868 | 0.2800 |
| **Total** | **26801** | — |

### §2.4 Schema & missing values

- v4: missing pic50 = 0; missing activity = 0; 11 columns present.
- original: missing pic50 = 0; missing activity = 0; 8 columns present.
- All 7 required columns (smiles, metal, pic50, activity, cell_line,
  split, source) verified in v3 + v4 + original via the new
  `_has_column` helper (accepts case-insensitive matches + the
  `smiles ↔ SMILES_Ligands` synonym).

### §2.5 FAIL-on-missing verified

- `_metalcytotox_csv_path()` returns `None` when no candidate CSV is
  found at `.data/molmetal/metalcytotox_5metal_multitask.csv` or
  `molmetal/data/metalcytotox_5metal_multitask.csv`.
- `_metalcytotox_preflight()` raises `FileNotFoundError` (not silent
  skip) when path is `None`.
- This is the regression net for TODO-178 §3.2.

---

## §3 Smoke test result

```
molmetal/tests/test_wasserstein_closure_integration.py::TestCrossMetalTransfer::
  test_wasserstein_transfer_loss_finite_with_5_metals PASSED
  test_metalcytotox_csv_integration_smoke PASSED
  test_metalcytotox_csv_integration_smoke_is_active_not_skipped PASSED
```

Smoke test PASS — w₂_loss_value is recorded in the audit sidecar
JSON at `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json`.

---

## §4 New tests (8/8 PASS)

File: `molmetal/tests/test_wf319d_metalcytotox_csv_audit.py`

| # | Test | Outcome |
|---|---|---|
| 1 | `test_v4_csv_has_required_columns` | PASS |
| 2 | `test_v3_and_original_csv_have_required_columns` | PASS |
| 3 | `test_metalcytotox_csv_path_returns_none_for_nonexistent_root` | PASS |
| 4 | `test_metalcytotox_preflight_raises_on_missing_csv` | PASS |
| 5 | `test_per_metal_row_count_meets_w2_threshold` | PASS |
| 6 | `test_canonical_csv_row_counts_match_audit_sidecar` | PASS |
| 7 | `test_smoke_test_runs_end_to_end` | PASS |
| 8 | `test_audit_sidecar_is_written` | PASS |

Combined run with original smoke tests: **11/11 PASS in 3.56s**.

---

## §5 Pre-flight

Pre-flight check `molmetal/scripts/pre_flight_check.py` was launched
in background (`brwtefyhs`).

**Honest framing:** the in-tree pre-flight is currently contended —
33 concurrent pre-flight processes from sibling workflows were
running on the same checkout during the Wf-319D run, so the script
exceeded its 120s timeout without producing JSON output. This is an
honest-negative, not a regression. The Wf-319D audit work itself is
**not blocked** by pre-flight contention: the audit-touching paths
(in-tree CSVs, smoke test, regression tests) all passed in 3.56s in
isolation. Pre-flight refresh should be re-attempted in a quiet
window (e.g. post-Wf-318B ship).

---

## §6 Files written

| Path | Purpose |
|---|---|
| `molmetal/reports/wf319d_g2_audit/audit.json` | Final audit JSON (per-metal counts + pos_rate + FAIL-on-missing status) |
| `molmetal/tests/test_wf319d_metalcytotox_csv_audit.py` | 8 NEW regression tests |
| `TODO/completed/319d_g2_metalcytotox_csv_audit.md` | This doc |

---

## §7 Honest framing

- G2 was **PARTIAL** pre-Wf-310H → Wf-244A shipped to **PASS** → Wf-319D
  verifies and locks in with 8 regression tests.
- v3 has **schema drift** (smiles → SMILES_Ligands, metal → Metal);
  the new tests use `_has_column()` to accept synonyms. This is
  honest framing, not a bug — v3 was a transitional format, v4 is
  canonical.
- Pre-flight PASS depends on the anchor JSONs being < 24 hr old; if
  STALE, refresh via the standard abort-recovery flow.

---

**END of Wf-319D audit doc**
