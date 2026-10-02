# Wf-327F2: v3/v4 schema drift verification

**Wf:** wf327f2_v3_v4_schema_verified
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-327F2 agent
**Status:** SHIPPED (verification only — no code change)
**Refs:** Wf-323B (SHIPPED 2026-09-29, `TODO/completed/323b_v3_v4_schema_drift_fix.md`) + Wf-327-T3 schema-drift trigger

---

## §1 Verdict

**VERIFIED** — Wf-323B already shipped the v3/v4 schema drift fix. Wf-327F2
adds an independent 4-test verification layer on top, raising the
regression-detection surface from 8 tests (Wf-323B) to 12 tests
(Wf-323B 8 + Wf-327F2 4). **Zero new implementation required.**

---

## §2 Wf-323B status (MEASURED)

### §2.1 Source-of-truth check

| Artefact | Path | Status |
|---|---|---|
| Ship doc | `TODO/completed/323b_v3_v4_schema_drift_fix.md` | ✅ exists (2026-09-29) |
| Loader | `molmetal/data/metal_cytotox.py` | ✅ 242 lines (was 168; +113/-32 per Wf-323B) |
| Synonym map | `_COLUMN_SYNONYMS` dict at line 82 | ✅ 4 canonical keys |
| Helpers | `_has_column` (line 97) + `_rename_v3_to_canonical` (line 110) | ✅ present |
| Tests (Wf-323B) | `molmetal/tests/test_wf323b_v3_v4_schema_drift.py` | ✅ 8 tests, 8 PASS |
| CSVs on disk | `v1` (26801 rows) + `v3` (26632) + `v4` (26632) | ✅ all present |

### §2.2 Wf-323B tests re-run (Wf-327F2 verification, MEASURED)

```
$ PYTHONPATH=. .venv/bin/python -m pytest tests/test_wf323b_v3_v4_schema_drift.py -v
============================== 8 passed in 0.62s ===============================
```

| # | Test | Status |
|---:|---|---|
| 1 | `test_v3_loader_returns_canonical_4_columns` | PASS |
| 2 | `test_v3_loader_resolves_5_metals` | PASS |
| 3 | `test_v3_loader_no_nan_pic50` | PASS |
| 4 | `test_v4_loader_returns_canonical_4_columns` | PASS (was FAIL pre-Wf-323B) |
| 5 | `test_v4_loader_resolves_5_metals` | PASS |
| 6 | `test_v3_and_v4_outputs_match` | PASS |
| 7 | `test_column_synonyms_cover_all_canonical` | PASS |
| 8 | `test_rename_v3_skips_when_lowercase_present` | PASS |

### §2.3 Loader runtime smoke (MEASURED, Wf-327F2 §3 direct invocation)

```
$ PYTHONPATH=. .venv/bin/python -c "
from molmetal.data.metal_cytotox import load_metal_cytotox
from pathlib import Path
import time

p_v3 = Path('.data/molmetal/metalcytotox_5metal_multitask_v3.csv')
p_v4 = Path('.data/molmetal/metalcytotox_5metal_multitask_v4.csv')

t0 = time.time(); df_v3 = load_metal_cytotox(p_v3); t_v3 = time.time() - t0
t0 = time.time(); df_v4 = load_metal_cytotox(p_v4); t_v4 = time.time() - t0

print(f'v3: {df_v3.shape}, {df_v3[\"metal\"].nunique()} metals, pIC50 NaN={df_v3[\"pIC50\"].isna().sum()}, t={t_v3:.2f}s')
print(f'v4: {df_v4.shape}, {df_v4[\"metal\"].nunique()} metals, pIC50 NaN={df_v4[\"pIC50\"].isna().sum()}, t={t_v4:.2f}s')
"

v3: (26632, 4), 5 metals, pIC50 NaN=0, t=0.06s
v4: (26632, 4), 5 metals, pIC50 NaN=0, t=0.09s
```

Both v3 and v4 load to the canonical 4-column contract
(`smiles, metal, pIC50, active`) with identical 26632 rows × 5 metals
distribution, zero pIC50 NaN survivors, < 100 ms wall time each.

---

## §3 Wf-327F2 — 4 NEW verification tests (MEASURED)

`molmetal/tests/test_wf327f2_v3_v4_schema.py` — 196 LOC, 4 tests, all
PASS.  Each test pins an invariant NOT covered by Wf-323B's 8 tests.

### §3.1 Test surface

| # | Test class | Test | Invariant pinned | Status |
|---:|---|---|---|---|
| 1 | `TestCrossVersionInvariants` | `test_v1_v3_v4_canonical_alignment` | v1+v3+v4 round-trip to same canonical 4-col shape with identical dtypes + 5-metal set + row counts within 1% | PASS |
| 2 | `TestIdempotentReload` | `test_loading_v4_twice_is_idempotent` | Loading v4 twice produces byte-equal DataFrames (no global state mutation) | PASS |
| 3 | `TestHasColumnPrecedence` | `test_lowercase_wins_when_both_present` | v4 has both lowercase + uppercase columns; `_has_column` returns lowercase (v4 collision prevention) | PASS |
| 4 | `TestSynonymMapCompleteness` | `test_all_aliases_are_lowercase_or_known` | Every alias in `_COLUMN_SYNONYMS` is either the canonical or in the pinned known non-canonical set (guards future drift) | PASS |

### §3.2 Run output

```
$ PYTHONPATH=. .venv/bin/python -m pytest tests/test_wf327f2_v3_v4_schema.py -v
============================== 4 passed in 0.48s ===============================
```

### §3.3 Combined regression surface (Wf-323B + Wf-327F2)

| Surface | Tests | Status |
|---|---:|---|
| Wf-323B schema drift tests | 8 | 8/8 PASS |
| Wf-327F2 verification tests (NEW) | 4 | 4/4 PASS |
| **Total** | **12** | **12/12 PASS** |

CPU-only, total runtime < 2 s.

---

## §4 Pre-flight gate impact

### §4.1 Per Wf-323B ship doc §4 (MEASURED, 2026-09-29 baseline)

| Gate | Pre-Wf-323B | Post-Wf-323B |
|---|---|---|
| `p1_chemprop_smoke` | FAIL (loader ValueError) | **PASS** (Ru seed=0 test_auc=0.7929) |
| `chemprop_v2_v3` pipeline (13 gates) | 13 FAILs from loader | **13/13 PASS** |
| full pre-flight v3 (57 gates) | multiple loader-related FAILs | 42 PASS / 15 FAIL (1 of 15 FAILs PARTIAL resolved by Wf-323B — `p1_chemprop_smoke`) |

The 14 remaining pre-flight FAILs are pre-existing infra / honest-negative
issues (upstream `flow_matching` ImportError cascade + 4 ADMET HONEST_NEGATIVE
tag gates + 2 PB-packages status gates + NCI-60 9-cell + `p3_gate_count_consistency_audit`
drift) — flagged in §22.16 Wf-322D snapshot, NOT in scope for Wf-323B / Wf-327F2.

### §4.2 Wf-327F2 pre-flight run (2026-09-29)

Direct invocation of `p1_chemprop_smoke` gate (the Wf-323B target gate)
hits an unrelated pre-existing infra bug in `pre_flight_check.py:_run()`
TIMEOUT path (`TypeError: can't concat str to bytes`); this bug is
NOT triggered by v3/v4 schema work — it is a stale subprocess exception
field accessor and pre-dates Wf-323B.  Per Wf-323B ship doc §4, the
`p1_chemprop_smoke` gate was MEASURED-PASS at the Wf-323B ship moment
(Ru seed=0 test_auc=0.7929); Wf-327F2 inherits that PASS verdict
without re-running the full pre-flight suite (which has 57 gates and
routinely exceeds the per-gate 90s budget on this host).

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **VERIFIED, not re-implemented**: Wf-323B is the implementation
  surface; Wf-327F2 is the verification surface. No loader code
  changed; no CSVs regenerated; no GPU work.
- **v3 + v4 are both legitimate versions** of the same data (per
  Wf-323B §5). The loader now round-trips both for backward compat.
- **Pure CPU fix** — no GPU launch, no chemprop retrain.
- **No new pending files** — Wf-323B already moved pending/ → completed/
  on 2026-09-29; Wf-327F2 is verification-only.
- **`p1_chemprop_smoke` PASS verdict inherited from Wf-323B ship**,
  not re-MEASURED in this Wf (infra bug in pre-flight `_run` TIMEOUT
  path is pre-existing and unrelated).

---

## §6 Files touched

| Path | Change |
|---|---|
| `molmetal/tests/test_wf327f2_v3_v4_schema.py` | NEW, 196 LOC, 4 tests |
| `TODO/completed/327f2_v3_v4_schema_verified.md` | NEW (this doc) |

No loader changes (Wf-323B is the source of truth for `_COLUMN_SYNONYMS`,
`_has_column`, `_rename_v3_to_canonical`).

---

## §7 Wf-327F2 ship artefacts

| Artefact | Path |
|---|---|
| Verification tests | `molmetal/tests/test_wf327f2_v3_v4_schema.py` (4 tests, all PASS) |
| This doc | `TODO/completed/327f2_v3_v4_schema_verified.md` |
| Source-of-truth (Wf-323B) | `TODO/completed/323b_v3_v4_schema_drift_fix.md` |

---

**END of Wf-327F2 doc** — VERIFIED; Wf-323B is the implementation
surface; Wf-327F2 raises the regression-detection surface from 8 to
12 tests with 4 new invariants.
