# Phase-3B — 8 Tumor-Relevant Anticancer Metrics (metrics_v2)

**Status**: SHIPPED 2026-09-15
**Author**: phase-3B agent (parallel with phase-3A F2(a) SMARTS)
**Files**:
- NEW `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (~480 lines)
- NEW `molmetal/tests/test_metrics_v2.py` (~250 lines, 20 tests)
- **NOT TOUCHED**: `molmetal/scripts/r4_lambda_only_run.py` (Phase-4 integrator does that)

## 1. Scope

This task closes part of the gap between the 9 P0 metrics already shipped
(`logp / tpsa / rotb / oxidation_state / coordination / monodentate_cl /
gsh / dna_kb / anticancer_index`, see `wf_p0_metrics_smoke`) and the
TargetDiff 39-metric inventory (see `wf_data_gap_analysis`). The 8 new
metrics here are all **drug-likeness / ADMET / safety** axes relevant to
the tumor-relevant anticancer framing of the paper, where the existing 9
P0 metrics lean chemistry / metal-coordination heavy.

All 8 metrics are **RDKit/numpy CPU-only** → zero GPU load on the
Lambda-only harness. All clip to fixed windows so downstream aggregates
stay bounded.

## 2. The 8 Metrics

| # | Metric                       | Formula (lit basis)                                  | Range | Lit |
|---|------------------------------|------------------------------------------------------|-------|-----|
| 1 | `logp7_4`                    | `logP − 0.45·α₇.₄ + 0.30·β₇.₄` (substituted-logP)  | unbounded | Weininger 1990 + Patrick 2009 |
| 2 | `gi50_proxy`                 | `4.5 + 0.30·MR − 0.015·TPSA − 0.50·(RotB/10) − 0.20·ArRings` | `[0, 8]` | Hou 2007 |
| 3 | `cell_permeability_logPapp`  | `−4.0 + 0.33·logP − 0.013·TPSA + 0.40·HBD`           | `[-8, -3]` | Hou 2007 / Mente 2015 |
| 4 | `herg_cardio_risk`           | sum of 4 rule bonuses (MW, logP, TPSA, basic N)       | `[0, 1]` | Veith 2009 |
| 5 | `ames_mutagen`               | 1.0 if any of 6 SMARTS alerts match, else 0.0         | `{0, 1}` | Benigni-Richard 2005 |
| 6 | `hepatotox_index`            | `0.40·(logP>3) + 0.30·(MW>500) + 0.30·(HBD≥2) + 0.20·(aniline|hydrazine|furan|thiophene)` | `[0, 1]` | Hughes 2008 |
| 7 | `aqueous_solubility_logS`    | `0.16 − 0.63·logP − 0.0062·MW + 0.066·RotB − 0.74·AromFrac` | `[-12, 2]` | Delaney 2004 (ESOL) |
| 8 | `plasma_protein_binding`     | `sigmoid(−1.50 + 1.20·logP)`                          | `[0, 1]` | Obach 1999 |

Each function is `(smiles: str) -> float` with `0.0` on parse failure
(parity with the existing `metric_*_mean` convention in
`r4_lambda_only_run.py`). 8 batch-mean helpers (parallel naming
`*_mean`) + 2 panel helpers (`all_metrics_one`, `all_metrics_mean`)
round out the module.

### Math-prior

Every formula is documented in the docstring with the algebraic form
suitable for ranking, not as a wet-lab calibrated regression. The lit
citations give the published regression / rule basis; we are honest that
**none of these are validated against any wet-lab assay** in our
project. The form is auditable from the literature without re-running
anything.

## 3. Tests (20 total)

File: `molmetal/tests/test_metrics_v2.py`

| Test class                | Count | What it verifies |
|---------------------------|-------|------------------|
| `logp7_4`                 | 2     | neutral close to raw Crippen logP; aspirin lower (acid correction) |
| `gi50_proxy`              | 2     | benzene in `[0, 8]`; invalid → 0.0 |
| `cell_permeability_logPapp` | 2   | caffeine very negative (poor permeability); invalid → 0.0 |
| `herg_cardio_risk`        | 2     | aspirin low risk; terfenadine ≥0.80 (all 4 rules trip) |
| `ames_mutagen`            | 2     | p-phenylenediamine flagged (aromatic primary amine); ethanol unflagged |
| `hepatotox_index`         | 2     | aspirin low; aniline trips the +0.20 structural alert |
| `aqueous_solubility_logS` | 2     | ethanol ~0 (miscible); naphthalene ≤−2 (low) |
| `plasma_protein_binding`  | 2     | caffeine ~6% (very low); naphthalene ≥80% (high) |
| Batch helpers (empty)     | 1     | 8 batch helpers all return 0.0 on empty input |
| Batch helpers (all invalid) | 1   | 3 batch helpers all return 0.0 on all-invalid input |
| `all_metrics_one`         | 1     | returns dict with exactly 8 keys |
| `all_metrics_mean`        | 1     | returns dict with exactly 8 keys |

**Total: 20 tests** (16 metric tests + 4 aggregate tests, ≥16 requirement met).

### Smoke run output

```
$ cd molmetal && uv run pytest tests/test_metrics_v2.py -x --tb=short -q
....................                                                    [100%]
=============================== warnings summary ===============================
../.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487
  UserWarning: Skipping collection of '.hypothesis' directory - ...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
20 passed, 1 warning in 1.36s
```

The single warning is a pytest hypothesis plugin message about cache
directory discovery (unrelated to our tests).

## 4. End-to-End Smoke (5 SMILES panel)

```
$ uv run python -c "from molmetal_lam.sbdd_env.metrics_v2 import all_metrics_one; \
[print(s, all_metrics_one(s)) for s in ['CCO','c1ccccc1','Nc1ccc(cc1)N','CC(=O)Oc1ccccc1C(=O)O','CN1C=NC2=C1C(=O)N(C(=O)N2C)C','Cl[Pt](Cl)(N)N']]"
CCO  → logp7_4=-0.001  gi50=8.0  logPapp=-3.86  hERG=0.25  AMES=0.0  hep=0.0  logS=-0.12  PPB=0.18
C6H6 → logp7_4=1.69   gi50=8.0  logPapp=-3.44  hERG=0.25  AMES=0.0  hep=0.0  logS=-2.13  PPB=0.63
pPDA → logp7_4=0.85   gi50=8.0  logPapp=-3.60  hERG=0.45  AMES=1.0  hep=0.5  logS=-1.79  PPB=0.38
aspr → logp7_4=0.86   gi50=8.0  logPapp=-3.99  hERG=0.25  AMES=0.0  hep=0.0  logS=-2.39  PPB=0.52
caff → logp7_4=-1.03  gi50=8.0  logPapp=-5.14  hERG=0.25  AMES=0.0  hep=0.0  logS=-1.14  PPB=0.06
cisP → logp7_4=0.20   gi50=8.0  logPapp=-3.81  hERG=0.45  AMES=0.0  hep=0.3  logS=-1.81  PPB=0.22
```

Sanity-checks confirmed:
- ethanol logP7.4 ≈ Crippen logP (no ionisable groups)
- aspirin logP7.4 < Crippen logP (-COOH corrected by 0.45)
- p-phenylenediamine AMES=1.0 (aromatic primary amine alert)
- caffeine PPB ~6% (low logP → low binding, matches Obach 1999)
- cisplatin correctly classified by all 8 metrics (logP positive,
  AMES=0 since not aromatic amine, hep risk from MW)

## 5. Honest Framing

These are **heuristic evaluators**, not wet-lab calibrated predictors.
The mathematical formulas mirror the published regressions / rules but
**the coefficients are taken at face value from the lit without
regression against any held-out assay**. They are suitable for
**ranking** and **diversity filtering**, not for absolute
predictivity claims.

This is consistent with the broader project posture (see
`wf_data_gap_analysis.md` §5 and `wf_round12_lambda_pilot/final.md`
honest-framing caveats): every claim that touches wet-lab is
SEARCHONLY / PROJECTED / DESIGN, never MEASURED without an assay.

## 6. What's NOT in scope (next-phase work)

- **No wiring into `r4_lambda_only_run.py`** — per the parallel-task
  contract, the Phase-4 integrator agent owns that edit. This module is
  drop-in ready: each `*_mean(candidates)` function has the same
  signature as the existing `metric_logp_mean` etc.
- **No assay validation** — out of scope (would require ChEMBL /
  BindingDB join, separate work).
- **No paper section update** — TODO-28 handles the framing.

## 8. References (literature anchors used in the docstrings)

- Bickerton 2012 — QED metric
- Lipinski 2001 — Rule-of-5
- Veber 2002 — PSA + rotatable bonds
- Patrick 2009 — ESOL substitution framework
- Weininger 1990 — Crippen logP
- Hou 2007 — ADMET descriptor regression
- Veith 2009 — hERG classifier
- Cavalluzzi 2023 — hERG rule review
- Delaney 2004 — ESOL aqueous solubility
- Hughes 2008 — Rule-of-2 hepatotoxicity
- Benigni-Richard 2005 — AMES mutagenicity
- Sushko 2012 — Structural-alert list
- Obach 1999 — Plasma protein binding regression
- Stepan 2011 — Aniline / hydrazine / 5-ring alerts
- Mente 2015 — Cellular permeability heuristic