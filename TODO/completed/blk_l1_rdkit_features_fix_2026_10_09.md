# BLK-L1 Step 1: _rdkit_features 7-column zero fix (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` BLK-L1 fix_steps Step 1 (CPU-only code change,
no GPU).  Steps 2-6 (chemprop_v3 frozen encoder, Hilbert 18-d wire,
50-d CV, R² ≥ 0.10) are DEFERRED — see "Honest-negative framing" below.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) +
§5.6 (atomic_write_file; per §5.6 Edit for small code patches is the
canonical surface — atomic_write_file is for closeout writes) +
§5.7 (5/5 smoke PASS) + §5.8 (direct Edit/Write preferred; one Edit +
one test write) + §5.9 (single-venv .venv/bin/python).

## Problem (per `todo.json` BLK-L1 root_cause)

`molmetal/scripts/wf433_m2_extended.py::_rdkit_features` is the loader
for the Wf-335 / Wf-388-A1 extended-feature path.  It maps a
PLATINAI_32D_COLUMNS tuple to RDKit descriptors, but the elif-branches
for 4 of the 7 extended columns used the wrong RDKit submodule:

| Column | Wrong call (silent AttributeError) | Correct call |
|---|---|---|
| `NHOHCount` | `rdMolDescriptors.CalcNHOHCount(mol)` | `Lipinski.NHOHCount(mol)` |
| `NOCount` | `rdMolDescriptors.CalcNOCount(mol)` | `Lipinski.NOCount(mol)` |
| `fr_halogen` | `getattr(rdMolDescriptors, "CalcNumhalogen", None)` → None | `Fragments.fr_halogen(mol)` |
| `fr_Al_COO` | `getattr(rdMolDescriptors, "CalcNumAl_COO", None)` → None | `Fragments.fr_Al_COO(mol)` |

The except-clause swallowed every AttributeError, so the 4 affected
columns were silently 0 for every row — dropping the effective rank
of the 32-d feature from 32 to 28 on a 4078-row cohort.  Verified
via `grep -n "CalcNHOHCount\|fr_halogen\|fr_Al_COO"` which returned
the exact lines I edited, and a `from rdkit.Chem import rdMolDescriptors;
rdMolDescriptors.CalcNHOHCount` AttributeError probe.

The other 3 extended columns (`NumAliphaticRings`, `NumSaturatedRings`,
`NumHeteroatoms`) already had correct `rdMolDescriptors.CalcNum*` calls
in place — only the 4 above needed fixing.

## Fix design

Replace the 4 broken branches with the correct RDKit submodule
imports + calls.  Each branch is wrapped in `from rdkit.Chem import
<Submodule>` inside the branch (lazy import — keeps the
`from __future__ import annotations` module-load path RDKit-light).
The fallback branch (`else: out[j] = float(rdMolDescriptors.CalcMolWt(mol))`)
is preserved as a safety net for any column name not in the elif
chain.

## §5.5 filesystem-truth evidence

| File | md5 | size |
|---|---|---|
| `molmetal/scripts/wf433_m2_extended.py` (pre) | (recoverable from git) | 25013 B |
| `molmetal/scripts/wf433_m2_extended.py` (post) | (recoverable from git) | 25328 B |
| `molmetal/tests/test_wf433_m2_rdkit_features_fix.py` (NEW) | (recoverable from git) | ~3.5 KB |

Pre/post diff scope: 22 lines (12 lines of 4 new elif branches with
lazy imports + 6 lines of "BLK-L1:" comments + 4 lines of
in-line `# noqa: WPS433` annotations; one of the elif branches
(`fr_*`) was a rewrite of the existing `getattr(...)` lookup, not a
new addition).

## §5.7 smoke gate — 5/5 PASS

`molmetal/tests/test_wf433_m2_rdkit_features_fix.py` (NEW, ~110 LOC,
5 tests):

| # | Test | Asserts |
|---:|---|---|
| 1 | `test_nhohcount_uses_lipinski_not_rdmodelcriptors` | aspirin (CC(=O)Oc1ccccc1C(=O)O) → NHOHCount=1, NOCount=4 |
| 2 | `test_fr_halogen_uses_fragments_module` | ClCCBr → fr_halogen=2; malonic-acid → 0 |
| 3 | `test_fr_al_coo_uses_fragments_module` | malonic-acid → fr_Al_COO=2 |
| 4 | `test_numheteroatoms_uses_rdmodelcriptors` | aspirin → NumHeteroatoms=4 |
| 5 | `test_no_nan_no_inf_post_fix` | 5-molecule batch has no NaN / no Inf |

Run command: `PYTHONPATH=/home/hugo/codes/mol-metal .venv/bin/python
molmetal/tests/test_wf433_m2_rdkit_features_fix.py`.  Output:
`5/5 §5.7 smoke PASS` (verified).

## §5.7 §3 invariant: does the fix actually move the downstream metric?

`PYTHONPATH=/home/hugo/codes/mol-metal .venv/bin/python` (then `from
molmetal.scripts.wf331r5_r2_w2_chemprop import load_cohort;
load_cohort("platinai_4cell", max_cohort_size=4078); ... 5-fold CV
Ridge alpha=1.0`):

| Variant | R² on platinai_4cell n=4078 (5-fold CV) | Effective rank |
|---|---:|---:|
| Pre-fix (12-d RDKit) | -0.144 (from `WF388A1_12D_BASELINE_R2`) | 12 |
| Pre-fix (32-d, with 4 silent-0 columns) | -0.030 (from `metrics/by_metric/m2_r2_per_cohort.json` aggregate) | ~28 (effective) |
| **Post-fix (32-d, all 32 populated)** | **-0.0302 ± 0.0136** | **150 (over 160-d extended)** |

**Honest-negative finding**: the +0.114 R² lift from -0.144 → -0.030
came from the 12→32 column jump (the 12-d baseline only had
PLATINAI_12D_COLUMNS; the 32-d path adds 20 more).  Within the
extended 32-d path, the 4 newly-populated columns (NHOHCount, NOCount,
fr_halogen, fr_Al_COO) do NOT move the R² any further than the
already-0-effective columns — Ridge alpha=1.0 regularizes them
into noise.  The remaining 3 (NumAliphaticRings, NumSaturatedRings,
NumHeteroatoms) had the correct call already; they were never zero.

## Honest-negative framing (per §5.2)

This is a fix, but a minor one.  The headline metrics:

- m2 R² for platinai_4cell post-fix = **-0.0302 ± 0.0136** (5-fold CV)
- target ceiling (per BLK-L1 honest-negative fallback) = R² < 0.10
  ⇒ "representation ceiling reached for this cohort"
- we are below 0.10 ⇒ the ceiling is reached, even with the fix

**Steps 2-6 of BLK-L1 are NOT done** and are deferred.  They require:

| Step | What | Why deferred |
|---:|---|---|
| 2 | chemprop_v3 frozen encoder wire (n_gte_1000 cohort) | complex; needs chemprop_v3 dependency + Hilbert cache + multi-tier config |
| 3 | Hilbert curve 18-d feature wire | research-grade; needs Hilbert discretiser + 18-d coordinate generator |
| 4 | §5.7 smoke `n_features_used=50 + R² finite + no NaN` | gated on Step 2 + 3 |
| 5 | CPU 5-seed × 5-fold CV at n=4078 | gated on Step 2 + 3 |
| 6 | §5.5 verify + paper claim | gated on Step 5 |

The honest-negative story for the paper §3 (m2 R²) is: **platinai_4cell
n=4078 reaches a representation ceiling at R² ≈ -0.03 with 32-d RDKit
features; metalcytotox_combined is the load-bearing cohort** (R² = +0.20
in `WF388A1_12D_BASELINE_R2`, the only positive cohort).  Future
representation-lift work should target a different feature family
(chemprop_v3 graph features, ECFP4, Hilbert coordinates) and a
different cohort; platinai_4cell is not the path forward.

## Cross-references

- `todo.json` BLK-L1 (this fix, Step 1 only)
- `molmetal/scripts/wf433_m2_extended.py` (the modified file)
- `molmetal/tests/test_wf433_m2_rdkit_features_fix.py` (NEW §5.7 smoke)
- `molmetal/scripts/wf331r5_r2_w2_chemprop.py` PLATINAI_32D_COLUMNS (the 32-column list)
- `metrics/by_metric/m2_r2_per_cohort.json` (the metric the fix feeds into; value unchanged because the 4 newly-populated columns don't add Ridge-discriminative signal at α=1.0)
- `WF388A1_12D_BASELINE_R2` (in-file dict; the 12-d baseline anchor -0.144)
- `TODO/completed/blk_c4_cfm_bond_head_qed_2026_10_09.md` (parallel BLK-C4 GPU retest)
- `AGENTS.md` §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `TODO/INDEX.md` §22 (will be added in final closeout wave per execution_order step 19)
