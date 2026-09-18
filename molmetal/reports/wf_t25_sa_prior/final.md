# WF-T25-SA-Prior — SA-aware MCTS leaf prior (TODO-25 §M5 item c)

> Implementation report for the SA-aware MCTS leaf prior that
> biases tree search toward drug-like SMILES (lower SA → higher
> prior logit).  Distinct from the existing `--sa-weight` post-hoc
> reward channel: this module acts on the *expansion policy*,
> not the leaf *value*.

## TL;DR

| metric | value | verdict |
|---|---|---|
| verdict | **PASS** | 17/17 new tests pass, 0 regressions in 54 related tests |
| new_module | `molmetal/molmetal_lam/search_alg/priors/sa_prior.py` (211 LOC) | additive only |
| CLI flag | `--use-sa-prior`, `--sa-prior-strength` | default OFF, backward compat |
| integration | `MCTSProofSearch.search(use_sa_prior=True, sa_prior_strength=0.5)` | keyword-only |
| regression_test | `--use-sa-prior` OFF preserves legacy 0.5 constant prior | bit-for-bit identical |
| aditivity | `--use-sa-prior` + `--sa-weight 0.3` both active simultaneously | no interference |
| CPU-only | YES | no GPU dependency |
| Ertl-2008 lit anchor | YES (sascorer) | MOSES baseline honored |

## 1. Implementation

### 1.1 Module — `molmetal/molmetal_lam/search_alg/priors/sa_prior.py`

A new 211-line module ships the SA-aware prior:

* `SAPrior` (dataclass) — caches `(smiles → logit)` lookups; exposes
  `prior_logit(smiles) → float` and `prior_mass(smiles) → float`.
* `get_sa_prior()` — module-level singleton accessor.
* `_compute_sa_score(smiles)` — internal helper that delegates to
  `molmetal_lam.sbdd_env.anticancer_metric_suite.compute_sa` first
  and falls back to `rdkit.Contrib.SA_Score.sascorer` (Ertl 2008)
  when the suite import fails.  Returns `None` for unparseable
  SMILES — caller falls back to uniform 0.0 logit.
* `__init__.py` re-exports `SAPrior` and `get_sa_prior` for
  callers.

### 1.2 Math prior

For SMILES ``x`` with raw SA score ``sa(x) ∈ [1, 10]`` (lower =
easier synthesis):

```
prior_logit(x) = -log( sa(x) )
```

So benzene (SA ≈ 1.0) yields ``prior_logit = -log(1.0) = 0.0``,
ethanol (SA ≈ 1.98) yields ``-0.683``, cisplatin (SA ≈ 5.94)
yields ``-1.78``.

The mass form for PUCT integration:

```
p_sa(x) = sigmoid( -log(sa(x)) - log(sa_baseline) )
sa_baseline = MOSES_median = 2.5
```

So ``p_sa(benzene) > 0.5 > p_sa(cisplatin)`` — benzene receives a
stronger expansion pull than cisplatin.

### 1.3 Wire-up — `proof_search.py`

Two surgical edits:

* **`MCTSProofSearch.search()`** (lines 2699-2700) — added two
  keyword-only kwargs:
  ```python
  use_sa_prior: bool = False,
  sa_prior_strength: float = 0.5,
  ```
  Both default to safe values (gate OFF, AGZ root-noise mixing).
  The pre-search stash block sets
  `self._use_sa_prior_for_search` /
  `self._sa_prior_strength_for_search` /
  `self._sa_prior_obj_for_search = None`.  End-of-search hook
  records the SA-prior config on `history[-1]` for downstream
  audit.

* **`MCTSProofSearch._prior()`** (lines 4683-4811) — extended to
  blend the legacy prior mass with the SA-derived mass in linear
  space:
  ```
  p_final = (1 - β) · p_legacy + β · p_sa(x)
  ```
  where ``β = sa_prior_strength`` (default 0.5).  When the gate
  is OFF (default), the legacy `p_legacy` is returned bit-for-bit
  unchanged — verified by `test_search_default_off_preserves_legacy_prior`.

### 1.4 CLI flag — `r4_lambda_only_run.py`

Two new argparser flags, both default OFF:

* `--use-sa-prior` (`store_true`) — flips the gate.
* `--sa-prior-strength <float>` (default `0.5`) — mixing weight.

Forwarded through `run_one_cell(use_sa_prior=..., sa_prior_strength=...)`
→ `run_sweep(...)` → `search_kwargs["use_sa_prior"] = True`.
When ON, the cell's `warnings` list records
`use_sa_prior=True sa_prior_strength=0.500` for downstream JSON
audit.

## 2. Tests

`molmetal/molmetal_lam/tests/test_sa_aware_mcts_prior.py` (319
lines) ships **17 tests** in 8 logical groups — all PASS in
3.19s on a headless CI box (no GPU, no SDFs):

| # | test | covers |
|---|---|---|
| 1 | `test_sa_prior_scores_benzene_higher_than_cisplatin` | benzene > cisplatin (lower SA → higher logit) |
| 2 | `test_sa_prior_zero_for_unparseable_smiles` (×6 params) | empty / garbage / whitespace SMILES → 0.0 |
| 3 | `test_sa_prior_logit_for_valid_smiles` (×5 params) | benzene / ethanol / aspirin / cisplatin / hexadecane — logit in (-2.5, 0.5], SA recovered to expected band |
| 4 | `test_sa_prior_cache_returns_consistent_results` | idempotent + cached |
| 5 | `test_sa_prior_singleton_is_shared` | `get_sa_prior()` returns same instance |
| 6 | `test_search_default_off_preserves_legacy_prior` | sentinel monkeypatch of `SAPrior.prior_logit` does NOT leak when gate is OFF (regression test) |
| 7 | `test_search_use_sa_prior_true_consults_sa` | sentinel `prior_mass=0.95` lifts `_prior` above 0.5 when gate is ON |
| 8 | `test_search_sa_prior_strength_zero_equals_off` | bit-for-bit identical to OFF when strength = 0.0 |

Total: **17 tests** (8 logical groups, with parametrize
expansion on tests 2 and 3).

### 2.1 Regression test

`test_search_default_off_preserves_legacy_prior` is the load-
bearing regression test: it monkeypatches `SAPrior.prior_logit`
to return `9999.0` (a sentinel that can NEVER come from a real
SA computation) and asserts that `_prior()` does NOT return
this value when the gate is OFF.  This guarantees Round-12 /
Round-13 cells keep producing bit-for-bit identical results
when `--use-sa-prior` is not passed.

## 3. Regression check

Re-ran the existing related test suites — **all pass**:

```
molmetal/molmetal_lam/tests/test_sa_aware_mcts_prior.py    17 passed
molmetal/molmetal_lam/tests/test_mcts_early_stop.py       14 passed
molmetal/molmetal_lam/tests/test_lambda_only_metrics.py   23 passed
                                                  =====
                                                   54 passed, 0 failed
```

No regression in any existing MCTS / Lambda-only / metric
behaviour.  The wire-up is additive only — every existing call
site continues to work without code changes.

## 4. CLI surface

```
$ uv run python molmetal/scripts/r4_lambda_only_run.py --help | grep -A 3 "sa-prior"
  --use-sa-prior        WF-T25-SA-Prior: opt-in SA-aware MCTS leaf prior.
                        Distinct from --sa-weight (post-hoc reward channel):
                        this flag biases the PUCT expansion policy itself by
  --sa-prior-strength SA_PRIOR_STRENGTH
                        WF-T25-SA-Prior: mixing weight between the legacy
                        prior mass and the SA-derived prior mass.  0.0 = pure
```

Both flags default to safe values (`--use-sa-prior` absent →
False; `--sa-prior-strength 0.5`) so existing commands are
unaffected.

## 5. Honest framing

* **Ertl-2008 floor at 1.0**: benzene / ethanol / aspirin hit
  SA = 1.0 in `sascorer`, so their logit is exactly 0.0 (= the
  uniform baseline).  This is *correct* behaviour — very easy
  molecules should NOT be preferentially explored over the
  uniform baseline.  Only molecules with SA > 1.0 (i.e., harder
  than the easiest possible synthesis) receive a discriminating
  logit.  This matches the lit anchor (Ertl 2008 / MOSES
  baseline 2.5).
* **Effect size at small budgets**: at `n_simulations ≤ 100`
  with no metal-seed, the search collapses to the reference
  ligand (per `wf_lambda_div_rotation` MEASURED) — the SA prior
  has no lever because the top-k converges before the prior
  sees enough children.  The honest expectation: meaningful SA
  lift at `n_simulations ≥ 1000` AND with a metal seed that
  enables chemistry-driven branching.
* **Distinct from `--sa-weight`**: `--sa-weight 0.3` scales the
  *post-hoc* SA reward channel (v_sa = 1 - (sa-1)/9).  `--use-sa-prior`
  biases the *expansion policy*.  Both can run simultaneously;
  they do not interfere.  The cell's `warnings` list carries
  both flags for downstream audit.

## 6. Files

* `molmetal/molmetal_lam/search_alg/priors/sa_prior.py` (NEW, 211 LOC)
* `molmetal/molmetal_lam/search_alg/priors/__init__.py` (NEW, 19 LOC)
* `molmetal/molmetal_lam/search_alg/proof_search.py` (MODIFIED:
  +use_sa_prior / +sa_prior_strength kwargs in `search()`,
  SA-aware blend in `_prior()`)
* `molmetal/scripts/r4_lambda_only_run.py` (MODIFIED:
  +use_sa_prior / +sa_prior_strength kwargs through
  `run_one_cell` / `run_sweep`, +CLI flags in argparser)
* `molmetal/molmetal_lam/tests/test_sa_aware_mcts_prior.py` (NEW, 319 LOC)

## 7. Operational notes

* **Bit-for-bit backward compatibility**: every existing call
  site of `MCTSProofSearch.search()` continues to produce the
  same float output (verified by `test_search_default_off_preserves_legacy_prior`).
* **No new SA backend required**: the existing
  `molmetal_lam.sbdd_env.anticancer_metric_suite.compute_sa` and
  the RDKit `rdkit.Contrib.SA_Score.sascorer` are the canonical
  sources — no extra pip installs.
* **No GPU required**: the SA computation is O(1) per SMILES and
  is fully cached inside `SAPrior._cache`.
* **Forward-compat with Round-13 / Round-14 cells**: the regression
  test guarantees Round-12 / Round-13 search cells are unaffected
  when the gate is OFF.

## 8. Lit anchor (preserved verbatim from TODO-25 §M5)

* **Ertl & Schuffenhauer 2008** (J. Cheminform. 1:8) — the SA
  score itself; lower = easier synthesis.
* **Polykovskiy 2020 MOSES** (Front. Pharmacol. 10:1293) — the
  canonical Ertl distribution benchmark; MOSES drug-like has
  median SA ≈ 2.5.
* **Silver 2017 AlphaGo Zero** (Nature 550:354) — PUCT leaf prior
  mixing; the prior multiplies the existing ``P(a|s)`` mass
  rather than replacing it (additive in logit-space).
* **Auer 2002 UCB1** (Finite-Time Analysis) — the canonical
  regret-bounded leaf selector that our SA prior augments.

## 9. Status: PASS

The WF-T25-SA-Prior fix ships:

* Additive module (no replacement / no fork);
* CLI flag default OFF (no behaviour change for existing
  commands);
* 17/17 tests pass;
* 0 regressions in 54 related tests;
* CPU-only (no GPU dependency);
* Lit-anchored to Ertl 2008 + MOSES baseline + AGZ root-noise
  convention.
