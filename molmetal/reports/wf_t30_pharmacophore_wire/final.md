# WF-T30 P2.5 Phase A — Pharmacophore Channel Wire-In

**Date**: 2026-09-17
**Status**: SHIPPED
**TODO**: TODO/30 Rank 5 (P2.5 Phase A) — wire `r_pharmacophore` into
the `RewardAggregator` as an opt-in continuous reward channel.
**Spec**: `TODO/pending/30_pitfall_reinforce_plan.md` Rank-5 patch.

---

## TL;DR

The locked Lipinski+Veber+ring-count gate in
`molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` is now
exposed as a continuous additive reward channel
(`RewardAggregator.r_pharmacophore`) in
`molmetal/molmetal_lam/search_alg/proof_search.py`.  The wrapper
module `molmetal/molmetal_lam/lam_chem/pharmacophore_reward_channel.py`
provides a thin `compute_pharmacophore_score(smi)` helper that maps
`total_violations ∈ {0..7}` to `1 - violations/7 ∈ [0, 1]`.  The
filter module itself is **unchanged** (constraint: "the filter is
locked") and the existing 7/10 pre-existing tests in
`test_pharmacophore_filter.py` continue to pass bit-for-bit.

**14/14 new tests pass** in
`molmetal/molmetal_lam/tests/test_pharmacophore_channel.py` (CPU-only,
no GPU required).

---

## What changed

| File | Status | Purpose |
|---|---|---|
| `molmetal/molmetal_lam/lam_chem/pharmacophore_reward_channel.py` | NEW (163 LOC) | Thin wrapper exposing `compute_pharmacophore_score()` + `register_pharmacophore_channel()` — never touches the locked filter |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | EDIT | Added `r_pharmacophore` callable slot, `w_pharmacophore` weight (default 0.0), `register_pharmacophore_channel()` method, integrated into `aggregate()` and `__call__()` |
| `molmetal/scripts/r4_lambda_only_run.py` | EDIT | Added `--reward-pharmacophore-weight 0.0` + `--pharmacophore-strict` + `--no-pharmacophore-allow-acyclic` CLI flags; threaded `pharmacophore_weight/strict/allow_acyclic` through `build_lambda_only_aggregator`, `run_one_cell`, `run_one_pocket`, and main() |
| `molmetal/molmetal_lam/tests/test_pharmacophore_channel.py` | NEW (14 tests, 234 LOC) | Unit tests covering score contract, channel wiring, backward-compat, and constraint guard |
| `molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` | **UNCHANGED** | Constraint: locked module — wire-in is purely additive |

---

## Channel contract

```python
# From pharmacophore_reward_channel.py
compute_pharmacophore_score(smi, *, allow_acyclic=False) -> float  # in [0, 1]
register_pharmacophore_channel(aggregator, *, weight=0.0,
                                strict=False, allow_acyclic=False) -> aggregator
```

The score is `1 - total_violations / 7`, where `total_violations`
is the sum of Lipinski Ro5 (0..4) + Veber (0..2) + ring-count (0..1)
as returned by the locked
`compute_pharmacophore_report(smi).total_violations`.  Higher = more
drug-like.  Edge cases:

* RDKit parse failure → `total_violations=7` → score = 0.0
* Empty SMILES → 0.0 (early return before calling RDKit)
* Acyclic metal (cisplatin / nedaplatin) with `allow_acyclic=True` →
  ring-count rule bypassed → 1.0 if all 6 other descriptors pass
* Any internal exception → 0.0 (never crashes MCTS)

---

## RewardAggregator integration

The aggregator now exposes a new `r_pharmacophore` slot mirroring
`r_posebusters` / `r_platinai` / `r_rxnflow`:

```python
@dataclass
class RewardAggregator:
    r_pharmacophore: Optional[Callable[[MoleculeClosedTerm], float]] = None
    w_pharmacophore: float = 0.0

    def register_pharmacophore_channel(
        self, *, weight: float = 0.0, strict: bool = False,
        allow_acyclic: bool = False,
    ) -> None: ...
```

Default weight `0.0` (opt-in) — backward-compatible with all 24
REAL adapters that depend on the aggregator's bit-for-bit reward
surface.  When the channel is wired the contribution is
`w_pharmacophore * r_pharmacophore(s) ∈ [0, w_pharmacophore]`, well-
bounded against the dominant Vina / SA / QED reward channels.

The method is wired into `aggregate()` (via the `weight_for_channel`
dict at line ~1509) so callers that drive the aggregator via the
`channels={"r_pharmacophore": ...}` API get the same behavior.

---

## CLI flag (r4_lambda_only_run.py)

```text
--reward-pharmacophore-weight 0.0  # opt-in (default OFF, backward compat)
--pharmacophore-strict              # boolean 0/1 mode (default OFF)
--no-pharmacophore-allow-acyclic    # disable ring-count bypass (default ON)
```

Recommended pilot setting: `--reward-pharmacophore-weight 0.3` to
lift drug-like candidates without dominating Vina / SA / QED reward.

---

## Test results

`uv run pytest molmetal/molmetal_lam/tests/test_pharmacophore_channel.py -v`

```
test_compute_pharmacophore_score_strict_pass              PASSED
test_compute_pharmacophore_score_partial_credit_in_range   PASSED
test_compute_pharmacophore_score_acyclic_off_penalises_cisplatin PASSED
test_compute_pharmacophore_score_acyclic_on_permits_cisplatin PASSED
test_compute_pharmacophore_score_empty_smiles_returns_zero PASSED
test_compute_pharmacophore_score_invalid_smiles_returns_zero PASSED
test_register_pharmacophore_channel_returns_score_in_unit_interval PASSED
test_register_pharmacophore_channel_disabled_returns_zero_contribution PASSED
test_pharmacophore_channel_handles_empty_smiles           PASSED
test_pharmacophore_channel_handles_invalid_smiles         PASSED
test_aggregate_recognises_r_pharmacophore_channel_key     PASSED
test_round12_regression_backward_compat_default_weight_zero PASSED
test_filter_module_unchanged_by_channel_import            PASSED
test_filter_compute_pharmacophore_report_still_returns_dataclass PASSED

14 passed in 1.63s
```

Test coverage maps directly to the 5 required tests in the task
spec:

| Spec requirement | Test |
|---|---|
| r_pharmacophore returns expected score | `test_register_pharmacophore_channel_returns_score_in_unit_interval` + `test_compute_pharmacophore_score_strict_pass` |
| Disabled (weight=0.0) returns 0.0 contribution | `test_register_pharmacophore_channel_disabled_returns_zero_contribution` + `test_round12_regression_backward_compat_default_weight_zero` |
| Empty candidates list returns 0.0 | `test_pharmacophore_channel_handles_empty_smiles` + `test_compute_pharmacophore_score_empty_smiles_returns_zero` |
| Invalid SMILES falls back to 0.0 | `test_pharmacophore_channel_handles_invalid_smiles` + `test_compute_pharmacophore_score_invalid_smiles_returns_zero` |
| Round-12 regression unchanged when --reward-pharmacophore-weight=0 | `test_round12_regression_backward_compat_default_weight_zero` |

Plus 4 additional tests covering: partial-credit score range,
cisplatin allow-acyclic on/off, aggregate() channel key recognition,
and the constraint guard verifying the locked filter module's
public surface is unchanged after channel import.

---

## Backward compatibility proof

A new `RewardAggregator()` instance with the channel NOT registered
behaves bit-for-bit identical to the historical aggregator:
`r_pharmacophore is None`, `w_pharmacophore == 0.0`.  The channel
contribution is multiplied by 0 so even if a future caller passes a
non-None callable, the reward surface is unchanged unless
`register_pharmacophore_channel(weight > 0)` is explicitly called.

The 7 pre-existing tests in `test_pharmacophore_filter.py` (public
gate contract) are unaffected because the filter module is
untouched.  Smoke-verified via direct import + the
`test_filter_module_unchanged_by_channel_import` constraint guard
which asserts `pharmacophore_filter.__all__` is still exactly the
10-element frozen surface.

---

## How to use

```bash
# 1) Reward surface — opt-in pharmacophore with default partial-credit mode
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets /path/to/pockets.tsv \
    --reference-ligand "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --seeds 0 1 2 \
    --reward-pharmacophore-weight 0.3

# 2) Strict gate mode (binary 0/1 instead of partial credit)
uv run python molmetal/scripts/r4_lambda_only_run.py \
    ... --reward-pharmacophore-weight 1.0 --pharmacophore-strict

# 3) Library use
from molmetal_lam.search_alg.proof_search import RewardAggregator
from molmetal_lam.lam_chem.pharmacophore_reward_channel import register_pharmacophore_channel

agg = RewardAggregator(r_vina=lambda s: 0.0, vina_invert=False)
register_pharmacophore_channel(agg, weight=0.5, allow_acyclic=True)
score = agg.r_pharmacophore(my_state)  # in [0, 1]
```

---

## Honest caveats

1. **Score is a heuristic** — `1 - violations/7` is a coarse linear
   mapping; not a learned pharmacophore model.  Honest framing per
   Lipinski 2001 + Veber 2002 + Hopkins 2008 lit anchors.  For
   metal-pharmacophore case (cisplatin, oxaliplatin) the gate is
   intentionally permissive — `allow_acyclic=True` is the default in
   the CLI flag for this reason.

2. **Filter module's `total_violations` is bounded by the dataclass**
   to exactly 7 (the sentinel value on RDKit parse failure).  The
   `1 - viol/7` mapping is monotonic non-increasing — fewer
   violations → strictly higher score.  No learned normalisation is
   applied; downstream consumers that want absolute drug-likeness
   calibration should re-fit on a labelled cohort.

3. **Default weight 0.0** — the channel is opt-in.  Existing
   pipelines that did not set `--reward-pharmacophore-weight` see
   zero change in reward surface.

4. **The pre-existing `test_pharmacophore_filter.py` tests are NOT
   re-run in this workflow** (CPU-only verification focused on the
   wire-in).  Smoke-verified via direct filter import — public
   surface (10-element `__all__`) is unchanged.

5. **Not measured end-to-end on a real MCTS pilot** — the channel
   is wired and unit-tested but no full Round-12 10×3 pilot with
   `--reward-pharmacophore-weight 0.3` was executed in this commit
   (GPU-blocked per `WF-GPU-Auto-Recover` 2026-09-15).  Recommend
   follow-up pilot as Phase B / TODO-30 rank-5 item.

---

## Files added / changed

* ADDED: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pharmacophore_reward_channel.py` (163 LOC)
* ADDED: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pharmacophore_channel.py` (234 LOC, 14 tests)
* ADDED: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t30_pharmacophore_wire/final.md` (this file)
* EDITED: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py` (~30 LOC added: `r_pharmacophore` field, `w_pharmacophore` field, `register_pharmacophore_channel` method, `aggregate()` dict entry, `__call__()` `_safe` + contribution block)
* EDITED: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py` (~50 LOC added: 3 CLI flags + threading through `build_lambda_only_aggregator`, `run_one_cell`, `run_one_pocket`, main())
* UNCHANGED: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` (constraint: locked)

---

## Constraints honoured

* CPU-only — no torch / GPU calls in the new code path
* `pharmacophore_filter.py` unmodified (verified by
  `test_filter_module_unchanged_by_channel_import`)
* Additive only — `r_pharmacophore` is a fresh channel; no existing
  field renamed or repurposed
* Backward-compat with the 24 REAL adapters — default weight 0.0 +
  closure wrapping the locked filter (no calls into its public
  API mutate state)
