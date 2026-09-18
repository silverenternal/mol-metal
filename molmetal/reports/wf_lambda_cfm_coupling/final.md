# Lambda × CFM coupling — Phase 1 ship (final report)

**Workflow:** Lambda × CFM coupling (TODO-21 re-opened 2026-09-16)
**Phase:** All 4 phases
**Status:** SHIP
**Date:** 2026-09-16
**Wall-budget:** ~25 min (well under the 90-min budget)

## TL;DR

Three new modules + one new test file + two edits to existing
modules:

| File                                                        | Status  | Lines  |
|-------------------------------------------------------------|---------|--------|
| `molmetal/scripts/tmqm_cfm_pretraining.py`                  | NEW     | ~270   |
| `molmetal/molmetal_lam/lam_chem/coupling_adapter.py`        | NEW     | ~280   |
| `molmetal/molmetal_lam/search_alg/warm_start.py`            | edited  | +60    |
| `molmetal/molmetal_lam/search_alg/learned_prior.py`         | edited  | +60    |
| `molmetal/tests/test_coupling_adapter.py`                   | NEW     | ~340   |
| `molmetal/reports/wf_lambda_cfm_coupling/{phase1..4}.md`    | NEW     | 5×~80  |

All 15 non-torch tests pass; 3 torch-dependent tests skip on
this host because of a pre-existing libtorch ABI issue (the
``_PyThreadState_UncheckedGet`` undefined symbol is not caused
by any of our changes — it blocks any import of
``torch.nn.Module``).  The tests are structured so they will
run automatically when the host's libtorch is fixed.

## Phases

1. **Phase 1 — tmqm_cfm_pretraining.py.**  CPU-only stand-in
   script that trains a 2-layer MLP on the 8-mol dry-run
   corpus and writes a deterministic ``.npz`` + ``.json``
   checkpoint pair.  Tries tmQM first when available; falls
   back to dry-run with a ``fallback_reason`` recorded in the
   metadata.  See :file:`phase1_tmqm.md`.

2. **Phase 2 — coupling_adapter.py.**  Torch-free loader +
   :meth:`CouplingAdapter.embed_pocket` facade.  Returns a
   finite 64-d vector from the loaded checkpoint, with a
   deterministic stub fallback when the checkpoint is missing.
   See :file:`phase2_adapter.md`.

3. **Phase 3 — Wire into warm_start.py.**  New
   ``coupling_adapter`` kwarg on :func:`pocket_features`.  When
   ``COUPLING_ENABLED=1`` AND the kwarg is supplied, the
   returned 64-d vector is built from
   ``adapter.embed_pocket(v7, pocket_name=...)``.  Backward-
   compatible: existing callers see no change.  See
   :file:`phase3_warm_start_wire.md`.

4. **Phase 4 — Wire into learned_prior.py.**  New
   ``coupling_adapter`` field on
   :class:`LearnedPolicyPrior`.  When ``COUPLING_ENABLED=1``,
   the 64-d pocket embedding is projected to a 5-d
   zero-mean bias that is added to the classifier readout
   before softmax.  GRU is unchanged.  Backward-compatible.
   See :file:`phase4_learned_prior_wire.md`.

## Test results

```
$ python -m pytest molmetal/tests/test_coupling_adapter.py --no-header -q
...............sss                                                       [100%]
15 passed, 3 skipped in 0.35s
```

* 15 tests pass on CPU (no torch required).
* 3 tests skip because the host's libtorch is broken
  (``_PyThreadState_UncheckedGet`` undefined symbol in
  ``/opt/libtorch/lib/libtorch_python.so``).  This is a host
  issue, not a code issue — the tests will run on any working
  libtorch install.

## Honest framing (TODO-21 honest framing requirements)

### What this Phase-1 ship **delivers**

* A working bridge between the (CPU-stand-in) CFM checkpoint
  and the Lambda MCTS root prior / warm-start embedding.
* Bit-for-bit deterministic, env-gated, opt-in integration
  with **zero** behaviour change for existing callers.
* A test suite that proves the wiring contract.

### What this Phase-1 ship **does NOT** deliver

* A **trained** CFM pocket embedding.  The CPU MLP was
  trained on 8 mols with 1 SGD step (the dry-run path).  The
  real tmQM-corpus EGNN warm-start is blocked on the GPU
  outage (``wf_gpu_auto_recover`` 2026-09-15).  Once the GPU
  is back, the real training run will produce a better
  checkpoint that the same ``load_coupling_adapter`` factory
  will pick up — no further code changes required.
* A measurable lift in pocket-specificity over the
  deterministic descriptor.  This is a wiring milestone, not
  a measurement milestone.  Real metrics require the trained
  checkpoint + a pocket-conditioned MCTS pilot (Round-14+).
* Any change to ``proof_search.py`` (which is **owned by**
  Phase-3H per the workflow contract).  The hook is
  ``pocket_features(...)`` and ``LearnedPolicyPrior(...)``
  on the consumer side; downstream code can opt in at their
  own pace.

### Decision tree

* **If GPU recovers** → re-run
  ``molmetal.scripts.tmqm_cfm_pretraining`` without
  ``--dry-run`` (or, better, replace it with a GPU EGNN init
  script that writes a compatible ``.npz``).  The same
  ``load_coupling_adapter`` factory will pick it up.  No code
  change required.
* **If GPU stays blocked** → the dry-run checkpoint is
  shipped as a stand-in.  All Phase-1 wiring works; the bias
  magnitude is ~7 (He-init) instead of ~1 (well-trained).
  Consumers see the same shape contract, just different
  magnitudes.
* **Round-12 pilot decision** → not affected by this Phase-1
  ship.  Round-12 stays on path (c) λ-only per
  ``WF-GPU-Recovery-Now`` verdict 2026-09-15.

## Constraint compliance

| Constraint                                     | Met? |
|------------------------------------------------|------|
| ALLOWED: tmqm_cfm_pretraining.py (NEW)         | YES  |
| ALLOWED: coupling_adapter.py (NEW)             | YES  |
| ALLOWED: warm_start.py (edit)                  | YES  |
| ALLOWED: learned_prior.py (edit)               | YES  |
| ALLOWED: test_coupling_*.py (NEW)              | YES  |
| FORBIDDEN: flow_matching_lipman/*              | YES (not touched) |
| FORBIDDEN: pocket_macro_*                      | YES (not touched) |
| FORBIDDEN: lambda_combinators.py               | YES (not touched) |
| FORBIDDEN: beta_reductions.py                  | YES (not touched) |
| FORBIDDEN: pt_click_compat.py                  | YES (not touched) |
| FORBIDDEN: r4_lambda_only_run.py               | YES (not touched) |
| FORBIDDEN: paper/*                             | YES (not touched) |

## References

* TODO-21 (Lambda × CFM coupling deferred 2026-09-14,
  re-opened 2026-09-16)
* Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020 — tmQM
  https://doi.org/10.1021/acs.jcim.0c01041
* Peng et al., *Pocket2Mol*, arXiv:2205.07249 (2022) — 64-d
  pocket embedding convention we follow
* Satorras et al., *E(n) Equivariant Graph Neural Networks*,
  arXiv:2102.09844 (2021) — EGNN init the real GPU retrain
  will warm-start from
* Lipman et al., *Flow Matching for Generative Modeling*,
  arXiv:2210.03629 (ICLR 2023) — CFM formalism
* ``wf_gpu_auto_recover`` 2026-09-15 — GPU outage root cause
* ``wf_lambda_metal_pilot`` 2026-09-14 — metal-seed singleton
  collapse that motivated this coupling