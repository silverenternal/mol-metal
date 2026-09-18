# Phase 4 — Wire coupling into learned_prior.py (5-d bias from the CFM pocket embedding)

**Workflow:** Lambda × CFM coupling (TODO-21 re-opened 2026-09-16)
**Phase:** 4 of 4 — learned_prior wiring
**Status:** DONE
**Date:** 2026-09-16

## Goal

Use the 64-d pocket embedding from
:meth:`CouplingAdapter.embed_pocket` to bias the 5-way classifier
readout of
:class:`molmetal_lam.search_alg.learned_prior.LearnedPolicyPrior`.
The bias is **zero-mean by construction** so it shifts the prior
toward rules that *the CFM embedding* thinks will fire, while
keeping the prior normalised.

## What changed

File: :file:`molmetal/molmetal_lam/search_alg/learned_prior.py`

* Added a private helper :func:`_coupling_env_enabled` that
  mirrors the gate in :mod:`warm_start`.
* Added a new field ``coupling_adapter: Optional[object] = None``
  to :class:`LearnedPolicyPrior`.  When the field is non-None
  AND the env gate is on, :meth:`__post_init__` computes a 5-d
  bias tensor by:
  1. Calling ``adapter.embed_pocket(None, pocket_name="coupling_default")``
     to get a 64-d vector.
  2. Reshaping into 5 blocks of 12 slots and taking the mean of
     each block → a 5-d vector.
  3. Subtracting the mean (zero-centring) so the bias does not
     favour any one rule a priori.
* The cached bias is added to the classifier readout **inside**
  :meth:`predict_proba` and :meth:`batch_predict_proba` before
  softmax.  Both call sites are guarded by an ``is not None``
  check so the bias is silently absent when the env gate is off.
* Added a new method :meth:`LearnedPolicyPrior.set_coupling_pocket`
  that refreshes the cached bias from a fresh pocket vector
  (useful when the pocket changes across a sweep).

## Backward compatibility

* All existing call sites that omit ``coupling_adapter`` see
  identical behaviour: the cached bias is ``None`` and the
  readout is unmodified.
* The GRU forward pass is **unchanged** — the bias is added
  *after* the GRU readout, not before.  This means the existing
  GRU training (the functional-group-overlap fit) is unaffected.
* The bias projection (5 block means + zero-centring) is
  deterministic given the same adapter and the same
  ``pocket_name``.

## Tests (Phase 4)

Three torch-dependent tests (skipped on this host because of a
pre-existing libtorch ABI issue — torch import fails with
``_PyThreadState_UncheckedGet`` undefined symbol):

* :func:`test_learned_prior_coupling_adapter_env_off_no_bias` —
  env off → ``_coupling_bias`` is ``None``.
* :func:`test_learned_prior_coupling_adapter_env_on_caches_bias`
  — env on → 5-d bias cached, zero-mean, refreshable via
  ``set_coupling_pocket``.
* :func:`test_learned_prior_predict_proba_uses_bias` —
  ``predict_proba`` still sums to 1 and every entry is finite.

## Honest framing

The 64-d → 5-d block-mean projection is the **simplest**
deterministic projection that is also pocket-specific.  A
learned projection (e.g. an MLP ``R^64 → R^5`` trained jointly
with the classifier) is queued for Phase-5 but is out of scope
here.

The bias is **zero-mean by construction** so adding it does not
shift the prior's mean — it just reshapes the per-rule
distribution.  When the underlying coupling MLP is fresh
(dry-run weights) the bias is essentially zero and the prior
collapses to its default behaviour; only when the coupling MLP
is well-trained will the bias be meaningful.

## Deliverables checklist

- [x] ``learned_prior.py`` accepts ``coupling_adapter`` field
- [x] ``COUPLING_ENABLED`` env gate (default off)
- [x] Backward-compatible: no behaviour change without opt-in
- [x] 5-d bias projection (block-mean + zero-centring)
- [x] New method ``set_coupling_pocket`` for per-pocket refresh
- [x] 3 unit tests added (test_coupling_adapter.py, torch-skipped on broken libtorch)
- [x] All non-torch tests pass (15/15)