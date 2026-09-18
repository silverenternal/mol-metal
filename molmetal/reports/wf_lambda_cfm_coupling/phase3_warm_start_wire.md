# Phase 3 — Wire coupling into warm_start.py (env-gated CFM pocket embedding)

**Workflow:** Lambda × CFM coupling (TODO-21 re-opened 2026-09-16)
**Phase:** 3 of 4 — warm_start wiring
**Status:** DONE
**Date:** 2026-09-16

## Goal

Make :func:`molmetal_lam.search_alg.warm_start.pocket_features`
consume the CFM-derived 64-d embedding from
:meth:`CouplingAdapter.embed_pocket` when ``COUPLING_ENABLED=1``
**and** a ``coupling_adapter`` kwarg is supplied.  When the env
var is unset / falsy the existing deterministic descriptor is
returned (backward-compat guarantee).

## What changed

File: :file:`molmetal/molmetal_lam/search_alg/warm_start.py`

* Added a private helper :func:`_coupling_env_enabled` that
  mirrors :func:`molmetal_lam.lam_chem.coupling_adapter.is_coupling_enabled`.
* Added a new kwarg ``coupling_adapter: Optional[Any] = None``
  to :func:`pocket_features`.  When the kwarg is non-None AND
  the env gate is on, the returned
  :class:`PocketFeatureVector.values` is built from
  ``adapter.embed_pocket(v7, pocket_name=...)`` (the 7-d
  hand-crafted descriptor becomes the *input* to the adapter).
* Failure modes:
  - Adapter returns wrong shape → fall back to the deterministic
    vector (never break the pipeline).
  - Adapter raises → caught, fall back silently.
  - NaN / inf in adapter output → already handled inside the
    adapter (``nan_to_num`` on the forward path).
* When the coupling is active the
  ``feature_names`` field is replaced by the single
  ``"coupling_adapter_embedding_64d"`` marker so consumers can
  tell at a glance whether the deterministic or the coupling path
  was used.

## Backward compatibility

* No existing call site is broken.  All callers that omit the
  ``coupling_adapter`` kwarg see the original deterministic
  vector regardless of the env var.
* Even when the env var is on, omitting ``coupling_adapter`` still
  returns the deterministic vector (so callers must opt in to
  *both* the env var and the kwarg).
* Determinism of the default path: still bit-for-bit reproducible
  given the same input residues.

## Tests (Phase 3)

* :func:`test_warm_start_default_is_deterministic` — same input →
  identical output (no coupling).
* :func:`test_warm_start_adapter_ignored_when_env_off` — env off →
  adapter kwarg is a no-op.
* :func:`test_warm_start_adapter_active_when_env_on` — env on →
  ``pocket_features.values`` comes from the adapter (different
  from the deterministic vector); ``feature_names`` becomes the
  coupling marker.
* :func:`test_warm_start_adapter_pocket_specificity` — different
  residue shells → different embeddings when the adapter is on.

## Honest framing

The Phase-3 wire is a **drop-in, opt-in** integration.  No
existing call site sees a behaviour change unless it explicitly
opts in.  When the real GPU CFM retrain lands, the
:func:`CouplingAdapter` will produce a richer 64-d vector and
the same wiring will pick it up automatically — no further code
changes required.

The adapter's *output magnitude* depends on the underlying MLP
weights.  The dry-run checkpoint produces embeddings of
magnitude ~7 (He-init embed layer + zero head); a real
training run will produce embeddings of magnitude ~1 (after
ReLU + linear projection normalisation).  Consumers should
not assume a specific magnitude.

## Deliverables checklist

- [x] ``warm_start.py`` accepts ``coupling_adapter`` kwarg
- [x] ``COUPLING_ENABLED`` env gate (default off)
- [x] Backward-compatible: no behaviour change without opt-in
- [x] Failure modes → graceful fallback to deterministic vector
- [x] 4 unit tests added (test_coupling_adapter.py)
- [x] All tests pass