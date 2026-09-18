# Phase 2 — coupling_adapter.py (the bridge between the CFM checkpoint and the Lambda MCTS)

**Workflow:** Lambda × CFM coupling (TODO-21 re-opened 2026-09-16)
**Phase:** 2 of 4 — adapter module
**Status:** DONE
**Date:** 2026-09-16

## Goal

Build a tiny torch-free loader + ``embed_pocket()`` facade that
sits between the CFM-derived checkpoint (Phase 1) and the Lambda
MCTS consumers (Phases 3 and 4).

## What shipped

File: :file:`molmetal/molmetal_lam/lam_chem/coupling_adapter.py`

Public API:

* :class:`CouplingAdapter` — dataclass holding the loaded
  ``embed_w``, ``embed_b``, ``head_w``, ``head_b``, ``input_mean``,
  ``input_std`` arrays.
* :meth:`CouplingAdapter.embed_features(x)` — 9-d → 64-d forward
  pass; returns a finite numpy vector; defensively
  ``nan_to_num``s the output.
* :meth:`CouplingAdapter.embed_pocket(features, pocket_name=...)` —
  convenience wrapper that takes either the 7-d hand-crafted
  descriptor, a 64-d raw vector, or ``None`` (deterministic
  fallback with a pocket-name hash shift so two pockets with
  empty features still get distinct embeddings).
* :func:`load_coupling_adapter(path, prefer_stub)` — factory.
  Falls back to a deterministic stub adapter when the .npz is
  missing so the wiring code never sees ``None``.
* :func:`ensure_checkpoint_exists(...)` — idempotently ensures a
  checkpoint exists; trains a fresh dry-run if not.
* :func:`is_coupling_enabled()` and :func:`coupling_enabled_or(...)`
  — env gate helpers (shared by Phases 3/4).

Constants:

* :data:`INPUT_DIM` = 9 (matches
  :data:`molmetal.scripts.tmqm_cfm_pretraining.INPUT_DIM`).
* :data:`EMBED_DIM` = 64 (matches
  :data:`molmetal_lam.search_alg.warm_start.POCKET_FEATURE_DIM`).
* :data:`DEFAULT_CHECKPOINT_PATH` =
  ``molmetal/checkpoints/coupling/coupling_mlp.npz``.

## Determinism guarantees

* ``embed_pocket(features, pocket_name)`` is bit-for-bit
  reproducible across processes and Python versions given the
  same loaded weights and the same inputs.
* The stub adapter's ``embed_pocket(None, pocket_name="X")``
  differs across pocket names — useful in tests where we want
  pocket-specificity but do not yet have a real checkpoint.

## Tests (Phase 2)

* :func:`test_load_coupling_adapter_returns_64d_finite` — shape
  ``(64,)``, every entry finite, ``is_stub=False`` when a real
  checkpoint exists.
* :func:`test_load_coupling_adapter_stub_when_missing` — missing
  ``.npz`` → deterministic stub, still ``(64,)`` and finite.
* :func:`test_stub_adapter_deterministic` — same pocket_name →
  identical vector.
* :func:`test_adapter_pocket_specificity` — different
  pocket_names → different fallback vectors.
* :func:`test_is_coupling_enabled_truthy` — covers all the env-var
  truthy / falsy spellings.
* :func:`test_ensure_checkpoint_exists_idempotent` — calling
  twice does not rewrite.

## Honest framing

The adapter is **inference-only**.  It loads weights from disk
and runs the 2-layer MLP in numpy.  No training happens inside
the adapter — that is owned by Phase 1's script.  The adapter
itself is the **bridge**, not the **model**.

When the real GPU CFM retrain lands, the same
``load_coupling_adapter`` factory will load the trained
checkpoint and ``embed_pocket`` will produce a *better*
64-d vector.  The shape contract and the API are unchanged.

## Deliverables checklist

- [x] NEW file ``molmetal/molmetal_lam/lam_chem/coupling_adapter.py``
- [x] Public API: ``load_coupling_adapter`` + ``embed_pocket``
- [x] Deterministic stub fallback
- [x] Env gate helper ``is_coupling_enabled``
- [x] 6 unit tests added (test_coupling_adapter.py)
- [x] All tests pass (15 pass + 3 torch-skipped)