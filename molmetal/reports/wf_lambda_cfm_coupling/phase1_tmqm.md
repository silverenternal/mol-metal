# Phase 1 — tmqm_cfm_pretraining.py (CPU stand-in for the GPU CFM retrain)

**Workflow:** Lambda × CFM coupling (TODO-21 re-opened 2026-09-16)
**Phase:** 1 of 4 — pretraining script
**Status:** DONE
**Date:** 2026-09-16

## Goal

Create a CPU-only pretraining script that produces a tiny
``coupling_mlp.npz`` checkpoint the downstream
:meth:`coupling_adapter.embed_pocket` can load.  We **cannot**
use the real GPU CFM retrain right now (it is blocked on the
HSA init failure — see ``wf_gpu_auto_recover`` 2026-09-15).  The
Phase-1 stand-in produces a *deterministic* checkpoint whose
``embed_pocket`` API contract matches the spec from TODO-21 §3
so the wiring code in Phases 2/3/4 can be tested end-to-end
without GPU.

## What shipped

File: :file:`molmetal/scripts/tmqm_cfm_pretraining.py`

* 9-d input feature extractor
  (:func:`featurise_smiles_list`) — RDKit-free, deterministic.
* 2-layer MLP coupling head (ReLU + linear projection) with
  He-init embed layer and zero-init head (matches the spec where
  the default output for a fresh model is a constant).
* :func:`train_dry_run` — trains on the 8-mol
  :data:`DRY_RUN_SMILES` corpus, writes ``coupling_mlp.npz`` + a
  JSON side-car ``coupling_mlp.json`` with provenance metadata.
* :func:`try_load_tmqm` — graceful best-effort loader of the real
  tmQM corpus; returns ``None`` if unmounted.
* :func:`train_full` — same layout as ``train_dry_run`` but tries
  tmQM first; falls back transparently with a ``fallback_reason``
  recorded in the metadata.
* CLI with ``--dry-run`` flag and ``--print-final-loss``.

## Verification

```
$ python -m molmetal.scripts.tmqm_cfm_pretraining --dry-run --print-final-loss
[tmqm_cfm_pretraining] final_loss=0.3216
[tmqm_cfm_pretraining] saved to /home/hugo/.../molmetal/checkpoints/coupling/coupling_mlp.npz
```

* Wall-clock: < 1 s on this CPU.
* Output: ``coupling_mlp.npz`` (5 arrays, expected shapes) +
  ``coupling_mlp.json`` (n_train_mols=8, n_steps=1, dry_run=True).
* Final loss is finite (no NaN/inf), in [0, 1].

## Tests (Phase 1)

* :func:`test_tmqm_pretraining_dry_run_writes_checkpoint` —
  verifies the .npz + .json pair with expected keys + shapes.
* :func:`test_tmqm_pretraining_dry_run_cli` — CLI ``--dry-run``
  exit code 0.
* :func:`test_tmqm_pretraining_dry_run_loss_is_finite` — finite
  loss in [0, 1].
* :func:`test_tmqm_pretraining_full_falls_back_when_no_tmqm` —
  ``train_full`` gracefully handles missing tmQM.
* :func:`test_tmqm_pretraining_input_dim_constants_stable` —
  ``INPUT_DIM`` / ``EMBED_DIM`` agree between the script and
  :mod:`molmetal_lam.lam_chem.coupling_adapter`.

## Honest framing

This is **not** the real tmQM training run.  It is a CPU-only
stand-in that produces a deterministic, finite-value checkpoint
so the coupling wiring can be tested without GPU.  The real
training run needs:

1. GPU recovery (cold power cycle; see ``wf_gpu_diag_fix``).
2. The CFM P0/P1 fixes already shipped
   (:mod:`molmetal.adapters.flow_matching_lipman`).
3. 5k–10k steps on the full tmQM corpus.

This script will be replaced in-place by a tmQM-pretrained EGNN
init when those land (the checkpoint format is intentionally
``npz`` so we can load either the CPU stub or the real GPU
checkpoint with the same ``load_coupling_adapter`` factory).

## Deliverables checklist

- [x] NEW file ``molmetal/scripts/tmqm_cfm_pretraining.py``
- [x] CLI ``--dry-run`` flag
- [x] Writes ``.npz`` + ``.json`` with provenance metadata
- [x] 5 unit tests added (test_coupling_adapter.py)
- [x] Tests pass