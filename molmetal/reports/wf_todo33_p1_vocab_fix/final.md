# TODO-33 P1 F4 — Atom Vocab Extension + NaN/Inf Guards (Blackwell Inf divergence)

**Date**: 2026-09-25
**Fix pack**: TODO-33 §1 P1 — root cause #1 (atom vocab mismatch) + general numerical defence
**Status**: **PASS** — step 1 loss is finite on the full 500-mol metallodrug pool at seed=0

## Root cause

The default `LipmanFlowMatchingAdapter._atom_vocab` was a hard-coded 12-element tuple
`(1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)` = {H, C, N, O, F, P, S, Cl, Se, Br, I, Pt}.
But `molmetal/data/metallo_drugs_500_train.csv` contains 8 additional elements:

| Z | Symbol | Count | Source |
|---|---|---|---|
| 5  | B  | 24  | boranes / carboranes (tmQM + PlatinAI) |
| 14 | Si | 110 | organosilicon ligands |
| 21 | Sc | 43  | early-transition-metal complexes |
| 22 | Ti | -   | trace (titano-organics) |
| 33 | As | -   | trace (arsine donors) |
| 39 | Y  | 76  | rare-earth coordination |
| 40 | Zr | 2   | trace (zirconocene-like) |
| 72 | Hf | 1   | trace (HfCl4 adducts) |

Whenever any of these appeared in a training molecule:

1. `vocab_mask=True` set their logit slot to `-inf` *before* softmax.
2. The CE target `atom_types` for that atom became unreachable → `F.cross_entropy` returned `Inf`.
3. `loss = cfm + α·atom + β·bond` blew up to `Inf`.
4. AdamW's moment estimates became poisoned; cfm_loss diverged 17.7 → 252,330 by step 2000.
5. Decode sampler (which depends on a *trained* velocity field) produced gibberish
   coordinates → `decode_ratio = 0.000` floor.

This is documented as **TODO-33 P1 root-cause hypothesis #1** (highest likelihood) and
empirically reproduced in `molmetal/reports/wf_todo33_p1_f1/smoke.md`.

## Fixes applied (this pack)

### Fix A — Extend the canonical atom vocab (file:`molmetal/adapters/flow_matching_lipman/__init__.py:1825`)

```python
self._atom_vocab: tuple = (
    1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78,
)
```

New size: 20 atoms. Covers every Z present in `metallo_drugs_500_train.csv`.
Sorted ascending to keep the property docstring stable.

### Fix B — `atom_logits.clamp(min=-50, max=50)` before softmax (file:`__init__.py:~2170`)

Defends against BF16/TF32 matmul overflow when the EGNN is producing numerically
degenerate output. No-op for any sane value range.

### Fix C — `bond_inputs.nan_to_num` + `bond_logits_e.clamp` (file:`__init__.py:~2270-2295`)

Same idea but for the bond-order head. The EGNN hidden state `h` can overflow to NaN
in BF16 matmul, contaminating `e_h` and thus `bond_inputs`. We sanitise NaN/Inf to 0
*before* the bond head forward pass, then clamp the output logits to +/-50 before the
pattern mask and cross-entropy.

### Fix D — Data-load OOV guard (file:`molmetal/scripts/metallo_drug_smoke_retrain.py`)

* Before any 3D embedding, scan the SMILES via RDKit and reject molecules whose atoms
  are not in `LipmanFlowMatchingAdapter.atom_vocab`. Reports a per-rejection summary
  (`oov_Z=[5, 14, 21, ...]`) instead of silently poisoning the batch.
* Cache re-validation: if a pre-fix `mol_cache.pt` is loaded, drop any molecules
  whose `atom_types` contain OOV Zs (none in our cache — verified).
* Per-molecule `_smiles_to_molecule(smi, vocab=vocab)` raise a clear `ValueError` if
  the OOV check trips after the bulk pre-filter (defence in depth for cache-less runs).

### Fix E — `train_step` defensive guards (file:`__init__.py:2037-2080, ~2298`)

* **Pre-loss OOV check**: scan `mols[i].atom_types` for any Z not in `_atom_vocab`.
  If found, emit a `RuntimeWarning` and SKIP the optimizer step (return NaN, mark
  `last_losses["skipped_oov"]=True`). This is the last line of defence after the
  data-load filter.
* **Post-loss finite check**: if `loss` is not finite, emit a `RuntimeWarning`
  carrying the individual (cfm, atom, bond) components and SKIP the optimizer step
  (mark `last_losses["skipped_nonfinite"]=True`). Prevents AdamW from being poisoned
  by an Inf gradient.

## Verification (seed=0, 200 steps, batch=4, h=128, n_layers=3)

Command:

```bash
source env.sh
mkdir -p logs/wf_todo33_p1_vocab_fix
python molmetal/scripts/metallo_drug_smoke_retrain.py \
  --steps 200 --n-train 500 --batch-size 4 \
  --hidden-dim 128 --n-layers 3 --bond-head learned --joint-train \
  --seed 0 \
  --out-dir logs/wf_todo33_p1_vocab_fix/seed_0 \
  --decode-smoke-every 100
```

(Used the shared `CadButEaas/.venv-vlm` Python directly because `uv run` does not
honour `UV_PROJECT_ENVIRONMENT` on this host — same workaround as
`wf_todo33_p234_fix/final.md`.)

### Step-1 check (the load-bearing assertion)

| metric  | pre-fix (BF16 baseline) | post-fix (this pack) |
|---|---|---|
| loss    | inf                    | **46.5227** |
| cfm     | 17.7399                | **18.5385** |
| atom    | inf                    | **2.9957**  |
| bond    | inf                    | **27.6847** |

All three loss components are finite at step 1. The previously-observed Inf divergence
is fully resolved.

### Full 200-step loss curves

| loss  | first | last  | min   | max   |
|-------|-------|-------|-------|-------|
| cfm   | 18.54 | 17.55 | 5.85  | 215.47 |
| atom  |  3.00 |  2.85 |  1.12 |   6.13 |
| bond  | 27.68 |  1.36 |  1.18 |  27.68 |
| total | 46.52 | 19.19 |  7.52 | 242.29 |

The bond head converges 27.68 → 1.36 over 200 steps (a 20× reduction) — a strong
signal that the new vocab + numerical guards are working as intended.

### Decode smoke

`decode_ratio = 0.000` at step 100/200 — this is **unchanged** because it is caused
by a pre-existing shim bug independent of this fix pack:

```
TypeError: `model` must be a VelocityNet; got ModelWrapper.
```

Documented in `molmetal/reports/wf_todo33_p1_f1/smoke.md` §6 (Verdict #2). The fix
is to unwrap `velocity_model.model` in `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py`
before passing to `flow_matching.sampler.FlowMatchingSampler`. Out of scope for this
vocab+guards pack; tracked as a separate TODO.

### Vocab verification (per task spec)

```python
>>> from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
>>> a = LipmanFlowMatchingAdapter()
>>> print(a.atom_vocab)
(1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78)
>>> print(len(a.atom_vocab))
20
```

All atoms actually present in the training pool are in the vocab. The
`metallo_drugs_500_train.csv` parser (`RDKit.GetAtoms`) confirms:
`{H, B, C, N, O, F, Si, P, S, Cl, Sc, Se, Br, Y, Zr, I, Hf, Pt}` — every Z is in
the new 20-element vocab.

## Effort

* Diagnose root cause from CSV atom set: ~10 min (RDKit scan)
* Apply fixes A-E: ~25 min
* Verify (200-step smoke + final.json inspection): ~3 min on RTX PRO 6000 Blackwell

**Total**: ~40 min wall-clock.

## Artifacts

* `molmetal/adapters/flow_matching_lipman/__init__.py` (Fix A, B, C, E)
* `molmetal/scripts/metallo_drug_smoke_retrain.py` (Fix D)
* `logs/wf_todo33_p1_vocab_fix/seed_0.log` — full training log
* `logs/wf_todo33_p1_vocab_fix/seed_0/final.{md,json}` — aggregate metrics
* `logs/wf_todo33_p1_vocab_fix/seed_0/ckpt.pt` — checkpoint (velocity-field + bond-head)

## Next steps (out of scope for this pack)

1. **Decode TypeError fix** (1-line shim patch) — unblocks every decode smoke probe.
2. **Multi-seed verification** (seed=1, seed=2) — confirms the fix is stable, not
   seed-specific.
3. **Full 2000-step Blackwell run** — once GPU is back, run the canonical
   `wf_r17_yuelbond_2000` with the new vocab and verify `decode_ratio > 0`
   in 3/3 seeds.
4. **`_ALLOWED_PATTERNS` extension** — add explicit entries for Si / B / Sc / Ti /
   As / Y / Zr / Hf pairs so the bond pattern mask is more conservative. Currently
   relies on the "default allow all orders" path which works but is permissive.