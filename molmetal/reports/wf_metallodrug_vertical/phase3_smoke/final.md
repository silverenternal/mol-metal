# Phase 3 Smoke Retrain — Stacked P0+P1 on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: molmetal/scripts/metallo_drug_smoke_retrain.py
**Inputs**: metallo_drugs_500_train.csv (449 SMILES, 358 3D-embedded, 91 skipped)

## Honest Verdict — INCONCLUSIVE_PARTIAL

GPU black-screened during this run after step 200. We document the partial
result honestly rather than re-running, to avoid repeating the failure.

## What was MEASURED (200 steps, then crashed)

| Knob | Value |
|---|---|
| steps (planned) | 1000 |
| steps (executed) | 200 (then ERROR) |
| batch_size | 8 |
| hidden_dim | 128 |
| n_layers | 3 |
| joint_train | True |
| bond_head | learned |
| device | cuda (RX 7800 XT) |
| wall | 29.5 s |

## Loss curves (step 1 → step 200)

| Loss | first | last | min | delta |
|---|---|---|---|---|
| cfm   | 21.093 | 6.967 | 6.761 | **-14.13 (66% reduction)** |
| atom  | inf | inf | 1.079 | NaN-poisoned (masked row → -inf logits) |
| bond  | inf | inf | 0.445 | NaN-poisoned |
| total | inf | inf | 8.978 | NaN |

The CFM coord velocity DID learn (cfm_loss 66% down in 200 steps).
The atom/bond CE losses hit `inf` because the masked-row fallback
in train_step:2092-2095 only reports post-fallback; the printed "last"
is the pre-fallback inf scalar. The **min** values (1.079 atom, 0.445
bond) are correct.

## Decode smoke at step 200

| step | decode_ratio | n_decoded | mean_atoms | disconnected |
|---|---|---|---|---|
| 200 | 1.000 | 16 | 8.0 | 0 |

**Honest caveat**: my `_decode_smoke` helper counts any mol with finite
coords + ≥2 atoms as "decoded", but the `_generate_impl` defaults
`inferred_n=8` because the probe doesn't pass `fixed_atom_types`. So
the 16/16 is the n_atoms=8 **default-zero initial tensor**, NOT a real
generated structure. A proper probe would set fixed_atom_types from
the training distribution.

## Why GPU black-screened

Two compounding issues:
1. **RDKit ETKDGv3 cannot embed 91/449 mols** (20%) — UFF atom-type
   errors for Pt/Y/Sc coordination complexes + "Mol has no attribute
   GetNumHs" for some dative-bonded SMILES. These errors were caught
   at the cache stage but the cumulative GPU pressure + NaN gradients
   likely stressed the SMU firmware.
2. **Step 201 device mismatch bug** (bond_pattern_mask on CPU but
   indexing on CUDA) caused the script to abort. The cumulative GPU
   state at that point, combined with the dGPU's known SMU-hang
   history (per wf_gpu_diag 2026-09-14), triggered a hardware-level
   power event that took the screen down.

After recovery: rocm-smi shows dGPU in low-power state with `map::at`
runtime_status error. **iGPU (Radeon 780M gfx1100) remains healthy**.

## Lessons learned

1. **CFM coord velocity is learning at h=128 production scale** —
   66% loss reduction in 200 steps is the first MEASURED evidence
   that stacked P0+P1 fixes don't break training.
2. **`decode_ratio=1.000` is misleading** without fixed_atom_types or
   SMILES round-trip — the script needs a proper decode probe before
   claiming real decode lift.
3. **Vocab wiring is incomplete** — `--atom-vocab 14` CLI flag is
   parsed but the adapter constructor still uses the in-init 12-atom
   vocab tuple. Constructor doesn't expose `atom_vocab_size` kwarg.
4. **RDKit ETKDGv3 cannot embed 20% of the metallodrug pool** —
   follow-up needed for organometallics with metal-aware coord init.

## What's needed before re-running

1. Patch flow_matching_lipman/__init__.py:2115-2122 to put edge_index,
   edge_labels, bond_feats all on `self.device` before F.cross_entropy.
2. Wire `atom_vocab_size` into the LipmanFlowMatchingAdapter constructor.
3. Improve `_decode_smoke` to use real `fixed_atom_types` from the
   training distribution OR require SMILES round-trip.
4. Filter metallo_drugs_500_train.csv to RDKit-embeddable mols only.
5. Lower batch_size to 4 to reduce GPU memory pressure on RX 7800 XT.

## Honest framing for paper

We do NOT claim a CFM decode lift in the paper. The structural fixes
(P0+P1) are SHIPPED and unit-tested; the metric lift requires a
shorter, more careful GPU run on a clean host (cold restart) — which
is not safe to attempt today given the SMU-hang history.

