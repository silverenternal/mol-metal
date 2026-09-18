# Audit: targetdiff (Guan 2023, ICLR)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/`
Our hand-rolled code: `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py` (2248 LOC)

## (1) What we already have hand-rolled that is redundant

The `flow_matching_lipman/__init__.py` module re-implements several
TargetDiff components from scratch:

- **EGNN layer** (we have `egnn_rocm.py` ~620 LOC + the VelocityNet uses an
  inline EGNN at lines ~1400–1900); targetdiff ships `models/egnn.py`
  (canonical EGNN, Satorras+ 2021) with `EquivariantBlock`, `EquivariantMPNN`,
  and a separate `models/uni_transformer.py` for attention-based equivariant
  message passing.
- **OT-CFM path** (we hand-roll `interpolate`, `velocity_target`,
  `lipman_loss`); targetdiff implements the same SE(3)-decomposed flow with
  `t_to_sigma` / `sigma_from_t` in `utils/diffusion_utils.py` and uses the
  standard Lipman conditional OT path.
- **Score model wrapper** (we have `molopt_score_model.py`-style logic
  embedded in our velocity net); targetdiff has `models/molopt_score_model.py`
  which is the same wrapped-EGNN pattern.
- **Sampling loop** (we use Euler integration with our own t-schedule);
  targetdiff uses `utils/sampling.py::sampling()` with 20–100 reverse steps
  and known-good hyperparameters.

## (2) What is actually different / better in our hand-rolled code

- **Bond-order head**: targetdiff has no concept of per-edge bond type
  prediction; we ship `BondOrderHead` (joint atom-type + bond-tensor flow).
  This is *the* thesis of path-(b) — keeping it.
- **Pocket-conditioned reference ligand resolver**: targetdiff conditions on
  raw pocket residues; we use a typed `reference_ligand_resolver` that
  extracts scaffold + metal seed → 3D coord anchor. Different prior.
- **MetalGeometryPrior + AquaExchange/MetalLigandExchange rules**: targetdiff
  has no notion of Pt(II)/Pt(IV) coordination; the closure-theorem
  (`closure.py`) and click-rule SMARTS are unique to Mol-Metal.
- **CFM mid-epoch decoder smoke**: targetdiff's sampling is post-hoc only;
  our `decode_smoke_every` step counter is a debugging affordance.
- **Cross-platform ROCm**: targetdiff assumes CUDA 11.6 + PyG 2.2 (Python
  3.8); we run on ROCm gfx1101 with custom Triton kernels and torch_scatter
  fallbacks. Direct import would break our GPU path.
- **Pretrained weights**: targetdiff checkpoints are CUDA / DataParallel;
  loading them requires the same `MolOptScoreModel` architecture (1024-dim,
  9 layers) — we deliberately run at 128-dim / 6 layers for RX 7800 XT.

## (3) Concrete 3-line patch plan (file + lines + import)

```
# DO NOT replace flow_matching_lipman/__init__.py wholesale.
# INSTEAD, cherry-pick the canonical EGNN:
from references.targetdiff.models.egnn import EquivariantBlock, EquivariantMPNN
# and swap into egnn_rocm.py:84-86 (3 lines)
```
Then add a thin `targetdiff_compat.py` adapter that re-exports
`EquivariantBlock` under our API (`EgnnLayer`) so `velocity_net.py`
callsites stay unchanged. **Do NOT** import targetdiff's
`molopt_score_model.py` or `uni_transformer.py` — they have
targetdiff-specific dataset loaders baked in.

## Verdict

**PARTIAL REUSE.** Import `EquivariantBlock` / `EquivariantMPNN` from
targetdiff's `models/egnn.py` for a canonical EGNN reference, but keep
our `egnn_rocm.py` + `velocity_net.py` + `BondOrderHead` + pocket/click
machinery. Replacement of the full module would lose our novelty (bond
joint, metal prior, closure) and reintroduce CUDA-only dependencies.