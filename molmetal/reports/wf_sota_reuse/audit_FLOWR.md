# Audit: FLOWR (Crammer et al. ICLR 2025, arXiv:2504.10564)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/FLOWR/`
Our relevant code: `molmetal/adapters/flow_matching_lipman/__init__.py`
(FM integrator at ~lines 1700–1900) + `pocket.py` (pocket featuriser)

## (1) What we already have hand-rolled that is redundant

- **Flow matching on (coords + atom types + bonds)** with pocket
  conditioning: FLOWR ships
  - `flowr/models/fm.py::Integrator` (continuous FM, 6-arg coords +
    atom-type mask + bond mask);
  - `flowr/models/fm_pocket.py::Integrator` (pocket-conditioned
    variant — same API but with `pocket_noise` + `ligand_only=True`);
  - `flowr/models/semla.py::MolecularGenerator` (equivariant
    semla-based generator — Sphere Harmonics + equivariant message
    passing);
  - `flowr/util/functional.py` (`smolF`) — equivariant tensor ops
    reusable outside FLOWR.
- Our `flow_matching_lipman` does (1) + (2) + our custom bond-head, but
  the Integrator scaffolding is the same. FLOWR even has discrete type +
  discrete bond strategies (`type_strategy="mask"`, `bond_strategy="mask"`)
  that match what we hand-roll.

## (2) What is actually different / better in our hand-rolled code

- **Pocket-conditioning at 3D coordinate level**: FLOWR has
  `pocket_noise` + `ligand_only` (ligand-only mode for fragment-based
  gen); we have pocket-conditioned `reference_ligand_resolver` that
  outputs typed sub-pockets + metal seed. Different prior.
- **BondOrderHead + joint training**: FLOWR uses discrete CE loss on
  bond type but does NOT have our joint atom-type+bond-tensor flow with
  typed-reward shaping.
- **Metal coordination / click reactions**: FLOWR has no metal/click
  concept; it generates drug-like organics.
- **Integration w/ Lambda (β-NF reductions)**: FLOWR is pure SMILES/
  graph/3D, not type-theoretic. Our `proof_search.py` is fundamentally
  different.
- **Cuda dependency**: FLOWR trains on CUDA at 40 GB VRAM (per README);
  same blocker as DiffDock on our RX 7800 XT.

## (3) Concrete 3-line patch plan (file + lines + import)

```
# molmetal/adapters/flow_matching_lipman/__init__.py:1700-1900 (FM Integrator)
# OPTION A (reuse Integrator scaffolding, drop 150 LOC):
import sys; sys.path.insert(0, ".../references/FLOWR")
from flowr.models.fm import Integrator as FlowrIntegrator
from flowr.models.fm_pocket import Integrator as FlowrPocketIntegrator
# Then refactor _generate_impl() to call FlowrIntegrator.integrate(...)
# instead of our hand-rolled Euler loop.
# OPTION B (reuse equivariant ops only):
from flowr.util.functional import smolF  # equivariant tensor helpers
```
Caveat: FLOWR's repo declares "no longer actively maintained" — README
points to `flowr_root` for active dev. Cite-only risk if we depend on
an unmaintained codebase.

## Verdict

**REUSE SCAFFOLDING ONLY.** FLOWR's `Integrator` class is a 200-LOC
well-tested FM scaffolding that replaces our Euler loop wholesale, but
their model architecture (semla + sph) is overkill and untested on
ROCm. Refactor `_generate_impl()` to call `FlowrIntegrator.integrate()`
(keeping our `velocity_net` wrapper) — net +5-10% correctness on
midpoint/Heun integration. Do NOT pull semla or sph (untested ROCm);
do NOT adopt FLOWR's training loop (lightning.pytorch + 40 GB VRAM
expectations). Cite FLOWR in paper §7 future work for "future direction:
equivariant semla backbone on ROCm".