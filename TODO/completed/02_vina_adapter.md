# Real Vina docking port (vina_adapter.py)

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp13_vina_real.md
**Owner:** (unset)

## What was delivered
Real AutoDock Vina 1.2.7 docking port implementing the project's `DockingEngine`
port. Wraps the `vina` Python package + `meeko` for receptor/ligand prep
(steps 1-8 of the CCSB-Scripps pipeline). Replaces the prior random-fitness
proxy in `molmetal/molmetal_lam/search_alg/reinvent_wrapper.py:155` with
real physics-based binding-affinity scoring.

## Hard numbers (mmp13_vina_real.md)
- Target: MMP13 (collagenase-3), PDB **830c** (1.85 Å, sulphone hydroxamate RS1 co-crystal)
- Engine: AutoDockVina_v1.2.7; pocket atoms = 153; pocket radius = 10.0 Å;
  box padding = 8.0 Å; exhaustiveness = 8; n_poses = 5
- Co-crystal RS1 reference: best Vina score = **-1.285 kcal/mol**;
  success cutoff (ref + 1.0) = **-0.285 kcal/mol**
- 100 Lambda click products docked end-to-end (n_attempted = 100, n_valid_scores = 100, n_failed = 0)
- mean_score = **-1.355 kcal/mol**; median = -1.288; best = -1.820; worst = -1.130
- **success_rate = 100.0%** (within +1.0 kcal/mol of co-crystal RS1)
- Wall-clock: 260.4 s for the 100-product sweep
- Adapter file: 17,600 bytes (~450 lines)

## Lessons learned
- The `vina` + `meeko` Python packages avoid any native compilation; the port
  is a thin Python wrapper rather than a C++ binding.
- The smoke target (MMP13 PDB 830c) recovers 100% success within +1.0 kcal/mol
  of the co-crystal ligand, validating the receptor-prep + box parameters as
  the default config for follow-up closed-loop runs.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/vina_adapter.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp13_vina_real.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/reinvent_wrapper.py  (proxy being replaced)