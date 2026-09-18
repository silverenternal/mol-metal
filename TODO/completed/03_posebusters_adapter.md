# PoseBusters pass-rate port (posebusters_adapter.py)

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md
**Owner:** (unset)

## What was delivered
Adapter wrapping `posebusters` (BSD; modes: redock / dock / mol / gen) as the
project's `MoleculeValidator` port. Returns a structured `ValidityReport` with
per-check results (bond lengths within ±25% ideal, ring planarity ≤0.25 Å,
internal steric clashes, strain energy, pocket volume overlap <7.5%).
Falls back to an RDKit geometry check when the `posebusters` package is not
installed, so the adapter is always usable.

## Hard numbers
- Adapter file: 9,660 bytes (~265 lines); `PoseBustersAdapter` class + `ValidityReport`
- Used in production by L7/L8 governance checks: **SANITY=1.000, 24/24 EMBED_SUCCESS** on the 12-tile internal check
- **Negative result (real, not a bug):** PoseBusters pass-rate on 13 CuAAC/SPAAC products
  is **0/13** — root cause is the ETKDGv3+UFF 3D embedding does not reach the MMFF94
  minimum for 1,2,3-triazoles. Engineering fix tracked in pending/06_posebusters_fix.md
  (swap UFF for `MMFF94OptimizeMolecule`).

## Lessons learned
- The 0/13 CuAAC result is a real negative finding, not a port bug — it surfaces
  an embedding-quality problem in the pipeline that was previously hidden.
- PoseBusters serves two roles in this project: (a) headline validity metric for
  paper tables, (b) fast-path filter in MCTS expansion (see pending/decisions.md D5).

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md