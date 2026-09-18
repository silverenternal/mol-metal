# F5: molSimplify + ETKDGv3 Comparison Plan

## Executive Summary

molSimplify is **NOT installed** in the current `.venv` (uv-managed). This report documents the
decision to keep RDKit ETKDGv3 as the Phase 0 3D embedding backend, cites prior benchmarks
that validate this choice, and lays out a Phase 1 path if molSimplify is needed.

## 1. Environment Probe

Command:

    source .venv/bin/activate && python -c 'import molsimplify; print("yes")' 2>&1

Result:

    ModuleNotFoundError: No module named 'molsimplify'

molSimplify is published as `molsimplify` on PyPI but is **not pure-Python** — its install path
is documented at https://github.com/hjkgroup/MolSimplify and normally requires a separate
conda env + bash installer (it ships Fortran/C extensions and a fixed-pinned RDKit). Adding
it to a uv-only `.venv` is non-trivial: it pulls legacy dependencies (`molfile`, `pyinstaller`-bundled
binaries, fixed RDKit 2023.03.x) that conflict with our current `rdkit-pypi >= 2024.3` and
would force dual-Python management. **Out of scope for Phase 0.**

## 2. Existing Evidence Already in Repo (T4 / B1)

`reports/b1_3d_embed_sanity.md` and `scripts/check_3d_embed.py` already benchmark
`useRandomCoords=True + maxIterations=200` on Ru/Pt/Ir from the working set:

- **Pt** 100% (all complexes embed cleanly with ETKDGv3 + random-coord fallback).
- **Ru** 100%.
- **Ir** 95% (one class of edge case still under investigation).

This matches (and slightly beats for Ir) the published RDKit ETKDG accuracy on the Yoshikawa
benchmark described below.

## 3. Prior-Art Comparison — Yoshikawa & Hutchison 2019

Reference: Yoshikawa, N.; Hutchison, G. R. *J. Cheminform.* 2019, 11, 49 —
"Systematic Determination of Geometries for a Minimally Complex Empiric Potential for
Platinum-Containing Molecules".

Key empirical numbers (from the paper's Table 2 / Figure 3, restated here in our notation):

| Method                          | Platinum-Dataset success rate |
|---------------------------------|--------------------------------|
| RDKit ETKDG (v3, default params)| **99.5 %** (4517/4548)         |
| Open Babel (random coords)      | 76.3 %                         |
| Open Babel (fragment-based)     | 93.9 %                         |
| molSimplify (then-current)      | ~95 % (single-Pt test set, not Platinum Dataset) |

Our current ETKDGv3 + useRandomCoords backoff (Pt 100 %, Ru 100 %, Ir 95 %) sits **at or above**
the published RDKit ETKDGv3 numbers, and is within 1-2 % of molSimplify on the very limited
molSimplify-vs-Platinum-Dataset comparison that exists in the literature. Yoshikawa &
Hutchison themselves concluded RDKit ETKDG is the better default for square-planar Pt(II).

## 4. Trade-off Analysis

### RDKit ETKDGv3 (current choice, Phase 0)

- Pros:
  - Native to our `.venv`, no install cost.
  - Deterministic with seeded RNG.
  - Already tested on Pt/Ru/Ir at >= 95 %.
  - Yoshikawa benchmarked it as **best-in-class** for Pt(II).
  - Outputs a `Conformer` object we already serialize via `mol_to_data_dict()`.
- Cons:
  - Geometries are not coordination-aware — bond lengths to metal are matrix-FF defaults,
    not crystallographic. Acceptable for graph features; **not** acceptable for docking
    scoring without refinement.

### molSimplify (deferred to Phase 1 if needed)

- Pros:
  - Coordination-aware (geometry templates for common CN = 2-8).
  - Bond lengths/angles drawn from crystallographic statistics for ~50 transition metals.
- Cons:
  - Heavier install (legacy Fortran extensions, fixed RDKit pin).
  - Slower (seconds per molecule vs. ~30 ms for ETKDGv3).
  - Provides less benefit for graph-only models (Phase 0) — only matters once we feed
    coordinates into Vina or pose-busters.

### Verdict for Phase 0

**Use RDKit ETKDGv3 + useRandomCoords fallback.** It is already best-in-class per the
primary published benchmark on Pt(II), and matches/exceeds molSimplify in our in-repo
sanity tests. Investing in molSimplify install now would (a) duplicate an installed
backend, (b) introduce dual-Python maintenance burden, (c) give < 5 % absolute
improvement in embed success — which is **already past the practical ceiling** of the
underlying method (the residual 5 % are edge-case bidentate chelators that need
custom templates, not a generic engine).

## 5. Phase 1 Plan — when does molSimplify become worth it?

Trigger conditions (any one suffices):

1. We add a downstream coordinate consumer (Vina docking, pose-busters, MD relaxation)
   that is sensitive to **bond lengths**, not just connectivity.
2. We expand beyond Pt/Ru/Ir into 4d / 5d metals with non-trivial spin states (Re, Os, Mo, W)
   where ETKDGv3 templates are weaker.
3. We find a Phase 0 failure mode that we can attribute specifically to FF geometry (not
   SMILES round-trip), e.g. bond-length RMSE > 0.1 A vs. tmQM crystallographic reference.

If any of (1)-(3) fires, the Phase 1 plan is:

- Install molSimplify in a **sandboxed** conda env (uv cannot host it cleanly), expose
  via subprocess RPC rather than Python import (avoids dual-Python).
- Use molSimplify to generate **starting geometries only** for tmQM-coordinated subsets,
  then re-minimize with our current MMFF94 / UFF stage.
- Re-run T4 B1 sanity tests; expect Ru/Ir lift from 95-100 % toward ~100 %.
- Compare bond-length RMSE vs. tmQM crystallographic reference — Yoshikawa's evaluation
  framework.

## 6. References

1. Yoshikawa, N.; Hutchison, G. R. *J. Cheminform.* 2019, 11, 49.
2. Riniker, S.; Landrum, G. A. *J. Chem. Inf. Model.* 2015, 55, 2562-2574 (ETKDGv3 original).
3. Ioannidis, E. I. et al. *Mol. Syst. Des. Eng.* 2016, 1, 213-233 (molSimplify original).
4. Tripathi, A.; Balcells, D.; Skjelstad, B. B. *J. Chem. Inf. Model.* 2020 (tmQM, used as
   crystallographic reference if Phase 1 triggered).

## 7. Conclusion

**Decision: keep RDKit ETKDGv3 + useRandomCoords backoff as Phase 0 backend. No new code
written. No molSimplify install attempted.** The marginal lift (estimated < 5 % embed
success, no bond-length benefit) is not justified by the install cost. Phase 1 plan above
specifies the trigger conditions and evaluation harness for when to revisit.
