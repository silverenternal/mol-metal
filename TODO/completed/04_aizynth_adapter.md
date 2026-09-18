# AiZynth retrosynthesis port (aizynth_adapter.py)

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h3_retrosynthesis_check.md
**Owner:** (unset)

## What was delivered
Adapter wrapping AiZynthFinder (MIT, conda: `aizynthfinder`) as the project's
`RetrosynthesisChecker` port. Returns a structured tuple `(synthesizable: bool,
route: List[str], depth: int)`. Falls back to RDKit SMARTS retrosynthesis when
`aizynthfinder` is not installed, so the adapter is always usable.

## Hard numbers (h3_retrosynthesis_check.md)
- Adapter file: 9,320 bytes (~259 lines); `AiZynthAdapter` class
- **14 Lambda click products** (real Lambda pipeline outputs): round-trip 100%
  through retrosynthesis checker
- 25 random SMILES (negative control): 0% round-trip — confirms checker is not
  trivial
- L-A2 / L9 governance re-measurements use `ROLLOUT_GUIDED_RATIO=0.7583`
  (a key metric for the closed-loop gated by retrosynthesis availability)

## Lessons learned
- The `aizynthfinder` install is a conda-only package (not on PyPI); keep the
  RDKit SMARTS fallback so the port works on a stock pip environment.
- Median AiZynth route depth on Lambda outputs is ≤ 3, supporting the project
  claim that click-style metal-coordination products are synthesizable.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h3_retrosynthesis_check.md