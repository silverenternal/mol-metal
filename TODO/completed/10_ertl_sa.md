# Ertl SA via RDKit Contrib sascorer.py

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h1_sa_score_ertl.md
**Owner:** (unset)

## What was delivered
Integrated Ertl & Schuffenhauer's synthetic-accessibility (SA) score by
vendoring the RDKit Contrib `sascorer.py` module and wrapping it in the
project's reward interface. Used as the paper-grade SA axis for Lambda
"click chemistry is synthesizable" claims — not a proxy.

## Hard numbers (h1_sa_score_ertl.md)
- **Mean SA on 12 click tiles = 2.93** (range 1.00–4.74)
- Sits inside the "drug-like synthesizable" band of Ertl 2009 Table 1
- Uses RDKit's Contrib `sascorer.py` **unmodified** — no re-implementation
- Adapter file: `molmetal/molmetal_lam/sbdd_env/sa_score.py`

## Lessons learned
- "Stop reinventing" — the Ertl SA score has a stable reference implementation
  in RDKit Contrib; the only project-specific code is the wrapper that puts
  it behind the project's reward interface.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h1_sa_score_ertl.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/sa_score.py