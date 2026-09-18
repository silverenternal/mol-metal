# Cite-only SOTA comparison with protocol-mismatch flags

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md
**Owner:** (unset)

## What was delivered
A cite-only comparison table of SOTA SBDD methods against the project's
Lambda pipeline. Every numeric cell carries an explicit protocol-mismatch
flag (split, target family, hardware, sample-size) so a reader can tell at a
glance whether the comparison is apples-to-apples. The companion file
`lambda_vs_sbdd_protocol_aligned.md` lays out the alignment methodology.

## Hard numbers (lambda_vs_sbdd_paper_numbers.md)
- SOTA Vina scores cited (NOT measured — DiffSBDD/Pocket2Mol/TargetDiff ckpts
  unreachable from the sandbox; live head-to-head blocked on
  DiffSBDD/Torch_geometric/Torch_scatter ROCm wheels):
  - **Pocket2Mol** (Peng 2022, CrossDocked100, n=100): -7.07
  - **TargetDiff** (Guan 2023, CrossDocked100, n=100): -8.45
  - **DiffSBDD** (Qin 2024, CrossDocked100, n=100): -7.62
  - **DecompDiff**: -8.39
  - **FLOWR**: -6.93
  - **MolCRAFT**: -9.25
  - **AlphaDrug**: -9.77
  - **TransDiffSBDD**: -9.37
- **7 explicit protocol-mismatch flags** raised (cross-pocket, different n, different
  split, different engine — Vina 1.2.7 vs QVina, etc.)
- **Lambda Vina column = NOT REPORTED** (no docking in the comparison env at
  write time); companion h4_paper_grade_comparison.md tracks the measured Lambda
  numbers as they land
- 4 unit tests pass on the protocol-alignment logic

## Lessons learned
- "Cite-only" beats "reproduce-and-cite" when the reproduction budget would consume
  the entire compute grant; the trade-off must be explicit in the README.
- Mismatch flags compound: target-family + split differences inflate the apparent
  gap between methods by 2-3× in some cases.
- The QVina swap (pending/04) is the highest-leverage fix because it removes a
  systemic bias from Lambda's own numbers before any comparison is made.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_protocol_aligned.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h4_paper_grade_comparison.md