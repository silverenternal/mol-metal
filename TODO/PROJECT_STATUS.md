# PROJECT_STATUS — pointer doc (compressed 2026-10-05)

**Status**: COMPRESSED to pointer doc (was 41,677 B, now < 4 KB).
**Source-of-truth for current project state**: see `TODO/INDEX.md` §0 (recent Wf timeline) + §22 (full table) + `AGENTS.md` §6 (Wf history pointer) + `docs/STATUS.md` (status roll-up).

Per AGENTS.md §5.10 (OmegaFold gap-fill) + §5.2 (honest framing), this file is a **pointer** to the canonical docs — not a duplicate of them. Every claim must originate in a JSON or report path; nothing is restated here.

## Six user questions — canonical pointers

| # | User question | Canonical doc |
|---|---|---|
| **Q1** | Which baseline do we use? | `TODO/03_baselines/COMPARISON_BASELINES.md` (Krasnov 2026 / Rusanov 2025 / DrugOOD 2022; ALIGNMENT-not-reproduction principle) |
| **Q2** | Lambda modules + baselines | `TODO/03_plans/LAMBDA_ROADMAP.md` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (M1-M44) |
| **Q3** | Click chemistry + comparison experiments | `TODO/13_lambda_clickchem/plan.md` + `MECHANISM_METRIC_MAPPING.md` §5-§6 |
| **Q4** | Model architecture vs vanilla FM | `docs/ARCHITECTURE.md` + `mol-metal-cfm-training` skill |
| **Q5** | 21 metrics — current state | `TODO/02_metrics/METRICS_TARGETS.md` (per-metric target + verdict gate + status) |
| **Q6** | Training framework + Triton status | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 + §7 + `TODO/03_plans/PRE_TRAINING_GATE_v2.md` |

## D-MPNN line

- Canonical closeout: `TODO/completed/dmpnn_line_overview.md` (2026-10-05, 4 KB — NEW ultracode consolidation)
- Defects identified: 4 trainer-level defects + 2 model-architecture defects (see overview §"Wf-ultracode 2026-10-05 fixes")
- Tests: `molmetal/tests/test_dmpnn_ultracode_fixes.py` (4 defects) + `test_dmpnn_metal_lines_cleanups.py` (2 arch defects) + `test_dmpnn_sha256_audit_scope.py` (audit scope)
- SHA256 audit script: `molmetal/scripts/dmpnn_sha256_audit.py` (CPU-only, 280 LOC)

## What NOT to look for here

- Detailed metric-by-metric state → see `TODO/02_metrics/METRICS_TARGETS.md`
- Wave timeline → see `docs/WF_HISTORY.md` + `TODO/INDEX.md` §22
- Honest-negatives → see `docs/HONEST_NEGATIVE_CATALOG.md`
- Decision log → see `TODO/decisions.md`
- Pre-training protocol gates → see `TODO/03_plans/PRE_TRAINING_GATE_v2.md` (canonical since Oct 4; PROTOCOL_VALIDATION.md / MOCK_TO_REAL_PLAN.md / HONEST_NEGATIVE_CLOSURE_PLAN.md were Oct 2 wave-staging planning docs superseded by the canonical pair)

## Status banner (compressed)

- **D-MPNN line**: 7/7 (metal, seed) pair aggregate MEASURED; 6 confirmed defects fixed; canonical closeout SHIPPED.
- **Tier-1 LEAD metrics**: m21 0.8007±0.0160 PASS + m17 0.7479 PASS + m1 0.7717±0.0380 cohort honest-negative + m3 -7.782 kcal/mol PASS + m15 0.6578 Å PASS + m14 1.000 PERFECT. See `metrics/by_metric/*.json`.
- **Wet-lab**: DEPRECATED per AGENTS.md §5.4 + §10 (Tier-2 paper target).