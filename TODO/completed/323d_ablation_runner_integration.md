# TODO-323D: Wf-319A ablation_runner aggregate.csv per-strat integration

**Date**: 2026-09-29
**Owner**: Wf-323D
**Workflow**: Wf-323D (per-strat claim-card wire)
**Status**: SHIPPED (CPU-only metadata integration; no GPU launch)

---

## §1 Goal

Wire per-strat Wf-319A ablation_runner findings into the paper §6.1 / §5 dashboard surface:
3 NEW CLAIM_CATALOG entries in `paper_dashboard.py` + 3 NEW per_strat_subaxes under
`section_5_ablation.json::wf319a_ablation_runner_axis`. Complements the existing
`claim_wf319a_ablation_table_aggregate` (added Wf-319A) with the per-strat breakdown.

Per WEAK_TO_STRONG §3.6: M12 / M13 strong signals (atom_loss / bond_loss lift on the
smoke); M14 honest-negative (--lr-warmup-steps flag missing from harness).

---

## §2 Edits shipped

### §2.1 paper_dashboard.py — 3 NEW CLAIM_CATALOG entries

| Claim ID | Section | Badge | Headline |
|---|---|---|---|
| `claim_wf319a_m12_atom_loss_+204` | §6.1 | MEASURED | atom_loss 2.443 → 7.425, Δ=+4.982 / +204% (smoke-scale artifact caveat) |
| `claim_wf319a_m13_bond_loss_+152` | §6.1 | MEASURED | bond_loss 1.254 → 3.162, Δ=+1.908 / +152% (bond-head early-warning signal) |
| `claim_wf319a_m14_lr_warmup_honest_negative` | §6.1 | honest-negative | −7% within noise; HONEST_NEGATIVE: flag missing from harness, ablation silently broken |

Inserted after the existing `claim_wf319a_ablation_table_aggregate` block (line 1402-1409).
Each entry carries explicit MEASURED-vs-PROJECTED tag + json_path trace per
party-twenty-spirit §3.5.

**Total CLAIM_CATALOG count**: 88 → 91 (3 NEW).

### §2.2 section_5_ablation.json — 3 NEW per_strat_subaxes

Added `per_strat_subaxes` object inside `ablation_axes_with_data.wf319a_ablation_runner_axis`:

| Sub-axis | Metric | Δ / ζ |
|---|---|---|
| `wf323d_m12_atom_loss_axis` | atom_loss_last | 2.443 → 7.425 (Δ=+4.982 / +204%) |
| `wf323d_m13_bond_loss_axis` | bond_loss_last | 1.254 → 3.162 (Δ=+1.908 / +152%) |
| `wf323d_m14_lr_warmup_axis` | atom_loss_last | 2.443 → 2.280 (Δ=−0.163 / −6.7%, HONEST_NEGATIVE) |

Also updated `verdict` + `workflow` fields in section_5_ablation.json:
- verdict: + "3 Wf-323D per-strat sub-axes under wf319a_ablation_runner_axis.per_strat_subaxes"
- workflow: "Wf-311B + Wf-312C + Wf-319A" → "Wf-311B + Wf-312C + Wf-319A + Wf-323D"

---

## §3 Numerical verification (party-twenty-spirit §3.5 trace)

All numbers trace to `molmetal/reports/wf_ablation_runner/aggregate.csv` (12 rows):

| ID | atom_loss_last | bond_loss_last | vs BASELINE |
|---|---|---|---|
| BASELINE | 2.443439245223999 | 1.2540581226348877 | ref |
| M12 | 7.42507791519165 | 1.2570602893829346 | atom_loss +204% |
| M13 | 1.2316175699234009 | 3.1617648601531982 | bond_loss +152% |
| M14 | 2.280184507369995 | 1.254127860069275 | atom_loss −7% (within noise; flag missing) |

Δ verification:
- M12 atom_loss: (7.425 − 2.443) / 2.443 = 2.039 = +204% ✓
- M13 bond_loss: (3.162 − 1.254) / 1.254 = 1.521 = +152% ✓
- M14 atom_loss: (2.280 − 2.443) / 2.443 = −0.067 = −6.7% (within noise) ✓

---

## §4 Pre-flight gate verification (Wf-319A gates)

Per AGENTS.md §4 — pre-flight v2 25/25 PASS gate (TODO-207). For this metadata
integration task, no retrain/panel/ablation launch is performed, so the lighter
`--pipeline frontier` pipeline suffices. The 2 Wf-319A-specific gates are manually
confirmed:

| Gate | Status | Evidence |
|---|---|---|
| `p3_ablation_runner_aggregate_csv_present` | PASS | aggregate.csv has 12 data rows (≥11 required); CSV parseable; header matches schema |
| `p3_ablation_runner_tests_present` | PASS | `molmetal/tests/test_wf319a_ablation_runner.py` exists |

Pre-flight frontier pipeline (24/24 PASS, 0 FAIL, 0 SKIP, total_wall_sec=22.83):
JSON at /tmp/wf323d_preflight.json (summary: 24/0/0/24, verdict=PASS).

The heavier `--pipeline all` (chemprop_v2_v3 + cfm + frontier) was not re-run for
this metadata task — the previous Wf-323D-affecting gates (paper_dashboard +
section_5_ablation) are not retrain-trigger gates.

---

## §5 Honest framing (party-twenty-spirit §3.5)

Per Wf-319A final.md § "Honest framing":
- **200-step smoke is NOT production** — decode_ratio=1.0 in every toggleable is an
  artifact of the smoke-scale (n=8 fixed generation), not a meaningful signal.
- **M12 / M13 strongest at smoke scale** — the atom_loss / bond_loss lift is the
  cleanest evidence the §5 mechanism matters; production 10k would regress m14.
- **M14 honest-negative is dual**:
  1. final.md §M14: "At 200 steps the warmup has little effect — the LR is already
     small (1e-4) and the CFM loss landscape is smooth. The §5 prediction would
     manifest in a 10k-step production run." (smoke-under-power limitation)
  2. section_5_ablation.json finding: "--lr-warmup-steps 0 is NOT in the harness
     — ablation has been silently broken since the flag was removed" (flag-missing
     bug)

  Both honest-negatives surfaced; M14 cited as honest-negative in paper §6.1.

- **M14 fix path** (deferred to Wf-323A): either add --lr-warmup-steps back to the
  harness OR remove M14 from ablation_runner extra_args.

---

## §6 Files touched

| Path | Edit |
|---|---|
| `molmetal/scripts/paper_dashboard.py` | +3 CLAIM_CATALOG entries (claim_wf319a_m12_atom_loss_+204, claim_wf319a_m13_bond_loss_+152, claim_wf319a_m14_lr_warmup_honest_negative) |
| `metrics/by_paper_section/section_5_ablation.json` | +per_strat_subaxes (3 axes); verdict + workflow fields updated |
| `TODO/completed/323d_ablation_runner_integration.md` | this document |

---

## §7 Status

- §5 ablation axis count: 11 → 14 (5 Wf-268C + 4 Wf-311B + 2 Wf-312C + 1 Wf-319A
  + 3 Wf-323D per-strat sub-axes).
- §6.1 claim card count: 1 (claim_wf319a_ablation_table_aggregate) → 4 (above + 3 NEW).
- Verdict: PARTIAL_ABLATION (unchanged; production 100×3 still deferred).
- Workflow: Wf-311B + Wf-312C + Wf-319A + Wf-323D.
- Pre-flight (frontier): 24/24 PASS.

**SHIPPED** (CPU-only; no GPU launch per AGENTS.md §5.1 kill_then_adapt directive).
