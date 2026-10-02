# TODO-320E — Wf-320E downstream metric lift dashboard

**Date**: 2026-09-29 (Wf-320E close-out per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIPPED (dashboard JSON + 6 paper claim cards + 12 tests + pre-flight CFM 6/6 PASS)
**Parent**: `TODO/completed/177_wf123_p3_m1m2_downstream_metric_lift.md` §3 (per-consumer spawn plan)
**Wf tag**: Wf-320E

---

## §1 Deliverable summary

| Deliverable | Path | Status |
|---|---|---|
| Unified downstream lift dashboard JSON | `molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json` | ✅ WRITTEN |
| 6 NEW paper claim cards in `paper_dashboard.py` | `molmetal/scripts/paper_dashboard.py` (lines after `claim_wf319g_*`) | ✅ ADDED |
| 4 NEW HONEST_NEGATIVE catalog entries in `paper_dashboard.py` | `molmetal/scripts/paper_dashboard.py` (after `hn_wf318g_*`) | ✅ ADDED |
| 12 NEW dashboard tests | `molmetal/tests/test_wf320e_downstream_lift_dashboard.py` | ✅ 12/12 PASS |
| Pre-flight v2 CFM pipeline | `molmetal/scripts/pre_flight_check.py --pipeline cfm` | ✅ 6/6 PASS |
| Documentation | `TODO/completed/320e_downstream_lift_dashboard.md` (this file) | ✅ WRITTEN |

---

## §2 Per-consumer lift table (verbatim from `dashboard.json`)

| TODO | Consumer metric | Baseline | M1+M2-enabled | Lift Δ% | Status | Badge |
|---|---|---:|---:|---:|---|---|
| TODO-119 | m14 cfm_decode_ratio | 0.9645 (Wf-274A 1-seed CPU PlatinAI) | 0.9980 (Wf-148 3-seed GPU NCI-60+DrugOOD) | +3.47% | PARTIAL | HONEST_NEGATIVE |
| TODO-120 | m1 Ru AUC | 0.7123 (chemprop v2 single-task) | 0.7860 (Wf-310A G5 50-ep n=2/3 seeds) | +10.35% | MEASURED | MEASURED |
| TODO-121 | m19 LOMO Rh bottleneck | 0.5 (BELOW_RANDOM baseline) | 0.34375 (Wf-176 Re fold seed=4 2-ep W₂=0) | −31.25% | PARTIAL | HONEST_NEGATIVE |
| TODO-122 | L5 Hilbert R² ≥ 0.85 | 0.0 (architectural signal-floor) | −5.6625 (Wf-122 P3 post-fix 4-cell) | n/a (signal-floor) | MEASURED | HONEST_NEGATIVE |
| TODO-123 | m15 RMSD docked pose | 1.5 Å (legacy Kabsch+Hungarian) | 0.6578 Å (Wf-246-B 5-pocket median, 5/5 < 2.0 Å) | −56.15% | MEASURED | MEASURED |
| TODO-125 | L12 OOD per-pocket Vina | 0.49 (5-pocket cross-pocket mean) | 0.0702 (Wf-144 P1 ex=32 pooled, 1/5 pockets) | −85.67% | PARTIAL | HONEST_NEGATIVE |

**Categorical counts**:
- n_consumers_tracked = 6
- n_me = 3 (TODO-120, TODO-122, TODO-123 status=MEASURED)
- n_partial = 3 (TODO-119, TODO-121, TODO-125 status=PARTIAL)
- n_projected = 0
- n_honest_negative = 0
- n_me_badge = 2 (TODO-120, TODO-123 badge=MEASURED)
- n_honest_negative_badge = 4 (TODO-119, TODO-121, TODO-122, TODO-125 badge=HONEST_NEGATIVE)
- n_partial_recovery = 1 (TODO-119 partial_recovery via Wf-148 superseding Wf-274A)

---

## §3 Honest framing per `party-twenty-spirit` §3.5

### §3.1 What this dashboard IS

- Aggregation of 6 strengthen-TODO consumer lifts that consume M1 (Wasserstein
  regularizer) + M2 (Catherine closure-bounded). All 6 baseline AND M1+M2-enabled
  values trace to on-disk JSON / report paths (every consumer carries
  `baseline_source` + `m1m2_source` keys).
- Per party-twenty-spirit §3.5: MEASURED vs PROJECTED labels are mandatory.
  Every consumer row carries `status` (MEASURED / PARTIAL / PROJECTED / HONEST_
  NEGATIVE) and `badge` (MEASURED / HONEST_NEGATIVE) tags.
- HONEST_NEGATIVE rows are NOT silently masked — each carries an
  `honest_framing_note` pointing at the on-disk source and explaining the
  negative result.
- Lift values use sign convention: positive = improvement; for RMSD / Vina-
  receptor-error metrics, NEGATIVE = IMPROVEMENT (lower is better).

### §3.2 What this dashboard IS NOT

- **NOT new measurement**. No new training, no new docking, no new featurization.
  All values are computed from existing JSON / report paths.
- **NOT a positive lift for all consumers**. 4/6 consumers are HONEST_NEGATIVE:
  TODO-119 (methodology change), TODO-121 (BELOW_RANDOM), TODO-122 (arch
  ceiling), TODO-125 (1/5 pockets pass).
- **NOT a project-status update**. Per-consumer lift is a per-TODO deliverable
  (Q4 2026 / R20 GPU exercise). The dashboard surfaces current state, not future
  targets.

### §3.3 Per-consumer honest-framing notes (verbatim from `dashboard.json`)

- **TODO-119**: "The Wf-274A regression_disclosure flags cfm_decode_real.json
  0.9645 as a SMALLER-cohort 1-seed × 818 CPU PlatinAI pt_subset measurement;
  per party-twenty-spirit §3.5 do NOT cite 0.9645 as LEAD — cite Wf-188/Wf-148.
  Therefore 'lift' of +0.0335 vs 0.9645 baseline is a methodology change (1-seed
  CPU vs 3-seed GPU 5991), NOT a real M1+M2 lift."

- **TODO-120**: "MEASURED lift of +0.0737 vs chemprop v2 baseline on Ru seed=0+1
  overlap (n=2/3 seeds). Krasnov 2026 baseline 0.81 PROJECTED gate still BELOW
  by -0.024; honest entry is partial-closure of the Krasnov gap."

- **TODO-121**: "W₂ weight was 0.0 (line 28 of metric file); W₂ was effectively
  disabled for this run. The architectural pattern (oversample Rh + Wasserstein
  transfer loss) is wired but the single 2-epoch 1-fold 1-seed run does NOT
  recover the BELOW_RANDOM gap."

- **TODO-122**: "ARCHITECTURAL_CEILING: 12-d RDKit signal |ρ| ≤ 0.019 on
  PlatinAI pIC50; linear R² upper bound = (0.019)² · 12 = 0.0043, hence R² ≥ 0
  mathematically infeasible on this feature stack. Hilbert L2 + ASTSR scale-guard
  prevents divergence but does NOT lift the architectural signal floor."

- **TODO-123**: "MEASURED docked_pose_rmsd 0.6578 Å cross-pocket median (3-seed ×
  5-pocket × 5 poses; Wf-246-B redesign with per-pocket std + difficulty flag).
  5/5 pockets pass the 2.0 Å SOTA gate (Trott & Olson 2010 Vina paper convention)."

- **TODO-125**: "Wf-144 P1 ex=32 lift confirms 1h36 PASSES (4/13 hits, 0.3077) but
  4/5 pockets honest-negative. Wf-140 closure-bounded cohort is SA-optimized for
  synth, NOT for strong pocket binding. per_pocket_vina.csv (Wf-248C) shows
  STUB_FALLBACK for all 75 rows — the per-pocket Wasserstein + closure_bias FULL
  pipeline is NOT executed end-to-end on the 5-pocket cohort."

---

## §4 Test coverage (12 NEW tests in `test_wf320e_downstream_lift_dashboard.py`)

| Test | Purpose |
|---|---|
| `test_todo119_m14_decode_ratio_consumer_schema` | Schema validation: TODO-119 |
| `test_todo120_m1_ru_auc_consumer_schema` | Schema validation: TODO-120 (MEASURED + Krasnov gap) |
| `test_todo121_m19_lomo_rh_consumer_schema` | Schema validation: TODO-121 (BELOW_RANDOM) |
| `test_todo122_l5_hilbert_r2_consumer_schema` | Schema validation: TODO-122 (signal-floor) |
| `test_todo123_m15_rmsd_consumer_schema` | Schema validation: TODO-123 (5/5 < 2.0 Å) |
| `test_todo125_l12_per_pocket_vina_consumer_schema` | Schema validation: TODO-125 (STUB_FALLBACK tag) |
| `test_lift_abs_calculation_matches_formula` | Lift arithmetic: `lift_abs = m1m2 - baseline` |
| `test_lift_pct_calculation_matches_formula` | Lift arithmetic: `lift_pct = (m1m2 - baseline) / baseline * 100`; signal-floor handling |
| `test_lift_sign_convention_for_lower_is_better_metrics` | RMSD: NEGATIVE lift = IMPROVEMENT |
| `test_status_classification_rules_are_explicit` | All 4 statuses documented with rules |
| `test_status_counts_match_n_consumers_tracked` | Categorical counts (status vs badge) must sum to 6 |
| `test_dashboard_schema_validation` | Top-level schema fields, TODO ids, summary table size |

Run output:
```
12 passed, 1 warning in 0.43s
```

---

## §5 Pre-flight v2 status

`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py --pipeline cfm`:
- summary.pass = 6
- summary.fail = 0
- summary.skip = 0
- summary.total = 6
- All 6 cfm gates PASS (script_exists, config_exists, decoder_exists,
  wasserstein_loss_exists, smoke_decoded, catherine_30metals).

(Note: Wf-320E is aggregation-only; the pre-flight CFM pipeline subset is
sufficient because no new measurement is introduced. The chemprop_v2_v3 +
frontier pipelines would block on GPU/CPU resource exhaustion in this session
window per `mol-metal-gpu-execution` §0.)

---

## §6 Paper claim cards added to `paper_dashboard.py`

6 NEW claim cards (verbatim from the script):
- `claim_wf320e_todo119_m14_lift_partial` (§4.5.6, HONEST_NEGATIVE)
- `claim_wf320e_todo120_m1_ru_lift_measured` (§2.1, MEASURED)
- `claim_wf320e_todo121_m19_lomo_rh_honest_negative` (§6.4, HONEST_NEGATIVE)
- `claim_wf320e_todo122_l5_hilbert_r2_arch_ceiling` (§6.5, HONEST_NEGATIVE)
- `claim_wf320e_todo123_m15_rmsd_measured` (§4.6, MEASURED)
- `claim_wf320e_todo125_l12_per_pocket_vina_partial_honest_negative` (§6.6, HONEST_NEGATIVE)

4 NEW HONEST_NEGATIVE catalog entries:
- `hn_wf320e_todo119_m14_0_9645_not_lead` (§9.BB)
- `hn_wf320e_todo121_m19_lomo_below_random` (§9.BC)
- `hn_wf320e_todo122_l5_arch_ceiling` (§9.BD)
- `hn_wf320e_todo125_l12_per_pocket_4_of_5_fail` (§9.BE)

---

## §7 Follow-up Q4 2026 / R20 GPU launches

Per `mol-metal-gpu-execution` §0 "kill then adapt" directive, all 4 GPU launches
require **explicit fresh user OK** before any launch > 1 hr wall:

| TODO | Wf follow-up | Wall (hr) | Hardware | Blocker |
|---|---|---:|---|---|
| TODO-119 | Wf-124 P4 / Wf-128 orphan | 6.0 | RTX 5090 + Blackwell | user OK |
| TODO-121 | Wf-125 P2 | 3.0 | RTX 5090 | user OK |
| TODO-122 | Wf-94 R20-P3 | 4.0 | Blackwell | user OK |
| TODO-125 | Wf-130 P3 | 6.0 | Blackwell | user OK + TODO-29 F2(a) GPU unblock |
| **Total** | | **19.0** | | |

(The 4th launcher TODO-120 G6 7.5hr + TODO-181 G5 5hr are tracked separately per
the standing "kill then adapt" directive; together with the Wf-320E follow-ups
the 2027 GPU backlog is ~37.8 hr total per Wf-258 §2.1.)

---

## §8 Cross-references

- `TODO/completed/177_wf123_p3_m1m2_downstream_metric_lift.md` §1 + §2 (current
  MEASURED state) — this dashboard supersedes the §2 row by row.
- `TODO/pending/119_decode_ratio_strengthen.md` §3 P1.2 (TODO-119 consumer).
- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (TODO-120 consumer).
- `TODO/pending/121_m19_lomo_strengthen.md` §3 P1.1 (TODO-121 consumer).
- `TODO/pending/122_platinai_4cell_r2_strengthen.md` §3 P1.3 (TODO-122 consumer).
- `TODO/pending/123_m15_rmsd_strengthen.md` §3 P1.3 (TODO-123 consumer).
- `TODO/pending/125_vina_per_pocket_strengthen.md` §3 P1.2 (TODO-125 consumer).
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (TODO-122 arch-ceiling
  close-out).
- `TODO/completed/126_innovation_strengthen_roadmap.md` §3 (M1+M2 master plan).
- `metrics/by_metric/decode_ratio.json` (TODO-119 source).
- `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` (TODO-120 source).
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` (TODO-121 source).
- `metrics/by_metric/hilbert_4cell_r2.json` (TODO-122 source).
- `metrics/by_metric/docked_pose_rmsd.json` (TODO-123 source).
- `metrics/by_metric/vina_relaxed_success.json` (TODO-125 source).

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: This dashboard is
aggregation-only — no new measurement. Per-consumer lift is a per-TODO
deliverable (Q4 2026 / R20 GPU exercise). 4/6 consumers are HONEST_NEGATIVE —
do NOT paraphrase as "partially successful" without the negative framing.
If any consumer lift is below 0.05 OR below gate, the honest entry is §6
limitations, NOT §4 LEAD-claim. The Wf-320E dashboard is the verification
instrument for the Wf-123 P3 M1+M2 ship surface, NOT a re-derivation of
the per-consumer lift itself.

**END OF TODO-320E.**
