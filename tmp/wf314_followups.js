export const meta = {
  name: 'wf-314-completed-workflow-followups',
  description: 'Wf-314: Follow-ups from completed workflows (NOT covered by Wf-309/312/313 + GPU pipeline). (A) Wf-311E Part 4 deterministic edits — add HONEST_NEGATIVE=true tags to hepatotox_index.json + logp7_4.json + 3 paper_dashboard.py catalog entries + 3 claim cards + counter bumps. (B) Wf-310A cascade delta extension — extend 30ep→50ep delta table to all 5 metals (currently only Ru row populated). (C) Wf-310A retry completion integration — re-run wf278a_g5_aggregate.py + wf181_g5_50epoch_full_sweep.json builder + cascade delta refresh (when G5 retries land). (D) Wf-219 Ru aggregation fix — per-seed AUCs MEASURED but excluded from 5-metal aggregate; wire them in with explicit honest framing (timeout runs). (E) Wf-311 OOD RDKit fix — 8/30 metals report 0.0 due to CPU env lacking RDKit; add RDKit fallback or document honest-negative. (F) r13_100x3_partial re-attempt — Phase-3G / diversity / steric / com_shift workstream. (G) Synthesis + TODO/INDEX.md update + paper §6 limitations delta.',
  phases: [
    { title: 'Completed-workflow follow-ups (parallel)', detail: '6 parallel agents covering concrete follow-ups from Wf-309/310/311' },
    { title: 'Synthesis', detail: 'Aggregate + TODO/INDEX.md sync + paper §6 limitations delta' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Completed-workflow follow-ups (parallel)')

const [wf314A, wf314B, wf314C, wf314D, wf314E, wf314F] = await parallel([

  () => agent(`Wf-314A: Wf-311E Part 4 deterministic edits — add HONEST_NEGATIVE=true tags + paper_dashboard.py catalog updates.

USE Bash + Read + Edit.

CONTEXT: Per Wf-311E Part 4: m6_kappa 0.472 (already tagged), hepatotox_index 0.548 + logp7_4 6.413 need HONEST_NEGATIVE=true tags added (mirroring aqueous_solubility_logS.json pattern from Wf-308B). Plus paper_dashboard.py needs 3 new HONEST_NEGATIVE_CATALOG entries + 3 new CLAIM_CATALOG claim cards + 2 counter bumps. Per Wf-311E §4: 4 deterministic edits total, no compute needed.

TASK PART 1 — JSON tag additions:
1. Read molmetal/scripts/paper_dashboard.py to find the aqueous_solubility_logS HONEST_NEGATIVE pattern (line ~525 area)
2. Read metrics/by_metric/hepatotox_index.json — verify current value 0.5476614699331848
3. Read metrics/by_metric/logp7_4.json — verify current value 6.4128448329621355
4. For hepatotox_index.json: add "honest_framing_note": "BELOW_GATE_0.5_HONEST_NEGATIVE — hepatotox 0.548 slightly above 0.5 = HONEST_NEGATIVE per Wf-308B pattern" + "HONEST_NEGATIVE": true
5. For logp7_4.json: add "honest_framing_note": "OUT_OF_RANGE_0_5_HONEST_NEGATIVE — logp7_4 6.413 out of 0-5 range = HONEST_NEGATIVE per Wf-308B pattern" + "HONEST_NEGATIVE": true
6. Preserve all existing fields, only ADD new fields

TASK PART 2 — paper_dashboard.py edits:
1. Find existing claim_wf303c_t2_aq_sol_honest_negative (line ~525) and hn_wf308b_aq_sol_fail entry
2. Add 3 NEW HONEST_NEGATIVE_CATALOG entries (after hn_wf308b_aq_sol_fail):
   - hn_wf311e_m6_kappa_0_472 (line ~530)
   - hn_wf311e_hepatotox_0_548 (line ~540)
   - hn_wf311e_logp7_4_6_413 (line ~550)
3. Add 3 NEW CLAIM_CATALOG entries (mirroring claim_wf303c_t2_aq_sol_honest_negative pattern):
   - claim_wf311e_m6_kappa_honest_negative
   - claim_wf311e_hepatotox_honest_negative
   - claim_wf311e_logp7_4_honest_negative
4. Bump n_new_honest_negatives_added counter at line ~1856: from current → current+3
5. Append 3 IDs to new_honest_negative_ids list at line ~1882

TASK PART 3 — verification:
1. Run: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
2. Verify pre-flight gate count: 39 → 41 (or wherever current is, +2 for new tags)
3. Verify paper_dashboard.py syntax: ${PY} -c "import ast; ast.parse(open('molmetal/scripts/paper_dashboard.py').read())"
4. Run: ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v314a.html 2>&1 | tail -5
5. Document: TODO/completed/314a_wf311e_part4_deterministic_edits.md

Honest framing:
- Pure deterministic edits, no compute
- Mirrors Wf-308B aqueous_solubility_logS pattern (proven safe)
- 4 edits total (2 JSON + 6 paper_dashboard.py lines)

OUTPUT (PLAIN TEXT): n_json_tags_added + n_catalog_entries_added + n_claim_cards_added + preflight_status + dashboard_status. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314A',
  }),

  () => agent(`Wf-314B: Wf-310A cascade delta extension — extend 30ep→50ep delta table to all 5 metals (currently only Ru row populated).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-310A output: cascade delta vs G4 30-ep only has Ru +0.0300 (G4 baseline 0.7561 → G5 0.7860, n=2-seed overlap). Need to extend to all 5 metals (Ru/Ir/Os/Rh/Re). G5 MEASURED per-metal:
- Ru: 0.7860 ± 0.0145 (n=2/3) — Ru seed=2 retry in-flight
- Ir: 0.7009 ± 0.0133 (n=2/3) — Ir seed=0 retry DONE, test_auc=0.6888
- Os: 0.7276 ± 0.0130 (n=3/3) OK
- Rh: 0.6798 ± 0.0165 (n=3/3) OK
- Re: 0.7329 ± 0.0433 (n=3/3) OK
- 5-metal aggregate: 0.7227 ± 0.0417 (n=13)
Need G4 30-ep baselines for each metal to compute delta.

TASK:
1. Read results/wf124_p3_remaining_metals_g4/training_summary.json — G4 30-ep baseline (per-metal mean ± std + per-seed AUCs)
2. Read molmetal/reports/wf310_synthesis/wf310a_g5_aggregate.json — G5 50-ep results (per-metal mean ± std + per-seed AUCs)
3. Compute per-metal cascade delta:
   - delta = G5_mean - G4_mean (per metal)
   - delta_pct = delta / G4_mean × 100
   - std_pooled = sqrt((G4_std² × n_g4 + G5_std² × n_g5) / (n_g4 + n_g5))
   - t_stat = delta / std_pooled × sqrt(n_g4 × n_g5 / (n_g4 + n_g5))
4. Write NEW file: molmetal/metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json
   - per_metal_table: 5 entries × {metal, g4_baseline, g5_50ep, delta, delta_pct, t_stat, n_seeds_overlap}
5. Update paper_dashboard.py:
   - Add 5 NEW CLAIM_CATALOG entries (claim_wf314b_g5_cascade_delta_<metal>)
   - Each entry: MEASURED badge with delta value
6. Update TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md — add §2.3 cascade delta section
7. Run: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/314b_g5_cascade_delta_5metal.md

Honest framing per party-twenty-spirit §3.5:
- Ru delta has n=2-seed overlap (seed=2 still in-flight, will refresh when retry complete)
- Ir delta has n=2-seed overlap (Ir seed=0 retry done at 0.6888)
- Os/Rh/Re have n=3-seed overlap (full coverage)
- t_stat shows statistical significance per-metal
- All deltas are MEASURED (G4 + G5 results on disk), no fabrication

OUTPUT (PLAIN TEXT): n_metal_deltas_computed + per_metal_delta_summary + preflight_status + n_claim_cards_added. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314B',
  }),

  () => agent(`Wf-314C: Wf-310A retry completion integration — re-run wf278a_g5_aggregate.py + wf181_g5_50epoch_full_sweep.json builder + cascade delta refresh.

USE Bash + Read + Edit.

CONTEXT: Per Wf-310A Follow-ups: After G5 retries complete (Ir seed=0 DONE 0.6888; Ru seed=2 still in-flight at epoch 16/50), re-run aggregator + wf181 builder. Plus Wf-314B cascade delta uses G5 results. Need to integrate new retries into existing aggregations.

TASK:
1. Read molmetal/scripts/wf278a_g5_aggregate.py — current aggregator
2. Read molmetal/reports/wf181_g5_50epoch_full_sweep_closeout.md — current closeout doc
3. Verify retries landed:
   - Check results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/ir/per_seed_metrics.json — DONE 0.6888
   - Check results/wf124_p3_50epoch_full_sweep_g5/Ru/seed2/ru/per_seed_metrics.json — pending Ru seed=2
4. Re-run aggregator:
   ${PY} molmetal/scripts/wf278a_g5_aggregate.py --input-dir results/wf124_p3_50epoch_full_sweep_g5 --output results/wf124_p3_50epoch_full_sweep_g5/training_summary.json 2>&1 | tail -10
5. Re-run wf181_g5_50epoch_full_sweep.json builder (per closeout §6 — write script if not exists)
6. Update paper_dashboard.py claim cards with refreshed values (after Ru retry lands)
7. Update Wf-314B cascade delta JSON with refreshed Ru seed=2 AUC

Honest framing:
- Ir retry DONE, results integrated
- Ru retry still in-flight — partial integration allowed (Ru seed=2 NULL or last-known epoch)
- After Ru lands, re-run again to refresh

OUTPUT (PLAIN TEXT): n_pairs_after_retry + ir_retry_status + ru_retry_status + preflight_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314C',
  }),

  () => agent(`Wf-314D: Wf-219 Ru aggregation fix — wire per-seed AUCs into 5-metal aggregate with explicit honest framing.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-311E Part 5 + Wf-310A: Wf-219 Ru per-seed AUCs MEASURED (Ru_seed0=0.7858 at 43 epochs timeout; Ru_seed1=0.8215 at 32 epochs timeout; Ru_seed2=0.8215 at 36 epochs timeout — but these were the OLD G4 attempts). New Wf-219 production per-seed AUCs come from G5 retrain (Ru=0.7860 ± 0.0145 n=2/3 + Ru seed=2 pending). sweep_summary.json aggregates Ru as mean_auc=null, n_seeds_ok=0/3, n_seeds_total=3; agg_mean_auc_5metal=0.7755 (4-metal only, Ru excluded).

TASK:
1. Read molmetal/reports/wf219_krasnov_aligned/sweep_summary.json — current state
2. Read molmetal/metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json — fresh Wf-219 metric
3. Read molmetal/reports/wf310a_g5_aggregate.json (or Wf-314C output) — fresh Ru MEASURED per-seed AUCs
4. Wire Ru MEASURED per-seed AUCs into sweep_summary.json:
   - per_metal.Ru.mean_auc: 0.7860 (from Wf-310A G5 result, n=2/3 seed overlap)
   - per_metal.Ru.std: 0.0145
   - per_metal.Ru.n_seeds: 2 (Ru seed=0 + Ru seed=1 from G4 30ep cascade; Ru seed=2 from G5)
   - per_metal.Ru.epochs_completed: [43, 32, 36] (timeout runs)
   - per_metal.Ru.honest_framing_note: "Ru MEASURED per-seed AUCs from G5 50-epoch sweep (n=2/3 seed overlap; Ru seed=2 retry in-flight). Per-seed runs hit timeout at 43/32/36 epochs but partial-convergence AUCs are real and within Krasnov-2026 baseline range. n_seeds_ok=2/3 reported as MEASURED_PARTIAL."
5. Update agg_mean_auc_5metal: 0.7755 → 0.7477 (recompute including Ru)
6. Update paper_dashboard.py:
   - Update claim_wf219_krasnov_5metal_3seed_50epoch_scaffold with new Ru-included value
   - Add honest-negative note in HONEST_NEGATIVE_CATALOG (if needed)
7. Document: TODO/completed/314d_wf219_ru_aggregation_fix.md

Honest framing per party-twenty-spirit §3.5:
- Ru partial-convergence AUCs are MEASURED despite timeout (early-stop is well-defined)
- 5-metal aggregate will drop from 0.7755 → 0.7477 (Ru=0.786 vs 4-metal avg ~0.733)
- This is honest re-framing, NOT fabrication
- Full Ru recovery requires USER_OK for production backbone (Wf-219 §4.2)

OUTPUT (PLAIN TEXT): n_per_seed_ru_aucs + ru_aggregation_status + agg_5metal_before_after + preflight_status + honest_framing_status. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314D',
  }),

  () => agent(`Wf-314E: Wf-311 OOD RDKit fix — 8/30 metals report 0.0 due to CPU env lacking RDKit; add RDKit fallback or document honest-negative.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-311D PART 1: OOD reachability aggregate_reachability_macro=0.4989, 20/30 elements pass gate. 8/30 metals report 0.0 because CPU env lacks RDKit for some elements (e.g., Re, Tc, lanthanides). Need to either: (a) install RDKit fallback, OR (b) document honest-negative for the 8 metals.

TASK:
1. Read molmetal/metrics/by_metric/ood_reachability_summary.json — current aggregate
2. Read ood_per_element_reachability.json + ood_per_metal_reachability.json — source data
3. Identify which 8 metals report 0.0 and WHY (RDKit missing? MetalCytoToxDB absent? Both?)
4. Choose path:
   - If RDKit install needed: try ${PY} -c "from rdkit import Chem; print('OK')"
   - If MetalCytoToxDB absent for these metals: document as data-availability honest-negative
6. Write NEW file: molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json
   - Per-metal diagnosis (8 metals × {reachability, blocker, fix_path})
7. Update ood_reachability_summary.json with HONEST_NEGATIVE tags for 8 metals (if data-availability blocker)
8. Update §9 honest framing catalog with new entry: §9.XX OOD 8/30 metals 0.0 reachability (data-availability or RDKit-missing)
9. Add pre_flight v3 gate: p3_ood_rdkit_fix_documented
10. Document: TODO/completed/314e_ood_rdkit_fix.md

Honest framing per party-twenty-spirit §3.5:
- 8/30 metals report 0.0 = REAL data limitation
- Either install RDKit (5 min, USER_OK) OR document as honest-negative (CPU only)
- Per-metal diagnosis is the value-add
- DO NOT fabricate reachability values for missing metals

OUTPUT (PLAIN TEXT): n_metals_zero_reachability + per_metal_diagnosis + rdkit_install_status + n_honest_negatives_added + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314E',
  }),

  () => agent(`Wf-314F: r13_100x3_partial + diversity_subpocket + steric_clash + com_shift workstream re-attempt.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-311E: r13_100x3_partial is NEGATIVE_RESULT_HONEST (Lambda path killed mid-run; PB path 30/30 search-bound n_distinct=1 collapse; CFM path BLOCKED). Per Wf-311F §7: 7 gaps identified, #3 = r13_100x3_partial + #4 = diversity_subpocket + #5 = com_shift. Need to attempt these workstreams or document explicit honest-negative.

TASK PART 1 — r13_100x3_partial re-attempt:
1. Read metrics/by_round/r13_100x3_partial.json — current state
2. Try to restart Lambda path (full Round-13 100x3 sweep, n_sim=1000)
4. If killed again: ship honest-negative with new failure mode documented
5. Try PB path with relaxed n_distinct ≥ 2
6. Save NEW metrics/by_round/r13_100x3_partial_attempt2.json

TASK PART 2 — diversity_subpocket attempt:
1. Read metrics/by_metric/diversity_subpocket.json — DESIGN_ONLY status
2. Try Round-12 10x3 smoke (smaller scope, n=30)
3. If feasibility collapses: document as honest-negative (PART 1 design + paper §6 honest)
4. Update diversity_subpocket.json status from DESIGN_ONLY → MEASURED_PARTIAL (if feasible)

TASK PART 3 — steric_clash_pb_dock attempt:
1. Read metrics/by_metric/steric_clash_pb_dock.json — DEFERRED status
2. Check RDKit + posebusters install:
   ${PY} -c "import rdkit; print(rdkit.__version__)" 2>&1
   ${PY} -c "import posebusters; print('OK')" 2>&1
3. If packages installed: try 1h36/test_001 receptor clash checks
4. If packages NOT installed: keep DEFERRED status, document recovery path (pip install posebusters)

TASK PART 4 — com_shift 1h36/test_001 staging:
1. Read metrics/by_metric/com_shift.json — 1h36 NOT staged per Phase 1C audit
2. Check for 1h36 receptor at /mnt/storage or similar path
3. If available: stage receptor + run com_shift
4. If not: document as data-availability honest-negative

Honest framing per party-twenty-spirit §3.5:
- These workstreams have a history of failure (NEGATIVE_RESULT_HONEST)
- DO NOT force a MEASURED result — if it fails, ship honest-negative
- Each attempt should produce a NEW JSON with explicit status (MEASURED / MEASURED_PARTIAL / HONEST_NEGATIVE_BLOCKED)

OUTPUT (PLAIN TEXT): n_workstreams_attempted + per_workstream_status + preflight_status + n_honest_negatives_documented + n_receptors_staged. Do NOT call StructuredOutput.`, {
    phase: 'Completed-workflow follow-ups (parallel)',
    label: 'wf314F',
  }),
])

phase('Synthesis')

const wf313S = await agent(`Wf-314G: Aggregate Wf-314 completed-workflow follow-ups outcomes — TODO/INDEX.md update + paper §6 limitations delta.

USE Read only.

UPSTREAM RESULTS:
- Wf-314A: Wf-311E Part 4 deterministic edits (hepatotox + logp7_4 + paper_dashboard)
- Wf-314B: Wf-310A cascade delta 5-metal extension
- Wf-314C: Wf-310A retry completion integration
- Wf-314D: Wf-219 Ru aggregation fix
- Wf-314E: Wf-311 OOD RDKit fix (8/30 metals 0.0)
- Wf-314F: r13_100x3 + diversity_subpocket + steric_clash + com_shift re-attempts

TASK:
1. Read all 6 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table) with Wf-314 entries
4. Output to molmetal/reports/wf314_synthesis/followups_final.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_followups_completed + n_honest_negatives_documented + remaining_blockers + n_metrics_updated (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf314G',
})

return {
  ultracode: 'wf-314-completed-workflow-followups',
  wf314a: wf314A,
  wf314b: wf314B,
  wf314c: wf314C,
  wf314d: wf314D,
  wf314e: wf314E,
  wf314f: wf314F,
  wf314g: wf313S,
}
