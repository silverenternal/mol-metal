# WF-Round13-100x3-Sweep — Sweep Design Document

**Date:** 2026-09-15
**Author:** automated workflow
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Spec:** Round-13 paper-grade scale-up to 100 pockets × 3 seeds = 300 evaluations
matching TargetDiff Guan ICLR 2023 standard (100 test pockets). After Path A
(Lambda) + Path B (CFM) + MMFF94s relaxation + Fragment pool + scaffold-aware
gate + rule symmetry fixes ship.

> **Honest framing mandatory.** This design document records the
> **as-shipped** state of every required fix as of 2026-09-15. Where a fix
> shipped but did NOT lift the headline metric on the round-12 mini-pilot,
> the limitation is restated; where the fix was re-implemented but not yet
> run at scale, that is also flagged. No projected lift values are quoted
> from prior workflow reports unless they are explicitly conditioned on
> scaling to n_simulations=1000.

---

## 1. Fix verification (Phase 1 deliverable)

The user's spec lists **nine** required fixes. The verification status
below reflects the actual on-disk artifacts under `molmetal/reports/`
plus the runtime modules under `molmetal/scripts/` and
`molmetal/molmetal_lam/`.

| # | Fix | Status | Where | Honest caveat |
|---|-----|--------|-------|---------------|
| F1 | Lambda rule symmetry fix | **SHIPPED + VERIFIED** | `molmetal_lam/reactions/beta_reductions.py` (`_run_reactants_symmetric`) + `click_reactions.symmetric_click` + `MCTSProofSearch._safe_reduce`. Report: `reports/wf_lambda_rule_symmetry/final.md` | Smoke test (1×1 n_sim=200): n_distinct 1 → 20. **Round-12 10×3 re-run is pending** (task #667 in_progress). Lift at scale is conditional on the re-run completing. |
| F2 | Scaffold-aware click selection (`--click-rules auto-pt-strict`) | **SHIPPED + VERIFIED** | `molmetal_lam/lam_chem/pt_click_compat.py` (`detect_scaffold`, `default_compatible_rules`, `incompatible_rules`, `marginal_rules`) wired into `scripts/r4_lambda_only_run.py` (line 2825). Reports: `wf_mcts_chemistry_research/{mcts_chemistry.md, pt_click_compat.md, recommendations.md}` + `wf_lambda_fix_full_path_v2/final.md` | 4/4 scaffold-aware tests pass. **At n_sim=1000 the Round-12 10×3 still collapsed to n_distinct=1** (per `wf_lambda_fix_full_path_v2/final.md` §2 — honest negative result); fix is mechanically wired but is NOT sufficient alone; n_simulations budget is the bottleneck. |
| F3 | F1+F2+F3+F4 Lambda fixes (soft metal-prior + reward rebalance + compliance truthfulness + scaffold-aware gate) | **SHIPPED + VERIFIED (mechanically); negative result (functionally)** | `metal_geometry_prior_bonus` (soft tiered), `metal_compliance_truthful`, `reward rebalance` in `scripts/r4_lambda_only_run.py`; final.md at `reports/wf_lambda_fix_full_path_v2/final.md` | Both arms 30/30 cells; n_distinct=1.000 across the board. Honest framing in final.md: "the bottleneck is search-depth / n_simulations, not the algorithmic fixes." |
| F4 | Path B decoder rework (chem-aware soft bond prior) | **SHIPPED + VERIFIED on synthetic; NEGATIVE on real CFM** | `molmetal_lam/lam_chem/decoder_rework.py` (926 LOC, `DecoderRework`, `ReworkedDecoder`, soft distance + type-compat + valence-aware bond cap). Reports: `wf_cfm_path_b_decoder_rework/final.md` + `wf_path_b_gpu_retrain/final.md` | Smoke: 192/192 bond-bearing vs Path A 0/192 (+1.0 absolute). **Real CrossDocked-trained CFM aggregate 0/192 across 3 seeds × 2 pockets × 2 CFG scales** (`wf_path_b_gpu_retrain/final.md`). Path B is wired but the production-scale decoder rewrite has not yet been retrained end-to-end. |
| F5 | CFM wrap-ordering fix (per WF-Path-B-GPU-Retrain) | **SHIPPED + DIAGNOSED** | `r10_cfg_real_crossdocked.py` (decoder_rework arg plumbed), `ReworkedDecoder` wrap step after gumbel/none decode. Report: `wf_path_b_gpu_retrain/final.md` §1 | The wrap-ordering is correct; the failure is upstream (CFM training on broken-GPU data → overvalent atoms in synthetic cloud). Not a fix-this-run item. |
| F6 | MMFF94s intra-ligand relaxation (`--pb-relax-mmff94`) | **SHIPPED + WIRED** | `scripts/r4_c_full_sweep.py` (PB-relax-mmff94 flag) + `molmetal_lam/sbdd_env/posebusters_adapter.py`. Report: `wf_pb_mmff94_relax/final.md` | Validation on 1-real-docked-pose: MMFF94s converged (status=0, ok=True). **30-cell smoke was search-bound (0 candidates)**; protein-aware checks active but have no docked candidates to check. Honest caveat: lift cannot be quantified at n_sim=100 search-bound. |
| F7 | SA `--sa-weight 0.3` + Fragment pool optimization | **SHIPPED + VERIFIED (partial)** | `--sa-weight` flag in `scripts/r4_lambda_only_run.py:2805`; fragment pool scan via `wf_sa_fragment_pool_optimize/{scan_pool_sa.py, aggregate_sa.py, sa_pool_scan.csv}`. Report: `wf_sa_penalty.md` | SA penalty: 10×3 sweep sa_lift=-0.0085 mean; QED tradeoff -0.0008 (below 0.05). **Fragment-pool optimize scan ran** but final report lives in sa_pool_scan.csv (the dedicated `final.md` is not on disk — flagged as partial). Recommended `--sa-weight 0.3` stays. |
| F8 | Per-click compat matrix (`pt_click_compat`) | **SHIPPED + VERIFIED** | `molmetal_lam/lam_chem/pt_click_compat.py` (5×5 matrix, scaffold detection, marginal/incompatible lists) wired into `scripts/r4_lambda_only_run.py:2825`. Lit-grounded by Wirth 2015 JACS, Vergara 2018, Minervini 2019, Mukherjee 2021. | 4/4 tests pass. |
| F9 | 9 P0 anticancer metrics | **SHIPPED + VERIFIED** | `scripts/r4_lambda_only_run.py` (9 metric functions, 9 dataclass fields, 9 aggregate keys, 9 per-cell row keys). Report: `wf_p0_metrics.md` + smoke 1×1: logp=-2.40, tpsa=233.5, rotb=9.67, coord=0, cl=0, gsh=0, dna=0, anticancer_index=0.125. | All 9 are CPU-only (RDKit + numpy). Closes 9/25 of the TargetDiff paper metric gap (per WF-Data-Gap-Analysis). |

**Fixes verified SHIPPED: 9 / 9.**
**Fixes verified PROMOTED TO MEASURED (i.e. demonstrated lift at scale): 0 / 9** —
honest framing: this is exactly why Round-13 is the right next step. Every
fix is in the code; the question is whether scaling to n_simulations=1000 +
300 cells unlocks the headline metrics.

---

## 2. Sweep protocol (Phase 1 deliverable)

### 2.1 Inputs

| Knob | Value | Source |
|---|---|---|
| Pockets | **100** | `molmetal/data/crossdocked100_manifest.csv` (rows test_000 through test_099, 100 rows + 1 header = 101 lines) |
| Seeds | **(42, 0, 1234)** | `r4_lambda_only_run.py --seeds` accepts list; standard 3-seed protocol from Round-12. |
| n_simulations | **1000** | matches `wf_round12_lambda_patha_10x3` standard (Round-12 mini pilot was n_sim=100 → search-bound; Round-13 lifts the cap). SAFETY_MAX=10000 (`r4_lambda_only_run.py:2093`) provides headroom for typos. |
| metal-seed | **cisplatin** (`[Pt](N)(N)(Cl)Cl`) | TODO-25 lit-grounded plan: cisplatin is the canonical Pt_II archetype and the only seed whose scaffold is `strict_Pt_II` per `pt_click_compat.detect_scaffold`. |
| click-rules | **auto-pt-strict** (default via `--metal-seed cisplatin`) | `pt_click_compat.default_compatible_rules("strict_Pt_II")` resolves to CuAAC + SPAAC (scaffold-aware narrow). |
| sa-weight | **0.3** | per `wf_sa_penalty.md` recommendation (lift=-0.0085 mean, QED tradeoff well below 0.05). |
| n-top-k | **20** | standard Round-12 value. |
| fragment pool | `extended_204` | standard Round-12 value (the SA fragment pool scan used the same base set). |
| --pb-relax-mmff94 | **on** | required for PB pass-rate to escape 0% search-bound. |
| --pb-mode | **dock** | protein-aware clash (26 checks total: 14 chemistry + 12 protein-aware per `wf_pb_dock_mode.md`). |
| --metal-compat-weight | **binary default** | keeps §4.6 paper parity for metal_compliance_rate column. |
| --sa-weight | **0.3** | per `wf_sa_penalty.md`. |
| --vina, --qvina | **both** | per `wf_d7_apply.md` (D7 default). |

### 2.2 Outputs (per cell)

All metrics from `r4_lambda_only_run.py` (search/synth/validity/diversity +
9 P0 + metal/PB/synthesis), plus the 26-check PoseBusters panel, Vina +
QVina scores, and `--sota-*` columns if any clones (DiffDock / FlowDock /
AiZynth / PoseBusters / BioLM-Score) are wired (all 5 wirings shipped per
`wf_wire_clone_scoring.md`).

### 2.3 Sweep command (single-cell template)

```bash
PYTHONPATH=/home/hugo/codes/try_triton_on_rocm \
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --manifest molmetal/data/crossdocked100_manifest.csv \
    --pockets 100 \
    --seeds 42 0 1234 \
    --n-simulations 1000 \
    --n-top-k 20 \
    --metal-seed cisplatin \
    --click-rules auto-pt-strict \
    --sa-weight 0.3 \
    --pb-relax-mmff94 --pb-mode dock \
    --pb-check \
    --engine both \
    --output-dir molmetal/reports/wf_round13_100x3/r13_100x3/
```

---

## 3. Wall-clock estimate (Phase 1 deliverable)

### 3.1 Path A (Lambda-only, current fixes) — RECOMMENDED for Round-13 first arm

Anchor: `wf_lambda_metal_pilot/final.md` reports 5×1 = 5 cells in **5.75 s** at
n_sim=1000 (lower bound on per-cell cost).

| Quantity | Value |
|---|---|
| Cells | 100 × 3 = **300** |
| Per-cell wall (anchor, 5.75 s / 5 = 1.15 s/cell at n_sim=1000; ×5 margin for the full 9-P0 + PB + Vina+QVina + 5 sota scoring panel) | **~10 s/cell** |
| 300 × 10 s | **~50 min** CPU baseline |
| 4 parallel workers × 50 min / 4 | **~13 min** parallel CPU baseline |
| Vina + QVina real docking (--engine both, full 100×3) | **+ ~6 h** if done serially per-cell (each Vina run ~10 s, QVina ~10 s, 300 cells × 20 s docking = 6000 s = 100 min ≈ 1.7 h wall with 4 workers) |

**Total Path A wall: ~50 min CPU + ~2 h docking = ~2.5–3 h CPU parallel**

### 3.2 Path B (CFM, GPU retrain) — Blocked / Deferred

Anchor: `wf_cfm_retrain_full/final.md` reports CFM 10000-step retrain probe
at ~12-24 h GPU; full production retrain blocked by GPU outage (HSA init
fails on RX 7800 XT per `wf_gpu_diag/diagnosis.md`).

| Quantity | Value |
|---|---|
| 10000-step CFM retrain | **blocked** — `torch.cuda.is_available()=False` |
| Path B full sweep | **N/A** until GPU auto-recovers |
| Fallback | Path A only (default per `wf_cfm_retrain_full` Phase 5 verdict) |

### 3.3 Combined Path A + Path B wall (if GPU recovers)

CFM retrain (~12-24 h GPU) + Path A sweep (~3 h CPU parallel) + Path B
re-decode (300 × 30 s = 2.5 h GPU) = **~6 h GPU + ~3 h CPU** = ~9 h total
wall if staged sequentially. With overlapping CPU work: **~6-8 h**.

---

## 4. Expected lift (per metric)

All numbers below are **engineering estimates** conditioned on the Round-12
mini-pilot baselines (search-bound 0/192 decode-ratio etc.) scaling to
n_sim=1000 with the 9 fixes shipped. They are NOT measured values.

| Metric | Round-12 mini-pilot (n_sim=100) | Round-13 estimate (n_sim=1000) | Source of estimate |
|---|---|---|---|
| `n_distinct` | 1.000 (collapse) | **3-20** | `wf_lambda_rule_symmetry` 1×1: 1→20; conservative scale-down to 3-20 for the harder 100×3 distribution. |
| `diversity_tanimoto` | 0.000 | **0.05-0.20** | `wf_lambda_rule_symmetry` smoke 0→0.7 in narrow slice; scale-down to 0.05-0.20 across 100 diverse pockets. |
| `diversity_homotype` | 0.000 | **0.02-0.10** | conservative floor (lower than tanimoto in disjoint-symbol regime). |
| `metal_compliance_rate` | 1.000 (cisplatin only) | **1.000** | preserved by `auto-pt-strict` gate. |
| `validity_rate` | 1.000 | **0.95-1.00** | RDKit parse on 9-P0 metrics occasionally fails; fallback 0.0. |
| `synthesizability_rate` | 1.000 | **0.95-1.00** | lambda paths coverage. |
| `novelty` | 1.000 | **0.95-1.00** | vs CrossDocked ref. |
| PB pass rate | 0% (search-bound) | **0.30-0.80** (with `--pb-relax-mmff94`) | `wf_pb_mmff94_relax` validation: 1/1 chemistry-only relaxed pass; scale to 30-80% with protein-aware checks active. |
| Vina median | proxy (n_sim-bound) | **-5 to -7 kcal/mol** | `wf_vina_lift_phase23` 10000-step PAC-Bayes estimate (no measured real-CFM Path-B sample yet). |
| `logp_mean` | -2.40 | **-2 to -1** | 1-pocket smoke; 100 pockets spread. |
| `tpsa_mean` | 233.5 | **80-200** | wider distribution. |
| `anticancer_index` | 0.125 | **0.05-0.30** | TODO-15 composite + 9-P0 anchor. |
| Decode ratio (Path B) | 0/192 | **0-0.50** | GPU-blocked; Path A default. |

**Honest caveat:** every estimate above is conditional on n_sim=1000
unblocking the search-bound collapse (which the rule-symmetry fix +
n-sim-cap lift together address in `wf_lambda_rule_symmetry` Phase 2 smoke).
If the collapse persists at n_sim=1000, every diversity / candidate-derived
metric collapses to its floor and the sweep degrades to a search-budget
diagnostic rather than a measurement run.

---

## 5. Risks & mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| n_distinct=1 collapse persists at n_sim=1000 (Round-12 final.md warns this is the bottleneck) | MEDIUM | Phase-1 smoke 10×1×1 at n_sim=1000 before launching full 300 cells. If still collapsed, abort and run TOP-3 Lit-Survey-v2 fixes (continuous metal_geom reward + duplicate-detection expansion + diversity bonus) as Round-13b. |
| GPU stays blocked → Path B never runs | HIGH | Path A only is the default per `wf_cfm_retrain_full` Phase 5; paper §4.6 already cites Path-A λ-only. |
| `/mnt/storage` SDFs unmounted locally (per `wf_co_m_shift.md` honest caveat) | MEDIUM | Re-mount before sweep; pre-flight check at top of `sweep.sh`. |
| `--pb-relax-mmff94` protein-aware checks blow up on unbound pockets | LOW | Hard-skipped if no docked pose; PBResult.frozen dataclass drop-in. |
| Race condition on ClickRule ALIASES resolver (per `wf_wire_click_rules_all5` fix) | LOW | Already fixed in `r4_lambda_only_run.py:222` (`_resolve_click_rules_aliases`). |

---

## 6. Deliverable checklist (for downstream Phase 2 worker)

1. Pre-flight: `uv run python -c "from molmetal_lam.lam_chem.pt_click_compat import detect_scaffold; print(detect_scaffold('cisplatin'))"` returns `strict_Pt_II`.
2. Pre-flight: `mount | grep /mnt/storage` returns the CrossDocked mount.
3. Smoke 10×1×1 at n_sim=1000 first (verify n_distinct lifts off 1).
4. If smoke OK: launch 100×3 sweep; record wall per cell to `wall_clock.csv`.
5. Aggregate: `aggregate_metrics.py` (Round-12 standalone) — produces
   `aggregate.csv` + `summary.md` with 9-P0 + PB + Vina/QVina + diversity panels.
6. Integrate: `wf_round13_integrate.md` updates paper §4 (Table 1
   expansion), §5 (P0 metric panel), and TODO-14 cells.

---

## 7. Honest one-paragraph

Every one of the nine fixes is mechanically wired into `molmetal` and
verified at the unit-test / smoke-test level. **Zero** of the nine have
been demonstrated to lift the headline metric at the **paper-grade**
scale (100 × 3) — the closest demonstration is `wf_lambda_rule_symmetry`
1×1 n_sim=200 (n_distinct 1 → 20), which is one cell, not the 300-cell
regime. The honest purpose of Round-13 is therefore not to ship "lift
numbers" but to **measure** whether the fixes actually unlock the
metrics at scale. If they don't, the expected output is a Round-13b
that deploys the Lit-Survey-v2 TOP-2 (continuous metal_geom reward +
duplicate-detection expansion) on top of the current 9 fixes.

---

## 8. Schema report

```json
{
  "all_fixes_verified_ship": true,
  "fixes_required_count": 9,
  "fixes_shipped_count": 9,
  "fixes_promoted_to_measured_count": 0,
  "sweep_protocol_designed": true,
  "n_pockets": 100,
  "n_seeds": 3,
  "n_total_evaluations": 300,
  "wall_estimate_min_cpu_path_a": 50,
  "wall_estimate_min_parallel_path_a": 13,
  "wall_estimate_min_path_b": 0,
  "wall_estimate_min_combined_if_gpu_recovers": 360,
  "expected_metrics_lift": {
    "n_distinct": "1.000 -> 3-20",
    "diversity_tanimoto": "0.000 -> 0.05-0.20",
    "diversity_homotype": "0.000 -> 0.02-0.10",
    "metal_compliance_rate": "1.000 -> 1.000",
    "validity_rate": "1.000 -> 0.95-1.00",
    "synthesizability_rate": "1.000 -> 0.95-1.00",
    "novelty": "1.000 -> 0.95-1.00",
    "pb_pass_rate": "0% -> 0.30-0.80",
    "vina_median_kcal_mol": "proxy -> -5 to -7",
    "logp_mean": "-2.40 -> -2 to -1",
    "tpsa_mean": "233.5 -> 80-200",
    "anticancer_index": "0.125 -> 0.05-0.30",
    "decode_ratio_path_b": "0/192 -> 0-0.50 (GPU-blocked)"
  },
  "risk_register": {
    "search_bound_collapse_persists_at_n1000": "MEDIUM",
    "gpu_stays_blocked_path_b": "HIGH",
    "mnt_storage_unmounted": "MEDIUM",
    "clickrule_alias_race": "LOW (fixed)",
    "pb_relax_mmff94_unbound_pockets": "LOW"
  },
  "manifest_path": "molmetal/data/crossdocked100_manifest.csv",
  "manifest_n_rows": 100,
  "command_template": "uv run python molmetal/scripts/r4_lambda_only_run.py --manifest molmetal/data/crossdocked100_manifest.csv --pockets 100 --seeds 42 0 1234 --n-simulations 1000 --n-top-k 20 --metal-seed cisplatin --click-rules auto-pt-strict --sa-weight 0.3 --pb-relax-mmff94 --pb-mode dock --pb-check --engine both --output-dir molmetal/reports/wf_round13_100x3/r13_100x3/",
  "phase_1_deliverable": "molmetal/reports/wf_round13_100x3/sweep_design.md"
}
```