# Wf-297D: Remediation Plan per Honest-Negative

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Goal**: For every honest-negative in `wf297c_gap_compute/gap_table.md`
(Section A — 15 items) and Section C — 6 orphan integration gaps, produce a
concrete remediation action targeting **field-average** (NOT SOTA — per user
request: "未必说要把他提到sota水平，只要提到这个领域里大家的平均水平就行了").
All actions should reference cloned projects where applicable.

**Prioritisation**: rank by `gap × effort_inverse` (largest gap + lowest
effort first), with sub-tiebreak on `clone_readiness` (whether the cloned
project already provides the needed primitive).

---

## Remediation Action Table

### TIER-1 (DO FIRST — big gap, low effort, clone-ready)

| # | Negative ID | Current → Target | Gap | Root cause | Remediation action | Owner (Wf-###) | Status | Effort |
|--|--|--|--|--|--|--|--|--|
| R01 | **N1** drugood_v3_scaffold OOD AUC | 0.4574 → 0.55–0.65 | −0.10 to −0.20 | D-MPNN lacks OOD-adversarial head; DANN clone present but not wired | Implement `molmetal/adapters/dann_head.py` (GradientReversalLayer + domain classifier) on top of `train_dmpnn_multitask_wasserstein.py`; reuse `--lambda-grl` flag (currently orphan-flag F2) | **Wf-298A** | PENDING | 30 LOC + 1 GPU-hr (1 seed × 50 epoch) |
| R02 | **N8** diversity_tanimoto (m8) | TBD → 0.50–0.70 | TBD | m8 runs in `wf188_m14_full_metalcytotox` but not aggregated to per-metal | Run `molmetal/scripts/cfm_property_panel.py` on the 3 existing checkpoints (Ru seed0/1/2, Ir seed0) and aggregate per-metal Tanimoto into `metrics/by_metric/diversity_tanimoto.json` | **Wf-298B** | PENDING | 15 LOC + 0.5 GPU-hr |
| R03 | **N12/N13/N14** ames_mutagen / cell_permeability_logPapp / aqueous_solubility_logS | not run → <0.30 / >−6.0 / >−4.0 | unknown (no retrain) | RDKit ADMET pass-through not wired | Add `molmetal/scripts/admet_pass_through.py`: take 449 metallo-drug SMILES (existing in `metallo_drugs_500_train.csv`), call `rdkit.Chem.Descriptors` for logP/Crippen + `rdkit.Chem.FilterCatalog` for mutagen; emit 3 JSON files | **Wf-298C** | PENDING | 80 LOC + 5 min CPU |
| R04 | **O1+O2** Orphan scripts + orphan flags | 10 + 16 → 0 | high | Wf-296A audit; no wrapper | Either delete unused flags (`--no-kendall`, `--no-dann`, `--chemprop-mode v3-fix`, `--chemprop-mode shared-encoder`, `--no-yuelbond`, `--no-wasserstein`, `--no-closure-bounded`, `--use-tanimoto-ood-filter`) OR add a `configs/wf296c_*.yaml` driving them; canonical recommendation: delete | **Wf-296C** | PENDING | 0.5 day (LOC ≈ 50 deletions) |
| R05 | **O3** Orphan tests (14 files not in pre_flight_check.py) | 14 → 0 | 0.62% pre-flight coverage | Wf-296A: only 3/483 tests gated | Add the 14 missing tests to `molmetal/scripts/pre_flight_check.py` §5 P2 tests gate | **Wf-296D** | PENDING | 1 day + 14 LOC |
| R06 | **O4** Orphan catalog (m14_anchor.json + wf253d m17) | 2 → 0 | paper_dashboard misses latest | Wf-296A: `paper_dashboard.py` only references `result.json`, NOT the superseding `m14_anchor.json` | Add 2 lines to `paper_dashboard.py` catalog: `claim_wf287a_m14_anchor` + `claim_wf253d_m17_drugood_v3_reeval` | **Wf-296E** | PENDING | 0.25 day + 2 LOC |

### TIER-2 (DO NEXT — moderate gap or moderate effort, clone helps)

| # | Negative ID | Current → Target | Gap | Root cause | Remediation action | Owner (Wf-###) | Status | Effort |
|--|--|--|--|--|--|--|--|--|
| R07 | **N2** cfm_real_vina_dock | PROMISED → −7.5 to −9.5 kcal/mol | unknown (not run) | Wf-188/Wf-269C: PocketEncoder not wired | Port FLOWR's PocketEncoder from `molmetal/references/FLOWR/` (commit `bbb8bfb`, MIT) → `molmetal/adapters/flow_matching_lipman/pocket_encoder.py`; integrate with existing `cfm_yuelbond_wasserstein.pt` checkpoint | **Wf-298** | PENDING | 200 LOC + 2 GPU-hr |
| R08 | **N3** flowr_vina_dock | not run → −7.0 to −8.5 kcal/mol | unknown | FLOWR clone present, port needed | Same as R07 (the FLOWR port IS the cfm_real_vina_dock backbone); produce dual-track FLOWR-eval + Mol-Metal-eval CSV | **Wf-298** | PENDING | shared with R07 |
| R09 | **N6** m18_per_residue_protein_metal_auc | unknown → 0.60–0.75 | unknown | Wf-252-Apply Fix #2: M12 module fixed but not re-run | Re-run `molmetal/scripts/m18_per_residue_eval.py` on 5-metal × 5-seed checkpoints from `wf252_research/fix_apply/` | **Wf-298D** | PENDING | 100 LOC + 1 GPU-hr |
| R10 | **N5** vina_score (crossdocked 100-pocket) | −1.35 (830c only) → −7.5 to −9.5 | +6.1 to +8.1 | single-pocket; CrossDocked2020 100-pocket corpus not staged | Stage CrossDocked2020 100-pocket subset (D3 mirror fallback documented); extend `molmetal/scripts/lambda_100pocket_sweep.py` to real Vina 1.2.7; parity N=50 already verified at r=0.9983 (Wf-11) | **Wf-298E** | PENDING | 1.5 days + 10 GPU-hr |
| R11 | **N4** cfm_decode_real (silent regression) | 0.9645 (1-seed/818 samples) → 0.85–0.95 | +0.0145 above (PASS but disclose) | Wf-271B methodology change not disclosed in metric JSON | Add `methodology_disclosure` block to `metrics/by_metric/cfm_decode_real.json` per Wf-271B recommendation; rerun on 3 seeds × 5991 samples on GPU cuda:0 to confirm | **Wf-298F** | PENDING | 0.25 day + 0.5 GPU-hr |
| R12 | **O5** pre_flight_check coverage (0.62%) | 3/483 → ≥30/483 | 0.62% → ~6% | Wf-296A: 3 P2 tests gated | Same as R05 (Wf-296D) + add per-wave-checklist hook for new tests; ship a `tests/SMOKE_REGISTRY.md` listing all 483 tests | **Wf-296D** | PENDING | 1 day + 30 LOC |

### TIER-3 (DO LATER — long-pole items, clone needed but heavier)

| # | Negative ID | Current → Target | Gap | Root cause | Remediation action | Owner (Wf-###) | Status | Effort |
|--|--|--|--|--|--|--|--|--|
| R13 | **N11** vina_min | in-progress → −7.5 to −9.5 | TBD | MGLTools / UFF minimization path not wired (M2 protocol-mismatch flag) | Wire `meeko`+`obabel` minimize step into `round11_parity_n50.py` per Wf-12 B-1 plan; produce min-score JSON on the 830c + 1h36 first, then scale | **Wf-298G** | PENDING | 1 day + 0.5 GPU-hr |
| R14 | **N9** docked_pose_rmsd (m11) | in-progress → <2.5 Å median | TBD | post-MMFF rigid-fragment RMSD module not built | Build module per Wf-13 B-1: add `molmetal/scripts/post_mmff_rmsd.py`; reuse chemprop v2's MMFF94 path (`.cloned/chemprop_v2/db0bb86`) | **Wf-298H** | PENDING | 2 days + 30 LOC |
| R15 | **N10** com_shift_<pocket> (m9) | per-pocket → <2.0 Å median | TBD | CoM-shift requires paired reference 3D coord (M4 protocol-mismatch flag) | Compute CoM-shift post-RDKit-embed on 5 tested pockets; reference docking staged for 1h36+830c (need 4rlu, 3eml, 3ny8 too) | **Wf-298I** | PENDING | 1 day + 30 LOC |
| R16 | **N15** affinity_pearson_r (m_score_repellent vs Vina) | running → 0.50–0.70 | TBD | Wf-124 P3 in-flight | Wait for Wf-124 P3 completion (Ru seed0 already 0.7613 vs 0.7123 baseline); aggregate into `metrics/by_metric/affinity_pearson_r.json` | **Wf-124 P3** | IN-FLIGHT | TBD (already in flight) |
| R17 | **N7** m6_kappa_shape_index (pIC50 correlation) | r=0.081 → descriptive only | claim retraction | Wf-253D validation: topological index, not learned score | Add `validation_verdict: NEGATIVE_WEAK` to metric JSON; cite honest-framing in paper §3 "do not claim as learned metallodrug score" | **Wf-298J** | PENDING | 0.1 day + 1 LOC |
| R18 | **O6** CFM arm decoder-bound (0/96 docked) | 0/96 → ≥50/96 | 0% | TODO-21 CFM retrain deferred | Run `Wf-CFM-Retrain-Diagnose` (5000-step retrain); if success → enables #22, #23; if fail → Lambda-only + cite-only is canonical Q1 path | **Wf-298K** | PENDING | 5 GPU-hr |

### TIER-4 (DEFER — env-blocked, long-term)

| # | Negative ID | Current → Target | Gap | Root cause | Remediation action | Owner | Status | Effort |
|--|--|--|--|--|--|--|--|--|
| R19 | **S1** CrossDocked2020 100-pk corpus | 2 → 100 | 50× scale | F1 corpus blocked | Download + extract; verify `crossdocked_full` loader handles 100-pocket batch | TODO-07 | BLOCKED | 0.5 day |
| R20 | **S4** click_rules all 5 (CuAAC only) | 1 → 5 | 4 missing | L-3 fragment library extension | Add SPAAC, thiol-ene, Suzuki, amide coupling click tiles to FRAGMENT_LIBRARY_200_TILES; extend `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | TODO-13 | BLOCKED | 1 day |
| R21 | **S5** synthesis_oracle (REINVENT4) | observability → wired | not live | R5 / D6 REINVENT4 env-blocked | Once D6 resolves, wire `lambda_specific.synthesis_oracle=true` into `molmetal/scripts/r4_lambda_only_run.py` | TODO-05 | BLOCKED | 1 wk after D6 |
| R22 | **S8** Vina-proxy → real Vina 1.2.7 | proxy −14.18 → real −7.5 to −9.5 | placeholder | F2 / D4 oracle | Replace placeholder with full Vina at `exhaustiveness=8`, `n_poses=9` for every MCTS candidate | TODO-12 | BLOCKED | 1–2 days |
| R23 | **S9** QVina ↔ Vina engine swap | Vina 1.2.7 → QVina | engine swap | D7 / TODO-04_qvina_swap | Once QVina binaries available, ~0.5 day swap | TODO-04 | BLOCKED | 0.5 day |
| R24 | **S6** symbolic_prior refit_every=5 | learned once → cadence | cadence knob | trivial | Add `refit_every=5` knob to `molmetal/molmetal_lam/search_alg/proof_search.py` | TODO-11 | PENDING | 1 hr + 5 LOC |

---

## Priority Ranking (gap × effort⁻¹ × clone_readiness)

| Rank | Action | Rationale |
|---|---|---|
| 1 | **R01** drugood OOD AUC lift (DANN) | gap × effort⁻¹ ratio very high (1 hr to +0.10 AUC potential), DANN clone immediately wire-able |
| 2 | **R03** ADMET pass-through (RDKit) | 3 honest-negatives closed in 5 min CPU; no retrain |
| 3 | **R06** paper_dashboard catalog wire | 0.25 day closes 2 catalog orphans |
| 4 | **R05 + R12** Pre-flight test gate expansion | low effort, high value (closes 14 orphan tests + lifts pre-flight coverage from 0.62% to 6%) |
| 5 | **R02** diversity_tanimoto aggregation | 15 LOC closes N8 TBD → measured |
| 6 | **R17** m6_kappa honest-framing claim | 1 LOC closes a claim retraction risk |
| 7 | **R04** orphan flags deletion | cleanup, prevents future drift |
| 8 | **R11** cfm_decode_real regression disclosure | 0.25 day closes disclosure gap |
| 9 | **R07 + R08** FLOWR PocketEncoder port (cfm_real_vina_dock + flowr_vina_dock) | biggest research lift, but FLOWR clone is ready and MIT-licensed |
| 10 | **R09** m18 per-residue re-run | 100 LOC closes unknown gap |
| 11 | **R10** CrossDocked 100-pocket staging | biggest scale gap, env-blocked |
| 12 | **R13** vina_min wire-up | 1 day closes M2 mismatch |
| 13 | **R14** docked_pose_rmsd module | 2 days, chemprop_v2 MMFF94 ready |
| 14 | **R15** com_shift_<pocket> | 1 day, depends on N5 staging |
| 15 | **R16** affinity_pearson_r (Wf-124 P3) | already in flight, no action |
| 16 | **R18** CFM arm retrain (TODO-21) | 5 GPU-hr, independent workstream |
| 17-22 | **R19-R24** env-blocked items | defer until gates resolve |

---

## Total Effort Estimate

| Tier | Count | Effort (CPU-hr) | Effort (GPU-hr) | LOC |
|------|------:|----------------:|----------------:|----:|
| TIER-1 (DO FIRST) | 6 | 2.5 | 1.5 | ~200 |
| TIER-2 (DO NEXT) | 6 | 3.5 | 14 | ~400 |
| TIER-3 (DO LATER) | 6 | 4 | 6 | ~150 |
| TIER-4 (DEFER) | 6 | TBD (env-blocked) | TBD | TBD |
| **TOTAL (active items)** | **18** | **10 CPU-hr** | **21.5 GPU-hr** | **~750 LOC** |

**Active remediation actions**: **18** (TIER-1+2+3 minus items already IN-FLIGHT)
**TIER-4 actions**: 6 (deferred pending env-block resolution)
**Priority ranking**: see table above; TIER-1 items 1–7 should ship this week.

---

## Clone-Project Reference Table (per the user's "参考我们clone的那些项目" request)

For each remediation action, the cloned project that informs the solution:

| Action | Clone used | Local path | License |
|--------|-----------|------------|---------|
| R01 (DANN adversarial head) | DANN + DANN_py3 | `.cloned/DANN/`, `.cloned/DANN_py3/` | MIT (Ganin ICML 2015) |
| R07/R08 (FLOWR PocketEncoder) | FLOWR | `molmetal/references/FLOWR/` | MIT (arXiv:2504.10564) |
| R07/R08 (Lipman flow matching) | flow_matching | `molmetal/references/flow_matching/` | CC-by-NC (Lipman ICLR 2023) |
| R09 (m18 re-run) | dgl-lifesci (D-MPNN backbone) | `molmetal/references/dgl-lifesci/` | Apache-2.0 |
| R10 (CrossDocked staging) | dream-rsi (cross-domain RSI) | `.cloned/dream-rsi/` | MIT |
| R11 (cfm_decode_real disclose) | chemprop_v2 (eval harness) | `.cloned/chemprop_v2/` | MIT |
| R14 (post-MMFF RMSD) | chemprop_v2 (MMFF94 utilities) | `.cloned/chemprop_v2/` | MIT |
| R15 (CoM-shift) | torchdrug (molecular 3D ops) | `molmetal/references/torchdrug/` | Apache-2.0 |
| R18 (CFM retrain) | dgl-source (DGL backend) + flow_matching | `.cloned/dgl-source/`, `molmetal/references/flow_matching/` | Apache-2.0 + CC-by-NC |
| R22 (real Vina) | dgl-lifesci (D-MPNN) for chemprop-style batched affinity | `molmetal/references/dgl-lifesci/` | Apache-2.0 |
| R23 (QVina) | dream-rsi (cross-domain reference ligands) | `.cloned/dream-rsi/` | MIT |

---

## Honest framing (per party-twenty-spirit §3.5)

1. **Field-avg not SOTA**: every target band here is from the cloned project's
   published numbers (DANN ICML 2015, FLOWR arXiv:2504.10564, chemprop_v2
   README, dgl-lifesci README Tab. 3, DrugOOD ICML 2022 Tab. 7). SOTA is out
   of scope per user request.
2. **No claim inflation**: items P1–P11 in `wf297c_gap_compute/gap_table.md`
   Section B are already at/above field-avg and will be cited as honest-
   positive in paper §3; items N1–N15 are the honest-negatives addressed here.
3. **Env-blocked transparency**: TIER-4 items (R19–R24) are explicitly NOT
   claimed as "fixed" — they are deferred pending env resolution (CrossDocked
   corpus, REINVENT4, QVina).
4. **MEASURED vs PROJECTED discipline**: all values in the gap table cite the
   source JSON; remediation actions cite the owner Wf-### and ETA.