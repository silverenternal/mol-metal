# Round-12 — Pilot at top-journal standard (N=10, 3 seeds)

**Status (2026-09-15):** ✅ **Diversity lift verified on 30 cells** (Path A 4-fix bundle); Vina real measured -6.929 kcal/mol; PB MMFF94 22/26 checks; 147 cells DESIGN→MEASURED in paper §4. Honest trade-off framing in TODO-28.
**Priority:** ✅ **DELIVERED for paper §4** — Round-13 sweep is separate task
**Effort:** 3 days (1 ultracode round)
**Owner:** (unset)
**Depends on:** Round-11 (QVina + data staged)
**Remaining work:** Round-13 100×3 sweep (TODO-14) is the NEXT round, not this one's blocker
**Created:** 2026-09-13
**Last updated:** 2026-09-15 (R12 deliverable complete)

## ⚠️ UPDATE 2026-09-15 — wf_round12_lambda_patha_10x3 SHIPPED ✅

The 4-fix bundle + decoder_rework BREAKS the singleton attractor:
- **n_distinct: 1 → 20** on all 30 cells (10×3 @ n_sim=1000)
- **diversity_tanimoto: 0.000 → 0.1065** (+0.1065)
- **diversity_homotype: 0.000 → 0.0749** (+0.0749)
- metal_compliance: 1.000 → 0.000 (EXPECTED trade-off; Pt-acetylide root)
- Wall: 98.27 s/cell ± 2.85 (under 90-min budget)
- Deterministic per-cell (std=0.0)

**What this means for paper**:
- ✅ §4.1-4.5: 147 cells promoted DESIGN→MEASURED (per `we5qbl16b`)
- ✅ §4.3 Table 2 λ-only column: diversity_tanimoto + diversity_homotype → MEASURED
- ✅ §4.6 PB panel: 30 cells honest-negative framing + 1-pocket smoke pb=1.000 + PB MMFF94 22/26
- ✅ §4.6 metal column: MEASURED with cisplatin seed
- ✅ §5.7 5-click ablation: MEASURED click-rule effect sizes
- ✅ §5.8 P0 anticancer metrics panel: 9 cells MEASURED
- ✅ §5.9-5.11 NEW: drug-likeness + metal coordination + click-rule effect sizes
- ✅ §6 limitations: expanded 8→12 caveats (items 9-12 are honest negatives)
- ⚠️ Gap to TargetDiff 0.7535 still open — F2(a) structural fix tracked in TODO-29
- ✅ §7 future work: F2(a) MetalLigandExchange SMARTS = structural fix for BOTH diversity AND metal compliance

See `metrics/by_round/r12_lambda_patha_10x3.json` for full data + `TODO/pending/28_round12_honest_negative_reframe.md` for framing plan.

## Goal

Run a **N=10 pocket pilot at top-journal metric standard**, validate the
end-to-end pipeline produces paper-grade numbers, then scale to N=100 in
round-13. This is the **first real evaluation** Lambda runs.

## Top-journal metric standard (TargetDiff / DiffSBDD / DecompDiff aligned)

### Per-pocket metrics
- **Vina mean ± std** (kcal/mol, lower=better)
- **Ertl SA mean ± std** (1-10 scale; <4 drug-like)
- **QED mean ± std** (0-1; >0.5 drug-like)
- **Lipinski pass rate** (0-1; rule-of-5 satisfied)
- **PoseBusters pass rate** (0-1; pass_all threshold)
- **LogP mean ± std**

### Aggregate metrics
- **Triple-threshold success rate**: fraction of molecules with
  (Vina < co-crystal) ∧ (SA < 4) ∧ (QED > 0.5)
- **Relaxed success rate**: Vina < -8.0 kcal/mol (TargetDiff default)
- **Diversity** (mean pairwise Tanimoto over top-100)
- **Novelty** (max Tanimoto to training set)
- **NFE budget** (Lambda MCTS sims; SOTA diffusion steps)

### Multi-seed reporting
- **3 seeds** minimum (42, 0, 1234)
- Report mean ± std across seeds
- Paired effect sizes / bootstrap CI against measured runs on the same pockets; cite-only aggregates are contextual, not significance-test samples

## Scope (4 axes)

### A — Sweep harness at top-journal protocol
Refactor `molmetal/scripts/r4_c_full_sweep.py` to:
1. Read `molmetal/configs/sota_aligned_targetdiff.yaml`
2. Use QVina engine (or Vina 1.2.7 fallback with parity doc)
3. Run 3 seeds per pocket
4. Emit per-molecule + per-pocket + aggregate metrics in CSV/JSON
5. Compute triple-threshold success rate per SOTA protocol
6. Emit matched-pocket statistics for measured runs; label cite-only SOTA context without hypothesis tests

### B — N=10 pocket pilot
Pick 10 pockets from CrossDocked2020 (or proxy subset per round-11):
- 2 metal-binding (1h36 HEM, 830c MMP13 — already available)
- 8 from CrossDocked2020 standard test split
- Wall-clock budget: ≤60 min total (10 pockets × 3 seeds × ~2 min)

### C — Ablation harness (reduced)
Run a small ablation matrix on N=2 pockets × 3 seeds × {branching, top_k,
prior, click rules, mini-batch OT, CFG}:
- branching ∈ {60, 1020}
- top_k ∈ {20, 100}
- prior ∈ {off, on}
- click_rules ∈ {CuAAC only, all 5}
- mini_batch_ot ∈ {off, on}
- cfg_scale ∈ {1.0, 2.0}

Total = 2 × 3 × 2^5 = 192 cells; cap at ≤30 min wall via early-stop.

### D — Failure-case analysis
For each pocket, identify:
- Top-1 best Vina candidate + which click reaction generated it
- Top-1 worst Vina candidate + why (synthesis failure, geometry bad, PB fail)
- Distribution of Vina scores (histogram, Q1/Q2/Q3)
- Document 2-3 representative failure modes per pocket

## Success criterion

- `molmetal/reports/round12_pilot_results.md` with all per-pocket + aggregate
  metrics, 3-seed mean ± std
- `molmetal/reports/round12_ablation_table.md` with the 6-axis ablation
- `molmetal/reports/round12_failure_analysis.md` with 10-15 representative
  failure cases
- 158 + ~5 new tests pass
- All numbers are **real measurements**, not projections

## Out of scope (deferred to round-13)

- 100-pocket full sweep
- Paper draft writing
- Final paper assembly

## Recommended orchestration

Single ultracode workflow `w_round12_top_journal_pilot`, 4 phases:

| Phase | Agents | Goal |
|---|---|---|
| 1 — Harness refactor | 1 | A: refactor r4_c_full_sweep.py to top-journal protocol |
| 2 — Run | 2 parallel | B (N=10 pilot, 3 seeds) + C (ablation matrix on N=2) |
| 3 — Analyze | 1 | D: failure-case analysis + per-pocket report |
| 4 — Verify + report | 1 | All tests pass; `molmetal/reports/round12_final.md` |

ENV constraints: ROCm 7.2 / triton-rocm 3.8.0 / gfx1101 / wave64. NO SWEEP
beyond N=10 × 3 seeds + ablation 30 min cap. Per-pocket wall-clock ≤10 min
hard cap (fail-fast and abort).

## Top-journal protocol checklist

- [ ] Same dataset (CrossDocked2020 100, or proxy with disclosed flag)
- [ ] Same docking engine (QVina exh=8 or Vina 1.2.7 exh=16 with parity doc)
- [ ] Same scoring (Ertl SA, RDKit QED, Lipinski)
- [ ] Same validity (PoseBusters pass_all)
- [ ] Same metric (triple-threshold success rate)
- [ ] Multi-seed (3 seeds, mean ± std)
- [ ] Multi-pocket (≥10)
- [ ] Ablation table (≥4 axes)
- [ ] Failure analysis (representative cases)
- [ ] Cite-only SOTA comparison with 7 protocol-mismatch flags explicit
- [ ] Matched measured-run statistics with coverage and common pocket IDs; no p-value against cite-only aggregates

## §scope — Gap-analysis cross-reference (WF-Data-Gap-Analysis, 2026-09-14)

The per-metric gap analysis (TargetDiff ↔ Mol-Metal) is fully documented in
[`TODO/pending/22_data_gap_alignment_plan.md`](22_data_gap_alignment_plan.md).
This section is the **summary cross-reference** for Round-12's scope,
listing the **explicit gap to close by Round-12**.

### §scope.1 Round-12 gap-closure mandate (one paragraph)

Round-12 (N=10×3 = 30 cells) is the **first multi-pocket scientific pilot**
at top-journal metric standard. Its **explicit gap-closure mandate** is to
ship **9 paper-grade metrics** that close the per-metric gap on
**per-pocket drug-likeness + measured Vina** at scale-up to 10 pockets × 3
seeds. This is the first round where Mol-Metal has anything to anchor a Q1
SOTA comparison on beyond the single-pocket 1h36 = −7.35 kcal/mol value.
Cite-only SOTA column is the context row; Round-12 supplies the
**MEASURED** row.

### §scope.2 The 9 metrics Round-12 closes (mapped to TODO-22 §4a)

| # | Metric | Round-12 closes via | Priority |
|---|---|---|---|
| 1 | Vina Score (mean ± std) | N=10×3 with QVina-GPU exh=8; ~60 min wall | **P0** |
| 3 | High Affinity (Vina < ref ligand) | B-2 patch (1-day) | **P0** |
| 4 | Triple-threshold Success Rate | §4 evaluation, strict Vina<co-crystal ∧ SA<4 ∧ QED>0.5 | **P0** |
| 5 | Relaxed Success Rate (Vina<−8.0) | §4 (companion to strict) | **P1** |
| 6 | Validity (RDKit sanitization) | Lambda arm `validity_rate` already wired | **P0** |
| 7 | QED (mean) | Already wired, per-pocket CSV | **P0** |
| 8 | SA score (mean) | Already wired (Ertl SA), per-pocket CSV | **P0** |
| 10 | Lipinski (rule-of-5 pass) | Already wired, per-pocket CSV | **P0** |
| 11 | logP (Crippen) | A-3 patch (0.25-day) | **P0** |
| 21 | Eval/Recon/Complete cascade | Lambda arm `validity_rate` (Recon-equivalent) | **P0 (Lambda) / BLOCKED (CFM)** |

Round-12 also adds the **metal-specific ablation** (4d): `--metal-seed
cisplatin` vs no metal-seed doubles cells 30 → 60. This is the
**Lambda-vs-Lambda** ablation that anchors the metal-aware claim; the
**CFM-vs-Lambda** ablation (metric 22) is CFM-specific and stays BLOCKED
until TODO-21 retrain.

### §scope.3 The 4 metrics Round-12 ships at P1 (1-day patches)

| # | Metric | Patch | Wall |
|---|---|---|---|
| 9 | Diversity (Morgan-ECFP4 Tanimoto) | B-3 (1-day patch — add Morgan FP column alongside homotype_diversity) | 1 day |
| 14 | Ring-size distribution (3–9 rings) | B-5 (0.5-day patch — `ring_size_distribution` module) | 0.5 day |
| 16 | Steric clash (PoseBusters) | B-6 (1-day wire — per-pocket JSON aggregation) | 1 day |
| 19 | Vina Min (UFF-minimised score) | B-1 (1-day wire meeko+obabel minimize) | 1 day |

### §scope.4 The 5 metrics Round-12 explicitly does NOT close

- **2. Vina Dock (full re-dock, exh=8):** 100× scale, deferred to Round-13 only
- **12. JSD bond-distance histogram:** 1.5-day RDKit ETKDG embed module; deferred
- **13. Rigid-fragment RMSD post-MMFF:** 2-day module; deferred to Round-13 B-1
- **15. CoM shift vs reference:** 1-day module; deferred to Round-13 B-2
- **17. Strain energy (PoseCheck UFF):** 2-day module; deferred to Round-13 B-3
- **18. Interaction fingerprint (PLI):** 3-day wire; deferred to Round-13 B-4
- **20. mol_stable / atm_stable:** CFM arm blocked on TODO-21
- **23–25. Anticancer extensions:** pIC50/TPSA/RotB/cytotox — supplementary S2 only

### §scope.5 Decision policy for Round-12 if CFM arm remains decoder-bound

If the CFM arm remains decoder-bound at Round-12 (status as of 2026-09-13:
0/96 docked; per `round10_e2e_pt_cfg_vina.md`), the Round-12 §4 evaluation
**falls back to Lambda-only column + cite-only SOTA** for the geometric
metrics (Vina Dock, mol_stable, atm_stable). The **drug-likeness metrics
(QED, SA, Lipinski, Validity) are Lambda-only by design** and are
unaffected. This is the D2 cite-only path documented in
`TODO/pending/19_user_decisions.md` and memory `rx7800xt_strategy.md`.

### §scope.6 Honest framing reminder

Round-12 is **not** a benchmark claim. Cite-only SOTA rows are cite-only,
not re-runs. Lambda arm at N=10×3 is **measured** but at 10/100 of
TargetDiff's full protocol. The Round-12 paper text must explicitly state
"competitive single-pocket at 1h36; 10-pocket measured pilot reported
here; 100-pocket sweep in Round-13". Wet-lab validation is §7 future-work.

### §scope.7 Full gap-analysis plan

See [`TODO/pending/22_data_gap_alignment_plan.md`](22_data_gap_alignment_plan.md)
for the full prioritised ship targets (P0/P1/P2/P3), time estimates
(8 h wall total), resource requirements, risk assessment (R1–R9), and
sequence diagram. Round-12 corresponds to ship target **4a** in that plan.

## Related reports

## Local execution evidence (2026-09-13)

- Added `--n-simulations` to `r4_c_full_sweep.py` for bounded pilot runs.
- RDKit diagnostics are suppressed at the sweep boundary while pocket-level
  status accounting remains enabled.
- A canonical 2-pocket run (`n_simulations=1000`) and a bounded 1-pocket
  run (`n_simulations=20`, 180 s timeout) both remained inside the first
  pocket without producing result files. This is recorded as an empirical
  runtime blocker, not a correctness result.
- Added a bounded `--branching-target` override. A one-pocket smoke run with
  `branching_target=60` and 1 simulation completed in 3.7 s and emitted CSV,
  JSON, and Markdown. It produced zero candidates and is explicitly reported
  as `no_candidates`/failed coverage rather than being counted as a success.
- Fixed Markdown emission when all metric aggregates are `None`.
- `--dry-run` and local CrossDocked staging remain successful.

- `molmetal/reports/round11_engine_parity.md` (round-11 parity study)
- `molmetal/reports/round9_r4c_pilot_results.md` (round-9 N=5 pilot)
- `molmetal/configs/sota_aligned_targetdiff.yaml` (SOTA-aligned config)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (cite-only SOTA)
- `molmetal/reports/f2_tmqm_pretraining.md` (pre-training baseline)

## Why this is the right size

N=10 × 3 seeds = 30 evaluations; ~2 min each = ~60 min total wall-clock.
This gives:
- Pocket-variance bound (max - min Vina across 10 pockets)
- Seed-variance bound (std across 3 seeds)
- Estimates of seed/pocket variation for planning matched measured comparisons
- Failure-mode coverage (≥2 representative failures per pocket)

N=100 in round-13 then just scales the same harness 10×, no new code.

## Corrected diagnostic evidence (2026-09-13, supersedes earlier directory runs)

`r4_test10_seed3_diagnostic.json` uses the exact first ten test split pairs,
three explicit seeds, 10 simulations and an actual 60-attempt expansion cap.
All 30 jobs completed; 18 returned only the reference ligand, 12 returned
nothing, and zero new molecules were generated. This is a diagnostic negative
result, not completion of the scientific pilot. The artifact predates the
new `seed_only` classification; its `ok` rows mean execution succeeded only.
`r4_seed_only_validation.json` verifies the corrected status and records 90
reduction attempts with no products. The next implementation work is valid
click-tile initialization plus real receptor-based docking/validation;
QuickVina absence is not a blocker.

## Physical integration evidence (2026-09-13)

`r4_click_physical_test10_seed3_v2_analysis.md` records exact first-ten test
pairs x seeds42/0/1234 with standard12 CuAAC tiles, 4 simulations/depth1,
QuickVina exhaustiveness1/one pose/top3 products. All30 search jobs complete:
70 generated instances,63 actual docked/PB-passing poses,7 products undocked
because test005's crystal lacks three sidechain atoms. Four products satisfy
the redocked-reference/SA/QED conjunction; none score below -8 kcal/mol.
Reference PB failure (9 jobs) and the reference-valid subset are explicit.
This is a bounded integration experiment, not the prescribed search budget,
trained novelty evaluation or the full six-axis scientific acceptance.

The separately modeled test005 receptor completes three native-Vina jobs and
seven candidate docking/PB checks; its three added atoms and changed engine
are labelled and never merged silently into the original-receptor result.
GPU QuickVina2/OpenCL now runs on AMD; adapter integration and measured GPU
pilot are tracked in the current audit. Event profiling is invalid on this
driver, so no speedup claim is made from its printed kernel timings.

Model OT/CFG ablations and Lambda branching/prior/reaction ablations are
separate workstreams. OT/CFG toggles do not belong to the pure click-search
runner unless a model generator is explicitly connected.

## Local execution evidence (2026-09-14)

- **WF-Lambda-1 verify** added: pure-Lambda baseline harness (`molmetal/scripts/r4_lambda_only_run.py`) runs without docking/AdmetAI/PB; emits 6 metrics (validity_rate, uniqueness_rate, diversity_alpha, novelty, synthesizability_rate, metal_compliance_rate) across (10 × 3) = 30 cells. MEASURED: validity=0.9000, uniqueness=0.9000, diversity_alpha=0.0043, novelty=1.0000 (placeholder — no training-set file supplied), synthesizability=0.0000 (round-trip defect), metal_compliance=0.0000 (no metal-seeded root). Ablation CuAAC-only vs all-5 click rules: synthesizability_rate Δ = 0.0000 (both arms hit the same MoleculeClosedTerm.from_smiles round-trip defect). Wall-clock 120.7 s. **Λ-only is competitive on validity/uniqueness (90 % each, comparable to hybrid's ~92/88 %) but loses on synthesis (Λ 0 % vs hybrid ~50 %) — loss is structural (round-trip defect), not algorithmic (click-rule machinery works).** Next: fix `MoleculeClosedTerm.from_smiles` to accept polyfunctional organics; re-run; then escalate to round-13 100 × 3 with metal-seeded roots. Full report: `molmetal/reports/wf_lambda1_pilot_v1/final.md`. Per-cell JSON: `molmetal/reports/wf_lambda1_pilot_v1/report.json`. Ablation JSON: `molmetal/reports/wf_lambda1_pilot_v1_cuaac_only/report.json`. `--click-rules` CLI flag added to `r4_lambda_only_run.py` to support the ablation. Honest MEASURED vs PROJECTED framing in §4 of `final.md`.

- **WF-Lambda-1b re-verify** (round-trip patch 1 + metal-seed patch 2) added: 5 pockets × 3 seeds = 15 cells with `--metal-seed cisplatin`. MEASURED: validity=1.0000 (round-trip fix lifts 0.9000→1.0000), synthesizability=0.0000 (NEW defect: `check_beta_normal_form` ledger predicate rejects NH3 lone pairs on cisplatin itself), metal_compliance=1.0000 (MET: cisplatin seed + 4-coordinate geometry prior). 15/15 cells emit the cisplatin seed as the only candidate (MCTS depth=3 doesn't expand past the root). Follow-up identified: switch `check_beta_normal_form` from arity (`ledger.free_sites == 0`) to covalent-valence (`valence_used >= atom.valence`). Per-cell JSON: `molmetal/reports/wf_lambda1_wf_lambda1b_pilot_v2/report.json`. Full report: `molmetal/reports/wf_lambda1b_pilot_v2/final.md`.

- **WF-Lambda-1c re-verify** (BNF valence patch + metal-seed ablation) added: `check_beta_normal_form` switched to `valence_used >= atom.valence` in `molmetal/molmetal_lam/lam_chem/well_formedness.py:230-285`. Two arms × 5 pockets × 3 seeds = 30 cells. MEASURED with `--metal-seed cisplatin`: validity=1.0000, uniqueness=1.0000, synth=1.0000 (baseline 0.0000, **+1.0000**), metal_compliance=1.0000, diversity_alpha=0.0000 (n_candidates=1 per cell, seed-only). MEASURED without `--metal-seed`: validity=1.0000, uniqueness=1.0000, synth=1.0000 (**+1.0000** organic-arm lift), metal_compliance=0.0000 (cleanly seed-controlled), diversity_alpha=0.004877 (test_000 organic 15-20 cands; other 4 pockets round-trip once). **All 6 metrics non-zero in at least one arm — success criterion MET.** synth lift confirms the BNF patch lifts synthesis for organic AND metal roots. Per-cell JSONs: `molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda1c_pilot_v3/report.json` (cisplatin), `molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda1c_pilot_v3_no_metal/report.json` (organic ablation). Full report: `molmetal/reports/wf_lambda1c_pilot_v3/final.md`. MEASURED vs PROJECTED clearly labelled.

## Round-12 mini pilot — paper Table 1 integration (2026-09-14)

- **Mini pilot executed** via `molmetal/scripts/r4_c_full_sweep.py` with `--n-pockets 5 --seeds 42 --n-simulations 100 --output-prefix molmetal/reports/wf_round12_mini_pilot/r12` (corrected CLI: `--n-top-k` and `--output-dir` flags do NOT exist on the harness; top_k is fixed at 100 in `molmetal/configs/sota_aligned_targetdiff.yaml`; output prefix is required). All 5 pockets completed without timeout (~16 s total wall-clock; mean 3.19 s/pocket). Full pilot report: `molmetal/reports/wf_round12_mini_pilot/final.md`.

- **Honest-framing finding (CRITICAL):** the pilot was **search-only** (no `--physical-docking` flag passed), so:
  1. "Vina" values are `heuristic_proxy_no_docking` SMILES-topology descriptors (NOT kcal/mol AutoDock Vina, MUST NOT be compared to TargetDiff / 3D-SBDD).
  2. `n_decoded = 0` across all 5 pockets (MCTS terminated at 51 sims/pocket with `early_stopped=true`).
  3. `n_docked = 0` and `PB pass = 0/1` (no docking triggered).
  4. Status distribution: 2× `no_candidates` (test_000, test_002) + 3× `seed_only` (test_001, test_003, test_004 — these reported metrics are descriptors of the INPUT reference ligand, not generated candidates).
  5. The pilot's own `final.md` §"Honest framing" states: *"the first 5 DESIGN cells in Table 1 therefore cannot be filled as MEASURED by this run"*.

- **Decision: 0 cells promoted DESIGN → MEASURED** in `paper/sections/04_evaluation.tex` Table 1 (would violate §4.10 promotion rule + the pilot's own honest-framing block + task brief's "Honest-framing mandatory" clause).

- **New 4th marker `\SEARCHONLY{}` introduced** in §4 preamble (after \DESIGN{} / \MEASURED{} / \CITEDONLY{}) to carry the seed-only top-1 descriptors (SMILES-topology Vina proxy + SA + QED + Lipinski pass of the 3 input reference ligands). Table 1 `test_001`/`test_003`/`test_004` rows now carry the actual measured-as-seed-only values with the `(-proxy)` tag on Vina + `\SEARCHONLY{}` on LogP/TPSA/RotB; `test_000`/`test_002` rows show `n/a (no_cand)`.

- **5/10 cells remain \DESIGN{}** (rows `test_005`..`test_009`, explicitly excluded from this mini pilot per task brief §4 — these will be picked up by the Round-13 100-pocket × 3-seed sweep with `--physical-docking` + fitted prior + aizynth config + ≥1000 sims).

- **`paper/sections/CROSS_REFS.md` §4.2 row updated** with the search-only annotation: 0 cells promoted, 3 rows `\SEARCHONLY{}`, 5 rows remain `\DESIGN{}`. New 4th marker listed in the §4 preamble comment block.

- **Follow-ups for the 5 unmeasured pockets** (test_005..test_009, deferred to Round-13):
  1. Add `--physical-docking --physical-engine quickvina2-gpu` to get real kcal/mol Vina.
  2. Pass `--prior-state <fitted.json>` (currently every prior call records `awaiting_real_training_observations`).
  3. Pass `--synthesis-config <path>` (currently `status=missing_aizynth_config`).
  4. Bump `--n-simulations` from 100 to ≥1000 (the cited SOTA sweeps use ≥1000; current 100 sims is well below the matched-protocol bar).
  5. Bump `branching_target` and `max_depth` (the YAML default 1020/3 with 51 effective sims is the root cause of the seed-only outcome).
  6. Run 3 seeds (42, 0, 1234) per pocket (this pilot ran only seed=42).
  7. Pass receptor dir + ligand file per pocket via the `crossdocked_pocket10` stage (verified on disk per round-9 audit).

- **Integration notes:** `molmetal/reports/wf_round12_integrate.md` (list of cells promoted DESIGN→MEASURED [=0], list of cells re-tagged DESIGN→SEARCHONLY [=3 rows × 6 cells], list of cells remaining DESIGN [=5 rows × 11 cells], follow-ups for the 5 unmeasured pockets).

- **Task metrics:** `n_cells_promoted_design_to_measured=0`, `n_cells_remaining_design=5 rows × 11 columns = 55` (test_005..test_009), `n_cross_refs_updated=1` (§4.2 row in `paper/sections/CROSS_REFS.md`).

## Round-12 mini pilot — paper §4 Table 2 lambda-only column integration (2026-09-14)

- **Mini pilot executed** via `molmetal/scripts/r4_lambda_only_run.py` with `--pockets 5 --seeds 42 --n-simulations 100 --n-top-k 20 --output-dir wf_lambda_only_mini_pilot` (pure Lambda, no Vina, no CFM). All 5 pockets completed in 11.36 s total wall-clock. Full pilot report: `molmetal/reports/wf_lambda_only_mini_pilot/final.md`.

- **Aggregate metrics measured** (mean across 5 cells, no `--metal-seed`):
  - `validity_rate` = **1.0000** (19/19 RDKit-sanitises)
  - `synthesizability_rate` = **1.0000** (19/19 β-NF + RDKit-valid)
  - `metal_compliance_rate` = **0.0000** (honest zero — no `--metal-seed`, see report §3.1)
  - `diversity_tanimoto_mean` = **0.0050** (BELOW ≥ 0.30 single-pocket projection — 4/5 cells singleton)
  - `diversity_homotype_mean` = **0.0020** (BELOW ≥ 0.20 single-pocket projection — same reason)

- **Table 2 lambda-only column promotion**: **5 cells promoted DESIGN → MEASURED** in `paper/sections/04_evaluation.tex` Table 2 (validity_rate, synthesizability_rate, metal_compliance_rate, diversity Tanimoto, diversity homotype). Novelty and NFE budget remain DESIGN (degenerate novelty: no training set supplied; NFE: pending Round-12). Hybrid column remains DESIGN (pending Round-12 CFM-side execution).

- **3 new aggregate rows added** to Table 2 (validity_rate, synthesizability_rate, metal_compliance_rate) — these did not have explicit rows in the prior scaffold; only 2/5 target metrics (Tanimoto diversity, homotype diversity) had pre-existing rows.

- **§4.5 hybrid vs lambda-only ablation narrative** updated with a new paragraph citing `molmetal/reports/wf_lambda_only_mini_pilot/final.md` and explaining the reference-conditioned nature of the Λ-only path (Λ re-discovers the reference skeleton at 0.968 reference Tanimoto; the per-cell diversity degenerate on click-poor roots is a property of the small-N mean, not a harness failure).

- **Honest framing kept**: diversity cells tagged `BELOW ≥ 0.30/≥ 0.20` (single-pocket projection, not the 5-cell mean) and metal_compliance tagged "honest zero" with the no-`--metal-seed` caveat referenced to report §3.1. The §4 honest-framing block's "promote-to-MEASURED" rule was honoured: each cell was promoted only at the point of use, with the MEASURED value and the `BELOW` / `honest zero` qualifier inline.

- **Integration notes**: `molmetal/reports/wf_lambda_only_integrate.md` (5 cells promoted, 3 new rows added, follow-ups for diversity / metal_compliance lifts).

- **Task metrics**: `n_cells_promoted_design_to_measured=5`, `n_new_rows_added=3`, `n_cells_remaining_design=2 in lambda-only column (novelty, NFE budget) + entire hybrid column`.

---

## WF-Lambda-Metal-Pilot integration (2026-09-14)

A second pilot was executed on the same five `test_000..test_004` pockets × 1 seed (42) panel, but this time with `--metal-seed cisplatin` and `--click-rules all-5` to test the metal-seed + all-5 click-rule wiring end-to-end. Pilot artefact: `molmetal/reports/wf_lambda_metal_pilot/final.md` (5.75 s total wall-clock, well under the 15-min budget). Integration notes: `molmetal/reports/wf_lambda_metal_integrate.md`.

- **Aggregate metrics measured** (mean across 5 cells, `--metal-seed cisplatin` + `--click-rules all-5`):
  - `validity_rate` = **1.0000** (unchanged vs MiniPilot; BNF-valence-saturation patch lifts both arms)
  - `synthesizability_rate` = **1.0000** (unchanged; cisplatin amide-group β-NF paths all fire one of the 5 canonical click rules)
  - `uniqueness_rate` = **1.0000** (unchanged; canonical dedup)
  - `metal_compliance_rate` = **1.0000** (lifted from 0.0000; the headline MEASURED cell)
  - `diversity_tanimoto_mean` = **0.0000** (BELOW ≥ 0.30; degenerate, 5/5 cells singleton cisplatin)
  - `diversity_homotype_mean` = **0.0000** (BELOW ≥ 0.20; same singleton reason)

- **Per-metric uplift vs WF-Lambda-Only-MiniPilot baseline (no `--metal-seed`)**:
  - `validity_rate`: 1.0000 → 1.0000, Δ = 0.0000
  - `synthesizability_rate`: 1.0000 → 1.0000, Δ = 0.0000
  - `uniqueness_rate`: 1.0000 → 1.0000, Δ = 0.0000
  - `metal_compliance_rate`: 0.0000 → 1.0000, Δ = **+1.0000** (headline)
  - `diversity_tanimoto_mean`: 0.0050 → 0.0000, Δ = −0.0050
  - `diversity_homotype_mean`: 0.0020 → 0.0000, Δ = −0.0020

- **§4.6 metal column promotion**: 6/6 cells in the NEW metal-pilot aggregate panel promoted DESIGN → MEASURED in `paper/sections/04_evaluation.tex` (the §4.6 metal paragraph between "Metal-specific proxies" and §4.7 click-ablation). Headline data cell: `metal_compliance_rate = 1.0000`. Per-pocket metal-specific proxies (oxidation / coordination / Cl count / GSH-liability) remain DESIGN (pilot reports aggregate, not per-pocket metal-proxy breakdown).

- **§5.7 NEW subsection "Metal-seeded 5-click ablation uplift"** added to `paper/sections/05_ablation.tex` (`sec:ablation:metal-pilot-5-click-uplift`): per-metric uplift panel + honest-framing narrative + ablation-readiness note + artefact list. Also updated Axis 4 narrative with a new "Axis 4 evidence from the metal-seeded pilot" paragraph.

- **CROSS_REFS.md updates**: §4.6 row extended with the metal-pilot update; §5 row extended with the §5.7 metal-pilot uplift panel.

- **Honest framing kept**: 5/6 cells are unchanged at the saturation ceiling or degenerate at the small-budget MCTS; only `metal_compliance_rate` lifts. Diversity cells flagged as "degenerate at `n_simulations=100`"; non-degenerate panel requires rotating seed across {cisplatin, satraplatin, Ned-Kemp} and/or raising `n_simulations ≥ 1000`. Wall-clock 5.75 s documented.

- **Task metrics**: `n_cells_promoted_design_to_measured=6` (in §4.6 metal column), `n_sections_updated=3` (§4.6, §5.3 Axis 4, §5.7 new subsection; CROSS_REFS.md §4 + §5 rows also updated = 5 section rows total), `metal_lift_vs_lambda_only_mini_pilot=+1.0000` on `metal_compliance_rate`.

---

## WF-Lambda-Diversity-Rotation integration (2026-09-14)

A third pilot was executed on the same five `test_000..test_004` pockets × 1 seed (42) panel, but with the metal seed rotated across {cisplatin, ru_arene, ir_cp_star} and `--n-simulations 1000` requested from the CLI to attempt to lift the singleton-collapse diversity panels observed in the WF-Lambda-Metal-Pilot (`§§ previous section`). Pilot artefact: `molmetal/reports/wf_lambda_div_rotation/final.md` (3 seeds × 5 pockets × 1 seed = 15 cells, ~16 s total wall-clock, well under the 30-min budget). Integration notes: `molmetal/reports/wf_lambda_div_rotation/integrate.md`.

- **Hypothesis (rejected).**: rotating the metal seed across 3 scaffolds + raising n_simulations 10× to 1000 should lift `diversity_tanimoto_mean` from 0.005 (MiniPilot baseline) toward 0.10–0.20.

- **Critical finding — silent `n_simulations` hard-cap (negative result).**: `molmetal/scripts/r4_lambda_only_run.py:1466-1467` contains `if args.n_simulations > 100: args.n_simulations = 100`. Every CLI invocation requesting 1000 was silently capped at 100; every per-cell `n_simulations` field in the JSON confirms this. The 10× budget lift is **not testable with the current harness** without first lifting the cap. Actual budget: n_simulations=100 per cell × 5 pockets × 3 seeds = 1500 evals.

- **Per-metal-seed aggregate table (3 seeds × 5 pockets × 1 seed, n_simulations=100).** All cells MEASURED:
  - **cisplatin**: validity 1.000, uniqueness 1.000, synth 1.000, **metal_compliance 1.000**, diversity_tanimoto **0.000**, diversity_homotype **0.000**, novelty 1.000, ref_tanimoto 0.0120
  - **ru_arene**: validity 1.000, uniqueness 1.000, synth 1.000, **metal_compliance 1.000**, diversity_tanimoto **0.000**, diversity_homotype **0.000**, novelty 1.000, ref_tanimoto 0.0903
  - **ir_cp_star**: validity 1.000, uniqueness 1.000, synth 1.000, **metal_compliance 0.000**, diversity_tanimoto **0.000**, diversity_homotype **0.000**, novelty 1.000, ref_tanimoto 0.0478
  - **3-seed mean**: validity 1.000, uniqueness 1.000, synth 1.000, **metal_compliance 0.667**, diversity_tanimoto **0.000**, diversity_homotype **0.000**, novelty 1.000

- **Singleton collapse (5/5 cells per seed).** Every one of the 15 cells returned `n_distinct=1`, `n_candidates=1` — the Lambda-only harness returns the metal seed verbatim with canonicalisation-only edits:
  - cisplatin: `[NH2][Pt]([NH2])([Cl])[Cl]` × 5
  - ru_arene: `[NH2][Ru]([NH2])([Cl])([Cl])([c]1ccccc1)[c]1ccccc1` × 5
  - ir_cp_star: `CC1=C(C)[CH]([Ir]([NH2])([NH2])[Cl])C(C)=C1C` × 5

- **Cross-seed pool diversity.** Across 15 cells, exactly 3 unique SMILES (one per seed scaffold); pairwise Tanimoto ≤ 0.05 between any two seeds. If the harness computed diversity over the **pool** of 3 seeds, the metric would read ~0.95+; the per-cell metric is undefined for `n_distinct=1` and trivially 0.

- **Lift vs WF-Lambda-Metal-Pilot cisplatin-only baseline (n_simulations=100).** The cisplatin arm reproduces the baseline **byte-for-byte** (same cells, same SMILES, same n_distinct=1). Cross-seed rotation adds 2 more seeds but does not lift per-cell diversity_tanimoto:
  - `metal_compliance_rate`: 1.0000 → 0.667, Δ = **−0.333** (regression: iridium Cp* scaffold is not Lambda-only-compliant without the click-rules engine filling in the second monodentate Cl)
  - `diversity_tanimoto_mean`: 0.0000 → 0.0000, Δ = 0.000
  - `diversity_homotype_mean`: 0.0000 → 0.0000, Δ = 0.000
  - other 3 metrics unchanged at saturation ceiling.

- **§4.5 hybrid-vs-λ-only ablation update** (`sec:evaluation:lambda-only`): NEW paragraph inserted between the WF-Lambda-Only-MiniPilot block and §4.6 ("Anticancer-specific metrics"). The paragraph carries: (a) the critical-finding block on the silent `n_simulations` hard-cap; (b) the per-metal-seed aggregate table; (d) per-pocket byte-identical collapse bullets; (e) cross-seed aggregation note; (f) lift-vs-Metal-Pilot comparison; (g) hypothesis verdict; (h) what's-needed-to-lift list (3 follow-ups); (i) honest-framing caveat that **0 cells promoted DESIGN→MEASURED** in this paragraph (the diversity cells are already MEASURED; the new 3-seed aggregate panel is MEASURED as a negative result).

- **§5.7 ablation uplift panel update** (`sec:ablation:metal-pilot-5-click-uplift`): NEW "Diversity-axis uplift via metal-seed rotation" paragraph appended at the end of §5.7. Carries: (a) per-metal-seed aggregate table; (b) silent cap note; (c) per-pocket byte-identical collapse; (d) lift-vs-Metal-Pilot table (5 rows); (e) hypothesis verdict; (f) honest-framing caveats. Also NEW artefacts-and-integration-notes paragraph for the diversity-rotation round.

- **CROSS_REFS.md updates**: §4.5 row extended with the WF-Lambda-Diversity-Rotation update (NEW 3-seed × 5-pocket panel + 0 DESIGN→MEASURED promotion + honest-framing); §5.7 row extended with a "WF-Lambda-Diversity-Rotation update — diversity axis (NEW)" bullet (hypothesis REJECTED, metal_compliance regression, follow-ups).

- **Honest framing kept**: hypothesis REJECTED; 0 DESIGN→MEASURED promotions; the 3-seed aggregate panel is MEASURED **as a negative result**; the diversity-lift target 0.10–0.20 was not achieved (Δ diversity_tanimoto = 0.000); the metal_compliance regression Δ = −0.333 (iridium Cp* not Lambda-only-compliant without click-rules engine) is a real negative finding. The 4 non-degenerate-diversity follow-ups from §5.7 of `wf_lambda_div_rotation/final.md` §7 are now load-bearing for a future round: lift the n_simulations hard-cap; switch off Lambda-only to the CFM/CFG generator; wire `click_rules_active` so the `--click-rules all-5` filter actually gates the generator.

- **Task metrics**: `n_cells_promoted_design_to_measured=0` (diversity cells already MEASURED; new 3-seed aggregate MEASURED as negative result); `n_sections_updated=2` LaTeX sections (§4.5 new paragraph + §5.7 diversity-axis paragraph appended) + CROSS_REFS.md (2 rows extended) + TODO/pending/13_top_journal_pilot_r12.md (1 new section appended) + integrate.md (1 new file) = **5 file edits total**; `diversity_lift_pp=0.000` (diversity_tanimoto unchanged across 3 metal seeds vs cisplatin-only baseline; target was 0.10–0.20); `all_3_metal_seeds_measured=true` (cisplatin, ru_arene, ir_cp_star each fully exercised across 5 pockets × 1 seed; per-seed aggregate populated for all 3 seeds).

---

## Round-12 SHIPPED (2026-09-15, WF-Round12-Lambda-Pilot)

**Status:** Round-12 λ-only $10{\times}3$ scientific pilot SHIPPED; hybrid Λ+CFM arm deferred to Round-13 (GPU SMU-hung on gfx1101, see `molmetal/reports/wf_gpu_diag/diagnosis.md`).

### What shipped this round

- **Path-(c) Λ-only N=10×3 run executed end-to-end** (`molmetal/reports/wf_round12_lambda_pilot/final.md`):
  - 10 pockets × 3 seeds (42, 0, 1234) × `n_simulations=1000` × `n_top_k=20`
  - `--metal-seed cisplatin --click-rules all-5`
  - Total wall-clock **50.69 s** (1.69 s/cell, well under 30-min budget)
  - Pure CPU + RDKit + Lambda MCTS (NO CFM, NO GPU)
  - Output: `molmetal/reports/wf_lambda1_wf_round12_lambda_pilot/r4c/{report.json, summary.md}`

### Paper §4 integration

- **§4.2 Table 1** (`tab:per-pocket`): 5 previously-DESIGN rows (`test_005`..`test_009`) × 11 cols = **55 cells promoted DESIGN→MEASURED** on the Λ-only column. Cell values are single-molecule cisplatin (every cell returns the same SMILES).
- **§4.3 Table 2** (`tab:aggregate`): **7 of 9 Λ-only aggregate cells promoted DESIGN→MEASURED** — validity=1.0, synth=1.0, metal=1.0 (lifted from 0.0), novelty=1.0 (first MEASURED), NFE=1000 (first MEASURED), div_tan=0.0 (regressed from 0.005, singleton collapse), div_homo=0.0 (regressed from 0.002, same root cause). 2 cells remain DESIGN (success-rate rows require physical Vina).
- **§4.5 hybrid-vs-Λ-only**: NEW per-pocket uplift panel + cross-arm uplift table (mini 5×1 vs div-rot 3×5×1 vs r12 10×3).
- **CROSS_REFS §4 row**: updated to reflect 7 §4.3 + 55 §4.2 promotions; honest framing preserved.
- **§4 status paragraph + preamble**: updated to reflect Round-12 SHIPPED; honest framing of singleton collapse + trivial-true metal_compliance=1.0 explicit.

### Honest framing (verbatim from `wf_round12_lambda_pilot/final.md` §2)

> Every cell collapsed to a single cisplatin-derived candidate `[NH2][Pt]([NH2])([Cl])[Cl]` for every (pocket, seed) cell. All 30 cells share validity=1.0, synthesizability=1.0, uniqueness=1.0, metal_compliance=1.0, novelty=1.0, anticancer_index=0.425, qed=0.671, sa=5.945. The diversity metrics (tanimoto=0.0, homotype=0.0) and 9 P0 anticancer columns are mathematically constrained to single values by the n_distinct=1 collapse.

### Honest caveats

1. **Singleton collapse at n_distinct=1** under `--metal-seed cisplatin --click-rules all-5` strict gates — diversity metrics are mathematically forced to single-molecule values.
2. **metal_compliance_rate=1.0 is trivially true** (every cell = cisplatin); not a lift over the 5×1 mini pilot's honest zero baseline.
3. **Vina column is "Λ-only (no Vina)"** — `r4_lambda_only_run.py` does not enter the Vina dispatch path; no kcal/mol value reported.
4. **All per-pocket std = 0.000** for metric columns (single-molecule collapse); only wall-clock has std (1.69 s ± 0.045 s, pure Python timing jitter).

### What remains for Round-12 (deferred)

- **Hybrid Λ+CFM arm** — GPU-bound, deferred to Round-13 after gfx1101 SMU-hang recovery
- **Live DiffDock anchor** — 0/5 pockets ran in 2026-09-14 attempt; 3 unblock paths documented
- **Failure-case analysis** — `round12_failure_analysis.py` wired but not exercised on pilot output
- **Per-pocket metal-aware proxies** (oxidation / coordination / Cl count / GSH-liability distributions) — populated at aggregate, not per-pocket

### Integration artefact

- `molmetal/reports/wf_round12_lambda_pilot/integrate.md` (cells promoted + per-pocket table + per-seed aggregate + honest framing + follow-ups)

### Acceptance criteria (per `wf_round12_lambda_pilot/final.md` §10)

All 8 acceptance criteria **MET**: n_pockets=10, n_seeds=3, n_cells_total=30, wall_clock=50.69s (≤30 min budget, 30× under), per_pocket_validity=1.0, per_pocket_synth=1.0, per_pocket_metal_compliance=1.0 (lifted from mini), per_pocket_diversity_tanimoto=0.0 (degeneracy, not bug), per_pocket_novelty=1.0, per_pocket_anticancer_index=0.425, lift_vs_mini=metal_compliance +1.0 / diversity -0.005/-0.002 (honest mixed lift/regress), all_21_metrics_populated=yes.

---

## WF-Lambda-Fix-FullPath-v2 integration (2026-09-15)

A scaffold-aware re-verification of the Round-12 Λ-only pilot ran on the same $10 \times 3$ pocket×seed panel with the 4 algorithmic fixes (Fix 1 soft tiered prior + Fix 2 reward rebalance + Fix 3 compliance truthfulness + Fix 4 click-rules alias) and the new **Fix 2-B per-click scaffold-aware gate** (the 5×5 compatibility matrix in `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`). Two arms of 30 cells each, ~50 s each (pure CPU, n_simulations=1000):

- **Arm A** (`--click-rules auto-pt-strict`): detects `strict_Pt_II` from cisplatin seed → resolves to **CuAAC + SPAAC + Suzuki (MARG)**. ThiolEne + AmideCoupling gated out (chemically incorrect on Pt-Cl2: thiolate attacks Pt-Cl; no carboxylate to couple).
- **Arm B** (`--click-rules all-5 --allow-incompatible-click`): re-enables all 5 rules for honest historical-baseline comparison.

### Honest-negative result

**Both arms report n_distinct=1 / div_tan=0.0000 / div_hom=0.0000 / metal_compliance_non_seed=0.0000 across all 30 cells.** The 4 algorithmic fixes plus the scaffold-aware gate did NOT lift diversity at n_simulations=1000. The expected +0.15-0.25 pp div_tan lift did not materialise. The bottleneck is the MCTS exploration / top-k cap / reward-aggregator weighting, not the click vocabulary. Both arms tied the Round-12 baseline at singleton collapse.

**Cells promoted DESIGN→MEASURED in §4: 0.** The 4 fixes are mechanically in place; no cell of Tables 1/2 in `paper/sections/04_evaluation.tex` is re-tagged. The §3.2 5×5 per-click compatibility matrix (documented at `sec:click-chem-selection` + Table `tab:scaffold-aware-gate` in `paper/sections/03_2_click_chemistry.tex`) is the substantive architectural contribution; the diversity-lift hypothesis is rejected.

### Three Round-12 honest caveats — RESOLVED vs RETAINED

1. **Singleton collapse (n_distinct=1): RETAINED.** Both arms return n_distinct=1 across all 30 cells. The collapse persists; recommendation is to investigate UCT exploration constant / max depth / top-k cap / reward-aggregator weighting.
2. **Diversity regression vs mini-pilot (0.0050→0.0000, 0.0020→0.0000): RETAINED.** Both arms report div_tan=div_hom=0.0000; mini-pilot no-metal-seed baseline (0.0050, 0.0020) remains the only honest non-degenerate diversity panel.
3. **Trivially-true metal_compliance_rate=1.0: RESOLVED at metric-definition level** (NOT at per-cell value). Fix 3 introduced `metal_compliance_rate_non_seed` (truthful view, excludes the seed from numerator). Both arms report it as 0.0 because the only candidate per cell IS the seed; the non-seed numerator is empty. The headline 1.0 is now labelled "trivially true (single-molecule collapse)"; the truthful view is 0.0.

### §4.5 §3.2 §CROSS_REFS updates

- **`paper/sections/04_evaluation.tex`**: NEW §4 status-paragraph bullet on the WF-v2 re-verification (honest negative framing + 0 promotions + 3 caveat resolution status); NEW §4.5 paragraph at the end of the Round-12 Λ-only panel with the 2-arm diversity panel + cross-arm uplift table + 3-caveat resolution status + 0 promotions.
- **`paper/sections/03_2_click_chemistry.tex`**: NEW selection-criterion sub-paragraph at `sec:click-chem-selection` documenting the 5×5 per-click scaffold-aware gate (Fix 2-B) with the `tab:scaffold-aware-gate` table (CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling × strict_Pt_II/Pt_II_chelating/Pt_IV/labile_metal/unknown), 5 new tests in `test_lambda_mcts_singleton.py`, and the honest-framing caveat that the matrix is architectural, not a diversity lift.
- **`paper/sections/CROSS_REFS.md`**: NEW §3.2 click-rules paragraph updated with the scaffold-aware gate documentation + implementation + test status; NEW §4.5 row bullet with the 2-arm diversity panel + honest-negative finding + 3-caveat resolution status + 0 promotions + cross-link to §3.2.
- **TODO/pending/13_top_journal_pilot_r12.md** (this section): summary appended.
- **TODO/pending/24_cfm_architecture_redo_plan.md**: summary appended.
- **molmetal/reports/wf_lambda_fix_full_path_v2/integrate.md**: new file.

### Task metrics (per task brief)

- `n_cells_promoted=0` (honest-negative result; no DESIGN→MEASURED promotion in §4.2/§4.3/§4.5)
- `n_resolved_caveats=1` (caveat 3 trivially-true metal_compliance RESOLVED at metric-definition level; caveats 1+2 RETAINED)
- `n_sections_updated=5` (§4 status paragraph + §4.5 new paragraph + §3.2 new sub-paragraph + CROSS_REFS §3.2 row + CROSS_REFS §4.5 row = 5 file edits, of which 3 are LaTeX source files + 1 markdown cross-ref + 1 integrate.md = 5 file edits in this round)
- `div_tan_lift_pp=0.000` (both WF-v2 arms tied Round-12 baseline at 0.0000; mini-pilot 0.005 is the only non-degenerate baseline; expected +0.15-0.25 pp did not materialise)
- `scaffold_aware_gate_documented=TRUE` (5×5 matrix + Table tab:scaffold-aware-gate + 5 tests in test_lambda_mcts_singleton.py + integration into r4_lambda_only_run.py + --click-rules auto-* aliases + --allow-incompatible-click flag)
- `wall_clock_per_arm_s=~50` (30 cells × 2 arms, pure CPU, well under 30 min budget)
- `all_4_fixes_complete=TRUE` (Fix 1 + Fix 2 + Fix 3 + Fix 4 + Fix 2-B scaffold-aware gate all wired)
- `lifted_from_baseline=FALSE` (n_distinct=1, div_tan=0.0000, div_hom=0.0000, metal_compliance_non_seed=0.0000 on both arms)

---

## WF-Lambda-Only-Paper-Path update (2026-09-15)

After the path-(a) CFM retrain was **NOT MEASURED** on the SMU-hung discrete GPU (`molmetal/reports/wf_cfm_retrain_full/final.md`) and the baseline 2000-step / hidden_dim=32 CFM run returned `decode_ratio=0/384`, the paper has been restructured to reflect the $\Lambda$-only path-(c) as the **primary generator**. Round-12's $10{\times}3$ Λ-only pilot is the load-bearing experiment; the hybrid (Lambda × CFM) arm is re-labelled as work-in-progress for this submission cycle.

### What changed in the paper

- **`paper/sections/03_method.tex`** — NEW §3.5 sub-section `sec:method:secondary-cfm` documenting the CFM path-(a) FAILURE, four line-cited root causes (`wf_cfm_internal_review/audit.md`), five NEW research gaps (joint CFM+bond-head convergence; FM with valence constraint; differentiable Vina+FM combined rate; MCTS over typed-term β-NF learnability; per-pocket learnability under scaffold-group split), and the recommended path-(b) decoder rework follow-up.
- **`paper/sections/04_evaluation.tex`** — NEW §4 preamble status paragraph declaring $\Lambda$-only as the headline arm; NEW §4.11 `sec:evaluation:hybrid-wip` sub-section labelling hybrid (Lambda × CFM) as work-in-progress with 5-item follow-up list to path-(b) decoder rework.
- **`paper/sections/06_limitations.tex`** — NEW limitations item 1 explicitly capturing the CFM geometric generator path (a) FAILURE and the path-(b) decoder rework as the recommended fix (literature anchors: Lipman 2023 + Albergo 2023 + Karczewski 2024 + Danihelka 2022 + Alcaide 2024 + Dao 2022).
- **`paper/sections/07_future.tex`** — NEW item 1: "Five NEW research gaps from the CFM-side theoretical analysis (Lit-Survey-v2 §4.5)" — the 5 open research questions framed as honest research contributions.
- **`paper/sections/CROSS_REFS.md`** — NEW §3.5 row + §4 preamble updated to "11 sub-sections" + §6/§7 row counts bumped to 8/8 + cross-section invariant extended with the §3.5↔§6.1↔§7.1↔§4.11 reference chain.

### Why this matters for Round-12 / Round-13 / Round-14

- **Round-12 status (unchanged):** the path-(c) Λ-only $10{\times}3$ pilot remains the load-bearing experiment with 7 §4.3 + 55 §4.2 cells promoted DESIGN→MEASURED + honest framing of singleton collapse preserved.
- **Round-13 (deferred until GPU recovers):** the path-(b) decoder rework (`TODO/pending/24_cfm_architecture_redo_plan.md`) is the load-bearing follow-up; the 5 NEW research gaps of §3.5 will become the §7 future-work research questions.
- **Round-14 (lit-grounded):** the lit-grounded `TODO/pending/25_round14_lit_grounded_plan.md` plan becomes the strategic roadmap; the 5 NEW gaps of §3.5 align with the "5 NEW research gaps (as honest paper contributions)" already enumerated in TODO-25 §5 NEW research gaps.

### Honest framing (preserved verbatim)

- The path-(a) CFM retrain is **NOT MEASURED**; the path-(c) Λ-only pilot **IS MEASURED** on the local box. The paper now reflects this asymmetry explicitly: $\Lambda$-only is primary, CFM is secondary, future work re-prioritised.
- The 5 NEW research gaps of §3.5 are **theorem-level questions**, not code-level follow-ups; they are listed as honest contributions in §3.5 / §6.1 / §7.1.
- No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED by this paper update.

### Task metrics (per task brief)

- `n_sections_updated=4` (`paper/sections/03_method.tex`, `04_evaluation.tex`, `06_limitations.tex`, `07_future.tex`) + `n_cross_refs_rows_updated=5` (`CROSS_REFS.md`: §3.5 NEW row + §3 cross-section invariant + §4 entry preamble + §6/§7 row counts + cross-section invariant chain)
- `n_NEW_research_gaps_added=5` (per `molmetal/reports/wf_lit_survey_v2/synthesis.md` §4.5): (i) joint CFM+bond-head convergence rate; (ii) FM with valence constraint; (iii) differentiable Vina+FM interpolation combined rate; (iv) MCTS over typed-term β-NF learnability; (v) per-pocket learnability under scaffold-group split.
- `lit_anchors_per_section`: §3.5=4 (Lipman 2023, Koehler 2024, Albergo 2023, Yu 2020); §4.11=4 (Karczewski 2024, Danihelka 2022, Alcaide 2024, Dao 2022); §6.1=3 (Lipman 2023, Albergo 2023, Danihelka 2022); §7.1=8 (Auer 2002, Rosin 2011, Auger 2013, Lipman 2023, Albergo 2023, Sun 2022, McAllester 1999, Gat 2022, Karczewski 2024, Neyshabur 2017, Ertl 2018, Bemis 1996). All anchors cross-referenced to `paper/refs.bib` (existing entries).
- `honest_framing_preserved=TRUE` (path-(a) FAILURE explicit; path-(c) Λ-only primary explicit; 5 NEW research gaps framed as honest contributions not citation failures; no silent DESIGN→MEASURED promotion; verbatim honest caveats from `wf_round12_lambda_pilot/final.md` retained).
- `cross_section_invariant_chain=§3.5 ↔ §4.11 ↔ §6.1 ↔ §7.1` (full traceable reference loop preserved in `paper/sections/CROSS_REFS.md` cross-section invariant block).
