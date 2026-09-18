# Current-state execution audit — 2026-09-13

This audit supersedes earlier blanket statements that all locally executable
TODO work was complete. Passing unit tests proves the tested behavior, not
full experimental or scientific acceptance criteria.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| ROCm environment | PyTorch 2.14.0+rocm7.2, HIP 7.2.53211, Triton 3.8.0; GPU smoke and model tests execute on gfx1101 | POT OT, ADMET, Tanimoto, OpenMM OpenCL, official REINVENT RNN and QuickVina2-GPU execute on AMD; invalid driver event timestamps preclude kernel timing claims |
| CrossDocked test data | Corrected manifest resolves all 100 test receptor/ligand pairs under `extracted/crossdocked_pocket10`; paths are checked, inputs hashed | Physical evaluation integrated; generation search itself still uses a generic binding gate rather than a measured pocket reward |
| QuickVina 2 | Official binary byte identity proved; both qvina/quickvina2 selectors find it; real PARP1/CCO docking succeeds | Vina vs QuickVina N=50 parity experiment; the two selector names are aliases, not independent engines |
| Docking adapter | Fixed `REMARK VINA RESULT` parsing and seed propagation to both engines | Saved pose graph/charges/stereo and multi-conformer energy correspondence repaired; native Vina logical seed 0 mapped to deterministic nonzero engine seed; N=50 engine parity remains |
| REINVENT4 | Official prior generated 5/5 valid molecules and learned NLL on gfx1101; fixed-input CPU/GPU max NLL difference 1.05e-5 | Dedicated learned NLL bridge verified on AMD, including invalid-input accounting and real deadlines; original RDKit multiproperty proxy is separate, NLL does not establish activity or synthesis |
| AiZynthFinder | Isolated AiZynth 4.4.1, official legacy_v3 USPTO model, matching 46,695 templates and 17,422,831-entry ZINC stock found real one-step routes for three smoke targets | Current-version asset URLs still fail; legacy protocol is explicitly separate. Equivalent torch AMD policy and actual GPU search verified; maintain legacy provenance |
| Model line | tmQM shape transfer, CFG/OT and ROCm kernels have focused tests; generation descriptors now use real decoded molecules | Real QM9 3-seed converged GPU OT comparison measured (90/90 residual <1e-3); effect mixed, Vina improvement remains unproved |
| Lambda line | Explicit search settings reach MCTS; all_5 selects five reactions; capped tile count governs actual expansion attempts; Ertl SA reporting | Real click-tile products and frozen linear-prior PUCT ablation measured; original PySR backend and configured learned synthesis remain separate requirements |
| Anticancer reporting | Model evaluation reports raw means/std, range fractions, invalid counts, descriptive MW flags; suite/report corrections in progress | Per-pocket/aggregate metal coordination, GSH and DNA measurements still need actual outputs; heuristics do not establish efficacy |
| Pilot infrastructure | Exact pair selection, explicit seeds, per-job subprocess timeouts, JSON checkpoints, config/input-checked resume, truthful proxy reporting | Reduced-budget N=10 x 3 physical run finished; scientific-budget pilot, broader ablations, novelty and metal-specific pocket coverage remain |
| 100-pocket evaluation and paper | Depend on validated pilot; draft text exists | Full experiment, statistics, ablation tables, paper assembly and referenced figures |
| Dedicated workflow/skills | project-todo-flow, molflow-rocm, molmetal-lambda and molmetal-amd-eval validated; project_workflow.md maintained | Continue updating operational evidence with code changes |

## Corrections to earlier reports

- A `quickvina2` PATH lookup failing did not establish absence: the vendored
  `qvina02` is the official QuickVina 2 binary. See
  `molmetal/reports/quickvina2_binary_identity.md`.
- Previously generated manifest paths pointed to the wrong extraction level.
  The old pilot selected alphabetically sorted directories rather than test
  split pairs. Those runs cannot count as CrossDocked100 test evaluations.
- Earlier r4 logs printed top_k/branching/prior settings that were not fully
  applied. Current JSON separates requested settings from execution metadata.
- Merely exceeding 180/300 seconds is not an external environment blocker.
  The corrected bounded diagnostic runs locally; longer scientific runs remain
  required work, subject to profiling and result-quality verification.
- Previous pilot SA values were heuristic proxies; the updated runner records
  `rdkit_contrib_ertl` as its SA backend. Descriptor docking proxy values are
  never compared to published docking energies, and Lipinski rate is not
  relabeled as an SBDD success rate.

## Focused verification

```sh
HIP_VISIBLE_DEVICES=0 uv run pytest -q molmetal/tests/test_r4c_pair_execution.py molmetal/tests/test_r4c_full_sweep.py molmetal/molmetal_lam/tests/test_lambda_sweep_config_dispatch.py molmetal/tests/test_validate_generate_metrics.py molmetal/molmetal_lam/tests/test_qvina_swap.py molmetal/molmetal_lam/tests/test_vina_adapter.py
# 61 passed (before subsequent seed/metric focused additions)
uv run python molmetal/scripts/r4_c_full_sweep.py --n-pockets 1 --dry-run
```

The active goal is not complete or externally blocked: multiple implementation,
validation, and experiment tasks remain actionable.

## Current verification results

- Full regression after the first dispatch/model/adapter corrections:
  **995 passed, 3 skipped, 1 xfailed** (497.09 s).
- Focused validation after the additional device, seed-only, checkpoint,
  timeout and descriptor corrections: **88 passed**.
- 100 CrossDocked test pairs / 200 paths validated from the local split.
- Bounded diagnostic: 10 exact test pairs x 3 seeds completed, 79.44 s summed
  job wall time; 18 seed-only returns, 12 empty returns, zero new molecules.
  See `molmetal/reports/r4_test10_seed3_failure_analysis.md`.
- Fresh seed-only validation confirms 90 reduction attempts returned zero
  products and exits nonzero for generation coverage, with full JSON/CSV/MD.
- Actual adapter smoke: QuickVina -2.600, Vina -2.502 kcal/mol on identical
  prepared receptor/ligand/box/seed at N=1. Full QuickVina `dock()` returns a
  valid 3-heavy-atom CCO Complex. This is not N=50 protocol parity.

## New measured evidence: physical evaluation, AMD and OT

- `r4_click_physical_test10_seed3_v2_analysis.{json,md,png}`: 30 generation
  jobs, 70 generated products, 63 physically docked and PB passing. Strict
  preparation failed for test_005 in all 3 seeds; its 7 products remain in
  all-generated coverage (63/70). 4 products beat the redocked reference and
  satisfy SA<4/QED>0.5. None score below -8 kcal/mol. Mean measured QuickVina
  score is -5.2365 kcal/mol (63 poses); this reduced-budget, pocket-independent
  search is not a SOTA result.
- Nine redocked reference poses fail PB (test_000/003/007 x3); their raw score
  comparisons and a separate reference-PB-passing subset are both disclosed.
  All 90 saved candidate/reference SDFs were independently checked for hash,
  readability and a single conformer. Seed variation uses common pocket IDs.
- `test005_modeled_sidechain/report.json`: PDBFixer added only ASP B101
  CG/OD1/OD2; all 247 original atoms remain unchanged. Real AMD OpenCL
  minimization used PDBFixer's soft geometry potential, not a binding energy.
- `r4_click_physical_test005_modeled_vina_seed3_analysis.*`: separate modeled
  receptor + native Vina experiment completed all 3 seeds, 7 generated poses
  and PB checks. QuickVina's preceding macrocycle reference failure (unsupported
  CG0 atom type) is retained separately; no silent engine fallback or merging
  with the original strict-receptor experiment.
- `r10_ot_qm9_convergence.md`: same OT objective at 2000 rather than 200
  iterations meets residual<=1e-3 for 90/90 GPU pairings (max .000247).
  Common-validation OT-minus-off difference is -0.025934 +/- .059338 sample
  SD across 3 seeds; one seed worsens. No consistent molecular-quality or
  docking benefit is inferred. Original unconverged measurements are retained.
- `linear_prior_development_ablation_20260913/`: 32 genuine reaction products
  supply development-only observations, disjoint from evaluation products.
  Frozen prior changes the actual PUCT path (4 calls vs 0) but all paired
  reward deltas are zero; backend is linear ridge, not PySR.
- GPU molecular diversity agrees with RDKit Tanimoto on CPU and gfx1101.
  Three dedicated skills and the workflow were updated and validated.

The goal remains active. Installation fixes and bounded measurements do not
close the full pilot, full100 experiment, learned-baseline or paper criteria.

## Additional GPU and integration evidence

- `r4_click_gpu_test10_seed3_analysis.md`: all30 generation jobs ran, with70
  generated instances /4 unique structures. Actual63 generated poses plus27
  redocked references have completed gfx1101 kernel traces. Original test005's
  7 undocked products remain in coverage (63/70). There are9 redocked triple
  successes,8 in the reference-PB-passing subset; mean measured score -5.1365.
  GPU lanes/depth and seed mapping differ from native QuickVina; these counts
  do not prove an engine advantage or speedup.
- One reference was rejected by a stdout progress-prefix/trace parsing bug.
  The original run stays unchanged; derived recovery reuses its saved output,
  verifies contemporaneous log hash, device, completed kernels and chemistry.
  The output PDBQT itself was first hashed at recovery time, so historical
  unchanged output bytes cannot be established from a contemporaneous hash.
- `reinvent4_learned_smoke/README.md`: official file SHA256 is verified despite
  a retained old internal xxhash warning. ROCm forwarding is traced; the main
  environment and official Python sources remain separate from dependency
  adaptation in the isolated environment.
- R4 now accepts a development-trained `--prior-state` for frozen test
  inference and explicit `--synthesis-oracle`/`--synthesis-config`. Configuration
  hashes are checked on resume and between jobs; real per-job execution
  metadata replaces the old unconditional claim that synthesis never ran.
- Synthesis rejection does not erase a generated product: all candidate
  records, selected counts and physical coverage denominators are preserved.
- Latest broad regression found2 failures (1134 passed,3 skipped,1 xpassed):
  process-global L4 fault-injection counters and the official/local
  `flow_matching` package collision. Targeted fixes are implemented; a new
  full run is pending model runtime stabilization. Genuine model batch
  independence and seeded sampling defects were repaired and focused-tested;
  old numeric reports retain their original runtime provenance.

## Model and learned-backend follow-through

- `model_import_batch_sampling_repairs_20260913.md` documents repaired batch
  independence, preserved public/private imports and seeded inference contracts.
  Focused42 tests pass with1 existing skipped. The new real QM9 experiment
  `r10_ot_qm9_graph_independent` keeps original sources, splits and training
  protocol; three OT-minus-off deltas are -0.058275/-0.093624/-0.056010,
  mean -0.069303 +/-0.021093 sample SD. All90 transport plans satisfy the
  unchanged convergence gate. This supersedes old model values for the fixed
  runtime, without claiming molecular/docking quality or full training.
- `reinvent_prior_adapter_amd/report.json`: dedicated GPU NLL RPC preserves
  seven input rows, including four explicit empty/invalid/OOV failures. Equal
  length grouping prevents EOS padding from changing per-molecule likelihood.
  Separate adapter tests exercise startup/I/O deadlines and process cleanup.
- `aizynth_real_backend_20260913/rocm_policy_comparison.json`: same official
  legacy weights on28 inputs, ONNX/ROCm max probability difference2.32e-6,
  top1/10/50 sets agree. `rocm_learned_smoke.json` records10 actual GPU policy
  calls and three true stock routes. Intermediate MCTS trajectories can differ
  under floating-point ordering; no speed or node-for-node parity is claimed.
- Physical recovery v2 adds the explicit missing historical PDBQT hash
  limitation. Its source is `r4_click_gpu_test10_seed3_trace_recovered_v2.json`,
  and separate `_analysis_v2` artifacts preserve the original run and earlier
  postprocessing. Aggregate energy/coverage counts are unchanged.
- Real in-search docking reward integration is underway with a separate
  bounded, canonical-SMILES cache. A configured but unmeasured provider cannot
  count as applied; independent chemistry generation continues with explicit
  failure status. Actual search-feedback experiments are still required.

## Controlled measured search reward (latest)

`measured_reward_control_20260913_v2/comparison.{json,md}` compares weight0
with weight0.4 on exact test001/002 × seeds42/0/1234. Both arms invoke the
same real GPU oracle, with equal configured limits and equal actual attempts
in all six pairs. Both retain14 generated instances and complete14 candidate
plus6 reference physical evaluations. In-search provider traces are verified
separately from post-search measurements.

All generated structure sets are identical and mean top1 paired score delta
is0.0 kcal/mol. This reduced-budget4sim/depth1 result establishes working
feedback and a negative ablation, not an improvement in exploration or binding.
No six-independent-samples significance claim is made for two pockets×three
nested seeds. The initial launcher failure is preserved in the preceding
non-v2 directory; corrected uv PATH and frozen runtime are documented in the
workflow and source manifest.

The next broad regression completed1170 passed,3 skipped,1 xpassed,1 failed.
Its one legacy Pt-ablation import failure was fixed and the failing smoke
passed separately. New measured reward integration/provider scopes also pass;
a green full-suite claim still awaits stabilization of the next model edits.

## Activity target audit

The independent `pic50_assay_audit_20260913` reproduces legacy selection:
356/2000 sampled labels have explicit inequality bounds in the raw assay
field but were used as exact IC50. Across4,833 canonical Ru structures,
3,984 span multiple cell lines,395 multiple exposure times and1,142 change
by>0.5 pIC50 between first/last source rows. The old keep-first target is
order/context dependent. New conditioned/censor-aware calibration is required;
old checkpoint metrics stay historical. The audit export retains source rows,
DOIs and conditions and must not be randomly split as independent observations.

## Complete training-chemistry index and conditioned activity baseline

- `training_novelty_20260913/report.md`: all100000 CrossDocked train SDFs and
  all100 test files were valid. Train chemistry is8765 unique graphs. The
  original4 generated structures/70 instances have0 exact graph and0 Murcko
  scaffold overlap; nearest training similarities0.323–0.400. Real GPU results
  match full-index RDKit within9.62e-9. This establishes novelty only to that
  indexed train set, not diverse generation or novelty to other models' data.
  The official test split retains34 graph-shared instances; file/pocket splits
  need not imply ligand-disjointness, and the original split is not relabelled.
- `pic50_conditioned_baseline_20260913`: new uncensored HeLa48h/dark cohort,
  702 formulations/383 scaffolds,3 scaffold-group splits, GPU dual ridge with
  validation-only alpha selection. Test RMSE0.5972/0.6116/0.4150 versus paired
  mean baselines0.6489/0.7601/0.5274. New task18 tracks neural calibration and
  external validation; these changed-target/split metrics do not directly
  demonstrate improvement over the old pooled pIC50 result.
- Real CFG generation currently fails chemical acceptance: the initial
  `r10_cfg_real_crossdocked`96 requests yield64 finite atom clouds and0 decoded
  molecules. A training/sampling atom-input mismatch was repaired in a
  separately labelled experiment, but chemical decoding remains0. The active
  model work is diagnosing that gap; passing shape/loss tests does not close it.

## WF-Data-Gap-Analysis — Q1 SBDD benchmark data gap (2026-09-14)

Pure-analysis workflow. Downloaded TargetDiff (arXiv:2303.03543 v1,
ICLR 2023 OpenReview kJqXEPXMsE0; cloned repo `142f1eb`, Jul 13 2023) as
benchmark, inventoried **all 25 reported metrics** (7 primary Table 1 + 10
extended follow-up + 8 out-of-band anticancer), then compared to
Mol-Metal current data inventory. Comprehensive gap-analysis TODO at
`TODO/pending/22_data_gap_alignment_plan.md`; per-metric 25-row gap table
at `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md`;
TargetDiff inventory at `molmetal/reports/provenance_targetdiff.md`;
high-level summary + file index at `molmetal/reports/wf_data_gap_analysis.md`.

### Headline numbers

- TargetDiff full protocol: **100 pockets × 100 mols × 1000 DDPM steps =
  10 M NFE**; reports Vina Dock −7.80 / −7.91 kcal/mol (Avg / Med),
  High Affinity 58.1 %, Triple-threshold success rate 10.5 %.
- Mol-Metal measured: **~250 cells** across 4 unique pockets (1h36, 830c,
  test_001, test_005 modeled); Lambda arm N=10×3 (3000 mols), CFM arm
  **decoder-bound (0/96 docked)**; Lambda arm `validity_rate` = 0.90,
  QED 0.857, SA 1.870, Vina parity N=50 at r = 0.9983.
- **Gap closure:**
  - **Round-12 (N=10×3) ships 9 P0 metrics:** Vina Score, High Affinity,
    Triple-threshold Success Rate, Relaxed Success Rate, Validity, QED,
    SA, Lipinski, logP, Recon cascade (Lambda arm).
  - **Round-13 (N=100×3) ships 17 metrics total:** Round-12's 9 + Vina
    Dock, Diversity (Morgan-ECFP4, B-3 patch), Steric clash (PoseBusters),
    Vina Min (UFF-minimised, B-1 patch).
  - **8 metrics defer** to supplementary or post-Round-13: JSD
    bond-distance, rigid-fragment RMSD, CoM shift, strain energy,
    interaction fingerprint (PLI), mol/atm stability (CFM-blocked),
    anticancer extensions (pIC50, TPSA/RotB, cytotox).
- **Cite-only SOTA column already ship** at
  `molmetal/reports/wf_3_citeonly_sota.tex` (paper-ready LaTeX fragment,
  9 SOTA rows + 2 Lambda rows, 7 protocol-mismatch flags M1–M7 embedded
  in §2 of the paper). Honest framing preserved throughout: cite-only
  rows have distinct shading; per-row footnote "*(not re-run by
  Mol-Metal; cited from <ref>)*".
- **Honest framing mandatory** — cite-only SOTA is context, never a
  one-sample significance-test null. Lambda arm is measured at N≤10
  pockets until Round-13 ships. CFM arm is decoder-bound; geometric
  column depends on TODO-21 retrain.

### Ship target summary (from TODO-22 §4)

| Target | Wall | What it closes |
|---|---|---|
| **4a. Round-12 N=10×3 pilot** | 1–2 h GPU | Vina Score, High Affinity, Triple/Relaxed Success Rate, Validity, QED, SA, Lipinski, logP, Recon cascade (Lambda) |
| **4b. Round-13 100×3 sweep** | 6 h GPU | All of 4a + Vina Dock, Diversity, Steric clash, Vina Min |
| **4c. Cite-only SOTA column** | 1 h (already ship) | Context column for all 7 TargetDiff primary metrics |
| **4d. Metal-specific ablation** | +30 min on 4a | `--metal-seed cisplatin` arm (Lambda-vs-Lambda, doubles cells 30→60) |
| **4e. Wet-lab validation** | 0 (out of scope) | §7 future-work declaration |

**Total: ~8 h GPU + ~30 CPU-hours parallelisable + 3 days serial paper
write-up** to ship the canonical Q1 comparison row. The plan reuses
existing harnesses (`r4_lambda_only_run.py`, `r4_c_full_sweep.py`,
`wf_3_citeonly_sota.tex`) and does NOT modify `molmetal/` source.
Pure-analysis-and-shipping workflow.

### Cross-references

- `TODO/pending/22_data_gap_alignment_plan.md` — prioritised closure path
  (P0/P1/P2/P3, time estimates, resource requirements, risk assessment
  R1–R9, sequence diagram).
- `TODO/pending/13_top_journal_pilot_r12.md` §scope — explicit gap to
  close by Round-12 (9 P0 metrics).
- `TODO/pending/14_full_100pocket_paper_r13.md` §3 — cross-reference to
  TODO-22; what Round-13 closes (17 metrics) and what it does NOT close
  (8 deferred).
- `molmetal/reports/wf_data_gap_analysis.md` — high-level summary + file
  index for the WF-Data-Gap-Analysis workflow.
- `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md` —
  full 25-row per-metric gap table.
- `molmetal/reports/provenance_targetdiff.md` — TargetDiff inventory
  (100-pocket × 100 mols protocol).
