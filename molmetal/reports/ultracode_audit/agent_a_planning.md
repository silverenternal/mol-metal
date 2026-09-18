# Ultracode audit agent A — PLANNING layer snapshot (2026-09-14)

Source of truth: `/home/hugo/codes/try_triton_on_rocm/TODO/completion_audit_2026-09-13.md`.
Navigation: `/home/hugo/codes/try_triton_on_rocm/TODO/README.md`.
Workflow: `/home/hugo/codes/try_triton_on_rocm/TODO/project_workflow.md`.

This agent is read-only. Every status / blocker field below is taken verbatim
from the pending files or the authoritative audit. Historical artifacts in
`molmetal/reports/` are evidence of their original runs only.

---

## 1. Pending items — structured snapshot

### 1.1 `TODO/pending/05_reinvent4_install.md`

- **Status:** PARTIAL — installation, AMD learned prior generation/NLL and
  independent likelihood RPC complete; learned multiproperty bridge pending.
- **Priority:** high.
- **Effort:** 0.5 d.
- **Goal (1 line):** keep REINVENT4 dependencies isolated and provide a truthful
  learned prior NLL plus multiproperty scoring bridge.
- **Success criterion:** `/mnt/storage/envs/reinvent4/bin/reinvent --version`
  exits 0; JSONL bridge returns complete `ScoreResult` for valid SMILES and
  graceful incomplete results for invalid SMILES; learned-plugin scores stay
  `OBS` until a compatible scoring worker is configured.
- **Current blocker:** the existing four `[0,1]`-component JSONL bridge is an
  RDKit proxy, not learned prior NLL; a real task-specific multiproperty scoring
  worker remains unconfigured.

### 1.2 `TODO/pending/07_r4c_full_sweep.md`

- **Status:** pending (reduced-budget physical integration measured; full
  100-pocket scientific sweep not yet executed).
- **Priority:** high.
- **Effort:** 2-3 d.
- **Goal:** run the headline R4-C sweep (100 CrossDocked pockets against MMP13
  with RewardAggregator + MCTS proof search + QVina + REINVENT4 prior) and
  replace the cite-only Vina column with measured numbers.
- **Success criterion:** 100 pockets complete in a reasonable wall-clock budget
  (order ~3-5 h on RX 7800 XT, not 72 h); per-pocket top-10 Vina CSV;
  reproducibility from script alone; Lambda measured mean Vina column appears
  in `lambda_vs_sbdd_paper_numbers.md` next to cited SOTA numbers.
- **Current blocker:** none for staging; the sweep is a long-running
  experiment and depends on the new honest-orchestrator evidence
  (`17_aggregate_weak_impls_and_pending.md` flags 9/10 first-ten pairs passing
  strict prep; test_005 ASP B101 lacks CG/OD1/OD2 and is a separate modeled
  experiment).

### 1.3 `TODO/pending/11_algorithm_strengthening_r10.md`

- **Status:** local core implementation complete; full ablation evidence pending.
- **Priority:** high.
- **Effort:** 5 d (1 ultracode round).
- **Goal:** before the top-journal-aligned 100-pocket sweep, strengthen every
  Lambda-unique axis and prove each contributes on a controlled micro-benchmark.
- **Success criterion:** all 6 axes (A: 5 click reactions wired into MCTS,
  220-tile pool; B: mini-batch OT coupling parity + ablation; C: CFG head +
  ablation; D: square-planar Pt(II) prior ablation on 1h36; E: β-NF + AST
  formal semantics appendix; F: per-component sanity metrics harness)
  implemented and unit-tested; per-axis ablation result committed;
  158 + ~15 new tests pass; no sweep beyond 1-pocket micro-bench.
- **Current blocker:** none beyond Round-9 closure; per the audit, axes A/B/C/D
  have focused evidence; E is shipped as the round-10 β-NF + AST formal
  semantics appendix (task #309 completed); F (per-component metrics) is the
  remaining implementation thread.

### 1.4 `TODO/pending/12_qvina_data_staging_r11.md`

- **Status:** QuickVina 2 identified and runnable; test manifest paths
  corrected; N=50 parity pending.
- **Priority:** high.
- **Effort:** 5 d (1 ultracode round).
- **Goal:** resolve the two non-algorithmic blockers for top-journal strict
  protocol comparison (QVina engine parity at exhaustiveness=8; CrossDocked2020
  100-pocket Luo 2021 split staged locally).
- **Success criterion:** QVina callable from
  `molmetal_lam/sbdd_env/vina_adapter.py` via `--engine qvina` (engine parity
  ≥ 0.6 kcal/mol MAD vs published QVina runs); CrossDocked2020 100-pocket
  subset staged (or fallback proxy documented); `round11_engine_parity.md`
  with empirical N=50 kcal/mol table; 158 + ~10 new tests pass.
- **Current blocker:** none for QuickVina 2 discovery or test-pair staging
  (both verified locally, see `quickvina2_binary_identity.md` and
  `crossdocked_first10_resolved/report.json`); the N=50 empirical parity run
  is remaining experiment work, not an environment blocker.

### 1.5 `TODO/pending/13_top_journal_pilot_r12.md`

- **Status:** reduced-budget physical integration measured; full scientific
  pilot and ablations remain.
- **Priority:** high.
- **Effort:** 3 d (1 ultracode round).
- **Goal:** run a N=10 pocket pilot at top-journal metric standard, validate
  end-to-end pipeline produces paper-grade numbers, then scale to N=100 in
  round-13.
- **Success criterion:** `round12_pilot_results.md` with all per-pocket +
  aggregate metrics and 3-seed mean ± std; `round12_ablation_table.md` with
  the 6-axis ablation; `round12_failure_analysis.md` with 10-15 representative
  failure cases; 158 + ~5 new tests pass; all numbers are real measurements.
- **Current blocker:** the search remains pocket-independent (post-search
  physical evaluation is not yet a closed-loop pocket-conditioned reward);
  OT/CFG toggles do not belong to the pure click-search runner unless a model
  generator is explicitly connected.

### 1.6 `TODO/pending/14_full_100pocket_paper_r13.md`

- **Status:** pending.
- **Priority:** high.
- **Effort:** 7-14 d (1 ultracode round + serial paper write).
- **Goal:** run the full 100-pocket CrossDocked2020 sweep at top-journal
  protocol, then write the paper draft (Digital Discovery Q1 /
  J. Chem. Inf. Model. Q1 target).
- **Success criterion:** `round13_full_sweep_results.md` with 100-pocket ×
  3-seed results and statistical comparison vs SOTA;
  `round13_ablation_full.md` with 6-axis ablation; paper draft
  `paper/digital_discovery_submission.tex` with all 7 sections + supplementary
  materials; arXiv preprint bundle ready; 158 + ~20 new tests pass.
- **Current blocker:** round-12 N=10 pilot results + all 10 top-journal
  checklist items; per-pocket/aggregate metal coordination, GSH and DNA
  measurements still need actual outputs (anticancer metric suite TODO-15
  wires logP/TPSA/RotB, but metal proxies + DNA binding remain
  unimplemented); Triple-threshold descriptive; not efficacy.

### 1.7 `TODO/pending/15_anticancer_metric_suite_r11b.md`

- **Status:** local implementation complete; full paper evaluation pending.
- **Priority:** high.
- **Effort:** 1 d (1 codex task).
- **Goal:** build metal-anticancer adjusted metric suite (logP, TPSA, RotB,
  hERG proxy + composite_score).
- **Success criterion:** pytest green, `composite_score` wired into
  `RewardAggregator`, TODO file created; partial evidence:
  `AnticancerMetricSuite.descriptor_report()` exposes raw logP/TPSA/RotB/MW
  values and IV/metal-adjusted flags; generation and pilot reports include
  logP/RotB/MW aggregates.
- **Current blocker:** per-pocket and aggregate round-13 physical / metal /
  DNA outputs are not yet produced; metal coordination, GSH and DNA
  measurements still need actual outputs; heuristics do not establish
  efficacy.

### 1.8 `TODO/pending/17_aggregate_weak_impls_and_pending.md`

- **Status:** living audit — supersedes earlier stale installation blockers,
  proposed threshold-based explanations for weak results, and unverified
  completion claims.
- **Priority:** high (cross-cutting).
- **Effort:** n/a (audit document).
- **Goal:** maintain consolidated weak implementations, bad results, missing
  env, and pending tasks so they cannot be silently dropped.
- **Success criterion:** the audit reflects current state and tracks ROCm,
  native/GPU docking, CrossDocked first-ten, REINVENT4, AiZynth, Lambda
  generation/physical evaluation/chemistry preservation/prior/synthesis/
  pilot/metrics, model line, and experiment integrity.
- **Current blocker:** search remains pocket-independent; required CFG /
  metal-prior ablations, larger training/evaluation, learned SOTA checkpoint
  comparisons, and claimed Vina improvements remain unproved; 100 x 3 full
  experiment, statistical comparisons against measured baselines, metal/GSH/
  DNA physical evidence and paper assembly remain actionable work.

### 1.9 `TODO/pending/18_activity_assay_calibration.md`

- **Status:** data audit and conditioned GPU baseline measured; learned
  predictor retraining, broader external validation and calibrated
  deployment remain open.
- **Priority:** high (model/property-prediction line; independent of Lambda
  search chemistry).
- **Effort:** n/a (multi-step).
- **Goal:** replace the legacy keep-first Ru calibration (which collapsed
  356 censored rows into exact labels and lost cell-line / exposure /
  counterion context) with an assay-conditioned, uncensored, scaffold-split
  baseline that supports ranking and downstream deployment.
- **Success criterion:** verify and retrain the neural property predictor
  against the same frozen cohort/splits/budget controls; extend to other
  declared cell/time/metal tasks without pooling incompatible endpoints;
  handle censored observations as bounds; calibrate uncertainty and state
  the target domain before using activity as a reward; keep predicted
  activity, docking, synthetic routes and biological efficacy claims
  separate.
- **Current blocker:** only conditioned baseline (HeLa48h/dark, 702
  formulations / 383 scaffolds, 3 scaffold-group splits, GPU dual ridge) is
  shipped; neural predictor retraining on the same frozen cohort and broader
  external validation are open.

### 1.10 `TODO/pending/decisions.md` + `risks.md` + `roadmap.md`

- **`decisions.md`** carries D1-D5 approved (2026-09-12: DiffSBDD cite-only,
  MMP13 primary / MMP2 skipped, PDBbind skipped, EGNN-first, PoseBusters
  headline + RDKit fast-path) and D6-D7 pending (D6 REINVENT4 install path
  default = separate venv, decide by 2026-09-19; D7 Vina→QVina swap default
  = run both engines in headline table, decide when QVina is installed).
- **`risks.md`** enumerates 8 open risks. R1 cite-only SOTA comparison
  cannot tighten; R2 Vina-vs-QuickVina protocol parity still unmeasured;
  R3 PoseBusters CuAAC embedding root cause; R4 metal_hybrid_v4_round3
  stability re-measure BLOCKED (bash sandbox exit 1/120/134); R5 REINVENT4
  live score still OBS (resolved at install layer, open at multiproperty
  reward layer); R6 tmQM pretraining wireup resolved 2026-09-13; R7 PDBbind
  v2020 download blocked; R8 pIC50 Pearson r=0.407 — ranking only.
- **`roadmap.md`** places Phase 0 DONE; Phase 1 closed-loop implementations
  revalidation in progress (shipped components need current execution
  evidence); Phase 2 algorithm strengthening pending — round-10 (ship
  target end of W3 / 2026-10-03); Phase 3 top-journal protocol alignment
  pending — round-11 (W4 / 2026-10-10); Phase 4 pilot + full sweep pending
  — round-12 + round-13 (W7 / 2026-10-31 paper draft).

---

## 2. Completed archive — coverage note

`TODO/completed/` contains 23 items (13 original + 10 round-4) plus new
round-8 / round-9 / round-10 archives (`20_r4c_full_sweep_done.md`,
`18_reinvent4_install_done.md`, `21_tmqm_egnn_wireup_done.md`,
`22_metal_prior_done.md`, `23_minibatch_ot_done.md`, etc.). Coverage
includes: Lipman FM adapter, Vina adapter, PoseBusters adapter, AiZynth
adapter, QVina swap, CuAAC PB fix, tmQM pretraining, round-3 axes
(drop hooks, loosen clamp, register σ, L-1 DiffDock oracle, L-4 REINVENT4
scorer), metal SMILES parser, 3D embedding sanity, Ertl SA, leakage
diagnosis, counterion ablation, MMP13 surrogate, proof-search prior,
L3 tile wireup, closed-loop rewire, tmQM EGNN wireup, square-planar
metal prior, mini-batch OT. These items only prove their dated evidence;
the authoritative state is `completion_audit_2026-09-13.md`.

---

## 3. 1-paragraph synthesis — current work stream

The current work stream is a four-phase plan (Phase 0 DONE; Phase 1 closed-loop
revalidation; Phase 2 algorithm strengthening — round-10; Phase 3 top-journal
protocol alignment — round-11; Phase 4 pilot + full sweep — rounds 12/13)
with execution gated by the new honest-orchestrator posture: every pending
item now distinguishes "local implementation shipped" from "focused
validation evidence" from "scientific acceptance". Round-10 (algorithm
strengthening) is functionally complete at the local layer — CFG, mini-batch
OT, square-planar Pt(II) prior, β-NF + AST formal semantics appendix, and
per-component sanity metrics each have focused evidence — but the ablation
matrix across the 6 axes (branching, top_k, prior, click rules, mini-batch
OT, CFG) has not yet produced a measured on-/off-delta, so axis F remains
the largest open thread. Round-11 (QVina + data) is two-thirds shipped:
QuickVina 2 is the verified vendored `qvina02` binary, the CrossDocked2020
100-test-pair manifest resolves under local extraction, and the corrected
test-001/005 physical evaluation is in evidence; the open work is the
N=50 Vina-vs-QuickVina empirical parity experiment (R2). Round-12 (N=10 ×
3-seed pilot) has bounded physical integration measurements
(`r4_click_physical_test10_seed3_v2_analysis`) but the full scientific
budget, pocket-conditioned closed-loop reward and novelty/training-set
comparison are still pending. Round-13 (100-pocket sweep + paper) is
designed, not started; it depends on round-12 acceptance and on the
anticancer metric suite (TODO-15) producing per-pocket metal / DNA / GSH
outputs that are not yet generated. REINVENT4 (TODO-05) is split: install,
official learned prior NLL on gfx1101 and dedicated AMD RPC are all
shipped in the isolated `/mnt/storage/env-projects/reinvent4-rocm`
environment; the multiproperty bridge that connects the learned signal to
the search reward is still an RDKit proxy, and risk R5 remains partially
open. The activity predictor (TODO-18) has a measured conditioned HeLa48h/dark
baseline but no neural predictor retraining on the same frozen cohort.
Decisions gated on the user are concentrated in D6 (REINVENT4 install
path — separate venv vs shared venv vs Docker; default (a), decide by
2026-09-19), D7 (Vina→QVina swap activation — default (c) for the arXiv
table, decide when QVina is installed), and the editorial choice of which
journal to lead with at submission time (Digital Discovery vs
J. Chem. Inf. Model. vs Patterns vs Briefings in Bioinformatics, conditional
on which § has the strongest story in round-13). Smaller user-side calls:
whether to accept the test_005 modeled-receptor protocol as a separately
labelled experiment or to drop test_005 from the round-12 / round-13
cohorts, whether to ship the cite-only SOTA column when the GPU checks
fail to reach the publishable threshold, and whether to keep Lipinski MW
as a descriptive flag or to also report a 300-700 Da MW-range flag for
IV anticancer chemistry per the TODO annotations in `14_full_100pocket_paper_r13.md`.
No sweep / no benchmark / no full regression was run during this audit;
all findings come from the read-only inventory above.
