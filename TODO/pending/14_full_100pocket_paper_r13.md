# Round-13 — Full 100-pocket sweep + paper draft

**Status:** ⚙️ **PARTIAL → STRUCTURAL_SHIP_2026-09-16 — 30+ cells MEASURED per WF-Lambda-Boost + sub-fix A/C + Deflex flags lift; paper-grade 300 cells still deferred (CPU-only)**
**Priority:** high
**Effort:** 7-14 days (1 ultracode round + serial paper write)
**Owner:** (unset)
**Depends on:** Round-12 (pilot at top-journal standard), F2(a) MetalLigandExchange SMARTS rule (TODO-29)
**Blockers:** F2(a) ship + serialize Lambda runs (no concurrent sweeps) + CFM path import fix

**Last updated:** 2026-09-16 (R15 update — sub-fix A + C + Deflex flags ship; paper-grade 300 still deferred to Round-16)

## ⚠️ UPDATE 2026-09-15 — Honest negative result

`wf_round13_100x3` finished with `paper_grade_data_ready: false`:

| Path | Status | n_evaluated |
|---|---|---:|
| Lambda | process killed mid-run at 55%, no final report.json | 0 MEASURED |
| PB | COMPLETE on 10 pockets × 3 seeds | 30 (all search-bound) |
| CFM | BLOCKED (ModuleNotFoundError + GPU HSA userland outage) | 0 |

**Critical insight**: PathA-10x3 diversity lift does NOT generalize. 30/30 cells on 10 NEW pockets (test_010..test_019) hit same singleton collapse (n_no_candidates=15 + n_seed_only=15).

**GPU is healthier than wf_gpu_auto_recover/final.md claimed**: under `uv run`, `torch.cuda.is_available()=True` with 2 devices. System Python (3.14) cannot import ROCm libtorch — `uv run` can.

See `molmetal/reports/wf_round13_100x3/final.md` for full verdict + path forward.

## Path forward (TODO-29 owns this)

1. **Ship F2(a) MetalLigandExchange SMARTS rule** (in-flight via ultracode Task A in workflow `w8579x29t`) — structural fix for 3-layer singleton attractor
2. **Serialize Lambda runs** — kill PID 110660, launch single n_sim=200 sweep, ETA ~110 min
3. **Fix `r10_cfg_real_crossdocked.py` import** (one-line `sys.path` or `PYTHONPATH`)
4. **Re-run Round-13** at n_sim=1000 after F2(a) ships

If F2(a) ships + Lambda n_sim=1000 sweep completes → Round-13 has a path to paper-grade data.

## Goal

Run the **full 100-pocket CrossDocked2020 sweep** at top-journal protocol,
then write the paper draft (Digital Discovery Q1 / J. Chem. Inf. Model. Q1
target). This is the **final deliverable** of the algorithm → experiment
pipeline.

## Top-journal metric standard (carried from round-12)

### Per-pocket (×100 pockets)
- Vina mean ± std (kcal/mol)
- Ertl SA mean ± std
- QED mean ± std
- Lipinski pass rate
- PoseBusters pass rate
- LogP mean ± std

<!-- TODO(anticancer-metrics): Add TPSA and rotatable-bond summaries using
     anticancer/IV-appropriate ranges (TPSA 60–150 Å²; RotB <10). Report the
     fraction of candidates in-range rather than applying oral-drug cutoffs. -->

### Aggregate (across 100 pockets × 3 seeds = 300 evals)
- Triple-threshold success rate (Vina < co-crystal ∧ SA < 4 ∧ QED > 0.5)
- Relaxed success rate (Vina < -8.0)
- Diversity (mean pairwise Tanimoto)
- Novelty (max Tanimoto to training)
- NFE budget

<!-- TODO(anticancer-metrics): Keep Lipinski MW as a descriptive statistic only;
     do not gate metal complexes at MW ≤500 Da. Add an MW-range flag
     (300–700 Da) and document retained high-MW candidates. -->

### Statistical reporting
- 3 seeds × 100 pockets = 300 evaluations
- Mean ± std across seeds (per-pocket + aggregate)
- 95% CI on aggregate metrics
- Paired effect sizes and 95% bootstrap confidence intervals against actually
  measured baselines on the same held-out pockets and matched budgets; resample
  at pocket level, retaining within-pocket seed dependence
- Cite-only aggregates are context, never a one-sample significance-test null;
  predefine multiple-comparison treatment for the actual measured comparisons

## Anticancer metric survey — TODO annotations

The triple-threshold is retained for comparability, but it does not cover
metal-anticancer liabilities. Complete these items before finalising the
100-pocket paper (source: `molmetal/reports/anticancer_vs_general_metrics_survey.md`):

- [x] **logP:** compute RDKit Crippen logP; report mean ± std and the fraction
  in the 2–5 anticancer target range; flag logP >5 as a potential hERG liability.
- [ ] **TPSA/RotB:** add TPSA (target 60–150 Å² for IV candidates) and RotB
  (<10) to per-pocket and aggregate tables; state that TPSA >75 is acceptable
  for IV anticancer chemistry.
- [ ] **Metal proxies:** report oxidation state, coordination number, geometry
  validity, monodentate Cl count (aquation proxy), and an S-donor/GSH liability
  flag. Mark reduction potential, trans effect, and LFSE as unimplemented if no
  reliable calculator is available.
- [ ] **DNA binding:** define a reproducible DNA-fragment docking/Kb proxy (or
  explicitly record it as unavailable), separate from protein-pocket Vina.
- [x] **Safety/efficacy limits:** label hERG, CYP450, PPB, NCI-60 GI50/TGI,
  selectivity index, and kinetic/redox measurements as predictor or wet-lab
  requirements; do not imply the sweep measures them.
- [x] **Route assumption:** state whether candidates target IV cytotoxic use.
  If the route becomes oral targeted therapy, restore conventional Lipinski and
  TPSA interpretation and rerun the metric analysis.

Partial implementation evidence (not full sweep acceptance): `AnticancerMetricSuite.descriptor_report()` exposes
raw logP/TPSA/RotB/MW values and IV/metal-adjusted flags; generation and pilot
reports include logP/RotB/MW aggregates. The current interpretation is IV
cytotoxic discovery, with oral Lipinski thresholds retained only as a
comparability flag. Generation descriptors are tested, but per-pocket and
aggregate round-13 physical/metal/DNA outputs are not yet produced.

## Scope (4 axes)

### A — Full 100-pocket sweep
Run `molmetal/scripts/r4_c_full_sweep.py --pockets 100 --seeds 3` at the
top-journal protocol (QVina exh=8 or Vina 1.2.7 exh=16 with parity doc).
Budget: ≤6 hours wall-clock (100 × 3 × 1.2 min = ~360 min). Fail-fast if
any single pocket takes >10 min; report the failure + use partial coverage.

### B — SOTA checkpoint re-runs (best-effort)
Try to fetch + run Pocket2Mol / TargetDiff / DiffSBDD / DecompDiff /
FLOWR checkpoints on the same 100 pockets (3 seeds each). If checkpoints
unreachable, ship the cite-only SOTA comparison with all 7 protocol-mismatch
flags explicit (per round-9 baseline).

### C — Ablation table (full 6-axis)
Re-run the round-12 ablation harness on N=10 pockets × 3 seeds × 6 axes:
- branching ∈ {60, 1020}
- top_k ∈ {20, 100}
- prior ∈ {off, on}
- click_rules ∈ {CuAAC only, all 5}
- mini_batch_ot ∈ {off, on}
- cfg_scale ∈ {1.0, 2.0}

Cap at 60 min wall via early-stop. Report per-cell mean ± std + effect size.

### D — Paper draft writing (3 sections in parallel)

#### D1 — Method §3
- 9-layer MLC formalism with β-NF + AST examples
- 5 click reactions (CuAAC, SPAAC, thiol-ene, Suzuki, amide coupling)
- EGNN + dative-bond edge type
- MetalGeometryPrior (Pt square-planar 90°, Ru/Ir octahedral)
- Mini-batch OT coupling (Tong 2023)
- MCTS VirtualLoss + TranspositionTable

#### D2 — Evaluation §4 + Ablation §5
- §4.1 tmQM pretraining (Tier 1A, already paper-ready)
- §4.2 Ru/Ir toxicity baseline (Tier 1D)
- §4.3 MMP13 830c (Tier 1B)
- §4.4 Click chemistry + PB + retrosynthesis (Tier 1C)
- §4.5 100-pocket sweep results (NEW)
  - **WF-3 (2026-09-14):** cite-only SOTA comparison table shipped at
    `molmetal/reports/wf_3_citeonly_sota.tex` (paper-ready LaTeX
    fragment with caption noting "Cite-only SOTA comparison; all
    numbers are cited, not re-run by Mol-Metal. 7 protocol-mismatch
    flags (M1)-(M7) are documented in §2"); 11 rows (9 SOTA + 2
    Mol-Metal), 11 columns (Paper | Dataset | Docking engine |
    Validity | Novelty | Diversity | Vina mean ± std | SA | QED |
    PB pass | N pockets × seeds); companion audit at
    `molmetal/reports/wf_3_citeonly_sota_consistency.md` confirms
    `all_have_provenance = TRUE` (9/9 SOTA rows carry at least one
    PROVENANCE-FROM field); 7 protocol-mismatch flags (M1-M7) and
    honest framing (CITED-ONLY vs MEASURED vs PROXY) preserved.
- §5 Ablation table (NEW)

#### D3 — Intro §1 + Related §2 + Limitations §6 + Future §7
- §1 Introduction: precious-metal drug design challenge + Lambda framework
- §2 Related work: SOTA 9-paper table + 7 protocol-mismatch flags
- §6 Limitations: TODO-05 reinvent4 (env-blocked); TODO-07 data staging
  (resolved in round-11); QVina parity (resolved or documented)
- §7 Future work: wet-lab validation; multi-metal extension; scalability

## Success criterion

- `molmetal/reports/round13_full_sweep_results.md` with 100-pocket × 3-seed
  results, all aggregate metrics, statistical comparison vs SOTA
- `molmetal/reports/round13_ablation_full.md` with 6-axis ablation
- Paper draft `paper/digital_discovery_submission.tex` (target Q1 IF 8.5)
  with all 7 sections + supplementary materials
- arXiv preprint submission ready (`.tex` + `.bib` + figures/)
- 158 + ~20 new tests pass
- All numbers are **real measurements**, not projections

## Out of scope

- Wet-lab validation (deferred per user constraint; out of sandbox scope)
- Multi-metal extension beyond Pt/Ru/Ir (potential future paper)
- Larger denoising-NFE parity study (Lambda is MCTS; not diffusion)

## Recommended orchestration

This round is **larger than prior rounds** because of the paper draft. Split
into 2 sub-rounds:

### Round-13a — Experiments (3-5 days)
| Phase | Agents | Goal |
|---|---|---|
| 1 — Sweep | 1 | A: full 100-pocket × 3-seed sweep (longest single agent) |
| 2 — SOTA re-runs | 1 | B: best-effort SOTA checkpoint fetch + run |
| 3 — Ablation | 1 | C: full 6-axis ablation on N=10 × 3 seeds |
| 4 — Verify + report | 1 | All tests pass; `molmetal/reports/round13a_final.md` |

### Round-13b — Paper draft (3-5 days, serial after 13a)
| Phase | Agents | Goal |
|---|---|---|
| 1 — Method + Eval + Ablation | 3 parallel | D1 + D2 sections |
| 2 — Intro + Related + Limitations | 2 parallel | D3 sections |
| 3 — Assembly + supplementary | 1 | Combine + arXiv-ready bundle |

## Top-journal submission targets (in priority order)

| Target | IF (2025) | Q | Open access | Decision basis |
|---|---:|---|---|---|
| **Digital Discovery** (RSC) | 8.5 | Q1 | yes | ML × chemistry专门刊，对 protocol flags 宽容 |
| **J. Chem. Inf. Model.** (ACS) | 5.6 | Q1 | no (optional OA) | ACS 老牌方法刊，覆盖最广 |
| **Patterns** (Cell Press) | 6.0 | Q1 | yes | Cell 旗下综合刊 |
| **Briefings in Bioinformatics** | 7.0 | Q1 | no (optional OA) | 如果 interpretability 故事强 |

**Pick at submission time** based on which § has the strongest story:
- If §4.5 (100-pocket sweep) is the lead → Digital Discovery
- If §4.1 (tmQM pretraining) + §4.4 (click chemistry) are the leads →
  J. Chem. Inf. Model.
- If §3 (9-layer MLC formalism) + β-NF proof are strong → Patterns

## ENV constraints

- ROCm 7.2 / triton-rocm 3.8.0 / gfx1101 / wave64
- "先别跑实验" relaxed for this round ONLY (100-pocket sweep is the
  experiment, per user directive "最后在上实验")
- Per-pocket wall-clock ≤10 min hard cap
- Total sweep budget ≤6 hours wall-clock
- Ablation budget ≤60 min wall-clock

## Honest caveats to include in paper

- Single-GPU ROCm 7.2 + RX 7800 XT (not a GPU cluster)
- 100-pocket subset is Luo 2021 split (per round-11 staging)
- Cite-only SOTA comparison if checkpoints unreachable
- QVina parity documented empirically (round-11)
- 7 protocol-mismatch flags explicit (per lambda_vs_sbdd_protocol_aligned.md §2)
- Throughput numbers are **measured** for this round (not projections)
- Anticancer metric caveat: logP/TPSA and metal-proxy values are computational
  screens; hERG, CYP450, PPB, NCI-60 selectivity, DNA-binding kinetics,
  aquation/redox potentials, and GSH resistance require validated predictors or
  wet-lab confirmation.

## Related reports

- `molmetal/reports/round12_pilot_results.md` (round-12 N=10 pilot)
- `molmetal/reports/round11_engine_parity.md` (QVina parity)
- `molmetal/reports/round9_r4c_pilot_results.md` (round-9 N=5)
- `molmetal/configs/sota_aligned_targetdiff.yaml` (SOTA-aligned config)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (cite-only SOTA)
- `molmetal/reports/f2_tmqm_pretraining.md` (pre-training baseline)
- `molmetal/reports/baseline_metrics_extension.md` (Ru/Ir baseline)
- [`TODO/pending/22_data_gap_alignment_plan.md`](22_data_gap_alignment_plan.md)
  (WF-Data-Gap-Analysis plan; this round = ship target 4b)
- `molmetal/reports/wf_data_gap_analysis.md` (high-level summary + file index)

## §3 — Gap-analysis cross-reference (WF-Data-Gap-Analysis, 2026-09-14)

The per-metric gap analysis (TargetDiff ↔ Mol-Metal) is fully documented in
[`TODO/pending/22_data_gap_alignment_plan.md`](22_data_gap_alignment_plan.md).
This section is the **summary cross-reference**, not the plan itself.

### §3.1 Gap summary in one paragraph

TargetDiff's 100-pocket × 100-mol × 1000-step diffusion protocol (10 M NFE
total) reports **25 inventoriable metrics** (7 primary + 10 extended + 8
out-of-band). Mol-Metal currently measures **12 metrics** at any scale
(n≤10 pockets, ~250 cells total). At Round-13 closure: **17 metrics ship
ready** (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 16 partial, 21, 22 internal) and
**8 metrics defer** to supplementary or post-Round-13 (12 JSD bond-distance,
13 RMSD, 15 CoM shift, 17 strain, 18 PLI, 20 atm/mol stability, 23–25
anticancer extensions). Cite-only SOTA column is **already ship** via
`wf_3_citeonly_sota.tex`. Lambda arm reports per-pocket Vina at N=10×3 from
Round-12; Round-13 scales to N=100×3 = 300 evals. CFM arm is
**decoder-bound (0/96 today)** and remains blocked on TODO-21.

### §3.2 What Round-13 closes (per-metric, mapped to TODO-22 §4b)

| Metric | TargetDiff scale | Round-13 closes via | Priority |
|---|---|---|---|
| Vina Score (mean ± std) | 100 × 1 × 100 | N=100×3 sweep, QVina-GPU exh=8 | **P0** |
| Vina Dock (full re-dock) | 100 × 1 × 100 | Same sweep, ~6 h wall | **P1** |
| Triple-threshold success rate | 100 × 1 × 100 | §4.5 evaluation, strict Vina<co-crystal ∧ SA<4 ∧ QED>0.5 | **P0** |
| Relaxed success rate (Vina<−8.0) | 100 × 1 × 100 | §4.5 (companion to strict) | **P1** |
| Validity | 100 × 1 × 100 | Lambda arm `validity_rate` | **P0** |
| QED (mean) | 100 × 1 × 100 | Already wired, scale only | **P0** |
| SA score (mean) | 100 × 1 × 100 | Already wired (Ertl SA), scale only | **P0** |
| Diversity (Morgan-ECFP4 Tanimoto) | 100 × 1 × 100 | B-3 patch (1-day) — current is Lambda-native homotype | **P1** |
| Lipinski (rule-of-5 pass) | 100 × 1 × 100 | Per-pocket CSV | **P0** |
| logP (Crippen) | 100 × 1 × 100 | Per-pocket CSV | **P0** |
| Steric clash (PoseBusters) | 100 × 1 × 100 | 22/22 wire to per-pocket JSON (test_001 confirmed) | **P1** |

<!-- PB-pass-rate status (WF-PB-Pass-Real-Dock, 2026-09-14, molmetal/reports/wf_pb_pass_real_dock/final.md):
     the PB pass rate is now \MEASURED{} on a real docked pose at the
     path-correctness smoke level (1/1 = 1.000 on pocket=test_000, seed=42,
     chemistry-validity only via PoseBusters mol mode, no protein clash).
     This is NOT the 100 × 1 × 100 production figure above; the production
     cell remains \DESIGN{} pending Round-13 sweep. The §4.6 integration at
     paper/sections/04_evaluation.tex carries the honest-framed \MEASURED{}
     single-pocket smoke value with three explicit caveats:
       (1) protein-blind mode (pb_mode="mol", no receptor);
       (2) one of 100 pockets exercised;
       (3) no SOTA comparison (cite-only PB row remains \CITEDONLY{}).
     The full per-pocket PB columns of Table 1 remain \DESIGN{} pending
     Round-13 10×3 (or 100×3) sweep. -->
**PB pass rate status (Round-14 / 2026-09-14):** \MEASURED{} on
real docked poses at single-pocket / single-seed smoke
(pocket=\texttt{test\_000}, seed=42, n\_pb\_pass=1/1, pb\_pass\_rate=1.000,
chemistry-validity only). Full 100×3 PB column remains \DESIGN{}
pending Round-13 sweep. Artefact: \texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/final.md};
integration: \texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/integrate.md}.
Honest-framing caveats: (i) protein-blind mode (no receptor clash),
(ii) 1-of-100 pockets, (iii) no SOTA anchor.

<!-- WF-PB-Pass-10x3-Smoke (2026-09-14, molmetal/reports/wf_pb_pass_10x3_smoke/final.md):
     re-ran the same --pb-check + --physical-docking path on
     10 pockets × 3 seeds = 30 cells to obtain the per-pocket
     mean ± std across seeds that the §4.6 protocol promises.
     Result: pb_pass_rate = None × 30 cells (every cell returned
     no_candidates or seed_only; zero accepted generated
     molecules per cell, so PoseBusters has nothing to evaluate).
     path-correctness wiring is now exercised at n=30 (extended
     from n=1 at WF-PB-Pass-Real-Dock). Per-cell PB production
     cells promoted to \MEASURED{} in this round: 0 / 100.
     The pipeline is search-bound, not PB-bound at the
     n_simulations=100 budget with the strict
     synthesis_oracle=smarts ∧ symbolic_prior=True gate
     combination. Three plausible causes (not yet ruled out):
     (a) under-budget MCTS at n_sim=100 vs the
     WF-Lambda-1c 1000–3000 budget that accepted candidates;
     (b) strict gate combination is conservative at small-N;
     (c) reference initialisation on click-poor pockets.
     Wall-clock: 144.3 s total / 4.81 s per cell; the
     100 × 3 = 300-cell production sweep is bounded at
     ~24 min wall on this box.
     Integration: paper/sections/04_evaluation.tex
     §\ref{sec:evaluation:pb-30cell} (new \subsubsection);
     molmetal/reports/wf_pb_pass_10x3_smoke/integrate.md.
     Follow-up: re-run with r4_lambda_only_run.py at
     --metal-seed cisplatin + n_simulations=1000 (after
     WF-Lift-N-Sim-Cap lifts the r4_c_full_sweep.py:1466-1467
     hard-cap) so that pb_pass_rate becomes a finite fraction.
-->
**PB production cells promoted \DESIGN{} $\to$ \MEASURED{}
(WF-PB-Pass-10x3-Smoke, 2026-09-14):** **0 / 100.**  Of the
30 cells executed (10 pockets $\times$ 3 seeds,
\texttt{test\_000..test\_009}, seeds $\{42, 0, 1234\}$),
every cell returned \texttt{pb\_pass\_rate\,=\,None} because
the strict-gate combination
(\texttt{synthesis\_oracle\,=\,smarts + symbolic\_prior\,=\,True +
n\_simulations\,=\,100}) accepts \emph{zero} generated
molecules per cell; PoseBusters therefore has nothing to
evaluate.  The path-correctness claim is extended from
$n{=}1$ to $n{=}30$ at the \emph{wiring} level only;
Table~1 \texttt{PB pass} cells remain \DESIGN{} pending a
search budget that accepts $\geq 1$ generated candidate per
cell.  Artefact: \texttt{molmetal/reports/wf\_pb\_pass\_10x3\_smoke/final.md};
integration: \texttt{molmetal/reports/wf\_pb\_pass\_10x3\_smoke/integrate.md}.
Wall-clock: $144.3$~s total / $4.81$~s per cell;
the $100 \times 3 = 300$-cell production sweep is bounded at
$\sim\!24$~min wall on this box.
| Eval/Recon/Complete cascade | 100 × 1 × 100 | Lambda arm `validity_rate` (Recon-equivalent) | **P0 (Lambda) / BLOCKED (CFM)** |

### §3.3 What Round-13 explicitly does NOT close (deferred)

- **JSD bond-distance histogram** (12): RDKit ETKDG embed + 8 bond-type histograms; deferred (Lambda has no raw 3D by design)
- **Rigid-fragment RMSD post-MMFF** (13): 2-day module; deferred
- **Ring-size distribution** (14): partial via `ring_size_distribution`; minor extension
- **CoM shift vs reference** (15): 1-day module; deferred
- **Strain energy (PoseCheck UFF)** (17): 2-day module; deferred
- **Interaction fingerprint (PLI, PoseCheck)** (18): 3-day wire; deferred
- **Vina Min (UFF-minimised)** (19): 1-day wire meeko+obabel minimize; can ship
- **mol_stable / atm_stable** (20): CFM arm blocked on TODO-21
- **Anticancer extensions (23–25):** pIC50, TPSA/RotB, cytotox — supplementary S2 only

### §3.4 Decision policy (carried from TODO-22 §7)

**If user does not authorise CFM retrain (TODO-21):** Lambda-only column +
cite-only SOTA + hybrid-paradigm framing is the canonical Q1 path. Lambda
arm + cite-only covers 11/25 metrics. Wet-lab validation is declared §7
future-work. **If user authorises CFM retrain:** ship geometric column at
Round-13 and gain metrics 20 (atm/mol stability) at +2-3 days retrain cost.

### §3.5 Resource budget (carried from TODO-22 §6)

- **Wall-clock:** 6 h GPU (QVina-GPU exh=8, ~10× speedup vs Vina 1.2.7)
- **GPU hours:** ~6 h on RX 7800 XT gfx1101 wave64
- **CPU cores:** 16 (100×3 cells parallelisable)
- **Memory:** 32 GB system RAM
- **Disk:** 50 GB (100×3 = 300 cells × per-cell artifacts)
- **Network:** 0 (offline; data already staged per round-11)
- **Dependencies:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0,
  QVina-GPU, PoseBusters, RDKit, meeko, ADFRsuite, OpenMM — all installed
  per Round-7 (TaskList #413)

### §3.6 Risk inheritance from TODO-22 §7

| Risk | Severity | Status |
|---|---|---|
| **R1: CFM decoder-bound (0/96 today)** | HIGH CONFIRMED | Fall back to Lambda-only + cite-only SOTA (D2 cite-only path) |
| **R2: QVina-GPU scaling unverified at N>50** | MEDIUM LOW | Round-12 pilot confirms; Round-13 scales only after |
| **R3: CrossDocked 100-pockets staging incomplete** | MEDIUM MEDIUM | Round-12 prep phase (1 day); 9/10 already resolved |
| **R7: Triple-threshold < 10.5 %** | MEDIUM MEDIUM | Report both strict AND relaxed; cite TODO-21 as future work |
| **R9: ROCm instability in 6 h sweep** | MEDIUM MEDIUM | Round-12 validates first; fall back to `--num-workers 1` |

### §3.7 Honest framing reminder

This round is **not** a benchmark claim. Cite-only SOTA rows are cite-only,
not re-runs. Lambda arm runs are measured but at N≤10 pockets until
Round-13 ships. The Q1 comparison row is anchored on **drug-likeness metrics
(QED, SA, Lipinski, Validity)** where Lambda arm is already at or near
SOTA, with binding affinity (Vina) framed as "competitive single-pocket;
100-pocket scale in Round-13". Wet-lab validation is §7 future-work.

### §3.8 Full gap-analysis plan

See [`TODO/pending/22_data_gap_alignment_plan.md`](22_data_gap_alignment_plan.md)
for the full prioritised ship targets (P0/P1/P2/P3), time estimates
(8 h wall total), resource requirements, risk assessment (R1–R9), and
sequence diagram. Round-13 corresponds to ship target **4b** in that plan.

---

## Round-13 scope unchanged (2026-09-15, post-Round-12 integration)

**Status:** Round-12 Λ-only $10{\times}3$ pilot SHIPPED 2026-09-15; Round-13 scope remains the full $100 \times 3$ scientific production sweep + paper draft (Digital Discovery Q1 / J. Chem. Inf. Model. Q1 target).

### Round-13 scope (UNCHANGED from 2026-09-13)

- **N=100 pocket sweep at top-journal metric standard** (3 seeds × 100 pockets = 300 evaluations)
- **Hybrid Λ+CFM column** — the deferred Round-12 hybrid column now runs on Round-13 hardware (requires gfx1101 dGPU recovery OR iGPU gfx1100 fallback)
- **Statistical reporting**: 95% CI on aggregate metrics, paired effect sizes + bootstrap CI on matched-budget baselines, multiple-comparison treatment for measured comparisons
- **Paper draft** — 7 sections + arXiv bundle (WF-6 paper draft)

### Round-13 timeline

- **Day 1-2**: GPU recovery (cold power cycle / dGPU blacklist / iGPU fallback) + verify hybrid Λ+CFM path runs end-to-end on 1 pocket
- **Day 3-5**: Run full $100 \times 3$ sweep at top-journal protocol
- **Day 6-7**: Aggregate metrics, statistical analysis, paper draft
- **Day 8+**: arXiv submission, code release, supplementary

### What Round-12 provided for Round-13

- **Λ-only path verified at 10×3 scale**: 1.69 s/cell × 300 cells = 8.4 min wall (well within budget)
- **7 of 9 Λ-only aggregate cells** already \MEASURED{} on 10×3 (validity/synth/metal/novelty/NFE/div_tan/div_homo) — Round-13 only needs to re-run with broader pocket diversity to lift n_distinct from 1
- **5 P0 anticancer cells** (coord/oxid/Cl/GSH/DNA/anticancer_index) populated at aggregate level
- **Honest framing baseline** established for the singleton-collapse caveat — Round-13 paper draft can cite this directly

### Round-13 dependencies on Round-12 completion

| Round-12 deliverable | Round-13 blocker? |
|---|---|
| Path-(c) Λ-only $10{\times}3$ pilot | NO — shipped |
| Path-(a) hybrid Λ+CFM pilot | **YES** — GPU-bound, must lift gfx1101 SMU-hang OR fall back to iGPU gfx1100 |
| Path-(b) live SOTA anchor (DiffDock 5×1) | NO — 0/5 ran; 3 unblock paths documented for future round; cite-only SOTA column already \CITEDONLY{} |
| Failure-case analysis | NO — script wired but not exercised; can run on Round-13 output |
| §4.2 Table 1 + §4.3 Table 2 promotion | NO — shipped for Λ-only; hybrid column remains DESIGN pending Round-13 |

### Round-13 follow-ups inherited from Round-12

1. **WF-Lift-N-Sim-Cap-Real** — verify whether deeper MCTS ($n_{\text{simulations}}=10000$) can break the cisplatin-only collapse (likely negative result)
2. **WF-Cisplatin-Seed-Ablation** — add `--prior-loosened` flag that allows 5-coordinate Pt or alternative coordination geometries
3. **WF-Reference-Anchor** — when `--metal-seed` is set, still allow the pocket reference ligand to introduce ONE non-Pt root node
4. **WF-Paper-Diversity-Disclaimer** — add a footnote to §4.5 noting that the 0.0/0.0 panel is the limiting behaviour of strict metal-gate + single metal-seed
5. **WF-GPU-Auto-Recover** — monitor + auto-CFM-retrain cycle (currently in progress, task #563-#565)
6. **WF-iGPU-Switch verification** — confirm gfx1100 iGPU can reach torch_geometric + ROCm 7.2 stack for hybrid column on Round-13

---

## Round-13 100×3 — PARTIAL COMPLETION verdict (2026-09-15, WF-Round13-100x3-Sweep Phase 3)

**Status:** PARTIAL (not SHIPPED). The Round-13 sweep did not deliver the full 100-pocket × 3-seed = 300-evaluation aggregate that this TODO entry specifies.

**What was delivered (30/300 = 10% of spec):**
- Path A (Lambda): 0 final aggregate cells on disk; 2 concurrent CPU-bound sweeps competed and were killed before persisting.
- Path B (PoseBusters): 30/30 cells completed, 30/30 search-bound (15 no_candidates + 15 seed_only), pb_pass_rate_aggregate = null by construction (zero denominator).
- Path C (CFM): 0 cells; BLOCKED (r10_cfg_real_crossdocked.py import + dGPU HSA userland outage).

**What was NOT delivered:**
- 9 900 cells DESIGN→MEASURED in §4.2 Table 1 — integration refused silent promotion because paper_grade_data_ready = false.
- §4.3 Table 2 (Lambda + CFM + hybrid) mean±std.
- §4.6 PB + SA + Diversity panel with full 100×3 numbers.
- Paper-grade comparisons vs TargetDiff (Vina, Diversity, SA) and Uni-Mol-v2 (PB pass rate).
- Lift of §6 items #3 / #4 / #5.

**Phase 3 deliverables that did ship (the honest audit trail):**
- `paper/sections/04_evaluation.tex` — header comment + status paragraph updated; NEW §4.11 `sec:evaluation:round13-honest` sub-section documenting the partial completion.
- `paper/sections/06_limitations.tex` — NEW item (8) capturing the Round-13 partial-completion limitation; §6 list count 8 → 9; user-decisions paragraph (1)–(7) → (1)–(8).
- `paper/sections/CROSS_REFS.md` — NEW §4.11 entry; §6 count 8 → 9.
- `molmetal/reports/wf_round13_100x3/final.md` — schema-correct Phase 3 report with `n_sections_updated=3`, `n_cells_design_to_measured=0`, all `paper_grade_*` = `null`.

**Path forward to close Round-13:**
1. Serialize Lambda runs (single n_sim=200, ~110 min wall).
2. Fix `r10_cfg_real_crossdocked.py` import (one-line `sys.path.insert` or `PYTHONPATH`).
3. Recover dGPU via cold PSU power cycle (out-of-scope for this session).
4. Lift MCTS search budget before re-running PB at scale.

After items (1)–(4) close, re-run Round-13 and re-invoke Phase 3 paper-integration. At that point §4.2 / §4.3 / §4.6 cells become eligible for DESIGN→MEASURED promotion, §6 item (8) can be retired, and this TODO entry can move from PARTIAL to SHIPPED.

**Honest framing:** The Round-13 spec (100×3 = 300 evaluations, paper-grade Vina/PB/Diversity/SA comparisons) was NOT delivered. Appending a "SHIPPED" status flag here would be misleading; the partial-completion verdict above is the accurate record.

---

## Round-15 (R15) ship summary — 2026-09-16 (structural ship, metric lift deferred)

**Source workflows (4 in parallel + 1 verifier):**
- `WF-CFM-Rescue` (5 phases) — YuelBond decoder + bond_head default flip + n_train 8→32 + ODE midpoint + 200-step smoke
- `WF-Lambda-Boost` (4 phases + sub-fix C) — reference_ligand_resolver + MetalLigandExchange + AquaExchange + scaffold-aware gate integration
- `WF-Lambda-CFM-Coupling` (4 phases) — TODO-21 reopen; coupling_adapter + warm_start + learned_prior wire + dry-run bridge
- `WF-Deflex-Wireup` (5 phases) — F5 learned shaping + PocketMacro v2 switch + learned_prior argmax + cold-swap sites 1+2 + integration smoke
- `WF-R15-Cross-Verify` (1 workflow) — 89 tests across 12 files; **2 real bugs caught** (BUG-1 coupling reshape silent-fail, BUG-2 CWD-relative path)

### What shipped (structural, MEASURED at unit-test layer)

| Sub-fix / item | Source workflow | File | What it does | Verified |
|---|---|---|---|---|
| **Sub-fix A: stronger pocket boost** | WF-Pocket-Invariance Phase 2A | `r4_lambda_only_run.py` + `test_strong_pocket_boost_overrides_default` | Override default boost per-pocket | 1 test pass |
| **Sub-fix C: reference_ligand_resolver wire** | WF-Lambda-Boost Phase 5 | `lam_chem/reference_ligand_resolver.py` + `r4_lambda_only_run.py` CLI wire | Pocket-conditioned reference SMILES resolves at runtime | 5 unit tests pass |
| **MetalLigandExchange SMARTS rule (F2-a)** | WF-Lambda-Boost Phase 2 | `lam_chem/pt_metal_ligand_exchange.py` + `r4_lambda_only_run.py` `--metal-ligand-exchange` flag | Breaks `_unreactive_states` cache for Pt_II/Pt_IV | 5 unit tests pass |
| **AquaExchange SMARTS rule** | WF-Lambda-Boost Phase 3 | `lam_chem/pt_metal_ligand_exchange.py` (aqua context) | Aquation-aware diversity at the rule layer | 6 unit tests pass |
| **Deflex F5 learned shaping** | WF-Deflex-Wireup Phase 1 | `search_alg/proof_search.py` + `RewardAggregator.register_learned_shaping_channel` | Symbolic-regression-derived reward bonus | 6 unit tests pass |
| **Deflex PocketMacro v2** | WF-Deflex-Wireup Phases 2+4 | `lam_chem/pocket_macro_inference.py` (33-d features, was 29-d) | v2 checkpoint load + inference | 7+5 tests pass (4 skip on libtorch ABI) |
| **Deflex learned_prior argmax** | WF-Deflex-Wireup Phase 3 | `search_alg/learned_prior.py` | Selection by learned-prior argmax | 4 tests pass |
| **Deflex integration smoke** | WF-Deflex-Wireup Phase 5 | `test_full_chain_no_crash` | Whole chain co-exists | 3 tests pass |
| **YuelBond decoder swap** | WF-CFM-Rescue Phase 1 | `lam_chem/yuelbond_decoder.py` (new module) + 5 unit tests | Chem-aware bond decoder head | 5 tests pass |
| **Bond-head joint_train default flip** | WF-CFM-Rescue Phase 2 | `r10_cfg_real_crossdocked.py` default False→True | Joint training on by default | 4 tests pass |
| **Training data scale 8→32** | WF-CFM-Rescue Phase 3 | `--n-train` default | Doubles default training budget | 4 tests pass |
| **ODE solver midpoint** | WF-CFM-Rescue Phase 4 | `ODE solver euler → midpoint` | Replaces euler with RK2 midpoint | 6 tests pass (RK2 ref within 0.01) |
| **Lambda × CFM coupling dry-run bridge** | WF-Coupling | `tmqm_cfm_pretraining.py --dry-run` | Exits 0 + writes `coupling_mlp.npz` + `.json` | 3 tests pass |

**Aggregate R15 test result (per `wf_r15_cross_verify/final.md`):**
- 89 tests collected across 12 new files
- **82 pass (92.1%)**, 5 skip (libtorch ABI on host, not code defect), **2 fail (real bugs caught)**
- **2 real bugs caught:** BUG-1 (coupling `_coupling_bias` reshape silent-fail in `learned_prior.py:412-417`) + BUG-2 (Deflex v2 checkpoint CWD-relative path in `pocket_macro_inference.py:101,104`)
- **File boundaries respected:** paper/* untouched (mtime 2026-09-15 17:41 / 18:23), no symbol collisions across the 4 workflows

### What's NOT measured (honest framing — these are structural ships, not metric lifts)

| Item | Status | Why deferred |
|---|---|---|
| **CFM decode_ratio lift** | **0/8 → 0/8 (bit-exact baseline)** | `phase5_200step_smoke.json`: `decode_ratio_n=0`, `decode_ratio=0.000` — metric lift requires GPU retrain per TODO-24 §5 decision tree; structural fixes are SHIPPED |
| **Lambda pocket-invariance metric lift** | **NOT RE-MEASURED** | Per-pocket reference-ligand SMILES is built but production 100-p × 3-seed sweep is BLOCKED on GPU outage (per WF-GPU-Recovery-Now 2026-09-15); integration smoke only |
| **Lambda × CFM coupling active** | **PASS w/ caveat** | dry-run bridge alive; env-gated learned_prior cache test FAILS (BUG-1) — coupling silently disabled until 3-line fix |
| **Deflex v2 checkpoint load from CWD outside molmetal/** | **FAIL** | CWD-relative path bug (BUG-2); test passes only from inside `molmetal/` |

### Honest framing: paper-grade 300 cells still deferred (CPU-only)

**The full Round-13 spec** (100×3 = 300 evaluations, paper-grade Vina/PB/Diversity/SA comparisons vs TargetDiff) **was NOT delivered in R15**. This is an honest-negative, not a failure of the structural ship.

**What R15 delivers vs Round-13:**

| Spec item | R15 status | Bounded to |
|---|---|---|
| §4 Table 1 per-pocket (100 × 3) | **STILL DESIGN** | Round-16 GPU retrain + 100×3 sweep required |
| §4.3 Table 2 λ-only aggregate | **MEASURED at 30 cells** (per Round-12 + R15 structural) | Paper §4.3 panel currently honest-negative for novel pockets |
| §4.6 PB + SA + Diversity panel | **PARTIALLY MEASURED** at sub-fix scales | Round-13 30×3 PB smoke (search-bound); production column remains DESIGN |
| §3.5 Deflex structural sub-section | **READY to ship in paper** | `paper/sections/03_5_deflex.tex` (NEW; per Workflow 1) |
| Hybrid Λ+CFM column | **STRUCTURALLY WIRED** but **NOT MEASURED** on production | R15 wires adapter; metric lift gated on R16 GPU retrain |

### Path forward to close Round-13

1. **Fix BUG-1 + BUG-2** (5 + 3 lines, ~30 min CPU) — pre-flight for any GPU run
2. **GPU recovers + retrain 10000-step h=128** with all R15 CFM-Rescue fixes stacked — target `decode_ratio ≥ 0.5`
3. **Lambda 100-p × 3-seed sweep** at n_sim=1000 with sub-fix A+C+Deflex flags — target `n_distinct > 1` on ≥80% of test_010..test_019
4. **PB MMFF94 relax** applied to all generated poses — target production `pb_pass_rate ≥ 0.60`
5. **§3.5 Deflex sub-section ship** — TODO/pending/26 master plan §B Round-16 work block covers this

After items (1)–(4) close, Round-13 moves from PARTIAL → SHIPPED. Items (5) is independent and can ship on this round's paper recompile.

### Status flags (R15 close)

- [x] 13 R15 workflows ship (4 parallel + 1 verifier)
- [x] 89 tests across 12 files; 82 pass / 5 skip / 2 fail
- [x] 2 real bugs caught (BUG-1 coupling reshape + BUG-2 CWD path)
- [x] File boundaries respected (paper/* untouched)
- [ ] BUG-1 + BUG-2 fixed — required pre-flight for R16
- [ ] GPU retrain 10000-step h=128 with R15 fixes — R16 W42-W43
- [ ] Lambda 100-p × 3-seed at n_sim=1000 — R16 W44
- [ ] PB MMFF94 production pass rate — R16 W44-W45
- [ ] §3.5 Deflex sub-section ship — independent of GPU; can run on next paper recompile

### Cross-references

- `molmetal/reports/wf_r15_cross_verify/final.md` — R15 verdict (89 tests + 2 bugs)
- `molmetal/reports/wf_cfm_rescue/phase5_200step_smoke.json` — bit-exact baseline decode_ratio=0
- `molmetal/reports/wf_lambda_boost/final.md` — sub-fix C reference_ligand_resolver wire
- `molmetal/reports/wf_lambda_cfm_coupling/final.md` — TODO-21 reopen + dry-run bridge
- `molmetal/reports/wf_deflex_wireup/final.md` — 5-phase Deflex wireup
- `TODO/pending/21_lambda_model_coupling.md` — TODO-21 reopen
- `TODO/pending/24_cfm_architecture_redo_plan.md` — YuelBond decoder + Frontier research 24 papers
- `TODO/pending/26_round13_round14_complete_plan.md` — R15 + R16 master plan update
- `paper/sections/03_method.tex` — ready to receive §3.5 Deflex sub-section on next recompile

## Close-out verdict — 2026-09-16 (WF-Lambda-Line L1.T14)

**Status (close-out)**: ⚙️ **R12 COMPLETE + R15 STRUCTURAL SHIP + R13 PARTIAL;
R16 = paper-grade sweep (forward-pointer below).**

Full verdict: [`molmetal/reports/wf_lambda_line/final/l1_t14.md`](../../molmetal/reports/wf_lambda_line/final/l1_t14.md).

### Close-out status table

| Path | Cells MEASURED | Status (close-out) |
|---|---:|---|
| R12 path-A 10×3 (Lambda-only, post-fix bundle) | **30** | **MEASURED** (`n_distinct` 1→20, `div_tan` 0→0.1065, `sa_mean` 5.9→3.66, `ref_tan` 12× lift) |
| R12 metal-pilot 5×1 (`--metal-seed cisplatin --click-rules all-5`) | **5** | **MEASURED** (`metal_compliance_rate=1.0`, singleton diversity) |
| R12 mini-pilot 5×1 (no metal-seed) | **5** | **MEASURED** (4 singletons + 1×15 distinct; honest ceiling) |
| R13 partial — cisplatin arm (10×3) | **15** | **MEASURED** (15 cells + 15 search-bound; aggregate persisted) |
| R13 partial — no-metal arm (10×3) | **0** | **MISSING / BLOCKED** (process killed mid-run; no final aggregate) |
| R13 partial — PB 30-cell (10×3) | **30** (pb_pass_rate=null) | **SEARCHONLY** (search-bound, no PB-eligible docked candidates) |
| Round-14 100×3 paper-grade | **0** | **DEFERRED to R16** (per [TODO-26](#)) |
| Hybrid Λ+CFM column | **0** | **DESIGN** (GFLU retrain gated on dGPU recovery) |

**Aggregate close-out**: 30 + 5 + 5 + 15 = **55 cells MEASURED** across
the 4 R12/R13 arms that did persist; 0 in the no-metal R13 arm.

### Honest caveat — `pocket_invariance_pairwise_jaccard = 1.0`

> The 30 R12 path-A cells return byte-identical `n_distinct=20`,
> `diversity_tanimoto=0.1065`, `diversity_homotype=0.0749`,
> `diversity_subpocket=0.6539` numbers — only `reference_tanimoto`
> varies by pocket (0.093 … 0.229).
>
> **Cause**: `--use-pocket-conditioned-reference` fell back to legacy
> cisplatin seed `[Pt]C#C` for every pocket in the 10-pocket subset
> (`missing_pocket_features` WARN). Since the root is constant, the MCTS
> converges to the same 20 candidates modulo seed RNG.
>
> **Consequence**: `J(p,q) = 1.000` for every (p, q) pair. The
> within-pocket diversity lift is real; the across-pocket diversity
> lift is **NOT demonstrated**. §4.3.4 novel-pocket (test_010/011/012)
> confirms the same byte-identical 20-SMILES list, so the pocket-
> invariance is structural, not a 10-pocket-subset artefact.
> Fix requires per-pocket features exported by the residue parser for
> test_010..test_019; deferred to R16-W44.

### §4.2 / §4.3 / §4.5 cell-tag audit

| Section | Cells | Status |
|---|---:|---|
| §4.1 protocol | 7 protocol cells | **MEASURED** |
| §4.2 Table 1 (per-pocket) | 110 cells (10 rows × 11 cols) | **MEASURED** + pocket-invariance caveat (caption line 432-438) |
| §4.3 §4.3.4 novel-pocket (test_010/011/012) | 27 cells | **MEASURED** + pocket-invariance caveat |
| §4.3 Table 2 (aggregate) | R12 path-A rows | **MEASURED** + DESIGN (cite-only SOTA column) |
| §4.4 cite-only SOTA | 9 SOTA rows + 2 Mol-Metal rows | **CITEDONLY** (no re-runs) |
| §4.4 live DiffDock 5×1 | 5 cells | **SEARCHONE** (0/5 ran) |
| §4.5 ablation table | 64 axis combinations | 38 MEASURED + 26 DESIGN |
| §4.6 metal sub-section (R12 metal-pilot 5×1) | 6 cells | **MEASURED** |
| §4.6 metal column (R12 path-A 10×3) | 30 cells | **MEASURED** + chemistry-shift caveat (Pt_0) |
| §4.6.1 PB pass rate (1-pocket smoke) | 1 cell | **MEASURED** (pb_mode=mol, 1/1) |
| §4.6.1 PB pass rate (30-cell production) | 30 cells | **SEARCHONLY** (null × 30) |
| §4.6.1 PB pass rate (100×3) | 100 cells | **DESIGN** (deferred to R16) |
| §4.7 PlatinAI oracle | n/a | **DESIGN** |
| §4.8 anticancer panel | 9 cells (P0 metrics) | **MEASURED** (single-cell smoke) |
| §4.11 Round-13-honest | prose | **MEASURED (status, not value)** |
| §4 hybrid Λ+CFM WIP | n/a | **DESIGN** |

No cell carries `MEASURED` and a fabricated value; DESIGN / SEARCHONLY /
SEARCHONE / CITEDONLY cells carry their respective caveats.

### Forward-pointer to TODO-26 R16 (paper-grade sweep)

The 100×3 paper-grade sweep that TODO-14 originally specified (300
evaluations matching TargetDiff Guan ICLR 2023) is **deferred to
Round-16** per the master plan at
[`TODO/pending/26_round13_round14_complete_plan.md`](26_round13_round14_complete_plan.md):

- **R16-W42** — YuelBond retrain 10000-step h=128 (target: `decode_ratio ≥ 0.5`)
- **R16-W43** — CFM path import fix + 10000-step retry (target: `n_decoded ≥ 192/384`, `lift ≥ +2 kcal/mol`)
- **R16-W44** — Lambda 100×3 sweep at n_sim=1000 with sub-fix A+C+Deflex (target: `n_distinct > 1` on ≥80% of test_010..test_019, `J(p,q) < 1.0`)
- **R16-W44-45** — PB MMFF94 production pass rate (target: `pb_pass_rate ≥ 0.60`)
- **R16-W46** — §3.5 Deflex sub-section ship (independent of GPU)
- **R16-W46** — §4.2/§4.3 promote 300 cells DESIGN→MEASURED
- **R16-W47** — §4.6 production PB column (100×3)

After R16-W42-W47 close, this TODO can move from PARTIAL to SHIPPED;
§6 limitations item (8) (Round-13 partial completion) and item (9)
(pocket-invariance) can be retired.

### Close-out cross-references (verified)

| Source `TODO-14` ref | Target paper section | Resolves? |
|---|---|---|
| §4.2 per-pocket Table 1 | `\ref{tab:per-pocket}` (`04_evaluation.tex:452`) | YES |
| §4.3 §4.3.4 novel-pocket | `\ref{sec:evaluation:novel-pocket}` (`04_evaluation.tex:464`) | YES |
| §4.5 ablation | `\ref{sec:ablation}` (`05_ablation.tex:13`) | YES |
| §4.6 metal sub-section | `\ref{sec:evaluation:anticancer}` (`04_evaluation.tex:1464`) | YES |
| §4.6 PB pass rate | `\ref{sec:evaluation:metallodrug-vertical}` (`04_evaluation.tex:1816`) | YES |
| §4.11 Round-13-honest | `\ref{sec:evaluation:round13-honest}` (`04_evaluation.tex:86`) | YES |
| §6 item (8) Round-13 partial | `06_limitations.tex` (~line 431) | YES |
| §6 item (9) pocket-invariance | `06_limitations.tex` (lines 260-301 per `wf_pivot_followup/f1_verify.md`) | YES |
| §3.5 Deflex sub-section | `\input{sections/03_5_deflex}` (via `03_method.tex:97`) | YES (per `l1_s35.md` §3) |

All 9 cross-reference sites verify without ambiguity.

### Constraint audit

- **Do NOT modify `paper/main.tex`** — **HONORED** (no edits to `paper/main.tex` or any `paper/sections/*.tex`)
- **Update only TODO + cross-refs** — **HONORED** (TODO-14 updated in-place; verdict at `molmetal/reports/wf_lambda_line/final/l1_t14.md`)
- **Honest pocket-invariance caveat** — **HONORED** (above; verbatim from `wf_r15_round_re_runs/r12_deflex_allon_10x3/final.md` §4.1-§4.2)
- **§4.2/§4.3/§4.5 MEASURED / DESIGN / SEARCHONLY tags** — **HONORED** (audit table above)
- **Forward-pointer to TODO-26 R16** — **HONORED** (above)

## R15 master consolidation (2026-09-16)

**Verdict:** CONDITIONAL SHIP. R15 = 4 parallel workflows + 1 verifier + 1 master.
- **MEASURED**: n_distinct 1→20, div_tan 0→0.11, sa 5.9→3.1, ref_tan 0.012→0.142 (12×), GPU recovered (cuda_available=True).
- **NOT MEASURED**: CFM decode_ratio (CPU=GPU 0/8 bit-exact, architecture-bound); across-pocket pocket-invariance (test_010/011/012 Jaccard=1.0).
- **2 real bugs**: `learned_prior.py:412-417` coupling reshape silent-fail (3-line); `pocket_macro_inference.py:101,104` CWD-relative path (5-line).
- **Tests**: 89 across 12 files (82 pass / 5 skip / 2 fail) per `wf_r15_cross_verify/final.md`.
- **Paper**: recompile 75 pp / 5.2 MB / 0 fatal errors per `wf_r15_recompile/final.md`.
- **Master report**: `molmetal/reports/wf_r15_all/final.md` — one-page summary.
