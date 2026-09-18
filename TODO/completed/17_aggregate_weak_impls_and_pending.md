# Active implementation and evidence gaps — 2026-09-13

This living audit supersedes earlier stale installation blockers, proposed
threshold-based explanations for weak results, and unverified completion claims.
Historical reports remain evidence of their original runs, not current behavior.
The user has authorized execution and parallel agents; earlier “do not run
experiments” or “wait for user to unlock” notes do not apply to this session.

## Environment and data

| Item | Verified state | Remaining action |
|---|---|---|
| ROCm | torch 2.14 / HIP 7.2 / gfx1101; model, ADMET and POT GPU operations execute | Track actual solver/device and convergence; profile before claiming speedups |
| OpenMM | OpenCL AMD context executes dynamics | Validate molecular force fields separately; this does not prove GPU docking |
| Native docking | Vendored binary is official QuickVina 2, available through qvina/quickvina2 aliases; native Vina available | N=50 independent-engine parity under matched protocol; aliases are not independent engines |
| GPU docking | Official QuickVina2-GPU 2.1 built for AMD; project adapter and 90 real gfx1101 search/grid calls traced | GNINA remains CUDA-only/CPU fallback; event timestamps invalid, no speedup claim. GPU/native protocols reported separately |
| CrossDocked | All 100 exact test pairs resolve under local storage extraction | Execute proper physical experiments; never select alphabetically sorted directories |
| First ten receptors | 9/10 pass strict or explicitly recorded conformer/name normalization | test_005 / original 4RN0 ASP B101 lacks CG/OD1/OD2 even in crystal; any completion must be separately labelled modeled atoms |
| REINVENT4 | Official pretrained RNN generates and scores NLL on gfx1101; dedicated isolated learned RPC tested | NLL is model likelihood; original RDKit multiproperty proxy is separate and cannot establish activity |
| AiZynth | Official legacy_v3 USPTO weights + matching templates + ZINC stock run true bounded route search; isolated Lambda gate executes | Current-version asset URLs still fail; legacy protocol explicit. Equivalent AMD policy verified, actual GPU strategy search being integrated |

Evidence: `molmetal/reports/amd_eval_capabilities.{json,md}`,
`crossdocked_first10_resolved/report.json`, `quickvina2_binary_identity.md`.

## Lambda line

- **Generated products:** `seed_strategy=click_tile` uses declared tiles and a
  real reaction witness; reference-ligand initialization is explicitly separate.
  Standard-12/CuAAC smoke produces three actual products. A truncated
  extended library with no viable reaction pair correctly fails initialization.
- **Physical evaluation:** `r4_c_full_sweep.py --physical-docking` prepares the
  exact receptor, redocks the reference under the same engine/box/seed, saves
  actual generated poses, and checks all configured PoseBusters booleans.
  test_001 smoke docked three products, all 22 PB checks passed for each,
  but none beat the redocked reference. This is a useful negative result.
- **Chemistry preservation:** Meeko conformers map to energy rows; domain
  molecules retain bonds/charges, and original RDKit docked poses retain stereo
  for SDF/PB. Reembedding SMILES is not docked-pose validation.
- **Search remains pocket-independent:** post-search physical evaluation is
  not a closed-loop pocket-conditioned reward. Connect that reward and compare
  under matched budgets before claiming SBDD search performance.
- **Prior:** a serializable linear descriptor prior now has a real PUCT path,
  development-only fitting and frozen test inference. It is not PySR symbolic
  regression. Real frozen-state ablation and the original symbolic backend
  acceptance criteria remain separate work.
- **Synthesis:** a true configured AiZynth gate and an explicitly labelled
  SMARTS heuristic are different backends; missing learned assets cannot count
  as successful execution of a requested learned gate.
- **Pilot:** bounded N=10 x 3-seed physical integration has been measured with
  reduced search/docking budgets. Full prescribed budget, novelty against
  training data, representative metal pockets and meaningful ablations remain.
- **Metrics:** report finite-value n/mean/sample SD and all-generated coverage
  denominators. New-to-seed is not training novelty. Triple-threshold and metal
  heuristic scores are descriptive research metrics, not clinical efficacy.

## Model line

- POT's invalid `ot.ot.sinkhorn` call was corrected to public `ot.sinkhorn`
  on GPU tensors using log-domain Sinkhorn; CPU discrete rounding is explicit.
- Synthetic and real QM9 experiments are separate artifacts. Real samples are
  checked against original SDF geometry/atom identities and train/validation
  canonical overlap is reported.
- Real-data calibration exposed insufficient Sinkhorn convergence at 200
  iterations. Increasing iterations under the same objective was measured;
  a valid discrete permutation alone does not establish a converged plan.
  Original runtime achieved90/90 convergence at2000 iterations. Newly found
  cross-graph aggregation defects require separately labelled model reruns.
- Required CFG/metal-prior ablations, larger training/evaluation, learned SOTA
  checkpoint comparisons, and claimed Vina improvements remain unproved.

## Experiment integrity and remaining deliverables

- Runtime code is frozen per experiment. The driver records dependency hashes
  and stops before mixing changed code into later jobs. The interrupted first
  physical run is retained for debugging and excluded from experiment totals.
- Partial search and completed pose records survive a physical timeout;
  independent jobs continue. One environment failure does not stop other lines.
- 100 x 3 full experiment, statistical comparisons against measured baselines,
  metal/GSH/DNA physical evidence and paper assembly remain actionable work.
- Avoid significance tests against cite-only published aggregates: they lack
  matched samples and often use different protocols. Do not relax thresholds
  retrospectively to turn negative results into success.

See `../completion_audit_2026-09-13.md`, `../project_workflow.md`, and atomic
pending tasks 05, 07, 11–15 for the live acceptance criteria. Archived tasks
prove only their dated evidence and must not override newly found defects.
