# Project Roadmap

**Status:** living doc
**Last updated:** 2026-09-13

## Phase 0 — Skeleton + ports [DONE]
**Status: complete.** All ports are written and adapters are MVP:
- `molmetal/ports/{generator,docking,predictor,scorer,design_loop,validator,retrosynthesis_checker}.py`
- `molmetal/adapters/{flow_matching_lipman,egnn_predictor,egnn_rocm,rdkit_predictor,mock}.py`
- `molmetal/molmetal_lam/sbdd_env/{vina_adapter,posebusters_adapter,aizynth_adapter,...}.py`
- Lipman 2023 FM clone at `molmetal/references/flow_matching/`
- See TODO/completed/01, 02, 03, 04 for the three external ports.

## Phase 1 — Closed-loop implementations [revalidation in progress]
**Status: archived implementations exist; end-to-end evidence remains incomplete.**
New audits corrected missing chemistry, failed PoseBusters embedding and a CPU OT fallback.
The following shipped components require the current execution evidence in the audit:
- TODO-01 closed_loop rewire (Vina + PB + ADMET)
- TODO-02 proof_search prior (RewardAggregator-based)
- TODO-03 L3 tile wireup (204-tile SMARTS-diverse pool)
- TODO-04 QVina swap (--engine flag)
- TODO-06 PoseBusters MMFF94 fix
- TODO-08 tmQM EGNN wireup (key-shape bridge in round-9)
- TODO-09 square-planar Pt(II) prior + dative-bond EGNN
- TODO-10 mini-batch OT coupling (Tong 2023)
See TODO/completed/01, 02, 03, 04, 06, 08, 09, 10.

## Phase 2 — Algorithm strengthening [pending — round-10]
**Status: design + pending items.** Goal: before the top-journal sweep,
strengthen every Lambda-unique axis and prove each contributes on a controlled
micro-benchmark. Model and Lambda ablations run independently of environment repairs.
- 5 click reactions fully wired into MCTS (CuAAC, SPAAC, thiol-ene, Suzuki,
  amide coupling)
- Mini-batch OT coupling ablation harness (Tong 2023)
- Classifier-free guidance (CFG)
- Square-planar Pt(II) prior ablation (prove it lowers Vina)
- β-NF + AST formal semantics appendix (confluence + strong normalization)
- Per-component sanity metrics harness (9-layer MLC)
See `TODO/pending/11_algorithm_strengthening_r10.md`.
**Ship target: end of W3 (2026-10-03).**

## Phase 3 — Top-journal protocol alignment [pending — round-11]
**Status: local native docking and exact test data are available.** QuickVina 2
binary identity and real adapter docking are verified. Learned integrations and
independent-engine parity still require measurement; they do not block other experiments.
- QVina binary install — **bundled static binary discovered and wired**
- REINVENT4 install — **official learned prior and dedicated NLL RPC verified on AMD; multiproperty scoring is separate**
- QuickVina 2 binary available; N=50 Vina-vs-QuickVina protocol parity remains
- CrossDocked2020 100-pocket staging (Luo 2021 split) — **staged in `/mnt/storage/data/molmetal`**
- Empirical engine parity study (N=5 × 10 mols × both engines)
- SOTA checkpoint fetch + smoke (best-effort, no eval)
See `TODO/pending/12_qvina_data_staging_r11.md`.
**Ship target: end of W4 (2026-10-10).**

## Phase 4 — Pilot + full sweep [pending — round-12 + round-13]
**Status: native and GPU physical integration have been measured.** New click products now pass through
real receptor docking and strict PoseBusters. First-ten receptor preparation is
9/10 successful with provenance; one genuinely incomplete crystal sidechain
is handled as a separate modeling experiment. Reduced-budget integration is
not the full scientific pilot.
- Round-12: N=10 × 3-seed pilot at top-journal metric standard, ablation
  matrix, failure-mode coverage. See
  `TODO/pending/13_top_journal_pilot_r12.md`.
- Round-13: Full 100-pocket × 3-seed sweep + paper draft (Digital
  Discovery Q1 / J. Chem. Inf. Model. Q1 target). See
  `TODO/pending/14_full_100pocket_paper_r13.md`.
**Ship target: end of W7 (2026-10-31) for paper draft.**

## Remaining environment/integration work
- REINVENT4 official learned RNN inference runs in the isolated ROCm project
  `/mnt/storage/env-projects/reinvent4-rocm`; its dedicated NLL RPC preserves
  invalid inputs and actual model/device metadata. This does not turn the
  original RDKit multiproperty bridge into learned activity scoring.
- AiZynthFinder4.4.1 with official legacy_v3 model/templates and ZINC stock
  executes real route search, including the isolated Lambda gate. Latest
  model assets remain inaccessible; keep legacy provenance explicit. Equivalent
  torch AMD policy inference has been verified and actual search integration
  is underway.
- Full100 remains pending pilot quality, closed-loop reward and the specified
  scientific protocol. Available data/docking and learned backends let these
  tasks progress without waiting for every optional backend.

## Cross-cutting
- See `risks.md` for open blockers
- See `decisions.md` for pending architecture choices (D6 REINVENT4 install
  path; D7 QVina swap activation — both default values set in round-9/11)
- See `17_aggregate_weak_impls_and_pending.md` for consolidated weak
  implementations, bad results, missing env, and pending tasks
- See `completed/` for the 23 archived items (rounds 1-8 closed)
- See TODO/README.md for navigation

Current environment/GPU and experimental evidence: `../completion_audit_2026-09-13.md` and `17_aggregate_weak_impls_and_pending.md`. The active goal is not complete.
