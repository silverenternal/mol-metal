# Pitfall Audit MASTER — gap consolidation 2026-09-17

**Scope**: Consolidate 6 per-layer audit reports (P1-P6, 22 pitfalls
across reaction rules, generator architecture, reward design, eval &
novelty, wet-lab validation, data & reproducibility). **Audit-only —
no code modifications.**

**Per-layer audits**:
- `p1_reaction_rules.md` (4 pitfalls) — REACTION_RULES 7 + SyntheMol 13
- `p2_generator_arch.md` (5) — MCTS + CFM + chem-aware decoder
- `p3_reward_design.md` (4) — 14-channel RewardAggregator
- `p4_eval_novelty.md` (3) — 33-cell Path A + 25 TargetDiff metrics
- `p5_wetlab_validation.md` (3) — paper §6/§7 honest framing
- `p6_data_reproducibility.md` (3) — splits + hashes + OOD protocol

**Total: 22 pitfalls across 6 layers.**

---

## TL;DR

**Status counts (22 pitfalls)**:
- **AVOIDED**: 2 (P2.1 reward hacking, P5.1 wet-lab claim)
- **PARTIAL**: 11 (plumbing shipped, opt-in or narrow scope)
- **OPEN**: 9 (no infra or wrong scope — must be engineered)

**High-level**:
- Lambda MCTS **structurally avoids** RL failure modes (no policy
  gradient, no learned reward model); `leaf_value_var` is a safety
  belt. **P2.1 = AVOIDED.**
- `pocket_macro_inference` + `reference_ligand_resolver` + 3D
  conformer path **DO** address 3D-aware generation, but opt-in.
  **P2.2 = PARTIAL.**
- §6 limitations + §7 future work disclaim wet-lab validation.
  **P5.1 = AVOIDED.**
- **Load-bearing OPEN**: P2.3 (layered), P2.4 (skeleton jump),
  P4.2 (patent axis), P3.3 (Pareto operator, 440 LOC unused).

---

## Layer-by-layer table

| Layer | AVOIDED | PARTIAL | OPEN | Total | Highest-EV OPEN |
|---|---|---|---|---|---|
| P1 Reaction rules    | 0 | 2 | 2 | 4 | P1.4 known-Pt regression tests (8h) |
| P2 Generator arch    | 1 | 2 | 2 | 5 | P2.3 layered generation (12h, root-cause of singleton collapse) |
| P3 Reward design     | 0 | 3 | 1 | 4 | P3.3 wire pareto.py (2h, 440 LOC unused) |
| P4 Eval & novelty    | 0 | 1 | 2 | 3 | P4.2 patent / known-Pt-drug axis (4-6h, closes 2 P4.x in one) |
| P5 Wet-lab validation| 1 | 1 | 1 | 3 | P5.2 wet-lab feedback plumbing (6h, future collaborator) |
| P6 Data & repro      | 0 | 3 | 0 | 3 | — (all PARTIAL; no OPEN) |

**Honest aggregate**: Lambda's *plumbing* is mature (44 test files,
1585 pytest pass, 13/14 testing packages, 14 reward channels, 23/25
TargetDiff metrics wired). Missing: *policy* (which channels default-
on), *action-space* diversity primitives (layered + edit), and the
*patent* + *wet-lab-feedback* axes for publication-grade honesty.

---

## Top 5 OPEN pitfalls ranked by engineering value (EV)

EV = paper-strengthening × implementation affordability. Opinionated.

### 1. P4.2 — Patent / known-Pt-drug similarity axis (EV = highest)
Closes P4.2 + P4.3 patent axis in one 4-6h CPU patch. Adds
`metric_max_sim_known_pt` + Bemis-Murcko `scaffold_in_known_pt_drugs`
+ patent-risk-flag column. Default metal-seed is cisplatin — *every*
generated molecule is a cisplatin derivative. Without MaxSim-vs-
cisplatin, `novelty=1.000` is meaningless. **Single highest-leverage
patch in this audit.**

### 2. P3.3 — Wire pareto.py into candidate ranker (EV = highest, cost = lowest)
440 LOC of NSGA-II in `pareto.py` with **zero import sites**. A 2-line
patch (import + call `rank_population` at `proof_search.py:3559-3593`)
plus a `--postprocess-pareto` CLI flag diversifies top-K by
construction — without needing to break pocket-invariance first.
Mitigates P4.1 sample-size problem via output diversity. **Cheapest,
broadest-blast-radius fix.**

### 3. P2.3 — Layered generation (scaffold → attach → decorate) (EV = high)
*Structural root cause* of singleton collapse (WF-Round12-Lambda-Pilot
`n_distinct=1` × 30 cells). All reward-side fixes cannot lift
`n_distinct` when the action space is one-step forward-only. Phase
classifier + phase-bias PUCT + per-phase metrics + tests + pilot =
12h CPU. **Largest-diversity-lift patch, medium risk.**

### 4. P1.4 — Known-Pt regression tests (EV = high, cost = low)
Pure insurance against future SMARTS drift. 8h CPU adds
`test_pt_metal_ligand_exchange_known_complexes` (cisplatin /
carboplatin / oxaliplatin reference set) + AquaExchange second-
aquation regression + 35-cell `pt_click_compat` smoke + priors sync.
Catches any future contributor who breaks Pt chemistry. **No GPU.**

### 5. P2.5 — MD relaxation + pharmacophore channel wire-in (EV = high)
Lifts PB pass-rate beyond the 60-80% MMFF94 ceiling (per
WF-PB-MMFF94-Relax). OpenMM already installed (per WF-Round-7); no new
deps. 16h CPU total but split-able: Phase A (`r_pharmacophore` wire-in,
2h) is independently shippable. **MD is the longest-pending stage —
installed Round-7, never consumed.**

### Honourable mentions (NOT in top 5)
- **P2.4** skeleton jump / fragment-swap (15h, med-high risk): high
  upside but high risk of invalid structures; sequence after P2.3.
- **P5.2** wet-lab feedback plumbing (6h): enables integration
  if/when collaborator provides data.
- **P1.1** non-click med-chem rules (10h): expansion, not bug fix;
  Round-15+ scope.

---

## What Lambda MCTS architecture DOES avoid (honest framing)

1. **RL instability**: Lambda is MCTS, not RL. No policy gradient, no
   REINFORCE, no critic, no replay buffer. The reward-hacking failure
   surface is structurally absent. `leaf_value_var` /
   `_leaf_value_history` (`proof_search.py:2657-2669`) are shipped as
   monitors but the underlying risk is zero. **P2.1 = AVOIDED.**

2. **3D-aware generation via pocket_macro**: `pocket_macro_inference`
   ships a per-pocket 32-d embedding (`pocket_macro_inference.py:387`)
   consumed by `modify_root_prior` at `proof_search.py:3014-3139`.
   When `use_pocket_macro=True` + `use_pocket_conditioned_reference=True`,
   the MCTS *root state* is swapped for a pocket-conditioned reference
   ligand SMILES via `reference_ligand_resolver.py:243`. No pure
   RL-from-scratch model has this — the search is conditioned on the
   pocket from rollout zero. **P2.2 = PARTIAL (opt-in).**

3. **Honest wet-lab framing**: §6 explicitly conditions therapeutic
   claims on future wet-lab validation. No molecule is claimed
   synthesised or tested. pIC50 column honestly labelled MEASURED-but-
   not-wet-lab with `pearson_r=0.181` ≤ ridge baseline 0.572 (per
   `wf_pic50_margin_sweep_verdict.md`). **P5.1 = AVOIDED.**

4. **Synthesis-as-proof**: every candidate is produced by a chain of
   β-reductions (each is a named reaction), so *every molecule has a
   proof of synthesizability* in the search tree. RL-from-scratch
   generators cannot make this guarantee.

5. **Closure-theorem property**: the productive space is provably
   closed under the reaction rules (per WF-Lambda-4). Property tests
   catch Au_III arity bugs that downstream unit tests miss. **Unique
   to Mol-Metal among generative models.**

---

## Recommended sequencing (12-week paper-grade plan)

**Tier 1 — ship next (≤ 1 dev-week, 64h CPU)**:
- P3.3 wire pareto.py (2h) — cheapest, broadest blast-radius
- P4.2 patent axis (4-6h) — closes 2 pitfalls, biggest reviewer defence
- P1.4 known-Pt regression tests (8h) — pure insurance
- P5.2 wet-lab plumbing Tier 1 (6h) — unlock future collaborator
- P2.5 pharmacophore wire-in Phase A (2h) — 1-line aggregator hook
- P2.3 layered generation (12h) — root-cause singleton collapse
- P2.2 always-on 3D/pocket (9.5h) — opt-in → default flip
- P1.3 buyable-gate wire-in (12h) — partial fix for singleton
- P5.3 `register_rule` + `vet_rule` (11h) — extensibility convention

**Tier 2 — next sprint (1-2 dev weeks, ~55h CPU)**:
- P1.2 FG-compat veto (9h) — after P1.3/P1.4; risk of over-narrowing
- P1.1 non-click med-chem rules (10h) — expansion, not bug fix
- P2.4 skeleton jump / fragment-swap (15h) — needs P2.3 first
- P2.5 MD relaxation Phases C-E (14h) — needs ROCm OpenMM validation
- P6.3 Tanimoto<0.4 + MMseqs2-30% (6h) — needs MMseqs2 binary
- P4.1 generalisation n=100×3 (8h) — depends on R-13 re-execution
- P5.2 wet-lab outreach (1h) — needs chemistry facility

---

## Cross-cutting observations

1. **Singleton collapse is the binding constraint**. P2.3 + P2.4 + P1.3
   are partial fixes for the same root cause: the search cannot
   generate diverse *edits* of a collapsed state. Reward-side fixes
   (P3.1 ADMET default-on, P3.2 novelty channel, P3.4 anti-collapse)
   are necessary but not sufficient without action-space primitives.

2. **Patent axis (P4.2) is the highest reviewer-defence EV**. Any
   reviewer looking at `novelty=1.000` will ask "novel compared to
   what?". Current answer: "CrossDocked2020 train split" — not
   credible patent risk.

3. **Pareto operator (P3.3) is the cheapest diversity lift** — 2h vs
   12h+ for layered generation. Should ship first.

4. **No pitfall is AVOIDED-only-by-honest-framing**. P2.1 and P5.1
   are AVOIDED because of *architectural choices* (MCTS-not-RL and
   no-wet-lab-by-design). The other 20 need engineering.

5. **22 pitfalls × 39-78h total CPU ≈ 1-2 dev-weeks**. All Tier-1+2
   fit a 2-week sprint. No GPU required.

---

## Files referenced

- `molmetal/reports/wf_pitfall_audit/p{1..6}_*.md` (6 per-layer audits)
- `TODO/pending/30_pitfall_reinforce_plan.md` (companion TODO)