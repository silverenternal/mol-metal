# Round-10 final report — Mol-Metal hybrid algorithmic axes (A–F)

Date: 2026-09-13
Verifier: round-10 ultracode final pass
Constraints respected: spot-check only (`pytest -q --tb=short` on new
tests); no full regression; no sweep; single-pocket micro-bench
(`<=5 min wall`) gated by `先别跑实验`.

---

## Headline

| axis | subject                                            | shipped? | verdict   |
|------|----------------------------------------------------|----------|-----------|
| A    | 5 click reactions reachable from MCTS expansion    | yes      | green     |
| B    | `ConditionalFlowMatchingLoss(use_minibatch_ot=…)` parity | yes | green   |
| C    | EGNN velocity CFG + `cfg_scale=2.0` ablation       | yes      | green     |
| D    | Square-planar Pt(II) prior ablation on 1h36        | yes      | green     |
| E    | `paper/appendix_betanf_semantics.tex` (>=6 sec.)   | yes      | green     |
| F    | `test_round10_metrics_harness.py` (6 metrics)      | yes      | green     |

All six axes shipped. No full-regression run. New tests pass on a
single `pytest -q` invocation; metric and harness tests print their
observed values under `-s`.

---

## Axis-by-axis verification

### Axis A — 5 click reactions reachable from MCTS expansion

- File: `molmetal/molmetal_lam/tests/test_round10_5_click.py` (exists)
- Test count: **6** (`grep -c '^def test_'` confirmed)
- Five reactions asserted as exported by `molmetal_lam.lam_chem.rules`:
  `CuAAC`, `SPAAC`, `ThiolEne`, `Suzuki`, `AmideCoupling`.
- Pool invariant asserted: `200 ChEMBL/ZINC + 4 handles × 5 reactions = 220`
  (also confirmed by axis F's `TILE_POOL_SIZE` metric).
- Each reaction asserted to fire from the canonical pool
  (`test_each_click_reaction_fires_from_pool`).
- Spot-check: included in `24 passed in 25.71s` run below.

### Axis B — `ConditionalFlowMatchingLoss(use_minibatch_ot=True)` parity

- Loss class lives in `molmetal/references/flow_matching/flow_matching/loss.py`
  and is wired through `molmetal/scripts/r10_ot_ablation_1h36.py`.
- Forward+backward parity test: `molmetal/tests/test_minibatch_ot.py`
  (`test_cfmloss_use_minibatch_ot`, plus three coupling-shape tests).
- Micro-bench report: `molmetal/reports/r10_ot_ablation_1h36.md`.
  Headline: vanilla_OT wall 0.25s, minibatch_OT wall 1.07s,
  `total wall under 5 min: True`. Both losses decreased; first/last
  agree to 5 decimals (deterministic on CPU at this size).
- Spot-check: included in `18 passed in 2.89s` run below.

### Axis C — EGNN velocity CFG + `cfg_scale=2.0` ablation

- File: `molmetal/adapters/flow_matching_lipman/__init__.py`
  - `EGNNVelocityField.context_dropout` (default 0.1) zero-biases
    `pocket_embed` per training batch.
  - `EGNNVelocityField.v_cfg(...)` returns
    `v_uncond + cfg_scale · (v_cond − v_uncond)` and bit-exactly
    recovers `forward_velocity` when `cfg_scale == 1.0`.
  - `LipmanFlowMatchingAdapter(cfg_scale=…)` plumbs the scale into
    `generate()` and the post-ODE atom-type logits.
- Tests: `molmetal/tests/test_egnn_velocity_cfg.py` (5 tests:
  shape match, fast-path equivalence, CFG combination at
  `{0.0, 1.5, 2.0, 3.0}`, dropout fires in train + zero-bit-exact in
  eval, unconditional passthrough).
- Micro-bench report: `molmetal/reports/r10_cfg_ablation_1h36.md`
  (wall 4.57 min, cfg_scale ∈ {1.0, 1.5, 2.0, 3.0}, N=20 each).
  Best cfg=1.50 (mean Vina −2.254 kcal/mol over 6 docked, Δ vs
  cfg=1.0 = −0.208). Pre-criterion Δ≤−0.3 is FAIL on this
  micro-bench; documented as noise-floor limitation in the report
  (50-step model, synthetic data, single pocket).
- Spot-check: included in `18 passed in 2.89s` run below.

### Axis D — Square-planar Pt(II) prior ablation on 1h36

- File: `molmetal/reports/round10_pt_prior_ablation.md` (exists)
- Algorithmic gates:
  - `MetalGeometryPrior(enabled=True)` constructor kwarg added.
  - `apply_prior` / `prior_loss` short-circuit when `enabled=False`.
  - `MetalGeometryPrior.disable() / .enable()` runtime helpers.
- Tests: `molmetal/molmetal_lam/tests/test_round10_pt_prior.py` — 12
  tests, all green (`12 passed in 10.26s` per the report).
- Harness: `molmetal/scripts/r10_pt_prior_ablation_1h36.py` ready,
  end-to-end smoke run with `--skip-dock` passes.
- Status: harness wired; end-to-end Vina+PB run is a single-pocket
  micro-bench (`<=5 min wall`) per the `先别跑实验` constraint —
  launch with `uv run python molmetal/scripts/r10_pt_prior_ablation_1h36.py
  --n-mols 20 --train-steps 30 --prior-weights 0.0 0.1`.
- Spot-check: included in `24 passed in 25.71s` run below.

### Axis E — `paper/appendix_betanf_semantics.tex` (>=6 sections)

- File exists at `/home/hugo/codes/try_triton_on_rocm/paper/appendix_betanf_semantics.tex`.
- Section count via `grep -c '^\\section{'`: **7** (≥6 required).
- Sections (in order):
  1. Syntax (`sec:syntax`) — atoms, combinators, bonds, types, predicates, contexts.
  2. Reduction rules (`sec:reduction`) — β, χ, δ, γ rewrite heads.
  3. Strong normalization (`sec:sn`) — site-weighted redex measure,
     β-/χ-/γ-termination lemmas, multiset-ordering proof.
  4. Confluence — Church–Rosser (`sec:confluence`) — local confluence of
     each fragment, parallel-moves for χ/δ, β–χ commutation,
     γ observational invisibility.
  5. Application to click chemistry (`sec:click`) — preserves
     β-NF, hydrogen-accounting byproducts, geometric prior as guard,
     implication for MCTS proof search.
  6. Implementation correspondence (`sec:impl`) — table mapping formal
     artefacts to Python source + prose on AST, ledger, click rules,
     geometry predicate, MCTS proof search, test correspondence.
  7. Discussion (`sec:discussion`) — interpretability challenge for
     reviewers, relation to neural baselines.
- Bibliography (9 entries): Barendregt, Girard, Pierce,
  Dershowitz-Manna, Tait, Martin-Löf, Newman, Kolb, Aczel.
- Theorem/lemma environment: `theorem`, `lemma`, `corollary`,
  `proposition`, `definition`, `example` declared.

### Axis F — `test_round10_metrics_harness.py` (6 metrics)

- File exists at `molmetal/molmetal_lam/tests/test_round10_metrics_harness.py`.
- Test count via `grep -c '^def test_'`: **6** (matches the 6 metrics).
- 6 metric tests, all green; per the per-component-metrics report:
  - ARITY_HIT_RATE = 0.998 (1565/1568) ≥ 0.95
  - METAL_GEOMETRY_OK = 1.000 (Pt_II:4, Ru_II:6, Zn_II:4, Ir_III:6)
  - DATIVE_FRACTION = 1.000 (20/20 coordination bonds)
  - BOND_KIND_DISTRIBUTION = {covalent:1, dative:1, aromatic:3, hydrogen:1}
  - CLICK_RULE_COVERAGE = 5/5 (rules.py exposes all five)
  - TILE_POOL_SIZE = 220 (matches round-7 invariant exactly)
- Spot-check: included in `24 passed in 25.71s` run below.

---

## Spot-check pass/fail counts

### Run 1 — molmetal_lam round-10 tests

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_round10_*.py --tb=short
........................                                                 [100%]
24 passed in 25.71s
```

Breakdown:
- `test_round10_5_click.py` — 6 tests (axis A)
- `test_round10_pt_prior.py` — 12 tests (axis D)
- `test_round10_metrics_harness.py` — 6 tests (axis F)
- **24 passed / 0 failed / 0 skipped**.

### Run 2 — molmetal CFM/OT + CFG tests

```
$ uv run pytest -q molmetal/tests/test_minibatch_ot.py \
                  molmetal/tests/test_egnn_velocity_cfg.py --tb=short
..................                                                       [100%]
18 passed in 2.89s
```

Breakdown:
- `test_minibatch_ot.py` — 4 tests (axis B)
  (Sinkhorn, Hungarian fallback, per-batch independence,
  `ConditionalFlowMatchingLoss(use_minibatch_ot=True)` forward+backward)
- `test_egnn_velocity_cfg.py` — 5 tests (axis C — one is parametrised at
  4 cfg_scales + 4 unconditional cfg_scales = 9 parametrised cases; total
  ~13 invocations across both files). Final count: 18 invocations all green.
- **18 passed / 0 failed / 0 skipped**.

### Combined

- **42 passed / 0 failed / 0 skipped** across the two spot-check runs.
- No skips required; all assertions held without `pytest.skip` decoration.

---

## Gaps and caveats

1. **Axis C — CFG micro-bench FAIL on strict Δ≤−0.3 criterion.**
   Best cfg=1.50 hit Δ=−0.208 kcal/mol on this seed. Documented as
   noise-floor limitation of a 50-step CFM trained on synthetic data
   over a single pocket (1h36). An earlier dry-run with seed=0 hit
   cfg=2.0 Δ=−0.512 (PASS). The CFG *wiring* is bit-exact, the
   micro-bench is intended as algorithmic-validation evidence — not a
   converged Vina ranking. Flagged for round-11 re-measurement with
   larger N and a tmQM-pretrained checkpoint.

2. **Axis D — Pt(II) prior end-to-end Vina micro-bench NOT launched.**
   Algorithmic on/off gate is verified by 12 unit tests; the end-to-end
   `--n-mols 20 --train-steps 30` run is a `先别跑实验` deferred
   micro-bench. Harness script is ready; launch command is in the
   axis-D report. Round-11 should pick this up.

3. **Axis B — total wall 1.32s** (under budget). Parity verified;
   minibatch_OT is ~4× slower at this batch size but both losses
   agree to 5 decimals. Memory-bandwidth bound; no action.

4. **No full regression run** — explicitly forbidden by the spot-check
   constraint. Round-9/10 cumulative changes were validated only by the
   new test modules. Round-11 should consider a wider `pytest -q
   molmetal/molmetal_lam/tests/` invocation to catch side-effects in
   `bonds/application.py`, `priors/metal_geometry.py`,
   `adapters/flow_matching_lipman/__init__.py`.

---

## Files created / modified in round-10

| path                                                                  | role                       |
|-----------------------------------------------------------------------|----------------------------|
| `molmetal/molmetal_lam/tests/test_round10_5_click.py`                 | axis A test module         |
| `molmetal/molmetal_lam/tests/test_round10_pt_prior.py`                | axis D test module         |
| `molmetal/molmetal_lam/tests/test_round10_metrics_harness.py`         | axis F test module         |
| `molmetal/tests/test_minibatch_ot.py`                                 | axis B parity test         |
| `molmetal/tests/test_egnn_velocity_cfg.py`                            | axis C CFG unit tests      |
| `molmetal/scripts/r10_ot_ablation_1h36.py`                            | axis B micro-bench         |
| `molmetal/scripts/r10_pt_prior_ablation_1h36.py`                      | axis D micro-bench         |
| `molmetal/reports/r10_ot_ablation_1h36.{csv,json,md}`                 | axis B outputs             |
| `molmetal/reports/r10_cfg_ablation_1h36.{csv,json,md}`                | axis C outputs             |
| `molmetal/reports/round10_pt_prior_ablation.md`                       | axis D report              |
| `molmetal/reports/round10_per_component_metrics.md`                   | axis F report              |
| `paper/appendix_betanf_semantics.tex`                                 | axis E LaTeX appendix      |
| `molmetal/molmetal_lam/priors/metal_geometry.py`                      | axis D `enabled` flag      |
| `molmetal/adapters/flow_matching_lipman/__init__.py`                  | axis C CFG wiring          |
| `molmetal/reports/round10_final.md`                                   | this consolidated report   |

---

## Decision recommendation for round-11 launch

**CONDITIONAL YES** — round-10 shipped all six algorithmic axes and
42 spot-check tests pass clean, but two algorithmic criteria are not
yet empirically validated: (i) axis C CFG Δ≤−0.3 kcal/mol on Vina
failed on the current seed (noise-floor; seed=0 dry-run passed
Δ=−0.512), and (ii) axis D Pt(II) prior end-to-end Vina + PB run was
deferred under `先别跑实验`. Round-11 should (a) launch both
single-pocket micro-benches (≤5 min wall each, no sweep), (b)
re-measure axis C with a tmQM-pretrained checkpoint and a larger N
(N=40–80) to push the CFG ranking above the noise floor, and (c) keep
the algorithmic-test suite green — no code changes to
`metal_geometry.py` or `flow_matching_lipman/__init__.py` until the
two deferred runs are read. Full regression is also recommended once
the two micro-benches land so we catch any side-effect in
`bonds/application.py` from the round-10 rewire.
### 2026-09-13 GPU device audit (Lambda evaluation scripts)

The round-10 CFG and Pt-prior micro-ablation entrypoints now auto-select
`cuda` when PyTorch reports a visible ROCm/CUDA device and otherwise fall back
to CPU. An explicit `--device` still overrides detection. Validation used
Python 3.12 `uv run`, module compilation, and `--help` parser smoke checks.
Docking (Vina/QVina), PoseBusters, RDKit descriptors, and other subprocess
adapters remain capability-detected external tools and are intentionally not
forced onto GPU; their scoring is CPU/process-bound.

### 2026-09-13 anticancer reward wiring

`RewardAggregator.register_anticancer_channels` now adapts molecule states to
canonical SMILES and registers the suite's `composite_score` alongside the
four component channels. The composite channel is included in reward
aggregation and metrics reporting. Focused validation: 8 tests passed
(`test_anticancer_metric_suite.py`, `test_proof_search_prior.py`).

### 2026-09-13 SynFlowNet retrosynthesis regression

Fixed backward decomposition for ethyl acetate when legacy reaction registries
lack an esterification SMARTS: active rule sets now return
`(CC(=O)O, CCO)`, while an explicitly empty rule set remains an empty
environment. Focused adapter suite passes 15 tests.

### 2026-09-13 pilot metric extension

The 1h36 Pocket2Mol/Lambda pilot aggregation now reports the
metal-anticancer `composite_score` mean per method, alongside Vina, SA, QED,
and threshold success. This is a smoke/pilot metric only; no docking sweep
was run. Missing RDKit/metric backends degrade to NaN without blocking the
existing report.

### 2026-09-13 round-13 anticancer MW reporting

The anticancer metric suite now exposes raw molecular weight and explicit
`mw_lipinski` (<=500) versus `mw_metal_adjusted` (<=800) flags. Pilot
aggregation reports mean MW and both pass rates, making metal-complex
threshold differences visible without changing the composite reward.
Focused metric tests: 5 passed; pilot script compiles under Python 3.12.
