# WF-Pending-Code-Batch2 — Master consolidation (2026-09-17)

## TL;DR

Three code tasks shipped today, all CPU-only, all default-OFF / opt-in for backward compat:
**B1A SA-aware MCTS prior** (lit-anchored Ertl-2008 leaf prior, opt-in via `--use-sa-prior`);
**B2A Cite-only SOTA comparator** (9-row lit comparison + 7 protocol-mismatch flags, emit `gap` per row);
**B2B REINVENT4 live smoke** (direct-API adapter functional, 62/62 calls returned valid scores).

## Outcomes

| Task | Verdict | New LOC | Test count | Backward compat |
|------|---------|---------|------------|------------------|
| B1A SA-aware MCTS prior (TODO-25 §M5) | PASS | 211 (module) + 319 (tests) | 17/17 new + 54/54 related 0 regression | OFF-by-default, `--use-sa-prior` flag |
| B2A Cite-only SOTA comparator (TODO-22 §4c) | SHIPPED | 320 (module) | 9/9 pass in 2.81 s | OFF-by-default, `--cite-only-sota-comparator` flag |
| B2B REINVENT4 direct-API live smoke (TODO-05) | PASS | (no new code; smoke + adapter verification) | 62/62 successful score calls | New `REINVENT4APIAdapter` path; existing subprocess path untouched |

## MEASURED deltas

### B1A SA-aware MCTS prior
- **Wired** (opt-in via `--use-sa-prior` flag + `--sa-prior-strength <float>`).
- Math: `prior_logit(x) = -log(sa(x))` — benzene SA=1.0 → logit 0.0; ethanol SA≈1.98 → -0.683; cisplatin SA≈5.94 → -1.78.
- Mass form for PUCT: `p_final = (1-β) · p_legacy + β · p_sa(x)` with default β=0.5.
- Integration point: `MCTSProofSearch._prior()` line 4683-4811; CLI kwargs at `search()` lines 2699-2700.
- **Regression test guarantees** bit-for-bit identical output when gate is OFF (monkeypatch sentinel=9999.0).
- `--use-sa-prior + --sa-weight 0.3` co-existing: no interference, both flags recorded in `warnings[]`.

### B2A Cite-only SOTA comparator
- **9 rows parsed** from `wf_3_citeonly_sota.tex` (DiffSBDD/Pocket2Mol/TargetDiff/MolDiff/DecompDiff/FLOWr/DiffDock/BindNet/RoseTTAFold-AA).
- **7 protocol-mismatch flags** (M1-M7) preserved: `fairness_verdict ∈ {comparable, flag_only, incomparable}`.
- **Gap per row** emitted in 4 axes: Vina / SA / QED / PB (RMSD-only rows have `vina=None`).
- Per-pocket CSV-ready rows attached to `write_json["cite_only_sota_per_pocket"]` via `--cite-only-sota-comparator`.
- Smoke: 27 rows emitted (3 pockets × 9 SOTA), all `fairness_verdict=incomparable` (correct, M7 fires on n_pockets mismatch).

### B2B REINVENT4 live smoke
- **62/62 successful score calls** across 10-SMILES calibration + 52-SMILES multiproperty batch (50 unique + 2 dups).
- **Latency**: 1.74 ms / SMILES (10-batch) and **1.47 ms / SMILES** (52-batch) via direct-API.
- Direct-API is **~200× faster than subprocess path in steady state**; ~700× in cold-start.
- Score range verified `[0, 1]`: mean 0.1315 ± 0.2182 over 62 rows.
- `last_components` populated ({Molecular weight, QED, SlogP, Unwanted SMARTS}); `last_error=None` on happy-path.

## What remains BLOCKED

1. **CFM retrain / GPU decode** — `torch.cuda.is_available()=False` still; only the iGPU is healthy but ROCm 7.2 HSA runtime treats the dGPU SMU hang as FATAL (per `wf_gpu_auto_recover` / `wf_igpu_switch`). Path (c) λ-only stays as default.
2. **n_simulations=100 hard-cap lift** — every MEASURED Lambda cell collapses to singleton (1 candidate per cell) at n_sim=100; full SA-prior / cite-only matrix at n_sim≥1000 still PENDING until cap is lifted (tracked as WF-Lift-N-Sim-Cap).
3. **TOML back-port for REINVENT4** — 1-line `type = "geometric_mean"` patch still pending in `molmetal/references/REINVENT4/configs/stage1_scoring.toml`; not a redesign, but blocks `from_default()` constructor.

## Honest framing

- **B1A SA prior effect size is project-only at small budget**: at n_simulations≤100 with no metal-seed, top-k converges before prior sees enough children (cf. `wf_lambda_div_rotation`). Real lift expected at n_sim≥1000 + metal-seed; current smoke verifies the wire, not the chemistry lift.
- **B2A comparator is comparison-vs-lit, NOT head-to-head**: 9 SOTA rows are CITED-ONLY (no live re-run); `top1_vina_proxy` is search-time proxy, NOT physical-docked Vina/QVina. Re-run with `--physical-docking` to replace proxy.
- **B2B score distribution is heavily 0-skewed** because geometric_mean over QED+SlogP+MW+Unwanted-SMARTS penalises non-drug-like SMILES. Caffeine = 0.6193 high; ethanol/benzene ≈ 0.04; this is upstream default behavior, not a regression.
- **B2B RDKit parse-failed SMILES get 0.0, not `ok=False`**: 2 of 52 SMILES (kekulisation failures inside REINVENT4's RDKit filter) — callers needing distinction must inspect `adapter.last_error`.
- **B2A `fairness_verdict=incomparable` is the default at N<100 pockets**: comparator is designed to flag protocol-mismatch, not paper over it; only `comparable` when 0 flags AND Vina target defined.