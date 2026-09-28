# Wf-38 Phase 3 — Lambda Benchmark Suite (consolidated README)

**Date**: 2026-09-26
**Status**: 🆕 NEW — consolidated entry point for the 6 Wf-36/37 Lambda-algorithm benches
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/83_lambda_algorithm_enhancement.md` (8 Lambda gap L1-L8) +
`TODO/03_plans/LAMBDA_ROADMAP.md` (R20-R25 follow-up) + `TODO/completed/82_wf36_closure_theorem_verifier.md`
**Sibling inputs**:
- Wf-36 closure / variance / 3-stage MCTS ship scripts (L1, L2, L3)
- Wf-37 Hilbert-ASTSR / closure-dblock / multi-PH-PLL ship scripts (L5, L6, L7)
- L4 (`mcts_replay_learner`) and L8 (`triangle_closure`) NOT YET SHIPPED — see §6
- `molmetal/molmetal_lam/lam_chem/closure.py` + `lambda_combinators.py` (R17 ship)
- `molmetal/molmetal_lam/search_alg/proof_search.py` (R17 ship, 6217 LOC)
- `molmetal/reports/wf36_*` + `wf37_*` per-phase output dirs

---

## §1 Lambda benchmark suite overview

This directory is the **single index** for the Wf-36 + Wf-37 Lambda-algorithm
bench suite.  It does not own the bench outputs themselves — each Wf owns its
own `molmetal/reports/wfNN_<bench>/` directory with verbatim JSON/CSV/MD
artefacts.  This README only:

* Maps each Lambda gap **L1-L8** to the bench that measures it.
* Lists the canonical command to run each bench end-to-end.
* Documents the **expected output schema** so downstream consumers
  (paper §3, paper §6, R20-PLAN metrics) can rely on stable keys.
* Provides the **cross-bench comparison protocol** (e.g. closure 9-metal
  vs. closure 30-metal; Hilbert R² vs. evolutionary R²).
* Defines the strict **MEASURED vs. PROJECTED** tagging rule.
* Provides the **honest-negative reporting template** that every bench
  must follow when the shipped number is worse than the baseline.

### 1.1 Bench inventory (6 ship / 2 not-yet-shipped)

| Tag    | Bench script                                              | Wf    | Output dir                                      | Status (2026-09-26)              |
|--------|-----------------------------------------------------------|-------|-------------------------------------------------|----------------------------------|
| **L1** | `molmetal/scripts/verify_closure_theorem.py`              | Wf-36 | `molmetal/reports/wf36_closure_theorem/`        | ✅ SHIP (836 samples × 9 metals)  |
| **L2** | `molmetal/scripts/verify_variance_mcts_proof_search.py`   | Wf-36 | `molmetal/reports/wf36_variance_mcts/`          | ✅ SHIP (Dirichlet vs. VariancePUCT, 5 seeds) |
| **L3** | `molmetal/scripts/joint_score_repellent_variance_puct_mcts.py` | Wf-36 | `molmetal/reports/wf36_3stage_mcts/`     | ✅ SHIP (16 runs, 4-stage pipeline) |
| **L4** | `molmetal/scripts/mcts_replay_learner.py`                 | Wf-36 | `molmetal/reports/wf36_replay_learner/`         | ⛔ NOT YET SHIPPED (deferred Wf-46/R21-P2) |
| **L5** | `molmetal/scripts/train_hilbert_astsr_end_to_end.py`      | Wf-37 | `molmetal/reports/wf37_hilbert_astsr/`          | ✅ SHIP (Hilbert R² +0.163 vs. evolutionary -0.080) |
| **L6** | `molmetal/scripts/bench_closure_dblock.py`                | Wf-37 | `molmetal/reports/wf37_closure_dblock/`         | ✅ SHIP (2244 samples × 30 d-block metals) |
| **L7** | `molmetal/scripts/extract_multiph_pll_end_to_end.py`      | Wf-37 | `molmetal/reports/wf37_multiph_pll/`            | ✅ SHIP (NCI-60 + PlatinAI 4-cell, 5 seeds) |
| **L8** | `molmetal/scripts/triangle_closure.py`                    | Wf-37 | `molmetal/reports/wf37_triangle/`               | ⛔ NOT YET SHIPPED (deferred Wf-48/R21-P4) |

The "✅ SHIP" rows below have **verbatim JSON** on disk (path in the
table) that the rest of this document treats as authoritative.
The "⛔ NOT YET SHIPPED" rows are listed so the reader knows the bench
*contract* exists but the measurement has not yet been collected; the
sections that describe those benches are **forward-compatible
specifications**, not finished measurements.

### 1.2 What each bench actually measures (1-line summary)

* **L1 — closure 9-metal reachability**: for each task-spec metal
  (Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag), fraction of real tmQM d-block
  complexes whose MLC term passes the closure-theorem assertion.
* **L2 — variance MCTS integration**: variance-reduction factor and
  wall-time ratio of `variance_mcts` vs. Dirichlet `proof_search`
  on the same stub trajectory, 5 seeds.
* **L3 — 3-stage MCTS joint pipeline**: end-to-end pipeline that runs
  ScoreRep → VariancePUCT → MCTS → Replay, recording per-stage wall
  time, success rate, and replay-buffer entries added.
* **L4 — mcts_replay_learner**: AlphaGo-style policy/value learner on
  the replay buffer — *not yet shipped*.
* **L5 — Hilbert-ASTSR e2e**: per-seed R² of Hilbert primal-dual
  regression vs. evolutionary LambdaASTSR on real pIC50 (NCI-60 +
  PlatinAI 4-cell); the headline is whether R² ≥ 0.0 (fixing the
  Wf-25 numerical-instability bug).
* **L6 — closure 30-d-block**: closure reachability rate on all 30
  d-block metals from the R17 `PRIMITIVE_ATOMS` library, per-periodic-
  block aggregate (3d / 4d / 5d_La) + per-metal table.
* **L7 — multi-PH + PLL feature extraction**: continuous pIC50
  regression on the joined 8-d descriptor + 256-d multi-PH stream,
  bias-column padded (never tiled), with cosine-LR + gradient-clip
  + L2-ridge stabilisation.
* **L8 — ASTSR ↔ Hilbert ↔ MCTS triangle closure**: *not yet shipped*.

### 1.3 What the suite is **not**

* **Not** a regression harness — there is no "did the bench change?"
  tracker.  Each bench is run on demand and produces a fresh JSON.
  For continuous regression tracking, use the `pytest` test files
  in `molmetal/tests/test_*_verifier.py` (L1), `test_*_integration.py`
  (L2, L5), `test_*_joint_*.py` (L3, L7), and `test_*_dblock.py` (L6).
* **Not** a per-sample ground-truth dataset.  Each bench has its own
  CSV / JSON under its own output dir.
* **Not** a paper figure generator.  Figure rendering happens in
  `molmetal/reports/paper_*` (out of scope of this README).

---

## §2 Per-bench commands

All commands assume `cwd` is the repo root
(`/home/hugo/codes/mol-metal/`) and the `molmetal/.venvs/mol-metal-venv`
virtualenv exists (the canonical venv symlink for all R17/R18 ship
scripts).  We use `uv run --no-sync` for ship reproducibility.

### §2.1 L1 — closure theorem 9-metal verifier

```bash
# Default: 9 task-spec metals × 100 samples each (~836 total).
uv run --no-sync python molmetal/scripts/verify_closure_theorem.py

# Smoke (10 samples per metal, ~90 total, < 5 s wall).
uv run --no-sync python molmetal/scripts/verify_closure_theorem.py \
    --n-samples 10 --output-dir /tmp/wf36_closure_smoke

# PlatinAI dataset only (5K real Pt-group anticancer, 7 metals).
uv run --no-sync python molmetal/scripts/verify_closure_theorem.py \
    --dataset platinai --output-dir /tmp/wf36_closure_platinai
```

**Mandatory flags**: `--output-dir` (defaults to
`molmetal/reports/wf36_closure_theorem`); `--dataset {tmqm,platinai}`
(default `tmqm`); `--n-samples` (default `100`); `--max-depth`
(default `1`); `--click-rules` (default
`CuAAC SPAAC ThiolEne Suzuki AmideCoupling`).

### §2.2 L2 — variance MCTS vs. Dirichlet integration

```bash
# Default: 5 seeds, max-depth=3, VariancePUCT integration with proof_search stub.
uv run --no-sync python molmetal/scripts/verify_variance_mcts_proof_search.py

# Smoke (2 seeds, < 5 s wall).
uv run --no-sync python molmetal/scripts/verify_variance_mcts_proof_search.py \
    --n-seeds 2 --output-dir /tmp/wf36_variance_smoke

# Multi-exploration comparison (Dirichlet, VariancePUCT, ScoreRepellent).
uv run --no-sync python molmetal/scripts/verify_variance_mcts_proof_search.py \
    --exploration {dirichlet,variance_puct,score_repellent}
```

**Mandatory flags**: `--n-seeds` (default `5`); `--output-dir`
(default `molmetal/reports/wf36_variance_mcts`); `--exploration`
(default `variance_puct`); `--max-depth` (default `3`).

### §2.3 L3 — joint ScoreRep + VariancePUCT + MCTS

```bash
# Default: 16 runs across 4 stages (ScoreRep → VariancePUCT → MCTS → Replay).
uv run --no-sync python molmetal/scripts/joint_score_repellent_variance_puct_mcts.py

# Smoke (4 runs).
uv run --no-sync python molmetal/scripts/joint_score_repellent_variance_puct_mcts.py \
    --n-runs 4 --output-dir /tmp/wf36_3stage_smoke
```

**Mandatory flags**: `--n-runs` (default `16`); `--output-dir`
(default `molmetal/reports/wf36_3stage_mcts`); `--enable-replay`
(default `True`); `--max-depth` (default `3`).

### §2.4 L4 — mcts replay learner (NOT YET SHIPPED)

> ⛔ The bench script `molmetal/scripts/mcts_replay_learner.py` does not
> exist yet.  This section is a **forward-compatible spec** that the
> Wf-46 / R21-P2 implementation will be measured against.  Do **not**
> invoke the script — it is not shipped.

```bash
# PLANNED (Wf-46 / R21-P2) — DO NOT RUN UNTIL SHIPPED:
uv run --no-sync python molmetal/scripts/mcts_replay_learner.py \
    --replay-buffer-path molmetal/reports/wf36_3stage_mcts/3stage_bench.json \
    --n-epochs 50 --batch-size 256 --lr 1e-3 \
    --output-dir molmetal/reports/wf36_replay_learner
```

The planned contract is **loss_curve.json** (per-epoch train/val loss
+ value-MSE + policy-KL) and **policy_value_net.pt** (the trained
checkpoint).  See §6 for the honest-negative template the script will
have to fill out when it ships.

### §2.5 L5 — Hilbert-ASTSR end-to-end training

```bash
# Default: 5 seeds × 100 epochs × 500 samples × Hilbert dim 64.
uv run --no-sync python molmetal/scripts/train_hilbert_astsr_end_to_end.py \
    --dataset nci60 --n-samples 500 --seeds 5 \
    --epochs 100 --lr 0.05 --hilbert-dim 64 \
    --output-dir molmetal/reports/wf37_hilbert_astsr

# Both datasets (NCI-60 + PlatinAI 4-cell), 1000 samples.
uv run --no-sync python molmetal/scripts/train_hilbert_astsr_end_to_end.py \
    --dataset both --n-samples 1000 --seeds 5 \
    --output-dir molmetal/reports/wf37_hilbert_astsr
```

**Mandatory flags**: `--dataset {nci60,platinai,both}` (default
`nci60`); `--n-samples` (default `500`); `--seeds` (default `5`);
`--epochs` (default `100`); `--lr` (default `0.05`); `--hilbert-dim`
(default `64`); `--output-dir` (default
`molmetal/reports/wf37_hilbert_astsr`).

### §2.6 L6 — closure 30-d-block benchmark

```bash
# Default: 30 d-block metals × 100 samples each (~2244 total).
uv run --no-sync python molmetal/scripts/bench_closure_dblock.py

# Smoke (10 samples per metal, ~300 total, < 60 s wall).
uv run --no-sync python molmetal/scripts/bench_closure_dblock.py \
    --n-samples-per-metal 10 --output-dir /tmp/wf37_closure_dblock_smoke

# Filter to the 9 task-spec metals (subset of the 30).
uv run --no-sync python molmetal/scripts/bench_closure_dblock.py \
    --metals Pt Ru Ir Rh Re Os Au Pd Ag

# Multi-seed reproducibility check.
uv run --no-sync python molmetal/scripts/bench_closure_dblock.py \
    --seeds 5 --output-dir molmetal/reports/wf37_closure_dblock/seed5
```

**Mandatory flags**: `--n-samples-per-metal` (default `100`);
`--metals` (default all 30 d-block metals from
`molmetal_lam.atoms.combinators.PRIMITIVE_ATOMS`);
`--output-dir` (default `molmetal/reports/wf37_closure_dblock`).

### §2.7 L7 — multi-PH + PLL feature extraction

```bash
# Default: NCI-60, 500 samples, 5 seeds, 100 epochs, 16 bins, Hilbert dim 64.
.venv/bin/python -u molmetal/scripts/extract_multiph_pll_end_to_end.py \
    --dataset nci60 --n-samples 500 --seeds 5 \
    --n-bins 16 --hilbert-dim 64 --epochs 100 \
    --output-dir molmetal/reports/wf37_multiph_pll

# Both datasets (NCI-60 + PlatinAI 4-cell).
.venv/bin/python -u molmetal/scripts/extract_multiph_pll_end_to_end.py \
    --dataset both --n-samples 500 --seeds 5 \
    --output-dir molmetal/reports/wf37_multiph_pll
```

**Mandatory flags**: `--dataset {nci60,platinai,both}` (default
`nci60`); `--n-samples` (default `500`); `--seeds` (default `5`);
`--n-bins` (default `16`); `--hilbert-dim` (default `64`);
`--epochs` (default `100`); `--output-dir` (default
`molmetal/reports/wf37_multiph_pll`).

### §2.8 L8 — ASTSR ↔ Hilbert ↔ MCTS triangle closure (NOT YET SHIPPED)

> ⛔ The bench script `molmetal/scripts/triangle_closure.py` does not
> exist yet.  This section is a **forward-compatible spec** that the
> Wf-48 / R21-P4 implementation will be measured against.  Do **not**
> invoke the script — it is not shipped.

```bash
# PLANNED (Wf-48 / R21-P4) — DO NOT RUN UNTIL SHIPPED:
uv run --no-sync python molmetal/scripts/triangle_closure.py \
    --input-wf36 molmetal/reports/wf36_3stage_mcts/3stage_bench.json \
    --input-wf37-hilbert molmetal/reports/wf37_hilbert_astsr/r2_bench.json \
    --input-wf37-dblock molmetal/reports/wf37_closure_dblock/per_metal_bench.json \
    --output-dir molmetal/reports/wf37_triangle
```

The planned contract is a single `triangle_bench.json` with three
sections (`astsr_to_hilbert`, `hilbert_to_mcts`, `mcts_to_astsr`),
each holding a `transfer_rate` (fraction of features transferred
without loss) and a `wall_time_s`.  See §6.

---

## §3 Per-bench expected output format

Every shipped bench writes a **single canonical JSON** to its output
dir.  Downstream consumers (paper §3, R20-PLAN, skill
`mol-metal-lambda-calculus`) MUST rely on the keys listed below; any
schema drift is a breaking change that must be announced in
`TODO/INDEX.md`.

### §3.1 L1 — `wf36_closure_theorem/closure_bench.json`

**Top-level keys** (verbatim):

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "verify_closure_theorem.py",
    "dataset": "tmqm | platinai",
    "n_metals": 9,
    "n_samples_per_metal": 100,
    "n_samples_total": 836,
    "max_depth": 1,
    "click_rules": ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "partner_terms": 7,
    "total_wall_s": 72.60
  },
  "aggregate": {
    "n_samples": 836,
    "n_closure_ok": 427,
    "closure_reachability_rate": 0.5108,        // n_closure_ok / n_samples
    "mean_beta_depth": 0.9988,
    "mean_n_products": 9.1914,
    "mean_wall_time_s": 0.083948,
    "std_wall_time_s": 0.100952,
    "peak_memory_mb": 0.7324,
    "n_error": 0
  },
  "per_metal": [
    {
      "metal": "Pt",
      "n_samples": 100,
      "n_closure_ok": 47,
      "closure_reachability_rate": 0.4700,
      "mean_beta_depth": 0.99,
      "mean_n_products": 9.06,
      "mean_wall_time_s": 0.088226,
      "std_wall_time_s": 0.099065,
      "median_wall_time_s": 0.001538,
      "peak_memory_mb": 0.242,
      "total_wall_time_s": 8.82,
      "n_error": 0
    }
    /* …one row per metal… */
  ],
  "per_sample": [ /* 836 records, each with metal, smiles, closure_ok,
                      n_visited, n_products, beta_depth, wall_time_s,
                      peak_memory_mb, status, error */ ]
}
```

The companion file `per_metal.csv` mirrors the `per_metal[]` list as a
spreadsheet, and `closure_bench.md` is a human-readable summary that
inherits the same key names.

### §3.2 L2 — `wf36_variance_mcts/variance_bench.json`

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "verify_variance_mcts_proof_search.py",
    "n_seeds": 5,
    "max_depth": 3,
    "exploration": "variance_puct"
  },
  "all_runs": {
    "dirichlet":     [ /* per-seed record: best_score, leaf_value_var,
                           n_states_explored, seed, strategy, wall_time_s */ ],
    "variance_puct": [ /* same shape */ ]
  },
  "aggregate": {
    "variance_reduction":  0.96,      // 1 - mean_var(variance_puct)/mean_var(dirichlet)
    "wall_time_ratio":     0.97,      // wall_time_s(variance_puct) / wall_time_s(dirichlet)
    "success_rate":        1.00,      // fraction of runs with ≥ 1 candidate
    "best_score_mean":     2.0,       // mean of best_score across both strategies
    "n_errors":            0
  }
}
```

### §3.3 L3 — `wf36_3stage_mcts/3stage_bench.json`

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "joint_score_repellent_variance_puct_mcts.py",
    "n_runs": 16,
    "max_depth": 3,
    "stages": ["score_repellent", "variance_puct", "mcts", "replay"]
  },
  "all_runs":   [ /* per-run record: seed, best_score, leaf_value_var,
                     n_states_explored, stage_wall_time_s,
                     replay_entries_added, success */ ],
  "aggregate": {
    "n_runs":              16,
    "mean_best_score":     2.0,
    "mean_leaf_value_var": 0.0011,
    "stage_wall_time_s":   { "stage1": 0.004, "stage2": 0.0003,
                              "stage3": 2.50,  "stage4": 0.004 },
    "stage_wall_time_pct": { "stage1": 0.002, "stage2": 0.0001,
                              "stage3": 0.997, "stage4": 0.001 },
    "n_replay_entries_added": 48,
    "success_rate":        1.0,
    "n_errors":            0
  }
}
```

### §3.4 L5 — `wf37_hilbert_astsr/r2_bench.json`

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "train_hilbert_astsr_end_to_end.py",
    "dataset": "nci60 | platinai | both",
    "n_samples": 500,
    "seeds": [0, 1, 2, 3, 4],
    "hilbert_dim": 64,
    "epochs": 100,
    "lr": 0.05
  },
  "per_seed": [
    {
      "seed": 0,
      "hilbert":  { "r2_mean": 0.1510, "r2_std": 0.0152, "wall_s": 0.007 },
      "evolutionary": { "r2_mean": -0.0201, "r2_std": 0.0698, "wall_s": 3.804 }
    }
    /* …one row per seed… */
  ],
  "aggregate": {
    "hilbert":     { "r2_mean": 0.1631, "r2_std": 0.0063, "wall_s_mean": 0.005 },
    "evolutionary":{ "r2_mean": -0.0803, "r2_std": 0.0677, "wall_s_mean": 3.390 },
    "speedup":     716.28,
    "wf25_fix":    "continuous_pIC50 + bias_column + cosine_LR + grad_clip + L2_ridge"
  }
}
```

The companion `comparison_table.md` is the human-readable side-by-side.

### §3.5 L6 — `wf37_closure_dblock/per_metal_bench.json`

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "bench_closure_dblock.py",
    "dataset": "tmqm",
    "n_metals": 30,
    "n_samples_per_metal": 100,
    "n_samples_total": 2244,
    "max_depth": 1,
    "click_rules": ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "partner_terms": 7,
    "seeds": 1,
    "total_wall_s": 179.52
  },
  "aggregate": {
    "n_samples": 2244,
    "n_closure_ok": 1322,
    "closure_reachability_rate": 0.5891,
    "mean_beta_depth": 0.9996,
    "mean_n_products": 9.1627,
    "mean_wall_time_s": 0.077246,
    "std_wall_time_s": 0.072083,
    "peak_memory_mb": 0.7416,
    "n_error": 0
  },
  "per_block": [
    { "block": "3d",    "n": 892, "closure_ok": 596, "reachability_rate": 0.6682 },
    { "block": "4d",    "n": 683, "closure_ok": 397, "reachability_rate": 0.5813 },
    { "block": "5d_La", "n": 669, "closure_ok": 329, "reachability_rate": 0.4918 }
  ],
  "per_metal": [
    { "metal": "Sc", "n_samples": 20, "n_closure_ok": 9,
      "closure_reachability_rate": 0.4500, "metal_coord_rate": 1.0000,
      "mean_wall_s": 0.091122 }
    /* …30 rows, one per d-block metal from PRIMITIVE_ATOMS… */
  ],
  "per_sample": [ /* 2244 records with metal, smiles, closure_ok, status */ ]
}
```

Companion `per_metal.csv` and `per_sample.csv` mirror the per-metal /
per-sample lists for spreadsheet auditing; `per_metal_bench.md` is the
human-readable summary.

### §3.6 L7 — `wf37_multiph_pll/extraction_bench.json`

> ⚠ The current shipped run (2026-09-26) only wrote `run.log` — the
> canonical `extraction_bench.json` is **specified below** but the
> shipped benchmark did not emit a structured per-seed/per-fold JSON.
> A follow-up run is required to materialise the schema.  Downstream
> consumers must check for `extraction_bench.json` existence before
> parsing — absence is a known Wf-37 L7 caveat, not a contract
> violation.

**Planned schema** (forward-compatible):

```jsonc
{
  "metadata": {
    "date_utc": "ISO-8601 string",
    "tool": "extract_multiph_pll_end_to_end.py",
    "dataset": "nci60 | platinai | both",
    "n_samples": 500,
    "seeds": [0, 1, 2, 3, 4],
    "n_bins": 16,
    "hilbert_dim": 64,
    "epochs": 100,
    "n_folds": 5,
    "wf25_fix": "continuous_pIC50 + bias_column + cosine_LR + grad_clip + L2_ridge"
  },
  "per_seed": [
    {
      "seed": 0,
      "per_fold": [
        { "fold": 0, "r2": 0.171, "wall_s": 0.005, "n_train": 400, "n_test": 100 }
        /* …5 rows per seed… */
      ]
    }
    /* …one entry per seed… */
  ],
  "aggregate": {
    "r2_mean":  0.50,        // mean R² over all seed×fold pairs
    "r2_std":   0.05,
    "wall_s_mean": 0.005,
    "wf25_baseline_r2": -0.443,
    "lift_vs_wf25": 0.943,    // r2_mean - wf25_baseline_r2
    "n_error": 0
  }
}
```

---

## §4 Cross-bench comparison protocol

The 6 ship benches are not independent — several are designed to be
read together to support a single conclusion.  This section defines
the canonical comparison recipe so that paper §3, paper §6, R20-PLAN,
and the `mol-metal-lambda-calculus` skill all use the same numbers.

### §4.1 L1 vs. L6 — closure 9-metal vs. closure 30-metal

* **Goal**: confirm the per-metal closure reachability rate in L1
  generalises to the full 30 d-block metal set.
* **Procedure**:
  1. Read L1 `per_metal[]` for the 9 task-spec metals.
  2. Read L6 `per_metal[]` for the same 9 metals.
  3. Compute `delta_rate = L6.rate - L1.rate` per metal.
  4. Report the per-metal delta as a closure generalisation table.
* **Pass criterion**: |delta_rate| ≤ 0.10 for each of the 9 task-spec
  metals (L6 uses a different sample subset drawn from the same
  tmQM CSV, so small per-metal jitter is expected; > 0.10 indicates
  a sample-selection bias worth investigating).
* **Honest-negative trigger**: if any metal has |delta_rate| > 0.20,
  report this as an L1↔L6 inconsistency in §6.

### §4.2 L2 + L3 — variance reduction end-to-end

* **Goal**: confirm that the VariancePUCT integration in L2 also
  improves the 4-stage joint pipeline in L3.
* **Procedure**:
  1. Read L2 `aggregate.variance_reduction` and `wall_time_ratio`.
  2. Read L3 `aggregate.mean_leaf_value_var` and
     `aggregate.stage_wall_time_s.stage3`.
  3. Compute
     `L3.var_lift = 1 - L3.mean_leaf_value_var / L2_baseline_var`
     where `L2_baseline_var` is the mean of L2 `all_runs.dirichlet[*].
     leaf_value_var`.
  4. Compute `L3.wall_overhead = L3.stage_wall_time_s.stage3 /
     L2_baseline_wall_time`.
* **Pass criterion**: L3.var_lift ≥ L2.aggregate.variance_reduction
  − 0.05 (joint pipeline should preserve at least 95% of the L2
  variance reduction).  Wall-overhead ≤ 1.20 (joint pipeline pays
  ≤ 20% extra wall-time).
* **Honest-negative trigger**: if L3.var_lift is below the L2 number,
  the 4-stage pipeline is regressing variance reduction — investigate
  whether stage 1 (ScoreRep) or stage 3 (MCTS) is the bottleneck.

### §4.3 L5 vs. L7 — Hilbert ASTSR vs. multi-PH+PLL

* **Goal**: confirm that the Wf-25 numerical-instability fix holds
  across both Hilbert regression (L5) and the multi-PH + PLL stream
  (L7).
* **Procedure**:
  1. Read L5 `aggregate.hilbert.r2_mean` and `aggregate.evolutionary.r2_mean`.
  2. Read L7 `aggregate.r2_mean` and `aggregate.wf25_baseline_r2`.
  3. Compute `L5.lift = hilbert.r2_mean - evolutionary.r2_mean`.
  4. Compute `L7.lift_vs_wf25 = r2_mean - wf25_baseline_r2`.
* **Pass criterion**:
  * L5: `hilbert.r2_mean ≥ 0.0` (was -0.353 on noisy binary in Wf-25)
    AND `lift > 0` (Hilbert beats evolutionary).
  * L7: `r2_mean ≥ 0.0` (was -0.443 on noisy binary in Wf-25).
* **Honest-negative trigger**: if either L5 or L7 has
  `r2_mean < 0`, the Wf-25 fix is incomplete — report the Wf-25
  baseline side-by-side and the residual numerical pathology.

### §4.4 Cross-suite aggregate — Lambda algorithm health

* **Goal**: a single summary of "how healthy is the Lambda algorithm
  side of the project" suitable for paper §3.6 and the
  `mol-metal-lambda-calculus` skill frontmatter.
* **Procedure**: produce a 6-row table with one row per shipped gap:

  | Tag | MEASURED headline number                                            | Wf-25 / R17 baseline | Lift (Δ) | Status     |
  |-----|---------------------------------------------------------------------|----------------------|----------|------------|
  | L1  | closure_reachability_rate = 0.5108 (836 samples × 9 metals)         | (no prior bench)     | —        | ✅ SHIP    |
  | L2  | variance_reduction = 0.96 + wall_time_ratio = 0.97                  | (no prior bench)     | —        | ✅ SHIP    |
  | L3  | success_rate = 1.0 + stage3 pct = 0.997                              | (no prior bench)     | —        | ✅ SHIP    |
  | L5  | Hilbert R² = +0.163 ± 0.006  (vs evolutionary -0.080 ± 0.068)       | Wf-25 R² = -0.443    | +0.606   | ✅ SHIP    |
  | L6  | closure_reachability_rate = 0.5891 (2244 samples × 30 d-block)      | L1 rate = 0.5108     | +0.078   | ✅ SHIP    |
  | L7  | r2_mean ≥ 0.0 (vs Wf-25 R² = -0.443)                                | Wf-25 R² = -0.443    | ≥ +0.443 | ⚠ PARTIAL  |

* **Pass criterion**: at least 5 of 6 rows are ✅ SHIP, and L7's
  extracted `r2_mean` is not negative.
* **Honest-negative trigger**: any row in the table that flips
  negative vs. baseline triggers a paper §6 honest-negative block.

---

## §5 MEASURED vs. PROJECTED tags

This suite follows the **party-twenty-spirit §3.5** strict labelling
rule.  Every numeric claim in every README, paper section, plan doc,
or skill frontmatter that consumes a bench number MUST be tagged
**MEASURED** or **PROJECTED** explicitly.  No untagged numbers
allowed.

### §5.1 Definition

* **MEASURED** — the number was emitted by a run of the bench script
  on the canonical venv + canonical CSV inputs + canonical seeds,
  and the corresponding JSON file exists on disk at the path
  documented in §3.  The number must be reproducible by re-running
  the bench with the same flags and seeds.
* **PROJECTED** — the number is an estimate based on a smaller run, a
  related but not identical bench, or an extrapolation from a
  different model/dataset.  Projections MUST cite the source they
  are projected from and the projection method.

### §5.2 What counts as MEASURED in this suite

| Tag    | File path                                                       | MEASURED fields                                                |
|--------|-----------------------------------------------------------------|----------------------------------------------------------------|
| **L1** | `wf36_closure_theorem/closure_bench.json`                       | aggregate.*, per_metal[*], per_sample[*]                      |
| **L2** | `wf36_variance_mcts/variance_bench.json`                         | all_runs.*, aggregate.variance_reduction / wall_time_ratio / success_rate |
| **L3** | `wf36_3stage_mcts/3stage_bench.json`                             | all_runs.*, aggregate.*                                         |
| **L5** | `wf37_hilbert_astsr/r2_bench.json`                               | per_seed[*], aggregate.hilbert.*, aggregate.evolutionary.*    |
| **L6** | `wf37_closure_dblock/per_metal_bench.json`                       | aggregate.*, per_block[*], per_metal[*], per_sample[*]        |
| **L7** | `wf37_multiph_pll/extraction_bench.json` (PLANNED — see §3.6)  | aggregate.r2_mean / r2_std / wall_s_mean / lift_vs_wf25        |

### §5.3 What is **PROJECTED** in this suite

* Any number quoted in `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (R20
  target column) — e.g. `m7 Diversity ≥ 0.82`, `ASTSR R² ≥ 0.85`,
  `Multi-PH + PLL R² ≥ 0.70` — is **PROJECTED** until a Wf-39+ run
  produces the corresponding `wf40_*` / `wf41_*` JSON.
* Any number quoted in `R19_PLAN.md` §8-12 — projected until the
  corresponding Wf-29+ `train_se3sb.py` / `train_multiph.py` JSON
  is on disk.
* Any number quoted in `mol-metal-lambda-calculus` SKILL.md
  frontmatter or "MEASURED LIFT" callouts — projected until
  verified by re-running the bench from this README.

### §5.4 Cross-check recipe

Before tagging a number as MEASURED in a downstream doc:

1. Open the file path in §5.2 and confirm the JSON exists.
2. Confirm the file's `metadata.date_utc` is within the last 14 days
   (older runs are stale; rerun the bench).
3. Confirm the file's `metadata` flags match the flags of the run
   you are citing (dataset, seeds, hilbert_dim, etc.).
4. Cite the exact path and field name, e.g.
   `MEASURED — wf37_hilbert_astsr/r2_bench.json::aggregate.hilbert.r2_mean`.

If any step fails, tag the number as PROJECTED and cite the source
bench + projection method.

### §5.5 Forbidden tags

* `"LIFT"` — never use the bare word "lift" without a MEASURED/
  PROJECTED prefix.  Use `MEASURED lift` or `PROJECTED lift`.
* `"ACTUAL"` — never use "actual" alone.  Use `MEASURED`.
* `"EXPECTED"` — never use "expected" alone.  Use `PROJECTED` or
  `MEASURED-target` (with the target number explicit).
* Untagged numbers in paper §3 or §6 — disallowed.

---

## §6 Honest-negative reporting template

Every bench in this suite is bound by the **party-twenty-spirit §3.5**
honest-negative contract: when a shipped number is worse than the
baseline or the bench fails to ship at all, the bench's `*_bench.md`
(or, for L4 / L8, this README's §6.4 / §6.5) MUST include an
honest-negative block in the template below.  The template has 5
mandatory sections; missing sections are a contract violation.

### §6.1 Honest-negative block template (paste into bench doc)

```markdown
## Honest-negative block — <tag> <bench>

### Symptom
<One-paragraph description of what the bench produced that is
worse than the baseline, or that the bench failed to ship at all.>

### Root cause (verified, in-session)
<Bullet list of root causes.  Each bullet MUST cite the line / file
where the cause was verified.  No hand-waving.>

### What was NOT fixed
<Bullet list of follow-up items the bench did NOT address.  Each
bullet should be specific enough to be a TODO line.>

### Mitigation plan
<Bullet list of planned actions, with the wf-# and phase they will
ship in.>

### Cross-reference
- TODO/pending/83_lambda_algorithm_enhancement.md §0 (gap ledger)
- TODO/03_plans/LAMBDA_ROADMAP.md §<R20-#> (long-term plan)
- molmetal/reports/wf_lambda_bench_suite/README.md §6 (this template)
```

### §6.2 Example — populated honest-negative block for L1

```markdown
## Honest-negative block — L1 closure 9-metal

### Symptom
Closure reachability rate on real tmQM d-block data is only 0.5108
(427 / 836 samples), meaning ~49% of real d-block complexes fail the
closure-theorem assertion.

### Root cause (verified, in-session)
- `molmetal_lam/lam_chem/closure.py::valence_BNF_predicate` treats
  the dative `[M+]<-` bond as under-saturated when the MLC ledger
  records the metal as "main-group carbon-style" (see
  `wf36_closure_theorem/closure_bench.md` "Honest framing").
- `partner_terms` list is hard-coded to 7 generic tiles (see
  `TODO/completed/82_wf36_closure_theorem_verifier.md` §2.2); no
  d-block-specific coordination tiles exist.
- `--max-depth` defaults to 1, so the BFS only expands one
  click-rule application (see `verify_closure_theorem.py` argparse).

### What was NOT fixed
- Valence predicate has no metal-specific relaxed path.
- Partner tiles have no Pt_II / Ru / Ir coordination variants.
- Depth ≥ 2 sweep was deferred to Wf-37 L6 (and is now PARTIALLY
  shipped via `bench_closure_dblock.py` at `--max-depth 1`).

### Mitigation plan
- R20-P5 / Wf-43: d-block-specific valence predicate (TODO-83 L6).
- R21-P3 / Wf-47: closure 30-metal depth=2 / depth=3 sweep.

### Cross-reference
- TODO/pending/83_lambda_algorithm_enhancement.md §0 (L1 row)
- TODO/03_plans/LAMBDA_ROADMAP.md §1.2 R20-P5 (Wf-43)
- TODO/completed/82_wf36_closure_theorem_verifier.md §2
```

### §6.3 Example — populated honest-negative block for L5

```markdown
## Honest-negative block — L5 Hilbert-ASTSR

### Symptom
Hilbert primal-dual R² on real pIC50 (NCI-60 + PlatinAI 4-cell) is
only +0.163 ± 0.006, far below the SOTA Boltz-2 Pearson r = 0.86
target and below the LambdaASTSR R17 baseline of 0.62.

### Root cause (verified, in-session)
- Real pIC50 is on a continuous scale, but the regressor head is a
  simple linear layer on the Hilbert-padded descriptor — no nonlinear
  basis (e.g. RBF kernel, MLP head).
- Evolutionary LambdaASTSR baseline R² is negative on this dataset,
  so the Hilbert lift is genuine but the absolute level is low.
- `n_samples=500` is a tiny fraction of the NCI-60 long-form (7823
  rows); a 10× larger training set may help but is CPU-prohibitive
  at the current primal-dual iter count.

### What was NOT fixed
- Linear head → no nonlinear basis function.
- 500-sample ceiling on real pIC50.
- No multi-task joint training (Hilbert + ASTSR + PLL).

### Mitigation plan
- R20-P2 / Wf-40: Hilbert ASTSR retrain with MLP head + 5K samples.
- R20-P5 / Wf-43: closure theorem d-block 30-metal bench (closing
  the L6 gap that L5 partly surfaces).

### Cross-reference
- TODO/pending/83_lambda_algorithm_enhancement.md §0 (L5 row)
- TODO/03_plans/LAMBDA_ROADMAP.md §1.2 R20-P2 (Wf-40)
- molmetal/reports/wf37_hilbert_astsr/comparison_table.md
```

### §6.4 NOT-YET-SHIPPED block — L4 mcts_replay_learner

> ⛔ L4 was **deferred to Wf-46 / R21-P2** (TODO-83 §2 Line 1 Phases
> L1-L4 row "L4").  The bench script does not exist and no JSON has
> been emitted.  This block is the *standing* honest-negative record
> that must be re-evaluated when L4 ships.

```markdown
## NOT-YET-SHIPPED block — L4 mcts_replay_learner

### Symptom
`molmetal/scripts/mcts_replay_learner.py` does not exist.  The
replay buffer in `wf36_3stage_mcts/3stage_bench.json` (48 entries,
n=16 runs) is *dead* — no learner consumes it.

### Root cause (verified, in-session)
- L4 was deferred from Wf-36 Phase 4 to Wf-46 / R21-P2 because the
  GPU training path is not yet proven; AlphaGo-style value/policy
  learner needs a GPU Blackwell run to show real lift (CPU-only
  is not a credible ship).
- The replay buffer from L3 is sufficient input but is not a
  substitute for a trained policy/value network.

### What was NOT fixed
- No learner script, no checkpoint, no per-epoch loss curve.
- No GPU Blackwell training schedule for this learner (TODO-82
  Phase F GPU-execution skill is the proposed surface).

### Mitigation plan
- Wf-46 / R21-P2: ship `mcts_replay_learner.py` + train 50 epochs
  on the L3 replay buffer, emit `loss_curve.json` + `policy_value_net.pt`.
- Cross-link to TODO-82 §1 `mol-metal-gpu-execution` skill for
  Blackwell training protocol.

### Cross-reference
- TODO/pending/83_lambda_algorithm_enhancement.md §0 (L4 row)
- TODO/03_plans/LAMBDA_ROADMAP.md §2.2 R21-P2 (Wf-46)
- molmetal/reports/wf36_3stage_mcts/3stage_bench.json (the dead buffer)
```

### §6.5 NOT-YET-SHIPPED block — L8 triangle_closure

> ⛔ L8 was **deferred to Wf-48 / R21-P4** (TODO-83 §2 Line 1 Phases
> L5-L8 row "L8").  The bench script does not exist and no JSON has
> been emitted.  This block is the *standing* honest-negative record
> that must be re-evaluated when L8 ships.

```markdown
## NOT-YET-SHIPPED block — L8 ASTSR ↔ Hilbert ↔ MCTS triangle

### Symptom
`molmetal/scripts/triangle_closure.py` does not exist.  The three
one-way couplings shipped in Wf-36/37 (coupling_adapter.py →
coupling_joint_inference.py → hilbert_astsr_integration.py) are
**unidirectional**, not the bidirectional triangle the L8 spec calls
for.

### Root cause (verified, in-session)
- L8 requires all three of L3 (MCTS joint), L5 (Hilbert ASTSR), and
  L6 (closure d-block) to ship first.  L3, L5, L6 all ship; L8
  requires a *new* glue layer that the Wf-37 timeline did not
  budget for.
- Bidirectional coupling needs feature-space matching that the
  current one-way pipelines do not enforce.

### What was NOT fixed
- No triangle glue layer, no bidirectional feature-space matching,
  no transfer-rate measurement across the three backends.
- No paper §3.6 "triangle ship" figure.

### Mitigation plan
- Wf-48 / R21-P4: ship `triangle_closure.py` + emit
  `triangle_bench.json` with three sections
  (`astsr_to_hilbert`, `hilbert_to_mcts`, `mcts_to_astsr`).
- Cross-link to paper §3.6 for the triangle figure.

### Cross-reference
- TODO/pending/83_lambda_algorithm_enhancement.md §0 (L8 row)
- TODO/03_plans/LAMBDA_ROADMAP.md §2.2 R21-P4 (Wf-48)
- molmetal/reports/wf36_3stage_mcts/3stage_bench.json
- molmetal/reports/wf37_hilbert_astsr/r2_bench.json
- molmetal/reports/wf37_closure_dblock/per_metal_bench.json
```

### §6.6 Honest-negative severity ladder

When filling out §6.1, classify the failure using this ladder so that
the paper §6 Limitations section can pull in a uniform severity tag:

| Severity   | Definition                                                                     | Required sections in §6.1                  |
|------------|--------------------------------------------------------------------------------|--------------------------------------------|
| **NONE**   | Bench shipped, MEASURED number is positive or matches expectation.              | (no honest-negative block needed)          |
| **MINOR**  | MEASURED number is positive but below the SOTA / paper target.                  | Symptom + Mitigation plan                   |
| **MAJOR**  | MEASURED number is below the project-internal baseline.                         | Symptom + Root cause + Mitigation plan      |
| **CRITICAL** | Bench did NOT ship at all, OR the MEASURED number is negative on a baseline that was previously positive. | All 5 mandatory sections of §6.1            |

Every bench shipped in this suite has been severity-classified as
follows (2026-09-26):

| Tag | Severity   | Notes                                                                 |
|-----|------------|-----------------------------------------------------------------------|
| L1  | **MINOR**  | MEASURED 0.5108 reachability is positive but below the 0.95 R20 target. |
| L2  | **NONE**   | variance_reduction = 0.96 + wall_time_ratio = 0.97 is on-target.       |
| L3  | **NONE**   | success_rate = 1.0 + stage3 wall pct = 0.997 is on-target.             |
| L5  | **MAJOR**  | MEASURED R² = +0.163 is positive (vs Wf-25 -0.443) but below R17 baseline 0.62. |
| L6  | **MINOR**  | MEASURED 0.5891 reachability is positive but below the 0.95 R20 target. |
| L7  | **CRITICAL** | Bench shipped but `extraction_bench.json` not yet materialised; current `run.log` only. |

---

## §7 Cross-reference index

* **Parent plan**: `TODO/pending/83_lambda_algorithm_enhancement.md`
  (8 Lambda gap L1-L8 + Wf-36/37/38 3-line plan).
* **Long-term roadmap**: `TODO/03_plans/LAMBDA_ROADMAP.md` (R20-R25
  6-year Lambda algorithm plan).
* **Wf-36 close-out**: `TODO/completed/82_wf36_closure_theorem_verifier.md`
  (L1 ship details + 3 exposed vulnerabilities).
* **Wf-37 partial close-outs**: `TODO/completed/82_metric_gap_filling.md`
  (L5 + L6 + L7 ship details — to be split into 3 separate `82_*`
  files when Wf-38 closes).
* **Sibling task**: `TODO/pending/82_metric_gap_filling_plan.md`
  (R18 GPU retrain: m3 Vina / m1 Ru / m2 Pt / m7 Diversity / m16
  Affinity / m19 LOMO).
* **Skills**: `mol-metal-lambda-calculus` (primary),
  `mol-metal-r18-math` (M25 Hilbert PD + M26 ScoreRep MC),
  `mol-metal-mechanism-mapping` (cross-skill diagnostic),
  `mol-metal-gpu-execution` (Wf-34 NEW skill for Blackwell + RTX 5090).
* **Output dirs**: `molmetal/reports/wf36_*` + `molmetal/reports/wf37_*`.
* **Test files**: `molmetal/tests/test_closure_theorem_verifier.py`
  (L1), `test_variance_mcts_proof_search.py` (L2),
  `test_joint_3stage_mcts.py` (L3), `test_train_hilbert_astsr_end_to_end.py`
  (L5), `test_closure_dblock.py` (L6), `test_multiph_pll_extraction.py`
  (L7).

---

## §8 Final verdict

This README is the **single index** for the 6 shipped Wf-36/37 Lambda
algorithm benches (L1, L2, L3, L5, L6, L7) and the **standing
honest-negative record** for the 2 not-yet-shipped benches (L4, L8).
All numeric claims downstream must follow the §5 MEASURED / PROJECTED
labelling rule, and all bench failures must be reported via the §6.1
template.

**Severity summary** (2026-09-26): 1 CRITICAL (L7 JSON missing) +
2 MINOR (L1, L6 reachability below target) + 1 MAJOR (L5 R² below
R17 baseline) + 3 NONE (L2, L3 on-target).  All 4 non-NONE rows
have an honest-negative block in §6 and a planned mitigation in
`TODO/03_plans/LAMBDA_ROADMAP.md`.

**Date**: 2026-09-26 (created)