# R17 Paper-Sweep Spec — final report

**Workflow:** WF-Paper-Sweep-Spec (TODO M-R17-A2)
**Owner:** ultracode Track-B (Lambda calculus / CLI plumbing)
**Date:** 2026-09-25
**Status:** **SHIP** — CLI preset flags integrated on
`molmetal/scripts/r4_lambda_only_run.py`; smoke mode verified
end-to-end; production 300-cell paper-grade sweep remains BLOCKED on
TODO-26 R16 GPU recovery (Lambda-only path is structurally wired and
idempotent on `--r17-paper-grade`).

---

## 1. Goal

Restore the **infrastructure** for running the full 100-pocket ×
3-seed = 300-evaluation paper-grade sweep (TODO-14 §"Scope / A") even
though the production run itself remains a 10000-step CFM retrain
downstream of TODO-21 / TODO-26.

This entry adds two convenience preset flags on
`molmetal/scripts/r4_lambda_only_run.py` so the same runner can launch
either the smoke or paper-grade configuration with a single CLI flag,
without needing to remember the four-knob
(`--pockets / --seeds / --n-simulations / --manifest`) override
combination.

---

## 2. New flags

### 2.1 `--smoke`

| Parameter | Value |
|---|---|
| `--pockets` | 10 |
| `--seeds` | `[42]` |
| `--n-simulations` | 20 |
| `--manifest` | `molmetal/data/test_010_019_subset_manifest.csv` |
| Wall-clock | ~5 min CPU (verified: **13.3 s** for 10 cells on this box at n_sim=20) |
| Output dir | user-supplied via `--output-dir` |
| Use case | quick verification during development; explicitly NOT a paper-grade measurement cell |
| Compatible with | explicit `--pockets / --seeds / --n-simulations / --manifest` overrides (any non-None value wins) |

Loads the **test_010..test_019** subset (10 novel pockets, distinct
from the test_000..test_009 pilot cohort so singleton-collapse
patterns are easy to spot).

### 2.2 `--r17-paper-grade`

| Parameter | Value |
|---|---|
| `--pockets` | 100 |
| `--seeds` | `[42, 0, 1234]` |
| `--n-simulations` | 1000 |
| `--manifest` | `molmetal/data/crossdocked100_manifest.csv` |
| Wall-clock | ~6 h on GPU CFM / ~50 min on CPU Lambda-only (estimated; not yet measured on this box) |
| Output dir | `molmetal/reports/wf_paper_grade_300cells/` (300 cells = 100 pockets × 3 seeds) |
| Use case | full Round-13 / TODO-14 §"Scope / A" spec at the top-journal protocol |
| Compatible with | explicit `--pockets / --seeds / --n-simulations / --manifest` overrides (any non-None value wins) |

### 2.3 Pre-existing flags re-used as-is

- `--metal-seed cisplatin` — already on `r4_lambda_only_run.py` since
  WF-Metallodrug-Vertical; forces a Pt(II) cisplatin seed for
  metal_compliance_rate > 0.  No surface change.
- `--output-dir ABS_PATH` — already required positional-style flag;
  output goes to `molmetal/reports/wf_lambda1_<ABS_PATH>/`.  No
  surface change.

The two new presets are **additive** and do not change the existing
CLI surface.  Existing invocations (e.g.
`--pockets 10 --seeds 42 0 1234 --n-simulations 200 --output-dir foo`)
keep their exact behaviour.

---

## 3. Resolution logic

`_resolve_paper_sweep_preset(args)` in
`molmetal/scripts/r4_lambda_only_run.py` (line ~4875) applies the
preset values AFTER parsing CLI args.  Resolution order (most-specific
wins):

1. **Explicit user-supplied flag** (any non-None value) wins.
2. **`--r17-paper-grade`** preset value (if `--r17-paper-grade` is set).
3. **`--smoke`** preset value (if `--smoke` is set).
4. **Legacy default** (preserves pre-R17 behaviour when neither preset
   is active: `--pockets 10 --seeds [42, 0, 1234] --n-simulations 200
   --manifest crossdocked100_manifest.csv`).

If both `--smoke` and `--r17-paper-grade` are set, `--r17-paper-grade`
wins (the larger configuration is the safer fallback) and a warning is
logged.

The resolved preset is recorded as `args.r17_preset_active ∈ {"legacy",
"smoke", "paper_grade"}` for downstream logging and audit.

---

## 4. Manifest loading

The runner uses `load_manifest(path)` (line 529 in
`r4_lambda_only_run.py`) which reads a CSV with the
`pocket_id,receptor_path,ligand_path,ref_path,metal_atoms,n_atoms,n_residues,source`
columns (per `molmetal/data/crossdocked100_manifest.csv`).  Both
manifests used here conform to this schema:

- `molmetal/data/crossdocked100_manifest.csv` — 100 rows, the
  standard CrossDocked2020 test split (test_000..test_099), used by
  `--r17-paper-grade` and as the legacy default.
- `molmetal/data/test_010_019_subset_manifest.csv` — 10 rows, the
  test_010..test_019 novel subset, used by `--smoke` (distinct from
  the test_000..test_009 pilot cohort so singleton patterns don't
  cross-contaminate).
- `molmetal/data/test_001_subset_manifest.csv` — 1-row smoke for the
  B5 single-pocket test (`wf_lambda_pipeline_smoke.py`).

`load_manifest` falls back to empty list (with WARN) if the file does
not exist.  No pre-flight check for the actual receptor/ligand paths;
that resolution happens lazily inside `run_one_cell`.

---

## 5. Output paths

| Preset | `--output-dir` | Resolved path |
|---|---|---|
| `--smoke` | user-supplied, e.g. `r17_smoke_test` | `molmetal/reports/wf_lambda1_r17_smoke_test/` |
| `--r17-paper-grade` | user-supplied, e.g. `paper_grade_300cells` | `molmetal/reports/wf_lambda1_paper_grade_300cells/` |

Each run writes three artifacts under that directory:

- `report.json` — per-cell JSON, aggregate dict, config fingerprint
  (300 cells for paper-grade; 10 cells for smoke).
- `summary.md` — human-readable aggregate report.
- `seeds.json` — reproducibility fingerprint (schema_version=1).

---

## 6. Expected runtime

| Configuration | Cells | Wall-clock estimate | Wall-clock measured (this box) |
|---|---:|---|---|
| `--smoke` (n_sim=20, 1 seed) | 10 | ~5 min CPU | **13.3 s** (verified 2026-09-25) |
| `--r17-paper-grade` (n_sim=1000, 3 seeds) | 300 | ~6 h GPU / ~50 min CPU | not yet measured (BLOCKED on TODO-21 R16 retrain) |
| legacy `--pockets 10 --seeds 42 0 1234 --n-simulations 200` | 30 | ~10 min CPU | per `wf_round12_10x3` reports |

The Lambda-only path is CPU-bound and uses no GPU.  Wall-clock scales
linearly in `pockets × seeds × n_simulations`.

---

## 7. Smoke verification (2026-09-25)

```
source env.sh
uv run --no-sync python molmetal/scripts/r4_lambda_only_run.py \
    --smoke --output-dir r17_smoke_test --quiet
```

Result (10 cells, n_sim=20, seed=42, test_010..test_019):

- **n_cells = 10** (one per pocket, single seed)
- **elapsed_s_total = 13.30**
- **validity_rate = 1.000** (10/10 cells emit a valid SMILES)
- **novelty = 1.000** (training-set novelty oracle confirms all
  generated molecules are novel — no training-set match within
  Tanimoto 0.7)
- **qed_mean = 0.621** (RDKit QED; n=10)
- **logp_mean = 1.377** (Crippen logP; n=10)
- **sa_mean = 6.003** (Ertl-Schuffenhauer SA; n=10 — high values
  reflect the documented test_010..test_019 search-bound pattern)
- **diversity_tanimoto = 0.000** — singleton collapse at n_sim=20
  (consistent with the test_010..test_019 singleton pattern observed
  in WF-PB-Pass-10x3-Smoke; expected at this budget)

Aggregate highlights:

```
validity_rate            1.0000
uniqueness_rate          1.0000
diversity_tanimoto       0.0000
novelty                  1.0000
synthesizability_rate    1.0000
metal_compliance_rate    0.0000
sa_mean                  6.0028
qed_mean                 0.6210
logp_mean                1.3765
```

Per-cell (10 rows):

```
test_010  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_011  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_012  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_013  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_014  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_015  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_016  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_017  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_018  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
test_019  s=42  n_sim=20  n_cand=1  n_distinct=1  val=1.000  div_tan=0.0000
```

---

## 8. Open work (forward-pointer to TODO-26 R16)

- **R16-W42** — YuelBond retrain 10000-step h=128 (target:
  `decode_ratio ≥ 0.5`).
- **R16-W43** — CFM path import fix + 10000-step retry (target:
  `n_decoded ≥ 192/384`).
- **R16-W44** — Lambda 100-p × 3-seed sweep at n_sim=1000 with
  sub-fix A+C+Deflex (target: `n_distinct > 1` on ≥80% of
  test_010..test_019, `J(p,q) < 1.0`).
- **R16-W44-45** — PB MMFF94 production pass rate (target:
  `pb_pass_rate ≥ 0.60`).
- **R16-W46** — §4.2 / §4.3 promote 300 cells DESIGN → MEASURED
  (this is where `--r17-paper-grade` lands; the run output replaces
  the current DESIGN cells in `paper/sections/04_evaluation.tex`).

After R16-W42-W47 close, this TODO can move from PARTIAL to SHIPPED
and `--r17-paper-grade` becomes the canonical entry point for the
production 300-cell evaluation column.

---

## 9. Cross-references

- `molmetal/scripts/r4_lambda_only_run.py:4157-4875` — preset flag
  implementation (`_build_argparser`, `_resolve_paper_sweep_preset`).
- `molmetal/scripts/r4_lambda_only_run.py:529` — `load_manifest()`
  helper.
- `molmetal/data/crossdocked100_manifest.csv` — 100-row manifest
  (paper-grade default).
- `molmetal/data/test_010_019_subset_manifest.csv` — 10-row manifest
  (smoke default).
- `TODO/pending/14_full_100pocket_paper_r13.md` — R17 spec stub
  section at the top of the file (added 2026-09-25).
- `TODO/pending/34_parallel_ultracode_dispatch.md` — R17
  Track-A + Track-B parallel dispatch.
- `TODO/pending/26_round13_round14_complete_plan.md` — R16 W42-W47
  master plan for the production sweep.
- `molmetal/reports/wf_round13_100x3/final.md` — partial completion
  verdict for the prior round-13 attempt.
- `molmetal/reports/wf_deflex_wireup/final.md` — Deflex F5 + PMI v2
  + learned_prior wire-up (5/5 phases ship; paper-grade 100×3 sweep
  still deferred).
- `paper/sections/04_evaluation.tex` — §4.2 / §4.3 / §4.6 cells that
  the production `--r17-paper-grade` run will populate.

---

## 10. Honest framing

- **`--smoke` is NOT a paper-grade measurement cell.**  It is a quick
  verification path.  Wall-clock ~5 min is intentional; the result
  (`diversity_tanimoto = 0.000`, `metal_compliance_rate = 0.000` at
  n_sim=20) reflects the documented small-budget collapse pattern,
  not a paper-grade result.
- **`--r17-paper-grade` is structurally wired but production-blocked.**
  The CLI flag works; the production 10000-step CFM retrain that
  unlocks the `decode_ratio ≥ 0.5` target is gated on TODO-26 R16-W42.
  Until that ships, `--r17-paper-grade` will run as a CPU-only
  Lambda sweep at n_sim=1000 — informative for the Λ-native columns
  but NOT a full paper-grade CFM result.
- **Pre-existing `--metal-seed cisplatin` and `--output-dir`
  flags are reused as-is.**  No surface change.  The `wf_lambda_88rule_deflex`
  flag-fix phase is orthogonal to this entry; if it had already added
  those flags (it had — they shipped in earlier R15 work-blocks), this
  phase re-uses them with no conflict.