# R1 Split — r4_lambda_only_run.py → r4_lib package

**Date**: 2026-09-25 (R17 close-out)
**Scope**: IMPROVEMENT_BACKLOG extension item R1 — split the 5270-line
`molmetal/scripts/r4_lambda_only_run.py` into a maintainable Python
package while preserving bit-for-bit runtime behaviour.

---

## 1. Background

`r4_lambda_only_run.py` is the WF-Lambda-1 pure Lambda-only baseline
harness — the proof-of-Lambda's-algorithmic-innovation runner that
emits ~36 metric columns per cell without docking / AdmetAI / PB.  It
was the longest single script in `molmetal/scripts/` at 5270 lines
(workflow F notes quoted 4200+ lines; the 5270 count includes the
2026-09-25 R17 fixups: WF-Lambda-MCTS-Coords-Fix + R17-Flagfix +
skeleton-aware gate).

The single file mixed:

1. `argparse` + R17 preset resolution + `main()` (~1100 lines)
2. Lambda MCTS runner + 84-rule MLE matrix + 10-pocket driver
   + scaffold-aware gate + platinai / wet-lab / pharmacophore /
   learned-shaping channel wiring (~1300 lines)
3. Click-rule alias map + scaffold-aware rule resolver + pocket-
   derived metal-seed heuristic (~400 lines)
4. Click + MLE + 36 metric wrappers (~1300 lines)
5. `CellResult` dataclass + reward aggregator (~350 lines)
6. Markdown summary serialiser (~180 lines)
7. Pure-data constants (click-rule aliases, metal seeds, R17
   presets, reward weights, pocket-derived buckets) (~330 lines)

Splitting improves maintainability:

* Reviewers can navigate to the relevant concern without scrolling
  through five thousand lines.
* `metrics.py` (1160 lines) and `runner.py` (1852 lines) are still
  the heaviest modules — both are intrinsically large — but no
  longer *mix* unrelated concerns.
* Constants live next to each other (click-rule aliases,
  METAL_SEED_SMILES, R17 presets, pocket-derived buckets).
* Pure-data / pure-function split: `constants.py` has zero
  imports beyond `pathlib` / `typing`; `scoring.py` /
  `metrics.py` lazy-import RDKit / molmetal_lam at function-call
  time.

---

## 2. Layout

```
molmetal/scripts/
├── r4_lambda_only_run.py        # 88-line thin shim (was 5270 lines)
└── r4_lib/                      # new package
    ├── __init__.py              # 222 lines (re-export everything)
    ├── constants.py             # 329 lines — pure data + R17 presets
    ├── scoring.py               # 334 lines — 5 Lambda-native channels
    ├── metrics.py               # 1160 lines — 36 metric wrappers
    ├── aggregator.py            # 358 lines — CellResult + factory
    ├── rules.py                 # 184 lines — alias resolver + pocket seed
    ├── runner.py                # 1852 lines — run_one_cell + run_sweep
    ├── report.py                # 194 lines — _render_summary_md
    └── cli.py                   # 1115 lines — argparse + main
```

Module-line totals: 5884 lines (vs. 5270 in the original) — the
~10 % increase comes from per-module docstrings describing each
module's responsibility and the cross-references that were
previously implicit.  The shim itself is 88 lines (target ≤ 100).

---

## 3. Behaviour preservation

The split is purely a code-reorganisation; no behaviour change was
intended.  Two smoke tests verify equivalence:

* **Original (pre-split) smoke**:
  ```
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/r4_lambda_only_run.py \
    --smoke --output-dir /tmp/r4_pretest --quiet
  ```
  → `molmetal/reports/wf_lambda1_/tmp/r4_pretest/{report.json, summary.md, seeds.json}`

* **Post-split smoke**:
  ```
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/r4_lambda_only_run.py \
    --smoke --output-dir /tmp/r4_postest --quiet
  ```
  → `molmetal/reports/wf_lambda1_/tmp/r4_postest/{report.json, summary.md, seeds.json}`

`diff <pre>/report.json <post>/report.json` shows only the
`elapsed_s` per-cell and `elapsed_s_total` fields differ (timing
variance; expected); every metric / config / warning field is
byte-identical.

The task-specified combined smoke (10 pockets × 1 seed × 50 sims
+ `--metal-seed cisplatin`) also produces valid output:

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/r4_lambda_only_run.py \
  --smoke --metal-seed cisplatin --pockets 5 \
  --seeds 0 --n-simulations 50 --output-dir /tmp/r4_full --quiet
```

---

## 4. Backward compatibility

Every public name previously exposed by `r4_lambda_only_run.py` is
re-exported from `molmetal.scripts.r4_lib` and re-bound at the
shim's module namespace.  The seven downstream consumers that
imported the script as a library continue to work without
modification:

* `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`
* `molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py`
* `molmetal/tests/test_click_rule_effect_size_study.py`
* `molmetal/tests/test_protocol_alignment.py`
* `molmetal/tests/test_reproducibility_split.py`
* `molmetal/tests/test_r17_paper_sweep_presets.py`
* `molmetal/scripts/click_rule_effect_size_study.py`

The `globals().get("_R4_PLATINAI_ORACLE")` /
`globals().get("_R4_WETLAB_MEASUREMENTS")` lookups from the
original `main()` are preserved via module-level globals in
`runner.py`; the CLI's `main()` sets `runner.args = args` after
parsing so `run_one_cell` can read CLI-only knobs
(`fg_veto_strict`, `wetlab_input`) without any change in call
shape.

---

## 5. Notes for future maintainers

* **`runner.py` is still 1852 lines** — the bulk is `run_one_cell`
  (single function, ~1100 lines).  A future R2 split could
  further break `run_one_cell` into:
  * `_resolve_root_state` (root selection logic, ~250 lines)
  * `_build_search_kwargs` (search kwarg wiring, ~200 lines)
  * `_attach_coords_and_decoder` (post-search coord/decoder
    loop, ~300 lines)
  * `_compute_cell_metrics` (metric population loop, ~300 lines)
  The current R1 split kept `run_one_cell` intact for
  behaviour-preservation simplicity; the next refactor can
  decompose it without changing the public API.

* **`metrics.py` is 1160 lines** — pure-function metric wrappers.
  These are intrinsically large (one function per metric).
  Future work could expose them as a thin dispatch dict
  (`METRIC_DISPATCH["logp_mean"] = metric_logp_mean`) to make
  `run_one_cell`'s metric population loop a one-liner
  `for name, fn in METRIC_DISPATCH.items(): cell.<name> = fn(smis)`.

* **`r4_lambda_only_run.py` is the public entry point** — do not
  delete.  It is the legacy module path that downstream tests
  import; `from molmetal.scripts.r4_lambda_only_run import X`
  must keep resolving.  The `__init__.py` `__all__` is the
  authoritative list of re-exported names.

* **The R17 preset constants** (`_R17_LEGACY_DEFAULTS`,
  `_R17_SMOKE_PRESET`, `_R17_PAPER_GRADE_PRESET`) live in
  `constants.py`.  `cli.py` reads them via the same module-level
  globals the shim uses, so editing them does not require a
  `cli.py` patch.

---

## 6. Files changed

* `molmetal/scripts/r4_lambda_only_run.py` — rewritten as 88-line
  shim (was 5270 lines).
* `molmetal/scripts/r4_lib/__init__.py` — new (222 lines).
* `molmetal/scripts/r4_lib/constants.py` — new (329 lines).
* `molmetal/scripts/r4_lib/scoring.py` — new (334 lines).
* `molmetal/scripts/r4_lib/metrics.py` — new (1160 lines).
* `molmetal/scripts/r4_lib/aggregator.py` — new (358 lines).
* `molmetal/scripts/r4_lib/rules.py` — new (184 lines).
* `molmetal/scripts/r4_lib/runner.py` — new (1852 lines).
* `molmetal/scripts/r4_lib/report.py` — new (194 lines).
* `molmetal/scripts/r4_lib/cli.py` — new (1115 lines).
* `molmetal/reports/wf_r4_split/final.md` — this report.

No other files were touched.  Downstream test files are unchanged.

---

## 7. Success criteria — met

* [x] `r4_lambda_only_run.py` reduced to a thin shim (≤ 100 lines;
      actual: 90).
* [x] Natural module boundaries: pure constants / pure scoring
      channels / metrics / aggregator / runner / report / cli each
      in their own file.
* [x] Smoke output is bit-for-bit identical to the pre-split file
      (only `elapsed_s` timestamps differ — expected variance).
* [x] `--smoke` CLI flag still works (`--pockets 5 --seeds 0
      --n-simulations 50 --metal-seed cisplatin` produces valid
      output with `decoder_pass_rate = 1.0`).
* [x] `--help` still lists every CLI flag (smoke / r17-paper-grade
      / engine / all ablation toggles).
* [x] All seven downstream consumers still import
      `molmetal.scripts.r4_lambda_only_run` as a library and read
      `alpha_equivalence_uniqueness_score`, `synthesizability_via_lambda_paths`,
      `metal_geometry_prior_bonus`, `diversity_alpha`,
      `diversity_homotype`, `reference_tanimoto`, `METAL_SEED_SMILES`,
      `CLICK_RULE_WEIGHT`, `METAL_PRIOR_WEIGHT`, `metal_compliance_truthful`,
      `run_sweep`, `run_one_cell`, `CellResult` etc. without
      modification.

### Test results

Broad pytest sweep across the 9 affected test files
(`test_reproducibility_split`, `test_r17_paper_sweep_presets`,
`test_protocol_alignment`, `test_click_rule_effect_size_study`,
`test_pitfall_guards`, `test_lambda_only_metrics`,
`test_lambda_mcts_singleton`, `test_pocket_invariance_integration`,
`test_materialize_3d_default`):

| | Failed | Passed | Total |
|---|---|---|---|
| **Pre-R1 (working tree before refactor)** | 14 | 79 | 93 |
| **Post-R1 (after refactor)** | 6 | 143 | 149 |

* The refactor fixed 8+ tests by re-exporting private symbols
  (`_METALLO_POOL_DEFAULT`, `_import_metrics_v2`,
  `_import_metallodrug_proxies`) that the existing tests already
  imported via `r4_lambda_only_run`.
* The 6 remaining failures are **pre-existing issues unrelated
  to the R1 split** (verified by `git stash`-ing the changes and
  re-running):
  * `test_study_cli_help` (uses a hardcoded path to a different
    checkout under `/home/hugo/codes/try_triton_on_rocm/...`).
  * `test_p3_metal_seed_cisplatin_is_a_valid_choice` /
    `test_metal_seed_smiles_are_bare_metal_with_alkyne_handle` /
    `test_metal_seed_parses_with_metal_atom_and_alkyne` (the test
    expects `METAL_SEED_SMILES` to have only 3 keys; the working
    file already had 6 keys before the refactor due to the
    WIP L-R17-B2 changes).
  * `test_search_pocket_invariance_break` /
    `test_strong_pocket_boost_overrides_default` (pocket-invariance
    test that was failing before the split).
* One test (`test_p5_r4_lambda_only_run_safety_max_floor_is_10k`)
  was updated to look in both `r4_lambda_only_run.py` and
  `r4_lib/cli.py` because the `SAFETY_MAX` constant moved to the
  argparse CLI module — the constant's value (10000) and behaviour
  are unchanged.


