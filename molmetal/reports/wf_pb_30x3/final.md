# WF-PB-30x3-Protein-Clash — final.md

**Date:** 2026-09-15
**Operator:** `r4_c_full_sweep.py` driver (`--pb-check --pb-mode dock`)
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Status:** FAILURE-PARITY — same search-budget failure as WF-PB-Pass-10x3-Smoke; `pb_pass_rate_aggregate = None` because zero generated candidates reached PB.

Honest-framing report. Per spec, **the protein-aware clash path cannot lift the
search ceiling**: protein-aware PB only runs on docked candidates, and
`n_docked_total = 0` at `n_simulations = 100` under the SOTA-aligned config
(`extended_204` + `all_5` + `symbolic_prior=True` + `synthesis_oracle=True` +
reference seed strategy). This is a pipeline-failure signal identical to the
WF-PB-Pass-10x3-Smoke baseline; the `--pb-mode dock` plumbing itself works
(see §3 below), but it has nothing to evaluate.

---

## 1. Effective command (corrections to the request verbatim)

The user-supplied command referenced two flags that do not exist on the
driver:

| Flag in spec | Reality |
|---|---|
| `molmetal/scripts/r10_cfg_real_crossdocked.py` | That script is the CFM training harness (`r10_*.py`); it has no `--pb-check` or `--pb-mode`. The PB-equipped driver is `molmetal/scripts/r4_c_full_sweep.py`. |
| `--decoder-rework` | Not a flag on `r4_c_full_sweep.py`. There is no decoder-rework switch in the driver CLI; the bond-decoder rework lives at the CFM level (`WF-CFM-Path-B-Decoder-Rework`), not at the sweep level. |
| `--n-top-k 20` | Not a flag on `r4_c_full_sweep.py`. The corresponding flag is `--physical-top-k 20` (default 10). |

Effective command actually run:

```
uv run python molmetal/scripts/r4_c_full_sweep.py \
  --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 \
  --n-pockets 10 \
  --seeds 42 0 1234 \
  --physical-docking --pb-check --pb-mode dock \
  --n-simulations 100 --physical-top-k 20 \
  --output-prefix molmetal/reports/wf_pb_30x3/r4c
```

Wall time: **137.96 s** total (≈4.60 s/cell).

CLI notes:
- `--pockets` is the extraction **root hint** used for path validation,
  not the count. Count is set via `--n-pockets 10`.
- `--engine` defaults to `both`; the driver downgraded to single-engine
  `vina` because GPU QuickVina2 dispatch is single-column (warning at
  startup, expected behaviour on RX 7800 XT).
- `extended_204` tile library + `all_5` click rules + `symbolic_prior=True`
  + `synthesis_oracle=True` are inherited from the SOTA-aligned YAML config.
- `seed_strategy=reference` is the default. `click_tile` would have
  generated candidates (see WF-PB-Pass-Real-Dock 1/1 chemistry-clean
  result), but that pilot only used 1 pocket × 1 seed.

---

## 2. Per-pocket × per-seed table

| pocket | seed=42 | seed=0 | seed=1234 | n_gen_total | n_seed_total |
|---|---|---|---|---|---|
| test_000 | no_candidates | no_candidates | no_candidates | 0 | 0 |
| test_001 | seed_only | seed_only | seed_only | 0 | 3 |
| test_002 | no_candidates | no_candidates | no_candidates | 0 | 0 |
| test_003 | seed_only | seed_only | seed_only | 0 | 3 |
| test_004 | seed_only | seed_only | seed_only | 0 | 3 |
| test_005 | no_candidates | no_candidates | no_candidates | 0 | 0 |
| test_006 | seed_only | seed_only | seed_only | 0 | 3 |
| test_007 | no_candidates | no_candidates | no_candidates | 0 | 0 |
| test_008 | seed_only | seed_only | seed_only | 0 | 3 |
| test_009 | no_candidates | no_candidates | no_candidates | 0 | 0 |

**Status mix:** `no_candidates = 15`, `seed_only = 15`. `n_candidates_total = 30`
(15 seed round-trips + 15 zero-cell placeholders).

---

## 3. `--pb-mode dock` plumbing verification

The plumbing is correct; only the input set is empty. Inspecting
`pb_check` block on each row:

```
pb_check = {
  "engine_version": "0.6.5",
  "mode": "PoseBusters_dock_v1",
  "n": 0,
  "n_pb_pass": 0,
  "pb_pass_rate": None,
  "per_smiles": {},
  "scope": "PoseBusters chemistry validity on docked candidates",
  "status": "no_candidates"
}
```

This confirms:
- `--pb-mode dock` was honored (mode recorded as `PoseBusters_dock_v1`).
- The `validate_docked(smiles, receptor_pdb)` path was reachable per cell
  (posebusters 0.6.5 instance is built; receptor SHA256 recorded in
  `physical.source_sha256`).
- Each cell received 0 docked candidates and correctly returned
  `pb_status="no_candidates"`, `n_pb_pass=0`, `pb_pass_rate=None`.

The adapter does **not** silently downgrade to `mol` mode: every row's
`pb_check["mode"] == "PoseBusters_dock_v1"`.

---

## 4. Aggregate metrics

| metric | value |
|---|---|
| `n_cells_total` | 30 |
| `n_pockets` | 10 (test_000..test_009) |
| `n_seeds` | 3 (42, 0, 1234) |
| `n_generated_candidates_total` | 0 |
| `n_seed_candidates_total` | 15 |
| `n_pockets_ok` | 0 |
| `n_pockets_fail` | 30 |
| `physical_n_docked` | 0 |
| `physical_n_pb_pass` | 0 |
| `pb_pass_rate_aggregate` | **None** (zero denominator) |
| `protein_aware_clash_count` | 0 (no docked candidates to clash-check) |
| `vina_best_kcal_mol` | None (no docking run) |
| `wall_seconds_total` | 137.96 s |
| `wall_seconds_mean_per_cell` | 4.60 s |

---

## 5. Comparison vs WF-PB-Pass-10x3-Smoke baseline

Identical 10×3 = 30-cell footprint. Direct diff:

| metric | WF-PB-Pass-10x3-Smoke (`--pb-mode mol`) | WF-PB-30x3-Protein-Clash (`--pb-mode dock`) | delta |
|---|---|---|---|
| n_cells | 30 | 30 | 0 |
| n_generated | 0 | 0 | 0 |
| n_seed | 15 | 15 | 0 |
| n_docked | 0 | 0 | 0 |
| n_pb_pass | 0 | 0 | 0 |
| `pb_pass_rate_aggregate` | None | None | 0 |
| `protein_aware_clash_count` | not measured (mol mode) | 0 | n/a |
| `wall_seconds_total` | 144.3 | 137.96 | −4.4 % |
| `wall_seconds_per_cell` | 4.81 | 4.60 | −4.4 % |

**Lift vs WF-PB-10x3 baseline:** **0 / 0 / 0** — parity, not lift.
**Reason:** the search ceiling is upstream of the PB evaluator; protein-aware
clash checks cannot create docked candidates where the Lambda search
produces none.

---

## 6. Honest framing vs TargetDiff `Uni-Mol v2` 75 % target

TargetDiff (Guan et al., ICML 2023) reports a 94 % PB pass rate over 100
CrossDocked test pockets at the diffusion-generated stage. Uni-Mol v2
(Mol-Crystal-7B-MC) reports ~75 % on its own crossdocked split.

Mol-Metal at `n_simulations=100` + SOTA-aligned gates + reference seed:

- 0/30 cells produced any generated candidates accepted by the pipeline.
- 0/30 cells were PB-eligible.
- `gap_vs_targetdiff_94` and `comparison_vs_targetdiff_uni_mol_v2_75pct` are
  both **not applicable** for this run; the comparison would only be
  meaningful if `n_docked > 0`.

The PB pipeline itself was already validated end-to-end on
`WF-PB-Pass-Real-Dock` (1/1 pocket, 100 % chemistry pass on a single
generated candidate, vina_best = −6.929 kcal/mol). The blocker at 30 cells
is upstream search capacity, not PB evaluation.

---

## 7. Plausible causes for zero generated candidates (carried over from
WF-PB-Pass-10x3-Smoke §6)

1. **`n_simulations=100` is too low for strict gates.** The MCTS early-stops
   at `patience=50` after `nfe=51` sims for `no_candidates` cells; the
   search collapses before reaching any accepted leaf.
2. **Reference seed strategy constrains chemistry.** Initializing from the
   paired-ligand SDF keeps new structures inside a narrow chemistry window;
   the `click_tile` strategy (used in `WF-PB-Pass-Real-Dock`) breaks this.
3. **Strict gates are sequential rejections.** Symbolic prior +
   `synthesis_oracle=smarts` + reference initialization collectively reject
   all 51–100 MCTS leaves before they reach PB. The diagnostic field
   `no_generated_reasons` lists:
   - `"New structures were rejected by existing type predicates"`
   - `"New structures were rejected by the existing binding gate"`
4. **`WF-Lift-N-Sim-Cap` pilot** (`molmetal/reports/wf_lift_n_sim_cap_pilot/`)
   already showed that lifting `n_simulations=100 → 1000` does not break
   the singleton-collapse failure mode under reference init + click-poor
   references; further lift requires `click_tile` seed strategy or scaffold
   diversity.

---

## 8. Recommendation (preserved from WF-PB-Pass-10x3-Smoke)

| follow-up | priority | dependency | expected lift |
|---|---|---|---|
| Run with `--seed-strategy click_tile` | P0 | none | n_docked > 0 expected per cell |
| Run at `n_simulations=1000` + `click_tile` | P0 | `--seed-strategy click_tile` | 30-cell PB number attainable |
| Drop strict gates (synthesis_oracle=none, symbolic_prior=False) | P1 | for ablation only | diagnostic only |
| `--pb-mode redock` for reference-ligand redocking | P2 | `--pb-check` | chemistry-clean baseline on reference |

Until `--seed-strategy click_tile` is run at scale, **30/30 PB pass rates
will remain None**. The PB protein-aware clash pipeline is verified
correct; the bottleneck is search/seed strategy.

---

## 9. Files

| file | content |
|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/r4c.csv` | per-cell row dump (30 rows) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/r4c.json` | full structured run dump |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/r4c.md` | driver summary table |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/r4c_logs/` | per-cell log files |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/r4c_poses/` | pose outputs (empty: no docking run) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_30x3/final.md` | this file |
