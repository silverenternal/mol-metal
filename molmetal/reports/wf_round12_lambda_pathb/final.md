# WF-Round12-Lambda-PathB — Round-12 Lambda Pilot with Path B (scaffold-aware + decoder-rework flag)

> **Honest-framing**: this is a MEASURED run on `2026-09-15`.
> Total wall-clock **50.44 s** for 30 cells (10 pockets × 3 seeds) at
> n_simulations=1000, n_top_k=20. The result is the same singleton
> collapse as the Round-12 Lambda Pilot baseline. **Path B does NOT lift
> n_distinct off 1 on this configuration.**

## 1. Configuration

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | 10 |
| `--seeds` | `[42, 0, 1234]` |
| `--n-simulations` | 1000 |
| `--n-top-k` | 20 |
| `--metal-seed` | `cisplatin` (SMILES `[Pt]C#C` — Fix 1 from WF-Lambda-Fix-Singleton) |
| `--click-rules` | `all-5` (resolves to `{AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne}`) |
| `--decoder-rework` | True (control-only flag — see §6) |
| `--output-dir` | `wf_round12_lambda_pathb/r4c` → `molmetal/reports/wf_lambda1_round12_pathb_r4c/` |
| Python | 3.12 (uv-managed) |
| ROCm / Triton | 7.2 / 3.8.0 / gfx1101 wave64 |

## 2. Wiring verification (preflight)

| item | status | evidence |
|---|---|---|
| `pt_click_compat` scaffold-aware gate wired | YES | `molmetal/scripts/r4_lambda_only_run.py:1716-1720` imports `detect_scaffold`, `default_compatible_rules`, `incompatible_rules` |
| scaffold detection runs on `auto-*` aliases | YES | `molmetal/scripts/r4_lambda_only_run.py:1705-1711` defines the 5 auto-alias set; gate triggers when user passes one of them |
| `--decoder-rework` flag exposed | YES (this run) | new argparse entry in `_build_argparser`; threaded through `run_one_cell` → `run_sweep` → `main`; recorded in `cell.warnings` for audit (`decoder_rework=True`) |
| `--decoder-rework` actually applied to Lambda candidates | NO — by design (see §6) | the chem-aware soft bond prior in `decoder_rework.py` operates on `(coords, Z)` tensors from the CFM module; the Lambda path produces typed `MoleculeClosedTerm` objects directly. No coordinate tensor exists on this entry point. |

The `--decoder-rework` flag was wired as a **control-only flag** for
audit purposes: the chem-aware soft bond prior in
`molmetal/molmetal_lam/lam_chem/decoder_rework.py` is a coordinate-space
decoder (it replaces the legacy 2.4 Å hard cutoff on CFM samples with a
soft distance mask × type-compatibility × valence-cap product). Lambda
MCTSProofSearch produces typed β-NF derivations and never instantiates a
coordinate tensor, so there is no place to apply decoder_rework in this
code path. The flag is therefore recorded in `cell.warnings` so the run
is auditable, but it does not change the candidate generation pipeline.
Use `molmetal/scripts/r10_cfg_real_crossdocked.py --decoder-rework` for
the CFM path where decoder_rework is meaningful.

## 3. CLI invocation (as requested)

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 0 1234 \
    --n-simulations 1000 --n-top-k 20 \
    --metal-seed cisplatin --click-rules all-5 \
    --decoder-rework \
    --output-dir round12_pathb_r4c
```

Note: the task brief specified `--output-prefix molmetal/reports/wf_round12_lambda_pathb/r4c`,
but the actual CLI flag is `--output-dir <name>` (the script joins it
with the `wf_lambda1_` prefix internally). The output is therefore at
`molmetal/reports/wf_lambda1_round12_pathb_r4c/` — the corresponding
per-run report directory under `molmetal/reports/wf_round12_lambda_pathb/`
contains this `final.md`.

## 4. Aggregate metrics (mean across 30 cells)

| metric | Path B (this run) | Round-12 Lambda Pilot baseline |
|---|---:|---:|
| n_cells | 30 | 5 |
| validity_rate | 1.0000 | 1.0000 |
| synthesizability_rate | 1.0000 | 1.0000 |
| uniqueness_rate | 1.0000 | 1.0000 |
| metal_compliance_rate | 0.0000 | 1.0000 |
| diversity_tanimoto_mean | 0.0000 | 0.0000 |
| diversity_homotype_mean | 0.0000 | 0.0000 |
| novelty | 1.0000 | 1.0000 |
| reference_tanimoto | 0.0014 | 0.0120 |
| anticancer_index | 0.1000 | n/a (not in baseline) |
| sa_mean | 6.0081 | n/a |
| qed_mean | 0.5087 | n/a |

Distinct SMILES across all 30 cells: **1** (`C#[C][Pt]`).
Distinct SMILES across all 5 baseline cells: **1** (`[NH2][Pt]([NH2])([Cl])[Cl]`).

> **Baseline note:** the "Round-12 Lambda Pilot baseline" cited in the
> task brief refers to `molmetal/reports/wf_lambda_metal_pilot/final.md`
> (5×1 with cisplatin + all-5 → n_distinct=1, div_tan=0.000). That
> pilot used the pre-Fix-1 cisplatin seed (`[NH2][Pt]([NH2])([Cl])[Cl]`).
> The current run uses the post-Fix-1 Pt-acetylide seed (`[Pt]C#C`),
> which round-1c made the canonical metal-acetylide carrier for the
> click rules. The aggregate `metal_compliance_rate` went 1.0000 → 0.0000
> because the Pt_0 acetylide does not satisfy the metal-geometry prior
> (coordination_number_mean=1.0 vs target=4 for Pt_II). This is a
> measurement of the Fix-1 effect on the compliance gate, not a Path B
> regression.

## 5. Per-cell panel (Path B, 30 cells)

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | syn | metal | ref_tan | oxid_state_dist |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| test_000 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_000 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_000 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_001 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_001 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_001 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_002 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_002 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_002 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_003 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_003 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_003 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_004 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_004 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_004 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_005 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.014 | Pt_0=1 |
| test_005 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.014 | Pt_0=1 |
| test_005 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.014 | Pt_0=1 |
| test_006 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_006 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_006 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_007 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_007 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_007 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_008 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_008 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_008 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_009 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_009 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |
| test_009 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | Pt_0=1 |

Total wall-clock: **50.44 s** (mean 1.68 s/cell, range 1.41 – 1.79 s).

## 6. Why Path B does not lift n_distinct on this configuration

Three structural reasons (each independently sufficient to block
diversity):

1. **`--decoder-rework` is not applicable to Lambda candidates.**
   The chem-aware soft bond prior replaces the CFM's 2.4 Å hard cutoff
   on (coords, Z) tensors. Lambda's MCTSProofSearch produces typed
   `MoleculeClosedTerm` objects directly from the MLC reduction
   grammar; no (coords, Z) pair is materialised on this code path, so
   there is no input for decoder_rework to consume. The flag was wired
   as a control-only audit hook (see `cell.warnings[decoder_rework=True]`).

2. **`--click-rules all-5` does NOT engage the scaffold-aware gate.**
   The scaffold-narrowing logic (lines 1705-1711) is only triggered when
   `--click-rules` contains one of the `auto-*` aliases (auto-pt-strict,
   auto-pt-iv, auto-pt-chelate, auto-labile, auto-unknown). Passing an
   explicit rule list (`all-5`) bypasses the scaffold detection entirely
   for backward compatibility with the WF-Wire-Click-Rules-All5 cell
   (per the in-source comment at line 1701-1703). To exercise the
   scaffold-aware gate, the user must pass e.g.
   `--click-rules auto-pt-strict` instead of `all-5`.

3. **Metal-seed is `[Pt]C#C` (Pt_0 acetylide), which MCTS collapses onto.**
   WF-Lambda-Fix-Singleton Fix 1 changed `METAL_SEED_SMILES["cisplatin"]`
   from the closed cisplatin square-planar seed
   (`[NH2][Pt]([NH2])([Cl])[Cl]`) to the open Pt-acetylide
   (`[Pt]C#C`) so that the click rules can attach to the dangling alkyne
   carbons. The MCTS at n_simulations=1000 still exhausts its budget
   with `C#[C][Pt]` as the only kept candidate across all 30 cells —
   no click-rule fires produce a kept child because the reward
   aggregator has no multi-step-path scoring that prefers the
   click-extended product over the bare Pt-acetylide seed. The
   WF-MCTS-Synth recommendations file (TODO-24) tracks this as an
   open follow-up; the recommended fix is to (a) give a non-zero
   reward shaping bonus for any candidate that fired >= 1 click rule
   AND (b) limit the singleton trap by penalising candidates whose
   `n_distinct_after_search=1` on cells with `n_simulations >= 1000`.

## 7. Verdict

| condition | result |
|---|---|
| Path B produces n_distinct > 1 on any cell | **NO** (n_distinct=1 on all 30 cells) |
| n_distinct_pathb vs n_distinct_baseline | **1 == 1** (no lift) |
| diversity_tanimoto_pathb vs div_tan_baseline | **0.000 == 0.000** (no lift) |
| diversity_homotype_pathb vs div_hom_baseline | **0.000 == 0.000** (no lift) |
| metal_compliance_rate_pathb vs baseline | **0.000 vs 1.000** (regressed) |
| lifted_from_baseline | **FALSE** |
| n_per_pocket_diverse_candidates | **0 / 30** cells (none) |

Path B as defined in this task brief does NOT lift the singleton
collapse on the Lambda path. The three structural blockers are listed
in §6 above; addressing them is the prerequisite for any future Path B
re-attempt.

## 8. Schema report

```yaml
n_distinct_pathb: 1
n_distinct_baseline: 1
diversity_tanimoto_pathb: 0.0000
diversity_tanimoto_baseline: 0.0000
diversity_homotype_pathb: 0.0000
diversity_homotype_baseline: 0.0000
metal_compliance_rate_pathb: 0.0000
metal_compliance_rate_baseline: 1.0000
lifted_from_baseline: false
n_per_pocket_diverse_candidates: "0 / 30 (all cells collapse to Pt_0 acetylide C#[C][Pt])"
n_cells_pathb: 30
n_cells_baseline: 5
elapsed_s_pathb: 50.44
elapsed_s_per_cell_mean: 1.68
verdict: NO_LIFT
structural_blockers:
  - decoder_rework_not_applicable_to_lambda_path
  - scaffold_aware_gate_bypassed_by_explicit_all-5_list
  - mcts_collapses_onto_Pt0_acetylide_seed_at_n_sim_1000
decoder_rework_intent_recorded: true
pt_click_compat_wiring_verified: true
scaffold_gate_engaged_this_run: false  # because click-rules=all-5 is explicit, not auto-*
```

## 9. Follow-ups

1. Re-run Path B with `--click-rules auto-pt-strict` instead of `all-5`
   so the scaffold-aware gate actually narrows the rule subset for the
   detected Pt_II strict category (will drop ThiolEne + AmideCoupling
   per the 5×5 Pt-click compat matrix in
   `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
2. Wire a chem-aware soft bond prior specifically for Lambda — i.e.
   replace the hard `mol_validity_score` rejection in MCTSProofSearch
   with a graded score that prefers typed-derived candidates over
   the bare seed. This is the Lambda-path analogue of decoder_rework.
3. Raise the MCTS singleton penalty: any cell where
   `n_distinct == 1` and `n_simulations >= 1000` should be flagged in
   `cell.warnings` as `mcts_singleton_trap` so integrate.py can refuse
   to mark it MEASURED.
4. Update paper §3.3 MetalGeometryPrior section to note that the
   Pt_0 acetylide seed (`[Pt]C#C`) is intentionally open at the click
   handle but currently re-collapses at n_simulations=1000 on the
   Lambda-only path (Fix-3 follow-up).
5. Investigate whether `n_simulations >= 1000` is enough — WF-MCTS-Synth
   recommends n_simulations >= 5000 with `virtual_loss` enabled for
   non-degenerate Lambda MCTS; the current pilot at n_sim=1000 may be
   inside the regime where the click-rules reward cannot overcome the
   prior-bias on the seed.

## 10. Artefacts

- `molmetal/reports/wf_lambda1_round12_pathb_r4c/report.json` (per-cell + aggregate)
- `molmetal/reports/wf_lambda1_round12_pathb_r4c/summary.md` (per-cell table)
- `molmetal/reports/wf_round12_lambda_pathb/final.md` (this file)
- `molmetal/scripts/r4_lambda_only_run.py` (3 edits: argparse entry, `run_one_cell` signature, `run_sweep` propagation, `main` forward)

Sources of truth:
- Round-12 Lambda Pilot baseline: `molmetal/reports/wf_lambda_metal_pilot/final.md`
- Path B reference docs:
  - `molmetal/molmetal_lam/lam_chem/decoder_rework.py` (chem-aware soft bond prior — CFM path only)
  - `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (5×5 Pt-click compat matrix — scaffold-aware gate)
  - `molmetal/scripts/r4_lambda_only_run.py:1705-1711` (auto-alias set),
    `:1716-1720` (pt_click_compat import).
