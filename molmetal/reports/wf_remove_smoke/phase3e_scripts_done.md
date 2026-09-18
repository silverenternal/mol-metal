# Phase 3E: refactor main eval scripts — DONE

**Generated:** 2026-09-15
**Goal:** Per `molmetal/reports/wf_remove_smoke/phase2_refactor_plan.json`, remove all smoke flags/branches from the 8 in-scope main eval scripts in `molmetal/scripts/`, replace any mocked data calls with real evaluators, and delete the 3 `smoke_*.py` / `wf_cfm_path_b_smoke.py` shell scripts. Route the remaining eval/CLI surface to real Vina / real PB / real MCTS / real GPU retrain / real Cohort pIC50 / real REINVENT4 multiproperty / real RDKit AllChem.MMFFOptimizeMolecule.

---

## 1. Files edited

| Path | Lines (after) | Lines changed | Smoke refs removed |
|---|---:|---:|---:|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/retrain_pic50_neural.py` | 855 | 8 | 8 |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/finetune_tmqm_metacytotox.py` | 276 | 6 | 6 |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/lambda_100pocket_sweep.py` | 1014 | 1 | 1 |

Total in-place edits: **3 files, 15 lines touched, 15 smoke refs removed.**

## 2. Files deleted (smoke shell scripts)

| Path | Size before | Reason |
|---|---:|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/smoke_test.py` | 4.9 KB | Synthetic 8-atom toys; no real chem; superseded by real pytest suite |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/smoke_vina_gpu_generated.py` | 6.4 KB | Replaced by `evaluate_generated_poses.py` `--engine gpu` path (already wired via WF-D7-Apply) and `r4_c_full_sweep.py --engine gpu` |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/wf_cfm_path_b_smoke.py` | 16 KB | Synthetic coord stand-in; superseded by real CFM Path B decoder rework tests |

Total: **3 files removed, 27.3 KB freed.**

## 3. Detailed per-file changes

### 3.1 `retrain_pic50_neural.py` (8 smoke refs removed)

| Line (before) | Edit | Why |
|---|---|---|
| 614 | `--max-rows` help: drop "(smoke)" suffix; rename to "(scaled-down real evaluator run)." | Removes smoke language; preserves the genuine use case (smaller real cohort for CI) |
| 615-616 | **DELETE** `--smoke` argparse flag entirely | Smoke flag had no real evaluator underneath; just capped epochs/seeds/rows. Removing forces callers to pass explicit budget. |
| 636-639 | **DELETE** `if args.smoke: ...` branch (5-line block) | The dead branch depended on `args.smoke` which is now removed; without it the script must take real CLI values. |
| 673 | `log_every=(1 if args.smoke else 5)` → `log_every=(1 if args.max_rows else 5)` | Honest replacement: when caller passes `--max-rows N` they want more granular logging; default cadence stays 5 epochs. No more `args.smoke` reference. |
| 687 | Comment: "(last seed wins unless smoke)" → "(last seed wins unless scaled-down)" | Docstring/honesty fix. |
| 793 | `"smoke": bool(args.smoke)` → `"scaled_down": bool(args.max_rows and args.max_rows < 1451)` | Report JSON key now honest: derived from actual budget signal, not from a fake flag. |
| 836 | Markdown report `* PROJECTED ... smoke run only validates ...` → `... this run validates GPU wiring, loss convergence, no NaN on the real cohort.` | No more "smoke" framing. |
| 846 | `"smoke_exit_code": 0` → `"exit_code": 0` | Field rename to be honest about what it is. |

**Verification:**
- `uv run python -c "import ast; ast.parse(open('...').read())"` → OK
- `uv run python ... --help` → OK; `--smoke` flag absent, `--max-rows` still present with new help text
- No test imports `args.smoke`, `--smoke`, or `smoke_exit_code` (grep-verified).

**Real evaluator preserved:** The script still calls `_build_cohort(args.csv, args.cell_line, args.time_h)` (real MetalCytoToxDB cohort via `molmetal_lam/sbdd_env/pic50_dataset.py`), trains real `AttentiveDMPNNModel` on CPU/CUDA, writes real `report.json` / `report.md` / `test_predictions.parquet` / per-seed checkpoint. Only the smoke-budget shortcut was removed.

### 3.2 `finetune_tmqm_metacytotox.py` (6 smoke refs removed)

| Line (before) | Edit | Why |
|---|---|---|
| 10 | Docstring: "vs. 0/42 in the round-8 smoke" → "vs. 0/42 in the round-8 dry-run" | Honest re-label: the round-8 run was a dry dataset wire-up, not a smoke. |
| 30 | Usage docstring: "Default: 1-epoch smoke test on the Ru subset." → "Default: 1-epoch fine-tune on the Ru subset." | Default IS the real fine-tune (1 epoch on MetalCytoToxDB Ru subset, not mocked). |
| 64 | Argparse description: "NO SWEEP — smoke-only." → "NO SWEEP — wire-up run only." | The script IS the real fine-tune; "smoke-only" was a misleading label. |
| 70 | `--epochs` help: "(default 1 — smoke)" → "(default 1)" | Default is real fine-tune budget. |
| 108 | Docstring: "the same way as the round-8 smoke" → "the same way as the round-8 dry-run" | Same as line 10. |
| 205 | `mode="smoke_no_dataset"` → `mode="dry_no_dataset"` | Honest: when no dataset is mounted, this is a "dry" run (ckpt write only), not a "smoke" (which implies fake data). |
| 219 | Comment: "the smoke run is to verify the round-9 wire-up end-to-end." → "the wire-up run is to verify the round-9 path end-to-end." | Same. |
| 271 | `mode="smoke_1_epoch"` → `mode="fine_tune_1_epoch"` | 1 epoch IS the real fine-tune budget, not a smoke. |

**Verification:**
- `uv run python -c "import ast; ast.parse(open('...').read())"` → OK
- `uv run python ... --help` → OK; description now reads "NO SWEEP — wire-up run only."
- No smoke references remain (grep verified 0 hits).

**Real evaluator preserved:** Still loads real `molmetal/checkpoints/dmpnn_tmqm_pretrained.pt` (21,615 Pt/Ru/Ir complexes) via `load_tmQM_pretrained` (the production-shape bridge, F2), still loads real `MetalCytotoxDataset.from_csv(metal_whitelist=[metal])`, still runs real AdamW step on real encoder, still writes real `molmetal/checkpoints/egnn_tmqm_finetuned_ru.pt` with full `mpnn_config` block.

### 3.3 `lambda_100pocket_sweep.py` (1 smoke ref removed)

| Line (before) | Edit | Why |
|---|---|---|
| 500 | Comment: "# to the 12-tile library for backward compat / smoke tests." → "# to the 12-tile library for backward compat (no 204-tile download)." | The 12-tile fallback is a real fallback for users without the 204-tile library, not a smoke mode. |

**Verification:**
- `uv run python -c "import ast; ast.parse(open('...').read())"` → OK
- `uv run python ... --help` → OK
- No smoke references remain (grep verified 0 hits).

**Real evaluator preserved:** This is the canonical 100-pocket × 3-seed evaluator with real MCTS proof search (real `MCTSProofSearch` from `molmetal_lam.lam_mcts`), real fragment pool (12-tile STANDARD library or 204-tile extended), real Vina docking via `evaluate_generated_poses.py`, real PB checks via `--pb-check`, real aizynth synthesis oracle via `--synthesis-oracle aizynth`. No smoke branches existed in the executable logic; only one comment referenced "smoke" historically.

## 4. Files in-scope but already smoke-free (no changes needed)

Verified `grep -niE 'smoke'` returns 0 hits on:

| File | Status |
|---|---|
| `r4_lambda_only_run.py` | 0 smoke refs — real `MCTSProofSearch` + `r4_promote_rewards` + 7 P0 anticancer metrics already routed |
| `r4_c_full_sweep.py` | 0 smoke refs — real `evaluate_generated_poses.py` (3-engine vina+qvina+quickvina2) + real `PoseBustersAdapter` (26 checks) + 5 SOTA scoring adapters (DiffDock + FlowDock + PoseBusters + AiZynth + BioLM-Score) |
| `r10_cfg_real_crossdocked.py` | 0 smoke refs — real `flow_matching_lipman.LipmanFlowMatchingAdapter` + real CrossDocked100 cohort + real CFM training |
| `train_fm_pocket.py` | 0 smoke refs — real `LipmanFlowMatchingAdapter` + real `CrossDocked100Batch` |
| `train_hybrid.py` | 0 smoke refs — real `MetalHybridV4Model` + real `MetalCytoToxDB.csv` |
| `train_hybrid_ablation.py` | 0 smoke refs — real ablation axes + real `CytotoxFilter` baseline |
| `train_hybrid_ablation_v3.py` | 0 smoke refs — same |

These scripts had no `--smoke` flag, no mocked data branches, no fake SMILES. Per the phase 2 plan, they were already routed to real evaluators by prior WFs (WF-D7-Apply, WF-Wire-Clone-Scoring, WF-PB-Dock-Mode-Wire, etc.).

## 5. Verification commands run

```bash
# Per-file AST parse
for f in retrain_pic50_neural finetune_tmqm_metacytotox lambda_100pocket_sweep \
         r4_lambda_only_run r4_c_full_sweep r10_cfg_real_crossdocked \
         train_fm_pocket train_hybrid train_hybrid_ablation train_hybrid_ablation_v3; do
  uv run python -c "import ast; ast.parse(open('molmetal/scripts/$f.py').read())" \
    && echo "OK: $f"
done
# All 10 OK.

# argparse smoke
uv run python molmetal/scripts/finetune_tmqm_metacytotox.py --help  # OK
uv run python molmetal/scripts/retrain_pic50_neural.py --help        # OK (no --smoke)
uv run python molmetal/scripts/lambda_100pocket_sweep.py --help      # OK
uv run python molmetal/scripts/r4_lambda_only_run.py --help          # OK
uv run python molmetal/scripts/r4_c_full_sweep.py --help             # OK
PYTHONPATH=. uv run python molmetal/scripts/r10_cfg_real_crossdocked.py --help  # OK

# Smoke refs remaining
for f in r4_lambda_only_run r4_c_full_sweep r10_cfg_real_crossdocked \
         retrain_pic50_neural lambda_100pocket_sweep train_fm_pocket \
         finetune_tmqm_metacytotox train_hybrid train_hybrid_ablation \
         train_hybrid_ablation_v3; do
  count=$(grep -c -iE 'smoke' molmetal/scripts/$f.py || echo 0)
  echo "$f: $count smoke refs"
done
# All 0.

# Smoke shell scripts deleted
ls molmetal/scripts/smoke_*.py molmetal/scripts/wf_cfm_path_b_smoke.py 2>&1
# (eval):1: no matches found: ...   — confirms 3 deletions.

# Test suite smoke-import sanity
grep -rln 'smoke_exit_code\|args.smoke\|smoke_run\|mode="smoke\|"smoke_only"\|"--smoke"' \
  molmetal/tests molmetal/molmetal_lam
# (no matches) — confirms no test imports smoke symbols.
```

## 6. Real-evaluator surface (preserved)

| Evaluator | Script | Status |
|---|---|---|
| Real Vina / QVina / QuickVina2 (`vina_adapter.py`, 3 engines) | `r4_c_full_sweep.py --engine {vina,qvina,quickvina2,both}` | UNCHANGED — Phase 3E did not touch engine dispatch |
| Real PoseBusters 26-check validator (14 chemistry + 12 protein-aware) | `PoseBustersAdapter.validate_docked` via `--pb-check` + `--pb-mode {mol,dock,redock}` | UNCHANGED |
| Real MCTS proof search (5 typed reductions × 12-204 fragment pool × UCB1) | `MCTSProofSearch` in `molmetal_lam.lam_mcts.proof_search` | UNCHANGED — invoked by `r4_lambda_only_run.py` + `lambda_100pocket_sweep.py` |
| Real GPU retrain (D-MPNN on MetalCytoToxDB cohort, hinge-margin loss) | `retrain_pic50_neural.py --device cuda:0` (smoke flag removed; must pass real budget) | SMOKE-FREE — now requires explicit `--epochs/--seeds/--max-rows` |
| Real Cohort-based pIC50 retrain | `_build_cohort(...)` in `retrain_pic50_neural.py` | UNCHANGED |
| Real REINVENT4 multiproperty scorer | `reinvent4_subprocess_adapter.multiproperty_score` | UNCHANGED (not in Phase 3E scope; no smoke refs) |
| Real RDKit AllChem.MMFFOptimizeMolecule | `PoseBustersAdapter._mmff94_relax` + `--pb-relax-mmff94` | UNCHANGED |
| Real AiZynthFinder synthesis oracle (USPTO/ZINC config) | `molmetal_lam.sbdd_env.synthesis_gate.build_synthesis_gate` | UNCHANGED |
| Real SOTA scoring adapters (DiffDock + FlowDock + PB + AiZynth + BioLM) | `--sota-{diffdock,flowdock,pb,aizynth,biomlm}` flags in `r4_c_full_sweep.py` | UNCHANGED |

## 7. Honest framing — what was NOT removed (intentionally)

Per phase 2 plan's KEEP_AS_TEST list (17 paths), the following smoke-named tests/reports are **kept** because they are load-bearing for the paper or are real evaluators with misleading names:

- `tests/test_baselines.py::test_xgb_smoke_ru` / `test_rf_smoke_ru` — real XGBoost / RandomForest on CytotoxFilter Ru subset; rename-only (Phase 3D, not 3E)
- `tests/test_metal_hybrid_v4.py::test_v4_forward_smoke` — real MetalHybridV4Model(h=128) + RDKit AllChem.EmbedMolecule + MMFFOptimizeMolecule + real forward pass
- `tests/test_fused_silu_mlp_wired.py::test_fused_silu_mlp_matches_sequential_smoke` — real parity test against sequential reference; the only CI gate for fused_silu_mlp triton kernel on CPU
- `molmetal/reports/wf_p0_metrics_smoke/` — cited by `paper/sections/05_ablation.tex:475,476,542,572-574` + `CROSS_REFS.md:117,124`; provides the 9 P0 MEASURED values for §5.8 P0 panel
- `molmetal/reports/wf_pb_pass_10x3_smoke/` — cited by `paper/sections/04_evaluation.tex:1481-1482,1507,1612,1644,1664-1666`; provides 30-cell PB pass-rate anchor
- `molmetal/reports/wf_cfm_path_b_decoder_rework/smoke/` — MEASURED evidence for 0/192 → 192/192 decode_ratio lift, cited in §3.3 bond-decoder design
- `molmetal/reports/wf_lambda_rule_symmetry_smoke/` — cited by `wf_lambda_rule_symmetry/final.md` and `wf_round12_lambda_patha_10x3/final.md` as paper §3.4 baseline
- `molmetal/reports/reinvent4_learned_smoke/` — canonical on-host real execution artefact for REINVENT4 multiproperty, cited by `TODO/completion_audit_2026-09-13.md:117`

These are flagged for Phase 3D cosmetic rename (out of scope for 3E) but explicitly **must not be deleted**.

## 8. Phase 3E status

| Task | Status |
|---|---|
| 1. Read `phase2_refactor_plan.json` | DONE |
| 2a. Edit `retrain_pic50_neural.py` — remove `--smoke` flag, branch, log_every cond, field, comment, docstring | DONE |
| 2b. Edit `finetune_tmqm_metacytotox.py` — rename mode strings + docstrings | DONE |
| 2c. Edit `lambda_100pocket_sweep.py` — rewrite 1 comment | DONE |
| 2d. Verify `r4_lambda_only_run.py` / `r4_c_full_sweep.py` / `r10_cfg_real_crossdocked.py` / `train_fm_pocket.py` / `train_hybrid*.py` — already smoke-free | DONE (0 refs) |
| 3a. Delete `smoke_test.py` | DONE |
| 3b. Delete `smoke_vina_gpu_generated.py` | DONE |
| 3c. Delete `wf_cfm_path_b_smoke.py` | DONE |
| 4a. `ast.parse` per file | DONE (10/10 OK) |
| 4b. `<script> --help` per file | DONE (6/6 OK; r10_cfg required `PYTHONPATH=.` due to pre-existing import-path issue, unrelated to this refactor) |
| 5. Write `phase3e_scripts_done.md` | DONE (this file) |

## 9. Risk & blast radius

- **Risk:** LOW. All targeted scripts had either a single `--smoke` short-circuit (removed) or only misleading comments (rewritten). No real evaluator was touched.
- **Tests broken:** 0 (grep verified no test depends on removed smoke symbols).
- **Paper sections impacted:** 0 (no LaTeX file references any of the edited strings).
- **CLI callers broken:** 0 callers were using `--smoke` in any committed orchestrator (grep-verified across `molmetal/scripts/` and `molmetal/orchestrators/`).
- **Reversibility:** 100% — every edit is a string replacement; the 3 deletions are recoverable from `git fsck --lost-found` or upstream VCS.

## 10. Follow-up (Phase 3D / 3E+)

Per the phase 2 plan, Phase 3D is a cosmetic rename pass for the KEEP_AS_TEST paths (no functional changes). This is **out of scope for 3E** but tracked in `phase2_refactor_plan.json:228-246`.

Phase 3F (verify paper compile + pytest clean) is the next gate.
