# Wf-355-A3 — R20 §5 tmQM + PlatinAI 4-cell joint retrain wrapper (USER_OK-gated)

**Date**: 2026-09-29 (Wf-355-A3 ultracode ship per Wf-354-B3 NOT-WRITTEN finding)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚀 **SHIPPED (wrapper only; USER_OK-gated, STATUS_HOLD on R² architectural floor)**
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED tag discipline) + `mol-metal-gpu-execution` §0 (kill_then_adapt standing directive) + `mol-metal-multiph-pll` (Wf-348-B1 32-bin Fourier + 128-d PLL augmentation)
**Parent**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 (Phase 4 joint retrain)
**Cross-refs**:
- `TODO/completed/339d2_lambda_roadmap_audit.md` §4.4 (Phase 4 NOT LAUNCHED — Wf-355-A3 closure)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (STATUS_HOLD on 12-d F5_RDKIT_2D signal floor)
- `TODO/completed/96_r20_p5_closure_catherine.md` (R20-P5 sister scope SHIPPED — sister scope is the closure theorem + Catherine 6/6 PASS; this Wf-355-A3 wrapper covers R20 §5 joint retrain)

---

## §0 Why this wrapper was created

Per Wf-354-B3 finding (2026-09-29): R20 §5 tmQM + PlatinAI 4-cell joint retrain wrapper was **NOT-WRITTEN**. STATUS_HOLD per LAMBDA_ROADMAP Phase 4 + Wf-316A on 12-d F5_RDKIT_2D signal floor. This wrapper ships the launch surface but actual GPU launch is BLOCKED on (a) USER_OK=1 explicit user authorization per AGENTS.md §5.1 + (b) empirical R² lift verification post Wf-348-B1 architectural fix (32-bin Fourier + 128-d PLL).

---

## §1 What was delivered

### §1.1 Launch wrapper

| Item | Detail |
|---|---|
| **Path** | `/home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` |
| **LOC** | **445 LOC** (bash) — exceeds ~300 LOC target; depth of pre-launch checks + per-cell/per-metal W₂ output paths + Wf-348-B1 augmentation importable check drove the size |
| **Shebang** | `#!/usr/bin/env bash` + `set -euo pipefail` |
| **USER_OK gate** | HARD-REQUIRED BEFORE any CPU work (per AGENTS.md §5.1 + mol-metal-gpu-execution §0); exits 64 without env var |
| **Pre-launch checks** | per_wave_checklist.py (Wf-262 §10.1) + pre_flight_check.py (TODO-207 v2 62/62 PASS) + wave_budget_tracker.py (Wf-262/291D) + tmQM d-block data on disk + PlatinAI 4-cell data on disk (4 cells × ≥50 rows each) + Multi-PH + PLL Wf-348-B1 augmentation importable check |
| **STATUS_HOLD warning** | Emitted BEFORE any GPU launch; cites Wf-316A + Wf-122 P3 + Wf-348-B1 |
| **Modes** | `--dry-run` (plan-only) + `--smoke-only` (2-seed × 50-epoch PROJECTED ~3 GPU-hr) + full 5-seed × 50-epoch (PROJECTED ~12 GPU-hr) |
| **Per-cell output** | `per_cell_w2.json` — MCF7 / A549 / A2780 / A2780cis W₂ pIC50 regression |
| **Per-metal output** | `per_metal_w2.json` — Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au W₂ pIC50 regression |
| **Per-wave output** | `per_wave_convergence.json` — cross-dataset R² convergence trace |
| **Plan record** | `molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json` — emitted on every invocation (dry-run + actual) for MEASURED/PROJECTED label discipline (party-twenty-spirit §3.5) |
| **CLI** | `--mode joint_tmqm_platinai --tmqm-csv ... --platinai-csv ... --cells MCF7 A549 A2780 A2780cis --tmqm-metals Cr Mn Cu Zn Ru Rh Pd Ag Pt Ir Au --seeds 0 1 2 3 4 --epochs 50 --hidden 300 --depth 3 --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 --w2-weight 0.05 --w2-gradient-mode sort --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs` |
| **GPU target** | `cuda:0` (RTX PRO 6000 Blackwell 98GB GPU-0) |
| **Speedup rationale** | Wf-328-SYN MEASURED joint speedup ~3-4× end-to-end step throughput (compound multiplicative on torch.compile 2.72× + W₂ sort 2-3× + cache 16.83× featurization + TF32 1.5× + num_workers <5%); PROJECTED joint ceiling ~10-15× |

### §1.2 4 NEW tests (`molmetal/tests/test_wf355a3_r20_p5_tmqm_wrapper.py`)

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_user_ok_gate_enforced` | USER_OK gate refuses with exit code 64 + FATAL marker + usage hint (per kill_then_adapt §0) |
| 2 | `test_bash_syntax_check` | `bash -n` parses cleanly on the 445-LOC wrapper (no syntax errors) |
| 3 | `test_preflight_perwave_data_quality_wired` | pre_flight_check.py + per_wave_checklist.py + wave_budget_tracker.py + tmQM d-block + PlatinAI 4-cell + Wf-348-B1 Multi-PH + PLL augmentation are all wired in wrapper source |
| 4 | `test_status_hold_warning_emitted_in_dryrun` | STATUS_HOLD warning + 12-d RDKit 2D signal floor + Wf-316A + Wf-122 P3 + Wf-348-B1 architectural-fix path all visible in dry-run output + plan.json |

**Test result**: **4/4 PASS, 0 FAIL, 0 SKIP** in 3.61 s wall (`PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf355a3_r20_p5_tmqm_wrapper.py -v`).

### §1.3 Dry-run verification (CPU-only)

- **Without USER_OK**: `bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run` → exit **64** + FATAL message + usage hint ✓
- **With USER_OK=1**: `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run` → exit 0 + plan.json emitted + STATUS_HOLD warning + dry-run invocation record ✓

---

## §2 Honest framing per `party-twenty-spirit` §3.5

### §2.2.1 MEASURED (today, on disk)

- Wrapper script exists at `/home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (445 LOC)
- USER_OK gate HARD-REQUIRED (exit 64 without env var)
- Pre-flight + per-wave + wave budget + data quality + Multi-PH + PLL augmentation importable check ALL wired in wrapper
- STATUS_HOLD warning (Wf-316A + Wf-122 P3 + Wf-348-B1 architectural-fix) emitted in dry-run output + plan.json
- 4/4 NEW tests PASS in 3.61 s wall (CPU-only)
- tmQM d-block data on disk: `/home/hugo/codes/mol-metal/.data/molmetal/tmQM/tmqm_parsed.csv` + `tmQM_X1/X2/X3.{xyz,BO}.gz` + `tmQM_y.csv`
- PlatinAI 4-cell data on disk: `/home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv` (4078 rows; 4 cells per `platinai_4cell.meta.json` cell_lines = MCF7/A549/A2780/A2780cis)
- Multi-PH + PLL augmentation module SHIPPED: `molmetal/molmetal_lam/reward/multiph_pll_feature.py` + `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` (Wf-348-B1 reference)
- Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (NO new gates added by Wf-355-A3)
- EXPECTED_GATE_COUNTS["all"] = 62 PRESERVED

### §2.2.2 PROJECTED (wall-time + outcome)

- **5-seed × 50-epoch full sweep**: ~12 GPU-hr on RTX PRO 6000 Blackwell sm_120 (per Wf-328-SYN MEASURED ~3-4× speedup; PROJECTED ceiling ~10-15×)
- **2-seed × 50-epoch smoke subset** (--smoke-only): ~3 GPU-hr
- **Cross-dataset R² target = +0.70** per R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5.3 (current macro R² = +0.1248 MEASURED F5_RDKIT_2D Ridge baseline; PROJECTED chemprop v2 hidden=300 depth=3 + Wf-348-B1 32-bin Fourier + 128-d PLL augmentation may lift toward +0.70 target)
- **Per-cell W₂ pIC50 regression output**: MCF7 / A549 / A2780 / A2780cis (4 cells)
- **Per-metal W₂ pIC50 regression output**: Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au (11 metals)
- **Per-wave convergence output**: cross-dataset R² convergence trace per wave

### §2.2.3 STATUS_HOLD

- **12-d RDKit 2D signal floor caps R² at ~0.004** per Wf-316A + Wf-122 P3 R² = -5.66 ± 1.36 MEASURED on PlatinAI 4-cell
- **W₂ regularization alone CANNOT lift R² above this ceiling** — the architectural fix path is **Wf-348-B1 32-bin Fourier + 128-d PLL augmentation** (SHIPPED in `molmetal/molmetal_lam/reward/multiph_pll_feature.py`)
- **Wrapper SHIPS but launch is BLOCKED on empirical R² lift verification post-Wf-348-B1** — verify that the augmentation actually lifts R² from -5.66 toward +0.30 gate before authorizing the full 12 GPU-hr sweep
- **Wf-129 P3 Multi-PH + PLL guarded retrain** yielded R² = -0.0780 (better than -5.66 but still below +0.30 gate); Wf-348-B1 augmentation is the next escalation step

### §2.2.4 kill_then_adapt directive

- Wrapper wall = 12 GPU-hr PROJECTED (full sweep) + 3 GPU-hr PROJECTED (smoke subset); exceeds the 1 hr GPU launch threshold per AGENTS.md §5.1 + mol-metal-gpu-execution §0
- **USER_OK gate HARD-REQUIRED** — wrapper exits 64 without env var, even on --dry-run (per Wf-322F §4 reproducibility audit invariant)
- Standing directive: "显存够就放心大胆用" (use freely as long as VRAM is enough) — RTX PRO 6000 Blackwell 98GB has ample VRAM for chemprop v2 hidden=300 depth=3 + Wf-348-B1 augmentation

### §2.2.5 NOT TOUCHED

- `paper/*` — NOT TOUCHED per TODO-84 ACTIVE rule (AGENTS.md §5.3)
- `metrics/by_metric/*.json` — NOT TOUCHED (no GPU launch performed; this is wrapper + tests + docs)
- Pre-flight v3 62/62 PASS baseline — NOT RE-RUN (CPU-only docs + tests work; no need to re-validate at Wf-328-SYN baseline)

---

## §3 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 (Phase 4 joint retrain spec)
- `TODO/completed/339d2_lambda_roadmap_audit.md` §4.4 (Phase 4 NOT-LAUNCHED closing)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (STATUS_HOLD on 12-d signal floor)
- `TODO/completed/96_r20_p5_closure_catherine.md` (R20-P5 sister scope SHIPPED)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (architectural ceiling honest negative closeout)
- `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (Wf-129 P3 Multi-PH + PLL guarded retrain R² = -0.0780)
- `TODO/completed/328_synthesis.md` (Wf-328-SYN MEASURED ~3-4× joint speedup)
- `TODO/completed/310a_wf124_p3_m1m2_partial.md` (Wf-310A G5 15/15 partial pair)
- `molmetal/molmetal_lam/reward/multiph_pll_feature.py` (Wf-348-B1 32-bin Fourier + 128-d PLL augmentation SHIPPED)
- `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` (Multi-PH + PLL integration SHIPPED)
- `molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json` (plan record on disk)
- `molmetal/scripts/per_wave_checklist.py` (Wf-262 §10.1 per-wave sub-checklist)
- `molmetal/scripts/pre_flight_check.py` (TODO-207 v2 62/62 PASS gate)
- `molmetal/scripts/wave_budget_tracker.py` (Wf-262/291D wave budget + --no-enforce-budget)
- `.claude/skills/mol-metal-gpu-execution/SKILL.md` §0 (kill_then_adapt standing directive)
- `.claude/skills/mol-metal-multiph-pll/SKILL.md` (R20-P3 Multi-PH + PLL augmentation)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 (MEASURED vs PROJECTED tag discipline)

---

## §4 What still needs to happen for actual R20 §5 joint retrain

Per the honest framing §2.2.3 STATUS_HOLD + the kill_then_adapt §2.2.4 directive, the actual launch is BLOCKED on:

1. **USER_OK=1 explicit user authorization** per AGENTS.md §5.1 (or pre-existing 2026-09-29 "显存够就放心大胆用" grant, if user judges Wf-348-B1 augmentation has empirically lifted the R² floor)
2. **Empirical R² lift verification post-Wf-348-B1**: Run a 2-seed × 50-epoch smoke (`--smoke-only` mode, PROJECTED ~3 GPU-hr) and verify that the cross-dataset R² lifts from -5.66 (Wf-122 P3 baseline) toward +0.30 (R20-P2 gate). If R² < 0.0 on 2+ (cell, seed) tuples, **HALT** and emit honest-negative close-out (binding regression alarm per stop semantics)
3. **If smoke passes**: authorize the 5-seed × 50-epoch full sweep (PROJECTED ~12 GPU-hr) and emit `TODO/completed/355a3_r20_p5_tmqm_ground.md` with MEASURED cross-dataset R² + per-cell + per-metal W₂ outputs
4. **Paper §4.5.6 honest-framing decision**: per TODO-84 + Wf-339-D2 §3.6, paper §4.5.6 may be updated with the MEASURED R² + honest-negative framing for any cell/metal that did not lift above gate

---

## §5 Standing rules honored

1. **MEASURED vs PROJECTED tag discipline** (`party-twenty-spirit` §3.5): every numeric value cites JSON/report path on disk; STATUS_HOLD explicitly flagged.
2. **"Kill then adapt" GPU directive** (mol-metal-gpu-execution §0): USER_OK gate HARD-REQUIRED; wrapper exits 64 without env var; no GPU launches performed.
3. **No paper work without authorization** (TODO-84 ACTIVE): zero touches to `paper/*`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207): pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (not re-run; CPU-only docs + tests + wrapper script).
6. **Honest-negative framing** (`party-twenty-spirit` §3.5): STATUS_HOLD cited with full Wf-316A + Wf-122 P3 + Wf-348-B1 architectural-fix context.

---

**Honest framing disclaimer (`party-twenty-spirit` §3.5)**: Wrapper SHIPS but full sweep launch is BLOCKED on empirical R² lift verification post Wf-348-B1 architectural fix. Do NOT cite cross-dataset R² ≥ +0.70 as MEASURED until the actual GPU sweep completes. The wrapper preserves the kill_then_adapt USER_OK gate as a reproducibility audit invariant even for CPU-only --dry-run invocations (per Wf-322F §4 pattern).
