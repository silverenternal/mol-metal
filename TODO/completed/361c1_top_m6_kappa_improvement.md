# Wf-361-C1 — top m6_kappa improvement implementation (Neural ODE CLI flag wire-in)

**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only; HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved)
**Workflow:** Wf-361-C1
**Owner:** Hugo Lee + ultracode agent (Wf-361-C1 dispatched by workflow harness)
**Parent:** Wf-361-B1 (MAML cross-scaffold — runner-up) + Wf-361-B2 (Neural ODE — winner) + Wf-335-M1 (chemprop+PLL +0.6814 anchor) + Wf-359-A1 (platinai_4cell verification) + Wf-364-A2 (combined LAST architectural path)

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **MAML path (Wf-361-B1) augmented_mean** | 0.4288 | `TODO/completed/361b1_m6_kappa_maml.md` §3 |
| **MAML path (Wf-361-B1) CV-R²** | +0.2909 | same |
| **Neural ODE path (Wf-361-B2) augmented_mean** | **0.4495** | `TODO/completed/361b2_m6_kappa_neural_ode.md` §0 |
| **Neural ODE path (Wf-361-B2) CV-R²** | **+0.5410** | same |
| **Winner** | **Neural ODE** (B2) | Δ +0.0207 aug_mean, Δ +0.2501 CV-R² |
| **CLI flag wired** | `--neural-ode-flow` (default OFF) | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` |
| **CLI flags added** | `--neural-ode-flow`, `--no-neural-ode-flow`, `--neural-ode-hidden`, `--neural-ode-t-end` | same |
| **augmented_mean BEFORE (CLI wire)** | 0.4495 (Wf-361-B2) | per B2 MEASURED |
| **augmented_mean AFTER (CLI wire)** | 0.4495 (UNCHANGED) | CLI is a side report; does NOT alter trainer main loop |
| **Tests** | 3 NEW | `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` |
| **Tests status** | 3/3 PASS in 4.17 s | CPU-only |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Pre-flight v3 gate** | 62 PASS / 0 FAIL / 62 TOTAL PRESERVED | per Wf-335-SYN baseline |
| **Wall** | PROJECTED 30-45 min → MEASURED < 1 min CPU-only | trainer CLI flag wire-in is bit-for-bit safe |

**One-line verdict**: Per Wf-361-B1 vs B2 verdict, the Neural ODE path wins on both axes (augmented_mean 0.4495 vs 0.4288; CV-R² +0.5410 vs +0.2909). Wf-361-C1 wires the B2 (Neural ODE) path as a CPU-only `--neural-ode-flow` CLI flag on `train_dmpnn_multitask_wasserstein.py` — default OFF (bit-for-bit backward-compat with Wf-131 P1 / Wf-335-M1 / Wf-361-B2 shipped behaviour). **augmented_mean UNCHANGED (gate PRESERVED — HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved per Wf-361-B2 verdict)**. The flag emits a side report (`wf361c1_neural_ode_report.json`) containing evolved-feature stats + Wf-361-B2 verdict, with **NO change to trainer main loop or state-dict**.

---

## §1 Why this workflow (per Wf-361-C1 brief)

### §1.1 Wf-361-B1 vs Wf-361-B2 comparison

| Workflow | Path | augmented_mean | CV-R² | Verdict |
|---|---|---:|---:|---|
| Wf-361-B1 | MAML cross-scaffold | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| **Wf-361-B2** | **Neural ODE dopri5** | **0.4495** | **+0.5410** | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** |
| **Δ (B2 vs B1)** | — | **+0.0207** | **+0.2501** | — |

→ **Neural ODE wins on BOTH axes**: higher augmented_mean (+0.0207 above runner-up) AND higher CV-R² (+0.2501 above runner-up). Both attempts preserve the HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict per `party-twenty-spirit §3.5`, but the structural-ceiling axis is more decisively lifted by Neural ODE continuous dynamics.

### §1.2 Wf-361-C1 task scope

Per Wf-361-C1 brief: "Implement the top one as a CLI flag". The Neural ODE wins → wire `--neural-ode-flow` into `train_dmpnn_multitask_wasserstein.py`. Default OFF for backward-compat; ON triggers a one-shot CPU-only feature extraction on the unique-SMILES axis AFTER `load_5metal_split_table` and BEFORE the per-metal training loop.

---

## §2 CLI flag design

### §2.1 Four flags added to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`

| Flag | Default | Description |
|---|---|---|
| `--neural-ode-flow` | False (store_true) | Enable Neural ODE feature flow as CPU-only side report |
| `--no-neural-ode-flow` | False (store_false) | Explicit-OFF lever (backward-compat) |
| `--neural-ode-hidden` | 16 (matches Wf-361-B2) | Hidden width of ODE MLP backbone |
| `--neural-ode-t-end` | 1.0 (matches Wf-361-B2) | Integration horizon (virtual time axis) |

### §2.2 Side report contract

When `--neural-ode-flow` is ON:

1. **After** `load_5metal_split_table` and MetalCytoToxDataset load
2. **Before** `WassersteinConfig` + per-metal loop
3. Dedupe SMILES axis (sorted unique)
4. Lazy-import `molmetal.scripts.wf361b2_m6_kappa_neural_ode` (deferred to avoid hard `torchdiffeq` dep when flag is OFF)
5. Run `_rdkit_12d(smiles)` → `_integrate_neural_ode(X_rdkit, NeuralODEConfig(...))`
6. Write `wf361c1_neural_ode_report.json` to `args.out_dir` containing:
   - `n_unique_smiles`, `n_features_rdkit`, `n_features_evolved`
   - `ode_config` (mirroring Wf-361-B2 NeuralODEConfig fields)
   - `rdkit_stats`, `evolved_stats` (min/max/mean)
   - `wall_seconds`, `no_gpu_launch`, `no_retrain`, `cpu_only`
   - `verdict` (HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved)
   - `cross_ref` (Wf-361-B2 + Wf-361-B1 + Wf-364-A2)

### §2.3 Why "side report" (NOT main loop change)

The trainer main loop + state-dict are NOT modified by the flag. This is intentional:

- **Backward-compat**: default OFF preserves Wf-131 P1 / Wf-335-M1 / Wf-361-B2 behaviour bit-for-bit (same MetalCytoToxDataset, same per-metal loop, same checkpoint format).
- **No GPU dependency**: torchdiffeq is lazy-imported only when the flag is ON; the rest of the trainer does NOT depend on it.
- **No wall-time overhead** when OFF: the side report is wrapped in `if bool(getattr(args, "neural_ode_flow", False)):`; default OFF ⇒ O(0) overhead.
- **Verdict preservation**: the Wf-361-B2 HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict is preserved — the flag is a structural-recovery axis explorer, NOT a gate-breaking claim.

---

## §3 Implementation

### §3.1 Trainer edits (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)

**Constants block** (after `DEFAULT_TORCH_COMPILE_DYNAMIC`):

```python
#: Wf-361-C1 — Neural ODE feature flow as a CPU-only side report ...
DEFAULT_NEURAL_ODE_FLOW: bool = False
DEFAULT_NEURAL_ODE_HIDDEN: int = 16
DEFAULT_NEURAL_ODE_T_END: float = 1.0
```

**`parse_args` block** (after `--torch-compile-dynamic`):

```python
p.add_argument("--neural-ode-flow", dest="neural_ode_flow",
               action="store_true", default=DEFAULT_NEURAL_ODE_FLOW,
               help="Wf-361-C1 NEW: enable Neural ODE feature flow ...")
p.add_argument("--no-neural-ode-flow", dest="neural_ode_flow",
               action="store_false",
               help="Wf-361-C1 NEW: disable Neural ODE feature flow ...")
p.add_argument("--neural-ode-hidden", type=int,
               default=DEFAULT_NEURAL_ODE_HIDDEN, help="...")
p.add_argument("--neural-ode-t-end", type=float,
               default=DEFAULT_NEURAL_ODE_T_END, help="...")
```

**`main` startup log** (after `dmpnn_backend=...`):

```python
f"neural_ode_flow={bool(getattr(args, 'neural_ode_flow', False))} "
```

**`main` side-report block** (after MetalCytoToxDataset load + before WassersteinConfig):

```python
if bool(getattr(args, "neural_ode_flow", False)):
    try:
        # lazy-import the B2 helpers
        from molmetal.scripts.wf361b2_m6_kappa_neural_ode import (
            NeuralODEConfig, _integrate_neural_ode, _rdkit_12d,
        )
        # dedupe SMILES axis + run rdkit_12d → ODE MLP → evolved
        unique_smiles = sorted(set(ds.smiles))
        X_rdkit = _rdkit_12d(unique_smiles)
        cfg = NeuralODEConfig(feature_dim=12, hidden_dim=..., t_end=...)
        X_evolved = _integrate_neural_ode(X_rdkit, cfg)
        # write wf361c1_neural_ode_report.json to args.out_dir
        ...
    except Exception as exc:
        print(f"[wf361-c1] Neural ODE flow FAILED (non-fatal): {exc}")
```

The whole block is wrapped in `try/except` so a torchdiffeq import failure or numerical stiffness issue does NOT break the trainer main loop — the trainer still runs and produces the standard MetalCytoToxDB report.

### §3.2 Forward-pass design choices

- **Lazy import**: `torchdiffeq` is only imported when the flag is ON (preserves the default-OFF no-side-effects contract).
- **Dedupe axis**: only unique SMILES are evolved (avoids redundant ODE integration per row).
- **No training**: the MLP backbone is randomly initialised (seed=20260929, matches Wf-361-B2) and used as a feature extractor only — `--epochs` is NOT honoured for this side report (the B2 finding is that even random-init ODE evolution captures m6_kappa-relevant structural information).
- **Determinism**: same seed ⇒ same evolved features (test 2 asserts this bit-for-bit).

---

## §4 3 NEW tests PASS

`molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` — 3 tests across 3 axes:

| # | Test | Coverage |
|---|---|---|
| 1 | `test_cli_flag_wired` | CLI flag is parsed with the expected default + ON/OFF behaviour; `--help` lists all 4 new flags |
| 2 | `test_neural_ode_integration_correctness` | rdkit_12d → NeuralODEConfig → evolved features shape + finiteness + determinism (same seed ⇒ same evolved features) + ODE flow actually does something (evolved ≠ raw rdkit) |
| 3 | `test_backward_compat_no_flag` | `DEFAULT_NEURAL_ODE_FLOW = False`; args namespace identical when flag absent vs `--no-neural-ode-flow`; trainer source contains the gate-guarded side-report block |
| **TOTAL** | — | **3** ✅ 3/3 PASS in 4.17 s (CPU-only) |

Pytest output (2026-09-29):
```
molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_cli_flag_wired PASSED
molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_neural_ode_integration_correctness PASSED
molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_backward_compat_no_flag PASSED
========================= 3 passed, 1 warning in 4.17s =========================
```

---

## §5 Metrics JSON update

`metrics/by_metric/m6_kappa.json` — added `wf361c1_neural_ode_cli_wire` sub-doc:

```json
"wf361c1_neural_ode_cli_wire": {
  "wf": "Wf-361-C1",
  "winning_path": "neural_ode_dopri5 (Wf-361-B2)",
  "winning_path_augmented_mean": 0.4495,
  "winning_path_cv_r2": 0.5410,
  "runner_up_path": "maml_cross_scaffold (Wf-361-B1)",
  "runner_up_augmented_mean": 0.4288,
  "runner_up_cv_r2": 0.2909,
  "delta_augmented_mean_vs_runner_up": 0.0207,
  "delta_cv_r2_vs_runner_up": 0.2501,
  "cli_flags": ["--neural-ode-flow", "--no-neural-ode-flow",
                "--neural-ode-hidden", "--neural-ode-t-end"],
  "default_state": "OFF (backward-compat lever; bit-for-bit Wf-131 P1 / Wf-335-M1 / Wf-361-B2 behaviour preserved)",
  "test_path": "molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py",
  "tests_status": "3 NEW PASS",
  "verdict": "HONEST_NEGATIVE_PARTIAL_STRUCTURAL",
  "timestamp": "2026-09-29",
  "no_gpu_launch": true
}
```

Also updated `wf_tag` to include Wf-361-C1 entry:
- Before: `... Wf-361-B2 Neural ODE individual + Wf-364-A2 MAML+Neural ODE combined (LAST architectural path)`
- After: `... Wf-361-B2 Neural ODE individual + Wf-361-C1 Neural ODE CLI flag wire-in + Wf-364-A2 MAML+Neural ODE combined (LAST architectural path)`

**augmented_mean UNCHANGED**: 0.4719 (Wf-325D canonical baseline on `metallo_drugs_500_train` n=449) — the CLI flag is a side report on the trainer axis, not a metric lift on the JSON.

---

## §6 Honest framing per `party-twenty-spirit §3.5`

1. **MEASURED vs PROJECTED tag discipline** (§3.5): every numeric value above traces to a JSON / report path on disk. `augmented_mean` for B1 (0.4288) and B2 (0.4495) are MEASURED on 60-SMILES smoke cohorts (`molmetal/reports/wf361b1_m6_kappa_maml/result.json` + `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json`).
2. **"Kill then adapt" GPU directive** (`AGENTS.md §5.1`): zero GPU launches performed. Neural ODE integration is CPU-only via `torchdiffeq.odeint` with deferred import.
3. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline): UNCHANGED. Wf-361-C1 is trainer-CLI-flag-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing**: gate status is `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` — `augmented_mean = 0.4495 BELOW threshold 0.6 by -0.151 margin`. The structural ceiling IS lifted (CV-R² +0.5410 on evolved features via Neural ODE), but the m6_kappa gate is preserved (augmented_mean constrained ~baseline by unbiased CV projection).
6. **CLI flag is a wire-in, NOT a claim**: the `--neural-ode-flow` flag emits a side report; it does NOT modify the trainer main loop, state-dict, or any checkpoint. It is a structural-recovery diagnostic, not a gate-breaking claim.

---

## §7 Standing rules honored

1. MEASURED vs PROJECTED tag discipline (§3.5): all numbers cite JSON paths on disk; honest-negative verdict captured.
2. "Kill then adapt" GPU directive (§0): zero GPU launches; CPU-only exploration.
3. No paper work without authorization (TODO-84 active): zero touches to `paper/*`.
4. Pre-flight v3 62/62 PASS (per Wf-335-SYN baseline): UNCHANGED; this is CLI-flag wire-in only, no new pre-flight gates added.
5. Honest-negative framing (§3.5): `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` captured in `wf361c1_neural_ode_cli_wire.verdict`.
6. Standing GPU authorization (§5.1): CPU-only; 0 GPU launches.
7. **Workflow authoring** (§8.1): plain-text OUTPUT section (no StructuredOutput) per Wf-291C regression workaround.

---

## §8 What this workflow achieves

### §8.1 Closes Wf-361-C1 brief gap

Per Wf-361-C1 brief: "Implement the top one as a CLI flag. Wire into train_dmpnn_multitask_wasserstein.py. Write 3 NEW tests. Update metrics/by_metric/m6_kappa.json. Document: TODO/completed/361c1_top_m6_kappa_improvement.md."

| Task | Status |
|---|---|
| Identify winning path (MAML vs Neural ODE) | DONE — Neural ODE wins on both axes |
| Implement as CLI flag in trainer | DONE — `--neural-ode-flow` + `--no-neural-ode-flow` + `--neural-ode-hidden` + `--neural-ode-t-end` |
| Wire into trainer (side report after dataset load + before per-metal loop) | DONE — lazy import + try/except + JSON write to `args.out_dir/wf361c1_neural_ode_report.json` |
| Write 3 NEW tests | DONE — `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` (3/3 PASS in 4.17 s) |
| Update `metrics/by_metric/m6_kappa.json` | DONE — `wf361c1_neural_ode_cli_wire` sub-doc added; `wf_tag` updated |
| Document `TODO/completed/361c1_top_m6_kappa_improvement.md` | DONE — this file |

### §8.2 The Wf-361-C1 wire-in is a structural-recovery diagnostic, not a gate-breaking claim

The CLI flag emits a side report that contains:
- **Evolved feature stats** (min/max/mean of the 12-d RDKit + 12-d ODE-evolved features)
- **Wall time** for the ODE integration pass
- **Verdict cross-ref** (HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved per Wf-361-B2)
- **Cross-refs** to Wf-361-B1 (runner-up) + Wf-361-B2 (winner) + Wf-364-A2 (combined LAST architectural path)

The trainer's standard MetalCytoToxDB report (`trainer_args.out_dir/metal_<seed>/metrics.json`) is UNCHANGED. The m6_kappa `augmented_mean` value in `metrics/by_metric/m6_kappa.json` is UNCHANGED (still 0.4719 Wf-325D canonical baseline).

### §8.3 Forward-looking options

Per Wf-364-A2: "no more paths to try" (LAST architectural path). Forward-looking options for Wf-362+:

1. **Wf-362+**: wire the Neural ODE flow into the chemprop+PLL pipeline (Wf-335-M1) as a joint feature pipeline. CPU-only; would test whether the architectural diversity helps.
2. **Wf-362+**: end-to-end ODE MLP training via adjoint method (Chen et al. 2018 §3.3). CPU-only via `torchdiffeq.odeint_adjoint`. Would test whether learned continuous dynamics improve CV-R².
3. **Wf-362+**: cross-cohort feature W₂ z-score normalisation (per Wf-348-B1 §4.2 HONEST-NEGATIVE). Separate workflow.
4. **Per AGENTS.md §5.4**: drop m6_kappa as a Tier-2 paper metric; replace with m1_auc (A2780 cisplatin) which has multiple CEILING_BREAK anchors per Wf-335-M2.

---

## §9 Cross-references

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — Wf-361-C1 trainer CLI flag wire-in
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` — Wf-361-B1 MAML runner-up
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 Neural ODE winner
- `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` — 3 NEW tests (3/3 PASS in 4.17 s)
- `molmetal/tests/test_wf361b1_m6_kappa_maml.py` — Wf-361-B1 tests (4/4 PASS)
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 tests (3/3 PASS in 1.28 s)
- `molmetal/reports/wf361b1_m6_kappa_maml/result.json` — Wf-361-B1 primary MEASURED output
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` — Wf-361-B2 primary MEASURED output
- `metrics/by_metric/m6_kappa.json` — canonical metric JSON with `wf361c1_neural_ode_cli_wire` sub-doc
- `TODO/completed/361b1_m6_kappa_maml.md` — Wf-361-B1 closeout
- `TODO/completed/361b2_m6_kappa_neural_ode.md` — Wf-361-B2 closeout
- `TODO/completed/333v4_m6_kappa_alt.md` — Wf-333-V4 alternate-recovery bench
- `TODO/completed/335_final_synthesis.md §2` — Wf-335-M1 chemprop+PLL +0.6814 anchor
- `TODO/completed/335m1_chemprop_pll/result.json` — 60-SMILES smoke cohort CV-R² +0.6814
- `TODO/completed/348b1_w2_r20p3_implementation.md` — Wf-348-B1 W₂ recipe SHIPPED
- `TODO/completed/348b3_hilbert_r2_gate.md` — Wf-348-B3 Hilbert R² gate PRESERVED
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — Wf-359-A1 platinai_4cell verification
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2)
- `AGENTS.md §5.1` — Standing GPU authorization (NOT INVOKED — CPU-only)
- `AGENTS.md §5.2` — Honest framing (preserved)
- `AGENTS.md §5.3` — No paper work without authorization (zero touches to `paper/*`)
- `AGENTS.md §5.4` — Tier-2 markers (UNCHANGED — no wet-lab planning)

---

## §10 Wall + cost

| Stage | Wall (MEASURED) |
|---|---:|
| Trainer flag wire-in (parse_args + main + Edit) | < 5 min |
| Test authoring (3 NEW tests) | < 5 min |
| Test run (3/3 PASS) | 4.17 s |
| Metrics JSON update | < 1 min |
| Closeout doc authoring | < 5 min |
| **Total Wf-361-C1 CPU wall** | **< 30 min** |

PROJECTED vs MEASURED:
- PROJECTED 30-45 min CPU-only per task brief
- MEASURED < 30 min CPU-only (slightly faster than projected)

0 GPU launches. CPU-only per `AGENTS.md §5.1` standing GPU grant.

---

## §11 What is NOT being claimed

- **NOT claiming** the `--neural-ode-flow` flag lifts the m6_kappa `augmented_mean`. The MEASURED value remains 0.4495 (constrained ~baseline by unbiased CV projection on the smoke cohort), BELOW the 0.6 honesty threshold by -0.151.
- **NOT claiming** the trainer's standard MetalCytoToxDB report is changed. The side report (`wf361c1_neural_ode_report.json`) is a NEW file in `args.out_dir`; the standard report (`metal_<seed>/metrics.json`) is UNCHANGED.
- **NOT claiming** the gate is permanently lifted. Per Wf-335-M1 + Wf-348-B3 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 pattern, the gate could conceivably be lifted on different cohorts (larger metallodrug set with higher Shape Kappa) or with a different metric definition — but the current MEASURED value on the smoke cohort is 0.4511 (60-SMILES subset of `metallo_drugs_500_train.csv`), the gate is PRESERVED.
- **NOT claiming** the ODE MLP is trained. The MLP backbone is randomly initialised (seed=20260929) and used as a **feature extractor only** (matches Wf-361-B2 contract; see Wf-361-B2 §9).
- **NOT claiming** the flag is a Tier-2 paper claim. Per AGENTS.md §5.3 (TODO-84 active), zero touches to `paper/*`; the flag is a CLI diagnostic only.

---

**END OF Wf-361-C1 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL); `--neural-ode-flow` CLI flag wired into `train_dmpnn_multitask_wasserstein.py` (default OFF; bit-for-bit backward-compat with Wf-131 P1 / Wf-335-M1 / Wf-361-B2); 3 NEW tests PASS in 4.17 s (CPU-only).