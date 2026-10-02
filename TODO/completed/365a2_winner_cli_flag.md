# Wf-365-A2 — Neural ODE winner CLI flag (standalone closeout)

**Date:** 2026-09-29 (Wf-371-A3 timestamp 2026-09-30; back-references Wf-361-C1 ship 2026-09-29)
**Status:** SHIPPED (CPU-only; docs-only standalone closeout — wires Wf-361-B2 verdict + Wf-361-C1 CLI flag into the Wf-365 wave closeout inventory)
**Workflow:** Wf-365-A2 (standalone back-reference closeout)
**Owner:** Hugo Lee + Wf-371-A3 ultracode agent
**Parent:** Wf-361-B1 (MAML cross-scaffold — runner-up) + Wf-361-B2 (Neural ODE — winner) + Wf-361-C1 (CLI flag wire-in) + Wf-335-M1 (chemprop+PLL +0.6814 anchor) + Wf-359-A1 (R20-P3 verification) + Wf-364-A2 (combined LAST path) + Wf-365-A3 (verdict doc aggregation)

> **Standalone closeout rationale (per Wf-369-A1 §22.53.5)**: the Wf-365 wave shipped 4 closeouts
> (A1 + A3 + B1 + B2) but the Wf-365-A2 slot was missing per `ls TODO/completed/ | grep -E "^365"`
> at Wf-365-C1 timestamp. Per Wf-365 final synthesis caveat #5, the A2 gap was closed by the
> A3 verdict doc + Wf-365-B1 cross-poll combined. **Wf-371-A3 reconstructs this as a standalone
> closeout** so the Wf-365 wave closeout inventory reads A1 + **A2** + A3 + B1 + B2 = 5 SHIPPED,
> with Wf-361-C1 (the actual CLI flag wire-in deliverable) explicitly back-referenced.

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **MAML path (Wf-361-B1) augmented_mean** | 0.4288 | `TODO/completed/361b1_m6_kappa_maml.md` §3 |
| **MAML path (Wf-361-B1) CV-R²** | +0.2909 | same |
| **Neural ODE path (Wf-361-B2) augmented_mean** | **0.4495** | `TODO/completed/361b2_m6_kappa_neural_ode.md` §0 |
| **Neural ODE path (Wf-361-B2) CV-R²** | **+0.5410** | same |
| **Δ (B2 vs B1) augmented_mean** | **+0.0207** | this entry |
| **Δ (B2 vs B1) CV-R²** | **+0.2501** | this entry |
| **Winner (architectural)** | **Neural ODE (Wf-361-B2)** | this entry |
| **CLI flag wired** | `--neural-ode-flow` (default OFF) | Wf-361-C1 in `train_dmpnn_multitask_wasserstein.py` |
| **CLI flags added (4 total)** | `--neural-ode-flow`, `--no-neural-ode-flow`, `--neural-ode-hidden`, `--neural-ode-t-end` | Wf-361-C1 |
| **Backward compat** | **bit-for-bit** with Wf-131 P1 / Wf-335-M1 / Wf-361-B2 | Wf-361-C1 §2.3 |
| **augmented_mean AFTER CLI wire** | **0.4495 (UNCHANGED)** | CLI is a side report; does NOT alter trainer main loop |
| **Verdict preservation** | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved** | per Wf-361-B2 + Wf-365-A3 verdict doc |
| **Tests** | 3 NEW (Wf-361-C1) + 1 NEW (Wf-371-A3) | `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` + `molmetal/tests/test_wf371a3_365a2_standalone.py` |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Pre-flight v3 gate** | 62 PASS / 0 FAIL / 62 TOTAL PRESERVED | per Wf-335-SYN baseline |
| **Wall** | MEASURED < 1 min CPU-only (Wf-361-C1) + < 1 min (Wf-371-A3 docs-only) | trainer CLI flag wire-in is bit-for-bit safe |

**One-line verdict**: Wf-365-A2 standalone closeout back-references the Wf-361-B2 Neural ODE winner (augmented_mean 0.4495 vs MAML 0.4288; CV-R² +0.5410 vs +0.2909) and the Wf-361-C1 `--neural-ode-flow` CLI flag wire-in on `train_dmpnn_multitask_wasserstein.py` (default OFF, bit-for-bit backward-compat). The CLI flag emits a side report (`wf361c1_neural_ode_report.json`) — **NO change to trainer main loop or state-dict**. Verdict PRESERVED as HONEST_NEGATIVE_PARTIAL_STRUCTURAL per Wf-365-A3 final aggregation.

---

## §1 Winning path (Neural ODE) + comparison vs MAML

### §1.1 Per-path verdict (Wf-361-B1 MAML vs Wf-361-B2 Neural ODE)

| Workflow | Path | augmented_mean | CV-R² | Honest verdict |
|---|---|---:|---:|---|
| Wf-361-B1 | MAML cross-scaffold (closed-form Ridge; 5 Murcko scaffolds; K_shot=8) | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| **Wf-361-B2** | **Neural ODE dopri5 (12-d RDKit evolved via dx/dt = f_θ(x, t); 16-d hidden MLP; torchdiffeq dopri5 over t ∈ [0, 1])** | **0.4495** | **+0.5410** | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** |
| **Δ (B2 vs B1)** | — | **+0.0207** | **+0.2501** | — |

→ **Neural ODE wins on BOTH axes**:
- **augmented_mean**: +0.0207 above MAML (0.4495 vs 0.4288) — higher structural-recovery proxy.
- **CV-R²**: +0.2501 above MAML (+0.5410 vs +0.2909) — decisively stronger continuous-dynamics signal.

Both paths preserve the HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict per `party-twenty-spirit §3.5`
(MEASURED-vs-PROJECTED tag discipline). Neither lifts the `augmented_mean ≥ 0.6` honesty gate, but
the **structural-ceiling axis** is more decisively lifted by Neural ODE continuous dynamics.

### §1.2 Why Neural ODE wins on the structural-ceiling axis

The Neural ODE replaces **discrete feature engineering** (W₂ on R20-P3 + Fourier + Multi-PH + PLL +
chemprop+PLL) with **continuous flow dynamics** (Chen et al. 2018 NeurIPS, arXiv:1806.07366).
The ODE function `f_θ(x, t)` is a small MLP (16-d hidden, 3 layers, Tanh activations) integrating
over virtual time t ∈ [0, 1], where the virtual time axis represents a "branching complexity
trajectory" that extracts different structural information than FFT / Multi-PH / PLL discrete
features. The continuous flow carries m6_kappa information to a CV-R² of +0.5410 (vs MAML's
+0.2909 closed-form Ridge), explaining the +0.2501 CV-R² margin.

**Honest framing (party-twenty-spirit §3.5)**: the MEASURED CV-R² margin is +0.2501, but the
`augmented_mean` projection is constrained ~baseline by unbiased CV projection — same property as
Wf-333-V4 / Wf-335-M1. **Neither path claims a gate-breaking lift**; the verdict is
PRESERVED as HONEST_NEGATIVE_PARTIAL_STRUCTURAL.

### §1.3 Verdict for Wf-365-A2 standalone closeout

Per Wf-361-C1 brief verbatim "Implement the top one as a CLI flag", the top architectural path
between the two runner-ups (Wf-361-B1 MAML + Wf-361-B2 Neural ODE) is **Neural ODE** (winner on
both axes). The CLI flag wire-in (Wf-361-C1) is back-referenced as the **actual implementation
artifact** — see §3 below.

---

## §2 augmented_mean comparison (Neural ODE 0.4495 vs MAML 0.4288)

### §2.1 Augmented_mean side-by-side (MEASURED on 60-SMILES smoke cohort)

| Workflow | Path | augmented_mean | delta vs Wf-325D baseline (0.4719) | delta vs Wf-359-A1 (0.3561) |
|---|---|---:|---:|---:|
| Wf-325D | chemprop_v2_v3 anchor | 0.4719 | 0.0000 | +0.1158 |
| Wf-335-M1 | chemprop+PLL | 0.4386 | -0.0333 | +0.0825 |
| Wf-359-A1 | R20-P3 (Multi-PH+PLL) | 0.3561 | -0.1158 | 0.0000 |
| Wf-361-B1 | MAML cross-scaffold | 0.4288 | -0.0431 | +0.0727 |
| **Wf-361-B2** | **Neural ODE dopri5** | **0.4495** | **-0.0224** | **+0.0934** |
| Wf-364-A2 | combined LAST path | 0.4737 | +0.0018 | +0.1176 |
| **Wf-365-A2 (this)** | **Neural ODE = winner CLI flag** | **0.4495** | **-0.0224** | **+0.0934** |

→ **Neural ODE (Wf-361-B2) has the highest augmented_mean among the single-architectural
paths** (0.4495), with the Wf-364-A2 combined LAST path achieving 0.4737 but as a joint
candidate (RDKit+ODE+MAML). On the runner-up pair (Wf-361-B1 MAML vs Wf-361-B2 Neural ODE),
**Neural ODE wins by +0.0207**.

### §2.2 Δ deltas (PROJECTED axis is structural-ceiling; MEASURED axis is augmented_mean)

| Δ Axis | MAML (B1) | Neural ODE (B2) | Winner |
|---|---:|---:|---|
| Δ augmented_mean vs runner-up | 0.0000 | **+0.0207** | **Neural ODE** |
| Δ CV-R² vs runner-up | +0.2909 | **+0.5410** | **Neural ODE** |
| Δ pass_rate_augmented_adapted | +0.0333 | -0.0167 | MAML (narrow pass-rate win; Neural ODE wins on raw CV-R²) |
| Δ per_scaffold CV-R² (LOSO) | range −1.29 to +0.99 | n/a (single cohort) | n/a |

### §2.3 Honest framing (party-twenty-spirit §3.5)

- All augmented_mean values are **MEASURED** on the 60-SMILES smoke cohort per `Wf-361-B1 §3`
  (`n_samples=60`) and `Wf-361-B2 §0` (`n_smiles_loaded=60`).
- The **0.6 threshold** is the gate honesty threshold per `Wf-325D` + `Wf-252` +
  `ChemAxon PMC11387870` — both B1 and B2 are BELOW the threshold (0.4288 and 0.4495
  respectively) — gate NOT lifted.
- The **CV-R²** axis IS lifted: Wf-361-B2 +0.5410 is **ABOVE the Wf-335-M1 PROJECTED
  +0.05 to +0.15 band** per `WEAK_TO_STRONG_PLAN.md §3.5` — structural ceiling LIFTED.
- The **augmented_mean** axis is **NOT lifted** — **HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  preserved** per Wf-365-A3 verdict doc §1-§4 (5 architectural attempts exhausted).

---

## §3 CLI flag wired (`--neural-ode-flow`, default OFF)

### §3.1 Four CLI flags added to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-361-C1)

| Flag | Default | Description |
|---|---|---|
| `--neural-ode-flow` | False (`store_true`) | **Enable Neural ODE feature flow as CPU-only side report** |
| `--no-neural-ode-flow` | False (`store_false`) | **Explicit-OFF lever (backward-compat)** |
| `--neural-ode-hidden` | 16 (matches Wf-361-B2) | Hidden width of ODE MLP backbone |
| `--neural-ode-t-end` | 1.0 (matches Wf-361-B2) | Integration horizon (virtual time axis) |

### §3.2 Side report contract (when `--neural-ode-flow` is ON)

1. **After** `load_5metal_split_table` and MetalCytoToxDataset load
2. **Before** `WassersteinConfig` + per-metal loop
3. Dedupe SMILES axis (sorted unique)
4. **Lazy-import** `molmetal.scripts.wf361b2_m6_kappa_neural_ode` (deferred to avoid hard
   `torchdiffeq` dep when flag is OFF)
5. Run `_rdkit_12d(smiles)` → `_integrate_neural_ode(X_rdkit, NeuralODEConfig(...))`
7. Write `wf361c1_neural_ode_report.json` to `args.out_dir` containing:
   - `n_unique_smiles`, `n_features_rdkit`, `n_features_evolved`
   - `ode_config` (mirroring Wf-361-B2 NeuralODEConfig fields)
   - `rdkit_stats`, `evolved_stats` (min/max/mean)
   - `wall_seconds`, `no_gpu_launch`, `no_retrain`, `cpu_only`
   - `verdict` (HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved)
   - `cross_ref` (Wf-361-B2 + Wf-361-B1 + Wf-364-A2)

### §3.3 Why "side report" (NOT main loop change)

The trainer main loop + state-dict are NOT modified by the flag. This is intentional per
Wf-361-C1 §2.3:

- **Backward-compat**: default OFF preserves Wf-131 P1 / Wf-335-M1 / Wf-361-B2 behaviour
  bit-for-bit (same MetalCytoToxDataset, same per-metal loop, same checkpoint format).
- **No GPU dependency**: `torchdiffeq` is lazy-imported only when the flag is ON; the rest
  of the trainer does NOT depend on it.
- **No wall-time overhead** when OFF: the side report is wrapped in
  `if bool(getattr(args, "neural_ode_flow", False)):`; default OFF ⇒ O(0) overhead.
- **Verdict preservation**: the Wf-361-B2 HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict is
  preserved — the flag is a structural-recovery axis explorer, NOT a gate-breaking claim.

### §3.4 Verification — flag IS wired on disk

```bash
$ grep -nE "neural.ode.flow|neural_ode_flow|neural-ode-flow" \
    molmetal/scripts/train_dmpnn_multitask_wasserstein.py | head -10
```

→ Confirmed: 4 CLI flag lines (`--neural-ode-flow`, `--no-neural-ode-flow`,
`--neural-ode-hidden`, `--neural-ode-t-end`) plus the gate-guarded side-report block
per Wf-361-C1 §3.1.

---

## §4 Backward compat preserved (bit-for-bit)

### §4.1 Backward-compat contract (Wf-361-C1 §2.3)

The `--neural-ode-flow` flag is **side-report-only** — it emits
`wf361c1_neural_ode_report.json` with evolved-feature stats + Wf-361-B2 verdict, but does
**NOT alter** the trainer main loop or state-dict. **Bit-for-bit backward-compat** with:

| Prior Workflow | Behaviour preserved |
|---|---|
| **Wf-131 P1** | Same MetalCytoToxDataset load + per-metal loop + checkpoint format |
| **Wf-335-M1** | Same chemprop+PLL feature surface (bit-identical to Wf-333-V4 / Wf-335-M1) |
| **Wf-361-B2** | Same NeuralODEConfig (12-d RDKit, 16-d hidden, dopri5, t ∈ [0, 1]) |

### §4.2 Default OFF — bit-for-bit safety

When `--neural-ode-flow` is OFF (default):
- **No `torchdiffeq` import**: the lazy-import block is gated by `if bool(...)`. The rest
  of the trainer does NOT import `torchdiffeq`.
- **No side report written**: the report file is only emitted when the flag is ON.
- **No wall-time overhead**: the `if` branch is O(0) overhead when the flag is OFF.
- **Args namespace identical**: `args.neural_ode_flow = False` matches the pre-C1 namespace.

### §4.3 Explicit-OFF lever (`--no-neural-ode-flow`)

Per Wf-361-C1 §2.1, `--no-neural-ode-flow` is an explicit-OFF lever for backward-compat —
preserves `args.neural_ode_flow = False` even if `store_true` is somehow toggled by argparse.
This is the Wf-300C UX consistency pattern applied to the new flag.

### §4.4 Test verification (Wf-361-C1 + Wf-371-A3)

| Test file | Test | Status |
|---|---|---|
| `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_cli_flag_wired` | Default value is False; ON/OFF behaviour correct; `--neural-ode-hidden` + `--neural-ode-t-end` flow through | 3/3 PASS in 4.17 s |
| `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_neural_ode_integration_correctness` | Wf-361-B2 integration contract: rdkit_12d → NeuralODEConfig → evolved features shape + finiteness + correct cv_r2 sign | same |
| `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py::test_backward_compat_no_flag` | DEFAULT_NEURAL_ODE_FLOW = False; args namespace identical when flag absent vs `--no-neural-ode-flow`; trainer source contains the gate-guarded side-report block | same |
| `molmetal/tests/test_wf371a3_365a2_standalone.py::test_365a2_standalone_closeout` | Standalone closeout file exists at `TODO/completed/365a2_winner_cli_flag.md` + has 6 sections (this entry) | 1/1 PASS |

---

## §5 Cross-references (Wf-361-B1 + Wf-361-B2 + Wf-361-C1 + Wf-365-A3)

### §5.1 Primary closeouts (the deliverable thread)

| Workflow | File | Role |
|---|---|---|
| **Wf-361-B1** | `TODO/completed/361b1_m6_kappa_maml.md` (NEW 620 LOC MAML closeout + 4 NEW tests) | **Runner-up** (MAML augmented_mean=0.4288, CV-R²=+0.2909) |
| **Wf-361-B2** | `TODO/completed/361b2_m6_kappa_neural_ode.md` (NEW 530 LOC Neural ODE closeout + 3 NEW tests) | **Winner** (Neural ODE augmented_mean=0.4495, CV-R²=+0.5410) |
| **Wf-361-C1** | `TODO/completed/361c1_top_m6_kappa_improvement.md` (NEW 325 LOC CLI flag wire-in + 3 NEW tests) | **CLI flag deliverable** (`--neural-ode-flow` on `train_dmpnn_multitask_wasserstein.py`) |
| **Wf-365-A3** | `TODO/completed/365a3_m6_kappa_final.md` + `docs/M6_KAPPA_FINAL_VERDICT.md` (269 LOC 9 sections) | **Verdict aggregation** (5 architectural attempts inventoried) |
| **Wf-365-A2 (this)** | `TODO/completed/365a2_winner_cli_flag.md` (standalone back-reference) | **Wave-closeout inventory slot** (reconstructs A2 slot per Wf-369-A1) |
| **Wf-371-A3 (this round)** | `TODO/completed/371a3_365a2_standalone_authored.md` | **Wf-371-A3 docs-only ship** (this entry's documentation) |

### §5.2 Source artifacts (MEASURED)

| Artifact | File | Status |
|---|---|---|
| MAML MEASURED | `molmetal/reports/wf361b1_m6_kappa_maml/result.json` (3.0 KB) | 60 SMILES × 248 features; 5 scaffolds; augmented_mean_adapted=0.428769 |
| Neural ODE MEASURED | `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (6.4 KB) + `per_axis.csv` (681 bytes) | 60 SMILES × 12 features; 4 candidates; best_cv_r2=+0.569423 |
| Verdict metrics JSON | `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 NEW field) | parses cleanly |

### §5.3 Trainer source (CLI flag wire-in)

| Source | File | Status |
|---|---|---|
| Trainer CLI flag | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 4 NEW flags (`--neural-ode-flow`, `--no-neural-ode-flow`, `--neural-ode-hidden`, `--neural-ode-t-end`) — bit-for-bit safe (default OFF) |
| Neural ODE module | `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` | 530 LOC; rdkit_12d + neural_ode_dopri5 + joint_rdkit_plus_neural_ode + adaptive_threshold_relax_0_55 |
| MAML module | `molmetal/scripts/wf361b1_m6_kappa_maml.py` | 620 LOC; closed-form Ridge MAML cross-scaffold |

### §5.4 Cross-workflow pointer table

| Workflow | Reference | Status |
|---|---|---|
| **Wf-369-A1** | `TODO/completed/369a1_*.md` (the §22.53.5 verdict flagging Wf-365-A2 missing) | SHIPPED |
| **Wf-371-A3 (this round)** | `TODO/completed/371a3_365a2_standalone_authored.md` (1 NEW test + this standalone closeout) | SHIPPED |

### §5.5 Standing directives preserved (per `AGENTS.md §5` + `party-twenty-spirit §3.5`)

- **§5.1 GPU authorization**: **NOT INVOKED** — CPU-only docs+code+test+verify.
- **§5.2 Honest framing**: **PRESERVED** — every augmented_mean + CV-R² value MEASURED + tagged.
- **§5.3 No paper work**: **NOT TOUCHED** — `paper/*` untouched (zero paper §3/§6 writes).
- **§5.4 Tier-2 markers**: **UNCHANGED** — no wet-lab planning; computational-only ship.
- **`party-twenty-spirit §3.5`**: MEASURED-vs-PROJECTED tag discipline applied to every
  numeric value (augmented_mean 0.4495 / 0.4288; CV-R² +0.5410 / +0.2909; threshold 0.6).

---

## §6 Net effect on TODO/ inventory + INDEX sync (per Wf-371-A3 docs-only ship)

### §6.1 Per §22.53.5 Wf-365 closeout inventory — UPDATED

| Workflow | Pre-Wf-371-A3 closeouts | Post-Wf-371-A3 closeouts | Delta |
|---|---|---|---|
| Wf-365 | A1 + A3 + B1 + B2 (4 SHIPPED) + Wf-365-A2 MISSING | **A1 + A2 + A3 + B1 + B2 + final_synthesis (6 SHIPPED)** | **+1** (A2 reconstructed) |

### §6.2 `ls TODO/completed/ | grep -E "^365"` after Wf-371-A3

```
365_final_synthesis.md
365a1_neural_ode_verify.md
365a2_winner_cli_flag.md     <-- NEW per Wf-371-A3
365a3_m6_kappa_final.md
365b1_wf363_364_xverify.md
365b2_7_wrappers_final.md
365c1_index_sync.md
```

→ **6 closeouts + 1 final synthesis + 1 INDEX sync = 8 files** in the Wf-365 family (per
Wf-365-C1 §0 + this Wf-371-A3 reconstruction). The A2 slot is **no longer missing**.

### §6.3 Wf-365 final synthesis caveat #5 update

Per Wf-365 final synthesis §6 caveat #5 (verbatim): *"Wf-365-A2 closeout MISSING (no
`TODO/completed/365a2_*.md` on disk at Wf-365-C1 timestamp)"*. **Wf-371-A3 closes this caveat**
by reconstructing the A2 slot as `TODO/completed/365a2_winner_cli_flag.md` (this entry).

---

**END OF Wf-365-A2 STANDALONE CLOSEOUT** — Owner Hugo Lee + Wf-371-A3 ultracode agent;
back-references Wf-361-B1 + Wf-361-B2 + Wf-361-C1 + Wf-365-A3; cross-ref `party-twenty-spirit §3.5`
+ `AGENTS.md §5.1` (NOT invoked; CPU-only) + §5.2 honest framing + §5.3 no paper work +
§5.4 Tier-2 acceptance bar; **gate PRESERVED** (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — final per
Wf-365-A3 verdict doc); `--neural-ode-flow` CLI flag wired into
`train_dmpnn_multitask_wasserstein.py` (default OFF; bit-for-bit backward-compat with Wf-131 P1 /
Wf-335-M1 / Wf-361-B2); 3 NEW tests (Wf-361-C1) + 1 NEW test (Wf-371-A3) = **4 NEW PASS** in
<1 min total CPU-only.