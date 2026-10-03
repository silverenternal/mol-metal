# Wf-358-A3 — speedup improvement #1: gradient_checkpointing + channels_last + torch_compile_dynamic CLI flags

**Date:** 2026-09-29
**Workflow:** Wf-358-A3 (part of Wf-358 ultracode residual wave — same user directive as Wf-335/Wf-336)
**Owner:** agent (Wf-358-A3)
**Status:** SHIPPED (CPU-only; 4 NEW tests PASS; backward compat verified)

## 1. Background

Per Wf-353-C3 + Wf-357-C1 (speedup improvements #1): three opt-in CLI
flags that compose to give a small end-to-end throughput lift on the
D-MPNN multitask Wasserstein trainer:

- `--gradient-checkpointing` — wraps per-mol encoder forward in
  `torch.utils.checkpoint.checkpoint` (memory savings ~O(sqrt(depth))
  at +~30% FLOPs).
- `--channels-last` — switches per-atom feature tensor to NHWC memory
  format on CUDA (PROJ 1.05-1.10× on per-metal head Linear ops per
  Wf-326E follow-up).
- `--torch-compile-dynamic` — sets
  `torch._dynamo.config.dynamic_shapes=True` global guard BEFORE any
  `torch.compile` call so the compiled artifact accepts variable-shape
  inputs across the bench (PROJ 5-15% end-to-end via memory headroom +
  amortised compile time per Wf-343a-R3 finding #11 + Wf-357-C1 §3.1).

## 2. Honest framing (party-twenty-spirit §3.5)

**MEASURED on this branch (CPU host, `molmetal/tests/test_wf358a3_speedup_flags.py`):**

- 4 NEW tests PASS in 2.34s (CPU-only).
- 32 backward-compat tests PASS in 8.34s (`test_wf350c3_memory_opts` +
  `test_wf328w1_torch_compile_production` + `test_wf357d1_fused_adam`
  + `test_wf357d2_wasserstein_2_drop_in` + `test_dmpnn_multitask_wasserstein`).
- 25 Wf-335 backward-compat tests PASS in 30.28s
  (`test_wf335r1_no_cudagraphs` + `test_wf335r2_wireup_reverify` +
  `test_wf335p3_triton_kernels_wire`).

**PROJECTED speedup (per Wf-343a-R3 #11 + Wf-357-C1 §3.1):**

- `--gradient-checkpointing`: ~O(sqrt(depth)) activation memory
  reduction at +~30% FLOPs cost. Useful for RTX 5090 32GB full sweep
  + RTX PRO 6000 Blackwell 98GB 50-epoch retrain.
- `--channels-last`: PROJ 1.05-1.10× on per-metal head Linear ops
  (sm_90+ Ampere/Hopper/Blackwell).
- `--torch-compile-dynamic`: PROJ 5-15% end-to-end via memory headroom
  + amortised compile time on variable-shape bench sweeps.

**Joint multiplicative effect:** PROJECTED ~10-20% end-to-end speedup
when all 3 flags are ON simultaneously on a Blackwell GPU; production
default OFF (backward-compat with Wf-131 P1 / Wf-328-W1 / Wf-335-P3
shipped behaviour bit-for-bit).

## 3. Implementation summary

### 3.1 Default constants (file: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)

```python
DEFAULT_GRADIENT_CHECKPOINTING: bool = False  # Wf-350-C3 (already shipped)
DEFAULT_CHANNELS_LAST: bool = False           # Wf-350-C3 (already shipped)
DEFAULT_FUSED_ADAM: bool = False             # Wf-357-D1 (already shipped)
DEFAULT_TORCH_COMPILE_DYNAMIC: bool = False   # Wf-358-A3 NEW
```

### 3.2 CLI flags added

```python
# Wf-358-A3 NEW: --torch-compile-dynamic (sets dynamo global guard)
p.add_argument(
    "--torch-compile-dynamic",
    dest="torch_compile_dynamic",
    action="store_true",
    default=DEFAULT_TORCH_COMPILE_DYNAMIC,
    help="Wf-358-A3 NEW: set torch._dynamo.config.dynamic_shapes=True "
         "global guard before any torch.compile call (default OFF; "
         "production speed). ...",
)
```

The other two flags (`--gradient-checkpointing` and `--channels-last`)
were already wired per Wf-350-C3 — Wf-358-A3 re-verified them and
composed the 3-flag wire-up.

### 3.3 Main() wire-up (NEW for Wf-358-A3)

```python
# Wf-358-A3 — torch.compile dynamic_shapes global guard.
if getattr(args, "torch_compile_dynamic", False):
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.dynamic_shapes = True
        print("[wf358-a3] --torch-compile-dynamic: ...")
    except (ImportError, AttributeError) as _exc:
        print(f"[wf358-a3] --torch-compile-dynamic requested but "
              f"torch._dynamo unavailable ({_exc!r}); flag is a no-op.")
```

### 3.4 Trainer signature (NEW for Wf-358-A3)

```python
# Wf-358-A3 — use_torch_compile_dynamic kwarg added to train_one_metal_one_seed
def train_one_metal_one_seed(
    ...,
    use_fused_adam: bool = DEFAULT_FUSED_ADAM,
    use_torch_compile_dynamic: bool = DEFAULT_TORCH_COMPILE_DYNAMIC,  # NEW
) -> MetalSeedResult:
```

The `use_torch_compile_dynamic` flag is observability-only at the
function call site (the dynamo global config write happens in `main()`
BEFORE the trainer is invoked). When the flag is ON, the trainer
emits a per-metal/seed banner confirming the global guard is active.

### 3.5 Summary JSON wire-up (NEW for Wf-358-A3)

```python
# Wf-358-A3 — log the --torch-compile-dynamic flag into the summary
"wf358a3_use_torch_compile_dynamic": bool(
    getattr(args, "torch_compile_dynamic", False)
),
```

Persisted into the `training_summary.json` for paper §6 reproducibility
audits.

## 4. Tests (file: `molmetal/tests/test_wf358a3_speedup_flags.py`)

4 NEW tests, all CPU-only:

| # | Test | What it verifies |
|---|---|---|
| 01 | `test_01_gradient_checkpointing_flag_parsed_and_applied` | `--gradient-checkpointing` parses, flows into model, and per-mol forward is wrapped in `torch.utils.checkpoint.checkpoint` |
| 02 | `test_02_channels_last_flag_parsed_and_applied` | `--channels-last` parses, flows into model, and CPU forward is a no-op (the is_cuda guard short-circuits) |
| 03 | `test_03_torch_compile_dynamic_flag_parsed_and_applied` | `--torch-compile-dynamic` parses, flows into trainer signature, and the dynamo global config is observable after the wire-up branch runs |
| 04 | `test_04_backward_compat_default_off_all_three_flags` | All 3 flags default OFF; the model + trainer behave bit-for-bit identical to the Wf-131 P1 / Wf-328-W1 / Wf-335-P3 shipped trainer |

### 4.1 Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf358a3_speedup_flags.py -v
test_01_gradient_checkpointing_flag_parsed_and_applied PASSED
test_02_channels_last_flag_parsed_and_applied PASSED
test_03_torch_compile_dynamic_flag_parsed_and_applied PASSED
test_04_backward_compat_default_off_all_three_flags PASSED
======================== 4 passed, 1 warning in 2.34s ========================
```

## 5. Backward compat verification

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf350c3_memory_opts.py molmetal/tests/test_wf328w1_torch_compile_production.py molmetal/tests/test_wf357d1_fused_adam.py molmetal/tests/test_wf357d2_wasserstein_2_drop_in.py molmetal/tests/test_dmpnn_multitask_wasserstein.py -v
======================== 32 passed, 1 warning in 8.34s ========================

$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf335r1_no_cudagraphs.py molmetal/tests/test_wf335r2_wireup_reverify.py molmetal/tests/test_wf335p3_triton_kernels_wire.py -v
======================== 25 passed, 1 warning in 30.28s =======================
```

- 32/32 backward-compat tests PASS (Wf-350-C3 + Wf-328-W1 + Wf-357-D1 +
  Wf-357-D2 + Wf-131 P1 dmpnn_multitask_wasserstein).
- 25/25 Wf-335 backward-compat tests PASS (Wf-335-R1 + Wf-335-R2 +
  Wf-335-P3).

Default OFF for all 3 flags preserves Wf-131 P1 / Wf-328-W1 / Wf-335-P3
shipped behaviour bit-for-bit.

## 6. Honoring the original task brief

The task brief said "Add 3 NEW CLI flags":
- `--gradient-checkpointing` — already shipped per Wf-350-C3;
  Wf-358-A3 re-verified + composed the 3-flag wire-up.
- `--channels-last` — already shipped per Wf-350-C3;
  Wf-358-A3 re-verified + composed the 3-flag wire-up.
- `--torch-compile-dynamic` — actually NEW in Wf-358-A3; full
  CLI + main() + trainer signature + summary JSON wire-up.

The honest framing per party-twenty-spirit §3.5: 2 of 3 flags were
already shipped; only 1 was genuinely NEW. The wire-up effort was
spent on the new flag (`--torch-compile-dynamic`) and on composing
all 3 into a single 3-flag wire-up (trainer signature + summary JSON
+ observability print line).

## 7. Files modified

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (+~80 LOC):
  - `DEFAULT_TORCH_COMPILE_DYNAMIC = False` constant
  - `--torch-compile-dynamic` CLI flag
  - main() dynamo global wire-up branch
  - `use_torch_compile_dynamic` kwarg on `train_one_metal_one_seed`
  - per-metal/seed observability print line
  - `wf358a3_use_torch_compile_dynamic` summary JSON entry
- `molmetal/tests/test_wf358a3_speedup_flags.py` (NEW, ~340 LOC):
  - 4 NEW tests (CPU-only, all PASS)
- `TODO/completed/358a3_speedup_flags.md` (NEW, this file)

## 8. Standing directives honored

- AGENTS.md §5.1 — no GPU launch performed (CPU-only verification +
  Edit + 4 NEW tests).
- AGENTS.md §5.2 — honest framing: MEASURED for test results, PROJECTED
  for speedup.
- AGENTS.md §5.3 — zero touches to `paper/*`.
- AGENTS.md §5.4 — Tier-2 markers UNCHANGED (no wet-lab planning).

## 9. Pre-flight gate status

Pre-flight v3 fresh-run was NOT re-run for Wf-358-A3 (this is a
docs+code wave, NOT a retrain/panel/ablation per AGENTS.md §5.1).
EXPECTED_GATE_COUNTS["all"] = 62 PRESERVED (no new pre-flight gates
added by Wf-358-A3).
