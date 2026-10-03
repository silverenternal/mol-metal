# Wf-350-C2 — NaN detection (`torch.autograd.detect_anomaly`) wire-up

**Date:** 2026-09-29
**Wave:** Wf-350 ultracode residual closeout (R3 §3.9 priority #3 follow-up)
**Author:** Wf-350-C2 (Harness spawned subagent)
**Status:** SHIPPED — CPU-only, 6/6 NEW tests PASS, backward compat preserved.

---

## §1 Context (per Wf-343a-R3 finding #2)

Wf-343a-R3 surfaced 4 ultracode residuals; finding #2 was:

> "NaN detection (`torch.autograd.detect_anomaly`) should be wired into
> trainer. Helps catch numerical instability early."

`torch.autograd.set_detect_anomaly(True)` enables the autograd engine's
NaN / Inf backward-pass detector. When a NaN gradient surfaces during
backward, autograd prints the offending op + its forward traceback so
the bug is localizable in seconds instead of hours. Cost: ~2-3×
backward overhead when enabled, plus the process must run with the flag
on throughout the failed run. Default should remain **OFF** (production
speed); the flag is opt-in.

---

## §2 Implementation

### §2.1 CLI flag — D-MPNN trainer

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (line ~2433):

```python
# Wf-350-C2 — NaN detection (torch.autograd.detect_anomaly).
# Default OFF for production speed (detect_anomaly adds significant
# runtime overhead per backward; ~2-3x slowdown on CUDA, more on CPU
# when no anomaly fires).  Use --detect-anomaly to surface the
# offending autograd op when a NaN / Inf is observed in the loss or
# gradients (Wf-343a-R3 finding #2: helpful for early diagnosis of
# numerical instability in AMP/BF16 / W₂ / Wasserstein regimes).
p.add_argument(
    "--detect-anomaly",
    dest="detect_anomaly",
    action="store_true",
    default=False,
    help=(
        "Wf-350-C2 NEW: enable torch.autograd.set_detect_anomaly(True) "
        "at main() startup.  Default OFF (production speed; "
        "detect_anomaly adds ~2-3x backward overhead).  Use for "
        "diagnosis when a NaN/Inf appears in the loss or gradients."
    ),
)
```

Wire-up in `main()` (~line 2571, AFTER CUDA_VISIBLE_DEVICES, BEFORE
device selection):

```python
if getattr(args, "detect_anomaly", False):
    torch.autograd.set_detect_anomaly(True)
    print(
        "[wf350-c2] --detect-anomaly: torch.autograd.set_detect_anomaly(True) "
        "ENABLED (NaN/Inf during backward will print the offending "
        "op; ~2-3x backward overhead).",
        flush=True,
    )
```

### §2.2 CLI flag — CFM trainer

`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (line ~1440):

```python
# Wf-350-C2 — NaN detection (torch.autograd.detect_anomaly).
p.add_argument("--detect-anomaly", action="store_true", default=False,
                help=(
                        "Wf-350-C2 NEW: enable "
                        "torch.autograd.set_detect_anomaly(True) at "
                        "main() startup.  Default OFF (production "
                        "speed; detect_anomaly adds ~2-3x backward "
                        "overhead).  Use for diagnosis when a NaN/Inf "
                        "appears in the loss or gradients."
                ))
```

Wire-up in `main()` (line ~1464, FIRST statement after `args = parse_args()`,
BEFORE Wf-268A ablation block):

```python
if getattr(args, "detect_anomaly", False):
    torch.autograd.set_detect_anomaly(True)
    print(
        "[wf350-c2] --detect-anomaly: torch.autograd.set_detect_anomaly(True) "
        "ENABLED (NaN/Inf during backward will print the offending "
        "op; ~2-3x backward overhead).",
        flush=True,
    )
```

---

## §3 Tests (`molmetal/tests/test_wf350c2_detect_anomaly.py`)

6 NEW tests (the brief asked for 3; we cover both DMPNN + CFM
trainers symmetrically for full coverage). All 6 PASS.

| Test | Verdict |
|---|---|
| `test_dmpnn_trainer_detect_anomaly_flag_parsed` | PASS — flag accepted by D-MPNN trainer, default False, --detect-anomaly flips True. |
| `test_cfm_trainer_detect_anomaly_flag_parsed` | PASS — flag accepted by CFM trainer, default False. |
| `test_dmpnn_main_calls_set_detect_anomaly_when_flag_on` | PASS — D-MPNN main() calls `torch.autograd.set_detect_anomaly(True)` when flag is on (mocked spy). |
| `test_backward_compat_no_flag_no_set_detect_anomaly` | PASS — when --detect-anomaly absent, NO call to `set_detect_anomaly` is made (backward compat preserved). |
| `test_cfm_main_calls_set_detect_anomaly_when_flag_on` | PASS — CFM main() same behaviour. |
| `test_cfm_backward_compat_no_flag` | PASS — CFM default OFF preserved. |

Run command:

```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf350c2_detect_anomaly.py -v
```

Result: **6 passed, 1 warning in 1.63s**.

---

## §5 Honest framing (party-twenty-spirit §3.5)

| Item | Tag | Value |
|---|---|---|
| Default state | MEASURED | OFF (production speed) — argparse default check |
| Backward compat | MEASURED | preserved (test #4: no set_detect_anomaly call without flag) |
| Tests written | MEASURED | 6 NEW in 1 file (`test_wf350c2_detect_anomaly.py`) |
| Tests passing | MEASURED | 6/6 PASS in 1.63s |
| Wall time | MEASURED | ~1.63s (CPU-only test run) — well under PROJECTED 15-25 min |
| GPU launches | MEASURED | 0 (per §5.1 standing GPU grant — diagnostic flag, no retrain) |
| Side-effects (flag OFF) | MEASURED | none — `getattr(args, "detect_anomaly", False)` default `False` ⇒ branch not entered |

---

## §6 Usage

```bash
# Production retrain (default — flag OFF, no overhead):
bash scripts/launch_wf124_p3_g5.sh --dry-run

# Debug a NaN/Inf in loss or gradients (flag ON, expect ~2-3× slower):
python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
    --detect-anomaly --epochs 1 --batch-size 8 ...

python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --detect-anomaly --epochs 1 --batch-size 4 ...
```

When a NaN/Inf surfaces during backward, autograd prints something like:

```
RuntimeError: Function 'AddmmBackward0' returned nan due to an
in-place operation. Traceback:
  File ".../train_dmpnn_multitask_wasserstein.py", line ..., in train
    loss = loss_fn(...)
```

…which immediately localizes the failing layer without binary-search
through checkpoints.

---

## §7 Related

- **Wf-335-R3** — CUDA Graphs hard-disable safety lever (`--no-cuda-graphs`).
  Same default-OFF, opt-in philosophy.
- **Wf-343a-R3** — Ultracode residual finding #2 (this Wf).
- **Wf-330-B4** — Original CUDA Graphs RuntimeError that motivated this
  diagnostic wire-up (numerical instability caught early ⇒ shorter debug loop).
- **AGENTS.md §5.1** — Standing GPU authorization (no GPU launches needed here;
  this is a CPU-only diagnostic wire-up).
