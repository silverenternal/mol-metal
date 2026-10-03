# Wf-350-C3 — Gradient checkpointing + channels_last wire-up

**Date:** 2026-09-29
**Tag:** wf350-ultracode-C3
**Status:** SHIPPED

## §1 Background

Per Wf-343a-R3 findings #4 + #8 (Wf-326E follow-up follow-up for memory
optimizations on the production trainer):

  - **#4 — gradient checkpointing** (memory savings): wrap the
    per-mol encoder forward in
    :func:`torch.utils.checkpoint.checkpoint` so activations are dropped
    from the autograd graph and recomputed during backward. Memory
    savings ~O(sqrt(depth)) at the cost of ~30% more FLOPs.
  - **#8 — channels_last memory format** (memory-format throughput
    lift): apply NHWC layout to the per-atom feature tensor on CUDA
    before the per-mol encoder forward. PROJ 1.05-1.10x on per-metal
    head Linear ops at sm_90+ (Ampere / Hopper / Blackwell).

Both are opt-in CLI flags. Production default for both = OFF (the
Wf-131 P1 / Wf-328-W1 / Wf-335-P3 shipped behaviour is preserved
bit-for-bit when neither flag is set).

## §2 CLI flags added

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:

| Flag | Default | Effect |
|---|---|---|
| `--gradient-checkpointing` | OFF | wrap per-mol encoder forward in `torch.utils.checkpoint.checkpoint` |
| `--channels-last` | OFF | apply `memory_format=torch.channels_last` to per-atom feature tensor on CUDA |

Both flow through:
- argparse → `args.gradient_checkpointing` / `args.channels_last`
- `train_one_metal_one_seed(..., use_gradient_checkpointing, use_channels_last)`
- `MultiTaskWassersteinModel.__init__(..., use_gradient_checkpointing, use_channels_last)`
- `model.encode_pool(...)` runtime hook
- `training_summary` JSON (paper §6 audit trail)
- per-seed `ckpt_path` payload (`gradient_checkpointing_used` /
  `channels_last_used` fields)

## §3 Wire-in detail

### §3.1 `MultiTaskWassersteinModel.__init__`

Two new ctor params (`use_gradient_checkpointing` /
`use_channels_last`) defaulting to `False`. Both are stored as
`self.use_gradient_checkpointing` / `self.use_channels_last` flags
that `encode_pool` checks at runtime. CPU-only construction path is
unchanged from Wf-131 P1 / Wf-328-W1 / Wf-335-P3 (no extra params
needed for backward compat).

### §3.2 `MultiTaskWassersteinModel.encode_pool`

Runtime application:

  - `use_channels_last` + CUDA: `atom_feats = atom_feats.contiguous(
    memory_format=torch.channels_last)` applied BEFORE the per-mol
    forward loop. CPU + flag-ON is a no-op (channels_last is
    CUDA-only for tensor cores).
  - `use_gradient_checkpointing` + activation requires_grad: the per-
    mol encoder forward is wrapped in
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
    When the activation does NOT require grad (eval mode), the wrapper
    is skipped (no backward through a checkpoint = wasted overhead).

`bond_feats` (2-D per-mol) does not see channels_last (it would be a
no-op on <4-D tensors); we record the intent via `self.use_channels_last`
for paper §6 audits but skip the `.to()` call.

### §3.3 `train_one_metal_one_seed`

Two new params (`use_gradient_checkpointing` / `use_channels_last`)
defaulting to `False`. Forwarded to `MultiTaskWassersteinModel.__init__`
verbatim. No call-site branches — defaults preserve Wf-131 P1 / Wf-
328-W1 / Wf-335-P3 behaviour bit-for-bit.

### §3.4 `main()` argparse + per-seed banner + checkpoint payload

`main()` parses both flags from CLI and forwards to
`train_one_metal_one_seed`. The startup banner line + per-seed
`training_summary` JSON + per-seed `ckpt_paved` payload all record the
state so paper §6 audits can detect a memory-tight run (RTX 5090
32GB full sweep / RTX PRO 6000 98GB 50-epoch retrain).

## §4 Tests added (4 NEW, 4/4 PASS)

`molmetal/tests/test_wf350c3_memory_opts.py`:

| Test | Contract |
|---|---|
| `test_01_gradient_checkpointing_flag_parsed_and_applied` | `--gradient-checkpointing` parses → flag stored on model → per-mol forward is wrapped in `torch.utils.checkpoint.checkpoint` (verified via spy) when input requires_grad |
| `test_02_channels_last_flag_parsed_and_applied` | `--channels-last` parses → flag stored on model → forward executes without error (channels_last is no-op on CPU) |
| `test_03_backward_compat_default_off` | both flags OFF (default): checkpoint wrapper is NOT invoked (0 spy calls); Wf-131 P1 / Wf-328-W1 / Wf-335-P3 behaviour preserved bit-for-bit |
| `test_04_combined_flags_memory_savings` | both flags ON simultaneously: checkpoint wrapper invoked + channels_last advertised; the two compose without conflict |

### §4.1 Test execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf350c3_memory_opts.py -v
========================= 4 passed, 1 warning in 1.31s =========================
```

### §4.2 Backward compat re-verification

The existing Wf-335-P3 test suite (`molmetal/tests/test_wf335p3_triton_kernels_wire.py`,
8 tests) was re-run after the wire-up to confirm no regression:

```
========================= 8 passed, 1 warning in 1.15s =========================
```

## §5 Honest framing

Per `party-twenty-spirit §3.5` + AGENTS.md §5.2:

  - **Both default OFF** — production speed; opt in for memory-
    throughput tight cases (RTX 5090 32GB full sweep / RTX PRO 6000
    98GB 50-epoch retrain).
  - **MEASURED:** 4/4 NEW tests PASS CPU + 8/8 Wf-335-P3 backward-
    compat tests PASS. Per-mol checkpoint wrapper is invoked exactly
    B times when flag ON + input requires_grad; 0 times when OFF.
  - **PROJECTED memory savings:** ~O(sqrt(depth)) activation memory
    per Wf-326E follow-up audit. GPU validation deferred to Wf-326G
    GPU bench (per AGENTS.md §5.1 standing GPU grant — VRAM check
    first).
  - **PROJ throughput lift:** 1.05-1.10x on per-metal head Linear ops
    on sm_90+ (Ampere / Hopper / Blackwell) per Wf-326E follow-up
    audit. RTX 5090 + RTX PRO 6000 Blackwell sm_120 GPU bench
    deferred to Wf-326G.
  - **CPU + flag-ON is a no-op for channels_last** (channels_last is
    CUDA-only for tensor cores). CPU + flag-ON for
    `gradient_checkpointing` adds overhead (re-runs forward on
    backward) — `~+30% FLOPs` cost; expected and honest-neg framed.

## §6 Files touched

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — 2 new
  constants (`DEFAULT_GRADIENT_CHECKPOINTING` /
  `DEFAULT_CHANNELS_LAST`), 2 new ctor params on
  `MultiTaskWassersteinModel.__init__`, runtime hook in
  `encode_pool`, 2 new params on `train_one_metal_one_seed`, 2 new
  CLI flags in `parse_args`, wire-in in `main()`,
  `training_summary` JSON log, per-seed checkpoint payload, banner log.
- `molmetal/tests/test_wf350c3_memory_opts.py` — NEW 4-test suite.
- `TODO/completed/350c3_memory_opts.md` — THIS closeout doc.

## §7 GPU verification deferred

Per AGENTS.md §5.1 standing GPU grant, this closeout is CPU-only
verification. GPU bench (deferred to Wf-326G):

  - `--gradient-checkpointing`: VRAM measurement on RTX 5090 32GB
    full sweep (Wf-181 G5 retrain) + RTX PRO 6000 Blackwell sm_120
    50-epoch retrain (Wf-131 P1 reproduce). PROJECTED ~50% memory
    reduction at depth=3.
  - `--channels-last`: throughput bench on RTX PRO 6000 Blackwell
    sm_120 at production batch size (2048). PROJECTED 1.05-1.10x on
    per-metal head Linear ops.

GPU launch BLOCKED on `kill_then_adapt` per §0 of the
Wf-263A launch protocol. Defer to Wf-326G when ready.

## §8 Pre-flight gate

EXPECTED_GATE_COUNTS["all"] 62 PRESERVED. No new pre-flight gates
added by Wf-350-C3. Per AGENTS.md §1.4 the gate is N/A (this is an
audit, not a retrain / panel / ablation launch).

---

**END OF Wf-350-C3 closeout** — owner Hugo Lee; cross-ref
`party-twenty-spirit §3.5` + AGENTS.md §5.1; CPU-only verification
complete; 4/4 NEW tests PASS + 8/8 Wf-335-P3 backward-compat tests
PASS.
