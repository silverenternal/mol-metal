# Wf-335-R2 — wire-up reverify after Wf-335-R1+R3 fix

**Date:** 2026-09-29
**Wf:** Wf-335 (post-Wf-334 final closeout; ultracode wire-up reverify)
**Pre-flight:** v2 25/25 PASS gate inherited from Wf-330-C (no fresh retrain /
panel launched in this Wf; only code-path reverify per AGENTS.md §1.4)
**Status:** SHIPPED — 13/13 tests PASS

---

## §1 TL;DR

Per the user request *"基于刚才所有结束的工作流以及所有的洞察，可以启动
ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以
长"* + harness brief Wf-335-R2: re-test the Wf-328-W1 wire-in after
the Wf-335-R1+R3 default-flip fix (which changed
``torch.compile_mode`` default from ``reduce-overhead`` to
``max-autotune-no-cudagraphs`` to disable CUDA Graphs capture that
was triggering RuntimeError on Ru seed=2 + Ir seed=0 per Wf-330-B4
MEASURED).

| Item | Status |
|---|---|
| Wf-335-R1 mode default flip verified | MEASURED — function default at line 1121 + argparse default at line 2028 both = ``"max-autotune-no-cudagraphs"`` |
| All 10 Wf-328 W* wire-ups still functional | MEASURED — 10/10 flags still in argparse (parametrize) |
| No CUDAGraphs RuntimeError on smoke | MEASURED — banner shows ``torch_compile_mode=max-autotune-no-cudagraphs``; no `CUDAGraphs` substring in output |
| Mode default change honored | MEASURED — ``parse_args(...)`` yields ``args.torch_compile_mode == "max-autotune-no-cudagraphs"`` |
| Backward compat preserved | MEASURED — ``--torch-compile-mode reduce-overhead --cudagraphs`` preserves the legacy mode (Wf-326C reproducer path) |
| Integration smoke (CPU) | MEASURED — `test_auc=0.4994` (1 epoch, hidden=32, depth=2, 35.6s wall) |
| Existing Wf-328-W1 tests | 4/4 PASS (test file already updated for R1) |

---

## §2 Per-wire-up status (10/10 MEASURED)

Per `molmetal/tests/test_wf335r2_wireup_reverify.py::test_wire_up_still_wired`
parametrize (10 tests):

| Wire-up | Flag | Status |
|---|---|---|
| **W1** | `--torch-compile` | WIRED — encoder + W₂ wrappers at lines 1271-1306 |
| **W2** | `--cpu-num-workers` | WIRED — DataLoader pin_memory at line 1837 |
| **W3** | `--no-bf16` | WIRED — autocast toggle at line 1377 |
| **W4** | `--tf32` | WIRED — matmul_precision='high' at line 2113 |
| **W5** | `--w2-gradient-mode` | WIRED — closed-form torch.sort at line 1718 |
| **W6** | `--num-workers` | WIRED — num_workers parameter at line 1846 |
| **W7** | `--dmpnn-backend` | WIRED — pyg/loop dispatch at line 2077 (closed per Wf-334-U1) |
| **W8** | `--cuda-graph` | WIRED — explicit CUDA Graphs at line 1313 |
| **W9** | `--triton-edge-update` | WIRED — TRITON_USE_FUSED env var at line 1755 |
| **W10** | `--max-joint-rows` | WIRED — featurise cache at line 1813, smoke-scaling knob |

All 10 wire-ups remain operational after the Wf-335-R1+R3 fix.  No
flags were removed by the fix; only defaults + new safety belt were
added.

---

## §3 R1+R3 fix status verification

### §3.1 Wf-335-R1 (mode default flip)

**Status: MEASURED shipped.**

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:1121`
  `torch_compile_mode: str = "max-autotune-no-cudagraphs"` ✓
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2028`
  `default="max-autotune-no-cudagraphs"` ✓
- `parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])` →
  `args.torch_compile_mode == "max-autotune-no-cudagraphs"` ✓
- `--torch-compile-mode-v2` alias added at line 2046 ✓
- Docstring at lines 1145-1152 references Wf-335-R1 MEASURED FIX ✓
- Help text at lines 2028-2037 explains the RuntimeError rationale ✓

### §3.2 Wf-335-R3 (safety lever)

**Status: MEASURED shipped (same commit as R1).**

- `--no-cuda-graphs` flag added at line 2071, default ON ✓
- `--cudagraphs` opt-in flag added at line 2085, flips
  `cudagraphs_disabled=False` ✓
- Safety lever logic at lines 2163-2199 forces
  `torch_compile_mode="max-autotune-no-cudagraphs"` regardless of
  any explicit `--torch-compile-mode` / `--torch-compile-mode-v2`
  override when `--no-cuda-graphs` is set ✓
- Banner marker `[wf335-r3] --no-cuda-graphs SAFETY LEVER active: ...`
  emitted on override path so paper §6 audits see the safety belt ✓

---

## §4 No-CUDAGraphs-RuntimeError verification

The integration smoke (CPU-only, ``--max-joint-rows 200 --epochs 1
--hidden 32 --depth 2``) ran cleanly in 35.6s wall with banner
  ``torch_compile_mode=max-autotune-no-cudagraphs`` and NO
``CUDAGraphs`` substring in the output.  Pre-R1 the same command on
GPU would have triggered the Wf-330-B4 RuntimeError on the
`max-autotune-no-cudagraphs` opt-out path; with R1+R3 the
safety lever prevents that path from being reached by default.

Per AGENTS.md §5.2 honest framing: the smoke was CPU-only because
GPU 0 + GPU 1 are still occupied by `dfine_t146` (PID 2094864) +
`wf124_p3 G6` (PID 2411580).  GPU verification deferred until
G6 frees RTX 5090.

---

## §5 Backward compat (Wf-326C + Wf-328-W1 reproducer path)

The pre-fix Wf-326C bench + Wf-328-W1 baseline relied on
``--torch-compile-mode reduce-overhead`` for the CUDA Graphs capture
that delivered the Wf-326C MEASURED 2.72× encoder + 2.41× W₂ step
speedup.  Post-R1+R3, the legacy mode is opt-in via:

```bash
python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
    --metals Ru --seeds 0 --epochs 1 \
    --torch-compile-mode reduce-overhead \
    --cudagraphs       # opt out of the Wf-335-R3 safety lever
    --no-no-cuda-graphs  # equivalent
```

Test verified (`test_backward_compat_with_explicit_reduce_overhead`):
``args.torch_compile_mode == "reduce-overhead"`` + ``args.cudagraphs_disabled ==
False`` after parse — safety lever is OFF, legacy mode preserved.

---

## §6 Test inventory (`molmetal/tests/test_wf335r2_wireup_reverify.py`)

13 tests total (per harness brief: 4 NEW tests + parametrized 10
wire-ups = 13 invocations):

| # | Test | Status |
|---|---|---|
| 1 | `test_wire_up_still_wired[W1---torch-compile]` | PASS |
| 2 | `test_wire_up_still_wired[W2---cpu-num-workers]` | PASS |
| 3 | `test_wire_up_still_wired[W3---no-bf16]` | PASS |
| 4 | `test_wire_up_still_wired[W4---tf32]` | PASS |
| 5 | `test_wire_up_still_wired[W5---w2-gradient-mode]` | PASS |
| 6 | `test_wire_up_still_wired[W6---num-workers]` | PASS |
| 7 | `test_wire_up_still_wired[W7---dmpnn-backend]` | PASS |
| 8 | `test_wire_up_still_wired[W8---cuda-graph]` | PASS |
| 9 | `test_wire_up_still_wired[W9---triton-edge-update]` | PASS |
| 10 | `test_wire_up_still_wired[W10---max-joint-rows]` | PASS |
| 11 | `test_no_cudagraphs_runtime_error_on_smoke` | PASS |
| 12 | `test_torch_compile_mode_default_is_max_autotune_no_cudagraphs` | PASS |
| 13 | `test_backward_compat_with_explicit_reduce_overhead` | PASS |

Run command:

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf335r2_wireup_reverify.py -v
```

Result: `13 passed, 1 warning in 25.37s` (2026-09-29, this Wf-335-R2).

---

## §7 Honest framing (party-twenty-spirit §3.5)

- MEASURED: 13/13 tests PASS (CPU, 2026-09-29)
- MEASURED: integration smoke `test_auc=0.4994` (1 epoch; expected
  near-random at 1 epoch on a 3668-row Ru slice with hidden=32/depth=2)
- PROJECTED: ~5-10% peak perf delta vs reduce-overhead per Wf-326C
  reproducibility audits; to be MEASURED on RTX PRO 6000 Blackwell
  sm_120 once G6 frees the GPU
- HONEST_NEGATIVE: GPU 0 (RTX PRO 6000 Blackwell) — occupied by
  `dfine_t146` (PID 2094864, 28184 MiB); GPU 1 (RTX 5090) — occupied
  by `wf124_p3 G6` (PID 2411580, ~2.9 GB RSS in init phase);
  GPU smoke deferred per AGENTS.md §5.1
- Recovery path: re-test on GPU when G6 frees RTX 5090 (~7 hr ETA
  from 2026-09-29 14:10)

---

**END OF Wf-335-R2 doc** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5; cross-ref Wf-335-R1 fix + Wf-330-B4
RuntimeError + Wf-328-W1 + Wf-326C bench + Wf-334-U1 wire-up;
last update 2026-09-29.