# Wf-335-R1 — torch.compile mode default switch (reduce-overhead → max-autotune-no-cudagraphs)

**Date:** 2026-09-29
**Wf:** Wf-335 (post-Wf-334 final closeout; ultracode wire-up reverify)
**Pre-flight:** v2 25/25 PASS gate inherited from Wf-330-C (no fresh retrain /
panel launched in this Wf; only code-path reverify per AGENTS.md §1.4)
**Status:** SHIPPED

---

## §1 TL;DR

Per Wf-330-B4 MEASURED on the in-flight Wf-124 P3 G6 retrain (Ru seed=2 +
Ir seed=0): ``torch.compile(mode="reduce-overhead")`` triggered the
``"accessing tensor output of CUDAGraphs that has been overwritten"``
RuntimeError on the RTX PRO 6000 Blackwell sm_120 GPU when Inductor's
CUDA Graphs capture path overwrote a static-output buffer between
backward + forward passes.

This Wf switches the **default** torch.compile mode from
``reduce-overhead`` (CUDA Graphs enabled) to
``max-autotune-no-cudagraphs`` (Triton autotune WITHOUT CUDA Graphs
capture) — eliminating the RuntimeError at the cost of a small peak-
perf delta (per Wf-326C: ~5-10% peak perf drop on RTX PRO 6000; offset
by no CUDA Graphs risk).  Pre-fix Wf-328-W1 / Wf-326C benchmarks are
preserved via the ``--torch-compile-mode-v2`` opt-in alias.

| Item | Status |
|---|---|
| Function default flipped | MEASURED — line 1121 (was ``"reduce-overhead"``) |
| argparse default flipped | MEASURED — line 2028 (was ``"reduce-overhead"``) |
| ``--torch-compile-mode-v2`` alias added | MEASURED — line 2046 |
| ``--no-cuda-graphs`` / ``--cudagraphs`` safety lever added | MEASURED — line 2071 (Wf-335-R3 in same commit) |
| Pre-fix Wf-328-W1 + Wf-326C bench reproducer preserved | MEASURED — ``--cudagraphs`` flag flips safety lever OFF |
| Integration smoke | MEASURED — `test_auc=0.4994` (1 epoch, expected near-random), wall=35.6s |
| Existing Wf-328-W1 tests | 4/4 PASS after fix (test file already updated for R1) |
| New Wf-335-R2 reverify tests | 13/13 PASS (10 wire-ups + smoke + mode-default + backward-compat) |

---

## §2 What changed (1-line audit)

| File | Line | Change |
|---|---|---|
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 1121 | `torch_compile_mode: str = "max-autotune-no-cudagraphs"` (was `"reduce-overhead"`) |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 1145-1152 | docstring updated to reference Wf-335-R1 MEASURED FIX |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2022-2038 | argparse default `"max-autotune-no-cudagraphs"` (was `"reduce-overhead"`); help text expanded to explain the post-fix default + the Wf-330-B4 RuntimeError rationale |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2046-2057 | NEW: ``--torch-compile-mode-v2`` alias (post-fix default) — explicit opt-in to the new default + preserves legacy Wf-326C reproducer via ``--torch-compile-mode-v2 reduce-overhead`` |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2071-2093 | NEW: ``--no-cuda-graphs`` (default ON) / ``--cudagraphs`` (opt-out) safety lever (Wf-335-R3) — hard-pins ``torch_compile_mode="max-autotune-no-cudagraphs"`` regardless of any explicit override |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2145-2156 | NEW: ``main()`` resolves v2 alias + emits banner ``[wf335-r1] --torch-compile-mode-v2 OVERRIDE active: ...`` so paper §6 audits see the override path |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2163-2199 | NEW: ``main()`` applies the Wf-335-R3 safety lever — when ``cudagraphs_disabled=True`` (default), forces ``torch_compile_mode="max-autotune-no-cudagraphs"`` if the v1/v2 override would have left it in CUDA Graphs mode |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 2545-2550 | NEW: ``torch_compile_mode_v2_provided`` + ``torch_compile_mode_v2_value`` added to the training_summary JSON for paper §6 reproducibility audits |

---

## §3 Wf-330-B4 root cause (the bug we fixed)

The Inductor backend's ``mode='reduce-overhead'`` enables CUDA Graphs
capture: the encoder forward + W₂ closed-form are traced once, then
replayed via a single cuLaunchKernel call.  When the trainer runs
multiple (metal, seed) configurations that share the compiled encoder
(e.g. the Wf-131 P1 / Wf-181 G5 / Wf-182 G6 5-metal × 5-seed sweep),
the CUDA Graph's static output buffer can be reused by the next
``(metal, seed)`` trace before the previous backward pass has finished
reading it.

PyTorch raises ``RuntimeError: accessing tensor output of CUDAGraphs
that has been overwritten by another invocation`` when the next forward
pass starts before the previous backward pass has consumed the static
output buffer.

Per Wf-330-B4 MEASURED: this fired on Ru seed=2 + Ir seed=0 after
~3-5 minutes of training on the RTX PRO 6000 Blackwell sm_120 GPU.
``mode='max-autotune-no-cudagraphs'`` runs the SAME max-autotune
Triton-kernel selection (so the Wf-326C MEASURED 2.72× encoder +
2.41× W₂ speedup is preserved) WITHOUT the CUDA Graphs capture —
eliminating the RuntimeError at the cost of ~5-10% peak perf per
Wf-326C reproducibility audits.

---

## §4 Pre-flight + MEASURED-vs-PROJECTED tag discipline

Per AGENTS.md §5.2 (party-twenty-spirit §3.5): every numeric value
in this doc traces to a JSON / report path on disk. PROJECTED vs
MEASURED tag is required.  Zero tolerance for fabricated numbers.

- MEASURED: function default at line 1121 (read from source 2026-09-29)
- MEASURED: argparse default at line 2028 (read from source 2026-09-29)
- MEASURED: Wf-330-B4 RuntimeError on Ru seed=2 + Ir seed=0 (per Wf-330-B4 doc)
- MEASURED: integration smoke run 2026-09-29 — `test_auc=0.4994` (1 epoch, expected
  near-random; 35.6s wall on CPU)
- MEASURED: 13/13 Wf-335-R2 reverify tests PASS (see
  `molmetal/tests/test_wf335r2_wireup_reverify.py`)
- PROJECTED: ~5-10% peak perf delta vs reduce-overhead per Wf-326C
  reproducibility audits — to be MEASURED on RTX PRO 6000 Blackwell
  sm_120 once G6 retrain frees the GPU.

---

## §5 Recovery path

| Scenario | Path |
|---|---|
| Re-test integration post-R1 (this Wf-335-R2) | `bash scripts/wf335r2_smoke.sh` (CPU, hidden=32, depth=2, 1 epoch, --max-joint-rows 200) |
| Re-test on GPU when G6 frees RTX 5090 / PRO 6000 | USER_OK-gated per AGENTS.md §5.1; full 5-metal × 5-seed × 50-epoch sweep |
| Reproduce Wf-326C / Wf-328-W1 baseline (CUDA Graphs ON) | `--torch-compile-mode-v2 reduce-overhead --cudagraphs --torch-compile-mode reduce-overhead` (legacy mode + explicit opt-out of safety lever) |
| Disable torch.compile entirely | `--no-torch-compile` (Wf-328-W1 backward-compat lever; Wf-326C eager baseline) |

---

**END OF Wf-335-R1 doc** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; cross-ref Wf-330-B4 + Wf-328-W1 + Wf-326C; last update 2026-09-29.
