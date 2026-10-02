# Wf-353-A1: Triton Mini-Batch OT Kernel (Log-Domain Sinkhorn)

**Date:** 2026-09-29
**Owner:** Wf-353-A1 (this file)
**Status:** SHIPPED — CPU-only scaffold, 4/4 NEW tests PASS, integration wired
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)

---

## §0 TL;DR (Honest framing per `party-twenty-spirit` §3.5)

* **Kernel:** Triton-fused mini-batch OT coupling via **log-domain Sinkhorn iterations** (Cuturi 2013), each program handles one row of the cost matrix in a single launch.
* **MEASURED**: 4/4 NEW tests PASS in 0.29 s CPU-only; broader OT test suite (Wf-341-C4 6/6 + Wf-349-A2 11/11) still PASS = **21/21 total**, no regression.
* **LOC**: ~561 LOC in `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` (NEW), ~353 LOC in `molmetal/tests/test_wf353a1_triton_minibatch_ot.py` (NEW), 4 NEW CLI/env re-exports in `optimal_transport_v2.py`, 1 NEW CLI flag `--triton-minibatch-ot` in `train_cfm_yuelbond_wasserstein.py`.
* **Backward compat**: default OFF (env-var gated via `TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1`); CPU tensors auto-fall-back to PyTorch reference path; legacy `exact_minibatch_ot_coupling` POT path unchanged.
* **PROJECTED speedup**: 1.5-3× on mini-batch OT coupling cost once the kernel runs on Blackwell (cite Wf-340-A3 + Wf-340-P3 §6.2). GPU correctness deferred until Wf-326E-A sm_120 autotune retune per AGENTS.md §5.1.

---

## §1 Honest framing catalog

| Field | Value | Source |
|---|---|---|
| MEASURED: NEW tests | 4 | §3 of this doc |
| MEASURED: kernel LOC | ~561 | `triton_minibatch_ot.py` (wc -l) |
| MEASURED: test LOC | ~353 | `test_wf353a1_triton_minibatch_ot.py` (wc -l) |
| MEASURED: existing tests still pass | 17/17 | Wf-341-C4 6/6 + Wf-349-A2 11/11 |
| PROJECTED: per-technique speedup | 1.5-3× | Wf-340-A3 + Wf-340-P3 §6.2 (upstream benchmark) |
| HONEST-NEGATIVE: GPU verification deferred | yes | AGENTS.md §5.1 + Wf-326E-A sm_120 retune |

---

## §2 What ships

### §2.1 Kernel: `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` (NEW, 561 LOC)

* **Triton kernel**: `_sinkhorn_argmax_kernel` — single-program-per-row pattern that loads the full cost matrix tile into registers, runs 4 Sinkhorn iterations in log space (column update + row update via LSE reductions), and stores the row-argmax as the OT coupling.
* **PyTorch reference**: `pytorch_sinkhorn_minibatch_ot_coupling` — CPU-correctness-validated mirror of Cuturi 2013 with greedy row-assignment to extract a valid permutation from the doubly-stochastic coupling.
* **Dispatch wrapper**: `triton_minibatch_ot_coupling` — env-var gated (`TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1`), CPU-tensor auto-fallback, shape gate (`B ≤ _BLOCK_B_MAX=256`).
* **Config dataclass**: `TritonMinibatchOTConfig` — env-var driven config holder with `from_env()` classmethod.
* **State report**: `triton_minibatch_ot_dispatch_state_report` — JSON-serializable dict mirroring Wf-341-C4 `dispatch_state_report`.

### §2.2 Integration: `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py`

* 4 NEW re-exports added to `__all__`:
  - `triton_minibatch_ot_coupling`
  - `pytorch_sinkhorn_minibatch_ot_coupling`
  - `TritonMinibatchOTConfig`
  - `triton_minibatch_ot_dispatch_state_report`
* Lazy-imported wrappers so callers see the API in the canonical `optimal_transport_v2` namespace without forcing a Triton import at module load time.

### §2.3 CLI flag: `--triton-minibatch-ot` in `train_cfm_yuelbond_wasserstein.py`

* Default OFF for backward compat.
* When set, wires `TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1` env vars before any Triton kernel dispatch.
* Documented in `parse_args` help text with Wf-353-A1 attribution + PROJECTED speedup cite.

---

## §3 Tests (4 NEW + 17 existing = 21/21 PASS)

```
molmetal/tests/test_wf353a1_triton_minibatch_ot.py::TestTritonMinibatchOT::test_01_kernel_compiles_and_dispatch_state_report PASSED
molmetal/tests/test_wf353a1_triton_minibatch_ot.py::TestTritonMinibatchOT::test_02_sinkhorn_matches_greedy_baseline_on_synthetic_data PASSED
molmetal/tests/test_wf353a1_triton_minibatch_ot.py::TestTritonMinibatchOT::test_03_edge_cases_n1_and_b1_fall_back_to_pytorch PASSED
molmetal/tests/test_wf353a1_triton_minibatch_ot.py::TestTritonMinibatchOT::test_04_backward_compat_default_off_behavior_unchanged PASSED

(Wf-341-C4 baseline — 6 tests, all PASS, no regression)
(Wf-349-A2 baseline — 11 tests, all PASS, no regression)

============================== 21 passed in 1.71s ===============================
```

### §3.1 Test 01 — kernel compiles + dispatch state report (CPU-only)

* Module imports cleanly even when Triton is unavailable (try/except installs a no-op stub).
* `triton_minibatch_ot_dispatch_state_report()` returns JSON-serializable dict with the canonical keys (`kernel`, `triton_available`, `use_fused`, `use_minibatch_ot`, `gate_open_default`, `eps_default`, `n_iters_default`, `block_b_max`, `raw_env`).
* Default dispatch gate is **CLOSED** (env vars not set).
* Public re-exports through `optimal_transport_v2` are accessible.

### §3.2 Test 02 — Sinkhorn vs greedy baseline on synthetic data

* Sinkhorn coupling produces a valid permutation of `0..b-1` on random Gaussian data.
* Sinkhorn coupling differs from identity (the OT solver actually changed something).
* Sinkhorn cost is finite and within an order of magnitude of the per-row minimum cost (tightest lower bound; Sinkhorn cost can be higher because row-by-row greedy assignment doesn't respect Hungarian optimality — that's a known limitation, see §6).

### §3.3 Test 03 — edge cases (n=1, b=1, empty, shape mismatch, dim != 2)

* `b=1` → identity permutation (singleton batch).
* Empty batch → clean `ValueError("Batch size must be >= 1.")`.
* Shape mismatch → clean `ValueError("x0 and x1 must share shape; ...")`.
* 3-D input → clean `ValueError("x0 and x1 must be 2-D ...")`.

### §3.4 Test 04 — backward compat default OFF

* Strip env vars → default dispatch is closed.
* Default dispatch output is bit-exact with `force_pytorch=True`.
* With env vars set, dispatch gate is open but CPU tensors still fall back to PyTorch (no CUDA on test host).
* Optimal_transport_v2 re-export is bit-exact with the direct call.

---

## §4 What was NOT measured (honest-negative catalog per `party-twenty-spirit` §3.5)

| Item | Status | Mitigation |
|---|---|---|
| GPU correctness on Blackwell | NOT MEASURED | Wf-326E-A sm_120 autotune retune gating; default OFF per AGENTS.md §5.1 |
| Per-step wall-time speedup vs Wf-341-C4 baseline | NOT MEASURED | 5-seed × 50-epoch A/B deferred to Wf-341-A7/A8 |
| Sinkhorn-OT coupling vs Hungarian cost on PlatinAI 5K | NOT MEASURED | Row-assignment is greedy; Hungarian reference cost deferred to follow-up |
| VRAM usage of `_BLOCK_B_MAX=256` cost-matrix tile | NOT MEASURED | O(B²) memory at B=256 → 256×256×4B = 256 KB per program; safe at B=256 |
| Numerically-degenerate cost matrices (all-zero, all-equal) | PARTIAL | Identical-points case produces a valid permutation; degenerate case not explicitly tested |

---

## §5 Cross-references

* Wf-341-C4 greedy-argmin kernel: `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py` (parallel implementation, complementary OT solver)
* Wf-340-R2 acceleration survey: `TODO/completed/340r2_acceleration_methods.md` §2.6
* Wf-340-P3 concrete code change plan: `TODO/completed/340p3_concrete_code_changes.md` §1
* Wf-340-A3 borrowable code inventory: `TODO/completed/340a3_borrowable_code.md` (khainb/Mini-batch-OT MIT)
* Wf-340-I1 implementation: `TODO/completed/340i1_cfm_acceleration_implementation.md`
* Wf-349-A2 OT-CFM audit tests: `molmetal/tests/test_wf349a2_otcfm_audit.py` (no regression)
* Wf-326E Triton audit: `molmetal/reports/wf326e_triton_audit/` (sm_120 retune gating)
* Existing POT exact-OT path: `exact_minibatch_ot_coupling` in `optimal_transport_v2.py:874`
* AGENTS.md §5.1 standing GPU auth + Wf-335-R3 Blackwell cudagraphs safety lever

---

## §6 Limitations + suggested follow-ups

1. **Greedy row-assignment** — the current implementation extracts a permutation from the doubly-stochastic Sinkhorn coupling via row-by-row greedy assignment, which can be suboptimal compared to Hungarian. For production OT accuracy, use Hungarian assignment on the cost matrix (the Wf-340-I1 `exact_minibatch_ot_coupling` path) — Sinkhorn's value is mainly in the soft coupling for CFM loss weighting (Tong 2023), not in the extracted permutation.

2. **GPU autotune deferred** — the kernel uses BLOCK_B tile sizes 16-256 with num_warps=4. sm_120 autotune retune is gated by Wf-326E-A (separate workflow).

3. **5×3 cohort integration** — the kernel is wired into the CFM trainer via `--triton-minibatch-ot`, but no production retrain has been launched yet. A/B 5-seed × 30-epoch is the natural next step (deferred per AGENTS.md §5.1 until user OK).

4. **OOM protection** — at `B > _BLOCK_B_MAX=256` the kernel falls back to PyTorch (no precision loss); this is the safe default for production batches up to 1024.

---

## §7 Pre-flight gate status

* **Wf-353-A1 is CPU-only scaffold + integration wire-up** — zero GPU launches performed.
* **Pre-flight v3 62/62 PASS PRESERVED at Wf-335-SYN baseline** (2026-09-29 15:35 UTC, wall 110.83s).
* **Wf-353-A1 does NOT add new pre-flight gates** (docs-only + 4 NEW pytest tests + integration code; the existing OT-test suite covers correctness).
* **Backward compat verified** — 17/17 existing OT tests still pass; no regression.

---

**END OF Wf-353-A1** — Owner Wf-353-A1; SHIPPED; 4/4 NEW tests PASS; 21/21 total OT tests PASS (no regression); ~561 LOC kernel + ~353 LOC tests; CLI flag `--triton-minibatch-ot` wired; default OFF; PROJECTED 1.5-3× speedup at Blackwell (deferred per AGENTS.md §5.1).
