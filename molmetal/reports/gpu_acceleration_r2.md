# GPU acceleration round 2 — wiring + batching verification

Date: 2026-09-12
Operator: Claude (sonnet 4.6)
GPU: AMD Radeon 780M (gfx1101, RDNA3, 16 GiB VRAM) — iGPU inside Ryzen 9 8945HS
ROCm driver: 7.2.4-zen2-1-zen
Torch: 2.14.0+rocm7.2
Triton: 3.8.0 (rocm build)

## Scope

This round wires four **new Triton kernels** and the **batched RDKit
helper module** so they are importable, parity-tested against PyTorch
references, and usable from any production code path that wants them.
No production call site is force-rewired — the new kernels are
**library-grade additions** with parity verified on a GPU/CPU fallback
path; downstream callers may opt in at their own pace.

Constraints honoured from `TODO/environment.md`:

- `num_warps` + `num_stages` only inside `@triton.autotune` (no
  `waves_per_eu`).
- No TMA / WGMMA / cluster launch / warp specialization.
- `from triton_kernels import ...` convention only — `import triton`
  is confined to leaf kernel modules.
- Existing autograd shims preserved (`aggregate_vectors`, `euler_step`,
  `rk4_step`, `fused_layer_norm`, `fused_rms_norm`).
- Spot-check only — no full sweep / benchmark run.

## 1. New Triton kernels wired

All four are imported and parity-tested under
`tests/test_new_kernels.py`.

| Module | Public API | Purpose |
|---|---|---|
| `triton_kernels.fused_dropout` | `fused_dropout_residual(x, residual, p, *, seed=None, offset=0)` | Fused `dropout(x) + residual` with deterministic Philox mask, autograd-replayable |
| `triton_kernels.fused_residual_add` | `fused_residual_add(x, residual, alpha=1.0, beta=1.0)` | Fused `α·x + β·residual` with trailing-axis broadcast + fused backward |
| `triton_kernels.batched_mlp` | `batched_fused_silu_mlp(x, w1s, b1s, w2s, b2s)` | K stacked SiLU MLPs in a single launch (sum across K) |
| `triton_kernels.batched_mlp` | `batched_fused_gelu_mlp(x, w1s, b1s, w2s, b2s)` | K stacked GELU-tanh MLPs in a single launch (sum across K) |

Each kernel ships with:

- `@triton.autotune` over `AUTOTUNE_CONFIGS` (`num_warps`, `num_stages` only).
- `_check_cuda_pointers` host-side guard — raises a friendly error if
  the caller routes a CPU tensor through (closes the TODO-T2 gap for
  the four new kernels).
- A `torch.autograd.Function` wrapper so the kernels integrate with
  any nn.Module forward.
- A CPU fallback path so the test suite still runs on a CPU-only host
  (and so the test suite can collect on CI sandboxes without ROCm).

`fused_dropout_residual` and `fused_residual_add` are **memory-bound
elementwise** — they cut one HBM round-trip relative to a torch
two-step (load x + load residual + store out). On gfx1101 the absolute
saving is small (single-digit μs at the harness sizes) but
production-shaped transformer blocks benefit because the kernel fuses
cleanly with the surrounding pointwise activation.

`batched_fused_silu_mlp` and `batched_fused_gelu_mlp` are
**launch-fusion** kernels (K distinct weight stacks, single launch)
used to amortize per-launch overhead on gfx1101 where wave-scheduler
setup dominates the per-MLP compute. The current public API still
sums the per-K outputs because the call site that motivated the
kernel (per-atom-type / per-edge-type heads in the EGNN conditioning
block) sums across heads after the per-head MLP. The `_BatchedMLPFunction`
backward path matches the sum-over-K semantics exactly.

## 2. RDKit operations batched

`molmetal/molmetal_lam/lam_chem/batched_rdkit.py` provides
intra-batch parallelism via `multiprocessing.Pool` ("fork" context)
with a serial fallback for environments where the pool fails
(sandboxed CI, Jupyter, Windows spawn re-entrancy). Every helper
returns numpy arrays in the caller's original SMILES order, so the
existing scalar call sites can be swapped one-for-one.

| Helper | Scalar equivalent | Returns |
|---|---|---|
| `batch_mol_from_smiles(smiles_list, n_workers=0)` | `Chem.MolFromSmiles(s)` per item | `list[Chem.Mol \| None]` |
| `batch_descriptors(mols, descriptor_name, n_workers=0)` | `_scalar_descriptor` | `np.ndarray[float]` |
| `batch_descriptors_from_smiles(smiles_list, descriptor_name, n_workers=0)` | `_scalar_descriptor` | `np.ndarray[float]` |
| `batch_qed(smiles_list, n_workers=0)` | `Descriptors.qed(mol)` | `np.ndarray[float]` |
| `batch_sa(smiles_list, n_workers=0)` | `_sa_proxy_one` (Ertl-style SA surrogate, MW+logP+RotB) | `np.ndarray[float]` |
| `batch_lipinski(smiles_list, n_workers=0)` | `_lipinski_one` (Ro5 boolean) | `np.ndarray[bool]` |
| `is_parallel_safe()` | — | `bool` (cached) |

Why not RDKit's `SA_Score` contribution? — it ships as an *optional*
contribution and is not bundled with the `rdkit-pypi` wheels. The
Ertl-style proxy used here matches `_sweep_helpers._sa_proxy` exactly
so callers see numerically identical results to the scalar path.

The 28 tests in
`molmetal/molmetal_lam/tests/test_batched_rdkit.py` cover:

- Length preservation, order preservation, empty input.
- Parity against the scalar `_scalar_descriptor` for `MW`, `LogP`,
  `qed`, SA proxy, Lipinski Ro5 mask.
- NaN / False returns on un-parseable SMILES.
- Sequential fallback when `multiprocessing.Pool` is unavailable
  (`monkeypatch.setattr` on `is_parallel_safe`).
- Case-insensitive descriptor names.
- `pool.map` failure → sequential fallback path.

## 3. Estimated throughput change (qualitative)

**No measurement was run** in this round (per the spot-check-only
constraint). The qualitative estimates below are extrapolations from
the r0 baseline (`molmetal/reports/triton_integration_r0.md`):

| Path | Before r2 | After r2 | Expected Δ (qualitative) |
|---|---|---|---|
| MCTS reward aggregation (100s of SMILES per `_batch_rewards` step) | One `Pool.starmap` task per candidate → one C++ binding round-trip per candidate | `batch_qed` / `batch_lipinski` / `batch_descriptors_from_smiles` → one pool dispatch per batch, K candidates per worker task | ~2–5× speedup on 4+ cores at batch sizes ≥64 (binding overhead dominates); parity at batch ≤8 (overhead floor) |
| Transformer / D-MPNN residual step | `out = torch.dropout(x) + res` then `out = out + bias` (two elementwise launches) | `fused_dropout_residual(x, res)` then `fused_residual_add(out, bias)` (one launch each) | ~1.3–1.5× speedup on elementwise-bound shapes (single HBM pass + single fused activation); parity on compute-bound shapes |
| EGNN per-atom / per-edge head MLPs | K separate `nn.Sequential` calls (K kernel launches) | `batched_fused_silu_mlp(x, w1s, b1s, w2s, b2s)` (one launch for K stacked MLPs) | ~K× launch-overhead reduction on gfx1101 (per-launch ~10–30 μs); parity on compute-bound shapes |
| General additive residual (ResNet-style, additive bias) | `torch.add(x, residual, alpha=...)` (one launch, scalar path) | `fused_residual_add(x, residual, alpha, beta)` (one launch, fused) | parity-to-marginal (≤10% faster); main win is single-launch fusion with downstream pointwise activation |

The batched RDKit helpers are expected to give the largest **practical**
speedup because RDKit's C++ binding round-trip is the dominant cost at
small-molecule sizes, not the descriptor math itself. A micro-bench of
`batch_qed([100 SMILES])` vs a serial loop is the obvious next-step
measurement, but it was not run in this round.

## 4. Verification summary

```
tests/test_batched_rdkit.py          28 / 28 pass
tests/test_new_kernels.py             9 /  9 pass   (parametrized: 27 cases)
tests/test_fused_ops.py               8 /  8 pass   (parametrized: 14 cases)
─────────────────────────────────────────────────────
total new test files                 45 / 45 pass
```

Import smoke test:

```
uv run python -c "from triton_kernels.fused_dropout import fused_dropout_residual; \
from triton_kernels.fused_residual_add import fused_residual_add; \
from triton_kernels.batched_mlp import batched_fused_silu_mlp, batched_fused_gelu_mlp; \
print('OK')"
→ OK
```

RDKit batched smoke test:

```
uv run python -c "from molmetal_lam.lam_chem.batched_rdkit import batch_qed, batch_lipinski; \
import numpy as np; \
print(batch_qed(['CCO', 'c1ccccc1'])); \
print(batch_lipinski(['CCO', 'c1ccccc1']))"
→ [0.40680797 0.44262837]
→ [ True  True]
```

Lambda + fused-ops full suite:

```
uv run pytest -q --ignore=molmetal/references molmetal/molmetal_lam/tests/ tests/test_fused_ops.py --tb=short
→ 261 passed, 6 failed in 56.19s
```

The 6 failures are pre-existing (none introduced by this round):

1. `test_baselines.py::test_compare_all_methods_runs` — drift in
   downstream LM frag-counter thresholds (`synthesis_success in
   (0.78, 1.0)`, got 0.75).
2. `test_baselines.py::test_lambda_sas_best` — Lambda SAS 2.63 vs
   DiffSBDD SAS 1.67 (numerical, unrelated to Triton wiring).
3. `test_baselines.py::test_predict_pic50_smoke` — NaN smoke
   assertion.
4. `test_baselines.py::test_sas_score_smoke` — `sas_score("CCO")`
   returns 1.98, not 1.0 (RDKit SA-score formula change vs the
   Ertl-style proxy the test asserts against).
5. `test_sweep_helpers.py::test_lipinski_pass_empty_string_returns_false`
   — semantic disagreement on `lipinski_pass("")` (empty SMILES
   semantics; not a Triton kernel).
6. `test_tile_properties.py::test_property_all_tiles_unique_smiles`
   — 6 duplicate SMILES in the static fragment library (data, not
   code).

All 6 failures trace to baseline / data / semantics issues; none
trace to `triton_kernels.*` or `molmetal_lam.lam_chem.batched_rdkit`.

## 5. New TODOs discovered

### TODO-G2: opt-in call sites for the four new kernels

The four kernels are library-grade and tested, but no production
call site currently uses them. Candidates for opt-in rewiring:

- `molmetal/models/dmpnn.py` residual tail — switch from
  `nn.Dropout + nn.Linear` residual to
  `fused_dropout_residual + fused_residual_add`.
- `molmetal/models/velocity_net.py` EGNN conditioning heads — three
  per-atom / per-edge / global MLPs could fold into one
  `batched_fused_silu_mlp(K=3)` call.
- `molmetal_lam/scripts/r4_c_full_sweep.py` MCTS reward aggregation
  — switch `_batch_rewards` step from per-candidate
  `Pool.starmap` to `batch_qed` / `batch_lipinski` /
  `batch_descriptors_from_smiles`.

### TODO-G3: micro-benchmark the batched RDKit helpers

The expected throughput gain (2–5×) is qualitative. A direct
micro-bench of `batch_qed(N=100)` vs a serial loop over
`Descriptors.qed(mol)` should land in `molmetal/reports/` and
update the r2 report.

### TODO-G4: r3 follow-up — `batched_fused_*_mlp` per-K outputs

The current API sums the per-K outputs. Several downstream paths
(per-atom-type message passing in the D-MPNN, multi-task regression
heads) want the per-K outputs preserved. Adding a
`batched_fused_silu_mlp_per_k` variant (returning `(K, M, H2)`)
unlocks those rewires without breaking the sum-over-K default.

## 6. Summary

- **4 new Triton kernels** wired and parity-tested:
  `fused_dropout_residual`, `fused_residual_add`,
  `batched_fused_silu_mlp`, `batched_fused_gelu_mlp`. All CPU-safe
  (transparent fallback) and GPU-accelerated via `@triton.autotune`.
- **7 batched RDKit helpers** wired under
  `molmetal_lam.lam_chem.batched_rdkit`: `batch_mol_from_smiles`,
  `batch_descriptors`, `batch_descriptors_from_smiles`, `batch_qed`,
  `batch_sa`, `batch_lipinski`, `is_parallel_safe`. All multiprocessing
  with serial fallback; all 28 parity / order / NaN / fallback tests
  pass.
- **45 / 45** new test cases pass (combined across
  `test_batched_rdkit.py`, `test_new_kernels.py`,
  `test_fused_ops.py`).
- **Lambda + fused-ops full suite**: 261 pass, 6 pre-existing
  failures (data / semantics, not Triton).
- **No production call site was rewired** — all kernels are
  opt-in library additions; the opt-in TODOs (TODO-G2, TODO-G3,
  TODO-G4) are listed for the next round.
- **Spot-check only** — no full sweep or benchmark was run.

## 7. Round 6 — wiring + algorithm fixes (verification only, no sweep)

Date: 2026-09-12
Operator: Claude (sonnet 4.6)
Constraints honoured: spot-check only, no sweep, no benchmark, no new deps.

### 7.1 Kernels wired

| Kernel | Call site(s) wired | Test file |
|---|---|---|
| `fused_silu_mlp` (triton_kernels.fused_mlp) | EGNN adapter, `train_fm_pocket.py`, `flow_matching_lipman` time_mlp | `molmetal/tests/test_fused_silu_mlp_wired.py` |
| `softmax_last_dim` (triton_kernels.fused_softmax) | Dirichlet sites in `proof_search.py` | `molmetal/molmetal_lam/tests/test_dirichlet_fused_softmax.py` |
| `aggregate_vectors` (triton_kernels.equivariant_ops) | EGNN scatter (T1-gated) | `molmetal/tests/test_egnn_coord_scatter_wired.py` |
| `rotation_from_axis_angle` (triton_kernels.equivariant_ops) | EGNN coord update | `molmetal/tests/test_egnn_coord_scatter_wired.py` |

### 7.2 Data fixes

- **6 duplicate SMILES in `fragment_pool.py` → distinct RDKit-equivalent replacements**
  (closes the data-side cause of the pre-existing
  `test_tile_properties.py::test_property_all_tiles_unique_smiles`
  failure from round 2).
- **Tile canonicalisation cache**: `molmetal/molmetal_lam/tile_lib/canonical_cache.py`
  ships `canonicalize(...)` + `precompute_library(...)` and is wired
  into the `proof_search.py` round-trip paths. Covered by
  `test_tile_canonical_cache.py` + `molmetal/tests/test_duplicate_tile_fix.py`.

### 7.3 Algorithm fixes

- **R3 PoseBusters + MMFF94**: `molmetal/validation/posebusters_runner.py`
  ships `check_posebusters(...)`. Wired into `RewardAggregator` and
  `closed_loop.py`. Covered by `test_posebusters_runner.py`.
- **MCTS VirtualLoss + TranspositionTable + recycle**:
  `_VirtualLoss` and `_TranspositionTable` ship in `proof_search.py`
  and are wired into `MCTSProofSearch`. Covered by
  `test_mcts_vloss_tt.py`.

### 7.4 Property tests

- `molmetal/molmetal_lam/tests/test_mcts_properties.py` — hypothesis-driven
  property tests for MCTS invariants.

### 7.5 Verification spot-check (no sweep, no benchmark)

```
uv run pytest -q --ignore=molmetal/references \
  molmetal/molmetal_lam/tests/test_tile_canonical_cache.py \
  molmetal/molmetal_lam/tests/test_posebusters_runner.py \
  molmetal/molmetal_lam/tests/test_mcts_vloss_tt.py \
  molmetal/molmetal_lam/tests/test_mcts_properties.py \
  molmetal/molmetal_lam/tests/test_dirichlet_fused_softmax.py \
  molmetal/tests/test_fused_silu_mlp_wired.py \
  molmetal/tests/test_egnn_coord_scatter_wired.py \
  molmetal/tests/test_duplicate_tile_fix.py \
  tests/test_fused_ops.py \
  tests/test_new_kernels.py \
  --tb=short
→ 78 passed in 19.95s
```

Per-file (collected from the parametrized run above):

| File | Pass |
|---|---|
| `molmetal/molmetal_lam/tests/test_tile_canonical_cache.py` | pass |
| `molmetal/molmetal_lam/tests/test_posebusters_runner.py` | pass |
| `molmetal/molmetal_lam/tests/test_mcts_vloss_tt.py` | pass |
| `molmetal/molmetal_lam/tests/test_mcts_properties.py` | pass |
| `molmetal/molmetal_lam/tests/test_dirichlet_fused_softmax.py` | pass |
| `molmetal/tests/test_fused_silu_mlp_wired.py` | pass |
| `molmetal/tests/test_egnn_coord_scatter_wired.py` | pass |
| `molmetal/tests/test_duplicate_tile_fix.py` | pass |
| `tests/test_fused_ops.py` | pass |
| `tests/test_new_kernels.py` | pass |
| **total** | **78 / 78 pass** |

Import smoke test:

```
from triton_kernels.fused_mlp import fused_silu_mlp                # OK
from triton_kernels.fused_softmax import softmax_last_dim          # OK
from triton_kernels.equivariant_ops import aggregate_vectors, rotation_from_axis_angle  # OK
from molmetal_lam.tile_lib.canonical_cache import canonicalize, precompute_library      # OK
from molmetal.validation.posebusters_runner import check_posebusters                     # OK
from molmetal_lam.search_alg.proof_search import MCTSProofSearch, _VirtualLoss, _TranspositionTable  # OK
→ OK
```

(`MCTSProofSearch` is a dataclass requiring `tile_library`, `rules`,
`target_predicates`, `binding_site` — it cannot be default-constructed
in the smoke test, which is correct for a search-object API. The
newly-wired fields `_VirtualLoss.value` and `_TranspositionTable._max_size`
were instantiated and inspected.)

### 7.6 Estimated throughput change (qualitative, NOT measured)

| Path | Before r6 | After r6 | Expected Δ (qualitative) |
|---|---|---|---|
| EGNN message MLP forward | `nn.Sequential` MLP per atom/edge | `fused_silu_mlp` autotuned Triton kernel | ~1.5–3× speedup at H=128-512 (single-launch + autotune-tuned `BLOCK_H`); parity at small H (overhead floor) |
| EGNN coord update | `torch.einsum` rotation matrix | `rotation_from_axis_angle` Triton | ~1.2–1.8× speedup (fewer intermediate materializations); parity for tiny graphs |
| EGNN scatter (T1-gated) | `torch.scatter_add_` over neighbours | `aggregate_vectors` Triton, gated by `T1` | ~1.3–2× speedup when N_edges ≫ N_nodes (single fused pass vs scatter+index); parity when T1=False (PyTorch path) |
| MCTS Dirichlet-noise root prior | `torch.softmax` + numpy alpha mixing | `softmax_last_dim` Triton | marginal (~5-15% on the softmax itself, which is small relative to MCTS step); main win is single-launch fusion with downstream sampling |
| MCTS leaf evaluation | naive `Q + c_puct * ...` with no virtual loss or recycle | `_VirtualLoss` + `_TranspositionTable` + recycle | ~1.5–3× simulation throughput (avoids re-evaluating transpositions; virtual loss prevents two sims from re-entering the same path) — qualitative, depends on branching factor |
| Tile canonicalisation on hot path | RDKit round-trip per candidate | `canonicalize` + `precompute_library` cache | ~10–100× speedup on cached tiles (in-memory hash lookup vs RDKit C++ binding round-trip); parity for cache misses |
| Reward aggregator (R3) | `QED + SA + Lipinski` only | `+ PoseBusters + MMFF94` geometric / steric checks | wall-time cost **increases** (PB is heavy); reward-signal quality **increases** — fewer invalid poses survive filtering, fewer wasted closed-loop steps |

### 7.7 Files added / verified

- `molmetal/molmetal_lam/tile_lib/canonical_cache.py` (added)
- `molmetal/validation/posebusters_runner.py` (added)
- `molmetal/molmetal_lam/tests/test_tile_canonical_cache.py` (added)
- `molmetal/molmetal_lam/tests/test_posebusters_runner.py` (added)
- `molmetal/molmetal_lam/tests/test_mcts_vloss_tt.py` (added)
- `molmetal/molmetal_lam/tests/test_mcts_properties.py` (added)
- `molmetal/molmetal_lam/tests/test_dirichlet_fused_softmax.py` (added)
- `molmetal/tests/test_fused_silu_mlp_wired.py` (added)
- `molmetal/tests/test_egnn_coord_scatter_wired.py` (added)
- `molmetal/tests/test_duplicate_tile_fix.py` (added)
- `triton_kernels/config.py` (existed, referenced)
- `molmetal/reports/gpu_acceleration_r2.md` (this report, section 7 appended)

### 7.8 Summary

- **All round-6 kernels wired**: `fused_silu_mlp`, `softmax_last_dim`,
  `aggregate_vectors`, `rotation_from_axis_angle`.
- **All round-6 data fixes shipped**: 6 duplicate SMILES → distinct
  replacements; tile canonicalisation cache shipped.
- **All round-6 algorithm fixes shipped**: R3 PoseBusters+MMFF94
  wired into `RewardAggregator` + `closed_loop.py`; MCTS
  `_VirtualLoss` + `_TranspositionTable` + recycle wired into
  `MCTSProofSearch`.
- **Property tests added** under `test_mcts_properties.py`.
- **78 / 78 spot-check tests pass** across 10 files.
- **No sweep, no benchmark, no new deps** (per environment constraint).

## Round-9 follow-up (2026-09-13)

Round-9 did NOT add new Triton kernels; it consolidated the GPU-side
wiring already shipped across rounds 5–8 and verified that the SOTA-aligned
closed-loop path is test-correct. Specifically: the round-8 Triton
kernels (`batched_fused_silu_mlp`, per-K variant, fused `dropout+residual`,
fused `α·x + β·residual`) are still parity-tested green
(`tests/test_new_kernels.py`: 22 pass; `tests/test_fused_ops.py`: 24 pass);
`tests/test_round5_kernels_wired.py` (8 pass) and
`tests/test_fused_silu_mlp_wired.py` (4 pass) confirm they remain wired
into the production call sites (`models/dmpnn.py` swaps
`nn.Sequential(Linear, SiLU, Linear)` → `_fused_silu_mlp`). The new
round-9 test files — `test_tmqm_shape_bridge.py`,
`test_r4c_full_sweep.py`, `test_sota_aligned.py`,
`test_tmqm_loading.py` — are pure pytest smoke tests (no Triton kernels);
they verify the tmQM shape bridge (`_shape_bridge_state_dict` in
`molmetal/adapters/flow_matching_lipman/__init__.py:270` lifts the DMPNN→
EGNN transfer from 0/42 → >40/43 keys), the r4c sweep orchestrator CLI
(`--help`/`--dry-run`), the frozen `SOTAAlignedConfig` dataclass, and
the baseline tmQM loading path. Total round-9 spot-check:
**180 pass / 1 skip / 1 xfail / 0 fail** across 25 test files in 34.40s
on the same gfx1101 / ROCm 7.2.4 / Torch 2.14.0+rocm7.2 / Triton 3.8.0
stack as this report. No new GPU throughput was measured in round-9; the
single sweep executed was the 2/5-pocket R4-C pilot, which is
CPU-bound (`MCTSProofSearch` rollout). See
`molmetal/reports/round9_final_report.md` for the consolidated
round-9 write-up.
