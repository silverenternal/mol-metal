# WF-CFM-GPU-Retrain — Phase 1: GPU sanity + diagnostic design

**Date**: 2026-09-15
**Author**: WF-CFM-GPU-Retrain
**Status**: Phase 1 (design) — Phase 2 (probe) pending
**Project root**: `/home/hugo/codes/try_triton_on_rocm`
**Stack**: uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64

---

## 1. Honest premise (no spin)

The full task spec names a script `molmetal/scripts/cfm_path_b_decoder_rework.py` that does **not** exist on disk. The Path-B decoder rework module lives at:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/decoder_rework.py`

The end-to-end CFM training script that exercises `r10_cfg_real_crossdocked.py` with `--decoder-rework` lives at:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py`

That is the script we will use for the diagnostic. It owns the `--train-steps`, `--hidden-dim`, `--seeds`, `--output-dir`, `--bond-head`, `--decoder-rework`, `--joint-train`, `--pb-relax-mmff94`, `--use-pcgrad`, `--use-tmqm-init` flags (line numbers 213-355). It does **not** expose a `--device` flag; device is auto-resolved by `LipmanFlowMatchingAdapter.setup()` via `verify_rocm_active()` + `get_device()` (line 1836-1883) and prefers an iGPU>dGPU>CPU policy.

This phase therefore documents the **probe command** for `r10_cfg_real_crossdocked.py`, not a script that does not exist.

---

## 2. GPU state — VERIFIED 2026-09-15

Output of `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` (just-now, this session):

```
cuda_available: True
device_count: 2
device_0: AMD Radeon Graphics
```

`torch.cuda.is_available()=True` and `device_count=2` are the two gates the spec calls out. Both **PASS**.

Prior `wf_gpu_recovery_now` probe (2026-09-15) showed the same state and successfully completed a 5000-step retrain in 125.10 s wall clock — see `molmetal/reports/wf_gpu_recovery_now/cfm_probe/report.json`. The GPU is therefore both **reachable** and **useful**, not merely visible to PyTorch.

Honest caveat: the previous GPU outage root cause was a firmware-level SMU hang (`wf_gpu_diag/diagnosis.md`). The 2026-09-15 recovery is **opportunistic**, not engineered — no permanent fix exists. A re-hang during this probe is possible. The diagnostic budget below is sized so that one re-hang does not strand the workflow.

---

## 3. P1 fixes — VERIFIED in place (file:line refs)

| Fix | Spec location | Verified location | Status |
|---|---|---|---|
| P1.1 hidden_dim default 128 | flow_matching_lipman/__init__.py lines 1014 + 1548 | line 1014 (`EGNNVelocityField.__init__`, `hidden_dim: int = 128`); also line 920 (`PocketEncoder`, `hidden_dim: int = 128`); line 254 (`LipmanFlowMatchingAdapter.load_pretrained_mpnn`, default 128); line 1854 emits `UserWarning` when `< 64` (P0-F4 safeguard) | **SHIP** |
| P1.2 vel_scale nn.Parameter | line 1109 | line 1128 — `self.vel_scale = nn.Parameter(torch.tensor(1.0))` with bounded-sigmoid mapping documented at lines 1108-1127; tanh saturation gate dropped | **SHIP** |
| P1.3 pocket_residue_embed + cross_attn | lines 1140-1156 | `self.pocket_residue_embed = nn.Linear(...)` at line 1183 (zero-init at 1190-1191); `self.cross_attn = nn.MultiheadAttention(...)` at line 1191; `self.cross_attn_norm = nn.LayerNorm(...)` at line 1204; used in forward at 1414-1433 | **SHIP** |
| P1.4 ConnectivityAwareDecoder wrapper | `_generate_impl` | line 1715-1724 (`accept_only_connected`, `keep_empty_connected` flags); import + invocation at 2512-2562 (`bond_decoder = BondAwareDecoder(...)` + `ConnectivityAwareDecoder as _ConnAwareDecoder` wrapper) | **SHIP** |

All four P1 fixes are present in the codebase with the right semantics. The grep also confirmed the absence of the historical `bonds=zeros` placeholder (P0-F5) — the `BondAwareDecoder.decode` is now wired at the line the audit identified (line 2017) and the `bonds=zeros` placeholder that was at line 2018 is gone.

---

## 4. CLI flags inventory (r10_cfg_real_crossdocked.py)

The diagnostic command needs a subset of these flags. Verified by `grep -n "add_argument"` over `r10_cfg_real_crossdocked.py`:

| Flag | Default | Type | Used in probe? |
|---|---|---|---|
| `--output-dir` | `molmetal/reports/r10_cfg_real_crossdocked` | `Path` | YES (override to `wf_cfm_gpu_retrain/phase2_probe/`) |
| `--train-steps` | `200` | `int` | YES (`5000`) |
| `--n-train` | `8` | `int` | YES (`32` — same as wf_gpu_recovery_now) |
| `--ode-steps` | `16` | `int` | default |
| `--lr` | `0.001` | `float` | YES (`0.0001` — same as wf_gpu_recovery_now) |
| `--hidden-dim` | `128` | `int` | YES (`128` — P1.1 default; explicit for documentation) |
| `--n-layers` | `1` | `int` | YES (`2` — keep production scale, not smoke) |
| `--n-samples` | `8` | `int` | YES (`16` — same as wf_gpu_recovery_now) |
| `--budget-seconds` | `300` | `float` | YES (`1500` — 25 min ceiling) |
| `--gpu-binary` | required | `Path` | YES (`scripts/_fake_vina.sh` — same as wf_gpu_recovery_now) |
| `--trace-library` | None | `Path` | NO (skip — adds noise) |
| `--seeds` | `[42, 0, 1234]` | `int list` | YES (`42 0 1234` — same triple) |
| `--vocab-mask` / `--no-vocab-mask` | True | `BooleanOptionalAction` | YES (`--vocab-mask` — P0-F3, mandatory) |
| `--bond-head` | `distance` | `['distance','learned']` | YES (`learned` — exercise the full bond-head path) |
| `--joint-train` | False | `BooleanOptionalAction` | YES (`--joint-train` — P0-F1) |
| `--bond-loss-weight` | `1.0` | `float` | default |
| `--bond-pattern-mask` / `--no-bond-pattern-mask` | True | `BooleanOptionalAction` | YES (`--bond-pattern-mask`) |
| `--connectivity-prior` | True | `BooleanOptionalAction` | default |
| `--use-pcgrad` / `--no-use-pcgrad` | False | `BooleanOptionalAction` | YES (`--use-pcgrad` — P2.3) |
| `--use-tmqm-init` / `--no-use-tmqm-init` | False | `BooleanOptionalAction` | NO (cheaper probe; defer to production) |
| `--pac-bayes-bound` / `--no-pac-bayes-bound` | False | `BooleanOptionalAction` | NO (defer to Round-13) |
| `--pac-bayes-delta` | `0.05` | `float` | NO |
| `--pac-bayes-loss-clamp` | `10.0` | `float` | NO |
| `--decoder-rework` / `--no-decoder-rework` | False | `BooleanOptionalAction` | YES (`--decoder-rework` — Path B chem-aware soft prior; the named-but-nonexistent script's logic lives here) |
| `--pb-relax-mmff94` / `--no-pb-relax-mmff94` | False | `BooleanOptionalAction` | NO (PB relax is post-decode; not load-bearing for the gate) |

No `--device` flag exists — auto-detect picks dGPU (gfx1101) via `get_device()` policy. That is fine: the probe does not need to pin a specific device.

---

## 5. Diagnostic budget

### 5.1 Step-count budget

| layer | spec | reasoning |
|---|---|---|
| **train-steps per seed** | `5000` | Same as wf_gpu_recovery_now (which probed 5000-step @ h=32 with the OLD architecture); reusing the same step count lets us isolate the effect of the P1 fixes (h=32 → h=128) without confounding step-count changes |
| **seeds** | `[42, 0, 1234]` | Same triple as wf_gpu_recovery_now; preserves seed-level reproducibility checks |
| **hidden-dim** | `128` | P1.1; the only architectural lift in this probe |
| **n-layers** | `2` | Production scale, not smoke (smoke uses 1) |
| **n-train** | `32` | Same as wf_gpu_recovery_now |
| **n-samples** | `16` | Same as wf_gpu_recovery_now |

Total training steps: `3 seeds × 5000 = 15000`.
Total generated samples: `3 seeds × 2 pockets × 2 CFG × 16 = 192` (matches wf_gpu_recovery_now).

### 5.2 Wall-time budget

From `wf_gpu_recovery_now/cfm_probe/report.json`:
- 5000-step retrain (3 seeds, h=32, n_layers=2) → **125.10 s wall** end-to-end
- Per-step cost ≈ `125 / 5000 = 25 ms/step` (aggregate across seeds)

For h=128 (4x wider hidden), the per-step cost on a single-GPU EGNN scales roughly **linearly with hidden_dim for the MLP heads and quadratically for the EGNN message passing** (O(H^2) per layer for the equivariant edge MLP). At n_layers=2 we expect roughly:
- MLP heads: ~4x cost
- EGNN message: ~4x cost per layer × 2 layers = ~8x worst case, ~4x amortized (memory-bound on gfx1101)
- Embedding lookup + AtomHead projection: ~4x

Conservative estimate: **2x wall-time at h=128 vs h=32** (well below the theoretical 4-8x because most of the 125s is fixed Python overhead + ODE solver stepping, not the matmuls).

- Projected wall: `125 × 2 ≈ 250 s ≈ 4.2 min` for one full probe
- Budget ceiling: 1500 s (25 min) via `--budget-seconds` — 6x safety margin for re-hang recovery
- Total Phase-2 budget: **15 min wall** (probe + buffer + report write)

### 5.3 Disk budget

`report.json` (with all 192 sample decode attempts) is ~50 KB. With checkpoints at every 1000 steps the directory totals ~5 MB. Generous allocation: 100 MB for `phase2_probe/`.

---

## 6. Expected gate criteria (honest)

Per `WF-CFM-Retrain-Diagnose` spec, the verdict on `decode_ratio = n_decoded / n_requested`:

| Verdict | decode_ratio | Action |
|---|---|---|
| **SUCCESS** | `> 0.5` | Kick off path (a) 10000-step + hidden-dim 64 full retrain |
| **PARTIAL** | `(0, 0.5]` | Hold path (c) λ-only, record evidence, write Phase-3 final.md |
| **FAILURE** | `= 0` | CFM structurally broken; switch to path (b) architectural rework (decoder_rework + ChemAware bond prior) |

Secondary metrics recorded but not gate-decisive:
- `bond_loss_last` — must be `0.000` (already saturated in wf_gpu_recovery_now; expect same)
- `atom_loss_last` — should drop from ~1.0 to < 0.9 if h=128 has more capacity for the categorical head
- `cfm_loss_last` — velocity-field MSE; h=128 may help here even if decode stays at 0 (the failure mode is geometric distance distribution, not velocity accuracy)
- `decode_status_counts` — the failure taxonomy. Prior run: 187 disconnected_distance_graph + 2 valence + 3 vocab. If `--decoder-rework` + h=128 still produces 187 disconnected, we know the decoder soft prior alone is not enough and Path-(b) full architectural rework is required.

### 6.1 Honest prior expectations

The wf_gpu_recovery_now 5000-step probe at h=32 produced `decode_ratio=0/192`. Lifting to h=128 has a **prior probability** of breaking the stalemate, but the audit at `wf_cfm_internal_review/audit.md` named **four root causes**:
- (A) BondOrderHead in_dim under-specified — **already fixed in P0-F2**
- (B) tanh velocity gate — **already fixed in P0 + P1.2**
- (C) `bonds=zeros` placeholder — **already fixed in P0-F1**
- (D) hidden_dim=32 + n_layers=2 + constant pocket bias — **partially fixed in P1.1+P1.3**

Root cause (D) is the only one the probe can lift, and even then the EGNN message-passing capacity gain at h=128 may not be enough to make 1.5-2.0 Å C-C distances emerge from the random prior within 5000 steps (the CFM convergence literature suggests ~10000-50000 steps for h=128 on CrossDocked-scale data). Honest prior: **decode_ratio expected to remain 0** at this budget. The probe value is the *evidence trail* — bond_loss, atom_loss, cfm_loss trajectories at h=128 vs h=32 — that informs the Phase-3 budget for the full 10000-step path.

---

## 7. Probe command (Phase 2)

```
PYTHONPATH=/home/hugo/codes/try_triton_on_rocm timeout 1500 \
  uv run python /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 0 1234 \
    --train-steps 5000 --n-train 32 --n-samples 16 \
    --hidden-dim 128 --n-layers 2 --lr 0.0001 \
    --vocab-mask --bond-head learned --bond-pattern-mask \
    --joint-train --use-pcgrad \
    --decoder-rework \
    --budget-seconds 1500 \
    --output-dir /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_gpu_retrain/phase2_probe/ \
    --gpu-binary /home/hugo/codes/try_triton_on_rocm/scripts/_fake_vina.sh
```

Differences from `wf_gpu_recovery_now`:
- `--hidden-dim 32` → `--hidden-dim 128` (P1.1)
- `--joint-train` added (P0-F1 + P2.3)
- `--use-pcgrad` added (P2.3)
- `--decoder-rework` added (Path-B soft prior)
- `--output-dir` redirected to `phase2_probe/`

Pre-flight: confirm the `device()` `str(DEFAULT_DEVICE)` fix from `wf_gpu_recovery_now` (line 1462-1463) is still in place; without it the probe dies in 14 ms.

---

## 8. Phase 2 deliverables (next workflow)

1. Run the probe command above.
2. Capture `report.json` to `phase2_probe/`.
3. Compare per-seed trajectories (bond_loss, atom_loss, cfm_loss, total) at h=128 vs the wf_gpu_recovery_now h=32 baselines.
4. Compare `decode_status_counts` — does the failure taxonomy shift away from `disconnected_distance_graph` (187/192 in the h=32 probe) when `--decoder-rework` is on?
5. Write `phase2_results.md` with the verdict per the gate in §6.
6. If SUCCESS → kick off `WF-CFM-Retrain-Full` (10000-step + h=64 budgeted separately).
7. If PARTIAL or FAILURE → update `TODO-21` and `TODO-24` with the lift evidence and the path forward (likely Path-B architectural rework beyond the chem-aware soft prior).

---

## 9. Files (paths)

| File | Purpose | Status |
|---|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/cfm_path_b_decoder_rework.py` | Script named in spec | **DOES NOT EXIST** (decoder rework is a module, not a script) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/decoder_rework.py` | Path-B decoder rework module | VERIFIED EXISTS, 76+ LOC, docstring confirms chem-aware soft prior |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py` | CFM training harness with all required flags | VERIFIED EXISTS, CLI flags inventoried in §4 |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py` | Lipman CFM adapter with all 4 P1 fixes | VERIFIED, P1 fixes at lines 1014, 1128, 1183-1204, 1715-1724 + 2512-2562 |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_gpu_retrain/` | This directory + probe output | CREATED this session |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_gpu_recovery_now/cfm_probe/report.json` | Prior 5000-step baseline (h=32) | READ — 125.10 s wall, decode_ratio=0/192 |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/audit.md` | Root-cause audit (4 failures) | READ — root causes (A)-(D) |
