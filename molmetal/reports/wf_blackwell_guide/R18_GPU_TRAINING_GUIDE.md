# R18 GPU Training Guide — Blackwell (RTX PRO 6000) and RX 7800 XT Fallback

**Date**: 2026-09-26
**Audience**: ultracode agents + Hugo running R18 Phase 1-4 GPU training on the local Blackwell host.
**Parent plan**: `TODO/03_plans/R18_PLAN.md` (8 weeks, single GPU, no R17 ultracode contention).

---

## §1. Hardware requirements

### §1.1 Primary — Blackwell RTX PRO 6000 (32 GB, sm_120)

| Spec | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 (Blackwell, `sm_120`) |
| VRAM | 32 GB GDDR7 (CUDA 13.4) |
| FP32 / bandwidth | ~120 TFLOPs (BF16 + tensor cores) / ~1.7 TB/s |
| Python | `torch==2.14.0+cu130` + `triton==3.8.0` (pre-installed in `/home/hugo/codes/CadButEaas/.venv-vlm`) |
| Budget | **30 min per ultracode workflow** (CI ceiling); ≤3 h for full R18 Phase 2 retrain |

The host has a secondary RTX 5090 (device 1) — intentionally **not** used by R18 retraining (Blackwell is the only QA'd profile). Reserve the RTX 5090 for non-overlapping R17 ultracode drain.

### §1.2 Fallback — RX 7800 XT (gfx1101, wave64, 16 GB)

Used only when Blackwell is offline (R16 SMU hang) or the 30-min CI budget cannot fit a full R18 retrain (R18_PLAN §risk). AMD Radeon RX 7800 XT (RDNA 3, `gfx1101`, 16 GB GDDR6, wave64). Software: ROCm 7.2.x, `torch==2.14.0+rocm7.2` + `triton-rocm==3.8.0` (ROCm-only PyPI index). Budget: 12-18 h per run (no 30-min CI ceiling). Triton is source-compatible across `triton-rocm` and `triton>=3.0` (CUDA) — only env-var maps differ (`HIP` vs `CUDA`).

---

## §2. Environment bootstrap

### §2.1 Source env.sh (Blackwell primary)

`env.sh` is the single source of truth for the Blackwell host:

```bash
cd /home/hugo/codes/mol-metal
source env.sh           # adds CUDA env vars, routes uv to .venv-vlm
nvidia-smi              # verify RTX PRO 6000 visible
```

What `env.sh` does:

1. `UV_PROJECT_ENVIRONMENT=/home/hugo/codes/CadButEaas/.venv-vlm` — routes `uv run` to the pre-existing venv (avoids re-downloading torch+cu130 ~3 GB).
2. `MOLMETAL_DATA_DIR` / `MOLMETAL_TMQM_DIR` — points loaders at the project-local `.data/molmetal/`.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces CUDA allocator fragmentation for BF16 matmul.
4. `MOLMETAL_CFM_AMP=1` + `MOLMETAL_CFM_AMP_DTYPE=bfloat16` + `MOLMETAL_EGNN_CHECKPOINT=1` — BF16 autocast + per-layer checkpoint (R17 verified).
5. `CUDA_VISIBLE_DEVICES=0` — pin to the RTX PRO 6000.
6. `VIRTUAL_ENV` mirror — forces uv 0.11.x to honour `UV_PROJECT_ENVIRONMENT`.

### §2.2 Verify GPU + stale-venv guard

```bash
# Blackwell
nvidia-smi | head -20
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True 'NVIDIA RTX PRO 6000 ...'

# ROCm fallback
rocm-smi
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True 'AMD Radeon RX 7800 XT'
```

`env.sh` warns if `.venv/` reappears (would shadow the shared env). Pre-commit hook: `molmetal/scripts/check_no_stale_venv.sh`.

---

## §3. R18 Phase 1 — Transport Relaxation (1 week)

### §3.1 Goal

Replace the R17 hard endpoint loss `||x_1 - x_data||^2` with the soft transport-relaxation OT penalty `λ · KL(p_T ‖ q_T(ε))` from Jiang-Xu-Zhang 2026 (arXiv:2602.08118). The W2.2 adapter `molmetal/adapters/transport_relaxation_integration.py` is the bridge — it wraps `learned_shaping` additively (no code change to `learned_shaping.py`).

### §3.2 Driver command

```bash
cd /home/hugo/codes/mol-metal && source env.sh

# CPU-only smoke (verify adapter contract; ~5 s)
uv run --no-sync python molmetal/scripts/verify_transport_relaxation_adapter.py

# Blackwell retrain (target m14 ≥ 0.5 smoke / m14 = 1.000 retrain)
python molmetal/adapters/transport_relaxation_integration.py \
    --train --epochs 50 --seeds 0,1,2
```

**Expected outcomes** (per `R18_PLAN.md` §2 Phase 1 success criteria):

| Metric | Smoke | Retrain |
|---|---|---|
| **m14 Decode ratio** | ≥ 0.5 (≥8/16 valid decodes) | **1.000** (no regression vs R17 close-out) |
| atom_loss (step 1) | finite | finite (R17 fixed via 20-atom vocab + `clamp(-50, 50)`) |
| cfm_loss last | < 50 (variance ↓ from R17 baseline 252330) | < 25 (smooth ε-annealing) |
| disconnected | 0/16 | 0/16 |

**ε-annealing schedule** (hard-coded in adapter): `eps = max(0.1, 1.0 - step/500)` — linear 1.0→0.1 over 500 steps, then constant 0.1. Wall-time budget: smoke ~4 min Blackwell / ~25 min RX 7800 XT; retrain 10k-step × 3 seeds ~1.5 h Blackwell / ~9 h RX 7800 XT.

---

## §4. R18 Phase 2 — SE(3) Schrödinger Bridge (3 weeks, BIGGEST)

### §4.1 Goal

Replace the R17 Euclidean CFM decoder head `X_t = (1-t) X_0 + t X_1` with the **SE(3) geodesic flow** from Mahmood 2026 (arXiv:2603.14049). The W2.1 adapter (`molmetal/adapters/se3_sb_integration.py`) wraps an EGNN velocity net and returns a twist `(ω, v)` ∈ se(3); `integrate_with_cfm` walks the SE(3) manifold via Euler or Heun (midpoint).

### §4.2 Driver command

```bash
cd /home/hugo/codes/mol-metal && source env.sh

# CPU-only smoke (~3 s; verifies SE(3) geodesic + Sinkhorn coupling)
uv run --no-sync python molmetal/scripts/verify_se3_sb_adapter.py
uv run --no-sync python molmetal/scripts/verify_se3_lipman.py

# Blackwell retrain (target m3 ≤ -10.5)
python molmetal/adapters/se3_sb_integration.py \
    --train --epochs 100 --seeds 0,1,2
```

**Expected outcomes** (per `R18_PLAN.md` §2 Phase 2 success criteria):

| Metric | R17 baseline | R18 target | Critical? |
|---|---|---|---|
| **m14 Decode ratio** | 1.000 | **1.000** (no regression) | **HARD GATE** |
| **m3 Vina Dock median** | -8.5 | **≤ -10.5** | **CRITICAL — R18 main push** |
| m11 n_distinct | 4-7 | ≥ 5 | |
| m8 PB-Valid | (R17 baseline) | ≥ 0.95 | |
| SE(3) orthogonality (det=1) | n/a | max orth err < 1e-3 | **HARD GATE** |

### §4.3 Weekly breakdown + wall-time budget

| Week | Task | Deliverable |
|---|---|---|
| **W3** | Lie group Euler + Heun midpoint in `se3_lie_group.py` (~200 LOC); unit tests | `test_se3_lie_group.py` (≥30 tests) |
| **W4** | `velocity_net.py` modified to emit se(3) 6-dim twist `(ω, v)`; 200-step smoke for SO(3) orthogonality + det=1 | `wf_se3_lipman_integration/final.md` |
| **W5** | 10k-step × 3-seed retrain + 100-pocket × 3-seed sweep + Table 2 R18 row | `wf_se3_sb_10000x3/final.md`, paper §4 update |

**Blackwell-only constraint**: R18 Phase 2 retrain must run in a **Thu-Sun 24 h window** (no ultracode contention, per R18_PLAN §risk). Wall-time budget: 200-step SE(3) smoke ~3 min Blackwell / ~15 min RX 7800 XT; 10k-step × 3 seeds retrain ~6 h Blackwell (single GPU, **§risk blacklist** — no ultracode concurrent) / ~36 h RX 7800 XT.

---

## §5. R18 Phase 3 — Multi-Parameter PH + Hilbert primal-dual (2 weeks)

### §5.1 Goal

Two CPU-side modules ship in Phase 3, but the **GPU retrain** is what lifts the wall-time -50% claim:

* **T5 Multi-PH** (Kim-Jung 2026, AAM 174:103023) — pocket multi-parameter persistent homology via Fourier-accelerated 2-PH filtration `(r, d) = (spatial radius, atomic-number coordination distance)`; aggregated into a `(n_bins, n_bins)` feature map.
* **T9 Hilbert primal-dual** (Sun-Zheng 2026, *数学学报* 53(4)) — deterministic O(1/k) primal-dual optimizer on a finite Fourier basis of ℋ = L²[0,1], replacing the F5/LambdaASTSR evolutionary loop.

### §5.2 Driver command

```bash
cd /home/hugo/codes/mol-metal && source env.sh

# CPU-only smoke (~10 s; verifies multi-PH SE(3) invariance + Hilbert O(1/k))
uv run --no-sync python molmetal/scripts/verify_multi_ph_adapter.py
uv run --no-sync python molmetal/scripts/verify_hilbert_primal_dual.py

# Blackwell retrain (target m7 ≥ 0.75, wall-time -50%)
python molmetal/adapters/multi_ph_integration.py --train --epochs 50
```

**Expected outcomes**:

| Metric | R17 baseline | R18 target |
|---|---|---|
| **m7 Diversity (Morgan ECFP4 1-Tanimoto)** | 0.65 | **≥ 0.75** |
| LambdaASTSR R² (F5 gate) | 1.000 | ≥ 0.62 (gate, no regression) |
| LambdaASTSR wall-time | evolutionary (30 gen) | **-50%** via Hilbert primal-dual |
| Hilbert convergence (theoretical) | n/a | **O(1/k)** (Sun-Zheng 2026) |
| Multi-PH complexity (N ≤ 100 atoms) | O(N²) brute | O(N log N) Fourier (Kim-Jung 2026) |

### §5.3 Wall-time budget

| Phase | Blackwell | RX 7800 XT fallback |
|---|---|---|
| Multi-PH Fourier sweep | ~45 min | ~4 h |
| Hilbert primal-dual 30-epoch | ~30 min | ~3 h |
| Joint P3 retrain × 3 seeds | ~2 h | ~12 h |

---

## §6. R18 Phase 4 — Paper final + close-out (1 week)

### §6.1 Goal

Update `paper/sections/03_method.tex` (§3.4 SE(3) SB + Wasserstein GF + Monge-Ampère), update Table 2 with the R18 row (m3, m7, m14, m11, m17, Deflex R² — **drop Tier-1 AUC columns** per honest framing), update 5 SKILL.md files +30 lines each, run `7/7 pitfall guards` one final time, and write `TODO/completed/42_r18_closeout.md`.

### §6.2 Driver commands

```bash
cd /home/hugo/codes/mol-metal && source env.sh

# 4-pass pdflatex + bibtex build (no GPU; can run any time)
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex   # pass 1
bibtex main                                                # pass 2
pdflatex -interaction=nonstopmode -halt-on-error main.tex   # pass 3
pdflatex -interaction=nonstopmode -halt-on-error main.tex   # pass 4
cd .. && ls -la paper/main.pdf
```

Or use the pre-baked script: `bash molmetal/scripts/build_paper.sh`. Expected: `=== Verify main.pdf ===`, `main.pdf` exists, 0 unresolved refs in `paper/main.log`.

### §6.3 R18 Table 2 row (target values)

| Metric | R17 close-out | R18 (Phase 2+3) |
|---|---|---|
| m3 Vina Dock median | -8.5 | ≤ -10.5 ✅ |
| m7 Diversity | 0.65 | ≥ 0.75 ✅ |
| m14 Decode ratio | 1.000 | 1.000 (no regression) ✅ |
| m11 n_distinct | 4-7 | ≥ 5 ✅ |
| m17 DrugOOD OOD | 0.891 / 0.697 / 0.748 | unchanged ✅ |
| Deflex R² | 1.0000 | ≥ 0.62 (gate, no regression) ✅ |

**Drop from Table 2** (per honest framing, R18_PLAN §honest): m1 Ru AUC, m2 Pt MCF-7/A549/A2780/A2780cis AUC. Mark "see §6 limitations" in §4. Wall-time budget: pdflatex 4 passes ~30 s · 5 SKILL.md × +30 lines ~2 h · 7/7 pitfall guard ~5 min · `wf_r18_closeout/final.md` synthesis ~1 h.

---

## §7. R18-specific monitoring (per workflow gate)

Each R18 ultracode workflow must surface the following diagnostic — the GPU run is **unrunnable** if the diagnostic fails:

| Wf | Module | Diagnostic | Gate |
|---|---|---|---|
| **Wf-1** | `se3_sb_integration` | **SE(3) orthogonality** (max orth err per epoch) | max orth err < 1e-3 each epoch |
| **Wf-2** | `transport_relaxation_integration` | KL(p_T ‖ q_T) trace (per-epoch) | KL monotone non-increasing |
| **Wf-9** | `hilbert_primal_dual` | **Convergence rate** at iter k | empirical ≤ O(1/k) (Sun-Zheng 2026) |
| **Wf-9** | `score_repellent_mc` | **Variance reduction** vs vanilla MCMC | ≥ 30% (Burch-Bardsley 2022 §4.3) |
| **Wf-10** | `w_flow_cfm` | **Wall-time speedup** over base solver | ≥ 10× (Lelarasmee 1982 / W-Flow paper) |
| **Wf-2.4** | `coordination_gnn` / `quotient_space_diffusion` | SO(3) equivariance check | max rotation drift < 1e-4 |

**Diagnostic logging hook**: each module writes its diagnostic to the per-run report dir (`molmetal/reports/wf_<id>/<seed>/diagnostics.jsonl`). **CI hook**: `molmetal/scripts/wf_pitfall_guard/check_all.py` reads the diagnostics JSONL and fails the workflow if any gate value is out of range.

### §7.1 Per-phase monitoring timeline

```
Phase 1 (W1-2) — Transport Relaxation
  Wf-2: KL(p_T ‖ q_T) trace, monotone non-increasing
  Wf-2: atom_loss + bond_loss finite check (post 2026-09-25 F4 fix)

Phase 2 (W3-5) — SE(3) Schrödinger Bridge
  Wf-1: SE(3) orthogonality each epoch (max orth err < 1e-3)
  Wf-1: det(R) = 1 sanity (no det drift over integration window)
  Wf-1: m3 Vina Dock median per epoch (≤ -10.5 target)

Phase 3 (W6-7) — Multi-PH + Hilbert primal-dual
  Wf-9: Hilbert convergence rate empirical vs theoretical O(1/k)
  Wf-9: Score-Repellent variance reduction ≥ 30%
  Wf-10: W-Flow wall-time speedup ≥ 10×
  Wf-2.4: Multi-PH feature map SE(3) invariance (rotate + assert equal)

Phase 4 (W8) — Paper + close-out
  pdflatex 4-pass: 0 unresolved refs
  7/7 pitfall guards PASS
  5 SKILL.md × +30 lines + MERGE_GATE_R18 PASS
```

---

## §8. Fallback — RX 7800 XT (gfx1101, wave64)

### §8.1 Environment

```bash
# (Re-)install the ROCm env — only on the AMD box
cd /home/hugo/codes/try_triton_on_rocm
uv sync
source environments/setup_env.sh    # ROCm 7.2 + torch 2.14 + triton-rocm 3.8
```

What `environments/setup_env.sh` (or `molmetal/__init__.py:_apply_rocm_gfx1101_env` on import) does: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (BF16 allocator hygiene); `TORCH_BLAS_PREFER_HIPBLASLT=1` (hipBLASLt routing); `PYTORCH_NO_HIP_MEMORY_CACHING` (16 GB VRAM ceiling); loads `triton_rocm` runtime (wave64 kernels; **no** `waves_per_eu`).

### §8.2 Smoke commands

```bash
source environments/setup_env.sh && rocm-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Re-run the same R18 adapters on the AMD path
uv run python molmetal/adapters/transport_relaxation_integration.py --train --epochs 50 --seeds 0,1,2
uv run python molmetal/adapters/se3_sb_integration.py --train --epochs 100 --seeds 0,1,2
uv run python molmetal/adapters/multi_ph_integration.py --train --epochs 50
```

### §8.3 Wave64 Triton note

The shipped `triton_kernels/*.py` (matmul, EGNN coord-update, scatter) are **gfx1101 wave64-only** — `num_warps=4` (256 threads) and **do not** query `waves_per_eu` (wave32-only). See `molmetal/triton_kernels/autotune.py` and `config.py` for wave-aware autotune defaults.

### §8.4 When to fall back (per `R18_PLAN.md` §risk)

| Trigger | Action |
|---|---|
| Blackwell OOM at 10k-step retrain | Drop to 5k-step + linear LR; ship incremental |
| Blackwell firmware hang (R16 SMU hang) | Switch to RX 7800 XT — flag in §6 limitations "computational constraint" |
| 30-min CI budget cannot fit | Submit RX 7800 XT pilot as paper §6 evidence (12-18 h run, no CI ceiling) |
| Phase 2 3-seed → 1-seed (R18_PLAN §risk reserve) | Honest "single-seed" annotation in §4 Table 2 R18 row |

---

## §9. Quick reference (TL;DR)

```bash
# Blackwell primary path (default)
cd /home/hugo/codes/mol-metal && source env.sh && nvidia-smi

# Phase 1 (1 wk): Transport Relaxation
python molmetal/adapters/transport_relaxation_integration.py --train --epochs 50 --seeds 0,1,2
# target: m14 ≥ 0.5 smoke / 1.000 retrain

# Phase 2 (3 wk, BIGGEST): SE(3) Schrödinger Bridge
python molmetal/adapters/se3_sb_integration.py --train --epochs 100 --seeds 0,1,2
# target: m3 Vina Dock median -8.5 → ≤ -10.5

# Phase 3 (2 wk): Multi-PH + Hilbert primal-dual
python molmetal/adapters/multi_ph_integration.py --train --epochs 50
# target: m7 Diversity ≥ 0.75, wall-time -50% (Hilbert O(1/k))

# Phase 4 (1 wk): Paper final
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex    # ×4
cd .. && bash molmetal/scripts/build_paper.sh

# RX 7800 XT fallback (R17 path)
cd /home/hugo/codes/try_triton_on_rocm
source environments/setup_env.sh && rocm-smi
# (same adapter commands; triton-rocm 3.8.0 wave64)
```

**Cross-refs**: `TODO/03_plans/R18_PLAN.md` (8-week Gantt) · `env.sh` (Blackwell env) · `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` (shim + vocab fixes) · `TODO/05_compute/7800xt_budget.md` (ROCm 7.2 + gfx1101 budget) · `TODO/completed/{54,62,63}_*.md` (Wf-1/9/10 ships) · `molmetal/scripts/wf_pitfall_guard/check_all.py` (7/7 CI gate).

---

**End R18 GPU Training Guide. Read with `TODO/03_plans/R18_PLAN.md` for the strategic 8-week timeline.**