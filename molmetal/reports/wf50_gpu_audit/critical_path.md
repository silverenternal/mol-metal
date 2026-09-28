# Wf-50 Critical Path — 5 Most-Impactful GPU Accelerations

> **Selection criterion**: `impact = per_call_cpu_seconds × workload_per_run × speedup`
> across the 46 evaluation surfaces audited in `gpu_readiness_matrix.md`.
> Each entry below has a measured (or estimated) wall-time saving and
> an explicit effort/risk profile.

The 5 accelerations below are **ranked by cost-benefit** (impact per
developer-week), not by absolute speed-up.

---

## #1 — `boltz_affinity` → `predict_pic50_batch` (1-line refactor)

| | |
|---|---|
| Surface | `molmetal/scoring/score_any.py::_w_boltz_affinity` |
| Current | `for smi in smiles_list: pic50.predict_pic50(smi)` — serial Python loop |
| Target | `pic50.predict_pic50_batch(smiles_list, batch_size=128)` |
| Effort | **0.5 dev-day** (refactor + tests) |
| Risk | LOW — `predict_pic50_batch` already exists, returns `[float]` |
| Expected speed-up | **5-10×** on the pIC50 column |
| Wall-time saving | 0.5-1.5 h/run (10 k-100 k SMILES × 50 ms → 5-10 ms each) |
| VRAM | 150 MB (D-MPNN + ckpt) |
| Required wheels | none (model + ckpt already on `cuda:0`) |

### Why this first

* Smallest change in the entire audit.
* Removes a serial CPU-loop on the **hottest per-SMILES metric** —
  boltz_affinity is invoked once per generated candidate.
* Sets a precedent: every wrapper that has a batched counterpart
  underneath should use it.

### Implementation sketch (NOT applied; Wf-50 Phase 2)

```python
def _w_boltz_affinity(ctx: Dict[str, Any]) -> List[float]:
    pic50 = _import_pic50()
    smiles_list = _as_smiles_list(ctx["molecules"])
    batch_size = int(ctx.get("batch_size", 128))
    return [float(v) for v in pic50.predict_pic50_batch(
        smiles_list, batch_size=batch_size,
    )]
```

---

## #2 — RDKit `DataStructs.BulkTanimotoSimilarity` for `diversity_tanimoto` + `reference_tanimoto` (3-line refactor)

| | |
|---|---|
| Surface | `molmetal/scoring/score_any.py` → RDKit `AllChem.GetMorganFingerprintAsBitVect` loop |
| Current | `for smi_a, smi_b in pairs: AllChem.GetMorganFingerprintAsBitVect(...); DataStructs.TanimotoSimilarity(...)` |
| Target | `DataStructs.BulkTanimotoSimilarity(query_fp, bulk_fps)` |
| Effort | **1 dev-day** (3 metric wrappers + tests) |
| Risk | LOW — `BulkTanimotoSimilarity` is the canonical RDKit bulk path |
| Expected speed-up | **10-50×** on pairwise columns |
| Wall-time saving | ~200 s/run × 10 000 mols (pairwise O(N²)) |
| VRAM | 0 (CPU lever) |
| Required wheels | none |

### Why second

* **Zero GPU work** — pure RDKit refactor. Bypasses the entire
  CUDA-toolchain dependency.
* Diversity is computed on every per-pocket summary, every sweep —
  this lever benefits **every** evaluation run, not just the ones that
  hit the GPU pipeline.
* The three wrappers (`diversity_tanimoto`, `reference_tanimoto`,
  `diversity_subpocket`) share the same fingerprint cache pattern, so
  the fix generalises.

### Implementation sketch (NOT applied; Wf-50 Phase 2)

```python
def bulk_tanimoto_pairwise(smiles_list, radius=2, n_bits=2048):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), radius, n_bits) for s in smiles_list]
    out = np.zeros((len(fps), len(fps)))
    for i, fp in enumerate(fps):
        out[i] = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return out
```

---

## #3 — `flowdock_pose` + `diffdock_pose` → switch to GPU device (wrapper change, vendor wheels)

| | |
|---|---|
| Surface | `molmetal/molmetal_lam/sbdd_env/flowdock_adapter.py` + `diffdock_adapter.py` |
| Current | `--device cpu` (FlowDock 40-step CPU ODE fallback) or subprocess default |
| Target | `--device cuda` (Blackwell) |
| Effort | **3 dev-days** (wrapper change + vendored repo check + GPU smoke test) |
| Risk | MEDIUM — FlowDock upstream depends on `torch_scatter` ROCm wheels; DiffDock-L on `torch_cluster` |
| Expected speed-up | **15-30×** (FlowDock) / **5-15×** (DiffDock) |
| Wall-time saving | ~6.5 h/run (5 000 ligands, single pocket) for FlowDock |
| VRAM | 4 000-6 000 MB (one model) |
| Required wheels | `torch_cluster`, `torch_scatter` for ROCm 7.2 |

### Why third

* **The per-ligand docking oracle** is the second-largest wall-time
  consumer after `vina_score`. Even on FlowDock's CPU ODE fallback the
  per-call cost is ~5 s.
* Both adapters already expose a `device` knob — the change is the
  *default value* plus the wrapper path (currently forks a subprocess
  per SMILES, which itself is a ~50 ms overhead per call).

### Implementation sketch (NOT applied; Wf-50 Phase 2)

```python
# in flowdock_adapter.py / diffdock_adapter.py
device: str = "cuda:0"   # was "cpu"
```

Then merge the subprocess-per-SMILES loop into a single batched call.

---

## #4 — Vina-GPU CUDA build (vendor-binary work)

| | |
|---|---|
| Surface | `molmetal/molmetal_lam/sbdd_env/vina_gpu_adapter.py::QuickVinaGPUAdapter` |
| Current | Vendored QuickVina2-GPU 2.1 (AMD-only OpenCL, refuses Blackwell) |
| Target | CUDA build of QuickVina2-GPU **or** AutoDock-GPU with NVIDIA support |
| Effort | **2-3 dev-weeks** (build script + ROCm/NVIDIA CUDA wheels + smoke + parity test) |
| Risk | HIGH — upstream binaries are not NVIDIA-tested; OpenCL→CUDA port is non-trivial |
| Expected speed-up | **40-80×** on `vina_score` |
| Wall-time saving | ~16 h/run (10 pockets × 10 000 ligands) |
| VRAM | 1 200 MB (single receptor cache) |
| Required wheels | CUDA-flavoured QuickVina2-GPU binary; OpenCL kernel source must be NVIDIA-compatible |

### Why fourth (not first)

* The biggest absolute win, but **the build is the blocker**. The
  current QuickVina2-GPU 2.1 binary is AMD-only; an NVIDIA build is a
  multi-week port.
* Pair with #3: once the ROCm `torch_scatter` wheels land for DiffDock
  the same env will also support a CUDA build of Vina-GPU.

### Implementation plan (NOT applied; candidate ultracode)

1. Fork QuickVina2-GPU OpenCL → CUDA translation (or use AutoDock-GPU
   which already has CUDA support).
2. Compile against CUDA 13.4 + NVIDIA Blackwell sm_120.
3. Smoke-test against the 1h36/3eml pocket to verify kcal/mol parity.
4. Wire into `vina_gpu_adapter.py` with a CUDA-detection guard that
   refuses to launch on a non-NVIDIA device.

---

## #5 — Hilbert / ASTSR + math-skill uniform `.to(device)` (single ultracode)

| | |
|---|---|
| Surface | `molmetal/molmetal_lam/reward/hilbert_*.py`, `lambda_astsr.py`, `astsr_multiph_hilbert.py`, `math_skills/*.py` |
| Current | docstring "CPU-only"; inputs constructed on CPU; test contracts assert CPU |
| Target | uniform `.to(device)` plumbing + relaxed test contracts |
| Effort | **1 dev-week** (one ultracode sweeping 14 modules) |
| Risk | LOW — math is GPU-portable; tests need `assert tensor.device.type in {"cpu", "cuda"}` |
| Expected speed-up | **3-10×** per module × 14 modules |
| Wall-time saving | ~1 h/run (cumulative across the proof-search pipeline) |
| VRAM | 50-200 MB (per module) |
| Required wheels | none (math is built on `torch.linalg` / `torch.cdist`) |

### Why fifth

* Largest *batch* of wins in the audit (14 modules, same change).
* Fits the ultracode pattern perfectly: a single PR that adds
  `.to(device)` everywhere the math is GPU-capable, with relaxed
  test contracts.
* Does NOT depend on any vendor binary or ROCm wheel — pure
  PyTorch.

### Implementation pattern (NOT applied; candidate ultracode)

```python
# 1. Add a device-resolver helper in math_skills/__init__.py:
def to_device_if_available(t: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
    return t.to(device) if torch.cuda.is_available() else t

# 2. Replace CPU-only test assertions:
- assert z.device.type == "cpu"
+ assert z.device.type in {"cpu", "cuda"} or torch.cuda.is_available()

# 3. Add a kwargs `device=None` to each public function and forward to .to().
```

---

## Cross-cutting

| | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Effort | 0.5 d | 1 d | 3 d | 2-3 wk | 1 wk |
| Risk | LOW | LOW | MED | HIGH | LOW |
| Speed-up | 5-10× | 10-50× | 5-30× | 40-80× | 3-10× |
| Wall-time saved per run | 0.5-1.5 h | 200 s | 6.5 h | 16 h | 1 h |
| GPU required | YES (cheap) | NO | YES | YES (binary) | YES (cheap) |
| New wheels | none | none | torch_scatter ROCm | Vina-GPU CUDA | none |
| Recommended ultracode | **YES** | **YES** | YES | NO (multi-week) | **YES** |

## Recommended sequence

1. **Ultracode A** (#1 + #2 + #5) — three small refactors that
   together buy 5-50× on the cheap-RAM, no-vendor-binary column.
   Estimated total wall-clock saving: **~3 h/run** without touching
   the docking oracle or any vendor binary.
2. **Wrapper change** (#3) — one-line wrapper edits + vendored repo
   check; the speed-up is contingent on ROCm `torch_scatter` wheels
   landing in parallel.
3. **Vendor-binary work** (#4) — the biggest absolute win but the
   longest critical path. Schedule after the ROCm wheels land.

## Verification

* `molmetal/tests/test_score_any.py` — must continue to pass after
  each ultracode PR (no mock-fallback invariant).
* `molmetal/tests/test_score_any_policy.py` — policy audit must
  continue to pass.
* After #1 lands: re-run a 1 000 SMILES boltz_affinity smoke and
  verify median latency drops from ~50 ms to ~5-10 ms.
* After #2 lands: re-run `diversity_tanimoto` on 1 000 generated
  SMILES and verify median per-pair latency drops by 10×.
* After #3 lands: re-run a 5-ligand FlowDock sweep on 1h36 and
  verify mean confidence parity vs. CPU ODE within 5%.

---

**Audit ran**: 2026-09-26. **No code modified.** The 5 entries above
are the recommended order of attack for Wf-50 Phase 2 ultracode
sprints. The single-RDKit refactor (#2) and the one-line wrapper
change for boltz_affinity (#1) are the cheapest "free lunch" wins;
they should land before any multi-week vendor-binary work.