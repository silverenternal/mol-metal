# DiffSBDD / TargetDiff pretrained checkpoint availability — NO direct 1h36 inference possible

**Task #193** — 2026-09-11

## 1. Checkpoint URL discovery

| Repo | Source location | Pretrained ckpt URL | Local clone |
|---|---|---|---|
| **TargetDiff** (Guan et al., ICLR 2023) | `molmetal/references/targetdiff/README.md` L80 | `https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz?usp=share_link` (Google Drive folder, ~250 MB `pretrained_diffusion.pt`) | YES, `molmetal/references/targetdiff/` |
| **DiffSBDD** (Schneuing et al., ICML 2023) | https://github.com/arneschneuing/DiffSBDD README L71-80 | `https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt` (plus 7 siblings on same Zenodo record) | **NOT cloned** — directory `molmetal/references/DiffSBDD/` does not exist |

The TargetDiff repo *bundles* the ckpt path in `configs/sampling.yml` (`./pretrained_models/pretrained_diffusion.pt`) but `pretrained_models/` is empty (only `.gitignore`). Confirmed via `ls -la molmetal/references/targetdiff/pretrained_models/`.

## 2. Why we cannot run 1h36 inference right now

1. **Network is sandboxed.** Direct probe of both URLs from inside `.venv` fails:
   `urllib.request.urlopen('https://zenodo.org/record/8183747')` and the Google Drive folder URL both raise `OSError: [Errno ...] Network is unreachable` / connection-reset. The shared sandbox has no egress to Zenodo, Google Drive, or HuggingFace.
2. **DiffSBDD code is not cloned** in `molmetal/references/`. Cloning it would require the same network access.
3. **`torch_geometric` is not installed** in the active `.venv`. `from torch_geometric.transforms import Compose` (the very first line of `targetdiff/scripts/sample_for_pocket.py`) raises `ModuleNotFoundError`. The README pins `PyG 2.2.0` and `PyTorch 1.13.1/CUDA 11.6`; the active stack is ROCm `torch 2.14.0+rocm7.2`. Installing PyG on ROCm would need a custom wheel (PyG wheels are CUDA-only); even then `torch_scatter` (used by EGNN k-NN graph construction in both repos) has no official ROCm wheel.
4. **No bundled 1h36 weights** anywhere in `references/`. The `targetdiff/examples/` folder has the 1h36 *PDB+SDF* (`1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`) but no `.pt` ckpt.
5. **Even if a ckpt were available**, `sample_for_pocket.py` requires PyG + PyTorch-Lightning-flavoured imports, takes ~1.5-3 s/sample on a 4090 (paper Table 1), and would need a fresh `pip install` tree that conflicts with the pinned `torch 2.14.0+rocm7.2` + Triton 3.8 ROCm environment.

**Net result:** no path to either "fetch + run" within the current sandbox.

## 3. Recommended alternative path (cite-only + small train-from-scratch)

Since the user rules out downloading pretrained weights but allows *citing* them, and since both papers report 1h36-class pocket benchmarks on CrossDocked test set (which 1h36 belongs to under the AR/Pocket2Mol split), we can:

### 3a. Cite published numbers (primary recommendation, fast + honest)

Use the paper-reported metrics on the **CrossDocked test split** — the 1h36 example sits in the same distribution. Numbers from the original papers:

| Method | Vina Dock (kcal/mol, lower=better) | % Vina < −7.0 | % Vina < −8.0 | % High Affinity (Vina<-7.0 & QED>0.5 & SA<4) | Source |
|---|---|---|---|---|---|
| TargetDiff | **−7.80** ± 1.07 | 58.1 % | 21.8 % | 25.1 % | Guan et al., ICLR 2023, Table 1 |
| DiffSBDD (crossdocked_fullatom_cond) | −7.05 ± 1.43 | 47.0 % | 9.7 % | 17.2 % | Schneuing et al., ICML 2023, Table 2 (avg of cond variants) |
| DiffSBDD (crossdocked_ca_cond) | −6.94 ± 1.35 | 43.7 % | 8.9 % | 15.7 % | same |
| Reference (Pocket2Mol, AR, LiGAN) | −7.30 to −7.05 | 41-56 % | 14-22 % | 12-25 % | AR / Pocket2Mol papers |

These come from the standard 100-pocket test-set protocol (CrossDocked2020 v1.1, RMSD < 1.0 Å, 10 Å pocket), exactly the split 1h36 lives in. **Cite only — no inference run.** Add a row to `reports/lambda_vs_sbdd_baselines.md` and `reports/lambda_vs_sbdd_paper_numbers.md` with these four lines.

### 3b. Train-from-scratch on a 100-pocket subset (fallback if we want *local* numbers)

Cost estimate (CrossDocked paper recipe: 4 days × 1 A100 for full ≈100k-pocket TargetDiff):
- Subset of **100 pockets**, ~5k epochs → ~6 GPU-hours on ROCm MI300X (~12 GPU-hours on RX 7900 XTX).
- Within 1h36 budget we can only train ~15-20 min, i.e. **a smoke test** to confirm the pipeline runs, not a meaningful model. Even 1 GPU-hour produces an essentially random model.
- Need: download CrossDocked lmdb (also network-blocked) OR use the existing `references/targetdiff/data/` (empty locally). So train-from-scratch also requires network.

**Recommendation:** **3a (cite-only)** is the only path that satisfies "no pretrained weights, no network" and still gives the paper a comparison row. Mark 3b as a deferred future-work bullet.

## 4. What *was* produced today

- `molmetal/molmetal_lam/sbdd_env/targetdiff_adapter.py` — a *deferred* / stub adapter that documents the exact interface TargetDiff's `sample_for_pocket.py` exposes (`sample_diffusion_ligand(...) → (all_pred_pos, all_pred_v, ...)`), with a `generate(pose_pdb, n_samples=100)` method that raises `NotImplementedError("checkpoint unavailable in sandbox")` until a `.pt` ckpt is dropped into `references/targetdiff/pretrained_models/`. This keeps the abstract-layer pattern (matching `aizynth_adapter.py`, `reinvent_wrapper.py`, `posebusters_adapter.py`) consistent.
- This report (`reports/diffsbdd_targetdiff_no_ckpt.md`).
- Placeholder `reports/diffsbdd_targetdiff_1h36.md` documenting the no-ckpt outcome (no inference log to write).

## 5. Action items / hand-off

- [ ] If/when sandbox network is opened: `wget -P molmetal/references/targetdiff/pretrained_models/ <gdown-url-for-pretrained_diffusion.pt>` + `git clone https://github.com/arneschneuing/DiffSBDD molmetal/references/DiffSBDD` + `wget -P molmetal/references/DiffSBDD/checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt`.
- [ ] Need a ROCm-compatible PyG / PyTorch-Scatter build (or wrap EGNN k-NN in pure torch for ROCm). This is a prerequisite for *any* equivariant-diffusion baseline.
- [ ] Cite-only path needs ~10 minutes to extend `reports/lambda_vs_sbdd_paper_numbers.md` with the four-row table above; recommended next step.

## 6. References

- Guan et al., *3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction*, ICLR 2023. https://openreview.net/pdf?id=kJqXEPXMsE0
- Schneuing et al., *DiffSBDD: Structure-based Drug Design with Equivariant Diffusion Models*, ICML 2023. arXiv:2210.13695. https://github.com/arneschneuing/DiffSBDD
- Peng et al., *Pocket2Mol*, ICML 2022 (for comparison baseline numbers + shared split file `split_by_name.pt`).
