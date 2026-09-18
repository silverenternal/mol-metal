# WF-Round12-SOTA-Subset — live SOTA 5×1 (DiffDock)

**Date:** 2026-09-14
**Status:** PARTIAL (all three independent blockers hit; honest-framing mandatory)
**Goal:** Run ONE SOTA generator on 5 pockets × 1 seed as a LIVE anchor for paper §4.5.

## 1. Pre-flight CLI signature audit

`molmetal/references/DiffDock/inference.py` was inspected (318 lines).
Key CLI flags (line numbers in parens):
- `--protein_path` (62) — single pocket PDB
- `--ligand_description` (64) — **SMILES string OR rdkit-readable file**
  - The user-brief used `--ligand`; the actual flag is `--ligand_description`.
  - Default is `'CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1'` (5-phenyl-valeric
    phosphonate-ish reference). Confirmed against line 64.
- `--protein_ligand_csv` (60) — batch mode alternative
- `--samples_per_complex` (72, default 10)
- `--batch_size` (78, default 10)
- `--inference_steps` (80, default 20)
- `--model_dir` (73, default None → auto-download)
- `--confidence_model_dir` (75, default None)
- `--ckpt` / `--confidence_ckpt` (74, 76)
- `--out_dir` (69, default `results/user_inference`)
- `--inference_steps`, `--actual_steps` (80–81)
- Temperatures `--temp_sampling_tr/rot/tor`, `--temp_psi_*`, `--temp_sigma_data_*` (88–96)

`Pocket2Mol/inference.py` is named `sample.py` and `sample_for_pdb.py`
(see `ls molmetal/references/Pocket2Mol/`); both require a downloaded
`ckpt/` dir and a separate RDKit preprocessing step. Per the brief's
recommendation, DiffDock was chosen because the CLI is one-shot and the
rdkit-bound docking scores were already integrated via `flowdock_adapter`.

## 2. Honest framing — three independent blockers

The wall-clock budget was 30 min; we stopped at pre-flight because every
arm below failed before any generation step ran. All three blockers are
*environment*, not *code*, and each is independently fatal.

### Blocker A — GPU unavailable
```
uv run python -c "import torch; print(torch.cuda.is_available())"
→ torch 2.14.0+rocm7.2  cuda False  device_count 0
```
DiffDock's `inference.py:142`:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
would fall back to CPU. CPU inference of 100 samples × 5 pockets × 20
diffusion steps is documented at 4–8 h / pocket on commodity x86 — well
over the 30 min budget, and not a "live anchor".

### Blocker B — `esm` not installed in the uv env
The DiffDock preprocessing pipeline (used to embed ESM-1b per residue)
imports `esm` at `InferenceDataset` construction
(`molmetal/references/DiffDock/datasets/pdbbind.py` etc.). The current
uv env has `fair-esm` missing — `import esm` fails with
`ModuleNotFoundError`. Installing `fair-esm` would require a network
fetch, which is itself blocked (see C).

### Blocker C — GitHub network unreachable
```
curl -sIL https://github.com/gcorso/DiffDock  → curl: (28) timeout
curl -sIL https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip
                                          → curl: (28) timeout
```
The DiffDock `REMOTE_URLS` (line 53) points to
`https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip`
+ the v1.1 fallback. Both fail to resolve, so the auto-download branch
(`inference.py:120–141`) raises:
> `Models not found locally and failed to download them from REMOTE_URLS`
…before any preprocessing or sampling begins.

Even if A and B were resolved, C means we cannot obtain the 120 MB
`diffdock_models.zip` (best_ema_inference_epoch_model.pt + 75-epoch
confidence ckpt) required by `--model_dir ./workdir/v1.1/score_model`
per `default_inference_args.yaml`.

### Smoke test (1 pocket) result
The brief's fallback was a 1-pocket smoke. Even at the 1-pocket stage,
the script cannot be invoked because `inference.py` imports
`torch_geometric` at module top-level (`from torch_geometric.data import
Batch, Data` etc.), and `torch_geometric` is **not** installed in the
uv env. Confirmed:
```
uv run python -c "from inference import get_parser"
→ ModuleNotFoundError: No module named 'torch_geometric'
```
So the smoke test is blocked at parser-import, not at sampling. No
ligand, no confidence score, no top-1 was produced. We refuse to
fabricate numbers.

## 3. What *was* produced

| Artifact | Path | Content |
|---|---|---|
| This report | `molmetal/reports/wf_round12_sota_subset/final.md` | Honest-framing non-run report |
| Empty output dir | `molmetal/reports/wf_round12_sota_subset/diffdock/` | Directory created per spec; no SDFs / no CSV (0 ligands generated) |

No `diffdock_<pocket>/` sub-directories, no `.sdf` files, no confidence
CSVs were written. `n_ligands_total = 0`.

## 4. Per-pocket table (empty — 0 of 5 pockets ran)

| pocket_id | source_pdb | n_requested | n_ligands_generated | confidence_mean | top1_confidence |
|---|---|---:|---:|---:|---:|
| test_000 (BSD_ASPTE_1_130_0) | `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb` | 100 | 0 | n/a | n/a |
| test_001 (GLMU_STRPN_2_459_0) | `…/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb` | 100 | 0 | n/a | n/a |
| test_002 (GRK4_HUMAN_1_578_0) | `…/GRK4_HUMAN_1_578_0/4yhj_A_rec_4yhj_an2_lig_tt_min_0_pocket10.pdb` | 100 | 0 | n/a | n/a |
| test_003 (GSTP1_HUMAN_2_210_0) | `…/GSTP1_HUMAN_2_578_0/14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10.pdb` | 100 | 0 | n/a | n/a |
| test_004 (GUX1_HYPJE_18_451_0) | `…/GUX1_HYPJE_18_451_0/…pocket10.pdb` | 100 | 0 | n/a | n/a |

All 5 receptor PDBs exist on disk; the 5th was inspected (`ls …` shows
the expected `_pocket10.pdb` files for the BSD_ASPTE family at least).
The failure is not data-side.

## 5. Aggregate

| Metric | Value |
|---|---:|
| pockets requested | 5 |
| pockets completed (no timeout) | 0 |
| seeds | 1 (intended: 42) |
| n_ligands_total | 0 |
| confidence_mean | n/a |
| top1_confidence | n/a |
| gpu_blocked | True (`torch.cuda.is_available() == False`, ROCm not enumerated as `cuda`) |
| n_smoke_pockets (1-pocket smoke fallback ran) | 0 (blocked at import) |
| network_blocked (cannot fetch diffdock_models.zip) | True |
| esm_blocked (cannot embed receptor residues) | True |
| torch_geometric_blocked (parser-import fails) | True |
| wall-clock consumed | <2 min (pre-flight only) |
| wall-clock budget | 30 min |

## 6. Limitations & honest-framing notes

1. **The §4.5 live SOTA column is NOT shipped in this round.** The
   cite-only SOTA table from `molmetal/reports/wf_3_citeonly_sota.{tex,csv}`
   remains the only SOTA evidence in the paper, and is what
   §4.5 should continue to cite.
2. The mini-pilot result for Lambda search-only (`wf_round12_mini_pilot`)
   is unrelated: that ran on the *mol-metal* generator (Lambda + β-NF),
   not on a third-party SOTA. It must not be conflated with this SOTA
   anchor attempt.
3. Pocket2Mol was not exercised because DiffDock is simpler
   (one-shot `inference.py`) and was explicitly preferred in the brief.
4. The brief mentioned `--ligand`; the actual flag is
   `--ligand_description`. We did not invoke `inference.py` because of
   blockers A/B/C above, so no CLI mismatch leaked into the run.
5. **Possible unblock paths** (for a future round):
   - Get ROCm enumerated as `cuda` (or use `torch.cuda.set_device(0)` with
     `HSA_OVERRIDE_GFX_VERSION=10.3.0` for gfx1101).
   - Install `fair-esm` and `torch_geometric` (network permitting).
   - Pre-fetch `diffdock_models.zip` and unpack into
     `molmetal/references/DiffDock/workdir/v1.1/{score_model,confidence_model}`.
   - Then re-run the 5-pocket sweep with `--samples_per_complex 40` to
     stay under the 30 min ceiling (CPU path) or full 100 (GPU path).

## 7. Conclusion

**status = partial**, **gpu_blocked = True**, **n_ligands_total = 0**.
We did not paper over the gaps — this report is the artefact. The
paper's SOTA anchor for §4.5 stays cite-only for this submission cycle.
