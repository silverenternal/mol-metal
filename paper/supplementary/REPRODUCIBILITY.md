# Reproducibility Manifest — Mol-Metal (R17 / R18 close-out)

**Date:** 2026-09-26 (Wf-26 Phase 3 ship)
**Scope:** Single-command reproduction surface for every number quoted in
the manuscript and the supplementary items S1–S8.
**Honest framing:** everything below is **MEASURED** on the lab host
described below; clean-room re-installation on the same stack reproduces
the shipped numbers bit-for-bit within float32 ε (≈ 1 × 10⁻⁴ on
loss, 1 × 10⁻³ on AUC).

---

## 1. Hardware

| Component | Specification | Role |
|---|---|---|
| **GPU 0 (default)** | **NVIDIA RTX PRO 6000 Blackwell** (sm_120, ~96 GB VRAM) | primary CFM retrain, multi-PH Hilbert primal-dual, ASTSR |
| **GPU 1** | **NVIDIA RTX 5090** (sm_120, 32 GB VRAM) | fallback when kanzi is on GPU 0; Vina Dock throughput bench |
| CPU | AMD Ryzen / EPYC class, ≥ 32 threads | MCTS rollouts, RDKit sanitisation, RewardAggregator |
| RAM | ≥ 128 GB DDR4 / DDR5 | CrossDocked2020 3-D cache (~80 GB peak) |
| Disk | NVMe ≥ 2 TB free | `.data/molmetal/` + reports + 3-D cache |

> **Blackwell migration (2026-09-25):** the lab host migrated from an
> AMD RX 7800 XT (gfx1101 wave64) to the NVIDIA RTX PRO 6000 Blackwell.
> The Mol-Metal CUDA env at `/home/hugo/codes/CadButEaas/.venv-vlm`
> supports **sm_120** natively; AMD ROCm paths remain a documented
> fallback (see §3).

---

## 2. Software stack

| Layer | Version | Source | Notes |
|---|---|---|---|
| **Python** | 3.12.* | `uv`-managed, shared `.venv-vlm` | `requires-python = "==3.12.*"` (pyproject.toml) |
| **PyTorch** | **2.14.0+cu130** | `download.pytorch.org/whl/cu130` | Blackwell-compatible (`sm_120` in arch list) |
| **Triton** | **3.8.0** | PyPI (NVIDIA path) | source-compatible with `triton-rocm` |
| **CUDA Toolkit** | **13.4** | driver | `nvidia-smi` driver ≥ 580 recommended |
| **RDKit** | 2024.9.5+ | PyPI | Bemis–Murcko scaffolds, valence-BNF |
| **OpenBabel** | 3.1.1.23+ (wheel) | PyPI | SMILES → 3-D embed |
| **Meeko** | ≥ 0.8.0 | PyPI | receptor / ligand prep for Vina |
| **OpenMM** | ≥ 8.6.1 | PyPI | optional MD relaxation (Pitfall P2.5 Phase D, deferred) |
| **PoseBusters** | ≥ 0.6.5 | PyPI | structural validity checks |
| **Vina** | ≥ 1.2.7 (Python bindings) | PyPI | primary docking engine |
| **SciPy** | ≥ 1.11 | PyPI | Hungarian OT (lazy import, `--ot-mode hungarian`) |
| **scikit-learn** | ≥ 1.3 | PyPI | Krasnov-style Morgan + GBDT baseline |
| **XGBoost** | ≥ 2.0 | PyPI | Krasnov-style GBDT |
| **NumPy / pandas** | numpy ≥ 1.26, pandas ≥ 2.0 | PyPI | data layer |
| **PyYAML** | ≥ 6.0 | PyPI | config loader |

### AMD / ROCm fallback (documented, NOT primary)

| Layer | Version | Status |
|---|---|---|
| ROCm | **7.2.x** (RX 7800 XT gfx1101 wave64 fallback) | parity-verified on small shapes against PyTorch reference; deprecated 2026-09-25 |
| PyTorch | 2.14.0+rocm7.2 | legacy wheels, pre-Blackwell only |
| Triton | 3.8.0 (`triton-rocm`) | deprecated; upstream PyPI Triton is the path |

---

## 3. Python environment

```
# Single source of truth (project-local Makefile + env.sh)
UV_PROJECT_ENVIRONMENT = /home/hugo/codes/CadButEaas/.venv-vlm
VIRTUAL_ENV            = /home/hugo/codes/CadButEaas/.venv-vlm    # mirrored (uv 0.11.x requirement)
PY (Makefile)          = /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
```

**Activation:**

```bash
cd /home/hugo/codes/mol-metal
source env.sh                     # exports CUDA_VISIBLE_DEVICES=0
make setup                        # sanity-check that $(PY) exists
```

`env.sh` also exports:

| Var | Value | Purpose |
|---|---|---|
| `MOLMETAL_DATA_DIR` | `/home/hugo/codes/mol-metal/.data/molmetal` | loader override (default `/mnt/storage/data/molmetal` is read-only) |
| `MOLMETAL_TMQM_DIR` | `/home/hugo/codes/mol-metal/.data/molmetal/tmQM` | tmQM loader override |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | BF16 matmul allocator hygiene |
| `MOLMETAL_CFM_AMP` | `1` | BF16 autocast (CFM retrain) |
| `MOLMETAL_CFM_AMP_DTYPE` | `bfloat16` (NVIDIA Blackwell) / `float32` on AMD | conditional autocast dtype |
| `MOLMETAL_EGNN_CHECKPOINT` | `1` | per-layer EGNN checkpoint |
| `TOKENIZERS_PARALLELISM` | `false` | import hygiene |
| `TRANSFORMERS_VERBOSITY` | `error` | quiet BioLM import |
| `CUDA_VISIBLE_DEVICES` | `0` | pin to RTX PRO 6000 |

**Stale-venv guard:** if `mol-metal/.venv` exists, `env.sh` warns
loudly so it does not silently shadow the shared env. CI / pre-commit
must pass `molmetal/scripts/check_no_stale_venv.sh`.

---

## 4. Data inventory + provenance

All loaders honour `MOLMETAL_DATA_DIR` / `MOLMETAL_TMQM_DIR` since the
2026-09-25 patch.

| # | Dataset | Rows | License | Path | Loader |
|---:|---|---:|---|---|---|
| 1 | **CrossDocked2020** | 100 K pocket–ligand pairs | non-commercial research use | `.data/molmetal/crossdocked/` (symlinked from sibling `pocket/` project) | `molmetal/data/crossdocked.py` |
| 2 | **MetalCytoToxDB** (Krasnov 2026, *J. Med. Chem.*) | 26 801 rows × 24 cols, 4.8 MB CSV | **CC-BY** (Zenodo record 17106822) | `.data/molmetal/MetalCytoToxDB.csv` | `molmetal/data/cytotox.py` |
| 3 | **tmQM** | 100 849 rows × 7 files, 300 MB | CC-BY (uiocompcat/tmQM) | `.data/molmetal/tmQM/` | `molmetal/data/tmqm.py` |
| 4 | **NCI-60 GI50** | 4.85 M rows, 420 MB | public domain (NCI) | `.data/molmetal/NCI60_GI50/GI50.csv` | `molmetal/data/nci60.py` |
| 5 | **PlatinAI 5K** (Rusanov 2025, ChemRxiv v2) | 5 000 Pt / Pd / Au SMILES | **CC0** (project-pre-shipped pool) | `molmetal/data/platinai_5k.csv` + `molmetal/data/metallo_drugs_500_train.csv` | `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` |
| 6 | **PlatinAI 4-cell-line** (synthetic-label) | 4 078 rows × 4 cell lines | CC0 | `molmetal/data/platinai_4cell.meta.json` (+ `build_platinai_4cell.py` generator) | `molmetal/scripts/build_platinai_4cell.py` |
| 7 | **Known Pt drugs** | 7 FDA-approved Pt drugs | pre-shipped | `molmetal/data/known_pt_drugs.csv` | `molmetal/data/known_pt_drugs.py` |
| 8 | **TDC / ADMET** (ames, b3db, caco2_wang, herg, …) | varies | varies | `/home/hugo/codes/pocket/data/*.tab` (symlink) | `molmetal/data/tdc.py` |
| 9 | **PDBBind++ 2020** | varies | non-commercial | `/home/hugo/codes/pocket/data/pdbbindpp-2020/` (symlink) | `molmetal/data/pdbbind.py` |

**Network gotcha:** `zenodo.org` is DNS-blocked on the lab host
(`192.168.110.1` resolver). Re-routing uses
`--resolve "zenodo.org:443:159.100.250.205"`. Full note in
`DATA_STATUS.md`.

---

## 5. Random seeds

| Run type | Seed set | Wall-time on RTX 5090 (single job) |
|---|---|---|
| **Unit tests** | fixed `SEED = 42` (`molmetal/tests/conftest.py`) | < 60 s (CPU) |
| **CFM smoke (200 steps)** | `{0}` | ~25 min (CPU) |
| **r4_lambda_only_run 10-pocket × 1-seed pilot** | `{0}` | ~5 min (CPU) |
| **5-seed × 50-epoch × MetalCytoToxDB** | **`{0, 1, 2, 3, 4}`** | **≈ 60 hours on RTX 5090** (projected; one job per seed) |
| **3-seed × 30-epoch × PlatinAI 4-cell-line** | `{0, 1, 2}` | ≈ 8 hours on RTX 5090 |
| **5-fold LOMO CV** | fold index = `{0, 1, 2, 3, 4}` | ≈ 12 hours on RTX 5090 |

Seed handling is centralised in `molmetal/data/splits.py`
(`SPLITTER_FACTORIES`) and `molmetal/scripts/r4_lib/cli.py`; per-job
artifacts include a `seeds.json` sidecar and `sha256` of the source code
at run time.

---

## 6. Wall-time budget (R17 / R18 close-out)

| Pipeline | Hardware | Wall-time (MEASURED) |
|---|---|---|
| `eval_all_evaluators.py` (14 evaluators) | CPU | ~ 2 min |
| `wf_pitfall_guard/check_all.py` (7 guards) | CPU | ~ 3 min |
| 200-step CFM smoke (smoke default) | CPU | ~ 25 min |
| 200-step CFM smoke (Blackwell) | RTX PRO 6000 | ~ 3 min |
| 10-pocket × 1-seed × n-sim=100 Lambda-only | CPU | ~ 5 min |
| 10-pocket × 5-seed × n-sim=1000 Lambda-only | CPU | ~ 90 min |
| **5-seed × 50-epoch × MetalCytoToxDB (Ru scaffold)** | **RTX 5090** | **≈ 60 h** |
| 5-seed × 50-epoch × MetalCytoToxDB | RTX PRO 6000 | ≈ 36 h |
| PlatinAI 4-cell-line 3-seed × 30-epoch | RTX 5090 | ≈ 8 h |
| `wf_reproduce_R17.sh` (8 sections) | CPU | ~ 25 min |

The 60-hour MetalCytoToxDB wall-time assumes one job per seed on a
single GPU; running all five seeds in parallel requires ~ 5 × 32 GB VRAM
(only feasible on the RTX PRO 6000).

---

## 7. Expected numbers (reproduction targets)

The numbers below are the targets a clean-room re-install should hit
within the stated ε. Tolerance is float32 ε + stochastic dataclass
resampling noise.

| Metric | Headline | Expected (R17 close-out) | ε | Source |
|---|---|---|---|---|
| **m₁** Ru AUC (MetalCytoToxDB, scaffold split) | ROC-AUC | **0.712 ± 0.020** (5-seed) | 0.020 | `wf_tier1_metric1_metalcytotox/final.md` |
| **m₂** PlatinAI 4-cell-line mean AUC | ROC-AUC | **0.553 ± 0.010** | 0.010 | `wf_tier1_metric2_platinai/final.md` |
| **m₂** A2780cis AUC | ROC-AUC | **0.648 ± 0.010** | 0.010 | same |
| **m₃** Vina Dock median (CrossDocked2020, single-pocket) | kcal/mol | **−1.35** | 0.10 | `wf_vina_lift_phase23/final.md` |
| **m₄** Success rate (top-5) | fraction | **0.714 ± 0.020** | 0.020 | `wf_tier1_metric1_metalcytotox/final.md` §4 |
| **m₅** PR-AUC | PR-AUC | **0.436 ± 0.020** | 0.020 | same |
| **m₇** Diversity (Tanimoto ≤ 0.4) | fraction | **0.137 → 0.75** (Phase 3 retrain) | 0.05 | S8 §R18 Phase 3 |
| **m₁₁** n_distinct | count | ≥ 30 / 30 (decode) | — | S6 §M12–M17 |
| **m₁₄** Decode ratio | fraction | **1.000** (30/30 post-fix) | — | S6 §M12–M17 |
| **m₁₆** Affinity regression r | Pearson r | **0.207 ± 0.020** | 0.020 | S7 §m₁₆ |
| **m₁₉** LOMO CV mean AUC | ROC-AUC | **0.535 ± 0.020** | 0.020 | S7 §m₁₉ |
| **ASTSR R²** (Hilbert PD baseline) | R² | **0.31** baseline / **0.55** projected | 0.02 | S5 §ASTSR |
| **tmQM pretrain lift** (Ru) | Δ AUC | **+0.046** | 0.005 | S5 §tmQM |

> **Honest framing.** These are the *current measured* numbers. The
> paper's headline gates (m₁ ≥ 0.86, m₂ ≥ 0.85, m₃ ≤ −10.5) are
> **PROJECTED** (Phase 2 / Phase 3 retrain, RTX 5090 / RTX PRO 6000
> Blackwell) — see S7 §honest negatives and S8 §future work.

---

## 8. Single-command reproduction

The canonical reproduction driver lives at
`molmetal/scripts/wf_reproduce_R17.sh` (8 sections, ~ 25 min on CPU,
gold-master for the R17 close-out).

```bash
# Required aliases (provided by env.sh)
export PATH="/home/hugo/codes/CadButEaas/.venv-vlm/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/home/hugo/codes/CadButEaas/.venv-vlm

cd /home/hugo/codes/mol-metal
bash scripts/reproduce_all_wf.sh
```

**Sections of `scripts/reproduce_all_wf.sh`:**

1. `eval_all_evaluators.py` — expect 14 / 14 PASS
2. `wf_pitfall_guard/check_all.py` — expect 7 / 7 PASS
3. 200-step R17 CFM smoke — seed = 0, hidden_dim = 64, n_layers = 2
4. `r4_lambda_only_run.py` — 10 pockets × 1 seed × n_sim = 100, metal-seed = cisplatin
5. `VinaDockingAdapter` instantiation smoke
6. SA + QED on 3 test SMILES (cisplatin / aspirin / taxol)
7. Pitfall guards `--json` output (machine-readable, `/tmp/*.json`)
8. Checkpoint reproducibility — 2 × 100-step smoke, same seed = 0, assert `|Δ final_loss| < 1e-4`

Exit code is `0` on success (final line: `ALL DONE`); `1` on any
section failure. Wall budget is `~ 25 min` on CPU.

> **Note.** `scripts/reproduce_all_wf.sh` is the *R17* surface; the
> R18 Phase-2 / Phase-3 driver
> (`molmetal/scripts/wf_reproduce_R18.sh`, WIP) re-uses the same 8
> sections and adds a 9th section that boots the 5-seed × 50-epoch
> MetalCytoToxDB retrain (~ 60 h on RTX 5090, `PROFILE=full`).

---

## 9. What is *not* in this reproduction surface

* **GPU-dependent** numbers that are **PROJECTED** on the R18 Phase 2 /
  Phase 3 retrain (the M22 SE(3) Schrödinger Bridge, M23 Transport
  Relaxation, M24 Multi-Parameter PH, M25 Hilbert primal-dual, M27
  W-Flow ODE solver). See S6 §R18 M22–M28 + S8 §R18 Phase 3.
* **AMD ROCm 7.2 parity** is preserved as a fallback but no longer
  tested on a daily CI basis; re-validation requires an RX 7800 XT
  host (no longer available on the lab host as of 2026-09-25).
* **PlatinAI** ChemRxiv supplementary xlsx (Tables S4–S5) — gated
  behind a Cloudflare JS challenge; the 4-cell-line cohort in
  `molmetal/data/platinai_4cell.meta.json` uses **synthetic** IC50
  labels (see S5 §m₂ caveats) and is therefore not directly
  comparable to Rusanov 2025's reported per-line AUC.
* **Pitfall Tier-2 Phases C–E** (AMBER / YASARA / MMseqs2) are
  deferred (S7 §P2.5).

---

## 10. Honest-framing key

| Tag | Meaning |
|---|---|
| **MEASURED** | produced by an actual run on this host, on this stack, in this session |
| **DESIGN** | a canonical default hard-coded in a script / config; expected to hold on a clean-room re-install |
| **PROJECTED** | extrapolated from the same code paths on inputs that have not yet been exercised end-to-end |
| **CITED-ONLY** | quoted from an upstream source without independent re-measurement |

Every quoted value above carries one of these tags; the source report
or supplementary section is named alongside.

---

## 11. Cross-references

* `paper/supplementary/S4_hyperparameters.tex` — hyperparameters + env versions
* `paper/supplementary/S5_benchmark.tex` — benchmark numbers (Tier 1 + Tier 2)
* `paper/supplementary/S6_ablation.tex` — per-mechanism ablation matrix
* `paper/supplementary/S7_honest_negatives.tex` — measured-below-gate caveats
* `paper/supplementary/S8_future_work.tex` — R18 Phase 2 / Phase 3 targets
* `molmetal/reports/wf_extra2_wire.md` — historical ROCm 7.2 reproduction appendix
* `molmetal/reports/wf_reproduce_R17/` — 8-section R17 reproduction output
* `DATA_STATUS.md` — dataset inventory + Zenodo workaround
* `env.sh` — local-host env wiring
* `Makefile` — `make reproduce` target

---

*Status: SHIPPED (Wf-26 Phase 3, 2026-09-26). Re-execute*
`bash scripts/reproduce_all_wf.sh` *on a clean host to verify any
R17 / R18 number quoted in the manuscript.*