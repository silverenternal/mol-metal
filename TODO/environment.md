# Environment — canonical measured facts

**Last measured:** 2026-10-03. Every fact in §1–§7 below was measured on this
host in the 2026-10-03 session, with `/home/hugo/codes/mol-metal/.venv/bin/python`
or the system tool named inline. Facts that stopped being true are preserved and
labelled STALE in **§8** rather than deleted (§5.5 filesystem-truth + the
project's history-preserving convention).

**Canonical interpreter** (AGENTS.md §5.9 single-venv rule):
`/home/hugo/codes/mol-metal/.venv/bin/python`. Do not use bare `python3`,
`CadButEaas` venvs, or `kanzi_venv` venvs in launchers or probes.

---

## 1. Host

Measured 2026-10-03 with `uname -srm`, `lscpu`, `free -h`, `df -h`, `du -sh`:

| Item | Measured value |
|---|---|
| Kernel | Linux 7.2.6-hardened1-1-hardened x86_64 |
| CPU | Intel Core i9-14900K, 32 CPUs |
| RAM | 94 GiB total; 63 GiB available at probe time |
| Root disk | /dev/nvme0n1p2 — 3.7 TB total, 1.6 TB free (59% used), mounted at `/home` |
| Repo size | 202 GB (`/home/hugo/codes/mol-metal`) |

## 2. GPUs (canonical)

Measured 2026-10-03 with
`nvidia-smi --query-gpu=index,name,memory.total --format=csv`:

| Index | Name | VRAM |
|---|---|---|
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 97887 MiB |
| GPU 1 | NVIDIA GeForce RTX 5090 | 32607 MiB |

- Driver 615.71.09; CUDA UMD version 13.4 (nvidia-smi header).
- torch probe: `torch.cuda.is_available()` → `True`; `device_count()` → 2;
  `cuda:0` capability `(12, 0)` (sm_120, Blackwell); arch list includes `sm_120`;
  a real 256×256 matmul on `cuda:0` succeeds.
- **Device selection is CUDA-style now.** Measured: `CUDA_VISIBLE_DEVICES=1`
  → 1 device (RTX 5090); `HIP_VISIBLE_DEVICES=1` has **no effect** (still 2
  devices). This is the reverse of the historical ROCm guidance (§8).
- VRAM budget check with `nvidia-smi` before any launch, per AGENTS.md §5.1
  (97887 MiB + 32607 MiB cards; OOM = abort). Launch wrappers select the device
  via `CUDA_VISIBLE_DEVICES` plus `--device` / `--cuda-visible-devices` flags
  (e.g. `scripts/launch_wf131_p3_pt_4cells.sh:446`,
  `scripts/launch_wf299_cfm_main.sh:513`).

## 3. Python / torch stack (canonical)

Measured 2026-10-03 with `.venv/bin/python -c ...` and `grep pyproject.toml`:

- Python 3.12.13 — `.venv/bin/python` is a symlink to a uv-managed CPython
  (`/home/hugo/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`).
  `.python-version` = `3.12`; `pyproject.toml` `requires-python = "==3.12.*"`.
- `uv` 0.11.8 at `/home/hugo/.local/bin/uv`; `uv.lock` present (344,202 bytes,
  regenerated 2026-10-03) — uv is the package manager of record.
- `torch 2.14.1+cu130`; `torch.version.hip` is `None`; CUDA available.
- `triton 3.8.0` — the PyPI `triton==3.8.0` pinned in `pyproject.toml`
  (NOT `triton-rocm`, which is historical — §8).
- numpy 2.5.3 · scipy 1.18.1 · rdkit 2026.03.6.
- `pyproject.toml` index: `pytorch-cu130` →
  `https://download.pytorch.org/whl/cu130`; pins `torch>=2.14,<2.15` +
  `triton==3.8.0`.

## 4. Data root (canonical)

Measured 2026-10-03 (`ls`, `wc -l`, `du -sh`, and a `_resolve_csv()` probe):

- **Canonical root:** `/home/hugo/codes/mol-metal/.data/molmetal`
  (i.e. `<repo>/.data/molmetal`).
- **Override:** `MOLMETAL_DATA_DIR` is honoured FIRST. Measured: with a temp
  dir holding a `MetalCytoToxDB.csv`, `MOLMETAL_DATA_DIR=/tmp/mm_env_probe`
  made `molmetal.data.metal_cytotox._resolve_csv()` return
  `/tmp/mm_env_probe/MetalCytoToxDB.csv`; with the var unset it returns
  `/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv`.
- `MetalCytoToxDB.csv`: 4,839,261 bytes, 26,801 data rows (`wc -l` = 26,802
  incl. header).
- Top-level contents (`du -sh`): chembl_29 21 GB · bindingdb_full 11 GB ·
  drugood 911 MB · NCI60_GI50 402 MB (contains `GI50.csv`) · tmQM 324 MB ·
  crossdocked 9.7 MB (2,466 entries incl. `split_by_name.pt`) · moleculenet
  6.0 MB · tdc_cache 5.9 MB · `metalcytotox_5metal_multitask_v4.csv` 6.6 MB ·
  `platinai_4cell*.csv` + `platinai_4cell_mbfinder.csv` +
  `platinai_selenium_real_labels.csv`.
- `molmetal/data/metal_cytotox.py::_resolve_csv` still lists a legacy fallback
  candidate `/mnt/storage/data/molmetal/MetalCytoToxDB.csv`, but
  **`/mnt/storage` does not exist on this host** (`ls /mnt/storage` → No such
  file or directory), so that candidate can never resolve.

## 5. QVina

- Canonical binary: `molmetal/bin/vina` (567-byte wrapper script). Measured
  2026-10-03: `molmetal/bin/vina --version` → `AutoDock Vina f458505-mod`.
- `MOLMETAL_QVINA_BIN` override is supported by
  `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (source-inspected
  2026-10-03).

## 6. Isolated toolchains (current state)

- **AiZynthFinder:** runner `environments/aizynth/run.sh` exists and executes
  `uv run --no-project --isolated --offline --python 3.12 --with-requirements
  requirements.lock.txt` (source-inspected 2026-10-03). Its configs
  (`molmetal/configs/aizynth_legacy_v3_{cpu,rocm}.yml`) still point at
  `/mnt/storage/data/molmetal/aizynth_public/...`, a path that does not exist.
  The only AiZynth asset located on this host today is
  `/home/hugo/codes/pocket/.local/share/aizynthfinder/zinc_stock.hdf5`;
  `uspto_legacy_converted.onnx` was not found under `/home/hugo/codes`
  (`find ... -maxdepth 6`). Treat the legacy-v3 asset set as NOT currently
  staged.
- **REINVENT4:** no install found under `/home/hugo`
  (`find -maxdepth 4 -iname "*reinvent*"` → empty, 2026-10-03). The historical
  `/mnt/storage/envs/reinvent4` path is dead (§8).

## 7. Re-measure commands (canonical probes)

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import triton, numpy, scipy, rdkit; print(triton.__version__, numpy.__version__, scipy.__version__, rdkit.__version__)"
.venv/bin/python -c "from molmetal.data.metal_cytotox import _resolve_csv; print(_resolve_csv())"
wc -l .data/molmetal/MetalCytoToxDB.csv
molmetal/bin/vina --version
```

## 8. Historical / STALE (preserved; do NOT cite as current)

`TODO/environment.md` was authored for the 2026-09-18 R16 close-out (single
git commit `4527e4f34e`). The ROCm→CUDA re-pin on this host landed 2026-10-03
(commit `09c2202a1a`, "Wf-A5 … rocm cleanup"). Everything below stopped being
true by 2026-10-03 and is kept only as history:

- ~~GPU target: AMD RX 7800 XT (gfx1101, RDNA3) + integrated Radeon 780M
  (gfx1103).~~ Replaced by the two NVIDIA GPUs in §2. The wave64 /
  `waves_per_eu` CDNA notes, MIOpen cache guidance, and
  `ROCTRACER_HOME` / `HSA_*` / `HIP_PATH` troubleshooting do not apply under
  CUDA.
- ~~`torch 2.14.0+rocm7.2`, HIP `7.2.53211`, `triton-rocm 3.8.0`, index
  `https://download.pytorch.org/whl/rocm7.2`.~~ Replaced by torch
  `2.14.1+cu130` + `triton==3.8.0` from the `pytorch-cu130` index (§3).
- ~~"ROCm does not honor `CUDA_VISIBLE_DEVICES`; restrict with
  `HIP_VISIBLE_DEVICES`."~~ Measured reversal: `CUDA_VISIBLE_DEVICES` works,
  `HIP_VISIBLE_DEVICES` is ignored (§2).
- ~~Data root `/mnt/storage/data/molmetal`; crossdocked staged there at
  8.3 GiB; REINVENT4 at `/mnt/storage/envs/reinvent4`; AiZynth assets under
  `/mnt/storage/data/molmetal/aizynth_public`.~~ `/mnt/storage` does not exist
  (2026-10-03). Crossdocked now measures 9.7 MB at
  `.data/molmetal/crossdocked`. The old "100 test pairs + 100,000 train pairs"
  loader claim was NOT re-verified in this update.
- ~~QVina at `molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02`.~~
  Absent; canonical binary is `molmetal/bin/vina` (§5).
- ~~Import-time banner `[molmetal] applied gfx1101 env vars: ...` firing on
  this NVIDIA host.~~ Fixed 2026-10-03 in `molmetal/__init__.py`: the summary
  now separates genuinely-applied gfx1101 vars from the unconditional general
  hygiene vars, and prints "no gfx1101 GPU visible; gfx1101 env vars skipped"
  when the AMD path is not hit.
- AiZynth legacy-v3 provenance record (2026-09-13: "28 inputs, max probability
  error 3.28e-6", official legacy v3 USPTO weights + 17,422,831-entry ZINC
  stock): historical measurements, preserved in
  `molmetal/reports/aizynth_real_backend_20260913/report.md` and
  `environments/aizynth/README.md`; do not reuse as current until re-measured.
