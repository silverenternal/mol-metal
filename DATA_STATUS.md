# Dataset status — local-host inventory (2026-09-25)

This file documents where each mol-metal dataset lives on this host. The
loaders consult env vars `MOLMETAL_DATA_DIR` and `MOLMETAL_TMQM_DIR`
(env.sh sets them to `.data/molmetal/` and `.data/molmetal/tmQM`).

| # | Dataset | Status | Path / source |
|---:|---|---|---|
| 1 | **CrossDocked2020** (100 K pocket-ligand pairs) | ✅ REUSED — symlinked from `/home/hugo/codes/pocket/data/crossdocked2020/` | `.data/molmetal/crossdocked/` (split_by_name.pt + crossdocked_pocket10.tar.gz + 2464 pocket dirs all symlinked) |
| 2 | **MetalCytoToxDB** (26801 rows, 24 cols, 4.8 MB CSV) | ✅ DOWNLOADED — Zenodo record `17106822` (Krasnov 2026, J. Med. Chem.) | `.data/molmetal/MetalCytoToxDB.csv` |
| 3 | **tmQM** (100849 rows, 7 files, 300 MB) | ✅ DOWNLOADED — `https://github.com/uiocompcat/tmQM` | `.data/molmetal/tmQM/{tmQM_y.csv, tmQM_X{1,2,3}.{xyz,BO}.gz}` |
| 4 | **NCI-60 GI50** (4.85 M rows, 420 MB CSV) | ✅ DOWNLOADED — `https://wiki.nci.nih.gov/.../GI50.zip` | `.data/molmetal/NCI60_GI50/GI50.csv` |
| 5 | **Known Pt drugs** (7 FDA-approved Pt drugs) | ✅ PRE-SHIPPED | `molmetal/data/known_pt_drugs.csv` |
| 6 | **metallo_drugs_500_train.csv** (PlatinAI 500-mol MaxMin pool, 49 KB) | ✅ PRE-SHIPPED | `molmetal/data/metallo_drugs_500_train.csv` |
| 7 | **metallo_scaffold_split/** (PlatinAI SMILES with scaffold split) | ✅ PRE-SHIPPED | `molmetal/data/metallo_scaffold_split/{train,val,test}.csv` |
| 8 | **PlatinAI full xlsx** (226 K SMILES + A2780/MCF7 predictions) | ⚠️ UNREACHABLE — README URL (`github.com/ai-safety-institute/PlatinAI`) returns 404; ChemRxiv gated; Zenodo has only PLATINAI Brand Book (unrelated). The pre-shipped `metallo_drugs_500_train.csv` covers the 500-mol subset used by the de novo vertical. | (none on this host) |
| 9 | **TDC / ADMET datasets** (ames, b3db, caco2_wang, herg, etc.) | ✅ AVAILABLE — `/home/hugo/codes/pocket/data/*.tab` if needed | (out of canonical tree) |
| 10 | **PDBBind++ 2020** (already extracted) | ✅ AVAILABLE — `/home/hugo/codes/pocket/data/pdbbindpp-2020/` | (out of canonical tree) |

## Why this layout

- **Project-local** (`.data/molmetal/`) instead of `/mnt/storage/data/molmetal/`
  because `/mnt` is read-only on this host and `sudo` requires a password.
- **Pre-existing data** (CrossDocked, ADMET, etc., downloaded for the
  sibling `pocket/` project) is **symlinked** rather than re-downloaded.
- **PlatinAI** has no working GitHub URL and its ChemRxiv supplement is
  Cloudflare-gated; the 500-mol pool pre-shipped in `molmetal/data/`
  covers the de novo vertical but **not** the full A2780/MCF7 oracle
  benchmarks in §4.6.1. If the user has the original xlsx files
  elsewhere on the host, drop them in `.data/molmetal/`.

## Network gotcha: Zenodo is DNS-blocked

`zenodo.org` resolves to `0.0.0.0` on this host (blocked at the local
DNS resolver `192.168.110.1`). Use this workaround:

```bash
ZENODO_IP=159.100.250.205   # public Cloudflare anycast IP
curl --resolve "zenodo.org:443:$ZENODO_IP" \
     -o file.csv "https://zenodo.org/records/.../files/file.csv/content"
```

(Other Zenodo subdomains like `doi.org` redirect via Cloudflare and work
fine; only direct `*.zenodo.org` is blocked.)

## Loader patches applied (2026-09-25)

The original loaders hardcoded `/mnt/storage/data/molmetal/...` despite a
comment claiming they honoured `MOLMETAL_DATA_DIR`. Patched to consult
the env var with the legacy path as fallback:

- `molmetal/data/cytotox.py:62-65` — `DEFAULT_DATA_DIR` / `DEFAULT_CSV` / `DEFAULT_CACHE_DIR`
- `molmetal/data/crossdocked.py:54-56` — `DEFAULT_DATA_DIR` / `DEFAULT_ARCHIVE` / `DEFAULT_EXTRACT_DIR`
- `molmetal/molmetal_lam/lam_chem/platinai_dataset.py:72-82` — `PLATINAI_MBFINDER_DEFAULT` / `PLATINAI_A2780_DEFAULT` / `PLATINAI_MCF7_DEFAULT` / `METAL_CYTOTOX_DEFAULT` / `NCI60_GI50_DEFAULT`
- `molmetal/data/tmqm.py:90` — `tmqm_dir()` already honoured `MOLMETAL_TMQM_DIR` ✓

## Verifying the loaders

```bash
source env.sh

# MetalCytoToxDB (canonical, 26801 rows)
uv run python -c "
from molmetal.molmetal_lam.lam_chem.platinai_dataset import MetalCytoToxDataset
ds = MetalCytoToxDataset(n_max=5000)
print('rows:', len(list(ds)))
print('Ru:', len(ds.get_by_metal('Ru')))
"

# tmQM (100849 rows)
uv run python -c "
from molmetal.data.tmqm import load_tmqm, summarize
print(summarize(load_tmqm()).per_metal)
"

# CrossDocked2020 (100000 pairs, symlinked from pocket/)
uv run python -c "
from molmetal.data.crossdocked import CrossDockedDataset
import os
ds = CrossDockedDataset()
s = ds[0]
print('pair:', s['pocket_pdb_path'])
print('exists:', os.path.exists(s['pocket_pdb_path']))
"

# NCI-60 GI50 (4.85M rows)
uv run python -c "
import pandas as pd
print(len(pd.read_csv('.data/molmetal/NCI60_GI50/GI50.csv', low_memory=False)))
"
```
