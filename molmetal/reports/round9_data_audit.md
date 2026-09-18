# Round-9 SBDD Pocket Data Audit (2026-09-13)

Read-only inventory of protein/ligand structure files staged in
`molmetal/`, `references/`, and `/mnt/storage/data/molmetal/` that are
usable for r4c (Lambda + SBDD) docking pilots. Cross-checked against
the user's named targets: **1h36, 830c, MMP13, MMP2, CA2**.

## (a) Inventory of staged pockets

### A.1 — CrossDocked2020 (Luo 2021 split, full)
**Location:** `/mnt/storage/data/molmetal/crossdocked/`

```
CrossDocked2020_cascadediff.zip     1.6 GB   (raw archive, dual-EOCD quirk)
crossdocked_pocket10.tar.gz         1.6 GB   (inner tar)
split_by_name.pt                    15 MB    {"train": [..], "test": [..]}
extracted/crossdocked_pocket10/     2,464 protein dirs   <-- pre-extracted
```

`split_by_name.pt` (loaded with venv torch) confirms:
- **train:** 100,000 pairs (1,907 unique UniProt-style protein IDs)
- **test:** 100 pairs (Luo 2021 official 100-pocket subset)
- Each pair = `(<pocket10.pdb>, <ligand.sdf>)` inside a per-protein folder
- Loader already exists: `molmetal/data/crossdocked.py`
  (handles the dual-EOCD zip, raw-deflate entries, prefers `extracted_dir`)

Metal-binding / Zn-coordinating pockets present in the staged set:

| Target | Train hits | Test hits | Notes |
|---|---|---|---|
| `MMP13_HUMAN_*` | 548 | 0 | Zn-coordinating collagenase; rich train material |
| `CAH2_HUMAN_*` (CA-II / carbonic anhydrase 2) | 869 | 0 | Zn-coordinating; large scaffold-diverse train set |
| Other `CAH*` isoforms | many | 0 | CAH1/4/5A/7/9/12/13/14 |
| Other `MMP*` (MMP8, MMP12) | yes | 0 | Zn-dependent endopeptidases |
| `830c` ligand token | 25 | 0 | hydroxamate MMP-13 inhibitor in 25 docked poses |
| `1h36` pocket token | 0 | 1 | hit is `SQHC_ALIAD_1_631_0/1h36_A_rec_1o79_r23_...` (scaffold-hopping example, not heat-shock HSP90-1h36) |
| `MMP2` / `MMP-2` | 0 | 0 | **NOT in CrossDocked2020** |
| `CA2` token | 30 | 0 | matches `_ca2` ligand names (Zn ion), not human CA2 protein |

### A.2 — Curated single-pockets (project-native)

| Path | Target | Use |
|---|---|---|
| `molmetal/data/mmp13_real/830c.pdb` (407 KB) | **830c / MMP-13** | Full holo crystal structure. HEADER: "MATRIX METALLOPROTEASE … COLLAGENASE-3 (MMP-13) COMPLEXED TO A SULPHONE-BASED HYDROXAMIC ACID". |
| `molmetal/data/mmp13_real/830c_ligand.pdb` (1.2 KB) | MMP-13 native ligand | Reference pose for RMSD / success-rate eval |
| `molmetal/data/mmp13_real/dock_mmp13.py` | — | Pre-existing docking script (R4 docking workflow) |
| `molmetal/data/mmp13_real/dock_results.csv` | — | Pre-existing benchmark output |
| `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` | **1h36 (HSP90)** | Pre-cropped pocket10 file, ready to dock against |
| `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0.sdf` | HSP90 native ligand | Reference pose |
| `molmetal/references/targetdiff/examples/3ug2_protein.pdb` + `3ug2_ligand.sdf` | 3ug2 (HSP90 paralog) | Extra HSP90 pocket |

### A.3 — Reference-repo pockets (NOT project-owned, low-priority)

These live inside `references/<Repo>/` clones (DiffDock, Pocket2Mol,
FlowDock, TankBind, etc.) and would only be used as last-resort
fallback. They are **not** redistributed and pull-only:

- **DiffDock examples:** 1a46, 1cbr, 6ahs, 6moa, 6o5u, 6w70, 1a0q (1 mol2 + sdf)
- **Pocket2Mol example:** 4yhj
- **Flowr examples:** bace, cdk2, ptp1b, tyk2 (+ ligand SDFs)
- **TankBind examples:** 6dlo, 6hd6 (single + HTVS)
- **RxnFlow examples:** 6oim
- **SynFlowNet examples:** 5np8, 7wl4, 8azr (KRAS)
- **Posebusters dataset (flowr / FLOWR):** 1ia1, 1of6, 1s3v, 1uou (protein + ligand SDFs)
- **REINVENT4 maize tutorial:** 1DB5, 6G5J, 8ZYP (apo + box PDBs)
- **DrugDesignAI-Benchmark:** BRAF ensemble (3SKC, 2FB8, 1UWH), 100+ snap PDBs — large metalloprotein-relevant subset
- **FlowDock amino-acid templates:** ALA..VAL (irrelevant for protein docking)

### A.4 — Cache/scratch — out of scope
- `molmetal/data/3d_cache/` (referenced by `temporal_grids.py`) — precomputed featurisation cache, not raw PDBs
- `molmetal/checkpoints/l2_persistent_tree/` — Lambda search tree dump
- `molmetal/checkpoints/*.pt` — model weights (Ru / hybrid / fm_pocket), not structure files

### A.5 — Verdict on the five named targets

| Target | Stage status | Best file to dock against |
|---|---|---|
| **1h36** (HSP90 N-term) | YES | `references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` (+ .sdf) |
| **830c / MMP-13** | YES | `molmetal/data/mmp13_real/830c.pdb` (+ `830c_ligand.pdb`) |
| **MMP13** (general) | YES — train only | pick any `MMP13_HUMAN_*/<…>_pocket10.pdb` in `extracted/crossdocked_pocket10/` (548 pairs) |
| **MMP2** | NOT in CrossDocked2020 | No pocket file in any `molmetal/`, `references/`, or `/mnt/storage/data/molmetal/` tree. Would need fresh PDB→pocket10 fetch (PDB IDs e.g. 1qib, 1hov, 3ayu). |
| **CA2 / CA-II** | YES — train only | pick any `CAH2_HUMAN_2_260_0/<…>_pocket10.pdb` (869 pairs, 1,907 unique proteins total). Note: literal "CA2" UniProt ID isn't a CrossDocked entry; `CAH2_HUMAN` *is* human carbonic anhydrase II. |

## (b) Max N for r4c pilot

**Bottleneck is not dataset size** — the staged CrossDocked2020 split is
already complete (100,000 train / 100 test) and on local NVMe.

Recommended r4c pilot sizing:

| Pilot | N (train) | N (test) | Wallclock target | Use |
|---|---|---|---|---|
| **Smoke (default for first run)** | 50 | 10 | < 5 min | Verify pipeline wires (smoke test) |
| **R4c mini** | 500 | 100 (full test) | 10–20 min | First SOTA-comparable signal |
| **R4c median** | 5,000 | 100 (full test) | 1–2 h | Statistically meaningful per-pocket Vina-RMSD |
| **R4c full** | 100,000 | 100 | 6–12 h | Match TargetDiff/DiffDock recipe |

The 100-pocket test set is fixed (Luo 2021 official). For train, the
loader streams from the pre-extracted dir, so disk I/O is not the
constraint — the constraint is per-pocket Vina / DiffDock cost. With
the existing `data/crossdocked.py` loader already wired and the
pocket10 cache hot on `/mnt/storage/`, the practical r4c pilot cap is
**N_train = 100,000, N_test = 100** (i.e. the entire Luo 2021 split).

If we want a *metal-binding-focused* sub-pilot instead:
- **Metal/zinc train pool:** grep the train list for `MMP|CAH|MMP|COB|IPNS|RENI` folders → 55 protein folders, ~5,000–7,000 pockets
- **CA-II-only:** `CAH2_HUMAN_2_260_0` → 869 pockets
- **MMP-13-only:** `MMP13_HUMAN_104_271_0` → 548 pockets

These are large enough for a 1-hour median pilot and a 12-hour full
metal-only sweep.

## (c) CrossDocked100 download feasibility

Probed with `curl -sI` (15 s timeout) on 2026-09-13:

| URL | HTTP | Verdict |
|---|---|---|
| `https://bits.csb.pitt.edu/` (root) | **200 OK** | host reachable |
| `https://bits.csb.pitt.edu/files/` | **200 OK** | directory listing served |
| `https://bits.csb.pitt.edu/files/crossdock2020/` | **200 OK** | directory served |
| `https://bits.csb.pitt.edu/files/crossdock2020/` contents | (parsed) | contains only `PDBBind2016_caches.tar.gz`, `PDBbind2016.tar.gz` — **no CrossDocked files** |
| `https://bits.csb.pitt.edu/files/crossdocked100.tar.gz` | 404 | not present |
| `https://bits.csb.pitt.edu/files/crossdocked_test.tar.gz` | 404 | not present |
| `https://bits.csb.pitt.edu/files/crossdocked_pocket10_test.tar.gz` | 404 | not present |
| `https://bits.csb.pitt.edu/files/CrossDocked2020_test.tar.gz` | 404 | not present |
| `https://zenodo.org/record/13871492/files/posebusters_benchmark_set.zip` | **301 MOVED PERMANENTLY** | Zenodo reachable; Posebusters via Zenodo is viable |

**Conclusion:**

- The Pitt CSB host is reachable, but the canonical "CrossDocked100 test
  tarball" URL is **not** published there anymore — only PDBbind
  archives remain in `crossdock2020/`.
- **A fresh download is NOT needed:** the 100-pocket Luo 2021 test set
  *plus* the full 100,000-pocket train set is already staged at
  `/mnt/storage/data/molmetal/crossdocked/` (1.6 GB zip + pre-extracted
  to 2,464 protein dirs, with `split_by_name.pt` matching exactly).
  This is the same archive the crossdocked loader was written for.
- For the wider 100-pocket benchmark families (DiffDock / EquiBind),
  the only large data tarball on `bits.csb.pitt.edu/files/` is
  `dkoes_equibind.tar.gz` (3.1 GB) — not needed for r4c.
- For PoseBusters benchmark (40 systems, useful for a "real-world
  cross-docking" extension later): Zenodo at record 13871492 returns
  301 — follow the redirect to grab `posebusters_benchmark_set.zip`.

## Recommended r4c data plan

1. **Use the staged CrossDocked2020 as-is.** No download.
   `molmetal/data/crossdocked.py` already loads it (pre-extracted path
   preferred). `split_by_name.pt` is the canonical Luo 2021 split.
2. **For the 1h36 / 830c demos:** point the orchestrator at the two
   curated files in A.2. These are pre-cropped / full-holo, ready to
   dock against without pocket10 re-extraction.
3. **For MMP2 (not staged):** add a one-time fetch step that pulls
   e.g. PDB 1qib / 1hov / 3ayu from RCSB and crops a pocket10. Defer
   unless the r4c pilot explicitly needs MMP2.
4. **Defer PoseBusters / Zenodo download** until round-10; not in scope
   for r4c but the URL is reachable.
