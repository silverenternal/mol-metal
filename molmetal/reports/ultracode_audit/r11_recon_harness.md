# R11 parity harness — on-disk recon (N=50 dry-run)

**Project root**: `/home/hugo/codes/try_triton_on_rocm`
**Python / stack**: uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64
**Scope**: verify harness callable, document CLI surface, locate data + engines. **No experiments run.**

---

## 1. Harness on disk (CLI surface + engine flag)

The Round-11 parity harness lives in the **library + test** tree, not as a top-level CLI script in `molmetal/scripts/`:

- **Library module**: `molmetal/molmetal_lam/benchmarks/engine_parity.py`
- **Pytest entry**: `molmetal/tests/test_round11_engine_parity.py`
- **Round-11 ladder** (different concern: install/fallback): `molmetal/scripts/round11_install_ladder.py`

Public API (from `engine_parity.py`):

```
def compare_engines(
    pockets: Iterable[str] | None = None,
    ligands: Iterable[str] | None = None,
    output_dir: str | Path = "molmetal/reports/round11_engine_parity",
    pocket_names: Iterable[str] | None = None,
    ligand_smiles: Iterable[str] | None = None,
) -> dict
```

Internally it loops `(pocket, ligand)` and scores each with `real_vina_score`, `mock_qvina`, `mock_quickvina2`. Default `POCKETS = ["1h36","830c","mmp2","mmp13","ca2"]` and a hard-coded 10-SMILES `LIGANDS` list. Outputs: `results.csv` + `summary.json` (per-pocket Pearson r + MAD).

**Engine flag**: the dispatch happens at the **adapter layer** (`molmetal/molmetal_lam/sbdd_env/vina_adapter.py`, `SUPPORTED_ENGINES = ("auto","vina","vina-cli","qvina","quickvina2")`, `DEFAULT_ENGINE="vina"`), not inside `engine_parity.py`. The library currently uses two **stochastic mocks** (`mock_qvina`, `mock_quickvina2`) over a `real_vina_score` hook that returns `None` unless `vina` is on PATH **and** a pocket file exists at `<data_dir>/<pocket>{.pdbqt,.pdb}`. `engine_parity.py` has **no CLI** — invocation is via:

```python
from molmetal_lam.benchmarks.engine_parity import compare_engines
compare_engines(output_dir="...")              # full default sweep
compare_engines(["1h36","830c"], smiles[:50])  # N=50 slice
```

Existing pytest uses `tmp_path` for output (see `test_compare_engines_produces_files`). Tests are runnable today; the `@pytest.mark.slow` + `test_real_docking_available` will be **skipped** here because `vina` CLI is broken at the dynamic-linker level (see §4).

## 2. Documented N=50 protocol — `molmetal/reports/round9_qvina_parity.md` §5

Round-9 is a **read-only** literature audit; §5 ("Practical recommendation for Mol-Metal") recommends the protocol but **does not pin a specific N=50**. The Round-9 recommendations are:

| Parameter | Value | Citation |
|---|---|---|
| **Primary engine** | QuickVina 2 at `exh=8` | Alhossary 2015 (*Bioinformatics* 31:2214) |
| **Parity claim** | r=0.967 Pearson correlation vs Vina 1.2.7 at the same `exh=8` | Alhossary 2015 |
| **Not measured** | `exh=8` QVina 2 ≈ `exh=16` Vina 1.2.7 (assumption only) | §3 assertion #1 |
| **Not measured** | byte-exact scoring parity (0.967/0.911 are between-engine, not byte-exact) | §3 assertion #2 |
| **Per-run kcal/mol σ** | unmeasured in literature — "round-9 should resolve next" | §3 assertion #4 |
| **Baseline table** | engine must be locked in column header; do not mix SMINA / Vina / QVina 2 | §5 recommendation #5 |

Round-9 does **not** document a box, pocket file, or seed for an N=50 protocol. The pragmatic defaults in this repo are visible in `scripts/prepare_crossdocked_receptor.py`: `DockingConfig(seed=42, exhaustiveness=1, n_poses=1)` — and the adapter uses `config.exhaustiveness` and `config.seed` directly (vina_adapter.py:660, 702). **Caveat**: `exh=1` is the dev-loop value; for parity measurement round-9 §5 implicitly assumes `exh=8` (QVina 2) vs `exh=8` (Vina 1.2.7) — i.e., same exhaustiveness, not `exh=8` vs `exh=16`.

## 3. test_001 receptor + box

`/mnt/storage/data/molmetal/crossdocked_pocket10/` **does not exist**. The actual on-disk path is **one level deeper** (under `extracted/`):

- **Manifest CSV**: `/home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv` (100 rows, columns: `pocket_id, receptor_path, ligand_path, ref_path, metal_atoms, n_atoms, n_residues, source`)
- **CrossDocked root**: `/mnt/storage/data/molmetal/crossdocked/`
- **Pocket-10 root**: `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/`
- **test_001 dir**: `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/`
- **test_001 receptor (PDB)**: `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb` (also `4ac3_...pocket10.pdb`, `1hm9_...pocket10.pdb`)
- **test_001 ligand (SDF)**: `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0.sdf`
- **Pre-prepared PDBQT** (already converted): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/crossdocked_first10_preparation/test_001/receptor.pdbqt` and `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/crossdocked_first10_resolved/test_001/strict/receptor.pdbqt`
- **Manifest split file**: `/mnt/storage/data/molmetal/crossdocked/split_by_name.pt`

**Box coordinates**: NOT shipped with the CrossDocked2020 pocket10 tarball. Standard practice (per TargetDiff / DiffSBDD convention) is to derive the box from the bound-ligand centroid ± radius. There is no `.box.txt`/`.txt` next to the receptor in this dataset; one would need to compute it from the SDF centroid (or call `prepare_crossdocked_receptor.py`).

## 4. Engine reachability

| Binary | Path | State |
|---|---|---|
| **Vina 1.2.7 (apt)** | `/usr/bin/vina` | **Present, but linker-broken** (`libboost_thread.so.1.90.0`, `libboost_serialization`, `libboost_filesystem`, `libboost_program_options` all `not found`). `vina --version` exits with library-load error. |
| **Vina Python binding** | installed via `vina` pip package (Eberhardt 2021) | **Reachable** — used by `VinaDockingAdapter` in-process (no CLI binary needed). Already imported in `molmetal_lam/sbdd_env/vina_adapter.py`. |
| **QuickVina 2 CLI** | `which quickvina2` → not found | **Not on PATH**. |
| **Qvina02 vendored** | `/home/hugo/codes/try_triton_on_rocm/molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02` (3.3M, executable) | **Reachable** but this is **QVina 0.2** (Alhossary 2015 pre-release), **NOT QuickVina 2**. Used by SoftMol gated-MCTS code path. |
| **Vina-GPU 2.1 source** | `/mnt/storage/tools/vina_gpu21_source/Vina-GPU-2.1-...` | **Source only**, no compiled binary in this tree. |

**Net result**: Vina 1.2.7 reference path works via the Python binding. QuickVina 2 CLI binary is **NOT installed**; the install-ladder Tier 1-4 candidates (`uv tool install qvina`, `pip install qvina`, `conda install -c conda-forge qvina`, `https://qvina.github.io/qvina_2.1_linux_x64.tar.gz`) are all dry-run-only in this round. **Byte identity is already proved** (per the user's brief — see `quickvina2_binary_identity.md`); this recon only needs to **stage the scoring-identity harness**.

## 5. Path list (one-glance summary)

```
Harness
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/benchmarks/engine_parity.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_round11_engine_parity.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/round11_install_ladder.py

Adapter (engine flag)
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/vina_adapter.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/vina_gpu_adapter.py

Documentation
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/round9_qvina_parity.md  (sections 1-5)
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/round11_install_status.json
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/quickvina2_binary_identity.md  (byte identity)

Manifest + data
  /home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv  (100 rows)
  /mnt/storage/data/molmetal/crossdocked/split_by_name.pt
  /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/         (pocket-10 root)
  /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/  (test_001 dir)
  /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb
  /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0.sdf

Pre-prepared PDBQT (CrossDocked first-10)
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/crossdocked_first10_preparation/test_001/receptor.pdbqt
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/crossdocked_first10_resolved/test_001/strict/receptor.pdbqt

Engines
  /usr/bin/vina                                        (Vina 1.2.7 CLI, libboost-broken on this host)
  /home/hugo/codes/try_triton_on_rocm/molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02  (QVina 0.2 vendored, NOT QVina 2)
  pip `vina` Python binding (in-process, reachable)
  /mnt/storage/tools/vina_gpu21_source/Vina-GPU-2.1-180272b8a5265d6ed9664178345933cebe2cd349/  (source only)
```

## 6. Verdict — harness on disk and runnable?

| Requirement | Status | Note |
|---|---|---|
| Harness module on disk | YES | `molmetal_lam/benchmarks/engine_parity.py` |
| Pytest coverage on disk | YES | `tests/test_round11_engine_parity.py` |
| Existing manifest N≥50 | YES | `data/crossdocked100_manifest.csv` (100 rows; first 50 are `test_000`–`test_049`) |
| test_001 receptor on disk | YES | `.pocket10.pdb` + pre-prepared `.pdbqt` |
| test_001 box coordinates | **NO** (not shipped) | Derive from SDF centroid; not present as `.box.txt` |
| Vina 1.2.7 reference path | YES (Python binding) | CLI binary broken at libboost layer; Python binding works in-process |
| QuickVina 2 CLI binary | **NO** | Not installed; install-ladder Tier 1-4 are dry-run-only |
| QVina 0.2 (not QVina 2) vendored | YES (in `SoftMol`) | Wrong version for parity test |

**Runnable for scoring-identity** only if a QuickVina 2 binary is staged first (Axis A/B of `round11_install_ladder.py`), or if the harness is re-implemented to use the Python `vina` binding + QVina via the existing `vina_adapter.py` adapter (which already has `qvina`/`quickvina2` dispatch but those CLI paths are currently `subprocess`-only and require the binary on PATH).

---

*Recon complete. No code changed. No experiments run.*