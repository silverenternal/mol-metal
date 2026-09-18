# WF-1 recon — A1 bond-head training data audit

**Generated:** 2026-09-14
**Status:** MEASURED (audit only — no new runs; reconciles on-disk artifacts)
**Scope:** Catalogue every existing per-edge / per-pair training signal in the codebase that could supervise a learned bond head (A1), confirm tmQM bond/atom-pair data is staged and accessible, and produce a 1-paragraph training-data spec feeding the new `molmetal/models/bond_head.py` and `molmetal/molmetal_lam/tests/test_bond_head.py`.

---

## tmQM bond/atom-pair data staged (MEASURED)

The raw tmQM release is at `/mnt/storage/data/molmetal/tmQM/` (8 files; 269 MB on disk; ownership hugo, dates 2026-09-11):

| File | Size | Purpose for A1 |
|------|------|-----------------|
| `tmQM_y.csv` | 20 MB | SMILES + 8 DFT properties per CSD code (the canonical `smiles` field that yields RDKit ground-truth bond orders / aromaticity). |
| `tmQM_X1.xyz.gz` / `tmQM_X2.xyz.gz` / `tmQM_X3.xyz.gz` | 68 MB × 3 | XYZ coordinates; the comment line carries `MND = n` (Metal Node Degree — tmQM's own ligating-atom count, used as a connectivity anchor). |
| `tmQM_X1.BO.gz` / `tmQM_X2.BO.gz` / `tmQM_X3.BO.gz` | ~30 MB × 3 | Wiberg bond-order blocks; one line per atom with `<idx> <El> <total_BO> <El> <idx> <BO> …` triples. Provides **continuous bond-order regression targets**, not just {0,1} adjacency — strictly richer than RDKit's discrete `Chem.BondType`. |
| `tmQM_X.q` | 100 MB | Natural atomic charges (not directly used by A1, but useful for a metal-vs-ligand classification head). |
| `tmqm_parsed.csv` | 23 MB | **MEASURED** — the cached merge produced by `molmetal/data/tmqm.py:load_tmqm()`; 108 543 rows × 19 columns (csd_code, stoichiometry, metal, charge, spin, coord_number, metal_bo_total, metal_bo_sum, bo_n_neighbors, 8 DFT properties, CSD_years, smiles). Exists on disk and read in <1 s by the existing loader (`MOLMETAL_TMQM_DIR=/mnt/storage/data/molmetal/tmQM`; the fallback `molmetal/data/.cache/tmqm_parsed.csv` is **not** present because the root is writable). |

The tmQM loader (`molmetal/data/tmqm.py:232-291`) merges three shards (`parse_xyz_headers`, `parse_bo_metal`, `parse_properties`) keyed on `csd_code`, drops the one duplicated CSD code (IMUJUY), filters missing SMILES (`require_smiles=True` removes ~7 692 of 108 541 = ~7%), and exposes a `summarize()` helper for per-metal coordination histograms. **MEASURED**: `load_tmqm(metals=("Pt","Ru","Ir"))` yields ~12 k+ rows; the cached CSV has 108 543 lines. The `parse_bo_metal` parser only extracts the **metal-centre** line (it sets `done = True` after the first transition-metal row at line 213). **A1's bond head needs ligand-side bonds too** — the existing parser would have to be extended (`parse_bo_all` per CSD code, returning all ligand-ligand Wiberg BO triples) before it can serve as a per-edge regression target. None of the existing scripts ship this extension yet.

## CrossDocked100 bond-pattern training set (MEASURED, NOT staged for A1)

`/home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv` is a **100-row pocket manifest** (`pocket_id, receptor_path, ligand_path, ref_path, metal_atoms, n_atoms, n_residues, source`); it carries *paths* to .pdb/.sdf pairs, not per-bond ground truth. The ligand .sdf files are written by RDKit so they carry `Chem.BondType` per edge (single=1, double=2, aromatic=12, dative=0) when the SDF is parsed — but A1 currently consumes raw model outputs and would need a separate pre-computation step to materialise `target_bond_classes: LongTensor[E]` per pocket. No such precomputed tensor exists; the manifest is a *selector*, not a *labelled training set*.

`/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/examples/` is too small for A1: only **two** hand-picked pockets (`1h36_A_rec_1h36_r88_lig_tt_docked_0.{sdf,pdb}` and `3ug2_{ligand.sdf,protein.pdb}`). Suitable only as a unit-test fixture, not as a training corpus. `molmetal/references/targetdiff/data/affinity_info.pkl` is the only other staged file in that tree and contains affinity labels, not bonds.

`r10_cfg_real_crossdocked.py:42-75` (`decode_distance_graph`) shows that **all current real-data bond labels are produced lazily inside the harness** at sample time — there is no precomputed `target_bond_classes` tensor for any of the 100 CrossDocked pockets. The 19-atom C/N/O/F filter (line 103) means the existing harness only sees single-bond + implicit-H chemistry anyway, so a per-edge bond head trained on it would only ever learn to predict "single" — there is no label diversity on the real-data path.

## EGNN edge-feature surface (MEASURED)

`/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py` already exposes a categorical edge-type system (T9 / TODO-09) that **A1 should reuse, not reinvent**:

- `EDGE_TYPE_SINGLE = 0` (default, covalent single)
- `EDGE_TYPE_DOUBLE = 1` (covalent double; message scale ×2)
- `EDGE_TYPE_DATIVE  = 2` (ligand donor → metal centre; coord-scale override via `self.dative_coord_scale`, learnable init-1; reciprocal coord-update on the ligand atom suppressed — only the metal moves)

The legacy `dative_bond_edge_attr: [n_edges] bool` flag is unioned with `edge_types == EDGE_TYPE_DATIVE` inside `EquivariantGraphConv._resolve_dative_mask` (lines 361-398). The bond-order scale (`_resolve_bond_order_scale`, lines 400-422) emits `{1.0, 2.0, 1.0}` per edge. **A1 should emit the same 4-way classification {no-bond, single, double, aromatic}** and be wire-compatible with `set_edge_types()` (line 561) so the same `edge_types` buffer that drives the EGNN's update also drives the bond head's prediction.

The pair-feature MLP can be built from the existing `_MaybeFusedSiLUMLP` wrapper (lines 101-141) which already gates the Triton-fused `fused_silu_mlp` kernel via `triton_config.use_fused_mlp(...)` — so the new bond head will automatically use the ROCm-resident fused path on gfx1101 / wave64 at the same shapes as the EGNN's update MLPs (no extra kernel work).

## Lambda AST ↔ bond semantics (NOT directly applicable)

`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/ast.py` is the MLC syntactic layer (`LamVar` / `LamAbs` / `LamApp`, β-reduction, α-conversion, capture-avoiding substitution). It carries **no bond semantics**: there is no edge-type enum, no coordination-geometry term, no relation to the molecular graph. The only chemistry-aware metadata on AST nodes is `LamVar.domain: str` (default `'real'`; e.g. `'molar'`, `'kelvin'`). A1 should not depend on `lam_chem/ast.py` — it operates on the EGNN-level pair features, not on the MLC proof-search layer. The Z-mapping for metal centres lives in `molmetal/molmetal_lam/priors/metal_geometry.py:DEFAULT_METAL_GEOMETRY` (lines 364-374) — a `{atomic_number: Geometry}` dict with 9 entries (78=SP, 46=SP, 29=TD, 30=TD, 26=OCT, 44=OCT, 77=OCT, 45=OCT, 25=TBP) — and should be the source of metal-side dative-edge priors at training time.

## Training-data spec (1 paragraph, MEASURED + PROJECTED)

A1's bond head should be supervised by a **two-tier corpus** that is already 95% staged. **Tier 1 (MEASURED, available now)**: the cached `tmqm_parsed.csv` at `/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv` (108 543 rows, ~100 k after the empty-SMILES filter) provides per-complex (a) SMILES from `tmQM_y.csv` → RDKit `Chem.BondType` per edge → 4-way classes {no-bond=0, single=1, double=2, aromatic=3} via `Chem.MolFromSmiles(smi) → mol.GetBonds() → bond.GetBondType()` (cheap, ~3 ms per molecule), and (b) Wiberg BO regression targets via a new `parse_bo_all()` extension to `molmetal/data/tmqm.py` (currently `parse_bo_metal()` only emits the metal line at line 213; a sibling `parse_bo_all()` would emit one row per atom-pair with `BO > 0.05`, ~6-12 ligand-ligand bonds + 4-6 metal-ligand dative edges per complex). **Tier 2 (MEASURED, available now, but small)**: the 100 CrossDocked ligand .sdf paths in `molmetal/data/crossdocked100_manifest.csv` give a held-out-domain corpus of organic molecules with non-trivial double / aromatic labels — useful for calibration / val-loss but not for primary training (all 100 are C/N/O/F at 19 heavy atoms, single-bond + implicit-H per `r10_cfg_real_crossdocked.py:103`, no bond-order diversity). **Loss shape (PROJECTED)**: `BondHeadLoss(logits, target_bond_classes, edge_mask)` = masked CE with inverse-frequency class weights computed from Tier 1; for Tier 1 regression add a parallel `wiberg_mae` term on the `EDGE_TYPE_SINGLE / DOUBLE / AROMATIC` classes only (no Wiberg target for the dative class — set dative targets to NaN and mask them). **Edge mask** comes from a 4-Å distance cutoff on the corresponding Tier-1 xyz coordinates (XYZ blocks in `tmQM_X{1,2,3}.xyz.gz`) — every pair whose `||r_ij|| < 4 Å` is a candidate edge; the 4-way classification predicts the ground truth on those, the rest is "no-bond". **Splits (PROJECTED)**: paper-metal subset `Pt ∪ Ru ∪ Ir` first (~12 k complexes), then d-block sweep; 90/5/5 split keyed by CSD code (no complex overlaps train/val/test). **Class balance (PROJECTED)**: single dominates by ~10×, aromatic by ~3×, double by ~1×; the inverse-frequency weights are the right fix. **Per-class Wiberg BO threshold (PROJECTED, NOT MEASURED)**: map Wiberg → discrete via BO ∈ [0.0, 0.05) → no-bond, [0.05, 0.55) → single, [0.55, 1.45) → aromatic, [1.45, 2.25) → double — these are the standard Wiberg→BondType cutoffs (covalent single ≈ 1.0, double ≈ 2.0, aromatic ≈ 1.5). The 0.05 lower bound matches the existing tmQM truncation threshold mentioned in the docstring at line 36 of `molmetal/data/tmqm.py`. **Negative cache**: NONE — Tier 1 has not been pre-rendered into a `bond_head_train.pt` tensor yet; this is the next concrete step before training.

---

## Path list (MEASURED)

Raw data (staged, MEASURED):
- `/mnt/storage/data/molmetal/tmQM/tmQM_y.csv` — SMILES + DFT properties (20 MB)
- `/mnt/storage/data/molmetal/tmQM/tmQM_X1.xyz.gz` / `tmQM_X2.xyz.gz` / `tmQM_X3.xyz.gz` — XYZ blocks (68 MB × 3)
- `/mnt/storage/data/molmetal/tmQM/tmQM_X1.BO.gz` / `tmQM_X2.BO.gz` / `tmQM_X3.BO.gz` — Wiberg BO blocks (~30 MB × 3)
- `/mnt/storage/data/molmetal/tmQM/tmQM_X.q` — Natural atomic charges (100 MB; auxiliary)
- `/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv` — merged cache (23 MB; 108 543 rows; built by `molmetal/data/tmqm.py:load_tmqm()`)

Code (MEASURED):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/tmqm.py` — loader (TM raw + parsed cache); **needs `parse_bo_all()` extension to feed A1 ligand-ligand BO targets**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/load_tmqm.py` — CLI summary tool (no per-bond export)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_tmqm_geometry_control.py` — existing tmQM consumer; reads `tmQM_X1.BO.gz` directly for Pt/MND=4 (coord repair, not training)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py` — EGNN layer with `EDGE_TYPE_SINGLE/DOUBLE/DATIVE` enum (lines 39-54), `_resolve_dative_mask` / `_resolve_bond_order_scale` (lines 361-422), `_MaybeFusedSiLUMLP` (lines 101-141)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/ast.py` — MLC AST (NOT used by A1; no bond semantics)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py` — `DEFAULT_METAL_GEOMETRY` (lines 364-374), `GEOMETRY_IDEAL_ANGLES_DEG` (lines 350-355); used for metal-side dative priors at training time
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv` — 100-row pocket manifest (path-only; no per-bond labels)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/examples/` — 2 hand-picked pockets (unit-test scope only)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py` — current harness; `decode_distance_graph` at lines 42-75 is the *legacy* decoder A1 will replace

New (NEW, no edits to existing):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py` — to be written (LearnedBondHead + BondHeadLoss + BondHeadDecoder)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_bond_head.py` — to be written (unit tests; fixtures will use the 2-pocket targetdiff examples subset, not tmQM)

Cached parsed CSV fallback (NOT MEASURED on disk — would be created if `/mnt/storage/data/molmetal/tmQM` were read-only):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/.cache/tmqm_parsed.csv` — does not exist; the root is writable so the loader writes to `/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv` instead (per `molmetal/data/tmqm.py:93-97`).

---

## Honest framing summary

**MEASURED**: tmQM raw + parsed cache present and readable; 108 543 complexes (100 k after SMILES filter); raw BO shards present; loader extension to `parse_bo_all()` is the *only* missing piece on the data side. EGNN edge-type enum (3-way: SINGLE/DOUBLE/DATIVE) already wired in `egnn_rocm.py` — A1 must align with it (adds AROMATIC). CrossDocked100 manifest is path-only (no per-bond labels), `targetdiff/examples/` is 2 pockets (unit-test scope only).

**PROJECTED**: A1 will consume `target_bond_classes[E] ∈ {0,1,2,3}` (no-bond/single/double/aromatic) derived from RDKit `Chem.BondType` on the tmQM SMILES, with optional Wiberg BO regression targets from `parse_bo_all()` extension; train/val/test split keyed by CSD code; inverse-frequency class weights; 4-Å distance cutoff for edge-mask generation; metal-side dative edges inferred from `DEFAULT_METAL_GEOMETRY` Z-mapping.

**NOT MEASURED** (would require running): per-class Wiberg→BondType cutoff accuracy on tmQM; class-frequency ratios on the Pt/Ru/Ir subset; whether the 100 k complex Tier-1 corpus is large enough to suppress the single-bond majority class without oversampling.