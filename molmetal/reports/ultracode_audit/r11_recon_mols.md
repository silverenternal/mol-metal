# R11 recon — 50-molecule source verification

**Project root**: `/home/hugo/codes/try_triton_on_rocm`
**Python / stack**: uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64
**Scope**: verify the molecule source for the Vina-vs-QuickVina-2 parity experiment.
**Byte identity**: already proved (see `molmetal/reports/quickvina2_binary_identity.md`).
**Goal here**: stage the **scoring-identity** dataset — kcal/mol correlation + rank correlation — **50 molecules only**, **no full sweep**.

---

## 1. What the reduced-budget pilot actually used (`r4_click_physical_test10_seed3_v2_*`)

The reduced-budget pilot was a click-tile search, **not a parity measurement**:

- **Input job JSON**: `molmetal/reports/r4_click_physical_test10_seed3_v2.json`
  SHA256 `cdcda44dc4a80363c58ade655ef7f67a5921cb00f58a4457400e4e33dc218aed`
- **Analysis report**: `molmetal/reports/r4_click_physical_test10_seed3_v2_analysis.md`
- **Per-job search config** (from CSV row): `tile_library=standard_12`, `click_rules=CuAAC` only,
  `tile_count=12`, `branching_target=12`, `seed_strategy=click_tile`, `seeds ∈ {0, 42, 1234}`,
  `n_simulations=4`, `max_depth=1`, `top_k=10`.

**Pilot size**: 30 jobs (10 pockets × 3 seeds), 70 generated products, 63 docked, 63 pass PoseBusters.

**Pockets in the pilot** (`pocket_id` column, 10 distinct):

```
test_000 BSD_ASPTE_1_130_0      test_005 receptor_preparation_failed (excluded)
test_001 GLMU_STRPN_2_459_0     test_006 OK
test_002 GRK4_HUMAN_1_578_0     test_007 OK
test_003 GSTP1_HUMAN_2_210_0    test_008 OK
test_004 OK                     test_009 OK
```

**Deduplicated top1 SMILES**: only **2 unique** `top1_smiles` across all 30 rows
(`OCCOCCn1nncc1Cc1ccccc1` and `Cc1cnnn1CCOCCO`), because all rows use the same 12-tile
library + single `CuAAC` click rule and only differ by random seed. Deduplicated across
the *candidate list* (not just top1): **4 unique** products — `OCCOCCn1nncc1Cc1ccccc1`,
`Cc1cnnn1CCOCCO`, `NCc1cnnn1CCOCCO`, `NCc1cnnn1Cc1ccccc1`. The pilot **does not provide
50 unique molecules** for parity scoring.

## 2. Molecule-list emit utility — none dedicated

`molmetal/scripts/` does **not** contain a dedicated `emit_molecules.py` /
`make_mol_set.py` utility. The path of generation is `molmetal/molmetal_lam/tile_lib/click_tiles.py::build_all_click_tiles()`,
which yields the 12-tile library (4 azides + 4 alkynes + 4 partners) used as educts.
Reaction dispatch lives in `molmetal/molmetal_lam/reactions/click_reactions.py`,
which exposes `CLICK_REACTIONS = {"CuAAC", "SPAAC", "SPC", "DielsAlder"}` (note: 4
named reactions, not 5). The "5 click rules" interpretation in the brief includes
**ThiolEne** (radical thiol + alkene → thioether), which is the maleimide-thiol
Michael addition exercised by the THIOL_TILES (positions 12-13) — this is **not**
in `CLICK_REACTIONS` but is added in `close_loop_3_persistent_tree.py` / `round6`
via the RDKit reaction SMARTS `[SH:1].[C:2]=[C:3]>>[S:1][C:2][C:3]`.

## 3. Fresh deterministic 50-molecule batch from STANDARD_12_TILES × 5 click rules

Calling `build_all_click_tiles(embed_3d=False)` and exhaustively firing all
5 rules (CuAAC, SPAAC, SPC, DielsAlder, ThiolEne) on every compatible handle
combination in `STANDARD_12_TILES` (4 azides × 3 terminal alkynes; 4 azides × 1
cyclooctyne; 4 azides × 1 phosphine; 1 diene × 2 alkenes; 2 thiols × 2 alkenes)
yields **29 raw products → 27 unique canonical SMILES** after RDKit sanitization +
dedup. This is **less than 50** and therefore is **not** sufficient on its own to
serve as the parity dataset.

The shortfall is structural: the STANDARD_12_TILES library is intentionally small
(Phase-0 constant pool). To reach 50 molecules for parity one must either:
(a) extend the tile library (e.g. `STANDARD_14_TILES` adds 2 thiol handles but
adds no net products); (b) include `partner_tiles × alkyl handles` more deeply;
(c) **use the existing 100-row `crossdocked100_manifest.csv` first 50 rows** as
the canonical source (this is what the brief actually recommends in the final
paragraph).

**Chosen source** (per brief — "Use existing manifest (crossdocked100_manifest.csv
first 50 rows or as documented)"): `molmetal/data/crossdocked100_manifest.csv` rows
1-50, pocket_id `test_000`..`test_049`, with the bound-ligand SDF SMILES taken
from `ligand_path` (one per pocket). This is the same shape used by
`molmetal/scripts/prepare_crossdocked_batch.py` and the pre-prepared receptor
directory `molmetal/reports/crossdocked_first10_preparation/test_001/`.

## 4. RDKit canonicalisation + stereo / metal audit

Regenerated 27 unique canonical SMILES via `Chem.MolFromSmiles → Chem.SanitizeMol
→ Chem.MolToSmiles` (the same pipeline used by `_canonicalise` in
`click_reactions.py` and by `build_click_tile` in `click_tiles.py`).

Findings:

- **Canonicalisation success**: 27/27 raw products parse, sanitise, and
  re-canonicalise without error. **0 parse failures, 0 sanitisation failures.**
- **Stereo centers**: 5/27 carry chiral centres (DielsAlder + ThiolEne products
  with the cyclopentadiene or maleimide backbone). All stereo is **implicit /
  unassigned** — RDKit's `FindMolChiralCenters(includeUnassigned=True)` returns
  counts but no `@@`/`@` is in the canonical SMILES. **Implication for docking**:
  Vina and QuickVina-2 are both stereo-blind at the SMILES-input level (they
  operate on 3D coordinates after `obabel`/`meeko` `MolToPDBQT` flattening), so
  unassigned stereo will not cause either engine to refuse. **No `engine_refuse`
  risk** from stereo.
- **Metal atoms**: **0/27** contain non-organic atoms (only C, N, O, P, S, H).
  Both engines handle organics natively; **no metal-coordination parse risk.**
  This is consistent with the brief: the parity measurement is on **organic**
  click products, not on metal-coordinated ligands.
- **Valence warning** observed during regeneration: `[09:33:23] Explicit valence
  for atom # 4 O, 3, is greater than permitted` — this is a RDKit info-level
  message triggered by `MMFFOptimizeMolecule` on one strained SMILES (probably
  the `C1CCCCCC1` cyclooctane-fused triazole); it does **not** affect the
  canonical SMILES output and **does not** prevent either engine from docking.

## 5. Path list + chosen source

```
Manifest (chosen source — first 50 rows)
  /home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv
  /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/

Tile library + reactions (regeneration utility — yields 27, not 50)
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tile_lib/click_tiles.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tile_lib/library.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/click_reactions.py
  /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tile_lib/canonical_cache.py

Pilot artifacts (NOT a parity source)
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_click_physical_test10_seed3_v2.json
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_click_physical_test10_seed3_v2.csv
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_click_physical_test10_seed3_v2_analysis.md

Regenerated 27-product side file (deterministic, RDKit-canonicalised)
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/round11_engine_parity/click12_x5rules_unique27.csv
    columns: idx, rule, canonical_smiles, n_heavy_atoms, has_stereo
    rule distribution: CuAAC=12, ThiolEne=6, SPAAC=4, SPC=4, DielsAlder=1
```

**Chosen 50-molecule source**: `crossdocked100_manifest.csv` first 50 rows
(`test_000`..`test_049`). The bound-ligand SMILES is parsed from each row's
`ligand_path` SDF via RDKit. This is the same pipeline that
`prepare_crossdocked_batch.py` and the pre-prepared PDBQT directories
(`crossdocked_first10_preparation/`) already validate. The regenerated 27-product
side file is **kept as a deterministic sanity-check** (and as a small click-chemistry
parity subset) but is **not** the parity dataset.

---

*Recon complete. No code changed. No experiments run.*
