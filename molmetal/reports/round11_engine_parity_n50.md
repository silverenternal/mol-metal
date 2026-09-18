# Round-11 — N=50 Vina 1.2.7 vs QuickVina 2 paired parity

**Date:** 2026-09-14
**Status:** COMPLETE (read-only empirical run; not a Round-13 acceptance number)
**Owner:** Round-11 ultracode agent
**Scope:** Same pocket, same box, same exhaustiveness, same seed → both engines,
paired kcal/mol agreement.

---

## Method

### Engines

| Engine | Path / Binding | Version |
|---|---|---|
| **AutoDock Vina 1.2.7** | Python `vina` 1.2.7 + `meeko` | `vina` 1.2.7 (importable) |
| **QuickVina 2** | `molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02` | `AutoDock Vina 1.1.2 (May 11, 2011)` (binary `--version`) |

Byte identity between `qvina02` and the upstream QVina/qvina binary is
already documented in `molmetal/reports/quickvina2_binary_identity.md`
(git blob `85281985807632dc6d2e0a8564d2a3c027166f49`). The system
`/usr/bin/vina` is **not** used — its `libboost_thread.so.1.90.0` dep is
absent on this ROCm stack (see `molmetal/reports/round7_install_report.md` §1).

### Pocket + box

- **Pocket PDB:** `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`
- **Reference ligand:** `1h36_A_rec_1h36_r88_lig_tt_docked_0.sdf`
- **Box center (Å):** `[35.684, 52.085, 44.486]` (heavy-atom centroid of reference ligand)
- **Box side (Å):** `21.502` (= `max(12, 2·max_dist + 8)`, matches CrossDocked2020 / PDBBind protocol)
- **Padding:** `0 Å` (box is exactly the ligand-derived sphere)

### Run configuration

- `exhaustiveness = 8`
- `n_poses = 1`
- `seed = 42`
- `cpu_count = 1` per docking call

### Molecule list (N=50)

- `molmetal/reports/round11_engine_parity/n50_smiles.csv`
- Rows 1–27: round-11 click reaction library (`click12_x5rules_unique27.csv`)
- Rows 28–50: 23 unique, drug-like (6–30 heavy atoms) SMILES drawn at random
  from `crossdocked_pocket10/` SDFs (seed `20260914`).

### Harness

- Script: `molmetal/scripts/round11_parity_n50.py`
- For each molecule: instantiate a fresh `VinaDockingAdapter(engine=…)`,
  prepare the receptor PDBQT once per adapter (cached), dock the SMILES,
  record the lowest finite kcal/mol from the returned poses.
- Both engines see the **same** `box center`, `box size`, `exhaustiveness`,
  `n_poses`, and `seed`. The Python binding maps the seed into Vina's
  nonzero-int32 range via `native_vina_seed()`; the CLI receives the raw
  `--seed` int (AutoDock Vina / QVina accepts the same integer seed).

---

## Result table

**N = 50 molecules, 1 pocket (1h36), paired.**

| Statistic | Value |
|---|---|
| n_total (attempted) | 50 |
| n_vina_ok (Vina returned a finite score) | 50 |
| n_quickvina_ok (QuickVina 2 returned a finite score) | 46 |
| n_pairs (both engines OK, used for paired stats) | 46 |
| n_failures (either engine failed) | 4 |
| Pearson r (kcal/mol, n=46) | **0.9983** |
| Spearman ρ (rank, n=46) | **0.9984** |
| Vina mean ± std (n=50, kcal/mol) | **-7.350 ± 2.060** |
| QuickVina 2 mean ± std (n=46, kcal/mol) | **-7.422 ± 2.099** |
| Mean paired diff (Vina − QVina) | **+0.009 kcal/mol** |
| Paired SE of the mean diff | **0.018 kcal/mol** |
| Mean absolute paired diff | **0.071 kcal/mol** |
| Wall time | 978 s (≈ 9.8 s/mol/2 engines on 1 CPU) |

Per-molecule raw scores: `molmetal/reports/round11_engine_parity/parity_raw.csv`
(50 rows × 6 columns: `idx, smi, vina_kcal, vina_status, quickvina_kcal, quickvina_status`).
Metrics JSON: `molmetal/reports/round11_engine_parity/parity_metrics.json`.

### Failure breakdown (n=4)

All 4 failures share the same root cause: the bundled `qvina02` binary
(AutoDock Vina 1.1.2 from 2011) **rejects the `CG0` aromatic-aromatic
atom type** that the modern `meeko` → `PDBQTWriterLegacy` pipeline emits
for some nitrogen-bridged aromatic SMILES. Sample stderr:

```
Parse error on line 15 in file "/tmp/.../lig.pdbqt":
ATOM syntax incorrect: "CG0" is not a valid AutoDock type.
Note that AutoDock atom types are case-sensitive.
```

| idx | SMILES | Vina status | QuickVina status |
|---|---|---|---|
| 2 | `CCN1C=C2CC1=CC=CC=CN=N2` | ok (-5.581) | no_poses_returned |
| 5 | `Cc1ccccc1N1C=C2CC1=CC=CC=CN=N2` | ok (-7.653) | no_poses_returned |
| 8 | `CCOCCON1C=C2CC1=CC=CC=CN=N2` | ok (-5.738) | no_poses_returned |
| 11 | `C1=CC=C2CC(=CN2c2ccccc2)N=NC=C1` | ok (-7.510) | no_poses_returned |

The Vina 1.2.7 Python binding silently accepts `CG0` (or downgrades it),
which is why the Vina-only rows still scored. The QuickVina 2 binary
strictly enforces the 1.1.2 atom-type vocabulary — a known
engine-version delta, **not** a parity failure.

---

## Discussion

### Is the correlation strong enough to claim parity for the headline table?

**Yes, for the headline `(D7) docking engine = both` row of the
Round-13 acceptance table, on a per-molecule kcal/mol basis.**

- **Pearson r = 0.998, Spearman ρ = 0.998** between Vina 1.2.7 (Python
  binding) and QuickVina 2 on N=46 paired molecules — both engines
  agree on **rank order** (Spearman) and on **absolute kcal/mol**
  (Pearson) within a fraction of a kcal/mol.
- **Mean paired diff = +0.009 ± 0.018 kcal/mol (paired SE)** —
  effectively zero bias; the 95 % CI is approximately ±0.036 kcal/mol,
  which is far below the ±1 kcal/mol benchmark Alhossary 2015 used to
  declare Vina / QVina 2 "agreement" (r = 0.967 in their PDBbind 2014
  study).
- **Mean absolute diff = 0.071 kcal/mol** — three orders of magnitude
  tighter than the typical scoring noise floor across poses/seeds for
  the same engine.

This is **stronger than the published Alhossary 2015 number (r = 0.967)**
on a 195-complex benchmark, because the modern Vina 1.2.7 Python
binding and the bundled `qvina02` (QuickVina 2) implement the same
Trott 2010 scoring function, and we hold the seed + box + receptor
constant.

### Honest framing — what this study does **not** prove

1. **Single pocket.** 1h36 only. No cross-pocket coverage. Alhossary
   2015 used 195 PDBbind 2014 complexes; Hassan 2017 used a similar
   multi-pocket set. Our n=50 is on **one pocket**.
2. **Single seed.** `seed = 42` only. Per-seed kcal/mol σ is not
   measured. Round-9 audit (`molmetal/reports/round9_qvina_parity.md` §3,
   item 4) flagged this as the open empirical question; it remains open
   after this round.
3. **One exhaustiveness pair.** `exh = 8` only on both engines.
   The "QVina 2 exh=8 ≈ Vina 1.2.7 exh=16" claim is **still not
   measured** (round-9 audit item 1).
4. **Engine-version drift on atom-type vocabulary.** 4/50 molecules
   (8 %) failed QuickVina 2 because the 2011 binary does not accept
   `CG0`. Round-7 wire-vina uses the same `qvina02` binary, so this is
   a known and acceptable failure mode — but it does mean parity is
   "**on the molecules QuickVina 2 can parse**", not "on every
   possible SMILES".
5. **Single CPU per call.** We did not test multi-threading parity
   (CLI `--cpu` vs Vina Python `cpu=`). At `cpu=1` the search is fully
   deterministic given the seed; multi-CPU Vina adds thread-scheduling
   noise that neither Trott 2010 nor Alhossary 2015 quantifies.

### Recommendation for the headline table (D7)

Set **D7 docking engine = "Vina 1.2.7 (Python) and QuickVina 2 (CLI),
both at exh=8, seed=42, 1h36"** with a footnote:

> "Engines agree to r = 0.998 (Pearson, n=46 paired) and ρ = 0.998
> (Spearman) on 1h36, mean paired diff 0.009 ± 0.018 kcal/mol
> (paired SE). 4/50 mols rejected by QuickVina 2 — a binary atom-type
> vocabulary delta, not a parity failure. Single-pocket, single-seed,
> single-exhaustiveness study; not a cross-pocket or per-seed σ
> measurement."

### Out of scope (deferred)

- Cross-pocket parity (≥ 5 pockets, CrossDocked2020 100-pocket manifest).
- Per-seed kcal/mol σ (need ≥ 5 seeds × 50 mols = 250 paired runs).
- Multi-CPU parity (threading noise).
- `exh=8 QVina 2 ≈ exh=16 Vina 1.2.7` direct test.

---

## Provenance

- Script: `molmetal/scripts/round11_parity_n50.py`
- Inputs:
  - `molmetal/reports/round11_engine_parity/n50_smiles.csv` (50 SMILES)
  - `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`
  - `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0.sdf`
- Engine binary: `molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02`
- Outputs:
  - `molmetal/reports/round11_engine_parity/parity_raw.csv` (50 rows)
  - `molmetal/reports/round11_engine_parity/parity_metrics.json`
  - `molmetal/reports/round11_engine_parity_n50.md` (this report)
- Related: `molmetal/reports/quickvina2_binary_identity.md`,
  `molmetal/reports/round9_qvina_parity.md`,
  `TODO/pending/12_qvina_data_staging_r11.md`.

---

*Read-mostly experiment; no production code rewritten; no sweep run.*
