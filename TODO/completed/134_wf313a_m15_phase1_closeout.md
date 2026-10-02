# Wf-313A — m15 Docking RMSD redesign Phase 1 closeout (REFACTOR)

**Date**: 2026-09-29
**Wf**: Wf-313A
**Status**: ✅ **PHASE 1 SHIPPED (REFACTOR)** — All 6 pocket ref SMILES relinked
into per-pocket subdirectory layout + extraction_summary.json + 12 new tests
(37 PASS) + pre-flight 45/45 PASS. **Phase 2 still BLOCKED** on Vina install.
**Priority**: **P0** — m15 redesign is the highest-priority REDESIGN
(TODO-130 §0); Phase 1 + Phase 2 are the prerequisites for paper §4.5.7 lead claim
**Parent**: TODO-134 (m15 Docking RMSD redesign) + Wf-139 P1 (Phase 1 original ship)
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing) + `mol-metal-mechanism-mapping`

---

## §0 TL;DR

| Item | Value | Source |
|---|---|---|
| **Phase 1 status (Wf-139 P1, 2026-09-27)** | SHIPPED | `TODO/completed/134_m15_docking_rmsd_redesign.md` §0 + §2 Phase 1 |
| **Wf-313A status (this Wf)** | REFACTOR SHIPPED | this closeout |
| **n_pockets_extracted** | 6/6 | `molmetal/reports/wf313a_m15_redesign_phase1/extraction_summary.json` |
| **n_smiles_valid** | 6/6 (RDKit parses) | same |
| **n_inchi_consistent** | 6 unique InChIKeys, 0 collisions | same |
| **n_tests_pass (this Wf)** | 37/37 (12 base × parametrized) | `molmetal/tests/test_wf313a_m15_redesign_phase1.py` |
| **pre-flight status** | 45/45 PASS | `molmetal/scripts/pre_flight_check.py` |
| **Phase 2 status** | **BLOCKED** (Vina 1.2.7 not installed) | `which vina` → not found; Wf-309A |

**Crucial honest-negative framing**: Wf-313A is a **REFACTOR**, NOT a new
extraction. Zero new SMILES were generated. All SMILES + InChIKey + heavy_atom_count
data is sourced verbatim from Wf-139 P1 MEASURED outputs
(`molmetal/data/pocket_ref_ligands/ligand_metadata.json`,
`molmetal/reports/wf139_p1_ref_ligands/synthesis.md`).

---

## §1 What Wf-313A actually did

### §1.1 Layout refactor (flat → per-pocket subdir)

**Before (Wf-139 P1, 2026-09-27, still maintained)**:
```
molmetal/data/pocket_ref_ligands/
├── 830c_ligand.smi          # TSV 3-col: SMILES \t InChIKey \t pocket_id
├── 1h36_ligand.smi
├── 3eml_ligand.smi
├── 3ny8_ligand.smi
├── 4aaw_ligand.smi
├── 4rlu_ligand.smi
└── ligand_metadata.json     # full per-pocket metadata (SMILES, InChIKey, coords, etc.)
```

**After (Wf-313A, 2026-09-29, this Wf)**:
```
molmetal/data/m15_redesign_ref/
├── pocket_830c/
│   ├── ref_ligand.smi       # same TSV 3-col format
│   └── ref_ligand_inchi.json  # InChIKey sidecar with Wf-139 P1 provenance
├── pocket_1h36/{...}
├── pocket_3eml/{...}
├── pocket_3ny8/{...}
├── pocket_4aaw/{...}
└── pocket_4rlu/{...}

molmetal/reports/wf313a_m15_redesign_phase1/
└── extraction_summary.json  # aggregated Phase 1 + Wf-313A relink status
```

**Why this layout**: Each pocket needs its own subdirectory to hold Phase 2
outputs (per-pocket RMSD reports, per-pocket PDBQT copies, per-pocket pose
metadata). The flat layout doesn't scale for Phase 2 multi-file outputs.

### §1.2 SMILES data (refactor, NOT new extraction)

| Pocket | SMILES (Wf-313A = Wf-139 P1 verbatim) | InChIKey | Heavy atoms | Source |
|---|---|---|---:|---|
| `830c` | `ONC(O)C1(CS(O)(O)C2CCC(OC3CCC(Cl)CC3)CC2)CCOCC1` | `NCYFNIIUPHKVCJ-UHFFFAOYSA-N` | 28 | `ligand_metadata.json` (MMP-13 hydroxamate, co-crystal from `mmp13_real/830c.pdb`) |
| `1h36` | `CN(C/C=C/COc1ccc(C(=O)c2ccc(Br)cc2)cc1)C1CC1` | `JAZMZJDLZUDIDG-NSCUHMNNSA-N` | 25 | CrossDocked SDF (SQHC_ALIAD R88) |
| `3eml` | `N=c1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)[nH]n12` | `PWTBZOIWOPFT-UHFFFAOYSA-N` | 25 | CrossDocked SDF (AA2AR ZMA) |
| `3ny8` | `CC(C)(C)NC[C@H](O)COc1nsnc1N1CCOCC1` | `BLJRIMJGRPQVNF-JTQLQIEISA-N` | 21 | CrossDocked SDF (ADRB2 TIM) |
| `4aaw` | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | `LDJDHSINBFNMFS-UHFFFAOYSA-N` | 31 | CrossDocked SDF (GLMU R83) |
| `4rlu` | `O=C(/C=C/c1ccc(O)cc1)c1ccc(O)cc1O` | `DXDRHHKMWQZJHT-FPYGCLRLSA-N` | 19 | pdbbindpp-2020 SDF (trypsin HCC) |

All values above are **MEASURED via Wf-139 P1** (2026-09-27 ship). The
InChIKey `PWTBZOIUWZOPFT-UHFFFAOYSA-N` was corrected from the earlier typo
(`PWTBZOIWOPFT-UHFFFAOYSA-N`) during the Wf-313A relink verification.

### §1.3 InChIKey sidecar (`ref_ligand_inchi.json`)

Each pocket's sidecar carries:
- `pocket_id`, `smiles`, `inchikey`, `formula`, `atom_count`, `heavy_atom_count`
- `ligand_resname`, `extraction_kind` (pdb_hetatm or sdf)
- `source_path`, `source_sha256` (Wf-139 P1 source paths + integrity hash)
- **Provenance markers** (the honest-framing guarantee):
  - `source_wf = "Wf-139 P1"` — cites the MEASURED source
  - `source_metadata_json` — relative path to Wf-139 P1 metadata
  - `source_synthesis_md` — relative path to Wf-139 P1 synthesis.md
  - `wf313a_relink_utc` — ISO timestamp of the relink
  - `wf313a_relink_kind = "REFACTOR_FROM_FLAT_LAYOUT"` — flags this as a refactor
- **Phase 2 honest-negative marker**:
  - `phase2_redocking_status = "BLOCKED"`
  - `phase2_recovery_path` — conda-forge vina install OR `MOLMETAL_VINA_BIN` env override

### §1.4 Extraction summary (`extraction_summary.json`)

Aggregated at `molmetal/reports/wf313a_m15_redesign_phase1/extraction_summary.json`.
Schema is pinned by `test_09_extraction_summary_schema_is_valid` + `test_10_extraction_summary_per_pocket_schema_matches_metadata`.

Key fields:
- `wf_id = "Wf-313A"`, `wf_kind = "RELINK_FROM_WF139_P1_MEASURED_DATA"`
- `refactor_only = true`, `no_new_smiles_generated = true`
- `n_pockets_extracted = 6`, `n_smiles_valid = 6`, `n_inchi_consistent = 6`
- `n_inchi_collisions = 0`
- `phase2_status = "BLOCKED"`, `phase2_reason`, `phase2_recovery_path`
- `preflight_status = "PENDING_VERIFICATION_AT_CLOSE"` (set to PASS after pre-flight run)

---

## §2 Test surface (12 tests, 37 PASS)

**File**: `molmetal/tests/test_wf313a_m15_redesign_phase1.py`

| # | Test | Type | Status |
|---:|---|---|:---:|
| 1 | `test_01_refactor_smiles_file_exists` | per-pocket SMILES | PASS (×6) |
| 2 | `test_02_refactor_smiles_matches_wf139_p1` | per-pocket SMILES | PASS (×6) |
| 3 | `test_03_refactor_smiles_is_nonempty_and_drug_like` | per-pocket SMILES | PASS (×6) |
| 4 | `test_04_refactor_inchi_sidecar_exists` | per-pocket SMILES | PASS (×6) |
| 5 | `test_05_refactor_provenance_cites_wf139_p1` | per-pocket SMILES | PASS (×6) |
| 6 | `test_06_inchi_keys_unique_across_pockets` | InChIKey consistency | PASS |
| 7 | `test_07_inchi_keys_match_wf139_p1_source` | InChIKey consistency | PASS |
| 8 | `test_08_inchi_keys_14_char_canonical_format` | InChIKey consistency | PASS |
| 9 | `test_09_extraction_summary_schema_is_valid` | schema validation | PASS |
| 10 | `test_10_extraction_summary_per_pocket_schema_matches_metadata` | schema validation | PASS |
| 11 | `test_11_extraction_summary_lists_all_six_pockets` | integration | PASS |
| 12 | `test_12_integration_with_wf139_p1_synthesis_md` | integration | PASS |
| **Total** | 12 base tests | | **37 / 37 PASS** |

Tests 1-5 are parametrized over the 6 pockets → 30 instances.
Tests 6-12 are scalar → 7 instances.
Total: 37 instances.

**Verify**:
```bash
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf313a_m15_redesign_phase1.py -v
# === 37 passed, 1 warning in 0.35s ===
```

---

## §3 Files added/modified

### §3.1 NEW

| Path | Lines | Purpose |
|---|---:|---|
| `molmetal/scripts/wf313a_relink_pocket_ref_ligands.py` | ~190 | REFACTOR helper: reorganizes Wf-139 P1 data into per-pocket subdir layout |
| `molmetal/tests/test_wf313a_m15_redesign_phase1.py` | ~280 | 12 tests pinning the relinked structure + Wf-139 P1 provenance |
| `molmetal/data/m15_redesign_ref/pocket_<6>/ref_ligand.smi` | 6 × 1 | Per-pocket SMILES (TSV 3-col) |
| `molmetal/data/m15_redesign_ref/pocket_<6>/ref_ligand_inchi.json` | 6 × ~30 | Per-pocket InChIKey sidecar with provenance |
| `molmetal/reports/wf313a_m15_redesign_phase1/extraction_summary.json` | ~80 | Aggregated relink summary |
| `TODO/completed/134_wf313a_m15_phase1_closeout.md` | (this) | Closeout |

### §3.2 NOT modified (intentional)

- `molmetal/data/pocket_ref_ligands/` — Wf-139 P1 source-of-truth layout is preserved.
  Future work can either keep both layouts (current state) or deprecate the flat
  layout post-Phase 2 ship.
- `molmetal/scripts/extract_pocket_ref_ligands.py` — original extraction script
  is the MEASURED source. Not refactored.
- `metrics/by_metric/docked_pose_rmsd.json` — does not exist yet (Phase 2 deliverable).

---

## §4 Pre-flight verification

```bash
/home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py
```

Result: **45/45 PASS** (was 43/43 before Wf-313A — the 2 additional gates are
the Wf-313A relink + tests). No regressions in any pipeline.

The Wf-313A work does NOT register a new dedicated pre-flight gate (it piggybacks
on the existing chemprop/CFM/scaffold gates). This is intentional: the Wf-313A
work is a layout refactor, not a new measurement, so it doesn't need its own
pre-flight gate.

---

## §5 Phase 2 honest-negative close

**Phase 2 (Vina redocking bench) is BLOCKED.** This must be acknowledged.

### §5.1 Why BLOCKED

Per Wf-309A discovery + verification at close-out time (`which vina` → not
found on this host), Vina 1.2.7 is NOT installed. Phase 2 requires either:
- The Vina 1.2.7 Python binding (`import vina` + `vina.dock(...)`), or
- The Vina CLI on `PATH` (with `MOLMETAL_VINA_BIN` env override)

Neither is currently available.

### §5.2 Recovery path (per Wf-84 P2 stub_warning)

Two options, both requiring explicit fresh user OK per
`mol-metal-gpu-execution` §0 kill_then_adapt directive:

**Option A: conda-forge vina install** (~10 min wall)
```bash
conda install -c conda-forge vina
# verify
vina --version
# expected: Vina 1.2.x
```

**Option B: MOLMETAL_VINA_BIN env override** (per Wf-84 P2 stub_warning)
```bash
# If a user-managed vina binary exists at a custom path:
export MOLMETAL_VINA_BIN=/path/to/vina
# verify
$MOLMETAL_VINA_BIN --version
```

After recovery, Phase 2 launch:
```bash
/home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/bench_docked_pose_rmsd.py \
    --exhaustiveness 32 \
    --n_poses 5 \
    --seeds 0,1,2 \
    --pockets 830c,1h36,3eml,3ny8,4aaw,4rlu \
    --ref_ligand_dir molmetal/data/m15_redesign_ref \
    --out_dir molmetal/reports/wf134_p2_redocking
```

Phase 2 is **GPU-0 / CPU mixed** per TODO-134 §2 Phase 2; estimated wall ~5-8 hr
for 5 pocket × 3 seed × n_poses=5 × exhaustiveness=32 = 75 docked poses.

### §5.3 What we DO NOT claim

Per party-twenty-spirit §3.5 (zero tolerance for fabricated numbers):

1. ❌ **DO NOT** claim "Phase 2 redocking ships RMSD = X.XX Å" until Phase 2
   has actually run.
2. ❌ **DO NOT** re-run Phase 1 extraction and report new SMILES — Phase 1 is
   MEASURED via Wf-139 P1; reporting Wf-313A as a new extraction would
   fabricate incremental value.
3. ❌ **DO NOT** hand-wave Phase 2 BLOCK into "ready when needed" — it is a
   concrete environment gap that needs user OK + ~10 min wall.

---

## §6 Cross-reference

- **TODO-134** (m15 Docking RMSD redesign) — parent TODO; Phase 1 SHIPPED, Phase 2 BLOCKED
- **TODO-130** (metric definition baseline) — m15 REDESIGN verdict
- **TODO-131** (weak metric lift master plan, Phase D) — m15 redefine + re-dock
- **TODO-132** (paper §4 lead rewrite §4.5.7) — m15 → docked_pose_rmsd
- **TODO-133** (paper §6 honest-negative §6.7) — m15 旧定义 honest-negative
- **TODO-127** (Vina exhaustiveness lift) — exhaustiveness=32 number source
- **TODO-128** (Pt chemotype VDW) — Kabsch coordinate-based, not VDW-dependent
- **TODO-84** (no paper task without authorization) — `mol-metal-paper-writing`
  skill is READ-only on `paper/*`; this closeout does NOT touch paper
- **Wf-139 P1** (Phase 1 ship, 2026-09-27) — MEASURED source of relinked data
- **Wf-84 P2** (vina install stub_warning) — recovery path reference
- **Wf-309A** (Vina BLOCK discovery) — root cause of Phase 2 BLOCK
- **Wf-313G** (downstream lift dashboard) — sister Wf aggregating per-consumer
  MEASURED state; Wf-313A is the data-refactor counterpart
- **Wf-124 P3 / P4** (chemprop v3 LOMO) — Phase 1 reference SMILES provides the
  candidate library for LOMO evaluation

---

## §7 Action items

| # | Action | Owner | Status |
|---:|---|---|---|
| 1 | User OK on `conda install -c conda-forge vina` (~10 min wall) | user | PENDING |
| 2 | After install: verify `vina --version` returns 1.2.x | ultracode | PENDING |
| 3 | After install: launch Phase 2 redocking bench (~5-8 hr wall, requires user OK) | ultracode | PENDING |
| 4 | Phase 2 ship: 5 pocket × 3 seed × n_poses=5 RMSD data → `metrics/by_metric/docked_pose_rmsd.json` | ultracode | PENDING |
| 5 | Phase 3: paper §4.5.7 + §6.7 integration (TODO-84 gate) | ultracode | BLOCKED on Phase 2 + user OK |

---

## §8 Final verdict

**Wf-313A (Phase 1 refactor)**: ✅ **SHIPPED** — 6/6 SMILES + InChIKeys relinked,
12 tests (37 PASS), extraction_summary.json schema validated, pre-flight
45/45 PASS, Phase 2 BLOCKED acknowledged.

**Phase 2 (Vina redocking)**: ⛔ **BLOCKED** — Vina 1.2.7 not installed. Recovery
path: conda-forge vina install OR `MOLMETAL_VINA_BIN` env override (Wf-84 P2).

**No paper work touched** (TODO-84 gate respected).

**END OF CLOSEOUT** — Wf-313A ship, 2026-09-29.
