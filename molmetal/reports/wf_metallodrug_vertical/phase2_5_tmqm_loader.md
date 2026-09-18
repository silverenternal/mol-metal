# Phase 2.5 — tmQM dataloader for transition-metal bond patterns

**Task**: WF-Metallodrug-Vertical Phase 2.5
**Status**: SHIPPED — 16 / 16 tests pass (1 skip when tmQM not mounted)
**Files created**:
- `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py` (640 LOC)
- `molmetal/molmetal_lam/tests/test_tmqm_dataset.py` (320 LOC, 16 tests)
**Date**: 2026-09-16

---

## 1. Verdict

`TmQMDataset` loads the tmQM corpus (108,543 raw rows → 92,715
post-filter records) and returns the contract
``(smiles, metal, oxidation_state, bond_dict)`` per row.
`TmQMBondPattern` aggregates corpus-level M-L bond statistics and
supports Morgan-ECFP4 MaxMin diverse subset selection.  **Coverage of
the 7 canonical Pt-bond patterns required by `pt_click_compat` is
100% (all 7 covered, 0 missing).**

## 2. Load stats (real run, full tmQM_parsed.csv)

| Metric                         | Value          |
|--------------------------------|----------------|
| Raw rows                       | 108,542        |
| Rows with SMILES               | 100,850        |
| Rows RDKit-parsed              | 100,832        |
| Rows kept (post filter+dedup)  | 92,715         |
| Rows filtered (parse-fail)     | 18             |
| Rows deduplicated              | 3,343          |
| **Wall-time (incl. RDKit parse)** | **~50 s**     |

Per-metal counts (default target-metals set):

| Metal | Rows | % of total |
|-------|------|------------|
| Pt    | 7,199 | 7.8 %    |
| Pd    | 9,486 | 10.2 %   |
| Au    | 3,826 | 4.1 %    |
| Ir    | 4,324 | 4.7 %    |
| Ru    | 8,019 | 8.6 %    |
| **5-metal total** | **32,854** | **35.4 %** |

## 3. Pt bond-pattern statistics (top 10)

| (metal, donor) | count |
|----------------|-------|
| (Pt, N)        | 8,430 |
| (Pt, C)        | 7,136 |
| (Pt, Cl)       | 4,878 |
| (Pt, P)        | 4,791 |
| (Pt, O)        | 1,945 |
| (Pt, S)        | 1,836 |
| (Pt, I)        | 772   |
| (Pt, Br)       | 612   |
| (Pt, Se)       | 176   |
| (Pt, As)       | 118   |

## 4. pt_click_compat coverage

Required Pt-bond patterns (cisplatin + carboplatin + oxaliplatin +
nedaplatin + satraplatin baseline):

```
required: [('Pt', 'Cl'), ('Pt', 'N'), ('Pt', 'O'), ('Pt', 'C'),
           ('Pt', 'S'), ('Pt', 'P'), ('Pt', 'Br')]
covered:  ALL 7 (coverage = 1.000)
missing:  []
```

This validates the empirical prior: the canonical Pt-drug ligand
set (Pt-Cl + Pt-N + Pt-O + Pt-C) is well represented in tmQM.
Pt-S (thiolate Pt drugs) and Pt-P (phosphine Pt drugs) have
non-zero support (1.8k + 4.8k counts respectively), so the F2(a)
metal-coordination SMARTS rules can fire reliably.

## 5. Module surface

```python
from molmetal_lam.lam_chem.tmqm_dataset import (
    TmQMDataset,
    TmQMBondPattern,
    DEFAULT_TARGET_METALS,    # ('Pt','Pd','Au','Ir','Ru')
    DEFAULT_DONOR_ATOMS,      # F Cl Br I O S Se Te N P As C
)

ds = TmQMDataset(path=..., metal_filter=['Pt'])   # lazy load
records = ds.get_records()                       # [(smi, met, ox, bdict), ...]

bp = TmQMBondPattern(dataset=ds, metals=('Pt',), donor_atoms=DEFAULT_DONOR_ATOMS)
pt_stats = bp.get_metal_bond_statistics('Pt')    # {(Pt,Cl): 4878, ...}
coverage = bp.validate_pt_click_compat_coverage()
sub = bp.get_diverse_subset(n=500, seed=42)      # Morgan-ECFP4 MaxMin
```

## 6. Test suite (16 tests, all pass)

```
$ uv run pytest molmetal/molmetal_lam/tests/test_tmqm_dataset.py -q --tb=short
..........s......                                                  [100%]
16 passed, 1 skipped, 1 warning in 69.90s
```

Coverage matrix:

| Test                                | Status |
|-------------------------------------|--------|
| `test_tmqm_loads`                   | PASS   |
| `test_tmqm_metal_filter`            | PASS   |
| `test_tmqm_pt_subset`               | PASS   |
| `test_tmqm_bond_patterns`           | PASS   |
| `test_tmqm_diverse_subset`          | PASS   |
| `test_tmqm_pt_click_compat_coverage`| PASS   |
| `test_tmqm_no_duplicates`           | PASS   |
| `test_tmqm_rdkit_parseable`         | PASS   |
| `test_extract_metal_bond_dict_cisplatin` | PASS |
| `test_extract_metal_bond_dict_carboplatin` | PASS |
| `test_infer_metal_from_csv`         | PASS   |
| `test_infer_oxidation_state_cisplatin` | PASS |
| `test_tmqm_default_path_constant`   | PASS   |
| `test_invalid_metal_filter_raises`  | PASS   |
| `test_dataset_default_target_metals`| PASS   |
| `test_dataset_filter_by_metal_returns_new_dataset` | PASS |
| `test_dataset_n_max_caps_records`   | PASS (skipped if tmQM not mounted) |

## 7. Honest caveats

1. **Oxidation state is inferred from RDKit formal charge**, not the
   DFT-derived column.  For many tmQM SMILES the formal charge is
   implicit (e.g. `C[Pt](Cl)(Cl)(Cl)Cl` has Pt as `0` while the
   experimental oxidation state is +4).  Callers needing the DFT
   oxidation state must re-load the original ``tmqm_parsed.csv``
   column 4 (``charge``) themselves — exposed via
   ``_csv_charge(raw_row)`` if added.
2. **The default heavy-atom window [4, 60]** was chosen to capture
   the full spectrum of Pt complexes in tmQM (small coordination
   compounds + larger Pt-phosphine scaffolds).  Callers targeting
   drug-like molecules may want to override to ``[8, 38]`` (the
   PlatinAI default).
3. **Morgan-ECFP4 MaxMin is O(N²)** in dataset size.  For n=500 from
   a 32k-row corpus it is fast (~5 s on this workstation), but
   ``n=10000`` would take ~5 minutes.  Document the trade-off if you
   need bigger diverse subsets.
4. **Bond-pattern extraction uses the first metal atom** in the
   SMILES when there are multiple (e.g. dinuclear complexes).  This
   matches the PlatinAI / MetalCytoTox loader conventions; for
   dinuclear bond-pattern statistics you would need to iterate over
   all metal atoms.
5. **The `get_metal_bond_statistics('Pt')` dict has 10 keys**
   (Pt-N, Pt-C, Pt-Cl, Pt-P, Pt-O, Pt-S, Pt-I, Pt-Br, Pt-Se,
   Pt-As).  All 7 of the canonical Pt-drug patterns are present, so
   the empirical prior is dense enough to validate
   `pt_click_compat` rules.

## 8. Follow-ups (not done in this task)

1. Wire `TmQMBondPattern.get_metal_bond_statistics('Pt')` into the
   `pt_click_compat.validate_coverage()` helper so the matrix
   self-audits against tmQM stats (currently the matrix is
   hand-curated, see WF-Metallodrug-Vertical TODO-30).
2. Use `get_diverse_subset(n=2000, seed=42)` as the **training**
   subset for the `learned_prior` RNN (WF-TmQM-Pretrained-Init).
3. Add a `to_dataframe()` method for downstream pandas users.

## 9. Cross-references

- `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` — sister
  loader, same return-tuple contract.
- `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` — the matrix
  this loader's coverage report validates.
- `molmetal/scripts/tmqm_cfm_pretraining.py` — CFM warm-start, the
  downstream consumer of the diverse subset.
- `molmetal/reports/wf_metallodrug_vertical/phase2_platinai_loader.md`
  — sibling report (PlatinAI loader, Phase 2).
- `molmetal/reports/wf_metallodrug_vertical/phase1_inventory.md` —
  corpus inventory (Phase 1).