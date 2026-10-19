# Dataset card — MSRS precious-metal potency corpus

**Version**: 1.0 (2026-10-19)
**Producer**: `molmetal/scripts/msrs_data_card.py`
**Measured artifact**: `metrics/by_metric/msrs_data_card.json`
**Raw file**: `data/msrs/measurements.parquet`
**Raw SHA-256**: see `provenance.long_sha256` in the artifact

Every number below is MEASURED from the parquet on 2026-10-19. Re-run the script
rather than editing this card.

---

## 1. What this dataset is

A curated long-format table of **half-maximal inhibitory concentrations (IC50)**
for precious-metal coordination compounds against cancer cell lines, assembled
from the metal-anticancer literature. One row is one measurement: a complex, a
cell line, a paper, a duration, and a potency.

The strong model (MET-1/MET-2) consumes a filtered view of this table. The
dataset card describes that view, not the raw 43,268-row file.

## 2. Composition

| Quantity | Value |
|---|---:|
| Total rows in the raw file | 43,268 |
| Endpoint = `potency` | 26,801 |
| Endpoint = `cisplatin_ratio` | 14,309 |
| Endpoint = `phototoxicity` | 2,158 |
| **Modeling rows** (filter in §4) | **21,404** |
| Metals | 5 |
| Cell lines | 711 |
| Ligands (`ligand_canonical`) | 5,861 |
| DOIs | 1,784 |
| Distinct exposure times | 28 |
| Value units | µM (all rows) |

Metal imbalance is severe:

| Metal | Rows | Share |
|---|---:|---:|
| Ru | 15,051 | 70.3% |
| Ir | 3,725 | 17.4% |
| Rh | 967 | 4.5% |
| Os | 900 | 4.2% |
| Re | 761 | 3.6% |

**Consequence.** Any aggregate metric is dominated by Ru. Within-metal metrics
(MET-3, the transfer line) are the only fair read; a pooled R² is mostly a Ru
model. Metal-pair analyses cover only three pairs (Os–Ru, Ir–Rh, Rh–Ru) because
the corpus rarely measures two metals on the same ligand/cell/paper/time.

## 3. Label definition

```
y = log10(IC50 / µM)        # lower = more potent
```

- IC50 is the half-maximal inhibitory concentration from the source paper.
- Values are already log-transformed in `value_log10`; the model never sees µM.
- A difference of 0.2091 log10 is the project's noise floor (≈1.62× in
  concentration); it is used as the minimum effect worth ranking.

## 4. Inclusion / exclusion

The modeling view is:

```
endpoint == "potency"
AND value_log10.notna()
AND censoring == "none"
AND ligand_main_fragment.notna()
AND ligand_canonical.notna()
AND oxidation_state.notna()
```

| Filter step | Rows |
|---|---:|
| Raw | 43,268 |
| Endpoint = potency | 26,801 |
| Censoring = none | 21,777 |
| Required structure fields present | 21,404 |

**Censoring is the largest exclusion.** 5,015 potency rows (18.7%) are
right-censored (`IC50 > threshold`) and 9 are left-censored. They are dropped,
not modelled. This is a **complete-case analysis**: the training set is biased
toward compounds potent enough to yield an exact IC50. A model trained here
should not be read as estimating potency for inactive compounds; it estimates
potency conditional on an IC50 having been reached in the assay window.

`cisplatin_ratio` and `phototoxicity` are separate endpoints used elsewhere
(the phototoxicity switch analysis). They are not in the regression target.

## 5. Feature blocks

The strong arm (`ligand+both`) is:

| Block | Content | Width |
|---|---|---:|
| Complex | ECFP4 of `ligand_canonical` (2048 bits) | 2,048 |
| Metal state | oxidation state, complex charge, d-electron count, valence electrons | 4 |
| Coordination | donor counts + geometry indicators | 22 |
| Metal label | LightGBM native categorical code | 1 |
| Cell label | LightGBM native categorical code | 1 |

`ligand_canonical` is the canonical SMILES of the **ligand and counterions**;
the metal atom is not in the SMILES (measured: 0 of 21,404 rows contain
`[Ru`/`[Ir`/`[Rh`/`[Os`/`[Re`/`[Pt`). Metal identity therefore reaches the model
only through the state, coordination and categorical blocks. This is what makes
the MET-6 ablation meaningful: the ECFP4 block cannot smuggle the metal in.

## 6. Splits

Four partitions, all produced by `msrs_generalization.scheme_folds`:

| Protocol | Group key | Folds | Groups |
|---|---|---:|---:|
| `ligand` | `ligand_canonical` | 5 | 5,861 |
| `doi` | `doi` (whole papers held out) | 5 | 1,784 |
| `scaffold` | Bemis–Murcko scaffold of `ligand_canonical` | 5 | 2,636 |
| `year` | year parsed from the DOI, leave-one-year-out | 16 | 16 |

Rows with no usable key get fold `-1`: they train in every fold and are never
evaluated. For `year` this is the 69.6% of rows whose DOI has no parseable year.

Leakage quantities for each split are in `docs/DATA_LEAKAGE_AUDIT.md` §8 and in
the artifact's `protocols` block. The headline: the `ligand` protocol leaves
89.4% of test rows in a paper that is also in train; that is why MET-2 measured
R² 0.5556 there and 0.3321 under `doi`.

## 7. Duplicates and ties

**Duplicates.** 304 keys on `(ligand, metal, cell, DOI, time)` occur more than
once (663 rows, 3.1%); 176 keys carry conflicting values (393 rows, 1.8%); 142
rows are exact duplicates. **No de-duplication is applied** — each row is a
measurement and replicates carry assay variance.

**No duplicate key crosses a fold under any of the four protocols** (measured:
0 crossing keys each). The key is nested inside every group key, so this is a
property of the partition, not a coincidence. 188 rows (0.9%) have no `time_h`
and cannot be keyed; they are still assigned to folds by ligand/DOI.

**Ties.** Decision-layer rankings use `sort_values(kind="stable")`, so exact
score ties break by dataset order. Pairwise Δ uses a fixed lexicographic metal
order (Ir < Os < Re < Rh < Ru) so a pair is counted once, never as both
`(a,b)` and `(b,a)`. Tie sensitivity for the classification line is measured
separately in `metal_activity_transfer_tie_sensitivity.json`.

## 8. Intended use

- **In scope**: ranking metals within a measured ligand/cell/paper cohort;
  estimating the effect of a metal swap (Δ) when both metals were measured;
  quantifying representation gains from metal state and coordination.
- **Out of scope**: prospective potency prediction for new chemotypes;
  extrapolation to metals absent from the corpus (only 5 metals, and Pt is
  blocked); any claim about compounds that never reached an IC50; clinical or
  wet-lab decisions (the project is computational-only by standing directive).

## 9. Limitations

1. **Metal imbalance** — 70% Ru; pooled metrics describe Ru.
2. **Complete-case selection** — 18.7% of potency rows censored and dropped.
3. **Few metal pairs** — 427 pairs, only three metal combinations.
4. **Paper-level structure** — 89.4% of test rows under ligand grouping share a
   DOI with train; only the `doi` protocol removes it.
5. **No independent external set** — MET-7 is still BLOCKED-PENDING-DATA; every
   split here is a re-partition of the same corpus.
6. **Cell-line recurrence** — 96–98% of test rows share a cell line with train
   in every protocol. Within-cohort decisions cancel this, pooled R² does not.
7. **Ligand reuse across papers** — 15.4% of DOI-holdout test rows have a ligand
   that appears in another paper's training rows. Not a leak, but the DOI
   protocol is not ligand-disjoint.

## 10. Provenance and maintenance

- Raw file: `data/msrs/measurements.parquet` (SHA-256 recorded in the artifact).
- Producer: `molmetal/scripts/msrs_data_card.py` (read-only; no mutation).
- Re-run after any change to the parquet or the filter:
  `OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_data_card.py`
- Downstream consumers: `msrs_seed_stability.py` (MET-1),
  `msrs_generalization.py` (MET-2), `msrs_delta_metrics.py` (MET-3),
  `msrs_calibration.py` (MET-4), `msrs_metal_ablation.py` (MET-6).
- License / redistribution: derived from published literature; cite the source
  DOIs. Not for redistribution without the original article terms.
