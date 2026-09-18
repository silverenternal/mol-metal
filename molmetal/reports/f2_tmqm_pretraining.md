# F2 — tmQM as a Coordination-Chemistry Pre-training Corpus

**Status:** done · **Date:** 2026-09-11 · **TODO ref:** F2 (P1)
**Hardware:** AMD Radeon Graphics (ROCm), `torch 2.14.0+rocm7.2`

## 1. Motivation

MetalCytoToxDB is small (single-digit thousands of rows per metal) and has no
signal about *coordination geometry* — yet coordination number and metal–ligand
bond strength are exactly what distinguishes a cytotoxic Pt(II) square-planar
complex from an inert Pt(IV) octahedral prodrug. tmQM gives us 108k DFT-computed
mononuclear transition-metal complexes for free, with two supervision signals
that are unavailable anywhere else at this scale.

**Reference:** Balcells & Skjelstad, *tmQM Dataset — Quantum Geometries and
Properties of 86k Transition Metal Complexes*, J. Chem. Inf. Model. 2020, 60,
6135–6146. TPSSh-D3BJ/def2-SVP. MIT licence.
Repo: https://github.com/uiocompcat/tmQM (the copy on disk is the extended 108k
release).

## 2. Data layout and what we extract

The release ships no ready-made table — the supervision has to be recovered from
three different file formats:

| File | Content | What we take |
| --- | --- | --- |
| `tmQM_y.csv` (`;`-sep) | 8 DFT properties + SMILES | `SMILES`, DFT props |
| `tmQM_X{1,2,3}.xyz.gz` | XYZ blocks; comment line holds `CSD_code / q / S / Stoichiometry / MND` | **`MND`** = metal node degree = coordination number; metal identity from `Stoichiometry` |
| `tmQM_X{1,2,3}.BO.gz` | Per-atom Wiberg bond orders | **metal total Wiberg BO** + per-neighbour BOs |
| `tmQM_X.q` | NBO charges | (not used) |

Two traps worth recording:

1. **The metal is not the first atom of the XYZ block.** Only ~65 % of entries
   put it first (`C`, `Br`, … appear as atom 1). The metal is identified by
   intersecting the `Stoichiometry` string with the 30 d-block symbols, which is
   unambiguous because tmQM is mononuclear by construction.
2. **One CSD code (`IMUJUY`) is duplicated in the BO shards**, which inflates the
   merge by one row. `load_tmqm` de-duplicates on `csd_code`.

Full parse of the three gzipped shard sets takes ~6 s and is cached to
`tmqm_parsed.csv` next to the raw data (~0.5 s on subsequent loads).

## 3. Dataset statistics

```
n_total (all 30 metals, raw)                 108,541
n_total (all metals, SMILES non-empty)       100,850
dropped for empty SMILES                       7,691
```

Pt/Ru/Ir subset used for pre-training — **21,617 complexes**:

| metal | n | mean CN | mean metal Wiberg BO | coordination-number histogram |
| --- | ---: | ---: | ---: | --- |
| Ru | 9,068 | 7.17 | 5.65 | 6: 3,956 (44 %) · 9: 2,596 (29 %) · 8: 1,367 (15 %) · 5: 745 (8 %) · 7: 210 (2 %) |
| Pt | 7,854 | 4.32 | 3.92 | **4: 6,440 (82 %)** · 6: 1,105 (14 %) · 5: 238 (3 %) · 3: 26 · 7: 20 |
| Ir | 4,695 | 6.49 | 4.69 | 6: 2,455 (52 %) · 8: 1,370 (29 %) · 4: 298 (6 %) · 5: 258 (5 %) · 7: 235 (5 %) |

Entries missing a Wiberg bond order: **0**.

The Pt distribution is a useful external sanity check: 82 % square-planar
4-coordinate reproduces the tmQM paper's statement that Ni/Pd/Pt prefer
4-coordination, and it is the geometry cisplatin itself adopts. The high Ru/Ir
counts at CN 8–9 come from η⁵-Cp*/η⁶-arene sandwich ligands, where every ring
carbon counts as a ligating atom — a known artefact of degree-based CN
definitions, not a parsing bug.

Full metal breakdown (all 30 metals, top of the list): Ni 10,902 · Pd 10,696 ·
Ru 9,800 · Pt 8,275 · Zn 7,482 · Fe 5,691 · Ir 5,232 · Rh 4,640 · Au 4,338 ·
Re 4,172. (Counts before the SMILES filter.)

Machine-readable copy: `molmetal/reports/f2_tmqm_stats.json`.

## 4. Pre-training setup

**Architecture.** The existing `molmetal.models.dmpnn.DirectedMPNN` encoder
(unchanged, 3 layers, hidden 128, dropout 0.1) plus a 2-output MLP head.

The readout is deliberately **metal-aware**: the head consumes
`[h_metal_atom ‖ mean-pooled h_graph]` rather than a plain graph pooling, because
both targets are properties *of the metal centre*. This is the inductive bias we
want to transfer to MetalCytoToxDB.

**Batching.** Graphs are batched as a block-diagonal disjoint union so a whole
minibatch is a single `forward_per_atom` call. Because `DirectedMPNN` is entirely
index/scatter based, this is numerically identical to per-molecule forwards but
~20× faster on ROCm (6.3 s/epoch over 19.5k graphs vs. a Python-loop estimate of
~2 min).

**Targets.** Both z-scored for a balanced joint MSE; metrics are reported back in
raw units.
`mean = [5.992 CN, 4.811 BO]`, `std = [1.767, 0.911]`.

| setting | value |
| --- | --- |
| samples featurised | 21,615 (2 of 21,617 SMILES unusable) |
| train / val | 19,454 / 2,161 (random 90/10, seed 42) |
| epochs / batch / lr | 10 / 32 / 1e-3 |
| optimiser | AdamW, weight decay 1e-5, grad-norm clip 5.0 |
| params | 624,706 (encoder 583,424) |
| wall clock | 15.6 s featurisation + ~6.3 s/epoch |

## 5. Loss curve

| epoch | train loss | val loss | val MAE(CN) | val MAE(BO) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1727 | 0.1107 | 0.214 | 0.278 |
| 2 | 0.0858 | 0.0677 | 0.151 | 0.222 |
| 3 | 0.0752 | 0.0635 | 0.122 | 0.226 |
| 4 | 0.0678 | 0.0585 | 0.102 | 0.211 |
| 5 | 0.0607 | 0.0561 | 0.152 | 0.197 |
| 6 | 0.0564 | 0.0513 | 0.131 | 0.188 |
| 7 | 0.0514 | 0.0541 | 0.170 | 0.177 |
| 8 | 0.0517 | 0.0498 | 0.102 | 0.186 |
| 9 | 0.0485 | 0.0450 | 0.149 | 0.165 |
| **10** | **0.0455** | **0.0455** | **0.132** | **0.176** |

Loss is the mean z-scored MSE over the two heads. Train and validation loss track
each other to three decimals at epoch 10 — no overfitting, and the curve is still
descending, so more epochs would help if we ever need a stronger encoder.

Held-out validation, in physical units:

| task | MAE | RMSE | R² | mean-predictor MAE |
| --- | ---: | ---: | ---: | ---: |
| coordination number (MND) | 0.132 | 0.206 | 0.9862 | 1.365 |
| metal Wiberg BO (total) | 0.176 | 0.254 | 0.9239 | 0.768 |

Rounding the CN prediction to the nearest integer gives **97.0 % exact-match**
accuracy on the 2,161 held-out complexes.

## 6. Honest caveat — the two heads are not equally hard

This must be stated before anyone quotes the R² = 0.986:

**The coordination-number head is close to a graph invariant.** Simply counting
the RDKit degree of the metal atom in the *input* SMILES reproduces tmQM's MND
for **92.0 %** of the 21,615 complexes (mean |degree − MND| = 0.105). The model's
97.0 % is a genuine improvement over that trivial readout — it learns to correct
the 8 % where SMILES bond perception and tmQM's connectivity disagree (mostly
dative `->` bonds and η-coordinated rings) — but it is *not* an 0.986-R² discovery.
Its real value is as an auxiliary task that forces the encoder to build a correct
metal-centred receptive field.

**The Wiberg bond-order head carries the actual DFT signal.** Total metal Wiberg
BO is not recoverable from the 2-D graph by any counting rule: it depends on
ligand field strength, back-bonding and oxidation state. Going from a
mean-predictor MAE of 0.768 to 0.176 (R² = 0.924) is real learned chemistry, and
it is the part of the pre-training that should transfer.

When the fine-tuning ablation is run (§7), the honest comparison is
*pretrained vs. random-init with identical fine-tuning budget and split* — per the
PMO lesson (Gao 2024) that "many claimed improvements are spurious". A single
seed will not settle it; plan for ≥5 seeds and report the spread.

## 7. Plan: how to use this for MetalCytoToxDB fine-tuning

The checkpoint stores the encoder separately from the head precisely so it can be
dropped into the downstream models:

```python
from molmetal.scripts.pretrain_coordination import load_pretrained_encoder
encoder, meta = load_pretrained_encoder()   # DirectedMPNN + metadata dict
```

Proposed sequence:

1. **Swap the encoder in.** `molmetal/baselines/dmpnn.py` and
   `molmetal/models/metal_hybrid_v3.py` both instantiate a `DirectedMPNN` with a
   fresh init. Add a `--pretrained` flag that loads
   `checkpoints/dmpnn_tmqm_pretrained.pt` instead. The atom/edge feature dims
   (39/6) already match, so no featuriser change is needed — this is the reason
   pre-training used `GraphFeaturizer` rather than a tmQM-specific encoding.
2. **Two fine-tuning regimes, both reported.**
   (a) *frozen encoder* + new pIC50/activity head — isolates how much of the
   downstream signal the coordination representation already contains;
   (b) *full fine-tune* at a lower encoder LR (1e-4 encoder / 1e-3 head) — the
   regime expected to win, with discriminative LR to avoid washing out the
   pre-trained weights in the first few hundred steps.
3. **Evaluate on the temporal split, not random.** The honest baseline target is
   Krasnov 0.81 AUC; the counterion ablation (T2 A2) already showed the temporal
   split moves AUC by 0.08, which is the same order as any plausible pre-training
   gain. Pre-training must be judged on temporal AUC with the counterion handling
   held fixed, otherwise the two effects are confounded.
4. **Metal coverage caveat.** MetalCytoToxDB has Ru/Ir/Rh/Os/Re and **no Pt**,
   while the Pt story runs through the cisplatin case study. The pre-trained
   encoder is the only component in the pipeline that has seen 7,854 real Pt
   complexes, so it is also the justification for making any Pt-directed
   prediction at all. State that explicitly in the paper rather than implying
   Pt performance was validated on cytotoxicity data.
5. **Optional extensions if the transfer is weak.** (i) Widen pre-training to all
   30 metals (100,850 complexes, ~5× data, ~30 s/epoch) and let the metal
   one-hot do the specialising; (ii) add HOMO–LUMO gap and `Metal_q` from
   `tmQM_y.csv` as extra auxiliary heads — both are already parsed and in the
   dataframe, no new I/O required; (iii) pull tmQMg-L (30k ligands) for a
   ligand-level tile library, which connects to the T2 L3 click-tile work.

## 8. Artefacts

| path | content |
| --- | --- |
| `molmetal/data/tmqm.py` | parser + `load_tmqm` / `filter_by_metal` / `summarize` |
| `molmetal/scripts/load_tmqm.py` | corpus-inspection CLI |
| `molmetal/scripts/pretrain_coordination.py` | pre-training CLI + `load_pretrained_encoder` |
| `molmetal/tests/test_tmqm_loading.py` | 3 tests (all pass, 1.8 s) |
| `molmetal/checkpoints/dmpnn_tmqm_pretrained.pt` | encoder + head + norm constants + history |
| `molmetal/reports/f2_tmqm_stats.json` | dataset statistics |
| `molmetal/reports/f2_pretrain_history.json` | full loss curve + run config |

Reproduce:

```bash
source .venv/bin/activate
python -m molmetal.scripts.load_tmqm
python -m molmetal.scripts.pretrain_coordination --epochs 10
python -m pytest molmetal/tests/test_tmqm_loading.py -q
```
