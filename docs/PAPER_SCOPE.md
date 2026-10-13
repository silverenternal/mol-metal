# Paper scope A / B and the shared-asset manifest

Decided 2026-10-11. Two papers, one repository for now; the repository split is
deferred until Paper A is submitted. Nothing here touches `paper/` (AGENTS.md
§5.3, TODO-84).

## Why two papers rather than one

The two lines carry evidence of very different strength, and a single paper
would either hide the strong half or be dragged down by the weak half.

| | Paper A — predictive | Paper B — generative |
|---|---|---|
| strongest number | cross-metal collapse, see below | bond EXISTENCE F1 0.9961 vs OpenBabel 0.9771 and exact-type rate 0.0978 vs 0.0300, same 900 complexes |
| honest ceiling of the weak half | — | macro-F1(type) 0.4723 still **behind** OpenBabel 0.5051; `gen_validity` reaches 0.59–0.62 but is measured at n=768 and swings 0.44–0.62 within a run |
| reviewer pool | cheminformatics / QSAR | generative modelling |
| readiness | submit-ready once §"open contradictions" is closed | **not ready** |

**Corrected 2026-10-11.** This table previously read "`bond_type_f1` 0.9194 vs OpenBabel
0.505". That pair differed in three variables at once — aggregation (pooled vs per-complex),
model scale (n_train 26715 vs 256) and cohort (192 vs 900 complexes). The production checkpoint
was re-scored under the bench's own protocol
(`molmetal/reports/tmqm_bond_perception_fullcorpus/`); the same checkpoint scores **0.4723**
there, not 0.9194. Full record in
`metrics/by_metric/tmqm_bond_perception.json::production_scale_comparison_2026_10_11`.

## PAPER A — scope

### Claim 1 (headline). Metal-specific activity does not transfer across metals.

MEASURED, `metrics/by_metric/metal_activity_transfer.json` and
`metal_activity_transfer_seed_stability.json`:

| metal | within-metal (5-fold) | cross-metal LOMO (3 seeds) |
|---|---|---|
| Ru | 0.8363 | 0.6174 |
| Ir | 0.8295 | 0.5341 |
| Rh | 0.7381 | 0.4919 |
| Os | 0.4900 | 0.5195 |
| Re | 0.9278 | ~0.46 |

Os and Re cross-metal AUC sit **below chance**. This axis has no organic
analogue — ligand-level activity models transfer across organic scaffolds, but
here the metal itself is the confound. That is the novelty.

### Claim 2. tmQM pre-training does NOT transfer to the activity head.

MEASURED, `metrics/by_metric/tmqm_pretrain_init_ab_ru.json`, 5 seeds x 30
epochs, paired, **FIXED Bemis-Murcko scaffold split** (desalt-aware,
BLK-PA12), `init:` the only difference:

```
tmqm   0.7778 +/- 0.0194
random 0.7872 +/- 0.0201
paired delta -0.0094   95% CI [-0.0280, +0.0116]   t(4)=-0.839   gate: FAIL
```

**Corrected 2026-10-12.** This claim previously read "paired delta **+0.0181**,
95% CI [+0.0035, +0.0328], 5/5 seeds positive, gate PASS", from the same pair of
configs under the earlier `largest_first` split. Both arms moved DOWN on the
fixed split (0.8189 -> 0.7778 and 0.8007 -> 0.7872), so this is a protocol
change, not run-to-run noise. The old positive result is preserved in the metric
JSON under `supersedes`.

**Why it matters more than a plain null result:** the earlier +0.0181 was
measured on a split that did not hold scaffold groups out, which is exactly the
condition under which memorising the pretraining distribution looks like
transfer. On a disjoint split there is no effect. Any claim that this encoder
transfers must now demonstrate it on a leak-free split, or not be made.

Transfer is still verified mechanically, not assumed -- both arms print
`loaded 12 backbone tensors from tmQM`. What failed is that the weights do not
help, not that they failed to load.

### Claim 3. Honest-negative against the published protocol.

MEASURED, `metrics/by_metric/metalcytotox_ru_auc.json::paper_protocol_measurement`:
under the paper's own **DOI-split 10-fold CV** on the Krasnov-2026 cohort we
score **0.7717 ± 0.0380** against their LightGBM **0.810 ± 0.024** — 0.0383
**below**, 3/10 folds at or above 0.81, different model class. This is reported,
not buried.

### Supporting

- `m21_nci60_9cell` 0.8007 ± 0.0160, target 0.70 — **PASS**
  (`metrics/by_metric/nci60_precious_9cell_auc.json`)
- Ir within-metal exceeds Krasnov 0.73 under **both** tie rules and all 5 seeds
  (conservative min 0.8237) — **safe to state**
- tmQM bond-perception reference table: 360 complexes, 6 metals, ground truth
  from tmQM `.BO` Wiberg shards; the file records that **no published number
  exists for this task on any transition-metal corpus**
  (`metrics/by_metric/tmqm_bond_perception.json`)

## PAPER A — open contradictions that must be closed before submission

1. **`0.8363` is recorded twice with opposite verdicts.**
   `metal_activity_transfer.json::within_metal_cv` records Ru 0.8363 / Ir 0.8295
   as MEASURED and its own tie-sensitivity file calls the Ir claim robust and
   the **Ru claim NOT robust** (conservative rule: 0.7880–0.8082, at or below
   Krasnov 0.81). Meanwhile `todo.json::withdrawn_claims_2026_10_10` withdraws
   the same numbers entirely as a random-split artefact. **Reconcile before
   writing.** The cross-metal collapse (Claim 1) is unaffected — it uses the
   cross column, not the within column.

2. **Tie rule is documented backwards.** `collapse_duplicates` uses
   `int(v.sum()*2 >= len(v))`, which sends ties to **ACTIVE**; four docstrings
   and configs state "ties → inactive". Both sides use the same rule so the
   comparison is not corrupted, but the documentation is false and a reviewer
   reading the code will find it.

3. **`cfm_decode_real.json` reports an aggregate with no data.**
   `n_nci60_loaded: 0`, `n_drugood_loaded: 0`, both paths null — the run loaded
   nothing and still wrote a number.

4. **`dmpnn_platinai_4cell_50ep` is routed at by nothing useful.** Its own
   recorded test_aucs are 0.4949 / 0.4799 / 0.5064 / 0.5765, and fed the real
   cohort its probabilities do not track the base rate at all (A2780 is 72.9%
   active, the model scores it 0.467). The strong PlatinAI model
   (`train_platinai_4cell.py`, 0.7050) persists **no weights** — `torch.save`
   appears zero times in that file.

## PAPER B — scope, and why it is not ready

Scope: flow-matching generation of metal complexes with an explicit bond-order
head, scored against real tmQM Wiberg truth on 900 held-out complexes (150 per
metal, six metals, non-hydrogen pair scope, per-complex aggregation).

**What the head actually earns, MEASURED at production scale**
(`molmetal/reports/tmqm_bond_perception_fullcorpus/`):

| | CFM (n_train 26715) | OpenBabel | verdict |
|---|---|---|---|
| bond EXISTENCE F1 | **0.9961** | 0.9771 | **win** |
| exact-type rate | **0.0978** | 0.0300 | **win, 3.3×** |
| macro-F1(type) | 0.4723 | **0.5051** | **loss, 0.0328 behind** |

The scale effect is real and large: the same protocol at n_train 256 gave
F1(exist) 0.6118 / type 0.2801 / exact 0.0011. The bond head went from failing
on every axis to winning two of three. It still **loses the axis it was
designed to win** — bond ORDER — and that is the honest-negative of this paper.

The previous framing here ("0.9194 vs OpenBabel 0.505") was withdrawn: those two
numbers used different aggregations, different model scales and different
cohorts, and the same checkpoint scores 0.4723 under this protocol.

**Not ready, and the reason is measured, not assumed.** Fragmentation is the
binding constraint. `gen_validity` reaches 0.59–0.62 on the three full-corpus
seeds at the same protocol (n=768, midpoint/50,
`metrics/by_metric/cfm_bond_type_f1_3seed.json`), which is HIGHER than the
0.23–0.38 this document previously quoted as the honest ceiling — that earlier
band came from `cfm_objective_terms` (9 epochs) and `cfm_fm_distort` (8
epochs), i.e. short runs, and quoting them as a ceiling was wrong. Within a
single run the value still swings 0.44–0.62, so it is not yet a tight number.

Every cheap lever has been tested and failed:

- valence budget is a **local optimum** — raising caps, lowering bond cost,
  withholding budget for the bridge, and removing the budget entirely are each
  worse on **both** connectivity and sanitisation
  (`metrics/by_metric/cfm_valence_budget_variant_ab.json`)
- train-time geometry distortion is worse on fragmentation at **8/8** matched
  epochs (neutral on `decode_ratio` and `gen_validity`, 4/8 each)
- `entropy_reg_weight` 0.01 is a **functional no-op** (todo.json BLK-C14)
- `assemble_loss` is exactly **0.0** at every epoch despite
  `assemble_enabled=1.0` — it is inert by construction, not by wiring

So the valence-budget work is a **design argument**, not a performance result,
and must not be presented as one.

## Shared-asset manifest

Two papers, two repositories later. These are the files that must not be
duplicated into both, because they are the same object appearing in both
stories:

| asset | path | used by |
|---|---|---|
| tmQM pretrained encoder | `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` | **A and B** |
| tmQM cloud loader | `molmetal/data/tmqm_clouds.py` (7/27 CFM scripts import it) | **A and B** |
| D-MPNN backbone | `molmetal/baselines/dmpnn.py` | A (heads) and B (none directly) |
| multitask model | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | A |
| featuriser + collate | `molmetal.baselines.dmpnn::{featurize_smiles_list, collate_graphs}` | A |
| inference scorer | `molmetal/scripts/de_novo_loop.py::{load_pic50_scorer, score_pic50}` | A |
| metal valence caps | `TMQM_MEASURED_METAL_CAPS` in `molmetal/molmetal_lam/lam_chem/decoder_rework.py` | **A and B** |
| tmQM bond-perception benchmark | `molmetal/scripts/bench_tmqm_bond_perception.py` | **A and B** |

## Sequencing

1. Close the four open contradictions in Paper A.
2. **Paper A cites Paper B's encoder explicitly.** Pretraining is shared; saying
   so up front is strictly better than a reviewer noticing the same checkpoint
   in two papers.
3. Submit Paper A.
4. Only then extract Paper B into a second repository depending on a shared
   core package. Splitting before A is submitted would hide the shared origin.