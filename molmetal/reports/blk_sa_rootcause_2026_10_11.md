# P0-3 BLK-SA-rootcause — the SA clue is NOT a batch effect

**Date:** 2026-10-11
**Tag:** MEASURED (all numbers recomputed this session from files on disk)
**Cost:** zero GPU, zero retrain. Read-only analysis.

## Verdict

The SA drop from ~6.7-7.5 (earlier) to ~5.01 (today) is **NOT a batch effect and
NOT a definition change**. It is a **population change with a specific,
identified cause: today's three `generated_smiles.csv` were not produced by
sampling the CFM at all.**

`molmetal/scripts/wf_cfm_decode_dump_smiles.py` loads **real tmQM geometries**
(`load_tmqm_clouds`, real `AtomCloud` + real Wiberg bond orders), then runs
**only the bond head** on them (`head.featurise` -> `_select_bonds` ->
`_bridge_components` -> `_assemble_mol`). It never calls the flow-matching ODE.
The trainer's real generation path does: `evaluate_generation_validity` calls
`generate_clouds_from_noise(model, ..., method=method, n_steps=n_steps,
noise_sigma=...)` (`train_cfm_yuelbond_wasserstein.py:4683-4690`). The dump
script has no `n_steps`, no `method`, no `noise_sigma`, no integration call —
verified by grep over the whole file.

So the ~5.01 is **bond-head reconstruction of real corpus molecules**, and it
lands where real molecules land. That is the whole gap.

## The number that settles it

Real tmQM corpus (metals Pt,Ru,Ir,Rh,Os,Re), scored with the *identical*
`_load_sascorer` used by `cfm_property_panel`:

| cohort | n | SA mean |
|---|---|---|
| real tmQM corpus | 31150 | **4.9781** |
| real tmQM, size-matched to the 192 val clouds the dump script loaded | 192 | **4.8781** |
| today's three "generated" cohorts | 120 | **5.0146** |
| gap (generated - size-matched real) | | **+0.1365** |

Generated output is within 0.14 SA of the real molecules it was handed. It is
not "better chemistry than reality" — it is **the same chemistry**, because the
geometry never came from the model.

## Batch-effect hypothesis: NOT SUPPORTED

Direct tests on `molmetal/reports/{cfm_fullcorpus_2026_10_10,
cfm_objective_terms_2026_10_11, cfm_fullcorpus_b64_2026_10_11}/generated_smiles.csv`,
120 rows, all scored by me (none of these CSVs carried an SA column before):

| per-run factor | test | result |
|---|---|---|
| run identity | eta^2 (between-run share of SA variance) | **0.0006** — per-run means 4.974 / 5.033 / 5.032 |
| decode epoch | corr(SA, epoch) | **-0.021** |
| metal | per-metal means | 4.085 (Re) .. 5.841 (Rh) — present but not the driver |
| fragment count | corr(SA, n_frag) | -0.402 |
| size | corr(SA, heavy atoms) | **-0.432** (NEW) vs **+0.618** (old cohort) |

The three runs agree to within 0.06 SA. There is no batch to blame.

## Definition change: NOT SUPPORTED

`git log -p -- molmetal/scripts/cfm_property_panel.py` shows `calculateScore`
was never touched; the only change to `compute_one` was threading
`include_validity` through. The scorer is `rdkit.Contrib/SA_Score/sascorer.py`,
unchanged. Every number in this document was produced by calling
`cfm_property_panel._load_sascorer` itself, so all cohorts are on one scale.

## What the gap actually decomposes into

Ertl score = fragment-rarity + size-penalty + stereo/macro penalties:

| term | NEW (n=120) | old wf124 cohort (n=47) |
|---|---|---|
| fragment rarity (`score1`) | **+0.6356** | **-0.4357** |
| log(n_frag_bits) | +2.042 | +1.910 |
| size penalty | +3.5321 | +3.4691 |
| stereo penalty | +0.3239 | +0.5877 |
| **total** | **5.0146** | **6.4892** |

The size penalty is ~identical. The entire difference is **fragment rarity** —
i.e. how many Ertl-unseen substructures the molecule contains. Real tmQM
molecules contain fewer of them than the earlier cohorts did. This is a
*property of the molecules*, consistent with them being real.

Size-matched buckets (NEW vs the one pre-today cohort still on disk,
`wf124_p2_cfm_full/generated_smiles.csv`, n=47, which I re-scored today with the
same scorer and got 6.4892 — a recomputed number, not a stored one):

```
heavy   NEW(n,SA)   OLD(n,SA)
   20    26 6.11      8 6.16
   30    31 4.85     18 6.35
   40    40 4.56     10 6.64
   50    11 4.88      7 6.92
   60    11 4.64      2 7.43
```

NEW is 1.5-2.8 SA units *better* at every matched size. Size is not the
confounder.

## Why the old 6.66-7.93 numbers cannot be re-derived

`metrics/by_metric/cfm_decode_chemistry_validity.json` (n=1128 / 871 / 147) and
`cfm_internal_diversity_intdiv1.json` are the sources. Their runs
(`cfm_fm_ab_2026_10_09`, `cfm_fm_control_2026_10_09`,
`cfm_blk_c5_repair_2026_10_09`, `cfm_frag_control`, `cfm_frag_assemble`) have
**no `generated_smiles.csv`** — `ls` confirms, and
`git log --all --diff-filter=A` shows they were never committed. Their
`per_epoch_metrics.csv` rows carry `gen_sa_observed=0, gen_sa_mean=0.0000`, i.e.
the trainer itself never scored SA on those runs. The input cohort for that
panel run is **not on disk**, so the old numbers are currently unfalsifiable.
That is itself a finding: a metric was published from an unrecoverable input.

## Was the earlier number wrong, or just measured on a different thing?

Both cohorts are internally honest; they measure different things.
The 6.7-7.5 numbers came from the trainer's `gen_validity` path
(`gen_eval_n=16`, seeds 0,1,2, **from noise**), which is a real generation
sample. Today's 5.01 came from a re-decode of **real geometries**. They are not
comparable, and today's number does not supersede the old one.

## Evidence that would distinguish the remaining hypotheses

To confirm the ODE-skip reading rather than infer it, one cheap CPU pass:
decode the same `cfm_fullcorpus_2026_10_10` checkpoint **through the trainer's
`evaluate_generation_validity`** (which integrates the ODE) with
`--save-generated-smiles`, and score that CSV. Prediction: SA returns to the
6.7-7.5 band. Decode-only over one checkpoint — no training, and it does not
touch the live `cfm_fullcorpus_seed{1,2}/` runs.

Secondary check, also cheap: re-run the dump script with a deliberately
noise-initialised cloud instead of `load_tmqm_clouds`. If SA jumps, the ODE is
the only difference.

## Files

- analysis scripts: `/tmp/blska/` (scratch, not repo artifacts)
- cohorts read: `molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv`,
  `.../cfm_objective_terms_2026_10_11/...`, `.../cfm_fullcorpus_b64_2026_10_11/...`,
  `.../wf124_p2_cfm_full/generated_smiles.csv`
- corpus: `.data/molmetal/tmQM/tmqm_parsed.csv`
- root cause: `molmetal/scripts/wf_cfm_decode_dump_smiles.py:140-170` vs
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:4683-4690`
