# CFM line: six rounds, what was actually learned, and what to build next

Written 2026-10-13 from disk. Every figure below carries its path. Nothing
here is projected.

> **2026-10-13 correction (same day, later).** §2 and OPT-1 were measured
> through `molmetal/eval/scoring/sascorer.py`, a vendored fork whose
> fragment term is a hardcoded constant — its Morgan fingerprint is folded
> to `fpSize=2048` while the Ertl table is keyed on sparse 32-bit ids, so
> **0 of 66 molecules match a single fragment bit**. Under the canonical
> scorer the SA sign reverses. See [§7](#7-correction-the-sa-fork-was-measuring-a-constant)
> and `metrics/by_metric/cfm_opt1_stereo_suppression.json`.
>
> §1's SA row is **not** affected: `cfm_sa_gap_decomposition.json` was
> computed through RDKit's bundled scorer (its `generated.sa_mean` is
> 6.4423348), so the 71.6% attribution stands. Two figures in §1/§2 were
> nonetheless wrong for other reasons and are fixed inline: the generated SA
> mean was 6.258 (that is the stereocentre count, not SA), and the
> stereocentre ratio was 31.1x on a 4,000-row prefix rather than 19.8x on
> the full corpus.

## 1. The pain points, measured

| target | measured | verdict on disk |
|---|---|---|
| QED | generated 0.0944 (n=66) | ROTB +0.0577 and ALERTS +0.0561 carry essentially all of it |
| SA | generated 6.4423 vs real 4.9781 (gap 1.4642) | 71.6% of the gap is the stereocentre penalty; fragment rarity points the WRONG way (-0.32) |
| ADMET | ames 0.0 vs 0.0349 real, logPapp -3.039 vs -3.680 | BETTER than real already, and the logPapp "win" is a clipping artefact |
| precious metal | 0 of 8 ADMET proxies read the metal | ADMET cannot express a metal-specific fact |

Paths: `metrics/by_metric/cfm_objective_inventory.json` (all four verdicts),
`metrics/by_metric/cfm_sa_stereocentre_control.json`,
`metrics/by_metric/cfm_free_generation_panel_seed0.json`,
`metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json`.

## 2. The stereocentre finding nobody was looking for

    generated   6.258 potential stereocentres per molecule, 97.0% of molecules have >=1   (n=66)
    real tmQM   0.316 potential stereocentres per molecule, 83.2% have none              (n=31150, FULL corpus)
    ratio       19.8x

Path: `metrics/by_metric/cfm_sa_stereocentre_control.json`
(`::corrections_2026_10_12[0]`). An earlier 31.1x figure used the first
4,000 rows of the filter, which is not a random sample; on that prefix real
is 0.201/molecule. Use 19.8x.

The consequence is a DIRECTION REVERSAL. SA punishes stereocentres. Real
anticancer Pt/Ru/Ir complexes are overwhelmingly achiral. So generating
chirality is not a weaker version of the fix -- it is the wrong direction
entirely. The lever is suppressing unassigned potential centres, not producing
better ones.

This sat unactioned for six rounds because the standing question was "how do
we get MORE aromatic bonds", and aromatic bonding was treated as the proxy for
ring content. It is not the same thing.

## 3. What six rounds of bond-head work actually produced

A correct negative, and one correction:

- The post-hoc aromatic logit offset is MEASURED DEAD. Bias 0.0 / 0.5 / 1.5 /
  3.0 gave 22 / 13 / 3 / 0 decodes of 64 with **0 aromatic bonds across all
  1061 bonds examined**. Path:
  `molmetal/reports/cfm_aromatic_bias_sweep_seed0/`.
- The head has a transition zone at Wiberg [1.46, 1.51], not a
  clean boundary at 1.48 as round 6 first reported. Above it the head is
  right (recall 0.66 to 0.86); below it the label itself is ambiguous.
  Path: `molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13/`.
- Metal labels are SOUND. Of 649,758 aromatic truth bonds over 33,429
  metal-containing complexes only 9,677 (1.49%) are metal-bound. Path:
  `molmetal/reports/wf_meta_metal_label_audit/audit.json`.
- bond-TYPE macro-F1 still LOSES to OpenBabel: 0.4784 +/- 0.0055 (3 seeds:
  0.4723 / 0.4829 / 0.4801) against 0.5051 on the identical per-complex
  protocol, i.e. -0.0267 behind and consistent across all three seeds.
  Path: `metrics/by_metric/tmqm_bond_perception_3seed.json`.

So the retrain slot should NOT go to the bond head. It should go where the
objective actually is.

## 4. The model-internal optimisation plan

Four changes, ordered by measured size of the lever. Each states what it
targets, what it costs, and what it would decide.

### OPT-1  Suppress unassigned potential stereocentres  (biggest measured lever)

- Targets: 71.6% of the SA gap AND the ALERTS share of the QED gap, together.
- Mechanism: the stereo penalty is `log10(n_unassigned_potential_centres+1)`
  (recorded in `cfm_sa_stereocentre_control.json`), so it is a COUNT, not a
  quality. A postprocess that removes unassigned potential centres from the
  decoded molecule is worth measuring BEFORE any retrain: it is decode-time,
  needs no training, and the metric already exists to score it.
- Cost: CPU, no GPU, no retrain. One script plus a red-checked test.
- Decides: how much of the 1.464 SA gap is reachable without touching the
  model at all. If most of it is, the retrain slot is freed for something
  that needs it.
- RISK: removing centres may raise QED or may not; QED ROTB counts rotatable
  bonds and deleting centres can break rings. Measure both, do not assume.

### OPT-2  Make rings, not just aromatic bonds

- Targets: the 98.5% non-aromatic figure and the ring content behind ROTB.
- Mechanism: aromatic BOND fraction is not ring fraction. A saturated ring
  fixes ROTB and ALERTS the same way an aromatic ring does, and the sampler
  currently produces neither. If the head is being asked for aromatics it
  cannot rank, the cheaper target is ring CLOSURE.
- Cost: CPU diagnostic first -- measure ring count in generated vs real. If
  generated rings are also near zero, this is a connectivity-closure problem,
  not a bond-TYPE problem.
- Decides: whether the bond head is relevant to the objective at all.

### OPT-3  Fix the metal sampling so the precious-metal axis becomes measurable

- Targets: the axis the user actually named and which is currently
  UNMEASURABLE.
- Mechanism: `min_per_metal` already exists and is guarded. The zero-Pt
  finding means a cohort can be drawn with no Pt at all, so any round claiming
  to score precious metals must first report `generated_metal_fraction[Pt]`.
- Cost: config change plus the existing guard.
- Decides: whether the precious-metal objective can be measured at all.

### OPT-4  Point the single retrain slot at bond_ce class-4 vs class-1

- Only after OPT-1 and OPT-2 have shown how much is reachable without it.
- Status: macro-F1(type) 0.4784 vs OpenBabel 0.5051 is a real honest
  negative and still needs addressing, but it is NOT what moves QED or SA.

## 5. What must NOT be claimed

- The 70.7% aromatic collapse is not "the head's fault". It relocates rather
  than fixes when the band moves.
- 22.8x pad_groups speedup is WITHDRAWN. The runnable figure is 4.78x and it
  is not equivalent on one seed (63 of 192 clouds flip).
- ADMET cannot be presented as a metal-specific result.
- "66% of generated bonds are triple" misreads
  `cfm_objective_inventory.json`, which reports the fraction of MOLECULES
  containing a triple bond, not the fraction of BONDS that are triple.

## 6. Ring size, not ring count (OPT-2, measured)

Round 7 treated aromaticity as the target. It is downstream. The measured
defect is ring SIZE:

| metric | generated | real | ratio |
|---|---|---|---|
| rings / mol | 1.64 | 4.19 | 0.39x |
| **5/6-membered rings** | **1.85%** | **96.92%** | **0.019x** |
| **3-membered rings** | **89.8%** | 0.5% | 180x |
| rotatable bonds | 16.86 | 5.18 | 3.26x |
| largest-fragment fraction | 0.9984 | 0.9996 | essentially identical |

Fragmentation is **not** the story — `largest_frag_frac` agrees to 4 dp.
ROTB 16.86 is ring size in disguise. A 3-ring has no aromatic degree of
freedom, so chasing aromatic bonds was chasing a symptom. This is why OPT-4
(bond-head retrain) should not get the retrain slot: a better head cannot
relabel a bond whose atoms never close a 5-ring.

Path: `metrics/by_metric/cfm_opt2_ring_content.json`,
`molmetal/reports/cfm_opt2_ring_content_2026_10_13/summary.json` (n=2000 of
the 31152-row six-metal tmQM corpus).

## 7. Correction: the SA fork was measuring a constant

Everything in §1's SA row and all of §2/OPT-1 was computed through
`molmetal/eval/scoring/sascorer.py`. That file does not implement Ertl 2009
and cannot run:

- It calls `GetCountFingerprint` on a `fpSize=2048` Morgan generator,
  producing 11-bit hashed ids. The Ertl table is keyed on the 32-bit ids
  `GetSparseCountFingerprint` produces. **MEASURED: 0 of 66 molecules match
  a single fragment bit** (canonical: 41.3485 bits/molecule, max 75). With
  `nf == 0` the fork returns its `else` branch, `score1 = 5.0`, for every
  molecule — the term Ertl built the method around is a constant.
- Removing the `fpSize` keyword does **not** fix it; it just defaults to
  2048. Only the sparse call matters.
- It cannot execute at all: its table is at a `.gitignore`'d path that is
  absent, so every call raises `FileNotFoundError`. Three modules import it
  and none can return a score — `pocket2mol_score`, `softmol_score`,
  `targetdiff_score`. `sa_penalty.py` escapes only because it tries RDKit's
  bundled scorer first.

Same 66 molecules, two implementations:

    canonical (rdkit sascorer)   SA mean 6.442335   <- every number in the repo
    in-tree fork                 SA mean 3.303854
    difference                   3.138481 SA units

Round 8's OPT-1 probe ran through the fork, which is why it reported a
baseline (3.3039) that appears nowhere else, and why it concluded SA moves
the **wrong way** (+1.115) under every intervention. Under the canonical
scorer the sign reverses, with a size-matched control — necessary because
SA carries a size penalty and the intervention deletes 6.26 atoms/molecule:

| arm | SA | d_SA | atoms/mol |
|---|---|---|---|
| baseline | 6.4423 | — | 37.15 |
| A: remove stereocentre atoms | 4.8278 | **-1.6145** | 30.89 |
| B: size-matched control (same count, non-stereocentre atoms) | 6.1021 | -0.3402 | 31.02 |
| **stereocentre-attributable** | | **-1.2743** | |

QED attributable: +0.0381 (A +0.1027 vs B +0.0645).

**What survives from round 8:** `set_tag` and `remove_annotation` stay
vacuous, and that is scorer-independent —
`FindMolChiralCenters(includeUnassigned=True)` counts assigned centres too,
so tagging cannot lower the count the penalty reads.

**What this does not claim:** dropping atoms is a *score-level* postprocess,
not a chemistry improvement — it deletes 6.26 atoms per molecule and leaves
a 12-atom fragment in the worst cases. What it establishes is narrower: the
stereocentre axis carries 1.27 SA units of real attributable signal, so a
sampler-side constraint has something to aim at. Whether a generator reaches
it *without* deleting chemistry is the re-opened question.

**Still open:** fix the fork or delete it and route every caller through
RDKit's bundled scorer. Not done here — both change what other numbers in
the repo mean, and the caller sweep is wider than this pass.

Paths: `molmetal/tests/test_sascorer_fork_divergence.py` (8 tests, three
mutation runs recorded in its docstring),
`metrics/by_metric/cfm_opt1_stereo_suppression.json`,
`molmetal/scripts/wf_cfm_opt1_stereo_suppression_2026_10_13.py`.

## 8. Revised priority

Round 7's ordering (OPT-1 > OPT-2 > OPT-3 > OPT-4) inverted to
OPT-3 > OPT-2 > OPT-1 > OPT-5 > OPT-4 after round 8, because OPT-1 was
believed infeasible and OPT-2 found the ring-size axis. The §7 correction
puts OPT-1 back in contention, but not at the top: the honest reading is that
its 1.27 SA units are reachable only by a postprocess that deletes
chemistry, which is a scorer intervention rather than a generator one. What
the user actually named — QED, ADMET, SA, precious-metal-specific — still
routes through OPT-3 first, because until `generated_metal_fraction[Pt]` is
non-zero the precious-metal axis cannot be measured at all.

