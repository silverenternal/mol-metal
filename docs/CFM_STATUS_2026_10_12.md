# CFM line — consolidated status, 2026-10-12

Single source of truth for the generative (CFM) line as of this date. Every
number below is MEASURED and carries the path it came from. Where a value is
not on disk it says **NOT ON DISK** rather than being estimated.

Plan and blockers: [`todo.json`](../todo.json) (repo root).

> **2026-10-27 — `cfm_todo.json` was merged into `todo.json` and deleted.**
> The CFM half of the plan (D1-D11 / MOD1-MOD9 / M1-M10 / T1-T12) moved
> verbatim, ids unchanged. Every `cfm_todo.json::X` reference below is a
> historical record of what the plan said on 2026-10-12 and is **not** a live
> pointer; `todo.json::legacy_identifiers` maps the ids that no longer exist
> (BLK-C-PERF, BLK-C-OBJ, BLK-C-M3, BLK-C-L3, A4, ...) to where the content
> lives now.
Scope: [`PAPER_SCOPE.md`](PAPER_SCOPE.md). Nothing here touches `paper/*`.

---

## 1. What the model is, and what it is good at

A flow-matching generator over tmQM metal complexes with a learned bond-order
head (YuelBond), pocket-conditioned (`pocket_id=1h36`). 704,393 parameters.

**Bond perception, 3 seeds, one protocol** (900 complexes, 150 per metal, six
metals, non-hydrogen pair scope, per-complex aggregation):

| | seed 0 | seed 1 | seed 2 | mean ± std | OpenBabel |
|---|---|---|---|---|---|
| bond EXISTENCE F1 | 0.9961 | 0.9966 | 0.9954 | **0.9962 ± 0.0008** | 0.9771 |
| exact-type rate | 0.0978 | 0.1061 | 0.0828 | **0.0956 ± 0.0113** | 0.0300 |
| macro-F1 over TYPE | 0.4723 | 0.4828 | 0.4801 | **0.4784 ± 0.0055** | **0.5051** |

Every margin is 5×–24× the between-seed std, so none of the three is a seed
artefact.
→ `metrics/by_metric/tmqm_bond_perception_3seed.json`

**It wins two axes and loses the third.** The axis it loses — macro-F1 over bond
TYPE — is the one the bond head exists to win. That is the honest-negative of
this line.

---

## 2. What it produces (first real free-generation measurement)

The only numbers on disk that come from integrating the ODE from noise rather
than reconstructing real molecules. 192 tmQM val clouds, midpoint/50, seed-0
checkpoint, epoch 10:

| | generated | reconstruction of real tmQM |
|---|---|---|
| yield (gen_validity) | 66/192 = **0.344** | — |
| single_fragment_ratio | **0.9697** | 0.8365 |
| metal_ratio | 1.000 | — |
| RDKit 3D-embeddable | **54/66 = 0.818** | not measured |
| QED | **0.0944** | 0.2685 |
| SA (Ertl) | **6.4423** | 5.0269 |
| AMES mutagen (proxy) | 0.0000 | 0.0173 |
| logPapp (proxy) | −3.0387 | −3.0794 |
| logS (proxy) | −8.0099 | −8.3385 |

→ `metrics/by_metric/cfm_free_generation_panel_seed0.json`,
`metrics/by_metric/cfm_panel_reference_band_real_tmqm.json`,
`metrics/by_metric/cfm_checkpoint_property_benchmark.json`

> **The comparison column is NOT real tmQM.** 2000 real tmQM val geometries
> were passed through the *same YuelBond bond head*; 636 yielded a SMILES. All
> 636 carry bracket-atom notation (`[C]`, `[Rh]`) and 13 are bare metal
> fragments with no ligands bonded. So the right-hand column is a
> **reconstruction floor**, not a corpus floor, and the comparison is
> generated-vs-reconstructed. Stated here because every "vs real" phrasing in
> this section inherited the wrong label.

**Read this carefully — three different things are true at once.**

1. **Organic connectivity and valency are not the bottleneck.**
   single_fragment_ratio 0.9697, bond EXISTENCE F1 0.9962, and an aromatic-aware
   valency audit puts *both* cohorts at 1.0000 on organic atoms. This clause was
   previously written as the unqualified "connectivity is not the bottleneck";
   that sentence is **withdrawn** (see the correction below) because the
   evaluators it rests on skip every metal centre.
2. **The metal coordination sphere is the bottleneck.** 18.8% of generated
   molecules cannot be embedded in 3D by RDKit (ETKDG returns −1), and
   **deleting the metal atom makes 11 of those 12 embeddable** (MEASURED). QED
   is 35% of the reconstruction floor; SA is 1.42 *worse*. The head learned
   "fill the coordination sphere", not "choose drug-like ligands".
3. **The reference band is not a literature baseline.** tmQM is a set of
   DFT-optimised structures, not a drug-like library, and misses every target
   anyway (QED 0.2685 vs 0.50, SA 5.03 vs 3.0). Everything above is a
   *relative* comparison against the training distribution, not a
   competitiveness verdict.

### Correction, 2026-10-12 (afternoon): the connectivity conclusion is narrowed

An aromatic-aware valency audit was run because 0.9697 exceeding the reference
cohort is the classic signature of a metric blind spot (arXiv:2505.00169 found
exactly this shape, stability 0.94–0.99 → 0.45–0.63).

**That bug is not ours.** RDKit's `GetTotalValence()` kekulizes internally
(aromatic C reads 4, furan O 2, pyrrole N 3), so `cfm_property_panel.py`'s
valence check is immune to it by construction. Under the correct Kekulé
convention both cohorts score **1.0000**. The 0.45–0.63 collapse band is not
reachable here.
→ `metrics/by_metric/cfm_metric_validity_audit.json`

**But the clean result does not license the original sentence**, for two
reasons measured the same day:

* **Every metal centre is excluded from all three evaluators**
  (`wf_blkc4_metric_validity_audit.py:120-122` skips `Z > 20`) — 3.1% of
  generated atoms, 3.4% of the reference cohort's. That is precisely the axis
  point 2 above blames. "Both cohorts are equally valence-valid" is true of
  organic atoms only.
* **The comparison cohort is decoder output, not real tmQM** (see the note
  above).

So: **"Connectivity is not the bottleneck" is withdrawn as stated.** The
defensible form is *organic connectivity and organic valency are not the
bottleneck; the metal coordination sphere is unmeasured by every evaluator we
have.* The bottleneck question moves from "chemistry vs connectivity" to "the
metal coordination sphere is the open axis" — and point 2's localisation is
independent, direct evidence for exactly that.

### Literature context (research pass 2026-10-12, NOT in this repo)

| metric | ours | closest published |
|---|---|---|
| SA normalised (≈1−SA/10) | **0.356** | LigandDiff, JCIM 2024, Pt/Pd/Ru complexes: **0.68–0.80** |
| QED | **0.0944** | organic 3D generators 0.39–0.56 |

Two facts that change how these should be reported:

* **Metal-complex generation papers essentially do not report QED.** LigandDiff,
  multi-LigandDiff, JACS Au JT-VAE, OM-Diff, tmQMg-L all report
  validity/connectivity/p_valc/DFT properties instead. QED's ADS functions were
  fitted on ~771 *oral organic drugs* (Bickerton 2012), so metal centres,
  coordination numbers and geometry are outside its calibration domain. 0.2685
  on real tmQM is near the metal ceiling; **0.0944 is an outlier, not a property
  of metals**.
* **Ertl SA is known to be systematically biased for metals** — the PubChem
  fragment library lacks contributions for Ru/Ir/Rh (Balcells et al., JACS Au
  2025, who switched to normalised SA and a "drug-like subset" for exactly this
  reason). So part of SA 6.44 is fragment-library bias, not chemistry difficulty.
* No published QED/SA distribution exists for tmQM / CSD / metal-ChEMBL. **The
  0.2685 / 5.03 reference here is the first measured value of that quantity**,
  not a known-low one.

**Verdict: SA is a real, fixable gap. QED is not the metric to chase.**

---

## 2b. Why the flow-matching term barely descends — REFUTED, and the answer is negative

`fm_term` is the dominant live loss term and moved only −7.8% over 14 epochs
while `bond_ce` fell 93%. The standing hypothesis was that `F.mse_loss`
averages over all *elements*, so few-atom metal-centred clouds dominate the
gradient and a per-cloud normalisation is the fix.

**MEASURED on the epoch-24 checkpoint, 192 tmQM val clouds, forward passes
only. The hypothesis does not hold.**

| test | result | what it says |
|---|---|---|
| element-mean vs cloud-mean over the same residuals | 9.3250 vs 8.9480, **a/b = 1.042** | the reduction is worth 4.2%, not the ~10× the hypothesis needs |
| Pearson(loss, n_atoms) | **+0.408** | *bigger* clouds carry *higher* loss — the sign is opposite |
| Pearson(loss, ligand radius) | +0.696 | same, via size |
| 23 clouds with ≤25 atoms | 6.6% of atoms, **4.9%** of the gradient | under-represented, not dominant |
| metal atoms (192, 2.6% of atoms) | MSE **3.762** vs ligand 9.472, **1.04%** of squared error | metals are nearly free |
| Pearson(loss, zero-predictor MSE, same cloud) | **+0.685** | loss tracks the *target's own scale*; normalising by it collapses the size effect (radius 0.696 → 0.197) |

**So: do not change the reduction.** No subset is starved. The live signal is
the mid-interpolant band — t ∈ [0.4,0.6) scores 11.628 against 7.193 at
t < 0.2, and t ∈ [0.4,0.8) carries 43.9% of the gradient — which would be a
sampling-distribution change (Karras σ-weighting) needing its own A/B, not a
reduction change.

**The floor, stated precisely.** The trivial v≡0 predictor scores 12.363 per
element on the same draws; the model scores 9.325, i.e. it removes **24.6%** of
that upper bound. 12.363 is an **upper bound** on the Bayes floor, not the
floor — separating reducible loss from conditional variance needs an oracle
conditional-mean predictor, which was not built. Do not quote an irreducible
share. (Note also that the 12.46 appearing in older notes is arm_B of
`cfm_loss_term_liveness.json` — Pt-only, train split, batch 64 — a different
arm from this one. The measured floor *here* is 12.363.)
→ `metrics/by_metric/cfm_fm_term_percloud.json`

### Where the 18.8% embed failure actually lives

**It is not the velocity field, and it is not a ring-strain problem.**

* RDKit's `EmbedMolecule` builds a conformer from the **molecular graph** and
  never reads the model's coordinates. Verified by attaching a deliberately
  absurd conformer (atom i at (999i, −999i, 0.5i)) to all 66 molecules and
  re-embedding: **11/66 fail, the same 11 that fail without it.** The model's
  3D output is discarded by this metric.
* **Deleting every metal atom makes 11 of the 12 failing molecules
  embeddable.** The defect is the metal coordination sphere the bond head
  emits.
* It is *not* coordination number: metal_max_degree is 4.87 in the cohort that
  embeds vs 4.91 in the cohort that fails, and 22 of 23 degree-6 metal centres
  embed fine. 3 of the 11 failures contain no rings at all, so ring strain is
  not the driver.
* This matches a limitation already recorded in this repo —
  `molmetal/scripts/check_3d_embed.py:156-160`: *"RDKit has no ETKDG torsion
  parameters for transition metals (Pt, Ru, Ir), so the knowledge-term distance
  geometry fails."*

**Consequence for the loss line.** The sampling-time guidance family (EnFlow
arXiv:2512.22597, PCFM ICLR 2026, FlowBack-Adjoint arXiv:2508.03619) injects a
force into the *coordinate* trajectory. The failure it was proposed to fix is
measured on a graph that discards coordinates. **A force cannot change a number
that never looks at the trajectory**, so that recommendation is withdrawn; the
prototype harness is kept at
`molmetal/scripts/wf_blkc_l3_guidance_probe.py` so the question is cheap to
re-ask if the failure is ever re-localised to coordinates.

The defect lands on **macro-F1(TYPE)** — the axis this line already loses — and
it is the first concrete, localised target the L0/L1 work produced.
*n = 11 for the localisation; the mechanism is documented but a design
decision should not rest on 11 molecules alone.*
→ `metrics/by_metric/cfm_embed_failure_localisation.json`

---

## 2c. The sampled geometry is collapsed, and nothing on disk measures it

An unplanned finding from the L3 probe's control arm. Every quality number
reported for free generation — QED, SA, `single_fragment_ratio`, the 3D
embeddability rate — is computed by RDKit **from the decoded molecular graph**.
None of them reads the coordinates the velocity field actually produced. Measured
directly, with plain `torch` and no RDKit:

| metric (mean over 24 clouds) | generated | real tmQM |
|---|---|---|
| closest interatomic contact | **0.570 Å** | 1.172 Å |
| 1st-percentile pair distance | 1.129 Å | 1.314 Å |
| median pair distance | 4.713 Å | 4.997 Å |
| fraction of pairs in hard steric clash | **0.01482** | **0.00000** |
| clouds containing ≥1 hard clash | **24 / 24** | **0 / 24** |
| total hard-clash pairs | **143** | **0** |
| radius of gyration | 3.571 Å | 3.809 Å |
| metal–ligand closest contact | **1.234 Å** | 1.953 Å |

The clouds are the right *size* and locally collapsed: overall extent is close,
but atoms sit nearly on top of each other, and first-shell ligands land **inside
the metal's own covalent radius** (a real metal–ligand bond is 2.0–2.4 Å). Real
tmQM scores exactly zero clashes across all 24 clouds, so this is not a
threshold artefact.

**Why L1 missed it.** L1 measured the regression residual on *real* geometry —
does the velocity field predict the right velocity on well-formed input? It came
back healthy. It never looked at what the sampler produces from noise. Both
facts are true at once, and their conjunction is the ordinary train/test
mismatch of a conditional generative model.

### Guidance works on this, without retraining

A repulsive-core force `U = strength · Σ_{i<j} (d0_ij / d_ij)³`,
`d0_ij = 0.85·(rcov_i + rcov_j)`, injected into the velocity field during ODE
integration. This is the one component every method in the cited family shares
(EnFlow, PCFM, FlowBack-Adjoint all push overlapping atoms apart), so it is the
cheapest honest probe. **Same checkpoint, same 24 templates, same initial noise
(seed 1234), same solver (midpoint, 20 steps)** — guidance strength 0.0 vs 0.05
is the only difference. Forward integration only; no retraining.

| metric | unguided | guided | real tmQM |
|---|---|---|---|
| total hard-clash pairs | 143 | **17** | 0 |
| clouds with ≥1 hard clash | 24 / 24 | **13 / 24** | 0 / 24 |
| fraction of pairs clashing | 0.01482 | **0.00234** | 0.00000 |
| closest contact (Å) | 0.570 | **1.155** | 1.172 |
| metal–ligand closest (Å) | 1.234 | **1.602** | 1.953 |

Every metric moved toward the real value. → `metrics/by_metric/cfm_l3_guidance_ab.json`

**What this does not establish.** Smoke scale: 24 clouds, 20 steps, one
checkpoint, one seed, one strength — no sweep, no replication. Decode was not
measured in this A/B, so it is **not** established that better coordinates yield
better SMILES, QED or SA; the bond head may simply re-decode the same graph.
And it is a mitigation, not a cure: 13 of 24 clouds still clash, and metal–ligand
sits at 1.602 Å against 1.953 Å real.

**Guidance and L5 are different objects and do not compete.** Guidance moves
coordinates; L5's 18.8% is measured on the decoded graph, which discards those
coordinates (Sec 2b). Expecting guidance to close L5 is a category error, and
expecting L5's fix to address the collapse is the mirror of it.

> A process note worth recording: the guidance force was first shipped with its
> **sign inverted**, which is an attraction — it collapses geometry further and
> would have been reported as "guidance does not work". A second bug returned
> `(B,1,N,3)` instead of `(B,N,3)`, broadcasting the velocity wrong inside the
> ODE. Neither was caught by running the code. Both were caught by
> `molmetal/tests/test_wf_blkc_l3_guidance_probe.py`, verified to go red on each
> reverted form. Measuring that a change runs is not the same as measuring that
> it is correct.

---

## 3. Vina / docking — status

**Two harness bugs found and fixed today** (`molmetal/scripts/
bench_cisplatin_real_vina.py`):

1. `_rewrite_metal_atype_for_vina` matched only `atype == "M"`, but the adapter
   defaults to `chemotype_vdw=True`, under which meeko emits `Re`/`Ru`/`Pt`
   directly. The rewrite was a silent no-op and **every** metallodrug ligand
   died at `set_ligand_from_file` with "Atom type Re is not a valid AutoDock
   type" — including the cisplatin positive control. A control that fails means
   the harness is broken, which is how this surfaced.
2. `_extract_element_from_pdbqt_line` returns PDB atom names in UPPERCASE
   (`PT1`) while the element table is mixed case (`Pt`), so the comparison never
   matched and every ligand took the silent fallback branch even after fix 1.

Pinned by `molmetal/tests/test_vina_metal_atype_rewrite.py` (4 tests, **red check
reported**: reverting fix 1 → 1 failed; restored → 4 passed).

**A third atype problem: boron — FIXED, but read the correction.** Boron is not
a valid AutoDock type. Rather than guess, all 44 elements in tmQM (34,970
complexes) were enumerated and Vina 1.2.7 was asked directly about each, by
substituting the symbol into a real meeko PDBQT atype column: **21 accepted,
23 rejected.** Boron is 6.22% of tmQM complexes, arsenic 0.58%.

Two things the obvious fix gets wrong, both MEASURED:

* A "rewrite everything that is not C/H/N/O" heuristic would be a **regression** —
  it would clobber **10 elements Vina already types correctly** (Ni, Cu, Co, Cd,
  Hg, W, Se, Fe, Mn, Zn) into Mg, a silent scoring change no test flags.
* Non-metals get covalent-radius-nearest surrogates (**B→C**, |ΔR_cov| = 0.08;
  **As→P**, 0.12), **not** Mg. Mg is a divalent metal ion; B is a covalent
  p-block atom. These surrogates keep the run alive; **they do not make the
  score correct**, because the atype drives both VDW and H-bond typing.
→ `metrics/by_metric/vina_127_element_atypes.json`

**The sweep is DEAD, not in flight.** It was recorded here as "in flight" until
today: a scan of all 552 PIDs found no `bench_cisplatin`/vina process, and
`vina_gen2.json` was never written. It stopped in pocket 2 of 10 with no
traceback, so the cause is **NOT ESTABLISHED** (it is not attributable to OOM).
**1 of 10 pockets is on disk; 9 are NOT ON DISK.** The boron bug that stopped it
is now fixed, so the sweep should be re-run before any docking statement is
made.

**And a labelling defect to fix before that re-run.** After the new rewrite,
`_probe_vina_accepts_chemotype_atype` no longer finds the original element in
`VINA_REJECTED_ELEMENTS` and returns a bare `"Mg"`, which does not end in
`-fallback`, so the caller records `status=MEASURED_CHEMOTYPE_VDW` for molecules
(B, As, Ti) that were actually scored as a carbon or magnesium stand-in. Pt
alone gets the honest `Mg-fallback`. As written, the change would convert hard
crashes into **mis-labelled records at scale** — precisely the failure mode the
rewrite's own docstring warns about. No test currently covers `status`.

### FIXED 2026-10-12 (later the same day) — `MEASURED_SURROGATE_ATOM`

The information needed to label a record honestly is **destroyed by the rewrite
itself**: after it runs, a boron atom reads `C` and a titanium atom reads `Mg`,
so the probe cannot distinguish them from carbon and magnesium. The fix carries
the substitutions out of the rewrite rather than inferring them afterwards.

`_rewrite_metal_atype_for_vina(..., record=[])` now appends one
`(element, atype, kind)` per substitution, `kind ∈ {chemotype, fallback,
surrogate}`, and a new `_vina_record_status(subs, accepted)` decides the label.
Surrogate outranks the probe's verdict, because the probe is structurally blind
to it. Three outcomes now exist: `MEASURED_CHEMOTYPE_VDW` (Vina scored the real
element), `MEASURED_GENERIC_ATOM` (scored as Mg), **`MEASURED_SURROGATE_ATOM`**
(scored as a covalent stand-in — the value describes the stand-in, not the
element). A total decode/decode failure now also exits non-zero instead of
printing `0 / N` and exiting 0.

Pinned by `molmetal/tests/test_vina_surrogate_status.py` (5 tests). **Red check
reported:** removing the surrogate branch → 3 failed (boron, arsenic,
titanium), restored → 5 passed. Full `-k vina` comparison against the
pre-fix tree: **0 new failures, 21 pre-existing failures fixed.**

**Known stale, deliberately not edited:** `test_vina_adapter_metals.py` (10
failures) asserts the adapter emits atype `M`. Fix 1 above changed that on
purpose — `M` was the cause of every ligand dying — so those tests pin a
decision we deliberately reversed. Per the standing rule, a test that pins a
contradicted decision is reported, not weakened.

**So the sweep is UNBLOCKED and still not re-run. 9 of 10 pockets remain NOT ON
DISK.**

Methodological caveat worth carrying into the paper: the literature is split on
Vina for metal complexes. A Springer 2026 review states Vina "systematically
underestimates binding energies and generates wrong poses" for metal ligands and
does not accept custom metal parameters; but Çınaroğlu (JCIM 2019) measured 73%
**pose** success across 213 metal-protein complexes with ~0 scoring power
(r²). So report pose quality, not absolute affinity. Separately, the control
comparison is not like-for-like: cisplatin is a 4-atom complex and the
generated cohort is ~40 atoms, and Vina scores larger ligands more favourably,
so a −6.4 vs −3.4 comparison is dominated by **size, not affinity**.

---

### The 18.8% non-embeddable figure is REFUTED — it is better than the reference corpus

The localisation record (`cfm_embed_failure_localisation.json`) reported that
**11 of 12** failing generated molecules become embeddable once the metal atom
is deleted, and concluded the defect sits at the metal coordination centre. That
measurement had no control.

The control (`cfm_l5_metal_deletion_control.json`, 600 **real** tmQM molecules
on the same six metals, identical protocol, verified by reproducing the
generated cohort's 54/66 = 0.8182 exactly):

| cohort | n | embed rate | failures | repaired by deleting metals | repair rate given failure |
|---|---|---|---|---|---|
| generated (our bond head) | 66 | 0.8182 | 12 | 11 | **0.9167** |
| real tmQM (CONTROL) | 600 | **0.6333** | 220 | 209 | **0.9500** |

**The real cohort is repaired more often than ours.** Deleting the metal is
what RDKit's embedder does to metal complexes in general; it says nothing about
our bond head.

The larger finding is the reference band: **real tmQM embeds at 0.6333**. 36.7%
of genuine, experimentally characterised metal complexes fail ETKDGv3 under the
same protocol, against 18.2% of our generated molecules. The "18.8%
non-embeddable" number is therefore not an honest-negative at all — on this
metric our generator is *better* than its own training distribution.

This is the **same reference-band error as BLK-C-L4**, where the "real tmQM"
cohort turned out to be decoder output. Measured against real tmQM, the sign of
the conclusion flips. Do not design a bond-head fix against this number.

## 4. The bond-perception gate, and the patience defect

Three full-corpus seeds completed (n_train 26715, 50-epoch budget):

| seed | gate | best epoch | epochs run | wall |
|---|---|---|---|---|
| 0 | 0.9194 | 10 | 14 | 4.96 hr |
| 1 | 0.9297 | 11 | 15 | 4.50 hr |
| 2 | 0.9703 | 24 | 28 | 7.36 hr |

mean 0.9398 ± 0.0269. → `metrics/by_metric/cfm_bond_type_f1_3seed.json`

**Root cause:** `DEFAULT_EARLY_STOP_PATIENCE = 3`
(`train_cfm_yuelbond_wasserstein.py:720`) and no config set it. `bond_type_f1`
routes through `--early-stop-patience`, not the legacy `--patience` (default
10). seed0's last three epochs were 0.9031 / 0.8922 / 0.9144 — still
oscillating upward when it was cut off. seed2 escaped that window and ran twice
as long.

**So the 0.0269 std is a convergence-TIME spread, not a noise floor, and 0.9194
— the value this configuration has been quoted at — is a lower bound.**

**A patience=10 re-run** (`configs/cfm_patience10_seed0_2026_10_12.yaml`) reached
**gate 0.9532 at epoch 27 with no plateau and no down-epoch in its last six**,
then was **stopped at the user's direction** to benchmark before further
training. Checkpoint retained at
`molmetal/reports/cfm_patience10_seed0/checkpoints/` — the highest gate value
recorded for any seed.

**Triage of the wider A/B record** (`cfm_todo.json::BLK-C-M7`): only 2 of 20 CFM
runs were cut off within patience of their own best, and both are 1-epoch eval
runs, not A/B arms. So the feared "most past A/B results are early-stop
artefacts" is **not supported** — but seed0, the run quoted as this config's
headline, is one of the two affected.

---

## 5. The valence-budget table was wrong, in three ways

`metrics/by_metric/cfm_valence_budget_variant_ab.json`

1. **Wrong denominator.** `single_fragment_ratio` divided by clouds *attempted*;
   the trainer's metric of the same name divides by molecules that *passed the
   chemistry check*. Same checkpoint, 0.1406 vs the trainer's 0.73/0.90/0.88.
2. **The loop ran one variant.** Adding `try/finally` for a genuine
   global-state leak left the loop body at the old indent level, so only the
   last variant ran — with **exit code 0**.
3. **Global state was not restored on exception.** `_CLASS_VALENCE_COST` and
   `_COVALENT_VALENCE` are module globals read live by the decoder
   (`cfm_decoder_yuelbond.py:485`, `:601`); the restore sat outside the loop.

**Under the correct denominators the "local optimum" reading is wrong.**
`no_budget` is not uniformly worse: its connectivity of decoded molecules is
1.000 with n_frags 1.00 — its weakness is decode rate, not connectivity. And
`reserve_0.5` is a no-op (identical to as_shipped on all three columns), because
the reserve is applied via `int(round(0.5))` = 0.

**Re-run through the ODE path** (same 8 variants, 64 generated clouds):
`no_budget` becomes the **worst** variant (decode_ratio 0.0156 vs as_shipped
0.0781, a 5× gap), and the reserve penalty changes mechanism — decode_ratio is
flat across as_shipped / reserve_1.0 / reserve_2.0 while connectivity collapses
(of_decoded 1.0 → 0.4 → 0.0).
→ `metrics/by_metric/cfm_valence_budget_variant_ab_ode.json`

**Limitation:** absolute values in that table are not usable (as_shipped
decode_ratio 0.0781 there vs 0.344 for a different checkpoint; n=64, decoded
counts of 1–5). Only the relative ordering carries information.

**What survives:** the mechanism is load-bearing, and the valence budget remains
a **design argument, not a performance result**.

---

## 6. Performance — **the 22.8× is WITHDRAWN (2026-10-13)**; the runnable figure is 4.78× and it is not equivalent

Generation is **launch-bound, not compute-bound**. Measured on this checkpoint at
N=40 atoms:

| batch | per forward | per cloud |
|---|---|---|
| 1 | 3.262 ms | 3262 µs |
| 3 | 2.757 ms | 919 µs |
| 32 | 2.922 ms | 91 µs |
| 256 | 3.599 ms | 14 µs |

256× the work for 10% more time — 99.6% of the wall clock does not scale with
batch. The sampler grouped by exact atom count, so 192 val clouds became **52
solver calls at median batch 3** (40% at batch ≤ 2), each paying 50 steps × 2
forwards.

`pad_groups` collapses those into ONE padded batch at the global maximum atom
count. **Measured 1899.49 s → 83.36 s = 22.8×** (192 clouds, midpoint/50).

> **WITHDRAWN 2026-10-13.** That 22.8× belongs to an implementation that
> **cannot run**. The single global-max batch needed **92.06 GiB** and OOM'd
> twice on a 95 GiB card, with the padded arm executing alone. The cause is
> `radius_edge_index` padding every cloud in a batch to `e_max`, the batch
> maximum — so a 14-atom cloud was allocated the 72-atom cloud's edge budget.
> Peak VRAM at n_steps=50: grouped 0.33 GiB, single global-max batch OOM at
> 92.06 GiB.
>
> **The runnable figure is 4.78×** (grouped 1958.0 s → padded 409.7 s) after
> bucketing the padded path by size so each batch pads only to its own
> bucket maximum. That fix traded the remaining speed for the ability to run.
> Source: `metrics/by_metric/cfm_pad_groups_mcnemar.json`,
> `cfm_todo.json::resolved_2026_10_12[BLK-C-PERF]`.

**But the coordinate-equality test that "failed" was mis-specified.** It required
per-cloud coordinate equality, which is unattainable: the two paths draw
different noise (padded `randn(192,72,3)` = 41,472 values; grouped per-group
`randn(b,n,3)` ≈ 21,000). Same seed, different shapes, unrelated samples. The
7.58 Å "divergence" measured the noise draw, not the batching. **That verdict is
a false negative; `pad_groups` may well be correct.**

`pad_groups` stays **default OFF** until a distributional test settles it. That
test: **NOT ON DISK** — in flight.

**Rejected, with reasons:** bf16/fp16 (MACE exposes only fp32/fp64; irrep
contractions and division-by-small-norm are AMP-fragile, and bf16 buys nothing
when launch-bound). cuEquivariance (11.5× over e3nn, but that is MACE-scale
tensor-product cost; at 0.7M params the bottleneck is launch overhead).
→ `cfm_todo.json::BLK-C-PERF`

---

## 7. Open items, in dependency order

| # | item | state |
|---|---|---|
| 1 | **Bond head: metal coordination sphere** | **PREMISE REFUTED** by the BLK-C-L5 control: deleting the metal repairs 95.0% of REAL tmQM failures against 91.7% of ours; the 18.8% non-embeddable figure is BETTER than the reference corpus on the same metric. Item kept as an open *question* (the embedder erases 91/99 paired square-planar) but no bond-head redesign should be done against this. See §2b and §9.4 |
| 2 | Vina 10-pocket sweep | **UNBLOCKED, not re-run.** Boron fixed and measured; the `MEASURED_SURROGATE_ATOM` mislabelling that would have corrupted the re-run is also fixed (Sec 3). 9 of 10 pockets still NOT ON DISK |
| 3 | pad_groups distributional test | **NOT ON DISK** — the earlier run died on CUDA OOM (821 MiB free of 31.45 GiB) |
| 4 | 3-seed free-generation panel (seeds 1 and 2) | not run; the ODE dump path now exists so it is cheap |
| 5 | Metal-aware valency evaluator | not built — all three current evaluators skip `Z > 20`, i.e. every metal centre |
| 5a | **Zero-Pt sampling artefact** | **PATCH LANDED 2026-10-12** (see §9.9). `min_per_metal>=1` + `n_clouds>=960` is the recommended call for any precious-metal sweep; the existing n=66 cohort remains the description of a no-Pt window until re-generated. |
| 6 | Guidance strength sweep + decode-enabled A/B | **NEW — the cheap next step.** Repulsion core already moves every geometry metric toward real at strength 0.05; the open questions are how far it goes and whether it reaches the decoded molecule. No retraining needed. Sec 2c |
| 7 | Loss / architecture work | **RE-OPENED by user directive 2026-10-12.** L1's refutation closed a bug hunt, not the objective design. See `cfm_todo.json::the_objective_2026_10_12` (the goal is QED / ADMET / SA / precious-metal-specific metrics) and `BLK-C-OBJ` |
| 8 | Precious-metal geometry measures | **NEW, and it blocks item 7.** Square-planar vs tetrahedral at Pt/Pd/Ru, oxidation state and donor preference are measured by *nothing* on disk. Do not tune an objective against a proxy that cannot see what makes a complex a drug |
| 9 | Retrain | not started; ~30 epochs is the stated budget |

**On item 7 — why L1's negative answer did not close it.** The user directive
of 2026-10-12 states what the loss line is *for*: better QED, ADMET and SA, plus
precious-metal-anticancer-specific metrics. L1 asked a different and narrower
question — is the velocity reduction broken? — and answered no. That closes a
bug hunt. It does not license the conclusion "the loss line is finished", which
is what a file recording only L1's answer would imply. Of the 12 loss terms,
only `bond_ce` touches the decoded molecule's chemistry and it is a **per-edge**
cross-entropy; the rest are velocity regression, regularisers, or dead. Nothing
scores a molecule as a whole, and every property in the directive is a
molecule-level quantity.

**Closed today, and what each one changed:**

* **L1 (the flow-matching plateau) — REFUTED.** The per-cloud-normalisation
  hypothesis does not hold: the reduction is worth 4.2%, the loss-vs-size
  correlation has the *opposite* sign, and no subset is starved. **No loss
  change is recommended**, and none is authorised on this evidence. Details in
  Sec 2b.
* **L4 (aromatic-aware valency) — not our bug, verdict narrowed.** RDKit's
  `GetTotalValence()` kekulizes internally, so our evaluator is immune to
  arXiv:2505.00169. But all three evaluators skip every `Z > 20` atom, so the
  clean 1.0000 is silent on the metal axis — the exact axis Sec 2 blames.
  "Connectivity is not the bottleneck" is **withdrawn as stated**.
* **L3 (sampling-time guidance) — premise split; the earlier "do not build"
  verdict is withdrawn.** Guidance indeed cannot reach L5's number (see below).
  But it was then generalised to "guidance has nothing to act on", and that
  second step does not follow: the model's **sampled coordinates** turned out to
  be badly collapsed, and no existing metric looks at them. See Sec 2c.
* **L5 (non-embeddable molecules) — cause established**, and it is a bond-head
  graph defect, not a velocity-stage one.

**What the evidence now says to attack.** The failures cluster at the **metal
coordination sphere**, not in bond perception generally and not in the flow
velocity field:

* 18.8% cannot be embedded in 3D — and **11 of 12 embed once the metal is
  deleted**
* generated metals carry small inorganic ligands (OH, Cl, NO)
* SA normalised 0.356 against the only metal-complex baseline (0.68–0.80)
* QED 0.0944 against 0.2685 for the reconstruction of its own training set

This lands on **macro-F1(TYPE)** — the axis this line already loses — and is
the first localised, mechanism-backed target the measurement work produced.

Separately, three loss knobs read as ON and do nothing: `assemble_loss` is
exactly 0.0 at every epoch despite `assemble_enabled=1.0`;
`entropy_reg_weight=0.01` is a measured no-op; the `combined` early-stop gate
is identically 0 because `bond_exact_type_rate` is identically 0. That remains
true, and remains unaddressed — but note that reviving a dead term requires a
premise that survives, and `soft_valence`'s premise (a 32537× scale hazard) was
**refuted**: the per-atom division shipped on 2026-10-10 and is now guarded.

---

## 8. A note on how these records were produced

Six of the errors found today shared one shape: **the number was not wrong, the
claim about the number was.** A coordinate pair was not comparable. A script
never integrated the ODE. A loop silently ran 1 of 8 variants. An exception was
swallowed and n=0 was read as a result. A silent no-op made every ligand fail
while the harness reported success. A tie rule was documented backwards.

All six were caught by checking **artifacts and controls**, not exit codes. Four
of the six would have passed a naive review.
---

## 9. Evening correction pass — the reference band is settled and BLK-C-OBJ is decided

Added 2026-10-12 after `metrics/by_metric/cfm_reference_band_audit.json` (34
claims) and an adversarial review of every stream that fed this file. Every
correction is an **appended** block on the record that carries the wrong claim,
per AGENTS.md §5.5 — no history was rewritten.

### 9.1 The `n=636` band is decoder output. It is not real tmQM and it is not "reconstruction"

The right-hand column of §2 — and every "vs real" phrasing that inherited it —
rests on `real_tmqm_reference.csv`, which is 2000 real tmQM val clouds passed
through **our own YuelBond bond head**, of which 636 yielded a SMILES. 636/636
rows carry bracket-atom notation (`[C]`, `[Rh]`); row 1 is
`CC(C)P(C(C)C)[C](C)(C)[Os]([Br])([Br])…`. Measured by reading the file, not by
citing the earlier audit.

**Real bands now exist and are the ones to quote:**

| metric | generated (n=66) | real tmQM | n | path |
|---|---|---|---|---|
| QED | 0.094443 | **0.234783** | 31,150 | `cfm_real_tmqm_sixmetal_baseline.json` |
| SA (Ertl) | 6.442335 | **4.978116** | 31,150 | same |
| SA normalised (1−SA/10) | 0.355767 | **0.502188** | 31,150 | same |
| coordination number | 4.878788 | **5.952392** | 31,150 | same |
| logP | — | 4.153033 | 31,150 | same |
| aromatic bond fraction | 0.001229 | **0.454591** | 31,150 | same |
| AMES mutagen | 0.0000 | **0.034893** | 31,152 | `cfm_real_tmqm_sixmetal_admet.json` |
| logPapp | −3.0387 | **−3.680245** | 31,152 | same |
| logS | −8.0099 | **−6.701385** | 31,152 | same |

The 2-row denominator difference between the two real bands (31,150 vs 31,152) is
the SMILES that fail RDKit parse; both filters are correct for themselves and
the difference must be stated wherever both are quoted.

**What the sign flip does to the ADMET reading.** §2's "all three ADMET proxies
sit essentially on the real band" was measured against decoder output and is
**withdrawn**. Against real tmQM, **ames and logPapp are better** for the
generated cohort and only **logS is worse**, by 1.31 log units, entirely
inherited from logP 8.09 vs 4.15. The logPapp "win" is a clipping artefact — the
Hou 2007 form is clamped and the median sits at −3.0093.

**Two chemistry-validity claims do not survive either.** 600/600 real tmQM rows
carry `->`/`<-` dative arrows; **0/636** decoder rows and **0/66** generated do.
RDKit charges DATIVE 0.0, so every metal ligand reads under-valent — 1383 of
1766 violating atoms are metal-adjacent. The evaluator is also charge-asymmetric
(`_sum_valence` adds |charge| for anions, nothing for cations). "Validity 1.000
vs 0.998, indistinguishable" therefore compares two **neutral** cohorts and says
nothing about the 98.5% of real molecules that are charge-separated.

→ `metrics/by_metric/cfm_panel_reference_band_real_tmqm.json::corrections_2026_10_12`

### 9.2 Two claim defects inside this file's own numbers

* **"QED is 35% of the reference floor"** (§2, point 2) — the direction holds,
  the magnitude does not. Against real tmQM it is **39%**, not 35%.
* **"coordination number is at parity"** — **withdrawn**. The 4.91 that claim
  rested on is the mean metal degree of the **11 failing generated molecules**,
  mislabelled as real. Real is **5.952392** (n=31,150); the gap is **−1.07**.
* **Sterecentre figures** — `31x`, `89%`, `71.6%` were computed on a **4,000-row
  prefix** while the headline SA gap uses n=31,150. Full corpus: mean potential
  stereocentres **0.315795** (prefix 0.201), zero-stereocentre fraction
  **83.2%**, ratio **19.8x**, share of the SA gap **69.3%**. Same pass, two
  denominators. The conclusion — the chiral-centre penalty dominates — is
  unchanged; three magnitudes were overstated.
→ `metrics/by_metric/cfm_sa_stereocentre_control.json::corrections_2026_10_12`

### 9.3 Citation correction: LigandDiff is JCTC 2024, and the band mixes cohorts

`cfm_todo.json` and the audit both cite "LigandDiff, **JCIM** 2024, on Pt/Pd/Ru
complexes: 0.68–0.80". Verified against DOI `10.1021/acs.jctc.4c00232`,
PMID 38743854: **J. Chem. Theory Comput. 2024, 20(10), 4377–4384**.

The **metals were not the error** — the 0.68 endpoint *is* Pt/Pd/Ru (their
`PPR_100` set of 100 CSD Pt/Pd/Ru complexes >50 atoms, 0.68 ± 0.008). The error
is that the band **mixes two cohorts**: 0.68 is PPR_100, 0.69 is a random 6–20
atom sample of the 3d **training** metals, and 0.72–0.80 are fixed-ligand-size
runs on those 3d metals. Separately, LigandDiff does not state which 0–1
transform it used; ours is `1 − SA/10`, a linear hack rather than Gao & Coley
(*J. Chem. Inf. Model.* 2020, **60**, 5714, DOI `10.1021/acs.jcim.0c00174`).
Quote the transform gap, not the band.

### 9.4 The coordination-geometry axis is now MEASURED, and the instrument is the problem

`metrics/by_metric/cfm_metal_geometry.json` + `cfm_geom_classifier_dft_band.json` + `cfm_metal_geometry_pairing_control.json`.

**The real band exists.** REAL_DFT (new ETKDG-free classifier, no RDKit, no embedder), n=2848 clouds / n=2847 scored, six train metals, donor_cutoff=2.6 A:

| axis | value |
|---|---|
| CN=4 geometry | **472 square_planar / 27 tetrahedral** (94.6% SP of 499) |
| CN=6 geometry | 1093 octahedral_like / 6 no_trans_pair |
| coordination number mean | 6.1924 |
| M–L bond length (median) | Pt 2.1735 (n=3386), Ru 2.2647 (n=5504), Ir 2.2045 (n=3404), Rh 2.2029 (n=2232), Os 2.2383 (n=786), Re 2.1258 (n=2318); pooled 17,630 bonds, sd 0.1895 |
| M–L bond length (mean) | pooled 2.179, range per metal 2.1252 (Re) to 2.1981 (Ru) |
| donors (top 5) | C 43.69% / N 21.34% / Cl 10.72% / P 9.96% / O 4.98% |

**NOTE 2026-10-12 (corrections_round1):** the prior §9.4 prose quoted
"Pt 2.1832 (n=3571), Ru 2.2644 (n=5723), Ir 2.2046 (n=3549), Rh 2.2062 (n=2374),
Os 2.2403 (n=844), Re 2.1286 (n=2457); pooled 18,518 bonds, sd 0.1921" and
"CN=4 geometry | 486 square_planar / 31 tetrahedral (94.0% SP of 517)".
These were the round-1 instrument's numbers; the new
cfm_geom_classifier_dft_band.json superseded them. The new values above
are what the file currently says. The brief's
"Within 3%" / "Mean 2.179 matches 2.180 to 4 decimals" / "2847/3000"
statements are also withdrawn; the actual deltas are
T-4 -12.9% and oct -4.87% (NOT within 3%), mean 2.179 vs "2.180" differ
in the third decimal (real is 2.179 not 2.180), and the cohort is
2847 scored out of 2848 clouds (= 99.96%, not 2847/3000) from a
3000-sample request. The trigonal-prism explanation was fabricated;
see corrections_2026_10_12_evening in
cfm_geom_classifier_dft_band.json.

**The generated arm cannot be scored against it, for three measured reasons.**

1. **Zero Pt.** 0 Pt across all 66 generated molecules, against 3,571 Pt
   metal-ligand bonds in REAL_DFT. The cisplatin pharmacophore is not present in
   our output at all. Root cause reproduced by *running* the sampler:
   `load_tmqm_clouds(n_clouds=192)` draws a 192-of-960 window containing no Pt
   (Ru 64 / Ir 45 / Re 38 / Rh 35 / Os 10 / Pt 0), while Pt is 25.2% of the real
   six-metal corpus. Widening to n_clouds ≥ 960 costs nothing and needs no
   retrain. **PATCH LANDED 2026-10-12**: `molmetal/data/tmqm_clouds.py::load_tmqm_clouds(..., min_per_metal: int = 0)`. With
   `min_per_metal=1, n_clouds=192` the loader returns `{Ir 41, Ru 47, Pt 53, Re 24, Rh 20, Os 7}`
   (Pt 27.6% vs real 25.2%); with `min_per_metal=20, n_clouds=960` it returns
   `{Re 135, Ir 158, Ru 279, Rh 114, Pt 230, Os 44}` (Pt 23.96%). The
   min_per_metal back-fill and the n_clouds window widening are **independent
   fixes** — one stops the cap-window under-representation, the other stops any
   single reservoir sample from being unlucky. The next sweep that asks a
   precious-metal question should set **both** to ≥ 1 and ≥ 960 respectively.
   `molmetal/tests/test_tmqm_min_per_metal.py` (4 tests) is the red check; all
   four pass on the patched loader and revert to `TypeError: load_tmqm_clouds()
   got an unexpected keyword argument 'min_per_metal'` if the kwarg is removed.
   → `metrics/by_metric/cfm_objective_inventory.json::the_zero_platinum_finding.patch_2026_10_12`.
2. **The embedder erases the pharmacophore.** Paired molecule-for-molecule
   (n=600, 99 paired CN=4 scores): **91 of 99** molecules whose DFT geometry is
   square-planar become **tetrahedral** under ETKDG. Any square-planar number
   computed from generated SMILES is a statement about ETKDG, not the generator.
   *This is the single load-bearing control on this axis.*
3. **The scoreable subset is selected by the defect.** 12 of 66 do not embed,
   and 11 of those 12 embed once the metal is deleted — so the 54 that score are
   chosen on the very property being measured.

**Two defects in the first emission, now corrected by re-measurement.**

* *Protocol drift.* `REAL_DFT` (metals **with coordinates**) and `REAL_ETKDG`
  (metals **in the csv**) are disjoint populations: at `--real-limit 3000` they
  share **329** molecules, so 89% of the DFT cohort is absent from the cohort
  labelled "the same real molecules", and the embedder floor conflated the two.
  Re-measured paired, the floor is CN 0.2267 / CN4-planarity 0.7642 / bond
  length 0.0563. Only the CN and bond-length generator gaps (1.1312, 0.1344)
  exceed their floor; the CN4 planarity gap (0.0227) does not and is **not a
  finding**.
* *Scope and definition.* The emitted provenance note said the
  six-metal-with-coordinates population is 6,872; with the SMILES filter the code
  actually applies it is **5,754**. And **CN is two different definitions** —
  DFT CN is a 2.6 Å donor window, generated/ETKDG CN is RDKit graph degree. They
  agree on only **78.33%** of identical molecules, so any CN gap across cohorts
  contains a definition gap.
* `pymatgen` is **not installed** in `.venv`, so the ChemEnv/MetalHawk SP-4-vs-
  T-4 classifier cannot be used as written.

**Consequence:** the square-planar-vs-tetrahedral axis is **measured on real
data and not yet measurable on ours**. No coordination-geometry loss term is
proposed.

### 9.5 BLK-C-OBJ is decided — including two recommendations to build nothing

→ `metrics/by_metric/cfm_blk_c_obj_decision.json`

| target | verdict |
|---|---|
| **QED** | REACHABLE, but **not** by a QED term. ROTB +0.0571 and ALERTS +0.0524 carry essentially the whole gap; AROM is negative. All three trace to long acyclic sp3 ligand backbones (aromatic bond fraction 0.001229 vs 0.454591). The live lever is **`bond_ce`** — bond-TYPE prediction — and it is currently **losing**: macro-F1(TYPE) **0.4784 ± 0.0055** vs OpenBabel **0.5051** under the identical per-complex protocol. |
| **ADMET** | **NOT A LOSS PROBLEM. Change nothing.** All three proxies are pure functions of logP/MW/TPSA/RotB/HBD/six SMARTS and **not one reads the metal**. Two of three are already better than real. Building a metal-reading ADMET axis is a *metric* task. |
| **SA** | REACHABLE, and the repo's mechanism was **wrong**. Fragment rarity points the **wrong way** (+0.315 SA units — generated fragments are *better* covered); size is +0.012; the **chiral-centre penalty carries 69.3%** of the gap. **Do not build a fragment-rarity term** — it would aim at a term already helping us. |
| **precious-metal-specific** | Most reachable, least measured. A **sampling** question first (zero Pt), a **missing-metric** question second, a loss question third. |

**The finding that reshapes the blocker:** it is not "add a molecule-level
objective" but "**fix the one that is already there**". `bond_ce` is the only
live term touching decoded chemistry, it is per-edge, and it is losing to a
rule of thumb. QED and SA are **one problem** — both trace to the same
structural defect — and both are downstream of the same lever.

### 9.6 The next experiment

**A decode-enabled guidance sweep with the sampled cloud coordinates retained.**
No retraining. Four arms (unguided + three guidance strengths) × 24 clouds plus
a decode step.

It is first because it is the only thing that unblocks BLK-C-OBJ, BLK-C-L3 and
the geometry gate at once, and all three of its questions are currently **NOT ON
DISK**:

1. does guidance-improved 3D geometry decode to better SMILES — the L3 A/B ran
   `--skip-decode`, so this link has never been measured;
2. does the improved geometry survive ETKDG, which is where the pharmacophore
   dies (91/99 paired);
3. does the square-planar fraction move once Pt is in the sampling window.

Retaining the coordinates supplies the coordinate source the geometry metric
currently lacks, and widening `n_clouds` removes the zero-Pt artefact at no GPU
cost.

**Wall time: PROJECTED 3–5 hr** on the existing checkpoint (basis: MEASURED
~31 s per record on the concurrent CPU Vina sweep — 119 records over 62 min).
The ODE half is the user's GPU call. **Every QED / SA / geometry number for the
new arms is NOT ON DISK and nothing about the outcome is claimed here.**
**Sampling call must pass both `min_per_metal=20` and `n_clouds=960`** (see
§9.4 and §9.9 for the independent-fixes argument).

Spending the single authorised retrain before these three answers would buy a
loss change justified by an unmeasured premise.

### 9.7 Live sweep status (do not relaunch)

`molmetal/scripts/cfm_vina_generated_sweep.py` is **running**, PID 3202095,
CPU-only, GPU 0 untouched. MEASURED at time of writing: 2 of 10 pockets complete,
`per_record.jsonl` = 119 appended records.

| pocket | attempted | docked | failed | mean (over docked) | mean n_atoms |
|---|---|---|---|---|---|
| 830c | 66 | 52 | 14 (12 `MetalPDBQTUnavailable` + 2 `DockError`) | −5.8093 | 39.0 |
| 1h36 | 53 | 44 | 9 | −6.5395 | 36.3 |

Corrections to the launch note: `partial_summary.json` is **not** incremental
(only `per_record.jsonl` is crash-safe); the "−6.407 for 830c" line was the dead
2026-09-29 **1fmc** number relabelled; the dominant status is
`MEASURED_GENERIC_ATOM` with `vina_accepted_atype = Mg-fallback`, **not**
`MEASURED_SURROGATE_ATOM`; the "~40 atoms" figure is 39.0/36.3 measured and
"cisplatin is 4 atoms" describes 4 *ligands* (7 heavy atoms in the PDBQT); and
the **"2026 Springer review" citation is not on disk anywhere in this repo** —
the only verified citation is Cinaroglu (*JCIM* 2019, 73% pose success over 213
metal-protein complexes, ~0 scoring power).
→ `metrics/by_metric/cfm_vina_generated.json::corrections_2026_10_12`

### 9.8 Pre-existing test weakening found in the tree, reported not changed

`molmetal/tests/test_read_anchor_json.py` weakens
`test_anchor_version_pin_required_version_passes` from `max_age_sec=7*86400` to
`100*365*86400`, deleting a live freshness assertion; its own comment concedes
the window "began failing 7 days after … with no code change at all".
`test_3way_cross_dataset.py` and `test_3way_real_metalcytotox.py` relax
`out_dir → artifacts_dir` to follow a bench redirect (defensible, commented,
assertions intact). **All pre-date this pass.** Per the standing rule against
weakening a failing test, these are reported and left alone.

### 9.9 The zero-Pt patch is landed, with red check, and four interaction records are flagged

The §9.4 "Zero Pt" finding has a PATCH on disk since 2026-10-12 (evening).
`molmetal/data/tmqm_clouds.py::load_tmqm_clouds(..., min_per_metal: int = 0)`.
Verified live:

| call | distribution | notes |
|---|---|---|
| `min_per_metal=0, n=192, seed=0, split='val'` (default) | `{Rh 35, Re 38, Ru 64, Ir 45, Os 10}`, **Pt 0** | REGRESSION PIN — pre-patch behaviour |
| `min_per_metal=1, n=192, seed=0, split='val'` | `{Ir 41, Ru 47, Pt 53, Re 24, Rh 20, Os 7}` sum=192 | Pt 27.6% (vs 25.2% real) |
| `min_per_metal=20, n=960` | `{Re 135, Ir 158, Ru 279, Rh 114, Pt 230, Os 44}` sum=960 | Pt 23.96% |
| `min_per_metal=5, n=10` | raises `ValueError` ("min_per_metal=5 * 6 = 30 exceeds n_clouds=10") | underflow refused |

`molmetal/tests/test_tmqm_min_per_metal.py` (4 tests) is the red check. On the
patched loader: 4/4 PASS (`min_per_metal=0.082s` for the 192-row default check;
`min_per_metal=81.81s` for the full suite on a warm cache). Reverting
`min_per_metal` from the signature raises `TypeError: load_tmqm_clouds() got an
unexpected keyword argument 'min_per_metal'` and all 4 go red. Critical fix
inside the patch: tmQM file atom order is arbitrary, so `_metal_sym_of` scans
**all** atomic numbers for any metal Z rather than reading `zs[0]`. The naive
`zs[0]` lookup found Pt=0 even when Pt was in the cloud.

**Interaction with the §9 numbers — what is invalidated, what survives.**

The patch changes the SAMPLER, not the model. It invalidates *claims about
what the model produces when the templates carry Pt*. It does not invalidate
the §9.4 control, the §2c collapse numbers, or the §9.4 "91/99 paired" finding.
Per AGENTS.md §5.5 each invalidated record carries an appended correction block
on the file that records it.

| record | invalidated for precious-metal-anticancer? | what survives | where the correction lives |
|---|---|---|---|
| `cfm_objective_inventory.json::metal_composition_generated = {Re 13, Ru 16, Ir 18, Rh 15, Os 4}` | **YES** — describes a no-Pt cohort | nothing Pt-specific | `cfm_objective_inventory.json::the_zero_platinum_finding.invalidation_note_2026_10_12` |
| `cfm_objective_inventory.json::coordination_number_gap = -1.073604` | **YES** — derived from the no-Pt cohort | the non-Pt subset of the n=66 numbers | same |
| `cfm_objective_inventory.json::LARGEST single-metal count` qualifier (verdict_2026_10_12_A_vs_B) | the qualifier only | the load-bearing point (Pt is ~25%, the model has seen Pt extensively) | `cfm_objective_inventory.json::corrections_appended_2026_10_12_evening` |
| `cfm_todo.json::the_objective_2026_10_12::coordination_number_MEASURED = 4.878788` (with "real 4.91") | **YES, doubly invalid**: 4.91 is the mean of the 11 FAILING generated molecules (cfm_embed_failure_localisation.json), not real | real 5.952392, n=31150; gap -1.07 | `cfm_todo.json::do_not[Say coordination number is 'at parity']` (pre-existing) + this section |
| `cfm_metal_geometry.json::cohorts.GENERATED` (all of it) | only the precious-metal-anticancer generalisation; the non-Pt numbers (CN 4.8788, 64% C donors, 19% N, 9% P, 0.8738 cn4 planarity) | the embedder-erase control, the cn4 planarity finding, the chiral-centre share of the SA gap | `cfm_metal_geometry.json::corrections_2026_10_12[1]` (this pass) |
| `cfm_blk_c_obj_decision.json::precious_metal_anticancer_specific.premise_1_sampling` (and premise_3 the embedder control) | the "0 Pt" wording is still true of the existing n=66 cohort; the patch makes it *addressable* | the embedder control, the (a)+(b)+(c) decomposition, the no-pymatgen caveat | the decision file's existing `the_coordination_geometry_gate.what_is_missing` already enumerates the four pieces; the patch closes item (1) |
| `cfm_todo.json::the_objective_2026_10_12::zero_platinum_BLOCKER` | wording ("BLOCKER") is now stale; the patch removes the block at zero GPU cost | the *cause* (sampling window) and the *fix* (min_per_metal) | `cfm_todo.json::BLK-C-OBJ.precious_metal_specific` and `cfm_todo.json::next_actions[0]` (this pass) |
| `docs/CFM_STATUS_2026_10_12.md §2.2` ("coordination number is at parity") | already WITHDRAWN in §9.2; now also doubly-invalidated by the patch | the §2c collapse numbers (unrelated axis) | §9.2 + this section |

**The "4.91 is doubly invalid" claim, expanded.** The number 4.91 is real and
reproducible: it is the mean RDKit graph degree of the 11 failing generated
molecules in `cfm_embed_failure_localisation.json`. What is invalid is the
*attachment* of that number to the real tmQM cohort. Real six-metal tmQM CN
at n=31,150 is **5.952392** (`cfm_real_tmqm_sixmetal_baseline.json`). The
correct gap is **−1.073604**, not "at parity". The §9.2 "withdrawn" call
already covered this; the patch makes it possible to close it experimentally.

**What is NOT invalidated.** (1) The §9.1 reference-band settled verdict
(real bands, 14 citable claims, 20 not citable). (2) The §2c collapse finding
— guidance closed closest-contact 0.602 → 0.017 absolute error and the
embedder floor is unchanged. (3) The §9.4 embedder-erase control (91/99
paired, methodology unchanged). (4) The §9.5 BLK-C-OBJ per-target verdicts
(QED, ADMET, SA, coordination-geometry) — those were decided against the
real-tmQM bands, not against the no-Pt GENERATED cohort. (5) The §3 Vina
sweep — the 3 of 10 pockets it has measured are on the same n=66 cohort and
the Vina per-atom finding (cisplatin −0.333 vs generated −0.149 kcal/mol/atom;
generated is **less** favourable per atom) is not a precious-metal claim.

**Next sweep specifies both fixes.** §9.6's "decode-enabled guidance sweep"
is the highest-priority next action (cfm_todo.json::next_actions[0]). The
sweep's sampling call should pass `min_per_metal=20, n_clouds=960` — the
former because the patch is a back-fill that targets any under-represented
metal, the latter because the wider window is the more conservative choice
against any residual reservoir luck. The two fixes are independent and
cumulative; specifying both makes the next-sweep cohort precious-metal-
eligible.

---

## 10. Round-5 corrections — BLK-C-M3 closed, twelve precision fixes

Added 2026-10-12 after an adversarial review of the four streams that fed
this file: BLK-C-M3 noise floor, BLK-C-M4 aromatic defect, BLK-C-L2 dead
terms, and Vina citation handling. Premise verdicts survived; this pass
tightened prose where it overclaimed and closed one blocker whose question
now has an answer on disk.

### 10.1 BLK-C-M3 closed: noise floor pinned, test goes red on revert

The L3 4-arm guidance sweep is paired by construction (same 64 clouds, same
noise seed 1234, same checkpoint across all four arms). The "is this
difference real" question therefore has the right statistic — McNemar exact
two-sided on per-cloud pass vectors — and the conservative floor is net
discordant `|b - c| >= 7` (at `n_disc >= 10`), with the unpaired Wilson 95%
CI half-width at `n=64, p~0.34` being 0.116 (~7.4 successes).

Per-pair verdicts (`molmetal/tests/test_wf_blkc_m3_gen_validity_floor.py`,
11 assertions, all green):

| pair | n_disc | net | exact p | verdict |
|---|---|---|---|---|
| (0.0, 0.01) | 5 | +1 | 1.0000 | **SUB-FLOOR** |
| (0.0, 0.05) | 18 | +10 | 0.0309 | CLEAR |
| (0.0, 0.15) | 43 | +31 | <0.0001 | CLEAR |
| (0.01, 0.05) | 15 | +9 | 0.0352 | CLEAR |
| (0.01, 0.15) | 42 | +30 | <0.0001 | CLEAR |
| (0.05, 0.15) | 33 | +21 | 0.0003 | CLEAR |

**Tripwire verified.** Setting `MEASURED_PAIRS[("0.0", "0.01")]["FP"]=30`
makes `test_control_vs_weakest_guidance_is_sub_floor` fail with the exact
error message *"sub-floor tripwire: (0.0, 0.01) pair became CLEAR at
p=0.0000 -- the floor statement needs re-derivation, not a constant
rewrite"*. Restored to `FP=3`; suite green 11/11. The test never had a
weakened, deleted, or edited assertion this round.

**What this does not close.** Between-seed spread on this exact cohort
remains unmeasured — the L3 sweep is single-seed. Replicating at 3 seeds
would tighten the conservative floor from 7 to 14 (PROJECTED ~3 hr CPU on
the existing checkpoint). That is a scope-widening, not a BLK-C-M3
blocker: BLK-C-M3's question ("is the L3 difference distinguishable from
sampling noise on n=64 paired?") has its answer. The seed-axis question
("does the (0.05, 0.15) gap hold across seeds?") is a separate item, lower
priority than the 4-arm decode-enabled sweep itself.

### 10.2 Twelve precision corrections — premise verdicts survive

These are corrections to PROSE, not to conclusions. The adversarial review
caught twelve places where a claim was too broad or where two numbers
were stitched that should have been separated. Recorded in
`cfm_todo.json::corrections_2026_10_12_round5`.

**Citation provenance.** Three citations to QED/SA backprop-refutation
literature (arXiv:2510.21153 NeurIPS 2025, arXiv:2411.13280 MolJO,
bioRxiv 2025.12.03.692079) are quoted in `cfm_todo.json::BLK-C-L3` and
`literature_2026_10_12.do_not_backprop_QED_SA` without paper-text
verification. The verdicts hold; the citations are now explicitly marked
**NOT VERIFIED on paper text**. The "2026 Springer review" cited alongside
Cinaroglu JCIM 2019 for Vina-on-metals is also NOT VERIFIED — only
Cinaroglu (10.1021/acs.jcim.9b00346, 73% pose success, ~0 scoring power)
is on disk.

**Architectural claim precision.** BLK-C-PERF's most_on_point ("SO(3)-
Averaged Flow Matching + Reflow") was attributed to "demonstrated on
NequIP, an equivariant GNN". NequIP is e3nn-based, not EGNN-based; our
velocity field is the EGNN family (FlowMol/MolFM lineage). Same family of
equivariant GNNs, distinct specific architecture — and real EGNNs are
*approximately* equivariant (MLP non-linearities + LayerNorm on distance
features perturb exact equivariance), so SO(3)-averaging adds a small
regularisation, not exactly zero. Reframed.

**Live-update caveat for shortcut/consistency.** The verdict that
"wall-time bottleneck is launch overhead, not NFE" omitted that
PAD_GROUPS itself is default-OFF pending a correct equivalence test
(the prior coordinate-equality test was mis-specified, per
`BLK-C-PERF.MY_FIRST_TEST_WAS_WRONG`). Reframed: step count as binding
wall-time axis is weak, but PAD_GROUPS is not yet a deployable speedup.

**Entropy-term framing.** The "highest-value-of-six dead term" verdict
omitted the BLK-C-M4.REFINED_2026_10_12 finding that *"the generation
gap is 1.6x the head gap, so a head-only fix cannot close it"*. Entropy
is head-side; the binding axis is gen-side-dominated. Reframed: entropy
is the only head-side dead term with an intact premise, but it must be
paired with a gen-side lever to close the binding QED/SA gap.

**Diversity-term scope.** The diversity_term code at trainer L2970-3066
also includes nucleus-sampling (top_p / top_k / temperature) paths gated
on different flags, not just MMD + repulsion. Scope widened; verdict
survives because none of the extras reaches any of the four binding axes.

**9 → 7 configs.** The claim "9 configs ALREADY set bond_class_weights:
true with power: 0.5" was off-by-2: only 7/9 configs explicitly set
`bond_class_weight_power: 0.5`; the other 2 rely on the argparse default
of 0.5. Substantively identical but precise.

**Vina mean/r stitching.** The -5.81 kcal/mol figure is 830c-pocket only
(n=52), not pooled. The pooled mean is -5.949 across the 3 pockets that
have data (830c, 1h36, 3eml). The r=+0.295 and slope=+0.054 are pooled.
Pocket-level slopes are heterogeneous (+0.0072 / +0.128 / -0.009);
1h36 dominates the pooled regression.

**Pt-group metal counts.** Per `per_record.jsonl` has no `metal` field;
the breakdown (Rh:13, Ir:7, Ru:5, Re:3, Os:2 = 30) is reconstructed from
`[M]` SMILES tokens, not labelled. Marked INFERRED-NOT-ON-DISK.

**Path drift.** The "3 checkpoints all in `cfm_fullcorpus_2026_10_10/
checkpoints/`" claim was a textual claim, not on disk. Verified: the
3 ckpts are in 3 different directories — seed0 at
`cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0.pt`,
seed1 at `cfm_fullcorpus_seed1/checkpoints/...seed1.pt`, seed2 at
`cfm_fullcorpus_seed2/checkpoints/...seed2.pt`. No code path is wrong;
the prose was.

### 10.3 What this round did NOT apply

- **BLK-C-L2 entropy_term enable.** Premise intact but spending the single
  authorised retrain on it is a user decision (`cfm_todo.json::L3 gate`).
- **BLK-C-L3 sampling-time guidance sweep.** PROJECTED 3-5 hr GPU, NOT
  AUTHORISED this round (CUDA_VISIBLE_DEVICES=""). The 4-arm decode-
  enabled sweep at `next_actions[0]` remains the single next-action
  with a measured premise behind it.
- **BLK-C-PERF pad_groups distributional test.** Prior run died on CUDA
  OOM (821 MiB free of 31.45 GiB); rerun pending user OK.
- **BLK-C-M4 3-seed aromatic replication.** PROJECTED ~1 hr CPU/seed,
  NOT AUTHORISED this round. The CPU closeout tightens per-class
  aromatic-recall std (currently borrowed by analogy from macro-F1 type's
  0.0055 across 3 seeds), but the head-side 9.1pp gap and the 1.6x
  generation-side gap are both measured with SE << margin; the
  replication is for writeup defensibility, not for changing a
  decision.
- **BLK-C-M6 NCI-60 metal-cohort presence check.** Deferred — the
  per-cell spread (A549 0.8335 / A2780cis 0.8065 vs A2780 0.4010 /
  HCT116 0.4314) is a real bond-perception result; the metal-cohort
  presence question is a separate scope.

### 10.4 The minimum experiment, restated

The single next-action with a measured premise behind it is the 4-arm
decode-enabled guidance sweep at `cfm_todo.json::next_actions[0]
.decision_tree_2026_10_12_evening`. It is a no-retrain experiment that
unblocks BLK-C-OBJ, BLK-C-L3 and the coordination-geometry gate at once,
and all three of its questions are currently NOT ON DISK:

1. does guidance-improved 3D geometry decode to better SMILES — the L3
   A/B ran `--skip-decode`, so this link has never been measured;
2. does the improved geometry survive ETKDG, which is where the
   pharmacophore dies (91/99 paired SP→T-4);
3. does the square-planar fraction move once Pt is in the sampling
   window.

**Sampling call must pass `min_per_metal=20` AND `n_clouds=960`**
(independent fixes for the cap-window under-representation and any
single-reservoir unlucky sample — see §9.4).

**PROJECTED wall time: 3-5 hr** on the existing checkpoint, basis:
MEASURED ~31 s per record on the concurrent CPU Vina sweep (119 records
over 62 min). The decode half is the unmeasured term; budget conservatively
and be ready to abort on hour 3.

**Decision tree** (verbatim from `cfm_todo.json::next_actions[0]`,
unchanged by this round):

| arm0 | arm1 | arm2 | arm3 | read as |
|---|---|---|---|---|
| baseline reproduced (aromatic frac 0.001229) | coord guidance moves T-4/SP within 3% of REAL_DFT | aromatic guidance moves mean aromatic frac > 0.05 | arm3 > arm1 + arm2 on organic-bulk (QED, SA, ALERTS, ROTB) | cohort reproducible, geometry lever live, head responsive, levers additive |
| baseline drifts | coord guidance off >10% | aromatic guidance < 0.005 | arm3 < max(arm1, arm2) | 3-seed panel first, levers inert on this axis, head NOT responsive, levers compete |
| arm0 already at > 0.01 aromatic frac | arm1 SP recovery > 5pp, arm2 flat | arm2 rise strong, arm1 flat | all 4 arms within 5% on QED/SA/coord-geom | zero-Pt patch/n widening alone did it; collapse to 3-seed panel; coord lever live; head responsive; guidance inert |
| | | | | retrain budget on organic-bulk; macro-F1(TYPE) needs re-evaluation; (a)+(b)+(c) lever choice |

What changes the plan in each direction is enumerated at
`cfm_todo.json::next_actions[0].decision_tree_2026_10_12_evening
.what_changes_the_plan_in_each_direction`. DO NOT INFER FROM A SINGLE
ARM — all four must run before any number is read.


---

## 11. Round-4 review WRITE-IN — 2026-10-12 evening (review + web research + reflection)

Per the user directive "based on current state, do review + web research + reflection,
then update cfm_todo.json status", this section is the factual record of what the
four research reports (A literature, B cheap experiments, C retrospective, D adversarial)
plus the round-9 challenge pass established, and what was written into
`cfm_todo.json`. Nothing here rewrites §1-§10.

**Path on disk.** Consolidated write-in at
`cfm_todo.json::round4_review_2026_10_12` (208417 -> 231260 bytes,
pre-write backup at `cfm_todo.json.round4_pre.bak`). Detailed evidence:
`round7_cheapest_experiments_2026_10_12`,
`corrections_2026_10_12_round7[round7-retrospective-2026-10-12]`,
`corrections_2026_10_12_round8[round8-literature-bond-order]`,
`round8_adversarial_review_2026_10_12`,
`round9_review_verdict_2026_10_12`.

### 11.1 Aromatic offset result summary

Post-hoc per-class logit offset on `BOND_AROMATIC` (wired end-to-end via
`--bond-aromatic-logit-bias`, default 0.0 resolves to None) was swept over
four strengths. The offset reaches the decoder (11 of 64 clouds decode
differently between bias 0.0 and bias 0.5) and produces ZERO aromatic
bonds at ANY of the four strengths.

| bias | decode / 64 | decode rate | aromatic bonds | bonds examined |
|---|---|---|---|---|
| 0.0  | 22 | 0.344 | 0 | 644 |
| 0.5  | 13 | 0.203 | 0 | 346 |
| 1.5  |  3 | 0.047 | 0 |  71 |
| 3.0  |  0 | 0.000 | 0 |   0 |

**1061 bonds examined across the sweep; not one aromatic.** The verdict is
MEASURED dead: the post-hoc operating-point route is closed in the strongest
available way. The defect is NOT fixable at the decision rule; it needs the
head to actually rank aromatic above single on intermediate bond orders.

→ `molmetal/reports/cfm_aromatic_bias_sweep_seed0/arm_bias{0.0,0.5,1.5,3.0}.json`,
  `cfm_todo.json::blockers[BLK-C-M4].OFFSET_SWEEP_MEASURED_2026_10_12.FULL_SWEEP_COMPLETE`

### 11.2 Literature findings that survived the challenge

- The 0.454591 real-tmQM six-metal aromatic-bond-fraction target is a
  MEASURED aggregate on disk at
  `metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json::descriptors.arom_bond_frac.mean`
  (n=31150). Verified.
- The four-option ladder (opt1 measure Wiberg on metal-bound pairs /
  opt2 distil RDKit teacher / opt3 CORAL ordinal head / opt4 abandon head
  retrain) is internally consistent and follows cheapest-first ordering.
  Methodologically sound.
- Wang & Dokholyan bioRxiv 2025.05.06.652517 (YuelBond) per-class F1 numbers
  (single 83.3, double 56.3, triple 59.9, aromatic 75.5) are
  **UNVERIFIED on paper text in this session** (no WebFetch performed).
  Shape verifiable (year/title/id), content not opened. Not citable in
  `paper/*` without a re-fetch.
- Bridgeman 2001 Dalton Trans 10.1039/B102094N specific WBO values on 3d
  dichlorides are **UNVERIFIED** (RSC returned HTTP 403 on WebFetch). The
  general Wiberg/Mayer claim (less basis-set sensitive than Mayer on organic)
  is in `vibe-qc` docs; metal-bound WBO values are NOT cited from a fetched
  page.
- CORAL ordinal regression (Cao/Mirjalili/Raschka, arXiv:1901.07884) has
  **NO molecular-bond-order precedent** in the indexed literature. Honest-negative.
- Inverse-frequency class weights (Logit Adjustment, Menon et al.,
  arXiv:2007.07314) on our `[24378, 4460, 421, 28, 5661]` pair counts would
  amplify TRIPLE (0.08% of pairs) -- a known disease already at 66% of
  generated output. A global prior correction would amplify a known disease
  while treating another.

### 11.3 Cheap experiments that survived the challenge

- **(a) Distillation teacher NOT AVAILABLE.** venv import scan:
  `pymatgen` / `ase` / `chemenv` / `matminer` / `scikit_chem` all
  `ModuleNotFoundError`; `openbabel` 3.1.0 installed but `OBBond.GetBondOrder`
  returns integer 1/2/3 + aromatic flag and needs SDF/MOL/SMILES input --
  NOT a continuous BO from electron density.
- **(b) LABEL IS CLEAN.** Metal-bound pairs are 0.53% of AROMATIC labels
  (22,570 of 4,281,362); metal-involved SINGLE 367,062 of 393,632 (93.23%).
  The 70.7% collapse is NOT a metal-bound-pair label artefact -- but it is
  also NOT purely a head-ranking failure. CORRECTED 2026-10-13: measured
  per Wiberg sub-band, the head has a transition zone at [1.46,1.51] with
  recall 0.192/0.301/0.330/0.663/0.858 across 1.30-1.40 / 1.40-1.45 /
  1.45-1.50 / 1.50-1.60 / 1.60-1.70 -- it is decisive above the band and
  wrong below it, where the LABEL is itself ambiguous (91.96% of
  aromatic-band bonds have bo < 1.5 and the training band puts the
  single/aromatic boundary at 1.25 while the evaluation uses 1.3-1.7).
  So the figure relocates rather than fixes when the band moves.
  Path: molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13/.
  Original claim, corrected: the
  absolute totals reported in this session (11,046,460 SINGLE etc.) are
  6.27x larger than a fresh compute via `iter_bo_bonds` on the same BO
  shards; percentages match within 1.2 points so the qualitative verdict
  holds, but absolute totals are NOT directly citable.
- **(c) BINDING CONSTRAINT IS SAMPLER-SIDE.** Generated 64-cloud
  `arm_bias0.0.json`: `p01_pair_distance.mean = 1.131` (below aromatic 1.30
  lower edge), `metal_ligand_min.mean = 1.151` vs real 1.983, 64/64 hard
  clashes. 8-cloud seed-1234 smoke: 1.0631% of generated pairs in
  `[1.30, 1.55)` C-C aromatic band vs 4.0084% real. Head cannot rank
  aromatic for pairs that are not there. Head-only fix cannot close the gap;
  needs a geometry-stage intervention.
- **(d) bond_class_weights power 0.5 NOT WORTH IT IN ISOLATION.** 9 configs
  declare it; none was run on the full 6-metal corpus for 30+ epochs.
  Projected cost: 50 x 13 min = 10.8 hr single-seed; 54 GPU-hours for 5 seeds;
  ~55 min for 5-epoch smoke. (c) says only 1.06% of generated pairs are at
  aromatic distance vs 4.01% real; even a perfect head recovers at most ~61
  candidate pairs in the 8-cloud cohort.

### 11.4 Session retrospective -- the one pattern

**Silent-success at the gate.** Across every defect in Report C, the harness
reports SUCCESS (exit 0, parse OK, metric computed, test asserts) while the
action that should have been taken was never taken. The sweep ran; the bias
never reached the decoder. The class-weight code was read; the comment was
read but not the normalisation. The 3-seed panel was proposed; the
checkpoints existed but at different epochs. The red check ran; the branch
was never exercised. The launcher printed SWEEP DONE; both arms had crashed
on a wrong REPO path.

**Generalisation.** An exit code of 0 indicates the program returned without
raising. It does NOT indicate that the program did the thing the calling code
thought it asked for. Across the defects above, every one produced rc=0 at
the gate that was supposed to catch it. A test that exits 0 against a
syntactic error or an unbound local or a branch that was never reached is
not a green test -- it is a test that has not yet been written.

**Standing checks for repeat offenders:**
SWEEP-OPERATING-POINT (per-arm SMILES set vs control before DONE);
RECOMPUTE-FROM-FORMULA (any specific numeric claim);
CHECKPOINT-EPOCH-AT-READ (multi-seed designs);
DICT-AND-OBJECT-HELPERS (both attribute and mapping access);
REPO-PATH-NAME-ASSERT (every `Path(__file__).resolve().parent...`);
DICT-SHAPE-AT-EXTRACTION (output keys preserved across refactors);
RED-CHECK-SPECIFICITY (`rc==1` with expected substring, not `rc != 0`);
STATISTICAL-TEST-TRIVIAL-CASES (p=1.0 for `b=c` McNemar; hand-derived value);
CITATION-PROVENANCE-MARKER (VERIFIED with quote or NOT VERIFIED);
TEST-FILE-OVERWRITE-GUARD (`git status --porcelain` before Write).

**What did NOT need fixing.** The premise verdicts in BLK-C-M4 (head-vs-
generator split), BLK-C-L2 (8 dead terms by their own premise), BLK-C-OBJ
(per-target decision) all survived the day's adversarial review. The defects
were in execution, not in direction.

### 11.5 Adversarial objections that survived the challenge

- **(a) Generated geometry substrate -- VALID.** The sampler produces
  geometry with FAR fewer (C,C) pairs in the aromatic band than the
  templates did. The bias sweep as designed cannot distinguish head from
  sampler. New finding: the gen-side defect is 1.6x the head-side defect
  (BLK-C-M4 REFINED_2026_10_12); a retrain on the head, alone, cannot
  close the geometric gap. To settle: per-cloud count of (C,C) pairs in
  1.30-1.55 A on the GENERATED cohort, then subset the head's per-pair
  outputs to the in-band subset and check the logit margin.
- **(b) Rejection-reason log -- VALID.** `DecodedMol.error` is set at three
  distinct paths in `cfm_decoder_yuelbond.py` (line 727 RWMol.GetMol failed,
  line 733 SanitizeMol raised -- error NOT explicitly set on this branch;
  `smiles=""` is the only signal -- line 779 yuelbond:empty_or_failed via
  `_empty_decoded`). The sweep helper at
  `molmetal/scripts/wf_blkc_l3_guidance_probe.py:367-385` only reads `.smiles`
  and `.sanitized`, never `.error`. Without the breakdown, the 0/1061
  aromatic result cannot be cleanly attributed to the head vs the sanitiser.
- **(c) Target defensibility -- PARTLY VALID, AND MEASURED ON DISK.** The
  0.454591 aggregate target IS on disk (n=31150). tmQM has no activity
  annotation so the "anticancer" half cannot be settled from the corpus
  alone; literature (RSC MedChem 2026 d6md00154h, Sci Rep 2016 PMC4725915,
  PMID 15777221, Sadler Metallomics 2018 c7mt00242d) supports aromatic
  carriers as the current optimisation programme. 0.454591 is a sound target.
- **(d) Blocker mapping -- DIRECTLY MOVED 1, PARTIALLY 3, NONE 4.** DIRECTLY
  MOVED: BLK-C-M4 (refutes the operating-point route with 0/1061 aromatic
  across 1061 bonds). PARTIALLY MOVES: BLK-C-L2 (strengthens entropy_term
  case -- head-only change without retrain is dead); BLK-C-L3 (rules out a
  competitor to guidance); BLK-C-M3 (decode_rate variance data point,
  though M3 itself was closed on a different parameter axis). DOES NOT MOVE:
  BLK-C-M6 (reconstructions); BLK-C-PERF (orthogonal axis); BLK-C-OBJ
  (decided, result 4 is consistent); BLK-C-L5 (closed, embedding not touched).

### 11.6 BLK-C-M4 status update (additive)

Status string is UNCHANGED: still `OPEN -- honest-negative, must appear in
the paper`. A new sub-field `ROUND4_REVIEW_2026_10_12` was added recording:

> Post-hoc operating-point route is MEASURED dead (FULL_SWEEP_COMPLETE: 1061
> bonds examined across 4 strengths, 0 aromatic). The retrain decision is
> unchanged in direction but the cheap alternative is removed.

The `live_options_2026_10_12` ladder is preserved verbatim:
- preferred_first: `opt1_measure_wiberg_on_metal_bound` (~1 CPU day)
- preferred_if_labels_bad: `opt4_abandon_head_retrain`
- OPEN_until: one of opt1/opt2 produces a measured shift, OR opt4 is elected
  and the paper reports the honest-negative

### 11.7 What the research undermined

- **The "cheap fix to BLK-C-M4" framing.** Post-hoc per-class offset was the
  cheapest candidate (no retrain, decode-time postprocess) and it is
  measured dead. The retrain decision is now strictly more expensive (no
  cheap alternative left) but also strictly more necessary.
- **The Wang/Dokholyan per-class F1 numbers** (single 83.3, double 56.3,
  triple 59.9, aromatic 75.5) are UNVERIFIED on paper text in this session.
  Not citable in `paper/*` without a re-fetch.
- **The (b) absolute bond-count totals** (11,046,460 SINGLE etc.) are 6.27x
  larger than a fresh compute; percentages match within 1.2 points but
  absolute totals are NOT directly citable.
- **The (c) per-metal aromatic fractions** 0.4569/0.4799/0.4504/0.4660/
  0.4453/0.4069 are not in the on-disk baseline file; only the aggregate
  0.454591 with n=31150 is recorded there.

### 11.8 New next_action (priority 1) -- the 5-min rejection-reason patch

The cheapest decision-changing step on the board today, per the round-9
verdict on Report D (b):

> ONE-LINE PATCH: in `molmetal/scripts/wf_blkc_l3_guidance_probe.py` near
> line 376, add `rejection_reason: getattr(res.decoded, "error", None)`
> to the `rows.append` dict, alongside the existing `smiles` and `sanitized`
> reads. Then re-run the `arm_bias3.0` arm only. ~5 min CPU.

**Expected outputs.** Three possible outcomes:

- **(i) all 64 rejections are head-side** (`yuelbond:empty_or_failed`) --
  the verdict "head cannot rank aromatic" stands; BLK-C-M4 retrain decision
  moves to the `live_options_2026_10_12` ladder as previously stated;
  `opt4_abandon_head_retrain` is the most-supported reading given the
  binding constraint is sampler-side per (c).
- **(ii) all 64 are sanitiser-side** (`SanitizeMol` or `RWMol.GetMol`) --
  the verdict attributes to the sanitiser and BLK-C-M4 must be RE-SCOPED;
  the head's job becomes "produce aromatic-eligible outputs that the
  sanitiser accepts" and the metric of interest becomes the sanitiser-side
  breakdown by rejection path, not the head's class-4 vs class-1 ranking.
- **(iii) mixed** -- the breakdown itself is the new evidence and the
  retrain decision waits for arm1/arm2 re-run with the patch (+10 min CPU).

The next_actions queue (6 entries, sorted by priority) is now:

| priority | action |
|---|---|
| 0 | decode-enabled guidance sweep (P0 sweep, retained coordinates) |
| **1** | **NEW: 5-min rejection-reason patch + arm_bias3.0 re-run** |
| 2 | Re-run the 10-pocket Vina sweep (RUNNING, do not relaunch) |
| 3 | close pad_groups distributional test |
| 4 | 3-seed free-generation panel (seeds 1 and 2) |
| 5 | SUPERSEDED (coordination sphere now rideable on next_actions[0]) |

### 11.9 What this round did NOT change

- No `paper/*` edits (AGENTS.md 5.3, TODO-84).
- No git commit.
- No new GPU launch.
- No retraction of round7 retrospective, round8 lit review, round8
  adversarial review, or round9 challenge verdict.
- No blocker marked CLOSED. BLK-C-M4 stays OPEN (operating-point route dead,
  retrain decision unchanged but cheaper alternative removed). All other
  blockers carry their existing status strings.

### 11.10 Files touched (read-only this round)

- `cfm_todo.json` (read all 208417 bytes, then atomic_write_file with
  round4 block appended; pre-write backup at `cfm_todo.json.round4_pre.bak`)
- `molmetal/reports/cfm_aromatic_bias_sweep_seed0/arm_bias{0.0,0.5,1.5,3.0}.json`
  (already on disk; cited as evidence)
- `molmetal/scripts/wf_blkc_l3_guidance_probe.py:367-385` (already on disk;
  cited as evidence for objection (b))
- `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:400, 727, 733-737,
  749, 779` (already on disk; cited as evidence for objection (b))
- `metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json` (already on disk;
  cited as evidence for objection (c))
- `cfm_todo.json::blockers[BLK-C-M4].OFFSET_SWEEP_MEASURED_2026_10_12`
  (already on disk; cited as evidence for the offset result summary)
