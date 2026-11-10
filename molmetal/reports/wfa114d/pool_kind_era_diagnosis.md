# Wf-A114D — the pool-kind rule is wrong for one pool (and it matters)

**Wave:** Wf-A114D (diagnosis input for the correction wave)
**Date:** 2026-10-30
**GPU used:** 0. Every number below is a CPU re-read of files already on disk.
**Status:** diagnosis COMPLETE; the fix ships in Wf-A114D proper.

---

## §1 The claim, in one line

`molmetal/scripts/cfm_pool_kind.py` classifies a pool by its CSV header and
says **`metal_z` in the header => RECONSTRUCTION**. That is true for two of the
three `metal_z`-headed pools on disk and **false for the third**, which is the
**pool of record behind the paper table `metrics/paper_tables/T7_generation.md`**
and behind `metrics/by_metric/cfm_generation_metrics.json` and
`metrics/by_metric/cfm_sa_guided.json`.

**The pool `cfm_fullcorpus_2026_10_10/generated_smiles.csv` (66 rows) is FREE
GENERATION, not a reconstruction.** The project's own artifact
`metrics/by_metric/cfm_free_generation_panel_seed0.json` (dated 2026-10-12)
already says so; the Wf-A114 rule contradicts it.

> **The shortest way to see the defect.** Two of the three artifacts that
> consume this pool already declare it to be generation —
> `cfm_sa_guided.json:/design/pool_of_record` = *"cfm_free_generation_panel_seed0.json
> (66 molecules, 192 attempts)"*, and
> `cfm_generation_metrics.json:/cohort/yield_source` =
> *"cfm_free_generation_panel_seed0.json:generation.gen_validity"*. The
> **pool-kind RULE is the only thing on disk that disagrees with them**, and it
> is the thing that the Wf-A114 audit then wrote into the report, the TODO
> index and the honest-negative catalog.

---

## §2 Six independent lines of evidence (all MEASURED)

### §2.1 The project's own artifact says "free generation"

`metrics/by_metric/cfm_free_generation_panel_seed0.json`:

| field | value |
|---|---|
| `date` | `2026-10-12` |
| `how` | *"molmetal/scripts/wf_cfm_decode_dump_smiles.py **after the 2026-10-12 fix**, replaying ODE integration from cfm_fullcorpus_2026_10_10 checkpoint (epoch 10), 192 tmQM val clouds, midpoint/50"* |
| `generation.n_attempted` | 192 |
| `generation.n_valid` | 66 |
| `generation.gen_validity` | 0.344 |
| `generation.single_fragment_ratio` | 0.9697 |
| `panel.sa_mean` | 6.4423 |
| `panel.qed_mean` | 0.0944 |

Its `comparison` block says *"generation 6.4423 vs real 5.0269"* and
*"generation 0.9697 single-fragment vs real 0.8365"* — it names the 66-pool a
**generation** cohort, and it compares it against the reconstruction band.

### §2.2 The file's `sanitized` column is structurally impossible in the reconstruction era

`molmetal/scripts/wf_cfm_decode_dump_smiles.py` has **two behaviours with one
header**:

| era | what it does | `sanitized` column | `metal_z` column |
|---|---|---|---|
| **pre-2026-10-12** | `head.featurise` -> `_select_bonds` -> `_bridge_components` -> `_assemble_mol` on the REAL tmQM clouds (bond head only, no sampler) | the bond head's own `decoded_mol.sanitized` | the metal's atomic number, read off the real cloud |
| **2026-10-12+** | `T.evaluate_generation_validity(...)` -> `generate_clouds_from_noise` (the real ODE sampler) | `row.get("sanitized", False)` — and `decode_chemistry`'s return dict has **no `sanitized` key**, so this is **always 0** | `row.get("metal_z", "")` from `decode_chemistry` |

Source: `git show b2547f6f^:molmetal/scripts/wf_cfm_decode_dump_smiles.py`
(lines 150-215) vs the working-tree file (lines 186-263); and
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:9107-9126`
(`decode_chemistry`'s return dict — 12 keys, none of them `sanitized`).

MEASURED on the three `metal_z`-headed pools:

| pool | rows | `sanitized` | `metal_z` | era |
|---|---|---|---|---|
| `cfm_fullcorpus_2026_10_10` | 66 | **0 for 66/66** | **'' (empty) for 66/66** | **2026-10-12+ = GENERATION** |
| `cfm_fullcorpus_b64_2026_10_11` | 40 | 1 for 40/40 | populated for 40/40 | pre = RECONSTRUCTION |
| `cfm_objective_terms_2026_10_11` | 43 | 1 for 43/43 | populated for 43/43 | pre = RECONSTRUCTION |

Two further structural facts, both MEASURED on 2026-10-30:

* **All three pools carry `n_fragments`, NOT `n_frags_raw`** — they were all
  written by the decode-dump script, so `n_frags_raw` is absent from every one
  of them. `metal_z` is therefore the *only* header marker any of them has,
  which is exactly why the header could not decide the question.
* Their `(epoch, seed)` keys are `10/0`, `8/0` and `5/0` respectively — the
  66-pool's `epoch 10, seed 0` matches `cfm_free_generation_panel_seed0.json`
  ("cfm_fullcorpus_2026_10_10 checkpoint (epoch 10) … seed-0") and the other
  two match no generation panel.

### §2.3 The file timestamps agree

```
-rw-r--r-- 6948  2026-10-12 03:04  cfm_fullcorpus_2026_10_10/generated_smiles.csv
-rw-r--r-- 4076  2026-10-11 16:24  cfm_fullcorpus_b64_2026_10_11/generated_smiles.csv
-rw-r--r-- 4586  2026-10-11 16:24  cfm_objective_terms_2026_10_11/generated_smiles.csv
```

The corrected script skips a run whose CSV already exists
(`"already has {out_csv.name}, skipped"`), so the two 2026-10-11 pools kept
their pre-fix content while the 66-row pool was re-written on 2026-10-12.

### §2.4 The commit timeline agrees

| commit | when | what |
|---|---|---|
| `b2547f6f` | 2026-10-11 19:42 | *"The decode dump never ran the ODE"* — **docstring only**; the code was still the reconstruction path at that moment |
| `0e232def` | 2026-10-13 | stages the working-tree change that made the script call `T.evaluate_generation_validity` (the ODE path) |

So the ODE path existed in the working tree from **2026-10-12 03:00** — the
66-pool's mtime — and was committed on 2026-10-13.

### §2.5 The sha256 chain agrees

`metrics/by_metric/cfm_generation_metrics.json` and
`metrics/by_metric/cfm_sa_guided.json` (both written 2026-10-19) record
`generated_sha256 = 6e17eb2d037937d0f0ed71de4aa0cbf68687af7edf882c385d4f2e44f539e8ef`.
The file on disk hashes to exactly that value — it has not been touched since
2026-10-12 03:04, so both artifacts scored the generation pool.

### §2.6 The project's own 2026-10-12 STATUS DOC calls the same numbers "generated"

`docs/CFM_STATUS_2026_10_12.md` §2 — *"What it produces (first real free-generation
measurement)"* — reads, verbatim:

> *"The only numbers on disk that come from **integrating the ODE from noise**
> rather than reconstructing real molecules. 192 tmQM val clouds, midpoint/50,
> seed-0 checkpoint, epoch 10:"*

and its table's first column is headed **"generated"**, carrying
`yield (gen_validity) 66/192 = 0.344` and `single_fragment_ratio 0.9697`,
against a right-hand column headed **"reconstruction of real tmQM"**. It names
`metrics/by_metric/cfm_free_generation_panel_seed0.json` as the source and
warns that the comparison column is *"a **reconstruction floor**, not a corpus
floor"*.

**This is the sharpest form of the defect.** The project established the label
correctly on **2026-10-12** — the same day the pool file was written (03:04) —
and the **Wf-A114 audit inverted it**. `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md:75`
and `TODO/INDEX.md:1033` now publish the *same* `0.9697` as

> *"Data | OK sound | **reconstruction pools on _real_ tmQM geometry**:
> single-fragment 0.9697 / 0.7500 / 0.6744"*

That sentence re-labels the generator's own output as a reconstruction, and
hands the paper a §1 whose "Data: sound" verdict is about the wrong cohort.
Two of the three numbers in it (0.7500, 0.6744) *are* genuine reconstruction
pools; the first (0.9697) is not — and it is the one that anchors the claim.

> **MEASURED cross-check (2026-10-30).** The 66-pool has **0 unparseable
> rows** of its 66 (`sanitized` = `'0'` on 66/66, `metal_z` = `''` on 66/66,
> RDKit parse rate 66/66), so moving it into the generation block changes that
> block's row total but not its unparseable count (still **1 of 21,445**).

---

## §3 What the misclassification does to the numbers

### §3.1 The pool set (13 pools on disk)

| kind | Wf-A114 said | corrected |
|---|---|---|
| generation | 9 | **10** (+ `cfm_fullcorpus_2026_10_10`) |
| reconstruction | 3 | **2** (- `cfm_fullcorpus_2026_10_10`) |
| unknown | 1 | 1 |

`metrics/by_metric/cfm_fragmentation_audit.json` -> `runs[*].pipeline` is the
field that carries this.

### §3.2 The SA-gate rate — the one number that was about to be published wrong

`metrics/by_metric/cfm_sa_guided.json` -> `combined`:

| | rows | SA<=4.5 | rate |
|---|---|---|---|
| on disk today (2026-10-19 artifact) | 2396 | 16 | **0.0067** |
| Wf-A114C Cluster P's proposed split (66-pool treated as reconstruction) | 2330 | 16 | **0.0069** |
| **corrected split (66-pool IS generation)** | **2396** | **16** | **0.0067** |

The 66-row pool contributes **0** molecules at SA <= 4.5
(`sa_counts_at_or_below = {"4.0": 0, "4.5": 0, ...}`), so the rate is
unchanged at **0.0067**; only the *membership* of the generation block changes
(it goes back to 2396 rows) and the reconstruction block becomes the two real
reconstruction pools (40 + 43 = **83** rows) instead of 66.

**Cluster P's 0.0067 -> 0.0069 is a regression and must not ship.**

#### §3.2.1 Why the on-disk artifact is already right (MEASURED, and it is not luck)

The 2026-10-19 artifact has **no `combined_by_kind` block** — the kind
machinery did not exist yet. Its `pools` array is exactly three pools, and all
three are **generation** pools:

| label | path | rows | header marker | kind |
|---|---|---|---|---|
| `panel_of_record` | `cfm_fullcorpus_2026_10_10` | 66 | `metal_z` | **generation** |
| `pcgrad_on` | `cfm_pcgrad_on_2026_10_14` | 1155 | `n_frags_raw` | generation |
| `pcgrad_off` | `cfm_pcgrad_off_2026_10_15` | 1175 | `n_frags_raw` | generation |

66 + 1155 + 1175 = **2396** = the `combined.n_scored`, and 16 of them are at
SA <= 4.5 -> **0.0067**. The artifact's own `/design/pool_of_record` field
says *"cfm_free_generation_panel_seed0.json (66 molecules, 192 attempts)"* — so
the artifact **already declares the 66-pool to be the generator's panel of
record**.

So the timeline is:

1. **2026-10-19** — the artifact pools its three default pools, all of which
   are generation. `combined` = 2396 -> **0.0067**. Correct.
2. **2026-10-30 (Wf-A114B)** — the kind machinery is added (`combined_by_kind`,
   "generation-only" rate, refusal on UNKNOWN). Under the **wrong** rule the
   66-pool is reconstruction, so a generation-only `combined` becomes
   **2330 -> 0.0069**. Wrong.
3. **2026-10-30 (Wf-A114D, this wave)** — under the **corrected** rule the
   66-pool is generation, so `combined` stays **2396 -> 0.0067**, and the
   reconstruction block becomes the two real reconstruction pools (83 rows).

**The correction therefore does not change a single published number in
`cfm_sa_guided.json` — it changes what the script would do if it were re-run
today.** That is the whole point: the wrong rule would silently rewrite a
correct headline the next time anyone re-ran the audit, and the number that
moves is the one in the paper table (`T7_generation.md`, row 13).

### §3.3 The generation-side gate

`cfm_fragmentation_audit.py` scores the generation pools only
(`PASS=0 / FAIL=9`). With the 66-pool reclassified as generation, its
end-to-end rate is **64 / 192 = 0.3333** (64 single-fragment rows out of the
decode-dump's own **192** attempts — *not* the trainer's 768 epoch-level
draws, which are a different population), so it still **FAIL**s the 0.50
threshold: **PASS=0 / FAIL=10**.

### §3.4 The paper table is RIGHT

`metrics/paper_tables/T7_generation.md` is titled *"Generation-side metrics
(66 survivors from 192 attempts)"* and every row cites
`cfm_generation_metrics.json`, which scores the 66-pool. **The table is
correct** — because the 66-pool IS the generator's output. It was never wrong;
the new rule was.

### §3.5 The row totals move; the unparseable count does not

| set | Wf-A114 said | corrected |
|---|---|---|
| generation pools | 9 pools / **21,379** rows / 1 unparseable | **10** pools / **21,445** rows / **1** unparseable |
| reconstruction pools | 3 pools / 149 rows | **2** pools / **83** rows |

(+66 rows, +0 unparseable — MEASURED, the 66-pool parses 66/66.)

### §3.6 Every doc site that carries the inverted label (the integrator's work list)

Grep of `docs/`, `TODO/` and `metrics/paper_tables/` on 2026-10-30, for the
exact strings the correction invalidates:

| file:line | current text | what it must become |
|---|---|---|
| `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md:42` | *"9 generation pools, PASS=0, FAIL=9"* | **10** pools, PASS=0, **FAIL=10** |
| `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md:75` | `cfm_fullcorpus_2026_10_10 \| 0.9697 \| PASS` (a reconstruction row) | moves to the **generation** table |
| `TODO/INDEX.md:1033` | *"Data … reconstruction pools on real tmQM geometry: 0.9697 / 0.7500 / 0.6744"* | 0.9697 leaves the reconstruction list |
| `TODO/INDEX.md:1038` | *"9 generation pools, PASS=0, FAIL=9"* | **10** / FAIL=**10** |
| `TODO/INDEX.md:1096` | *"the 9 generation pools hold 21,379 rows, 21,378 parseable, 1 unparseable"* | **10** pools / **21,445** / **21,444** parseable |
| `TODO/completed/wfa114_cfm_fragmentation_audit.md:22,27,84,86` | the same four claims | as above |
| `TODO/pending/wfa114_cfm_fragmentation_fix_verification.md:80` | *"generation-side gate: 9 pools, PASS=0, FAIL=9"* | **10** / FAIL=**10** |
| `docs/HONEST_NEGATIVE_CATALOG.md:12922, 12926, 12980-12983, 13260, 13397, 13487, 13490, 13493` | *"9 generation pools, PASS=0, FAIL=9"*, *"3 pools, PASS=2 / FAIL=1, ratios 0.9697 / 0.75 / …"*, *"9 generation pools … 21,379 rows"* | as above; the catalog gains **+1** entry (§6.202) recording the inversion and its correction |
| `docs/CFM_STATUS_2026_10_12.md:66,90,108` | *"single_fragment_ratio 0.9697"* under the **generated** column | **already correct** — do NOT touch |

The last row is the point: the 2026-10-12 status doc got this right and the
2026-10-30 audit got it wrong. The correction restores the 2026-10-12 label.

---

## §4 The fix design (for Wf-A114D)

### §4.1 The rule owner

`molmetal/scripts/cfm_pool_kind.py` must stop reading `metal_z` as an
unconditional reconstruction marker, because **one script wrote one header in
two eras**. The rule must become evidence-based and must **refuse** when the
evidence disagrees:

1. an explicit `pool_kind` column, if present, wins (forward-compatible);
2. `n_frags_raw` in the header => **generation** (the trainer's free-generation
   path — unambiguous, one writer);
3. `metal_z` in the header => decide by the CONTENT, and require the two
   content signals to AGREE:
   * `sanitized == 1` **and** `metal_z` populated => **reconstruction**
     (the bond head's own flag on a real geometry — impossible in the ODE era,
     where `decode_chemistry` has no `sanitized` key);
   * `sanitized == 0` **and** `metal_z` empty => **generation** (the ODE era);
   * anything else => **unknown**, with a reason that names the disagreement —
     never silently bucketed;
4. neither marker => **unknown**, as today.

Reading the content means reading the first data row, so `classify_pool` must
stop being strictly header-only (it may read the header plus **at most one**
data row).

### §4.2 The root cause

`molmetal/scripts/wf_cfm_decode_dump_smiles.py` should write an explicit
`pool_kind` column (and the number of attempts it drew) so that no future pool
can be ambiguous again. The header-only rule was a symptom; the two-behaviour
script is the cause.

### §4.4 The code sites that assert the false rule in prose (the fix list)

The false claim is not confined to the rule module — it is written into
**docstrings and comments** in the consumer scripts, and one of them is the
reason the wrong label propagated into the docs.

| file:line | the false prose | note |
|---|---|---|
| `molmetal/scripts/cfm_pool_kind.py` (module docstring) | *"loads REAL tmQM geometries … and runs only the bond head … It never calls the sampler"* | **the rule owner's own banner** |
| `molmetal/scripts/c7_scale_matched_pools.py:22-25` | *"writes a CSV with a `metal_z` column by loading REAL tmQM geometries and running only the bond head — a reconstruction, never the sampler … **Every full-corpus pool on disk is a reconstruction**"* | dated *"V1.1 CORRECTION (2026-10-20)"* — i.e. the false rule was authored **8 days after** the 2026-10-12 status doc got it right, and it inverted that doc |
| `molmetal/scripts/c7_scale_matched_pools.py:629-630` | the runtime message *"header contains 'metal_z' -> reconstruction; 'n_frags_raw' -> generation"* | must print the rule owner's **reason string** instead of re-deriving the verdict |
| `molmetal/scripts/c7_fullgen_pool_quality.py:301-308` | *"MEASURED: the main pool is NOT always a generation pool — `cfm_fullcorpus_2026_10_10/generated_smiles.csv` carries `metal_z`, i.e. it is a RECONSTRUCTION pool, so its numbers must not be read as the generator's quality."* | **the A114B verification round's own D1 note is the misclassification** — it is right that the main pool must be *labelled*, and wrong about which label |
| `molmetal/scripts/c7_fullgen_pool_quality.py:254-255, 265-270` | `--recon-pool` "Defaults to the two full-corpus reconstruction pools on disk" | **already correct** — the defaults are `cfm_fullcorpus_b64_2026_10_11` + `cfm_objective_terms_2026_10_11`, and :282-287 already REFUSES a non-reconstruction pool. Leave the defaults alone. |

**One subtlety the consumers must not miss.** The 66-pool's connectivity column
is **`n_fragments`**, not **`n_frags_raw`** — it was written by the decode-dump
script, not by the trainer. So the measure *"the trainer's own `n_frags_raw == 1`
column"* is **None** for it, and its connectivity must be read **off the
molecule** (`Chem.GetMolFrags`), exactly as `c7_fullgen_pool_quality.py:85-93`
already does. A consumer that moves the 66-pool into the generation block
without re-deriving its connectivity will silently score it `None` and drop it
from the denominator.

### §4.5 What must NOT change

* `metrics/paper_tables/T7_generation.md` — already correct;
* `metrics/by_metric/cfm_generation_metrics.json`'s numbers — they were
  computed on the generation pool and are right. What may change is its
  `generation_side` / verdict SCOPE (Cluster P refused to publish
  generation-side claims because it believed the subject was a reconstruction);
* the census (`metrics/by_metric/cfm_run_census.json`) — it reads the trainer's
  own per-epoch metrics, not the pools, so the pool-kind rule does not touch
  it.

---

## §5 One observation this diagnosis does NOT explain (recorded, not fixed)

In the 66-pool, `decode_chemistry`'s `metal_z` is empty for **every** row, so
the row's metal was not matched against the template's `seed_metal_z`. The
trainer's own `events.jsonl` for the same run and epoch reports
`metal_missing_rate: 0.0` and `n_metal_seeded: 250`, so the template metals
were found on the trainer's side. The decode-dump path pairs the sampled
clouds with the template metals by index, and `generate_clouds_from_noise`
groups samples by atom count — the pairing can drift. This is **unmeasured**
here and is NOT part of the pool-kind fix; it is recorded so it is not
mistaken for a solved problem.
