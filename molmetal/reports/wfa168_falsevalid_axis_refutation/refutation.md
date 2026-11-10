# Wf-A168 -- REFUTATION of the "our false-valid / speck decomposition" branch

**Verdict: NOT REFUTED.** Every headline NUMBER in the branch report was re-derived
independently from the raw pool and matches to the last digit. Four prose/note-level
concerns survive; none of them moves a number.

Branch under review: `molmetal/scripts/wfa168_falsevalid_axis.py` +
`molmetal/tests/test_wfa168_falsevalid_axis.py` +
`molmetal/reports/wfa168_falsevalid_axis/{falsevalid_axis.json,report.md}`.

**0 GPU.** CPU-only, RDKit-only. No torch, no checkpoint, no decode. Every python /
pytest call in this refutation ran with `CUDA_VISIBLE_DEVICES=""`.

## Checks

| # | check | verdict | evidence |
|---|---|---|---|
| C1 | re-derive the numbers from the raw pool | **PASS** | independent script `/tmp/wfa168_refute/rederive.py` (does NOT import the branch script) reproduces all 38 checked values exactly |
| C2 | is the published artifact a cross-check, not a source? | **PASS** | mutation M2 (numerator replaced by the literal 2663) -> 3 negative controls + the vacuity guard FAIL |
| C3 | does the regression test bite? | **PASS_WITH_GAP** | 15 passed; M1/M2/M3/M7 caught; **M4/M5/M6 not caught** |
| C4 | is the comparison overstated? | **PASS_WITH_CONCERN** | the artifact quotes **zero** frontier numbers (0 hits for frontier / arXiv / 92.7 / http / sha256); concern C1 below |
| C5 | was metrics/ registry/ paper/ touched? | **PASS** | newest mtime under metrics/+registry/ = `registry/metrics.yaml` @ 2026-11-01 10:40:55, **65 min before** the branch's files (11:46/11:47); `git status --porcelain paper/` empty |
| C6 | was a GPU used? | **PASS** | the GPU-violation log still has exactly its 24 historical lines (pid=700241, 550 MiB), mtime 2026-10-31 17:52 -- no NEW line |
| C7 | "ruff clean" | **PASS** | pinned ruff v0.6.9 from the pre-commit env: `All checks passed!` |

### C1 -- the re-derivation (the decisive check)

Re-implemented three ways: `rdMolStandardize.LargestFragmentChooser`; a first-principles
total-atom-count-with-hydrogens rule (ties to the last fragment); the max-heavy-atoms rule.
Metal vocabulary re-read with `ast` from `molmetal/molmetal_lam/lam_chem/platinai_dataset.py`.
Source of truth: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` (12669 data rows).

| quantity | branch | independent | published artifact |
|---|---|---|---|
| buckets <=2 / 3-10 / >=11 heavy | 2663 / 2 / 10003 | **2663 / 2 / 10003** | 2663 / 2 / 10003 |
| heavy mean / median / min / max | 34.39161667192927 / 38.0 / 1 / 89 | **same** | same |
| raw single-fragment | 1078 (0.08509630565203663) | **same** | same |
| raw / kept heavy total | 561976 / 435673 | **same** | same |
| mass discarded | 0.22474803194442466 | **same** | 0.22474803194442472 (delta 5.55e-17, float-expression ULP) |
| false-valid rate | 2663/12668 = 0.210215 | **0.21021471424060625** | -- |
| speck composition | 2602 C + 61 Si, 0 with a metal, 0 single-fragment raw, raw-frags 21-89 | **same** | -- |
| ligand-only analogue | 2663 le2, 0 new, 2663 largest<=2, 7040 split | **same** | -- |
| metal content of survivors | 9399 with / 3269 without (606 drug-sized) | **same** | -- |
| UFF | 0 ok, 12668 x "ValueError: Bad Conformer Id" | **same** | same |
| chooser vs max-heavy | 261 different picks, 2 different sizes, rule agrees 12668/12668 | **same** | -- |

### C3 -- mutation battery (the branch's own test file, no branch file modified)

| mutant | result |
|---|---|
| M1 survivor heavy <- raw heavy | 3 failed (2 negative controls + drift check) |
| M2 numerator copied from the published artifact | 4 failed (3 negative controls + vacuity guard) |
| M3 metal vocabulary polluted with C and Si | 2 failed (1 negative control + drift check) |
| M7 survivor replaced by the whole molecule | 3 failed (2 negative controls + drift check) |
| M4 UFF block made vacuous | **15 PASSED -- not caught** |
| M5 per-epoch axis collapsed to epoch 0 | **15 PASSED -- not caught** |
| M6 mass retention inverted | **15 PASSED -- not caught** |

## Surviving concerns

**C1 (MEDIUM) -- the comparability paragraph splices two catalog sections and drops the
total METRIC break.** `report.md` section 7 attributes a "POST-PROCESSING break" to
`HONEST_NEGATIVE_CATALOG.md` **6.264**; section 6.264 contains **zero** occurrences of the
string "post" -- its four breaks are object / denominator / chemistry / bond-rule+raw-geometry.
The authoritative four-break set (`DENOMINATOR, METRIC, POST-PROCESSING, HYDROGENS`) lives in
**6.265** section 4 and on disk at
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` (`"breaks": [...]`).
Two consequences: (a) the branch's "the other three breaks" are 6.264's breaks 1-3, not
6.265's set, and the **METRIC break -- which the on-disk matrix calls "a total METRIC break"
("THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL ... NEITHER is a
connected-component count") -- is omitted entirely**; (b) "the frontier band is
POST-PROCESSED" is contradicted for the connectivity cell by the same matrix's EF-TALFM cell
("The C. = 92.98 % cell is measured ... BEFORE the largest-component retention ... a
post-retention count would be 100 % by construction, and the paper reports 92.98 %"), and by
its HLTF cell (the headline pair straddles raw 85.5 % and post-processed 92.2 %). The
HYDROGENS bullet is also inverted for this arm: the branch's own pipeline step 2 is
`Chem.AddHs` (explicit H) and the frontier recipe in the matrix is "largest fragment +
explicit hydrogens + UFF relaxation" -- the same family. NOT a refutation: the branch quotes
no frontier percentage, never places the 100 % beside the band, and its `reader_conclusion`
forbids quoting the 100 % alone. But a downstream writer trusting section 7 would carry a
mis-cited break list and would miss the break that forbids the comparison outright.

**C2 (MEDIUM) -- the artifact's UFF note carries a false causal clause.**
`falsevalid_axis.json` -> `uff` -> `note`: "the pool is SMILES-derived and carries no
conformer, **and UFF has no parameters for the metals**". Measured otherwise: after
`AllChem.EmbedMolecule(ETKDGv3)`, `UFFOptimizeMolecule` succeeds on **60/60** metal-bearing
survivors that embedded (metal elements Ir, Os, Pt, Re, Rh, Ru; 0 UFF failures) and on 4/4
metal-free survivors. The measured cause of all 12668 failures is the **missing conformer
alone**; the metal is not a blocker in RDKit's UFF. (The clause is inherited from
`metrics/by_metric/cfm_fragmentation_postproc_arm.json`'s `uff_note`.) The NUMBER it
accompanies -- `uff_ok_rate = 0.0`, 12668 x "ValueError: Bad Conformer Id" -- is confirmed.

**C3 (LOW_MEDIUM) -- the per-epoch prose describes a non-monotone series as a decline.**
"the speck rate falls from 0.5639 (epoch 0) to 0.0085 (epoch 17)": the series also **rises**
at epoch 1 (0.7443 -- the MAX), epoch 9 (0.1236) and epoch 16 (0.0284). The artifact's own
`per_epoch_summary.max_epoch_speck_rate = 0.7443181818181818` holds the peak, but the prose
never names it, and the worst epoch (74.4 %) is 3.5x the pooled 21.0 % the branch asks readers
to handle carefully. `per_epoch_summary` keys only `first_epoch`/`last_epoch`, so the peak's
epoch id (1) is not in the summary.

**C4 (LOW) -- three artifact blocks have no code-level regression guard.** The `uff`,
`per_epoch` and `mass_retention` assertions read the frozen artifact JSON; the drift check
re-derives only 7 fields on 256 sampled rows -- so M4/M5/M6 all pass 15/15. The speck
decomposition itself is guarded.

## Files

* this refutation: `molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.json` + `refutation.md`
* re-derivation (scratch, not committed): `/tmp/wfa168_refute/rederive.py`, `uff_probe.py`, `uff_probe3.py`, `mut_plugin.py`

---

## Wf-A187 CORRECTION (2026-11-01) — this refutation's quotations of the matrix's X1 wording are SUPERSEDED

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation. This section is **ADDITIVE**: every line above it is unchanged and the pre-existing
content of this file is a **byte-identical PREFIX** of the current file (AGENTS.md §5.5 rule #6 —
append-only; a superseded rendering is preserved VERBATIM and corrected ADDITIVELY + DATED).

### The superseded passage, NAMED, and quoted VERBATIM

**Superseded passage:** "Surviving concerns" → **C1 (MEDIUM)**, in the clause that quotes the A158
matrix's HLTF cell. Quoted **VERBATIM** (nothing above this section was edited):

> Two consequences: (a) the branch's "the other three breaks" are 6.264's breaks 1-3, not
> 6.265's set, and the **METRIC break -- which the on-disk matrix calls "a total METRIC break"
> ("THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL ... NEITHER is a
> connected-component count") -- is omitted entirely**; (b) "the frontier band is

The same X1 wording is quoted, VERBATIM, in the companion artefact
`molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.json` →
`surviving_concerns[0].evidence[2]`, and paraphrased in the closing clause of
`surviving_concerns[0].why_it_is_not_a_refutation` as "that the 92.7-94 % band is not a connectivity band
at all". Both are preserved unchanged inside a dated key
(`wfa187_supersession_2026_11_01`) in that JSON.

### Why they are superseded

The **matrix's LIVE rendering changed**. In
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` the live
`band_provenance.headline_finding` is now **READING-QUALIFIED**, and the X1 wording survives only
inside the dated key `band_provenance.headline_finding_superseded_2026_11_01`. The quotations above are
therefore quotations of a **superseded rendering**: they are kept for the record, and they must no
longer be presented as the matrix's current wording. **X1** -- "NEITHER is a connected-component
count" / "THE BAND IS NOT A CONNECTIVITY BAND" / "the 92.7-94 % band is not a connectivity band at all"
-- is **TRUE under Reading A** (`band_provenance`) and **FALSE under Reading B** (`sources`.`EF-TALFM`).

### What is NOT withdrawn — the direction of the concern stands

The **DIRECTION** of surviving concern C1 is **NOT withdrawn**. It remains true that "the frontier band
is POST-PROCESSED" is **contradicted for the connectivity cell** by the same matrix's EF-TALFM
POST-PROCESSING cell ("The C. = 92.98 % cell is measured on the reconstructed graph BEFORE the
largest-component retention ... a post-retention count would be 100 % by construction, and the paper
reports 92.98 %"), and it remains true that the METRIC break is the break the matrix itself calls
total. **Only the QUOTATION of the matrix's superseded wording is superseded.**

### The load-bearing conclusion, restated in reading-safe terms

Under **BOTH** competing readings of the band, the METRIC break holds: the band's endpoints are **NOT
our quantity** -- **OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing
else** -- so no post-processing alignment can rescue the comparison. And because the band's provenance
is **NOT settled on disk**, the break list may not be asserted blanket-ward: the two readings disagree
on the numerals (92.7/94.0 vs 92.98/94.33) and therefore on which frontier cells the band names.

**A "RAW" claim must be READING-CONDITIONAL.** An unconditional "both arms are RAW" is itself a
defect.

### LOCATORS

**JSON PATHS, never line numbers**: `band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. A line number is a
**perishable** locator -- the record grows, so a line-number locator into it is **stale by
construction** the moment it is written.

### This artefact is HAND-MAINTAINED -- there is NO producer (MEASURED, 2026-11-01)

* `git grep -n -I -- 'falsevalid_axis_refutation' -- '*.py' '*.sh'` -> **no hit**. No script in the
  repository produces `refutation.md` or `refutation.json`.
* `molmetal/scripts/wfa168_falsevalid_axis.py` is the **branch under review**, not this artefact's
  producer: it writes `molmetal/reports/wfa168_falsevalid_axis/{{falsevalid_axis.json,report.md}}`, a
  **different** directory.
* The whole `molmetal/reports/wfa168_falsevalid_axis_refutation/` directory is **untracked**
  (`git status --porcelain` -> `?? molmetal/reports/wfa168_falsevalid_axis_refutation/`).

**Consequence:** these artefacts are corrected **BY HAND** and **APPENDED**. No script regenerates or
re-verifies them, and there is no producer/artefact parity check to run. The dated key
`wfa187_supersession_2026_11_01` in `refutation.json` is the machine-readable counterpart of this
section.

### The coupled test file -- ONE added regression test

Wf-A187 added ONE regression test to the branch's coupled test file molmetal/tests/test_wfa168_falsevalid_axis.py -- a consumer-side guard on this dated key (the refutation artefacts are hand-maintained, so this test is their only automated guard). It touches neither the branch script nor the frozen artefact, so it cannot affect the mutation battery. The '15 passed' figure recorded in the C3 row is the count AT THE TIME OF THE REFUTATION (2026-11-01, before Wf-A187) and stands as the historical measurement; the mutation battery was NOT re-run (its mutant harness is scratch and uncommitted), so rows M1-M7 stand unchanged.

### What this correction does NOT change

**No number in this refutation moves.** Every check verdict (C1-C7), every re-derived value, the
mutation battery (M1-M7), the four surviving concerns (C1-C4) and the verdict **NOT REFUTED** are
untouched. Only the **quotation of the matrix's superseded wording** is corrected, additively and
dated.

Nothing above this section was edited or removed (AGENTS.md §5.5 rule #6 -- append-only).

---

## Wf-A194 CORRECTION (2026-11-02) — the X-class census of THIS refutation, and the closure of the defect it cites

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation. **ADDITIVE + DATED** (AGENTS.md §5.5 rule #6): every line above this section is unchanged
and remains a byte-identical PREFIX of this file.

**This artefact is HAND-MAINTAINNED** — `git grep -n -I -- 'falsevalid_axis_refutation' -- '*.py' '*.sh'`
returns **no hit** (re-measured 2026-11-02): no script produces `refutation.md` or `refutation.json`, so
there is no producer/artefact parity to run. The correction is applied **BY HAND** and recorded here, DATED.
The machine-readable counterpart of this section is the dated key `wfa194_closure_2026_11_02` in
`refutation.json`.

### 1. The X-class census of THIS artefact — every hit, classified

| file:line | rendering | class | action |
|---|---|---|---|
| `refutation.md:69-75` — the C1 clause | the A158 matrix's X1 wording, quoted | **REFUTATION-QUOTE** | **not edited** — a refutation quoting a defect is not a claim — **and SUPERSEDED**: the very same clause is quoted VERBATIM in this file's `## Wf-A187 CORRECTION (2026-11-01)` section, under the heading `### The superseded passage, NAMED, and quoted VERBATIM`, where the same X1 wording is DATED as a quotation of the matrix's now-superseded live rendering (see `### Why they are superseded`). **Row amended by Wf-A202 (2026-11-02)**; the prior rendering of this row is preserved VERBATIM in `refutation.json` → `wfa202_closure_2026_11_02.r4_census_row_prior_rendering_verbatim` and in the `## Wf-A202 CORRECTION (2026-11-02)` section below |
| `refutation.md:133` | "that the 92.7-94 % band is not a connectivity band at all" | **DATED-SUPERSESSION** (inside `## Wf-A187 CORRECTION (2026-11-01)`) | not edited |
| `refutation.md:145` | the X1 list | **MUST-NOT-ASSERT-LIST** (inside the Wf-A187 section) | not edited |
| `refutation.md:165` | "both arms are RAW" | **DATED-SUPERSESSION** (inside the Wf-A187 section) | not edited |
| `refutation.json` → `surviving_concerns[0].evidence[2]` | the matrix's X1 wording, quoted | **REFUTATION-QUOTE** | not edited |
| `refutation.json` → `surviving_concerns[0].why_it_is_not_a_refutation` | a paraphrase of X1 | **REFUTATION-QUOTE** | not edited |
| `refutation.json` → `wfa187_supersession_2026_11_01` (4 entries) | the same renderings, preserved | **DATED-SUPERSESSION** | not edited |

**Result: this artefact holds ZERO live X-class claims.** Every X-class rendering it carries is a quotation
of a defect inside a refutation, a must-not-assert entry, or a dated supersession — none of which is a live
claim. Editing them would destroy the record of what the matrix's superseded wording said, which is the whole
point of a refutation record (AGENTS.md §5.5 rule #6).

### 2. The direction of surviving concern C1 is NOT withdrawn

What the Wf-A187 section superseded was the **quotation of the matrix's superseded wording**, not the concern.
The concern still stands in reading-safe terms: under **BOTH** competing readings of the band, the band's
endpoints are **NOT our quantity** — **OUR 8.51 % IS a connected-component count on the RAW emitted SMILES,
and nothing else** — so no post-processing alignment can rescue the comparison, and the METRIC break is the
break the matrix itself calls total. The band list may not be asserted blanket-ward: the two readings
disagree on the numerals (92.7/94.0 vs 92.98/94.33) and therefore on which frontier cells the band names.

**A "RAW" claim must be READING-CONDITIONAL.** An unconditional "both arms are RAW" is itself a defect.

### 3. The reading-safe replacement — quoted VERBATIM from the record

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
(`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
`band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
`sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
connected-component count on the RAW emitted SMILES, and nothing else.

### 4. The defect this refutation's C1 and the axis-mapping synthesis's D1 cite is now CLOSED at source

Both cite `molmetal/reports/wfa161_frontier_labelling/report.md:139`, where the live `## Comparability
breaks` item 2 asserted that the frontier's connectivity rates are computed on the atom-count largest
connected component after RDKit sanitization (EF-TALFM) and that the published band comes from a DIFFERENT
perception branch (Open Babel). **Wf-A194 (2026-11-02) repaired that producer and re-published that
artefact**: the live item 2 now carries the reading-safe form, and the superseded sentence survives VERBATIM
in the artefact's own `## Wf-A176 CORRECTION (2026-11-01)` section. The sentence therefore sits at **no fixed
line number** — cite the section, not a line.

### 5. What this correction does NOT change

**No number in this refutation moves.** Every check verdict (C1-C7), every re-derived value, the mutation
battery (M1-M7), the four surviving concerns (C1-C4) and the verdict **NOT REFUTED** are untouched. The
`refutation.json` gains exactly ONE dated key and changes no existing leaf; its numeric-leaf path set is
byte-identical before and after. Only the **quotation record** is annotated, additively and dated.

Nothing above this section was edited or removed (AGENTS.md §5.5 rule #6 -- append-only).


## Wf-A202 CORRECTION (2026-11-02) — the census pointer (R4), the byte-identity of the "verbatim" quote (R3), the guard's exemption window (R2), and the producer-artefact parity of the dated docstring (D4)

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation, no file under `metrics/` touched.

**ADDITIVE + DATED** (AGENTS.md §5.5 rule #6), with ONE declared in-place amendment:

- Every line ABOVE the `## Wf-A194 CORRECTION (2026-11-02)` section is unchanged and remains a
  **byte-identical PREFIX** of this file.
- **ONE** line inside the Wf-A194 section was amended in place: the `refutation.md:69-75` census
  row, which was missing the pointer at the Wf-A187 supersession. Its **prior rendering is
  preserved VERBATIM** in §2 below and in `refutation.json` →
  `wfa202_closure_2026_11_02.r4_census_row_prior_rendering_verbatim`.
- Everything else in this section is new.

### 1. R4 — the census row now POINTS at the Wf-A187 supersession

**BEFORE** — the row as it stood, preserved VERBATIM (this is the whole of the prior rendering of
that row):

```
| `refutation.md:69-75` — the C1 clause | the A158 matrix's X1 wording, quoted | **REFUTATION-QUOTE** | **not edited** — a refutation quoting a defect is not a claim |
```

The JSON's prior rendering of the same census row is a different string — it is the JSON's own
`x_class_census_of_this_artefact` prose, not the markdown row — and it is preserved VERBATIM in
`refutation.json` → `wfa202_closure_2026_11_02.r4_census_row_prior_rendering_verbatim`:

```
ZERO live X-class claims. refutation.md:69-75 (the C1 clause) = REFUTATION-QUOTE; refutation.md:133 = DATED-SUPERSESSION (inside the Wf-A187 CORRECTION section); refutation.md:145 = MUST-NOT-ASSERT-LIST; refutation.md:165 = DATED-SUPERSESSION; surviving_concerns[0].evidence[2] = REFUTATION-QUOTE; surviving_concerns[0].why_it_is_not_a_refutation = REFUTATION-QUOTE; wfa187_supersession_2026_11_01 (four entries) = DATED-SUPERSESSION. None was edited.
```

**AFTER** (the row as it now stands):

```
| `refutation.md:69-75` — the C1 clause | the A158 matrix's X1 wording, quoted | **REFUTATION-QUOTE** | **not edited** — a refutation quoting a defect is not a claim — **and SUPERSEDED**: the very same clause is quoted VERBATIM in this file's `## Wf-A187 CORRECTION (2026-11-01)` section, under the heading `### The superseded passage, NAMED, and quoted VERBATIM`, where the same X1 wording is DATED as a quotation of the matrix's now-superseded live rendering (see `### Why they are superseded`). **Row amended by Wf-A202 (2026-11-02)**; the prior rendering of this row is preserved VERBATIM in `refutation.json` → `wfa202_closure_2026_11_02.r4_census_row_prior_rendering_verbatim` and in the `## Wf-A202 CORRECTION (2026-11-02)` section below |
```

**Why the pointer was missing.** The `refutation.md:69-75` clause is the C1 clause that quotes the
A158 matrix's X1 wording ("NEITHER is a connected-component count") inside the attribution "which
the on-disk matrix calls". That SAME clause is quoted VERBATIM, as a blockquote, in this file's
`## Wf-A187 CORRECTION (2026-11-01)` section under the heading `### The superseded passage,
NAMED, and quoted VERBATIM`, where it is DATED as a quotation of the matrix's now-superseded live
rendering (the matrix's live `band_provenance.headline_finding` was repaired by Wf-A187). The
census row said only "not edited", which is true but incomplete: it did not say that this
refutation's own quotation is itself now dated. The row now says both.

### 2. R3 — the "quoted VERBATIM" claim, measured byte-for-byte

Three renderings of the same sentence exist on disk. All three were measured, not eyeballed:

| rendering | path | chars | backticks | double spaces | byte-identical to the source? |
|---|---|---|---|---|---|
| **SOURCE** | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` → `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement` | **951** | 14 | 0 | — (it IS the source) |
| `refutation.json` → `wfa194_closure_2026_11_02`.`reading_safe_replacement_verbatim` (BEFORE Wf-A202) | `refutation.json` | **939** | 0 | 2 | **NO** — the backticks had been stripped and two double spaces collapsed to one |
| `refutation.md` §3 "The reading-safe replacement — quoted VERBATIM from the record" (the Wf-A194 section) | `refutation.md` | **953** unwrapped | 14 | 2 | **NO** — same words, but two double spaces where the source has single spaces |

**The JSON key was NAMED `..._verbatim` and was not verbatim.** Wf-A202 (2026-11-02) replaced its
value with the source string **byte-for-byte**:

- characters **939 → 951**
- sha256 `228a80b3a047d601b514dcc05b951a1689aa6e6d0625e846222708e20883ab65` →
  `ba280c4e9b1d9da225b5e8c06a0d0b07bc9e202a2c61bc2706336fcc816bd041`

The prior 939-character rendering is preserved VERBATIM in `refutation.json` →
`wfa202_closure_2026_11_02.r3_reading_safe_replacement_verbatim_superseded_2026_11_02_wfa194`,
and the **source file was NOT edited** — `molmetal/reports/wfa158_frontier_postproc_matrix/`
belongs to another concurrently-running wave.

**The `refutation.md` §3 blockquote was NOT rewritten in place**: it sits inside the dated
Wf-A194 section, and rule #6 forbids an in-place rewrite of a dated block. It is 953 characters
unwrapped against the source's 951 — the word sequence is identical and the two differences are
both double-space-vs-single-space — and it is left exactly as Wf-A194 published it. The
machine-readable copy now carries the exact bytes.

### 3. R3 — the source's OWN count is not settled on disk, and that conflict is NOT resolved here

The byte-identical source string asserts a **settled count**: it says the matrix record "holds TWO
competing readings" and reasons "Under BOTH readings ...". Byte-identity and count-neutrality are
therefore in tension, and the tension is resolved BY SEPARATION, not by editing the source:

- the quoted bytes are kept byte-identical to the source, so the key that is NAMED "verbatim"
  really is verbatim;
- the count caveat is recorded in a dated sibling key
  (`refutation.json` → `wfa202_closure_2026_11_02.r3_the_sources_own_count_is_not_settled_on_disk`).

The repo's own canonical clause on the count, quoted VERBATIM from
`molmetal/reports/wfa175_synthesis/_write_md.py` → `_X5_COUNT_NOT_SETTLED` (Wf-A196; read by AST
parse — the module was **not** imported, because its module-level code rewrites
`molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md`, a file this wave does not own):

> The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals.

**XW-A202-01 — NAMED CROSS-WAVE FINDING** (owner: the wave that owns
`molmetal/reports/wfa158_frontier_postproc_matrix/`): the canonical reading-safe statement at
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`
asserts a settled count of TWO competing readings, while the repo's own Wf-A196 clause says the
NUMBER of competing readings is NOT settled on disk and the primary-source verification
(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` →
`invariants.competing_readings`) records a FURTHER reading. Wf-A202 did not edit that tree and
does not resolve the conflict; the count-neutral wording belongs to the owning wave.

### 4. R2 — the guard's exemption window is now STRUCTURAL, not a character radius

`molmetal/tests/test_wfa194_x_closure_scope2.py` → `_statement_window` used a fixed ±300
**character** radius. A radius cannot tell "the qualifier belongs to THIS statement" from "the
qualifier belongs to the statement NEXT DOOR": a rendering inserted immediately after a
reading-conditional line inherited that line's qualifier and was silently exempt. The window is now
the enclosing **STATEMENT** — the nearest sentence terminators on either side, clipped at the hard
markdown boundaries (blank line / heading / list item / table row).

MEASURED on the repo corpus, the live FORBIDDEN matches are two, and both are still exempt for a
legitimate reason:

| site | class | exemption | why it is legitimate |
|---|---|---|---|
| `synthesis.md` line 127 | X1 | the qualifier `THIS reading` | it is IN THE SAME STATEMENT: "Under THIS reading no endpoint is a connected-component count." |
| `synthesis.md` line 334 (the D1 row) | X3 | the defect marker `**contradicted**` | it is IN THE SAME STATEMENT: the row that NAMES the defect |

Against a copy carrying one rendering inserted immediately after the reading-conditional line 129,
the OLD radius window exempted the inserted line via `Reading 2` and `THIS reading` bleeding
across two sentence boundaries; the NEW statement scope does not, and the guard FAILS as it must.

### 5. D4 — producer-artefact parity of the dated Wf-A194 docstring (NOT repaired here)

**XW-A202-02 — NAMED CROSS-WAVE FINDING** (owner: the wave that owns
`molmetal/scripts/wf_a155_ligand_connected_rate.py` and
`molmetal/scripts/wf_a155_posebusters_validity.py`): both producers define
`DOCSTRING_SUPERSEDED_2026_11_02_WFA194`, and both emit their OTHER dated rule-#6 records into
their artefacts — but NEITHER emits the Wf-A194 dated docstring record, and NEITHER published
artefact carries the superseded docstring text. A constant that exists but is never emitted is
the producer-artefact parity defect class: a re-run drops the dated history with no diff to
review.

The producers' own constant records the refusal (`emitted_into_the_artifact: "no -- deliberately"`,
with the reason that emitting it would change the artefact's bytes, its sha256 and its registry
row without changing a measurement), so the disposition is the owning wave's call — emit it and
re-publish, or record the refusal as a named exception.

Wf-A202 changed the TEST, not the producers: the assertion now tests that the producer **emits**
the dated docstring into its artefact, instead of merely that the constant exists. That test
**FAILS** as of this writing, by design:

```
molmetal/tests/test_wfa194_mirror_closure.py::test_wfa194_docstring_supersession_is_emitted_into_the_published_artifact
```

A green test that only checks a constant's existence IS the defect, so no such green test is
shipped.

### 6. What this correction does NOT change

**No number in this refutation moves.** Every check verdict (C1-C7), every re-derived value, the
mutation battery (M1-M7), the four surviving concerns (C1-C4) and the verdict **NOT REFUTED** are
untouched. `refutation.json` gains ONE dated key (`wfa202_closure_2026_11_02`), and every leaf of
it is a STRING — the numeric-leaf path set is byte-identical before and after (81 numeric leaves).
Only the **quotation records** are annotated, additively and dated.

**No rendering was edited.** The C1 clause, the D1 row and the matrix quotations are exactly as
Wf-A187 and Wf-A194 left them; what changed is a POINTER and a byte-identical copy.

### 7. What Wf-A202 MEASURED (2026-11-02)

**The tightening changed no existing verdict.** `molmetal/tests/test_wfa194_x_closure_scope2.py`
collects **17 cases** (12 test functions; `test_synthesis_replaced_rendering_survives_verbatim_in_a_dated_position`
is parametrised over 6 keys and `test_wfa171_replaced_rendering_survives_verbatim_in_a_dated_position`
over 1). All 17 pass with the repaired statement window, and all 17 also passed with the OLD ±300
character radius window in place — the defect was that the OLD window was green for the WRONG reason,
not that it was red. Of the 17, **4 exercise the window** (`test_synthesis_...`, `test_wfa171_...`,
`test_refutation_md_...`, `test_wfa161_report_...`); the fifth guard (`test_refutation_json_...`)
scans each JSON string leaf whole and never calls `_statement_window`; the other 12 are
dated-position / verbatim / producer-constant / JSON-leaf tests that never call the guard. The 4
window-exercising guards stay green because the corpus carries **3** live FORBIDDEN matches in total
and each one's exempting marker sits in the SAME statement:

| corpus | live matches | exemption | statement window | old ±300 radius window |
|---|---|---|---|---|
| `synthesis.md` line 127 | 1 (X1) | `THIS reading` | 62 chars | 642 chars |
| `synthesis.md` line 334 (the D1 row) | 1 (X3) | `**contradicted**` | 324 chars | 690 chars |
| `refutation.md` line 70 | 1 (X1) | `on-disk matrix calls` | 443 chars | 638 chars |
| `wfa171` report.md, `wfa161` report.md | 0 | — | — | — |

**The bite results.** Each repair was reverted, the matching residual test was observed to FAIL, and
the file was restored with its sha256 unchanged:

| repair | reverted | residual test | result | restored sha256 (at bite time) |
|---|---|---|---|---|
| R2 | `_statement_window` → the old ±300 character radius | `test_r2_a_rendering_inserted_next_to_a_reading_conditional_line_is_caught` | FAILED (as required) | `402af4e8…c018bd` |
| R2 | the same | `test_r2_the_exemption_window_does_not_span_the_neighbouring_statement` | FAILED (as required) | `402af4e8…c018bd` |
| R3 | the "verbatim" key → the prior 939-char string | `test_r3_the_reading_safe_replacement_key_is_byte_identical_to_the_source` | FAILED (as required) | `a1f5c361…b11f4` |
| R4 | the census row → the prior "not edited" wording | `test_r4_the_md_census_row_points_at_the_wfa187_supersession` | FAILED (as required) | `348d1d85…bcaeb7` |
| D4 | the emission assertion → the constant-existence-only assertion | `test_d4_the_mirror_guard_asserts_emission_not_merely_constant_existence` | FAILED (as required) | `bd967a17…e58efed` |

**The `metrics/` baseline drifted, and NOT because of this wave.** The required invariant was
`git status --porcelain metrics/ | md5sum` = `a022e54fd2938b9f333fd8438f6d3d27` before and after.
The value observed during the wave was `753b04b2a241039eb538601f35bb0504`. A brute-force search over
every 1-, 2- and 3-line removal from the observed porcelain reproduced the required baseline with
**exactly one** match: the two lines ` M metrics/by_metric/metal_lomo_auc_wasserstein.json` and
` M metrics/by_metric/nci60_precious_9cell_auc.json`. Each of those two files gained a single
`headline_caveat` JSON key against HEAD in a CONCURRENT wave (mtimes 2026-11-02 02:33:34 and
02:31:08). Wf-A202 wrote NO file under `metrics/`; the residual test therefore compares the porcelain
**with those two named paths EXCLUDED**, which is stable whether the concurrent wave's edit stands or
is reverted. NAMED cross-wave observation, not a Wf-A202 defect.
