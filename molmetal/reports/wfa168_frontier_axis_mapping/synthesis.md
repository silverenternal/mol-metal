# Wf-A168 — SYNTHESIS: the two frontier axes (HLTF + EF-TALFM) mapped onto our c7 pool

**Status:** 0 GPU (CPU-only, `CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
read-only w.r.t. every tree except this file. Every number below traces to a file on disk or to a retrieval
record in a branch `sources.json`. Three branches + three adversarial refutations were run; **all three
branches were returned NOT REFUTED**. This document is the SYNTHESIS layer only — it introduces no new
measurement.

---

## 0. VERDICT (one paragraph)

The frontier papers make **two** axes available, and our repo sits on **one** of them with a measured
price that the frontier does not print. Axis 1 (**POST-PROCESSING**) is a shared, real effect
(HLTF +6.7 pts on GEOM-DRUGS; ours +91.49 pts) whose *magnitudes are NOT comparable* — ours is
fragment-selection **step 1 only** on a stream that was 8.51 % single-fragment to begin with, and the
price is 22.47 % of raw heavy mass discarded and 21.02 % of survivors left as ≤2-heavy-atom specks.
Axis 2 (**FALSE-VALID**) exists in HLTF only, and HLTF never *counts* it — it infers it from the
PB-Valid-vs-Val gap; our 21.02 % is a **count**, and is an ANALOGUE, not the same metric (different
direction, class, measurement and population). The repo's published **92.7–94 % band's provenance is
NOT settled on disk**: the repo holds **TWO competing readings** of which frontier cells the band
names, and they do not even agree on the numerals, so no single provenance claim about it may be
asserted blanket-ward. Under BOTH readings the band is **EXTERNAL**, is **NOT one homogeneous
quantity**, and is **NOT on our scale** — OUR 8.51 % IS a connected-component count on the RAW emitted SMILES.

---

## 1. PINNED FRONTIER NUMBERS

### 1.1 HLTF — arXiv:2603.27113v1 (Awasthi / Lobo / Zhukov; accepted ICLR 2026 AI4Mat Workshop)

**Retrieval record (live, primary source):**
`https://arxiv.org/e-print/2603.27113` → HTTP **200**, final `https://arxiv.org/src/2603.27113`,
**504186 B**, sha256 `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329`,
2026-11-01T03:40:28Z.
`https://arxiv.org/abs/2603.27113` → HTTP **200**, **41764 B**, sha256
`374ece252cad92826dcc1f36a321ca971a650a7897acb6eec57581502d714804`.
Record + per-pin needles/sites: `molmetal/reports/wfa168_hltf_axis/sources.json` (15/15 pins, 13/13
context pins; the refutation re-ran `pin_all` against its own download → 15/15, 0 site mismatches).

**Pins (value | site | denominator | post-processing | hydrogens):**

| pin | value | site | denominator | post-proc | H |
|---|---|---|---|---|---|
| QM9 Atom stability | 98.8 ± 0.1 % | `tables/qm9_benchmarks_og.tex:41` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 Molecule stability | 86.8 ± 0.3 % | `tables/qm9_benchmarks_og.tex:41` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 Validity | 93.8 ± 0.6 % | `tables/qm9_benchmarks_og.tex:41` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 Validity&Uniqueness (HLTF col) | 92.9 ± 0.2 % | `tables/qm9_benchmarks_og.tex:41` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 Validity&Uniqueness (GeoLDM col) | **92.7 ± 0.5 %** | `tables/qm9_benchmarks_og.tex:41` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 PB-Valid (HLTF col) | **94.0 ± 0.5 %** | `tables/qm9_benchmarks_og.tex:66` | 10,000 QM9 samples | NONE (raw) | not stated |
| QM9 Novel | 56.0 ± 0.7 % | `tables/qm9_benchmarks_og.tex:65` | 10,000 QM9 samples | NONE (raw) | not stated |
| GEOM-DRUGS Valid raw → +PP | **85.5 → 92.2 %** | `tables/geom_drugs_results.tex:25` (raw) / `:26` (+PP) | 100,000 GEOM-DRUGS samples | raw = NONE; +PP = largest fragment + explicit H + UFF (`sections/experiments.tex:8`) | raw implicit; +PP explicit |
| GEOM-DRUGS Valid&Uniq raw → +PP | **85.4 → 92.0 %** | `tables/geom_drugs_results.tex:25` / `:26` | 100,000 | as above | as above |
| GEOM-DRUGS Valid&Uniq&Novel raw → +PP | **85.0 → 91.2 %** | `tables/geom_drugs_results.tex:25` / `:26` | 100,000 | as above | as above |

Invalidity breakdown "Disconn. 35 % → 12 %" is a share of **INVALID samples only**
(`sections/appendix.tex:516,518`) — it is NOT a connectivity rate over all samples.

### 1.2 EF-TALFM — arXiv:2609.08333v4 (Yao / Gruich / Goldsmith / Wang; v4 = 21 Oct 2026)

**Retrieval record (live, primary source):**
`https://arxiv.org/e-print/2609.08333v4` → HTTP **200**, **8508562 B**, sha256
`e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224`;
`https://arxiv.org/abs/2609.08333v4` → HTTP **200**, **41623 B**, sha256
`54e9eb0a9a7cda5f2e46d48d6e44a5ce769c6661b906b5b29d3dba17c6c18a90`.
Record: `molmetal/reports/wfa168_eftalfm_axis/sources.json`. The refutation re-fetched both
independently: **byte-identical digests**, all 26 cited lines character-for-character, and
re-ran the branch script from its own cache → `sources.json` diff **0 lines** ⇒ not hand-edited.

**The four pinned numbers (all four FOUND; none NOT-FOUND):**

| pin | value | what it is | site | denominator |
|---|---|---|---|---|
| **P1** | **87.90 %** | "PoseBusters-verified novel yield" = unique + novel + sanitized + PoseBusters-pass | `src/abstract.tex:1`; body `src/results_unconditional.tex:35`; metric def `src/appendix.tex:53` | **ALL** generated samples (10,000) |
| **P2** | **92.98 %** | Connectivity "C." = *"% of molecules with exactly one connected component, i.e., one fragment"* | table row `src/appendix.tex:115`; definition `src/appendix.tex:60` | **ALL** generated samples (10,000) |
| **P3** | **94.33 %** | N.&U.&S. = sanitized + nonduplicate + absent-from-training, **before** the joint PoseBusters pass | caption `src/appendix.tex:89`; table row `src/appendix.tex:127` | **ALL** generated samples (10,000) → **9,433** molecules |
| **P4** | **93.13 %** | "All PoseBusters checks" jointly (six individual rates 97.22–99.93 %) | row `src/appendix.tex:171`; panel (b) caption `src/appendix.tex:92` | the **N.&U.&S. SUBSET (9,433)**, NOT all samples |

**The abstract/table/caption reconciliation (this is the 87.9 vs 92.98 vs 94.33 question):** these are
**three different quantities over three different denominators**, and the paper is internally
consistent:

* 87.90 % is the paper's **primary metric**, over all 10,000 samples.
* 94.33 % is the N.&U.&S. count (9,433 / 10,000), also over all samples but *without* the joint
  PoseBusters pass.
* 93.13 % is the joint PoseBusters pass **over the 9,433 subset only**.
* Arithmetic chain: `9433 × 93.13 % / 10000 = 87.8495 % ≈ 87.90 %` — the abstract headline is
  reconstructible from the two table cells.
* **92.98 %** is a *different axis* from all three: it is **connectivity** (one connected component),
  counted **RAW**. Evidence it is raw: (i) the pipeline always retains the largest component, so a
  post-retention connectivity count would be **100 %** by construction and the paper reports
  92.98 %; (ii) `src/appendix.tex:60` defines it as "exactly one connected component, i.e., one
  fragment"; (iii) the paper's own Appendix-B property-conditioned population counts "92,221
  connected molecules" whose graphs "pass strict sanitization **before** geometry refinement".
  The *additional-checkpoints* caption at `src/appendix.tex:183` grammatically attaches the
  largest-component-retention clause to connectivity too — that caption is **ambiguous, not the
  number**.
* **Denominator rule (quotable, `src/appendix.tex:55`):** *"Connectivity, sanitization, unique
  sanitized yield and novel unique sanitized yield use all generated samples as their denominator;
  individual PoseBusters check rates use the novel, unique, sanitized subset."*
* **The "89.4 %" search-summary number is WRONG — NOT-FOUND:** literal `89.4` occurs **0×** in the
  e-print and **0×** on the abs page. The paper says 87.9 %. (The `0.8942` coincidence is real and
  lives at `src/appendix.tex:832`.)
* **Bond rule (correction, already catalogued in §6.265):** `src/appendix.tex:1043`
  `\tau_{ij} \gets \lambda_\pi (r^{cov}_i + r^{cov}_j)` — `\lambda_\pi` occurs **exactly once** in the
  source and **no** `lambda*pi` product exists ⇒ **π is a SUBSCRIPT (the policy π), not a factor**.
  Our repo deviation is a **flat 3.5 Å** candidate cutoff (`yuelbond_decoder.py:263`); the Cordero
  table (19 elements, same as `molmetal/data/batch.py`) has max pair-sum **2.78 Å** (I–I, Z=53), so
  at λ=1 the paper's rule is **STRICTER** than our flat cutoff; the minimum λ reaching 3.5 Å is
  **3.5/2.78 = 1.2590**. The "all-atom" branch does not use τ (Open Babel, `:1016`, `:1018`).
  Our "Appendix C.2.1, Algorithm 3" citation is **CORRECT** (C.2.1 = "Molecular graph
  reconstruction", `:987`; the float is written under C.2.2 `:1028` but introduced in C.2.1 `:1006`;
  the reconstruction float is the 3rd `\begin{algorithm}` → Algorithm 3).

### 1.3 What the repo's published **92.7–94 %** band actually is (Wf-A158, on disk)

`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` → JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement` — the LIVE rendering of `band_provenance`.`headline_finding` is the same reading-safe clause; the X1 wording survives only at `band_provenance`.`headline_finding_superseded_2026_11_01`.

* **Reading 1 candidate cells** (`band_provenance`.`endpoint_candidates`): **92.7 %** = HLTF
  `arXiv:2603.27113v1` Table 1(b) row **"Val&Uniq (%)"**, **GeoLDM** column — a per-molecule
  *RDKit-sanitisable AND non-duplicate* rate over 10,000 QM9 samples; **94 %** = HLTF Table 1(b) row
  **"PB-Valid (%)"**, **HLTF (ours)** column — a **six-check PoseBusters pose verdict** (bond
  lengths/angles, steric clash, aromatic & double-bond flatness, internal energy) over 10,000 QM9
  samples, which *contains* a connectivity check as one of six but is not that check's rate; plus
  YuelBond (bioRxiv 10.1101/2025.05.06.652517v1) 92.7 % as a **per-bond-type classification F1**
  over 1,000 coordinate-noised molecules. Under THIS reading no endpoint is a connected-component count.
* **Reading 2 candidate cells** (`sources`.`EF-TALFM`): EF-TALFM arXiv:2609.08333v4 Table 3(a) RAW
  connectivity `C. = 92.98 %` (counted BEFORE the largest-component retention) + the post-processed `N. & U. & S. = 94.33 %` companion cell. Under THIS reading the lower endpoint IS a connectivity count.

---

## 2. THE TWO AXES THE FRONTIER MAKES AVAILABLE, AND WHERE OUR NUMBERS LAND

### 2.1 AXIS 1 — POST-PROCESSING (shared axis; magnitudes NOT comparable)

| | frontier | ours |
|---|---|---|
| HLTF effect | Valid **85.5 → 92.2 %** (+6.7 pts); V&U 85.4 → 92.0 (+6.6); V&U&N 85.0 → 91.2 (+6.2) — `tables/geom_drugs_results.tex:25-26`, N=100,000 GEOM-DRUGS samples, PP = largest fragment + explicit H + UFF | raw single-fragment **0.08509630565203663 (1078/12668)** → post-processed connected **1.0 (12668/12668)** = **+91.4903694348 pts**, ratio **13.655279×** — `metrics/by_metric/cfm_fragmentation_postproc_arm.json` |
| price paid | **NOT REPORTED.** The paper prints no mass-retention and no survivor-shape number for its +PP arm; grep for mass/retention/survivor over the source = 0 hits ⇒ the fragment-selection component of its +6.7 pts is **UNQUANTIFIED in the paper** | 22.47 % of raw heavy mass discarded; 21.02 % of survivors are ≤2-heavy-atom specks |
| steps actually run | 3 (largest fragment + explicit H + UFF) | **1 of 3** — `uff_ok_rate_over_postprocessed = 0.0` (12,668/12,668 "ValueError: Bad Conformer Id") |

**COMPARABLE AT ALL? NO — as a magnitude.** The larger number says our RAW stream was *more
fragmented*, not that our post-processing is better. Four independent breaks: population
(all-organic PCQM4Mv2 latent-flow samples vs our metal complexes), denominator (10,000 samples vs
12,668 parsed rows of one concatenated 18-epoch pool), chemistry (no metals vs 100 % of rows
carrying Ru/Rh/Re/Os/Ir/Pt), bond rule, and step coverage (3 steps vs 1).

**A correction to the brief's framing of the frontier's PP:** the paper's PP is *not* "a geometry
relaxation" — its own definition (`sections/experiments.tex:8`) is the same three-step recipe our arm
implements (same source: Buttenschoen et al. arXiv:2505.00518), whose **first step IS mass-deleting
fragment selection**. The paper nonetheless attributes the gain to geometry alone and claims PP
"cannot repair … disconnected graphs" — which its own pipeline's first step does repair, by deleting
mass.

### 2.2 AXIS 2 — FALSE-VALID (HLTF only; our analogue is a COUNT, theirs is an INFERENCE)

**Frontier, verbatim.** "Explicit topology generation also reduces ``false-valid'' samples that pass
RDKit sanitization but fail stricter checks." (`iclr2025_conference.tex:55`; also `:65`, `:83`.)
"Higher PB-Valid at comparable RDKit validity indicates fewer ``false-valid'' samples that sanitize
in RDKit but violate stricter structural constraints at comparable headline validity."
(`sections/experiments.tex:25`; rationale `:27`.) **The paper never COUNTS false-valids** — it
INFERS them from the PB-Valid-vs-Val gap. All 5 "false-valid" mentions in the source are prose.

**Ours (ANALOGUE — explicitly NOT the same metric).** Our post-processed arm reports **100.00 %
"connected" (12668/12668)**, of which **2663/12668 = 21.02 %** are ≤2-heavy-atom specks — one
connected component that is still a fragment of a fragment.

**Where the analogy breaks (all four are real):**
1. **DIRECTION** — the paper's false-valid is a *second, stricter check on the SAME molecule*
   (PoseBusters on top of RDKit sanitisation). Ours scores a *DIFFERENT* molecule (the largest
   fragment), so "100 % connected" is a statement about a molecule the sampler never emitted.
2. **CLASS** — the paper's false-valid presupposes a chemically implausible but structurally *whole*
   molecule (a strained ring, a bad bond length). A ≤2-heavy speck is not a false-positive validation;
   it is a **degenerate survivor of a deletion**. Different failure class; the paper's term does not
   cover it.
3. **MEASUREMENT** — theirs is an inference over a pair of numbers; ours is a **count** over the
   survivor set.
4. **POPULATION** — the paper's own limitation excludes our corpus ("performance on unusual
   chemistries, charged/metal-containing compounds, or biomolecules remains to be established");
   its 100,000 GEOM-DRUGS samples contain no metals. Plus denominator and hydrogens differ.

### 2.3 Comparability verdict per axis

| axis | shared? | magnitudes comparable? | verdict |
|---|---|---|---|
| POST-PROCESSING | **YES** — both sides have a raw and a post-processed reading of the same recipe | **NO** (13.66× is a fragmentation of OUR raw stream, not a PP-quality score) | shared *structure*, incomparable *size*; our side carries a measured price the frontier does not print |
| FALSE-VALID | **PARTIAL** — HLTF names the concept; we can count its analogue | **NO** — different direction / class / measurement / population | ours is an ANALOGUE and must be labelled as one |
| CONNECTIVITY (the band) | **NO** | **NO** | not a connectivity band under Reading 1 (A158); under Reading 2 the lower endpoint IS a connectivity count — the band's source cells are NOT settled on disk |

---

## 3. OUR MEASURED DECOMPOSITION — WITH THE DENOMINATOR AT EVERY STEP

Source pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`
Artifact: `metrics/by_metric/cfm_fragmentation_postproc_arm.json` (cross-check) +
`molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json` (primary; re-derived from the pool).

| step | quantity | value | denominator at this step |
|---|---|---|---|
| 0 | rows in the pool CSV | 12,669 | — |
| 1 | RDKit-parseable rows | **12,668** | 12,669 CSV rows (1 unparseable) |
| 2 | RAW single-fragment (single connected component) | **1078 / 12,668 = 0.08509630565203663 (8.51 %)** | 12,668 parsed rows |
| 3 | POST-PROCESSED connected | **12,668 / 12,668 = 1.000000 (100.00 %)**; `n_multicomponent_after_pp = 0`, `n_pp_failed = 0` | 12,668 parsed rows (same denominator — the denominator does **not** change across the axis) |
| 4 | UFF (step 3 of the recipe) | **0 / 12,668 ok** (`uff_ok_rate = 0.0`), all "ValueError: Bad Conformer Id" ⇒ the arm measures **steps 1–2 only** | 12,668 post-processed rows |

**Speck split** (denominator = 12,668 survivors):
* **≤ 2 heavy atoms = 2,663** — and **every one is exactly 1 heavy atom** (histogram {1: 2663}); composition **2,602 lone C + 61 lone Si**; **0** contain a metal; **0** were single-fragment in the RAW SMILES (each came out of a 21-to-89-fragment raw molecule).
* **3–10 heavy atoms = 2**
* **≥ 11 heavy atoms = 10,003** (78.96 %)
* Survivor heavy-atom mean 34.39161667192927, median 38.0, min 1, max 89. **Bimodal**: a drug-sized molecule or a ≤2-atom speck; almost nothing in between.

**False-valid rate = 2,663 / 12,668 = 0.21021471424060625 (21.0215 %)** — the share of the 100 %
"connected" arm that is still a fragment. Denominator = 12,668 post-processed survivors.

**Ligand-only analogue** (metal atom(s) deleted from the SURVIVOR; denominator = 12,668 survivors):
* still ≤ 2 heavy after metal removal: **2,663 / 12,668 = 21.0215 %**
* **NEW specks created by metal removal: 0**
* survivors whose largest ligand fragment is ≤ 2 heavy: **2,663**
* survivors that **split into > 1 fragment** when the metal is removed: **7,040**
* metal-free survivors: **3,269** (2,663 specks + 606 drug-sized); metal-bearing survivors: **9,399**, of which **0** are specks
⇒ **the false-valid arm is NOT a metal-coordination artifact.**

**Mass retention** (denominator = raw heavy-atom mass of the 12,668-row pool):
* raw heavy total **561,976** → kept **435,673** = **0.775252** → **discarded 0.224748 (22.47 %)**
* per-molecule heavy-atom retention **mean 0.7395022201805989**, **median 0.926829268292683**
* the published `fraction_of_raw_mass_discarded = 0.22474803194442472` and the re-derivation
  `(raw-kept)/raw = 0.22474803194442466` differ by 5.55e-17 (a float-expression ULP, not a
  measurement difference).

**Epoch axis (mandatory whenever the pooled rate is quoted):** the pooled 21.02 % is a **TIME AVERAGE
over an improving generator**. Per-epoch speck rate: epoch 0 = 0.5639 → epoch 17 = 0.0085, but the
series is **non-monotone** — the **maximum is epoch 1 = 0.7443 (74.4 %, 3.5× the pooled figure)**,
with local rises at epoch 9 (0.1236) and epoch 16 (0.0284). The pooled figure must never be quoted
without the epoch axis.

**Reader conclusion (must be printed beside the 100 %):** a reader of "post-processed connected rate
= 100 %" alone would conclude all 12,668 molecules are single meaningful molecules; in fact **2,663
(21.02 %) are a SINGLE heavy atom** and the post-processing **discarded 22.47 % of the raw heavy-atom
mass** to get there. So the speck rate, the survivor-size histogram, the discarded-mass fraction and
the per-epoch axis must be printed beside the 100 %, and **the 100 % must never be quoted alone**.

---

## 4. PROPOSED HOUSE FORMAT (PROPOSAL ONLY — not applied to any metric, JSON or document)

**Template** (HLTF branch, corrected over Wf-A161's; the two new mandatory slots are the
false-valid/speck count and the mass price, and the last two slots are **axis-local** so a raw line
prints "n/a" instead of borrowing post-processed numbers):

```
connectivity {v:.2f}% = {k}/{N} single-component rows; denominator: {denominator}; post-processing: {postproc}; hydrogens: {hydrogens}; definition: exactly one RDKit connected component; specks: {specks_clause}; mass: {mass_clause}; source: {path}
```

What this adds over Wf-A161's `connectivity {v:.2f}% (single-fragment rate over N={n} decoded samples;
post-processing: …; denominator: …; hydrogens: …; definition: exactly one connected component)`:
N **and** the denominator rule become **one** slot-pair that cannot disagree (the Wf-A153 census
defect: `runs` is a LIST); a **FALSE-VALID / SPECK** slot; a **MASS-RETENTION** slot; the definition
names the tool (RDKit); the two new slots are **axis-local**.

**Filled example — our RAW arm:**

```
connectivity 8.51% = 1078/12668 single-component rows; denominator: RDKit-parseable rows of a CONCATENATED 18-epoch generation pool (12669 CSV rows, 1 unparseable) -- not one draw of N; post-processing: NONE (raw, pre-LargestFragmentChooser); hydrogens: implicit; definition: exactly one RDKit connected component; specks: n/a (no survivor set exists before post-processing); mass: n/a (nothing is discarded before post-processing); source: /home/hugo/codes/mol-metal/metrics/by_metric/cfm_fragmentation_postproc_arm.json
```

**Filled example — our POST-PROCESSED arm:**

```
connectivity 100.00% = 12668/12668 single-component rows; denominator: RDKit-parseable rows of a CONCATENATED 18-epoch generation pool (12669 CSV rows, 1 unparseable) -- not one draw of N; post-processing: LargestFragmentChooser -> explicit H -> UFF (UFF ok 0/12668: no UFF parameters for this corpus' metals); hydrogens: explicit; definition: exactly one RDKit connected component; specks: 2663/12668 = 21.02% of the survivors are <=2-heavy-atom specks (a 'connected' molecule that is a fragment of a fragment); mass: 22.47% of the raw heavy mass discarded (pooled); per-molecule heavy-atom retention mean 73.95%, median 92.68%; source: /home/hugo/codes/mol-metal/metrics/by_metric/cfm_fragmentation_postproc_arm.json
```

> ⚠️ **CORRECTION to the filled PP example (from the false-valid refutation, C2).** The clause
> *"UFF ok 0/12668: no UFF parameters for this corpus' metals"* is **FALSE as a causal claim**:
> after `AllChem.EmbedMolecule(ETKDGv3)`, `UFFOptimizeMolecule` succeeds on **60/60** metal-bearing
> survivors (Ir/Os/Pt/Re/Rh/Ru) and **4/4** metal-free ones — the sole cause of all 12,668 failures
> is the **missing conformer**. The **number** (`uff_ok_rate = 0.0`) is confirmed; the clause must be
> rewritten to "the pool is SMILES-derived and carries no conformer" before this template is applied.
> The same false clause is inherited from `metrics/by_metric/cfm_fragmentation_postproc_arm.json`
> (`uff_note`).

---

## 5. WHAT THE REPO MAY AND MAY NOT SAY ABOUT THE 92.7–94 % BAND

### MAY say

1. **"The 92.7–94 % band's provenance is NOT settled on disk: the repo holds TWO competing readings of
   which frontier cells the band names, and they do not even agree on the numerals."** Under BOTH
   readings the band is **EXTERNAL**, is **NOT one homogeneous quantity**, and is **NOT on our scale** —
   OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. Cite A158 by
   JSON PATH `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`.
2. **"Our 8.51 % and our 100 % sit on the same axis as EF-TALFM's 92.98 % (raw connectivity)."**
   True — same metric definition ("exactly one connected component"), and both RAW.
3. **"EF-TALFM's 92.98 % is a RAW connectivity count, measured before largest-component retention."**
   True, with the three independent pieces of evidence in §1.2.
4. **Name the four breaks (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS) whenever our rate is
   printed.** Authoritative source: `docs/HONEST_NEGATIVE_CATALOG.md` §6.265 §4 + A158
   `frontier_matrix.json` (`"breaks": ["DENOMINATOR","METRIC","POST-PROCESSING","HYDROGENS"]`).
5. **Say our post-processing has a measured price and the frontier's does not.** "HLTF prints no
   mass-retention and no survivor-shape number for its +PP arm, so the fragment-selection component
   of its +6.7 pts is UNQUANTIFIED in the paper; we can say it is not reported, we cannot say it is
   small."

### MAY NOT say

1. **Never write "92.7–94 %" beside any of our rates without naming all four breaks.** §6.265 §4:
   *"Any document that writes '92.7–94 %' beside one of our rates without naming these four is
   mis-framed."*
2. **Never assert "the band is POST-PROCESSED" as a blanket claim** — §6.265 §4's POST-PROCESSING
   row says so, but the band's **source cells are NOT settled on disk**: under Reading 1 HLTF's
   92.7 % and 94.0 % candidates are RAW (`tables/qm9_benchmarks_og.tex:41,66`, no post-processing);
   under Reading 2 EF-TALFM's 92.98 % candidate is RAW and its 94.33 % companion cell IS
   post-processed, and HLTF's +PP column (92.2 / 92.0 / 91.2) IS post-processed. The honest
   statement is **per-cell AND reading-conditional**; the same cell-dependence applies to the
   HYDROGENS break (raw implicit vs +PP explicit). *(SYNTHESIS conclusion from the pinned cells.)*
3. **Never assert "the band is RAW" either** — same reason, same cells, same reading-conditional rule.
4. **Never quote the 89.4 % search-summary number** — NOT-FOUND in the primary record (0 hits in the
   e-print, 0 on the abs page). The paper says **87.9 %**.
5. **Never quote "13.66×" or "+91.49 pts" as a win over the frontier**, and never publish the
   "100 %" without the speck rate, the mass price and the epoch axis beside it.
6. **Never say the frontier's false-valid axis is a measured rate.** HLTF never counts it; it infers
   it from the PB-Valid-vs-Val gap. Ours is a count over a different population.
7. **Never repeat the "UFF has no parameters for the metals" clause** (refutation C2, §4 above).
8. **Never say the frontier's connectivity is computed after largest-component retention.** Wf-A161's
   `report.md` said exactly that for EF-TALFM and was **WRONG** (A158 + the EF-TALFM branch); **Wf-A194 (2026-11-02) REPAIRED it at source** — the live item 2 no longer carries that sentence, the superseded sentence survives VERBATIM in that file's `## Wf-A176 CORRECTION (2026-11-01)` section, and the sentence sits at no fixed line number (cite the section, not a line).

---

## 6. DEFECTS THE REFUTATIONS FOUND (must be repaired by the main thread)

| # | where | defect | severity |
|---|---|---|---|
| D1 | `molmetal/reports/wfa161_frontier_labelling/report.md` → `## Comparability breaks` item 2 (no line number) | "The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM)" — **contradicted**: EF-TALFM's 92.98 % is RAW (A158 matrix + EF-TALFM branch). **CLOSED by Wf-A194 (2026-11-02)** — the live item 2 no longer carries this sentence; the superseded sentence survives VERBATIM in that file's `## Wf-A176 CORRECTION (2026-11-01)` section | MEDIUM → **CLOSED (2026-11-02)** |
| D2 | `molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json` → `uff.note` (and `metrics/by_metric/cfm_fragmentation_postproc_arm.json` → `uff_note`, from which it is inherited) | "UFF has no parameters for the metals" is **false**; measured cause is the missing conformer (UFF succeeds 60/60 metal-bearing + 4/4 metal-free after ETKDGv3 embedding) | MEDIUM |
| D3 | `molmetal/reports/wfa168_falsevalid_axis/report.md` §7 | attributes a "POST-PROCESSING break" to catalog **§6.264**, which contains **zero** occurrences of "post"; the authoritative four-break set is **§6.265 §4** + A158 matrix. It also **omits the METRIC break**, which A158 calls *"a total METRIC break"* ("the band's upper endpoint is not a connectivity number at all") | MEDIUM |
| D4 | `molmetal/reports/wfa168_falsevalid_axis/report.md` §1b + `falsevalid_axis.json` → `per_epoch_summary.note` | "falls from 0.5639 → 0.0085" describes a **non-monotone** series; the **maximum is epoch 1 = 0.7443 (74.4 %)**, never named in prose; `per_epoch_summary` exposes only `first_epoch`/`last_epoch`, so the peak's epoch id is not in the summary | LOW-MED |
| D5 | `molmetal/scripts/wfa168_eftalfm_axis.py:710` (`render_report`) | the script **crashes** (`TypeError` on `ck['abstract_pct']`) on the very NOT-FOUND path it provides for — `sources.json` is written, `report.md` is not | LOW (latent) |
| D6 | `molmetal/scripts/wfa168_eftalfm_axis.py` §2a break 2 | the branch's break 2 ("parsed rows only, 1 unparseable excluded of 12,669") is **more accurate** than catalog §6.264 §6 break 2 ("over ALL samples drawn") — a supersession that was **not recorded** (8.51 % unaffected) | LOW (doc drift) |
| D7 | `molmetal/reports/wfa168_falsevalid_axis/report.md` §7 + `falsevalid_axis.json` | the HYDROGENS bullet is **inverted** for this arm: step 2 is `Chem.AddHs` (explicit H) and the frontier recipe in the matrix is "largest fragment + explicit hydrogens + UFF relaxation" — the same family, not a break | LOW |
| D8 | branch guards | the HLTF and EF-TALFM branches ship **no test of their own** (the false-valid branch ships 15, but M4/M5/M6 — the UFF block, the epoch axis and the mass-retention block — are asserted against the frozen JSON only and are **not caught** by mutation) | MEDIUM (test-coverage) |
| D9 | `molmetal/reports/wfa168_hltf_axis/` + `wfa168_eftalfm_axis/` | no CI guard against a future arXiv v2 re-pinning these numbers | LOW |

---

## 7. FILES ON DISK (all created by this wave)

**Branch A — HLTF (post-processing + false-valid axes):**
* `/home/hugo/codes/mol-metal/molmetal/scripts/wfa168_hltf_axis.py`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_hltf_axis/sources.json`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_hltf_axis/report.md`

**Branch B — EF-TALFM (four-number reconciliation):**
* `/home/hugo/codes/mol-metal/molmetal/scripts/wfa168_eftalfm_axis.py`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_eftalfm_axis/sources.json`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_eftalfm_axis/report.md`

**Branch C — our false-valid / speck decomposition:**
* `/home/hugo/codes/mol-metal/molmetal/scripts/wfa168_falsevalid_axis.py`
* `/home/hugo/codes/mol-metal/molmetal/tests/test_wfa168_falsevalid_axis.py` (15 tests)
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_falsevalid_axis/report.md`

**Refutations (all three returned NOT REFUTED):**
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_hltf_axis_refutation/refutation.{json,md}`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_eftalfm_axis_refutation/refutation.{json,md}`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.{json,md}`

**This synthesis:**
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md`

**Pre-existing, referenced (not modified):**
* `/home/hugo/codes/mol-metal/metrics/by_metric/cfm_fragmentation_postproc_arm.json`
* `/home/hugo/codes/mol-metal/molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`
* `/home/hugo/codes/mol-metal/molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.{json,md}`
* `/home/hugo/codes/mol-metal/docs/HONEST_NEGATIVE_CATALOG.md` §6.264, §6.265 §4

**Trees verified untouched:** `metrics/`, `registry/`, `paper/`, `docs/`, `TODO/`, `configs/`,
`AGENTS.md`, `README.md` — 0 modifications from this wave. GPU-violation log unchanged (24 historical
lines, pid=700241, mtime 2026-10-31 17:52 — no new line).

---

## 8. WHAT THE MAIN THREAD MUST DO NEXT

1. **Repair D1–D7** (the two doc-truth defects D1/D3, the false causal clause D2, the non-monotone
   epoch prose D4, the latent crash D5, the break-2 supersession D6, the inverted HYDROGENS bullet D7).
   Per §5.5 Rule 6 these are **append-only corrections with dates**, never silent edits.
2. **Re-quote the band everywhere it appears.** Sweep the repo by VALUE (`92.7`, `94 %`, `92.98`,
   `94.33`, `87.9`) — not by file list — and re-state each site with the four breaks, the cell's
   post-processing, and its denominator. A prior wave's failure mode was to re-quote every document
   EXCEPT its own home document and ship a false "UNCHANGED" banner.
3. **Apply the house format** to our two arms (the template in §4, with the D2-corrected UFF clause),
   and to the c8 arm (529/2112 = 0.250473 metal-inclusive, 60/2112 = 0.028409 ligand-only) as a
   separate population. **NOTE: the format is a PROPOSAL — it is currently UNENFORCED (refutation S6).**
   Decide whether to enforce it with a guard before quoting it.
4. **Decide the EF-TALFM false-valid reading.** P4 (93.13 % over the 9,433 subset) and P1 (87.90 % over
   all 10,000) have **different denominators** — the 5.23-pt gap between them is **NOT** a
   false-valid count and must not be presented as one. If a frontier false-valid rate is wanted, it
   must be derived from a same-denominator pair (PB-Valid vs Val within the HLTF QM9 table), and
   labelled as the frontier's *inference*, not a count.
5. **Catalog entry.** Append a new §6.266 to `docs/HONEST_NEGATIVE_CATALOG.md` recording: the four
   pinned EF-TALFM numbers + their sites/denominators, the HLTF pin set, our decomposition, the
   per-cell post-processing finding, and defects D1–D7. Update the AGENTS.md §5.2 pointer and
   `TODO/INDEX.md` per §2 of AGENTS.md.
6. **Do NOT touch `paper/*`** without explicit user authorization (TODO-84 / §5.3). The paper-side
   work — replacing a bare "92.7–94 %" with the four-break sentence — needs that OK first.
7. **No GPU.** Nothing in this wave needs one, and none was used.

---

## 9. HONEST NEGATIVES OF THIS SYNTHESIS

* The band-interpretation cell-dependence (§5 item 2) is a **SYNTHESIS** conclusion derived by
  cross-reading the pinned HLTF and EF-TALFM cells against the A158 matrix; **no branch asserted it**
  and no branch refuted it. It should be treated as a proposal to the main thread, not as a settled
  catalog fact, until a re-quote wave lands it in the catalog.
* The house format (§4) is a **PROPOSAL**. It has been applied to **nothing** in this repo. It is not
  registry-citable and no guard enforces it.
* The EF-TALFM bond threshold `λ_π` is **not numerically recoverable** from the paper (no numeric
  value is given); only its *form* (`λ_π` = policy subscript, product of a scaled covalent-radius sum)
  is pinned. Our deviation's exact λ-equivalent (3.5/2.78 = 1.2590) is a *threshold-equivalence*
  computation, not the paper's λ.
* The HLTF pin set contains **no** mass-retention or survivor-shape number, and the HLTF +PP arm's
  fragment-selection component therefore remains **UNQUANTIFIED in the paper** — we can say it is not
  reported; we cannot say it is small.
* Our false-valid analogue is **not** the frontier's metric. Every sentence above that places the two
  on one axis is labelled ANALOGUE for that reason.


---

## Wf-A180 CORRECTION (2026-11-01) — the published band's provenance is NOT settled on disk

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation. This section is **ADDITIVE**: every line above it is unchanged and the pre-existing
content of this file is a **byte-identical PREFIX** of the current file (AGENTS.md §5.5 rule #6 —
append-only; a correction preserves the superseded rendering VERBATIM and corrects ADDITIVELY + DATED).

### SUPERSEDED (2026-11-01, Wf-A180)

The five passages below are **SUPERSEDED** by the reading-safe clause that follows. They are
preserved **VERBATIM** — not deleted, not silently edited — and must no longer be quoted as fact.

**SUPERSEDED §0 (the VERDICT paragraph):**

> direction, class, measurement and population). The repo's published **92.7–94 % band is not a
> connectivity band at all** (its endpoints are a Validity&Uniqueness rate and a six-check PoseBusters
> pose verdict) and, per Wf-A168, is **cell-heterogeneous in post-processing**: some cells are RAW,
> some are POST-PROCESSED, so the standing "the band is post-processed" break cannot be asserted
> blanket-ward in either direction.

**SUPERSEDED §1.3 ("What the repo's published **92.7–94 %** band actually is"):**

> ### 1.3 What the repo's published **92.7–94 %** band actually is (Wf-A158, on disk)
>
> `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` →
> `band_provenance.headline_finding`: **"THE BAND IS NOT A CONNECTIVITY BAND."**
>
> * **92.7 %** endpoint = HLTF `arXiv:2603.27113v1` Table 1(b) row **"Val&Uniq (%)"**, **GeoLDM**
>   column — a per-molecule *RDKit-sanitisable AND non-duplicate* rate over 10,000 QM9 samples.
> * **94 %** endpoint = HLTF Table 1(b) row **"PB-Valid (%)"**, **HLTF (ours)** column — a **six-check
>   PoseBusters pose verdict** (bond lengths/angles, steric clash, aromatic & double-bond flatness,
>   internal energy) over 10,000 QM9 samples. It *contains* a connectivity check as one of six, but the
>   number is not that check's rate.
> * A **third** 92.7 % exists in the same repo scan: YuelBond (bioRxiv 10.1101/2025.05.06.652517v1) quotes
>   92.7 % as a **per-bond-type classification F1** over 1,000 coordinate-noised molecules.
> * A158 also records that the repo's usage of the band **mixes in EF-TALFM's 92.98 % (raw connectivity)
>   with 94.33 % (post-processed N.&U.&S.)** — "NOT one homogeneous quantity".

**SUPERSEDED §2.3 (the comparability table row):**

> | CONNECTIVITY (the band) | **NO** | **NO** | the band is not a connectivity band (A158) |

**SUPERSEDED §5 ("MAY say", item 1):**

> 1. **"The 92.7–94 % band is not a connectivity band."** Cite A158: the 92.7 % endpoint is HLTF Table
>    1(b) "Val&Uniq (%)" GeoLDM column (a per-molecule RDKit-sanitisable-AND-non-duplicate rate over
>    10,000 QM9 samples); the 94 % endpoint is HLTF Table 1(b) "PB-Valid (%)" HLTF column (a six-check
>    PoseBusters pose verdict over 10,000 QM9 samples); a third 92.7 % is YuelBond's per-bond-type
>    classification F1. None is a connected-component count.

**SUPERSEDED §6 (the defects table, row D3):**

> | D3 | `molmetal/reports/wfa168_falsevalid_axis/report.md` §7 | attributes a "POST-PROCESSING break" to catalog **§6.264**, which contains **zero** occurrences of "post"; the authoritative four-break set is **§6.265 §4** + A158 matrix. It also **omits the METRIC break**, which A158 calls *"a total METRIC break"* ("the band's upper endpoint is not a connectivity number at all") | MEDIUM |

**Why they are superseded.** Each of the five passages asserts ONE provenance reading for the
frontier cells the "92.7–94 %" band names as if it were settled. The repo holds **TWO COMPETING
readings** of those cells and they do not even agree on the numerals, so the band's source cells are
**NOT settled on disk**:

* **Reading A** — `frontier_matrix.json` → `band_provenance`: GeoLDM's Validity & Uniq **92.7 %** and
  HLTF's PB-Valid **94.0 %** on QM9, both `post_processed: false` (plus YuelBond's per-bond-type
  classification **F1 92.7 %** as a third candidate for the same numeral). Under this reading **no**
  endpoint is a connected-component count.
* **Reading B** — the same file → `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a)):
  RAW connectivity `C. = 92.98 %`, counted **BEFORE** the largest-component retention, with the
  post-processed `N. & U. & S. = 94.33 %` companion cell. Under this reading the lower endpoint **IS**
  a connectivity count.

So the §0 verdict, the §1.3 endpoint assignment, the §2.3 row, the §5 item 1 and the D3 row's
"total METRIC break" gloss are each **TRUE under one reading and FALSE (or unset) under the other**.
A published sentence must not assert either reading as fact.

### CORRECTED — the reading-safe clause (the only form that may be quoted)

> the published band's provenance is NOT settled on disk: the frontier
> post-processing matrix record (`frontier_matrix.json`) holds TWO competing
> readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %,
> both `post_processed: false`, plus YuelBond's per-bond-type classification
> F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW
> connectivity `C. = 92.98 %`, counted BEFORE the largest-component
> retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).
> They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the
> band's source cells are NOT settled on disk. Under BOTH readings the band is
> EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR
> 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing
> else.

### THE INVARIANTS THAT HOLD UNDER BOTH READINGS

These four, and only these, may be asserted as fact:

1. the band is **EXTERNAL** to Mol-Metal;
2. it is **NOT one homogeneous quantity**;
3. it is **NOT on our scale** — our number **IS** a connected-component count on the RAW emitted
   SMILES;
4. the band's **source cells are NOT settled on disk**.

### CLAIMS THAT ARE FALSE OR SETTLED-WRONG AND MUST NOT BE ASSERTED

* **X1** — "neither endpoint is a connected-component count" / "None of the three is a
  connected-component count" / "THE BAND IS NOT A CONNECTIVITY BAND": **TRUE under Reading A, FALSE
  under Reading B.** (This is the assertion carried by the SUPERSEDED §0, §1.3, §2.3 and §5
  passages above, and by A158's `band_provenance.headline_finding`.)
* **X2** — "the published band is measured on the published POST-PROCESSED stream" / "our number is
  not post-processed and theirs is": **FALSE** — both band endpoints carry `post_processed: false`.
* **X3** — "their Connectivity is computed AFTER retaining the atom-count largest connected
  component": **REFUTED on disk** — EF-TALFM's 92.98 % connectivity is a RAW single-component rate,
  counted BEFORE the largest-component retention (see `molmetal/reports/wfa168_eftalfm_axis/report.md`
  and `molmetal/reports/wfa171_frontier_postproc/report.md`). The paper CAPTION's sentence is
  ambiguous; the NUMBER is RAW.
* **X4** — "the band is minted here: lower endpoint 92.7 % = GeoLDM's Val&Uniq, upper endpoint 94 %
  = HLTF's own PB-Valid": **asserts ONE reading as settled.** Qualify it, or replace it with the
  reading-safe clause above.

### LOCATORS

Use **JSON PATHS, never line numbers**: `band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. The token
`frontier_axis_mapping` is **NOT** a key of that record — it survives only inside a string value, so
pointing a reader at it is a **DEFECT**.

### WHAT IS NOT CHANGED BY THIS CORRECTION

Every number in this document is untouched, and every measurement above is untouched: the 8.51 %
RAW single-fragment rate (1078/12668), the 100.00 % post-processed rate (12668/12668), the 21.02 %
speck rate, the 22.47 % discarded-mass price, the per-epoch axis, the HLTF and EF-TALFM pin sets and
the D1–D9 defect table. Only the **interpretation** of the "92.7–94 %" band's provenance is corrected,
and only additively and dated.

Nothing above this section was edited or removed (AGENTS.md §5.5 rule #6 — append-only).
---

## Wf-A187 CORRECTION (2026-11-01) — §1.3's live X1 rendering, and this file's PROVENANCE

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation. This section is **ADDITIVE**: every line above it is unchanged and the pre-existing
content of this file is a **byte-identical PREFIX** of the current file (AGENTS.md §5.5 rule #6 —
append-only; a correction preserves the superseded rendering VERBATIM and corrects ADDITIVELY + DATED).

### This artefact is HAND-MAINTAINED — there is NO producer (MEASURED, 2026-11-01)

Measured on disk, 2026-11-01, from the repository root:

* `git ls-files molmetal/reports/wfa168_frontier_axis_mapping/` → **no tracked file**. The whole
  directory is **untracked** (`git status --porcelain molmetal/reports/wfa168_frontier_axis_mapping/`
  → `?? molmetal/reports/wfa168_frontier_axis_mapping/`) and its only member is `synthesis.md`.
* `git grep -n -I -- "wfa168_frontier_axis_mapping" -- '*.py' '*.sh'` → **no hit**. No script in the
  repository produces this file, so there is **no producer/artefact parity check to run** for it
  (contrast `molmetal/reports/wfa161_frontier_labelling/report.md`, whose producer
  `molmetal/scripts/wfa161_frontier_labelling.py` was taught to re-emit its dated correction sections).
* The three sibling `wfa168_*` scripts each write to a **different** directory —
  `molmetal/scripts/wfa168_eftalfm_axis.py` → `reports/wfa168_eftalfm_axis`,
  `molmetal/scripts/wfa168_hltf_axis.py` → `reports/wfa168_hltf_axis`,
  `molmetal/scripts/wfa168_falsevalid_axis.py` → `reports/wfa168_falsevalid_axis`. **None of them
  writes here.**
* The token `frontier_axis_mapping` survives in the repository only inside **string values** (and one
  mention in `docs/HONEST_NEGATIVE_CATALOG.md`); it is **NOT** a key of the A158 record and it is
  **NOT** an output path.

**Consequence for a reader:** every correction to this file must be applied **BY HAND** and **APPENDED**
with a DATED section. No script will regenerate, overwrite, or re-verify it. Do not expect a
producer-parity diff here; there is no producer to diff against.

### The superseded §1.3 claim — NAMED, and quoted VERBATIM

**Superseded passage:** §1.3 "What the repo's published **92.7–94 %** band actually is (Wf-A158, on
disk)". Its opening sentence is the **X1** rendering, quoted **VERBATIM**:

> `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` →
> `band_provenance.headline_finding`: **"THE BAND IS NOT A CONNECTIVITY BAND."**

and §5 ("MAY say", item 1) repeats it, VERBATIM:

> 1. **"The 92.7–94 % band is not a connectivity band."** Cite A158: the 92.7 % endpoint is HLTF Table
>    1(b) "Val&Uniq (%)" GeoLDM column (a per-molecule RDKit-sanitisable-AND-non-duplicate rate over
>    10,000 QM9 samples); the 94 % endpoint is HLTF Table 1(b) "PB-Valid (%)" HLTF column (a six-check
>    PoseBusters pose verdict over 10,000 QM9 samples); a third 92.7 % is YuelBond's per-bond-type
>    classification F1. None is a connected-component count.

The last sentence — **"None is a connected-component count."** — is the **X1** claim itself. It is
**TRUE under Reading A and FALSE under Reading B**, so it may **not** be asserted. The
`## Wf-A180 CORRECTION (2026-11-01)` section above already quotes §1.3 **in full** and names it
**SUPERSEDED**; that dated section is **CORRECT and must be KEPT** — under AGENTS.md §5.5 rule #6 a
superseded rendering is preserved VERBATIM inside a dated record, never silently deleted, and never
silently edited in place. This section does **not** re-open §1.3: it adds the provenance facts and the
replacement wording that a reader of §1.3 needs in order to know the passage is superseded.

### The reading-safe replacement — the only form that may be quoted

> the published band's provenance is NOT settled on disk: the frontier
> post-processing matrix record (`frontier_matrix.json`) holds TWO competing
> readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %,
> both `post_processed: false`, plus YuelBond's per-bond-type classification
> F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW
> connectivity `C. = 92.98 %`, counted BEFORE the largest-component
> retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).
> They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the
> band's source cells are NOT settled on disk. Under BOTH readings the band is
> EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR
> 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing
> else.

### The four invariants that hold under BOTH readings

1. the band is **EXTERNAL** to Mol-Metal — it is a FRONTIER composite, not our measurement;
2. it is **NOT one homogeneous quantity** — under Reading A its two endpoints are two different
   observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B
   one endpoint is a raw connectivity count while the other is a post-processed composite;
3. it is **NOT on our scale** — our number **IS** a connected-component count on the RAW emitted
   SMILES, and nothing else;
4. the band's **source cells are NOT settled on disk**.

**A "RAW" claim must be READING-CONDITIONAL.** An unconditional "both arms are RAW" is itself a
defect: it is true of Reading A's two `post_processed: false` cells and of Reading B's `C. = 92.98 %`
cell, but the two readings disagree on which cells the band names at all.

### LOCATORS for this file

**JSON PATHS, never line numbers.** In
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` the two competing readings
are at `band_provenance` and `sources`.`EF-TALFM`. A line number is a **perishable** locator — the
record grows, so every line-number locator into it is **stale by construction** the moment it is
written. §1.3's `band_provenance.headline_finding` and the §5 item-1 citation above are therefore
superseded **as locators as well as as claims**.

### What this correction does NOT change

No number in this document is touched and no measurement above is touched: the 8.51 % RAW
single-fragment rate (1078/12668), the 100.00 % post-processed rate (12668/12668), the 21.02 % speck
rate, the 22.47 % discarded-mass price, the per-epoch axis, the HLTF and EF-TALFM pin sets and the
D1–D9 defect table. Only the **interpretation** of the "92.7–94 %" band's provenance is corrected,
and only additively and dated.

Nothing above this section was edited or removed (AGENTS.md §5.5 rule #6 — append-only).

---

## Wf-A194 CORRECTION (2026-11-02) — the live body no longer carries the X1 / X4 renderings; every replaced passage is preserved VERBATIM below

**Status:** 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`,
no git mutation. **This file is HAND-MAINTAINNED** — `molmetal/reports/wfa168_frontier_axis_mapping/` has no
`tools/` directory and no producer script, so there is no producer/artefact parity to run: the correction is
applied **BY HAND** and recorded **here, DATED** (AGENTS.md §5.5 rule #6).

**This edit is LOUD, not silent.** Eight passages in the live body of this file were replaced in place with the
reading-safe form. Each replaced passage is quoted **VERBATIM** in §3 below, so every superseded rendering is
preserved in a DATED position and **nothing is deleted**. Every replacement is **line-count neutral**, so every
line-number locator into this file that existed before this section still resolves to the same line.

### 1. Why the replacement was required

The repo holds **TWO competing readings** of which frontier cells the "92.7–94 %" band names — JSON PATHs
`band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. They do not even agree on the numerals.
Two renderings were therefore **FALSE as settled claims**:

* **X1** — "the band is not a connectivity band" / "None is a connected-component count" /
  "THE BAND IS NOT A CONNECTIVITY BAND": **TRUE under Reading 1, FALSE under Reading 2** (whose lower endpoint
  IS a raw connectivity count, `C. = 92.98 %`).
* **X4** — the band names GeoLDM's Validity & Uniq (lower) and HLTF's PB-Valid (upper): **asserts ONE reading
  as settled**.

The canonical reading-safe replacement is quoted **VERBATIM** from
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

### 2. What the replacements changed — and what they did NOT

* **Changed:** the *interpretation* of the band's provenance, in the eight passages listed in §3.
* **NOT changed:** every NUMBER in this file. The 8.51 % RAW single-fragment rate (1078/12668), the 100.00 %
  post-processed rate (12668/12668), the 21.02 % speck rate, the 22.47 % discarded-mass price, the per-epoch
  axis, the HLTF and EF-TALFM pin sets and the D1–D9 defect table all stand as printed. The seven frontier
  cells, their locators and their denominators in §1.3 are unchanged.
* **NOT changed:** the §7-cell table, the four comparability breaks, the axis verdicts other than the one
  CONNECTIVITY row, and the MAY-NOT prohibitions themselves — only their *justification* was made
  reading-conditional.
* **NOT changed:** the `## Wf-A180 CORRECTION (2026-11-01)` and `## Wf-A187 CORRECTION (2026-11-01)` dated
  sections above. They are **not** rewritten (AGENTS.md §5.5 rule #6). Where one of them quotes a passage this
  wave also replaced, the two are now consistent: the dated section quotes the **pre-repair** text, this
  section quotes the **same** pre-repair text, and both point at the same replacement.

### 3. The replaced passages, VERBATIM (superseded renderings preserved)

**§0 VERDICT — was lines 20-24 — X1:**

> direction, class, measurement and population). The repo's published **92.7–94 % band is not a
> connectivity band at all** (its endpoints are a Validity&Uniqueness rate and a six-check PoseBusters
> pose verdict) and, per Wf-A168, is **cell-heterogeneous in post-processing**: some cells are RAW,
> some are POST-PROCESSED, so the standing "the band is post-processed" break cannot be asserted
> blanket-ward in either direction.

**§1.3 lead-in — was lines 117-118 — X1, quoted as the matrix's LIVE wording:**

> `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` →
> `band_provenance.headline_finding`: **"THE BAND IS NOT A CONNECTIVITY BAND."**

**§1.3 candidate-cell bullets — were lines 120-129 — X4:**

> * **92.7 %** endpoint = HLTF `arXiv:2603.27113v1` Table 1(b) row **"Val&Uniq (%)"**, **GeoLDM**
>   column — a per-molecule *RDKit-sanitisable AND non-duplicate* rate over 10,000 QM9 samples.
> * **94 %** endpoint = HLTF Table 1(b) row **"PB-Valid (%)"**, **HLTF (ours)** column — a **six-check
>   PoseBusters pose verdict** (bond lengths/angles, steric clash, aromatic & double-bond flatness,
>   internal energy) over 10,000 QM9 samples. It *contains* a connectivity check as one of six, but the
>   number is not that check's rate.
> * A **third** 92.7 % exists in the same repo scan: YuelBond (bioRxiv 10.1101/2025.05.06.652517v1) quotes
>   92.7 % as a **per-bond-type classification F1** over 1,000 coordinate-noised molecules.
> * A158 also records that the repo's usage of the band **mixes in EF-TALFM's 92.98 % (raw connectivity)
>   with 94.33 % (post-processed N.&U.&S.)** — "NOT one homogeneous quantity".

**§2.3 comparability table, CONNECTIVITY row — was line 189 — X1:**

> | CONNECTIVITY (the band) | **NO** | **NO** | the band is not a connectivity band (A158) |

**§5 "MAY say" item 1 — was lines 288-292 — X1:**

> 1. **"The 92.7–94 % band is not a connectivity band."** Cite A158: the 92.7 % endpoint is HLTF Table
>    1(b) "Val&Uniq (%)" GeoLDM column (a per-molecule RDKit-sanitisable-AND-non-duplicate rate over
>    10,000 QM9 samples); the 94 % endpoint is HLTF Table 1(b) "PB-Valid (%)" HLTF column (a six-check
>    PoseBusters pose verdict over 10,000 QM9 samples); a third 92.7 % is YuelBond's per-bond-type
>    classification F1. None is a connected-component count.

**§5 "MAY NOT say" items 2-3 — were lines 310-317 — X4 / MIRROR:**

> 2. **Never assert "the band is POST-PROCESSED" as a blanket claim** — §6.265 §4's POST-PROCESSING
>    row says so, but Wf-A168 shows the band is **cell-heterogeneous**: HLTF's 92.7 % and 94.0 %
>    endpoints are RAW (`tables/qm9_benchmarks_og.tex:41,66`, no post-processing), EF-TALFM's 92.98 % is
>    RAW, while HLTF's +PP column (92.2 / 92.0 / 91.2) IS post-processed. The honest statement is
>    **per-cell**, and the same cell-dependence applies to the HYDROGENS break (raw implicit vs +PP
>    explicit). *(SYNTHESIS conclusion derived from the pinned cells; not a branch-established
>    statement.)*
> 3. **Never assert "the band is RAW" either** — same reason, same cells.

**§5 "MAY NOT say" item 8 — was lines 325-326 — stale locator (`report.md:139`):**

> 8. **Never say the frontier's connectivity is computed after largest-component retention.** Wf-A161's
>    `report.md:139` says exactly that for EF-TALFM and is **WRONG** (A158 + the EF-TALFM branch).

**§6 defect table, row D1 — was line 334 — stale locator + open status:**

> | D1 | `molmetal/reports/wfa161_frontier_labelling/report.md:139` | "The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM)" — **contradicted**: EF-TALFM's 92.98 % is RAW (A158 matrix + EF-TALFM branch) | MEDIUM |

### 4. The coupled defect D1 — closed at source on 2026-11-02

D1 pointed at `molmetal/reports/wfa161_frontier_labelling/report.md:139`, where the live `## Comparability
breaks` item 2 asserted that the frontier's connectivity rates are computed on the atom-count largest connected
component after RDKit sanitization, and that the published band comes from a DIFFERENT perception branch (Open
Babel). **Wf-A194 repaired that producer and re-published that artefact**: the live item 2 now carries the
reading-safe form, and the superseded sentence survives VERBATIM in the artefact's own
`## Wf-A176 CORRECTION (2026-11-01)` section. The D1 row above is therefore marked **CLOSED**, and its locator
is now a **section path**, not a line number.

### 5. CROSS-WAVE CONSEQUENCE (MEASURED, 2026-11-02) — reported, NOT edited here

`molmetal/reports/wfa175_band_provenance/_build_sites.py` (Wf-A195's file, outside this scope) cites this file by
line number at lines **20, 115, 284, 288, 307, 308, 407**. Every one of those line numbers is **unchanged** by
this wave, because every replacement is line-count neutral. Two of the cited sites change **meaning** and their
owner must re-read them: line **20** (the §0 VERDICT) and line **288** (§5 "MAY say" item 1) now carry the
reading-safe form instead of the X1 form.

The same file cites `molmetal/reports/wfa161_frontier_labelling/report.md` at line **139** and
`molmetal/scripts/wfa161_frontier_labelling.py` at lines **7** and **934**. Line 139 of the report no longer
carries the defect sentence and the producer's line numbers moved when the dated sections were added; those
three citations are now **stale** and their owner must re-point them. This wave does not edit that file.

### 6. What this correction does NOT change

No number in this document is touched and no measurement above is touched: the 8.51 % RAW single-fragment rate
(1078/12668), the 100.00 % post-processed rate (12668/12668), the 21.02 % speck rate, the 22.47 %
discarded-mass price, the per-epoch axis, the HLTF and EF-TALFM pin sets and the D1–D9 defect table. Only the
**interpretation** of the "92.7–94 %" band's provenance is corrected, and only additively and dated.

Nothing above this section was deleted (AGENTS.md §5.5 rule #6 — append-only).

### 7. AMENDMENT (2026-11-02) — precision on §6's "no number touched" sentence

§6 states "No number in this document is touched". Read as a claim about **MEASUREMENTS** that sentence is
exact: the 8.51 %, 100.00 %, 21.02 % and 22.47 % values and every frontier-cell numeral stand as printed.
Read as a claim about **numeral tokens** it needs one correction, which is recorded here ADDITIVELY — the §6
sentence itself is not rewritten (AGENTS.md §5.5 rule #6):

* The live §0 no longer carries the wave attribution "per Wf-A168" (the token `168`), because that attribution
  belonged to the superseded clause "…is **cell-heterogeneous in post-processing**". It survives VERBATIM in
  the dated positions quoted in §3 (the superseded §0 is quoted there in full, twice over this file).
* The live §0 and the live §5 "MAY say" item 1 now each carry "OUR 8.51 %", which is the canonical clause's own
  wording. `8.51 %` was already printed 5 times in this file's live body OUTSIDE the passages this wave
  replaced (live lines 16, 203, 264, 293 and 339 of the post-edit live part), so no numeral was introduced that
  this file did not already contain.

No MEASUREMENT changed, and no numeral inside a replaced passage was re-derived: the numerals of the replaced
§0, §1.3, §2.3, §5-item-1, §5-item-8 and §6-D1 passages are reproduced verbatim in §3 above.
