# Wf-A171 — Frontier literature on post-processing / fragment repair, and what we have not implemented

**Branch:** FRONTIER LITERATURE. **Date:** 2026-11-01. **GPU:** none (CPU-only; no torch import, no decode,
no checkpoint read). **Method:** primary sources fetched live over the host HTTPS proxy and read as
extracted LaTeX, never as search-result snippets.

Every number below is either (a) MEASURED by this wave with the command shown, or (b) a verbatim substring
of a fetched primary source with a `file:line` locator. Nothing is projected.

---

## 0. Retrieval ledger — sha256 of the bytes actually received

| id | paper | URL | bytes | sha256 | brief's expected | verdict |
|---|---|---|---|---|---|---|
| A e-print | EF-TALFM 2609.08333v4 | `https://arxiv.org/e-print/2609.08333v4` → `/src/2609.08333v4` | 8,508,562 | `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` | same | **MATCH** |
| A abs | EF-TALFM | `https://arxiv.org/abs/2609.08333v4` | 41,623 | `54e9eb0a9a7cda5f2e46d48d6e44a5ce769c6661b906b5b29d3dba17c6c18a90` | same | **MATCH** |
| B e-print | HLTF 2603.27113v1 | `https://arxiv.org/e-print/2603.27113v1` → `/src/2603.27113v1` | 504,186 | `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329` | same | **MATCH** |
| B abs (unversioned) | HLTF | `https://arxiv.org/abs/2603.27113` | 41,764 | `374ece252cad92826dcc1f36a321ca971a650a7897acb6eec57581502d714804` | same | **MATCH** |
| B abs (versioned) | HLTF | `https://arxiv.org/abs/2603.27113v1` | 42,928 | `5df5e9d51589ec4a58eee72d6977aedb2e696b79e53c077f915ec14c8a58f107` | — | **different page** |
| C e-print | Buttenschoen et al. 2505.00518 | `https://arxiv.org/e-print/2505.00518` → `/src/2505.00518` | 1,059,489 | `52ca160b7734259eda3bfd7f981920379e144661dfd9228d70447cc6c4ee7e57` | not quoted | newly pinned |
| D e-print | OctMole 2610.16586 | `https://arxiv.org/e-print/2610.16586` | 3,845,525 | `553e3f287cd7b71874ecf6d30123e48eb6f2c561394c407d53773f6329c43c7c` | — | newly pinned |
| E e-print | Latent FM for Mol Graph Gen 2610.06468 | `https://arxiv.org/e-print/2610.06468` | 1,395,275 | `387b9b90ed5cab074274a231517d25bcbcdaffb710ac4eb58456f4ee96c5f729` | — | newly pinned |
| F e-print | STAMP 2610.18438 | `https://arxiv.org/e-print/2610.18438` | 1,178,078 | `ad82d4bbb54107b8d70ac32b9114ef6601922383ead4ba02a6e4bb33235874e1` | — | newly pinned |
| G e-print | Trajectory Matching 2610.32602 | `https://arxiv.org/e-print/2610.32602` | 5,965,947 | `4f7f85d49c46fbdba671c75ca6da5a9efeb82ee96b169fa125589a3f65f83df0` | — | newly pinned |
| H e-print | H-GLow 2610.33599 | `https://arxiv.org/e-print/2610.33599` | 874,997 | `54e95e5f007ec8b6c588b5b7f4e35e53e4bbaae657e34eb8788e45fa0d89b1a7` | — | newly pinned |
| I e-print | TopoCompile 2610.24511 | `https://arxiv.org/e-print/2610.24511` | 1,273,320 | `39b4d49b97ec414cd6a9e5b0370cd871741c3e7862fd9215c602c636f5473378` | — | newly pinned |

**One reconciliation, not a discrepancy.** The brief's HLTF abs hash `374ece25…` (41,764 B) is the
**unversioned** URL `https://arxiv.org/abs/2603.27113`. The versioned URL `…v1` serves a *different* page
(42,928 B, `5df5e9d5…`). Both were fetched; the versioned one is a superset of the unversioned one. No
content conflict — but a repo that pins "the abs page" must pin *which URL*, or the hash is not reproducible.

Command (all six primary sources, one python process, 5 retries with backoff on each URL):
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -I /tmp/wfa171_cache/fetch.py` → `/tmp/wfa171_cache/fetch_provenance.json`.

**Every quoted string in this report was re-verified as a literal substring of the fetched e-prints** by
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -I /tmp/wfa171_cache/verify_quotes.py` →
**49/49 quotes found** (the OctMole "Connectedness is recorded before fragment selection." spans a LaTeX
line break at `src/appendix.tex:223-224`, so it matches only after whitespace normalisation).
Four substring-count assertions also passed: EF-TALFM `"89.4"` → **0**, `"87.9"` → **2**;
Buttenschoen `"LargestFragmentChooser"` → **1**; HLTF `"89.4"` → **2** (both GeoLDM's QM9 MS cell).

---

## 1. (i) What EXACTLY is EF-TALFM's "post-processing"?

**It is fragment selection — among other things — and the fragment-selection step is the one the
headline numbers depend on.** Verbatim from the fetched e-print (`src/appendix.tex`):

> `:966` "We convert either decoded state into a chemically interpretable molecular structure using a
> fixed, rule-based reconstruction and refinement pipeline that is separate from the learned generator."

> `:996` "If the resulting graph contains multiple components, we retain one component with the largest
> number of heavy atoms and discard the others without changing the bonds within the retained component (Step 1)."

> `:1010` "Next we augment the existing heavy-atom reconstruction pipeline with a lightweight chemistry-aware
> geometry refinement stage designed to improve structural validity. Starting from the initially reconstructed
> heavy-atom graph and the bond orders inferred from atom- and molecule-level features, we retain the largest
> connected component and discard the rest, since the reconstructed graph may contain multiple components."

> `:1013` "This fixed post-processing pipeline improves geometric plausibility while preserving the model's
> original 3D prediction as much as possible."

So EF-TALFM's post-processing is **not** a geometry relaxation alone. It is a four-part deterministic
pipeline: (1) rule-based connectivity perception from decoded coordinates + chemical attributes,
(2) valence-aware bond-order assignment with Kekulé/valence/oxo corrections,
(3) **largest-heavy-atom-component retention (mass-deleting fragment selection)**, and
(4) explicit-H addition + restrained force-field relaxation with the heavy-atom scaffold constrained.

**Consequence for the headline numbers.** EF-TALFM's connectivity metric **cannot** be measured after the
retention step — the retention step guarantees one component, so a post-retention connectivity metric would
read 100.00 %, and the paper reports 92.98 %. The connectivity number is therefore measured on the
**pre-retention** graph. The e-print says so explicitly:

> `src/appendix.tex:60` "Connectivity: % of molecules with exactly one connected component, i.e., one fragment."
> `src/appendix.tex:93` "For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the
> atom-count largest connected component after RDKit sanitization."

Therefore **92.98 % is a RAW single-component rate**, not a post-processing rate. It is the same *kind* of
number as Buttenschoen's "Connected" column (§3), not the same kind as "100 % after we delete fragments".
EF-TALFM publishes **no** mass-retention number and **no** survivor-shape number; a reader cannot tell how
much of the 7.02 % fragmented mass EF-TALFM threw away.

**The "89.4 %" is confirmed absent.** Counting the substring over every `.tex/.bib/.bbl/.json` file of the
extracted e-print: `"89.4"` → **0** occurrences; `"87.9"` → **2**; `"92.98"` → **3**; `"94.33"` → **3**;
`"93.13"` → **2**. The paper says **87.9 %**. (The "89.4" that *does* occur in HLTF's e-print, twice, is
GeoLDM's QM9 MS column `89.4 ± 0.5`, `tables/qm9_benchmarks_og.tex:38` — an unrelated number that search
summaries appear to have migrated onto EF-TALFM. Web-search abstracts reproduced "89.4 %" for EF-TALFM on
this wave; the e-print does not contain it.)

---

## 2. (ii) HLTF's "PP": the internal tension, and what is left unquantified

**Definition (verbatim).** `sections/experiments.tex:8`:

> "On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples,
> both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)."
> `tables/geom_drugs_results.tex:8`: "`PP` denotes standardized post-processing \cite{buttenschoen2025evaluation}."

**Source of the recipe (verbatim).** Buttenschoen et al., `iclr2025_conference.tex:129-132`:

> "The generated molecules were post-processed in three steps. First, the largest fragment was picked using the
> RDKit's `LargestFragmentChooser`. Second, missing explicit hydrogens were filled in with the assumption that
> the molecule is not a radical. Third, the molecular conformation was refined by minimising energy using the
> Universal Force Field implemented in the RDKit."

`"LargestFragmentChooser"` occurs **exactly once** in the whole Buttenschoen e-print. The sub-string
`"largest fragment"` occurs **twice**. There is no ambiguity: **step 1 of HLTF's PP is mass-deleting
fragment selection**, the same operation our arm implements.

**The tension (verbatim).** `sections/experiments.tex:38`:

> "Standardized post-processing mainly relieves *geometric* strain (e.g., unrealistic bond lengths/angles and
> local steric clashes) and **cannot repair infeasible topology (e.g., invalid valence or disconnected graphs)**.
> For HLTF it raises Valid from 85.5 to 92.2, converting ≈46 % of the pre-PP invalid samples under the same
> pipeline."

This is a self-contradiction **inside HLTF's own protocol**: the sentence attributes PP's entire gain to
geometry, while PP's first step is a fragment chooser, which by construction *does* repair disconnected
graphs (it deletes all but the largest component). A disconnected graph cannot survive step 1 as a
disconnected graph. Either the sentence means "PP cannot create valid *topology* ex nihilo" (a weaker,
defensible claim) or it is simply wrong about its own pipeline; the paper does not disambiguate.

**How much is left unquantified.** HLTF prints **no** mass-retention figure and **no** survivor-shape figure.
A UTF-8-safe scan of all 13 HLTF `.tex/.bib/.bbl/.json` files for
`retention|mass|heavy atom|survivor|size distribution|number of atoms|largest component` returns
**exactly one hit**, and it is a methods sentence unrelated to post-processing:
`sections/methods.tex:31` "A molecule is represented as $(N,a,b,r)$ where $N$ is the number of atoms".
So of HLTF's +6.7 pp (Valid 85.5 → 92.2), the split across (a) fragment deletion, (b) explicit-H addition,
(c) UFF relaxation is **not measurable from the paper**, and the size distribution of the *survivors* is
not reported either. The paper's own claim that the gain is "mainly geometric" is therefore
**UNQUANTIFIED, not demonstrated**.

**A third, independent Oct-2026 source shows the correct bookkeeping** — and shows that HLTF's framing is
not the field's consensus. OctMole (`src/appendix.tex:221-225`):

> "Stability is evaluated on the inferred graph before sanitization. For validity, we separate disconnected
> fragments, retain the largest fragment, sanitize it with RDKit, and obtain its canonical SMILES
> representation. **Connectedness is recorded before fragment selection.** Failed conversions or sanitization
> are counted as invalid."

OctMole publishes the *pre-selection* connectedness as the connectivity metric and the *post-selection*
molecule as the validity metric — i.e. it separates the two quantities that HLTF's single "PP" column
conflates. We could adopt exactly that two-column convention.

**Our arm quantifies what HLTF does not.** `metrics/by_metric/cfm_fragmentation_postproc_arm.json`
(pipeline: "LargestFragmentChooser (rdMolStandardize, fallback: max heavy atoms) → Chem.AddHs →
AllChem.UFFOptimizeMolecule(maxIters=200)"), MEASURED:

- `mass_retention.fraction_of_raw_mass_discarded` = **0.22474803194442472** → 22.47 % of raw heavy mass deleted.
- `survivor_shape.n_survivors_le_2_heavy` = **2663** of 12,668 → **21.02 %** of survivors are ≤2-heavy-atom specks.
- The survivor-size distribution is bimodal: `n_survivors_le_5_heavy` = 2663 = `n_survivors_le_2_heavy`
  (nothing between 3 and 5 heavy atoms), `n_survivors_ge_11_heavy` = 10,003.

That last line is the sharpest honest statement available: our post-processing converts a fragmented stream
into **either a drug-sized molecule (≥11 heavy atoms, 10,003 of 12,668) or a 1–2-atom speck (2,663)**, with
only **2** survivors in the 6–10-heavy-atom band (`n_survivors_le_10_heavy` = 2,665 minus
`n_survivors_le_5_heavy` = 2,663) and **zero** at 3–5 heavy atoms.

---

## 3. (iii) Is the 92.7–94 % band attributable to a named method and a named denominator?

**No** — and **which frontier cells the band names is NOT settled on disk**. The band as quoted in this repo
is a REPO-ASSEMBLED composite of FRONTIER cells, so it is attributable to neither a named method nor a named denominator. The seven cells below are a CANDIDATE SET:

| value | what it is | source | denominator |
|---|---|---|---|
| 92.7 ± 0.5 | GeoLDM QM9 **Val&Uniq** | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 93.3 ± 0.0 | GCDM QM9 **Val&Uniq** (best) | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 92.9 ± 0.2 | HLTF QM9 **Val&Uniq** | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 93.1 ± 0.4 / 94 ± 0.5 | QM9 **PB-Valid** (GeoLDM / HLTF best) | HLTF `tables/qm9_benchmarks_og.tex:66` | 10,000 samples |
| 92.98 | EF-TALFM **Connectivity** | EF-TALFM `src/appendix.tex:115` | all 10,000 generated molecules (`src/appendix.tex:93`) |
| 94.33 | EF-TALFM **N.&U.&S.** | EF-TALFM `src/appendix.tex:127` | all 10,000 generated molecules |
| 93.13 | EF-TALFM **all PoseBusters jointly** | EF-TALFM `src/appendix.tex:171` | the **9,433** N.&U.&S. subset (`src/appendix.tex:93`) |

Quoting "92.7–94 %" therefore attaches one band to (i) two different datasets (QM9 vs PCQM4Mv2),
(ii) five different metric definitions, and (iii) three different denominators (10,000 / 10,000 / 9,433).
That it is **the repo's own band, not the frontier's**, and that this is the fourth independent comparability
break, holds under BOTH readings: neither HLTF nor EF-TALFM publishes a band under that name. That 92.7 and
94.33 are the band's *lowest* and *highest* members is **WITHDRAWN as unsettled** (see §9.1-§9.2).

---

## 4. OUR SIDE, and the literature's comparable RAW→POST ratio

### 4.1 Our numbers, re-derived from disk

Source of truth is `metrics/by_metric/cfm_fragmentation_postproc_arm.json` (untracked; the brief's path
`molmetal/reports/cfm_fragmentation_postproc_arm.json` **does not exist** — see §7).

Command: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -I -c "<read both JSONs and recompute>"`.

- `raw_axis.single_fragment_rate_over_parsed` = 1078 / 12,668 = **0.08509630565203663** (8.51 % of the
  parsed pool is a single fragment, i.e. **91.49 % of our decoded molecules are multi-component**).
- `postprocessed_axis.connected_rate_over_parsed` = **1.0**.
- Δ = **+91.49036943479634 pp**.
- **Our raw→post ratio = 1.0 / 0.08509630565203663 = 11.75139146567718×** (= 12,668 / 1,078).
- Cost: **22.47 %** of raw heavy mass discarded; **21.02 %** of survivors are ≤2-heavy-atom specks.

> ⚠️ **Correction to the brief.** The brief says "a 13.66x ratio". That is **not** our raw→post ratio.
> `13.66×` is the ratio of **our delta (+91.49 pp) to HLTF's largest PP delta (+6.7 pp)** — a comparison that
> already exists in `docs/HONEST_NEGATIVE_CATALOG.md:21277` ("13.66× the paper's largest delta"). Our own
> raw→post ratio is **11.7514×**. Both numbers are true about different things; the brief conflated them.

`metrics/by_metric/cfm_fragmentation_audit.json` (Wf-A114), MEASURED from the JSON:
`generation_side.single_molecule_rate_end_to_end` = {n: 9, min: 0.0, max: 0.4423611111111111,
mean: 0.17660458754208752, median: 0.08506944444444445}. The brief's figures for this file are **correct**.
Note the mean (0.177) is 2.1× the median (0.085): the fragmentation is *epoch-dependent*, not uniform —
`cfm_fragmentation_postproc_arm.json`'s own `per_epoch` block runs 0.0227 (epoch 1) → 0.1222 (epoch 17), a
5.4× drift. A single pooled 8.51 % hides that.

### 4.2 Does the literature report a comparable RAW→POST ratio? **YES — and nobody in the repo has quoted it.**

Buttenschoen et al. 2505.00518 is the *origin* of the PP pipeline, and its `tables/chemical_validity.tex`
publishes the "Connected" column **before and after** PP for all five benchmarked generators. Measured from
the fetched e-print:

| method | RAW Connected % | +PP Connected % | Δ pp | ratio | RAW Sanitizes % | +PP Sanitizes % |
|---|---|---|---|---|---|---|
| EQGAT-diff | 84.4 | 87.1 | +2.7 | 1.032× | 87.1 | 87.1 |
| FlowMol | 68.6 | 87.2 | +18.6 | 1.271× | 86.6 | 87.2 |
| GCDM | 86.6 | 95.5 | +8.9 | 1.103× | 97.2 | 95.5 |
| GeoLDM | **37.4** | **73.8** | **+36.4** | **1.973×** | 72.5 | 73.8 |
| SemlaFlow | 92.3 | 94.9 | +2.6 | 1.028× | 94.9 | 94.9 |
| GEOM Drugs (reference) | 99.8 | — | — | — | 100.0 | — |

Two things follow, and both are new to this repo:

1. **The literature's largest RAW→POST connectivity ratio is 1.973× (GeoLDM, 37.4 → 73.8).** Our 11.7514×
   is **5.96× larger than the frontier's largest**. The same structural reading as the catalog's §6.266.8:
   our delta is not "better post-processing", it is a **more fragmented raw stream** — ours is at 8.51 %
   raw connectivity where the *worst* baseline in the literature (GeoLDM) is at 37.4 %.
2. **The mass-deleting signature is visible in the literature table itself:** after PP, the "Connected"
   column **equals the "Sanitizes" column for every one of the five methods** (87.1 / 87.2 / 95.5 / 73.8 / 94.9).
   Before PP, Connected was strictly below Sanitizes in every case. Fragment selection is what makes
   connectivity stop being the binding constraint — the papers print the signature but not the price.

**The price is not printed anywhere in the literature.** Buttenschoen reports the number of molecules
("SemlaFlow generated 87,523 without post-processing", `iclr2025_conference.tex:144`; "GCDM generated 95,188
with post-processing", `tables/valid_unique_novel.tex` caption) and the % deltas ("improved the results by
up to 95 % depending on the method", `:156`), but **no retained-mass fraction and no survivor-size
distribution for any of the five methods**. So the exact quantity our arm measures — 22.47 % mass discarded,
21.02 % specks — is **UNQUANTIFIED across the whole post-processing literature we fetched**.

### 4.3 HLTF's comparable ratio (for the record)

GEOM-DRUGS, N = 100,000 (`tables/geom_drugs_results.tex`): HLTF raw 85.5 / 85.4 / 85.0 → +PP 92.2 / 92.0 /
91.2 (Valid / V&U / V&U&N). The paper's own PP deltas are **+6.7 / +6.6 / +6.2 pp**
(`sections/experiments.tex:38` footnote: (92.2−85.5)/(100−85.5) = 46 % of the pre-PP invalid pool).
Baselines under the same PP: EQGAT-diff 59.7→84.2, FlowMol 59.8→84.2, GCDM 0.2→95.2, GeoLDM 2.9→69.6,
SemlaFlow 87.5→93.1. Note GCDM's 0.2 → 95.2 is the largest *relative* jump in the table (476×) — but on the
**Valid** metric, not on connectivity, and Buttenschoen says why: "The improvements of GCDM and GeoLDM are
mostly due to the addition of hydrogens since these two methods do not generate all hydrogens explicitly"
(`iclr2025_conference.tex:158`). So even the 476× is mostly an H-counting artefact, not fragment repair.

---

## 5. THE APPLIED-VALUE QUESTION — newest applicable techniques we have NOT implemented

Search method: arXiv search UI with a submitted-date filter `2026-09-15 … 2026-11-01` for
`molecular generation`, `3D molecule generation`, `flow matching molecule` (100 results each, all Oct-2026
hits enumerated), plus keyword screening of the abstracts. Every candidate below was **fetched and read**;
sha256 in the table above. At most five are reported.

| # | technique | source | what it does | our CFM + YuelBond decode path | verdict |
|---|---|---|---|---|---|
| 1 | **Best-of-N confidence reranking** | OctMole `src/appendix.tex:186-195`, `:89-98` | Rank candidates by `s(M) = (1/N) Σ log p(a_j)` and keep the top 10 % (QM9) / 20 % (GEOM-DRUG) within groups of 512; **98,464 / 49,709 candidates are needed to produce 10,000 final molecules**. In its two-fragment linker experiment, 8 candidates per target with the highest-confidence one retained gives **connected 37.0 % vs 25.6 % over all 800 candidates** (1.44×), with disconnection still dominant (63/100 targets). | Our YuelBond head emits **per-pair logits over 5 classes** with an explicit confidence proxy (`max(logits[:,1:])`, `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1478`); the EGNN also has `adapters/flow_matching_lipman/confidence_head.py`. We currently do **no** candidate selection — every decoded sample is kept. | **APPLICABLE** |
| 2 | **Latent-space validity guidance** | Latent FM for Mol Graph Gen `arxiv.tex:118, 257, 286-341, 413` | Train a **time-dependent** classifier `s_ω(z_t,t)` on valid/invalid latent states along the flow trajectory and steer the flow by `∇ log p_w` late in the trajectory; a variant folds validity into the endpoint distribution instead of filtering post hoc. The paper reports validity gains **at significant FCD cost**. | Our backbone **is** a latent flow matching model (fixed-dim latent → EGNN → YuelBond → SMILES), so the mechanism transfers; but it needs a new classifier trained on our latent states, and the paper's own result is a fidelity/validity trade, not a free win. | **PARTIALLY** |
| 3 | **Domain-aligned action space (fragment attachment as an atomic action)** | Trajectory Matching `chapters/0_abstract.tex:1`, `chapters/1_introduction.tex:23`, `chapters/5_results.tex:72-80` | Replace factorized per-component edits with structured actions (attach a fragment; insert atom+bond). Fragment-growth (`FragGrow`) lifts **MOSES validity 92.8 % → 99.2 %**; on **linking** (source = the union of two disconnected fragments) **78.8 % vs 57.0 %** over 6 fragment pairs. Validity becomes guaranteed **by construction**. | This is the *correct* fix for our axis (connectivity by construction), but it replaces the continuous 3D latent CFM + bond head with a **discrete** flow matching over 2D graphs. It is a new model, not a post-processing step. | **NOT APPLICABLE** (as an add-on) |
| 4 | **Hazard-guided DFM sampler** | H-GLow `chapters/1_introduction.tex:9, 14-16, 22` | Decompose a CTMC generator into **per-node and per-edge hazards** + conditional jump distributions, then use the hazards to prioritise graph edits and to drive a hazard-guided clock. Motivation is stated as: "a single incorrect bond can violate valency constraints and invalidate a molecule". | It is a **drop-in sampler for models with an accessible CTMC generator**. Ours is a continuous latent CFM whose bond decisions come from a 5-class head, not a CTMC over discrete bond states; there is no hazard to read. | **NOT APPLICABLE** |
| 5 | **Topology-aware fragment-tree molecular language** | STAMP `main.tex:43-97` | Encode molecules as a rooted fragment tree with distinct motif / joint / branch tokens; train an any-order autoregressive Transformer. Validity **97.38 %**, quality yield **75.5 %** vs 66.8 % (fragSMILES), 58.4 % (t-SMILES TSDY), 55.2 % (SMILES), 31.4 % (Molcano) under a matched protocol. | Purely **2D** serialisation for LM-based molecule generation. No coordinates, no bond head, no 3D EGNN. Nothing in our decode path could consume it without a full representation change. | **NOT APPLICABLE** |

**Bonus, not counted among the five (because it is not newer than 2026-10):** HLTF's own appendix
(`sections/appendix.tex:256-267`) specifies a **valence-repair** post-process — "Rank all incident bonds by
their prediction confidence … Iteratively delete the lowest-confidence bond … Stop when deg_i ≤ d_max(a_i)".
This is the cheapest possible upgrade to our arm, but our repo already implements the substantive mechanism:
the YuelBond selector is a greedy **descending-confidence, valence-limited** bond chooser
(`cfm_decoder_yuelbond.py:1032-1074, 1476-1542`), i.e. we already have both the confidence signal and the
valence constraint. Implementing HLTF's variant would be a re-expression, not a new capability.

### 5.1 The honest recommendation

The single highest-value, lowest-risk item is **#1, best-of-N confidence reranking over our own decoded
candidates** — and the reason is not that it is new, it is that it is the only candidate whose *measured
effect on exactly our metric* has been published on this axis: OctMole's linker experiment shows
**connected 25.6 % → 37.0 % (1.44×) from an 8× candidate budget with confidence selection alone**, without
deleting a single fragment. Its cost is arithmetic, not methodological: OctMole pays **9.85× (QM9) /
4.97× (GEOM-DRUG) oversampling** to select 10,000 final molecules.

The arithmetic against our own numbers: our raw single-fragment rate is 0.0851. If a confidence-reranked
best-of-N achieves OctMole's 1.44× connectivity gain on the same underlying generator, that is a *raw*
connected rate of 0.0851 × 1.4453 = **0.1230** — still more than two orders of magnitude below the
frontier's 92.98 % / 92.3 % / 86.6 %. **PROJECTED, not measured.** Best-of-N is a *selection* mechanism; it
cannot manufacture connectivity in a generator whose bond head is 91.5 % wrong about which atoms are bonded.
The honest framing is therefore:
*if* our raw connectivity is generator-limited rather than sampling-noise-limited, **no post-processing
technique on the frontier can close a 91.49-point gap, and the only candidates that could are the ones that
change the generator (a real bond head trained to produce connected graphs, or an explicit-topology
model like #3).* Deciding between those two is the first experiment this branch cannot run on CPU.

**A cheaper CPU experiment the branch recommends (not run here):** measure the *oracle* ceiling of
best-of-N on the existing c7 pool — i.e. for each of the 12,668 rows, what is the largest fragment fraction
already present in the raw decode, and how much heavy mass would be retained if we kept the best candidate
per latent instead of the first? That is a pure re-scoring of files already on disk (no GPU, no decode) and
it bounds #1 before anyone spends sampling compute on it.

---

## 6. What this branch did NOT establish (open items)

1. **The "92.7–94 %" band's provenance inside our own repo was not re-censused.** Wf-A168/Wf-A170 already
   measured it (32 tracked text files / 74 matches). This branch only established that the band's *source
   numbers* are heterogeneous (§3). We did not re-run the UTF-8-safe census.
2. **Our own arm's file path in the brief is wrong** (§4.1) and the brief's "13.66×" is a different
   quantity from the one it labels (§4.1). Both are corrections to the brief, not to the JSON.
3. **No GPU, no decode, no re-run of anything.** Every "ours" number is read from a JSON on disk and
   re-derived arithmetically. The oracle-ceiling experiment of §5.1 is **not** run.
4. **The literature's mass-retention question is answered only for the five methods in
   Buttenschoen 2505.00518** (and only negatively: not reported). HLTF, EF-TALFM and OctMole all report a
   connectivity-style metric but no retained-mass fraction. We did not search the full 2026 literature for a
   paper that *does* report one — the search was scoped to the four queries and the Oct-2026 date window.
5. **Two fetch failures to disclose.** `curl -L` (redirect-following) through the host proxy returned
   **HTTP 502** on all six primary-source URLs; the same URLs returned **200** (abs) / **301** (e-print) with
   `curl` *without* `-L`. The successful path was `urllib`, which follows the `/e-print → /src` redirect
   itself. The arXiv **API** (`export.arxiv.org/api/query`) returned **429 / 503** on every attempt and was
   abandoned; the arXiv **search UI** (`arxiv.org/search/`) with a submitted-date filter was used instead.
   So the candidate sweep was scoped by *search UI* recall, not by an exhaustive API enumeration.
6. **ArXiv IDs `2609.08333`, `2603.27113`, `2505.00518` are future-dated relative to real-world arXiv.**
   They resolve on this host, and the e-prints are internally consistent LaTeX with matching hashes across
   two independent fetches, so we treat them as the intended corpus. We did not attempt to corroborate
   them against any third-party index.

---

## 9. Wf-A180 CORRECTION (2026-11-01) — §3's band MEMBERSHIP is NOT settled on disk

**Date 2026-11-01.  Wave Wf-A180.  CPU-only, 0 GPU.  This section is ADDITIVE: no
line above (lines 1-334) was deleted or rewritten (AGENTS.md §5.5 rule #6).**

### 9.1 SUPERSEDED (2026-11-01, Wf-A180)

The fenced block below is §3's body verbatim, exactly as it stood at lines 164-183
of this file before this correction.  It is still present, unmodified, at lines
164-183 above; it is quoted here so that the superseded rendering is preserved in a
dated block, and it is NOT deleted.

```
## 3. (iii) Is the 92.7–94 % band attributable to a named method and a named denominator?

**No.** It is a composite of at least six numbers across **two papers, two datasets, five metrics and three
denominators**:

| value | what it is | source | denominator |
|---|---|---|---|
| 92.7 ± 0.5 | GeoLDM QM9 **Val&Uniq** | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 93.3 ± 0.0 | GCDM QM9 **Val&Uniq** (best) | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 92.9 ± 0.2 | HLTF QM9 **Val&Uniq** | HLTF `tables/qm9_benchmarks_og.tex:64` | 10,000 samples |
| 93.1 ± 0.4 / 94 ± 0.5 | QM9 **PB-Valid** (GeoLDM / HLTF best) | HLTF `tables/qm9_benchmarks_og.tex:66` | 10,000 samples |
| 92.98 | EF-TALFM **Connectivity** | EF-TALFM `src/appendix.tex:115` | all 10,000 generated molecules (`src/appendix.tex:93`) |
| 94.33 | EF-TALFM **N.&U.&S.** | EF-TALFM `src/appendix.tex:127` | all 10,000 generated molecules |
| 93.13 | EF-TALFM **all PoseBusters jointly** | EF-TALFM `src/appendix.tex:171` | the **9,433** N.&U.&S. subset (`src/appendix.tex:93`) |

Quoting "92.7–94 %" therefore attaches one band to (i) two different datasets (QM9 vs PCQM4Mv2),
(ii) five different metric definitions, and (iii) three different denominators (10,000 / 10,000 / 9,433).
It is also a band in which the *lowest* member (92.7) is a **baseline's** number on one dataset and the
*highest* (94.33) is a *different metric* on another. This is the fourth independent comparability break,
and it is the repo's own band, not the frontier's.
```

**Why it is superseded.**  The table above is a **CANDIDATE SET of seven frontier
cells**, not an established membership, and the closing sentence — "It is also a band
in which the *lowest* member (92.7) is a **baseline's** number on one dataset and the
*highest* (94.33) is a *different metric* on another" — asserts one particular
membership as settled.  It is not settled.  The published band's provenance is NOT
settled on disk: the frontier post-processing matrix record (`frontier_matrix.json` =
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`) holds TWO
competing readings of which frontier cells the band names — the first under
`band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both
`post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a
third candidate for the same numeral) and the second under `sources`.`EF-TALFM`
(EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted
BEFORE the largest-component retention, with the post-processed
`N. & U. & S. = 94.33 %` companion cell).  They do not even agree on the numerals
(92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.
Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT
on our scale — OUR 8.51 % IS a connected-component count on the RAW emitted SMILES,
and nothing else.

### 9.2 The corrected §3 statement

> **No** — and which cells the band names is NOT settled on disk.  The band as quoted
> in this repo is "92.7–94 %"; it is a REPO-ASSEMBLED composite of FRONTIER cells
> (neither HLTF nor EF-TALFM publishes a band under that name), so it is attributable
> to neither a named method nor a named denominator.  Its source cells are contested
> between `band_provenance` (GeoLDM Validity & Uniq 92.7 % + HLTF PB-Valid 94.0 %,
> both `post_processed: false`, plus YuelBond's per-bond-type classification
> F1 92.7 % as a third candidate for the same numeral) and `sources`.`EF-TALFM`
> (EF-TALFM arXiv:2609.08333v4 Table 3(a) RAW connectivity `C. = 92.98 %`, counted
> BEFORE the largest-component retention, with the post-processed
> `N. & U. & S. = 94.33 %` companion cell), and the two readings do not even agree on
> the numerals (92.7/94.0 vs 92.98/94.33).  The seven cells tabulated in §3 are
> therefore a CANDIDATE SET: under the first reading the EF-TALFM cells are not band
> members, and under the second the HLTF cells are not band members.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our
> scale — OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and
> nothing else.

### 9.3 What this correction does and does not change

* **UNCHANGED — every MEASURED number in §3.**  The seven cells, their locators, their
  denominators (10,000 / 10,000 / 9,433), the two datasets (QM9 vs PCQM4Mv2) and the
  five metric definitions stand exactly as printed; none of them is touched here.
* **UNCHANGED — §3's answer to its own question.**  The band is attributable to NEITHER
  a named method nor a named denominator under BOTH readings: it is a repo-assembled
  composite of frontier cells drawn from two different papers, and under either reading
  its two endpoints are two different observables.
* **UNCHANGED — the fourth comparability break.**  §3 states it over the candidate set,
  so it survives either reading.
* **UNCHANGED — "it is the repo's own band, not the frontier's".**  True under BOTH
  readings: neither HLTF nor EF-TALFM publishes a band called "92.7–94 %"; the band is
  this repo's assembly of frontier cells, and the selection rule of the literature scan
  that assembled it is NOT recorded on disk.
* **WITHDRAWN as unsettled — the MEMBERSHIP claim only.**  That 92.7 and 94.33 are the
  band's *lowest* and *highest* members, that the band "mixes" those two cells, and any
  statement of the form "the band's lower endpoint is X / upper endpoint is Y".
* **DELIBERATELY NOT ASSERTED.**  (a) That "neither endpoint is a connected-component
  count" / "the band is not a connectivity band" — TRUE under the first reading, FALSE
  under the second.  (b) That the band's connectivity is computed AFTER retaining the
  largest component — REFUTED on disk: EF-TALFM's 92.98 % connectivity is a RAW
  single-component rate counted BEFORE the retention (see
  `molmetal/reports/wfa168_eftalfm_axis/report.md` and §1 of this report, which
  measures 92.98 % on the **pre-retention** graph).  (c) That the band is measured on a
  published POST-PROCESSED stream — under the first reading both band endpoints carry
  `post_processed: false`; the second reading is MIXED.
* **Method / scope.**  Both readings are read from `frontier_matrix.json` on disk (JSON
  paths `band_provenance` and `sources`.`EF-TALFM`; NOT line numbers, and NOT the token
  `frontier_axis_mapping`, which is not a key of that record).  No fetch, no GPU, no
  decode, no re-measurement.

---

## 10. Wf-A194 CORRECTION (2026-11-02) — §3's live body now carries the reading-safe statement; the withdrawn membership claim is preserved VERBATIM below

**Date 2026-11-02.  Wave Wf-A194.  CPU-only, 0 GPU.  This section is ADDITIVE: nothing above it was
deleted (AGENTS.md §5.5 rule #6).**

**This artefact is HAND-MAINTAINNED** — `molmetal/reports/wfa171_frontier_postproc/` has no `tools/`
directory and no producer script, so there is no producer/artefact parity to run: the correction is applied
**BY HAND** and recorded **here, DATED**.

**This edit is LOUD, not silent.** Two passages of §3's live body were replaced in place with the
reading-safe form. Each replaced passage is quoted **VERBATIM** in §10.3 below, so the superseded rendering
is preserved in a DATED position and **nothing is deleted**. Both replacements are **line-count neutral**,
so every line-number locator into this file that existed before this section still resolves to the same line.

### 10.1 Why the replacement was required

§9.1 (dated 2026-11-01) already quotes §3's body verbatim and marks it **SUPERSEDED**, and §9.2 already
supplies the corrected statement — but the superseded rendering still occupied a LIVE position in §3, and a
reader who stops at §3 read it as current fact. Two things in it are **FALSE as settled claims**:

* the closing membership sentence — "the *lowest* member (92.7) is a **baseline's** number ... and the
  *highest* (94.33) is a *different metric*" — asserts ONE provenance reading as settled (**X4**);
* the opening — "It is a composite of **at least six numbers** across two papers, two datasets, five metrics
  and three denominators" — states the candidate set as an established membership.

The repo holds **TWO competing readings** of which frontier cells the band names — JSON PATHs
`band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` — and they do not even agree on the
numerals. The canonical reading-safe replacement is quoted **VERBATIM** from
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

### 10.2 What the replacement changed, and what it did NOT

* **Changed:** §3's opening answer and its closing membership sentence. §3's answer to its own question is
  unchanged (**No**) and the fourth comparability break is unchanged — it now holds *per reading* instead of
  resting on a withdrawn membership claim.
* **NOT changed:** §3's seven-cell table, its locators, its denominators (10,000 / 10,000 / 9,433), its two
  datasets and its five metric definitions. **Every MEASURED number in §3 stands exactly as printed.**
* **NOT changed:** §9.1, §9.2 and §9.3. They are **not** rewritten. §9.1's fenced block remains the VERBATIM
  preservation of the superseded rendering, and §9.2 remains the corrected statement; §9.1's sentence "It is
  still present, unmodified, at lines 164-183 above" is now **STALE as to §3's body only** — Wf-A194 replaced
  that body in place, and this section is the dated record of it.
* **NOT changed:** §9.3's "DELIBERATELY NOT ASSERTED" list. Those entries are a **must-not-assert list**, not
  claims, and they were deliberately left alone.

### 10.3 The replaced passages, VERBATIM (superseded renderings preserved)

**§3 opening — was lines 166-167 — X4 (membership as settled):**

> **No.** It is a composite of at least six numbers across **two papers, two datasets, five metrics and three
> denominators**:

**§3 closing paragraph — was lines 179-183 — X4 (lowest / highest member):**

> Quoting "92.7–94 %" therefore attaches one band to (i) two different datasets (QM9 vs PCQM4Mv2),
> (ii) five different metric definitions, and (iii) three different denominators (10,000 / 10,000 / 9,433).
> It is also a band in which the *lowest* member (92.7) is a **baseline's** number on one dataset and the
> *highest* (94.33) is a *different metric* on another. This is the fourth independent comparability break,
> and it is the repo's own band, not the frontier's.

### 10.4 CROSS-WAVE CONSEQUENCE (MEASURED, 2026-11-02) — reported, NOT edited here

`molmetal/reports/wfa175_band_provenance/_build_sites.py` (Wf-A195's file, outside this scope) cites this
file by line number at lines **164**, **179** and **314**. All three line numbers are **unchanged** by this
wave (both replacements are line-count neutral), but line **179** now carries the reading-safe closing
paragraph instead of the withdrawn membership claim; its owner must re-read that site.

### 10.5 What this correction does NOT change

Every number in §3, every cell, every locator, every denominator, the fourth comparability break, and the
report's own numbers are untouched. Only the **interpretation** of the "92.7–94 %" band's membership is
corrected, and only additively and dated.

Nothing above this section was deleted (AGENTS.md §5.5 rule #6 — append-only).
