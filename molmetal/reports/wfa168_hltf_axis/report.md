# Wf-A168 (branch HLTF) — the post-processing axis and the false-valid axis for arXiv:2603.27113

**Generated:** 2026-11-01T03:41:26+00:00 · **Tag:** MEASURED (primary-source retrieval + on-disk metrics) · **GPU:** 0 (CPU-only)

## 0. Retrieval record

| artefact | URL | HTTP (live) | sha256 of the retrieved bytes | bytes | retrieved (UTC, live) |
|---|---|---|---|---|---|
| `eprint` | https://arxiv.org/e-print/2603.27113 | `200` | `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329` | 504186 | 2026-11-01T03:40:28+00:00 |
| | ↳ served from cache on this run; HTTP status and timestamp replayed from the sidecar written at the live fetch | | | | |
| `abs` | https://arxiv.org/abs/2603.27113 | `200` | `374ece252cad92826dcc1f36a321ca971a650a7897acb6eec57581502d714804` | 41764 | 2026-11-01T03:40:28+00:00 |
| | ↳ served from cache on this run; HTTP status and timestamp replayed from the sidecar written at the live fetch | | | | |

- `eprint` final URL after redirect: `https://arxiv.org/src/2603.27113`
- `abs` final URL after redirect: `https://arxiv.org/abs/2603.27113`

## 1. Pinned percentages (every one located in the retrieved LaTeX source)

| id | metric | value | dataset | denominator | post-processing | hydrogens | LaTeX site |
|---|---|---|---|---|---|---|---|
| `qm9_as` | atom stability (AS) | **98.8 %** (`98.8\pm0.1`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw; no LargestFragmentChooser / explicit H / UFF on the QM9 numbers) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:41` |
| `qm9_ms` | molecule stability (MS) | **86.8 %** (`86.8\pm0.3`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:41` |
| `qm9_val` | RDKit validity (Val) | **93.8 %** (`93.8\pm0.6`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:41` |
| `qm9_val_uniq` | RDKit valid-and-unique (Val&Uniq) | **92.9 %** (`92.9\pm0.2`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:41` |
| `qm9_pb_valid` | PoseBusters validity (PB-Valid) | **94.0 %** (`94\pm0.5`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:66` |
| `qm9_novel` | novelty (Novel) | **56.0 %** (`56\pm0.7`) | QM9 | 10,000 unconditional samples (QM9); the fraction is over samples | NONE (raw) | NOT STATED by the paper for the QM9 arms | `tables/qm9_benchmarks_og.tex:65` |
| `geom_valid_raw` | Valid (raw, no post-processing) | **85.5 %** (`{HLTF (ours)} & \underline{85.5} & \underline{85.4} & \underline{85.0}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | NONE (raw, pre-PP) | implicit at this stage (the PP step is what adds explicit H) | `tables/geom_drugs_results.tex:25` |
| `geom_vu_raw` | Valid&Unique (raw, no post-processing) | **85.4 %** (`\underline{85.4}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | NONE (raw, pre-PP) | implicit at this stage | `tables/geom_drugs_results.tex:25` |
| `geom_vun_raw` | Valid&Unique&Novel (raw, no post-processing) | **85.0 %** (`\underline{85.0}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | NONE (raw, pre-PP) | implicit at this stage | `tables/geom_drugs_results.tex:25` |
| `geom_valid_pp` | Valid (+PP) | **92.2 %** (`\quad \emph{+PP} & \underline{92.2} & \underline{92.0} & \underline{91.2}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | standardized PP = largest fragment + explicit hydrogens + UFF relaxation | EXPLICIT (added by the PP step) | `tables/geom_drugs_results.tex:26` |
| `geom_vu_pp` | Valid&Unique (+PP) | **92.0 %** (`\underline{92.0}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | standardized PP = largest fragment + explicit hydrogens + UFF relaxation | EXPLICIT (added by the PP step) | `tables/geom_drugs_results.tex:26` |
| `geom_vun_pp` | Valid&Unique&Novel (+PP) | **91.2 %** (`\underline{91.2}`) | GEOM-DRUGS | 100,000 samples (GEOM-DRUGS); the fraction is over samples | standardized PP = largest fragment + explicit hydrogens + UFF relaxation | EXPLICIT (added by the PP step) | `tables/geom_drugs_results.tex:26` |
| `abstract_geom_pair` | abstract restatement of the GEOM-DRUGS raw/PP pair | **92.2 %** (`85.5\%/85.0\% validity/valid-unique-novel without post-processing and \textbf{92.2\%/91.2\%} after standardized relaxation`) | GEOM-DRUGS | as in the table (100,000 samples); the abstract does not restate N | the abstract calls the PP 'standardized relaxation'; the experiments section defines it as largest fragment + explicit H + UFF | explicit in the +PP arm (per the experiments section) | `iclr2025_conference.tex:55` |
| `geom_invalid_breakdown_disconn` | Disconnected share of INVALID samples (invalidity cause breakdown) | **35.0 %** (`No energy guidance & 40 & 35 & 15 & 10`) | GEOM-DRUGS | INVALID samples only ('Percentages among invalid samples'), NOT all samples; the absolute N of the invalid set is NOT printed | NONE stated for this table (it is an ablation of the raw sampler) | NOT STATED for this table | `sections/appendix.tex:516` |
| `geom_invalid_breakdown_disconn_guided` | Disconnected share of INVALID samples (default guidance) | **12.0 %** (`+$E_{\text{chem}}+E_{\text{cons}}$ & 14 & 12 & 8 & 66`) | GEOM-DRUGS | INVALID samples only; absolute N not printed | NONE stated for this table | NOT STATED for this table | `sections/appendix.tex:518` |

## 2. Context pins (the metadata above is itself pinned, not asserted)

- `ctx_denominators` — sample counts for both benchmarks — `sections/experiments.tex:8`
- `ctx_pp_definition` — the post-processing pipeline definition (largest fragment + explicit H + UFF) — `sections/experiments.tex:8`
- `ctx_pp_effect_claim` — the paper's own attribution of what the PP step does — `sections/experiments.tex:38`
- `ctx_pp_delta` — the paper's own PP delta and its footnote conversion fraction — `sections/experiments.tex:38`
- `ctx_false_valid_abstract` — the 'false-valid' concept, verbatim, in the abstract — `iclr2025_conference.tex:55`
- `ctx_false_valid_ethics` — the 'false-valid' concept, verbatim, in the Ethics Statement — `iclr2025_conference.tex:65`
- `ctx_false_valid_experiments` — how the paper OPERATIONALISES 'false-valid' (inferred from the PB-Valid vs Val gap, never counted) — `sections/experiments.tex:25`
- `ctx_false_valid_pb_rationale` — the paper's rationale for the false-valid axis, verbatim — `sections/experiments.tex:27`
- `ctx_pb_stricter` — the paper's statement that PoseBusters is STRICTER than RDKit sanitization — `sections/experiments.tex:27`
- `ctx_metal_exclusion` — the paper's own limitation that metal-containing compounds are OUT OF SCOPE — `sections/appendix.tex:591`
- `ctx_geom_pipeline_caveat` — the paper's own caveat that GEOM-DRUGS absolute metrics are affected by pipeline issues — `sections/appendix.tex:589`
- `ctx_invalid_cause_in_text` — the body text that ties the invalidity-cause breakdown to GEOM-DRUGS — `sections/experiments.tex:50`
- `ctx_ablation_default_qm9` — the ablation table's default row, which disagrees with the main table on the QM9 columns — `tables/ablations.tex:14`

## 3. The post-processing axis

| metric | raw | +PP | delta (pts) | share of pre-PP invalid recovered |
|---|---|---|---|---|
| Valid | 85.5 % | 92.2 % | **+6.7** | 46.2069 % |
| Valid&Unique | 85.4 % | 92.0 % | **+6.6** | 45.2055 % |
| Valid&Unique&Novel | 85.0 % | 91.2 % | **+6.2** | 41.3333 % |

**OURS:** raw single-fragment rate 0.08509630565203663 (1078/12668) → post-processed connected rate 1.0 (12668/12668) = **+91.49 pts** over the SAME 12668-row denominator (exact: 91.4903694348).

**Magnitude ratio:** +91.49 pts (ours) / +6.7 pts (HLTF Valid) = **13.66×** (exact: 13.655279).

> **COMPARABLE AT ALL? NO.** NOT COMPARABLE as a magnitude. The two deltas are not the same measurement: theirs is the FULL effect of a three-step post-processing pipeline (largest fragment + explicit H + UFF) on a 100,000-sample drug-like pool where 85.5 % of samples were ALREADY valid; ours is the effect of the SAME pipeline's first step on a 12,668-row metal-complex pool where only 8.51 % of rows were already single-fragment. A larger delta here means our RAW stream was more fragmented, not that our post-processing is better.

> ⚠️ **CORRECTION TO THE BRIEF.** The brief framed HLTF's post-processing as 'a geometry relaxation'. The paper's OWN definition (sections/experiments.tex) is 'standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)' -- i.e. the SAME three steps our arm implements, with the SAME source (Buttenschoen et al., arXiv:2505.00518). Its first step IS fragment selection, which is mass-deleting. So the two post-processing axes are the same KIND of operation, and the difference is in the regime and the price, not in the kind.

> ⚠️ **INTERNAL TENSION IN THE PAPER.** The paper attributes the PP gain to geometry only -- 'Standardized post-processing mainly relieves geometric strain ... and cannot repair infeasible topology (e.g., invalid valence or disconnected graphs)' -- while its own pipeline's FIRST step (largest fragment) DOES repair disconnected graphs, and does so by deleting mass. The paper prints no mass-retention or survivor-shape number for its +PP arm, so the fragment-selection component of its +6.7 pts is UNQUANTIFIED in the paper.

**Honest one-sentence reading:**

> The two deltas measure different things on different populations under different baselines — HLTF's +6.7 pts is a *post-processing effect on a stream that was already 85.5 % valid* over 100,000 drug-like samples, while our +91.49 pts is the *fragment-selection step alone* rescuing a stream that was only 8.51 % single-fragment over 12668 metal-complex rows — so a reader must not conclude "we beat the frontier": the larger number says our RAW stream is more fragmented, and the price is 22.47 % of the raw heavy mass discarded with 21.02 % of the survivors left as ≤2-heavy-atom specks.

## 4. The false-valid axis

> **Abstract** (`iclr2025_conference.tex:55`):
>
> Explicit topology generation also reduces ``false-valid’’ samples that pass RDKit sanitization but fail stricter checks.

> **Ethics Statement** (`iclr2025_conference.tex:65`):
>
> ``false-valid'' structures that pass basic sanitization but fail stricter checks

> **Experiments, PoseBusters paragraph** (`sections/experiments.tex:25`):
>
> Higher PB-Valid at comparable RDKit validity indicates fewer ``false-valid'' samples that sanitize in RDKit but violate stricter structural constraints at comparable headline validity.

> **Experiments, 'Why PoseBusters matters here'** (`sections/experiments.tex:27`):
>
> PoseBusters applies stricter structural checks than RDKit, which can sanitize chemically implausible structures, so improvements in PB-Valid (at comparable RDKit validity) indicate fewer false-valid samples rather than simply exploiting the sanitizer

**OUR-side analogue (LABELLED ANALOGUE, NOT THE SAME METRIC):** our post-processed arm reports 100.00 % "connected" (12668/12668), of which 2663/12668 = 21.02 % of survivors are ≤2-heavy-atom specks — a molecule that is "one connected component" and is still a fragment of a fragment.

**Where the analogy goes:**
- Both are validity predicates satisfied by a structure a chemist would not accept as a molecule: a single heavy atom has exactly one connected component, so the predicate passes vacuously.
- Both are properties of a POST-PROCESSED stream, not of the sampler's raw output.
- Both are one-step predicates (component count) that cannot see molecule-level viability.

**Where it breaks:**
- DIRECTION: the paper's false-valid is a SECOND, STRICTER check applied to the SAME molecule (PoseBusters on top of RDKit sanitization). Our analogue is not a stricter check on the same molecule — after LargestFragmentChooser we are scoring a DIFFERENT molecule (the largest fragment), so '100 % connected' is a statement about a molecule the sampler never emitted.
- CLASS: the paper's false-valid presupposes a chemically implausible but structurally WHOLE molecule (a strained ring, a bad bond length). A ≤2-heavy-atom speck is not a false-positive validation — it is a degenerate survivor of a DELETION. Different failure class; the paper's term does not cover it.
- MEASUREMENT: the paper never counts false-valid samples. It INFERS them from the gap between PB-Valid and RDKit Val ('Higher PB-Valid (at comparable RDKit validity) indicates fewer false-valid samples'), so its 'false-valid axis' is an inference over a pair of numbers, whereas ours is a COUNT over the survivor set.
- POPULATION: the paper's own limitation excludes our corpus: 'performance on unusual chemistries, charged/metal-containing compounds, or biomolecules remains to be established' (sections/appendix.tex). Its 100,000 GEOM-DRUGS samples contain no metals.
- DENOMINATOR + HYDROGENS: theirs is 100,000 samples from a training-set-distribution sampling of small drug-like molecules, explicit H after PP; ours is 12,668 RDKit-parseable rows of a CONCATENATED 18-EPOCH generation pool (12,669 CSV rows, 1 unparseable) of a metal-complex generator -- 100 % of the pool's rows carry a cohort metal (Ru/Rh/Re/Os/Ir/Pt), and the rows are pooled across epochs rather than drawn once.

## 5. The corrected house format

**Prior (Wf-A161):** `connectivity {v:.2f}% (single-fragment rate over N={n} decoded samples; post-processing: {postproc}; denominator: {denominator}; hydrogens: {hydrogens}; definition: exactly one connected component)`

**What the prior format missed:**
- N and the DENOMINATOR RULE were separate slots, so a number could be printed with a denominator rule that did not match N (the Wf-A153 census defect: 'runs' is a LIST).
- no FALSE-VALID / SPECK slot: the format could print '100 % connected' with no indication that 21.02 % of those 'connected' survivors are <=2-heavy-atom specks.
- no MASS-RETENTION slot: the price of the post-processing was invisible.
- the definition said 'exactly one connected component' without naming the tool (RDKit).
- the slots must be AXIS-LOCAL: the speck count and the mass-retention clause describe the POST-PROCESSED arm only, so a RAW-arm line must print 'n/a' there rather than borrow the post-processed numbers (borrowing them is an axis mix, the defect class §6.250-§6.253 catalogues).

**Corrected:**

```
connectivity {v:.2f}% = {k}/{N} single-component rows; denominator: {denominator}; post-processing: {postproc}; hydrogens: {hydrogens}; definition: exactly one RDKit connected component; specks: {specks_clause}; mass: {mass_clause}; source: {path}
```

**Filled — our RAW arm:**

```
connectivity 8.51% = 1078/12668 single-component rows; denominator: RDKit-parseable rows of a CONCATENATED 18-epoch generation pool (12669 CSV rows, 1 unparseable) -- not one draw of N; post-processing: NONE (raw, pre-LargestFragmentChooser); hydrogens: implicit; definition: exactly one RDKit connected component; specks: n/a (no survivor set exists before post-processing); mass: n/a (nothing is discarded before post-processing); source: /home/hugo/codes/mol-metal/metrics/by_metric/cfm_fragmentation_postproc_arm.json
```

**Filled — our POST-PROCESSED arm:**

```
connectivity 100.00% = 12668/12668 single-component rows; denominator: RDKit-parseable rows of a CONCATENATED 18-epoch generation pool (12669 CSV rows, 1 unparseable) -- not one draw of N; post-processing: LargestFragmentChooser -> explicit H -> UFF (UFF ok 0/12668: no UFF parameters for this corpus' metals); hydrogens: explicit; definition: exactly one RDKit connected component; specks: 2663/12668 = 21.02% of the survivors are <=2-heavy-atom specks (a 'connected' molecule that is a fragment of a fragment); mass: 22.47% of the raw heavy mass discarded (pooled); per-molecule heavy-atom retention mean 73.95%, median 92.68%; source: /home/hugo/codes/mol-metal/metrics/by_metric/cfm_fragmentation_postproc_arm.json
```

## 6. Honest negatives / UNPINNED

- The QM9 arms carry NO stated hydrogen convention: the paper never says whether the 10,000 QM9 samples are evaluated with explicit or implicit H. Hydrogens are therefore UNPINNED for every QM9 number here.
- The GEOM-DRUGS raw arms are likewise not labelled as to hydrogens; only the +PP arm is explicitly 'explicit hydrogens'. The raw arm is therefore reported as 'implicit at this stage' — the paper's own PP definition adds explicit H, so the raw stream precedes it.
- The invalidity-cause breakdown ('Disconn. 35 % / 20 % / 12 %') is a percentage of INVALID samples, not of all samples, and the paper prints no absolute N for the invalid set. It is NOT comparable to our raw single-fragment rate, which is over parsed rows.
- HLTF prints NO mass-retention and NO survivor-shape number for its +PP arm. The fragment-selection component of its +6.7 pts is therefore UNQUANTIFIED in the paper; we can say it is not reported, we cannot say it is small.
- The paper's ablation table (tables/ablations.tex, default rows) prints QM9 V&U 93.2 and QM9 PB 92.4 for the SAME default configuration the main table reports as Val&Uniq 92.9 and PB-Valid 94. The paper prints two different numbers for the same default config; the headline 94.0 is the main-table one.
- Our own c7 pool (12669 rows / 18 epochs) and the c8 pool (2112 rows / 3 epochs) are DIFFERENT pools and their denominators are not mixed here.
- This branch read `metrics/by_metric/cfm_fragmentation_postproc_arm.json` and `metrics/by_metric/wfa155_ligand_connected_rate.json` from disk and recomputed nothing that those artifacts already measure; every derived number here is a subtraction or a ratio of those on-disk values.
- A parallel branch of this wave extracted a DIFFERENT paper (arXiv:2608.00657) into the shared /tmp/wfa168_probe directory during this run; those files were excluded and the HLTF source was re-extracted into a private directory and re-hashed. The tarball sha256 recorded here is of the re-extracted, verified copy.

## 7. Files

- `script` → `molmetal/scripts/wfa168_hltf_axis.py`
- `sources_json` → `molmetal/reports/wfa168_hltf_axis/sources.json`
- `report_md` → `molmetal/reports/wfa168_hltf_axis/report.md`
