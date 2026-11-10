# Wf-A163 — narrative-truth scan of the published metrics tree

**Date:** 2026-11-01 · **Branch:** N (Wf-A163 parallel ultracode) · **CPU-only**
(`CUDA_VISIBLE_DEVICES=""`; no torch import, no CUDA, no decode, no trainer, no `os.walk`).

---

## 0. Bottom line

The Wf-A155 class **generalises**, but not in the way a tree scan can see it.

| Question | Measured answer (2026-11-01) |
|---|---|
| Denominator | **583 JSON artifacts** under `metrics/by_metric/` + `metrics/paper_tables/` (583 parsed, **0 unreadable**) |
| Artifacts with ≥1 flag (scanner) | **169** flagged / **414** clean; **482** flags total |
| Hard contradictions of an artifact's own fields | **0** — the two precise rules (pool misattribution, stale cross-reference) fire **0 times** in 583 artifacts (guard, independent implementation) |
| Hand adjudication | sample = **top 10 artifacts by flag count = 139 of 482 flags (28.9 %)** → **0 TRUE_POSITIVE**, **106 BENIGN**, **33 UNBACKED_OWN_MEASUREMENT** |
| Cross-check vs the two REPAIRED artifacts | **0 flags on each** (scanner *and* guard) → the class is closed there |
| Where the defect actually lives | **in the TOOL.** 4 armed scripts carry pool-specific literals (§6) |
| Patch | `molmetal/reports/wfa163_narrative_truth_scan/patch_narrative_guard.diff` (`git apply --check` PASS on a scratch copy, NOT applied) |

**The honest claim this supports:** *"the published metrics tree contains prose numbers that no
machine field of the same artifact backs (33 of 139 hand-adjudicated flags in the top decile),
and 4 producer scripts can emit false prose for a non-default pool."*

**The claim it does NOT support:** *"the published metrics tree is untrustworthy."* — 0 of 583
artifacts contains a prose number that contradicts the artifact's own fields, and the guard found
0 stale cross-references. A 0 %-precision-on-the-strong-class scanner does not license that claim,
and this report does not make it.

---

## 1. The class

An artifact is a machine record (numbers, keys) **plus** prose (strings). A number rendered in a
prose string is a **claim**. If the claim has no counterpart anywhere in the artifact's own machine
fields, the prose is **unbacked**: it was written by a human or by a tool constant, not derived
from the run the artifact describes. When the producer is re-run on a different population, an
unbacked prose number becomes **false**, because nothing recomputes it.

Wf-A155 found this in one tool (`wf_a155_ligand_connected_rate.py`, repaired by Wf-A162). Wf-A163
asks whether the same thing happens elsewhere.

---

## 2. Method

Two **independent** implementations (neither imports the other), CPU-only:

### 2.1 `molmetal/scripts/wfa163_narrative_truth_scan.py` — the census

Enumerates `metrics/by_metric/` and `metrics/paper_tables/` with `Path.iterdir()` (never
`os.walk`), parses every `*.json`, walks every string value, extracts integer/float tokens, and
asks whether the token appears anywhere in the artifact's own machine fields (numbers, keys, or
*other* strings). Rules:

| Rule | Meaning |
|---|---|
| `NARRATIVE_NUMBER_NO_MACHINE_COUNTERPART` | an integer in a prose string with no machine counterpart |
| `NARRATIVE_PERCENT_NO_MACHINE_COUNTERPART` | the same for a percentage |
| `SELF_CONTRADICTING_PROSE_POOL` | prose names a pool the artifact's own `source_pool` does not name, and no other machine field cross-references it |
| `SELF_CONTRADICTING_PROSE_COUNT` | prose claims a population size that contradicts the artifact's own denominators |
| `SELF_CONTRADICTING_PROSE_DATE` | prose names a date that is not the artifact's own `date` (reported UNSURE: cross-referencing another run's date is legitimate) |
| `SELF_CONTRADICTING_PROSE_SIBLING` | a prose number whose clause matches a *semantically named* machine key that holds a different value |

Precision levers (each of these was added after it removed only false positives): claim-position
detection (a number standing in front of a unit word / `%` / an `A/B` ratio / an `N of M`);
percent scaling only when the token is literally followed by `%`; a strict-subset word match for
the sibling rule; clause-break clipping; a generic-word stoplist (English function words, count-unit
nouns, file extensions); exclusion of ranges, ratio members, identifiers, dates and file paths.

### 2.2 `molmetal/scripts/wfa163_narrative_truth_guard.py` — the proposed repo-wide guard

A library any artifact producer can call. Two modes:

- **`declared`** — the producer names its narrative paths (the Wf-A162-repaired a155 tool already
  does this with its `NARRATIVE_PATHS` constant). Every number in those paths must be backed.
  **Precise. This is the hard gate.**
- **`auto`** — no declaration; only count-claim positions are considered. **Loose: advisory.**

Three checks, by severity:

- `pool_misattribution` (**contradiction**) — prose names a pool the artifact's own `source_pool`
  does not name, with no legitimate cross-reference.
- `stale_cross_reference` (**contradiction**) — prose quotes a number attributed to another
  artifact as `file.json -> /jsonpath`; the guard **resolves the cited artifact** and checks the
  quote still matches. This is the only rule that can verify a number that legitimately lives
  elsewhere (it is what makes the Wf-A155 "8.51 %" claim checkable at all).
- `count_contradiction` / `unbacked_number` / `unbacked_percent` (advisory).

### 2.3 Tool-layer scan

A tool is **ARMED** if it exposes a `--pool` argument *and* carries a pool-specific literal inside
a string literal (module docstring, path defaults and bare path fragments excluded). 863 scripts
checked.

---

## 3. Denominator

| Item | Count | How |
|---|---|---|
| Artifacts enumerated | **583** | `Path.iterdir()` over `metrics/by_metric/` + `metrics/paper_tables/`, `suffix == ".json"` |
| Parsed | **583** | `json.loads` |
| Unreadable | **0** | — |
| Scripts checked (tool layer) | **863** | `molmetal/scripts/*.py` + `molmetal/**/scripts/*.py`, `iterdir`/`glob`, no `os.walk` |

---

## 4. Census (scanner)

| | artifacts | flags |
|---|---|---|
| with ≥1 flag | **169** | **482** |
| clean | **414** | — |

| Rule | flags |
|---|---|
| `NARRATIVE_NUMBER_NO_MACHINE_COUNTERPART` | 285 |
| `NARRATIVE_PERCENT_NO_MACHINE_COUNTERPART` | 64 |
| `SELF_CONTRADICTING_PROSE_DATE` | 64 |
| `SELF_CONTRADICTING_PROSE_SIBLING` | 57 |
| `SELF_CONTRADICTING_PROSE_COUNT` | 12 |
| `SELF_CONTRADICTING_PROSE_POOL` | 0 |

## 4b. Census (guard — independent implementation)

| Rule | severity | flags | artifacts |
|---|---|---|---|
| `pool_misattribution` | contradiction | **0** | 0 |
| `stale_cross_reference` | contradiction | **0** | 0 |
| `count_contradiction` | advisory | **19** | 13 |
| `unbacked_number` + `unbacked_percent` | advisory | **474** | 172 |

**0 blocking findings over 583 artifacts** — the reason the proposed promotion gate (§9) can be
enabled repo-wide without blocking a single existing promotion.

---

## 5. Hand adjudication (task B)

**Sample:** the 10 artifacts with the most flags = **139 of the 482 flags (28.9 %)**. Each flag was
read in place in its artifact; the machine-field claim was re-verified on disk (e.g.
`metrics/by_metric/msrs_variance_decomposition.json` has 429 numeric leaves and **no** field equal
to 5,965, 2,217 or 309; `metrics/by_metric/cfm_metal_geometry.json` has 458 numeric leaves and no
field equal to 329, 2,671, 33,429 or 8,275).

| Verdict | n | what it is |
|---|---|---|
| **TRUE_POSITIVE** (prose contradicts the artifact's own fields) | **0** | — |
| **BENIGN** | **106** | identifiers (`TODO-140/141/142/143/144`, `Wf-A79/83/84`), dates naming a code path or a checkpoint mtime, a file size (`11,542 bytes`), an external dataset size (`26,801` MetalCytoToxDB rows), a protocol ratio (`70/10/20` scaffolds), the exact rational of a float stored in a *key* (`59/175` = 0.33714285714285713 — verified `True`), a sibling key that holds a **different quantity** (`n_replicate_groups` = 316/120/124 overall vs the prose's per-metal 4/5), and an artifact with **zero** numeric leaves (`cfm_blk_c_obj_decision.json` — every number is unbacked *by construction*, the rule cannot apply) |
| **UNBACKED_OWN_MEASUREMENT** (a real measurement stated only in prose, no machine field) | **33** | e.g. `msrs_variance_decomposition.json` "309 replicate groups" / "379 of 5,965 potency ligands" / "2,217 of 21,406 units (10.4 %)"; `cfm_metal_geometry.json` "the two draws share MEASURED 329 molecules" / "2,671 of 3,000"; `cfm_l3_decode_post_repair.json` "57/64 failures"; `cfm_panel_reference_band_real_tmqm.json` "1,383 of 1,766 violating atoms" |

The tree-wide **strongest** class (`SELF_CONTRADICTING_PROSE_COUNT`, 12 flags) was adjudicated in
full: **0 TRUE_POSITIVE, 8 BENIGN, 4 UNBACKED_OWN_MEASUREMENT**.

> The scanner's own automatic label said `TRUE_POSITIVE` for 12 flags tree-wide (3 of them in the
> sample). **Hand adjudication downgraded all 3** — they are derived or cross-population counts
> that the artifact simply does not store (`329` = the intersection of two 3,000-row cohorts whose
> own denominators are `n_molecules = 3000`; `33,429` = a figure quoted from a pinned test
> docstring about a *different* population). The scanner's automatic TRUE_POSITIVE label is
> therefore **not** reliable either, and is reported here only as a triage aid.

**Precision statement:** on the 139-flag sample, precision of the *strong* class (prose contradicts
the artifact's own fields) = **0/139**. Precision of the *soft* class (a number the artifact
cannot back, whatever its cause) = **33/139 ≈ 24 %** actionable, the rest benign. A scanner with
24 % actionable precision cannot be a blanket hard gate — which is exactly why the proposed guard
ships with a `declared` mode for the hard gate and keeps `auto` advisory.

---

## 6. The finding that matters: the defect lives in the TOOL

The pre-repair c8 artifact was **internally consistent** — for its own (default) pool every literal
happened to be true, so no scanner over the published JSONs could tell that the prose was not
computed. **The artifact-level census therefore has ≈0 recall for the Wf-A155 class.** The class
is only visible at the *producer*, which is why the guard is a library the producer calls.

The tool-layer scan finds **4 ARMED scripts** (expose `--pool` **and** carry a pool-specific
literal in a string literal). Each will emit false prose if re-run on a pool other than the one
whose numbers were hard-coded:

| Script | Literal sites (file:line) | The hard-coded claim |
|---|---|---|
| `molmetal/scripts/wf_a155_posebusters_validity.py` | L153, L154, L740, L1000, L1005, L1068, L1121, L1167 | "ONE pool of **2112** rows from THREE epoch-wise decodes of ONE checkpoint (**704** rows each …)", "(**529/2112** = 25.05 %)" |
| `molmetal/scripts/wf_a155_metal_coordination_sphere.py` | L283, L313 | "pool of **2112** rows from ONE post-hoc decode of the C8 checkpoint", "(9 runs, not 1 pool of **2112** rows)" |
| `molmetal/scripts/wfa156_bond_threshold_frontier.py` | L1008, L1009, L1010 | "**529/2112** rows with one fragment, **2082/2112** metal atoms sharing a fragment with a non-metal, **30/2112** metal atoms alone, **0/2112** rows …" |
| `molmetal/scripts/wfa142_fragflex_cminus1_probe.py` | L979, L1066 | "primary cohort = epoch 17 (**704** rows); 'all' = all 18 epochs" |

`molmetal/scripts/wfa158_population_reconciliation.py` was **not** counted as armed: its pool is a
module constant with no `--pool` argument, i.e. pool-specific *by design*.

Each of the four already receives the counts it needs — e.g. `wf_a155_posebusters_validity.py`'s
`build_artifact` (def 913) is called with `n_data_rows`, `records` and `epoch_labels` — so the fix
is to **compute** the numbers, not to re-type them. **No diff is proposed for these four scripts
here** (they are other branches' or the main thread's files); they are reported as findings.

---

## 7. Cross-check against the two repaired artifacts (task E)

The cross-check was run with **two independent implementations** (the scanner above and the guard,
which does not import it).

| Artifact | scanner flags | guard findings |
|---|---|---|
| `metrics/by_metric/wfa155_ligand_connected_rate.json` (**repaired c8**, published) | **0** | **0** (pool 0, stale 0, count 0, unbacked 0) |
| `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json` (**repaired c7**, 12,668 rows) | **0** | **0** (pool 0, stale 0, count 0, unbacked 0) |
| `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json` (**pre-repair c8**, preserved by Wf-A162) | **1** — `NARRATIVE_PERCENT_NO_MACHINE_COUNTERPART`, `/purpose`, token `8.51` | 1 soft (`unbacked_percent` `8.51` in `/purpose`) |
| `molmetal/reports/wfa163_narrative_truth_scan/counterfactual_pre_repair_c7.json` (**COUNTERFACTUAL** pre-repair c7) | **5** — 3 × `SELF_CONTRADICTING_PROSE_COUNT` (token `2112` at `/caveats/[2]`, `/comparability/breaks/[0]/ours`, `/not_comparable_to/8_51_percent_headline`), 1 × `SELF_CONTRADICTING_PROSE_POOL` (`cfm_c8_fullgen_2026_10_24`), 1 × `SELF_CONTRADICTING_PROSE_SIBLING` (`60` at `/caveats/[4]`) | 3 blocking-advisory — 1 × `pool_misattribution` (the c8 pool named by a c7 artifact) + 2 × `count_contradiction` (`2112` at the same two sites) |

**Zero flags on both repaired artifacts = CONFIRMATION** that the class was closed there, under two
independent implementations.

### 7.1 The pre-repair c7 artifact does NOT exist on disk

Stated plainly, as required: **the pre-repair c7 artifact is not on disk.** Only three
`ligand_connected_rate*` files exist in the repository; the two relevant ones are **untracked** and
have **no git history**, and the repaired c7 artifact was written by Wf-A162 *after* the defect was
fixed. **No pre-repair c7 artifact was reconstructed from memory.**

What exists instead is a clearly-labelled **COUNTERFACTUAL**, built mechanically (not from memory)
by the scanner's `--counterfactual` mode from three on-disk sources:

1. the preserved pre-repair c8 artifact
   (`…/wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json`),
2. the measured pre/post key-set diff between the pre-repair c8 artifact and the repaired c7
   artifact, and
3. the authoritative table of the 7 pre-repair literal sites and their c7 falsity in
   `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/report.md` §3.

Its provenance is in `counterfactual_pre_repair_c7.meta.json` (notice + reverted-site list); the
artifact itself carries no metadata keys. **It is a demonstration of scanner behaviour, not the
original, and must never be cited as the original.** The scanner's `--counterfactual` flag builds
it reproducibly from the sources above.

### 7.2 Recall limitation (stated, not hidden)

The counterfactual is flagged on **4 of the 7** documented pre-repair sites. The two **semantic**
sites — the identity claims "this is the 8.51 % headline" and "this is the 25.05 % fragment
count" — and site 7 ("Four independent breaks", true in both pools) are **out of reach of a
number-counterpart scanner**: they are claims about *identity*, not about a number the artifact
lacks. A scanner that "catches prose numbers" cannot be expected to catch them, and this report
does not claim it does.

---

## 8. Precision caveat (what may and may not be said)

| Statement | Supported? |
|---|---|
| "583 published artifacts contain prose numbers with no machine counterpart" | **YES** — 482 flags / 169 artifacts (scanner); 474 / 172 (guard) |
| "Some of those are real measurements stated only in prose" | **YES** — 33/139 in the adjudicated sample |
| "The published metrics tree contains false numbers" | **NO** — 0 hard contradictions in 583 artifacts; the 139-flag adjudication found 0 |
| "The published metrics tree is untrustworthy" | **NO** — not supported, and not claimed |
| "4 producer scripts can emit false prose on a non-default pool" | **YES** — 4 armed scripts, 19 literal sites, file:line cited in §6 |

---

## 9. REPAIR SPEC

### 9.1 The rule to enforce

> Every number rendered in a published artifact's **narrative** prose must be derivable from that
> artifact's **own machine fields** (or, for an explicitly attributed cross-reference, from the
> artifact it cites).

### 9.2 The guard (proposed, in the patch)

`molmetal/scripts/wfa163_narrative_truth_guard.py` — a producer-side library:

```python
from wfa163_narrative_truth_guard import assert_narrative_truth

assert_narrative_truth(
    artifact,
    narrative_paths=[("purpose",), ("caveats", "*")],   # declared mode = HARD gate
    artifact_name="wfa155_ligand_connected_rate",
)
```

- **declared mode** → raises `NarrativeTruthError` on an unbacked number or a pool
  misattribution. Precise; use it.
- **auto mode** → no raise by default (`strict=False`); a warning report only.
- `narrative_truth_report(artifact, ...)` returns the same findings without raising, for logging.

### 9.3 The wiring (proposed, in the patch)

`molmetal/scripts/promote_metrics.py` — the **single entry point** into `metrics/by_metric/`.
The gate runs on the payload **before** the write and blocks only on the two CONTRADICTION classes:

- `pool_misattribution` (the artifact names a pool its own `source_pool` does not name), and
- `stale_cross_reference` (a quoted number that no longer matches the artifact it cites).

On a block the promotion returns `status = "BLOCKED_NARRATIVE_TRUTH"` and **nothing is written**.
Measured basis: **0 blocking findings across the 583 published artifacts**, so this gate does not
block any existing promotion.

`promote_metrics.py` was **not modified** by this branch — the hunk is a proposal inside
`patch_narrative_guard.diff` for the main thread to apply.

### 9.4 What a producer should do (the actual fix for §6)

Replace each hard-coded literal with a computed one, and declare the narrative paths. Example
(the shape Wf-A162 used for the a155 tool):

```python
NARRATIVE_PATHS = (("purpose",), ("comparability", "breaks", "*", "ours"))
artifact["purpose"] = f"ONE pool of {n_rows} rows from ONE post-hoc decode ({pool_path})."
assert_narrative_truth(artifact, narrative_paths=NARRATIVE_PATHS,
                        artifact_name="<tool>")
```

### 9.5 Regression gate

`molmetal/tests/test_wfa163_narrative_truth.py` — **15 tests, all passing** (`CUDA_VISIBLE_DEVICES=""`,
`pytest molmetal/tests/test_wfa163_narrative_truth.py -q` → `15 passed`). They pin: the pre-repair
shape fails and the repaired shape passes; pool misattribution is detected and a legitimate
cross-referenced pool is not; a count contradiction is detected; auto mode is advisory and does not
raise; a consistent cross-reference passes and a stale one is flagged; the guard module contains no
`torch`/`cuda` reference and its import is side-effect-free; and the two repaired artifacts on disk
score 0.

---

## 10. Files

| Path | What |
|---|---|
| `molmetal/scripts/wfa163_narrative_truth_scan.py` | the census scanner (this branch's tool) |
| `molmetal/scripts/wfa163_narrative_truth_guard.py` | the proposed repo-wide guard (NEW; in the patch) |
| `molmetal/tests/test_wfa163_narrative_truth.py` | the proposed regression test (NEW; in the patch) |
| `molmetal/reports/wfa163_narrative_truth_scan/narrative_truth_scan.json` | the census: denominator, per-artifact findings, tool-layer scan, guard census, cross-check |
| `molmetal/reports/wfa163_narrative_truth_scan/patch_narrative_guard.diff` | **the patch spec** (guard + test + `promote_metrics.py` wiring) |
| `molmetal/reports/wfa163_narrative_truth_scan/counterfactual_pre_repair_c7.json` | the labelled COUNTERFACTUAL (not the original) |
| `molmetal/reports/wfa163_narrative_truth_scan/counterfactual_pre_repair_c7.meta.json` | its provenance (notice + reverted sites) |
| `molmetal/reports/wfa163_narrative_truth_scan/report.md` | this report |

**Patch verification (MEASURED 2026-11-01):** the diff was written to a scratch copy containing
only `molmetal/scripts/promote_metrics.py` (and empty `molmetal/tests/`), and
`git apply --check -v` returned **PASS** for all three files — **without applying it**. A second
scratch copy was then patched for real, and the result was verified: `ast.parse` clean on all
three files, and `pytest molmetal/tests/test_wfa163_narrative_truth.py -q` → **11 passed, 4
skipped** (the 4 skips are the tests that need the real repository's published artifacts, absent
from the scratch copy).

**Not touched by this branch** (per the branch's file-ownership rules): `paper/**`, `docs/**`,
`AGENTS.md`, `README.md`, `TODO/**`, `configs/**`, `metrics/**`, `registry/**`,
`molmetal/reports/run_ledger.jsonl`, `molmetal/molmetal_lam/**`,
`molmetal/scripts/pre_flight_check.py`, `molmetal/scripts/wf343c1_pre_retrain_check.py`,
`molmetal/scripts/wf_a155_ligand_connected_rate.py`,
`molmetal/tests/test_wfa155_decoder_bond_candidacy.py`.
