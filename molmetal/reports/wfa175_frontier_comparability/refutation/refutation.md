# Wf-A175 — REFUTATION of branch FRONTIER (adversarial)

**Date:** 2026-11-01 · **GPU:** 0 (CPU-only) · **Target:** `molmetal/reports/wfa175_frontier_comparability/report.md`
**Verdicts:** REFUTED 0 · SURVIVED 3 · IMPRECISE 3 — detail in `refutation.json`.

**Method.** Re-fetched 3 e-prints + 1 abs page + verified 1 cached PDF by sha256 against the report's own §0 ledger; confirmed the cached extracted text is byte-identical to the re-fetched tarball (4 files); re-read every cited locator at the cited line; re-derived every ratio; ran an independent band census over **all 9,551 HEAD blobs**, **all 134 paper/ files on disk**, **all 10,067 non-ignored working-tree files** and **95,714 gitignored text files** with the branch's regex **and** a broader one.

## Per-claim

| claim | verdict | one-line |
|---|---|---|
| 1 — 9 papers widened | **SURVIVED** | Every cited quote is verbatim at the cited line and byte-tied to a sha I reproduced. But DecompDiff's verdict is `n/a` and SemlaFlow has no §1 row, so "each with a RAW/POST verdict" holds for 8 of 9. |
| 2 — 11.75139× is 5.955312× the frontier max | **SURVIVED** | All load-bearing numbers re-derive exactly (11.75139146567718, 1.9732620320855614, 5.955312206183287, 133.0×, 1.08952×). 4 cells in §2.1's ratio column contradict the branch's own `tools/ratios.json`. |
| 2b — our RAW 8.51 % below every frontier RAW | **IMPRECISE** | The branch's own §1 row 3 measures TABASCO connectivity = **0.00** (verdict RAW) at 10–20 steps — below ours. The comparison also mixes denominators (DiffSBDD = P(connected\|valid) ≈ 61.0 % unconditional; ours = P(connected\|parsed)). |
| 3 — band is frontier-sourced, never in paper/, never at HEAD | **IMPRECISE** | Provenance + both negatives CONFIRMED and strengthened (I scanned all 9,551 HEAD blobs and all 134 paper/ files → 0). But the report says the 11 gitignored paper/ files "were scanned" (its own census says 123, not 134); "0 of 10,060 committed blobs" mislabels the denominator (HEAD = 9,551); "94.0 is 0 of 134 in paper/" is false (1 SVG); the census excluded gitignored paths repo-wide (26 more band-carrying files, none in paper/); and the mis-attribution is **not** single — `molmetal/scripts/wfa161_frontier_labelling.py` -> PYTHON CONSTANT `WFA176_CORRECTION_SECTION` carried "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel)…" and would re-emit it (that defect was LIVE when this refutation was written on 2026-11-01; the producer was repaired on 2026-11-02 and the string now survives only inside that DATED constant — see the Wf-A200 section below). |
| 4 — 87.9 % confirmed, 89.4 % is GeoLDM's | **SURVIVED** | I reproduced `89.4`=0 / `87.9`=2 on the e-print and `89.4`=0 / `87.9`=3 on the live abs page; the 87.9 sentence is verbatim. Defects: locator path is `src/results_unconditional.tex:35`, not `sections/…`; the abs-page half is **not archived** (no bytes, no sha in §0). |
| 5 — EF-TALFM's sentence contradicts its table | **IMPRECISE** | Conclusion right (92.98 % must be RAW; no RAW/POST pair exists). Premise wrong: `src/appendix.tex:60` states "Connectivity: % of molecules with exactly one connected component, i.e., one fragment." — :183 is a loose caption, not a contradiction. |

## The single most important correction

**Our "same 3-step PP as Buttenschoen" is 2 of 3 steps.** `metrics/by_metric/cfm_fragmentation_postproc_arm.json` records `uff_ok_rate_over_postprocessed = 0.0` and `uff_reasons {"ValueError: Bad Conformer Id": 12668}` — the UFF step ran on **0 of 12,668** molecules (no UFF parameters for this corpus's metal complexes). `report.md` mentions "UFF" exactly once and never discloses this, so §6.2's "our ratio is directly comparable to that column" is unsound as written. (The connectivity rate itself is fixed by step 1, so 11.75139× is unaffected.)

## Reproduction

```
molmetal/reports/wfa175_frontier_comparability/refutation/tools/census_indep.py   # band census, HEAD + paper/ + gitignored
molmetal/reports/wfa175_frontier_comparability/refutation/tools/show.py A:file:lo:hi  # cited-locator reader
```

---

## Wf-A200 CROSS-WAVE LOCATOR REPAIR (2026-11-02) -- this file's line-number locators are demoted to durable locators

**The defect class.**  A LIVE `<path>:<line>` citation into another wave's
artefact, where that artefact was re-rendered and the line number now
points at a DIFFERENT sentence while the citation still looks plausible.

**What was found.**  The claim-3 row of this refutation cited
`molmetal/scripts/wfa161_frontier_labelling.py:934`.  MEASURED 2026-11-02
on disk, line 934 now reads:

```text
     validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
```

and the mis-attribution string is at line **913**, inside the python
constant `WFA176_CORRECTION_SECTION` (declared at line 905):

```text
WFA176_CORRECTION_SECTION = """\
> 2. **Post-processing.** The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.
```

The durable locator that replaces the line number is the CONSTANT NAME
plus its sub-heading; a line number into another wave's mutable file
rotates on every edit above it.

**The superseded rendering, preserved VERBATIM (AGENTS.md §5.5 rule #6):**

```text
| 3 — band is frontier-sourced, never in paper/, never at HEAD | **IMPRECISE** | Provenance + both negatives CONFIRMED and strengthened (I scanned all 9,551 HEAD blobs and all 134 paper/ files → 0). But the report says the 11 gitignored paper/ files "were scanned" (its own census says 123, not 134); "0 of 10,060 committed blobs" mislabels the denominator (HEAD = 9,551); "94.0 is 0 of 134 in paper/" is false (1 SVG); the census excluded gitignored paths repo-wide (26 more band-carrying files, none in paper/); and the mis-attribution is **not** single — `molmetal/scripts/wfa161_frontier_labelling.py:934` still carries "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel)…" and would re-emit it. |
```

**The dated correction.**  The claim "the defect is still live in code"
was TRUE when this refutation was written (2026-11-01) and is no longer
true: the producer's LIVE "Comparability breaks" item 2 is the REPAIRED
two-reading text, and the mis-attribution survives only inside the DATED
`WFA176_CORRECTION_SECTION` constant, so it is no longer re-emitted as a
live claim.  The `counter_evidence` entry in `refutation.json` carries the
same correction, and the superseded strings survive VERBATIM in its
`crosswave_locator_repair_2026_11_02_wfa200` key.

### Wf-A200b -- the SECOND live locator this wave found (the JSON's
live counter-evidence), and the dated record that replaced it

The corrected auditor reports **exactly one** live locator left in this
branch, and it is in the JSON, not in this markdown file:

```text
refutation/refutation.json  line 84  ->  report.md : 139
```

`verdicts.CLAIM_3.counter_evidence[3]` opens by quoting the OLD title --
"'THE ONLY MIS-ATTRIBUTION IS report.md:139' IS INCOMPLETE".  The bare
filename resolved to THIS branch's own `report.md`, and MEASURED on
2026-11-02 that file's line 139 is an **EMPTY line** -- the locator pointed
at nothing at all.  The sentence meant the Wf-A161 record.

**The superseded rendering, preserved VERBATIM (§5.5 rule #6),** in
`refutation.json` -> `crosswave_locator_repair_2026_11_02_wfa200` ->
`superseded_live_counter_evidence_3_verbatim_2026_11_02_wfa200_b`:

```text
'THE ONLY MIS-ATTRIBUTION IS report.md:139' IS INCOMPLETE, AND THE DEFECT WAS STILL LIVE IN CODE WHEN THIS REFUTATION WAS WRITTEN (2026-11-01). molmetal/scripts/wfa161_frontier_labelling.py -> PYTHON CONSTANT 'WFA176_CORRECTION_SECTION' carried the same string 'stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a ' -- the generator that wrote the wfa161 report. The mis-attribution was therefore still in the producer and would have been re-emitted on the next run; AGENTS.md 5.5 append-only in the report cannot close it. DATED UPDATE 2026-11-02 (Wf-A200): the producer's LIVE 'Comparability breaks' item 2 is now the REPAIRED two-reading text, and the mis-attribution string survives ONLY inside the DATED 'WFA176_CORRECTION_SECTION' constant, so it is no longer re-emitted as a live claim; the superseded line-number locators are preserved VERBATIM in crosswave_locator_repair_2026_11_02_wfa200. (Independently: I swept all 39 band-carrying files for an 'our'-attribution context; the only rendered document that calls the band ours is wfa161's report, and the only other literal occurrence is wfa161's generator script.)
```

**The durable locator that replaced it** (in the LIVE `counter_evidence[3]`):

```text
'THE ONLY MIS-ATTRIBUTION IS THE Wf-A161 RECORD'S DATED `## Wf-A176 CORRECTION` SECTION' IS INCOMPLETE, AND THE DEFECT WAS STILL LIVE IN CODE WHEN THIS REFUTATION WAS WRITTEN (2026-11-01). molmetal/scripts/wfa161_frontier_labelling.py -> PYTHON CONSTANT 'WFA176_CORRECTION_SECTION' carried the same string 'stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a ' -- the generator that wrote the wfa161 report. The mis-attribution was therefore still in the producer and would have been re-em ...
```

The verdicts are untouched: REFUTED 0 / SURVIVED 3 / IMPRECISE 3, and no
NUMERAL of this refutation changed.

### Wf-A200e -- two DATED corrections to this branch's own Wf-A200 record
(the live index label, and the live-drift count)

Both corrections are ADDITIVE and DATED (AGENTS.md 5.5 rule #6).  Nothing
above this section was edited; the superseded renderings are quoted
VERBATIM below and the corrections live here.

**Correction 1 -- the live list index label is off by one.**  The dated
record's key NAMES say `what_i_checked[3]` and `counter_evidence[3]`, but
MEASURED on disk on 2026-11-02 the repaired entries are at index **4** (the
FIFTH element of a FIVE-element list):

```text
verdicts.CLAIM_3.what_i_checked   length 5   repaired entry at index 4
verdicts.CLAIM_3.counter_evidence length 5   repaired entry at index 4
```

An index label is a LOCATOR into a list: naming the wrong element points at
a DIFFERENT sentence -- the same defect class this wave repairs.  The key
NAMES are preserved VERBATIM (append-only) and the correction is recorded
in `refutation.json` -> `crosswave_locator_record_correction_2026_11_02_wfa200e`
-> `correction_1_the_live_list_index_label`.  The CONTENT of those two
entries is unchanged.

**Correction 2 -- the live-drift count.**  The superseded rendering,
preserved VERBATIM:

```text
The corrected auditor reports **exactly one** live locator left in this
branch, and it is in the JSON, not in this markdown file:
```

That was TRUE of the auditor as it stood when the Wf-A200b section was
written (2026-11-02, before the Wf-A200d CLAIM-ATTRIBUTION repair), when
the occurrence was classified `RESOLVES-BUT-DIFFERS`.  MEASURED NOW over
this branch (`wfa200_crosswave_locator_audit.py --paths
molmetal/reports/wfa175_frontier_comparability`; 64 files in the census, 38
scanned, 226 locator hits, wall 1.606 s):

| class | count |
|---|---|
| `RESOLVES-BUT-DIFFERS` (live drift) | **0** |
| `RESOLVES-NO-CLAIM` | 2 |
| `TARGET-IS-SELF` | 2 |
| `DATED-SUPERSESSION` | 70 |
| `SNAPSHOT` | 152 |
| `RESOLVES-AND-MATCHES` / `TARGET-ABSENT` / `TARGET-AMBIGUOUS` | 0 |

The count is now zero for TWO independent reasons, both MEASURED.  (1) After
the Wf-A200b repair the perishable locator survives ONLY inside the DATED
key `crosswave_locator_repair_2026_11_02_wfa200`, so the auditor
classifies it `DATED-SUPERSESSION` -- a supersession record that AGENTS.md
5.5 rule #6 requires to survive.  (2) MEASURED on a /tmp COPY of this tree
with the OLD title restored into the LIVE leaf, the occurrence is
`RESOLVES-NO-CLAIM`: the locator sits INSIDE the quoted old title, so it is
a POINTER (a sub-reference inside a quote about something else), not a
machine-verifiable claim.

**The REPAIR stands.**  The count changed; the defect did not.  Two measured
facts are unaffected by either correction: the bare filename `report.md` was
AMBIGUOUS -- it did not identify WHICH branch's report was meant -- and this
branch's own `report.md` line 139 is an **EMPTY line**, so the locator
pointed at nothing at all.  The durable locator (the Wf-A161 record's DATED
`## Wf-A176 CORRECTION` section) is still the right repair.

The repo-wide record of the same run -- 108 live drift in 46 citing files,
none of them in this branch -- is published at
`molmetal/reports/wfa200_crosswave_locator_audit/report.json` and
`report.md`.

### Wf-A200f -- the numeral 94.0 is NOT unique to HLTF (a MEASURED
counter-example), and the branch's live-drift count after that correction

**A MEASURED counter-example to this branch's own record.**  The dated
primary-source record (`sources.json` ->
`crosswave_locator_repair_2026_11_02_wfa200` ->
`primary_source_reverification_2026_11_02` -> `cells_measured`["94.0"]) says:

```text
the numeral 94.0 appears LITERALLY only in HLTF's abstract (iclr2025_conference.tex:55, 'improving PoseBusters validity to 94.0%'); the table cell prints '94\pm0.5' (qm9_benchmarks_og.tex:66, row 'PB-Valid (%)', HLTF column).  Same cell, two renderings.
```

Read over the frontier corpus this branch pins, "LITERALLY only" is **FALSE**.
MEASURED on disk on 2026-11-02 with the repo's own reader
(`refutation/tools/show.py` over the pinned `cache/2609.05097v1.txt.gz`;
CPU-only, **no network call**):

```text
NEAT-POCKET arXiv 2609.05097v1  sections/results.tex:23   CrossDocked & 94.0 $\pm$ 4.7
NEAT-POCKET arXiv 2609.05097v1  sections/results.tex:126  Overall, 94.0\% of the CrossDocked test-set ligands are PB valid.
```

with the e-print pinned at sha256
`bd978841f62a7d75e8635714e7336a6e3abec39b854ec3f2c90974890f2b7fdb` and
10,632,429 bytes; the tarball member `sections/results.tex` is recorded at
40,202 bytes and the ON-DISK extracted text is 40,208 bytes with sha256
`d9d65a1db56b9c9ed5a8ad2bd48c0623a5a281377f381299b36d79129ef96e5d` (the cached
`.txt.gz` is the tarball FLATTENED and re-encoded, so the two byte counts
differ by 6).

**The corrected rendering is SCOPED**: WITHIN HLTF arXiv 2603.27113v1 the
numeral 94.0 appears literally only in the abstract; OVER THE FRONTIER CORPUS
it is not unique to HLTF.  The superseded sentence above is preserved VERBATIM
in `sources.json` ->
`primary_source_reverification_correction_2026_11_02_wfa200f` ->
`superseded_rendering_verbatim` and was NOT rewritten (AGENTS.md 5.5 rule #6).

**What this does NOT change.**  NEAT-POCKET's 94.0 is a DIFFERENT quantity --
the PB-validity of the CrossDocked TEST-SET REFERENCE LIGANDS (real docked
molecules), on a DIFFERENT test set -- so it is a THIRD candidate for the
numeral 94.0, listed next to the branch's existing third candidate for 92.7
(YuelBond's per-bond-type classification F1 92.7 %).  Readings A (92.7 / 94.0)
and B (92.98 / 94.33) still disagree on the numerals; the band's endpoint
cells are still NOT settled on disk.  The verdicts are untouched: REFUTED 0 /
SURVIVED 3 / IMPRECISE 3, and no NUMERAL of this refutation changed.

**The branch's live-drift count after this correction.**  Re-measured with the
final auditor (`wfa200_crosswave_locator_audit.py --paths
molmetal/reports/wfa175_frontier_comparability`; 64 files in the census, 38
scanned, 256 locator hits, wall 1.62 s):

| class | count |
|---|---|
| `RESOLVES-BUT-DIFFERS` (live drift) | **0** |
| `RESOLVES-NO-CLAIM` | 2 |
| `TARGET-IS-SELF` | 2 |
| `DATED-SUPERSESSION` | 100 |
| `SNAPSHOT` | 152 |
| `RESOLVES-AND-MATCHES` / `TARGET-ABSENT` / `TARGET-AMBIGUOUS` | 0 |

The counts moved from the Wf-A200e measurement (226 hits, 70
DATED-SUPERSESSION) because the Wf-A200f dated key and the DATED
sections that carry it add 30 line-number
locators, ALL of which classify as `DATED-SUPERSESSION` -- a dated record's
locators are historical by construction.  The live drift is still **0**.
