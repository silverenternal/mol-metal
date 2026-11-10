# Wf-A164 -- REFUTATION of BRANCH B (`molmetal/scripts/wf_a155_metal_coordination_sphere.py`)

**VERDICT: NOT REFUTED.**  Every load-bearing claim was independently re-derived from disk
and from the PRIMARY record and reproduced.  Findings are in the branch's own census METADATA
and in residual wording/coverage -- none re-enables the pool-specific-literal defect class.

## Checks (1-5, in the order required)

| # | Check | Result |
|---|---|---|
| 1 | literals gone + replacements COMPUTED | PASS (2 census-metadata defects: D1, D2) |
| 2 | regression test passes AND its negative control fails the pre-repair behaviour | PASS |
| 3 | the guard fires on an unbacked prose number | PASS |
| 4 | default pool still runs, still well-formed | PASS |
| 5 | scope: no metrics/ write, no registry/ edit, no other branch's file touched | PASS |

## Evidence

1. **Literals gone.**  No `8.51` / `25.05` / `2112` / `|r| < 0.09` in the script; the surviving
   `92.7-94 %` is the labelled frontier constant (`:305`).  `25.05` <- re-measured `529/2112`
   (33.33 for a 6-row pool); `8.51` <- `end_to_end_headline.median = 0.08506944444444445`,
   `n_runs = 9`, read at run time from `metrics/by_metric/cfm_fragmentation_audit.json`
   (independently re-read: identical values).
   **Default-pool numeric identity re-derived**: 64093 common leaves, **0** numeric leaves
   changed, 15 string leaves changed, 145 added, 0 removed -- the branch's diff report is exact,
   and all 15 of its `old` strings match the published pre-repair artifact byte-for-byte.
   **Frontier provenance re-verified live**: arXiv:2603.27113v1 PDF sha256
   `03cc2f6c...b1da2` == the branch's cited `retrieval_sha256`; Table 1(b) rows read verbatim
   (`Val&Uniq 92.7+/-0.5 | 93.3+/-0.0* | 92.9+/-0.2`, `PB-Valid 93.1+/-0.4 | 91.9+/-0.5 | 94+/-0.5*`);
   section 4.1 + "10,000 unconditional samples" confirmed; EF-TALFM `87.9 %` counter-quote confirmed
   against arXiv:2609.08333v4.
2. **Test bites.**  `pytest molmetal/tests/test_wfa164_coordination_sphere_narrative.py -q`
   -> **15 passed** (3.26 s, `CUDA_VISIBLE_DEVICES=""`).  Negative control present and
   load-bearing: the pre-repair `breaks`/`nots` make the guard FAIL naming `2112`/`8.51`/`25.05`;
   a second control shows the naive prefix resolution would miss
   `comparability_statement.breaks[0].ours`.
3. **Guard not vacuous.**  Independent injections: `77.31` -> `pass=False`,
   `unbacked_numbers={'77.31': [...]}`; `12345.67` -> `pass=False`.  An unresolved narrative
   path -> `status: FAIL`, exit 1.
4. **Nothing broke.**  Fresh default-pool run -> exit 0, `PASS (46 prose numbers, 42816 machine
   numbers)`, 1264768 chars, all 12 legacy top-level keys present.  Different pool -> PASS, 6 rows,
   33.33 %, none of the 4 stale literals, and the wording switches to
   `n = 4 metal atoms, so the approximation is ROUGH (n is NOT in the thousands)`.
5. **Scope.**  `find metrics registry -newermt '2026-11-01 11:30'` -> **empty**.
   `metrics/by_metric/wfa155_metal_coordination_sphere.json` mtime 00:56 (untouched);
   `registry/metrics.yaml` mtime 10:40 (untouched); the conflicting legacy test
   `test_wfa155_metal_coordination_sphere.py` mtime 00:55 (untouched -- correctly left to its
   owner).  Branch B's writes: its script, its own test, its own report dir.

## Defects found in the branch

- **D1 (minor)** 12 of the census's 15 `src_line` anchors do not point at their site in the
  shipped script (deltas -9..+785; e.g. `finding_5` claimed L1239, actually L2024).  The branch's
  OUTPUT repeats the stale numbers -- the wave's own rule 9 ("N sites with a file:line for each")
  is not met.
- **D2 (minor)** the census's denominator does not reproduce: measured 915 `ast.Constant` (str)
  + 38 f-strings = **953** string literals, not the claimed 916 + 38 = 954.
- **D3 (moderate)** the repaired `break_3_post_processing` says the paper "reports every model
  twice, once RAW and once POST-PROCESSED, so 'the frontier number' is one of two numbers
  depending on the arm".  The primary PDF's section 4.1 says the raw/PP pair is GEOM-DRUGS
  (Table 2, 100,000 samples); QM9 (Table 1(b), 10,000 samples) is raw-only -- so the 92.7-94 %
  band has ONE arm.  The artifact's own `frontier_reference.primary_record.arms` field says so;
  the two disagree, and the frontier block is outside `NARRATIVE_PATHS` so the guard misses it.
  The OLD text was strictly FALSE, so this is an improvement, not a regression.
- **D4 (minor)** the `arms` quote inserts a colon the record does not have ("On QM9: we report"
  vs the record's "On QM9 we report").
- **D5 (moderate)** the guard is loose for INTEGER prose tokens: 2117 distinct integer machine
  numbers, 2112 of the integers 1..5000 coincidentally present, so an integer literal of the
  original defect's shape can be "backed" by an unrelated count.  2-decimal tokens are strict
  (98/5000 coincidentally backed).  A net, not a proof.
- **D6 (info)** the narrative set is a chosen subset: 2150 string leaves with numeric tokens sit
  outside it; the guard reports 52 prose leaves and never reports the unchecked count.

## Surviving concerns

- **C1 (blocking for the repo)** the test tree is RED: `test_wfa155_metal_coordination_sphere.py::
  test_json_schema` FAILS (1 failed, 23 passed, 5.35 s) at
  `molmetal/tests/test_wfa155_metal_coordination_sphere.py:351` -- `assert "8.51" in
  nots["not_8_51_percent"]`.  Correctly escalated to the owning branch, but not yet fixed.
- **C2 (high)** the PUBLISHED artifact still carries the defect: `metrics/by_metric/
  wfa155_metal_coordination_sphere.json` (untouched, `tag: MEASURED`) still states "the 8.51 %",
  "the 25.05 %", "ONE pool of 2112 rows ... of the C8 checkpoint" and "Four independent breaks".
  The repaired tool exists; the published tree still serves the unrepaired artifact.
- **C3 (moderate)** D3 + D5 above.

Reviewer: 0 GPU throughout (`CUDA_VISIBLE_DEVICES=""` on every call, no torch import, no decode,
no trainer).  Full machine-readable record: `refutation.json`.
