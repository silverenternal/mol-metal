# Wf-A164 — synthesis (pool-specific literals in four `tag: MEASURED` tools)

Scope: read-only except this file. 0 GPU; single venv `/home/hugo/codes/mol-metal/.venv/bin/python`;
every command prefixed `CUDA_VISIBLE_DEVICES=""`. Sources: the 4 branch reports + 4 refutation
artifacts on disk (`molmetal/reports/wfa164_{branch_a,coordination_sphere,c,fragflex_cminus1}_refutation/`).
Every number below is MEASURED and traces to a path on disk.

---

## 1. Per-tool verdicts

### A — `molmetal/scripts/wf_a155_posebusters_validity.py` (1646 → 2777 LOC)
* **Literals removed**: 32 lines of the 1646-line pre-repair file carried a pool-specific literal;
  **9** of them reached the `tag: MEASURED` artifact (empirically proven: the pre-repair snapshot on
  a 5-row copy pool emitted "704 rows each = 2112", "6 seeds per epoch", `cfm_yuelbond_wasserstein_seed0`,
  "…/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"). Pre-repair `file:line`:
  L153-155, L739-740, L988, L999-1001, L1005-1006, L1062, L1068, L1071-1072, L1074, L1121, L1167-1169.
  Pre-repair file preserved at `molmetal/reports/wfa164_posebusters_narrative/pre_repair_script_snapshot.py.txt` (1646 lines).
* **Test bites**: YES. `molmetal/tests/test_wfa164_posebusters_narrative.py` — 9 passed (5.68 s, per
  `default_pool_diff.json:regression_test`); my combined run of the 4 new files = **44 passed** (20.38 s).
  Negative control restores the old literal → guard FAIL, `unbacked_numbers {2112, 704, 6}` + `stale_paths`.
* **Refuter**: **NOT REFUTED** (`wfa164_branch_a_refutation/refutation.json`: C1.1/C2-C7 PASS, C1.2 REFUTED).
* **Default-pool artifact text**: **CHANGES**. `metrics/by_metric/wfa155_posebusters_validity.json`
  sha256 `3772fa65…3817b` — unchanged on disk (re-verified by sha256sum).
  **Caveat on the headline number**: the 1262 diff lines compare the published FULL-pool artifact against
  the repaired **BOUNDED 20-row** artifact (`default_pool_diff.json:bound.disclosure`), so 1262 is not the
  repair's own delta. The honest same-pool/same-bound/same-seed original-vs-repaired delta is **566 lines**
  with measured counts IDENTICAL (1/20 strict, 1/13 scored, 7 embed-failed, 9 FF-failed).
  ⇒ **REPUBLISH: YES** (a full-pool repaired artifact does not exist yet).
* **REGRESSION the branch and its refuter both missed**: `COMPARABILITY_BREAKS` existed at
  `pre_repair_script_snapshot.py.txt:144` and was removed from the tool ⇒
  `molmetal/tests/test_wfa155_posebusters_validity.py` now fails **at collection** with
  `ImportError: cannot import name 'COMPARABILITY_BREAKS'` (measured). It also pins the two renamed keys
  `not_a_restatement_of.the_8_51_pct_headline` / `.the_25_05_pct_fragment_count` at `:589-590`.
* **Defect of the same class that survives** (on disk in the branch's own evidence file
  `wfa164_posebusters_narrative/default_pool_bounded_repaired.json`): `.not_a_restatement_of.single_pool_fragment_count`
  = **"529/20 = 25.05 %"** and **"all 20 rows of the pool"** — the numerator 529 is the POOL's, the denominator
  20 is `counts.n_rows_in_scope` (the scope's own single-fragment rows = 5). 529/20 = 2645 %, and "all 20 rows
  of the pool" is false for the 2112-row pool. The guard **passed** it (`narrative_self_consistency.pass = True`,
  `unbacked_numbers {}`): backing ≠ correctness.

### B — `molmetal/scripts/wf_a155_metal_coordination_sphere.py` (2570 LOC)
* **Literals removed**: **15** sites; denominator 954 string literals (916 `ast.Constant` + 38 f-strings) —
  the refuter re-measured **953** (915 + 38), i.e. the branch's denominator is off by one.
  Census: `molmetal/reports/wfa164_coordination_sphere_narrative/pool_specific_literal_census.json`
  (`n_sites_repaired: 15`, `n_pool_specific_literals_remaining_in_the_artifact_emitting_code: 0`).
  Pre-repair `src_line` anchors: L15, L431, L452, L467, L497, L565, L566, L567, L1239, L1459, L1842, L1872,
  L2149, L2157, L2178. **12 of 15 do not resolve in the repaired file** (e.g. `finding_5`: census L1239 →
  repaired L2024) — they are pre-repair coordinates.
* **Test bites**: YES. `molmetal/tests/test_wfa164_coordination_sphere_narrative.py` — 15 passed (3.26 s);
  negative control names 2112/8.51/25.05.
* **Refuter**: **NOT REFUTED** (every load-bearing claim re-derived; frontier PDF sha256
  `03cc2f6c…b1da2` == branch's `retrieval_sha256`).
* **Default-pool text**: **CHANGES** (`default_pool_diff.json:verdict.default_pool_text_changes = true`):
  64093 common leaves, **42787 numeric leaves, 0 changed**, 15 string leaves changed, 145 added, 0 removed.
  Published sha256 `c2a49126…1c2b4` unchanged on disk. ⇒ **REPUBLISH: YES**.
* **Repo is RED**: `molmetal/tests/test_wfa155_metal_coordination_sphere.py::test_json_schema` FAILS
  (measured **1 failed, 23 passed**) at `:351` `assert "8.51" in nots["not_8_51_percent"]`; `:351-352` pin
  the defect.

### C — `molmetal/scripts/wfa156_bond_threshold_frontier.py` (1944 LOC)
* **Literals removed**: **6 sites on 4 surfaces**. Denominator: 1453 pre-repair lines (**not traceable to disk** —
  refuter concern 6; the script is untracked and was overwritten in place), 1944 now, 1053 string literals /
  186 digit-bearing. Sites: L291-292 (caveat "2,112-row" + pool path), L309-310 (caveat "every one of our pool
  rows carries a metal (metal_ratio 1.0)" — **FALSE** on c7: 12668/12669), L1006-L1012
  (`population_cross_check.note` 529/2112, 2082/2112, 30/2112, 0/2112), L1075-L1076 (hard-coded master.log
  quotes "518,276 params"), L1099 (hard-coded run-dir contents), L50/L61 (docstring).
* **Test bites**: YES. `molmetal/tests/test_wfa164_bond_threshold_narrative.py` — 10 passed; an independent
  negative control that restores the pre-repair note/caveat makes **3 of 10 FAIL**.
* **Refuter**: **NOT REFUTED** (5/5 checks PASS; re-scanned the c7 12,669-row pool).
* **Default-pool text**: **CHANGES** (`default_pool_diff.json:verdict.text_changed = true`): 1260 measurement
  leaves compared, **0 changed / 0 removed**; 46 machine leaves added; 2 changed (the two narrative strings);
  markdown 313→314 with **27** changed lines.
  ⇒ **REPUBLISH: YES** — but the published copy is `molmetal/reports/wfa156_bond_threshold_frontier/{report.json,report.md}`
  (**not** under `metrics/`). There is **no registry row** to re-point (see §5).

### D — `molmetal/scripts/wfa142_fragflex_cminus1_probe.py` (1598 LOC)
* **Literals removed**: 10 pool/cohort literal sites of **582 string literals, 50 digit-bearing**
  (refuter reproduced 582/50 exactly); **7 repaired, 3 deliberately left**. Repaired (pre-repair `file:line`):
  L785-787 (`load_pool` docstring), L979-981 (`--cohort` help), L1066, L1069, L1073-1074, L1081
  (CSV **line** count → measured **record** count), L899-906 ("this function reproduces it").
  Census: `molmetal/reports/wfa164_fragflex_cminus1_narrative/literal_census.json`.
* **Test bites**: YES. `molmetal/tests/test_wfa164_fragflex_cminus1_narrative.py` — 10 passed; the refuter's own
  experiment (guard disabled + old literal → `rc=0, status=PASS`) makes the branch's `assert rc == 1` fail
  pre-repair. With the pre-existing `test_wfa142_fragflex_cminus1.py` (15) → 25 passed (my run of the C+D
  legacy pair: **26 passed**).
* **Refuter**: **NOT REFUTED**; but the sub-guard `cohort_detail_recheck` is **tautological** (L1380 and L1469
  both call the same pure `derive_cohort_detail`, defined L912).
* **Default-pool text**: **CHANGES BY ADDITION** (`default_pool_diff.json`): 360 of 367 shared leaves
  byte-identical, **0 changed**, 61 leaves only in the new artifact; `cohort_detail` byte-identical
  ("epoch 17 rows of the pool (704 rows)"); `verdict.default_pool_numeric_results_change = false`;
  pool sha256 `2d7b7c44…7384` identical.
  ⇒ **REPUBLISH: YES** — published copy `molmetal/reports/wfa142_fragflex_cminus1/probe.json` (**not** under `metrics/`).

**Cross-cutting measured state of the repo (my own runs, CPU-only):**
`registry.py check` → **PASS, 579 current / 0 stale / 0 missing**; 4 new Wf-A164 test files → **44 passed**;
`test_wfa155_metal_coordination_sphere.py` → **1 failed, 23 passed**; `test_wfa155_posebusters_validity.py` →
**collection ERROR (ImportError COMPARABILITY_BREAKS)**; `test_wfa156_bond_threshold_frontier.py` +
`test_wfa142_fragflex_cminus1.py` → **26 passed**. No branch touched `metrics/`, `registry/`, `docs/`, `TODO/`,
`paper/`, `configs/` (the two published `metrics/` artifacts still carry sha256 `3772fa65…` / `c2a49126…`).

---

## 2. Concerns that SURVIVED a refutation

1. **A — scope-vs-pool ratio conflation** (branch A's own evidence file): "529/20 = 25.05 %" + "all 20 rows of the
   pool" (`default_pool_bounded_repaired.json`). Same defect class as the wave; still on disk.
2. **A — guard blind spot on internal ratio consistency** (refuter C3): the guard checks *backing*, not *correctness*.
3. **A — `COMPARABILITY_BREAKS` removed ⇒ legacy suite RED at collection** (measured ImportError). Reported by
   neither branch A nor refuter A.
4. **A — `pre_repair_script_snapshot.py.txt` is an executable second copy of a measurement tool** (kept in-repo because
   the script is untracked in git).
5. **A — docstring-only pool literals remain** at `:343/:453/:538/:638/:966/:1452` (never emitted).
6. **B — repo RED**: `test_wfa155_metal_coordination_sphere.py::test_json_schema` fails at `:351` (`:351-352` pin the defect).
7. **B — published artifacts still carry the stale literals**: `metrics/by_metric/wfa155_metal_coordination_sphere.json`
   → 8.51 ×2, 25.05 ×2, "2112 rows" ×2; `metrics/by_metric/wfa155_posebusters_validity.json` → 2112 ×15, 25.05 ×3,
   "704 rows" ×5, pool path ×2.
8. **B — D3 (material)**: the repaired `break_3` states the paper "reports every model twice, once RAW and once
   POST-PROCESSED", but arXiv:2603.27113v1 §4.1 puts the raw/PP pair on **GEOM-DRUGS only** — the artifact's own
   `frontier_reference.primary_record.arms` field says so, so the 92.7-94 % band has one arm.
9. **B — D5**: the guard is loose for **integer** prose tokens (2112 of 1..5000 coincide with machine numbers).
10. **B — D2**: the census denominator does not reproduce (953 measured vs 954 claimed).
11. **B — D1**: 12 of 15 census `src_line` anchors do not resolve in the repaired file.
12. **B — D6**: narrative coverage is a chosen subset (2150 string leaves with numbers outside `NARRATIVE_PATHS`).
13. **C — guard checks BACKING not CORRECTNESS** (a false "6/6 rows with one fragment" passes; truth 3/6).
14. **C — a FAIL run leaves a stale `report.md`** in a reused out-dir (written only on the PASS branch).
15. **C — `FRONTIER_BAND "92.7-94 %"` left in place**, and `report.md:44` still calls it "the frontier band" with no
    restatement label.
16. **C — `render_markdown:1557` still hard-codes "directory holds no coordinate dump"** outside `NARRATIVE_PATHS`
    (true for all 12 pools on disk → latent).
17. **C — "1,453 lines pre-repair" is not traceable** (untracked script, no snapshot).
18. **D — `cohort_detail_recheck` is tautological** (L1380/L1469 → `derive_cohort_detail`).
19. **D — "backed" ≠ "correct"**: `'spanning 3 epochs'` on an 18-epoch pool and `'(12,669 rows)'` for a 704-row
    cohort both PASS; small ints (0,1,4,5,10,20,142,2026) are spuriously backed by the pool path / `method.gpu`.
20. **D — raw diff is 7 changed leaves**, not 0; `compare_default_pool.py:52-54` excludes `wall_seconds*` /
    `generated_utc` without disclosing it in prose.
21. **D — the script, its pre-existing test and `probe.json` are all untracked in git** → this repair is a
    working-tree change only, with no diff and no pre-repair snapshot on disk.
22. **Registry — none of the four has a registry row**: `registry/metrics.yaml` has **0** hits for `wfa155`,
    `wfa156`, `wfa142` → **there is no registry sha to re-point**; `registry.py check` still PASSes (579 current),
    which means the "published" artifacts are not registry-citable.
23. **Two different "8.51 %" numbers**: runtime read `median = 0.08506944444444445` (n = 9,
    `metrics/by_metric/cfm_fragmentation_audit.json`) vs `0.08509630565203663` at
    `/blind_validity_signature/3/single_fragment_ratio`, quoted as 8.51 % in
    `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:345`.

---

## 3. Sites deliberately LEFT ALONE (with the reason)

* **FRONTIER constants** (92.7-94.0; 92.98/98.25/98.21/94.33; 99.9±0.1; 91.9±0.7) — A, B, C. They describe the
  FRONTIER papers, not our pool, so they are true for every pool; both arXiv records re-fetched live 2026-11-01
  (A: 2609.08333 = Yao et al.; 2508.12629 = Dunn & Koes; B/C: 2603.27113v1, sha256 `03cc2f6c…`).
* **DEFAULT_POOL path constants** — A L114-120, B `DEFAULT_POOL`, C L46/L183, D L26-31: the tool's default INPUT,
  not a claim; a path is an identifier.
* **Date-stamped MEASURED docstrings outside the artifact** — A pre-L34-37/47-54/235/562-564/1318; C L50/L61
  (reworded to "the DEFAULT pool"); D L14-18 (FragFlex arXiv:2610.15284 C-1 rule quote, verified verbatim against
  the primary arXiv record).
* **D: epoch filter `'17'`** (pre-L799-800) — the cohort NAME is definitional; the count that followed it was the defect.
* **B: legacy key names `not_8_51_percent` / `not_25_05_percent`** — the values are computed; renaming them would
  break more surfaces.
* **C: code line references** (`yuelbond_decoder.py:799` … `:19250`) — source citations, not measurements.
* **D: residual stale tokens at L833, L916-917, L925** — repair-history docstrings/comments, never emitted
  (`wfa156_text_scan`: 1 file scanned, 0 skipped, 2 matches/pattern).
* **B: `molmetal/tests/test_wfa155_metal_coordination_sphere.py:351-352`** — not branch B's file; escalated to the owner.

---

## 4. Exact file footprint per branch (all untracked in git)

* **A** — modified `molmetal/scripts/wf_a155_posebusters_validity.py` (1646→2777 lines);
  created `molmetal/tests/test_wfa164_posebusters_narrative.py` (401 lines);
  created `molmetal/reports/wfa164_posebusters_narrative/{default_pool_diff.json, default_pool_bounded_repaired.json,
  default_pool_bounded_original.json, pre_repair_script_snapshot.py.txt}`.
* **B** — modified `molmetal/scripts/wf_a155_metal_coordination_sphere.py` (2570 lines);
  created `molmetal/tests/test_wfa164_coordination_sphere_narrative.py` (480);
  created `molmetal/reports/wfa164_coordination_sphere_narrative/{default_pool_artifact.json, default_pool_diff.json,
  pool_specific_literal_census.json, synthetic_pool_artifact.json, synthetic_pool/}`.
* **C** — modified `molmetal/scripts/wfa156_bond_threshold_frontier.py` (1944);
  created `molmetal/tests/test_wfa164_bond_threshold_narrative.py` (465);
  created `molmetal/reports/wfa164_bond_threshold_narrative/{report.json, report.md, per_row.json, per_row.csv,
  default_pool_diff.json, c7_pool_check/}`.
* **D** — modified `molmetal/scripts/wfa142_fragflex_cminus1_probe.py` (1598);
  created `molmetal/tests/test_wfa164_fragflex_cminus1_narrative.py` (283);
  created `molmetal/reports/wfa164_fragflex_cminus1_narrative/{compare_default_pool.py, default_pool_diff.json,
  default_pool_probe.json, default_pool_run.log, literal_census.json, real_pool_all_stratified_limit1.json,
  real_pool_stratified.log}`.
* **Refuters** (code untouched): `molmetal/reports/wfa164_branch_a_refutation/{refutation.json, refutation.md}`,
  `molmetal/reports/wfa164_coordination_sphere_refutation/{refutation.json, refutation.md, default_pool_recheck.json}`,
  `molmetal/reports/wfa164_c_refutation/{refutation.json, refutation.md}`,
  `molmetal/reports/wfa164_fragflex_cminus1_refutation/{refutation.json, refutation.md}`.

---

## 5. Exact republication work the main thread must do (paths only)

1. `metrics/by_metric/wfa155_posebusters_validity.json` — regenerate by a **FULL-pool** run of the repaired
   `molmetal/scripts/wf_a155_posebusters_validity.py` (the 20-row bounded artifact is not a publishable stand-in).
2. `metrics/by_metric/wfa155_metal_coordination_sphere.json` — regenerate by the repaired
   `molmetal/scripts/wf_a155_metal_coordination_sphere.py` (its `--out` default is this path).
3. `molmetal/reports/wfa156_bond_threshold_frontier/report.json` and `…/report.md` — refresh from
   `molmetal/reports/wfa164_bond_threshold_narrative/report.json` / `report.md` (or re-run the tool with `--out` there).
4. `molmetal/reports/wfa142_fragflex_cminus1/probe.json` — refresh from
   `molmetal/reports/wfa164_fragflex_cminus1_narrative/default_pool_probe.json` (and `probe_all_epochs.json` if the
   all-epochs cohort is republished).
5. `molmetal/tests/test_wfa155_posebusters_validity.py` — restore the `COMPARABILITY_BREAKS` import surface and the two
   key names at `:589-590`, or the suite stays RED at collection.
6. `molmetal/tests/test_wfa155_metal_coordination_sphere.py` `:351-352` — replace the two literal assertions with the
   computed fields (e.g. `population.single_fragment_rate_percent`).
7. `registry/metrics.yaml` — **no sha to re-point** (0 rows for wfa155 / wfa156 / wfa142). If these four are to be
   citable, ADD rows (today `registry.py check` = PASS, 579 current).
8. `docs/HONEST_NEGATIVE_CATALOG.md:20668` — re-verify the citation into `wfa156_bond_threshold_frontier/report.json`
   after republication.
9. `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:345` and `:358` — that matrix reads
   `metrics/by_metric/wfa155_metal_coordination_sphere.json -> /comparability_statement`, which changes on republication.

---

## 6. A reviewer should still check (no branch covered)

* **A**: the surviving "529/20 = 25.05 %" / "all 20 rows of the pool" string — fix the scope-vs-pool numerator/denominator
  and the prose wording **before** the full-pool republication, otherwise the published artifact repeats the defect.
* **A**: whether `COMPARABILITY_BREAKS` removal was intended (the tool still emits `n_independent_comparability_breaks`
  in the pre-repair snapshot; the repaired tool must expose an equivalent machine field).
* **All four**: the guard family checks *backing*, not *correctness* — consider an independent re-derivation of every
  rendered ratio (the pattern branch D used in `derive_cohort_detail`) before publishing.
* **B**: the `break_3` RAW/POST-PROCESSED wording vs arXiv:2603.27113v1 §4.1 + Table 1(b) (D3) and the inserted colon in
  the `arms` quote (D4).
* **C**: whether the 12,669-row c7 run (`wfa164_bond_threshold_narrative/c7_pool_check/`) should become the published
  artifact — the old "every row carries a metal" claim is FALSE there (12668/12669).
* **D**: replace the tautological `cohort_detail_recheck` with a check that can actually fail.
* **All four**: none of the four artifacts has a registry row → decide whether they are citable
  (`docs/METRIC_CITABILITY_RULE.md`) or should be registered.
* **All four**: the pre-repair denominators disagree across three parties (A 1646 lines + snapshot on disk; C 1453
  lines with no snapshot; B 954 vs 953) — pin one number per file before quoting a census.
* **Cross-repo**: the two distinct "8.51 %" values (0.0850694 vs 0.0850963) and the `wfa158` matrix that consumes the
  coordination-sphere artifact.
* **Hygiene**: whether `molmetal/reports/wfa164_posebusters_narrative/pre_repair_script_snapshot.py.txt` (an executable
  copy of a measurement tool) should stay in-repo, and whether the four untracked scripts should be committed so the
  repairs become diffable.
