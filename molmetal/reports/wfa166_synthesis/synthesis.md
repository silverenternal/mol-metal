# Wf-A166 — Synthesis brief (run-ledger provenance)

MEASURED 2026-11-01 on the live tree, 0 GPU, single venv
(`/home/hugo/codes/mol-metal/.venv/bin/python`), read-only apart from this file.
Every number below traces to a path on disk; the re-derivation commands are in §8.

Ledger under audit: `/home/hugo/codes/mol-metal/molmetal/reports/run_ledger.jsonl`
— sha256 `de2272dd125fe184bc52ac17e020346d9351cf5a15e7ae7f2fa87167c82d0809`,
mtime 2026-10-31 09:54:13, `git diff --numstat` = `10 0`. **UNTOUCHED by both branches.**

---

## 1. Column contract — one line per column (12 columns, all present in 418/418 rows)

`HIST` = historical snapshot (frozen at write time) · `LIVE` = pointer into the current
tree (resolved at read time) · `UNDEF` = undefined anywhere in the repo.
Writer = `molmetal/scripts/run_ledger.py`.

| # | column | kind | producer (file:line) | validated by (file:line) |
|---|---|---|---|---|
| 1 | `metric_id` | LIVE join key | caller's metric id; seed path takes the `registry/metrics.yaml` keys — `run_ledger.py:214`, `:627` | `run_ledger.py:305-326` (`staleness()` joins ledger→registry on this key) — the only join key between the two |
| 2 | `value` | HIST the number | the recorded number; seed path copies the registry row verbatim — `run_ledger.py:215`, `:642` | `run_ledger.py:322` (`_same_value`) — decides the "reformatted" vs "stale" split; the only column carrying the number |
| 3 | `source_json` | LIVE path | repo-relative path or bare stem; seed path copies the registry row's `source_json` — `run_ledger.py:216`, `:630` | `run_ledger.py:107-118` (`_resolve_source_json`) + `:517-528` (`uncommitted_provenance` resolves it and compares shas) |
| 4 | `source_json_sha256` | HIST content hash | `_sha256_file(src_path)` — `run_ledger.py:217`, `:199-200`; seed path passes the **registry's** sha through verbatim `:644` | `run_ledger.py:320` (vs registry row) + `:523-528` (vs disk AND vs git history) — **the only content-level pointer, and the only column with a validator** |
| 5 | `git_sha` | HIST = `git rev-parse HEAD` **AT THE MOMENT THE LEDGER ROW WAS APPENDED** | `_git_sha()` — `run_ledger.py:89-104`; assigned at `:201-202`; emitted `:218` | **NOTHING.** Echoed only at `run_ledger.py:547`, `:874`, `molmetal/scripts/wfa156_ledger_provenance_guard.py:576`. `registry.py` has **0** `run_ledger` hits. Not the commit that produced the value; not the commit that touched the metric file. |
| 6 | `config_sha` | HIST file hash, **PATH NOT STORED** | `sha256(config_path)` — `run_ledger.py:220`, `:207-212`; seed path takes `trainer_config` from `registry/protocols.yaml` `:639`, `:646` | **NOTHING** — with the path absent the hash is not self-describing. Measured: 6/418 rows carry one; **0/6** still match the file the path resolves to now |
| 7 | `data_sha` | HIST file hash, **PATH NOT STORED** | `sha256` of the file resolved from the metric's `data_source` — `run_ledger.py:219`, `:203-206`, `:121-143` | **NOTHING** (path echoed only into `extra.data_source`, `:651`). Measured: 30/418 carry one; 30/30 still match |
| 8 | `timestamp` | HIST | `_now_utc()` — `run_ledger.py:223`, `:75-76`; seed path passes the **source JSON's own** timestamp, else the registry's — `:633-638` | **NOTHING**; echoed at `:317`. For seed rows it dates the **VALUE**, not the row |
| 9 | `protocol_id` | **UNDEFINED** | the `protocol_id` kwarg — `run_ledger.py:221`, `:176` — **no caller ever passes it** | **NOTHING**: no producer, no consumer, no validator, no doc. `null` in **418/418** rows; `registry/protocols.yaml` has its own protocol rows this column never joins to |
| 10 | `host` | HIST | `'ZhaoLab'` — 418/418 rows | nothing (unvalidated) |
| 11 | `seed` | HIST "baseline re-point, not a run" | `True` in **408/418** rows | nothing (unvalidated) |
| 12 | `extra` | HIST free-form | echoed fields incl. `data_source` — `:651` | nothing (unvalidated) |

---

## 2. Census counts per axis (denominator stated on every line)

Denominators: **418 ledger rows · 396 distinct metrics · 396 latest rows · 22 superseded rows ·
398 distinct `source_json` paths · 416 distinct `source_json_sha256` · 6 distinct `git_sha` values.**

| axis | count (of 418 rows) | count (of 396 latest rows) |
|---|---|---|
| A `source_json_sha256` == disk now | 387 match / **31 differ** | 385 match / 11 differ |
| B sha == a version this repo has committed | 409 match / **9 never-committed** | 396 match / 0 differ |
| C the recorded `git_sha` exists as a commit | **418 / 418** | 396 / 396 |
| D the recorded `git_sha` **touches its own `source_json`** | **1** (row 23, `l5_hilbert_r2`, superseded) | **0 / 396** |
| E the recorded `git_sha` is reachable from a branch/tag | 417 / **1 unreachable** (row 408, `975414bc6fb8ef20ca53429f287a4d3297184cad`) | 395 / 1 |
| F bulk-constant `git_sha` | **384 / 418 = `d86c879a` (91.9 %)** — all appended in one sitting, `f55f063e` (ledger 22→408) | — |
| G `data_sha` | 30 present / 30 match | 25 match / 371 n/a |
| G `config_sha` | 6 present / **0 match** | 1 unresolved / 395 n/a |
| source files missing on disk | **0** | 0 |
| content at the recorded commit (`commit_content_matches`) | 398 match / 18 differ / 2 absent | 388 match / 7 differ / 1 absent |
| candidate agreement | **A==B 0/54 · A==C 0/418 · B==C 0** | — |
| resolver cross-check | two independent git resolvers AGREE (398 paths, 754 blobs) | — |

Candidate disagreement (`census.json:git_sha_candidates`): A = "HEAD when the ledger ROW was
appended", B = the metric JSON's own `provenance.git_commit` (only **54/418** metric JSONs carry
it), C = "the commit whose tree held this content". The three answer three different questions and
**never agree**.

---

## 3. THE answer: would the proposed pre-flight gate BLOCK LAUNCHES on the current tree?

**YES — and it is worse than "proposed":**

1. **The gate is NOT installed.** `grep -c "WF_RUN_LEDGER_GIT_PROVENANCE\|wfa166_run_ledger_provenance"`
   on `molmetal/scripts/pre_flight_check.py` → **0**. Live tree: `len(build_gates("all")) == 129`,
   `len(build_gates("launch")) == 131`, `EXPECTED_GATE_COUNTS` 129 / 131.
2. **The tree's pre-flight is ALREADY RED — for an unrelated reason.** A live
   `--pipeline all` run on this tree measures **117 PASS / 1 FAIL / 11 SKIP = 129 gates**
   (`/tmp/wfa166_baseline_all.txt`, EXIT=1). The single FAIL is
   `p3_ssot_path_reference_check`: the Wf-A169 catalog entry §6.266 cites
   `molmetal/scripts/auto_emit_section_6_minor.py`, **which does not exist on disk** (5 failing
   assertions, reproduced by `molmetal/scripts/wf_a131_path_reference_guard_runner.py`).
   **This is Wf-A169's defect, not Wf-A166's** — but it means "launches are already blocked"
   is true today for a different reason.
3. **With the RECOMMENDED patch applied (verified live on a scratch copy, `/tmp/wfa166_synth_verify/`):**
   `build_gates("all") == 130`, `build_gates("launch") == 132`, and the new gate returns
   **FAIL** with `ledger_rows=418 metrics=396 latest_rows=396 OPEN_metrics=8` —
   `cfm_bond_accuracy, chembl29_metal_scan, m14_decode_ratio, m19_lomo, m1_ru_auc,
   m3_vina_dock, m6_kappa, m7_diversity`. With `WF_RUN_LEDGER_GIT_PROVENANCE=0` → **SKIP**.
4. **With the STRICT patch applied (verified live, `/tmp/wfa166_synth_strict/`):**
   **FAIL**, `OPEN_metrics=396` — the gate fires on **every** metric (100 % false positives).

⇒ **Both candidate gates FLIP the pre-flight to FAIL and block every launch.** The strict rule
blocks on 396/396 metrics; the recommended rule blocks on 8/396 metrics (7 real content
mismatches + 1 unreachable commit).

---

## 4. Is the TRUE commit for a historical row recoverable from the repo?

**Two different questions; only one of them has an answer.**

**(a) The commit the MEASUREMENT RAN ON — NOT recoverable for ANY row (0/418).**
The ledger never recorded it. The metric JSON's own `provenance.git_commit` exists for only
**54/418** rows and **disagrees with the ledger's `git_sha` 54/54 (A==B 0/54)**; it is itself
only "HEAD at metric-write time". ⇒ The A162 "compute it, don't assert it" pattern **cannot**
be applied to this column: it computes a different quantity. Any patch that "recomputes"
`git_sha` as "the producing commit" would **fabricate** a provenance the repo never held
(AGENTS.md §5.2 zero-tolerance).

**(b) The commit whose tree HELD the metric file at the row's `source_json_sha256` — computable
for 409/418, with two caveats.**
- 9 rows have **no holder at all** (their content was never committed) ⇒ those rows are
  **unverifiable**, not repairable.
- The census's "exactly one holding commit (409/418)" is an **artifact of git history
  simplification**: `molmetal/scripts/wfa166_run_ledger_census.py:379` uses
  `git rev-list --all -- <path>`, which prunes the merge `8289a7f5`. Under `--full-history`
  **5/418 rows have TWO commits holding the IDENTICAL content**: ledger rows
  **54 `cfm_cross_dataset_transfer`, 57 `cfm_decode_real`, 68 `cfm_internal_diversity_intdiv1`,
  74 `cfm_loss_term_liveness`, 310 `platinai_4cell_auc`**. Verified live: for
  `cfm_cross_dataset_transfer.json`, commits `8289a7f5` and `24d32fa2` both hold blob
  sha256 `6240f0af3df6206113e2d2514aa687958aa913fbb06a210d059d292dd8b517b9` (= the row's
  `source_json_sha256`). The **blob-INTRODUCING** commit (`git log --all --diff-filter=A`) IS
  unique for all 5 (live-verified) ⇒ that is the method to pin if a column is added.

⇒ **The repair is "append a new column", never "rewrite `git_sha`".**
The column must be named for what it holds (`source_json_blob_commit` / "the commit that held
the metric file"), **not** "the commit that produced the run". `git_sha` stays as what it
always was: the repo HEAD when the row was appended. The 9 never-committed rows and the
1 unreachable-commit row (`chembl29_metal_scan`, `975414bc`) must be marked **UNVERIFIABLE**.
The ledger is append-only (AGENTS.md §5.5 rule #6), so the correction mechanism is the
ledger's own append-only **correction-row** pattern (precedent: the 9 Wf-A123 rows) — but
note those 9 rows all record `1b9470a0`, which touches **none** of their 9 `source_json`
paths (axis D = False 9/9), i.e. they **re-record the same anti-pattern**; a correction row
must carry the sha256 of the source file **at** the correcting commit, not HEAD.

---

## 5. Blast radius — every file:line that pins a gate count

Current counts: **129 (`all`) / 131 (`launch`)** → target **130 / 132** (verified on the
patched scratch copy).

### 5.1 Code (both patch specs already change these two)

| file:line | current text | required text |
|---|---|---|
| `molmetal/scripts/pre_flight_check.py:5803` | `    "all": 129,` | `    "all": 130,` |
| `molmetal/scripts/pre_flight_check.py:5876` | `    "launch": 131,` | `    "launch": 132,` |

### 5.2 Enforced test pins

| file:line | current text | required text | state today |
|---|---|---|---|
| `molmetal/tests/test_wf415a3_preflight_fail_audit.py:150` | `    assert preflight_summary["total"] == 129, (` | `== 130` | GREEN |
| `molmetal/tests/test_wfa159_timeout_verdict_census.py:430` | `    assert len(pre_flight.build_gates("all")) == 129` | `== 130` | GREEN |
| `molmetal/tests/test_wfa159_timeout_verdict_census.py:431` | `    assert len(pre_flight.build_gates("launch")) == 131` | `== 132` | GREEN |
| `molmetal/tests/test_wfa128_preflight_classification.py:482-485` | `== 127` / `== 129` | `130` / `132` | **ALREADY RED** |
| `molmetal/tests/test_wfa127_residue_closure.py:152-155` | `== 127` / `== 129` | `130` / `132` | **ALREADY RED** |

### 5.3 **MISSED by `blast_radius.json`** (found by refutation #2, re-verified RED by me)

| file:line | current text | required text | state today |
|---|---|---|---|
| `molmetal/tests/test_wf433a1_gate_v2_wire.py:108` | `    assert _pfc.EXPECTED_GATE_COUNTS["all"] == 127, (` | `== 130` | **ALREADY RED** |
| `molmetal/tests/test_wf433a1_gate_v2_wire.py:120` | `    assert len(set(names)) == 127, (` | `== 130` | **ALREADY RED** |
| `molmetal/tests/test_wf350c1_gpu_env_probe.py:334` | `    assert preflight.EXPECTED_GATE_COUNTS["all"] == 94, (` | `== 130` | **ALREADY RED** |

The branch's artifact lists **20** entries (`blast_radius.json` B1–B20) while its report prose
says "17-entry" — the artifact is the longer list, but it is still **incomplete by 3 pins**.

### 5.4 Documents (main thread owns)

| file:line | current text | required text | note |
|---|---|---|---|
| `TODO/completed/321d_preflight_gate_audit.md:47` | `### §2.1 \`all\` pipeline — 129 gates` | `— 130 gates` | **test-enforced** (`test_wf327f11_gate_count_consistency.py`) |
| `TODO/completed/321d_preflight_gate_audit.md:440` | `\| \`n_gates_total\` (per-pipeline COUNT) \| 129 (\`all\`) / 25 (\`frontier\`) / 18 (\`chemprop_v2_v3\`) / 8 (\`cfm\`) — plus the \`launch\` pipeline at 131 (`launch\` = …` | `130 (\`all\`) … at 132 (\`launch\`)` | **test-enforced** (same test, §6 table) |
| `TODO/completed/321d_preflight_gate_audit.md:50-53, :660` | live-count prose (`== 129`, `== 131`, "129 gates") | 130 / 132 | prose only, not test-enforced — stale if left |
| `AGENTS.md:36` | `… **Pre-flight v2 PASS gate — 129 gates on the` | `130 gates on the` | **main thread owns — DO NOT EDIT from a branch** |
| `AGENTS.md:37` | `` `all` pipeline (131 on `launch`)** `` | `` (132 on `launch`) `` | same |
| `docs/HONEST_NEGATIVE_CATALOG.md` | — | **APPEND** a dated Wf-A166 entry (append-only, §5.5 rule #6); latest today is §6.266 (Wf-A169) | its live-count quotes (16158/16172/16756/16924/18152/18193) are stale vs 129/131 — a separate honest-negative for the catalog owner |
| `molmetal/scripts/pre_flight_check.py:12358`, `molmetal/tests/test_wf415a3_preflight_fail_audit.py:36, :137, :247`, `molmetal/tests/test_wfa157_help_probe_concurrency.py:25` | "129 gates" prose | 130 | prose only |

`molmetal/reports/**` historical reports (e.g. `wfa162_c7_pool_ligand_connected_rate/report.md:345`
"129 gates ran with ZERO GPU allocation") are **append-only history** — do not edit.

---

## 6. Recommendation

**Recommended: Option R — ship `patch_preflight_gate_recommended.diff` (the provenance-TRUE rule
(i)+(ii)+(iii′)), but ship it DOWNGRADED** (`WF_RUN_LEDGER_GIT_PROVENANCE=0` → SKIP), and close the
8 open findings through the ledger's own append-only correction-row mechanism; remove the
downgrade when the 8 are closed.

Reason: the 8 are **real defects** worth a gate (7 files whose recorded sha ≠ the content at the
recorded commit; 1 commit that exists but is reachable from no branch or tag); 396 are a
**semantics error** (the rule "the recorded `git_sha` must have touched the file" is simply the
wrong question, because `git_sha` means "HEAD when the row was appended" — 0/396 latest rows
satisfy it). A gate that fires on 100 % of rows is a gate nobody reads; the repo's own
`p0_gate_inventory_audit` also counts opt-in gates as **zombies**, so a permanently-advisory
gate is wrong too. R gives a gate that is enforced *in principle*, blocks on a bounded 8, and
has a mechanical closure path.

**Rejected options and why**

| option | rejected because |
|---|---|
| **Strict rule (iii) as-is** | blocks on **396/396 = 100 %** of metrics. MEASURED: 0/396 latest rows have a `git_sha` that touched their `source_json`. It encodes a false premise about the column's meaning. (The patch is still shipped as a spec, with a docstring naming the 396/396 number and the warning that it FAILS today.) |
| **Permanently report-only / opt-in (the neighbouring `p3_run_ledger_staleness` pattern)** | the repo's own `p0_gate_inventory_audit` counts opt-in gates as zombies, and the sibling `p3_run_ledger_provenance` already BLOCKS by default on OPEN findings. A gate nobody can fail is not a gate. |
| **Rewrite `git_sha` in place** | violates the append-only ledger (AGENTS.md §5.5 rule #6) **and** asserts a quantity the repo never recorded (AGENTS.md §5.2 zero-tolerance on fabricated numbers). |
| **"Compute the producing commit" (the Wf-A162 pattern)** | computes a **different** quantity: no row's producing commit is recoverable (0/418), and the metric JSON's own `provenance.git_commit` disagrees 54/54. |
| **Re-run every metric to re-derive `git_sha`** | 396 metrics × GPU-weeks, and it does not recover the historical commit either — it only writes a NEW one, reproducing the same defect one generation later. |

---

## 7. Files created by each branch (and what the main thread must do next)

### Branch 1 — census (provenance census)
- `molmetal/scripts/wfa166_run_ledger_census.py`
- `molmetal/reports/wfa166_run_ledger_census/census.json`, `census.md`
- `molmetal/tests/test_wfa166_run_ledger_census.py` (5 passed, 0.66 s, CPU-only)

### Branch 2 — validator + gate spec
- `molmetal/scripts/wfa166_run_ledger_provenance.py` (`audit_ledger(...)`, self-contained, 9 axes/row)
- `molmetal/tests/test_wfa166_run_ledger_provenance.py` (11 passed, 4.65 s, `CUDA_VISIBLE_DEVICES=""`)
- `molmetal/reports/wfa166_run_ledger_provenance/patch_preflight_gate.diff` (strict, spec only)
- `molmetal/reports/wfa166_run_ledger_provenance/patch_preflight_gate_recommended.diff` (provenance-TRUE, spec only)
- `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md`, `blast_radius.json`, `audit_real_ledger.json`

### Refutation artifacts (both branches' reports say "NOT REFUTED")
- `molmetal/reports/wfa166_run_ledger_census_refutation/refutation.json`, `refutation.md`
- `molmetal/reports/wfa166_b2_refutation/refutation.json`, `refutation.md`

### NOT created / NOT touched
`pre_flight_check.py`, `run_ledger.jsonl`, `AGENTS.md`, `TODO/**`, `docs/**`, `registry/**`,
`metrics/**`, `configs/**`, `paper/**` — all read-only. Neither patch was applied to the working
tree (verified: the A166 gate has 0 grep hits in `pre_flight_check.py`).

### Main thread — next actions, in order

1. **Fix the pre-existing RED first** (`p3_ssot_path_reference_check` → missing
   `molmetal/scripts/auto_emit_section_6_minor.py`). That is Wf-A169's defect; do not let Wf-A166
   be blamed for it and do not stack a second FAIL on top of it.
2. **Add the 3 missed pins (§5.3) to `blast_radius.json`** and correct the "17-entry" prose to 20
   (then 23).
3. **Apply `patch_preflight_gate_recommended.diff`** to `molmetal/scripts/pre_flight_check.py`
   (it also carries the two constant bumps, 5803 / 5876), then apply the §5.2 + §5.3 pin updates
   in lockstep in the same edit.
4. **Ship it downgraded** — `WF_RUN_LEDGER_GIT_PROVENANCE=0` — until the 8 are closed; then remove
   the downgrade. (The patch already implements the downgrade switch.)
5. **Close the 8** by appending correction rows to the ledger (append-only). Two of them need
   non-correction action: `chembl29_metal_scan`'s `975414bc` is reachable from **no branch or
   tag** (it survives only in the reflog) — either make it reachable or mark the row
   **unverifiable**; and the 7 content-mismatch rows need the sha256 of the source file **at**
   the correcting commit, not HEAD (the 9 Wf-A123 correction rows all recorded `1b9470a0`, which
   touches none of their own source paths — the same anti-pattern).
6. **Decide the new-column question** (§4b): append `source_json_blob_commit` (method: the
   blob-**introducing** commit, `git log --all --diff-filter=A`, unique for the 5 ambiguous rows;
   **never** rewrite `git_sha`). This is a schema change to an append-only artifact and should
   be its own follow-on wave, not folded into the gate patch.
7. **Write the honest-negative entry** — APPEND a dated §6.267 Wf-A166 entry to
   `docs/HONEST_NEGATIVE_CATALOG.md` (append-only, §5.5 rule #6).
8. **Update `AGENTS.md:36-37`** to 130 / 132 — **only after** the gate is actually applied,
   otherwise it is a false claim. The branch correctly did not touch AGENTS.md.
9. **No user decision is required** for steps 1-5 and 7 (0 GPU, CPU-only, in-repo). Ask the user
   only if you want to close the 8 by **re-running** metrics (a GPU + wall-budget decision) rather
   than by appending correction rows.

---

## 8. Re-derivation commands used for this brief (0 GPU, single venv)

```
sha256sum molmetal/reports/run_ledger.jsonl
wc -l molmetal/reports/run_ledger.jsonl
python -c "import json,collections; rows=[json.loads(l) for l in open('molmetal/reports/run_ledger.jsonl')]; ..."
python -c "import importlib.util; ...; print(len(m.build_gates('all')), len(m.build_gates('launch')))"   # 129 / 131
python -c "... git cat-file --batch ... sha256(blob) vs row['source_json_sha256'] ..."                    # 8 open
git cat-file -t 975414bc6fb8ef20ca53429f287a4d3297184cad ; git rev-list --all | grep -c 975414bc          # commit / 0
git rev-list --all -- <path>  vs  git rev-list --all --full-history -- <path>                            # 1 vs 2
git log --all --diff-filter=A --format=%H -- <path>                                                     # unique introducer
git apply --check <patch> ; git apply <patch>  (scratch /tmp/wfa166_synth_verify, /tmp/wfa166_synth_strict)
CUDA_VISIBLE_DEVICES="" python molmetal/scripts/pre_flight_check.py --pipeline all                        # 117P/1F/11S
CUDA_VISIBLE_DEVICES="" python -m pytest molmetal/tests/test_wf433a1_gate_v2_wire.py ... -q                # 5 stale-pin tests RED
```
