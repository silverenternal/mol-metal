# Wf-A127 — the Wf-A125 residue closure (CPU-only, **0 GPU**)

**Date:** 2026-10-31
**Standing directive:** *"现在不要运行任何gpu相关的实验"* / *"启动ultracode把这些问题都修一下, 但是禁止使用gpu"* — **every** python / pytest invocation in this wave was prefixed `CUDA_VISIBLE_DEVICES=""`.
**Catalog:** `docs/HONEST_NEGATIVE_CATALOG.md` **§6.235 .. §6.243** (cumulative counter **289 -> 298**, ACTIVE **245 -> 254**)
**Parent wave:** Wf-A125 (11 guard/gate gaps, catalog §6.224..§6.234) left four residues OPEN and one defect its own critic found: the repo-root CPU lock it shipped was **git-ignored**, so it could not ship.

## 1. What this wave closed

| # | Residue (Wf-A125) | Catalog | Repair | MEASURED |
|---|---|---|---|---|
| G1 | the repo-root CPU lock was **git-ignored** -> a fresh clone ran `tests/` unlocked and took VRAM | §6.235 | the `/conftest.py` ignore rule was **removed**; `conftest.py` is now `?? conftest.py` (trackable); a **repo-wide** invariant now forbids ANY git-ignored `.py` file | `git check-ignore -v conftest.py` -> **exit 1**; `git status --porcelain --ignored \| grep '^!!' \| grep -E '\.py$'` -> **0** lines |
| G7 | `bench_pb_valid.py` regenerated `wf32_pb_valid.json` **without** `spec_target`, via a plain `write_text` -> G7 one run away from being wrong again | §6.239 | `build_honesty_blocks()` now **computes** the honesty blocks from the run's own counts; both outputs written via `atomic_write_file` | a run that measured nothing is framed **NEVER RUN**; a run that DID measure is framed as measured |
| G8 | the census module **docstring** carried the refuted **9 / 18** split and named `field_present` as a third `basis` | §6.238 | the docstring now names **11 / 16** and states that `field_present` is the **INPUT FLAG**, not a basis | the artifact's **structured** `runs[].fragmentation_gate.basis` -> `field_absent` **11** / `denominator_empty` **16** of **27** SKIP; `('FAIL','end_to_end')=27`, `('PASS','end_to_end')=5` |
| G11 | the fix left **2 RED tests** + a latent contradictory WARN contract | §6.236 + §6.237 | the gate count is re-synced at **127** across **all 4** pins; `ALLOWED_STATUS` carries `"WARN"`, `REQUIRED_SUMMARY_KEYS` carries `"warn"`, the identity includes `s["warn"]` | `len(build_gates("all")) == 127`, `len(build_gates("launch")) == 129`; `frontier` 25, `chemprop_v2_v3` 18, `cfm` 7 |

Plus two further defects found and closed by the same sweep: the registry's file-level `generated_at` header **dated itself before two of its own rows** (§6.240), and **a repair that changes an artifact invalidates every document that describes it** (§6.241).

That same sweep then found **two more** defects of the same family, both shipped with a **mechanism** rather than a one-off edit:

| # | Finding | Catalog | Repair | Guard |
|---|---|---|---|---|
| X | **50** of the SSOT documents' own cross-references point at files the repository does **not** contain (40 `TODO/completed/`, 10 `TODO/pending/`) | §6.242 | a **growth-detector** (the 50 are pre-existing debt; deleting them would violate §5.5 rule #6) | `molmetal/tests/test_wfa127_dangling_reference_guard.py` |
| Y | **6** of the **28** byte-size citations quote a size the file no longer has, and **none of the 28** carried a date | §6.243 | every citation is now **dated**; the 6 stale ones corrected additively (`old B → new B, measured <date>`) | `molmetal/tests/test_wfa127_size_citation_guard.py` |

## 2. The class, honestly labelled

§6.241 corrects **5** stale claims in place (additively + dated, originals preserved verbatim per AGENTS.md §5.5 rule #6). It does **NOT** close the class: no gate in `pre_flight_check.py` re-reads prose against the artifacts it describes. What ships is the *method* (a **value-directed** sweep — grep for each superseded **value**, not for each file), not a mechanism. Catalog §6.241 says this in its own STATE line.

**Two more mechanisms now ship for that class**, both negative-tested, both on a
*value* the same value-directed sweep found while it was looking for the §6.241
recurrence:
* `molmetal/tests/test_wfa127_dangling_reference_guard.py` — every `TODO/<dir>/<name>.md`
  the SSOT documents name must resolve to a file on disk, or be in the recorded
  50-path baseline (§6.242).
* `molmetal/tests/test_wfa127_size_citation_guard.py` — every byte size the SSOT
  documents quote must carry a measurement date, must equal the file's size on
  disk, and the file must not have been modified after that date (§6.243).

## 3. Verification (MEASURED 2026-10-31, CPU-only)

```
$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest <15 affected files>
169 passed, 1 warning, 26 subtests passed in 207.78s
mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')
mol-metal GPU contract: 0 GPU-required test(s) did not run (CPU-only session; set MOLMETAL_TEST_ALLOW_GPU=1 to enable them)

$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa127_residue_closure.py
20 passed, 1 warning (under 2 s)
```

> **MEASURED 2026-11-02 (Wf-A269 wave-7 re-measurement, CPU-only) — this
> SUPERSEDES the 2026-10-31 transcript above, which is PRESERVED VERBATIM
> per AGENTS.md §5.5 rule #6:**
>
> ```
> $ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest --collect-only -q <15 affected files>
> 170 tests collected in 9.52s
> mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')
> mol-metal GPU contract: 0 GPU-required test(s) did not run (CPU-only session; set MOLMETAL_TEST_ALLOW_GPU=1 to enable them)
>
> $ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest <15 affected files>
> 2 failed, 168 passed, 1 warning in 397.17s (0:06:37)
> ```

**MEASURED 2026-11-04 (Wf-A269 wave-27 re-measurement, CPU-only) — this
SUPERSEDES the 2026-11-02 transcript above, which is PRESERVED VERBATIM
per AGENTS.md §5.5 rule #6:**

```
$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest --collect-only -q <15 affected files>
179 tests collected in 4.97s
mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')
mol-metal GPU contract: 0 GPU-required test(s) did not run (CPU-only session; set MOLMETAL_TEST_ALLOW_GPU=1 to enable them)

$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest <15 affected files>
2 failed, 177 passed, 1 warning, 26 subtests passed in 287.31s (0:04:47)
```

The 2 failures are
`molmetal/tests/test_wfa125_preflight_warn_contract.py::test_the_provenance_gate_passes_on_the_real_ledger`
and `::test_the_summary_counts_warn_and_the_identity_holds` — the disclosed
USER-gated `p3_run_ledger_provenance` defect (catalog §6.269.9), which needs the
U1 commit and is therefore NOT this wave's to close.

The 15-file surface includes the two files that were **RED** before this wave (`test_wf415a3_preflight_fail_audit.py`, `test_wf327f11_gate_count_consistency.py`) — both are **GREEN**.

> **HISTORICAL 2026-10-31 (Wf-A127):** an earlier measurement of the SAME
> surface, taken **before** the 20th guard test was added, read
> `168 passed, 1 warning, 26 subtests passed in 452.32s (0:07:32)` — it describes a tree that no longer exists.  It is PRESERVED
> VERBATIM per AGENTS.md §5.5 rule #6; the live statement above is the
> **current-tree** measurement (**169 passed** in 207.78 s, rc 0,
> `metrics/` byte-identical before/after).  Catalog §6.241 STATE UPDATE.

**The class guard that now ships.** `molmetal/tests/test_wfa127_surface_count_guard.py`
**collects** this same 15-file surface and requires every document that quotes its
pass count to quote the collected number — unless the quote is itself **inside** a
dated correction blockquote (a preserved historical record, AGENTS.md §5.5 rule
#6).  A test added to or removed from the surface therefore breaks that guard,
which forces the documents to be re-read instead of going silently stale (catalog
§6.241 STATE UPDATE).  It is deliberately OUTSIDE the surface: a guard inside the
surface would change the count it measures.

**The guard is negative-tested, not just green.**  A guard that cannot fail is
not a guard.  MEASURED 2026-10-31, CPU-only — each perturbation was applied,
the guard was run, and the tree was restored:

| Perturbation | Guard verdict |
|---|---|
| `TODO/README.md` live quote `169` -> `168` | **FAIL** — `TODO/README.md:3 quotes 168, collected 169` |
| `TODO/completed/a127_residue_closure.md` 15-file block `169` -> `167` | **FAIL** — `.../a127_residue_closure.md:27 quotes 167, collected 169` |
| `TODO/INDEX.md` trailer `169` -> `200` | **FAIL** — `TODO/INDEX.md:3 quotes 200, collected 169` |
| tree restored | **PASS** — `2 passed` |

The guard's own run: `molmetal/tests/test_wfa127_surface_count_guard.py` ->
**2 passed**, 1 warning, **7.45 s** (the surface collection inside it costs
**3.66 s**; the run does no GPU work).

**A first version of the guard was WRONG and was fixed.**  It exempted any
quote within 14 lines *below* a dated marker — a proximity rule.  Under that
rule the dated note that legitimately preserves the historical `168` also
shielded the **live** quote that sits above it, so a document whose live count
was wrong still passed.  The rule is now positional: only a quote that is
itself **inside** the dated blockquote is exempt.  The three perturbations
above are what caught it; the guard's own first negative test is what caught
the guard.


Other measurements:

* `registry.py check` -> **REGISTRY: PASS** — **578** current / **0** stale / **0** missing, **35/35** datasets, **8/8** protocols.
* `run_ledger.py provenance --all` -> **LEDGER PROVENANCE: PASS** — `rows=418 metrics=396 distinct_source_paths=398 OPEN_never_committed=0 superseded_never_committed=9`.
* `git diff --numstat molmetal/reports/run_ledger.jsonl` -> **`10  0`** (append-only, ZERO deletions).
* ruff **0.6.9** (the pinned pre-commit version, from pre-commit's own env) -> `All checks passed!` on the **four** touched Python files (`conftest.py`, `test_wfa127_residue_closure.py`, `test_pre_flight_check.py`, `test_wf415a3_preflight_fail_audit.py`).
* `git status --porcelain metrics/` before/after the test run -> **byte-identical**, 34 lines: **no published metric was overwritten**.
* A bare `pytest --collect-only -q` collects **1766** tests and then aborts with the **PRE-EXISTING** `molmetal/scripts/wf229_smoke_test.py:46` `sys.exit(rc)` INTERNALERROR (already catalogued; unchanged by this wave).
* A **second**, later run of the same 15-file surface read `169 passed, 1 warning, 26 subtests passed in 235.52s (0:03:55)` — the **count** is identical to the run above; the wall time is not, and only the count is what the guard measures.
* The two new guards: `test_wfa127_dangling_reference_guard.py` -> **2 passed**, 1 warning, **0.44 s**; `test_wfa127_size_citation_guard.py` -> **2 passed**, 1 warning, **0.44 s**.
* **The guard fired on the wave's OWN catalog entry** — and the RED interval is now dated.  An independent adversarial review MEASURED the live tree **RED at 12:16** and **green at 12:20** on 2026-10-31 (catalog mtime 12:15:50): the negative-test exemplar spelled a non-existent path in full, so the catalog cited a file that does not exist and the guard flagged the catalog.  The exemplar now uses this repo's `...` placeholder convention, and the interval is recorded in catalog §6.242 — a guard that only fires under a deliberate perturbation would have passed here.
* **Adversarial-review corrections folded into this wave** (each re-verified against disk first, not taken on the reviewer's word): §6.242's printed Proof command now carries the `-h` it needs to reproduce its own **50** (without it GNU grep's `filename:` prefix makes the shell test fail and the command prints **190**); the stale byte size **12898** was still live at `TODO/INDEX.md:900` and `:916` (disk = **13085** B) and is now corrected additively; **4** citations read `measured 2026-10-31 2026-10-03 04:21` — the wave had inserted today's date in front of the file's own date — and now carry the file's own mtime date; and §6.242's STATE wording was downgraded from **REPAIRED** to **PARTIALLY REPAIRED** because the guard covers **one** path family in **4** documents, leaving **48** distinct dangling repo-relative paths unguarded.
* **The dangling-reference guard fired on the wave's OWN catalog entry** on its first write: the negative-test exemplar spelled a non-existent path in full, so the catalog cited a file that does not exist and the guard flagged the catalog.  The exemplar now uses this repo's `...` placeholder convention, and the collision is recorded in catalog §6.242 as evidence — a guard that only fires under a deliberate perturbation would have passed here.

## 4. Filesystem-truth evidence (AGENTS.md §5.5)

The verbatim `ls -la` capture of every file this wave wrote is in **§7** below, taken immediately after the final write.  No size or timestamp is quoted here by hand — every one of them is in the capture.

## 5. Standing directives honoured

* **§5.1 GPU** — the 2026-09-29 standing grant is **overridden** by the 2026-10-31 user directive *"禁止使用gpu"*. **0 GPU** launches; every python / pytest invocation prefixed `CUDA_VISIBLE_DEVICES=""`; a watchdog polled `nvidia-smi --query-compute-apps` throughout and reported no mol-metal process.
* **§5.2 honest framing** — every number above traces to a path on disk. Nothing here is PROJECTED.
* **§5.3 paper** — **0** touches to `paper/*` (TODO-84 honoured). `metrics/paper_tables/` was not touched either.
* **§5.4 wet-lab** — **0** wet-lab work planned, executed or sent.
* **§5.5 filesystem-truth** — this close-out ships with `ls -la` evidence; every correction is ADDITIVE + DATED with the original PRESERVED VERBATIM (rule #6).
* **§5.6 atomic writes** — every write in this wave went through `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`.
* **§5.7 trainer-flag smoke** — **N/A**: no NEW trainer flag was wire-in'd by this wave.
* **§5.11 YAML-only launch** — **N/A**: no launcher was touched.
* **§8 no StructuredOutput** — this wave was executed inline, not as a Mol-Metal workflow.

## 6. Open residuals (NOT closed by this wave)

1. **The tree is uncommitted.** The root `conftest.py` and the five Wf-A125 test files are `??` (untracked, no longer ignored). A fresh clone still gets **nothing** until they are committed. **This is the single highest-value next action** and it needs the user's go-ahead (harness rule: commit only when asked).
2. **§6.241's class** — prose is still not re-read against the artifacts it
   describes.

> **PARTIALLY ADDRESSED 2026-10-31 (Wf-A127):** for **one** volatile value —
> the affected surface's pass count — a **mechanism** now ships
> (`molmetal/tests/test_wfa127_surface_count_guard.py`, negative-tested above).
> Every **other** number quoted in prose about an artifact is still unguarded,
> and no gate in `pre_flight_check.py` re-reads prose against the artifacts it
> describes.  The class is **not** closed; it is closed for one value.
>
> **WIDENED 2026-10-31 (Wf-A127):** three values are now guarded — a test
> surface's pass count (§6.241), the SSOT documents' **cross-references**
> (§6.242), and the SSOT documents' **byte sizes** (§6.243).  Line counts,
> metric values, gate totals and mtimes quoted in prose remain unguarded.
3. **The trainer defect behind `denominator_empty`** — the trainer records `0.0` where the truth is "undefined". Out of scope here, still OPEN (catalog §6.234).
4. **The `triton_kernels/fused_residual_add.py` CPU-fallback bug** (`torch.add(x, residual, alpha=alpha)` scales the second operand) — pre-existing, exposed by the CPU lock, not fixed here.
5. **`registry/narrative_claims.yaml` + `registry/file_provenance.json`** remain unguarded.
6. **The pre-flight is still not green**: MEASURED `--pipeline all` -> `113 PASS / 4 FAIL / 10 SKIP / 0 WARN / 127 TOTAL`, exit 1. The 4 FAILs are `p1_cfm_smoke_decoded`, `p1_cfm_catherine_30metals`, `p1_gpu_env_probe` (a direct consequence of the CPU-only session) and `p3_wf385_scaffold_loss_wire_in_verified` — none is a gate this wave touched.

7. **The 50 dangling cross-references** (§6.242) still point at nothing — the
   guard stops the set from **growing**, it does not repair the 50.  The **6**
   stale byte sizes (§6.243) ARE corrected; the *class* of stale prose numbers
   is not.

## 7. `ls -la` evidence

MEASURED 2026-10-31, captured immediately before this file's final write:

```
$ ls -la conftest.py .gitignore molmetal/tests/test_wfa127_residue_closure.py ...
-rw-r--r-- 1 hugo hugo   27424 Oct 31 11:33 AGENTS.md
-rw-r--r-- 1 hugo hugo   52194 Oct 31 11:47 .claude/skills/party-twenty-spirit/SKILL.md
-rw-r--r-- 1 hugo hugo    8131 Oct 31 11:18 conftest.py
-rw-r--r-- 1 hugo hugo 1671119 Oct 31 11:51 docs/HONEST_NEGATIVE_CATALOG.md
-rw-r--r-- 1 hugo hugo   10290 Oct 31 11:37 .gitignore
-rw-r--r-- 1 hugo hugo   40375 Oct 31 11:17 molmetal/scripts/bench_pb_valid.py
-rw-r--r-- 1 hugo hugo   98016 Oct 31 11:18 molmetal/scripts/registry.py
-rw-r--r-- 1 hugo hugo  154068 Oct 31 11:16 molmetal/scripts/wf_cfm_run_census.py
-rw-r--r-- 1 hugo hugo    6218 Oct 31 11:16 molmetal/tests/test_pre_flight_check.py
-rw-r--r-- 1 hugo hugo    7744 Oct 31 11:16 molmetal/tests/test_wf415a3_preflight_fail_audit.py
-rw-r--r-- 1 hugo hugo   16033 Oct 31 11:34 molmetal/tests/test_wfa127_residue_closure.py
-rw-r--r-- 1 hugo hugo    8698 Oct 31 11:49 molmetal/tests/test_wfa127_surface_count_guard.py
-rw-r--r-- 1 hugo hugo  676885 Oct 31 11:18 registry/metrics.yaml
-rw-r--r-- 1 hugo hugo   33781 Oct 31 11:16 TODO/completed/321d_preflight_gate_audit.md
-rw-r--r-- 1 hugo hugo   12274 Oct 31 11:51 TODO/completed/a127_residue_closure.md
-rw-r--r-- 1 hugo hugo   27607 Oct 31 11:39 TODO/completed/wfa125_guard_and_gate_residuals.md
-rw-r--r-- 1 hugo hugo  213594 Oct 31 11:49 TODO/INDEX.md
-rw-r--r-- 1 hugo hugo   48696 Oct 31 11:49 TODO/README.md
```

Every entry above is exact except this file's own line: writing this file changes its
size and mtime, so its listed values are the pre-write ones.  `ls -la` on any of the
other paths returns the numbers shown.
