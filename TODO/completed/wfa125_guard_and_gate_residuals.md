# Wf-A125 — Guard & Gate Residuals — Close-Out (2026-10-31)

**Wave**: Wf-A125 (CPU-only, **0 GPU**)
**Close-out date**: 2026-10-31
**Status**: ✅ **SHIPPED in the working tree** — **11** gaps addressed (G1..G11): **7** REPAIRED, **3** PARTIAL, **1** NOT-FIXED. **The wave's pre-flight is NOT green** (see §4) and it left **2 RED tests** of its own behind (see §5).
**Catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` **§6.224 .. §6.234** (cumulative counter **278 → 289**, ACTIVE **234 → 245**)
**Navigation**: `TODO/INDEX.md` **§22.167**
**Filesystem evidence**: `molmetal/reports/wfa125_closeout.md`

> **GPU statement.** **0 GPU experiments run; every python/pytest invocation prefixed with `CUDA_VISIBLE_DEVICES=""`.**
> MEASURED 2026-10-31 — `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i mol-metal` prints **nothing** (no mol-metal compute process at all). No training, sampling, or sweep was launched. No watchdog was started.

---

## 1. What was broken

Wf-A125 was assigned **eleven** gaps in the repo's guard / gate layer. All eleven are catalogued individually (catalog **§6.224..§6.234**); the table below is the index.

| # | gap | one-line defect | STATE |
|---|---|---|---|
| **G1** | root CPU lock | the `tests/` tree had **no** CPU lock — the only lock lives in `molmetal/tests/conftest.py`, which pytest does not load for `tests/` | **NOT-FIXED** (§6.224) |
| **G2** | guard coverage | the published-tree guard's roots were `metrics/by_metric`, `metrics/paper_tables`, `molmetal/metrics/by_metric` — **`registry/` was uncovered**, so a subprocess `registry generate` that re-stamped and restored the registry left no trace | **REPAIRED** (§6.225) |
| **G3** | guard diff kind | the guard compared **sha256 only**, so a write-then-restore that returned identical bytes was invisible | **REPAIRED** (§6.226) |
| **G4** | declaration semantics | all **3** entries in `PUBLISHED_WRITE_DECLARATIONS.json` are **script-keyed**, so any declaration authorised any module to write | **REPAIRED** (§6.227) |
| **G5** | ledger provenance gate | `run_ledger.uncommitted_provenance()` had **no production caller** — `pre_flight_check.py` imported only `check_ledger`, so a ledger row naming a never-committed file version could not block a launch | **REPAIRED** (§6.228) |
| **G6** | metric sha re-point | `registry.py` had **no** `repoint` verb; re-pointing one legitimately-regenerated metric meant either a 16k-line registry rewrite or a hand-edited YAML | **REPAIRED** (§6.229) |
| **G7** | PB-valid framing | `wf32_pb_valid.json` claimed **"5/5 pockets PASS"** against a **one-pocket** artifact, and its 100-cell sweep was **never run** | **REPAIRED**, not durable (§6.230) |
| **G8** | census SKIP basis | the census conflated "field never recorded" with "value recorded over an empty denominator" into **one** reason string | **REPAIRED** (§6.231) |
| **G9** | silent GPU skips | `grep -rn pytest_terminal_summary` → **0** hits; **9** GPU-required tests skipped and pytest still exited **0** with nothing recorded | **PARTIAL** (§6.232) |
| **G10** | guard regression coverage | the published-tree guard had **ZERO** regression coverage in the repo; its only proof was an **uncommitted** `/tmp` shell probe | **PARTIAL** (§6.233) |
| **G11** | WARN status | a `WARN` gate was counted in **none** of PASS / FAIL / SKIP, so `pass + fail + skip < total` and the summary was internally inconsistent; the run-ledger staleness gate **returned WARN instead of blocking** | **PARTIAL** (§6.234) |

---

## 2. What was repaired, and what proves it

### G2 / G3 / G4 — the published-tree guard (`molmetal/tests/conftest.py`)

| sub-gap | change | anchor |
|---|---|---|
| G2 | `PUBLISHED_FILE_PATHS` — **4** curated per-file paths with per-file `authorised_writers`: `registry/metrics.yaml`, `registry/datasets.yaml`, `registry/protocols.yaml`, `molmetal/reports/run_ledger.jsonl` | `molmetal/tests/conftest.py:338` |
| G3 | the snapshot now records the stat **quadruple** `(sha256, mtime_ns, size, ino)`; a new diff kind **`touched`** fires when the sha is identical but the quadruple moved | `molmetal/tests/conftest.py:453` |
| G4 | `is_test_module_key` (:493) / `split_declarations` (:522) / `load_capability_records` (:541) — a declaration authorises a write **only** if its key is in the active test-module set | `molmetal/tests/conftest.py:493-550` |

* **Proven by**: `molmetal/tests/test_wfa125_published_tree_guard.py` (NEW, 15657 B, sha256 `3e6ecb5de7aa48459fdcd04f9d2b6b3c70135f06829c332f82d4b56cee510c50`). MEASURED 2026-10-31 (CPU-only): that file + `molmetal/tests/test_published_tree_guard.py` + `molmetal/tests/test_registry_generate_import_path.py` → **55 passed, 0 failed**.
* `molmetal/tests/PUBLISHED_WRITE_DECLARATIONS.json` was **NOT** edited — byte-identical (sha256 `53ba26a6466128d741be1eb5609a4609b30876bedb6142b157904394690e0de1`, 1432 B, mtime Oct 31 05:23). MEASURED: its **3** script-keyed entries are now **0 authorisations / 3 capability records**.
* `molmetal/reports/` could NOT be added as a tree root: MEASURED — **486** files under `molmetal/scripts/` mention `molmetal/reports` or `"reports"`.

### G5 — the ledger-provenance launch gate

* NEW gate `p3_run_ledger_provenance` (`molmetal/scripts/pre_flight_check.py:8804-8850`), registered in **both** pipelines; it **FAILs** iff `OPEN_never_committed > 0`.
* `EXPECTED_GATE_COUNTS` moved `all` **126 → 127** and `launch` **128 → 129** (`molmetal/scripts/pre_flight_check.py:5034-5039`).
* **MEASURED 2026-10-31**: `p3_run_ledger_provenance` → **PASS**, evidence `ledger_rows=418 metrics=396 OPEN_never_committed=0 superseded_never_committed=9`; `molmetal/reports/run_ledger.jsonl` → **418** rows.
* **Proven by**: `molmetal/tests/test_wfa125_preflight_warn_contract.py` (NEW, 12095 B, sha256 `c0f27de499a5c6dc7a081cbfb29740fda61c1b5c6a0318dec6ef2977c7196a0c`).

### G6 — `registry.py repoint`

* NEW verb `registry.py repoint <metric_id> [--apply]` (`molmetal/scripts/registry.py:2297`) — a **surgical single-line** sha edit, no YAML round-trip, no re-stamping of other rows.
* **MEASURED 2026-10-31**: a sandbox `--dry-run` is **byte-identical**; `--apply` changes **2 lines in 1 of 578 rows**. Both re-pointed rows match the on-disk sha exactly — `registry/metrics.yaml:15811` → `d6eec79e…` (== `metrics/by_metric/wf32_pb_valid.json`) and `registry/metrics.yaml:3167` → `e85c529e…` (== `metrics/by_metric/cfm_run_census.json`).
* `registry.py check` → **REGISTRY: PASS** — **578** current / **0** stale / **0** missing; **35/35** datasets; **8/8** protocols.
* **Proven by**: `molmetal/tests/test_wfa125_registry_sha_refresh.py` (NEW, 14057 B, sha256 `8be2bc53696aa819a2036690c3bbb4bfa5068e6f78d3cb11fe94b6bdc1d7fa5b`).

### G7 — the PB-valid artifact

* `spec_target` added, **copied verbatim** from `molmetal/reports/wf32_m8_summary.json` (which carries `status: "PARTIAL"`). MEASURED: the two `spec_target` blocks compare **equal**; the string **"5/5 pockets PASS"** now occurs **0** times in the artifact.
* `honest_framing` rewritten to say the artifact measured **1 pocket x 1 candidate x 2 pose modes = 2 evaluations**, **both skipped**, PoseBusters validity **NOT MEASURED** (`pb_valid = null`, `n_checks = 0`, `pass_rate = 0.0`), and the **100-cell sweep was NEVER RUN**.
* **Proven by**: `molmetal/tests/test_wfa125_registry_sha_refresh.py`; the deferral is independently pinned by `molmetal/tests/test_wf32_phase3_pb_valid.py:183` (`test_wf32_full_sweep_is_recorded_as_not_run`).

### G8 — the census SKIP basis

* `molmetal/scripts/wf_cfm_run_census.py` now emits **two** bases — `field_absent` and `denominator_empty` — with distinct reason strings.
* **The split is 11 / 16, NOT 9 / 18.** MEASURED 2026-10-31 on the published artifact (independent recomputation over all **27** SKIP runs): **11 `field_absent` / 16 `denominator_empty`**, **0** mismatches; the artifact's own `changelog.wf_a125` records the same. The DESIGN plan's 9/18 came from reading `events.jsonl` directly; the census's own normalised per-epoch rows are the record of account.
* Full distribution MEASURED on the published artifact: `('SKIP','field_absent') = 11`, `('SKIP','denominator_empty') = 16`, `('FAIL','end_to_end') = 27`, `('PASS','end_to_end') = 5` (**59** runs).
* **Proven by**: `molmetal/tests/test_wfa125_registry_sha_refresh.py`.

### G9 — the GPU-skip announcement

* NEW repo-root `conftest.py` adds `pytest_collection_modifyitems` (`:125`, applying the `gpu` marker to `GPU_REQUIRED_MODULES` at `:94-97`) and `pytest_terminal_summary` (`:180`), printing **exactly one** line: `mol-metal GPU contract: N GPU-required test(s) did not run (…)`.
* **MEASURED 2026-10-31**: `pytest -q molmetal/tests/test_wf25_gpu_smoke.py molmetal/tests/test_rocm_egnn.py` → contract line **N = 9** and pytest's own `9 skipped`; `pytest -q tests/test_cpu_tensor_rejection.py` → `2 passed, 4 skipped`, contract line **N = 4**.
* **Proven by**: `molmetal/tests/test_wfa125_gpu_contract.py` (NEW, 8957 B, sha256 `9cf8bbab5a2a5128832ddbc263f9c983f17bee860c7f1dc3e8bd6c79a165caef`).

### G10 — the guard's regression coverage

* **Proven by**: `molmetal/tests/test_wfa125_published_tree_guard.py` (NEW, 15657 B, sha256 `3e6ecb5de7aa48459fdcd04f9d2b6b3c70135f06829c332f82d4b56cee510c50`).
* MEASURED: the wave's **5** new test files → **59 passed, 1 warning in 43.03 s** (CPU-only).

### G11 — WARN as a first-class status

* `n_warn` computed (`molmetal/scripts/pre_flight_check.py:15563`); the summary gains a `"warn"` key (`:15579`); `n_pass` / `n_fail` / `n_skip` at `:15554-15556`. The identity `pass + fail + skip + warn == total` now **holds**.
* `gate_run_ledger_staleness` (`:8698`) now returns **PASS/FAIL, never WARN** — it **blocks**.
* The evidence is bounded: MEASURED 2026-10-31 — the **HEAD** implementation, executed against the current ledger, produced **6405** characters; the working tree produces **673** (ending `… and 168 more (full list: run_ledger.py status)`).
* **MEASURED 2026-10-31** — the **HEAD** implementation of `gate_run_ledger_staleness`, extracted from git and executed against the current module's globals, returned **WARN**. `molmetal/tests/test_run_ledger.py:158` (`test_preflight_gate_enforced_by_default`) asserts `r.status in ("PASS","FAIL")`; that file is **unmodified by this wave** (`git diff molmetal/tests/test_run_ledger.py` is empty), so the assertion **could not have held at HEAD** and **does hold now**. Fixed honestly by the status change, not by relaxing the assertion.

### The wave's own new test files — MEASURED 2026-10-31, CPU-only

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  molmetal/tests/test_wfa125_root_conftest_gpu_lock.py \
  molmetal/tests/test_wfa125_gpu_contract.py \
  molmetal/tests/test_wfa125_published_tree_guard.py \
  molmetal/tests/test_wfa125_preflight_warn_contract.py \
  molmetal/tests/test_wfa125_registry_sha_refresh.py -q
→ 59 passed, 1 warning in 43.03s
```

---

## 3. What was NOT repaired

| gap | why not | what it would take |
|---|---|---|
| **G1** | the repair is on disk but **git-ignored**: `.gitignore:78` is `/conftest.py`. MEASURED: `git check-ignore -v conftest.py` → `.gitignore:78:/conftest.py	conftest.py`; `git ls-files` returns only `molmetal/conftest.py`, `molmetal/tests/conftest.py`, `molmetal/tests/conftest_wf_integration.py` | negate `.gitignore:78` + `git add conftest.py` (1 line, outside the wave's owned set) |
| **G9** | the mechanism lives in the **same git-ignored** root `conftest.py` | same one-line fix as G1 |
| **G10** | the coverage **exists in the working tree** but HEAD has **none** of it — all **5** new test files are untracked (`??`) and the wave does not commit | commit the wave's files |
| **G11** | the mechanism is repaired but the wave **introduced 2 RED tests** and left a **latent** contradictory contract (both files outside its owned set) | see §5 |

---

## 4. The pre-flight is NOT green — stated plainly

MEASURED 2026-10-31, full `--pipeline all` (`CUDA_VISIBLE_DEVICES=""`):

```
summary = {"pass": 113, "fail": 4, "skip": 10, "warn": 0, "total": 127,
           "total_wall_sec": 102.8784}
pre-flight exit code = 1
identity pass + fail + skip + warn == total  →  TRUE
```

The **4 FAILs**:

| gate | evidence | is it this wave's? |
|---|---|---|
| `p1_cfm_smoke_decoded` | HISTORICAL frozen Wf-188 artifact (cuda; 26,799/26,801 mols; decode_ratio=0.995821; UNTRACKED file; artifact's own `all_gates_passed=false`) | **no** — pre-existing |
| `p1_cfm_catherine_30metals` | `AnchorStaleError: anchor stale: age=2821958.1s > max_age_sec=2592000.0` | **no** — pre-existing |
| `p1_gpu_env_probe` | `torch.cuda.is_available == False` on every probed venv | **yes, by consequence** — the CPU-only session of this wave; it would PASS on a GPU host |
| `p3_wf385_scaffold_loss_wire_in_verified` | `loss did not decrease over 50-step smoke: loss[0]=1.4227431118488312, loss[-1]=1.4205920100212097` | **no** — pre-existing, unrelated to a gate this wave touched |

**This wave therefore does NOT have a green pre-flight, and this close-out does not claim one.** A green pre-flight and a CPU-only wave are not simultaneously observable: `p1_gpu_env_probe` FAILs here for that reason alone.

---

## 5. What remains OPEN

### 5.1 Two RED tests this wave introduced (both files outside the wave's owned set)

MEASURED 2026-10-31 (CPU-only):

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  molmetal/tests/test_wf415a3_preflight_fail_audit.py \
  molmetal/tests/test_wf327f11_gate_count_consistency.py \
  molmetal/tests/test_pre_flight_check.py -q
→ 2 failed, 7 passed, 1 warning
FAILED molmetal/tests/test_wf415a3_preflight_fail_audit.py::test_preflight_total_count
FAILED molmetal/tests/test_wf327f11_gate_count_consistency.py::TestAuditDocSync::test_audit_doc_table_count_matches_constant
```

1. `molmetal/tests/test_wf415a3_preflight_fail_audit.py:78` pins `preflight_summary["total"] == 117`; the code now has **127** → `assert 127 == 117`.
2. `molmetal/tests/test_wf327f11_gate_count_consistency.py::TestAuditDocSync::test_audit_doc_table_count_matches_constant` → `audit doc §2.1 'all' pipeline count drift: doc=126 vs code=127`. Cause: the wave bumped `EXPECTED_GATE_COUNTS` but not the SSOT doc the test pins — `TODO/completed/321d_preflight_gate_audit.md:47` ("126 gates"), `:53` and `:393` ("126 (`all`) … plus the `launch` pipeline at 128").

> The wave's own critic named (1). **The close-out found (2) independently** — it was in the wave's plan as a REQUIRED OUT-OF-SET edit that was never applied.

### 5.2 The three REQUIRED FIXES for §5.1

1. `TODO/completed/321d_preflight_gate_audit.md:47` → **127 gates**; `:53` / `:393` → **127** (`all`) / **129** (`launch`).
2. `molmetal/tests/test_wf415a3_preflight_fail_audit.py:78` → **117** becomes **127** (and the docstring at :7, :15, :66-80, :126-128).
3. `molmetal/tests/test_pre_flight_check.py:30` → add `"WARN"` to `ALLOWED_STATUS`; `:105-110` → include `s["warn"]` in the identity.

### 5.3 A LATENT (not-yet-red) contradictory contract

`molmetal/tests/test_pre_flight_check.py:30` `ALLOWED_STATUS = {"PASS", "FAIL", "SKIP"}` lacks `"WARN"`, and `:105-110` asserts `pass + fail + skip == len(gates)` — a **contradictory contract** with the wave's own new test. WARN **is** reachable from a REGISTERED gate: `gate_p3_protocol_alignment_yaml_target_in_baseline_range` returns WARN at `molmetal/scripts/pre_flight_check.py:10228` and is registered at `:13006`. MEASURED 2026-10-31 on this tree that gate currently returns **PASS** ("all 5 yamls have eval targets within baseline paper ranges", 76 chars), so the latency is not exercised — the fix is required for correctness, not optional.

### 5.4 The rest of the OPEN set

| # | OPEN item | anchor / measurement |
|---|---|---|
| 1 | the repo-root CPU lock is **git-ignored** (G1, G9) | `.gitignore:78`; `git check-ignore -v conftest.py` |
| 2 | the guard's coverage is **untracked** (G10) | all **5** new test files are `??` |
| 3 | `bench_pb_valid.py` regenerates `wf32_pb_valid.json` **without** `spec_target` → G7 is one run away from being wrong again | `molmetal/scripts/bench_pb_valid.py:874` (plain `write_text`, not `atomic_write_file` — §5.6) |
| 4 | the census module **docstring** still carries the REFUTED "9 of the 27" / "18 of the 27" split, and names a third basis `field_present` the artifact never emits | `molmetal/scripts/wf_cfm_run_census.py:423` and `:427` |
| 5 | the **trainer** defect behind `denominator_empty` (recording `0.0` where the truth is "undefined") is OUT OF SCOPE — the census can only report the degeneracy | — |
| 6 | `registry/narrative_claims.yaml` + `registry/file_provenance.json` are **unguarded**; the exclusion is now test-locked but unrecorded in prose | `molmetal/tests/test_wfa125_published_tree_guard.py:165, :168` |
| 7 | **ledger coverage gap**: **182** published metrics have **no ledger row**; the gate is GREEN while the wave's own two edited artifacts have **zero** provenance record | `p3_run_ledger_staleness` evidence `no_run=182` |
| 8 | `registry.py repoint` does **not** bump the file-level `generated_at`, which now predates two of its own rows | `registry/metrics.yaml:2` = `2026-10-30T18:17:10+00:00` |
| 9 | a PRE-EXISTING CPU-fallback bug exposed by the CPU lock: `torch.add(x, residual, alpha=alpha)` scales the **second** operand | `triton_kernels/fused_residual_add.py:286, :292, :294, :309` |
| 10 | **3** new ruff 0.6.9 `F401` in the wave's new files | `conftest.py:52` `typing.Dict`; `molmetal/tests/test_wfa125_preflight_warn_contract.py:36` `os`; `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py:36` `pytest` |
| 11 | the census artifact carries **no version field**, so G8's re-generation has no version bump | `metrics/by_metric/cfm_run_census.json` |
| 12 | the per-evaluation note **"posebusters not installed"** in `wf32_pb_valid.json` is **STALE**: MEASURED — `posebusters` **0.6.5** and `meeko` are both importable in `.venv` today. The operative blocker is the LICENSE-gated meeko/autobabel install the artifact's deferral block cites. The **100-cell sweep was NEVER RUN** | `metrics/by_metric/wf32_pb_valid.json` `per_evaluation[*].note` |
| 13 | PRE-EXISTING: `pytest --collect-only -q tests molmetal` exits **3** (INTERNALERROR) because `molmetal/scripts/wf229_smoke_test.py:46` calls `sys.exit(rc)` at import; **1766** tests collect before the abort. `pytest --collect-only -q tests` alone → **447** tests, rc 0 | `molmetal/scripts/wf229_smoke_test.py:46` |

> **SUPERSEDED 2026-10-31 (Wf-A127):** rows 1 (the git-ignored CPU lock), 3
> (the `spec_target` gap), 4 (the census docstring's refuted 9/18 split and the
> phantom `field_present` basis) and the two-RED-tests + latent-WARN-contract
> residue are **REPAIRED** — catalog §6.235, §6.239, §6.238, §6.236 and §6.237
> respectively.  The `registry.py repoint` / `generated_at` header gap is
> **REPAIRED** — catalog §6.240.  Row 13 (the `wf229_smoke_test.py` INTERNALERROR)
> is **STILL TRUE and PRE-EXISTING**: MEASURED 2026-10-31, a bare
> `pytest --collect-only -q` collects **1766** tests and then aborts with the same
> `sys.exit(rc)` at import; `pytest --collect-only -q tests` alone still exits 0
> with **447** tests.  Row 5 (the trainer defect behind `denominator_empty`) and
> row 2's *untracked* status remain **OPEN** until the tree is committed.  The
> table above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.

---

## 6. The five FINAL CHECKS — run 2026-10-31, quoted verbatim

### CHECK 1 — `grep -c "^## §6." docs/HONEST_NEGATIVE_CATALOG.md` and the highest `## §6.NNN`

```
grep -c "^## §6." docs/HONEST_NEGATIVE_CATALOG.md   ->  125
highest §6.NNN                                      ->  234
catalog's OWN counter (line 5)                      ->  COUNTER 278 → 289
```

**RESULT: the literal check CANNOT be satisfied, and the divergence is reported rather than papered over.**

The catalog defines its own count in **two** places, both read from the file (AGENTS.md §2: "the catalog's own cumulative counter"):

1. the **status banner on line 5** — `**COUNTER 278 → 289 (Wf-A125, 2026-10-31 — see §6.224..§6.234; …)**`, prepended **additively** with the Wf-A126 banner preserved verbatim beneath it (§5.5 rule #6);
2. a per-entry `**§6.NNN catalog-count math.** Net **+1**. Cumulative **X -> Y**. ACTIVE **A -> B**.` line.

MEASURED 2026-10-31 — the chain is internally consistent with **+1** per entry and terminates exactly at the banner: `277→278` (§6.223) → `278→279` (§6.224) → … → `288→289` (§6.234); ACTIVE `233→234` → … → `244→245`. **0** of the **40** parsable `catalog-count math` rows has a cumulative or ACTIVE delta other than **+1**.

The `grep -c "^## §6."` count is a **different quantity**: it counts only the entries written at `##` heading level (**125** h2 + **125** h3 + **37** h4 = **287** headings, **223** distinct `§6.NNN` ids). The catalog's counter is a **cumulative counter**, not a heading census — entries have been added at three heading levels and some ids are reused, so the two can never be equal. **The number written (289) is the catalog's own count**, and it is the number the three AGENTS.md sentences carry. The divergence is recorded here as an OPEN measurement, not silently reconciled.

### CHECK 2 — the three AGENTS.md count sentences

```
60:| "What's the honest-negative catalog?" | … (**289 entries** (245 active) — the catalog's own cumulative counter; latest = §6.234 Wf-A125 2026-10-31, the guard/gate residuals: 7 of the 11 gaps REPAIRED, 3 PARTIAL, 1 NOT-FIXED — the repo-root CPU lock is git-ignored, the published-tree guard's coverage is untracked, and the wave left 2 RED tests plus a latent WARN contract OPEN; counted 2026-10-31) |
169:See `docs/HONEST_NEGATIVE_CATALOG.md` (**289 entries** (245 active) — …; counted 2026-10-31).
367:- "What's the canonical honest-negative catalog?" → `docs/HONEST_NEGATIVE_CATALOG.md` (**289 entries** (245 active) — …; counted 2026-10-31). 6 sections: …
```

> **SUPERSEDED 2026-10-31 (Wf-A127):** the three sentences quoted above are the
> **HISTORICAL Wf-A125 snapshot** of `AGENTS.md`, PRESERVED VERBATIM per
> AGENTS.md §5.5 rule #6.  The catalog now stands at **296 entries**
> (**252 active**) — catalog §6.235..§6.241, cumulative **289 -> 296**.  The
> three sentences in `AGENTS.md` are still **byte-identical to each other**
> (they were re-verified at **296 / 252**); only their content moved on.
>
> **⚠️ COUNTER SUPERSEDED AGAIN (2026-10-31, Wf-A133).**  The sentence "The
> catalog now stands at **296 entries** (**252 active**)" above is the
> **HISTORICAL Wf-A127 snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5 rule
> #6.  The catalog now stands at **308 entries** (**263 active**) — catalog
> §6.246..§6.253, cumulative **300 -> 308** (the Wf-A129 + Wf-A128 wave took it
> to 300 / 255 first).  `AGENTS.md`'s three count lines are **byte-identical**
> to each other at **308 / 263**.
>
> **⚠️ COUNTER SUPERSEDED A THIRD TIME (2026-10-31, Wf-A141).**  The
> "308 entries (263 active)" line above is the **HISTORICAL Wf-A133
> snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.  The catalog
> now stands at **309 entries** (**264 active**) — catalog §6.254, cumulative
> **308 -> 309**.  §6.254 is the dated correction that the Wf-A140 F3
> adversarial re-check forced on §6.253.
>
> **⚠️ COUNTER SUPERSEDED A FOURTH TIME (2026-10-31, Wf-A141 guard-repair
> sub-wave).**  The "309 entries (264 active)" line above is the
> **HISTORICAL Wf-A141 snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5
> rule #6.  The catalog now stands at **310 entries** (**265 active**) —
> catalog §6.255, cumulative **309 -> 310**.  §6.255 records the episode in
> which the wf #131 path-reference guard caught **§6.254's own** new dangling
> citation.
>
> **⚠️ COUNTER SUPERSEDED A FIFTH TIME (2026-10-31, Wf-A141 guard-repair
> sub-wave).**  The "310 entries (265 active)" line above is the
> **HISTORICAL snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.
> The catalog now stands at **311 entries** (**266 active**) — catalog
> §6.256, cumulative **310 -> 311**.  §6.256 records the pre-flight's
> narrative-claims gate running the checker without `--strict`.
>
> **⚠️ COUNTER SUPERSEDED A SIXTH TIME (2026-10-31, Wf-A143
> citation-repair wave).**  The "311 entries (266 active)" line above
> is the **HISTORICAL snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5
> rule #6.  The catalog now stands at **312 entries** (**267 active**) —
> catalog §6.257, cumulative **311 -> 312**.  §6.257 records the YuelBond
> lit anchor printing a title that matches neither the preprint nor the
> published paper, the wrong author initials, a truncated DOI, the wrong
> RDKit entry point for a bond-order baseline, and a malformed arXiv id
> — repaired in 11 files, with the 3.5 Å-vs-3 Å cutoff deviation
> DISCLOSED but not changed, and the `paper/*` site left BLOCKED.
>
> **⚠️ COUNTER SUPERSEDED A SEVENTH TIME (2026-10-31, Wf-A145
> self-citation-repair wave).**  The "312 entries (267 active)" line
> above is the **HISTORICAL snapshot**, PRESERVED VERBATIM per
> AGENTS.md §5.5 rule #6.  The catalog now stands at **313 entries**
> (**268 active**) — catalog §6.258, cumulative **312 -> 313**.
> §6.258 records the catalog's own tmQM record printing a DOI that
> **404s from Crossref** (`10.1021/acs.jcim.0c00265`) against the
> record its own source module already prints correctly, and the
> YuelBond env test pinning a **59-gate-stale** hard-coded gate count
> (68 vs the registry's 127) — the test had been RED since the count
> first passed 68.  Both repaired CPU-only; the three remaining
> `cite_helper.py` records are independently VERIFIED against Crossref
> and left to the concurrent Wf-A140 wave's write scope.
>
> **⚠️ COUNTER SUPERSEDED AN EIGHTH TIME (2026-10-31, Wf-A146
> adversarial-review wave).**  The "313 entries (269 active)" line
> above is the **HISTORICAL snapshot**, PRESERVED VERBATIM per
> AGENTS.md §5.5 rule #6.  The catalog now stands at **314 entries**
> (**269 active**) — catalog §6.259, cumulative **313 -> 314**.  The
> adversarial review of the YuelBond citations against the paper's own
> extracted text found a bullet that quoted **"78.3 % failure rate"** —
> a percentage the paper never prints (it prints **783/1,000**), and a
> token that is the paper's **2D-reassignment F1** (Fig 4b / Table S3) —
> plus a baseline mis-attribution (the 217/783 split belongs to
> **RDKit**, not to `xyz2mol` directly), plus a **3.5 A-vs-3 A**
> candidate-edge-cutoff deviation that was declared at the DECODER
> definition sites but not at `DEFAULT_EDGE_CUTOFF` — the constant the
> YAML `edge_cutoff` key actually feeds.  All repaired CPU-only; the
> cutoff VALUE is deliberately unchanged (re-validation needs a GPU
> run this chain is forbidden to do).
>
> **⚠️ COUNTER SUPERSEDED A NINTH TIME (2026-10-31, Wf-A147
> 1-ulp-table wave).**  The "314 entries (270 active)" line above is
> the **HISTORICAL snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5
> rule #6.  The catalog now stands at **315 entries** (**270 active**)
> — catalog §6.260, cumulative **314 -> 315**.  §6.249's 1-ulp slip
> was closed in the pre-flight's COMMENT by Wf-A135 but was still live
> in **two TABLES** — catalog §6.249 row 2 and `TODO/INDEX.md` §22
> row 2 — which quoted `0.08508958875996528`, a value that appears
> NOWHERE on disk (the artifact carries `0.08508958875996527`, which
> is also the exact IEEE-754 double for `1078 / 12669`).  Both tables
> corrected; the deliberate one-ulp regression guard in
> `test_wfa133_reporting_matrix.py` left UNTOUCHED on purpose.
>
> **⚠️ COUNTER SUPERSEDED A TENTH TIME (2026-10-31, Wf-A151
> two-entry wave: Wf-A148 + Wf-A150).**  The "315 entries
> (270 active)" line above is the **HISTORICAL snapshot**, PRESERVED
> VERBATIM per AGENTS.md §5.5 rule #6.  The catalog now stands at
> **317 entries** (**272 active**) — catalog **§6.261** (Wf-A148,
> cumulative **315 -> 316**) and **§6.262** (Wf-A150, cumulative
> **316 -> 317**).  §6.261 records the Wf-A148 citation-and-code
> repair wave (11 files): the repo's FlowMol3 arXiv id pointed at a
> **compressible-fluid NPE paper** (2508.16916), the parameter count
> was **10× off** (66 M vs the paper's 6 M), FlowMol3's *validity*
> was mis-attributed to RDKit `DetermineConnectivity` when the paper
> defines it as *the percent of molecules that can be sanitized by
> rdkit*, a **bioRxiv DOI** was printed as an **arXiv id**, the
> YuelBond **year** was 2025 in two lines while two citation records
> in the same file already said **2026**, and `DetermineConnectivity`
> was described as a **sibling** of `DetermineBonds` when RDKit
> calls it **one call deep inside** it (DetermineBonds.cpp L543/L544).
> §6.262 records Wf-A150's `clean_only.best_epoch` **undeclared
> population** defect (the block computed over "CLEAN **AND** LAST-epoch
> rate" while its own `predicate` names "CLEAN **AND** BEST-epoch
> rate") and the **102-site** census re-quote that followed it across
> **5 documents** in **54 write operations**.  All repaired CPU-only;
> **0 GPU launches, 0 `paper/*` touches, 0 wet-lab**.
>
> **⚠️ COUNTER SUPERSEDED AN ELEVENTH TIME (2026-10-31, Wf-A151
> single-entry wave).**  The "317 entries (272 active)" line above is
> the **HISTORICAL snapshot**, PRESERVED VERBATIM per AGENTS.md §5.5
> rule #6.  The catalog now stands at **318 entries** (**273 active**) —
> catalog **§6.263** (Wf-A151, cumulative **317 -> 318**).  §6.263
> records the Wf-A151 bibliography re-verification: **6 of the 24 rows**
> in `molmetal/reports/wf_24paper_refs/final.md` carried a wrong cell —
> row 1's arXiv id pointed at a **compressible-fluid NPE paper**
> (2508.16916), row 20's at an **aerospace** paper (2303.16983), row 15's
> first author was **Zhang** where the list is **Jingyuan Zhou**, row 2's
> venue asserted a **JCIM paper that does not exist**, row 13 carried **no
> title** while denying its peer-reviewed **TMLR 2025** version, and the
> table's banner promised *"all arXiv-verified"*.  **Two** of the six
> proposed repairs were **REFUTED and NOT applied** (row 18's author cell
> was already correct; `2303.16982` is the **Uni-Mol+** paper by a
> different first author).  The same wave found that the Wf-A148
> "0 YuelBond sites in `paper/build_log.txt`" dismissal was itself
> **false** — the repo's PATH `grep` is **ugrep with `-I`**, which skips
> the repo's non-UTF-8 tracked text files; the true count is **68** over
> **256 716** lines, and the exposure surface is **189** tracked
> non-UTF-8 files of which only **3** are text.  Repaired in **4 files**,
> all CPU-only; **0 GPU launches, 0 `paper/*` touches, 0 wet-lab**.

**RESULT: PASS.** MEASURED — the three extracted sentences are **byte-identical** (`len(set(sents)) == 1` over `re.findall(r"\*\*289 entries\*\* \(245 active\).*?counted 2026-10-31", …)`), all read **289 entries (245 active)**, and **0** occurrences of the old `278 entries` string remain.

### CHECK 3 — `git diff --numstat docs/HONEST_NEGATIVE_CATALOG.md`

```
3806	38	docs/HONEST_NEGATIVE_CATALOG.md
```

**RESULT: the deletions column is **38**, NOT 0 — and those **38** deletions are PRE-EXISTING, not this wave's.**

MEASURED 2026-10-31 — the pre-wave snapshot (`/tmp/a125_catalog_before.md`, sha256 `0be5bce4c66eb44496624bda7947008848e0891144e70d54a0cd49b039c1178c`, 15713 lines) compared against HEAD gives:

```
git diff --no-index --numstat /tmp/head_catalog.md /tmp/a125_catalog_before.md
-> 2869	38	/tmp/{head_catalog.md => a125_catalog_before.md}
```

So HEAD → pre-wave = **2869 / 38**; HEAD → now = **3806 / 38**. **The deletions did not move: this wave deleted 0 lines.** The **38** deletions are Wf-A124 / Wf-A126's, inherited from the working tree at the moment this wave started.

The wave's own edit, measured against the pre-wave snapshot, is **additive**:

```
diff /tmp/a125_catalog_before.md docs/HONEST_NEGATIVE_CATALOG.md | grep -c "^< "   ->  1
```

The single removed line is **line 5**, the status banner — and the replacement **preserves the entire historical banner verbatim** and prepends the new clause (the file's own §5.5 rule #6 pattern, matching the Wf-A126 and Wf-A119B banners above it). No existing **entry** was edited or deleted; **§6.224..§6.234** are appended after §6.223.

### CHECK 4 — `git status --porcelain -- paper/`

```
0
```

**RESULT: PASS** — `paper/` is untouched (AGENTS.md §5.3 / TODO-84). `metrics/paper_tables/` was also not touched by this close-out.

### CHECK 5 — `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i mol-metal`

```
(no output; grep exit code 1 = nothing matched)
```

**RESULT: PASS** — no mol-metal compute process exists on the host at all. No watchdog was started.

### Summary

| check | result |
|---|---|
| 1 — catalog count vs the `^## §6.` grep | **PARTIAL** — the catalog's own counter (**278 → 289**) is correct and chain-consistent; the literal `grep -c` reading (**125**) is a different quantity (heading census) and **cannot** equal the cumulative counter. Reported, not reconciled. |
| 2 — three AGENTS.md count sentences | **PASS** — byte-identical, all **289 / 245** |
| 3 — catalog deletions column | **PARTIAL** — **38** deletions vs HEAD, all **pre-existing**; this wave's own delta vs its pre-wave snapshot is **1 replaced line (the banner, additively)** and **0** deleted lines |
| 4 — `paper/` clean | **PASS** |
| 5 — no mol-metal GPU process | **PASS** |

---

## 7. Standing directives re-confirmed

* **§5.1 / this wave's brief — GPU FORBIDDEN.** **0 GPU experiments run.** Every python / pytest invocation prefixed `CUDA_VISIBLE_DEVICES=""`. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i mol-metal` → **empty**. No watchdog started.
* **§5.2 — MEASURED vs PROJECTED.** Every number in this file traces to a path on disk or to a git object; PROJECTED values are labelled as such. No metric JSON was written by this close-out.
* **§5.3 / TODO-84 — no paper work.** `git status --porcelain -- paper/` → **empty**. `metrics/paper_tables/` was **not** touched by this close-out.
* **§5.4 — no wet-lab.** Zero wet-lab protocol content planned, executed, or written.
* **§5.5 — filesystem truth.** Every claim above is backed by an `ls -la` / `sha256sum` / git query actually run on 2026-10-31; see `molmetal/reports/wfa125_closeout.md`. Historical catalog text **PRESERVED VERBATIM** (rule #6): the only line of `docs/HONEST_NEGATIVE_CATALOG.md` this wave changed outside its own appended block is the **status banner**, which was extended **additively** (the Wf-A126 banner text is preserved verbatim after the new clause).
* **§5.6 — atomic writes.** This close-out file, `molmetal/reports/wfa125_closeout.md`, `TODO/INDEX.md`, `AGENTS.md`, and the catalog append were all written through `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (POSIX `os.replace`). No file was truncated in place.
* **§5.7 — 50-step CPU smoke.** **Not required by this wave**: it wire-in'd **no new trainer flag**. (The pre-flight's own `p3_wf385_scaffold_loss_wire_in_verified` gate ran its own 50-step CPU smoke and reported `loss did not decrease` — a pre-existing failure, recorded in §4.)
* **§5.9 — single venv.** Every invocation used `/home/hugo/codes/mol-metal/.venv/bin/python`; no bare `python`.
* **§5.11 — YAML-only launch.** Not applicable: this wave launched **no** training or inference.
