# Wf-A171 — The GPU-lock guard blind spot (A167)

Date: 2026-11-01. CPU-only, no GPU was touched (`CUDA_VISIBLE_DEVICES=""` on every
invocation; the module imports no `torch` and never invokes `nvidia-smi`).

---

## 0. PREMISE CORRECTION — the defect is in the A167 guard, not A165

The branch brief says the archived-vs-live classification lives in the **A165** guard
(`molmetal/scripts/wfa165_gpu_lock_guard.py`).  It does not.

    $ grep -cn "archived\|ARCHIVE" molmetal/scripts/wfa165_gpu_lock_guard.py
    0

A165 has **no archive concept at all**.  Its only path prefixes are a *reporting*
breakdown: `SCOPE_PREFIXES = ("molmetal/", "scripts/", "tests/")` (line 1239).

The path-prefix archive heuristic lives in the **A167** guard
`molmetal/scripts/wfa167_command_prefix_guard.py`:

| line | code |
|---|---|
| 120-123 | `ARCHIVE_PREFIXES: Tuple[str, ...] = ("TODO/archive/", "results/")` |
| 727 | `archived = path.startswith(ARCHIVE_PREFIXES)` |
| 765 | `klass = "ARCHIVED" if archived else semantic` |
| 997-1002 | `"real_defeaters_live": sum(1 for s in real_defeaters if not s["archived"])` / `"real_defeaters_archived": ...` |

Therefore the staged patch targets **A167**, not A165.  Everything below is measured
against A167.

---

## 1. THE DEFECT — a path substring decides a security-relevant verdict

`results/` is hard-coded as an archive prefix, so **every** file under `results/` is
filed as `ARCHIVED` regardless of what the tracked text says about it.  The guard then
publishes `real_defeaters_live: 0` — a green light — for a live, tracked, executable
launcher that can switch the GPU back on.

The file that exposes it:

    $ git ls-files -s results/cfm_full_retrain/launch_loop.sh
    100755 5b5b487beb12aafbb09d692640e9aa31e3008505 0  results/cfm_full_retrain/launch_loop.sh
    $ git ls-files -s TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh
    100755 97e6d9293b662cf437f433bffff701418f5df669 0  TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh

Both are mode `100755` — **the executable bit cannot separate live from archived**
(that is why reachability + a *declaration* are needed, not a path shape).

Reachability of the results launcher (3 tracked files name it by full path):

    $ git ls-files -z | <python: read each tracked file, look for the needle>
    tracked_files_total = 9550
    referencing_files = 3
      docs/HONEST_NEGATIVE_CATALOG.md
      molmetal/reports/wf279_synthesis/followup_summary.md
      molmetal/reports/wf279d_launcher_integration/integration_report.md

**No tracked text declares `results/` archival.**  The only trees declared archival are
`TODO/04_history/`, `TODO/archive/`, `TODO/archive/wf437_simplify/`, `scripts/_retired/`.

### 1.1 The false green, MEASURED on a counterfactual tree

The working-tree copy of `results/cfm_full_retrain/launch_loop.sh` had **already been
repaired by the main thread** before this branch ran (HEAD still holds
`CUDA_VISIBLE_DEVICES=0 ...`).  To measure the defect I built a scratch copy of the repo
(`/tmp/a171_counter`) carrying the HEAD (pre-repair) content of that one file, and ran
the guard there.  Nothing else differs.

    $ CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa167_command_prefix_guard.py \
        --repo-root /tmp/a171_counter --json --quiet

| | UNPATCHED A167 | PATCHED A167 (staged patch applied to a scratch copy) |
|---|---|---|
| `real_defeaters` | 12 | 12 |
| `real_defeaters_live` | **0** | **1** |
| `real_defeaters_archived` | 12 | 11 |
| `by_class` | `{ALREADY_LOCK_PRESERVING: 45, ARCHIVED: 22, BENIGN: 142}` — **no `REAL_DEFEATER` key at all** | `{ALREADY_LOCK_PRESERVING: 39, ARCHIVED: 40, BENIGN: 129, REAL_DEFEATER: 1}` |
| the results site | `class=ARCHIVED`, `value_raw="0"` | `class=REAL_DEFEATER`, `value_raw="0"` |

The defect is not a cosmetic label: the unpatched guard has **no `REAL_DEFEATER` class in
its published histogram at all** on that tree, because the class was overwritten by
`ARCHIVED` (line 765).  A reader of the JSON sees only `ARCHIVED: 22` and concludes
"nothing to fix".

On the **current** live tree the unpatched guard reports
`real_defeaters=11, real_defeaters_live=0, real_defeaters_archived=11` over
`tracked_sh_files_total=113` — the 0 is *accidentally* right today only because the
results launcher was repaired by hand; the guard still cannot tell the two apart.

> **CORRECTION — 2026-11-04 (Wf-A269 wave-21).  The paragraph above is PRESERVED VERBATIM
> (AGENTS.md §5.5 rule #6); its "live = 0" was FALSE on the live tree for the interval
> 2026-11-01 → 2026-11-04.**
>
> The sentence says the 0 is "accidentally right today".  It was **accidentally right on
> 2026-11-01** and then **wrong**: **5 launchers** were moved back out of
> `TODO/archive/wf437_simplify/` into `scripts/`, so 10 of the 11 archived defeaters became
> **LIVE**.  MEASURED 2026-11-04 by a fresh `scan_repository()`: `real_defeaters` **13**,
> `real_defeaters_live` **10**, `real_defeaters_archived` **3**, `sites_total` **258** over
> `tracked_sh_files_total=113`.
>
> **⇒ THE "ACCIDENTALLY RIGHT" IS THE POINT, NOT AN ASIDE.**  A headline that is right *by
> accident* is right until the tree moves, and nothing in the repo would have said otherwise —
> `molmetal/tests/test_wfa167_command_prefix_guard.py` compares the pinned census against a fresh
> scan (it **was RED** for those 3 days) but **no runner executes it**: CI names 12 explicit test
> files and none of them is this one, `make test` is the only directory-level runner and **no CI
> job calls it**, and `pre_flight_check.py` names neither the guard nor its test.  The guard's
> own CLI exits **1** on this tree and is likewise called by nobody.
>
> **REPAIRED (2026-11-04).**  The 10 live sites were repaired with the guard's own
> `lock_preserving_form`; MEASURED after: `real_defeaters_live` **0**,
> `real_defeaters_archived` **3** (untouched), `sites_total` **268**; **mutation-proved both
> ways** (revert all 10 ⇒ the guard test FAILS; revert one ⇒ FAILS; restore ⇒ 19 passed).  The
> census was re-pinned and the 2026-11-01 rendering preserved byte-for-byte at
> `census_superseded_2026-11-01.json`.
>
> **STILL OPEN.**  The guard's `ARCHIVE_PREFIXES` path heuristic does not list
> **`scripts/_retired/`**, which the repo's own **CITED** archival policy does
> (`scripts/_yaml_only_gate.py:108` + `:131`).  NOT changed in this wave — changing a classifier
> to move a number is the anti-pattern, and the correct repair (a CITED exclusion table) needs its
> own census re-pin.

---

## 2. INDEPENDENT CENSUS (my own scanner, not the A167 one)

Search method: `git ls-files -z` (never `os.walk`), UTF-8-safe per-file decode with
explicit accounting for non-UTF-8 / too-large / missing files, shell region tagging
(code / dquote / squote / comment) so a mention is never an assignment.

    $ CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa171_gpu_lock_guard_path.py \
        --repo-root /home/hugo/codes/mol-metal --json --quiet

**Denominator** (`census.json` → `denominator`): `tracked_files_total=9550`,
`text_candidate_files=7643`, `files_scanned=7583`, `files_not_text_candidate=1907`,
`files_skipped_too_large=15`, `files_skipped_non_utf8=1`
(`molmetal/reports/wf_final_paper_build/build_log.txt`),
`files_missing_on_disk=44` (43 `OpenCL` device files + 1 `flow_matching` directory).

**Sites** (`counts`): `sites_total=221` over `files_with_defeaters=75`; by language
`shell=193, python=28`; by bucket
`DEFEATER=9, POTENTIAL_DEFEATER=5, LOCK_HONOURING=63, NEUTRAL_PRESET=11, NOT_AN_ASSIGNMENT=133`.

**Classification of the four categories asked for**:

| category | count | meaning |
|---|---|---|
| (a) forces a GPU on | 9 proven (+5 potential) | the value provably replaces the ambient mask |
| (b) preserves an empty value | 63 | `${VAR-…}` / bare self-reference / empty literal |
| (c) presets a device without overriding | 11 | `os.environ.get(CVD, …)` reads, non-device literals |
| (not an assignment) | 133 | mentions, comments, string literals, argv-position assignments |

**Liveness split**: `defeaters_total=14`, `defeaters_live=3`,
`defeaters_archived=11` (9 proven + 2 potential).
`self_check` — all four invariants true (`defeater_split_sums`,
`live_plus_archived_equals_total`, `no_path_prefix_archive_rule`,
`sites_equal_sum_of_buckets`).

**Cross-validation**: the 11 archived defeaters are *exactly* A167's 11 real defeaters
(same files, same lines) — the two independent scanners agree on the archived set.
A167 scans only `.sh` (113 files); my census also covers the 28 Python sites A167
never sees.

### 2.1 The 3 LIVE defeater sites (all POTENTIAL, all Python)

| site | value | verdict |
|---|---|---|
| `molmetal/molmetal_lam/sbdd_env/training_utils.py:136` | `DEFAULT_CUDA_VISIBLE_DEVICES` | **not** a defeater under the lock: guarded by `cur = os.environ.get(CVD)` / `if cur is not None: return cur.strip()` (lines 133-136) — an empty ambient lock returns early |
| `molmetal/molmetal_lam/sbdd_env/training_utils.py:217` | `chosen` | **not** a defeater under the lock: `env_value is not None` → `chosen = env_value.strip()` → `""` is written back verbatim |
| `molmetal/scripts/_common_train_args.py:585` | `str(cvd)` | **reachable defeat path**: fires only when `--cuda-visible-devices` is supplied, and 2 tracked configs supply it — `configs/platinai_4cell_desc_v1_rerun.yaml:60` and `configs/train_cisplatin_resistance_v2_v1.yaml:47` both set `cuda_visible_devices: "0"` |

**Live PROVEN defeaters on the current tree: 0.**

### 2.2 A fifth mechanism the brief did not name (and a correction to my own first census)

My first census reported `scripts/launch_wf308c_todo220_gpu.sh:125` as a live PROVEN
defeater (`CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}"`).  **That was my
error**, and A167 was right to call it BENIGN: the assignment sits in *argument*
position after `setsid nohup`, so POSIX passes it as an argv element and the
environment is untouched.  Measured with plain bash:

    $ CUDA_VISIBLE_DEVICES="" bash -c 'setsid nohup env | grep "^CUDA_VISIBLE_DEVICES="'
    CUDA_VISIBLE_DEVICES=                 # still EMPTY -> the lock survived
    $ CUDA_VISIBLE_DEVICES="" bash -c 'CUDA_VISIBLE_DEVICES=0 env | grep "^CUDA_VISIBLE_DEVICES="'
    CUDA_VISIBLE_DEVICES=0                # the command-word prefix DID override it

My scanner now models command-word position (`_in_command_word_position`), and the
site is reclassified `ARG_POSITION_ASSIGNMENT` → not an environment assignment.
**Stated limitation**: a wrapper that re-interprets argv assignments as environment
assignments (`sudo VAR=value cmd`) would be a false negative — no tracked shell script
uses `sudo` (`git ls-files '*.sh' | xargs grep -ln sudo` → empty).

---

## 3. THE NEW CLASSIFICATION (evidence, not a path substring)

`molmetal/scripts/wfa171_gpu_lock_guard_path.py` — torch-free, stdlib-only, CPU-only.

**Default polarity = LIVE.**  A file is ARCHIVED only when tracked text *declares* it
archival **and** nothing live revives it.  Three declaration kinds:

1. **in-tree marker** — a tracked `README*`/`INDEX*` whose title line carries an
   archival term (e.g. `scripts/_retired/README.md`);
2. **external declaration** — a tracked sentence naming a real tracked directory
   (`archived under \`TODO/archive/\``);
3. **sibling-index declaration** — a directory index that names its own immediate
   subdirectory *and* an archival term *and* an annotation marker
   (`TODO/INDEX.md` → `archive/  ← 6 STALE plans superseded by new …`).

A declaration names a tree that **must exist in the tracked tree**, so a sentence that
merely contains the word "archived" cannot invent an archive.

**Revival rule**: a full-path reference from a tracked, non-declared file revives an
**executable** target (a live script that still calls an archived launcher).  A live
*document* mentioning an archived document does **not** revive it (3 `TODO/archive/*.md`
files are referenced by live docs and stay archived).

**Result**: `results/` is **not** a declared archive tree →
`results/cfm_full_retrain/launch_loop.sh` is classified
`archived=false, basis=NO_ARCHIVAL_DECLARATION, executable=true`.
`TODO/archive/` **is** declared (by `TODO/INDEX.md`) → every file in it is archived.

**Disagreement with the brief, stated**: the brief asks for a LIVE/ARCHIVED verdict
driven by path shape ("`results/` is treated as archived").  I argue the path shape is
the bug and the default must be LIVE: an *unproven* "archived" claim silently hides
live defeat sites, and the only way `results/` got into `ARCHIVE_PREFIXES` was by
being typed in — no evidence in the repo declares that tree archival.

---

## 4. THE STAGED PATCH (written, NOT applied)

`molmetal/reports/wfa171_gpu_lock_guard/patch_guard_classification.diff` (4 624 B) —
a unified diff against `molmetal/scripts/wfa167_command_prefix_guard.py`:

1. docstring: the ARCHIVED class now means "declared archival", not "under a prefix";
2. a guarded import of `build_liveness_index` / `classify_liveness` from the A171 module
   (with a `sys.path` fallback for direct-script invocation);
3. `archived = path.startswith(ARCHIVE_PREFIXES)` marked PROVISIONAL;
4. inside `scan_repository`, **before** the `by_*` aggregation, every site's
   `archived` / `class` is rewritten from the evidence-based verdict and a
   `liveness_basis` is attached;
5. a `"liveness"` block published in the report dict.

Verification on a scratch git repo (`/tmp/a171_applycheck`):

    $ git apply --check molmetal/reports/wfa171_gpu_lock_guard/patch_guard_classification.diff
    GIT_APPLY_CHECK=PASS (final artifact)
    $ git apply molmetal/reports/wfa171_gpu_lock_guard/patch_guard_classification.diff
    GIT_APPLY=PASS

**New live-defeater count the patch yields**: on the counterfactual tree
`real_defeaters_live` goes **0 → 1** and `by_class` gains `REAL_DEFEATER: 1`
(§1.1).  On the current live tree the count stays 0 — because the one live defeater
(the results launcher) has already been repaired by hand in the working tree; the patch
fixes the *classification*, not the file.

The patch was NOT applied to the live tree.  (Note: `wfa167_command_prefix_guard.py` is
**untracked** in git — `git status --porcelain` shows `??` — so the patch applies to the
file as it sits on disk.)

---

## 5. TESTS

    $ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa171_gpu_lock_guard_path.py -q
    27 passed, 1 warning in 12.57s

Covers: `results/cfm_full_retrain/launch_loop.sh` is **not** archived; `TODO/archive/`
**is**; `results/` is never declared archived; `scripts/_retired/` is archived via its
own README; the four defeat mechanisms; `${VAR-default}` (not a defeater) vs
`${VAR:-default}` (a defeater); a real bash proof that the lock-preserving form leaves
an empty `CUDA_VISIBLE_DEVICES` empty and falls back only when the variable is unset;
the argv-position correction; and the hard constraints (no `torch`, no `nvidia-smi`, no
`os.walk`).

---

## 6. WHAT I DELIBERATELY DID NOT DO

* **Did not modify** `wfa165_gpu_lock_guard.py`, `wfa167_command_prefix_guard.py`, or any
  file outside my ownership list.  The classification change lives in a NEW module; the
  A167 change is a staged patch only.
* **Did not repair** `results/cfm_full_retrain/launch_loop.sh` — the main thread had
  already repaired it; touching it would have been an unrequested edit.
* **Did not touch** `paper/**`, `docs/**`, `AGENTS.md`, `TODO/**`, `configs/**`,
  `metrics/**`, `registry/**`, `run_ledger.jsonl`, `pre_flight_check.py`,
  `test_wf350c1_gpu_env_probe.py`, or any Wf-A170-owned path.
* **Did not run** the full pytest suite (it burns GPU VRAM and promotes metrics) — only
  my own test file.
* **Did not commit**, push, stash, checkout, clean or reset.

## 7. OPEN ITEMS / UNCERTAINTIES (not smoothed over)

1. **The 3 live POTENTIAL sites are not proven defeaters.**  `_common_train_args.py:585`
   is the only one with a reachable path (2 tracked configs set
   `cuda_visible_devices: "0"`); the two `training_utils.py` sites are
   lock-preserving for an EMPTY ambient lock by code reading, not by execution.
2. **44 tracked paths are missing on disk** (43 `OpenCL` files + 1 `flow_matching`
   directory) and **1 tracked file is non-UTF-8** — those are counted, not scanned.
   They are device/blob files, so no CVD site is lost, but the denominator is not
   100 % of the tracked tree.
3. **The scanner does not join backslash-continued logical lines** (stated precision
   bound).  A continued assignment whose `=` sits on the next physical line is reported
   at that line, and its command-word position is judged from that line alone.
4. **The counterfactual in §1.1 is a scratch copy** (`/tmp/a171_counter`) with the HEAD
   content of exactly one file; the numbers are reproducible there and nowhere else.
5. **The staged patch was verified with `git apply` on a scratch repo, not with the
   repo's own pre-flight** — the A167 guard is untracked, so `pre_flight_check.py` has
   no gate for it.
6. `detect_truncation` is happy with the patch only because it ends with a one-line
   comment; `git apply --check` tolerates it (verified rc=0).


> **CORRECTION #2 — 2026-11-04 (Wf-A269 wave-21, later the same day).  THE BLOCK ABOVE IS PRESERVED
> VERBATIM (AGENTS.md §5.5 rule #6); its ATTRIBUTION is measurably wrong, and it is corrected
> here.**
>
> The block above says "**5 launchers** were moved back out of `TODO/archive/wf437_simplify/` into
> `scripts/`, so 10 of the 11 archived defeaters became **LIVE**".  **That split is wrong.**
> MEASURED 2026-11-04 by comparing the census's **own** `snapshot.per_file_sha256` against HEAD's
> blobs and against today's tree:
>
> * **8 of the 10 live sites** are the five moved launchers — the **values are identical** to the
>   census's (`"${gpu##cuda:}"`, `1`, `"${WF310C_T179_CUDA_VISIBLE_DEVICES:-1}"` ×2,
>   `"${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}"` ×3, `"${idx}"`); only the **path** changed, so the
>   ARCHIVED/LIVE **classifier** flipped.  The census's sha256 for these five matches HEAD's
>   `TODO/archive/wf437_simplify/` blobs **exactly**.
> * **2 of the 10** are **re-armed regressions**: `scripts/launch_wf310f_frontier_math_gpu.sh` and
>   `scripts/_retired/launch_wf_fix3lines_cfm.sh` were recorded by the 2026-11-01 census as
>   **`ALREADY_LOCK_PRESERVING`** (line 249 `"${CUDA_VISIBLE_DEVICES-${gpu##cuda:}}"`, line 62
>   `"${CUDA_VISIBLE_DEVICES-0}"`) ⇒ they had **already been repaired**, and by 2026-11-04 they were
>   back to the defeater form.  `git log --all -S` returns **0 commits** for all three repaired
>   forms ⇒ **the repair was never committed and the repository holds no trail of it**.
> * ⇒ only **8 of the 11** archived defeaters left the archived class (**3 remain archived**).
>
> **⇒ ROOT CAUSE, STATED PRECISELY: THE CENSUS'S `snapshot.git_head` IS A FALSE PROVENANCE CLAIM.**
> It names commit `1b9470a0` while describing content that is in **no commit** — the census's
> `per_file_sha256` for the two re-armed files matches **neither** HEAD's blob **nor** today's
> tree.  The census is therefore **not reproducible from its own provenance**, and it cannot be
> re-derived from the tree it names.
>
> **⇒ AND THE RE-PINNED CENSUS REPEATS THE DEFECT.**  The new `census.json` (sha256
> `4df7eaef3ac91cfd…`) again stamps `git_head = 1b9470a0` while the tree it scanned carries
> **1352 dirty paths**, **5 staged renames** and **2 uncommitted repairs**.  Its `per_file_sha256`
> **does** match the tree it scanned ⇒ the per-file field is honest; the `git_head` field is not.
> **NOT FIXED** — the fix belongs in the guard's `snapshot` builder.
>
> **Full reading: `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.31.**
