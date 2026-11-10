# Wf-A167 — SYNTHESIS (DECISION-READY)

**Tag: MEASURED (2026-11-01). 0 GPU. Single venv `/home/hugo/codes/mol-metal/.venv/bin/python`.**
Repo HEAD `1b9470a0e2129e3c4d982da4453e034a6cc12e24`. This agent wrote nothing except this file.
Every number below traces to a path on disk; the path is named in each bullet.

---

## 1. RECOMMENDATION (lead)

**Land branch 1's 20-site repair + land the extension patch on the A165 guard. Do NOT land
branch 2's standalone `wfa167_command_prefix_guard.py` as a shipped tool as-is.**

Reason, in one line: **the repair is the fix; the tool was never the missing piece.** The existing
A165 guard already detects this class at **11/11 (100 %)** coverage
(`molmetal/reports/wfa167_command_prefix_guard/crosscheck_a165.json` → `step2_a165_coverage_of_my_real_defeaters.coverage`),
so shipping a second, differently-classifying scanner buys nothing — and branch 2's scanner carries a
**path-based `results/` bucket that I MEASURED to produce a false green** (§7 below): injecting a live
defeater into a scratch `results/` dir yields `real_defeaters_live: 0` while the A165 guard patched
with branch 2's own extension reports `defeats_ambient_empty_mask` 11 → **12**.

Concretely, the main thread should:
1. keep branch 1's working-tree repair of 19 `scripts/*.sh` (already applied, §5);
2. `git apply molmetal/reports/wfa167_command_prefix_guard/patch_guard_extension.diff` — it touches
   **exactly one file**, `molmetal/scripts/wfa165_gpu_lock_guard.py` (§6);
3. `git add molmetal/scripts/wfa165_gpu_lock_guard.py` — it is currently **UNTRACKED** (`git status`
   → `??`), and it is the file the patch modifies, so the patch cannot land until it is tracked;
4. keep branch 2's `census.json` / `crosscheck_a165.json` as **evidence artifacts** (they are the
   census of record), and keep its `wfa167_command_prefix_guard.py` only as a non-shipped analysis
   helper, or fix its `results/` bucket before shipping.

---

## 2. COMMAND-PREFIX CENSUS — DENOMINATOR + FOUR CLASSIFICATIONS

**DENOMINATOR (MEASURED, re-derived independently from `git ls-files -z -- '*.sh'`):**

| Quantity | Value | Source |
|---|---|---|
| Tracked `*.sh` files | **113** | `git ls-files -z -- '*.sh'`; `census.json` → `denominator.tracked_sh_files_total` = 113 |
| Files read | **113** | `census.json` → `denominator.files_read` |
| Files unreadable | **0** | `denominator.files_unreadable` |
| Files decoded lossily | **0** | `denominator.files_decoded_lossy`; `enumeration_errors` = `[]` |
| Token sites | **210** over **149** unique `file:line` | `census.json` → `counts.sites_total` = 210, `sites` = 210 records; independently re-counted: 210 literal `CUDA_VISIBLE_DEVICES` tokens over 149 lines |

**THE FOUR-WAY CLASSIFICATION — disjoint, over all 210 sites** (derived from `census.json` → `sites[]`;
branch 2 reports the same partition under two different axes, so both are given):

| Class | Sites | Meaning |
|---|---|---|
| **REAL_DEFEATER** | **11** | hard-forces the GPU; **all 11 are in `TODO/archive/`**; **live = 0** |
| **BENIGN** | **142** | comment / quoted string / non-assignment |
| **ARCHIVED** (other) | **12** | archived path, not a defeater |
| **ALREADY_OK** | **45** | already lock-preserving (`${VAR-default}` no-colon form) |
| | **210** | total |

> **CORRECTION — 2026-11-04 (Wf-A269 wave-21).  THE TABLE ABOVE WAS TRUE ON 2026-11-01 AND BECAME
> FALSE WHILE IT SAT ON DISK.**  The superseded rendering is preserved **verbatim** above (AGENTS.md
> §5.5 rule #6); nothing above was edited or deleted.
>
> **WHAT HAPPENED.**  The census is a **snapshot of a tree**, and the tree moved: **5 launchers** were
> moved back out of `TODO/archive/wf437_simplify/` into `scripts/`, so the 11 archived defeaters of
> 2026-11-01 became **10 LIVE** defeaters.  MEASURED 2026-11-04 by a fresh
> `wfa167_command_prefix_guard.scan_repository()`: `sites_total` **258** (not 210),
> `real_defeaters` **13** (not 11), `real_defeaters_live` **10** (not **0**),
> `real_defeaters_archived` **3** (not 11).  ⇒ the headline **"all 11 are in `TODO/archive/`"** and
> **"live = 0"** was **FALSE** on the live tree for the interval 2026-11-01 → 2026-11-04.
>
> **WHY NOBODY NOTICED.**  `molmetal/tests/test_wfa167_command_prefix_guard.py` compares the pinned
> census against a fresh scan — it **was RED** for those 3 days — but **no runner executes it**:
> CI names **12** explicit test files (none of them this one), `make test` is the only
> directory-level runner and **no CI job calls it**, and `pre_flight_check.py` never names the guard
> or its test.  ⇒ this is the §6.269.29 **"a guard that no runner executes"** class, on a second
> guard.  The guard's own CLI also exits **1** on this tree (3 archived defeaters remain) and **no
> runner calls it either**.
>
> **REPAIRED.**  The **10 live** sites were repaired with the guard's OWN recommended form
> (`lock_preserving_form`, the no-colon `${VAR-default}` form ⇒ an explicitly empty ambient mask
> survives).  MEASURED after the repair: `real_defeaters_live` = **0**,
> `real_defeaters_archived` = **3** (the `TODO/archive/wf437_simplify/` sites, deliberately NOT
> touched), `sites_total` = **268**, `by_class` = {ARCHIVED 10, BENIGN 202,
> ALREADY_LOCK_PRESERVING 56}.  **MUTATION-PROVED both ways**: reverting all 10 ⇒ the guard test
> FAILS; reverting **one** site ⇒ the guard test FAILS; restoring ⇒ **19 passed**.
> `molmetal/reports/wfa167_command_prefix_guard/census.json` was re-pinned from the fresh scan; the
> 2026-11-01 rendering is preserved **byte-for-byte** at
> `census_superseded_2026-11-01.json` (sha256 `d9d98142fe8c75de…`) and its counts are copied
> verbatim into the new census's `superseded` block.
>
> **⇒ THE COUNTS ABOVE ARE NOW A COVERAGE CENSUS, NOT A DEFECT COUNT** (§6.269.13): `sites_total`
> **rose** 210 → 268 across the two repairs (each repaired site now carries the token twice — as the
> assignment and inside the quoted expansion) while the **live defect count fell** 10 → 0.
>
> **STILL OPEN.**  The guard's `ARCHIVE_PREFIXES` is a **path heuristic** = `("TODO/archive/",
> "results/")`; the repo's own **canonical, CITED** archival list names **`scripts/_retired/`**
> (`scripts/_yaml_only_gate.py:108` + `:131`; `docs/HONEST_NEGATIVE_CATALOG.md` §6.269 "the
> ARCHIVAL exclusion is CITED, not HEURISTIC … a path heuristic is a known-false instrument here").
> ⇒ the guard classifies a tree the repo's own policy calls archival as **live**.  NOT changed in
> this wave: altering a classifier mid-census to move a number is the anti-pattern this repo
> distrusts; the correct repair (a CITED exclusion table mirroring `_yaml_only_gate.py`) is a
> separate change that deserves its own census re-pin.  The one `scripts/_retired/` site was
> repaired anyway, so it is harmless under **either** classification — and the repair is
> load-bearing for the **documented re-run path** (`scripts/_retired/README.md`: "If you need to
> re-run one of these: `git mv scripts/_retired/<name>.sh scripts/<name>.sh`" ⇒ the file becomes a
> LIVE launcher in `scripts/`, where its old line was a live CPU-lock defeater).

Branch 2's own two axes, for traceability (`census.json` → `counts`):
`by_class` = {ARCHIVED 23, BENIGN 142, ALREADY_LOCK_PRESERVING 45};
`by_semantic_class` = {REAL_DEFEATER 11, BENIGN 153, ALREADY_LOCK_PRESERVING 46};
`real_defeaters_live` = **0**, `real_defeaters_archived` = **11**.
**Location × semantics cross-tab (MEASURED from `sites[]`):** live = 187 sites (142 BENIGN + 45 ALREADY_OK,
**0 REAL_DEFEATER**); archived = 23 sites (11 BENIGN + 11 REAL_DEFEATER + 1 ALREADY_OK).

**Scope note:** the census is **tracked** `*.sh` only. It does NOT cover the **449** `*.sh` under
`.claude/worktrees/` (gitignored). Re-scanning those with branch 2's own scanner I measured
**129 command-prefix real defeaters across 108 files** (76 files / 81 sites on non-archived paths).
**0 live tracked code references `.claude/worktrees`** (`git grep -n "\.claude/worktrees" -- 'scripts/*.sh' 'molmetal/scripts/*.py'` → 0 hits),
so this is not a live defect of the main checkout — but an operator running from a worktree gets the
lock-defeating behaviour. Honest-negative, scope caveat (branch-2 refutation C3 claims "72 files";
my own count is 108 files / 129 sites — the refutation's figure is not reproducible as stated).

---

## 3. DIFF AGAINST Wf-A165's 32-SITE LIST — BOTH DIRECTIONS

A165's list of record: `molmetal/reports/wfa165_shell_launchers/proof.json` → `census.command_prefix_census`
(`sites_total` = 32, buckets: 20 in live `scripts/`, 11 in `TODO/archive/`, 1 in `results/`).

**File level — EMPTY IN BOTH DIRECTIONS (MEASURED):**
- A165-not-mine = **[]** ; mine-not-A165 = **[]**
- Identical 27-file set, identical per-file site counts, 32 == 32 sites.
- Verified three ways: (a) branch 2's `sites[]` filtered to `shape == COMMAND_PREFIX` vs A165's `sites[]`;
  (b) branch 1's `census.json` `live_real_defeaters` (20) vs A165's live 20 and `archived_real_defeaters` (12)
  vs A165's archived 12 — both directions empty in both splits; (c) **the A165 guard itself re-run against
  a `git archive HEAD` snapshot** yields exactly **32** `shell_prefix_assignment` findings, set-equal to
  A165's own 32-site list in both directions.

**Line level — 20 shifted, 12 identical.** The 20 live sites moved by **+1** line (**+2** on the second
`scripts/launch_wf308c_todo220_gpu.sh` site) because branch 1's repair inserts a one-line `# Wf-A167: …`
WHY comment above each site. The 12 archived/results sites are byte-identical and line-identical
(branch 1 did not touch them). **No site was added and none was lost.**

---

## 4. ARCHIVED-SCRIPT POLICY VERDICT

**VERDICT: archived scripts are NOT LIVE and were correctly left alone. 11 archived real defeaters remain.**

Grep evidence (all MEASURED on the working tree):
- **0** live tracked `*.sh` reference `TODO/archive/` — `git grep -ln "TODO/archive" -- '*.sh' ':!TODO/archive/*'` → **0**.
- **0** live tracked `*.sh` invoke an archived launcher *by its archived path*; the only `TODO/archive`
  mentions outside the archive itself are **prose in 4 test files**
  (`molmetal/tests/test_wf327_ultracode_closeout.py:60`, `test_wf328w2_launcher_num_workers.py:86-87`,
  `test_wf333t3_trainer_verify.py:275`, `test_wf337g2_launcher_v2.py:49,186`) — no execution.
- Each of the 7 archived launchers named in the census has **no live twin** (the live path is absent from
  disk AND from `git ls-files`); the archived copy under `TODO/archive/wf437_simplify/` is the only copy.
- **8 dead refs** to archived launchers' old live paths (targets missing from disk **and** from
  `git ls-files`): `scripts/gpu_pipeline_master.sh:209,224,231,237,246,261` (6 executable `bash` lines
  that would fail with "No such file or directory" if run today), `scripts/gpu_smoke_when_vram_free.sh:211`,
  `scripts/run_wf264_cfm_property_panel.sh:71`. **PRE-EXISTING** — all three host files are unmodified
  vs HEAD (`git status --porcelain` blank for them) — and **not introduced by Wf-A167**. Branch 1's
  refutation S2 counted **8**, not 4: my own enumeration confirms **8** (6 + 1 + 1). This is a real
  honest-negative for the main thread, and it is a **different defect class** (dangling path) from the
  GPU-lock class this wave was chartered for.

**Consequence for the policy:** the 11 archived real defeaters are a **documented, deliberate
non-repair**. They are real device-forcing sites and they must be re-repaired if any of those launchers
is ever restored to `scripts/`.

---

## 5. REPAIR COUNT + THREE-WAY PROOF

**REPAIRED: 20 sites across 19 files**, all under `scripts/` (none in `results/`, none archived).
- Site list: `molmetal/reports/wfa167_command_prefix_scripts/proof.json` → `sites[]` (20 entries) and
  `files[]` (19 entries). `proof_summary` = {sites 20, pass_count 20, fail_count 0, files_any_fail 0,
  all_three_pass true}.
- Every one of the 19 files shows ` M` in `git status --porcelain -- scripts/*.sh` (19 of 19 confirmed).
- `bash -n` on all 19 files: **0 failures** (re-verified).

**THREE-WAY PROOF — MEASURED 20/20 PASS**, independently re-derived by me (not read from `proof.json`):
- ambient **unset** → the **original** pre-repair default (e.g. `"0"`, `"1"`, or the site’s computed default);
- ambient **`""`** → **empty** (lock honoured);
- ambient **`"1"`** → **`"1"`**.

Reproduction: I evaluated each repaired expression under `env -u CUDA_VISIBLE_DEVICES` / `""` / `"1"`
with each site's own `extra_env` binding, comparing against the **pre-repair** value taken from
`git show HEAD:`. Result **20/20 PASS**. (`proof.json` → `ambient_unset_original_mask`,
`ambient_unset_repaired_mask`, `ambient_empty_repaired_mask`, `ambient_one_repaired_mask`.)

**Two honest caveats on the proof, both verified on disk:**
- 6 of the 20 defaults are **computed** (`${idx}` ×3, `${gpu##cuda:}` ×2, `${gpu}` ×1, plus
  `${WF308C_CUDA_VISIBLE_DEVICES:-0}` ×2). For these, the repaired expression embeds the pre-repair
  expression **verbatim** as the default, so the "unset → original default" property holds for **any**
  value of that variable — but a naive harness that does **not** bind that variable reports 14/20, not
  20/20. I measured exactly this: 14/20 with unbound vars, **20/20 with the per-site binding**. The
  branch's number is right; the trap is real.
- The lock-preserving form is `${VAR-default}` (**no colon**). 2 sites in
  `scripts/launch_wf308c_todo220_gpu.sh` are nested
  (`${CUDA_VISIBLE_DEVICES-${WF308C_CUDA_VISIBLE_DEVICES:-0}}`) — the **outer** form is the
  no-colon one, so the lock is honoured; the inner `:-` is the *default expression*, not the gate.

---

## 6. DOES THE EXISTING A165 GUARD ALREADY DETECT THIS CLASS?

**YES — MEASURED COVERAGE 11/11 (100 %).** Source:
`molmetal/reports/wfa167_command_prefix_guard/crosscheck_a165.json` →
`step2_a165_coverage_of_my_real_defeaters.coverage` = `"11/11"`, both difference directions empty,
verdict recorded as *"the A165 guard ALREADY detects the command-prefix class; the tool was never the
missing piece for the live sites — the REPAIR was."*

**I re-derived this independently** rather than trusting the file:
- Ran `molmetal/scripts/wfa165_gpu_lock_guard.py --json` on the live working tree →
  `shell_prefix_assignment` findings = **11**, and their `(file,line)` set is **set-equal** to branch 2's
  `real_defeater_sites` (11) — `A165-not-mine` = `[]`, `mine-not-A165` = `[]`.
- Ran the same guard against a `git archive HEAD` snapshot → **32** `shell_prefix_assignment` findings,
  set-equal to A165's own 32-site list in both directions.
- **So: the A165 guard has always detected this class.** The tool was never the missing piece; the
  **repair** was. That is the single most decision-relevant fact in this wave.

### THE EXTENSION PATCH

- **`git apply --check` → rc = 0** (working tree, and again on a scratch copy of the same file).
- **Touches EXACTLY ONE FILE**: `molmetal/scripts/wfa165_gpu_lock_guard.py`
  (`grep -c "^--- "` on the diff = **1**; header is
  `--- a/molmetal/scripts/wfa165_gpu_lock_guard.py` / `+++ b/molmetal/scripts/wfa165_gpu_lock_guard.py`).
  It adds a `command_prefix_census` **report section only** — it is deliberately **not** promoted into
  `findings`, so M1/M2 semantics and the report's vacuity invariant are preserved.
- **Behaviour-neutral, MEASURED**: on a scratch copy, the patched guard's report differs from the
  unpatched guard's in exactly two ways — the `generated` timestamp, and the **added**
  `command_prefix_census` key. Every other key is byte-identical (`keys with changed content
  (excl generated)` = `[]`). The guard's **exit code is unchanged** (1 before, 1 after, on the working
  tree — the tree still carries 33 `defeats_ambient_empty_mask` findings, 22 of them in
  `molmetal/tests/*.py` under mechanism `M3_env_dropping_spawn`, which is a **different class** and out
  of this wave's scope).
- The patched census reports **61** sites (arg_position 1 / command_prefix 32 / export 28) with
  `by_effect` = {forces_blank 3, not_an_env_assignment 1, preserves_ambient 46, unmask 11} and
  `defeats_ambient_empty_mask` = **11**; denominator 113 / 113 / 0 unreadable / 0 lossy.
- **NOT applied to the working tree** (`applied_to_working_tree: false`). The main thread must apply it.
- **LANDING BLOCKER**: `molmetal/scripts/wfa165_gpu_lock_guard.py` is currently **UNTRACKED**
  (`git status` → `?? molmetal/scripts/wfa165_gpu_lock_guard.py`; `git ls-files --error-unmatch` fails).
  `git apply` succeeds on the untracked file as-is (rc = 0, verified), but the file must be `git add`ed
  in the same commit so the patch and its target land together. sha256 of the current target:
  `2112191ccbaac04e08cdec4a57e286fb405ee0aaa6bdf8f3dd0d5d35aabb13a5`.

---

## 7. THE HONEST-NEGATIVE THAT CHANGES THE RECOMMENDATION

**Branch 2's shipped scanner buckets `results/` as ARCHIVED by PATH** — `ARCHIVE_PREFIXES = ("TODO/archive/", "results/")`
in `molmetal/scripts/wfa167_command_prefix_guard.py`. I measured the consequence on a scratch tree
(all 113 tracked `*.sh` copied to `/tmp/a167inj`, one live defeater injected at
`results/probe/live_loop.sh`):

| Tool on the injected tree | Result |
|---|---|
| branch 2's `wfa167_command_prefix_guard.py` | `real_defeaters` 12, **`real_defeaters_live` 0**, `real_defeaters_archived` 12 — **FALSE GREEN**: a live tracked executable defeats the lock and the tool reports live = 0 |
| branch 2's own extension applied to the A165 guard | `defeats_ambient_empty_mask` **12** — **the injected site IS caught** (it names `results/probe/live_loop.sh:2` as `M1_hard_override`, effect `unmask`) |

This is the **same defect class the main thread already caught and recorded** in
`docs/HONEST_NEGATIVE_CATALOG.md` §6.266.7 (2026-11-01): *"a guard that reports `live: 0` while a
tracked executable file still defeats the lock is worse than no guard — it manufactures false
assurance."* The `results/` bucket is still a **path** heuristic; the structural lesson was not
carried into branch 2's tool. **Therefore: prefer the A165 guard + extension patch (which classifies by
reachability) over shipping branch 2's standalone scanner as a gate.**

---

## 8. OPTIONS I REJECTED, AND WHY

| Rejected option | Why rejected |
|---|---|
| **Repair the 11 archived real defeaters** (7 files under `TODO/archive/wf437_simplify/`) | Archived = read-only historical artifacts; repairing them would silently rewrite history. Leave as a documented honest-negative; re-repair on restore to `scripts/`. |
| **Fix the 8 dead refs to archived launchers' old live paths** | Out of this wave's class (dangling path, not GPU-lock) and **pre-existing** (the 3 host files are unmodified vs HEAD). Correct call: log as an honest-negative with the `gpu_pipeline_master.sh` 6-job breakage stated explicitly. |
| **Ship branch 2's standalone scanner as the gate** | Its `results/` path bucket demonstrably reports `live: 0` on a live defeater (§7). Shipping it would re-introduce the exact false-assurance defect §6.266.7 records. |
| **Apply the extension patch in this wave** | HARD RULE: read-only — this agent writes only this file. `git apply --check` rc = 0 is the readiness proof; the apply is the main thread's call. |
| **Treat the 5 failing `test_wfa131_repo_path_reference_guard.py` tests as a Wf-A167 regression** | **Cross-wave, not ours.** The offending citation is `molmetal/scripts/auto_emit_section_6_minor.py`, added by the working-tree catalog diff; the file **never existed** (`git log --all` → empty) and the catalog's mtime (12:14:49) is **after** both refutations (12:07:59 / 12:05:46). The failure is the catalog citing a never-existing path. Fix the citation, not Wf-A167. |
| **Treat branch 2's `crosscheck_a165.json` `arg_position_sites` (50 coords) as the arg-position census** | **Mis-labelled.** Cross-checked against branch 2's own `sites[]`: of those 50 coords, **47 are `QUOTED_STRING` and 3 are `EXPORT_ASSIGNMENT`**; exactly **1** is truly `ARG_POSITION_ASSIGNMENT` (`scripts/launch_wf308c_todo220_gpu.sh:125`). Confirmed independently by re-running branch 2's scanner: `by_shape.ARG_POSITION_ASSIGNMENT` = **1**. (Branch-2 refutation C2 raised this; I confirmed it numerically.) |
| **Ship the `results/`-bucket fix as part of landing branch 2's scanner** | If the main thread wants that scanner shipped, it must first classify `results/` by **reachability/executability**, not path — a code change beyond this wave's read-only scope. |

---

## 9. FILES EACH BRANCH CREATED (all untracked at time of writing)

**Branch 1 (repair):**
- `molmetal/reports/wfa167_command_prefix_scripts/wfa167_command_prefix_repair.py` (26 087 B)
- `molmetal/reports/wfa167_command_prefix_scripts/census.json` (58 083 B)
- `molmetal/reports/wfa167_command_prefix_scripts/proof.json` (30 255 B)
- `molmetal/tests/test_wfa167_command_prefix_scripts.py` (24 530 B)
- **modified 19** `scripts/*.sh` (the 20 repaired sites)

**Branch 2 (guard + patch):**
- `molmetal/scripts/wfa167_command_prefix_guard.py` (39 996 B)
- `molmetal/tests/test_wfa167_command_prefix_guard.py` (16 236 B)
- `molmetal/reports/wfa167_command_prefix_guard/census.json` (149 596 B)
- `molmetal/reports/wfa167_command_prefix_guard/crosscheck_a165.json` (13 001 B)
- `molmetal/reports/wfa167_command_prefix_guard/patch_guard_extension.diff` (13 666 B)

**Refutation 1:** `molmetal/reports/wfa167_branch1_command_prefix_refutation/{refutation.json, refutation.md}`
**Refutation 2:** `molmetal/reports/wfa167_branch2_refutation/{refutation.json, refutation.md}`

**Pre-existing from Wf-A165 (also untracked, also needed):**
`molmetal/scripts/wfa165_gpu_lock_guard.py` (55 810 B), `molmetal/tests/test_wfa165_gpu_lock_guard.py`,
`molmetal/tests/test_wfa165_shell_launchers.py`, `molmetal/tests/test_wfa165_start_shell.py`,
`molmetal/reports/wfa165_{gpu_lock_guard,shell_launchers,start_shell,synthesis,b_refutation,c_refutation,branchA_refutation}/`.

**Tests (MEASURED, CPU-only, `CUDA_VISIBLE_DEVICES=""`):**
`pytest molmetal/tests/test_wfa167_command_prefix_scripts.py molmetal/tests/test_wfa167_command_prefix_guard.py` → **30 passed**.
The 8-file Wf-A167/A165 sibling set → **108 passed / 2 skipped**. The 9-file set adds
`test_wfa131_repo_path_reference_guard.py` → **5 failures, all inside that one file**, attributable to
the cross-wave catalog citation described in §8.

---

## 10. WHAT THE MAIN THREAD MUST DO NEXT (ordered)

1. **Land the repair.** The 19 modified `scripts/*.sh` are already in the working tree and verified
   (20/20 three-way, `bash -n` clean, 0 live real defeaters). Commit them with the WHY comments.
2. **Track + patch the A165 guard.** `git add molmetal/scripts/wfa165_gpu_lock_guard.py` (currently `??`),
   then `git apply molmetal/reports/wfa167_command_prefix_guard/patch_guard_extension.diff`
   (`git apply --check` rc = 0, verified in-tree and on scratch). Re-run the guard afterwards and
   confirm the only new key is `command_prefix_census`.
3. **Decide branch 2's scanner.** Recommendation: do **not** ship `wfa167_command_prefix_guard.py`
   as a gate until its `results/` bucket is reachability-based (§7). Keep `census.json` +
   `crosscheck_a165.json` + `patch_guard_extension.diff` as the wave's evidence.
4. **Wire the census into the pre-flight.** NEITHER guard is referenced by
   `molmetal/scripts/pre_flight_check.py` today (`grep -c` = **0**). The census is measured but not
   enforced. `molmetal/scripts/pre_flight_check.py` also carries a **2 429-line working-tree diff**
   (2 275 insertions) of unknown authorship, inside this wave — the main thread should attribute it
   before committing it (branch-2 refutation C6).
5. **Log the honest negatives** in `docs/HONEST_NEGATIVE_CATALOG.md` (§6.266.x): the 11 archived real
   defeaters left in place by policy; the 8 dead refs (6 of them live `bash` lines in
   `scripts/gpu_pipeline_master.sh`); the 129 command-prefix defeaters in the gitignored
   `.claude/worktrees/` copies (out of the tracked-file census); the `wf308c:125` argv-position
   launcher (`setsid nohup CUDA_VISIBLE_DEVICES=…` is an **argv element**, so `nohup` exits 127 —
   a **broken invocation**, not a lock defeat); the two `nvidia-smi -i` index uses of
   `${CUDA_VISIBLE_DEVICES:-0}` (`scripts/launch_wf299_cfm_main.sh:560`,
   `scripts/launch_wf310f_frontier_math_gpu.sh:276` — reads device 0 on an empty lock, allocates
   nothing, so not a defeater); and the mis-labelled `arg_position_sites` field in
   `crosscheck_a165.json` (50 coords, only 1 genuinely arg-position).
6. **Fix the cross-wave citation** `molmetal/scripts/auto_emit_section_6_minor.py` in
   `docs/HONEST_NEGATIVE_CATALOG.md` (§6.266.5) — it names a file that never existed, which is what
   turns `test_wfa131_repo_path_reference_guard.py` red. Not Wf-A167's regression; Wf-A167's sibling
   tests are 30/30 green.

---

## 11. REPRODUCTION (CPU-only, 0 GPU)

```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  molmetal/tests/test_wfa167_command_prefix_scripts.py \
  molmetal/tests/test_wfa167_command_prefix_guard.py -q          # 30 passed
git apply --check molmetal/reports/wfa167_command_prefix_guard/patch_guard_extension.diff   # rc 0
git ls-files -z -- '*.sh' | tr '\0' '\n' | grep -c .              # 113
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa165_gpu_lock_guard.py --json --advisory --quiet --out /tmp/a167_guard_live.json
```
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` → the only compute PIDs belong to
`/home/hugo/codes/planB` (a different repo); **Wf-A167 allocated 0 MiB of GPU memory.**


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
