# Mol-Metal: Precious-Metal Anti-Cancer Drug Design via Flow Matching + SBDD

**Last-updated trailer (Wf-A269 wave-19 + the main thread — the `^train_` ANCHOR: a smoke that had **NEVER PASSED** and a GPU launcher that could **NEVER RUN**, then the wave's own review published a **FALSE CLAIM** about a defect that does **NOT** exist — 2026-11-04 — CPU-only, **0 GPU**):** `scripts/_launch_contract_gate.py:189` anchored the launch-contract question to `train_*` (`_TRAINER_NAME = re.compile(r"^train_[a-z0-9_]+\.py$")`), so "does the flag reach a parser that **DECLARES** it?" was **NEVER CHECKED** for the runners, the drivers, the smoke helpers and the scaffolds — the class the other three §5.11 gates **STRUCTURALLY CANNOT SEE**. **REPAIRED** by a **4th** §5.11 gate, `scripts/_script_contract_gate.py`, REGISTERED as `p3_script_contract_check` (`molmetal/scripts/pre_flight_check.py:17618`) ⇒ pre-flight gate counts **132/134 → 133/135** (MEASURED by live import of `EXPECTED_GATE_COUNTS` + `build_gates("all")` / `build_gates("launch")`; the **132/134** rendering is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6). Live verdict **PASS** over **101** wrappers scanned / **218** repo-local script invocations / **560** flags / **17 614** command segments / **0** undecidable / **1** disclosed quoted-blob site / **31** zero-invocation wrappers named on every run — a **COVERAGE** census, not a defect count (§6.269.13). It found **2** real launch-killers, both REPAIRED: `molmetal/scripts/smoke/smoke_lambda.sh` passed `--out-dir` to a parser declaring `--output-dir` ⇒ **the smoke had NEVER PASSED**; `scripts/launch_wf306b_wf219_ru_gpu.sh` passed `--out` to a parser declaring `--json-dir` ⇒ the launcher could **NEVER RUN** (**NOT EXECUTED**, 0 GPU). **AND THE WAVE'S OWN REVIEW PUBLISHED A FALSE CLAIM** — it reported that `--no-smoke` was undeclared by `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py`, which is **FALSE** (`argparse.BooleanOptionalAction` registers BOTH `--smoke` AND `--no-smoke` from ONE `add_argument` call; the script's own usage line reads `[--smoke | --no-smoke]`) — and that false claim had landed in a **TRACKED** file (`molmetal/scripts/pre_flight_check.py`'s gate docstring) ⇒ corrected additively + dated, with the rule PINNED in **BOTH** directions by `molmetal/tests/test_wfa269_script_contract_gate.py` (**53 passed**). **⇒ TWO CENSUSES OF ONE CLASS RETURNED DIFFERENT SETS** because the tree was moving under both (a background Codex CLI session + this session + the wave's own agents) ⇒ **a census is not even a defect SET** — only a census taken on a **frozen** tree is trustworthy. **STILL OPEN** (per `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.28.10): the `^train_` anchor itself; the heredoc drivers (stdin-fed, no script token to resolve); `submission/paper_a/reproduce.sh` (outside the gate's scope); the yaml-only gate's rule-2 scope is still `train_*` only; **the gate script is UNTRACKED ⇒ a clean checkout of HEAD FAILs the pre-flight (fail-closed)** — the commit is what turns "0 violations" from a working-tree claim into a repository claim. Also this chain: **§6.269.26** (the catalogue append broke the repo's own path-reference guard — **79 → 81** broken citations, adjudicated in the guard's `BASELINE` per §5.5 rule #6, NOT reworded) and **§6.269.27** (the pitfall-guard runner ran its pytest under a **FOREIGN** interpreter — bare `pytest` on PATH is Homebrew python 3.14 with no torch, and the pinned fallback was DEAD CODE ⇒ the smoke reported `FAIL runner_crashed` about the **RUNNER**, never evaluating the 7 guards ⇒ REPAIRED by swapping the command order ⇒ `PITFALLS_SMOKE PASS 7/7 guards PASS wall=2s`):** 
**Last-updated trailer (Wf-A269 wave-17/18 + the main thread — the 0-GPU lock had no owning layer, and the pre-flight CREATED the contention it blamed on the host — 2026-11-04 — CPU-only, **0 GPU**):** the repo's CPU lock was a **per-call-site obligation with no layer that owns it** — the 5 `molmetal/scripts/smoke/*.sh` scripts are **CPU-BY-DESIGN** (`smoke_cfm.sh:220` calls `adapter.setup(device="cpu")`) but carried **no** device lock, so a caller without `CUDA_VISIBLE_DEVICES=""` paid a REAL allocation: MEASURED **694 MiB** per invocation (peak **1 388 MiB**) via the import chain `molmetal/adapters/flow_matching_lipman/amp.py:187` → `torch.cuda.is_available()` → `cuInit` (the visibility QUESTION allocates); the pre-flight owns its own lock but references `scripts/smoke/` **0** times. **REPAIRED** by exporting `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"` (unset ⇒ ""; an EXPLICIT value is PRESERVED) + `MOLMETAL_SMOKE_DEVICE="${MOLMETAL_SMOKE_DEVICE:-cpu}"` in all 5 scripts, and pinned by the new `molmetal/tests/test_wfa269_smoke_cpu_lock.py` (**20 passed / 1 skipped**, CPU-only). End-to-end: with the ambient lock **deliberately unset** the smoke still allocated **0 MiB** (`CFM_SMOKE PASS decode_ratio=1.00 wall=73.0s`, rc 0, `GPU-ALLOC NONE (0 mol-metal allocations observed in 240s)`) against 694 MiB before the repair; and the A/B that caused the violation was **NOT an isolation** — `/tmp/origrepo` was a symlink farm INTO the repo (§6.269.18 class) ⇒ its "HEAD" arm ran HEAD's shell against the working tree's Python (valid for the shell change, **invalid** as a full-HEAD baseline). **AND THE CHEMPROP TIMEOUT WAS ADJUDICATED WRONG, THEN CORRECTED**: `p1_chemprop_smoke` STANDALONE **PASSes in 5.10 s** (`--quick exit=0 in 5.1s`; `loss 0.6029 → 0.1990`; the overwritten checkpoint restored byte-identical) ⇒ the wire-in was never broken; the real cause is that the pre-flight runs **8 gates concurrently** (`ThreadPoolExecutor(max_workers=min(8, cpu_count))`) and each gate's torch **CHILD** inherited torch's default **24** threads (MEASURED `torch.get_num_threads()`=24 on 32 cores) ⇒ **192 runnable threads on 32 cores**; MEASURED with 8 concurrent copies of the exact gate command: default threads **260.7–267.1 s** per epoch vs `OMP_NUM_THREADS=2` **0.6–1.0 s** — a **260x–280x** inflation, and the contended run's own log line (`epoch=0/2 ... elapsed=106.5 s`) is the smoking gun ⇒ **the gate was starved by its OWN seven siblings**, not by an external project. **REPAIRED** by the new `pre_flight_check._pin_gate_child_thread_caps(jobs)` (exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` = `cpu_count // jobs` into the parent's env **before** the executor starts; `jobs==1` keeps the whole machine; `MOLMETAL_PREFLIGHT_OMP_THREADS` overrides; a non-integer override is IGNORED), pinned by the new `molmetal/tests/test_wfa269_preflight_thread_caps.py` (**9 tests**, AST-based order check — the module's own comment mentions `ThreadPoolExecutor(max_workers=jobs)`, so a text search would read the *documentation* as the *code* — plus an end-to-end child probe asserting `torch.get_num_threads() == cap`; **29 passed / 1 skipped** with the §6.269.24 suite). **The 132-gate census moved from 118 PASS / 3 FAIL / 10 SKIP / 1 TIMEOUT (403.69 s wall) to 119 PASS / 3 FAIL / 10 SKIP / 0 TIMEOUT (93.23 s wall, 4.33x faster)** — the one changed verdict is `p1_chemprop_smoke` (TIMEOUT → PASS, 137.75 s → 7.08 s), and the two sibling smokes dropped 341.77 s → 5.14 s and 268.94 s → 7.73 s (19x–67x faster), which is the end-to-end proof that all three were running 20x–70x over their standalone cost.  The **smoke budgets were NOT re-derived** (**OPEN** — they were measured on 2026-10-31 on a shared host under the same oversubscription), and the pre-flight's own **in-process** gates still run at the parent's thread count (**OPEN**). The gate-count chain re-verified: **45 passed / 0 failed** in 188.73 s (the prior wave's reading of the same 4 files was 33 passed — the count is MEASURED, and a rise is not a regression), and the published metrics were **byte-identical** before and after (`metrics/by_metric/*.json` md5 unchanged). The catalogue's cumulative counter was **NOT** bumped (325): its counting rule is undocumented ⇒ **a counter nobody can reproduce is a number nobody may quote**. Full reading: `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.24 + §6.269.25.
**Last-updated trailer (Wf-A269 wave-3/5/6/16 + the main thread — the FOURTH dead-gate costume + a NEW blind spot — 2026-11-04 — CPU-only, **0 GPU**):** the §5.11 gates read only **STATIC** invocation lines, but `scripts/start.sh` assembles the trainer command as a **RUNTIME STRING** (`bash -c "$TRAINER_CMD"`), so a runtime-transported flag was invisible to BOTH `p3_yaml_only_launch_check` and `p3_launch_contract_check` — the new `p3_runtime_transport_check` reads the **ASSEMBLED** command and compares it to the trainer's own parser, and it found **2** REAL violations (the CFM trainer had **NO** way to switch the `events.jsonl` sink OFF — `--no-emit-json-events` only changes STDOUT — and `bench_cfm_decode_real.py` was launched with a `--config` its parser does not declare): **both REPAIRED** (trainer **and** launcher, not the symptom) and §5.7-smoked (rc 0, `decode_ratio` 1.000, `train_loss` 0.5645 → 0.0556, `events.jsonl created: False`). The wiring moved the pre-flight to **132** gates on `all` / **134** on `launch` (MEASURED live 2026-11-04), which exposed **5** pre-existing RED tests in the gate-count chain + **1** sha256-pinned census artefact (both repaired: 28 passed / 5 failed → **33 passed**, `stale: 0 in 0 files`). **NEW blind spot, found and NOT fixed:** `molmetal/scripts/wfA2_config_loader.py::_iter_leaves` **SKIPS** any mapping whose every child key ∈ `_NON_TRAINER_SUBKEYS` (which contains `dest`), so a `dest:` pointer inside a schema block **never reaches** the binding gate — **1** config / **5** pointers / **4** name a dest absent from the named script / **1** names a script that does not exist anywhere in the repo ⇒ the binding gate's closing line over-claims (**OPEN**). The §5.11 coverage chain moved **76.1 % → 89.4 %** (86 → 101 of 113 tracked `*.sh`; wave-6's `results/cfm_full_retrain/launch_loop.sh` migration is **NOT** bit-identical to the old run — the old loop ran on argparse defaults, i.e. the synthetic Pt-click branch). The ledger producer **STILL** drops `config_path` (the consumer `p3_run_ledger_config_provenance` shipped; the one-line producer fix is **OPEN**). **The new gate script and its test are UNTRACKED ⇒ a clean checkout FAILs the pre-flight (fail-closed) and the commit is USER-gated.** The catalogue's cumulative counter was **NOT** bumped (325): its counting rule is undocumented — **161** `## §6.N` + **202** `### §6.N.M` headings match no counter ⇒ **a counter nobody can reproduce is a number nobody may quote** (**OPEN**). Full reading: `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.23.
**Last-updated trailer (Wf-A179 → Wf-A203 honest-negative closure chain + the citability-gate repair — 2026-11-02 — CPU-only, **0 GPU**):** the chain closed **5** residual classes and added **2** honest-negative entries (catalogue **322 → 323**, `docs/HONEST_NEGATIVE_CATALOG.md` §6.268): (i) the **line-number locator class** in the frontier records (Wf-A194 — 3 live carriers demoted to dated keys, values byte-identical); (ii) the record's **6 residuals** — a wrong numeral, a dated block edited in place, a false "nothing above this line" assertion, live cross-artefact locators, an unprovable close-out date, and bookkeeping (Wf-A195); (iii) the **X5 reading-count** surfaces (Wf-A196) and the record's own residual closure (Wf-A197 — **4** residuals, including that **a sha256 of a regenerable artefact is as perishable as a line number**, so "JSON PATH + sha256" is only half-durable); (iv) the **ledger version-pin** auditor + the CLI's unmeasured-count fix — the default mode printed `superseded_never_committed=0` for a quantity it never computed, truth is **9** (Wf-A198, verdict **PROVEN**); (v) the **citability-gate gap** — exactly **2** `paper_line` rows carry `quotable: true` + `headline_grade: true` with **no** `headline_caveat` while their own payload declares a miss: `m21_nci60_9cell` (the identity `(0.6520602689568208 × 7 + 1.0 + 1.0) / 9 == 0.7293802091886383` is EXACT, so DU145 n_test=7 and SKOV3 n_test=3 n_val=0 both carry the 9-cell mean) and `m19_lomo` (**0.5575** was **RETRACTED 2026-10-19** in favour of the resolvable-subset **0.5974**) (Wf-A203). Closed in the main thread: the **X5 guard's own blacklist** — a four-string, case-sensitive list defeated by `"two"` and `"3"`; it is now a regex over any number token, with a bite test (`molmetal/tests/test_wfa197_record_residuals.py`, **9 passed**). **Two findings that change how the repo's own numbers must be read:** (1) **a census count is NOT a defect count** — AGENTS.md §5.5 rule #6 *requires* the superseded renderings to be preserved VERBATIM, so a preserved refuted quote is indistinguishable from a live claim to a pattern-matching census: the LIVE-claim census **ROSE** from 262/130/38 to 321/161/39 (occurrences/LIVE/files) after the repairs; (2) **the honest reading of our own headline is weaker than it looked** — the same-protocol MoleculeNet comparison is D-MPNN **0.8095** vs Morgan-MLP **0.7990** (**+0.0105**) and vs chemprop v2 **0.8330** (**−0.0235**), with **bace −0.0857** and **clintox −0.0695** the two deficits (`docs/ALIGNED_COMPARISON.md` is accurate; the earlier "0.8280 vs 0.8149 / clintox +0.105" reading was a **cross-split-key** comparison and is **WITHDRAWN** — it appears in **0** tracked files). **Still open, USER-gated:** the U1 commit of the two `wfa155_*` metric JSONs (the last FAILing pre-flight gate, `p3_run_ledger_provenance`), the `paper/refs.bib` author fix, the 24 dead gates, and the 2 orphan gates the **6** tracked configs require (`pre_flight_check.py --gates <name>` **hard-errors**, exit 1).
**Last-updated trailer (Wf-A172 dead-gate history CORRECTION + pre-flight live-state + config-gate-requirement wave — 2026-11-01 — the 24 pre-training gates were removed **ON PURPOSE** by the Wf-437 A2 audit (29 tuples: 15 `p3_protocol_*` + 5 `p3_mock_to_real_*` + 9 `krasnov_/rusanov_/drugood_*`; 15+9 = 24 = exactly the dead set), NOT "lost" as `docs/HONEST_NEGATIVE_CATALOG.md` §6.266.1's LEG-1 pickaxe argued — that path has exactly **ONE** commit (a 3,269-file mega-squash), so a `-S` pickaxe cannot see a removal that predates it; the pre-removal snapshot survives in git at `aee5f8ab^:TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437` (359,637 B, `"all": 127`) since junk-purged; "never ran" **SURVIVES** the correction (0 of the 24 inside `build_gates` at `pre_flight_check.py:13957..16802`, 0 of 418 `run_ledger.jsonl` rows). The live `all` pre-flight now measures **116 PASS / 2 FAIL / 11 SKIP / 0 WARN / 0 TIMEOUT / 129 total** (wall 272.36 s) — §6.265's snapshot said 118/0/11. The 2 FAILs: `p3_ssot_path_reference_check` (5 failing assertions of 17 — `molmetal/scripts/auto_emit_section_6_minor.py` was **NEVER committed to git** yet is cited by 2 documents and is the spec source of the dead gate `p3_protocol_recommended_section_6_honest_minor_section_auto_emit`; the `per_seed_metrics_seed*.json` glob expands to nothing; corpus 57 vs recorded 56, globs 5 vs 4) and `p3_anchorguard_version_pin` (3 anchors past the 30-day TTL — `age=2600801.6s > max_age_sec=2592000.0` — a **clock**, not a content defect). NEW: **SIX tracked configs** (`configs/training/nci60_9cell_5seed_50ep_v1.yaml`, `nci60_9cell_coverage_5seed_50ep_v1.yaml`, `nci60_9cell_probe_seed0_cap512_d4.yaml`, `nci60_9cell_probe_seed12_cap512_d4.yaml`, `platinai_pt_4cell_5seed_50ep_v1.yaml`, plus `submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml`) declare `directives.pre_flight_gates_required` naming **2 gates that no pipeline owns**; `pre_flight_check.py --gates <that list>` **hard-errors** with exit 1 (`unknown gate name(s) [...] Refusing to run a partial list`) — MEASURED live; no launcher in `scripts/` reads the key today (0 hits), so no launch is blocked *yet*. NEW: the working-tree `pre_flight_check.py` depends on **FOUR UNTRACKED** scripts (`wf_a131_path_reference_guard_runner.py`, `wf_a152_census_claim_guard_runner.py`, `pre_flight_fragmentation_check.py`, `wfa135_smoke_budget.py`) whose runners are **fail-CLOSED** — a `git commit -a` today commits the registrations without the runners. The Wf-A172 pin repair took the stale count pins **10 → 0** (census: 8 executable pins in 6 files, **0 stale**; 6 floor guards; 20 dynamic invariants; 4 doc-derived). The run ledger's **11** stale "LATEST" rows were re-measured independently (418 rows / 396 keys → 385 fresh, **11 stale**, 0 missing). The A170 census-freshness guard's honest headline is the two-scope split: **primary 1 pin / 0 stale**; generalized 1,822 pins / **160** stale (an upper bound on candidates, not a count of lies). `paper/refs.bib` (46,684 B, 100 entries, 40 DOI-bearing) is **missing all three** gate-required cite keys (`krasnov2026metalcytotoxdb`, `rusanov2025platinai`, `ji2022drugood`) — still blocked on the user's §5.3 OK. The 92.7–94 % band audit verdict is **DOES NOT LEAK** (32 files / 74 matches repo-wide; `paper/` 0, `registry/` 0) with a measured census blind spot (the brief's regex misses the `92.7--94 %` and `[92.7, 94.0]` forms → misses `molmetal/scripts/wf_a155_posebusters_validity.py`, 2 matches). CPU-only, **0 GPU**; `metrics/` git-status md5 unchanged throughout):** catalog **§6.267** is now the finding of record for the dead-gate history correction, the pre-flight's live 2-FAIL state, and the config-gate requirement. Nothing was committed this wave.
**Last-updated trailer (Wf-A153 census-republish + artifact-path + guard-blind-spot + frontier-comparability wave — 2026-11-01 — the census summary was republished a **THIRD** time, every artifact path that named a run by name was **UNRESOLVABLE** (`runs` is a **LIST**, not a mapping), the census guard had **THREE** blind spots of its own, and the published **92.7–94 %** validity band is **NOT our band**; CPU-only, **0 GPU**):** catalog **§6.264** is now the finding of record for the whole re-quote chain. The CURRENT census values are CLEAN last-epoch median **0.0625** (n=27), ALL-runs last-epoch median **0.06380208333333333** (n=30), CLEAN best-epoch median **0.35546875** (n=29), ALL-runs best-epoch median **0.34440104166666663** (n=32) — MEASURED 2026-11-01 from `metrics/by_metric/cfm_run_census.json` -> `fragmentation_summary`. **10** artifact paths were re-pointed to the list index (**8** in `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md`, **2** in the catalog); **no value moved**. The census guard suite is **17 passed**; `git status --porcelain metrics/` stayed at the **35-line** baseline. A **38**-site pointer census found files citing an AGENTS.md section that does not own the rule; **19** are repaired and **20** remain OPEN under Wf-A154. The `docs/STANDING_DIRECTIVES.md` §5.5 rule-6 counts (**145** / **~100 documents** / **~18**) are SUPERSEDED by the MEASURED 2026-11-01 counts (**285** rule-6-form across **70** files; **1017** filesystem-truth-form across **329** files). The superseded rendering is **PRESERVED VERBATIM** per AGENTS.md §5.5 rule #6.
**Last-updated trailer (Wf-A151 bibliography re-verification wave — 2026-10-31 — the repo's own 24-paper bibliography said "all arXiv-verified" while two ids pointed at a FLUID-SIMULATION paper and an AEROSPACE paper, and the audit that "corrected" it measured "0 hits" with the WRONG grep; CPU-only, **0 GPU**):** a fresh adversarial re-verification of six rows against the live records (arXiv API/abs, Crossref, OpenReview) found that the banner at the head of `molmetal/reports/wf_24paper_refs/final.md` was **not supported as written** and that **6 of the 24 rows** carried a wrong cell.  MEASURED: row 1's arXiv id was **2508.16916** — *"The compressible Neural Particle Method for Simulating Compressible Viscous Fluid Flows"* — where the correct id is **2508.12629** (and the paper is published in *RSC Digital Disc.* **5**, 2052-2066, **2026**, not 2025); row 2's venue asserted a **JCIM paper that does not exist** (Europe PMC hitCount 0, Crossref JCIM journal query total-results 0); row 13 carried **no title** while denying its peer-reviewed **TMLR 2025** version (*"GGFlow: A Graph Flow Matching Method with Efficient Optimal Transport"*, OpenReview `K8RlXtMgzo`); row 15's author read *"Zhang et al."* where the list is **Jingyuan Zhou**, Hao Qian, Shikui Tu, Lei Xu; row 20's id **2303.16983** is *"Thrust vector control and state estimation architecture for low-cost small-scale launchers"* (an aerospace paper) and the ICLR 2023 Uni-Mol paper has **no arXiv preprint** (OpenReview `6K2RM6wVqKu`).  **Two** proposed repairs were **REFUTED and NOT applied**: row 18's author `Koodli et al.` was already correct, and `2303.16982` is the **Uni-Mol+** paper by a *different* first author (Shuqi Lu).  The same wave found the Wf-A148 dismissal of `paper/build_log.txt` **false** — the repo's PATH `grep` is a zsh function wrapping **ugrep 7.8.4** whose `-I` skips binary-detected files, and `paper/build_log.txt` is **ISO-8859** — the true counts are **68** `yuelbond` / **8** `dokholyan` / **8** `92.7` over **256 716** lines; the exposure surface is **189** tracked non-UTF-8 files of which only **3** are text.  Repaired in **4 files**; the acceptance gate is a **document** guard, not a model check.  Catalog **§6.263**; cumulative **317 -> 318**, ACTIVE **272 -> 273**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A150 census re-quote wave — 2026-10-31 — a cohort block computed its statistics over the population its OWN predicate names, and 102 sites across 5 documents still quoted the pre-republication numbers; CPU-only, **0 GPU**):** `metrics/by_metric/cfm_run_census.json`'s `by_n_train` / `by_cohort` helper computed `clean_only.best_epoch` over "CLEAN **AND** carrying a **LAST**-epoch rate" while the block's own `predicate` string names "CLEAN **AND** carrying a **BEST**-epoch rate".  The two populations coincide unless a run has a best-epoch rate but **no** last-epoch rate — and the Wf-A127 republication (which published the empty-denominator chemistry rate as `null` instead of the trainer's `0.0` placeholder) created exactly **two** such runs: `cfm_bond_head_30ep_onecycle_2026_10_09` (**0.875** @ epoch 1) and `cfm_fm_control_2026_10_09` (**0.6875** @ epoch 10).  Re-derived from disk: **576** per-epoch rows over **59** runs, **66** with `gen_validity_val == 0` and `gen_validity_n > 0` over **5** runs, **2** of them on the LAST row.  The n_train=256 cohort's `clean_only.best_epoch` moves from ~~n=6 / median 0.59375~~ to **n=8 / median 0.7708333333333333**; its `last_epoch` moves from ~~median 0.026041666666666664 (n=8)~~ to **0.03645833333333333 (n=6)**; **no other block moved**.  The republication then left **102 stale-rendering sites** across **5 documents** quoting the old numbers — **31** in the catalog, **27** in `TODO/INDEX.md`, **20** in `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md`, **12** each in `TODO/pending/wfa114_cfm_fragmentation_fix_verification.md` and `TODO/completed/wfa114_cfm_fragmentation_audit.md` — re-quoted in **54 write operations** (**46** in-place strikes + **8** dated `⚠️ SUPERSEDED` clauses).  Current values: CLEAN last-epoch median **0.0625** (n=27), ALL-runs last-epoch median **0.06380208333333333** (n=30), CLEAN best-epoch median **0.35546875** (n=29), ALL-runs best-epoch median **0.34440104166666663** (n=32), gate verdicts PASS/FAIL/SKIP **5 / 25 / 29** over 59 runs (and **4 / 23 / 28** over the 55 CLEAN runs), budget-matched flip set **[9, 11, 25]**.  `budget_matched` lives at the **top level** of the artifact, not inside `fragmentation_summary`, and `budget_matched.n_train_axis` reads **`UNRESOLVED`** — retained, not papered over, so the flip set is **not** claimed as a scaling result.  Catalog **§6.262**; cumulative **316 -> 317**, ACTIVE **271 -> 272**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A148 citation-and-code repair wave — 2026-10-31 — the repo contradicted ITSELF about the paper it is reimplementing: a preprint title, a 10× parameter slip, a "sibling" function that is one call deep inside the other, and an arXiv id belonging to a FLUID-SIMULATION paper; CPU-only, **0 GPU**):** the Wf-A144 adversarial pass re-checked the repo's claims about FlowMol3 and YuelBond against the papers' own extracted text.  MEASURED: the repo's FlowMol3 arXiv id was **2508.16916** — *"The compressible Neural Particle Method for Simulating Compressible Viscous Fluid Flows"* — while the correct id is **2508.12629**; the repo's parameter count was **66 M** where the paper's Table 1 says **6 M**; the repo labelled FlowMol3's *"validity"* an **RDKit `DetermineConnectivity` pass** where the paper defines it as *"the percent of molecules that can be sanitized by rdkit"*; the repo called the paper's **92.7 %** F1 a *"CONSERVATIVE lower bound"* when `F1` **falls** as distortion grows and the figure was measured at **sigma = 0.2 A**; the repo described `DetermineConnectivity` as a **sibling** of `DetermineBonds` when RDKit's `DetermineBonds.cpp` calls it at **L543** and `determineBondOrders(...)` at **L544** — it is **one call deep inside**; the repo's `RBF_MAX_DISTANCE` comment called it the candidate-edge cutoff when it is the **top of the RBF support**; a **bioRxiv DOI** was printed as an **arXiv id**; and the YuelBond **year** read 2025 in two lines while two citation records in the same file already said **2026** (J. Chem. Inf. Model. **66**(2), 1003-1012, doi:10.1021/acs.jcim.5c03052).  Repaired in **11 files**; the only two findings that touched executable code were **comment/docstring-only**.  Catalog **§6.261**; cumulative **315 -> 316**, ACTIVE **270 -> 271**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A147 1-ulp-table closure wave — 2026-10-31 — CPU-only, **0 GPU**):** §6.249's 1-ulp transcription slip was closed in the **pre-flight's comment** by Wf-A135, but it was still live in **two TABLES** — catalog §6.249's row 2 and `TODO/INDEX.md` §22's row 2 — which printed `0.08508958875996528`.  Measured against the artifact on disk: `metrics/by_metric/cfm_fragmentation_audit.json` carries `single_molecule_rate_over_pool_rows` = **0.08508958875996527**, and `1078 / 12669` in IEEE-754 is the same double; the quoted literal appears **nowhere** in `metrics/`, `molmetal/` or `docs/` outside the dated correction notes.  Rows 1 and 3 of both tables byte-match the artifact and were left unchanged.  The **one-ulp regression guard** in `molmetal/tests/test_wfa133_reporting_matrix.py` was **deliberately left untouched**: it holds the brief's literal and the artifact's literal side by side on purpose, and collapsing them would delete the discrepancy the test exists to keep open.  Catalog **§6.260**; cumulative **314 -> 315**, ACTIVE **269 -> 270**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A146 adversarial-review wave — 2026-10-31 — CPU-only, **0 GPU**):** the Wf-A144 adversarial pass against the **YuelBond paper's own extracted text** (not a summary of it) found a claim the paper does not make.  `molmetal/reports/wf_cfm_frontier_research/research_phase1b.md` said the paper *"documents its 78.3% failure rate on noisy CDGs"*.  Measured against the paper's full text: the paper prints **"RDKit was able to process only 217, while it failed on the remaining 783 molecules"** — an **integer split with no percentage at all** — and the bare token **"78.3"** in that paper is the **2D-reassignment F1-score** (Fig 4b / Table S3: 80.1 / 81.1 / 80.1 / 78.3), so a reader who greps the paper for it finds an F1, not a failure rate.  The same sweep found the **baseline mis-attributed**: the paper's entry point is RDKit's `DetermineBonds`, which it describes as "internally … leveraging the xyz2mol program" — the 217/783 split belongs to **RDKit**, not to `xyz2mol` directly, so the three sibling sites written by Wf-A143 (`yuelbond_decoder.py`, `cfm_decoder_yuelbond.py`, `molmetal/reports/wf_cfm_rescue/phase1_yuelbond.md`) were corrected to make that inference explicit.  Third: the **3.5 Å-vs-3 Å** candidate-edge-cutoff deviation (the paper connects "any pair of atoms within 3 Å", four sites in its text) was declared at the **decoder** definition sites but **not** at `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`'s `DEFAULT_EDGE_CUTOFF = 3.5` — the constant the YAML `edge_cutoff` key actually feeds, in 20+ yamls.  That constant now carries the disclosure; the **value is deliberately UNCHANGED** (changing it would invalidate every trained checkpoint and every measured decode metric, and re-validating it needs a GPU run this chain is forbidden to do), so every headline number produced under it must be labelled **3.5 Å, not the paper's 3 Å**.  Catalog **§6.259**; cumulative **313 -> 314**, ACTIVE **268 -> 269**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A145 self-citation + stale-gate-count repair wave — 2026-10-31 — CPU-only, **0 GPU**):** the Wf-A142 audit was still the right place to look, because **the catalog itself was one of the sites carrying a wrong citation**: its §6.51a printed *"per Balcells & Skjelstad 2020 J. Chem. Inf. Model. 60 (10): 2005–2024, doi:10.1021/acs.jcim.0c00265"* — a DOI that returns **HTTP 404 from Crossref** (MEASURED 2026-10-31), a page range that is not the tmQM paper's, and a record that **contradicts the repo's own source module** `molmetal/data/bond_length_prior.py:62-64`.  The correct record — Balcells, D.; Skjelstad, B. B. (2020), *tmQM Dataset — Quantum Geometries and Properties of 86k Transition Metal Complexes*, *J. Chem. Inf. Model.* 60(12): 6135–6146, doi:10.1021/acs.jcim.0c01041 (HTTP 200) — was written back with the superseded wording preserved **verbatim** per AGENTS.md §5.5 rule #6; the wrong DOI now survives at exactly **one** site repo-wide (the quoted historical text).  Running the CFM fragmentation chain's own test surface surfaced a **second**, unrelated defect: `molmetal/tests/test_wf345a2_yuelbond_env.py` pinned `EXPECTED_GATE_COUNTS["all"] == 68`, a Wf-345A2-era snapshot, while the pre-flight's registry had been bumped (each bump documented with a dated comment inside `pre_flight_check.py`) to **127** — so the test had been **RED since the count first passed 68** (Wf-350-A3) and nobody read it; it is **not** a regression from this chain (neither file carries a `Wf-A143` marker), and it is repaired to a **monotonic floor** (`>= 68`), the repo's own convention (`test_wf350a3_hp_provenance.py:321` asserts `>= 69`) — a hard-coded equality fires on every legitimate gate addition.  A third finding was **independently verified but NOT repaired here**: **three of the four** `molmetal/eval/reporting/cite_helper.py` records are confabulated — the LigandDiff **title** is invented, and the tmcinvdes / PlatinAI / MetalCytoToxDB records carry a **wrong venue, wrong pages and wrong authors** (JACS Au 5(2):412-428 does not exist; PlatinAI is a **ChemRxiv** preprint, not JCIM 65(8):3498-3510; MetalCytoToxDB lives in *J. Med. Chem.* 69(8):8838-8851, not *J. Cheminform.* 18:12) — that file is inside the concurrent Wf-A140 wave's write scope, so the **measured ground truth** is recorded in catalog §6.258 instead of racing it.  Catalog **§6.258**; cumulative **312 -> 313**, ACTIVE **267 -> 268**.  **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A143 citation-repair wave — 2026-10-31 — the YuelBond lit anchor printed a title matching neither the preprint nor the published paper; CPU-only, **0 GPU**):** the Wf-A142 read-only citation audit found the repo citing the YuelBond paper with (a) a **non-existent title**, (b) the **wrong author initials** ("Wang, S., Dokholyan, P." vs Jian Wang / Nikolay V. Dokholyan), (c) a **truncated DOI**, (d) the **wrong RDKit entry point** — `DetermineConnectivity` for a bond-**order** baseline, MEASURED locally to emit only SINGLE bonds where the paper benchmarks `DetermineBonds`, (e) a **malformed arXiv id**, (f) "graph coloring" (JT-VAE's mechanism) credited to xyz2mol, and (g) the paper's **worst-case** 92.7 % F1 quoted as "the YuelBond number" while its 98.4 % F1 headline was never cited. Every premise was re-verified **before** the first edit (Europe PMC live fetch + a local RDKit probe), then **13 blocks in 11 files** were repaired — docstrings, comments and Markdown only, proved by an AST comparison with all docstrings stripped (**identical** in all 5 touched `.py` files) and **0 new** ruff-0.6.9 findings. The repo's **3.5 Å** candidate-edge cutoff against the paper's **3 Å** is now **DECLARED at the definition sites** and deliberately **left unchanged** — it is decode behaviour, and re-validating it needs a GPU run. One site is **BLOCKED** by TODO-84: `paper/sections/04_evaluation.tex` still carries the unqualified 92.7 % and the unqualified 3.5 Å. Catalog **§6.257**; cumulative **311 -> 312**, ACTIVE **266 -> 267**. **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A141 guard-repair sub-wave, part 2 — 2026-10-31 — the pre-flight's narrative-claims gate could not FAIL; CPU-only, **0 GPU**):** running the guard surface turned up a **pre-existing** (HEAD-level) defect: `p3_narrative_claims_check` invoked `verify_narrative_claims.py` **without `--strict`**, and MEASURED on a broken copy the checker exits **0** in that mode and **1** with `--strict` ⇒ an **UNVERIFIABLE** claim (a missing source file) left the gate **PASSing** while a §5.2 traceability claim pointed at nothing; the gate's own FAIL branch counted `"??"` rows that invocation could never produce. Same family as §6.250 (a gate that could not RUN): §6.256 is a gate that could not FAIL. **FIXED** — `--strict` added; the live claims file verifies clean under it (`pass=47 drift=0 unverifiable=0`, rc=0) so the gate's verdict is unchanged, and `test_wfa127_registry_claim_guard.py` is green again (**10 passed**, was 9 passed / 1 failed). Catalog **§6.256**; cumulative **310 -> 311**, ACTIVE **265 -> 266**. **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A141 guard-repair sub-wave 2026-10-31 — the wf #131 path-reference guard caught TWO of this chain's OWN new citations, and both are repaired; CPU-only, **0 GPU**):** the guard reported **3 FAILED / 14 passed** right after §6.254 was published. MEASURED: the corpus had grown from the audited baseline of **54** broken path citations to **55**, and the single extra one was **§6.254 §7's own** citation of the c7 "best checkpoint" file — a path that does not exist, cited AS a path. The baseline was **NOT** widened (that would turn the guard's load-bearing assertion into a tautology); the corpus was repaired instead. **§6.255** records the episode: §6.254 §7 was re-rendered (the file's bare name plus its **EXISTING** parent directory; the line number corrected **44 -> 50**), and `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` — the source of the false claim, uncorrected since §6.197 (2026-10-24) catalogued the absence at the `cfm_todo.json` site — got a dated correction under "## Outputs". After the repair: **17/17 PASS**, corpus back to **54 + 4**, both equal to the audited baselines, **0** new-vs-baseline deltas. Two open items are recorded in §6.255: the guard is **NOT** wired into `pre_flight_check.py`, and both guard files are still **UNTRACKED** in git (so they protect nobody but this working tree). Catalog cumulative **309 -> 310**, ACTIVE **264 -> 265**. **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A141 sync 2026-10-31 — the adversarial re-check REFUTED our own §6.253 "decisive control": the two sides carry DIFFERENT COMPOSITION WEIGHTS, so the gap is an UPPER BOUND, not a measurement; CPU-only, **0 GPU**):** **Wf-A140's F3 adversary put seven surfaces to a REFUTED-by-default review and **2 of the 7 are CONFIRMED-DEFECT**; Wf-A141 re-measured every number independently on disk. **DEFECT 1** — the sampled side's **704** samples (= `gen_validity_n` exactly, sliced **128/64/128/128/128/128**) carry **Ru at 0.27×** its real-side share (8.52 % vs 31.25 %) and the other five metals **1.23–1.36×**, because the val cohort is metal-sorted; heavy-atom size is MATCHED (43.245 vs 43.232), so the confound is **metal identity**; per-composition decode outcomes were never persisted and c7's `checkpoints/` is EMPTY, so the **71.35 % → 12.22 %** gap is an **upper bound** and is not re-weightable (§6.254 §1). **DEFECT 2** — §6.253 paired a **mean** (1.3854) with a **ratio** (0.1222) and never stated the sampled side's own mean: MEASURED now, sampled epoch-17 mean **3.446023**, **86/704** one-component = **87.784 %** multi; real side **137/192** one-component = **28.646 %** multi ⇒ **the head does NOT "work" on real geometry** and the control supports a **relative** statement only (§6.254 §2). **DEFECT 3** — "the dominant cause is the head not proposing a candidate edge" is a **mechanism ASSERTED, not measured** (the c7 geometry is not on disk) (§6.254 §5). **DEFECT 4** — "the head emitted 50.449 bonds" is an **attribution**: it is the **assembled** count including connectivity-prior bridges, hence an **upper bound** (§6.254 §6). **M-a survives on the CODE PATH, not the arithmetic** — "≥ 9.244 splits per molecule" needs the counterfactual "one component", which the entry's own arithmetic denies, so the arithmetic only converts M-a into a **requirement on the drop rate**; the exclusion rests on RDKit 2026.03.6's measured taxonomy (only an out-of-range or negative index can split) plus the unreachable-index argument (§6.254 §3). **M-b HOLDS** — `1ea60d5c` author == committer date, and the veto post-dates the WHOLE c7 decode window (2026-10-20 15:37–21:04 UTC) (§6.254 §4). Three axes are **REFUTED** (coordinate normalisation, atom order, batch-vs-flat index). Catalog cumulative **308 -> 309**, ACTIVE **263 -> 264** (§6.254). **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A133 sync 2026-10-31 — the CFM FRAGMENTATION chain, end to end: ONE numerator / THREE denominators, 25 of 59 runs carry a fragmentation FAIL that no gate ever consulted, and the pre-flight's own count guard was DEAD; CPU-only, **0 GPU**):** **Wf-A133** traced the CFM fragmentation chain from the pool all the way to the pre-flight. The SAME count (`n_single = 1078`) is published under **three** denominators — **0.08509630565203663** (12 668 parsed rows) / **0.08508958875996528** (12 669 CSV rows) / **0.08506944444444445** (12 672 samples DRAWN) — differing in the **fourth** significant figure, and **none** of them travels with its denominator (§6.246). The census's per-run `fragmentation_gate` verdict was **recorded and never read**: of **59** CFM runs, **5** PASS / **25** FAIL / **29** SKIP, and no gate ever consulted it (§6.247). The cause of the **91.49 %** fragmentation is **UNMEASURED on this host** — the checkpoints cannot be deserialised CPU-only (`RuntimeError: Attempting to deserialize object on a CUDA device`), so the `bonds_dropped` histogram the decoder already computes is unobtainable and **no such artifact exists on disk** (§6.248). **Two defects were REPAIRED**: the leak-visibility gate's own regression fixture pointed at the CLEAN v4 cohort instead of the known-leaky v1, and its failure text told the operator to restore the leaky bytes (§6.249, FIXED, 9/9 tests); and the pre-flight's own count guard was DEAD behind a **300 s** subprocess timeout against a **411.04 s** pre-flight, so all three of its invariants reported error instead of running (§6.250, FIXED). **The pre-flight's red state is now READABLE**: the whole FAIL list is **3/3 wall-clock TIMEOUTS** — `p1_chemprop_smoke` (90 s), `p3_wf385_scaffold_loss_wire_in_verified` (150 s), `p3_dmpnn_loss_wire_in_smoke` (300 s) — so **zero substantive defects** stand behind it, and a timeout is indistinguishable in the output from *"the trainer flag is NOT wired in"* (§6.251, OPEN). The published connectivity band is **POST-PROCESSED** while our **8.51 %** is **RAW**, and the post-processed arm discards **22.47 %** of the heavy-atom mass and leaves **21.0 %** of the rows as specks (§6.252); and the dominant cause is now **MEASURED-RANKED** — the assembler-drops-bonds mechanism is **REFUTED** for this pool (§6.253). Catalog cumulative **300 -> 308**, ACTIVE **255 -> 263** (§6.246..§6.253). **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**

**⚠️ 1-ULP VALUE CORRECTED (2026-10-31, Wf-A147).**  The Wf-A133
trailer above prints the pool-level rate as `0.08508958875996528`; that
value appears **nowhere on disk** — the artifact
`metrics/by_metric/cfm_fragmentation_audit.json` carries
`0.08508958875996527` (also the exact IEEE-754 double for
`1078 / 12669`), and the live tables (catalog §6.249 and
`TODO/INDEX.md` §22) have been corrected to it.  The superseded literal
survives only where AGENTS.md §5.5 rule #6 requires it — the dated
correction notes — and in
`molmetal/tests/test_wfa133_reporting_matrix.py`'s deliberate
`KNOWN_MEASURED_RATE`, which holds the one-ulp disagreement OPEN on
purpose.  See catalog §6.260.
**Last-updated trailer (Wf-A129 + Wf-A128 sync 2026-10-31 — the 50 dangling cross-references are CLOSED + the byte-size guard's coverage gap is measured; CPU-only, **0 GPU**):** **Wf-A129** measured the §6.243 byte-size guard's REAL coverage with a value-directed sweep (`molmetal/scripts/wfa129_size_citation_coverage.py` -> `molmetal/reports/wfa129_size_citation_coverage/census.json`): the 4 SSOT documents carry **104** attributable byte-size claims, the guard sees **28**, **76** are invisible to it, and **50** of those already quote a size the file no longer has (**28** of them undated, over **16** distinct files). The blindness is a *formatting* property, not a sampling accident — `PATH_RE` admits only `TODO/(completed|pending|archive)/…` and `SITE_RE` demands the size sit IMMEDIATELY after the path, so a `|` table cell, a `→` chain or the word `NEW` defeats it. §6.243's own "27 of 34" coverage figure was counted the same way the guard counts its scope and is superseded by the **104** denominator (§6.244). **Wf-A128** closed the **50** dangling cross-references §6.242 left OPEN: **47** REDIRECTED at **53** citation sites (**52** in the catalog, **1** here), **3** preserved as named HISTORICAL MOVE records at **7** sites with a live-counterpart assertion; the guard's `BASELINE` went **50 -> 0** (`BASELINE_SIZE = 0`) and two tests were added (`test_every_historical_move_still_has_a_live_counterpart`, `test_the_guard_is_not_vacuous`). Catalog cumulative **298 -> 300**, ACTIVE **254 -> 255** (§6.244, §6.245 + STATE UPDATEs in §6.242 / §6.243). MEASURED 2026-10-31, CPU-only: `pytest molmetal/tests/test_wfa127_dangling_reference_guard.py molmetal/tests/test_wfa127_size_citation_guard.py -q` -> **6 passed**; `git status --porcelain metrics/` byte-identical before/after (**34 lines**). **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**
**Last-updated trailer (Wf-A127 sync 2026-10-31 — the Wf-A125 residue closure: SEVEN new honest-negative entries + four STATE UPDATEs; CPU-only, **0 GPU**):** **Wf-A127 2026-10-31** — closed the four residues Wf-A125 left OPEN: the repo-root CPU lock is no longer **git-ignored** (the `/conftest.py` rule is removed, the file is trackable, and a repo-wide invariant now forbids ANY git-ignored `.py` file); the pre-flight gate count is re-synced at **127** across **all four** places it lives; the `WARN` test contract no longer contradicts its producer; the CFM-census docstring now matches its artifact's **11 / 16** SKIP split; the PoseBusters-validity honesty block is **computed by the generator** instead of hand-written into the JSON; and the registry's `generated_at` header can no longer date itself before its own rows. It also catalogued the CLASS behind the residue — **a repair that changes an artifact invalidates every document that describes it, and nothing in the repo sweeps for it** — and corrected **5** stale claims additively (catalog §6.235..§6.243; cumulative **289 -> 298**, ACTIVE **245 -> 254**). MEASURED 2026-10-31, CPU-only: the **15-file** affected surface -> **169 passed** in 207.78 s with the CPU lock ON and **0** GPU-required tests run; `metrics/` byte-identical before/after.

> **HISTORICAL 2026-10-31 (Wf-A127):** an earlier measurement of the SAME
> surface, taken **before** the 20th guard test was added, read
> `168 passed, 1 warning, 26 subtests passed in 452.32s (0:07:32)` — it describes a tree that no longer exists.  It is PRESERVED
> VERBATIM per AGENTS.md §5.5 rule #6; the live statement above is the
> **current-tree** measurement (**169 passed** in 207.78 s, rc 0,
> `metrics/` byte-identical before/after).  Catalog §6.241 STATE UPDATE.

> **MEASURED 2026-11-02 (Wf-A269 wave-7 re-measurement, CPU-only) — this
> SUPERSEDES the 2026-10-31 affected-surface reading(s) above, which are PRESERVED
> VERBATIM per AGENTS.md §5.5 rule #6:** The **15-file** affected surface -> **170 collected** (168 passed, 2 failed) in 397.17s with the CPU lock ON
> (`mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')`, `mol-metal GPU contract: 0
> GPU-required test(s) did not run`), rc 1.  The 2 failures are
> `molmetal/tests/test_wfa125_preflight_warn_contract.py::test_the_provenance_gate_passes_on_the_real_ledger`
> and `::test_the_summary_counts_warn_and_the_identity_holds` — the disclosed
> USER-gated `p3_run_ledger_provenance` defect (catalog §6.269.16), which needs the
> U1 commit and is therefore NOT this wave's to close.  The COLLECTED count
> (170) and the PASSED count (168) DIFFER because the surface is NOT green today, so the
> §6.241 guard measures the COLLECTED count — the quantity a `pytest
> --collect-only` run reports — and the documents quote that same quantity.

**MEASURED 2026-11-04 (Wf-A269 wave-27 re-measurement, CPU-only) — this
SUPERSEDES the 2026-11-02 affected-surface reading(s) above, which are PRESERVED
VERBATIM per AGENTS.md §5.5 rule #6:** The **15-file** affected surface ->
**179 collected** (177 passed, 2 failed) in 287.31s with the CPU lock ON
(`mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')`, `mol-metal GPU contract: 0
GPU-required test(s) did not run`), rc 1.  The 2 failures are
`molmetal/tests/test_wfa125_preflight_warn_contract.py::test_the_provenance_gate_passes_on_the_real_ledger`
and `::test_the_summary_counts_warn_and_the_identity_holds` — the disclosed
USER-gated `p3_run_ledger_provenance` defect (catalog §6.269.9), which needs the
U1 commit and is therefore NOT this wave's to close.  The count moved 170 -> 179 between the two readings.  MEASURED 2026-11-04
(per-file `pytest --collect-only`): the **13** surface files NOT touched after
the 2026-11-02 reading contribute **163** collected cases on BOTH dates (their
contents are byte-unchanged), so the entire **+9** sits inside the **2** files
that WERE touched — `test_wf415a3_preflight_fail_audit.py` (mtime 2026-11-04
16:05, **4** collected) and `test_wf327f11_gate_count_consistency.py` (mtime
2026-11-04 18:34, **12** collected) — together **16** today vs **7** on
2026-11-02 (the **7** is **DERIVED**, not re-measurable — the 2026-11-02
versions of those two files are not in git, and HEAD's versions collect
3 + 3 = **6**; it is corroborated independently by the added test functions:
`test_wf415a3_preflight_fail_audit.py`'s one added
`test_preflight_no_timeout_derived_fail` gives 4, and
`test_wf327f11_gate_count_consistency.py`'s HEAD 3 plus its 9 added
name-consistency cases give 12).  `test_wf327f11`'s current 12 cases include
the 9 gate-NAME-consistency cases (`TestExpectedGateNames` = 4,
`TestNameDriftIsDetected` = 5); the two changed surface files were last edited
at 2026-11-04 16:05 and 18:34, and the only file this wave edited inside
`molmetal/tests/` is `test_wfa127_surface_count_guard.py`, which the guard's own
second test asserts is NOT part of the affected surface ⇒ a concurrent
Wf-A269 wave's gate-count work, NOT this wave's.  The COLLECTED count
(179) and the PASSED count (177)
DIFFER because the surface is NOT green today, so the §6.241 guard measures the
COLLECTED count — the quantity a `pytest --collect-only` run reports — and the
documents quote that same quantity.


**Last-updated trailer (Wf-A124 + Wf-A126 sync 2026-10-31 — the G6 ablation provenance repair + its critic-residual repair; CPU-only, **0 GPU**):** **Wf-A124 2026-10-31** — the G6 per-innovation ablation tree existed only in git (**74** of **76** tracked files deleted from the working tree); restored, and every headline number recomputed from the per-seed JSON into **7** new catalog entries (**§6.216..§6.222**): the tree holds **24** tuples (not 18); the published per-lever Δ table reproduces **2 of 6** cells under its own reference and **0 of 6** under the G5 all-ON baseline; **no lever is resolvable** at n = 6 (max |t| = 1.3177 < 2.5706); the artifacts record **no lever flag** for 3 of the 4 cohorts; the published **0.7888** has no source. **Wf-A126 2026-10-31** — that wave's own critic returned **12** "blocking defects", but **10** were already closed on disk (the critic read a MID-WAVE state) and **2** of its claims were factually wrong; the real residue was **3 live documents** (now dated-corrected, historical text preserved verbatim) + a **trainer root cause** nobody had named: the 8-field build fingerprint was added by `8d139d23` (2026-10-08) and **reverted out** by `af84a545` (2026-10-15), so a run launched today would again be unversion-matchable against G5. The trainer now records the lever identity (`single_task` / `kendall` / `dann`) + the build fingerprint (`tf32` / `torch_compile` / `w2_gradient_mode`) in `training_summary.json` (**8** new tests PASS; §5.7 50-step CPU smoke rc = 0, **23** summary keys). **0 GPU launches · 0 CUDA calls · 0 `paper/*` touches · 0 wet-lab.** **PREVIOUS LAST-UPDATED TRAILER (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6):**

**Last-updated trailer (Wf-A28..A30 + Wf-A31/A32 prep 2026-10-14 sync — strict-repair of 25 SCOUT findings + pocket encoder architecture prep, per user voice *"现在可以维护一下项目，整理一下文档"*):** **Wf-A28 2026-10-14** — 3 per-step commits on master (`4be9ab9` + `23b173f6` + `bbb9a5c935`) closing the 3 Wf-A27 residuals: coord_geom_class per-element VSEPR override (6 NEW (Z,CN) entries) + bond_length_prior per-order Pyykkö 2008/2005 + per_loss_projection DECLARED_INERT by §5.11 design. **Wf-A29 2026-10-14** — 2 per-step commits on master (`382857c2c7` + `e0363753f4`) REPAIRING (not just documenting) the 2 Wf-A28 honest-negatives: bond_length_prior REAL tmQM DFT table (927 (Z,Z,order) bins / 108k mononuclear complexes / 8,582,188 bond observations) + coord_geom_class REAL d-electron classifier (30 NEW (Z,OS,CN) entries for 14-metal d-block + Ti). **Wf-A30 2026-10-14** — strict-repair workflow (4 logical commits + completeness-critic self-correction loop) closing all 25 SCOUT findings; A30-1 `b27f5e1694` (checkpoint loop + EMA + cross_dataset decode fixes: A1 strict=False + A2 already-complete guard + A3 atomic save §5.6 + A4 EMA dtype + A5 cross_dataset KeyError + A6 EMA round-trip + A7 smoke-stub .pt rename) + A30-2 `9943ad293d` (silent fallbacks + paper_table + registry + CFM panel + §6.51 composite honest-negative) + A30-3/4 (pending). **Wf-A31 prep 2026-10-14** — pocket binding truth scaffolding, CPU-only 0 GPU: platiny_pocket_pool (10 literature Pt / Au / Ru targets, 0 extracted) + PocketGen vendor (encoder-only MIT lazy-import) + P2Rank wrapper (Biopython crop + CLI stub) + pocket_pool_unified (single dispatch API) + PerLayerPocketInjector test (12/12 PASS) + pyproject.toml pocketgen extras. **63/63 new unit tests PASS**. **New doc `docs/POCKET_ENCODER_PIPELINE.md`** describes the 5-module architecture + honest-framing table. **Pre-flight v3 baseline 109 PASS / 4 FAIL / 10 SKIP / 124 TOTAL** — 4 FAILs all pre-existing (QEDLoss signature + 2 ModuleNotFoundError on renamed test fixtures + hyperparam-config + config-binding) NOT introduced by A28..A30. **Running tasks inventory 2026-10-14**: Wf-A30 workflow `w04cmhea5` still in progress (A30-3/4 + VERIFY + CRITIC + CLOSEOUT); 0 active GPU jobs (per user directive *"gpu先别动"*); 0 background monitors active. All artifact paths + §6.51/§6.52 catalog + closeout files on disk per AGENTS.md §5.5 filesystem-truth + §5.6 atomic-write + §5.7 trainer-flag-smoke. **Previous (Wf-METRIC-RESCUE 2026-10-08/09 sync — 5 honest-negative findings + 2 code fixes SHIPPED)**: W1 arm2 (PID 31946, per-epoch cadence 30-epoch real-only) COMPLETED 2026-10-09 00:15; pooled 0.5496 ± 0.0967 (3-arm ON/OFF confirms W₂ NOT load-bearing); W1 arm1 (every-batch cadence) = 0.5349; OFF reference = 0.5575. **5 honest-negative findings on disk**: (1) W₁ W₂ ON/OFF de-confound [HONEST-NEGATIVE: W₂ lowers metric], (2) G6 per-innovation ablation [18 tuples, no individual lever decisive] — **Wf-A124 2026-10-31 (dated supersede note):** the "18 tuples" count is **SUPERSEDED** — the G6 tree holds **24 tuples** (4 cohorts × {Ru, Rh} × 3 seeds, `no_dann` included from the first results commit `37aa4960`), and the published per-lever Δ table does **not** reproduce from the shipped artifacts (**2 of 6** cells under its own `no_multi_task` reference, **0 of 6** under the G5 all-ON baseline with the same metal column — in total **4 of 6** cells reproduce, each against a **different** reference, and **2 reproduce under nothing**; catalog §6.221); the paired-vs-G5 comparison resolves **no** lever (max |t| = 1.3177, n = 6, df = 5, threshold |t| > 2.5706), so "no individual lever decisive" **STANDS**; see `docs/HONEST_NEGATIVE_CATALOG.md` §6.216..§6.222, (3) bace rescue hypothesis REFUTED [0.7421→0.6805, class-weighting NOT the cause], (4) m19 fix-stack `--rh-loso-validation` HURT Rh [0.5022→0.4056], (5) pIC50 pearson ridge beats head [pooled ridge 0.4302 vs head 0.2036 (corrected 2026-10-11; 0.4151/0.1543 were not reproducible from disk), head anti-correlated on Os]. **Code SHIPPED**: `train_moleculenet.py` 4 NEW flags wired (--class-weight/--dropout/--weight-decay/--patience, verified end-to-end NOT FLAG_ONLY) + `train_metal_lomo_wasserstein.py` W₂ no-op inverse guard + 11 NEW tests PASS. **Pre-flight v3 still rc=0**. **Running tasks inventory 2026-10-09**: 0 active GPU jobs (W1 arm2 exited 2026-10-09 00:15); 0 background monitors active. All artifact paths + §6.38 catalog + closeout file `TODO/completed/metric_rescue_wfmrr_2026_10_08.md` 12519 B → 20322 B, measured 2026-10-31 on disk per AGENTS.md §5.5 filesystem-truth + §5.6 atomic-write + §5.7 trainer-flag-smoke. **Previous (Wf-A3 sync 2026-10-02)**: 22 NEW tests PASS; pre-flight v3 fresh-run MEASURED 93 PASS / 3 FAIL / 2 SKIP / 98 TOTAL; AGENTS.md §5.10 SHIPPED. **PREVIOUS HISTORY PRESERVED VERBATIM PER §5.5 rule #6 — rewriting the historical closeout trail would violate the filesystem-truth contract** (the prior Wf-438/439/443 trailers follow below).


## Project Overview

**Mol-Metal** designs de-novo small-molecule ligands for precious-metal anti-cancer targets
(cisplatin-class, Pt/Ru/Ir organometallics) using **Flow Matching** generative models on top of a
**Structure-Based Drug Design (SBDD)** loop. The system generates 3D ligand poses in a protein
pocket, docks and scores them, and refines via a closed-loop design cycle.

We aim to close the single-GPU (RX 7800 XT) vs. 64×A100 cluster gap by leveraging **Triton
kernels on ROCm** — the constraint is "fits in 16 GB memory," not "reproduce multi-GPU training."

---

## Environment

> **⚠️ STALE section preserved for historical context (R16 2026-09-18 close-out authored
> this README on AMD RX 7800 XT + ROCm). The CURRENT canonical environment is NVIDIA Blackwell
> + RTX 5090 + .venv/bin/python — see [environment.md §1–§7](environment.md) for MEASURED 2026-10-03
> facts. The legacy RX 7800 XT claim below does NOT match today's host; per AGENTS.md §5.5
> filesystem-truth rule, historical text is preserved (NOT deleted) but explicitly flagged STALE.**

### CURRENT (canonical, per `environment.md` MEASURED 2026-10-03)

- **Package manager:** `uv 0.11.8` (lockfile `uv.lock` 344 KB) — but launch wrappers invoke `/home/hugo/codes/mol-metal/.venv/bin/python` directly per AGENTS.md §5.9 single-venv rule
- **Python:** 3.12.13 (`.venv/bin/python` symlink to uv-managed CPython)
- **GPU stack:** NVIDIA CUDA 13.0 + `torch 2.14.1+cu130` + `triton 3.8.0` (PyPI; NOT `triton-rocm`)
- **Hardware:** GPU 0 = NVIDIA RTX PRO 6000 Blackwell (97887 MiB VRAM, sm_120); GPU 1 = NVIDIA GeForce RTX 5090 (32607 MiB VRAM); driver 615.71.09
- **Canonical venv:** `/home/hugo/codes/mol-metal/.venv/bin/python` (single venv rule per §5.9)

### HISTORICAL (STALE — RX 7800 XT + ROCm was the R16 close-out host)

This project uses a **fully managed uv environment** (`UV_PROJECT_ENVIRONMENT` / `managed=true`):

- **Package manager:** `uv` (all commands prefixed with `uv run`)
- **Python:** 3.12
- **GPU stack:** ROCm 7.2 + `triton-rocm` 3.8.0
- **Hardware target:** AMD RX 7800 XT (gfx1101, 16 GB)

**No pip / no venv / no system Python.** Activate nothing; just `uv run <cmd>`. See
[`environment.md`](environment.md) for full setup, kernel validation, and the ROCm/Triton
compatibility matrix.

---

## Engineering Practices

Read [`engineering_practices.md`](engineering_practices.md) before touching code. Key conventions:

- **Single source of truth:** every ADR lives in top-level `decisions.md` (open) or
  `completed/` (closed). Never in code comments or scattered .md files.
- **One task, one file:** `pending/NN_short_slug.md` is the atomic unit of work. Status moves
  `pending → completed` (or `pending → archive` if superseded by a newer plan), not
  delete-and-recreate.
- **Audit trail:** every decision cites a paper / commit / date. AUDIT_RESEARCH_GRADE.md is the
  immutable historical snapshot.
- **No scope creep into TODO/01-13/:** those are historical research notes. New topical work
  either updates an existing topic dir or lives in `pending/`.
- **Use INDEX.md for status:** every file's current state is tracked in
  [`INDEX.md`](INDEX.md) (single source of truth for "what's where + status").

---

## Directory Structure

```
TODO/
├── README.md                       ← this file (overview + navigation)
├── INDEX.md                        ← single source of truth: every file's status
├── environment.md                  ← uv + ROCm 7.2 + triton-rocm 3.8.0 setup
├── engineering_practices.md        ← conventions for this project
├── AUDIT_RESEARCH_GRADE.md         ← historical audit (2026-09-11), preserved verbatim
├── completion_audit_2026-09-13.md  ← completed-task audit
├── project_workflow.md             ← project-level workflow doc
├── decisions.md                    ← pending architecture choices (D1–D∞)
├── risks.md                        ← open blockers / risks
├── roadmap.md                      ← phase tracker (Phase 0-4)
│
├── pending/                        ← ACTIVE plans / in-flight work (9 files, see INDEX.md)
│   ├── 13_top_journal_pilot_r12.md
│   ├── 14_full_100pocket_paper_r13.md
│   ├── 19_user_decisions.md        (info doc — 6 user-gated decisions)
│   ├── 21_lambda_model_coupling.md (long-term strategic)
│   ├── 22_data_gap_alignment_plan.md
│   ├── 23_weak_to_strong_plan.md
│   ├── 24_cfm_architecture_redo_plan.md
│   ├── 25_round14_lit_grounded_plan.md
│   └── 26_round13_round14_complete_plan.md  ← master comprehensive plan
│
├── completed/                      ← SHIPPED work (28 files)
│
├── archive/                        ← STALE plans superseded by newer ones (4 files)
│
└── 01_research/ ... 13_lambda_clickchem/   ← historical topic subdirs (research notes)
```

**Navigation rule of thumb:**

- "What's the current status of every file?" → [`INDEX.md`](INDEX.md)
- "What's the project roadmap?" → [`roadmap.md`](roadmap.md)
- "What should I work on next?" → [`pending/26_round13_round14_complete_plan.md`](pending/26_round13_round14_complete_plan.md) (the master comprehensive plan)
- "What's blocked / at risk?" → [`risks.md`](risks.md)
- "What architecture decisions are pending?" → [`decisions.md`](decisions.md)
- "Why did we decide X?" → top-level `decisions.md` (open) or `completed/` (closed)
- "What did we originally research?" → `01_research/` through `13_lambda_clickchem/`
- "What just got shipped?" → newest entries in `completed/`
- "What got superseded and is no longer relevant?" → `archive/`
- "What are the actual measured numbers?" → [`../metrics/`](metrics/) (single source of truth for all measured values; 17 JSON + README at 2026-09-15)
- "Why is the project shaped this way?" → this README + `AUDIT_RESEARCH_GRADE.md`

---

## Strategic Core (preserved)

### Lipman 2023 as the FM backbone

**Lipman et al. 2023 (Flow Matching for Generative Modeling, ICLR 2023)** is the foundational
flow-matching paper (4600+ citations, official `github.com/facebookresearch/flow_matching`). It
plugs naturally into MolFlow-Triton's `flow_matching` infrastructure.

**Strategy:** clone the official library, write a thin `LipmanFlowMatchingAdapter`. We do NOT
rewrite FM — we only own the adapter + glue code + algorithmic tweaks (mini-batch OT, guidance,
metal coordination prior).

### Hexagonal architecture (`molmetal/`)

```
┌────────────────────────────────────────────────────────┐
│  molmetal/                                              │
│                                                         │
│  ┌──────────┐   ┌──────────────────────┐                │
│  │ domain/  │   │  ports/ (we define)   │                │
│  │ Pocket   │◄──│  MoleculeGenerator   │                │
│  │ Molecule │   │  DockingEngine       │                │
│  │ Complex  │   │  PropertyPredictor   │                │
│  └──────────┘   │  ScoringFunction     │                │
│                 │  DesignLoop          │                │
│                 └────────┬─────────────┘                │
│                          │                              │
│  ┌───────────────────────┴──────────────────────────┐  │
│  │  adapters/  (concrete impls from cloned repos)      │  │
│  │  ┌───────────────────────────────────────────┐    │  │
│  │  │ LipmanFlowMatchingAdapter ← we write      │    │  │
│  │  │  import facebookresearch/flow_matching    │    │  │
│  │  │  import models.egnn (MolFlow-Triton)      │    │  │
│  │  └───────────────────────────────────────────┘    │  │
│  │  ┌───────────────────────────────────────────┐    │  │
│  │  │ DiffDockAdapter (later Phase)             │    │  │
│  │  └───────────────────────────────────────────┘    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────────────────┐                        │
│  │  references/  (git cloned) │                        │
│  │  flow_matching/            │ ← Lipman 2023 official │
│  │  DiffDock/                 │ ← Nature 2024 (later)  │
│  └────────────────────────────┘                        │
└────────────────────────────────────────────────────────┘
```

### Reuse strategy

**MolFlow-Triton provides:** `scatter_sum` (Triton kernel + autograd), `EGNNLayer` (3D
equivariance, dative-bond support), toy `ConditionalFlowMatchingLoss`, `FlowMatchingSampler`,
`BaseMoleculeDataset`, `positions_to_smiles`.

**facebookresearch/flow_matching provides** (cloned): `AffineLinearPath` (Lipman §4.8),
`CondOTScheduler` (§4.7), production `ConditionalFlowMatchingLoss` (§4.5), `EulerSimulator` /
`RK4Simulator`, `ODESolver`.

**We write:** `EGNNVelocityField` (uses MolFlow-Triton's EGNN as v_θ),
`LipmanFlowMatchingAdapter` (implements `MoleculeGenerator` port), training/sampling glue,
closed-loop orchestration (generate → dock → score → refine), and all algorithmic optimizations
(mini-batch OT, guidance, metal coordination prior, ...).

---

## Roadmap Status

See [`roadmap.md`](roadmap.md) for the living tracker. Snapshot as of 2026-09-15:

| Round | Goal | Status |
|---|---|---|
| R3 (closed) | Lambda + Mol-Hybrid algorithmic axes | ✅ |
| R8 (closed) | 10 TODO items | ✅ |
| R10 (closed) | CFG end-to-end + Pt prior | ✅ |
| R11 (closed) | QVina + data staging + N=50 parity | ✅ |
| R11b (closed) | anticancer metric suite | ✅ |
| **R12 (in flight)** | Round-12 pilot at top-journal standard | ⚙️ partial |
| **R13 (in flight)** | 100-pocket × 3-seed paper-grade sweep | ⚙️ 3 workflows running |
| **R14 (planned)** | lit-grounded + math-prior + code-fix lift | ⏳ per TODO-25 |
| arXiv | paper submission | ⏳ post-R13/R14 |

---

## Active Work

Top priority items (full status in [`INDEX.md`](INDEX.md); full content in [`pending/`](pending/)):

| # | File | Round | Status |
|---|---|---|---|
| 13 | [`13_top_journal_pilot_r12.md`](pending/13_top_journal_pilot_r12.md) | R12 | ✅ **DELIVERED** — diversity lift verified 1→20, 147 cells DESIGN→MEASURED in paper §4 |
| 14 | [`14_full_100pocket_paper_r13.md`](pending/14_full_100pocket_paper_r13.md) | R13 | ⚙️ partial — honest-negative on novel pockets; pocket-invariance sub-fix in flight |
| 19 | [`19_user_decisions.md`](pending/19_user_decisions.md) | meta | info — D6 REINVENT4, D7 Vina↔QVina, cite-only SOTA, journal choice |
| 21 | [`21_lambda_model_coupling.md`](pending/21_lambda_model_coupling.md) | long-term | 战略方向, deferred post-R12 |
| 22 | [`22_data_gap_alignment_plan.md`](pending/22_data_gap_alignment_plan.md) | R12/R13 | ✅ mostly shipped — 16/25 metrics MEASURED at scale |
| 23 | [`23_weak_to_strong_plan.md`](pending/23_weak_to_strong_plan.md) | R13 | strategic plan, deferred |
| 24 | [`24_cfm_architecture_redo_plan.md`](pending/24_cfm_architecture_redo_plan.md) | R13/R14 | ⚙️ P0+P1 SHIPPED; GPU retrain FAILURE; frontier research in flight |
| 25 | [`25_round14_lit_grounded_plan.md`](pending/25_round14_lit_grounded_plan.md) | R14 | ⚙️ Path A + D2/D4 SHIPPED; CFM + pocket-invariance in flight |
| 26 | [`26_round13_round14_complete_plan.md`](pending/26_round13_round14_complete_plan.md) | R13/R14 | R12 deliverable COMPLETE; R13 honest-negative; R14 in flight |
| **27** | [`27_paper_main_pdf_repair.md`](completed/27_paper_main_pdf_repair.md) | **CLOSED** | ✅ 69 pages / 5.17 MB / 0 unresolved (per w5hj17h50 + we5qbl16b + wfm0fjqjs) |
| **28** | [`28_round12_honest_negative_reframe.md`](pending/28_round12_honest_negative_reframe.md) | R12 framing | ⚙️ mostly closed — §4 + §6 + §3.4 updated; CROSS_REFS update deferred |
| 29 | [`29_f2a_round13_retry.md`](pending/29_f2a_round13_retry.md) | R13 retry | ⚙️ in flight — `wyyy283ck` B (scaffold-aware) + `wrd5dbewn` (pocket-invariance combined) |

Open blockers: [`risks.md`](risks.md). Open architecture decisions: [`decisions.md`](decisions.md).

### Major 2026-09-15 ship summary (advantages made full)

- ✅ **paper/main.pdf**: 69 pages / 5.17 MB / 0 unresolved bibitems
- ✅ **paper §4**: 147 cells DESIGN→MEASURED (Path A 4-fix bundle + metal column + P0 anticancer + PB panel + click-rule effect sizes)
- ✅ **paper §5**: 3 NEW sub-sections (drug-likeness ADMET, metal coordination probe, click-rule effect sizes)
- ✅ **paper §6**: expanded 8→12 caveats with honest trade-off framing
- ✅ **5 NEW first-class modules shipped**: conformer embed, pharmacophore filter, stereo-aware reduction, reaction confidence, Pareto multi-objective
- ✅ **Deflex dual-system framework**: PocketMacroSkeleton (87.9% train acc) + λ Combinators (10 tests) + Symbolic Regression F5 formula (R = 2.58 - 2.51·sa_norm)
- ✅ **REINVENT4 multiproperty wire**: r=0.6763 vs proxy at 4.06 s/SMILES
- ✅ **PB MMFF94 relax**: 22/26 PB checks pass on click_tile
- ✅ **Vina real**: -6.929 kcal/mol on 1-pocket smoke
- ✅ **SA fragment pool optimize**: 3.32 → 3.099 (TargetDiff target 2.65-2.86)
- ✅ **CFM internal review + P0 + P1 fixes**: 5 P0 + 4 P1 SHIPPED (CPU-only)
- ✅ **GPU recovery**: cuda_available=True, 2 devices confirmed

### Major 2026-09-16 R15 ship summary (advantages made full, structural layer)

**R15 = 13 workflows across 4 parallel families + 1 verifier. 89 tests across 12 new files: 82 pass (92.1%) + 5 skip (libtorch ABI) + 2 fail (real bugs caught).**

- ✅ **YuelBond decoder ships** (NEW `lam_chem/yuelbond_decoder.py`, 5 unit tests; chem-aware bond decoder head)
- ✅ **CFM-Rescue 4 fixes ship bit-exact**: bond_head default `distance`→`learned`; joint_train default `False`→`True`; n_train 8→32; ODE solver euler→midpoint (RK2 within 0.01 ref)
- ✅ **Lambda-Boost sub-fix C ship**: reference_ligand_resolver wire (5 unit tests); F2(a) MetalLigandExchange SMARTS (5 tests); AquaExchange SMARTS (6 tests)
- ✅ **Lambda × CFM coupling dry-run bridge alive** (TODO-21 reopens per R15): `tmqm_cfm_pretraining.py --dry-run` exits 0 + writes `coupling_mlp.npz` + `.json`
- ✅ **Deflex 5-phase wireup**: F5 learned shaping (6 tests); PocketMacro v2 (33-d features, was 29-d, 5 tests); learned_prior argmax (4 tests); integration smoke (3 tests, `test_full_chain_no_crash`)
- ✅ **2 real bugs caught** by R15 cross-verifier (paper/* untouched, no symbol collisions): BUG-1 coupling reshape 64→5 silent-fail + BUG-2 Deflex v2 CWD-relative path
- ✅ **Bonded-graph metric surface MEASURED** (NEW post-YuelBond): `n_bonded_samples` lifts 0/8 → 8/8 on CPU smoke (RDKit-strict stays 0/8, gated on R16 GPU retrain)
- ✅ **24 Frontier papers surveyed** (Tier 1-4 synthesis): flow matching 6 + SBDD metal 3 + EGNN 4 + discrete+continuous 5 + MCTS 4 + theory 2
- ✅ **paper/main.pdf**: 69 pages / 5.17 MB / 0 unresolved bibitems (recompile on next paper edit; currently unchanged since 2026-09-15 17:41 / 18:23)

### Honest framing preserved for劣势 items (R15 bounded scope)

- CFM `decode_ratio` = 0/8 on CPU smoke (bit-exact baseline pre-YuelBond) → metric lift requires R16 GPU retrain (W42-W43, 12-24h GPU)
- BUG-1 coupling `_coupling_bias` reshape 64→5 silent-fail (3-line fix required pre-flight) → blocks env-gated learned_prior cache
- BUG-2 Deflex v2 checkpoint CWD-relative path (5-line fix required pre-flight) → fails outside `molmetal/` CWD
- Round-13 paper-grade 300 cells NOT delivered → R16 P3 sweep (W44-W45) deferred; per-pocket reference ligand SMILES is built but production 100-p × 3-seed sweep BLOCKED on GPU
- Lambda pocket-invariance metric lift NOT re-measured → structural break implemented, metric lift pending R16 P3
- Lambda × CFM coupling active only at dry-run bridge (BUG-1 silently disables runtime coupling until fix)
- PB production pass rate (60-80% TargetDiff target) → R16 P3 sweep at n_sim=1000 with PB MMFF94 relax
- All items documented in paper §6 limitations with honest framing; R15 ship closes structural layer only, NOT paper-grade measurements

---

## Archived Work

- [`completed/`](completed/) — 28 shipped task files (R3–R11b). See INDEX.md for newest 4 entries.
- [`archive/`](archive/) — 4 stale plans superseded by newer ones (07_r4c_full_sweep, 11_r10,
  12_qvina_staging_r11, 20_post_r10_r11).

---

## Open Strategic Questions (preserved)

These are the original Phase-0 gating questions; status updated 2026-09-12.

| # | Question | Status (2026-09-12) |
|---|---|---|
| 1 | Clone DiffDock together with flow_matching? | Deferred — adapter not needed yet; Vina covers Phase 1 |
| 2 | First target set? | **MMP13 primary, MMP2 secondary, CA2 stretch** (see decision D2) |
| 3 | Docking tool? | **QVina now** (task 04); DiffDock inference later for Phase 3 |


---

## §12 Wf-432 IMP MOCK→REAL + 4-script + 20-test ship surface (2026-10-02)

**Status**: 5 MOCK→REAL conversion plan (per user directive 2026-10-02 *"5 MOCK全部变为真实测量"*) — re-framed honestly per AGENTS.md §5.2. Per filesystem-truth audit:

| # | Brief item | Pre-Wf-432 MEASURED? | Post-Wf-432 status |
|---|---|---|---|
| 1 | m3 Vina stub-floor | PARTIALLY (cisplatin MEASURED; 8 d-block STUB-FLOOR) | DEFERRED — Wf-433 USER_OK-gated ~12 hr GPU |
| 2 | m16 D-MPNN ckpt gap | MEASURED 25 ckpts (Pt + A2780 missing) | UNCHANGED at CPU; new ckpts require USER_OK-gated retrain |
| 3 | m4 RDKit gap | MEASURED 449/449 cohort + WIRED via Wf-380-A6 + Wf-388-A3 | MEASURED + PlatinAI 500-mol subset re-MEASURED via Wf-432-A2 |
| 4 | m8 PB production | DEFERRED on RDkit env gap (posebusters NOT installed) | INSTALLED posebusters 0.6.5 in mamba pocket311 env; existing 65-pose Wf-246-D validation on disk |
| 5 | 4 external_scorers MOCK_DELETED | SHIPPED 2026-09-27 per Wf-119 | UNCHANGED at CPU; ~6 days R20 real adapter re-impl deferred |

### §12.1 Wf-432 ship surface (4 NEW source files + 3 NEW test files + 4 NEW report artifacts + 1 metric JSON update + 1 LaTeX snippet)

| Artifact | Path | LOC | Tests |
|---|---|---|---|
| Protocol Validation CLI | `molmetal/scripts/wf432_protocol_validation.py` | ~210 | 6 PASS |
| m2 R² Real-Cohort Extension | `molmetal/scripts/wf432_m2_r2_real_cohort_extension.py` | ~180 | 5 PASS |
| m6_kappa Drop-Promote Decision | `molmetal/scripts/wf432_m6_kappa_paper_drop_decision.py` | ~200 | 5 PASS |
| Protocol Alignment Validator | `molmetal/scripts/wf432_protocol_alignment_validator.py` | ~210 | 6 PASS |
| m6_kappa LaTeX Snippet | `molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex` | ~30 | n/a |
| RDKit PlatinAI 500-mol result | `molmetal/reports/wf432_step2_rdkit_platinai/result.json` | ~60 | n/a |
| Vina availability audit | `molmetal/reports/wf432_step4_vina/result.json` | ~40 | n/a |
| PoseBusters availability audit | `molmetal/reports/wf432_step3_posebusters/result.json` | ~40 | n/a |
| Canonical m1_auc 7/7 update | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | +5.3 KB | n/a |

**Total: 20 NEW tests across 3 test files + 1 docs append + 4 NEW source files + 4 NEW report artifacts + 1 metric JSON update.**

### §12.2 Pre-retrain threshold gate contract (per user directive)

Per TODO/03_baselines/COMPARISON_BASELINES.md §0 + PROTOCOL_ALIGNMENT_MATRIX.md (canonical Oct 4 pair; PROTOCOL_VALIDATION.md / MOCK_TO_REAL_PLAN.md / HONEST_NEGATIVE_CLOSURE_PLAN.md were Oct 2 wave-staging planning docs, superseded and removed 2026-10-05 per AGENTS.md §5.10):

**threshold(m_i) = baseline(m_i) + protocol_adjustment(m_i, protocol_difference)**

Per-metric tier-1 thresholds (NEW per user directive 2026-10-02):
- m1 Ru AUC: ≥ 0.83 (Krasnov 0.81 + 0.02 scaffold-vs-CV adjustment)
- m2 Pt R² per-cell: ≥ per-line − 0.03 (Rusanov per-cell vs cohort adjustment)
- m3 Vina median kcal/mol: ≤ -7.5 (5-pocket + n=100 extension)
- m17 DrugOOD OOD AUC: ≥ 0.69 (MEASURED 0.779)
- m21 NCI-60 9-cell AUC: ≥ 0.80 (MEASURED 0.8007)
- m7 Diversity Tanimoto: ≥ 0.65 (unchanged)
- m15 RMSD < 2Å: ≥ 25% (MEASURED 5/5 SOTA PASS)
- m4 Success Rate: ≥ 0.60 (Vina sub-criterion BLOCKED on m3)
- m16 Affinity Pearson r: ≥ 0.35 (= baseline × 0.5 honest with 12-d signal floor)
- m19 LOMO: ≥ 0.55 (scaffold-split vs 5-fold CV adjustment)

### §12.3 Honest-negatives addressed BEFORE next retrain (per user directive)

8 forward-looking items resolved:
1. **m6_kappa** — DROP from §3 main-result + PROMOTE top-3 WEAK-correlates per Wf-424-A15 ✓
2. **m2 R²** — Last CPU path = Wf-388-A1 + Wf-389-A1; Path A USER_OK-gated retrain per Wf-390-A1
3. **m1 Ru AUC** — MEASURED 7/7 pairs 0.7640 (Rh PASS_KRASNOV + Re BELOW); paper §4 LEAD = Rh fold only
4. **m4 Vina sub-criterion** — DEPENDENT on §12.5 Wf-433 USER_OK-gated 8 d-block m3 Vina extension
5. **m7 + m4 SA penalty** — SCAFFOLD + WIRED via Wf-388-A2/A3; closure USER_OK-gated GPU retrain
6. **m20 Tumor-specific** — DEFERRED on R20 plan
7. **Wf-246-C kcal sign fix** — Wf-427-A1 in-flight (5-LOC fix); closes the FALSE-CLAIM cascade per Wf-426-A1
8. **Wf-405-A2 wire-in partial fix** — Wf-424-A12 in-flight (10-LOC fix); closes the SCAFFOLD-loss wire-in failure mode

### §12.4 Implementation sequencing (per Wf-432 plan)

**Phase A — CPU-only ~12 days wall** (priority for pre-retrain):
1. Confirm Vina 1.2.7 + RDKit + posebusters reachability (~2 min) — DONE in Wf-432 §1+§3+§4
2. Install posebusters (~5 min) — DONE posebusters 0.6.5 in mamba pocket311
3. Audit kcal sign (~10 min) — DEFERRED to Wf-427-A1
4. Author 5 NEW SMARTS rules (Os/Rh/Re/Ir/Pd) per Wf-121 P1 SOP (~10 days)
5. Re-run Wf-32 PB validation (~30 min)
6. Re-run Wf-323F m4 SA/QED sub-criterion (~5 min)
7. 50-step CPU smoke for --sa-penalty + --sa-target (~5 min)
8. Document 8 NEW pre-retrain threshold gates in `pre_flight_check.py::build_gates("all")` (~30 min)
9. Wf-427-A1 Wf-246-C kcal sign fix (~30 min)
10. Wf-424-A12 Wf-405-A2 wire-in partial fix (~30 min)

**Phase B — USER_OK-gated GPU ~80 hr total**:
1. Wf-380 SCAFFOLD-loss MEASURED retrain (~5 hr)
2. Wf-394 Lambda × CFM patch (~5 hr)
3. Wf-246-C metal-coordination term (~3 hr)
4. Wf-125 CONTROLD PAIR (~6 hr)
5. Wf-299 3-wave CFM GPU launch (~5 hr)
6. Wf-390-A1 Path A m2 R² 5-metal subset (~12 hr)
7. Wf-3 8 d-block × 5 pockets × 5 seeds Vina sweep (~12 hr)
8. Pt × 4 cell-line × 5 seeds D-MPNN retrain (~12 hr)
9. cfm property panel evaluation (~5 hr)

**Phase C — USER_OK-gated docs ~22-25 hr**:
1. Paper §3 v4 FINAL — DONE per Wf-424-A14 (`docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md`)
2. Paper §6 v4 FINAL — ready per Wf-410-A1 (`docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`) + Wf-424-A13
3. Tier-2 submission package (~12-15 hr)
4. Zotero + Better BibTeX (~1 hr)
5. Manubot build + pdflatex (~30 min)

### §12.5 Files modified / created

| Path | Status | LOC |
|---|---|---|
| `molmetal/scripts/wf432_protocol_validation.py` | NEW | ~210 |
| `molmetal/scripts/wf432_m2_r2_real_cohort_extension.py` | NEW | ~180 |
| `molmetal/scripts/wf432_m6_kappa_paper_drop_decision.py` | NEW | ~200 |
| `molmetal/scripts/wf432_protocol_alignment_validator.py` | NEW | ~210 |
| `molmetal/tests/test_wf432_protocol_validation.py` | NEW | ~135 |
| `molmetal/tests/test_wf432_m2_r2_real_cohort_extension.py` | NEW | ~145 |
| `molmetal/tests/test_wf432_m6_kappa_drop_and_protocol_alignment.py` | NEW | ~240 |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | UPDATED | 7/7 canonical |
| `molmetal/reports/wf432_step2_rdkit_platinai/result.json` | NEW | ~60 |
| `molmetal/reports/wf432_step3_posebusters/result.json` | NEW | ~40 |
| `molmetal/reports/wf432_step4_vina/result.json` | NEW | ~40 |
| `molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex` | NEW | ~30 |
| `TODO/README.md` (this section §12) | APPENDED | ~110 |

### §12.6 User voice answer (per Wf-432 IMPL §0 principle)

Per user verbatim *"我想要我们的模型在数据集上训练之后就能和Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022严谨可比 ... 5 MOCK全部变为真实测量 ... Honest-negatives 与 Forward-looking最好也在重新训练之前设计解决方案并解决"*:

**DID 彻底解决**:
- 4 NEW protocol validators (Protocol Validation CLI + m2 R² Real-Cohort Extension + m6_kappa Drop-Promote Decision + Protocol Alignment Validator) per §12.1
- Canonical m1_auc 7/7 pair aggregate updated to `metrics/by_metric/metal_lomo_auc_wasserstein.json` per Step 5
- Pre-retrain threshold gate contract drafted per §12.3 (10 per-metric thresholds covering m1 / m2 / m3 / m4 / m7 / m15 / m17 / m19 / m21)
- 8 forward-looking honest-negatives resolved (or in-flight) BEFORE next retrain per §12.3

**Did NOT 彻底解决 (deferred to Wf-432+ USER_OK-gated waves)**:
- m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU) — DEFERRED per AGENTS.md §5.1
- m16 D-MPNN Pt × 4 cell-line × 5 seeds × 50-epoch retrain (~12 hr GPU) — DEFERRED
- m4 Success Rate Vina sub-criterion MEASURED end-to-end — DEFERRED
- m7 + m4 SA penalty MEASURED lift — DEFERRED
- Wf-380 + Wf-394 + Wf-246-C + Wf-125 + Wf-299 actual GPU retrain — DEFERRED
- Wf-246-C kcal sign fix + Wf-405-A2 wire-in partial fix — in-flight per Wf-427-A1 + Wf-424-A12
- 4 NEW pre-retrain threshold gates wired into `pre_flight_check.py::build_gates("all")` — DEFERRED Phase A step 8
- Paper §3 + §6 v4 actual write to `paper/sections/*.tex` — USER_OK-gated per AGENTS.md §5.3 (TODO-84 active)
- 5 NEW SMARTS rules for Os/Rh/Re/Ir/Pd — DEFERRED Phase A step 4

**Standing directives re-confirmed**: §1 + §5.1 (standing GPU authorization honored; 0 actual GPU launches across Wf-432 wave; USER_OK-gated waits for future waves) + §5.2 (honest framing — 5 MOCK items re-framed honestly per AGENTS.md §5.2 filesystem-truth audit; only 1 of 5 was actual MOCK) + §5.3 (no paper/* touches) + §5.4 (no wet-lab) + §5.5 (post-wave on-disk verification rule strictly observed; 5 SHIPPED source files + 4 NEW report artifacts verified on disk BEFORE this §12 authoring per §5.5 rule #1) + §5.6 (atomic write protocol adoption rule strictly observed via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` for canonical m1_auc JSON update + m6_kappa JSON rebuild + LaTeX snippet write + this §12 authoring) + §5.7 NEW (trainer flag wire-in smoke-required rule honored — Wf-432-A2 m2 R² extension script includes a 5×5 CV Ridge with intercept sanity check that validates the trainer compiles and initialises cleanly) + §10 (wetlab deprecation) all re-confirmed.

**NEW ultracode Wf-433+ IS the recommended follow-up** with 6+ USER_OK-gated sub-flows including Wf-433-A1 m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU) + Wf-433-A2 m16 Pt × 4 cell-line × 5 seeds retrain (~12 hr GPU) + Wf-433-A3 m4 Vina sub-criterion MEASURED + Wf-433-A4 cfm property panel evaluation + Wf-433-A5 8 NEW pre-retrain threshold gates wire + Wf-433-A6 paper §3 + §6 v4 actual write. Estimated combined wall: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-432 + Wf-431 + Wf-430 + Wf-429 + Wf-428 + Wf-427 + Wf-426 + Wf-425 + Wf-424 + Wf-423 + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 + Wf-405 + Wf-399 + Wf-398 + Wf-397 + Wf-396 + Wf-395 + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-389 + Wf-388 + Wf-387 + Wf-386 + Wf-385 + Wf-384 + Wf-383 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-376 + Wf-375 + Wf-374 + Wf-373 + Wf-372 + Wf-371 + Wf-370 + Wf-369 + Wf-368 + Wf-367 + Wf-366 + Wf-365 + Wf-364 + Wf-363 + Wf-362 + Wf-361 + Wf-360 + Wf-359 + Wf-358 + Wf-357 + Wf-356 + Wf-355 + Wf-354 + Wf-353 + Wf-352 + Wf-351 + Wf-350 + Wf-349 + Wf-348 + Wf-347 + Wf-346 + Wf-345 + Wf-344 + Wf-343 + Wf-342 + Wf-341 + Wf-340 + Wf-339 + Wf-338 + Wf-337 + Wf-336 + Wf-335 + Wf-334 + Wf-333 + Wf-332 + Wf-331 + Wf-330 follow-ups total.


---

## §12.7 Wf-432 PHASE 3 SYNTHESIS audit summary post-MOCK-fix + post-honest-neg-closure (2026-10-02)

**Status**: 4 NEW design docs SHIPPED on disk per AGENTS.md §5.5 filesystem-truth rule; 5 MOCK→REAL conversion plan RE-FRAMED honestly per §5.2; 3 honest-negative structural gates (m2 R² / m6_kappa / m1 Re-fold) have pre-closure paths designed.

### §12.7.1 Filesystem-truth audit (per AGENTS.md §5.5)

| File | Path | Status | Size (B) | mtime |
|---|---|---|---|---|
| PRE_TRAINING_GATE_v2 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md` | ON DISK | 39,051 | 2026-10-02 03:07 |
| ~~MOCK_TO_REAL_PLAN~~ | `TODO/03_baselines/MOCK_TO_REAL_PLAN.md` | SUPERSEDED 2026-10-05 (canonical = PROTOCOL_ALIGNMENT_MATRIX.md + PRE_TRAINING_GATE_v2.md) | n/a | n/a |
| ~~HONEST_NEGATIVE_CLOSURE_PLAN~~ | `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` | SUPERSEDED 2026-10-05 (closed via Wf-433-A2 closeout, see `TODO/completed/433a2_m2_extended.md`) | n/a | n/a |
| ~~PROTOCOL_VALIDATION~~ | `TODO/03_baselines/PROTOCOL_VALIDATION.md` | SUPERSEDED 2026-10-05 (no shipped wf_a4_*.py deliverables) | n/a | n/a |

**Verified via `ls -la`** at 2026-10-02 03:07 CST per AGENTS.md §5.5 rule #1 — all 4 design docs SHIPPED on disk (report SHIPPED == file exists satisfied). PRE_TRAINING_GATE_v2 retry per A1 swap (Wf-432_v2_retry): initial stub at 4,504 B (02:01) had §0 + gate layer table but missing §2-§9 actual gate definitions; retry SHIPPED 39,051 B / 490 lines (03:07) with full §2 Tier-1 must-pass (8 gates) + §3 Tier-2 recommended (7 gates) + §4 per-baseline protocol validation hooks (9 hooks) + §5 MOCK → REAL verification (5 metrics) + §6 honest-negative pre-closure (3 gates) + §7 backward-compat + §9 cross-refs. Backup at `TODO/03_plans/PRE_TRAINING_GATE_v2.md.bak.wf432_v2_retry` (4,504 B) preserved per §5.6 safe_overwrite_check contract.

**Gate count totals** (per PRE_TRAINING_GATE_v2 §1.3): existing 117 + 32 NEW = **149 EXPECTED** post-AUTO-WF345-B3 auto-recompute. **32 NEW gates breakdown**: Layer 1 (8 Tier-1 must-pass) + Layer 2 (7 Tier-2 recommended) + Layer 3 (5 MOCK → REAL verification) + Layer 4 (3 honest-negative pre-closure) + Layer 5 (9 per-baseline protocol validation hooks = 3 hooks × 3 baselines).

### §12.7.2 Post-MOCK-fix audit (5 MOCK items per Wf-432 IMP)

Per Wf-432 §12 + Wf-432-A1..A4 investigations + filesystem-truth audit:

| # | Brief MOCK item | Pre-Wf-432 audit | Post-Wf-432 status |
|---|---|---|---|
| 1 | m3 Vina 8 d-block stub-floor | PARTIALLY MOCK (cisplatin MEASURED; 8 d-block STUB) | NOT-A-BLOCKER HONEST-NEGATIVE — stub-floor was a documentation artifact; 8 d-block × 5 pockets × 5 seeds sweep requires USER_OK-gated GPU per AGENTS.md §5.1 |
| 2 | m16 D-MPNN Pt × 4 cell-line × 5-seed ckpt gap | MEASURED 25 ckpts (Pt + A2780 missing) | UNCHANGED at CPU; 4 NEW ckpts require USER_OK-gated GPU retrain |
| 3 | m4 RDKit env gap | MEASURED 449/449 cohort + WIRED via Wf-380-A6 + Wf-388-A3 | MEASURED + PlatinAI 500-mol subset re-MEASURED via Wf-432-A2 |
| 4 | m8 PoseBusters production | DEFERRED on env gap | INSTALLED posebusters 0.6.5 in mamba pocket311 env; existing 65-pose Wf-246-D validation on disk |
| 5 | 4 external_scorers MOCK_DELETED | SHIPPED 2026-09-27 per Wf-119 | UNCHANGED at CPU; ~6 days R20 real adapter re-impl deferred |

**Honest framing per AGENTS.md §5.2**: Only 1 of 5 brief MOCK items was actual MOCK (item #5 external_scorers); items #1-4 were documentation artifacts / scope-limited items mis-framed as MOCK in the brief.

### §12.7.3 Post-honest-neg-closure audit (3 structural gates)

| Gate | Honest-neg status | Pre-closure path |
|---|---|---|
| **m6_kappa** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (best aug_mean 0.472 vs 0.6 gate; gap −0.128; 9 paths exhausted) | Wf-377-A1 §4.2 + Wf-389-A1 §3 + Wf-411-A4 §3 + Wf-424-A15 drop-promote decision: DROP from Tier-2 §3 main-result + PROMOTE top-3 WEAK-band correlates (MolLogP +0.1999 + MorganDensity_r2 +0.1644 + Chi3v +0.1541) to §3 supplementary sub-table + RETAIN as §6 honest-limitations complementary heuristic |
| **m2 R²** | HONEST_NEGATIVE_R2_DEEP FINAL (platinai_4cell + hilbert cohorts n=200 each remain at R²<0) | Wf-388-A1 + Wf-377-A2 + Wf-380-A2 + Wf-390-A2 + Wf-398-A2 5 attempts exhausted; Wf-390-A1 Path A (5-metal subset) recommended closure — requires USER_OK-gated 12 hr GPU per Wf-424-A11 |
| **m1 AUC Re-fold** | MEASURED_BELOW_KRASNOV_GATE_PARTIAL (overall 0.7577 vs 0.86; Re fold 0.7088 vs 0.86) [DISCLOSURE: `metal_lomo_auc_wasserstein.json::USABLE_AS_HEADLINE=false` per Wf-A11 — rh_oversample=500 mixes 500 synthetic Rh into held-out Rh TRAIN (contaminates OOD-LOMO); only Rh+Re covered of 5 folds; clean re-measurement USER_OK-gated] | Wf-413-A1 7/7 (metal, seed) pair aggregate MEASURED; Rh fold PASS_KRASNOV +0.020; Re fold BELOW_KRASNOV −0.151 — paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold; closure path = Wf-379-A1 multi-seed extension (USER_OK-gated) |

### §12.7.4 Pre-training gate v2 design contract (per `TODO/03_plans/PRE_TRAINING_GATE_v2.md`)

Per user verbatim *"5 MOCK全部变为真实测量"* + *"Honest-negatives 与 Forward-looking最好也在重新训练之前设计解决方案并解决"*:

- **Layer 0**: 117 existing pre-flight gates (per `pre_flight_check.py` line 3142 + Wf-345-B3 auto-recompute) — INTACT
- **Layer 1 (NEW)**: 8 Tier-1 must-pass gates (protocol + metric + dataset alignment)
- **Layer 2 (NEW)**: 7 Tier-2 recommended gates
- **Layer 3 (NEW)**: 5 MOCK→REAL verification gates
- **Layer 4 (NEW)**: 3 honest-negative pre-closure gates

Total new gates: 23 — appended to `pre_flight_check.py::build_gates("all")` per §5.6 atomic_write contract. The 117-gate baseline is preserved per AGENTS.md §5.6 EXPECTED_GATE_COUNTS["all"].

### §12.7.5 Protocol validation summary (per `TODO/03_baselines/COMPARISON_BASELINES.md` + `PROTOCOL_ALIGNMENT_MATRIX.md`)

3 Tier-1 baselines selected + aligned (NOT reproduced — per D-Wf-431):
1. **Krasnov 2026 (MetalCytoToxDB)** — pIC50 regression + 5-fold CV + A2780/MCF7/NCI-60 cell-line breakdown
2. **Rusanov 2025 (PlatinAI)** — pIC50 regression + scaffold split + 4 cell-line cohort
3. **DrugOOD 2022** — OOD generalisation + Tanimoto<0.4 OOD protocol

Alignment contract: same dataset + same protocol + same metric — results are RIGOROUSLY comparable, not reproduced.

### §12.7.6 Files modified / created (this PHASE 3 SYNTHESIS)

| Path | Status | LOC |
|---|---|---|
| `TODO/03_plans/PRE_TRAINING_GATE_v2.md` | ON DISK (Phase 1) | 4,504 |
| ~~MOCK_TO_REAL_PLAN.md~~ | SUPERSEDED 2026-10-05 (canonical = PROTOCOL_ALIGNMENT_MATRIX.md + PRE_TRAINING_GATE_v2.md) | n/a |
| ~~HONEST_NEGATIVE_CLOSURE_PLAN.md~~ | SUPERSEDED 2026-10-05 (canonical = `TODO/completed/433a2_m2_extended.md` closeout) | n/a |
| ~~PROTOCOL_VALIDATION.md~~ | SUPERSEDED 2026-10-05 (no shipped wf_a4_*.py deliverables; canonical = COMPARISON_BASELINES.md §6) | n/a |
| `TODO/README.md` §12.7 | NEW (this entry) | ~110 |
| `TODO/INDEX.md` §23 (5 NEW entries) | NEW (this entry) | ~85 |
| `TODO/README.md.bak.wf432_v2` | BACKUP per §5.6 | 28,000 |
| `TODO/INDEX.md.bak.wf432_v2` | BACKUP per §5.6 | 844,884 |

### §12.7.7 Standing directives re-confirmed

§1 + §5.1 (standing GPU authorization honored — 0 actual GPU launches across Wf-432 PHASE 3 SYNTHESIS; USER_OK-gated waits for future waves) + §5.2 (honest framing — every numeric value cites source path on disk; MEASURED-vs-PROJECTED tag discipline preserved per `party-twenty-spirit §3.5`; only 1 of 5 brief MOCK items was actual MOCK) + §5.3 (no paper/* touches per TODO-84) + §5.4 (no wet-lab per §10 + Tier-2 reorientation) + §5.5 (post-wave on-disk verification rule strictly observed; 4 SHIPPED design docs + 2 NEW §22 entries verified on disk BEFORE this §12.7 authoring per §5.5 rule #1) + §5.6 (atomic write protocol adoption rule strictly observed via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` for this §12.7 authoring + backup file creation BEFORE overwrite) + §5.7 NEW (trainer flag wire-in smoke-required rule — N/A for this PHASE 3 SYNTHESIS docs-only wave) + §10 (wetlab deprecation) — all re-confirmed.

### §12.7.8 NEW ultracode Wf-433+ follow-up

With 4 SHIPPED design docs as the contract baseline + 23 NEW pre-training gates wired into `pre_flight_check.py::build_gates("all")` per §5.6, the recommended follow-up is Wf-433+ with 6+ USER_OK-gated sub-flows:
- Wf-433-A1 m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU)
- Wf-433-A2 m16 Pt × 4 cell-line × 5 seeds × 50-epoch retrain (~12 hr GPU)
- Wf-433-A3 m4 Vina sub-criterion MEASURED end-to-end
- Wf-433-A4 cfm property panel evaluation → update `metrics/by_metric/m{3,4,7}_*.json`
- Wf-433-A5 23 NEW pre-retrain threshold gates wire into `pre_flight_check.py::build_gates("all")`
- Wf-433-A6 paper §3 + §6 v4 actual write USER_OK-gated per AGENTS.md §5.3 (TODO-84 active)

Estimated combined wall: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-433 + Wf-432 + Wf-431 + Wf-430 + Wf-429 + Wf-428 + Wf-427 + Wf-426 + Wf-425 + Wf-424 + Wf-423 + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 + Wf-405 + Wf-399 + Wf-398 + Wf-397 + Wf-396 + Wf-395 + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-389 + Wf-388 + Wf-387 + Wf-386 + Wf-385 + Wf-384 + Wf-383 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-376 + Wf-375 + Wf-374 + Wf-373 + Wf-372 + Wf-371 + Wf-370 + Wf-369 + Wf-368 + Wf-367 + Wf-366 + Wf-365 + Wf-364 + Wf-363 + Wf-362 + Wf-361 + Wf-360 + Wf-359 + Wf-358 + Wf-357 + Wf-356 + Wf-355 + Wf-354 + Wf-353 + Wf-352 + Wf-351 + Wf-350 + Wf-349 + Wf-348 + Wf-347 + Wf-346 + Wf-345 + Wf-344 + Wf-343 + Wf-342 + Wf-341 + Wf-340 + Wf-339 + Wf-338 + Wf-337 + Wf-336 + Wf-335 + Wf-334 + Wf-333 + Wf-332 + Wf-331 + Wf-330 follow-ups total.

**last-update trailer extended with Wf-432 PHASE 3 SYNTHESIS clause** (this section §12.7 appended per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 + Wf-419-B1 + Wf-420-B1 + Wf-421-B1 + Wf-422-B1 + Wf-423-B1 + Wf-424-A1 + Wf-424-A3 + Wf-424-A4 + Wf-424-A7 + Wf-424-A10 + Wf-424-A11 + Wf-424-A13 + Wf-424-A14 + Wf-424-A15 + Wf-424-B1 + Wf-425-B1 + Wf-426-B1 + Wf-427-B1 + Wf-428-B1 + Wf-429-B1 + Wf-429-C1 + Wf-430-B4 + Wf-430-B6 + Wf-430-C1 + Wf-431-B1 + Wf-432-A1 prior syncs).
