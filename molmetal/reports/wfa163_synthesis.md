# Wf-A163 — SYNTHESIS

- tag: **MEASURED** (each number cited with its own path) · date: **2026-11-01** · wave: Wf-A163
- role: synthesis agent · GPU: **0** (every command prefixed `CUDA_VISIBLE_DEVICES=""`, single venv
  `/home/hugo/codes/mol-metal/.venv/bin/python`, no `os.walk`, no whole-suite pytest)
- inputs: branches N / G / V + refuters `ref1` (N), `gpucheck` (G), `v3` (V)
- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`
- this file: `molmetal/reports/wfa163_synthesis.md`

**Adjudication rule applied:** a claim survives only if no refuter refuted it. Where a refuter and a
branch disagree the refuter wins, *unless* the refuter's own evidence is weaker — those cases are
called out explicitly and demoted to **UNRESOLVED** below.

---

## 1. WHAT SURVIVED

### S1. The c7 headline REPRODUCES bit-for-bit under two independent implementations (branch V)

No refutation succeeded; the refuter's own implementation is methodologically the *opposite* of
the branch's (refuter: RDKit `Chem.GetMolFrags` + `RWMol.RemoveAtom`; branch: own BFS over a
hand-built bond adjacency, asserted on the parsed AST), so this is a genuine two-implementation
agreement, not a re-run.

| quantity | value | on disk at |
|---|---|---|
| metal-inclusive single-fragment rate | **0.08509630565203663** (1078 / 12668) | `ligand_connected_rate_c7.json:/metal_inclusive/single_fragment_rate` |
| ligand-only single-fragment rate | **0.02083991158825387** (264 / 12668) | `/ligand_only/single_fragment_rate` |
| analysed denominator | **12668** | `/denominators/n_analyzed` |
| McNemar exact two-sided p | **8.635435522167073e-197** | `/paired/mcnemar_exact/p_value_two_sided` |
| contingency a/b/c/d | 220 / 44 / 858 / 11546 | `/paired/contingency` |
| per-epoch `raw_single_rate` + `n_input` vs `metrics/by_metric/cfm_fragmentation_postproc_arm.json` | **18/18 MATCH** | same |

**I re-read the four headline numbers off the artefact myself** (not from a branch report) and they
are identical to branch V's and to refuter V's independent re-derivation. The denominator probe is
also confirmed: 12669 CSV rows − **1** RDKit-unparseable row
(`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv:9848`, epoch 13, "Maximum BFS
search size exceeded") = 12668; using 12669 would give 0.08508958875996527 (delta −6.7e−06).

**Surviving caveats (must ride with the number):**
- The ligand-only arm is a **metal-rule-dependent** quantity. It is invariant across the four
  "genuine" vocabularies but that agreement is **structural, not discovered**: all four induce the
  *same* 6-symbol set (Ir, Os, Pt, Re, Rh, Ru) on this pool. Only the metalloid-widened **strawman**
  discriminates: LO 0.02083991158825387 → 0.018866435112093464 (n 264 → 239). Quote the LO rate
  only under a stated metal predicate.
- The metal-inclusive column of the predicate probe is a **tautology** (`components(adj, all_idx)`
  never consults `metal_symbols`) — its "identical under all four vocabularies" row is not evidence.
- The McNemar **value** is confirmed via a third route (symmetry identity, never evaluating the
  threshold rule); the **independence claim** is only partial — refuter V's AST diff finds 2 of 9
  body statements textually identical, i.e. two writings of the same standard algorithm. Wording
  must be tightened; the number stands.
- The 1903 free-halide rows **dilute** the ligand-only arm (numerator unchanged, denominator +1903,
  LO rate +17.68 % relative) — they do not inflate it.

### S2. The Wf-A155 class IS closed on both repaired artefacts (branch N's cross-check)

I re-ran the guard directly against all four artefacts:

| artefact | pool_misattribution | stale_cross_reference | count_contradiction |
|---|---|---|---|
| `metrics/by_metric/wfa155_ligand_connected_rate.json` (repaired c8) | 0 | 0 | 0 |
| `…/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json` (repaired c7) | 0 | 0 | 0 |
| `…/wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json` (pre-repair c8) | 0 | 0 | 0 |
| `…/wfa163_narrative_truth_scan/counterfactual_pre_repair_c7.json` (labelled COUNTERFACTUAL) | **1** | 0 | **2** |

Refuter N reproduced this twice (branch scanner + its own no-branch-code implementation: 28 prose
numeric tokens per artefact, 0 unbacked) and confirmed the counterfactual rebuilds byte-identically
and re-scans to its claimed 5 flags. **The pre-repair c7 artefact genuinely does not exist on disk**
(0 tracked, empty `git log --all`) — no reconstruction from memory. The two **semantic** sites
("this is the 8.51 % headline", "this is the 25.05 % fragment count") are out of a number-counterpart
scanner's reach, as branch N states.

### S3. Branch N's census numbers all reproduce exactly (denominator 583)

583 artefacts enumerated / 583 parsed / 0 unreadable · 169 flagged / 414 clean / 482 flags ·
guard 0 / 0 / 19 / 474 · hand adjudication sample 139 of 482 flags (28.9 %) → **0 TRUE_POSITIVE**,
106 BENIGN, 33 UNBACKED_OWN_MEASUREMENT · the tree-wide strongest class (12 COUNT flags)
adjudicated in full → 0 / 8 / 4. Every one of these reproduced on disk under refuter N.

The branch's **refusal** of the "the published metrics tree is untrustworthy" claim also survives
and is correct: 0 of 583 artefacts contains a prose number contradicting its own fields. Its stated
precision — strong class **0/139**, soft class ≈24 % actionable — is honest and is the reason the
proposed gate ships `declared` (hard) + `auto` (advisory) modes.

### S4. Wf-A162's landed pre-flight lock is REAL (branch G §4) — no diff proposed for that path

I re-verified all four claims **by execution** under `CUDA_VISIBLE_DEVICES=""`:
`_session_cuda_is_locked()` → `True`; `_smoke_device()` → `"cpu"`; **`torch` not in `sys.modules`
before or after** (proving the no-subprocess short-circuit, `pre_flight_check.py:13136-13138`);
`_visible_device_count_hint()` → 0 under a `−1` mask. The NVML env is at `:13211`; both consuming
gates return FAIL with "(smoke NOT run)" at `:13437` and `:17017`. **No diff is proposed for
`pre_flight_check.py`** (read-only path).

### S5. The GPU-lock answer ("YES, via 2 mechanisms, no probe is one") survives

The parent-propagation qualification is stated correctly: a child inherits the lock **only if the
parent had it**, so the exposure lives at the entry points that build a *fresh* env. The 5
`${VAR:-}` shell sites were re-checked and exactly **one** is a genuine session-lock defeater:
`scripts/launch_wf308c_todo220_gpu.sh:125`. `molmetal/scripts/_cuda_default.py` sets `CVD="0"`
only when the variable is **unset** (an explicit `""` counts as set and is honoured).

### S6. Branch G's patch applies cleanly, is NOT applied, and does not touch the read-only path

`git apply --check molmetal/reports/wfa163_gpu_tooling_census/patch_smoke_device_lock.diff` → exit
**0** (clean) on the current working tree; the choke point is **not** present in
`molmetal/tests/conftest.py`; the only `pre_flight_check` mention in the diff is a comment. Its 12
own regression tests pass.

---

## 2. WHAT DID NOT SURVIVE — HARD FINDINGS (defect confirmed on disk)

### H1. Branch G's census API table is a hardcoded 5-name list → "10 primary-context sites" is 10 of 95 (severity HIGH)

`molmetal/scripts/wfa163_gpu_tooling_census.py:96` — `PROBE_APIS` = `{is_available, device_count,
current_device, set_device, init}`. The census's own `hits` array contains **0** entries for
`synchronize`, `get_device_name`, `get_device_properties`, `get_device_capability` (I counted the
`api` field across all 2025 hits). The refuter derived the true set by transitive closure over
`_lazy_init()` in the installed torch and measured **95** primary-context-creating sites in the
census's own scope; **81 of the 95 are absent from the census hits entirely**. §7.1 uses "10" as
the basis for attributing the ~550 MiB watchdog observation to a primary context, and the report
discloses no API-table limitation. §2.1 is the load-bearing section for the report's central
question.

*Refuter's evidence strength:* the **mechanism** is verified by me (5-name table; 4 APIs with 0
census hits). The **number 95** is the refuter's own measurement — I did not independently re-derive
it, so treat "95" as MEASURED-BY-REFUTER and "10 of 95" as the direction that is confirmed.

### H2. Import-time probes are **6**, not 5 (severity MEDIUM)

`molmetal/molmetal_lam/sbdd_env/protein_metal_binding.py:208` — `device: str = "cuda" if
torch.cuda.is_available() else "cpu"` is a **`@dataclass` field default**, evaluated at
class-creation time = import time. The census records the hit but tags it `module_level: false`
because `_index_module_level_lines` (`:372-383`) skips `ast.ClassDef` spans. The site appears in
**neither** the count of 5 **nor** §5.0's uncovered-paths table. **Verified by me by reading the
source.**

### H3. The child-spawn population is **24 in 15 files**, not 28 in 18 (severity MEDIUM)

All four contested sites **do** copy the parent environment, so "inherits NOTHING, including the
lock" is false for them. **Verified by me by reading the source:**
- `molmetal/tests/test_refactored_train_scripts.py:313` and `:390` — `env = dict(__import__("os").environ)`
  (the census's `_resolve_env_node` substring-tests `"os.environ"`, absent from `("os").environ`);
- `molmetal/tests/test_registry_generate_import_path.py:54` — `child_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}` over `os.environ`;
- `molmetal/tests/test_wf432a2_w246_real_trainer.py:26` — `_run_wrapper` does `os.environ.copy()`.

The census over-counts the population by 4 (17 %). **The patch's choke point is still correct for
the real 24** — only the rationale count and the diff's header comment are wrong.

### H4. The scope disclosure points at the harmless exclusions (severity MEDIUM)

**7 tracked, primary-context-creating call sites lie outside the census's scope prefixes** — I
confirmed each is tracked and each is a real GPU-path creator: `main.py:28`, `main.py:29`,
`models/_scatter.py:89`, `models/_scatter.py:159`, `triton_kernels/backend.py:55`,
`triton_kernels/backend.py:73`, `submission/paper_a/molmetal/__init__.py:95`. The report names only
`paper/`, `docs/`, `tools/` as excluded — never `models/`, `triton_kernels/`, `submission/`,
`main.py`. Plus `triton_kernels/tests/test_kernel_suite.py:12` (an import-time probe outside scope).

### H5. **NEW, and the most urgent:** the *landed* blocking rule `stale_cross_reference` has a measured recall hole

This is not in any branch or refuter report as a *live* issue, because the wiring landed **after**
both. `molmetal/scripts/promote_metrics.py:990-1005` now blocks a promotion on
`pool_misattribution` + `stale_cross_reference`, and that second rule **cannot fire on the class it
was built for**. I reproduced the hole by executing the guard's own functions:

- `quoted` (guard `:749-754`) mixes an **int token** (the integer part of the quoted number) with
  the float token;
- `_consistent_with` (`:659-670`) accepts an int token matching **any** numeric leaf of the cited
  value;
- so a stale quote is silently accepted whenever its integer part collides.

Against the real published citation (`cfm_fragmentation_audit.json -> /summary/generation_side/single_molecule_rate_end_to_end`,
cited value `{n: 9, min: 0.0, max: 0.442…, mean: 0.1766…, median: 0.08506944444444445}`):

| quoted | findings | should be |
|---|---|---|
| `= 0.07777777777777778` | **0** | 1 (median is 0.085069…) |
| `= 9.5` | **0** | 1 (n = 9) |
| `= 0.5` | **0** | 1 |
| `= 1.7777777777777778` | 1 | 1 |

Mechanism: the quoted float `0.0777…` is tokenised with an int sibling `0`, and the cited
`min: 0.0` swallows it (`abs(0 − 0.0) ≤ tol`). **This rule quotes rates.** Its tree-wide
denominator is **3 citation sites in 583 artefacts** (I counted `_CITE_RE` matches across the whole
tree myself). The sibling blocking rule has ≈0 recall for the Wf-A155 class too: the preserved
**pre-repair c8** artefact scores `pool=0 / stale=0 / count=0` (its "8.51 %" was caught only by the
**advisory** `unbacked_percent`). **Net: a live blocking gate whose two rules fire 0 times on the
tree and cannot detect the class they were written for.** The "0 blocking findings across 583" basis
is therefore a statement about a weak instrument, not about the tree's health.

### H6. Branch N's "4 ARMED scripts" is a hardcoded-marker-list **lower bound**, and it misses an on-disk instance of the very class it hunts

`molmetal/scripts/wfa163_narrative_truth_scan.py:1026-1035` — `TOOL_LITERAL_MARKERS` is **8
Wf-A155-specific strings** ("2112", "cfm_c8_fullgen_2026_10_24", "704 rows", "529/2112", …), while
§2.3 states the ARMED rule in general terms. Under the report's **own** rule, a script that was on
disk **before** the 11:15 scan is armed and was missed: **`molmetal/scripts/wfa163_c7_rederive.py`**
(branch V's own script, mtime 10:43). **Verified by me by reading the source:** it takes `--pool`
(`:563`) and hard-codes pool-specific numbers in **prose string literals** while its sibling
sentences use f-strings —
- `:1213` `"…move, 17/18 epochs…"` (18 = this pool's epoch count),
- `:1217` `"The 1903 free-halide rows…"` (1903 = this pool's count),
- `:1220` `"The denominator is 12668, not 12669…"` (both counts),
- `:1224` `"(delta -6.7e-06)"` (this pool's delta),
- contrasted with the computed `:844`, `:1051`, `:1054`.

That is the **Wf-A155 defect class, in a `tag: MEASURED` artefact, in this wave's own output**,
missed by the scanner written to find it. Refuter N's general AST detector reports **9** armed
scripts; 2 of those are the scanner's own marker literals and the detector's own pattern strings,
`wfa168_falsevalid_axis.py` post-dates the scan (mtime 11:46 > 11:15), and `wfa133_…matrix.py`'s
hits are docstring-only (excluded by branch N's own rule). **The defensible statement is
"≥ 5 armed scripts" (the branch's 4 + branch V's own), not "4"** — the census is a lower bound.
Separately, §3's claim that the tool layer was scanned over `molmetal/scripts/*.py` **+
`molmetal/**/scripts/*.py`** is false: the code globs only the former.

### H7. Branch N's §6 file:line citations are all stale

`build_artifact` is now **def 1817**, not 913; the four armed scripts were rewritten by **Wf-A164**
after the 11:15 report (their docstrings now read "Wf-A164 (2026-11-01): this block used to be
four hard-coded strings") and their literals are computed (`_pool_phrase`). The **finding** was
true when measured; the **citations** do not reproduce today. Confirmed by me.

### H8. Branch N's patch **no longer applies** to the current tree — because the wiring has already LANDED

`git apply --check …/patch_narrative_guard.diff` → *"already exists in working directory"* for the
guard and the test (the branch's own untracked outputs) and *"patch does not apply"* for
`molmetal/scripts/promote_metrics.py:65`. **That hunk is already live**: `promote_metrics.py`
(mtime 11:33) contains the `BLOCKED_NARRATIVE_TRUTH` gate at `:990-1005`. Against **HEAD** the
diff is exit 0, so the branch's scratch-copy claim reproduces. **Consequence for the main thread:
the patch must not be re-applied; the wiring is live and must be fixed forward (H5).**

### H9. Branch N's "published metrics tree" framing covers **583 of 604** artefacts

I counted: `metrics/by_metric` 582 + `metrics/paper_tables` 1 = **583** (the branch's scope);
the tree also holds `by_paper_section` 3 + `by_round` 15 + `sota_comparisons` 3 → **604** total.
Refuter N's re-scan of the branch's own scanner over the wider set gives **178 flagged / 497
flags** (vs the branch's 169 / 482) — so the branch's precision statement is measured on a scope
that omits 21 published artefacts. (The census also covers only **2 of the 5** `metrics/` subdirs.)

### H10. Branch N's §9.5 test claim is overstated

The test file pins only **one** of the two repaired artefacts (`grep 'ligand_connected_rate_c7\|wfa162'`
in `molmetal/tests/test_wfa163_narrative_truth.py` → 0 hits). The c7 zero is a §7 table row, not a
regression test.

### H11. Durability gap (Finding G, refuter V)

`.gitignore:158` `molmetal/reports/**/*.json` gitignores **both** `wfa163_c7_rederive/rederive.json`
**and** the artefact under test `wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`;
both are untracked and the repo's own whitelist mechanism (`.gitignore:159-165`) was not applied.
**A clean clone cannot re-verify the 13/13 table.** Also: the repo-root `conftest.py` is untracked
and must be `git add`-ed for branch G's patch to be meaningful.

---

## 3. SPECULATIVE ONES (named as a risk, NOT demonstrated on disk)

- **SP1.** Branch N's §6 concern that the four armed scripts "will emit false prose if re-run on a
  pool other than the one whose numbers were hard-coded" is now **untestable as stated** — Wf-A164
  rewrote them to compute the numbers, so the risk appears retired. Not demonstrated either way
  (no re-run on a non-default pool was performed in this wave).
- **SP2.** Branch G's §7.1 attribution of the ~500-600 MiB watchdog observation to a primary
  context rests on "10 primary-context sites"; after H1 that attribution is **unsupported**, but
  no measurement shows the 95 sites do allocate — the 550 MiB observation itself is not
  re-attributed here.
- **SP3.** The guard's ≈0 recall for the Wf-A155 class (pre-repair c8 scores `pool=0`) shows the
  *rule* cannot detect a self-consistent-but-uncomputed prose number; it does **not** by itself
  show that any published artefact contains such a number.

---

## 4. UNRESOLVED (a branch's own reviewer said UNSURE, or the refuter's evidence is weaker)

- **U1 — branch G's file-count denominator.** Branch: 6339 enumerated / 3011 decoded. Refuter
  measured 6380 / 2930 on the current tree and marked it **UNSURE**; the branch itself discloses
  it as a snapshot. I did not reproduce either figure (concurrent branches moved the tree between
  runs, and the refuter's own summary and `verdict.md` disagree with each other: 6391 vs 6380).
  **The denominator is neither confirmed nor refuted — it is a moving target.**
- **U2 — the corrected primary-context count "95".** The refuter's number, not the branch's. The
  *direction* (10 of a much larger set) is confirmed by me; the exact figure is not
  independently re-derived. Treat as MEASURED-BY-REFUTER.
- **U3 — branch N's "863 scripts checked"** vs the refuter's 873: both are snapshots of a tree
  that other branches were mutating. Not comparable.
- **U4 — branch V's McNemar *independence* wording.** The value is confirmed; the claim that the
  two routines are algorithmically independent is **partially refuted** (2 of 9 body statements
  identical). The number stands; the wording must be corrected.

---

## 5. WHAT THE MAIN THREAD MUST DO — ORDER OF OPERATIONS

**Step 0 (do this first).** **Do NOT re-apply `patch_narrative_guard.diff`** — it exits 1 on the
current tree because the guard, the test and the `promote_metrics.py` wiring **are already on
disk** (H8). Treat the guard as **live production code** and fix it forward.

**Step 1 — H5 (highest severity, live code).** In
`molmetal/scripts/wfa163_narrative_truth_guard.py::find_stale_cross_references`, stop the integer-part
token from suppressing a stale quote: drop the int token when a float token covers the same span
(or require the int token to match the cited value with no float sibling), and require the *float*
token to match. Re-verify by injecting the four quotes above into the real citation site and
asserting **4** findings, with the good quote still passing. Then re-run the 583-artefact census
and confirm the blocking count against the **3** citation sites tree-wide (not 583), and add a
regression test pinning the hole. Re-run
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa163_narrative_truth.py -q`
(15 passed today) — the new test must fail before the fix and pass after.

**Step 2 — H2, H3, H4 (branch G), then apply its patch.** The patch still applies cleanly (exit 0)
and is unapplied, so the order is: (a) widen `PROBE_APIS` to the torch-derived set
(`synchronize`, `get_device_name`, `get_device_properties`, `get_device_capability` + the existing
five) and re-run the census so §7.1's 550 MiB attribution rests on the full set; (b) fix
`_index_module_level_lines` to include `ast.ClassDef` bodies (→ 6 import-time probes) and add
`protein_metal_binding.py:208` to §5.0's uncovered-paths table; (c) fix `_resolve_env_node`'s
`dict(...)` branch so `("os").environ` and dict-comprehension-over-`os.environ` resolve to a full
env copy (28 → 24 in 15 files) and **correct the diff's header comment**; (d) widen the scope
prefixes to the 7 tracked out-of-scope sites, or state them as a known exclusion. Only then apply
`patch_smoke_device_lock.diff` and re-run
`molmetal/tests/test_wfa163_gpu_tooling_census.py` (12 passed) plus a functional re-check of the
three touched files. `molmetal/tests/conftest.py` is already modified by another branch — the
patch's hunk context is against the **working tree**, not HEAD; the repo-root `conftest.py` is
untracked and must be `git add`-ed.

**Step 3 — H6 (branch V's own output).** Compute the prose numbers in
`molmetal/scripts/wfa163_c7_rederive.py` — replace `"17/18 epochs"`, `"1903"`, `"12668"/"12669"`
and `"-6.7e-06"` with f-strings over the already-computed values — and re-issue
`molmetal/reports/wfa163_c7_rederive/report.md` through `atomic_write_file`. Then replace
`TOOL_LITERAL_MARKERS` in the scanner with the general AST rule and re-run the ARMED census over
`molmetal/scripts/*.py`. The four Wf-A155 tools themselves need no patch (Wf-A164 already
computed them) but their §6 citations must be re-quoted against the current source (H7).

**Step 4 — H9, H10 (branch N's report).** Re-label the scope as "**583 of 604** published
artefacts", re-run the scanner over the wider 604 (refuter measured 178 flagged / 497 flags), add
the **c7** artefact to the test file, and re-issue §6 with dated citations or an explicit
"superseded by Wf-A164" note. Do not re-apply the diff (Step 0).

**Step 5 — durability (H11).** Add `.gitignore` exceptions for
`molmetal/reports/wfa163_c7_rederive/**` and
`molmetal/reports/wfa162_c7_pool_ligand_connected_rate/**` so the 13/13 reproduction table is
re-verifiable from a clean clone; `git add` the repo-root `conftest.py`.

**Step 6 — re-verify everything, CPU-only, no whole-suite pytest.**
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa163_narrative_truth.py molmetal/tests/test_wfa163_gpu_tooling_census.py molmetal/tests/test_wfa163_c7_rederive.py -q`
→ then `python molmetal/scripts/pre_flight_check.py` (read-only; do not modify), then
`python molmetal/scripts/registry.py check` (579 current / 0 stale per `AGENTS.md` §2).

---

## 6. HONEST FRAMING OF THE WAVE'S HEADLINE

**MEASURED, survives:** the c7 ligand-connected-rate headline
(0.08509630565203663 metal-inclusive / 0.02083991158825387 ligand-only over 12668 analysed rows of a
12669-row pool; McNemar exact p = 8.635435522167073e-197) reproduces **bit-for-bit** under two
independently written implementations, and the Wf-A155 narrative-defect class is **closed on both
repaired artefacts** (0 flags under two independent instruments).

**MEASURED, does not survive:** branch G's census is **not a census** (a 5-name API table over a
95-site population; 5 import-time probes vs 6; 28 vs 24 child spawns; 7 tracked sites excluded
without disclosure), and branch N's ARMED-script count is a **hardcoded-marker lower bound** that
misses this wave's own output. The **live** blocking gate that branch N proposed has a **measured
recall hole** on the class it was built for.

**Not claimed:** no published metric is shown to be false. The finding of this wave is that the
**instruments** were weaker than their denominators claimed — which is the honest-negative form.

---

## 7. CITATION

- branches: `molmetal/reports/wfa163_narrative_truth_scan/report.md` (+ `narrative_truth_scan.json`,
  `patch_narrative_guard.diff`), `molmetal/reports/wfa163_gpu_tooling_census/report.md` (+
  `gpu_tooling_census.json`, `patch_smoke_device_lock.diff`), `molmetal/reports/wfa163_c7_rederive/report.md`
  (+ `rederive.json`)
- refuters: `molmetal/reports/wfa163_refute_ref1/verdict.md` (+ `verdict.json`,
  `general_armed_detector.json`, `full_metrics_tree_scan.json`, `strict_citation_check.json`),
  `molmetal/reports/wfa163_refute_gpucheck/verdict.md` (+ `verdict.json`, `independent_scan.json`),
  `molmetal/reports/wfa163_refute_v3/verdict.md` (+ `verdict.json`)
- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`
- prior wave: `molmetal/reports/wfa160_bond_threshold_independent_verification/report.md`
- this synthesis: `molmetal/reports/wfa163_synthesis.md`
