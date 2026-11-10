# Wf-A163 branch N — ADVERSARIAL REVIEW (reviewer `ref1`)

**Date:** 2026-11-01 · **Target:** `molmetal/reports/wfa163_narrative_truth_scan/` ·
**CPU-only** (`CUDA_VISIBLE_DEVICES=""`; no torch import, no CUDA, no decode, no trainer, no `os.walk`).
**Verdict:** the branch is **substantially honest** — the cross-check it was asked to run is REAL and
reproduces under two independent implementations, and its headline framing is *more* conservative than
the brief expected. Five material defects, none of them a fabricated number.

| # | Claim | Verdict |
|---|---|---|
| C1 | Denominator 583 artifacts, 583 parsed, 0 unreadable | CONFIRMED (scope caveat, C15) |
| C2 | 169 flagged / 414 clean / 482 flags + rule split | CONFIRMED |
| C3 | Hand adjudication 139/482 (28.9 %), 0 TP / 106 / 33 | CONFIRMED |
| C4 | Precision stated as 0/139 strong, 33/139 soft; refuses the strong claim | CONFIRMED |
| C5 | **Cross-check: 0 flags on BOTH repaired artifacts (scanner AND guard)** | **CONFIRMED** |
| C6 | 4 ARMED scripts, 863 scripts checked | PARTIALLY REFUTED |
| C7 | §6 file:line citations | REFUTED AS STATED (stale) |
| C8 | `wfa158_population_reconciliation.py` not armed | CONFIRMED |
| C9 | Pre-repair c7 not on disk; counterfactual reproducible | CONFIRMED |
| C10 | Diff applies cleanly, NOT applied, avoids READ-ONLY paths | CONFIRMED in substance / STALE in detail |
| C11 | 15 tests pass; "the two repaired artifacts on disk score 0" | PARTIALLY REFUTED |
| C12 | 0 blocking findings over 583; gate blocks no promotion | PARTIALLY REFUTED |
| C13 | No write into a forbidden path | CONFIRMED |
| C14 | §5 numeric-leaf examples | CONFIRMED |
| C15 | "the published metrics tree" framing | PARTIALLY REFUTED (undercount) |

---

## 1. The cross-check (the brief's suspect e) — **CONFIRMED, twice**

`metrics/by_metric/wfa155_ligand_connected_rate.json` and
`molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json` score **0 flags**
under the branch's scanner **and** 0 under the guard — reproduced by me:

```
$ .venv/bin/python molmetal/scripts/wfa163_narrative_truth_scan.py --files <the 4 targets> --out …/branch_scanner_crosscheck.json
scanned 4 artifacts (0 unreadable) … total flags: 6
   → repaired c8: 0 | repaired c7: 0 | pre-repair c8: 1 (PERCENT 8.51 @ /purpose) | counterfactual: 5
guard:  pool 0 stale 0 count 0 unbacked 0 | pool 0 stale 0 count 0 unbacked 0 | … | … | 1 pool + 2 count
```

and, with **my own implementation** (`molmetal/scripts/wfa163_refute_ref1.py backing`, no branch code):
both repaired artifacts carry **28 prose numeric tokens and 0 unbacked**. The counterfactual rebuilds
**byte-identically** from the two on-disk sources and re-scans to the same 5 flags
(3 × `SELF_CONTRADICTING_PROSE_COUNT` `2112`, 1 × `…POOL` `cfm_c8_fullgen_2026_10_24`,
1 × `…SIBLING` `60`). **The class is closed on both repaired artifacts.** The pre-repair c7 artifact
genuinely does not exist on disk (3 artifact JSONs, 0 tracked, empty `git log --all`).

## 2. Where the branch is wrong

**(a) C6 — the tool-layer census is a marker list, not a detector.** `scan_tools` implements
"ARMED" as a **fixed 8-string list** of Wf-A155 pool literals
(`wfa163_narrative_truth_scan.py` L1026-1035 `TOOL_LITERAL_MARKERS`). `report.md` §2.3 states the rule
in general terms ("carries a pool-specific literal inside a string literal") and never discloses this.
Measured consequence: **a script that matches the report's OWN rule but not the marker list was on disk
before the scan** — `molmetal/scripts/wfa163_c7_rederive.py` (mtime 10:43; the scan's JSON is 11:15).
It takes `--pool` as a `Path` (a real data pool) and hard-codes pool-specific prose into
`molmetal/reports/wfa163_c7_rederive/report.md` L149 ("0 rows in the pool have fewer than 2 heavy
atoms"), L150 ("17/18 epochs"), L152 ("The denominator is 12668, not 12669"). My general AST detector
finds **8** `--pool` scripts carrying a pool-specific literal (vs the branch's **4**). §8's
"4 producer scripts can emit false prose on a non-default pool" is **true**; "the tool-layer scan finds
4" is a **lower bound presented as a census**. Separately, §3 says the tool layer was scanned over
`molmetal/scripts/*.py` **+ `molmetal/**/scripts/*.py`** — the code globs only the former
(`scan_tools` L1043).

**(b) C12 — the proposed guard's `stale_cross_reference` rule has a recall hole.** `quoted` is built
from `int_tokens(window) + float_tokens(window)` and `_consistent_with` (guard L659-670) accepts an
**int** token that equals **any** numeric leaf of the cited artifact. So a stale quote is silently
accepted whenever its *integer part* collides. Injected into the real published citation
(`cfm_fragmentation_audit.json -> /summary/generation_side/single_molecule_rate_end_to_end`, whose
cited value is `{'n': 9, 'min': 0.0, 'max': 0.442…, 'mean': 0.1766…, 'median': 0.08506944444444445}`):

| quoted | findings | should be |
|---|---|---|
| `= 0.07777777777777778` | **0** | 1 (median is 0.085069…) |
| `= 9.5` | **0** | 1 (n = 9) |
| `= 0.5` | **0** | 1 |
| `= 1.7777777777777778` | 1 | 1 |

The quoted value "0.085069…" is tokenised as int `0` **and** float `0.0850…`; the int `0` matches the
cited `min: 0.0`, so the finding is suppressed. The class this rule exists for quotes *rates*. Its
denominator is also **3 citation sites in the whole published tree**, not 583 — §4b shows
"stale_cross_reference | 0 | 0" with no denominator. (My own strict checker, no integer-part
shortcut, over all 583 artifacts: 3 citations, 0 mismatches — the "0 stale" verdict survives on the
current tree, but the rule that produces it is weak.) Also: the gate has ≈0 recall for the Wf-A155
class itself — the preserved **pre-repair c8** artifact yields `pool_misattribution = 0`, because it is
self-consistent for its own pool (§6 admits this for the census; §9.3 reads as a generalisation).

**(c) C7 — every §6 file:line citation is stale.** All 15 cited lines now hold unrelated content. The
four armed scripts were rewritten **after** the branch's 11:15 report — `wf_a155_posebusters_validity.py`
11:37, `wf_a155_metal_coordination_sphere.py` 11:50, `wfa142_fragflex_cminus1_probe.py` 11:28,
`wfa156_bond_threshold_frontier.py` 11:40 — by Wf-A164, which says so in its own docstrings
("Wf-A164 (2026-11-01): this block used to be four hard-coded strings", L342-347 / L449-455); the
prose is now computed (`_pool_phrase`, L402-427). `build_artifact (def 913)` is now **def 1817**
(still taking `n_data_rows` / `records` / `epoch_labels`, L1822-1823, so the *substance* of "the counts
are already available" holds). The finding was TRUE when measured; the citations do not reproduce today.

**(d) C10 — the patch no longer applies.** `git apply --check` **exit 0** against HEAD's
`promote_metrics.py` (hunks at offsets -20/-36) — the branch's scratch-copy claim reproduces. Against
the **current** tree it is **exit 1**: the two "new" files report "already exists in working directory"
(they are the branch's own untracked outputs) and the `promote_metrics.py` hunk no longer matches —
that hunk **is** in the working-tree file (mtime 11:33, after the 11:15 report). The diff's file list
is exactly NEW guard + NEW test + EDIT `promote_metrics.py`: **no READ-ONLY path is touched**, and the
branch did not modify `promote_metrics.py` before writing its report.

**(e) C11 / C15 — two smaller overstatements.** (i) §9.5 says the test file pins "the two repaired
artifacts on disk score 0"; only **one** is asserted (`test_repaired_a155_tool_artifact_passes_if_present`);
`grep 'ligand_connected_rate_c7\|wfa162'` in the test file → **0 hits**. The c7 zero lives in the §7
table, not in a test. (ii) The census covers **2 of the 5** `metrics/` subdirs; the branch's OWN scanner
over all **605** published metric JSONs (`by_paper_section` 3, `by_round` 15, `sota_comparisons` 3,
`by_metric/_audit` 1 excluded) gives **178 flagged / 427 clean / 497 flags** (NARRATIVE_NUMBER 300, not
285) — so "the published metrics tree" framing undercounts by 9 artifacts and 15 flags.

## 3. What survives (NOT refuted)

* Every number in §0/§3/§4/§4b/§5/§7 reproduces **exactly** from disk: 583/169/414/482, the rule split,
  the top-10 sample = **139/482 = 0.2884**, verdicts 106/33/0, the 12 tree-wide TRUE_POSITIVE flags
  (3 in the sample), guard 0/0/19/474, and §5's numeric-leaf examples (429 / 458 / 0; 59/175 → True).
* The branch **admits** its precision (0 % on the strong class, ~24 % on the soft class), states the
  recall limitation of the counterfactual (4 of 7 sites), labels the counterfactual as NOT the original,
  and **refuses** the "the metrics tree is untrustworthy" claim in §8. No fabrication found.
* **No forbidden-path write**: the branch's footprint is 9 new untracked files; the 10:40-11:25 window's
  other touches are `registry/metrics.yaml` 10:40, `AGENTS.md` 10:42,
  `TODO/completed/321d_preflight_gate_audit.md` 10:42, `docs/HONEST_NEGATIVE_CATALOG.md` 10:45 (its diff
  attributes the Wf-A163 entries to **BRANCH V**), `wfa163_c7_rederive.py` 10:43 — other waves, not branch N.

## 4. Files

| Path | What |
|---|---|
| `molmetal/reports/wfa163_refute_ref1/verdict.json` | 15 claims, verdict + evidence + my independently measured number |
| `molmetal/reports/wfa163_refute_ref1/verdict.md` | this file |
| `molmetal/reports/wfa163_refute_ref1/branch_scanner_crosscheck.json` | my run of the branch scanner on the 4 cross-check targets |
| `molmetal/reports/wfa163_refute_ref1/full_metrics_tree_scan.json` | the branch scanner over all 605 metrics JSONs |
| `molmetal/reports/wfa163_refute_ref1/general_armed_detector.json` | my general armed-script detector (8 vs 4) |
| `molmetal/reports/wfa163_refute_ref1/strict_citation_check.json` | my strict cross-reference checker (3 citations, 0 mismatches) |
| `molmetal/scripts/wfa163_refute_ref1.py` | my tool: `armed` / `backing` / `cite` sub-commands (CPU-only) |
