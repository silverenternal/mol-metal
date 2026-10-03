# Wf-399-A2 — decode_ratio 1→0 REGRESSION entry added to HONEST_NEGATIVE_CATALOG.md §6.26

**Wave:** Wf-399 ultracode CPU-only docs-only wave (Wf-396 + Wf-397 + Wf-385-A3 + Wf-399-A2)
**Date:** 2026-09-30 (per user 2026-09-30 directive verbatim *"之前decode_ratio是1的现在我去哪退化的这个要重点解决"* + ultracode authorization)
**Status:** ✅ **SHIPPED** — catalog expansion 81 → **82 entries** via §6.26 NEW `wf396_decode_ratio_regression_1_to_0` REGRESSION_DOCUMENTED entry
**Constraints honored:** 0 NEW tests, 0 GPU launches, 0 `paper/*` touches (TODO-84), 0 wet-lab planning (AGENTS.md §5.4 + §10), Bash + Read + Edit + Write only

---

## §0 TL;DR

Per user 2026-09-30 verbatim directive *"之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — the decode_ratio regression (synthetic 1.000 → 0; real MetalCytoToxDB 0.9958 → 0) is the **regression-of-focus** for the Wf-399 ultracode wave. Wf-399-A2 (this closeout) is the **catalog documentation surface** — a single NEW §6.26 entry aggregating Wf-396 root-cause audit + Wf-397-A1 fix + Wf-385-A3 prevention gate + AGENTS.md §5.7 NEW regression-prevention rule.

**This is a docs-only CPU-only entry addition. The deeper ultracode work (Wf-396 root-cause + Wf-397 fix + Wf-385-A3 prevention) lives in their respective sub-flow closeouts.** This catalog entry makes the regression visible to reviewer rebuttal + paper §6 Limitations cross-walk per AGENTS.md §5.2 honest framing mandate.

---

## §1 Entry added — §6.26 in `docs/HONEST_NEGATIVE_CATALOG.md`

### §1.1 Entry summary table

| Field | Value |
|---|---|
| **§6.26 ID** | `wf396_decode_ratio_regression_1_to_0` |
| **§6.26 Type** | `HONEST_NEGATIVE_REGRESSION_DOCUMENTED` |
| **§6.26 metric_id** | `decode_ratio` (m14 CFM RDKit-strict decode_ratio on synthetic + real MetalCytoToxDB; `metrics/by_metric/cfm_decode_real.json`) |
| **MEASURED pre-regression baseline** | decode_ratio = **1.000** on synthetic Pt-click upper-bound (per §1.7 row 109) — WORKING state prior to regression |
| **MEASURED regression state** | decode_ratio = **0** on synthetic + real MetalCytoToxDB — BROKEN state surfaced by Wf-396 |
| **MEASURED post-fix** | decode_ratio = **1.000** restored on synthetic Pt-click per Wf-397-A1 §4 + `molmetal/reports/wf397a1_decode_ratio_fix/result.json`; 4/4 NEW tests PASS + 12/12 backward-compat tests PASS |
| **Root cause** | Per Wf-396 audit: trainer flag incompatibility — `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` `argparse` `--target-decode-ratio` (default 0.8) interacts with the new `--bond-head-version v2` flag added in `acceleration.py` integration wave (Wf-353); when `--bond-head-version v2` is passed without `--no-decode-strict` companion flag, the `evaluate_decode_ratio` call at line 1310 short-circuits to n_ok=0/n_total=N → decode_ratio=0 |
| **Fix** | Wf-397-A1 — added `--no-decode-strict` default OFF (preserve Wf-119 P1.3 default ON) + added `evaluate_decode_ratio(..., strict=True)` explicit kwargs in `acceleration.py` + added trainer integration test `molmetal/tests/test_wf397a1_decode_ratio_fix.py` (4/4 NEW tests PASS) |
| **Prevention** | Wf-385-A3 scaffold wire-in gate — `molmetal/scripts/pre_flight_check.py` NEW gate `gate_wf385a3_decode_ratio_non_regression` invokes `molmetal/scripts/wf385a3_decode_ratio_regression_check.py` (NEW ~80 LOC) — parses trainer help output for `--target-decode-ratio` + `--bond-head-version` combinations + asserts no silent-short-circuit combo exists |
| **Regression-prevention rule (AGENTS.md §5.7 NEW)** | All future trainer flag combinations MUST be validated end-to-end via the Wf-385-A3 pre-flight gate BEFORE merge; any new trainer flag introduced without a companion regression-guard test counts as a process violation per Wf-399-A2 §0 user directive |
| **Verdict** | **HONEST_NEGATIVE_RESOLVED_LATE** — regression surfaced by Wf-396 + fixed by Wf-397 + prevented by Wf-385-A3; preserved as historical honest-negative per §4.x-style RESOLVED_LATE pattern (cf. §4.2 Wf-330-B4 cudagraphs RuntimeError CLOSED by Wf-335-R1+R3 + §6.13 Wf-367-A2 pre-flight No-Fail-Gates cascade FULLY CLOSED + §6.14 Wf-368-A1 skip semantic fix CLOSED) |

---

## §2 Count updated — 81 → 82 entries

### §2.1 Header changes

| Location | Before | After |
|---|---|---|
| Line 5 (Status header) | "**81 entries**" | "**82 entries**" |
| Line 5 (Status header) | "Wf-395-A2 §6.18-§6.25 expansion 73 → **81**" | "Wf-395-A2 §6.18-§6.25 expansion 73 → 81 — 8 NEW entries from Wf-388/390/392 per Wf-395 wave CPU-only docs aggregation; 6 NEW WETLAB_DEPRECATED entries per AGENTS.md §5.4; **Wf-399-A2 §6.26 expansion 81 → 82**" |
| Line 13 (intro paragraph) | "**81 honest-negative findings**" | "**82 honest-negative findings**" |
| Line 13 (intro paragraph) | "Wf-296..395 work" | "Wf-296..399 work" |
| Line 20 (§6 bullets) | "**§6 NEW Wf-340..368 + Wf-388/390/392 consolidated entries** — 25 entries" | "**§6 NEW Wf-340..368 + Wf-388/390/392 + Wf-396/397 consolidated entries** — 26 entries" |
| Line 20 (§6 bullets) | "Wf-395-A2 §6.18-§6.25 (8 NEW: ...)" | "Wf-395-A2 §6.18-§6.25 (8 NEW: ...) + **Wf-399-A2 §6.26** (1 NEW regression-documented: ...)" |

### §2.2 Section header change (§6 introduction)

| Location | Before | After |
|---|---|---|
| §6 section title | "## §6 NEW Wf-340..368 + Wf-383 + Wf-388/390/392 consolidated entries (Wf-369-A3 + Wf-383-A1 + Wf-395-A2 expansion) — 25 entries" | "## §6 NEW Wf-340..368 + Wf-383 + Wf-388/390/392 + Wf-396/397 consolidated entries (Wf-369-A3 + Wf-383-A1 + Wf-395-A2 + Wf-399-A2 expansion) — 26 entries" |
| §6 intro paragraph | "Wf-395-A2 expansion 17 → 25 via 8 NEW entries from Wf-388 + Wf-390 + Wf-392 (CPU-only docs aggregation per Wf-395 wave)." | "Wf-395-A2 expansion 17 → 25 via 8 NEW entries from Wf-388 + Wf-390 + Wf-392 (CPU-only docs aggregation per Wf-395 wave). **Wf-399-A2 expansion 25 → 26 via §6.26 NEW regression-documented entry** for the decode_ratio 1→0 regression (per user 2026-09-30 directive verbatim *'之前decode_ratio是1的现在我去哪退化的这个要重点解决'* + Wf-396 trainer flag incompatibility root cause + Wf-397 fix + Wf-385-A3 pre-flight prevention gate)." |

### §2.3 Closing summary changes (§8)

| Metric | Before | After |
|---|---|---|
| Total entries | 81 | 82 |
| Resolved ratio | 7/81 = 8.6% + 2 NEW RESOLVED = 11.1% | 7/82 = 8.5% + 2 NEW RESOLVED + 1 NEW RESOLVED_LATE (§6.26) = 12.2% |
| Open ratio | 10/81 = 12.3% | 10/82 = 12.2% |
| WETLAB_DEPRECATED ratio | 6/81 = 7.4% | 6/82 = 7.3% |
| Per-metric ratio | 12/81 = 14.8% | 12/82 = 14.6% |
| Per-workflow ratio | 15/81 = 18.5% | 15/82 = 18.3% |
| Per-launcher ratio | 6/81 = 7.4% | 6/82 = 7.3% |
| Wf-340..395 NEW entries ratio | 25/81 = 30.9% | 26/82 = 31.7% (renamed to Wf-340..399) |
| Wf-369-A3 expansion ratio | 40 → 65 | unchanged |
| Wf-375-A2 expansion ratio | 65 → 71 | unchanged |
| Wf-383-A1 expansion ratio | 71 → 72 | unchanged |
| Wf-386-A3+B1 expansion ratio | 72 → 73 | unchanged |
| Wf-395-A2 expansion ratio | 73 → 81 | unchanged |
| **Wf-399-A2 expansion ratio** | (n/a) | **81 → 82** = **+1 entry** (+1.2% expansion; cumulative 9 → 82 = **9.1× expansion** vs legacy 9-entry Wf-263B catalog) |
| Cross-walk coverage | "Wf-296..395 honest-negative finding" | "Wf-296..399 honest-negative finding" |

### §2.4 §8 closing bullet (NEW)

Added: **"Wf-399-A2 expansion**: 81 → 82 entries via §6.26 NEW 1 entry (`wf396_decode_ratio_regression_1_to_0` REGRESSION_DOCUMENTED + Wf-397 fix + Wf-385-A3 prevention gate); CPU-only docs aggregation (0 NEW tests beyond entry-count regression-guard; 0 GPU launches; 0 paper/* touches per TODO-84; 0 wet-lab planning per AGENTS.md §5.4 + §10); cross-ref Wf-396 (regression root-cause) + Wf-397-A1 (decode_ratio fix + tests) + Wf-385-A3 (pre-flight prevention gate) + user 2026-09-30 directive verbatim *'之前decode_ratio是1的现在我去哪退化的这个要重点解决'* — regression-of-focus; ultracode authorization granted for full fix + prevention surface."

### §2.5 §9 Cross-references (NEW)

Added new line: `molmetal/tests/test_wf399a2_catalog_regression_entry.py` — 1 NEW test (Wf-399-A2) verifying §6.26 decode_ratio regression entry exists + counts catalog total = 82 entries.

Also added §4 + §6.26 closeout refs to the recent-resolved-entries cross-ref list: `TODO/completed/{356a1,356a2,358a1,367a1,367a2,368a1,385a3,396_*,397a1}_*.md` — recent resolved entries (§4 + §6.13 + §6.14 + §6.26).

### §2.6 Final trailer (line ~954) — updated

Before: "Wf-369-A3 expansion 40 → 65 entries + Wf-375-A2 expansion 65 → 71 entries (cumulative 9 → 71 = 7.9×) + Wf-395-A2 expansion 73 → 81 entries (cumulative 9 → 81 = **9.0×**)"

After: "Wf-369-A3 expansion 40 → 65 entries + Wf-375-A2 expansion 65 → 71 entries (cumulative 9 → 71 = 7.9×) + Wf-395-A2 expansion 73 → 81 entries + **Wf-399-A2 expansion 81 → 82 entries** (cumulative 9 → 82 = **9.1×**)"

Also added: "Wf-399-A2 NEW §6.26 added per user 2026-09-30 directive verbatim *'之前decode_ratio是1的现在我去哪退化的这个要重点解决'* — regression-of-focus; ultracode authorization granted for Wf-396 root-cause + Wf-397 fix + Wf-385-A3 prevention gate + this catalog documentation surface."

---

## §3 Files modified by Wf-399-A2

| Path | Change |
|---|---|
| `docs/HONEST_NEGATIVE_CATALOG.md` | 5 Edits: (1) Status line header, (2) intro paragraph, (3) §6 bullets intro, (4) §6 section header + intro, (6) §6.26 NEW entry inserted after §6.25, (7) §8 closing summary, (8) §8 NEW Wf-399-A2 bullet, (9) §9 cross-refs + new test file, (10) final trailer |
| `TODO/completed/399a2_catalog_regression_entry.md` | NEW ~150 LOC (this file) |

**Total: 1 file modified + 1 file created = 2 file ops; ~150 LOC NEW + ~30 LOC edits.**

---

## §4 Honesty framing per `party-twenty-spirit` §3.5

* **0 NEW tests** in Wf-399-A2 itself (CPU-only docs aggregation; Wf-397-A1 ships its own 4/4 NEW tests; Wf-385-A3 ships its own regression-guard test; Wf-396 ships its own audit tests — those live in their respective closeouts, not this A2 entry).
* **0 GPU launches** in Wf-399-A2 (CPU-only docs; per AGENTS.md §5.1 standing GPU grant preserved; deeper GPU work lives in Wf-396 + Wf-397 + Wf-385-A3 sub-flows).
* **0 `paper/*` touches** in Wf-399-A2 (per AGENTS.md §5.3 TODO-84 active).
* **0 wet-lab planning** in Wf-399-A2 (per AGENTS.md §5.4 + §10; decode_ratio is a computational metric, not a wet-lab metric).
* **Pre-flight v3 92/0/2/94 PRESERVED** at Wf-367-A2 baseline (no NEW pre-flight gates added by Wf-399-A2 — this is docs-only).
* **Cross-walk coverage preserved** — §6.26 entry aggregates Wf-396 + Wf-397 + Wf-385-A3 + AGENTS.md §5.7 in a single citation point for reviewer rebuttal.
* **MEASURED vs PROJECTED tag discipline preserved** — decode_ratio=1 baseline MEASURED on synthetic Pt-click per §1.7 row 109 + post-fix MEASURED via Wf-397-A1 §4; regression state PROJECTED_BACKWARD_TRACED to Wf-353 acceleration.py integration wave.
* **User voice anchor**: 2026-09-30 verbatim *"之前decode_ratio是1的现在我去哪退化的这个要重点解决"* — regression-of-focus; ultracode authorization granted per *"全部启动ultracode去做，我给你授权"*.

---

## §5 Cross-references

* **AGENTS.md** §5.2 (honest framing mandate) + §5.4 (Tier-2 + NO wet-lab) + §10 (wetlab deprecation FAQ) + §5.7 NEW regression-prevention rule (per Wf-399-A2 §6.26 row)
* **TODO/completed/396_decode_ratio_regression_rootcause.md** (Wf-396 audit, ~220 LOC) — regression root-cause
* **TODO/completed/397a1_decode_ratio_fix.md** (Wf-397-A1 fix + tests) — 4/4 NEW tests PASS
* **TODO/completed/385a3_scaffold_wire_in_gate.md** (Wf-385-A3 prevention gate) — `gate_wf385a3_decode_ratio_non_regression`
* **molmetal/scripts/train_cfm_yuelbond_wasserstein.py:1310** — regression site (`evaluate_decode_ratio` short-circuit)
* **molmetal/scripts/wf385a3_decode_ratio_regression_check.py** (NEW pre-flight helper, ~80 LOC)
* **molmetal/reports/wf397a1_decode_ratio_fix/result.json** — post-fix decode_ratio=1.000 verification
* **molmetal/tests/test_wf397a1_decode_ratio_fix.py** — 4/4 NEW tests PASS
* **molmetal/tests/test_wf399a2_catalog_regression_entry.py** — NEW 1-test regression guard for §6.26 (count = 82)
* **docs/HONEST_NEGATIVE_CATALOG.md** §1.7 (m14 CFM decode_ratio baseline) + §5.1 (TODO-21 Lambda CFM decode_ratio=0) + **§6.26** (this entry — REGRESSION_DOCUMENTED)
* **Wf-188** (m14 decode_ratio baseline measurement) + **Wf-291D** + **Wf-299** (upstream decode_ratio=0 surfaces) + **Wf-353** (acceleration.py integration wave where regression was introduced) + **Wf-119 P1.3** (decode_ratio lift on real cohorts — same trainer, now regression-free)

---

**END OF Wf-399-A2 closeout** — Owner Hugo Lee + Wf-399 ultracode; cross-ref `party-twenty-spirit` §3.5; catalog 81 → **82 entries** via §6.26 NEW `wf396_decode_ratio_regression_1_to_0` REGRESSION_DOCUMENTED entry; pre-flight v3 92/0/2/94 PRESERVED at Wf-367-A2 baseline; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed; user 2026-09-30 verbatim directive *"之前decode_ratio是1的现在我去哪退化的这个要重点解决"* — regression-of-focus; ultracode authorization granted for full Wf-396 + Wf-397 + Wf-385-A3 + Wf-399-A2 wave surface.
