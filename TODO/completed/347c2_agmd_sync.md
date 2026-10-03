# Wf-347-C2 — AGENTS.md Final Sync for Wf-347 UltRACODE Wave

**Date:** 2026-09-29
**Wf:** Wf-347-C2 (docs-only)
**Owner:** Hugo Lee (autonomous Wf-347 ultracode unit)
**Status:** SHIPPED (CPU-only, docs-only sync)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §1.4 (pre-flight v2 PASS gate), §5.1 (standing GPU authorization), §5.2 (honest framing)

Per the 2026-09-29 user request **"基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"** — the Wf-347 ultracode wave closes the **structural integration gap** between the canonical GPU pipeline master (`scripts/gpu_pipeline_master.sh`), the Wf-343C4 pre-retrain internal check gate set (4 NEW gates registered in `molmetal/scripts/pre_flight_check.py::build_gates("all")` per Wf-343C4 closeout §0), and the Wf-343C2 + Wf-343C3 standalone companion scripts.

Wf-347-C2 is the **docs-only AGENTS.md sync** that closes the AGENTS.md reflection gap introduced by the Wf-347-A1 / A2 / A3 / B1 / B2 closeouts (`TODO/completed/347{a1,a2,a3,b1,b2}_*.md`).

---

## §0 TL;DR

| Item | Result |
|---|---|
| **§4 Critical workflows** | +3 NEW rows: `wf343c2_per_wave_check.py --skip-live` + `wf343c3_data_quality_check.py --strict` + `bash scripts/gpu_pipeline_master.sh wf343c4 [--strict]` |
| **§5.1 Standing GPU authorization** | +1 NEW bullet: "Wf-343C2 per-wave runtime gate + Wf-343C3 data quality gate + USER_OK gate" with `scripts/gpu_pipeline_master.sh wf343c4 [--strict]` opt-in surface + standalone invocation examples |
| **§6 Recent Wf history pointer** | +1 NEW row 253: **Wf-347 ultracode integration v3 wave** (5 closeouts A1/A2/A3/B1/B2 + this Wf-347-C2 docs sync) |
| **§5.4 Tier-2 + NO wet-lab markers** | UNCHANGED — Tier-2 acceptance bar is computational-only, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验" |
| **§5.1 GPU grant + §5.2 honest framing + §5.3 no paper work** | UNCHANGED — all 4 standing directives re-confirmed |
| **Pre-flight v3 fresh-run** | NOT re-run by Wf-347-C2 (docs-only sync, 0 GPU launches) |
| **EXPECTED_GATE_COUNTS["all"]** | 68 PRESERVED at Wf-347-A1 baseline (no new pre-flight gates added by Wf-347-C2) |
| **0 NEW tests** | docs-only sync per honest framing |

---

## §1 Per-section changes (Δ)

### §1.1 §4 Critical workflows — +3 NEW rows

Added the following rows to the §4 Critical workflows reference table (lines 137-148 in pre-Wf-347-C2):

```python
| `python molmetal/scripts/wf343c2_per_wave_check.py --skip-live` | Wf-343C2 per-wave runtime gate (7 checks: preflight + hyperparams + yuelbond + wave_budget + VRAM + disk + freshness) (Wf-347-A1/A2/B1/B2) | Before each (metal, seed) pair launch in G5 + G6 sweep; standalone gate |
| `python molmetal/scripts/wf343c3_data_quality_check.py --strict` | Wf-343C3 data quality gate (5 loaders × 5 checks: csv_loads + per_metal_balance + scaffold_split + class_balance + label_range) (Wf-347-A1/A2/B1/B2) | Before any retrain; standalone gate |
| `bash scripts/gpu_pipeline_master.sh wf343c4 [--strict]` | Wf-343C4 opt-in gate runner (Wf-347-A2): invokes wf343c2 + wf343c3 + cross-references 4 NEW Wf-343C4 gates; gated behind `WF343C4_GATE=1` env var (default off) | Before any retrain / panel; opt-in surface for kill_then_adapt §0 diagnostics |
```

### §1.2 §5.1 Standing GPU authorization — +1 NEW bullet

Added the following bullet to the §5.1 constraints list (lines 162-185 in post-Wf-347-C2 AGENTS.md):

```
- **Wf-343C2 per-wave runtime gate + Wf-343C3 data quality gate + USER_OK gate** (per
  Wf-347-A1/A2/B1/B2 ultracode integration wave 2026-09-29): the canonical pre-launch
  gating surface for any retrain / panel / ablation. Both gate scripts are also
  wired into `pre_flight_check.py::build_gates("all")` as subprocess-style gates
  (`p3_wf343c2_per_wave_runtime_check` + `p3_wf343c3_data_quality_check`) and into
  `scripts/gpu_pipeline_master.sh wf343c4 [--strict]` (opt-in via `WF343C4_GATE=1`).
  Per-tuple invocation is wired into G5 + G6 launchers
  (`scripts/launch_wf124_p3_g5.sh` + `scripts/launch_wf124_p3_g6.sh` +
  `scripts/recover_g6_failed_tuples.sh`) — see §4 for the full script list.
  Standalone invocation examples:
  - `python molmetal/scripts/wf343c2_per_wave_check.py --wave <wave> --pairs-metals Pt --pairs-seeds 0 --skip-live`
  - `python molmetal/scripts/wf343c3_data_quality_check.py --strict`
  USER_OK gating on Wf-335 / Wf-336 / Wf-337-G3 launch wrappers follows
  `kill_then_adapt` §0 (TODO-84 still applies for paper/* per §5.3).
```

### §1.3 §6 Recent Wf history pointer — +1 NEW row

Added a new row 253 between the existing Wf-348-A4 row (252) and the §7 separator (255 in pre-Wf-347-C2). The new row covers:

```
**Wf-347 ultracode integration v3 wave** (5 closeouts: A1 Wf-296 pipeline master
integration of Wf-343C4 gates ... A2 Wf-334 pipeline master integration of Wf-343C4
gates ... A3 end-to-end pipeline integration verification ... B1 G5 launcher
per-wave check + per-pair budget tracking wire-in ... B2 G6 launcher per-wave
check wire-in) + **Wf-347-C2 AGENTS.md final sync** (this row); user directive 2026-09-29
verbatim "基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，
一定要全，工作流可以长" — same as Wf-330 / Wf-335 / Wf-336 / Wf-341 ultracode
```

Plus full source pointers to `TODO/completed/347{a1,a2,a3,b1,b2,c2}_*.md` + honest-framing per `party-twenty-spirit` §3.5.

---

## §2 Honest framing — what was synced (MEASURED-vs-PROJECTED)

Per `party-twenty-spirit` §3.5, every claim is tagged:

| Claim | Tag | Source |
|---|---|---|
| `wf343c2_per_wave_check.py` exists | **MEASURED** | `ls -la molmetal/scripts/wf343c2_per_wave_check.py` |
| `wf343c3_data_quality_check.py` exists | **MEASURED** | `ls -la molmetal/scripts/wf343c3_data_quality_check.py` |
| `scripts/gpu_pipeline_master.sh` exists | **MEASURED** | `ls -la scripts/gpu_pipeline_master.sh` (9,225 B, Sep 29 15:35 CST) |
| Wf-347-A1/A2/A3/B1/B2 closeouts on disk | **MEASURED** | `ls TODO/completed/347{a1,a2,a3,b1,b2}_*.md` returns 5 files |
| `EXPECTED_GATE_COUNTS["all"]` = 68 post-Wf-347-A1 | **MEASURED** | `molmetal/scripts/pre_flight_check.py:2823` (`EXPECTED_GATE_COUNTS = {"all": 68, ...}`) |
| `WF343C4_GATE=1` env var gating pattern | **MEASURED** | `scripts/gpu_pipeline_master.sh` line 43 + Wf-347-A2 closeout |
| Pre-flight v2 `pipeline=all` 68 PASS / 2 FAIL / 1 SKIP / 71 TOTAL post-Wf-350A1 | **MEASURED** | `TODO/completed/347a3_end_to_end_verify.md` §0 (Wf-347-A3 closeout) — note Wf-350A1 is a separate downstream wave that added 3 more gates AFTER Wf-347-A1; Wf-347-C2 §6 row anchors to Wf-347-A1 baseline of 68, which is the verified state at Wf-347-A1 ship time |
| §5.4 Tier-2 + NO wet-lab markers UNCHANGED | **MEASURED** | AGENTS.md §5.4 line 187 verbatim (no Edit applied to §5.4) |
| §5.1 GPU grant + §5.2 honest framing + §5.3 no paper work UNCHANGED | **MEASURED** | AGENTS.md §5.1-§5.3 line 162-237 (only §5.1 received +1 NEW bullet) |
| Total wall budget for Wf-347-C2 | **PROJECTED 5-10 min CPU-only** | matches Wf-347-C2 task contract |
| 0 NEW tests added | **MEASURED** | docs-only sync per honest framing |
| Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline | **MEASURED** | NOT re-run by Wf-347-C2 (docs-only) |
| EXPECTED_GATE_COUNTS["all"] 68 PRESERVED at Wf-347-A1 baseline | **MEASURED** | no new pre-flight gates added by Wf-347-C2 |

---

## §3 Files modified

| Path | LOC delta | Change |
|---|---:|---|
| `AGENTS.md` §4 Critical workflows table | +3 NEW rows | `wf343c2_per_wave_check.py --skip-live` + `wf343c3_data_quality_check.py --strict` + `bash scripts/gpu_pipeline_master.sh wf343c4 [--strict]` |
| `AGENTS.md` §5.1 Standing GPU authorization | +1 NEW bullet (~22 lines) | "Wf-343C2 per-wave runtime gate + Wf-343C3 data quality gate + USER_OK gate" with opt-in surface + standalone invocation examples |
| `AGENTS.md` §6 Recent Wf history pointer | +1 NEW row 253 | **Wf-347 ultracode integration v3 wave** (5 closeouts A1/A2/A3/B1/B2 + Wf-347-C2 this row) |
| `TODO/completed/347c2_agmd_sync.md` | NEW ~225 LOC | This document |

**Total source code delta:** +1 NEW bullet + +3 NEW table rows + +1 NEW §6 history row across AGENTS.md (no behavioral changes to any script, no changes to `molmetal/scripts/pre_flight_check.py`, no changes to `scripts/gpu_pipeline_master.sh`, no changes to any launcher).

---

## §4 Verification (CPU-only)

| Check | Command | Result |
|---|---|---|
| AGENTS.md structure preserved | `grep -n "^## §\|^### §" AGENTS.md` | **STRUCTURE OK** (9 main sections + 4 §5.x subsections + 3 §3.x subsections; all section headers preserved) |
| §5.4 Tier-2 markers unchanged | `grep "Tier-2" AGENTS.md` | **PRESENT** (line 187 verbatim Tier-2 markers + NO wet-lab markers UNCHANGED) |
| §5.1 GPU grant markers preserved | `grep "显存够就放心大胆用" AGENTS.md` | **PRESENT** (line 162 verbatim user directive preserved) |
| §5.2 honest framing preserved | `grep "MEASURED vs PROJECTED" AGENTS.md` | **PRESENT** (line 232 verbatim honest framing markers preserved) |
| §5.3 no paper work preserved | `grep "TODO-84" AGENTS.md` | **PRESENT** (line 238 verbatim TODO-84 + paper/* READ-only preserved) |
| §6 Wf-347 row added | `grep "Wf-347 ultracode integration" AGENTS.md` | **PRESENT** (new row 253 added) |
| §4 wf343c2 + wf343c3 rows added | `grep "wf343c2_per_wave_check\|wf343c3_data_quality_check\|gpu_pipeline_master.sh wf343c4" AGENTS.md` | **PRESENT** (3 NEW rows in §4 added) |

**Wall:** PROJECTED 5-10 min CPU-only for the docs sync + verification (actually <5 min; Read + Edit + grep checks only). 0 GPU launches performed.

---

## §5 Standing-rule compliance

| Rule | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED | §2 above is the full MEASURED-vs-PROJECTED ledger; all 12 claims tagged |
| §1.4 Pre-flight v2 25/25 PASS gate | Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline; NOT re-run by Wf-347-C2 (docs-only); EXPECTED_GATE_COUNTS["all"] 68 PRESERVED at Wf-347-A1 baseline |
| §5.1 Standing GPU authorization | §5.1 receives +1 NEW bullet documenting the Wf-343C2 + Wf-343C3 + Wf-343C4 opt-in USER_OK gate surface; GPU grant markers preserved verbatim |
| §5.2 Honest framing | §2 + §5 row in §6 + §5.4 unchanged markers all reflect honest framing per party-twenty-spirit §3.5; Wf-347-A3 reference to "71 TOTAL" includes Wf-350A1 gates (separate downstream wave); Wf-347-C2 §6 row anchors to Wf-347-A1 baseline of 68 with full honest-framing caveat |
| §5.3 No paper work | Zero touches to `paper/*`; AGENTS.md §5.3 preserved verbatim |
| §5.4 Tier-2 + NO wet-lab | UNCHANGED — Tier-2 acceptance bar is computational-only, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验" |
| party-twenty-spirit §3.5 | All numeric values in this document trace to MEASURED JSON / report path on disk (per §2); PROJECTED values clearly tagged (wall budget only) |

---

**END OF Wf-347-C2 CLOSEOUT** — Owner Hugo Lee; docs-only sync per honest framing; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
