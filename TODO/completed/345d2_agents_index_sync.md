# Wf-345-D2 — AGENTS.md + TODO/INDEX.md final sync (12 closeouts)

**Date:** 2026-09-29
**Wave:** Wf-345 ultracode residual closeout wave (D2 sub-flow)
**Status:** SHIPPED (docs-only sync; 0 NEW tests; 0 GPU launches)
**Owner:** Hugo Lee + Wf-345-D2 ultracode agent
**Cross-refs:** AGENTS.md §1.4 (pre-flight PASS) + §5.2 (honest framing) + §6 (recent Wf history pointer — this entry); TODO/INDEX.md §22.42 (this entry); `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED framing)

---

## §1 Problem statement

Per Wf-345-D2 task brief, the Wf-345 ultracode wave produced **12 closeouts on disk** (`A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+C4+D1` at `TODO/completed/345*` per `ls -la TODO/completed/ | grep -E "345[a-z][0-9]_" | sort` 2026-09-29 17:14 CST), but the prior `§6` row 254 + the file-list-only mentions in earlier sync waves covered only **5 closeouts** (`A1+A2+A3+A4+B2`). The 7 missing closeouts — `B1 + B3 + C1 + C2 + C3 + C4 + D1` — needed §22-layer INDEX entry + AGENTS.md §6 row extension.

Wf-345-D2 closes this documentation gap with:

1. **AGENTS.md §6 row 254 extension** — Wf-345 header expanded from "5 closeouts A1+A2+A3+A4+B2" to "12 closeouts A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+C4+D1" with detailed per-phase verdicts (Phase A1+A2+A3+A4 gate fixes, Phase B1 scaffold regression, Phase B2+B3 code-review findings, Phase C1+C2+C3+C4 anchor refreshes, Phase D1 pre-flight verify).
2. **TODO/INDEX.md §22.42 NEW entry** — 12-row per-closeout verdict table + aggregate Wf-345 totals + 7-entry honest-negative catalog + 4-row pre-flight delta + 5-row net-effect table + cross-link.
3. **MEASURED test pass-rates updated** — `Wf-345 = 18 NEW + 2 UPDATED tests PASS (CPU-only)` (was `6 NEW + 1 UPDATED` in the prior §6 row).
4. **Wf-345 closeout file list updated** at AGENTS.md §6 source column — `TODO/completed/345{a1,a2,a3,a4,b1,b2,b3,c1,c2,c3,c4,d1}_*.md` (was `{a1,a2,a3,a4,b2}_*.md`).

---

## §2 Changes shipped (this D2 sync)

### §2.1 AGENTS.md §6 row 254

**File:** `/home/hugo/codes/mol-metal/AGENTS.md` line ~254

Three Edits applied:

1. **Wf-345 header expanded** — "(5 closeouts A1+A2+A3+A4+B2 at `TODO/completed/345{a1,a2,a3,a4,b2}_*.md`; A1 fix ... + B2 `pre_flight_check.py` Wf-321D docstring dedup ...)" → "(12 closeouts A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+C4+D1 at `TODO/completed/345{a1,a2,a3,a4,b1,b2,b3,c1,c2,c3,c4,d1}_*.md`; user directive ... same as Wf-330/335/336/341/347/348/349/354 ultracode; **Phase A1+A2+A3+A4** (4 NEW Wf-343C4 gate fixes per wx63q8fwe ultracode review §3+§4): A1 ... + A2 ... + A3 ... + A4 ...; **Phase B1** (1 pre-existing regression fix): B1 ...; **Phase B2+B3** (2 code-review findings per wx63q8fwe §4): B2 ... + B3 ...; **Phase C1+C2+C3+C4** (4 stale anchor refreshes per Wf-345-A3 audit §2.2): C1 ... + C2 ... + C3 ... + C4 ...; **Phase D1**: Wf-345-D1 pre-flight fresh-run verification ... 3 FAILs escalated to Wf-346 follow-up ... 1 SKIP = `data_2_1_6_bindingdb_external_honest_negative` PRESERVED per AGENTS.md §5.4)".

2. **File list at end of §6 row 254** — "+ Wf-345 closeouts at `TODO/completed/345{a1,a2,a3,a4,b2}_*.md`" → "+ Wf-345 closeouts at `TODO/completed/345{a1,a2,a3,a4,b1,b2,b3,c1,c2,c3,c4,d1}_*.md`".

3. **MEASURED test pass-rate line** — "Wf-345 = 6 NEW + 1 UPDATED test PASS (CPU-only);" → "Wf-345 = 18 NEW + 2 UPDATED tests PASS (CPU-only);".

### §2.2 TODO/INDEX.md §22.42 NEW entry

**File:** `/home/hugo/codes/mol-metal/TODO/INDEX.md` line ~4935

One Edit applied: appended new `### §22.42` section between existing `### §22.41` (Wf-353) and `## §23` (Wf-316B). Section structure follows the established pattern from §22.39 / §22.40 / §22.41 (per-closeout verdict table + aggregate totals + honest-negative catalog + pre-flight delta + net-effect table + cross-link).

### §2.3 TODO/completed/345d2_agents_index_sync.md (this file)

**File:** `/home/hugo/codes/mol-metal/TODO/completed/345d2_agents_index_sync.md` — NEW ≤1 page closeout doc.

---

## §3 Honest framing per `party-twenty-spirit` §3.5

**MEASURED today (2026-09-29, this Wf-345-D2 sync)**:

- AGENTS.md §6 row 254 extended (Wf-345 narrative now covers all 12 closeouts, was 5 closeouts)
- TODO/INDEX.md §22.42 appended between §22.41 (Wf-353) and §23 (Wf-316B)
- 0/0 NEW tests added (docs-only sync, per the brief's wall projection)
- 0/0 GPU launches performed (CPU-only docs sync per §5.1 + AGENTS.md standing directives)
- 0/0 NEW pre-flight gates added (no code change; no gate change; EXPECTED_GATE_COUNTS["all"] 93 PRESERVED at Wf-353-B1 baseline)
- 1 NEW §22 INDEX entry appended + 1 AGENTS.md §6 row extended + 1 NEW closeout doc (this file)

**MEASURED across the Wf-345 ship wave (prior to this D2 sync)**:

- 12 NEW closeouts SHIPPED (A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+C4+D1 per `ls -la TODO/completed/345*`)
- 18 NEW + 2 UPDATED tests added (A1=4 + A2=3 + A3=3 + A4=3 + B1=2 + B3=3 = 18 NEW; A2 + A4 each updated 1 existing test = 2 UPDATED; B2 + C1+C2+C3+C4 + D1 = 0 NEW)
- 1 GPU launch performed (Wf-345-C3 Wf-170 per-metal singletask refresh on RTX 5090 GPU-1, 9.47 min wall, within AGENTS.md §5.1 standing authorization)
- 0/0 NEW pre-flight gates added (rename + env-aware SKIP + dynamic compute only; EXPECTED_GATE_COUNTS["all"] 93 PRESERVED at Wf-353-B1 baseline)
- 4/4 stale anchors refreshed (Wf-121 / Wf-143 / Wf-170 / Wf-67 P4.3 — all under 0.01h post-refresh; bit-identical kcal/mol except C3 which has CUDA non-determinism drift)

**Honest-negative caveats preserved (per `party-twenty-spirit` §3.5 — 7 entries, fully cross-referenced in §22.42.3.3)**:

1. **Wf-345-C3 Wf-170 per-metal HONEST-NEGATIVE drift** (5/5 per-metal means drift > 1e-3 due to CUDA non-determinism; mean_auc_across_metals drift = -0.0007 within 1e-3 NOT flagged; no effect on Wf-124 P3 multi-task lift narrative)
2. **Wf-345-A3 anchor-freshness 24h STALE by design** (4/11 anchors STALE at audit-write time per AGENTS.md §1.4 STALE flag rule; pre-existing honest-negative preserved through Phase 3 refresh)
3. **Wf-345-D1 P2 `p2_tests_pass` argparse `%`-escape infra bug** (test-infrastructure issue, NOT a code regression; Wf-346 escalation: escape literal `%` as `%%` in argparse help text)
4. **Wf-345-D1 P3 wave-budget over-budget ledger gap** (NOT blocking pre-flight PASS for gates outside `wf343c4` pipeline; Wf-346 escalation: refresh budget ledger with current-quarter allocation or split Wf-347 cohort into smaller sessions)
5. **Wf-345-D1 P3 wf343c2 strict-mode assertion noise** (NOT blocking pre-flight PASS for gates outside `wf347a1` pipeline; Wf-346 escalation: investigate `--strict` assertion surface for Wf-347A1)
6. **Wf-345-D1 P1 BindingDB external preserved** (SKIP per AGENTS.md §5.4 + Wf-322D §3 — documented honest-negative, NOT a regression)
7. **Wf-143 chemotype VDW collapse-to-Mg-fallback preserved** (Vina 1.2.7 cannot natively accept `Pt` atype per §22.23.5 / Wf-333 / Wf-335-SYN; Vina acceptance of `Pt` BLOCKED on Vina upstream)

---

## §4 Pre-flight gate status

| Pipeline | Pre-Wf-345 | Post-Wf-345-D2 | Source |
|---|---:|---:|---|
| `EXPECTED_GATE_COUNTS["all"]` | 93 (Wf-353-B1 baseline) | **93** | `molmetal/scripts/pre_flight_check.py:3117` (unchanged; Wf-345-D2 is docs-only) |
| `EXPECTED_GATE_COUNTS["frontier"]` | 25 | 25 | unchanged |
| `EXPECTED_GATE_COUNTS["chemprop_v2_v3"]` | 14 | 14 | unchanged |
| `EXPECTED_GATE_COUNTS["cfm"]` | 6 | 6 | unchanged |

**Pre-flight v3 fresh-run at Wf-345-D1 timestamp (2026-09-29T09:11:36Z)**: **90 PASS / 3 FAIL / 1 SKIP / 94 TOTAL** wall 147.32 s (per `molmetal/reports/wf345d1_preflight_freshrun.txt`). The 3 FAILs are Wf-346 follow-up scope (per §3 entries 3-5).

---

## §5 Wall + tests accounting

| Item | Value |
|---|---|
| Wall (CPU-only docs sync) | PROJECTED 5-10 min; ACTUAL ~3 min (3 Edits on AGENTS.md + 1 Edit on TODO/INDEX.md + 1 NEW closeout doc) |
| Tests | 0 NEW (docs-only sync per the brief) |
| GPU launches | 0 (CPU-only docs sync per §5.1 + AGENTS.md standing directives) |
| Pre-flight gate count delta | 0 (docs-only; EXPECTED_GATE_COUNTS["all"] 93 PRESERVED at Wf-353-B1 baseline) |
| AGENTS.md §6 row 254 delta | +1 phase (5 closeouts → 12 closeouts) + per-phase detailed verdicts (Phase A1+A2+A3+A4 / Phase B1 / Phase B2+B3 / Phase C1+C2+C3+C4 / Phase D1) + honest framing line updated |
| TODO/INDEX.md §22.42 delta | NEW entry appended (~115 lines; 12-row verdict table + 5-row aggregate + 7-entry caveats + 4-row pre-flight delta + 5-row net-effect + cross-link) |
| Files touched | AGENTS.md (3 Edits) + TODO/INDEX.md (1 Edit append) + TODO/completed/345d2_agents_index_sync.md (NEW) |

---

## §6 Cross-references

- `AGENTS.md §6 row 254` — Wf-345 extended narrative (12 closeouts, was 5 closeouts) — this D2 sync
- `TODO/INDEX.md §22.42` — NEW INDEX entry with 12-row per-closeout verdict table — this D2 sync
- `TODO/completed/345{a1,a2,a3,a4,b1,b2,b3,c1,c2,c3,c4,d1}_*.md` — 12 Wf-345 closeouts — primary sources
- `molmetal/scripts/pre_flight_check.py:2940-3131` — KEY_ALIASES + _resolve_key per Wf-345-A1
- `molmetal/scripts/pre_flight_check.py:3072` — `gate_wf343c4_yuelbond_decoder_env` per Wf-345-A2
- `molmetal/scripts/pre_flight_check.py:561` — `gate_scaffold_zero_leak` with stderr capture per Wf-345-B1
- `molmetal/scripts/pre_flight_check.py:3337-3522` — `gate_wf343c4_wave_budget_remaining_nonneg` 3-way verdict per Wf-345-A4
- `molmetal/scripts/pre_flight_check.py:3033` — EXPECTED_GATE_COUNTS dict retained for backward compat per Wf-345-B3
- `molmetal/scripts/pre_flight_check.py:5687` — `gate_count_consistency_audit` signature with `Optional` expected_total per Wf-345-B3
- `molmetal/reports/wf345a3_anchor_audit/audit.json` — anchor audit artifact per Wf-345-A3
- `molmetal/reports/wf345d1_preflight_freshrun.txt` — pre-flight fresh-run JSON output per Wf-345-D1
- `molmetal/tests/test_wf345{a1,a2,a3,a4,b1,b3}_*.py` — 6 NEW test files; 18 NEW tests total per Wf-345
- `molmetal/tests/test_wf343c4_integration.py` — 2 UPDATED existing tests per Wf-345-A2 + Wf-345-A4
- `configs/cfm_full_retrain.yaml:73` — NEW `train.batch_size: 16` per Wf-345-A1
- AGENTS.md §1.4 pre-flight PASS gate + §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar
- `party-twenty-spirit` §3.5 (honest framing gate)
- TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table — this entry's §22.42 home)

---

**END OF Wf-345-D2 SHIP SURFACE** — Owner Hugo Lee + Wf-345-D2 ultracode agent; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.2 (honest framing); last update 2026-09-29.