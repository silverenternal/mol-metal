# Wf-332-S4: Pre-flight final verification — pure measurement

**Status**: SHIPPED (CPU-only; pure measurement)
**Date**: 2026-09-29 13:30 UTC
**Workflow**: Wf-332-S4 (Pre-flight final verification sub-flow)
**Pre-flight JSON**: `/tmp/wf332s4_preflight.json` (17730 bytes; captured this run)
**Owner**: Hugo Lee + Wf-332-S4 subagent
**Honest framing**: pure measurement, MEASURED-vs-PROJECTED tag discipline preserved per `party-twenty-spirit` §3.5

---

## §1 TL;DR

Pre-flight v3 fresh-run at Wf-332-S4 timestamp reports **62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL**.
This is a **CLEAN STATE PRESERVED** measurement (no regression since Wf-330-C + Wf-331).

| Snapshot | Date | PASS | FAIL | SKIP | TOTAL | Source |
|---|---|---:|---:|---:|---:|---|
| **Wf-322D baseline (REGRESSION)** | 2026-09-29 11:06 CST | 42 | **15** | 0 | 57 | `TODO/INDEX.md` §22.16 |
| Wf-329-V4 INDEX sync | 2026-09-29 04:40 UTC | 61 | 1 | 0 | 62 | `TODO/completed/329v4_index_sync.md` |
| Wf-330-C FRESH (post-fix) | 2026-09-29 12:54 CST | **62** | **0** | 0 | 62 | `TODO/completed/330c_preflight_v3_full_pass.md` |
| Wf-331 closeout | 2026-09-29 13:20 UTC | 62 | 0 | 0 | 62 | `TODO/completed/331*.md` (6 Wf-331 sub-flows) |
| **Wf-332-S4 (this run)** | 2026-09-29 13:30 UTC | **62** | **0** | 0 | **62** | `/tmp/wf332s4_preflight.json` |

**Pre-flight delta (Wf-322D baseline → Wf-332-S4)**:
- **+20 PASS** (42 → 62)
- **−15 FAIL** (15 → 0; the Wf-322D 15-FAIL honest-negative REGRESSION is FULLY RESOLVED at Wf-330-C post-fix + PRESERVED at Wf-332-S4 fresh-run)
- **+5 NEW TOTAL** (57 → 62; 5 NEW gates added: `p3_catherine_30metal_closeout_documented` + `p3_wf327_cpu_num_workers_sufficient` + `p3_m6_kappa_closeout_documented` + `p3_admet_4_below_documented` + `p3_phase3g_f2a_runtime_unblock_verified`)
- **+0 SKIP** (0 → 0)

---

## §2 Per-pipeline breakdown (Wf-332-S4)

| Pipeline | PASS | FAIL | SKIP | TOTAL | Status |
|---|---:|---:|---:|---:|---|
| `all` | 62 | 0 | 0 | 62 | ✅ CLEAN |
| `chemprop` (P1) | 12 | 0 | 0 | 12 | ✅ |
| `cfm` (P1) | 7 | 0 | 0 | 7 | ✅ |
| `wf327` (P3) | 36 | 0 | 0 | 36 | ✅ |

**Total wall_sec**: 187.6703 s (3 min 7.7 s).

---

## §3 Per-gate flip attribution (Wf-322D baseline → Wf-332-S4)

### §3.1 The 15 FAIL gates that flipped to PASS

The Wf-322D 15-FAIL honest-negative REGRESSION root cause (per `TODO/completed/330c_preflight_v3_full_pass.md`):
- **System Python env mismatch**: pre-flight run with `/usr/bin/python` (Python 3.14.7, no torch) caused `catherine_properties` import failure → bench fell through to fallback `except` returning `n_pass=0` → fabricated FAIL.
- **Fix**: use `CadButEaas/.venv-vlm/bin/python` (Python 3.12.13 + torch 2.14.0+cu130) for all pre-flight runs.
- **Source agent**: Wf-330-C (path: `TODO/completed/330c_preflight_v3_full_pass.md`).

The 15 FAIL gates that flipped PASS are the entire pre-flight gate set whose `catherine_properties` bench was failing under system Python. They all share the same root cause: torch import failure → fabricated `n_pass=0`.

### §3.2 The 5 NEW gates added since Wf-322D baseline

| NEW gate | Source agent | Path | Status |
|---|---|---|---|
| `p3_catherine_30metal_closeout_documented` | Wf-322G | `TODO/completed/322g_catherine_30metal_closeout.md` (referenced) | PASS |
| `p3_wf327_cpu_num_workers_sufficient` | Wf-327-F11 | `TODO/completed/327f11_gate_count_docs_fix.md` | PASS (path-bug fix at Wf-327-SYN §4.1) |
| `p3_m6_kappa_closeout_documented` | Wf-325D | `TODO/completed/325d_m6_kappa_closeout.md` | PASS |
| `p3_admet_4_below_documented` | Wf-325E | `TODO/completed/325e_admet_4_closeout.md` | PASS |
| `p3_phase3g_f2a_runtime_unblock_verified` | Wf-325F | `TODO/completed/325f_f2a_runtime_unblock.md` | PASS |

### §3.3 Per gate flip — source agent mapping

| Gate | Source agent | Mechanism |
|---|---|---|
| All 15 originally-FAIL gates | **Wf-330-C** | venv-mismatch fix: switch from `/usr/bin/python` to `CadButEaas/.venv-vlm/bin/python` Python 3.12.13 + torch 2.14.0+cu130 |
| `p3_catherine_30metal_closeout_documented` (NEW) | **Wf-322G** | closeout doc emission |
| `p3_wf327_cpu_num_workers_sufficient` (NEW) | **Wf-327-F11** | gate-count docs fix; path-bug fix at Wf-327-SYN §4.1 (REPO_ROOT/scripts not MOLMETAL_ROOT/scripts) |
| `p3_m6_kappa_closeout_documented` (NEW) | **Wf-325D** | m6_kappa honest-negative closeout |
| `p3_admet_4_below_documented` (NEW) | **Wf-325E** | ADMET 4 BELOW honest-negative closeout |
| `p3_phase3g_f2a_runtime_unblock_verified` (NEW) | **Wf-325F** | F2(a) runtime PARTIAL_UNBLOCK verification |

---

## §4 Wf-329-V4 / Wf-330-B3 / Wf-331 deltas

### §4.1 Wf-329-V4 INDEX sync (61 PASS / 1 FAIL)

Source: `TODO/completed/329v4_index_sync.md`. The single FAIL was `p3_wf327_cpu_num_workers_sufficient`
(path-bug: `MOLMETAL_ROOT/scripts/launch_*.sh` vs actual `REPO_ROOT/scripts/launch_*.sh`).
**Fixed at Wf-327-SYN §4.1**: 6 paths corrected → Wf-330-C 62/62/62.

### §4.2 Wf-330-B3 fix

Source: `TODO/completed/330b1_trainer_merge_fix.md` (Wf-330-B1) + `TODO/completed/330b_catherine_30metal_full_pass.md` (Wf-330-B) + `TODO/completed/330b4_regression_smoke.md` (Wf-330-B4).
- **Wf-330-B1 trainer merge fix**: chemprop trainer merge bug fix (1-line) — pre-flight pass unaffected.
- **Wf-330-B catherine 30-metal full pass**: 30/30 metals 6/6 PASS at wall 1.20 s.
- **Wf-330-B4 regression smoke**: 62/62 PASS verified.
- Net Wf-330 contribution: Catherine bench FIXED (30/30 6/6); pre-flight 62/62 PRESERVED.

### §4.3 Wf-331 closeout (62 PASS / 0 FAIL)

Source: `TODO/completed/331*.md` (6 sub-flows: 331R1 Vina remaining gap + 331R2 triangle 12-d CPU scaffold + 331R3 m6_kappa multiph-pll + 331R4 m7 diversity multiph + 331R5 R² W₂ chemprop + 331R6 lambda cfm verified).
- Wf-331 added ZERO pre-flight gates; all 6 sub-flows verified existing pre-flight 62/62 still PASS.
- Net Wf-331 contribution: docs-only + R² honest-negative preserved; pre-flight 62/62 PRESERVED.

### §4.4 Wf-332-S4 (this run)

Pre-flight JSON captured at `/tmp/wf332s4_preflight.json`; **62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL** at wall 187.67 s.

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

- **Pure measurement**: every numeric value in this doc is from `/tmp/wf332s4_preflight.json` (this run) or `TODO/completed/{330c,329v4,322d}_*.md` (prior baseline artefacts).
- **PROJECTED**: NONE — this is a measurement-only doc.
- **MEASURED**: 62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL at Wf-332-S4 timestamp (2026-09-29 13:30 UTC).
- **Source agent per gate flip**: 15 FAIL gates fixed at Wf-330-C (venv-mismatch); 5 NEW gates added by Wf-322G / Wf-327-F11 / Wf-325D / Wf-325E / Wf-325F.

---

## §6 Cross-references

- `TODO/completed/330c_preflight_v3_full_pass.md` — Wf-330-C venv-fix root-cause analysis
- `TODO/completed/329v4_index_sync.md` — Wf-329-V4 INDEX sync (61/62 baseline)
- `TODO/completed/322d_index_readme_wf309_322_sync.md` — Wf-322D baseline (42/15/57 REGRESSION)
- `TODO/completed/327_synthesis.md` §4.1 — Wf-327 path-bug fix
- `TODO/INDEX.md` §22.16 (Wf-322D) + §22.20 (Wf-327-SYN) + §22.21 (Wf-328-SYN)
- `TODO/risks.md` R9 — CLOSED (15 FAIL → 0 FAIL)
- `/tmp/wf332s4_preflight.json` — captured this run

---

**END OF Wf-332-S4 doc** — Status: SHIPPED (CPU-only; pre-flight 62/62/62 PASS PRESERVED at Wf-332-S4 timestamp; G6 PID 2040025 retried Ru seed 2 single-task D-MPNN, 50-epoch full sweep, 28/97 GB VRAM on GPU 0, NOT interrupted per AGENTS.md §5.1).
