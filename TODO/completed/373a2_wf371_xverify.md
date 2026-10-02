# Wf-373-A2 — Wf-371 closeout verification (cross-Wf audit)

**Date:** 2026-09-30 (Wf-373-A2 timestamp)
**Owner:** Hugo Lee + Wf-373-A2 verify agent
**Workflow:** Wf-373-A2 (cross-Wf verification; READ-only audit)
**Status:** SHIPPED — 2/4 closeouts present + 2 MISSING flagged HONEST-NEGATIVE
**Pre-flight:** NOT re-run (verification only; no source patches)
**Cross-ref:** `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.2 + §5.3 + §5.4` +
Wf-371 task brief + Wf-369-A1 §1.1

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Wf-371 closeouts PRESENT** | 2/4 (A2 + A3) | `ls TODO/completed/ \| grep ^371` |
| **Wf-371 closeouts MISSING** | 2/4 (A1 + B1) | `ls TODO/completed/ \| grep ^371` |
| **A2 closeout** | `TODO/completed/371a2_368c1_standalone_authored.md` (131 lines, 6 sections, SHIPPED) | this verify |
| **A3 closeout** | `TODO/completed/371a3_365a2_standalone_authored.md` (227 lines, 7 sections, SHIPPED) | this verify |
| **A1 expected file** | `TODO/completed/371a1_m6_kappa_degenerate.md` (NOT ON DISK) | MISSING |
| **B1 expected file** | `TODO/completed/371b1_final_synthesis.md` (NOT ON DISK) | MISSING |
| **Total closeouts** | 840 files in `TODO/completed/` | `ls TODO/completed/ \| wc -l` |
| **Tests status** | 0 NEW (verify only) | this verify |
| **GPU launches** | 0 (READ-only audit) | AGENTS.md §5.1 |
| **paper/* touches** | 0 | AGENTS.md §5.3 TODO-84 active |
| **Wall** | ~5 min CPU-only | this verify |

**One-line verdict**: Wf-371 shipped 2/4 agents (A2 + A3 SHIPPED with substantial
closeouts); A1 + B1 are MISSING from disk and flagged HONEST-NEGATIVE per
party-twenty-spirit §3.5. Per the user's question "those problems completely resolved,
if not launch new ultracode to follow up" — Wf-371 is INCOMPLETE; recommend spawning
Wf-374 follow-up ultracode for A1 (m6_kappa degenerate) + B1 (final synthesis).

---

## §1 Per-agent verification

### §1.1 Wf-371-A1 (m6_kappa degenerate) — MISSING HONEST-NEGATIVE

| Brief item | Status |
|---|---|
| Expected file: `TODO/completed/371a1_m6_kappa_degenerate.md` | NOT ON DISK |
| `ls TODO/completed/ \| grep ^371a1` | (empty) |
| `find . -name "371a1*"` | (only chemprop cache keys, no closeout) |

**Verdict:** HONEST-NEGATIVE — A1 was scoped to write a standalone closeout for the
m6_kappa degenerate case (presumably a Wf-365-A1 caveat gap surfaced by Wf-369-A1),
but the closeout file was NOT authored.

### §1.2 Wf-371-A2 (Wf-368-C1 standalone) — SHIPPED

| Brief item | Status |
|---|---|
| File: `TODO/completed/371a2_368c1_standalone_authored.md` | ✅ ON DISK (131 lines) |
| Standalone closeout written: `TODO/completed/368c1_final_synthesis.md` | ✅ REFERENCED in §0 |
| NEW test: `molmetal/tests/test_wf371a2_368c1_standalone.py` | ✅ 1 NEW test PASS in 0.32 s |
| Pre-flight 92/0/2/94 cited | ✅ §1 verbatim |
| m6_kappa + m2 R² HONEST-NEGATIVE preserved | ✅ §2 |
| 7 USER_OK-gated launches referenced | ✅ §3 |
| AGENTS.md §5.1..§5.4 re-confirmed | ✅ §5 |

**Verdict:** SHIPPED — Wf-371-A2 reconstructed the Wf-368-C1 final_synthesis closeout
(6 sections, ~280 LOC) and authored 1 NEW test.

### §1.3 Wf-371-A3 (Wf-365-A2 standalone) — SHIPPED

| Brief item | Status |
|---|---|
| File: `TODO/completed/371a3_365a2_standalone_authored.md` | ✅ ON DISK (227 lines) |
| Standalone closeout written: `TODO/completed/365a2_winner_cli_flag.md` | ✅ REFERENCED in §0 |
| NEW test: `molmetal/tests/test_wf371a3_365a2_standalone.py` | ✅ 1 NEW test PASS in 0.32 s |
| CLI flag `--neural-ode-flow` already wired in Wf-361-C1 (back-referenced) | ✅ §0 |
| 7 sections (TL;DR + §1-§6) | ✅ §0 |
| Pre-flight v3 62/0/62 PRESERVED | ✅ §0 |

**Verdict:** SHIPPED — Wf-371-A3 reconstructed the Wf-365-A2 standalone closeout for
the CLI flag wire-in (6 sections + §6 bonus TODO inventory sync) and authored 1 NEW
test.

### §1.4 Wf-371-B1 (final synthesis) — MISSING HONEST-NEGATIVE

| Brief item | Status |
|---|---|
| Expected file: `TODO/completed/371b1_final_synthesis.md` OR `TODO/completed/371_final_synthesis.md` | NOT ON DISK |
| `ls TODO/completed/ \| grep ^371b1` | (empty) |
| `ls TODO/completed/ \| grep ^371_` | (empty) |

**Verdict:** HONEST-NEGATIVE — B1 (final synthesis aggregating A1+A2+A3 verdicts) was
not authored. With A1 also missing, the synthesis would have been incomplete anyway.

---

## §2 Honest framing (party-twenty-spirit §3.5)

**MEASURED:**
- 2/4 Wf-371 closeouts present on disk (A2 + A3).
- 2/4 Wf-371 closeouts missing from disk (A1 + B1) — HONEST-NEGATIVE flagged.
- Total `TODO/completed/` files: 840.

**PROJECTED:**
- Wall: ~5 min CPU-only for this verify (MEASURED).
- Wall for Wf-374 follow-up ultracode: ~15-30 min CPU-only (PROJECTED, docs+test only).

**Follow-up recommendation:**
Per the user's question "those problems completely resolved, if not launch new
ultracode to follow up", Wf-371 is INCOMPLETE. Recommend spawning **Wf-374**
follow-up ultracode to close:
1. Wf-371-A1 (m6_kappa degenerate standalone closeout) — 1 NEW file + 1 NEW test
2. Wf-371-B1 (final synthesis aggregating all 3 standalone closeouts) — 1 NEW file

PROJECTED Wf-374 wall: ~15-30 min CPU-only (docs + tests; no GPU launches; no
paper/* touches per AGENTS.md §5.1 + §5.3).

---

## §3 Cross-references

- `party-twenty-spirit §3.5` — honest-negative framing
- `AGENTS.md §5.1` — kill_then_adapt GPU budget (no GPU launch needed for docs)
- `AGENTS.md §5.2` — MEASURED-vs-PROJECTED tag discipline
- `AGENTS.md §5.3` — TODO-84 active; zero `paper/*` touches
- `AGENTS.md §5.4` — Tier-2 acceptance bar; computational-only
- Wf-371 task brief — 4 agents A1+A2+A3+B1
- Wf-369-A1 §1.1 — Wf-368-C1 + Wf-365-A2 final_synthesis gaps surfaced
- Wf-370-B1 §22.64 — cross-Wf synthesis precedent

---

## §4 Final verdict

| Agent | Status | File |
|---|---|---|
| Wf-371-A1 (m6_kappa degenerate) | **MISSING HONEST-NEGATIVE** | (gap) |
| Wf-371-A2 (Wf-368-C1 standalone) | ✅ SHIPPED | `TODO/completed/371a2_368c1_standalone_authored.md` |
| Wf-371-A3 (Wf-365-A2 standalone) | ✅ SHIPPED | `TODO/completed/371a3_365a2_standalone_authored.md` |
| Wf-371-B1 (final synthesis) | **MISSING HONEST-NEGATIVE** | (gap) |

**Wf-371 ship verdict:** PARTIAL — 2/4 SHIPPED + 2/4 MISSING. Follow-up ultracode
recommended (Wf-374 or equivalent) to close the A1 + B1 gaps.
