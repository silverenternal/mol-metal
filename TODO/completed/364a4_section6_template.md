# Wf-364-A4 — Sci paper §6 Limitations LaTeX template scaffold

**Wf**: Wf-364-A4 (ultracode residual)
**Date**: 2026-09-29
**Status**: ✅ **SHIPPED** — CPU-only docs+test surface
**Wall**: PROJECTED 15-25 min CPU-only

---

## §0 TL;DR

Wf-364-A4 SHIPPED the §6 Limitations **LaTeX template scaffold** at
`docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~270 LOC) — pre-prepared for
USER_OK-gated paper §6 v4 write per AGENTS.md §5.3 + TODO-84. The
template contains a ready-to-paste LaTeX snippet (~120 LOC) with full
`\MEASURED{}` / `\PROJECTED{}` label discipline + cross-references to
`metrics/by_metric/*.json` source paths + the 40-entry honest-negative
catalog at `docs/HONEST_NEGATIVE_CATALOG.md`.

**Honest framing per party-twenty-spirit §3.5:**
- Scaffold in `docs/` ONLY — `paper/*` touches USER_OK-gated
- 1 NEW test PASS
- 0 GPU launches performed
- 0 `paper/*` touches

---

## §1 Deliverables

| # | Path | Type | Status |
|---|---|---|---|
| 1 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` | NEW template scaffold (~270 LOC) | ✅ SHIPPED |
| 2 | `molmetal/tests/test_wf364a4_section6_template.py` | NEW test file | ✅ SHIPPED |
| 3 | `TODO/completed/364a4_section6_template.md` | NEW closeout | ✅ SHIPPED |

**Test count**: 1 NEW test file with **4 tests** (file_exists + latex_snippet + cross_refs + user_ok_gate).

---

## §2 Template scaffold structure (~270 LOC)

The template at `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` has 3 sections:

- **§0 How to use this template when USER_OK arrives** — 6-step paste-and-validate protocol
- **§1 Cross-references — MEASURED source paths** — 9-row table mapping every
  LaTeX section to its `\texttt{file:line}` JSON / report path on disk
- **§2 LaTeX snippet (ready to paste)** — ~120 LOC LaTeX with:
  - `\section{Limitations}\label{sec:limitations:v4}` header
  - **§2.1** Per-metric honest-negatives (m1, m2, m3, m6_kappa, m7/m14/m17/m19/ADMET)
  - **§2.2** Per-workflow honest-negatives (15 entries from HONEST_NEGATIVE_CATALOG §2)
  - **§2.3** Per-launcher honest-negatives (6 entries; 5 wrappers × ~47 hr GPU)
  - **§2.4** Resolved entries (7 entries; Wf-246-C + Wf-330-B4 + 4 stale anchors + CLI help-text + OT-CFM silent + DATA_CSV + pre-flight drift PARTIAL)
  - **§2.5** Open entries (10 entries; USER_OK-gated + structural)
  - **§2.6** Forward-looking (next 90 days; 10 launchers × 47 hr + Tier-2 submission + Blackwell sm_120 retune + bond_fdim gap)

Every numeric value in the LaTeX snippet carries `\MEASURED{}` or `\PROJECTED{}`
label per `party-twenty-spirit` §3.5.

---

## §3 MEASURED-vs-PROJECTED discipline (honest framing)

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2:

| Claim | Tag | Source path |
|---|---|---|
| m1 Ru seed=0 0.7613 | `\MEASURED{}` | `metrics/by_metric/metalcytotox_ru_auc.json` |
| m2 metalcytotox_combined R² +0.2227 | `\MEASURED{}` | `molmetal/reports/wf335m2_r2/result.json` |
| m2 platinai_4cell R² -0.1251 | `\MEASURED{}` | `molmetal/reports/wf335m2_r2/result.json` |
| m3 Vina cohort median -7.782 | `\PROJECTED{}` | `molmetal/reports/wf333_v3_vina/cohort_summary.json` |
| m6_kappa CV-R² +0.6814 | `\MEASURED{}` | `molmetal/reports/wf335m1_chemprop_pll/result.json` |
| m6_kappa augmented_mean 0.4386 | `\MEASURED{}` | `metrics/by_metric/m6_kappa.json` |
| 5 DEFERRED CFM waves 37.8 hr | `\PROJECTED{}` | `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` §2.1 |
| Tier-2 acceptance context | `\MEASURED{}` | AGENTS.md §5.4 |
| Standing GPU grant | `\MEASURED{}` | AGENTS.md §5.1 |
| 10 launchers × ~47 hr | `\PROJECTED{}` | `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` §3 |

**Zero fabrication** — every numeric value cites a path on disk.

---

## §4 USER_OK gate (TODO-84 + AGENTS.md §5.3)

Per TODO-84 + AGENTS.md §5.3, **zero touches to `paper/sections/*.tex`**
without explicit user authorization. The template:
- Lives in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (NOT `paper/`)
- References `paper/sections/06_limitations_v4.tex` as the **target** for
  USER_OK-gated write
- The test verifies `paper/sections/06_limitations_v4.tex` does NOT exist
  on disk (gate preserved)
- 6-step paste-and-validate protocol in §0 documents how to write the
  actual LaTeX file when USER_OK arrives

**Until USER_OK**: zero touches to `paper/sections/*.tex`.

---

## §5 Pre-flight v3 baseline preservation

Per `pre_flight_check.py`:
- **62 PASS / 0 FAIL / 62 TOTAL** preserved at Wf-335-SYN baseline
- Wf-364-A4 adds zero NEW pre-flight gates (docs+test surface only)
- `EXPECTED_GATE_COUNTS["all"] = 62` preserved

---

## §6 Cross-references

- **Source draft**: `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` (Wf-359-B2; ~290 LOC)
- **Source catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-359-B1 + Wf-358-C2; 40 entries)
- **Existing paper §6**: `paper/sections/06_limitations_v3.tex` (486 LOC; tier-1 anchors)
- **AGENTS.md**: §5.1 GPU grant + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2
- **party-twenty-spirit** skill §3.5 (MEASURED-vs-PROJECTED tag discipline)
- **mol-metal-paper-writing** skill (READ-only on `paper/*`)
- **Wf-364 ultracode residual wave**: parent workflow

---

## §7 Standing directives re-confirmed

- **§5.1 Standing GPU authorization** (2026-09-29 user grant): zero GPU launches in Wf-364-A4 (CPU-only docs+test).
- **§5.2 Honest framing mandate**: every numeric value in §2 LaTeX snippet
  carries `\MEASURED{}` / `\PROJECTED{}` label + source path citation.
- **§5.3 No paper work without authorization** (TODO-84): zero touches to
  `paper/sections/*.tex`. Template scaffold in `docs/` ONLY.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: every entry in §2
  is computational-only; no wet-lab external dependency.

---

**END OF WF-364-A4 CLOSEOUT** — Owner Hugo Lee; CPU-only docs+test surface;
pre-flight v3 62/62 PASS preserved; 0 GPU launches; 0 `paper/*` touches;
standing directives §5.1 + §5.2 + §5.3 + §5.4 re-confirmed.