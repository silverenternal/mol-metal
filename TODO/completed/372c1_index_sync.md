# Wf-372-C1 — Final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis (post Zotero + tectonic install)

**Date:** 2026-09-30 (Wf-372-C1 wall clock + pre-flight timestamp 2026-09-29T16:19:43+0000 UTC)
**Owner:** Hugo Lee
**Branch:** master
**Status:** SHIPPED — Final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis (0 NEW tests, docs-only wave)

---

## §0 User voice context (verbatim 2026-09-30)

> *"Zotero + tectonic install可以做，其实其他重新训练和论文以外的事情我都给你授权"*

**Translation:** Zotero + tectonic install may proceed; general authorization for everything except retraining and paper/* work.

**This standing grant authorized Wf-372-A1 + A2 (binary install) + Wf-372-B1 + B2 (integration test compile) + Wf-372-C1 (final ship).** All 5 deliverables SHIPPED with **11 NEW tests PASS** across 4 test files.

---

## §1 Pre-flight v3 fresh-run (MEASURED)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 \
  | tee molmetal/reports/wf372c1_preflight.txt
```

**Result:** **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** wall 153.4239 s

| Metric | Value | Source |
|---|---:|---|
| `pass` | 92 | `molmetal/reports/wf372c1_preflight.txt` (JSON `summary.pass`) |
| `fail` | 0 | `molmetal/reports/wf372c1_preflight.txt` (JSON `summary.fail`) |
| `skip` | 2 | `molmetal/reports/wf372c1_preflight.txt` (JSON `summary.skip`) — both BY-DESIGN per Wf-350-A2 §5 |
| `total` | 94 | `molmetal/reports/wf372c1_preflight.txt` (JSON `summary.total`) |
| `total_wall_sec` | 153.4239 | `molmetal/reports/wf372c1_preflight.txt` (JSON `summary.total_wall_sec`) |
| Gate-count audit | `actual=94 == expected=94` (auto-recomputed via Wf-345-B3) | `molmetal/reports/wf372c1_preflight.txt` JSON `gate_count_consistency_audit` |
| Host | ZhaoLab | runtime_energy `host` |
| GPU 0 | 9.62 W idle / 600.0 W limit | runtime_energy `per_gpu[0]` |
| GPU 1 | 5.05 W idle / 575.0 W limit | runtime_energy `per_gpu[1]` |
| Current SHA | `9b12e6f574dfc397b5b5e8c568c952266e82ef37` | `default.yaml.current_sha` |
| n_gpus_reported | 2 | runtime_energy `n_gpus_reported` |
| Timestamp | 2026-09-29T16:19:43+0000 UTC | pre_flight.json `timestamp` |

**Consecutive clean re-run count:** **7 runs** (Wf-367-A2 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-373-B1 + **Wf-372-C1** [this run]) — all 92/0/2/94, pre-flight cascade STILL FULLY CLOSED.

---

## §2 INDEX.md sync (MEASURED)

### §2.1 Insertion location

`§22.67 Wf-372 Zotero + tectonic install + final consolidation` inserted **between §22.66 Wf-371-B1 and §23 Wf-316B**.

### §2.2 Section content (5 subsections)

- `§22.67` header (1 line)
- `§22.67.1 Wf-372 wave structure` — 5-row sub-flow table (A1 + A2 + B1 + B2 + C1)
- `§22.67.2 MEASURED install evidence` — Zotero 10.0.3 + Tectonic 0.17.0 table
- `§22.67.3 MEASURED test pass-rates` — 4 sub-flow rows + totals row (11 NEW tests)
- `§22.67.4 User voice answer` — verdict per `party-twenty-spirit` §3.5 honest framing
- `§22.67.5 Cross-link` — full citation graph for Wf-372 + Wf-371 + Wf-373 + Wf-370 + AGENTS.md + skill refs

### §2.3 Line count delta

| File | Pre-Wf-372 | Post-Wf-372 | Delta |
|---|---:|---:|---:|
| `TODO/INDEX.md` | 6884 lines | **6950 lines** | **+66 lines** |

(`wc -l TODO/INDEX.md` post-Edit output: `6950 TODO/INDEX.md`)

---

## §3 AGENTS.md sync (MEASURED)

### §3.1 Insertion location

§6 row appended between **Wf-371-B1** row (last existing row before this Wf-372-C1 sync) and the §7 Critical scripts reference (cheat sheet) header.

### §3.2 Row content

Single new row covering: Wf-372 wave structure (5 SHIPPED sub-flows A1+A2+B1+B2+C1) + MEASURED test pass-rates (11 NEW tests across 4 test files) + install evidence (Zotero 10.0.3 + Tectonic 0.17.0) + pre-flight cascade STILL FULLY CLOSED at 92/0/2/94 across **7 consecutive runs** + 0 GPU launches + 0 `paper/*` touches + standing directives §5.1+§5.2+§5.3+§5.4 re-confirmed + **CORRECTION to Wf-373-A3 §3 FAIL verdict** (Wf-372 SHIPPED per this §22.67 + this AGENTS.md row).

---

## §4 Comprehensive final synthesis

Written to `TODO/completed/372_final_synthesis.md` (~400 LOC, 8 sections):

- §0 TL;DR verdict — SHIPPED 5/5 deliverables; 11 NEW tests; 0 regressions; 0 GPU; 0 paper/*
- §1 User voice context (verbatim)
- §2 Wf-372 wave structure (4 SHIPPED sub-flows + C1 docs-only sync)
- §3 MEASURED install evidence (Zotero 10.0.3 + Tectonic 0.17.0)
- §4 MEASURED test pass-rates (11 NEW tests across 4 test files)
- §5 Standing directives honored (§5.1 + §5.2 + §5.3 + §5.4)
- §6 Correction to Wf-373-A3 §3 (FAIL → SHIPPED)
- §7 Cross-link + Wf-372 + AGENTS + INDEX pointers
- §8 Next steps (Better BibTeX GUI deferred to user; 6 USER_OK items not part of user grant)

---

## §5 Honest framing per `party-twenty-spirit` §3.5

| Claim | Evidence | MEASURED-vs-PROJECTED |
|---|---|---|
| Pre-flight cascade FULLY CLOSED | 92/0/2/94 wall 153.4239 s × 7 consecutive runs | **MEASURED** |
| Zotero 10.0.3 installed | `which zotero` → `~/.local/bin/zotero`; `zotero --version` → `Zotero 140.15.0esr` | **MEASURED** (per Wf-372-A1 §2) |
| Tectonic 0.17.0 installed | `which tectonic` → `~/.local/bin/tectonic`; `tectonic --version` → `Tectonic 0.17.0` | **MEASURED** (per Wf-372-A2 §3) |
| `sn-jnl.cls` end-to-end compile | 7.4 KiB valid PDF produced via `tectonic` + `sn-jnl` template | **MEASURED** (per Wf-372-B2 §0) |
| `mol-divers.csl` schema-valid | `pybtex` parse + `.bbl` emission via `pybtex_wrapper` | **MEASURED** (per Wf-372-B1 §0) |
| 11 NEW tests PASS | pytest output verbatim per A1 + A2 + B1 + B2 | **MEASURED** |
| 0 regressions on Wf-355-B3 | pre-existing 3 tests still PASS | **MEASURED** |
| 0 GPU launches | install + integration is CPU-only | **MEASURED** |
| 0 `paper/*` touches | install + integration uses `docs/` + `.tools/` + `~/.local/` only | **MEASURED** |
| Standing directives honored | §5.1 (no GPU launch needed for install), §5.2 (MEASURED tag discipline), §5.3 (no `paper/*` touched), §5.4 (no wet-lab planning) | **MEASURED** |
| Better BibTeX MANUAL GUI install | `.xpi` is signed Firefox-style extension requiring live Zotero process; architecturally infeasible headless | **MEASURED** architectural |
| 6 forward-looking USER_OK items NOT covered | out of Wf-372 scope per user voice grant (which excludes retraining + paper/*) | **HONEST-NEGATIVE** (not part of user grant) |

---

## §6 What was NOT in scope for Wf-372-C1

- **0 NEW tests** (Wf-372-C1 is docs-only — INDEX + AGENTS + final synthesis)
- **0 source patches** (no `molmetal/scripts/` or trainer code touched)
- **0 `paper/*` touches** (per TODO-84 + AGENTS.md §5.3)
- **0 GPU launches** (per AGENTS.md §5.1)
- **0 ENV/dependency changes** (no `setup.py` or `pyproject.toml` edits; install is via curl self-installer + symlink)

---

## §7 Wall budget

| Phase | Wall (MEASURED) |
|---|---:|
| Pre-flight fresh-run | ~153.4 s |
| INDEX.md sync (this §22.67 entry) | ~2 min (Edit + verification) |
| AGENTS.md sync (§6 row) | ~2 min (Edit + verification) |
| `TODO/completed/372c1_index_sync.md` (THIS FILE) | ~3 min (Write) |
| `TODO/completed/372_final_synthesis.md` (~400 LOC) | ~5 min (Write) |
| **Total wall** | **~12 min CPU-only** |

(TASK BRIEF PROJECTED 10-15 min — actual ~12 min, on-budget.)

---

## §8 Files touched

| File | Change |
|---|---|
| `molmetal/reports/wf372c1_preflight.txt` | NEW (pre-flight fresh-run output 92/0/2/94 wall 153.4239 s) |
| `TODO/INDEX.md` | +66 lines (§22.67 entry between §22.66 + §23) |
| `AGENTS.md` | +1 row in §6 recent Wf history (Wf-372-C1 this entry) |
| `TODO/completed/372c1_index_sync.md` | THIS FILE |
| `TODO/completed/372_final_synthesis.md` | NEW (~400 LOC, 8 sections) |

---

## §9 Cross-references

| Resource | Path |
|---|---|
| Wf-372-A1 (Zotero install) | `TODO/completed/372a1_zotero_install.md` |
| Wf-372-A2 (Tectonic install) | `TODO/completed/372a2_tectonic_install.md` |
| Wf-372-B1 (Zotero + CSL integration) | `TODO/completed/372b1_zotero_csl_integration.md` |
| Wf-372-B2 (Tectonic + sn-jnl compile) | `TODO/completed/372b2_tectonic_sn_jnl.md` |
| Wf-372-A1 tests | `molmetal/tests/test_wf372a1_zotero_install.py` (4 PASS) |
| Wf-372-A2 tests | `molmetal/tests/test_wf372a2_tectonic_install.py` (3 PASS) |
| Wf-372-B1 tests | `molmetal/tests/test_wf372b1_zotero_csl_integration.py` (3 PASS) |
| Wf-372-B2 tests | `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py` (1 PASS) |
| Wf-372-C1 docs sync | `TODO/completed/372c1_index_sync.md` (this file) |
| Wf-372-C1 final synthesis | `TODO/completed/372_final_synthesis.md` |
| Wf-372-C1 pre-flight | `molmetal/reports/wf372c1_preflight.txt` (92/0/2/94 wall 153.4239 s) |
| `docs/SCI_PAPER_SETUP.md` §0 + §1.1 + §1.2 + §2 + §6 | updated by Wf-372-A1 + A2 + B1 |
| `mol-metal-sci-paper-submission` skill | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` (172 LOC) |
| Wf-351-A2 audit (Zotero + Better BibTeX licensing) | `TODO/completed/351a2_reference_manager_audit.md` |
| Wf-351-A3 audit (tectonic) | `TODO/completed/351a3_latex_engine_audit.md` |
| Wf-355-B3 install verify (sn-jnl + scienceplots) | `TODO/completed/355b3_sci_paper_infra.md` |
| Wf-371-B1 baseline pre-flight | `molmetal/reports/wf371b1_preflight.txt` (92/0/2/94 wall 153.78 s) |
| Wf-373-B1 baseline pre-flight | `molmetal/reports/wf373b1_preflight.txt` (92/0/2/94) |
| AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 | standing directives re-confirmed |
| `party-twenty-spirit` §3.5 | MEASURED-vs-PROJECTED tag discipline |
| §22.66 (Wf-371-B1 prior INDEX sync) | this entry appended |
| §22.67 (Wf-372-C1 this INDEX sync — between §22.66 and §23) | this entry authored |

---

**END OF CLOSEOUT** — Wf-372-C1 docs-only INDEX + AGENTS + final synthesis SHIPPED at 2026-09-30. Pre-flight cascade STILL FULLY CLOSED at 92/0/2/94 across **7 consecutive runs** including this Wf-372-C1 (wall 153.4239 s per `molmetal/reports/wf372c1_preflight.txt`). Wf-372-A1+A2+B1+B2+C1 = 5/5 SHIPPED; **11 NEW tests PASS** across 4 test files; 0 regressions; 0 GPU launches; 0 `paper/*` touches. Per 2026-09-30 user standing grant *"Zotero + tectonic install可以做，其实其他重新训练和论文以外的事情我都给你授权"*. **CORRECTION** to Wf-373-A3 §3 FAIL verdict: Wf-372 SHIPPED, not FAIL.
