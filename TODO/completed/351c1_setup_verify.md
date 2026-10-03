# Wf-351-C1: SCI paper submission setup verification

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Wall:** PROJECTED 5-10 min CPU-only; **actual ≈ 2 min** (docs-only verification)
**Verification only — zero code changes.**

---

## §1 Per-path status

| # | Path | Status | Notes |
|---|---|---|---|
| 1 | `.tools/latex/sn-jnl.cls` | **MISSING** | Deferred to Wf-352 per 351a1 §6 "Out of scope" — `.tex` integration needs user OK per TODO-84. Audit doc verified `DanySK/template-latex-springer-nature-sn-jnl` on GitHub (Apache-2.0, 3★, pushed 2026-07-21), but the `.cls` itself was NOT mirrored locally yet (audit-only wave, not download wave). |
| 2 | `.tools/latex/test_minimal.tex` | **MISSING** | Same as above — minimal LaTeX smoke file deferred to Wf-352. |
| 3 | `.tools/sci-paper/styles/mol-divers.csl` | **EXISTS** | 9775 B, 2026-09-29 16:35 — Mol. Divers. CSL Style. |
| 4 | `.tools/sci-paper/refs/references.bib` | **EXISTS** | 3165 B, 2026-09-29 16:37 — BibTeX reference DB. |
| 5 | `.tools/sci-paper/bin/pybtex-compile.py` | **EXISTS** | 4058 B, 2026-09-29 16:37, executable — pybtex compile helper. |
| 6 | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` | **EXISTS** | 16089 B, 2026-09-29 16:36 — 8 sections + YAML frontmatter per Wf-351-B3. |
| 7 | `AGENTS.md` — `mol-metal-sci-paper-submission` registered | **PASS** | grep found 1 entry in §3.2 + 1 entry in §6 (Wf-351-B3 ship surface line). |

**Summary:** 5/7 paths PASS (3 Zotero + 1 skill + 1 AGENTS.md update).
**MISSING:** 2/7 paths are LaTeX files — both **correctly deferred to Wf-352** per 351a1 §6.

---

## §2 Test results

```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf351b2_sci_paper.py molmetal/tests/test_wf351b3_sci_skill.py -v
```

```
collected 9 items

molmetal/tests/test_wf351b2_sci_paper.py::test_pybtex_compile_cli_roundtrip      PASSED [ 11%]
molmetal/tests/test_wf351b2_sci_paper.py::test_references_bib_format              PASSED [ 22%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_exists                    PASSED [ 33%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_has_frontmatter           PASSED [ 44%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_has_minimum_5_sections     PASSED [ 55%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_has_workflow_stages       PASSED [ 66%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_readonly_on_paper         PASSED [ 77%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_cites_open_source_audits PASSED [ 88%]
molmetal/tests/test_wf351b3_sci_skill.py::test_skill_md_decision_discipline       PASSED [100%]

========================= 9 passed, 1 warning in 0.83s =========================
```

**Verdict:** **9/9 PASS / 0 FAIL / 0 SKIP** in 0.83 s wall.

---

## §3 AGENTS.md update status

`grep "mol-metal-sci-paper-submission" AGENTS.md` returned 2 matches:

1. **`§3.2 Domain / surface skills` row** — lists `mol-metal-sci-paper-submission` alongside the 28 baseline + 8 NEW R20/R21 skills. Description: "SCI paper submission workflow — end-to-end build → validate → submit → archive using audited open-source tools (sn-jnl + tectonic + Zotero + matplotlib/scienceplots). READ-only on `paper/*` per TODO-84 + §5.3."
2. **`§6 Recent Wf history pointer` — Wf-351-B3 line** — included in the `2026-09-29` row that also covers Wf-296..343.

**Verdict:** AGENTS.md is **correctly synced** per Wf-336-D2 AFD pattern + Wf-351-B3 §5.

---

## §4 Integration verdict

| Component | Status |
|---|---|
| `mol-metal-sci-paper-submission` skill wired | YES (1 row in §3.2 + 1 in §6) |
| Zotero + Better BibTeX + CSL + pybtex stack | YES (3 files in `.tools/sci-paper/`) |
| Skill audit-trail (open-source licenses) | YES (test `test_skill_md_cites_open_source_audits` PASS) |
| Skill READ-only discipline on `paper/*` | YES (test `test_skill_md_readonly_on_paper` PASS) |
| Frontmatter + ≥ 5 sections + workflow stages | YES (3 tests PASS) |
| BibTeX format + pybtex round-trip | YES (2 tests PASS) |
| **LaTeX template (`sn-jnl.cls`)** | **DEFERRED to Wf-352** (audit-only Wf-351-A1; download + integration needs user OK per TODO-84) |

**Overall integration verdict: PASS (with deferred items flagged).**

The Wf-351 wave shipped:
- 4 audit docs (A1-A4) at `TODO/completed/351a{1..4}_*.md`
- 2 setup + skill docs (B2 + B3) at `TODO/completed/351b{2,3}_*.md`
- 1 NEW skill (`mol-metal-sci-paper-submission`)
- 9 NEW tests (2 in `test_wf351b2_sci_paper.py` + 7 in `test_wf351b3_sci_skill.py`)

**Deferral note** (per 351a1 §6 + 351b2 §6): LaTeX `.cls` + `.tex` integration, BibTeX style swap (`unsrtnat` → `splncs04.bst` / `spbasic.bst`), and Pandoc + Vale prose-lint are deferred to Wf-352+ — these require user authorization for `paper/*` edits per TODO-84 + AGENTS.md §5.3.

---

## §5 Pre-flight + standing-directive invariants
- Pre-flight v3: **NOT re-run** (verification only, no code change). Per Wf-335-SYN baseline: **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED. EXPECTED_GATE_COUNTS["all"] 62 PRESERVED (Wf-351 adds zero new gates).
- AGENTS.md §5.1 GPU grant — N/A (verification is CPU-only, no GPU launch).
- AGENTS.md §5.2 honest framing — every numeric value above traces to either a `Bash` output or `pytest` output in this verification step.
- AGENTS.md §5.3 no paper work — RESPECTED (zero touches to `paper/*`).
- AGENTS.md §5.4 Tier-2 journal target — Mol. Divers. CSL style configured; full submission workflow deferred to Wf-352.

---

## §6 Cross-references
- `TODO/completed/351a1_latex_template_audit.md` — LaTeX template audit (defers `.cls` download to Wf-352)
- `TODO/completed/351a2_reference_manager_audit.md` — Zotero + Better BibTeX + JabRef + papis audit
- `TODO/completed/351a3_latex_engine_audit.md` — tectonic + latexmk build engine audit
- `TODO/completed/351a4_figure_style_audit.md` — matplotlib + scienceplots figure-style audit
- `TODO/completed/351b2_zotero_setup.md` — Zotero + Better BibTeX + CSL + pybtex actual setup
- `TODO/completed/351b3_sci_skill.md` — skill wiring + tests + AGENTS.md sync
- `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` — the new skill itself (16089 B, 8 sections)
- AGENTS.md §3.2 (skill catalog row) + §6 (Wf-351-B3 ship-surface line)

---

**Wall:** ≈ 2 min CPU-only (faster than projected 5-10 min — verification was straightforward).
**Ship verdict:** Wf-351-C1 SHIPPED — pipeline-ready; LaTeX stage correctly deferred to Wf-352 per TODO-84 + §5.3.
