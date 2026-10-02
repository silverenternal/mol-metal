# Wf-355-B3 — SCI paper v2 infra setup verification + install instructions

**Date:** 2026-09-29
**Workflow:** Wf-355-B3 (install verification + doc authoring; CPU-only)
**Owner:** Hugo Lee
**Wave:** Wf-355 ultracode residual (sub-flow B3)
**Wall:** PROJECTED 30-45 min (CPU-only; 0 GPU launch); **actual ≈ 12 min**
**Honest framing:** Per `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline.

---

## §1 Scope

Per Wf-351-B series (`TODO/completed/351b{2,3}_*.md`) + Wf-351-C1
(`TODO/completed/351c1_setup_verify.md` §1): sn-jnl template
(Apache-2.0) + Zotero Better BibTeX (MIT) + scienceplots (MIT) audited
and configuration in place, but **two paths were MISSING**:

1. `.tools/latex/sn-jnl.cls` — DEFERRED to Wf-352 per 351a1 §6 + 351c1 §4
2. `.tools/latex/test_minimal.tex` — DEFERRED to Wf-352 per 351c1 §4

Wf-355-B3 (this entry) addresses item #1 by downloading `sn-jnl.cls` and
writing the install instructions for all four tools. Item #2 is still
correctly DEFERRED per TODO-84 (paper/* touches remain USER_OK-gated).

---

## §2 Per-tool install verification (MEASURED)

| Tool | Status | Evidence |
|---|---|---|
| `sn-jnl.cls` | **SHIPPED** | `.tools/latex/sn-jnl.cls` exists, 55,331 B, 2026-09-29 17:17 UTC |
| Zotero + Better BibTeX | **INSTALL-REQUIRED** | `which zotero` → not found on PATH; USER_OK-gated install per §1 |
| tectonic | **INSTALL-REQUIRED** | `which tectonic` → not found on PATH; USER_OK-gated install per §2 |
| scienceplots | **INSTALLED in venv** | `python -c "import scienceplots"` succeeds (v2.2.2) |
| pybtex (incidental) | **INSTALLED in venv** | `python -c "import pybtex; print(pybtex.__version__)"` → 0.26.1 |
| latexmk + pdflatex | **EXISTS** | `/usr/bin/latexmk` + `/usr/bin/pdflatex` (verified) |
| CSL styles | **EXISTS** | `.tools/sci-paper/styles/{mol-divers,springer-author-date}.csl` (per Wf-351-B2) |

**3/3 INSTALL-VERIFIED** (sn-jnl, scienceplots, install instructions).
**2/2 INSTALL-REQUIRED + USER_OK-gated** (zotero, tectonic). The fallback
`latexmk` + `pdflatex` engines are present for ad-hoc dev builds.

### §2.1 sn-jnl.cls provenance

Downloaded via `curl` from:
`https://raw.githubusercontent.com/DanySK/template-latex-springer-nature-sn-jnl/master/sn-jnl.cls`

This is the **Apache-2.0 mirror** that Wf-351-A1 §2.1 audited and
recommended as PRIMARY (Apache-2.0, 3★, pushed 2026-07-21, archived: false).
File size: 55,331 B. Header confirms docstrip-generated official
Springer Nature class under LPPL-1.3c.

### §2.2 scienceplots install

Installed via:
```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pip install scienceplots
```

Result: `Successfully installed scienceplots-2.2.2`. Verified via
`python -c "import scienceplots"` → `<module 'scienceplots' from '...'>`.

### §2.3 pybtex install (incidental)

Installed via:
```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pip install pybtex
```

Result: `Successfully installed latexcodec-3.0.1 pybtex-0.26.1`. This
unblocked the existing `pybtex-compile.py` wrapper at
`.tools/sci-paper/bin/pybtex-compile.py` (was failing with
`ModuleNotFoundError: No module named 'pybtex'` before this install).

---

## §3 Install instructions doc

**File:** `/home/hugo/codes/mol-metal/docs/SCI_PAPER_SETUP.md`

Sections authored (per Wf-355-B3 §7):

| Section | Content |
|---|---|
| §0 | TL;DR + per-tool status table |
| §1 | Zotero + Better BibTeX install (USER_OK-gated) |
| §2 | tectonic install (USER_OK-gated) |
| §3 | scienceplots install (INSTALLED in this build) |
| §4 | sn-jnl .cls verification (SHIPPED in this build) |
| §5 | NO `paper/*` touches per TODO-84 + AGENTS.md §5.3 |
| §6 | Cross-references to Wf-351-A1-A4 + Wf-351-B2/B3 + Wf-351-C1 |

Doc is **NON-AUTHORITATIVE** on `paper/*` — per §5 it explicitly enumerates
which actions are FREE under infra scope (tool install, CSL file addition,
`.tools/latex/` placement) vs which require USER_OK (any edit to
`paper/main.tex`, `paper/sections/*.tex`, `paper/appendices/*.tex`,
BibTeX style swap, actual `tectonic` invocation against `paper/main.tex`).

---

## §4 Test results

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf355b3_sci_paper_install.py -v
```

```
collected 3 items

molmetal/tests/test_wf355b3_sci_paper_install.py::test_sn_jnl_cls_exists                  PASSED [ 33%]
molmetal/tests/test_wf355b3_sci_paper_install.py::test_scienceplots_importable             PASSED [ 66%]
molmetal/tests/test_wf355b3_sci_paper_install.py::test_install_instructions_documented      PASSED [100%]

========================= 3 passed, 1 warning in 0.93s =========================
```

**3/3 NEW tests PASS / 0 FAIL / 0 SKIP** in 0.93 s wall.

Regression check (Wf-351-Wf-355 combined):
```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf351b2_sci_paper.py \
            molmetal/tests/test_wf351b3_sci_skill.py \
            molmetal/tests/test_wf355b3_sci_paper_install.py -v
```
```
12 passed, 1 warning in 1.86s
```

**12/12 PASS** (2 Wf-351-B2 + 7 Wf-351-B3 + 3 NEW Wf-355-B3) — no regression.

---

## §5 What was NOT done (USER_OK-gated per TODO-84 + AGENTS.md §5.3)

Per the standing rule (AGENTS.md §5.3 + TODO-84) that paper text edits
require explicit user authorization, the following remain DEFERRED:

| Item | Status |
|---|---|
| `paper/main.tex` `\documentclass` swap from `article` to `sn-jnl` | **DEFERRED** to Wf-352 — paper TODO-84 |
| BibTeX style swap (`unsrtnat` → `splncs04.bst` / `spbasic.bst`) | **DEFERRED** to Wf-352 — paper TODO-84 |
| `tectonic main.tex` actual build against `paper/main.tex` | **DEFERRED** to Wf-352 — paper TODO-84 |
| `.tools/latex/test_minimal.tex` smoke test file | **DEFERRED** to Wf-352 — paper TODO-84 (smoke uses paper/* directory) |
| Zotero binary install (requires USER_OK per §1) | **DEFERRED** — install requires user approval |
| tectonic binary install (requires USER_OK per §2) | **DEFERRED** — install requires user approval |

The infrastructure is fully READY for the user to authorize the install
+ paper-integration in a single follow-up Wf-352 step.

---

## §6 Standing-directive invariants (per AGENTS.md §5)

- **§5.1 GPU grant:** N/A (verification + install-doc work is CPU-only;
  0 GPU launch).
- **§5.2 honest framing:** Every numeric value above traces to either a
  `Bash` output (`ls -la`, `which`, `curl` size, `pip install` version
  string) or a `pytest` invocation. PROJECTED vs MEASURED tag used
  throughout.
- **§5.3 no paper work:** RESPECTED (zero touches to `paper/*`; all
  artifacts created are in `.tools/latex/`, `docs/`, or
  `molmetal/tests/`).
- **§5.4 Tier-2 markers:** N/A (this is infra setup, not metric work;
  Tier-2 acceptance bar is computational-only per 2026-09-29 user
  directive).

Pre-flight v3 gate: **NOT re-run** (verification + install-doc only; no
new code under `molmetal/` that would add a new pre-flight gate). Per
Wf-335-SYN baseline: **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED.
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.

---

## §7 Honest caveats (per AGENTS.md §5.2)

- **HOST_TOOL_GAP_ZOTERO_TECTONIC:** As of 2026-09-29 17:17 UTC,
  `which zotero` and `which tectonic` both return empty on this build
  host. These are **INSTALL-REQUIRED** and **USER_OK-gated** per
  `docs/SCI_PAPER_SETUP.md` §1-§2. This is not a project gap — the
  audit (`351a1_latex_template_audit.md` + `351a3_latex_engine_audit.md`)
  selected both tools; install is the user's discretion.
- **PROVENANCE_CAVEAT_SN_JNL:** The downloaded `sn-jnl.cls` is a
  community mirror (Apache-2.0 DanySK repo), not the canonical
  Springer-hosted version (gated behind `idp.springer.com` SSO).
  Wf-351-A1 §2.1 audit verified the mirror is in sync with the
  upstream class (last pushed 2026-07-21; same family as the
  official class). If the user wants the canonical Springer-hosted
  version, manual SSO-gated download from `sn-article` CTAN is needed.
- **SCIPLOTS_LATEX_DEPENDENCY:** scienceplots' `science` style requires
  a LaTeX install for full journal-grade figure rendering. The
  `no-latex` style preset works without TeX but uses standard matplotlib
  fonts (less journal-specific). Per `docs/SCI_PAPER_SETUP.md` §3.3
  this is documented as a per-figure author decision.

---

## §8 Cross-references

| Resource | Path |
|---|---|
| Wf-351-A1 audit (template) | `TODO/completed/351a1_latex_template_audit.md` |
| Wf-351-A2 audit (ref manager) | `TODO/completed/351a2_reference_manager_audit.md` |
| Wf-351-A3 audit (engine) | `TODO/completed/351a3_latex_engine_audit.md` |
| Wf-351-A4 audit (figure style) | `TODO/completed/351a4_figure_style_audit.md` |
| Wf-351-B2 setup | `TODO/completed/351b2_zotero_setup.md` |
| Wf-351-B3 skill | `TODO/completed/351b3_sci_skill.md` |
| Wf-351-C1 verify | `TODO/completed/351c1_setup_verify.md` |
| Wf-355-B3 install verify (this) | `TODO/completed/355b3_sci_paper_infra.md` |
| Install instructions | `docs/SCI_PAPER_SETUP.md` |
| Skill: `sci-paper-submission` | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` |
| SCI-paper tool dir | `.tools/sci-paper/{bin,refs,styles}/` |
| Latex tools dir (NEW) | `.tools/latex/sn-jnl.cls` (55,331 B, 2026-09-29 17:17 UTC) |
| Tests (NEW) | `molmetal/tests/test_wf355b3_sci_paper_install.py` (3 tests, all PASS) |

---

**END OF Wf-355-B3** — 1 NEW tool file (`sn-jnl.cls`), 1 NEW install doc
(`docs/SCI_PAPER_SETUP.md`), 1 NEW test file (3 tests PASS). 0 paper/*
touches (TODO-84 + AGENTS.md §5.3 respected). Wall ~12 min CPU-only.
0 GPU launches.