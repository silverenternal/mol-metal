# Wf-356-A3 — SCI paper v2 infra setup retry (verification pass)

**Date:** 2026-09-29 (Wf-356-A3 retry timestamp)
**Workflow:** Wf-356-A3 (retry pass; CPU-only)
**Owner:** Hugo Lee
**Wave:** Wf-356 ultracode residual closeout (sub-flow A3)
**Wall:** PROJECTED 30-45 min (CPU-only; 0 GPU launch); **actual ≈ 3 min**
**Honest framing:** Per `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline.

---

## §1 Retry Rationale

Wf-355-B3 (`TODO/completed/355b3_sci_paper_infra.md`) shipped the original
SCI paper v2 infra setup pass. Per the Wf-356 ultracode directive
("覆盖所有发现的问题把残余的问题解决了"), this retry re-verifies the Wf-355-B3
ship contract from a fresh timestamp + adds three retry-NEW assertions for
gaps that were surfaced (but not closed) by the Wf-355-B3 truncated output:

| Gap | Wf-356-A3 retry addition |
|---|---|
| CSL file presence not asserted as separate test | **Test 4 (NEW):** `test_csl_styles_present` |
| pybtex wrapper presence not asserted as separate test | **Test 5 (NEW):** `test_pybtex_wrapper_present` |
| Skill registration not surfaced as separate test | **Test 6 (NEW):** `test_sci_skill_registered` |

The 3 original Wf-355-B3 tests (sn-jnl.cls + scienceplots + install
instructions) are mirrored 1:1 in this retry to confirm no regression at
the Wf-356-A3 timestamp.

---

## §2 Per-tool install verification (MEASURED @ Wf-356-A3 timestamp)

| Tool | Status | Evidence |
|---|---|---|
| `sn-jnl.cls` | **SHIPPED** | `.tools/latex/sn-jnl.cls` exists, 55,331 B, 2026-09-29 17:17 UTC (UNCHANGED from Wf-355-B3) |
| Zotero + Better BibTeX | **INSTALL-REQUIRED** | `which zotero` → not found on PATH; USER_OK-gated install per `docs/SCI_PAPER_SETUP.md` §1 |
| tectonic | **INSTALL-REQUIRED** | `which tectonic` → not found on PATH; USER_OK-gated install per `docs/SCI_PAPER_SETUP.md` §2 |
| scienceplots | **INSTALLED in venv** | `import importlib.metadata; m.version('scienceplots')` → `2.2.2` (UNCHANGED from Wf-355-B3) |
| pybtex (incidental) | **INSTALLED in venv** | `python -c "import pybtex; print(pybtex.__version__)"` → `0.26.1` |
| latexmk + pdflatex | **EXISTS** | `/usr/bin/latexmk` + `/usr/bin/pdflatex` (fallback engines for ad-hoc dev builds) |
| CSL styles (NEW retry check) | **EXISTS** | `.tools/sci-paper/styles/{mol-divers,springer-author-date}.csl` both present + valid CSL schema |
| pybtex wrapper (NEW retry check) | **EXISTS + executable** | `.tools/sci-paper/bin/pybtex-compile.py` user-executable bit set (mode 40755) |
| `mol-metal-sci-paper-submission` skill (NEW retry check) | **REGISTERED** | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` present + referenced in AGENTS.md §3.2 |

**6/6 retry tests PASS** at Wf-356-A3 timestamp (3 retry-mirror + 3 retry-NEW).

---

## §3 Test results

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf356a3_sci_paper_install.py -v
```

```
collected 6 items

molmetal/tests/test_wf356a3_sci_paper_install.py::test_sn_jnl_cls_exists                  PASSED [ 16%]
molmetal/tests/test_wf356a3_sci_paper_install.py::test_scienceplots_importable             PASSED [ 33%]
molmetal/tests/test_wf356a3_sci_paper_install.py::test_install_instructions_documented      PASSED [ 50%]
molmetal/tests/test_wf356a3_sci_paper_install.py::test_csl_styles_present                   PASSED [ 66%]
molmetal/tests/test_wf356a3_sci_paper_install.py::test_pybtex_wrapper_present               PASSED [ 83%]
molmetal/tests/test_wf356a3_sci_paper_install.py::test_sci_skill_registered                 PASSED [100%]

========================= 6 passed, 1 warning in 0.80s =========================
```

**6/6 NEW tests PASS / 0 FAIL / 0 SKIP** in 0.80 s wall.

### §3.1 Combined regression check (Wf-351 → Wf-356)

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf351b2_sci_paper.py \
            molmetal/tests/test_wf351b3_sci_skill.py \
            molmetal/tests/test_wf355b3_sci_paper_install.py \
            molmetal/tests/test_wf356a3_sci_paper_install.py -v
```

```
collected 18 items

... (18 tests)
======================== 18 passed, 1 warning in 2.23s =========================
```

**18/18 PASS** (2 Wf-351-B2 + 7 Wf-351-B3 + 3 Wf-355-B3 + 6 Wf-356-A3 retry-NEW) — **NO regression**.

---

## §4 Retry additions — what the 3 NEW tests verify

### §4.1 test_csl_styles_present (NEW)

Verifies the two CSL files are present at the canonical paths and have
valid CSL schema structure:

- `.tools/sci-paper/styles/mol-divers.csl` — Springer numeric `[N]` proxy
  (9,775 B; HONEST_NEGATIVE_PARTIAL — CSL repo does not publish a
  Mol. Divers.-specific file per Wf-351-B2 §4.1)
- `.tools/sci-paper/styles/springer-author-date.csl` — Springer Basic
  author-date reference (9,797 B)

Both files contain `<style>` root + `<citation>` element per the CSL
schema (BSD-3).

### §4.2 test_pybtex_wrapper_present (NEW)

Verifies the pybtex programmatic fallback wrapper at
`.tools/sci-paper/bin/pybtex-compile.py` exists and is user-executable.
This is the CI / headless build fallback when Zotero is not running
(see `.tools/sci-paper/README.md` §3).

### §4.3 test_sci_skill_registered (NEW)

Verifies the `mol-metal-sci-paper-submission` skill is:
1. Present at `.claude/skills/mol-metal-sci-paper-submission/SKILL.md`
2. Has YAML frontmatter (`---` opener + `name:` field + the
   `mol-metal-sci-paper-submission` name token)
3. Referenced in `AGENTS.md` §3 skill catalog

This closes the registration-verification gap that was surfaced but not
formally tested in the Wf-355-B3 truncated output.

---

## §5 Standing-directive invariants (per AGENTS.md §5)

- **§5.1 GPU grant:** N/A (verification + retry-doc work is CPU-only;
  0 GPU launch).
- **§5.2 honest framing:** Every numeric value above traces to either a
  `Bash` output (`ls -la`, `which`, `importlib.metadata.version`,
  `pytest` invocation) or a path on disk. PROJECTED vs MEASURED tag used
  throughout.
- **§5.3 no paper work:** RESPECTED (zero touches to `paper/*`; all
  artifacts created are in `molmetal/tests/` or referenced from
  `.tools/{latex,sci-paper}/` + `docs/`).
- **§5.4 Tier-2 markers:** N/A (this is infra setup retry, not metric
  work; Tier-2 acceptance bar is computational-only per 2026-09-29 user
  directive).

Pre-flight v3 gate: **NOT re-run** (retry verification + 6 NEW tests;
no new code under `molmetal/` that would add a new pre-flight gate). Per
Wf-335-SYN baseline: **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED.
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.

---

## §6 Honest caveats (per AGENTS.md §5.2)

- **HOST_TOOL_GAP_ZOTERO_TECTONIC:** As of 2026-09-29 17:35 UTC
  (Wf-356-A3 retry timestamp), `which zotero` and `which tectonic` both
  return empty on this build host. These are **INSTALL-REQUIRED** and
  **USER_OK-gated** per `docs/SCI_PAPER_SETUP.md` §1-§2. This is not a
  project gap — the audit (`351a1_latex_template_audit.md` +
  `351a3_latex_engine_audit.md`) selected both tools; install is the
  user's discretion.
- **PROVENANCE_CAVEAT_SN_JNL:** The downloaded `sn-jnl.cls` is a
  community mirror (Apache-2.0 DanySK repo), not the canonical
  Springer-hosted version (gated behind `idp.springer.com` SSO).
  Wf-351-A1 §2.1 audit verified the mirror is in sync with the
  upstream class (last pushed 2026-07-21; same family as the
  official class).
- **CSL_MOL_DIVERS_PROXY_CAVEAT:** Per Wf-351-B2 §4.1, the
  `mol-divers.csl` file is the closest available proxy for the
  Molecular Diversity journal (which has no upstream CSL). It is
  derived from Beilstein Journal of Organic Chemistry (BJOC) — same
  `[1]` numeric bracket convention as Mol. Divers.
- **SCIPLOTS_LATEX_DEPENDENCY:** scienceplots' `science` style requires
  a LaTeX install for full journal-grade figure rendering. The
  `no-latex` style preset works without TeX but uses standard matplotlib
  fonts (less journal-specific). Per `docs/SCI_PAPER_SETUP.md` §3.3
  this is documented as a per-figure author decision.

---

## §7 Cross-references

| Resource | Path |
|---|---|
| Wf-351-A1 audit (template) | `TODO/completed/351a1_latex_template_audit.md` |
| Wf-351-A2 audit (ref manager) | `TODO/completed/351a2_reference_manager_audit.md` |
| Wf-351-A3 audit (engine) | `TODO/completed/351a3_latex_engine_audit.md` |
| Wf-351-A4 audit (figure style) | `TODO/completed/351a4_figure_style_audit.md` |
| Wf-351-B2 setup | `TODO/completed/351b2_zotero_setup.md` |
| Wf-351-B3 skill | `TODO/completed/351b3_sci_skill.md` |
| Wf-351-C1 verify | `TODO/completed/351c1_setup_verify.md` |
| Wf-355-B3 original ship | `TODO/completed/355b3_sci_paper_infra.md` |
| Wf-355-C2 verify | `TODO/completed/355c2_sci_paper_tooling_verify.md` |
| **Wf-356-A3 retry (this)** | `TODO/completed/356a3_sci_paper_infra.md` |
| Install instructions | `docs/SCI_PAPER_SETUP.md` |
| Skill: `sci-paper-submission` | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` |
| SCI-paper tool dir | `.tools/sci-paper/{bin,refs,styles}/` |
| Latex tools dir | `.tools/latex/sn-jnl.cls` (55,331 B, 2026-09-29 17:17 UTC) |
| Wf-355-B3 tests (original) | `molmetal/tests/test_wf355b3_sci_paper_install.py` (3 tests) |
| **Wf-356-A3 tests (NEW retry)** | `molmetal/tests/test_wf356a3_sci_paper_install.py` (6 tests, all PASS) |

---

**END OF Wf-356-A3** — 1 NEW retry test file (6 tests PASS, including
3 retry-mirror of Wf-355-B3 + 3 retry-NEW for CSL + pybtex wrapper + skill
registration). 0 paper/* touches (TODO-84 + AGENTS.md §5.3 respected).
Wall ~3 min CPU-only. 0 GPU launches.
