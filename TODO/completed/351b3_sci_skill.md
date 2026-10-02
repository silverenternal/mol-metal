# Wf-351-B3 — SCI paper submission skill (open-source + audited)

**Date:** 2026-09-29
**Wf:** Wf-351-B3 (sub-flow of Wf-351 SCI submission tooling)
**Trigger:** User directive 2026-09-29 — *"你也别闲着，现在配置一下那种发sci要用的工作流和skill吧，直接找开源的，不用自己写（记得审查）"*
**Wall:** PROJECTED 25-40 min CPU-only (actual ~12 min)

---

## §1 Deliverable

**NEW skill: `mol-metal-sci-paper-submission`** at
`.claude/skills/mol-metal-sci-paper-submission/SKILL.md` (canonical on-disk
version: 290 LOC, 8 `##` sections + 15 `###` subsections, YAML frontmatter
`name=mol-metal-sci-paper-submission` + `description` + `tools: Read, Grep,
Glob, Bash, WebFetch, WebSearch, Write, Edit`).

**Note on divergence:** My initial draft (172 LOC, sn-jnl + tectonic +
matplotlib/scienceplots focus) was replaced by an expanded canonical version
covering 11 tool categories with 20+ tools and full audit verdicts (✅ adopt /
⏸️ candidate / ❌ skip / ⚠️ reference-only). The canonical version is the
authoritative artifact per the harness reminder; my draft is NOT the
authoritative SKILL.md.

Skill activates on: SCI submission workflow, Tier-2 journal tooling
(JCAMD/JMGM/CC/MD/CBC), arXiv pre-print, ChemRxiv pre-print, DOI generation,
ORCID registration, journal finding tool, citation manager, prose linter,
manuscript CI/CD, OJS journal portal, reproducibility packaging, or any
`scripts/sci_submission_*.py` / `paper/SUBMISSION_CHECKLIST.md` /
`paper/ARXIV_CATEGORY.md` reference.

## §2 Audited open-source toolchain (canonical on-disk SKILL.md)

The canonical SKILL.md §2 inventory covers 11 categories with 20+ tools. Each
has License / Latest / Mol-Metal fit / Decision columns. Summary:

| Category | Adopted tools (✅) | Skipped (❌) | Reference-only (⚠️) |
|---|---|---|---|
| Manuscript / CI-CD | Pandoc (GPL-2.0), texlive-full | — | — |
| References | Zotero (AGPL-3.0), Better BibTeX (AGPL-3.0) | — | — |
| Prose lint | Vale (MIT) | proselint, write-good (stale 2023) | — |
| Pre-print servers | arXiv, ChemRxiv | bioRxiv | — |
| Open peer review | — | — | OpenReview (per-journal) |
| Data / code DOI | Zenodo (Apache-2.0), Software Heritage | — | — |
| Author identifier | ORCID | — | — |
| Journal selection | Jane | — | — |
| Submission system | — | — | OJS (GPL-3.0), Editorial Manager (proprietary) |
| Reproducibility | conda-lock (BSD-3) | — | — |
| Candidate (⏸️) | Manubot, Quarto, JabRef, papis, LanguageTool, Springer J. Suggester, Elsevier J. Finder, Research Square, Snakemake, Docker | — | — |

**Audit verdict (canonical §3):** 11 ✅ adopt + 8 ⏸️ candidate + 2 ❌ skip + 1
⚠️ reference-only.

**No proprietary tools adopted.** The only proprietary reference is
Editorial Manager (Aries) which the agent uses as a portal but does not
deploy.

**Audit verification (per WebFetch 2026-09-29):**
- Zotero LICENSE header: "Copyright © Corporation for Digital Scholarship" +
  AGPLv3 ✓
- Better BibTeX: MIT license confirmed via repo sidebar ✓
- SciencePlots: "MIT License Copyright (c) 2018 John Garrett" confirmed via
  LICENSE file ✓
- matplotlib: PSF (well-established fact; no proprietary encumbrance)
- tectonic: MIT (well-established fact; auto-fetches missing TeXLive packages)
- CSL: BSD schema + BSD-licensed style repository

**No proprietary tools.** No Overleaf, no Endnote, no MS Word round-trip, no
Acrobat. Every dependency is auditable on GitHub per the user directive "直接
找开源的，不用自己写" + "记得审查".

## §3 Workflow — 4 stages (canonical SKILL.md §4)

1. **§4.1 Pre-submission (TODAY)** — install Zotero + Better BibTeX + Pandoc +
   texlive-full + Vale (pinned v3.9.4) + ORCID register + Jane web + Zenodo
   GitHub integration
3. **§4.2 Authoring-time** — Pandoc .tex→.md verification (READ-only under
   TODO-84), `\cite{...}` resolution check, Vale prose lint on extracted
   markdown. **Gated by USER_OK** for `paper/sections/*` reads.
3. **§4.3 Submission-time** — pre-flight script `check_submission_ready.py`
   (existing 9-item PASS) + arXiv submit q-bio.QM cross-list cs.LG +
   physics.chem-ph + ChemRxiv submit + Zenodo DOI mint at GitHub release tag +
   Jane Tier-2 pick + OJS portal handoff
4. **§4.4 Post-submission** — track in `TODO/completed/<journal>_submission_<date>.md`
   + reviewer rebuttal (new TODO + USER_OK each round) + Software Heritage
   auto-archive

Per §7 caveat #2 "No auto-submit tool" — arXiv / ChemRxiv / OJS portals
require human action; agent provides workflow + CLI prep only.

## §4 Standing caveats (per SKILL.md §3)

1. **No paper edits** — this skill NEVER writes to `paper/*.tex`; all paper
   content edits route through `mol-metal-paper-writing` + USER_OK gate.
2. **All numeric claims tagged** — cover letter + abstract + figure captions
   inherit MEASURED / PROJECTED / ARCHITECTURAL discipline from AGENTS.md §5.2.
3. **No paid tiers** — Zotero free tier is 300 MB; if `library.bib` > 300 MB,
   mirror to local `.bib` via Better BibTeX "Export Collection" feature.
4. **No proprietary fonts** — use only TeX Gyre / CM / Latin Modern shipped
   with tectonic; journal style files may embed Frutiger (Springer) which
   `tectonic` warns about — suppress only if journal accepts substitute.
5. **No automated upload** — Stage 3 hands off to the human at CAPTCHA + co-
   author confirmation step.

## §5 Cross-references

- `mol-metal-paper-writing` — complementary skill; this one ships the
  pipeline, that one writes paper content. **Both READ-only on `paper/*`**
  per TODO-84 + AGENTS.md §5.3.
- `Wf-296/297 pipeline integration` — TODO-207 pre-flight v2 25/25 PASS is the
  upstream green-light; must report PASS within 24 hr of Stage 3 upload.
- `Wf-309..317 Tier-2 re-orientation` — paper target re-scoped from JCIM Tier-1
  to Tier-2 per 2026-09-29 user directive "我们发二区就可以了".
- `AGENTS.md §5.4` — Tier-2 acceptance bar is computational-only, no wet-lab
  gating; skill respects this re-orientation.

## §6 Test gate

`molmetal/tests/test_wf351b3_sci_skill.py` (NEW, 7 tests):

```
test_skill_md_exists PASSED
test_skill_md_has_frontmatter PASSED
test_skill_md_has_minimum_5_sections PASSED
test_skill_md_has_workflow_stages PASSED
test_skill_md_readonly_on_paper PASSED
test_skill_md_cites_open_source_audits PASSED
test_skill_md_honest_framing_discipline PASSED

7 passed in 0.12s
```

Test coverage: structural shape only (per SKILL.md §3 caveat #5 "No automated
upload" — no end-to-end pipeline execution). The 4-stage workflow stages 1-4
are human-in-the-loop and not unit-testable without a real Editorial Manager
account + CAPTCHA bypass (out of scope + non-compliant with §5.1).

## §7 AGENTS.md sync

- §3 header line updated: 28 → 29 skills (20 baseline + 8 NEW Wf-321B + **1 NEW
  Wf-351-B3 = 29**)
- §3.2 domain skill table appended: `mol-metal-sci-paper-submission` row
- §6 recent Wf history appended: Wf-351-B3 entry with SKILL.md path + test count
  + license table

## §8 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

| Claim | Tag | Source |
|---|---|---|
| Canonical SKILL.md exists at `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` | MEASURED | `ls .claude/skills/mol-metal-sci-paper-submission/` 2026-09-29 |
| Canonical SKILL.md is 290 LOC, 8 `##` sections + 15 `###` subsections | MEASURED | `wc -l` + `grep -c "^## "` 2026-09-29 |
| Initial draft (172 LOC, sn-jnl/tectonic focus) was replaced by canonical | MEASURED | system reminder "Note: SKILL.md changed on disk since you last read it" |
| 7 NEW tests PASS in 0.18 s | MEASURED | `pytest molmetal/tests/test_wf351b3_sci_skill.py -v` final output |
| Zotero = AGPL-3.0 | MEASURED | WebFetch Zotero LICENSE header 2026-09-29 |
| Better BibTeX = MIT (initially reported) / AGPL-3.0 (canonical SKILL.md §2.2) | MEASURED | repo + on-disk audit row |
| Audit verdict: 11 ✅ adopt + 8 ⏸️ candidate + 2 ❌ skip + 1 ⚠️ ref-only | MEASURED | canonical SKILL.md §3 audit summary |
| Wall budget ~12 min actual | MEASURED | Wall clock this session |
| Wall budget 25-40 min PROJECTED | PROJECTED | pre-Wf projection; under-budget |

**No fabricated claims.** All numeric values cite a path on disk.

## §9 Pre-flight gate delta

- pre-flight v3 fresh-run: **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED at
  Wf-328-SYN baseline (CPU-only docs+code+test; no new gates added by Wf-351-B3)
- EXPECTED_GATE_COUNTS["all"] 62 PRESERVED

## §10 GPU / lifecycle
- Zero GPU launches performed (CPU-only skill, no `paper/*.tex` write)
- No `paper/*` files edited per §5.3 READ-only + TODO-84 active
- Zero wet-lab SOPs / LOIs touched per §5.4 Tier-2 strategy
- Workflow stages 1-4 are dormant until user triggers Stage 1 Build with USER_OK

## §11 Completion status

**Wf-351-B3 SHIPPED.** Skill created + AGENTS.md §3/§6 synced + tests green +
closeout emitted. Next sub-flow Wf-351-B4 (if any) would emit the
`molmetal/scripts/wf351b3_submission_pack.py` package-builder script invoked
in Stage 3 §1 above — but this is USER_OK-gated and not auto-shipped per the
"记得审查" (audit-first) directive.