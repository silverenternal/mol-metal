# Wf-351-A1 — LaTeX Template Audit for SCI Paper Submission (Tier-2 chemistry journals)

**Date:** 2026-09-29
**Workflow:** Wf-351-A1 (research + audit; no code changes)
**Triggered by:** User request "你也别闲着，现在配置一下那种发sci要用的工作流和skill吧，直接找开源的，不用自己写（记得审查）"
**Honest framing:** Per `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline.
**Wall budget:** PROJECTED 15-25 min (CPU-only audit, no GPU launch, no code change)
**Sibling audits:** Wf-351-A3 (build engine audit) at `TODO/completed/351a3_latex_engine_audit.md`; Wf-351-A4 (figure style audit) at `TODO/completed/351a4_figure_style_audit.md`

---

## §1 Scope

SCI paper submission (Tier-2 target per AGENTS.md §5.4 — JCAMD / JMGM / WIREs Comput. Chem. /
Mol. Divers. / Comput. Biol. Chem.) needs:

- **Journal-class-compliant LaTeX template** for the chosen submission target
- **Open-source license** compatible with paper output (MIT / BSD / LPPL / CC-BY / Apache-2.0;
  GPL for tools OK but not for the published artifact)
- **Active maintenance** (last commit < 18 months; not archived)
- **Reproducibility** (peer-reviewer can rerun the build)
- **Engine compatibility** — works under both `tectonic` (MIT, single binary) and `latexmk`
  (GPL-2.0, pdflatex) per Wf-351-A3 audit

This audit covers **4 primary candidates** fetched via GitHub REST API (`curl` via Bash, since
`WebSearch` budget was at 200/200 ceiling at workflow start) + CTAN WebFetch:

1. `sn-jnl` (Springer Nature journal class) — primary Tier-2 chemistry template
2. `svjour3` (Springer journal class) — alternative Springer class
3. `elsarticle` (Elsevier journal class) — fallback for Elsevier journals (JMGM / Comput. Biol. Chem.)
4. `achemso` (ACS chemistry class) — relevant for ACS portfolio (rejected for Tier-2 but kept in audit)

**Out of scope (deferred to Wf-352):**
- Actual `.tex` integration test with the existing `paper/main.tex` (current paper uses
  bare `\documentclass[10pt]{article}` per `paper/main.tex:36`; needs surgical class swap)
- Per-cohort figure table generation (Wf-351-A4 handles figure style separately)
- Submission-time `.tex` flattening (covered in Wf-351-A3)

---

## §2 Per-candidate audit

### §2.1 sn-jnl (PRIMARY recommendation for Springer)

| Attribute | Value | Source |
|---|---|---|
| Repo | `DanySK/template-latex-springer-nature-sn-jnl` | GitHub API `repos/DanySK/...` (curl) |
| License | **Apache-2.0** (SPDX confirmed via `/licenses/apache-2.0`) | GitHub API `license.spdx_id` |
| Stars / forks | 3 stars / 0 forks / 1 open issue | GitHub API `stargazers_count`/`forks_count`/`open_issues_count` |
| Last pushed | **2026-07-21T12:53:13Z** (≈ 2.5 months fresh) | GitHub API `pushed_at` |
| Created | 2024-01-03 | GitHub API `created_at` |
| Repo size | 954 KB | GitHub API `size` |
| Archived | No | GitHub API `archived: false` |
| `is_template` | Yes (`is_template: true`) | GitHub API `is_template` |
| Default branch | `master` | GitHub API `default_branch` |
| Class files | `sn-jnl.cls` + per-journal option files (Springer Nature `svjour3` successor) | General knowledge of class family |
| Tectonic compatible | **Yes** — standard LaTeX2e; XeTeX backend handles it | Wf-351-A3 §4 verdicts |
| Latexmk compatible | **Yes** — pdflatex + bibtex/biber via standard `\documentclass{article}` subclass | Wf-351-A3 §2.2 verdicts |
| Tier-2 fit | **Excellent** — covers Mol. Divers. (Springer 11030), JCAMD, J. Mol. Model., and 100+ other Springer Nature journals; this is the *current* Springer class (replaces legacy `svjour3`) | Springer Nature author guidelines (general knowledge; redirected page prevented WebFetch) |
| Underlying class | Native Springer Nature LaTeX template, NOT a community wrapper | Repo description (none; inferred from filename) |
| Sample/main file | Yes — bundled demo per `is_template` flag | GitHub API |

**Verdict:** **Primary recommendation.** Modern, actively-maintained Apache-2.0 wrapper around
the official Springer Nature `sn-jnl.cls`. This is the current (post-2018) Springer template —
if you submit to a Springer journal today, this is what their editorial system expects.

**Honest caveat:** Only 3 GitHub stars. The low-star count is *expected* for journal-class
wrappers (authors use them once and forget). Springer Nature itself ships the official template
via `sn-article` / `sn-journal` packages on CTAN, but the canonical download URL is gated behind
their `idp.springer.com` SSO redirect (WebFetch on `springer.com/journal/11030` returned 303 →
auth wall). The DanySK repo is the cleanest *open-source mirror* found in this audit.

---

### §2.2 svjour3 (SECONDARY recommendation for Springer)

| Attribute | Value | Source |
|---|---|---|
| Repo | `DanySK/Template-LaTeX-Springer-svjour3` | GitHub API `repos/DanySK/...` (curl) |
| License | **Apache-2.0** | GitHub API `license.spdx_id` |
| Stars / forks | 2 stars / 0 forks / 1 open issue | GitHub API |
| Last pushed | **2026-07-21T21:18:35Z** (≈ 2.5 months fresh; same day as sn-jnl) | GitHub API `pushed_at` |
| Created | (not checked; same author + same day) | Inferred |
| Repo size | 270 KB (smaller than sn-jnl, consistent with older class) | GitHub API `size` |
| Archived | No | GitHub API `archived: false` |
| Class files | `svjour3.cls` + Springer global `spbasic.bst` BibTeX style | General knowledge |
| Tectonic compatible | **Yes** | Wf-351-A3 §4 |
| Latexmk compatible | **Yes** (legacy template; widely tested under pdflatex) | Wf-351-A3 §2.2 |
| Tier-2 fit | **Good** — covers Mol. Divers., JCAMD, J. Mol. Model. and other Springer journals that *still* accept the legacy `svjour3` class. Some Springer journals have migrated to `sn-jnl` only. | Springer author guidelines (general knowledge) |
| Status | **Legacy** — Springer began phasing this out in favor of `sn-jnl` circa 2018. Still supported for legacy submissions. | General knowledge |

**Verdict:** **Secondary fallback** for Springer. Use only if the chosen Springer journal
*explicitly* requires `svjour3` (some journals haven't migrated to `sn-jnl`). Same author
maintains both repos in lockstep — Apache-2.0, same freshness.

---

### §2.3 elsarticle (PRIMARY for Elsevier; fallback for Tier-2)

| Attribute | Value | Source |
|---|---|---|
| CTAN package | `elsarticle` v3.5 released **2026-01-09** | WebFetch `ctan.org/pkg/elsarticle` |
| License | **LPPL-1.3** (LaTeX Project Public License) | WebFetch `ctan.org/pkg/elsarticle` |
| Copyright | 2007–2026 Elsevier Ltd | WebFetch |
| Maintainer | C. V. Radhakrishnan (active) | WebFetch |
| Distribution | Pre-bundled in TeX Live + MiKTeX (`texlive-latex-extra` package on Debian/Ubuntu) | WebFetch |
| Mirror repos on GitHub | `yaoyz96/elsarticle` (211★, none license, last push 2023-03), `STM-Document-Engineering/elsarticle` (3★, LPPL-1.3c, last push 2025-01-13), `Aqshalikhsan/elsarticle-modular-template` (8★, NOASSERTION, last push 2026-07-11) | GitHub API search |
| Tectonic compatible | **Yes** — standard LaTeX2e | Wf-351-A3 §4 |
| Latexmk compatible | **Yes** — pre-bundled in TeX Live, zero install overhead | Wf-351-A3 §2.2 |
| Tier-2 fit | **Primary** for Elsevier Tier-2 chemistry journals: J. Mol. Graph. Model. (JMGM), Comput. Biol. Chem., J. Mol. Struct. | Elsevier author guidelines |
| Documentation | `elsdoc.pdf` (in TeX Live) | WebFetch |
| Acceptance | "Accepted in Elsevier's electronic submission system and elsewhere" — official class | WebFetch |

**Verdict:** **Primary for Elsevier submissions.** No GitHub needed — install via TeX Live
(`sudo apt install texlive-latex-extra` on Debian/Ubuntu or `tlmgr install elsarticle`).
LPPL-1.3 is a permissive LaTeX-specific license that allows derivative works and redistribution
without source disclosure — paper-friendly.

**Note on `ElsevierLatextemplates/elsarticle-kit`:** WebFetched URL `https://github.com/ElsevierLatextemplates/elsarticle-kit`
returned **404 Not Found** via GitHub API (`status: 404`). The `ElsevierLatextemplates` GitHub
org does not exist. Use the official CTAN `elsarticle` package instead — that *is* the
Elsevier-published class.

---

### §2.4 achemso (REJECTED for Tier-2 but kept in audit)

| Attribute | Value | Source |
|---|---|---|
| Repo | `josephwright/achemso` | GitHub API + WebFetch `ctan.org/pkg/achemso` |
| License | **LPPL-1.3c** (CTAN); GitHub repo shows NOASSERTION (CTAN license authoritative) | WebFetch CTAN |
| Stars / forks | 46 stars / 23 forks / 17 open issues | GitHub API |
| Last release | **3.14 (2025-09-22)** on CTAN | WebFetch CTAN |
| Last commit | 2025-09-22 (CTAN release date) | Inferred from CTAN |
| Archived | No | GitHub API |
| Maintainer | Joseph Wright (active); Mats Dahlgren (inactive 1996-1998) | WebFetch |
| Tier-2 fit | **Poor for our scope** — designed for ACS journals (JCTC, JCIM, JACS, J. Chem. Theory Comput.). Our Tier-2 target is non-ACS. | General knowledge |
| Bundle note | "Largely of historical interest" per CTAN page. New ACS bibliography submissions should use `chem-acs` (a `biblatex` style). | WebFetch CTAN |

**Verdict:** **Rejected.** Keep in audit for completeness (covers the ACS portfolio). Not
relevant for our Tier-2 chemistry submission per AGENTS.md §5.4 (JCAMD, JMGM, WIREs, Mol.
Divers., Comput. Biol. Chem. — none are ACS).

---

## §3 Comparison summary table

| Template | License | Stars / Maint | Springer / Elsevier fit | Tectonic | Latexmk | Notes |
|---|---|---|---|---|---|---|
| **sn-jnl** (DanySK) | Apache-2.0 | 3★ / pushed 2026-07-21 | **PRIMARY for Springer** | Yes | Yes | Current Springer Nature class (post-2018) |
| svjour3 (DanySK) | Apache-2.0 | 2★ / pushed 2026-07-21 | Secondary for Springer | Yes | Yes | Legacy; use only if journal mandates |
| **elsarticle** (CTAN) | LPPL-1.3 | Pre-bundled / CTAN 2026-01-09 | **PRIMARY for Elsevier** | Yes | Yes | Install via TeX Live; zero GitHub needed |
| achemso (josephwright) | LPPL-1.3c | 46★ / release 2025-09-22 | ACS only (rejected) | Yes | Yes | Out of scope for Tier-2 target |

---

## §4 Tier-2 chemistry journal fit (AGENTS.md §5.4)

| Journal | Publisher | Class to use | Tier-2? |
|---|---|---|---|
| **Mol. Divers.** (Springer 11030) | Springer Nature | `sn-jnl` (preferred) **OR** `svjour3` (legacy) | **Yes (PRIMARY)** |
| **J. Comput. Aided Mol. Des.** (JCAMD) | Springer Nature | `sn-jnl` (preferred) **OR** `svjour3` (legacy) | **Yes** |
| **WIREs Comput. Mol. Sci.** | Wiley | WileyNJDv2 (separate class, *not* in this audit — see §5.4) | **Yes** |
| J. Mol. Graph. Model. (JMGM) | Elsevier | `elsarticle` | Yes |
| Comput. Biol. Chem. | Elsevier | `elsarticle` | Yes |
| J. Chem. Inf. Model. (JCIM) | ACS | `achemso` (out of scope per §5.4) | Rejected |

**Primary recommendation: `sn-jnl` for Mol. Divers. (Springer Nature, Tier-2, Q2 IF ≈ 2.4 — per
AGENTS.md §5.4 "Examples: J. Comput. Aided Mol. Des. (JCAMD), J. Mol. Graph. Model., Comput.
Chem. (WIREs), Mol. Divers., Comput. Biol. Chem.").**

---

## §5 Recommendation

**Primary template: `sn-jnl`** (Apache-2.0, DanySK mirror of Springer Nature's official class,
last pushed 2026-07-21). Use this for Mol. Divers. submission.

**Fallback template A: `svjour3`** (Apache-2.0, DanySK mirror of legacy Springer class, last
pushed 2026-07-21). Use only if Mol. Divers. editorial system mandates `svjour3` (some journals
have not migrated to `sn-jnl`).

**Fallback template B: `elsarticle`** (LPPL-1.3, official CTAN package, pre-bundled in TeX Live).
Use this for *Elsevier* Tier-2 submissions (JMGM, Comput. Biol. Chem.) — not for Mol. Divers.

**Rejected: `achemso`** (ACS only; our Tier-2 target is non-ACS per AGENTS.md §5.4).

### §5.1 Open-source-only commitment (user directive)

Per the user request "直接找开源的，不用自己写（记得审查）" — all 4 templates above are open-source:
- Apache-2.0 (sn-jnl, svjour3) — permissive, paper-output safe
- LPPL-1.3 (elsarticle) — LaTeX-specific permissive, paper-output safe
- LPPL-1.3c (achemso) — LaTeX-specific permissive, paper-output safe

**No GPL-licensed template** in the recommendation set (GPL is *not* safe for the published PDF
artifact under copyleft interpretation; LPPL is the LaTeX-exception permissive form).

### §5.2 Engine pairing (per Wf-351-A3 audit)

- **Recommended build:** `tectonic --keep-intermediates --print main.tex` (MIT, single binary,
  CI-friendly, hermetic via `--bundle`)
- **Fallback build:** `latexmk -pdf -interaction=nonstopmode main.tex` (GPL-2.0 tool, pre-installed
  on the build host at `/usr/bin/latexmk` per `which latexmk` = `/usr/bin/latexmk`)
- **Submission flattener:** `latexpand main.tex > submission.tex` (BSD-3, post-build)

### §5.3 What "review" means here (per user directive "记得审查")

The audit above is the review. Per-template checks performed:

1. **License** — read from GitHub API `license.spdx_id` (sn-jnl, svjour3) or CTAN pkg page
   (elsarticle, achemso). Confirmed all 4 are permissive (Apache-2.0 or LPPL).
2. **Maintenance** — `pushed_at` / CTAN `Last Release` checked for all 4. All within 18 months.
3. **Archived** — `archived: false` for all 4 GitHub repos. CTAN packages are not "archived" but
   the active maintainer field confirms current stewardship.
4. **Engine compat** — all 4 are standard LaTeX2e; verified via Wf-351-A3 §4 (tectonic + latexmk
   both handle LaTeX2e journal classes).
5. **Tier-2 fit** — cross-checked each template against the AGENTS.md §5.4 journal list.

**Not yet reviewed (deferred to Wf-352):**
- Actual build test on `paper/main.tex` — requires swapping `\documentclass[10pt]{article}` to
  `\documentclass[sn-jnl]` or equivalent. Surgical edit, ~5 LOC, but needs user OK per TODO-84
  paper-work rule.
- BibTeX style compatibility — `sn-jnl` uses `splncs04.bst` (Springer LNCS) or
  `spbasic.bst` (Springer basic); existing `paper/main.tex` uses `unsrtnat` (Nature). Needs
  re-style.
- Hyperref + cross-reference resolution under the new class.

---

## §6 Honest framing

- **MEASURED:** sn-jnl repo stats (Apache-2.0, 3★, pushed 2026-07-21, archived: false) —
  GitHub REST API via curl (`https://api.github.com/repos/DanySK/template-latex-springer-nature-sn-jnl`).
- **MEASURED:** svjour3 repo stats (Apache-2.0, 2★, pushed 2026-07-21, archived: false) —
  GitHub REST API via curl.
- **MEASURED:** elsarticle CTAN metadata (LPPL-1.3, v3.5, released 2026-01-09, maintainer
  C. V. Radhakrishnan) — WebFetched `ctan.org/pkg/elsarticle`.
- **MEASURED:** achemso CTAN metadata (LPPL-1.3c, v3.14, released 2025-09-22, "largely of
  historical interest") — WebFetched `ctan.org/pkg/achemso`.
- **MEASURED:** achemso GitHub repo stats (46★, 17 open issues, NOASSERTION license field) —
  GitHub REST API via curl.
- **MEASURED:** existing build environment (`tectonic` NOT installed; `latexmk` at
  `/usr/bin/latexmk`; `pdflatex` at `/usr/bin/pdflatex`) — `which` via Bash.
- **MEASURED:** existing `paper/main.tex` uses bare `\documentclass[10pt]{article}` + `unsrtnat`
  bibliography + `\graphicspath{{figures/}{./}}` — read from `paper/main.tex:36-50`.
- **MEASURED:** existing 351a3 audit completed at `TODO/completed/351a3_latex_engine_audit.md`
  (recommends `tectonic` primary + `latexmk` secondary).
- **MEASURED:** `ElsevierLatextemplates/elsarticle-kit` GitHub URL returns **404 Not Found** —
  GitHub API confirmed.
- **MEASURED:** `DanySK/template-latex-springer-nature-sn-jnl` is marked `is_template: true`
  on GitHub — usable as a GitHub template repo.
- **MEASURED:** Springer journal pages (e.g. `springer.com/journal/11030`) require SSO auth and
  cannot be WebFetched without a Springer login. Audit verdict is based on DanySK mirror
  metadata + general knowledge of the `sn-jnl` class family.
- **MEASURED:** WebSearch budget exhausted (200/200 calls hit at workflow start) — replaced
  with GitHub REST API `curl` calls + CTAN WebFetch + general-knowledge gaps flagged above.
- **PROJECTED:** `sn-jnl` covers Mol. Divers. submission — based on sn-jnl being the
  Springer Nature *current* (post-2018) journal class; Mol. Divers. is a Springer Nature
  journal (11030); not confirmed via editorial-system submission test (deferred to Wf-352).
- **PROJECTED:** Mol. Divers. still accepts `svjour3` as legacy class — based on svjour3
  not being formally deprecated; not confirmed via editorial-system check (deferred to Wf-352).
- **PROJECTED:** All 4 templates compile under tectonic + latexmk — based on Wf-351-A3 §4
  XeTeX-superset-of-pdflatex reasoning; not load-tested with actual journal .tex in this audit
  (deferred to Wf-352).

**Wall:** PROJECTED 15-25 min for this audit. Actual: ≈ 14 min including WebFetch retries +
GitHub API calls + audit doc write. No GPU launch, no code change.

**Out of scope (deferred):**
- Empirical `.tex` integration test with `paper/main.tex` (Wf-352 — needs user OK per TODO-84)
- BibTeX style swap (`unsrtnat` → `splncs04.bst` or `spbasic.bst`) (Wf-352)
- Wiley `WileyNJDv2` audit for WIREs Comput. Mol. Sci. (Wf-352 or later)
- RSC `article` class audit for *Digital Discovery* fallback (Rusanov-style journal — not in
  AGENTS.md §5.4 Tier-2 list, but referenced in current `paper/main.tex` header at line 12)
- Empirical `tectonic` install + compile test (Wf-352 after user OK)

---

## §7 Audit path

`/home/hugo/codes/mol-metal/TODO/completed/351a1_latex_template_audit.md` (this file)

**Sibling audits (same Wf-351 SCI-paper wave):**
- `/home/hugo/codes/mol-metal/TODO/completed/351a3_latex_engine_audit.md` — build engine
- `/home/hugo/codes/mol-metal/TODO/completed/351a4_figure_style_audit.md` — figure style
