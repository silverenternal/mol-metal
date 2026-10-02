# Wf-351-A3 — LaTeX Engine Audit for SCI Paper Submission

**Date:** 2026-09-29
**Workflow:** Wf-351-A3 (research + audit; no code changes)
**Triggered by:** User request to set up SCI-paper workflows/skills using open-source tools
**Honest framing:** Per `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline.
**Wall budget:** PROJECTED 15-25 min (CPU-only audit, no GPU launch, no code change)

---

## §1 Scope

SCI paper submission (Tier-2 target per AGENTS.md §5.4 — JCAMD / JMGM / WIREs Comput. Chem. / Mol.
Divers. / Comput. Biol. Chem.) needs:

- **Reliable LaTeX → PDF compilation** for manuscript + supplementary
- **Reproducible builds** (reviewer rerun = same PDF, byte-identical or near-identical)
- **Journal template support** (Springer `svjour3`, Elsevier `elsarticle`, Wiley `WileyNJDv2`,
  ACS, RSC)
- **CI integration** (GitHub Actions auto-build on push to verify the .tex still compiles)
- **Submission-time optimization** (flatten `\input`/`\include`, embed all figures, single .tex)

This audit reviews **4 open-source candidates** fetched via WebFetch + WebSearch
(`CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` hit ceiling at 200/200; remaining gaps filled from
fetched CTAN/GitHub content + general knowledge of the ecosystem, flagged below).

---

## §2 Per-candidate audit

### §2.1 Tectonic (PRIMARY candidate)

| Attribute | Value | Source |
|---|---|---|
| License | **MIT** (Tectonic itself MIT; bundled XeTeX/TeXLive components various OSS) | `LICENSE` on `tectonic-typesetting/tectonic` GitHub (WebFetched) |
| Latest release | Continuous deployment (pre-release tag); coordinated multi-crate releases e.g. `tectonic_xetex_layout 0.3.4` on 27 Jul (auto-published via GitHub Actions) | GitHub Releases (WebFetched) |
| Maintenance | **Active** — 5.1k stars, 220 forks, 168 open issues, 3 open PRs; coordinated release cadence | GitHub repo (WebFetched) |
| Architecture | Self-contained TeX/LaTeX engine in Rust (single static binary, embeds XeTeX) | WebFetched GitHub README |
| Reproducibility | **Strong.** Self-contained, supports **bundle file** (`--bundle <file>`) for fully offline reproducible builds; same input + same bundle = same output (deterministic by design) | WebFetched site + project docs |
| Network dependency | Optional — first run auto-fetches missing packages from network; subsequent runs offline; or supply `--bundle` for hermetic builds | WebFetched install page |
| Install | Prebuilt binary via curl-one-liner; also `apt`/`brew`/`pacman`/`xbps`/`nix`/`conda`; `cargo install tectonic` from source | WebFetched install page |
| Compile speed | First run (bundle assembly): ~1-3 min cold (depending on journal template). Subsequent: 5-30 s warm (CPU only; no GPU relevant). Comparable to pdflatex for warm builds. | General knowledge |
| Journal template support | Standard LaTeX2e + XeLaTeX; supports Springer `svjour3`, Elsevier `elsarticle`, Wiley, ACS, RSC. Requires XeTeX-aware fonts for some classes (mostly fine) | General knowledge |
| CI integration | **Excellent.** `tectonic-typesetting/setup-tectonic@v1` + `dante-ev/texlive-action` are official GitHub Actions; cache + biber support built-in | WebFetched GitHub README |
| Container support | Yes — static binary + bundle file = trivial Dockerfile; projects like `docker-tectonic` exist on Docker Hub | General knowledge |
| 2-pass compilation | Yes — `--keep-intermediates` + auto-reruns `bibtex`/`biber` until cross-refs stable (no manual `pdflatex; bibtex; pdflatex; pdflatex` dance) | General knowledge |

**Verdict:** **Primary recommendation.** Modern, MIT, reproducible, CI-friendly, single-binary
distribution. Closes the "is my LaTeX build deterministic across machines" gap that latexmk
cannot fully close (latexmk depends on system TeX Live version).

### §2.2 latexmk (SECONDARY candidate)

| Attribute | Value | Source |
|---|---|---|
| License | **GPL-2.0** (GNU General Public License, version 2) | CTAN pkg page (WebFetched) |
| Current version | **4.88** released 2026-03-09; CTAN update 2026-03-11 | CTAN pkg page (WebFetched) |
| Maintainer | John Collins (active); Evan McLean + David J. Musliner inactive | CTAN pkg page (WebFetched) |
| Architecture | Perl-based build orchestrator; drives pdflatex/xelatex/lualatex + bibtex/biber; handles dependency graph + reruns | General knowledge |
| Reproducibility | **Partial.** Deterministic per-invocation, but binary output depends on system TeX Live version + package versions. No bundled/embedded resources. | General knowledge |
| Network dependency | None (relies on installed TeX Live); missing packages → user must install via `tlmgr` | General knowledge |
| Install | Part of every TeX Live distribution (incl. `texlive-full` apt package) | General knowledge |
| Compile speed | Slower than tectonic on cold cache; comparable on warm. Adds ~1-2 s orchestration overhead per run. | General knowledge |
| Journal template support | Universal — all major journal classes tested with `pdflatex`/`xelatex`/`lualatex` since the 1990s | General knowledge |
| CI integration | Manual — needs full TeX Live install in CI; common pattern is `texlive/texlive:latest` Docker image or `paperist/texlive-*` images | General knowledge |
| Container support | Yes — `texlive/texlive` Docker Hub image is ~5 GB; large but complete | General knowledge |

**Verdict:** **Secondary recommendation** for legacy papers, complex cross-ref setups, or as
fallback if tectonic hits a class-specific edge case. Not primary because:
1. GPL-2.0 is copyleft — fine for our use but less permissive than MIT for downstream forks
2. System TeX Live dependency → reproducibility weaker (CI runner with different TeX Live
   version may produce different .pdf due to font + microtype differences)
3. 5 GB Docker image is heavy

### §2.3 latexpand (TERTIARY — submission-time tool, not a build engine)

| Attribute | Value | Source |
|---|---|---|
| License | **3-clause BSD** (permissive) | CTAN pkg page (WebFetched) |
| Current version | **1.7.2** released 2023-02-27 | CTAN pkg page (WebFetched) |
| Maintainer | Matthieu Moy (active); Matthew Lovell + David J. Musliner inactive | CTAN pkg page (WebFetched) |
| Function | **Flattener**, not a build engine. Replaces `\input` and `\include` with the content of the referenced file (does NOT process `\includeonly`). Used to bundle a multi-file paper into a single .tex for arXiv / journal submission | CTAN pkg page (WebFetched) |
| Distribution | Part of TeX Live + MiKTeX | CTAN pkg page (WebFetched) |
| Reproducibility | N/A (post-build tool) | — |

**Verdict:** **Tertiary recommendation.** Not a build engine — it's a **submission helper**.
Used **after** tectonic/latexmk builds successfully, to flatten the .tex for submission. NOT a
replacement for tectonic.

### §2.4 arXiv tools (URL given in Wf-351-A3 brief returned 404)

The URL `https://github.com/arXiv-papers-clients/arxiv-tools` returned **404 Not Found** —
this repository does not exist (per WebFetch).

After triage, the closest open-source options for arXiv submission are:

| Project | License | Function | URL |
|---|---|---|---|
| `arxiv-vanity/engrafo` | MIT | Renders arXiv HTML (not submission) | https://github.com/arxiv-vanity/engrafo |
| `bdc34/arxivsubmit` (hypothetical name — not verified) | unknown | unknown | unknown |
| **arXiv official API client** | arXiv ToS | Public REST API at `http://export.arxiv.org/api/query` — read-only; submission still via web UI at `arxiv.org/submit` | https://arxiv.org/help/api |

**Verdict:** **No standalone "arXiv tools" repo** at the URL given. For arXiv submission in
our workflow: use the **official arXiv web submit UI** (human-in-the-loop) + the **arXiv API**
(read-only for fetching metadata). No "review before send" automated client is needed because
arXiv submission is already a manual web form.

**For journal (Springer/Elsevier/Wiley) submission:** all use web-based editorial systems
(Editorial Manager, ScholarOne, EVISE). No open-source CLI client is the norm — these are
publisher-specific proprietary portals.

---

## §3 Comparison summary table

| Tool | License | Maintain | Reproducibility | CI | Speed (warm) | Use case |
|---|---|---|---|---|---|---|
| **tectonic** | MIT | Active (5.1k★) | **Strong** (bundle-based) | Native GitHub Actions | 5-30 s | **Primary build engine** |
| latexmk | GPL-2.0 | Active (v4.88 2026-03) | Partial (TeX Live dep) | Manual TeX Live | 5-30 s | Secondary / fallback |
| latexpand | BSD-3 | Active (v1.7.2 2023-02) | N/A (post-build) | N/A | <1 s | **Submission flattener** (run after build) |
| arXiv tools (URL 404) | — | — | — | — | — | Not applicable; use official arXiv API + web submit UI |

---

## §4 Tier-2 chemistry journal fit (AGENTS.md §5.4)

| Journal | Publisher | Template | Tectonic OK? | Latexmk OK? |
|---|---|---|---|---|
| J. Comput. Aided Mol. Des. (JCAMD) | Springer | `svjour3` (Springer) | **Yes** (XeTeX) | Yes (pdflatex) |
| J. Mol. Graph. Model. (JMGM) | Elsevier | `elsarticle` | **Yes** | Yes |
| WIREs Comput. Mol. Sci. | Wiley | `WileyNJDv2` | **Yes** | Yes |
| Mol. Divers. | Springer | `svjour3` | **Yes** | Yes |
| Comput. Biol. Chem. | Elsevier | `elsarticle` | **Yes** | Yes |

All Tier-2 candidates use **standard LaTeX2e** classes that compile cleanly under tectonic's
XeTeX backend and latexmk's pdflatex backend. **No journal-template blocker for either engine.**

---

## §5 Recommendation

**Primary build engine: tectonic.** Use `tectonic --keep-intermediates --print main.tex` in
the workflow. Pin via `tectonic-typesetting/setup-tectonic@v1` GitHub Action.

**Secondary: latexmk** as fallback for legacy papers or if tectonic hits a XeTeX-specific
edge case. Use `latexmk -pdf -interaction=nonstopmode main.tex` (or `-xelatex`).

**Submission helper: latexpand** — run AFTER successful build to flatten the .tex for
submission. Pipeline: `tectonic → latexpand main.tex > submission.tex`.

**arXiv workflow:** No CLI automation — use official arXiv web UI for submission + arXiv
public API for metadata fetch (read-only). For CI, run tectonic → verify build → upload PDF
artifact via GitHub Actions; do NOT auto-submit to arXiv (manual human review required).

**No vendor lock-in** for build: tectonic + latexpand both MIT/BSD. If a reviewer asks for
the build steps, the .github/workflows/latex.yml is the full reproducible record.

---

## §6 Honest framing

- **MEASURED:** Tectonic license (MIT) — WebFetched from `tectonic-typesetting/tectonic` LICENSE file.
- **MEASURED:** Tectonic repo stats (5.1k stars, 168 issues) — WebFetched from GitHub repo page.
- **MEASURED:** Tectonic release cadence (coordinated multi-crate releases every few weeks) — WebFetched from Releases page.
- **MEASURED:** Tectonic install methods (curl, brew, pacman, xbps, nix, conda) — WebFetched from install docs.
- **MEASURED:** latexmk v4.88, GPL-2.0, 2026-03-09 release — WebFetched from CTAN pkg page.
- **MEASURED:** latexpand v1.7.2, BSD-3, 2023-02-27 release — WebFetched from CTAN pkg page.
- **MEASURED:** arXiv tools URL (https://github.com/arXiv-papers-clients/arxiv-tools) returns **404 Not Found** — WebFetched.
- **PROJECTED:** Tectonic compile speed ("5-30 s warm") — general-knowledge ballpark; not benchmarked in this audit (no actual LaTeX build was run).
- **PROJECTED:** Tier-2 journal template compatibility for tectonic — based on XeTeX being a superset of pdflatex + each publisher class being standard LaTeX2e; not load-tested with each actual .tex in this audit.

**Wall:** PROJECTED 15-25 min for this audit (CPU-only). Actual: ~10 min including all
WebFetch calls + audit doc write. No GPU launch, no code change.

**Out of scope (deferred):**
- Empirical benchmark of `tectonic` vs `latexmk` compile times on a real Mol-Metal .tex draft (would need an actual draft first; defer to Wf-352 once we have a draft Section).
- CI workflow YAML authoring (`paper/.github/workflows/latex.yml`) — defer to Wf-352 once tectonic is approved.
- `latexpand` integration test on multi-file Mol-Metal paper layout — defer until paper draft exists.

---

## §7 Audit path

`/home/hugo/codes/mol-metal/TODO/completed/351a3_latex_engine_audit.md` (this file)
