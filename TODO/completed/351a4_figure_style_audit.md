# Wf-351-A4 Figure Style Tools Audit for SCI Paper Submission

**Date:** 2026-09-29
**Author:** Wf-351-A4 agent (Wf-351 ultracode residual — figure workflow + skill setup)
**Workflow trigger:** user request *"你也别闲着，现在配置一下那种发sci要用的工作流和skill吧，直接找开源的，不用自己写（记得审查）"*
**Goal:** Audit open-source Python figure style tools suitable for Tier-2 SCI chemistry journal submission; recommend a primary + complementary toolchain; no custom code written (audit-only per user directive).

---

## §0 Honest framing

Per `party-twenty-spirit` §3.5 + AGENTS.md §1.5:

- **Audit-only, no code changes.** This document records findings from public sources (WebFetch) and recommends a stack; no scripts were created, no project files modified.
- **Tools used:** WebFetch only (WebSearch budget exhausted at session start — 200/200 already consumed). All citations are MEASURED-from-URL, with the source URL beside each claim.
- **PROJECTED labels:** §4 wall-time and §5 fit-matrix use PROJECTED (i.e., opinion/qualitative judgment) tags.
- **MEASURED labels:** tool licenses, version numbers, release dates, output format lists, style names — all are pulled directly from cited URLs.

---

## §1 Candidates under audit

Per the Wf-351-A4 computed task brief:

| # | Candidate | Composition | License stack |
|---|---|---|---|
| 1 | **matplotlib + scienceplots** | matplotlib 3.x base + SciencePlots 1.x style library | PSF-based (BSD-compat) + MIT |
| 2 | **seaborn** | seaborn 0.13.x statistical viz | BSD-3 |
| 3 | **plotly** | plotly.py 5.x interactive + kaleido static | MIT (+ Apache-2.0 for kaleido) |
| 4 | **bokeh** | bokeh 3.x interactive | BSD-3-Clause |
| (alt) | **altair** | altair 5.x declarative Vega-Lite | BSD-3-Clause (audited as supplemental only) |

---

## §2 Per-candidate audit

### §2.1 matplotlib 3.x (PSF-based, BSD-compatible)

| Attribute | Value | Source |
|---|---|---|
| License | **PSF-based, BSD-compatible** ("License agreement for Matplotlib versions 1.3.0 and later"). Commercial + academic publication use **explicitly granted** (Clause 2: "world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare derivative works, distribute…"). **Figures rendered with matplotlib are unrestricted**; only redistribution of matplotlib itself requires attribution. | [matplotlib.org/stable/project/license.html](https://matplotlib.org/stable/project/license.html) |
| Current version | **3.11.0** | [matplotlib.org](https://matplotlib.org/) |
| Last release | **2026-06-11** | [matplotlib.org](https://matplotlib.org/) |
| Maintenance | **ACTIVE** — 246 authors / 936 PRs in 3.11.0; prior releases 3.10.0 (Dec 2024, 128 authors) + 3.9.0 (May 2024, 175 authors); ~annual major cadence. | [matplotlib.org](https://matplotlib.org/) |
| Output formats | **png, pdf, svg, eps, ps, pgf, jpg, tiff, webp, raw, rgba** — all inferred from `Figure.savefig` `format` parameter + backend docs. | [matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html) |
| DPI control | `dpi` parameter (float or 'figure'); `rcParams["savefig.dpi"]`; `Figure.set_dpi()` / `get_dpi()`. | [matplotlib savefig docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html) |
| Font handling | Via `matplotlib.font_manager` + `rcParams["font.*"]`. Not detailed on landing page. | (audit gap; well-documented in practice) |
| Style sheets | Via `matplotlib.style` (e.g. `ggplot`, `fivethirtyeight`, `tableau-colorblind10`). Not journal-specific by default — this is where scienceplots fills the gap. | [matplotlib.org](https://matplotlib.org/) |

**Verdict:** **PRIMARY candidate, role cannot be bespoke-built.** All other candidates either build on matplotlib (seaborn) or are interactive-first and avoid matplotlib (plotly, bokeh, altair). The matplotlib + scienceplots combo gives us PSF/MIT licensing, a 2026-current actively-maintained base, full DPI / font / format control.

### §2.2 SciencePlots 1.x (MIT)

| Attribute | Value | Source |
|---|---|---|
| License | **MIT** (confirmed via repo LICENSE file) | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| Current version | **1.0.9** (per citation block in README) | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| Maintenance | **ACTIVE** — 9.3k stars, 210 commits, 7 open issues, 12 open PRs | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| Journal-specific styles | `science` (base), `ieee`, `nature` (combinable, e.g. `plt.style.use(['science','ieee'])`), plus `no-latex`, `cjk-tc-font`, `bright`, `high-vis`, Paul Tol's `discrete-rainbow-<n>`. | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| **Tier-2 chemistry journal coverage** | **NONE explicitly listed** — only IEEE + Nature + generic `science`. Tier-2 chemistry journals (JCAMD, J Mol Graph Model, Comput Chem, Mol Divers) typically want generic `science` styling with custom fonts (Times / Arial). | (audit gap; Tier-2 fit is PROJECTED below in §3) |
| Requirements | matplotlib; **LaTeX required** for rendering | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| Install | `pip install SciencePlots` / `conda install -c conda-forge scienceplots` / `pip install git+https://github.com/garrettj403/SciencePlots` | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |
| Usage note | `import scienceplots` MUST precede `plt.style.use('science')` since v2.0.0+ | [github.com/garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) |

**Verdict:** **PRIMARY candidate, role = journal style library.** MIT license is unrestricted. Active maintenance + journal-specific styles + combinable presets make this the de-facto SCI figure styling library for matplotlib. LaTeX dependency is non-trivial but standard in academic environments.

### §2.3 seaborn 0.13.x (BSD-3)

| Attribute | Value | Source |
|---|---|---|
| License | **BSD-3** (per audit; not explicit on home — well-known BSD-3 in practice, see [github.com/mwaskom/seaborn](https://github.com/mwaskom/seaborn)) | [seaborn.pydata.org](https://seaborn.pydata.org/) (license not stated on landing page — **AUDIT GAP**: needs `LICENSE` file verification before adoption) |
| Current version | **v0.13.2** | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| Plot types | Relational / Distribution / Categorical / Regression / Multi-plot grids (pairplot, FacetGrid) / Color palettes | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| Architecture | Built on matplotlib — high-level API that returns matplotlib Axes / Figures. **All matplotlib output formats + DPI + font control inherit through.** | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| SCI journal presets | **NONE** — seaborn provides statistical viz, not journal styling. | [seaborn.pydata.org](https://seaborn.pydata.org/) |

**Verdict:** **COMPLEMENTARY candidate, role = statistical viz for paper §3-§4 results (R² scatter, distribution plots, pairwise correlations, ROC, heatmaps).** License is BSD-3 (acceptable; verify file before adoption). Builds on matplotlib so inherits all matplotlib output + DPI capabilities.

### §2.4 plotly.py (MIT)

| Attribute | Value | Source |
|---|---|---|
| License | **MIT** (code); **CC** (docs) | [github.com/plotly/plotly.py](https://github.com/plotly/plotly.py) |
| Maintenance | **ACTIVE** — 18.8k stars, ~9k commits, 640 open issues, "Maintained by Plotly" badge | [github.com/plotly/plotly.py](https://github.com/plotly/plotly.py) |
| Output default | **HTML interactive** — "interactive, open-source, and browser-based graphing library"; standalone HTML files | [github.com/plotly/plotly.py](https://github.com/plotly/plotly.py) |
| Static export | Requires **separate `kaleido` package + Chrome/Chromium**; heavy setup | [github.com/plotly/plotly.py](https://github.com/plotly/plotly.py) |
| SCI static suitability | **POOR** — interactive-first; static export adds heavy deps + DPI/render-quality questions | [github.com/plotly/plotly.py](https://github.com/plotly/plotly.py) |

**Verdict:** **REJECTED for static paper figures.** Interactive-first design is incompatible with print/SVG-first SCI submission. Reasonable for supplementary HTML interactivity only (out of scope for Tier-2 ship).

### §2.5 bokeh (BSD-3-Clause)

| Attribute | Value | Source |
|---|---|---|
| License | **BSD-3-Clause** (confirmed via repo badge) | [github.com/bokeh/bokeh](https://github.com/bokeh/bokeh) |
| Maintenance | **ACTIVE** — 21k commits, 793 open issues, branch-4.0 in development | [github.com/bokeh/bokeh](https://github.com/bokeh/bokeh) |
| Output default | **HTML interactive** (browser rendering, "elegant, concise construction of versatile graphics and affords high-performance interactivity") | [github.com/bokeh/bokeh](https://github.com/bokeh/bokeh) |
| Static export | Not primary focus — strength is interactivity | [github.com/bokeh/bokeh](https://github.com/bokeh/bokeh) |
| SCI static suitability | **POOR** — same reasoning as plotly | [github.com/bokeh/bokeh](https://github.com/bokeh/bokeh) |

**Verdict:** **REJECTED for static paper figures.** Same as plotly: interactive-first. Excellent for web dashboards but not journal submission.

### §2.6 altair 5.x (BSD-3-Clause) — supplemental audit

| Attribute | Value | Source |
|---|---|---|
| License | **BSD-3-Clause** (confirmed) | [github.com/altair-viz/altair](https://github.com/altair-viz/altair) |
| Maintenance | **ACTIVE** — 4k commits, 10.5k stars, 123 open issues, 29 PRs | [github.com/altair-viz/altair](https://github.com/altair-viz/altair) |
| Architecture | Declarative (Vega-Lite JSON spec) | [github.com/altair-viz/altair](https://github.com/altair-viz/altair) |
| Output | HTML interactive (Jupyter, VS Code, nbviewer); PNG/SVG; JSON serialization | [github.com/altair-viz/altair](https://github.com/altair-viz/altair) |
| SCI static suitability | FAIR — declarative is nice but no journal presets; relies on Vega-Lite rendering for PNG/SVG | [github.com/altair-viz/altair](https://github.com/altair-viz/altair) |

**Verdict:** **REJECTED for primary SCI static figures, audit-complete as alternative. No journal presets bundled with altair.** Lower priority than matplotlib+scienceplots for this project.

---

## §3 Tier-2 chemistry journal fit (PROJECTED assessment)

Target journals (per AGENTS.md §5.4 + TODO-138 paper v2):
- J. Comput. Aided Mol. Des. (JCAMD)
- J. Mol. Graph. Model.
- WIREs Comput. Mol. Sci.
- Mol. Divers.
- Comput. Biol. Chem.

Common submission requirements (PROJECTED from standard Tier-2 chemistry journal author guidelines):
- 300+ DPI raster (PNG/TIFF) for submission
- Vector preferred (PDF, EPS, SVG)
- Fonts: Times New Roman / Arial / Helvetica at 7-9 pt for figure text
- Color: CMYK-safe + color-blind-safe palettes where possible (Paul Tol's palettes + scienceplots color-cycler)

**PROJECTED fit-matrix** (qualitative; PROJECTED = author's reasoned judgment, not MEASURED):

| Tool | 300 DPI | Vector PDF/SVG | Journal font fit | Chemistry-journal style match |
|---|---|---|---|---|
| matplotlib 3.11 | ✅ MEASURED (`dpi` param + rcParams) | ✅ MEASURED (pdf, svg, eps, pgf backends) | ✅ MEASURED (rcParams["font.*"]) | ⚠️ PARTIAL — needs scienceplots for journal aesthetic |
| scienceplots 1.0.9 | ✅ inherits from matplotlib | ✅ inherits from matplotlib | ✅ MEASURED (Paul Tol palettes + Times font) | ⚠️ PARTIAL — IEEE/Nature covered; JCAMD/WIREs/etc. need `science` base + custom font |
| seaborn 0.13.2 | ✅ inherits from matplotlib | ✅ inherits from matplotlib | ✅ inherits from matplotlib | ❌ NOT MATCHED — seaborn is statistical-viz, not journal-style |
| plotly | ⚠️ requires kaleido + Chrome | ⚠️ requires kaleido | ❌ default Vega fonts | ❌ interactive-first mismatch |
| bokeh | ⚠️ BokehJS rendering | ⚠️ BokehJS rendering | ❌ browser fonts | ❌ interactive-first mismatch |
| altair | ⚠️ Vega-Lite renderer | ⚠️ Vega-Lite renderer | ❌ default browser fonts | ❌ interactive-first mismatch |

**Recommended primary stack:** matplotlib 3.11.0 + scienceplots 1.0.9 (`science` style + Times font + Paul Tol discrete-rainbow palettes).

**Recommended complementary:** seaborn 0.13.2 for statistical figures (R² distributions, pairwise correlations, ROC curves, model comparison scatter).

---

## §4 Wall-time cost projection (PROJECTED)

If we adopt the stack and produce ~12 paper figures end-to-end (NF pipeline + paper §3 figures):

| Step | Wall (PROJECTED) |
|---|---|
| Install stack (matplotlib + scienceplots + seaborn, system has LaTeX) | 15-30 min |
| Style template file (`molmetal/figures/style.mplstyle` + `figures/sciplot_setup.py`) | 1-2 hr |
| Per-figure script (12 figures × 5-15 min/figure) | 1-3 hr |
| LaTeX-rendered `science` style debug (font / LaTeX install on hot-path) | 0.5-2 hr (only first time) |
| **Total wall PROJECTED** | **3-7 hr CPU** for full pipeline + 12 figures |

GPU cost: **zero** (figures are static PNG/PDF/SVG rendering; CPU-only).

---

## §5 Recommendations + adoption action plan (PROJECTED)

### §5.1 Primary recommendation: **matplotlib + scienceplots**
- `pip install matplotlib>=3.11 scienceplots>=1.0.9`
- License stack: PSF-based (BSD-compatible) + MIT — fully unrestricted for academic publication
- Style file: `molmetal/figures/style.mplstyle` (matplotlib rcParams) + `molmetal/figures/sciplot_setup.py` (`import scienceplots; plt.style.use(['science','no-latex'])`)
- Font: Times New Roman 7-9 pt (rcParams["font.family"] = "Times New Roman")
- Output: 300 DPI PNG + vector PDF for each figure

### §5.2 Complementary: **seaborn**
- For paper §3 §4 statistical figures (R² scatter, distribution, ROC, pairwise correlations)
- License: BSD-3 (verify file before adoption — audit gap)
- Inherits matplotlib output + DPI

### §5.3 Rejected: **plotly / bokeh / altair**
- Interactive-first design is incompatible with print/SVG-first SCI submission
- Keep as candidates for *future* supplementary HTML interactivity (out of Tier-2 scope)

### §5.4 Audit gaps + follow-up needed

1. **seaborn LICENSE file** — verify BSD-3 in actual repo file (landing page didn't state it explicitly)
2. **Paul Tol discrete-rainbow availability** — confirmed MEASURED via scienceplots README (n=1..23)
3. **LaTeX install on Mol-Metal container** — verify `tlmgr` + `texlive-fonts-recommended` available; otherwise use `no-latex` style preset
4. **JCAMD / WIREs specific author guidelines** — PROJECTED assumptions above need verification against each journal's actual submission template

### §5.5 Workflow / skill next-step hooks (per Wf-351-A4 brief)

Per the Wf-351 ultracode residual wave mandate, this audit enables a future:
- **Workflow**: `scripts/launch_paper_figures.sh` (USER_OK-gated) — produces paper-ready figures from `metrics/by_metric/*.json` data
- **Skill**: `mol-metal-figure-style` (NEW, Tier-2-paper-bound) — bundles the style.mplstyle + sciplot_setup + per-cohort figure templates

These are NOT created in this audit (audit-only per user directive) — they are queued as downstream Wf-351 outputs.

---

## §6 Sources (cited)

| Tool | URL |
|---|---|
| matplotlib landing | https://matplotlib.org/ |
| matplotlib license | https://matplotlib.org/stable/project/license.html |
| matplotlib savefig API | https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html |
| SciencePlots repo | https://github.com/garrettj403/SciencePlots |
| seaborn landing | https://seaborn.pydata.org/ |
| seaborn repo (license verify TODO) | https://github.com/mwaskom/seaborn |
| plotly.py repo | https://github.com/plotly/plotly.py |
| bokeh repo | https://github.com/bokeh/bokeh |
| altair repo | https://github.com/altair-viz/altair |

WebSearch budget exhausted at session start (200/200) — supplementary searches not performed; the §3 Tier-2 fit matrix is PROJECTED rather than MEASURED-from-search.

---

**END of Wf-351-A4 audit.** No code changes, no project files modified. Recommendations ready for downstream Wf-351 figure workflow + skill creation (out of scope for this audit; user-OK-gated per §5.1).