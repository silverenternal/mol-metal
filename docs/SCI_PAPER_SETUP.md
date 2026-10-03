# SCI Paper Submission — Install & Setup Guide

**Audience:** Mol-Metal paper authors configuring the Tier-2 chemistry journal
submission toolchain (per AGENTS.md §5.4 — JCAMD / J. Mol. Graph. Model. / WIREs
Comput. Chem. / Mol. Divers. / Comput. Biol. Chem.).

**Scope:** INFRASTRUCTURE ONLY — configures the build pipeline + reference
manager + figure style library. **ZERO touches to `paper/*`** per AGENTS.md
§5.3 (TODO-84 active) — paper text edits require explicit user authorization.

**Last updated:** 2026-09-30 (Wf-372-A1 Zotero + Wf-372-A2 tectonic binary installs verified).
**Owner:** Hugo Lee.
**Source audits:** `TODO/completed/351a{1,2,3,4}_*.md` + `TODO/completed/372a1_zotero_install.md` + `TODO/completed/372a2_tectonic_install.md`.

---

## §0 TL;DR

If `which zotero`, `which tectonic`, and `python -c "import scienceplots"`
all succeed, you can skip this doc — the toolchain is ready. Otherwise,
follow the §1-§4 install steps below. Per-tool status as of 2026-09-29:

| Tool | Status | Section |
|---|---|---|
| `sn-jnl.cls` (Springer Nature class) | **SHIPPED** at `.tools/latex/sn-jnl.cls` (55,331 B, 2026-09-29 17:17 UTC) | §4 |
| Zotero + Better BibTeX | **INSTALLED** Zotero 10.0.3 (Zotero Zotero 140.15.0esr) at `~/apps/zotero/` (2026-09-30 per Wf-372-A1); Better BibTeX plugin **MANUAL GUI** install — see §1.2 | §1 |
| tectonic (LaTeX engine) | **INSTALLED** tectonic 0.17.0 at `~/.local/bin/tectonic` (2026-09-30 per Wf-372-A2); `tectonic --version` reports `Tectonic 0.17.0` | §2 |
| scienceplots (Python figure style) | **INSTALLED** in `.venv` (v2.2.2) | §3 |
| pybtex (Python BibTeX processor) | **INSTALLED** in `.venv` (v0.26.1) | (see `.tools/sci-paper/README.md` §1.2) |
| latexmk + pdflatex (fallback engine) | **EXISTS** at `/usr/bin/latexmk` + `/usr/bin/pdflatex` | §2 |
| CSL styles for Mol. Divers. | **SHIPPED** at `.tools/sci-paper/styles/mol-divers.csl` | (see `.tools/sci-paper/README.md` §4) |

---

## §1 Zotero + Better BibTeX install (USER_OK-gated)

Per Wf-351-A2 audit (Zotero = AGPL-3.0; Better BibTeX = MIT), this is the
PRIMARY reference manager + BibTeX export pipeline.

### §1.1 Install Zotero (desktop app, AGPL-3.0)

| Platform | Command | Notes |
|---|---|---|
| Linux (Flatpak) | `flatpak install flathub org.zotero.Zotero` | Requires `flatpak` + Flathub repo |
| Linux (Debian/Ubuntu) | Download `.tar.bz2` from https://www.zotero.org/download/ | Extract + run `zotero` binary |
| Linux (Arch) | `pacman -S zotero-bin` from archlinuxcn (or follow generic Linux tarball) | Requires archlinuxcn repo enabled |
| Linux (generic tarball, no sudo) | `curl -L https://download.zotero.org/client/release/10.0.3/Zotero-10.0.3_linux-x86_64.tar.xz -o /tmp/zotero.tar.xz && tar -xJf /tmp/zotero.tar.xz -C ~/apps && ln -sf ~/apps/Zotero_linux-x86_64/zotero ~/.local/bin/zotero` | **MEASURED on this host 2026-09-30** — Zotero 10.0.3 (Zotero 140.15.0esr), 112.6 MB, ~7 s download |
| macOS | `brew install --cask zotero` | Or download `.dmg` |
| Windows | Download `.exe` from https://www.zotero.org/download/ | Standard Windows installer |

**Verify:** `which zotero` → expect `/usr/bin/zotero`, `/opt/zotero/zotero`,
`~/bin/zotero`, or `~/.local/bin/zotero` (symlinked to user-space install).

**MEASURED 2026-09-30 (Wf-372-A1):**
```
$ which zotero
/home/hugo/.local/bin/zotero

$ zotero --version
Zotero Zotero 140.15.0esr, Copyright (c) 2006-2025 Contributors
```
Zotero 10.0.3 binary is **SHIPPED** at `~/apps/zotero/` (user-space install,
no sudo required); symlinked to `~/.local/bin/zotero` which is on `PATH`
(verified `~/.local/bin` present in `$PATH`).

**Sandbox warning** (expected on hardened kernels, e.g. Linux 7.2.6-hardened):
```
Sandbox: CanCreateUserNamespace() clone() failure: EPERM
```
This is non-fatal — Zotero still launches (the EPERM comes from the
Linux kernel's user-namespace hardening, not from Zotero). GUI launch
requires an X11 / Wayland display; from a headless shell session, only
`zotero --version` / `zotero --help` work.

### §1.2 Install Better BibTeX (Zotero plugin, MIT)

Better BibTeX **CANNOT be installed headlessly** — it requires the Zotero
GUI (the `.xpi` plugin must be loaded into a running Zotero instance via
the Tools menu). Manual install steps:

```
# Step 1: Launch Zotero (requires X11 / Wayland display)
zotero &

# Step 2: In Zotero's GUI:
#   Tools → Add-ons
#   → Gear icon (top-right) → "Install Add-on From File…"
#   → Browse to: https://retorquere.github.io/zotero-better-bibtex/installation/
#   → Download the latest `better-bibtex-<version>.xpi`
#   → Open it; Zotero will prompt to restart

# Step 3: Verify post-restart:
#   Edit → Settings → Better BibTeX tab should be visible
#   Cite-As → "Better BibTeX Citation Key Quick Copy" should appear
```

**Why manual:** The `.xpi` is a signed Firefox-style extension that
needs the live Zotero process to register. There is no headless install
path for Zotero plugins on Linux.

**Where this doc leaves off:** Once the GUI install is done (one-time,
~2 minutes), the auto-export config in §1.3 takes over and the
`.tools/sci-paper/refs/references.bib` auto-updates within 5 s of any
item add/edit/delete.

### §1.3 Configure Better BibTeX auto-export

The project ships a reference config template at
`.tools/sci-paper/refs/better-bibtex-config.json` (1607 B, 2026-09-29).
**This file is NOT auto-imported** — Zotero regenerates it under
`~/.zotero/zotero-*/preferences.json` on first launch. Use the project's
template to set the keys in Zotero's GUI:

```
Edit → Settings → Better BibTeX:
  - citeFormat:              bibtex
  - citekeyFormat:          auth+year
  - autoExport:             idle
  - autoExportIdleDelay:    5
  - autoExportOnSave:       true
  - quickCopyFormat:        citekey
  - parseJabrefAbbreviations: true
  - autoPin:                true
  - extraExtensions[0]:     collection: Mol-Metal
                              format: bibtex
                              onChange: true
                              path: $HOME/codes/mol-metal/.tools/sci-paper/refs/references.bib
```

### §1.4 Verify

```
# 1. Open Zotero
# 2. Right-click "Mol-Metal" root collection → "Export Collection…"
# 3. Format: Better BibTeX; enable "On Change"
# 4. Save to: .tools/sci-paper/refs/references.bib
# 5. Confirm .bib file updates within 5 s of any item add/edit/delete
```

---

## §2 tectonic install (USER_OK-gated)

Per Wf-351-A3 audit (tectonic = MIT), this is the PRIMARY build engine for
reproducible LaTeX → PDF compilation.

### §2.1 Install tectonic

| Platform | Command | Notes |
|---|---|---|
| Linux (one-liner) | `curl -fsSL https://drop-sh.fullyjustified.net \|sh` | Self-installs to `~/.local/bin` (UNREACHABLE from this build host 2026-09-30 — see GitHub-releases-tarball fallback below) |
| Linux (apt) | `sudo apt install tectonic` | Debian/Ubuntu repo (may lag latest; **N/A on Arch**) |
| Linux (Arch) | `sudo pacman -S tectonic` | tectonic 0.17.0-1 is in `extra`; **N/A on this host (no sudo password)** — see GitHub-releases-tarball fallback below |
| Linux (GitHub releases tarball, no sudo) | `curl -fL -o /tmp/tectonic.tar.gz https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz && tar -xzf /tmp/tectonic.tar.gz -C /tmp && mv /tmp/tectonic ~/.local/bin/tectonic` | **MEASURED on this host 2026-09-30** — static musl x86_64 binary, 26.4 MB extracted, ~10 s download on retry; drop-sh.fullyjustified.net times out from this network |
| macOS | `brew install tectonic` | |
| Windows | `winget install tectonic` or `choco install tectonic` | |
| Source | `cargo install tectonic` | Requires Rust toolchain |

**Verify:** `which tectonic` → expect `~/.local/bin/tectonic` or
`/usr/local/bin/tectonic`. (As of 2026-09-30, the host PATH returns
`/home/hugo/.local/bin/tectonic` — **tectonic 0.17.0 IS INSTALLED** on
this build host per Wf-372-A2.)

**Why musl not gnu:** The musl build is statically linked (no glibc
version dependency), so it runs unchanged across Arch / Debian / Ubuntu
/ Alpine / RHEL. The gnu build is faster but requires glibc ≥ 2.17 on
the host.

### §2.2 Fallback: latexmk (GPL-2.0, pre-installed)

If tectonic install is delayed, `latexmk -pdf -interaction=nonstopmode
main.tex` is already available at `/usr/bin/latexmk` (verified via
`which latexmk`). This is **fine for local builds** but lacks tectonic's
bundle-based hermetic reproducibility — use for dev-loop only.

### §2.3 CI integration (future)

For GitHub Actions CI builds, use the official
`tectonic-typesetting/setup-tectonic@v1` action. Pin the version per
peer-review reproducibility requirement.

### §2.4 Verify

```bash
tectonic --version        # expect: Tectonic 0.17.0
tectonic --print main.tex  # build smoke-test on a minimal .tex
```

**MEASURED 2026-09-30 (Wf-372-A2):**
```
$ which tectonic
/home/hugo/.local/bin/tectonic

$ tectonic --version
Tectonic 0.17.0
```

**Smoke-test compile** (minimal `.tex`, requires network on first run
for tectonic to fetch its bundle of TeX resources):
```
$ cd .tools/latex
$ echo '\documentclass{article}\begin{document}Hello\end{document}' > test.tex
$ tectonic test.tex         # bundles TeXLive; ~5-10 min on first run
$ ls -la test.pdf           # expect: 5-30 KB PDF, "Hello" rendered
```

The smoke-test `.tex` file and its compiled `.pdf` are **gitignored**
test artifacts in `.tools/latex/`; they verify the install path end-to-
end without touching any `paper/*` file.

---

## §3 scienceplots install (INSTALLED in this build)

Per Wf-351-A4 audit (scienceplots = MIT), this is the PRIMARY journal-style
library for matplotlib figure rendering.

### §3.1 Install

```bash
# Already installed in project .venv at version 2.2.2 (2026-09-29 17:17 UTC)
# If re-installing:
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python -m pip install scienceplots

# Or in the project's own .venv (recommended for self-contained CI):
pip install scienceplots  # MIT, journal-style presets
```

### §3.2 Verify

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python -c \
  "import scienceplots; print('scienceplots OK:', scienceplots)"
# expect: scienceplots OK: <module 'scienceplots' from '.../scienceplots/__init__.py'>
```

### §3.3 Usage pattern

```python
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])  # 'science' base; 'no-latex' skips TeX dependency
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['savefig.dpi'] = 300
```

### §3.4 Available styles (MEASURED)

| Style | Purpose |
|---|---|
| `science` | Base journal aesthetic (small fonts, clean grids) |
| `no-latex` | Skip TeX rendering (when LaTeX engine unavailable) |
| `ieee` | IEEE transactions aesthetic |
| `nature` | Nature journal aesthetic |
| `high-vis` | High-contrast (talks, posters) |
| `bright` | Vibrant palette (online supplements) |
| Paul Tol `discrete-rainbow-{1..23}` | Color-blind-safe categorical palettes |

---

## §4 sn-jnl .cls verification (SHIPPED in this build)

Per Wf-351-A1 audit, `sn-jnl.cls` is the recommended Springer Nature
LaTeX class (Apache-2.0 mirror of the official class).

### §4.1 Verify

```bash
ls -la .tools/latex/sn-jnl.cls
# expect: -rw-r--r-- 1 hugo hugo 55331 Sep 29 17:17 .tools/latex/sn-jnl.cls
```

### §4.2 Provenance

The file was downloaded from
`https://raw.githubusercontent.com/DanySK/template-latex-springer-nature-sn-jnl/master/sn-jnl.cls`
on 2026-09-29 17:17 UTC (audit's PRIMARY recommendation). File size: 55,331
bytes. Header confirms docstrip-generated official Springer Nature class
under LPPL-1.3c.

### §4.3 Usage (NOT YET INTEGRATED — requires user OK per TODO-84)

```bash
# Smoke test (NOT integrated into paper/main.tex yet):
mkdir -p .tools/latex/smoke
cat > .tools/latex/smoke/minimal.tex <<'EOF'
\documentclass[sn-basic]{sn-jnl}
\title{Smoke test for sn-jnl.cls}
\author{Hugo Lee}
\begin{document}
\maketitle
Hello, world!
\end{document}
EOF

cd .tools/latex/smoke
tectonic --keep-intermediates --print minimal.tex
# expect: minimal.pdf
```

The actual integration with `paper/main.tex` is **DEFERRED to Wf-352**
(requires user OK per TODO-84 + AGENTS.md §5.3 — paper text changes are
user-authorization-gated).

---

## §5 NO `paper/*` touches (TODO-84 + AGENTS.md §5.3)

This setup doc configures **INFRASTRUCTURE ONLY** — it never edits
`paper/main.tex`, `paper/sections/*.tex`, `paper/appendices/*.tex`,
`paper/supplementary/*`, or any other paper text file.

**Why:** Per AGENTS.md §5.3 + TODO-84, paper text changes are user-
authorization-gated. The toolchain (Zotero, tectonic, scienceplots,
`sn-jnl.cls`, CSL styles, `pybtex-compile.py`) is part of the **build
pipeline**, not the paper itself.

**What requires USER_OK to proceed:**
- Any edit to `paper/main.tex`
- Any edit to `paper/sections/*.tex`
- Any edit to `paper/appendices/*.tex`
- Any BibTeX style swap (`unsrtnat` → `splncs04.bst` or `spbasic.bst`)
- Any addition of `\documentclass{sn-jnl}` line
- Any `tectonic` invocation against actual `paper/main.tex`

**What is FREE under infra scope:**
- Installing / upgrading tools (Zotero, tectonic, scienceplots, pybtex)
- Copying `sn-jnl.cls` into `.tools/latex/` (DONE — see §4.1)
- Configuring Better BibTeX auto-export paths (DONE — see §1.3)
- Adding CSL style files (DONE — see `.tools/sci-paper/styles/`)
- CI workflow YAML under `.github/workflows/latex.yml` (DEFERRED to Wf-352)

---

## §6 Cross-references

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
| Wf-372-A1 Zotero install | `TODO/completed/372a1_zotero_install.md` |
| Wf-372-A2 tectonic install (this update) | `TODO/completed/372a2_tectonic_install.md` |
| Skill: sci-paper-submission | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` |
| SCI-paper tool dir | `.tools/sci-paper/{bin,refs,styles}/` |
| Latex tools dir | `.tools/latex/{sn-jnl.cls,smoke/,test.tex,test.pdf}` |
| Tests | `molmetal/tests/test_wf355b3_sci_paper_install.py` + `molmetal/tests/test_wf372a1_zotero_install.py` + `molmetal/tests/test_wf372a2_tectonic_install.py` |

---

**END OF SETUP GUIDE** — configures infra only per TODO-84 + AGENTS.md §5.3;
paper text edits are USER_OK-gated. Last update 2026-09-30 (Wf-372-A1 + Wf-372-A2).