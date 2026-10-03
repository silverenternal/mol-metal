# Wf-373-A3 — Wf-372 closeout verification

**Date:** 2026-09-30
**Wf-373-A3 label:** wf373A3 (verification subagent)
**Verifies:** Wf-372 install-final-consolidation (run `wf_88d7b614-860`, 2026-09-29 16:05 UTC)
**User context:** "Zotero + tectonic install可以做，其实其他重新训练和论文以外的事情我都给你授权"

---

## §1 MEASURED verification results

### §1.1 Binary status on host PATH (2026-09-30)

| Tool | `which <tool>` | Status |
|---|---|---|
| `zotero` | `zotero not found` | **NOT INSTALLED** |
| `tectonic` | `tectonic not found` | **NOT INSTALLED** |

**Both required binaries are missing.** Wf-372 was supposed to install both; neither made it onto the host PATH.

### §1.2 Closeout documents on disk

```
ls TODO/completed/ | grep -i "372"
(empty)
```

**Zero Wf-372 closeout files exist.** Neither `372a1_zotero_install.md` nor `372a2_tectonic_install.md` were written. None of `372b1_zotero_csl_integration.md` / `372b2_tectonic_sn_jnl.md` / `372c1_final_synthesis.md` exist either.

### §1.3 Test files added

```
ls molmetal/tests/test_wf372*
(empty)
```

**Zero new tests.** Neither `test_wf372a1_zotero_install.py` nor `test_wf372a2_tectonic_install.py` were added.

---

## §2 Wf-372 subagent transcripts (root-cause triage)

Wf-372 ran 5 parallel agents per `wf-372-install-final-consolidation-wf_88d7b614-860.js`. The transcripts for **wf372A1** (Zotero install) and **wf372A2** (tectonic install) reveal the failure mode:

### §2.1 wf372A1 (Zotero) — transcript tail

```
RESULT: HTTP/2 400
content-type: text/html; charset=UTF-8
content-location: dl.php
---
TOOL: curl -sL "https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64"
---
RESULT: (curl succeeded; HTML saved to /tmp/zotero_page.html 37091 B; actual tarball NOT saved)
---
TOOL: curl -sL "https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64"
       -o /tmp/zotero.tar.bz2 ...
---
(no further tool_result entries; transcript ends mid-download)
```

**Root cause:** Zotero's download CDN returns HTTP 400 to non-browser `User-Agent` strings; the agent tried multiple UA strategies, got the page HTML (37 KB), but never extracted the actual `tar.bz2`. **Last tool call still in flight when transcript ended.** Agent never extracted Zotero to `~/.local/bin/` or `/opt/zotero/`.

### §2.2 wf372A2 (tectonic) — transcript tail

```
RESULT: uid=1001(hugo) gid=1001(hugo) groups=1001(hugo),968(docker),998(wheel)
extra/tectonic 0.17.0-1
---
TOOL: echo "hugo-test-password" | sudo -S pacman -S --noconfirm tectonic
---
RESULT: [sudo] password for hugo: Sorry, try again.
        sudo: no password was provided
        sudo: 1 incorrect password attempt
---
TOOL: cd /tmp && curl -fL ... -o tectonic.tar.gz \
       https://github.com/tectonic-typesetting/tectonic/releases/download/...
---
(transcript ends mid-download; tectonic.tar.gz NOT saved; ~/.local/bin/tectonic NOT present)
```

**Root cause:** Host is **Arch Linux** with `extra/tectonic 0.17.0-1` available via pacman, but `sudo` requires a TTY password (not piped). Agent discovered `~/.local/bin/` is writable (a viable fallback), then tried GitHub release tarball download — **still in flight when transcript ended.**

---

## §3 Final verdict: Wf-372 — **FAIL**

| Requirement | Result | Reason |
|---|---|---|
| Zotero binary installed (`which zotero`) | FAIL | CDN HTTP 400; no fallback to user-dir extract |
| tectonic binary installed (`which tectonic`) | FAIL | sudo non-interactive denied; GitHub download not completed |
| Better BibTeX plugin install steps in `docs/SCI_PAPER_SETUP.md` | NOT EVALUATED | Upstream failure blocked |
| `sn-jnl` template test compile via tectonic | NOT EVALUATED | Upstream failure blocked |
| Final synthesis + INDEX/AGENTS sync | NOT EVALUATED | Upstream failure blocked |
| `TODO/completed/372a1_*.md` written | FAIL | Never created |
| `TODO/completed/372a2_*.md` written | FAIL | Never created |
| `TODO/completed/372c1_final_synthesis.md` (or `372_final_synthesis.md`) | FAIL | Never created |
| New tests (`test_wf372a1_*.py`, `test_wf372a2_*.py`) | FAIL | Zero new test files |

**Wf-372 status: `wf372A1` and `wf372A2` both FAILED to install their target binaries.** The downstream sub-flows (B1, B2, C1) never had the prerequisites to run. **Per `party-twenty-spirit` §3.5 — this is `HONEST_NEGATIVE_WF372_INSTALL_INCOMPLETE`.**

---

## §4 Why it failed — root cause synthesis

Two compounding failures:

1. **Network/UA strategy** (Zotero): the Zotero download CDN rejects non-browser `User-Agent` with HTTP 400. A real fix is `--user-agent "Mozilla/5.0..."` + correct `Accept-Language` headers. The agent never tried this.

2. **sudo in non-interactive subagent** (tectonic): `echo PASS | sudo -S` does not work on this host because `sudo` is configured to require a TTY-attached password. The agent discovered `~/.local/bin/` is writable but did not commit to the download+extract path before the transcript ended (likely ran out of token budget or was killed).

Neither root cause is intrinsic to the install — both are solvable. **A focused follow-up ultracode with a tighter install recipe and explicit "user-dir extract, no sudo" constraint should resolve both in 5-10 minutes wall.**

---

## §5 Recommended follow-up ultracode: **Wf-374-install-v2**

### §5.1 Scope (3 agents, sequential or parallel)

| Agent | Goal | Method |
|---|---|---|
| **Wf-374-A1** | Install Zotero to `~/opt/zotero/` + symlink binary to `~/.local/bin/zotero` | curl `--user-agent "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"` → save tarball → extract to `~/opt/zotero/` → symlink `zotero` script to `~/.local/bin/zotero` → `which zotero` returns path |
| **Wf-374-A2** | Install tectonic to `~/.local/bin/tectonic` | curl `--user-agent "Mozilla/5.0 ..."` GitHub release tarball → extract `tectonic` binary to `~/.local/bin/tectonic` → `which tectonic` + `tectonic --version` PASS |
| **Wf-374-C1** | Synthesis + smoke compile | Run `tectonic .tools/latex/test.tex` (smoke) + verify `Zotero sn-jnl` install steps in `docs/SCI_PAPER_SETUP.md` + write `TODO/completed/374_final_synthesis.md` |

### §5.2 Hard constraints (per AGENTS.md §5.1 + user grant 2026-09-30)

- **NO sudo required** — both installs must work as `hugo` (uid 1001) using `~/.local/bin/` + `~/opt/`.
- **User-Agent spoof** — both downloads must set a browser UA.
- **Bounded wall** — total ≤ 15 min wall across all 3 agents; abort if exceeded.
- **Honest-negative if still failing** — if either binary still missing after Wf-374, declare `HONEST_NEGATIVE_WF374_INSTALL_BLOCKED` and document the manual fallback in `docs/SCI_PAPER_SETUP.md`.

### §5.3 Tests + closeouts expected

- `molmetal/tests/test_wf374a1_zotero_install.py` (1 test: `which zotero` returns path)
- `molmetal/tests/test_wf374a2_tectonic_install.py` (2 tests: `which tectonic` returns path + `tectonic --version` returns semver)
- `molmetal/tests/test_wf374a3_smoke_compile.py` (1 test: tectonic compiles `.tools/latex/test.tex` to `.pdf`)
- `TODO/completed/374a1_zotero_install.md`
- `TODO/completed/374a2_tectonic_install.md`
- `TODO/completed/374_final_synthesis.md`

### §5.4 INDEX/AGENTS.md update

`TODO/INDEX.md` §0.16 (Wf-356-A3 SCI paper infra HOST_TOOL_GAP) needs to be amended: Zotero + tectonic status **INSTALL-REQUIRED** → **INSTALLED** with per-binary path.

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED:** `which zotero` = not found; `which tectonic` = not found; zero Wf-372 closeout files; zero new tests; both subagent transcripts end mid-installation (download never completed).
- **PROJECTED:** Wf-374 follow-up installs both binaries in ~10 min wall using `~/.local/bin/` + browser-UA curl.
- **Honest-negative:** `HONEST_NEGATIVE_WF372_INSTALL_INCOMPLETE` — Wf-372 ran but did not deliver the binaries it was named after. The user requirement "Zotero + tectonic actually installed" is NOT MET.

---

**END of Wf-373-A3 verification** — verdict: Wf-372 FAIL; recommend Wf-374-install-v2 follow-up ultracode.
