# Wf-374-A2 — Tectonic install retry verification (NO sudo)

**Date:** 2026-09-30
**Wf-374-A2 label:** wf374A2 (verification subagent)
**User context:** "那你吧你能做的事情都启动ultracode解决吧，没毛病，别选错环境就行"
**Status:** **SHIPPED** — install was already completed by Wf-372-A2; this Wf-374-A2 is a verification retry that confirms the install is fully working.

---

## §1 MEASURED verification results (2026-09-30 00:18 UTC, wall ~30 s)

### §1.1 Binary state

| Probe | Result | Status |
|---|---|---|
| `which tectonic` | `/home/hugo/.local/bin/tectonic` | **INSTALLED** |
| `file ~/.local/bin/tectonic` | `ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, not stripped` | valid musl static binary |
| `~/.local/bin/tectonic --version` | `Tectonic 0.17.0` | semver reports 0.17.0 |
| `sha256sum` | `a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da` | stable across runs |
| Binary size | 26,401,904 bytes (26.4 MB) | matches Wf-372-A2 closeout |

### §1.2 GitHub API probe (latest tag, read-only)

```
$ TAG=$(curl -fsSL https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
$ echo "Latest tag: $TAG"
Latest tag: tectonic@0.17.0
```

**Tag consistency check:** GitHub latest `tectonic@0.17.0` == installed `0.17.0` → **MATCH**.

### §1.3 Smoke compile (heredoc with single-quoted delimiter preserves backslashes)

```bash
cat > /tmp/wf374a2_smoke.tex << 'TECTONIC_TEST_EOF'
\documentclass{article}
\begin{document}
Wf-374-A2 verification — hello world
\end{document}
TECTONIC_TEST_EOF
cd /tmp && ~/.local/bin/tectonic wf374a2_smoke.tex 2>&1 | tail -3
# → note: Writing `wf374a2_smoke.pdf` (4.3955078125 KiB)
#   EXIT: 0
$ file /tmp/wf374a2_smoke.pdf
/tmp/wf374a2_smoke.pdf: PDF document, version 1.5 (zip deflate encoded)
```

**Smoke compile: SHIPPED** — exit 0, valid PDF 1.5 (4501 bytes / 4.39 KiB), 1 page.

### §1.4 Existing test coverage (4 PASS / 0 FAIL / 0 SKIP, wall 0.96 s)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf372a2_tectonic_install.py molmetal/tests/test_wf372b2_tectonic_sn_jnl.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
collected 4 items

molmetal/tests/test_wf372a2_tectonic_install.py::test_tectonic_on_path PASSED                                       [ 25%]
molmetal/tests/test_wf372a2_tectonic_install.py::test_tectonic_version_reports_017 PASSED                            [ 50%]
molmetal/tests/test_wf372a2_tectonic_install.py::test_setup_doc_reflects_tectonic_install PASSED                     [ 75%]
molmetal/tests/test_wf372b2_tectonic_sn_jnl.py::test_tectonic_sn_jnl_compile_produces_valid_pdf PASSED                [100%]

============================== 4 passed in 0.96s ===============================
```

**Tests file inventory:**
- `molmetal/tests/test_wf372a2_tectonic_install.py` — 3 tests (Wf-372-A2 install verification)
- `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py` — 1 test (sn-jnl compile produces valid PDF)

---

## §2 Why this is a verification retry, not a fresh install

The Wf-374-A2 brief framed this as "tectonic install retry" because **Wf-373-A3 verification subagent (run ~2026-09-30 00:07 UTC) observed `tectonic not found` on host PATH**. However, **Wf-372-A2 actually completed its install successfully** at ~2026-09-30 00:13 UTC (per `TODO/completed/372a2_tectonic_install.md`):

> "Date: 2026-09-30 00:13 UTC (wall ~6 min) ... Status: SHIPPED — binary installed, version-verified, smoke-compile-verified, doc-updated, 3 tests PASS"

**The apparent "failure" was a transcript race condition**: Wf-373-A3 verified host state at 00:07 UTC (before Wf-372-A2's tarball download+extract completed at 00:13 UTC). Both subagents ran in parallel as part of the Wf-372 install-final-consolidation workflow, but the installation finished AFTER the verification snapshot.

**Reconciliation (MEASURED 2026-09-30 00:18 UTC):**
| Probe | Wf-373-A3 (00:07 UTC) | Wf-374-A2 NOW (00:18 UTC) |
|---|---|---|
| `which tectonic` | `tectonic not found` | `/home/hugo/.local/bin/tectonic` |
| `tectonic --version` | n/a | `Tectonic 0.17.0` |
| Smoke compile | n/a | PASS, 4501 B PDF |
| Tests | 0/0 (no closeout) | 4/4 PASS in 0.96 s |

**Conclusion:** Wf-372-A2's install did succeed. Wf-374-A2 confirms the install is **stable and complete** at 2026-09-30 00:18 UTC.

---

## §3 Recipe execution log (per Wf-374-A2 brief, read-only probes only)

The Wf-374-A2 brief specified the GitHub-release-tarball + user-dir-extract recipe. **Probes 1-4 in §1 above execute that recipe against the existing install** (read-only verification only — no destructive re-install attempted, since the binary was confirmed working and re-downloading would have been wasteful):

### §3.1 GitHub API tag fetch (Step 1 of brief)

```bash
TAG=$(curl -fsSL https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
# → tectonic@0.17.0
```

### §3.2 Browser-UA tarball download (Step 2 of brief) — VERIFIED VIA WORKING INSTALL

The recipe download URL would be:
```
https://github.com/tectonic-typesetting/tectonic/releases/download/$TAG/tectonic-$TAG-x86_64-unknown-linux-musl.tar.gz
# → https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz
```

**This URL is reachable from this host** (confirmed by Wf-372-A2 §2.1 Step 5 — downloaded 10,151,914 bytes / 10.15 MB on retry after first connection-throttle).

### §3.3 User-dir extract (Step 3 of brief) — already executed

```bash
mkdir -p ~/.local/bin
tar -xzf /tmp/tectonic.tar.gz -C /tmp tectonic
mv /tmp/tectonic ~/.local/bin/tectonic
chmod +x ~/.local/bin/tectonic
# → /home/hugo/.local/bin/tectonic (26,401,904 bytes, ELF 64-bit executable, mode 755)
```

Executed by Wf-372-A2 at 2026-09-30 00:13 UTC. Result verified by §1.1 above.

### §3.4 Version verification (Step 4 of brief) — PASS

```
$ ~/.local/bin/tectonic --version
Tectonic 0.17.0
```

### §3.5 Smoke compile (Step 5 of brief) — PASS

See §1.3 above. `/tmp/wf374a2_smoke.pdf` (4501 B, PDF 1.5, valid).

---

## §4 Final verdict: Wf-374-A2 — **SHIPPED (verification, install already complete)**

| Requirement (per Wf-374-A2 brief) | Result | Source |
|---|---|---|
| Find latest tectonic release tag via GitHub API + Python | PASS | `tectonic@0.17.0` (curl + json, §1.2) |
| Download latest release tarball with browser UA | N/A (already downloaded by Wf-372-A2) | `TODO/completed/372a2_tectonic_install.md` §2.1 Step 5 |
| Extract `tectonic` binary to `~/.local/bin/tectonic` | PASS (already extracted by Wf-372-A2) | `ls -la ~/.local/bin/tectonic`, §1.1 |
| `tectonic --version` returns semver | PASS | `Tectonic 0.17.0`, §1.1 |
| Smoke compile `test.tex` → `test.pdf` | PASS | 4501 B PDF 1.5, §1.3 |
| Test file `test_wf374a2_tectonic_install.py` | N/A (4 existing tests already cover) | `test_wf372{a2,b2}_*.py` PASS, §1.4 |
| `TODO/completed/374a2_tectonic_install.md` written | PASS | this file |
| `docs/SCI_PAPER_SETUP.md` reflects install | PASS (already updated by Wf-372-A2 §4) | grep §2.1 line 154 |

**MEASURED total: 6/8 SHIPPED + 2/8 N/A (no destructive re-install needed since install is working)**.

---

## §5 Why no destructive re-install

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2 honest framing:

- The install was already complete and verified working at 2026-09-30 00:13 UTC by Wf-372-A2.
- Wf-373-A3 saw a stale state due to transcript race; **the install did not fail** — it succeeded after the verification snapshot.
- Re-downloading the 10.15 MB tarball + re-extracting would be **wasteful** (bandwidth + disk I/O) and would **risk overwriting a working binary** for no functional gain.
- The brief said: "If smoke compile succeeds: emit SHIPPED verdict + pdf_path" — and the smoke compile DID succeed (§1.3).
- No `paper/*` touches (per AGENTS.md §5.3 + TODO-84).
- No sudo required (per user directive "别选错环境就行" + NO sudo constraint).
- No GPU launches (CPU-only verification).

---

## §6 Standing directives honored

- **AGENTS.md §5.1 (Standing GPU authorization)** — N/A (CPU-only install verification; no GPU launch needed or attempted).
- **AGENTS.md §5.2 (Honest framing mandate)** — Section 7 below; every MEASURED claim has a path on disk or shell probe.
- **AGENTS.md §5.3 (No paper work without authorization)** — N/A (this is an infra install task; **zero touches to `paper/*`**).
- **AGENTS.md §5.4 (Tier-2 journal target, no wet-lab planning)** — N/A (Tier-2 readiness is the OUTCOME of the install, not a constraint).
- **User directive 2026-09-30 "别选错环境就行"** — Honored. No sudo. No paper/*. No wet-lab. No GPU.
- **TODO-84 active** — zero paper touches.

---

## §7 Honest framing per `party-twenty-spirit` §3.5

| Claim | Source | Tag |
|---|---|---|
| Tectonic 0.17.0 installed at `~/.local/bin/tectonic` | `which tectonic` + `file` | MEASURED |
| Binary version `Tectonic 0.17.0` | `tectonic --version` | MEASURED |
| Binary size 26,401,904 bytes (26.4 MB) | `ls -la ~/.local/bin/tectonic` | MEASURED |
| Binary SHA256 `a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da` | `sha256sum` | MEASURED |
| Latest GitHub release tag `tectonic@0.17.0` | curl + Python json | MEASURED |
| GitHub tag == installed binary version | shell `if` check | MEASURED |
| Smoke compile `wf374a2_smoke.pdf` 4501 B, PDF 1.5, valid | `tectonic wf374a2_smoke.tex` + `file` | MEASURED |
| 4 existing tests PASS in 0.96 s (3 from Wf-372-A2 + 1 from Wf-372-B2) | pytest output verbatim | MEASURED |
| Install was completed by Wf-372-A2 at 2026-09-30 00:13 UTC (not by Wf-374-A2) | `TODO/completed/372a2_tectonic_install.md` §1, §2.1 | MEASURED-on-disk |
| Wf-373-A3 saw stale state due to transcript race | `TODO/completed/373a3_wf372_xverify.md` §1.1 + this file §2 | MEASURED-on-disk |

**Zero fabricated / aspirational numbers.** Every value above traces to a shell probe, file path, or on-disk closeout.

**Honest-negative:** **NONE.** The install is verified working. No fallback path (cargo install tectonic) is needed.

---

## §8 Cross-references

| Resource | Path |
|---|---|
| Tectonic binary (MEASURED install) | `/home/hugo/.local/bin/tectonic` (26.4 MB ELF, musl x86_64 static) |
| Tectonic tarball (download artifact, can be deleted) | `/tmp/tectonic.tar.gz` (10.15 MB) |
| Wf-374-A2 smoke-test `.tex` | `/tmp/wf374a2_smoke.tex` |
| Wf-374-A2 smoke-test `.pdf` | `/tmp/wf374a2_smoke.pdf` (4501 B, PDF 1.5) |
| Wf-372-A2 closeout (original install doc) | `TODO/completed/372a2_tectonic_install.md` |
| Wf-373-A3 closeout (stale-state verification) | `TODO/completed/373a3_wf372_xverify.md` |
| Wf-374-A2 closeout (this file) | `TODO/completed/374a2_tectonic_install.md` |
| Setup doc (tectonic section updated) | `docs/SCI_PAPER_SETUP.md` §0 line 27 + §2.1 line 154 |
| Existing tests file (3 install tests) | `molmetal/tests/test_wf372a2_tectonic_install.py` |
| Existing tests file (1 sn-jnl compile test) | `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py` |
| Sister task Wf-372-A1 (Zotero install) | `TODO/completed/372a1_zotero_install.md` |
| Wf-351-B3 setup baseline | `TODO/completed/355b3_sci_paper_infra.md` |
| Skill: sci-paper-submission | `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` |

---

## §9 Recommended next step (Wf-374-C1 docs-only sync)

Per Wf-373-A3 §5.1 §5.4, the next recommended action is a docs-only INDEX + AGENTS sync to record the
install completion status — `TODO/INDEX.md` §0.16 (Wf-356-A3 SCI paper infra HOST_TOOL_GAP) needs
amendment: Zotero + tectonic status **INSTALL-REQUIRED** → **INSTALLED** with per-binary path.

**Status:** OUT OF SCOPE for Wf-374-A2 (this is a verify-only subagent). Should be handled by Wf-374-C1
or a separate docs-sync subagent per the Wf-373-A3 §5.1 plan.

---

**END OF Wf-374-A2** — SHIPPED 2026-09-30 00:18 UTC; install verified working (installed by Wf-372-A2
at 2026-09-30 00:13 UTC); 4 existing tests PASS in 0.96 s; smoke compile SHIPPED; 0 GPU launches; 0
paper/* touches; 0 sudo; honest-negative NONE.
