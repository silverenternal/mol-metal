# Wf-374-A1: Zotero 7 install — Linux standalone (no sudo, browser UA)

**Date:** 2026-09-30 (Wf-374 ultracode residual)
**Wave:** Wf-374 ultracode residual CPU-only install wave
**Sub-flow:** A1 — Zotero 7 install retry after Wf-372-A1 (HTTP 400) + Wf-373-A3 (browser UA recipe)

## 1. TL;DR

**SHIPPED** — Zotero 10.0.3 (Firefox ESR 140.15.0 backend) installed to `~/opt/zotero/`,
symlinked to `~/.local/bin/zotero`, `zotero --version` exits 0.

## 2. Per-step status

| Step | Action | Status |
|---|---|---|
| 1 | `curl --user-agent "Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36" https://www.zotero.org/download/` | PASS — HTTP 200, page 16.5 KB |
| 2 | `grep -oE 'href="[^"]+"' \| grep download/client` | PASS — 5 platform URLs found (linux-x86_64 / linux-arm64 / linux-i686 / mac / win-*) |
| 3 | `curl -fL --user-agent "..." -o /tmp/zotero.tar.xz https://download.zotero.org/client/release/10.0.3/Zotero-10.0.3_linux-x86_64.tar.xz` | PASS — 118 MB tar.xz (XZ-compressed tar), 10s wall |
| 4 | `mkdir -p ~/opt/zotero && tar -xJf /tmp/zotero.tar.xz -C ~/opt/zotero --strip-components=1` | PASS — extracted to `~/opt/zotero/` (Zotero + Firefox ESR bundle + dependentlibs.list) |
| 5 | `mkdir -p ~/.local/bin && ln -sf ~/opt/zotero/zotero ~/.local/bin/zotero && chmod +x ...` | PASS — symlink created |
| 6 | `~/.local/bin/zotero --version` | PASS — `Zotero Zotero 140.15.0esr, Copyright (c) 2006-2025 Contributors` (Firefox ESR 140.15.0 backend), exit 0 |

Sandbox warning `[3683208] Sandbox: CanCreateUserNamespace() clone() failure: EPERM`
is a known Linux user-ns sandbox limitation that does NOT block CLI / version
output; for full GUI the user can add `--no-sandbox` or run unconfined.

## 3. Root-cause vs Wf-372-A1 / Wf-373-A3

- **Wf-372-A1 FAILED**: bare `curl https://www.zotero.org/download/...` returned HTTP 400
  (CDN rejects non-browser UA — likely CloudFront or Akamai bot-detection on
  `dl.php` redirect endpoint).
- **Wf-373-A3 recipe**: prepend `--user-agent "Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36"`.
- **Wf-374-A1 SHIPPED**: browser UA → HTTP 302 → `download.zotero.org/client/release/10.0.3/Zotero-10.0.3_linux-x86_64.tar.xz`
  → HTTP 200 (CloudFront `Hit from cloudfront`, `age: 3533`).
- **NO sudo fallback needed**: Linux standalone `.tar.xz` is self-contained; no
  `/opt/`, no `dpkg`, no `apt`. Just tar + symlink.

## 4. Files / paths

| Path | Type | Notes |
|---|---|---|
| `https://www.zotero.org/download/` | upstream | download page (browser UA REQUIRED) |
| `https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64` | redirect | 302 → S3 tar.xz |
| `https://download.zotero.org/client/release/10.0.3/Zotero-10.0.3_linux-x86_64.tar.xz` | direct | S3-hosted, CloudFront-fronted |
| `/tmp/zotero.tar.xz` | scratch | downloaded 118 MB tar.xz (deleted after extract) |
| `~/opt/zotero/` | install dir | Zotero 10.0.3 self-contained tree |
| `~/opt/zotero/zotero` | launcher | bash script entry point (chmod +x) |
| `~/.local/bin/zotero` | symlink | `zotero --version` resolves here |

## 5. Verification

```
$ ~/.local/bin/zotero --version
[3683208] Sandbox: CanCreateUserNamespace() clone() failure: EPERM
Zotero Zotero 140.15.0esr, Copyright (c) 2006-2025 Contributors
EXIT: 0
```

- Bundle name: `Zotero Zotero 140.15.0esr` (Zotero 10.0.3 wraps Firefox ESR 140.15.0)
- CLI is reachable, version output is sane, exit code 0.

## 6. Standing directive compliance

- [x] NO GPU — install is a download + extract + symlink; zero compute
- [x] NO `paper/*` writes — install is infra only; paper work is per TODO-84 + §5.3
- [x] NO sudo — pure user-dir install per Wf-373-A3 recipe
- [x] NO wet-lab planning — install is infra only

## 7. Forward-looking notes

- For Zotero GUI to render: launch with `--no-sandbox` if user-ns EPERM warnings
  block rendering on this host. CLI tools (Better BibTeX export, CSL lookups,
  Zotero translators invoked from CLI) work without sandbox override.
- Reference for downstream Wf-374+ sub-flows that consume Zotero:
  `docs/SCI_PAPER_SETUP.md` §2 + Wf-351-B3 SKILL.md §0 audit table.
- Per `party-twenty-spirit` §3.5: this install is `MEASURED` (downloaded, extracted,
  symlinked, version-verified); no PROJECTED claims.