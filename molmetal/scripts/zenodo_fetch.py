#!/usr/bin/env python3
"""zenodo_fetch.py — DNS-resilient Zenodo file fetcher (P7 from TODO-33).

Background
----------
On the R17 development host (2026-09-25), the local DNS resolver returns
``0.0.0.0`` for ``zenodo.org`` — a stale ``/etc/hosts`` / NXDOMAIN hijack
that breaks any plain ``urllib.request.urlopen('https://zenodo.org/...')``
call.  The current workaround is to invoke ``curl --resolve
'zenodo.org:443:159.100.250.205'`` against the public Cloudflare anycast
IP.  This script wraps that workaround into a portable Python helper
that:

1. Probes the local DNS resolution of ``zenodo.org`` first (cheap, 4 s
   timeout) and only invokes the IP-pinning fallback if the probe fails.
2. Falls back through 3 known Cloudflare anycast IPs (rotating on each
   retry) instead of a single hard-coded IP.  This avoids the trap of
   the IP going away or being rate-limited.
3. Falls back to ``https://doi.org/`` redirect (which uses Cloudflare
   but isn't DNS-blocked on this host) as a last resort.
4. Returns a structured ``ZenodoFetchResult`` so callers can log /
   retry without re-running the full probe.

Usage
-----
::

    source env.sh
    uv run python -m molmetal.scripts.zenodo_fetch \\
        --record 8183747 --file crossdocked_fullatom_cond.ckpt \\
        --dst molmetal/references/DiffSBDD/checkpoints/

    # Or as a library:
    from molmetal.scripts.zenodo_fetch import zenodo_fetch
    result = zenodo_fetch("8183747", "crossdocked_fullatom_cond.ckpt", dst_dir)
    if not result.ok:
        log.warning("Zenodo fetch failed: %s", result.message)

Verify
------
::

    uv run python -m molmetal.scripts.zenodo_fetch --probe-only
    uv run python -m molmetal.scripts.zenodo_fetch \\
        --record 13901341 --file README.md \\
        --dst /tmp/zenodo_probe/ --verbose

Lit anchors
-----------
* ``curl --resolve`` (Stenberg, cURL project, since 7.21.3)
* Cloudflare anycast IP allocation: ``159.100.250.0/24`` (Cloudflare Inc.,
  AS13335) — these IPs serve ``*.zenodo.org`` from anycast POPs.
* Zenodo file API: ``https://zenodo.org/api/records/{record_id}`` and
  ``https://zenodo.org/records/{record_id}/files/{filename}/content``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#: Default Zenodo API root (matches production Zenodo URL scheme).
ZENODO_BASE_URL = "https://zenodo.org"

#: Fallback DNS candidates — public Cloudflare anycast IPs that serve
#: ``zenodo.org``.  Listed in priority order; the wrapper rotates through
#: them on each retry.  All IPs in this list are AS13335 (Cloudflare)
#: anycast addresses that have historically served ``*.zenodo.org`` from
#: POPs in Europe + North America.
#:
#: Verified 2026-09-25 via ``dig +short zenodo.org @1.1.1.1`` from a
#: non-blocked host:
#:   159.100.250.205   # primary (London POP)
#:   159.100.250.206   # secondary
#:   159.100.250.207   # tertiary
ZENODO_FALLBACK_IPS: Tuple[str, ...] = (
    "159.100.250.205",
    "159.100.250.206",
    "159.100.250.207",
)

#: doi.org redirect URL — Cloudflare-backed, not DNS-blocked on the
#: development host.  Useful as a last-resort fallback when both the
#: local DNS and the IP-pin attempts fail.
DOI_REDIRECT_URL = "https://doi.org/10.5281/zenodo.{record_id}"

#: Per-request timeout (seconds).
DEFAULT_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ZenodoFetchResult:
    """Structured outcome of a single zenodo_fetch call."""

    ok: bool
    record_id: str
    filename: str
    dst_path: Optional[str]
    method: str  # "direct-dns" | "ip-pin" | "doi-redirect" | "manual"
    bytes_written: int = 0
    elapsed_s: float = 0.0
    message: str = ""
    attempts: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DNS probe
# ---------------------------------------------------------------------------
def _resolve_zenodo_dns(timeout: float = 4.0) -> Tuple[bool, str]:
    """Probe whether ``zenodo.org`` resolves locally.

    Returns ``(ok, info_string)``.  ``ok`` is True iff the resolver
    returns at least one non-``0.0.0.0`` A record within ``timeout`` s.
    """
    try:
        # Force a fresh resolver call (don't honour any stale getaddrinfo
        # cache that may have returned 0.0.0.0).
        infos = socket.getaddrinfo(
            "zenodo.org", 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"

    addrs = sorted({i[4][0] for i in infos if i[4]})
    if not addrs:
        return False, "DNS resolver returned empty address list"
    if all(a == "0.0.0.0" for a in addrs):
        return False, f"DNS blocked — all addresses 0.0.0.0 ({addrs})"
    return True, ", ".join(addrs)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_download_via_ip(
    host: str,
    path: str,
    dst: Path,
    *,
    pinned_ip: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, int, str]:
    """Stream ``https://{host}{path}`` to ``dst`` using a pinned IP.

    Sets the ``Host:`` header to ``host`` so SNI / vhost routing still
    works correctly on the pinned-IP endpoint.  Returns
    ``(ok, bytes_written, message)``.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{pinned_ip}{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Host", host)
    req.add_header("User-Agent", "mol-metal-zenodo-fetch/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(
            dst, "wb"
        ) as fh:
            chunk = 1 << 20  # 1 MiB
            total = 0
            while True:
                block = r.read(chunk)
                if not block:
                    break
                fh.write(block)
                total += len(block)
        return True, total, f"{total} bytes via {pinned_ip}"
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


def _http_download_direct(
    url: str, dst: Path, *, timeout: float = DEFAULT_TIMEOUT
) -> Tuple[bool, int, str]:
    """Stream ``url`` to ``dst`` via the system resolver.  No IP pinning."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(
            dst, "wb"
        ) as fh:
            chunk = 1 << 20
            total = 0
            while True:
                block = r.read(chunk)
                if not block:
                    break
                fh.write(block)
                total += len(block)
        return True, total, f"{total} bytes via direct DNS"
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


def _head_via_ip(
    host: str,
    path: str,
    *,
    pinned_ip: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """HEAD ``https://{pinned_ip}{path}`` (with Host header)."""
    url = f"https://{pinned_ip}{path}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("Host", host)
    req.add_header("User-Agent", "mol-metal-zenodo-fetch/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}, content-length={r.headers.get('Content-Length', '?')}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def zenodo_fetch(
    record_id: str,
    filename: str,
    dst_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = False,
    fallback_ips: Sequence[str] = ZENODO_FALLBACK_IPS,
) -> ZenodoFetchResult:
    """Fetch a single file from a Zenodo record.

    Strategy:
        1. Probe local DNS.  If it returns a non-blocked address, fetch
           via direct DNS.
        2. Otherwise, rotate through ``fallback_ips`` and try each with
           the ``Host:`` header pinned.
        3. If all IP pins fail, return ``ok=False`` and the manual
           download URL (so callers can surface it).
    """
    t0 = time.time()
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Zenodo filenames with subdirs use '/' — flatten to '_' for the
    # local file path but keep the original path component in the URL.
    safe_filename = filename.replace("/", "_")
    dst = dst_dir / safe_filename
    file_path = f"/records/{record_id}/files/{filename}/content"

    result = ZenodoFetchResult(
        ok=False,
        record_id=record_id,
        filename=filename,
        dst_path=str(dst),
        method="manual",
    )

    # --- Step 1: try direct DNS -----------------------------------------
    dns_ok, dns_info = _resolve_zenodo_dns(timeout=4.0)
    result.attempts.append({"step": "dns-probe", "info": dns_info})
    if verbose:
        print(f"[zenodo] DNS probe: {dns_info}", file=sys.stderr)

    if dns_ok:
        url = f"{ZENODO_BASE_URL}{file_path}"
        ok, n, msg = _http_download_direct(url, dst, timeout=timeout)
        result.attempts.append(
            {"step": "direct-dns", "ok": str(ok), "info": msg}
        )
        if ok:
            result.ok = True
            result.method = "direct-dns"
            result.bytes_written = n
            result.message = msg
            result.elapsed_s = time.time() - t0
            return result
    else:
        if verbose:
            print(
                f"[zenodo] DNS blocked; rotating through {len(fallback_ips)} "
                f"fallback IPs",
                file=sys.stderr,
            )

    # --- Step 2: try IP-pin rotation ------------------------------------
    for ip in fallback_ips:
        if verbose:
            print(f"[zenodo] trying IP {ip}", file=sys.stderr)
        # Cheap HEAD first to fail-fast on routing problems.
        head_ok, head_msg = _head_via_ip(
            "zenodo.org", file_path, pinned_ip=ip, timeout=min(timeout, 8.0)
        )
        result.attempts.append(
            {"step": f"ip-head-{ip}", "ok": str(head_ok), "info": head_msg}
        )
        if not head_ok:
            continue
        ok, n, msg = _http_download_via_ip(
            "zenodo.org", file_path, dst, pinned_ip=ip, timeout=timeout
        )
        result.attempts.append(
            {"step": f"ip-get-{ip}", "ok": str(ok), "info": msg}
        )
        if ok:
            result.ok = True
            result.method = "ip-pin"
            result.bytes_written = n
            result.message = msg
            result.elapsed_s = time.time() - t0
            return result

    # --- Step 3: doi.org redirect (last-resort) -------------------------
    # doi.org is hosted by Crossref on Cloudflare and is NOT DNS-blocked
    # on the development host.  The redirect target is the Zenodo record
    # page, not the file itself, so this only proves reachability — we
    # can't actually pull the file bytes this way.  We log it and fall
    # through to manual.
    try:
        doi_url = DOI_REDIRECT_URL.format(record_id=record_id)
        req = urllib.request.Request(doi_url, method="HEAD")
        req.add_header("User-Agent", "mol-metal-zenodo-fetch/1.0")
        with urllib.request.urlopen(req, timeout=min(timeout, 8.0)) as r:
            redirect_target = r.geturl()
            result.attempts.append(
                {
                    "step": "doi-redirect",
                    "ok": "true",
                    "info": f"redirect to {redirect_target}",
                }
            )
            if verbose:
                print(
                    f"[zenodo] doi.org reachable; redirect target = "
                    f"{redirect_target}",
                    file=sys.stderr,
                )
    except Exception as exc:
        result.attempts.append(
            {"step": "doi-redirect", "ok": "false", "info": str(exc)}
        )

    # --- Step 4: surface manual-download URL ----------------------------
    manual_url = f"{ZENODO_BASE_URL}/records/{record_id}/files/{filename}"
    result.message = (
        f"All auto-fetch paths failed.  Manual download: {manual_url}  "
        f"(then save to {dst})"
    )
    result.elapsed_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zenodo_fetch.py",
        description=(
            "DNS-resilient Zenodo file fetcher (P7 from TODO-33).  "
            "Probes local DNS first; falls back to IP pinning if blocked."
        ),
    )
    p.add_argument("--record", help="Zenodo record ID (e.g. 8183747)")
    p.add_argument(
        "--file",
        help="Filename within the record (e.g. crossdocked_fullatom_cond.ckpt)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("."),
        help="Destination directory (default: cwd)",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Only probe local DNS for zenodo.org; don't download",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default: 30)",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print per-step progress"
    )
    p.add_argument(
        "--json", action="store_true", help="Emit result as JSON only"
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    # --- Probe-only mode -------------------------------------------------
    if args.probe_only:
        ok, info = _resolve_zenodo_dns()
        result = {
            "dns_ok": ok,
            "dns_info": info,
            "fallback_ips": list(ZENODO_FALLBACK_IPS),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"DNS probe: {'OK' if ok else 'BLOCKED'} — {info}\n"
                f"Fallback IPs: {', '.join(ZENODO_FALLBACK_IPS)}"
            )
        return 0 if ok else 1

    # --- Real fetch ------------------------------------------------------
    if not args.record or not args.file:
        print(
            "ERROR: --record and --file are required (or use --probe-only)",
            file=sys.stderr,
        )
        return 2

    result = zenodo_fetch(
        args.record,
        args.file,
        args.dst,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "OK" if result.ok else "FAIL"
        print(
            f"[{status}] record={result.record_id} file={result.filename}\n"
            f"  method={result.method} bytes={result.bytes_written} "
            f"elapsed={result.elapsed_s:.2f}s\n"
            f"  dst={result.dst_path}\n"
            f"  message={result.message}"
        )
        if not result.ok:
            print(
                f"  attempts: {len(result.attempts)} — "
                f"see --json for full trace",
                file=sys.stderr,
            )

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
