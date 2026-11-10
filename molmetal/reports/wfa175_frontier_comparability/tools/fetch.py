#!/usr/bin/env python3
"""Wf-A175 FRONTIER branch — live primary-record fetch helper (CPU-only).

Rules honoured:
  * every python invocation is CUDA_VISIBLE_DEVICES="" + .venv/bin/python -I
  * nothing is written outside molmetal/reports/wfa175_frontier_comparability/
  * e-print bytes are read in memory (tarfile); only sha256 + the extracted
    .tex/.txt text is persisted.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request

PROXY = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
_opener = urllib.request.build_opener(urllib.request.ProxyHandler(PROXY))
UA = "mol-metal-wfa175/1.0 (frontier comparability census)"


def _get(url: str, timeout: int = 60, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            r = _opener.open(req, timeout=timeout)
            return r.status, r.geturl(), r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def api_search(query: str, max_results: int = 10):
    """arXiv API query.  quote the value strings with %22 already."""
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + query
        + "&max_results=%d&sortBy=relevance" % max_results
    )
    _, _, b = _get(url)
    s = b.decode("utf-8", "replace")
    ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", s)
    titles = [re.sub(r"\s+", " ", t).strip() for t in re.findall(r"<title>(.*?)</title>", s, re.S)][1:]
    pubs = [re.sub(r"\s+", " ", t).strip() for t in re.findall(r"<published>(.*?)</published>", s, re.S)][1:]
    out = []
    for i, (aid, t) in enumerate(zip(ids, titles)):
        out.append({"arxiv": aid, "title": t, "published": pubs[i] if i < len(pubs) else ""})
    return out


def fetch_abs(arxiv_id: str):
    url = "https://arxiv.org/abs/" + arxiv_id
    st, final, b = _get(url)
    return {"url": url, "final_url": final, "status": st, "bytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest(), "html": b.decode("utf-8", "replace")}


def fetch_eprint(arxiv_id: str):
    """Return {url, final_url, bytes, sha256, files: {name: text}}."""
    url = "https://arxiv.org/e-print/" + arxiv_id
    st, final, b = _get(url)
    rec = {"url": url, "final_url": final, "status": st, "bytes": len(b),
           "sha256": hashlib.sha256(b).hexdigest(), "files": {}}
    if b[:2] == b"\x1f\x8b" or b[:5] == b"ustar"[:5]:
        pass
    try:
        tf = tarfile.open(fileobj=io.BytesIO(b), mode="r:*")
        for m in tf.getmembers():
            if not m.isfile() or m.size > 20_000_000:
                continue
            name = m.name
            if not re.search(r"\.(tex|bib|bbl|txt|json|md|yaml|yml|csv)$", name, re.I):
                continue
            try:
                data = tf.extractfile(m).read()
            except Exception:  # noqa: BLE001
                continue
            rec["files"][name] = data.decode("utf-8", "replace")
        tf.close()
    except Exception as e:  # noqa: BLE001
        rec["extract_error"] = "%s: %s" % (type(e).__name__, e)
        rec["head"] = b[:400].decode("utf-8", "replace")
    return rec


def grep(files: dict, pattern: str, ctx: int = 0, limit: int = 200):
    """Regex grep over the extracted text map; returns [(name, line_no, line)]."""
    rx = re.compile(pattern, re.I)
    out = []
    for name, text in sorted(files.items()):
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                out.append((name, i, line.strip()))
                if len(out) >= limit:
                    return out
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        for r in api_search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10):
            print(r["arxiv"], "|", r["published"][:10], "|", r["title"][:100])
    elif cmd == "eprint":
        rec = fetch_eprint(sys.argv[2])
        print(json.dumps({k: v for k, v in rec.items() if k != "files"}, indent=1))
        print("n_files", len(rec["files"]))
        for n in sorted(rec["files"])[:60]:
            print("  ", n, len(rec["files"][n]))
