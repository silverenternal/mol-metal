#!/usr/bin/env python3
"""Wf-A175 BIBLIOGRAPHY branch — paper/refs.bib DOI integrity audit.

CPU-only. READ-ONLY on paper/** and .tools/sci-paper/**.
Writes only into molmetal/reports/wfa175_refs_bib_audit/.

Resolver strategy per DOI (primary records only, per hard rule 7):
  1. DOI resolver landing page (https://doi.org/<doi>) with redirect follow;
     title scraped from <meta name="citation_title"> or <title>.
  2. Crossref REST API (https://api.crossref.org/works/<doi>) — the DOI
     registration agency's publisher-deposited record.
  3. arXiv API for 10.48550/arXiv.* DOIs (primary arXiv record).
  4. bioRxiv API for 10.1101/* DOIs (primary preprint-server record).

Every record records http_status + resolver_title + bibtex_title.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

REPO = Path("/home/hugo/codes/mol-metal")
BIB = REPO / "paper" / "refs.bib"
OUT = REPO / "molmetal" / "reports" / "wfa175_refs_bib_audit"

UA = "molmetal-wfa175-audit/1.0 (bibliography integrity audit; contact: local)"
HEADERS = {"User-Agent": UA, "Accept": "*/*"}

TIMEOUT = 40


# ---------------------------------------------------------------- parsing ---
def parse_bib(text: str) -> list[dict]:
    """Split a .bib into entries. Returns dicts with key, type, fields, line."""
    entries: list[dict] = []
    i = 0
    n = len(text)
    lines = text.splitlines()
    # line number of each char offset
    line_of = []
    ln = 1
    for ch in text:
        line_of.append(ln)
        if ch == "\n":
            ln += 1
    while i < n:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").search(text, i)
        if not m:
            break
        etype, key = m.group(1), m.group(2)
        # brace matching from the entry's own opening brace
        ob = text.index("{", m.start())
        depth = 0
        j = ob
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[m.end() : j]
        entries.append(
            {
                "key": key,
                "type": etype,
                "body": body,
                "line": line_of[m.start()],
            }
        )
        i = j + 1
    return entries


FIELD_RE = re.compile(r"(\w[\w-]*)\s*=\s*", re.S)


def parse_fields(body: str) -> dict:
    """Extract `field = {value}` / `field = "value"` / `field = bare`."""
    fields: dict[str, str] = {}
    for m in FIELD_RE.finditer(body):
        name = m.group(1).lower()
        k = m.end()
        if k < len(body) and body[k] == "{":
            depth = 0
            j = k
            while j < len(body):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            val = body[k + 1 : j]
            fields[name] = " ".join(val.split())
        elif k < len(body) and body[k] == '"':
            j = body.find('"', k + 1)
            val = body[k + 1 : j]
            fields[name] = " ".join(val.split())
        else:
            j = body.find(",", k)
            val = body[k : j if j != -1 else len(body)]
            fields[name] = " ".join(val.split())
    return fields


def norm_title(t: str) -> str:
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def titles_match(a: str, b: str) -> bool:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # tolerate subtitle truncation / trailing punctuation / case
    if na.startswith(nb) or nb.startswith(na):
        # require >= 12 chars of overlap to count as a match
        shorter = min(len(na), len(nb))
        return shorter >= 12
    return False


# --------------------------------------------------------------- resolvers ---
def resolver_landing(doi: str, sess: requests.Session) -> dict:
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/")
    try:
        r = sess.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        return {"http_status": None, "resolver_title": "", "final_url": "",
                "error": f"{type(exc).__name__}: {exc}"}
    title = ""
    low = r.text.lower()
    m = re.search(r'<meta\s+name="citation_title"\s+content="([^"]*)"', r.text, re.I)
    if not m:
        m = re.search(r'<meta\s+property="og:title"\s+content="([^"]*)"', r.text, re.I)
    if not m:
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    if not title and "doi not found" in low:
        title = "DOI Not Found"
    return {"http_status": r.status_code, "resolver_title": title,
            "final_url": r.url, "error": ""}


def crossref(doi: str, sess: requests.Session) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        r = sess.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return {"http_status": None, "title": "", "error": f"{type(exc).__name__}: {exc}"}
    if r.status_code != 200:
        return {"http_status": r.status_code, "title": "", "error": "non-200"}
    try:
        msg = r.json()["message"]
    except Exception as exc:  # noqa: BLE001
        return {"http_status": r.status_code, "title": "",
                "error": f"json: {type(exc).__name__}"}
    title = ""
    if msg.get("title"):
        title = " ".join(msg["title"])
    return {
        "http_status": r.status_code,
        "title": title,
        "container": " ".join(msg.get("container-title") or []),
        "year": (msg.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "volume": msg.get("volume", ""),
        "issue": msg.get("issue", ""),
        "page": msg.get("page", ""),
        "publisher": msg.get("publisher", ""),
        "type": msg.get("type", ""),
        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in msg.get("author", [])],
        "error": "",
    }


def arxiv(doi: str, sess: requests.Session) -> dict:
    aid = doi.split("arXiv.")[-1]
    url = "http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid)
    try:
        r = sess.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return {"http_status": None, "title": "", "error": f"{type(exc).__name__}: {exc}"}
    if r.status_code != 200:
        return {"http_status": r.status_code, "title": "", "error": "non-200"}
    m = re.search(r"<entry>.*?<title>(.*?)</title>", r.text, re.S)
    title = " ".join(m.group(1).split()) if m else ""
    return {"http_status": r.status_code, "title": title, "error": ""}


def biorxiv(doi: str, sess: requests.Session) -> dict:
    url = "https://api.biorxiv.org/details/biorxiv/" + urllib.parse.quote(doi, safe="/")
    try:
        r = sess.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return {"http_status": None, "title": "", "error": f"{type(exc).__name__}: {exc}"}
    if r.status_code != 200:
        return {"http_status": r.status_code, "title": "", "error": "non-200"}
    try:
        coll = r.json().get("collection") or []
    except Exception as exc:  # noqa: BLE001
        return {"http_status": r.status_code, "title": "", "error": "json"}
    if not coll:
        return {"http_status": r.status_code, "title": "", "error": "empty collection"}
    rec = coll[0]
    return {
        "http_status": r.status_code,
        "title": rec.get("title", ""),
        "authors": rec.get("authors", ""),
        "date": rec.get("date", ""),
        "category": rec.get("category", ""),
        "version": rec.get("version", ""),
        "error": "",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text = BIB.read_text(encoding="utf-8")
    entries = parse_bib(text)
    sess = requests.Session()

    doi_rows = []
    for e in entries:
        fields = parse_fields(e["body"])
        doi = fields.get("doi", "").strip()
        if not doi:
            continue
        doi = doi.rstrip(".")
        row = {
            "cite_key": e["key"],
            "entry_type": e["type"],
            "line": e["line"],
            "doi": doi,
            "bibtex_title": fields.get("title", ""),
            "bibtex_journal": fields.get("journal", fields.get("booktitle", "")),
            "bibtex_year": fields.get("year", ""),
            "bibtex_volume": fields.get("volume", ""),
            "bibtex_number": fields.get("number", ""),
            "bibtex_pages": fields.get("pages", ""),
            "bibtex_url": fields.get("url", ""),
            "resolved": False,
            "http_status": None,
            "resolver_title": "",
            "resolver_source": "",
            "crossref": {},
            "match": False,
            "note": "",
        }

        # 1. DOI resolver landing page
        land = resolver_landing(doi, sess)
        row["http_status"] = land["http_status"]
        row["resolver_title"] = land["resolver_title"]
        row["resolver_source"] = "doi.org landing page"
        row["final_url"] = land["final_url"]
        if land["error"]:
            row["note"] = f"landing error: {land['error']}"

        # 2. Crossref (registration agency record)
        cr = crossref(doi, sess)
        row["crossref"] = {k: v for k, v in cr.items() if k != "error"}
        if not row["resolver_title"] and cr.get("title"):
            row["resolver_title"] = cr["title"]
            row["resolver_source"] = "crossref API"
        if land["http_status"] in (200, 301, 302, 303, 307, 308):
            row["resolved"] = True
        elif cr.get("http_status") == 200:
            row["resolved"] = True
            if row["resolver_source"] == "doi.org landing page" and not row["resolver_title"]:
                row["resolver_source"] = "crossref API"

        # 3/4. primary server APIs for arXiv / bioRxiv prefixes
        if doi.lower().startswith("10.48550/arxiv."):
            ax = arxiv(doi, sess)
            row["arxiv"] = ax
            if ax.get("title"):
                row["resolved"] = True
                row["resolver_title"] = ax["title"]
                row["resolver_source"] = "arXiv API"
        if doi.lower().startswith("10.1101/"):
            bx = biorxiv(doi, sess)
            row["biorxiv"] = bx
            if bx.get("title"):
                row["resolved"] = True
                row["resolver_title"] = bx["title"]
                row["resolver_source"] = "bioRxiv API"

        row["match"] = titles_match(row["resolver_title"], row["bibtex_title"])
        if not row["match"]:
            if not row["resolved"]:
                row["note"] = (row["note"] + " | " if row["note"] else "") + \
                    "DOI did not resolve; title could not be verified"
            else:
                row["note"] = (row["note"] + " | " if row["note"] else "") + \
                    "TITLE MISMATCH: resolver returned a different work"
        doi_rows.append(row)
        sys.stderr.write(f"{e['key']:<34} {doi:<40} {row['http_status']} "
                         f"match={row['match']}\n")
        sys.stderr.flush()
        time.sleep(0.3)

    payload = {"doi_entries": doi_rows}
    (OUT / "_audit_raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"parsed {len(entries)} entries, {len(doi_rows)} DOI-bearing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
