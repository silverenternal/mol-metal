#!/usr/bin/env python3
"""Wf-A175 — sixth pass: narrow lookups for the 4 remaining wrong-DOI entries."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets6_raw.json"
res: dict = {}


def cr_items(params):
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(params), headers=H, timeout=45)
    items = []
    for m in (r.json().get("message", {}).get("items", []) if r.status_code == 200 else []):
        items.append({"doi": m.get("DOI", ""), "title": " ".join(m.get("title") or []),
                      "container": " ".join(m.get("container-title") or []),
                      "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                      "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
                      "publisher": m.get("publisher", ""),
                      "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])]})
    return {"status": r.status_code, "n": len(items), "items": items}


def cr_q(**kw):
    kw.setdefault("rows", 20)
    return cr_items(kw)


def plain(url):
    r = s.get(url, headers=H, timeout=45)
    metas = dict(re.findall(r'<meta\s+name="citation_([a-z_]+)"\s+content="([^"]*)"', r.text))
    tm = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    return {"url": url, "status": r.status_code,
            "html_title": " ".join(tm.group(1).split()) if tm else "",
            "citation": {k: metas.get(k, "") for k in ("title", "journal_title", "volume", "issue",
                                                      "firstpage", "lastpage", "doi")},
            "authors": re.findall(r'<meta\s+name="citation_author"\s+content="([^"]*)"', r.text),
            "dois_found": sorted(set(re.findall(r"10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+", r.text)))[:40],
            "len": len(r.text)}


def main():
    # A. "The barrier of objects" — Origins of Life, ISSN 0020-9637, 1996
    res["origins_1996"] = cr_q(**{"filter": "issn:0020-9637,from-pub-date:1996-01-01,until-pub-date:1996-12-31",
                                  "rows": 100, "select": "DOI,title,container-title,issued,volume,issue,page,author"})
    # B. Rizzuti/Sportelli/Guzzi — Coord Chem Rev 2009
    res["rizzuti"] = cr_q(**{"query.author": "Rizzuti", "rows": 20})
    res["rizzuti_sportelli"] = cr_q(**{"query.bibliographic": "Rizzuti Sportelli Guzzi free-energy studies metallo-drugs", "rows": 10})
    # C. OpenMM 8 — Eastman, PLOS Comput Biol 2023
    res["eastman_plos"] = cr_q(**{"query.author": "Eastman", "query.container-title": "PLOS Computational Biology",
                                  "rows": 30})
    # D. RoseTTAFold-AA — bioRxiv
    res["biorxiv_search"] = plain("https://www.biorxiv.org/search/RoseTTAFold-AA")
    res["biorxiv_search2"] = plain("https://www.biorxiv.org/search/Accurate+protein+complex+structure+prediction")
    # E. arXiv search for RoseTTAFold-AA (may be on arXiv too)
    r = s.get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": 'all:"RoseTTAFold-AA"', "max_results": 10}), headers=H, timeout=45)
    res["arxiv_rosettafoldaa"] = re.findall(r"(?s)<entry>(.*?)</entry>", r.text)
    # F. arXiv AttentiveFP
    r2 = s.get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": 'ti:"AttentiveFP"', "max_results": 10}), headers=H, timeout=45)
    res["arxiv_attentivefp"] = re.findall(r"(?s)<entry>(.*?)</entry>", r2.text)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({k: (v if not isinstance(v, (list,)) else [str(x)[:2000] for x in v]) for k, v in res.items()},
                  fh, indent=2, ensure_ascii=False)
    for it in res["origins_1996"]["items"]:
        if "barrier" in it["title"].lower() or "object" in it["title"].lower():
            print("ORIGINS-MATCH", it)
    print("origins_1996 n =", res["origins_1996"]["n"])
    for it in res["rizzuti"]["items"][:12]:
        print("RIZZUTI", it["doi"], "|", it["title"][:80], "|", it["container"][:32], it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    print("---")
    for it in res["rizzuti_sportelli"]["items"][:8]:
        print("FEP-BIB", it["doi"], "|", it["title"][:80], "|", it["container"][:32], it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    print("---")
    for it in res["eastman_plos"]["items"][:30]:
        print("EASTMAN-PLOS", it["doi"], "|", it["title"][:80], "|", it["year"], "v", it["volume"], it["page"])
    print("---")
    for k in ("biorxiv_search", "biorxiv_search2"):
        print(k, res[k]["status"], res[k]["html_title"][:90])
        print("   dois:", res[k]["dois_found"][:20])
        print("   ct:  ", res[k]["citation"])
    for k in ("arxiv_rosettafoldaa", "arxiv_attentivefp"):
        print("###", k, len(res[k]))
        for e in res[k][:6]:
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            i = re.search(r"<id>(.*?)</id>", e, re.S)
            p = re.search(r"<published>(.*?)</published>", e, re.S)
            print("   ", i.group(1).strip() if i else "", "|", " ".join(t.group(1).split())[:85] if t else "", "|", p.group(1) if p else "")


if __name__ == "__main__":
    main()
