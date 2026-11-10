#!/usr/bin/env python3
"""Wf-A175 — resolve the CORRECT primary records for the defective / missing keys."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

UA = "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"
H = {"User-Agent": UA}
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets_raw.json"
s = requests.Session()


def crossref_by_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    r = s.get(url, headers=H, timeout=40)
    if r.status_code != 200:
        return {"doi": doi, "status": r.status_code, "msg": r.text[:120]}
    m = r.json()["message"]
    return {
        "doi": doi, "status": 200,
        "title": " ".join(m.get("title") or []),
        "container": " ".join(m.get("container-title") or []),
        "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "volume": m.get("volume", ""), "issue": m.get("issue", ""),
        "page": m.get("page", ""), "publisher": m.get("publisher", ""),
        "type": m.get("type", ""),
        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
        "url": m.get("URL", ""),
    }


def crossref_search(query, rows=5):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": rows,
         "select": "DOI,title,container-title,issued,volume,issue,page,publisher,type,author,URL"})
    r = s.get(url, headers=H, timeout=40)
    if r.status_code != 200:
        return {"query": query, "status": r.status_code}
    out = []
    for m in r.json()["message"]["items"]:
        out.append({
            "doi": m.get("DOI", ""),
            "title": " ".join(m.get("title") or []),
            "container": " ".join(m.get("container-title") or []),
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""),
            "page": m.get("page", ""), "publisher": m.get("publisher", ""),
            "type": m.get("type", ""),
            "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
            "url": m.get("URL", ""),
        })
    return {"query": query, "status": 200, "items": out}


def arxiv(aid):
    url = "http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid)
    r = s.get(url, headers=H, timeout=40)
    m = re.search(r"<entry>(.*?)</entry>", r.text, re.S)
    if not m:
        return {"arxiv_id": aid, "status": r.status_code, "found": False}
    e = m.group(1)
    def g(tag):
        mm = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
        return " ".join(mm.group(1).split()) if mm else ""
    auths = re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)
    return {
        "arxiv_id": aid, "status": r.status_code, "found": True,
        "title": g("title"), "summary": g("summary")[:400],
        "published": g("published"), "updated": g("updated"),
        "authors": [a.strip() for a in auths], "id": g("id"),
        "journal_ref": g("arxiv:journal_ref"), "doi": g("arxiv:doi"),
        "categories": re.findall(r'term="([^"]+)"', e),
    }


def biorxiv_search(title, rows=5):
    url = "https://api.biorxiv.org/details/biorxiv/" + urllib.parse.quote("0")
    return None


def biorxiv_title(title_q):
    """bioRxiv has no title search in the public API; use the site's search endpoint."""
    url = "https://www.biorxiv.org/search/" + urllib.parse.quote(title_q, safe="")
    r = s.get(url, headers=H, timeout=40)
    dois = sorted(set(re.findall(r"10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+", r.text)))
    titles = re.findall(r'<span class="highwire-cite-title">(.*?)</span>', r.text, re.S)
    titles = [" ".join(t.split()) for t in titles]
    return {"query": title_q, "status": r.status_code, "dois": dois[:rows*3], "titles": titles[:rows]}


def chemrxiv(doi):
    url = "https://chemrxiv.org/engage/chemrxiv/article-details/" + urllib.parse.quote(doi, safe="")
    r = s.get(url, headers=H, timeout=40)
    return {"doi": doi, "status": r.status_code,
            "title": (re.search(r"<title>(.*?)</title>", r.text, re.S).group(1).strip()
                      if re.search(r"<title>(.*?)</title>", r.text, re.S) else ""),
            "len": len(r.text)}


TARGETS = {
    "by_doi": [
        "10.1021/acs.jmedchem.5c02755",
        "10.26434/chemrxiv-2025-pp32k",
        "10.1038/s41467-024-49069-0",
        "10.1371/journal.pcbi.1012853",
        "10.1021/cr9600577",
        "10.1101/2024.07.17.603964",
        "10.1007/BF02473273",
        "10.1016/j.ccr.2009.03.006",
        "10.1609/aaai.v37i.v37i7.25970",
        "10.48550/arXiv.2201.09637",
    ],
    "search": [
        "PoseBusters AI-based docking methods fail to generate physically valid poses Nature Communications",
        "Breslow Dong Biomimetic reactions catalyzed by metallo-enzymes and their models Chemical Reviews",
        "Fontana Caputi The barrier of objects from dynamical systems to bounded organizations",
        "Free-energy studies of metallo-drugs coordination chemistry reviews",
        "OpenMM 8 molecular dynamics simulation with heterogeneous hardware",
        "RoseTTAFold-AA accurate protein complex structure prediction",
        "MetalCytoToxDB curated database metal-complex cytotoxicity",
        "PlatinAI platinum cytotoxicity dataset",
    ],
    "arxiv": ["2201.09637", "2002.07177", "2407.17145"],
}


def main():
    res = {"by_doi": {}, "search": {}, "arxiv": {}}
    for d in TARGETS["by_doi"]:
        res["by_doi"][d] = crossref_by_doi(d)
        print("DOI", d, "->", json.dumps(res["by_doi"][d])[:260], file=sys.stderr)
    for q in TARGETS["search"]:
        res["search"][q] = crossref_search(q, rows=5)
        print("SEARCH", q, "->", file=sys.stderr)
        for it in (res["search"][q].get("items") or [])[:5]:
            print("    ", it["doi"], "|", it["title"][:95], "|", it["container"][:40],
                  "|", it["year"], it["volume"], it["page"], file=sys.stderr)
    for a in TARGETS["arxiv"]:
        res["arxiv"][a] = arxiv(a)
        print("ARXIV", a, "->", json.dumps(res["arxiv"][a])[:300], file=sys.stderr)
    res["biorxiv_search"] = biorxiv_title("RoseTTAFold-AA accurate protein complex structure prediction")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
