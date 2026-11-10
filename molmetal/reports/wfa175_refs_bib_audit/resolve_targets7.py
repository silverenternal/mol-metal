#!/usr/bin/env python3
"""Wf-A175 — seventh pass: final lookups + build the consolidated audit.json."""
from __future__ import annotations
import html, json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
RD = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit"
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


def pubmed(term, retmax=15):
    e = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
              params={"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}, headers=H, timeout=45)
    ids = e.json().get("esearchresult", {}).get("idlist", [])
    arts = []
    if ids:
        f = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                  params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, headers=H, timeout=60)
        for a in re.findall(r"(?s)<PubmedArticle>.*?</PubmedArticle>", f.text):
            def g(tag):
                m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", a, re.S)
                return " ".join(m.group(1).split()) if m else ""
            dois = re.findall(r"<ArticleId IdType=\"doi\">(.*?)</ArticleId>", a)
            arts.append({"pmid": g("PMID"), "title": g("ArticleTitle"), "journal": g("Title"),
                         "year": g("Year"), "volume": g("Volume"), "issue": g("Issue"),
                         "pages": g("MedlinePgn"), "doi": dois[0] if dois else "",
                         "authors": [f"{fn} {ln}" for ln, fn in re.findall(
                             r"(?s)<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>.*?</Author>", a)]})
    return {"term": term, "ids": ids, "articles": arts}


def biorxiv_sweep(d1, d2, needle):
    cursor = 0
    hits, seen, total = [], 0, None
    for _ in range(40):
        url = f"https://api.biorxiv.org/details/biorxiv/{d1}/{d2}/{cursor}"
        r = s.get(url, headers=H, timeout=60)
        if r.status_code != 200:
            break
        j = r.json()
        msg = j.get("messages", [{}])[0]
        total = msg.get("total", total)
        coll = j.get("collection", [])
        seen += len(coll)
        for i in coll:
            if needle in (i.get("title", "") + " " + i.get("doi", "")).lower():
                hits.append(i)
        nxt = msg.get("next_cursor")
        if nxt is None or not coll or nxt == cursor:
            break
        cursor = nxt
    return {"total": total, "scanned": seen, "hits": hits}


def main():
    # OpenMM titles in PubMed
    res["pubmed_openmm_titles"] = pubmed('OpenMM[Title] AND "2015"[PDAT] : "2026"[PDAT]', retmax=30)
    # Breslow Chem Rev 1998
    res["pubmed_breslow_cr1998"] = pubmed('Breslow[au] AND "Chem Rev"[jour] AND 1998[dp]')
    res["cr_breslow_cr1998"] = cr_items({"query.bibliographic": "Breslow Dong Chemical Reviews 1998 metalloenzymes biomimetic",
                                         "filter": "issn:0009-2645,from-pub-date:1998-01-01,until-pub-date:1998-12-31",
                                         "rows": 100})
    # "The barrier of objects" by Fontana & Buss
    res["cr_barrier_fontana"] = cr_items({"query.bibliographic": "Fontana Buss The barrier of objects dynamical systems bounded organizations",
                                          "rows": 20})
    res["cr_origins_all"] = cr_items({"query.container-title": "Origins of Life and Evolution of the Biospheres",
                                      "query.author": "Fontana", "rows": 30})
    # "Free-energy studies of metallo-drugs" CCR 2009 — by volume/issue/page
    res["cr_ccr_253"] = cr_items({"filter": "issn:0010-8565,from-pub-date:2009-01-01,until-pub-date:2009-12-31",
                                   "rows": 200, "select": "DOI,title,container-title,issued,volume,issue,page,author"})
    # RoseTTAFold-AA on bioRxiv, date window 2024-07-17..2024-07-18
    res["biorxiv_rfaa"] = biorxiv_sweep("2024-07-17", "2024-07-18", "rosettafold")
    with open(f"{RD}/_targets7_raw.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("### pubmed OpenMM titles", res["pubmed_openmm_titles"]["ids"])
    for a in res["pubmed_openmm_titles"]["articles"][:20]:
        print("   ", a["pmid"], "|", a["title"][:76], "|", a["journal"], a["year"], "v", a["volume"], "p", a["pages"], "|", a["doi"])
    print("### pubmed Breslow CR1998", res["pubmed_breslow_cr1998"]["ids"])
    for a in res["pubmed_breslow_cr1998"]["articles"][:10]:
        print("   ", a["pmid"], "|", a["title"][:80], "|", a["journal"], a["year"], "v", a["volume"], "p", a["pages"], "|", a["doi"], "|", a["authors"][:3])
    print("### cr_breslow_cr1998 n=", res["cr_breslow_cr1998"]["n"])
    for it in res["cr_breslow_cr1998"]["items"][:12]:
        print("   ", it["doi"], "|", it["title"][:78], "|", it["year"], it["volume"], it["page"], "|", it["authors"][:3])
    print("### cr_barrier_fontana n=", res["cr_barrier_fontana"]["n"])
    for it in res["cr_barrier_fontana"]["items"][:8]:
        print("   ", it["doi"], "|", it["title"][:78], "|", it["container"][:34], it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    print("### cr_origins_all n=", res["cr_origins_all"]["n"])
    for it in res["cr_origins_all"]["items"][:12]:
        print("   ", it["doi"], "|", it["title"][:78], "|", it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    print("### cr_ccr_253 n=", res["cr_ccr_253"]["n"])
    for it in res["cr_ccr_253"]["items"]:
        if "free" in it["title"].lower() or "metallo-drug" in it["title"].lower() or it["page"].startswith("21"):
            print("   ", it["doi"], "|", it["title"][:78], "|", it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    print("### biorxiv_rfaa total=", res["biorxiv_rfaa"]["total"], "scanned=", res["biorxiv_rfaa"]["scanned"], "hits=", len(res["biorxiv_rfaa"]["hits"]))
    for h in res["biorxiv_rfaa"]["hits"][:10]:
        print("   ", h.get("doi"), "|", h.get("title", "")[:90], "|", h.get("date"), h.get("version"), "|", h.get("authors", "")[:60])


if __name__ == "__main__":
    main()
