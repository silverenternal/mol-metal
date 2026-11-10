#!/usr/bin/env python3
"""Wf-A175 — eighth pass: settle openmm DOI, PII->DOI, Springer 'barrier of objects', RoseTTAFold-AA."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
RD = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit"
res: dict = {}


def pubmed(term, retmax=8):
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
                         "pages": g("MedlinePgn"), "doi": dois[0] if dois else ""})
    return {"term": term, "ids": ids, "articles": arts}


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


def page_meta(url):
    r = s.get(url, headers=H, timeout=50)
    body = r.text
    metas = dict(re.findall(r'<meta\s+(?:name|property)="(?:citation_[a-z_]+|og:[a-z]+|dc\.[a-z]+)"\s+content="([^"]*)"', body))
    tm = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    return {"url": url, "status": r.status_code,
            "html_title": " ".join(tm.group(1).split()) if tm else "",
            "citation_title": (re.search(r'<meta\s+name="citation_title"\s+content="([^"]*)"', body) or [None, ""])[1]
            if re.search(r'<meta\s+name="citation_title"\s+content="([^"]*)"', body) else "",
            "dois_found": sorted(set(re.findall(r"10\.\d{4,9}/[^\s\"'<>]+", body)))[:40],
            "len": len(body)}


def main():
    for doi in ["10.1371/journal.pcbi.1011617", "10.1021/acs.jcim.9b00237",
                "10.1021/acs.jmedchem.9b00959", "10.1039/d3sc04185a",
                "10.1021/acs.jcim.3c01250", "10.1002/jcc.21334", "10.1021/cr970011j"]:
        res["pm_" + doi] = pubmed(f"{doi}[doi]")
    # ScienceDirect PII -> DOI (publisher's own page)
    res["sd_pii"] = page_meta("https://www.sciencedirect.com/science/article/pii/S0010854509000715")
    res["sd_doi_ccr2009_03_006"] = page_meta("https://www.sciencedirect.com/science/article/pii/S0010854509000716")
    # Springer: 'The barrier of objects'
    res["springer_search"] = page_meta("https://link.springer.com/search?query=" + urllib.parse.quote("The barrier of objects dynamical systems bounded organizations"))
    # bioRxiv prefix 10.1101 with RoseTTAFold-AA title
    res["cr_rfaa_biorxiv"] = cr_items({"query.title": "RoseTTAFold-AA", "filter": "prefix:10.1101", "rows": 20})
    res["cr_rfaa_all"] = cr_items({"query.bibliographic": "RoseTTAFold-AA accurate protein complex structure prediction", "rows": 20})
    with open(f"{RD}/_targets8_raw.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k in [k for k in res if k.startswith("pm_")]:
        v = res[k]
        print("###", k, v["ids"])
        for a in v["articles"][:3]:
            print("   ", a["pmid"], "|", a["title"][:78], "|", a["journal"], a["year"], "v", a["volume"], "p", a["pages"])
    for k in ("sd_pii", "sd_doi_ccr2009_03_006", "springer_search"):
        print("###", k, res[k]["status"], "|", res[k]["html_title"][:90])
        print("    ct:", res[k]["citation_title"][:110])
        print("    dois:", res[k]["dois_found"][:14])
    for k in ("cr_rfaa_biorxiv", "cr_rfaa_all"):
        print("###", k, res[k]["n"])
        for it in res[k]["items"][:10]:
            print("   ", it["doi"], "|", it["title"][:82], "|", it["year"], "|", it["authors"][:2])


if __name__ == "__main__":
    main()
