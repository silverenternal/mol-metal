#!/usr/bin/env python3
"""Wf-A175 — fifth pass: correct DOIs for fontana2006, fep_metallo, openmm, rosettafoldaa, metallodrug_review."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets5_raw.json"
res: dict = {}


def cr(title=None, container=None, author=None, rows=10, **extra):
    kw = {"rows": rows}
    if title:
        kw["query.title"] = title
    if container:
        kw["query.container-title"] = container
    if author:
        kw["query.author"] = author
    kw.update(extra)
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(kw), headers=H, timeout=45)
    items = []
    for m in (r.json().get("message", {}).get("items", []) if r.status_code == 200 else []):
        items.append({"doi": m.get("DOI", ""), "title": " ".join(m.get("title") or []),
                      "container": " ".join(m.get("container-title") or []),
                      "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                      "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
                      "publisher": m.get("publisher", ""),
                      "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])]})
    return {"status": r.status_code, "items": items}


def pubmed(term, retmax=10):
    e = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
              params={"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax},
              headers=H, timeout=45)
    ids = e.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"term": term, "ids": []}
    f = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
              params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, headers=H, timeout=60)
    arts = []
    for a in re.findall(r"(?s)<PubmedArticle>.*?</PubmedArticle>", f.text):
        def g(tag):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", a, re.S)
            return " ".join(m.group(1).split()) if m else ""
        dois = re.findall(r"<ArticleId IdType=\"doi\">(.*?)</ArticleId>", a)
        au = re.findall(r"(?s)<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>.*?</Author>", a)
        arts.append({"pmid": g("PMID"), "title": g("ArticleTitle"), "journal": g("Title"),
                     "year": g("Year"), "volume": g("Volume"), "issue": g("Issue"),
                     "pages": g("MedlinePgn"), "doi": dois[0] if dois else "",
                     "authors": [f"{fn} {ln}" for ln, fn in au]})
    return {"term": term, "ids": ids, "articles": arts}


def plain(url):
    r = s.get(url, headers=H, timeout=45)
    metas = dict(re.findall(r'<meta\s+name="citation_([a-z_]+)"\s+content="([^"]*)"', r.text))
    tm = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    return {"url": url, "status": r.status_code,
            "html_title": " ".join(tm.group(1).split()) if tm else "",
            "citation": {k: metas.get(k, "") for k in ("title", "journal_title", "volume", "issue",
                                                      "firstpage", "lastpage", "doi")},
            "authors": re.findall(r'<meta\s+name="citation_author"\s+content="([^"]*)"', r.text),
            "dois_found": sorted(set(re.findall(r"10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+", r.text)))[:30],
            "len": len(r.text)}


def main():
    # 1. "The barrier of objects" — Origins of Life and Evolution of the Biospheres
    res["barrier_origins"] = cr(title="The barrier of objects from dynamical systems to bounded organizations",
                                container="Origins of Life and Evolution of the Biospheres", rows=10)
    res["barrier_any"] = cr(title="The barrier of objects", rows=10)
    res["barrier_pubmed"] = pubmed('"The barrier of objects"[Title]')
    # 2. "Free-energy studies of metallo-drugs" — Coord Chem Rev 2009
    res["fep_any"] = cr(title="Free-energy studies of metallo-drugs", rows=10)
    res["fep_ccr"] = cr(title="Free energy studies of metallo-drugs",
                        container="Coordination Chemistry Reviews", rows=10)
    res["fep_pubmed"] = pubmed('"Free-energy studies of metallo-drugs"[Title]')
    # 3. OpenMM 8
    res["openmm_pubmed"] = pubmed("OpenMM[Title]", retmax=20)
    res["openmm_bib"] = cr(title="OpenMM 8 molecular dynamics simulation with heterogeneous hardware",
                           container="PLOS Computational Biology", rows=10)
    res["openmm_any"] = cr(title="OpenMM 8", rows=10)
    # 4. RoseTTAFold-AA — bioRxiv search
    res["biorxiv_search_rosettafold"] = plain("https://www.biorxiv.org/search/RoseTTAFold-AA")
    res["biorxiv_search_aa"] = plain("https://www.biorxiv.org/search/Accurate%20protein%20complex%20structure%20prediction")
    # 5. Chemical Reviews 1998 census for the Breslow metallo-enzymes paper
    res["chemrev1998"] = cr(container="Chemical Reviews", rows=200,
                            **{"filter": "from-pub-date:1998-01-01,until-pub-date:1998-12-31"})
    # 6. DrugOOD AAAI + arXiv confirm
    res["aaai_drugood_pdf_meta"] = plain("https://ojs.aaai.org/index.php/AAAI/article/view/25970")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k in ("barrier_origins", "barrier_any", "fep_any", "fep_ccr", "openmm_bib", "openmm_any"):
        print("###", k, res[k]["status"])
        for it in res[k]["items"][:6]:
            print("   ", it["doi"], "|", it["title"][:86], "|", it["container"][:38],
                  it["year"], "v" + str(it["volume"]), "p" + str(it["page"]), "|", (it["authors"] or [""])[:2])
    for k in ("barrier_pubmed", "fep_pubmed"):
        print("###", k, res[k]["ids"])
        for a in res[k]["articles"][:4]:
            print("   ", a["pmid"], a["title"][:80], "|", a["journal"], a["year"], a["volume"], a["pages"], a["doi"])
    print("### openmm_pubmed", res["openmm_pubmed"]["ids"])
    for a in res["openmm_pubmed"]["articles"][:20]:
        print("   ", a["pmid"], "|", a["title"][:78], "|", a["journal"], a["year"], "v", a["volume"], "p", a["pages"], "|", a["doi"])
    print("### biorxiv_search_rosettafold")
    for k in ("biorxiv_search_rosettafold", "biorxiv_search_aa"):
        print("  ", k, res[k]["status"], res[k]["html_title"][:80], res[k]["dois_found"][:12])
    hits = [i for i in res["chemrev1998"]["items"]
            if "metallo" in i["title"].lower() or "biomimetic" in i["title"].lower()]
    print("### chemrev1998 items:", len(res["chemrev1998"]["items"]))
    for i in hits:
        print("   ", i["doi"], "|", i["title"][:88], "|", i["year"], i["volume"], i["page"], "|", i["authors"][:3])


if __name__ == "__main__":
    main()
