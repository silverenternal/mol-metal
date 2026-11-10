#!/usr/bin/env python3
"""Wf-A175 — third pass: correct DOIs for the remaining defective entries."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets3_raw.json"
res: dict = {}


def cr(title=None, author=None, container=None, rows=8, extra=None):
    kw = {"rows": rows}
    if title:
        kw["query.title"] = title
    if author:
        kw["query.author"] = author
    if container:
        kw["query.container-title"] = container
    if extra:
        kw.update(extra)
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(kw), headers=H, timeout=40)
    if r.status_code != 200:
        return {"status": r.status_code, "items": []}
    out = []
    for m in r.json()["message"]["items"]:
        out.append({
            "doi": m.get("DOI", ""), "title": " ".join(m.get("title") or []),
            "container": " ".join(m.get("container-title") or []),
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
            "publisher": m.get("publisher", ""), "type": m.get("type", ""),
            "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
        })
    return {"status": 200, "items": out}


def cr_work(doi):
    r = s.get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""), headers=H, timeout=40)
    if r.status_code != 200:
        return {"doi": doi, "status": r.status_code}
    m = r.json()["message"]
    return {"doi": doi, "status": 200, "title": " ".join(m.get("title") or []),
            "container": " ".join(m.get("container-title") or []),
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
            "publisher": m.get("publisher", ""), "type": m.get("type", ""),
            "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])]}


def biorxiv_range(d1, d2, cursor=0):
    url = f"https://api.biorxiv.org/details/biorxiv/{d1}/{d2}/{cursor}"
    r = s.get(url, headers=H, timeout=60)
    if r.status_code != 200:
        return {"status": r.status_code, "items": []}
    j = r.json()
    items = j.get("collection", [])
    return {"status": 200, "total": j.get("messages", [{}])[0].get("total"), "items": items,
            "next_cursor": j.get("messages", [{}])[0].get("next_cursor")}


def plain(url):
    r = s.get(url, headers=H, timeout=45)
    body = r.text
    metas = dict(re.findall(r'<meta\s+name="citation_([a-z_]+)"\s+content="([^"]*)"', body))
    tm = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    return {"url": url, "status": r.status_code,
            "html_title": " ".join(tm.group(1).split()) if tm else "",
            "citation": {k: metas.get(k, "") for k in ("title", "journal_title", "volume", "issue",
                                                      "firstpage", "lastpage", "doi", "publication_date")},
            "authors": re.findall(r'<meta\s+name="citation_author"\s+content="([^"]*)"', body),
            "len": len(body)}


def main():
    # --- correct DOIs for defective entries ---
    res["openmm_title"] = cr(title="OpenMM 8 Molecular dynamics simulation with heterogeneous hardware", rows=8)
    res["openmm_plos_title"] = cr(title="OpenMM 8", container="PLOS Computational Biology", rows=8)
    res["posebusters_buttenschoen"] = cr(author="Buttenschoen", rows=20)
    res["posebusters_title_all"] = cr(title="PoseBusters AI-based docking methods fail to generate physically valid poses or generalise to novel sequences", rows=10)
    res["breslow_title"] = cr(title="Biomimetic Reactions Catalyzed by Metallo-enzymes and Their Models", rows=8)
    res["fontana_barrier_title"] = cr(title="The barrier of objects from dynamical systems to bounded organizations", rows=8)
    res["freeenergy_metallo_title"] = cr(title="Free-energy studies of metallo-drugs", rows=8)
    res["rusanov_scidata_claim"] = cr_work("10.1038/s41597-025-05012-9")
    res["krasnov_jcim_claim2"] = cr_work("10.1021/acs.jcim.5c01942")
    # --- bioRxiv day window around 2024-07-17 ---
    day = biorxiv_range("2024-07-17", "2024-07-18", 0)
    res["biorxiv_20240717_count"] = day["total"]
    hits = [i for i in day["items"] if "rosettafold" in (i.get("title", "") + i.get("doi", "")).lower()]
    res["biorxiv_20240717_rosettafold_hits"] = hits
    # --- PLOS search for OpenMM 8 ---
    res["plos_search_openmm"] = plain("https://journals.plos.org/ploscompbiol/browse/openmm")
    # --- arXiv full record for DrugOOD ---
    r = s.get("http://export.arxiv.org/api/query?id_list=2201.09637", headers=H, timeout=40)
    res["arxiv_drugood_raw"] = r.text
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k in ("openmm_title", "openmm_plos_title", "posebusters_buttenschoen",
              "posebusters_title_all", "breslow_title", "fontana_barrier_title",
              "freeenergy_metallo_title"):
        print("###", k)
        for it in res[k]["items"][:8]:
            print("   ", it["doi"], "|", it["title"][:85], "|", it["container"][:32],
                  "|", it["year"], it["volume"], it["page"], "|", (it["authors"] or [""])[0])
        print()
    print("rusanov_scidata_claim:", json.dumps(res["rusanov_scidata_claim"])[:400])
    print("krasnov_jcim_claim2:", json.dumps(res["krasnov_jcim_claim2"])[:200])
    print("biorxiv_20240717 total:", res["biorxiv_20240717_count"])
    print("rosettafold hits:", json.dumps(res["biorxiv_20240717_rosettafold_hits"], indent=1)[:1500])


if __name__ == "__main__":
    main()
