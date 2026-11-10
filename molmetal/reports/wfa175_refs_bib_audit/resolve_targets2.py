#!/usr/bin/env python3
"""Wf-A175 — second pass: locate the CORRECT primary records."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

UA = "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"
H = {"User-Agent": UA}
s = requests.Session()
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets2_raw.json"
res: dict = {}


def cr_work(doi):
    r = s.get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""), headers=H, timeout=40)
    if r.status_code != 200:
        return {"doi": doi, "status": r.status_code}
    m = r.json()["message"]
    return {
        "doi": doi, "status": 200,
        "title": " ".join(m.get("title") or []),
        "container": " ".join(m.get("container-title") or []),
        "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
        "publisher": m.get("publisher", ""), "type": m.get("type", ""),
        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
        "abstract": re.sub(r"<[^>]+>", " ", m.get("abstract", "") or "")[:2500],
        "url": m.get("URL", ""),
    }


def cr_query(**kw):
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(kw), headers=H, timeout=40)
    if r.status_code != 200:
        return {"status": r.status_code}
    out = []
    for m in r.json()["message"]["items"]:
        out.append({
            "doi": m.get("DOI", ""),
            "title": " ".join(m.get("title") or []),
            "container": " ".join(m.get("container-title") or []),
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
            "publisher": m.get("publisher", ""), "type": m.get("type", ""),
            "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
        })
    return {"status": 200, "items": out}


def arxiv_search(title, rows=8):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f'ti:"{title}"', "start": 0, "max_results": rows})
    r = s.get(url, headers=H, timeout=40)
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    out = []
    for e in entries:
        def g(tag):
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return " ".join(mm.group(1).split()) if mm else ""
        out.append({"id": g("id"), "title": g("title"), "published": g("published"),
                    "authors": [a.strip() for a in re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)],
                    "doi": g("arxiv:doi"), "journal_ref": g("arxiv:journal_ref")})
    return {"status": r.status_code, "items": out}


def epmc(doi):
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           + urllib.parse.urlencode({"query": f'DOI:"{doi}"', "resultType": "core", "format": "json"}))
    r = s.get(url, headers=H, timeout=40)
    if r.status_code != 200:
        return {"doi": doi, "status": r.status_code}
    j = r.json()
    resu = j.get("resultList", {}).get("result", [])
    if not resu:
        return {"doi": doi, "status": 200, "hits": 0}
    r0 = resu[0]
    return {"doi": doi, "status": 200, "hits": len(resu),
            "title": r0.get("title"), "journal": r0.get("journalInfo", {}).get("journal", {}).get("title"),
            "year": r0.get("pubYear"), "volume": r0.get("journalInfo", {}).get("volume"),
            "issue": r0.get("journalInfo", {}).get("issue"),
            "pages": r0.get("pageInfo"), "pmid": r0.get("pmid"),
            "authors": [a.get("fullName") for a in r0.get("authorList", {}).get("author", [])][:30],
            "abstract": (r0.get("abstractText") or "")[:2500],
            "source": r0.get("source")}


def plain(url, name):
    r = s.get(url, headers=H, timeout=40)
    body = r.text
    out = {"url": url, "status": r.status_code, "len": len(body)}
    tm = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    out["html_title"] = " ".join(tm.group(1).split()) if tm else ""
    # citation_* metas
    metas = dict(re.findall(r'<meta\s+name="citation_([a-z_]+)"\s+content="([^"]*)"', body))
    out["citation"] = {k: metas.get(k, "") for k in
                       ("title", "journal_title", "publication_date", "volume", "issue",
                        "firstpage", "lastpage", "doi", "author", "pdf_url", "conference")}
    out["authors"] = re.findall(r'<meta\s+name="citation_author"\s+content="([^"]*)"', body)
    out["abstract_frag"] = re.findall(r'(?s)<div[^>]*(?:abstract|Abstract)[^>]*>(.*?)</div>', body)[:1]
    return out


def main():
    # 1. corrected DOIs for defective entries
    res["krasnov_jmc"] = cr_work("10.1021/acs.jmedchem.5c02755")
    res["krasnov_jcim_claim"] = cr_work("10.1021/acs.jcim.5c01942")
    res["rusanov_chemrxiv"] = cr_work("10.26434/chemrxiv-2025-pp32k")
    res["posebusters_natcomm"] = cr_work("10.1038/s41467-024-49069-0")
    res["metallodrug_review"] = cr_work("10.1021/cr9600577")
    res["openmm_1011617"] = cr_work("10.1371/journal.pcbi.1011617")
    # 2. title searches
    res["cr_openmm"] = cr_query(**{"query.bibliographic": "OpenMM 8 molecular dynamics simulation heterogeneous hardware",
                                   "rows": 8})
    res["cr_posebusters_natcomm"] = cr_query(**{"query.bibliographic": "PoseBusters AI-based docking methods physically valid poses",
                                                "query.container-title": "Nature Communications", "rows": 8})
    res["cr_breslow"] = cr_query(**{"query.bibliographic": "Biomimetic reactions catalyzed by metallo-enzymes and their models",
                                     "query.author": "Breslow", "rows": 8})
    res["cr_fontana_barrier"] = cr_query(**{"query.bibliographic": "The barrier of objects from dynamical systems to bounded organizations",
                                             "query.author": "Fontana", "rows": 8})
    res["cr_freeenergy_metallo"] = cr_query(**{"query.bibliographic": "Free energy studies of metallo-drugs", "rows": 8})
    res["cr_rosettafoldaa"] = cr_query(**{"query.bibliographic": "RoseTTAFold-AA accurate protein complex structure prediction", "rows": 8})
    res["cr_metalcytotoxdb"] = cr_query(**{"query.bibliographic": "MetalCytoToxDB metal complex cytotoxicity database", "rows": 8})
    # 3. arXiv title searches
    res["arxiv_attentivefp"] = arxiv_search("AttentiveFP")
    res["arxiv_rosettafoldaa"] = arxiv_search("RoseTTAFold-AA")
    res["arxiv_drugood"] = arxiv_search("DrugOOD")
    # 4. Europe PMC primary-index records
    for d in ["10.1021/acs.jmedchem.5c02755", "10.1021/acs.jcim.9b00237",
              "10.1021/cr9600577", "10.1371/journal.pcbi.1011617"]:
        res["epmc_" + d] = epmc(d)
    # 5. publisher landing pages
    res["aaai_drugood"] = plain("https://ojs.aaai.org/index.php/AAAI/article/view/25970", "aaai")
    res["plos_openmm_1011617"] = plain("https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011617", "plos")
    res["chemrxiv_pp32k"] = plain("https://chemrxiv.org/engage/chemrxiv/article-details/60c74d7a7c7d4a2c8d3ba5b1", "chemrxiv")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:14000])


if __name__ == "__main__":
    main()
