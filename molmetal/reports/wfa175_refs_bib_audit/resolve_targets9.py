#!/usr/bin/env python3
"""Wf-A175 — ninth pass: PII->DOI for fep_metallo, RoseTTAFold-AA bioRxiv DOI, Breslow PubDate."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
RD = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit"
res: dict = {}


def pubmed_xml(pmid):
    f = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
              params={"db": "pubmed", "id": pmid, "retmode": "xml"}, headers=H, timeout=60)
    a = re.search(r"(?s)<PubmedArticle>.*?</PubmedArticle>", f.text)
    if not a:
        return {"pmid": pmid, "found": False}
    x = a.group(0)
    def g(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", x, re.S)
        return " ".join(m.group(1).split()) if m else ""
    return {"pmid": pmid, "found": True, "title": g("ArticleTitle"), "journal": g("Title"),
            "volume": g("Volume"), "issue": g("Issue"), "pages": g("MedlinePgn"),
            "pubdate_year": g("PubDate/Year"), "pubdate_medline": g("PubDate/MedlineDate"),
            "articledate": g("ArticleDate/Year"),
            "doi": (re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', x) or [None, ""])[1],
            "authors": [f"{fn} {ln}" for ln, fn in re.findall(
                r"(?s)<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>.*?</Author>", x)],
            "affil": g("Affiliation"), "lang": g("Language")}


def cr_items(params):
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(params), headers=H, timeout=45)
    items = []
    for m in (r.json().get("message", {}).get("items", []) if r.status_code == 200 else []):
        items.append({"doi": m.get("DOI", ""), "title": " ".join(m.get("title") or []),
                      "container": " ".join(m.get("container-title") or []),
                      "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                      "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
                      "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])]})
    return {"status": r.status_code, "n": len(items), "items": items}


def main():
    res["breslow_xml"] = pubmed_xml("11848956")
    res["krasnov_xml"] = pubmed_xml("41969144")
    res["posebusters_xml"] = pubmed_xml("38425520")
    res["dmpnn_xml"] = pubmed_xml("31361484")
    res["openmm_plos_xml"] = pubmed_xml("37943957")
    # PII -> DOI via Crossref alternative-id filter
    res["pii_altid"] = cr_items({"filter": "alternative-id:S0010854509000715", "rows": 5})
    res["pii_search"] = cr_items({"query.bibliographic": "Free-energy studies of metallo-drugs Coordination Chemistry Reviews 2009 253 2097",
                                   "rows": 20})
    res["cr_crr_2009_free"] = cr_items({"query.container-title": "Coordination Chemistry Reviews",
                                        "query.bibliographic": "free-energy studies metallo-drugs", "rows": 20})
    # RoseTTAFold-AA
    res["cr_rfaa_short"] = cr_items({"query.bibliographic": "RoseTTAFold-AA", "rows": 20})
    res["cr_rfaa_prefix"] = cr_items({"query.bibliographic": "RoseTTAFold-AA", "filter": "prefix:10.1101", "rows": 20})
    # bioRxiv single-day raw
    r = s.get("https://api.biorxiv.org/details/biorxiv/2024-07-17/2024-07-17/0", headers=H, timeout=60)
    j = r.json()
    res["biorxiv_0717_raw"] = {"messages": j.get("messages"), "n": len(j.get("collection", [])),
                               "titles": [i.get("title", "")[:100] for i in j.get("collection", [])][:60]}
    with open(f"{RD}/_targets9_raw.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k in ("breslow_xml", "krasnov_xml", "posebusters_xml", "dmpnn_xml", "openmm_plos_xml"):
        print("###", k, json.dumps(res[k], ensure_ascii=False)[:520])
    for k in ("pii_altid", "pii_search", "cr_crr_2009_free"):
        print("###", k, res[k]["n"])
        for it in res[k]["items"][:6]:
            print("   ", it["doi"], "|", it["title"][:74], "|", it["year"], "v", it["volume"], "p", it["page"], "|", it["authors"][:3])
    for k in ("cr_rfaa_short", "cr_rfaa_prefix"):
        print("###", k, res[k]["n"])
        for it in res[k]["items"][:10]:
            print("   ", it["doi"], "|", it["title"][:78], "|", it["year"], "|", it["authors"][:2])
    print("### biorxiv 2024-07-17", res["biorxiv_0717_raw"]["messages"], "n=", res["biorxiv_0717_raw"]["n"])
    for t in res["biorxiv_0717_raw"]["titles"][:40]:
        print("   ", t)


if __name__ == "__main__":
    main()
