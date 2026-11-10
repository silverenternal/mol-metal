#!/usr/bin/env python3
"""Wf-A175 — fourth pass: DOI handle registry + PubMed + remaining lookups."""
from __future__ import annotations
import json, re, sys, urllib.parse
import requests

H = {"User-Agent": "molmetal-wfa175-audit/1.0 (bibliography integrity audit)"}
s = requests.Session()
OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_targets4_raw.json"
res: dict = {}


def handle(doi):
    """Authoritative DOI Handle registry API (primary, part of the DOI system)."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/")
    r = s.get(url, headers=H, timeout=40)
    if r.status_code != 200:
        return {"doi": doi, "status": r.status_code, "registered": False, "targets": []}
    j = r.json()
    rc = j.get("responseCode", 0)
    vals = j.get("values", [])
    targets = []
    for v in vals:
        for t in v.get("data", {}).get("value", []):
            if not isinstance(t, dict):
                continue
            if t.get("type") in ("URL", "FS", "Handle"):
                targets.append({"type": t.get("type"), "format": t.get("format"),
                                "data": t.get("data")})
    return {"doi": doi, "status": r.status_code, "responseCode": rc,
            "registered": rc == 1, "targets": targets}


def pubmed(term, retmax=8):
    e = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
              params={"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax},
              headers=H, timeout=40)
    ids = e.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"term": term, "ids": []}
    f = s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
              params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
              headers=H, timeout=60)
    xml = f.text
    arts = []
    for a in re.findall(r"(?s)<PubmedArticle>.*?</PubmedArticle>", xml):
        def g(tag):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", a, re.S)
            return " ".join(m.group(1).split()) if m else ""
        dois = re.findall(r"<ArticleId IdType=\"doi\">(.*?)</ArticleId>", a)
        ids2 = re.findall(r"<ArticleId IdType=\"pmc\">(.*?)</ArticleId>", a)
        au = re.findall(r"(?s)<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>.*?</Author>", a)
        abst = " ".join(re.findall(r"(?s)<AbstractText[^>]*>(.*?)</AbstractText>", a))
        abst = re.sub(r"<[^>]+>", " ", abst)
        arts.append({"pmid": g("PMID"), "title": g("ArticleTitle"), "journal": g("Title"),
                     "year": g("Year"), "volume": g("Volume"), "issue": g("Issue"),
                     "pages": g("MedlinePgn"), "doi": dois[0] if dois else "", "pmc": ids2[0] if ids2 else "",
                     "authors": [f"{fn} {ln}" for ln, fn in au],
                     "abstract": " ".join(abst.split())[:2200]})
    return {"term": term, "ids": ids, "articles": arts}


def cr(title=None, author=None, container=None, rows=8):
    kw = {"rows": rows}
    if title:
        kw["query.title"] = title
    if author:
        kw["query.author"] = author
    if container:
        kw["query.container-title"] = container
    r = s.get("https://api.crossref.org/works?" + urllib.parse.urlencode(kw), headers=H, timeout=40)
    out = []
    for m in (r.json().get("message", {}).get("items", []) if r.status_code == 200 else []):
        out.append({"doi": m.get("DOI", ""), "title": " ".join(m.get("title") or []),
                    "container": " ".join(m.get("container-title") or []),
                    "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "volume": m.get("volume", ""), "issue": m.get("issue", ""),
                    "page": m.get("page", ""), "publisher": m.get("publisher", ""),
                    "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])]})
    return {"status": r.status_code, "items": out}


def plos_search(q):
    r = s.get("https://journals.plos.org/search?q=" + urllib.parse.quote(q), headers=H, timeout=45)
    dois = sorted(set(re.findall(r"10\.1371/journal\.pcbi\.\d+", r.text)))
    titles = re.findall(r'(?s)<span class="title">(.*?)</span>', r.text)
    return {"status": r.status_code, "dois": dois[:40],
            "titles": [" ".join(re.sub("<[^>]+>", " ", t).split()) for t in titles][:20]}


def biorxiv_day_sweep(d1, d2, needle):
    cursor = 0
    hits = []
    seen = 0
    while True:
        url = f"https://api.biorxiv.org/details/biorxiv/{d1}/{d2}/{cursor}"
        r = s.get(url, headers=H, timeout=60)
        if r.status_code != 200:
            break
        j = r.json()
        coll = j.get("collection", [])
        seen += len(coll)
        for i in coll:
            if needle in (i.get("title", "") + " " + i.get("doi", "") + " " + i.get("authors", "")).lower():
                hits.append(i)
        nxt = j.get("messages", [{}])[0].get("next_cursor")
        if nxt is None or not coll:
            break
        cursor = nxt
        if cursor > 400:
            break
    return {"scanned": seen, "hits": hits}


def main():
    audit = json.load(open("/home/hugo/codes/mol-metal/molmetal/reports/wfa175_refs_bib_audit/_audit_raw.json"))
    dois = [r["doi"] for r in audit["doi_entries"]]
    res["handles"] = {d: handle(d) for d in dois}
    for d in dois:
        h = res["handles"][d]
        print(f"{d:<40} rc={h.get('responseCode')} registered={h['registered']} "
              f"targets={[t['data'] for t in h['targets']][:1]}", file=sys.stderr)
    res["pubmed_breslow"] = pubmed('"Biomimetic Reactions Catalyzed by Metallo-enzymes and Their Models"[Title]')
    res["pubmed_metalcytotoxdb"] = pubmed('10.1021/acs.jmedchem.5c02755[DOI]')
    res["pubmed_posebusters"] = pubmed('"PoseBusters"[Title]')
    res["pubmed_openmm8"] = pubmed('"OpenMM 8"[Title]')
    res["cr_posebusters_title"] = cr(title="PoseBusters", rows=10)
    res["cr_breslow_title2"] = cr(title="Biomimetic Reactions Catalyzed by Metallo-enzymes and Their Models", rows=10)
    res["cr_openmm_plos"] = cr(title="OpenMM", container="PLOS Computational Biology", rows=20)
    res["plos_openmm_search"] = plos_search("OpenMM 8 heterogeneous hardware")
    res["biorxiv_sweep"] = biorxiv_day_sweep("2024-07-17", "2024-07-19", "rosettafold")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k in ("pubmed_breslow", "pubmed_metalcytotoxdb", "pubmed_posebusters", "pubmed_openmm8"):
        print("###", k, res[k].get("ids"))
        for a in res[k].get("articles", [])[:4]:
            print("   ", a["pmid"], "|", a["title"][:90], "|", a["journal"], a["year"],
                  "v", a["volume"], "p", a["pages"], "| doi:", a["doi"])
            print("      au:", ", ".join(a["authors"][:6]))
            if a["abstract"]:
                print("      abs:", a["abstract"][:400])
    print("### cr_posebusters_title")
    for it in res["cr_posebusters_title"]["items"][:8]:
        print("   ", it["doi"], "|", it["title"][:88], "|", it["container"][:30], it["year"], it["page"])
    print("### cr_breslow_title2")
    for it in res["cr_breslow_title2"]["items"][:8]:
        print("   ", it["doi"], "|", it["title"][:88], "|", it["container"][:30], it["year"], it["volume"], it["page"])
    print("### cr_openmm_plos")
    for it in res["cr_openmm_plos"]["items"][:10]:
        print("   ", it["doi"], "|", it["title"][:88], "|", it["year"], it["volume"], it["page"])
    print("### plos_openmm_search", res["plos_openmm_search"]["dois"][:20], res["plos_openmm_search"]["titles"][:6])
    print("### biorxiv_sweep scanned", res["biorxiv_sweep"]["scanned"], "hits", len(res["biorxiv_sweep"]["hits"]))
    for h in res["biorxiv_sweep"]["hits"][:8]:
        print("   ", h.get("doi"), "|", h.get("title", "")[:90], "|", h.get("date"), h.get("version"))


if __name__ == "__main__":
    main()
