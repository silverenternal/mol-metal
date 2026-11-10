#!/usr/bin/env python3
"""Wf-A175 — re-measure the 92.7-94 % band census (UTF-8-safe, no os.walk).

CPU-only. Writes ONLY under molmetal/reports/wfa175_frontier_comparability/.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = "/home/hugo/codes/mol-metal"
sys.path.insert(0, os.path.join(ROOT, "molmetal", "scripts"))
from wfa156_text_scan import iter_repo_files, scan  # noqa: E402

OUT = "/home/hugo/codes/mol-metal/molmetal/reports/wfa175_frontier_comparability"

# the band, in every rendering we have seen: "92.7-94", "92.7–94", "92.7 - 94", "92.7 to 94", "92.7-94 %"
BAND = re.compile(r"92\.7\s*(?:-|–|—|to)\s*94(?:\.0)?\s*%?", re.I)


def census(paths, label):
    hits = []
    for r in scan(ROOT, BAND.pattern, files=paths):
        if r.matches:
            hits.append({"path": r.path, "matches": r.matches,
                         "line_numbers": r.line_numbers[:12], "kind": r.kind})
    return {"label": label, "n_files_scanned": len(paths), "n_files_with_band": len(hits),
            "hits": sorted(hits, key=lambda h: h["path"])}


def main():
    all_files = iter_repo_files(ROOT, untracked=True)
    paper_files = [p for p in all_files if p.startswith("paper/")]
    work = census(all_files, "working tree (tracked + untracked, non-ignored)")
    pap = census(paper_files, "paper/ subtree")

    # committed-at-HEAD census: read blobs straight out of git, no working tree
    head_hits = []
    for p in all_files:
        try:
            blob = subprocess.run(["git", "-C", ROOT, "show", "HEAD:" + p],
                                  capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if blob.returncode != 0:
            continue
        txt = blob.stdout.decode("utf-8", "replace")
        n = len(BAND.findall(txt))
        if n:
            head_hits.append({"path": p, "matches": n})
    head = {"label": "HEAD (committed blobs, all tracked files)",
            "n_files_scanned": len(all_files), "n_files_with_band": len(head_hits),
            "hits": sorted(head_hits, key=lambda h: h["path"])}

    rec = {"pattern": BAND.pattern, "working_tree": work, "paper_subtree": pap, "head": head}
    out = os.path.join(OUT, "band_census.json")
    sys.path.insert(0, os.path.join(ROOT, "molmetal", "scripts"))
    from wf386a3_atomic_write import atomic_write_file, detect_truncation, safe_overwrite_check
    content = json.dumps(rec, indent=1, ensure_ascii=False)
    risk = safe_overwrite_check(out, content)
    trunc = detect_truncation(content)
    atomic_write_file(out, content)
    print(json.dumps({"working_tree_files_with_band": work["n_files_with_band"],
                      "working_tree_files_scanned": work["n_files_scanned"],
                      "paper_files_scanned": pap["n_files_scanned"],
                      "paper_files_with_band": pap["n_files_with_band"],
                      "head_files_with_band": head["n_files_with_band"],
                      "head_files_scanned": head["n_files_scanned"],
                      "overwrite_risk": risk, "truncation": trunc}, indent=1))
    print("\nfirst 8 working-tree hits:")
    for h in work["hits"][:8]:
        print("  ", h["path"], h["matches"], h["line_numbers"][:4])


if __name__ == "__main__":
    main()
