#!/usr/bin/env python3
"""Wf-A175 FRONTIER — batch fetch + text cache, then keyword mining.

Writes ONLY under molmetal/reports/wfa175_frontier_comparability/.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fetch import fetch_eprint  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)

KEY = re.compile(
    r"connectivit|connected|fragment|largest fragment|single fragment|"
    r"disconnected|multi[- ]component|sanitiz|validity|valid rate|"
    r"chemically valid|valid molecule|% *valid",
    re.I,
)


def get(aid: str, refresh: bool = False):
    p = os.path.join(CACHE, aid.replace("/", "_") + ".json")
    if os.path.exists(p) and not refresh:
        return json.load(open(p))
    rec = fetch_eprint(aid)
    rec["arxiv"] = aid
    files = rec.pop("files")
    rec["file_names"] = sorted(files)
    rec["file_bytes"] = {k: len(v) for k, v in files.items()}
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    # text cache separately (gzipped) to keep size sane
    import gzip
    tp = os.path.join(CACHE, aid.replace("/", "_") + ".txt.gz")
    with gzip.open(tp + ".tmp", "wt", encoding="utf-8") as fh:
        for k in sorted(files):
            fh.write("\n=====FILE %s=====\n" % k)
            fh.write(files[k])
    os.replace(tp + ".tmp", tp)
    return rec


def load_text(aid: str) -> str:
    import gzip
    tp = os.path.join(CACHE, aid.replace("/", "_") + ".txt.gz")
    with gzip.open(tp, "rt", encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fetch":
        for aid in sys.argv[2:]:
            try:
                r = get(aid)
                print(aid, "bytes", r["bytes"], "sha", r["sha256"][:12],
                      "nfiles", len(r.get("file_names", [])),
                      r.get("extract_error", ""))
            except Exception as e:  # noqa: BLE001
                print(aid, "FAIL", type(e).__name__, str(e)[:90])
    elif mode == "mine":
        aid = sys.argv[2]
        pat = sys.argv[3]
        rx = re.compile(pat, re.I)
        txt = load_text(aid)
        name = ""
        ln = 0
        for line in txt.splitlines():
            if line.startswith("=====FILE "):
                name = line[11:-5]
                ln = 0
                continue
            ln += 1
            if rx.search(line):
                print("%s:%d: %s" % (name, ln, line.strip()[:400]))
