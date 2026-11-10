#!/usr/bin/env python3
"""Wf-A175 — multi-paper phrase scan over the cached e-print text."""
import gzip
import os
import re
import sys

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def files_of(aid):
    txt = gzip.open(os.path.join(CACHE, aid.replace("/", "_") + ".txt.gz"), "rt", encoding="utf-8").read()
    parts = re.split(r"\n=====FILE (.+?)=====\n", txt)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def scan(aid, pattern, maxlen=320, ctx=0):
    rx = re.compile(pattern, re.I)
    out = []
    for name, text in sorted(files_of(aid).items()):
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                out.append((name, i, line.strip()))
    return out


if __name__ == "__main__":
    pat = sys.argv[1]
    aids = sys.argv[2:]
    for aid in aids:
        hits = scan(aid, pat)
        print("#### %s  (%d hits)" % (aid, len(hits)))
        for n, i, l in hits[:40]:
            print("   %s:%d: %s" % (n, i, l[:320]))
