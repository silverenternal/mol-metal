#!/usr/bin/env python
"""Wf-A163 branch-N ADVERSARIAL REFUTATION tool (reviewer: ref1).

Three INDEPENDENT checks, none of which reuses the branch's own logic:

  armed   -- a GENERAL armed-script detector (the branch hard-codes an 8-string
             marker list of Wf-A155 pool literals; this one derives pool
             names and count claims from the AST of every script that exposes
             a --pool argument).
  backing -- for one artifact: is every number rendered in a prose string
             backed by that artifact's own machine fields?
  cite    -- for the whole published tree: for every prose that cites another
             artifact as `file.json -> /jsonpath`, does the quoted number
             still match the cited value?  (No integer-part shortcut: the
             branch guard's int-token match suppresses a stale quote whenever
             the quoted number's integer part equals ANY cited leaf.)

CPU-only.  No torch, no CUDA.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "molmetal" / "scripts"

POOLNAME = re.compile(r"\b(cfm_[a-z0-9]+_[a-z0-9_]*\d{4}_\d{2}_\d{2})\b", re.IGNORECASE)
COUNT_CLAIM = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\s+"
    r"(rows|samples|molecules|compounds|structures|entries|atoms|fragments|"
    r"ligands|poses|decodes|epochs|seeds|runs|records)\b", re.IGNORECASE)


# ---------------------------------------------------------------- armed ----
def string_literals(text: str):
    tree = ast.parse(text)
    doc_end = 0
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        doc_end = tree.body[0].end_lineno or 0
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.lineno <= doc_end:
                continue
            out.append((node.lineno, node.value))
    return out


def armed_scan():
    armed, checked, with_pool = [], 0, []
    for p in sorted(SCRIPTS.glob("*.py")):
        checked += 1
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if '"--pool"' not in text and "'--pool'" not in text:
            continue
        with_pool.append(p.name)
        hits = []
        try:
            lits = string_literals(text)
        except Exception:
            lits = []
        for lineno, val in lits:
            if not val or len(val) < 12:
                continue
            for m in POOLNAME.finditer(val):
                hits.append({"line": lineno, "kind": "pool_name", "text": m.group(0),
                             "literal": val[:160]})
            for m in COUNT_CLAIM.finditer(val):
                hits.append({"line": lineno, "kind": "count_claim",
                             "text": m.group(0), "literal": val[:160]})
        if hits:
            armed.append({"script": str(p.relative_to(ROOT)), "n": len(hits), "hits": hits})
    return {"scripts_checked": checked, "scripts_with_pool_arg": len(with_pool),
            "with_pool": with_pool, "armed_by_general_rule": len(armed), "armed": armed}


# -------------------------------------------------------------- backing ----
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")


def leaves(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + (i,))
    else:
        yield p, o


def backing_check(path):
    obj = json.loads(Path(path).read_text())
    V, T, keys = set(), [], []

    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.append(str(k))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)
    rec(obj)
    for _p, v in leaves(obj):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            V.add(float(v))
        elif isinstance(v, str):
            T.append(v)
    blob = " ".join(keys + T)
    unbacked, total = [], 0
    for p, v in leaves(obj):
        if not isinstance(v, str):
            continue
        for m in NUM.finditer(v):
            tok = m.group(0)
            total += 1
            fv = float(tok.replace(",", ""))
            if "." not in tok:
                if any(abs(x - int(tok.replace(",", ""))) < 1e-9 for x in V):
                    continue
            else:
                if any(abs(x - fv) < 1e-9 for x in V):
                    continue
                if any(abs(x * 100 - fv) < 1e-9 for x in V):
                    continue
            if re.search(r"(?<![0-9.,])" + re.escape(tok) + r"(?![0-9.,])", blob):
                continue
            unbacked.append(("/".join(map(str, p)), tok, v[max(0, m.start() - 50):m.start() + 60]))
    return total, unbacked


# ----------------------------------------------------------------- cite ----
CITE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.json)"
    r"\s*(?:->|=>)\s*"
    r"(?P<jsonpath>/[A-Za-z0-9_.\-/]*)")


def cited_numbers(cited):
    out = []
    if isinstance(cited, bool):
        return out
    if isinstance(cited, (int, float)):
        return [("value", float(cited))]
    if isinstance(cited, dict):
        for k, v in cited.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append((str(k), float(v)))
            elif isinstance(v, dict):
                out += [(f"{k}.{kk}", float(vv)) for kk, vv in v.items()
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool)]
        return out
    if isinstance(cited, list):
        for i, v in enumerate(cited):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((str(i), float(v)))
    return out


def consistent(cf, val, dec):
    tol = 0.5 * (10 ** (-dec)) if dec else 0.0
    return (abs(cf - val) <= tol + 1e-12
            or abs(cf * 100 - val) <= tol + 1e-12
            or abs(cf / 100 - val) <= tol + 1e-12)


def jsonpath_get(obj, jp):
    cur = obj
    for seg in jp.strip("/").split("/"):
        if seg:
            cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    return cur


def cite_scan():
    paths = []
    for d in ["metrics/by_metric", "metrics/paper_tables"]:
        dd = ROOT / d
        if dd.is_dir():
            paths += [p for p in sorted(dd.iterdir())
                      if p.is_file() and p.suffix == ".json"]
    n_cite = n_unres = 0
    mismatches = []
    for p in paths:
        try:
            art = json.loads(p.read_text())
        except Exception:
            continue
        for _pp, text in leaves(art):
            if not isinstance(text, str):
                continue
            for m in CITE.finditer(text):
                n_cite += 1
                rel, jp = m.group("path"), m.group("jsonpath")
                cp = ROOT / rel
                if not cp.exists():
                    n_unres += 1
                    continue
                try:
                    cited = jsonpath_get(json.loads(cp.read_text()), jp)
                except Exception:
                    n_unres += 1
                    continue
                cn = cited_numbers(cited)
                if not cn:
                    continue
                tail = text[m.end(): m.end() + 320]
                sm = re.search(r"\.\s", tail)
                if sm:
                    tail = tail[: sm.start() + 1]
                quoted = []
                for mm in NUM.finditer(tail):
                    tok = mm.group(0)
                    dec = len(mm.group(2)) if mm.group(2) else 0
                    quoted.append((tok, float(tok.replace(",", "")), dec))
                if not quoted:
                    continue
                bad = [(tok, val) for tok, val, dec in quoted
                       if not any(consistent(cf, val, dec) for _l, cf in cn)]
                if bad and len(bad) == len(quoted):
                    mismatches.append({
                        "artifact": str(p.relative_to(ROOT)), "path": jp,
                        "cited": {l: c for l, c in cn}, "quoted_bad": bad,
                        "excerpt": text[:220].replace("\n", " ")})
    return {"artifacts_scanned": len(paths), "citations_found": n_cite,
            "unresolvable": n_unres, "n_strict_mismatches": len(mismatches),
            "mismatches": mismatches}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "armed"
    if mode == "armed":
        print(json.dumps(armed_scan(), indent=2))
    elif mode == "backing":
        for _p in sys.argv[2:]:
            _t, _u = backing_check(_p)
            print("=" * 70)
            print(_p)
            print("  prose numeric tokens:", _t, " UNBACKED:", len(_u))
            for _pp, _tok, _ex in _u:
                print("    ", _pp, "token=", _tok, "|", _ex.replace("\n", " "))
    elif mode == "cite":
        print(json.dumps(cite_scan(), indent=2))
    else:
        raise SystemExit(f"unknown mode {mode}")
