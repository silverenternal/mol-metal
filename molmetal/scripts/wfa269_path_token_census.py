#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors. Licensed under the Apache-2.0 License.
r"""Wf-A269 wave-32b — the PATH-TOKEN census, made REPRODUCIBLE.

WHY THIS FILE EXISTS (catalogue §6.269.42)
-------------------------------------------
``molmetal/tests/wfa131_path_reference_guard.py`` extracts repo-relative path
citations with one regex (``PATH_TOKEN``).  Before 2026-11-07 that regex's
lookahead excluded ``.`` as well as word/path characters, so a citation at the
END OF A SENTENCE (``see docs/x.md.``) produced **NO TOKEN AT ALL**: the greedy
character class swallowed the sentence period, the backtrack that would have
yielded ``docs/x.md`` was rejected by the lookahead, and the whole match FAILED.

A broken pointer in that position was therefore **INVISIBLE** to the guard —
not "extracted then rejected", but never tokenised at all.  The regex was
repaired (the lookahead is now ``(?![\w/\\\\-])``), and this script measures
BOTH sides of that repair so the claim is re-derivable from a clean checkout.

WHAT THIS SCRIPT MEASURES (MEASURED 2026-11-07, CPU-only, 0 GPU)
-----------------------------------------------------------------
1. **RAW token counts** under the PRE-FIX regex vs the LIVE one, per document.
   The pre-fix regex suppressed a raw match for essentially every
   sentence-final word-like token in the corpus.
2. **ADMITTED token counts** under both regexes, and the ADDED / REMOVED sets.
   If the admitted set is unchanged, the guard's VERDICT cannot move — which is
   why the repair needed **no BASELINE adjudication**.
3. **The end-dot census** — how many emitted tokens end in ``.``.  The token's
   final character class is ``[A-Za-z0-9_]``, so this count is **0 by
   construction**: the FIRST version of this census was vacuous, and a census
   that cannot fail is not a census.
4. **The LOOSE path+period census** — a repo-rooted path immediately followed
   by a sentence period, with no backtick and no closing paren to terminate
   the token.  This is the shape the pre-fix regex could not see.  The script
   reports how many such shapes exist, how many are invisible to the guard,
   and how many of those dangle (the path does not exist).
5. **The GHOST probe** — the same broken path written in 8 citation shapes, so
   the reader can see which shapes the LIVE guard sees.

CPU-only. 0 GPU. Read-only on the repo (it only reads files and git-free
resolver state).  Writes one JSON artefact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "molmetal/tests/wfa131_path_reference_guard.py"

#: The four SSOT documents the path-reference guard scans.  ``AGENTS.md`` is
#: NOT among them (the guard's own ``DOCUMENTS`` tuple) — so a citation in
#: AGENTS.md is outside this census by construction, and that is disclosed
#: rather than silently assumed.
DOCUMENTS = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "TODO/INDEX.md",
    "TODO/README.md",
    "docs/WF_HISTORY.md",
)

#: The PRE-FIX token regex (the one in force before 2026-11-07).  The ONLY
#: difference from the live one is the lookahead: it excluded ``.`` as well,
#: which made a sentence-final citation produce no token at all.  Embedded
#: here so the delta is reproducible from a clean checkout — the pre-fix
#: module itself is not committed.
PATH_TOKEN_PREFIX = re.compile(
    r"(?<![\w./\\-])"
    r"(?P<path>[A-Za-z0-9_.][A-Za-z0-9_.\-/]*[A-Za-z0-9_])"
    r"(?![\w./\\-])"
)

#: A LOOSE path-shaped scan: a repo-rooted-looking path immediately followed
#: by a sentence period (no space, no backtick, no closing paren).  This is
#: the shape the pre-fix regex could not tokenise.
LOOSE_PATH_PERIOD = re.compile(
    r"(?<![\w./\\-])"
    r"(?P<path>[A-Za-z0-9_][A-Za-z0-9_.\-/]*\.[A-Za-z0-9_]+)"
    r"\.(?![A-Za-z0-9_])"
)

#: The 8 citation shapes the GHOST probe writes.  The probe path does not
#: exist, so "seen" means "the guard tokenised it" and a shape that is NOT
#: seen is a shape in which a broken citation is invisible.
GHOST = "molmetal/tests/wfa269_does_not_exist_zzz.py"
GHOST_SHAPES = {
    "sentence-final (period)": "See %s." % GHOST,
    "mid-sentence (space)": "See %s and more." % GHOST,
    "mid-sentence (comma)": "See %s, and more." % GHOST,
    "backticked mid": "See `%s` and more." % GHOST,
    "backticked sentence-final": "See `%s`." % GHOST,
    "parenthesised mid": "See (%s) and more." % GHOST,
    "parenthesised sentence-final": "See (%s)." % GHOST,
    "line-final bare": "See %s" % GHOST,
}


def load_guard(path: Path = GUARD):
    """Import the live path-reference guard module by file path."""
    spec = importlib.util.spec_from_file_location("wfa131_path_ref_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the guard module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _raw_tokens(pattern: re.Pattern, text: str) -> list[str]:
    return [m.group("path") for m in pattern.finditer(text)]


def _line_spans(pattern: re.Pattern, text: str) -> list[tuple[int, int, int, str]]:
    """``(line_no, start, end, path)`` for every token the pattern emits.

    **PER LINE, on purpose.**  The guard itself scans line by line
    (``Resolver.extract`` splits on newlines), so a span is only comparable
    with another span on the SAME line.  Measuring the census over the whole
    text at once would compare spans across line boundaries and could report
    a false overlap.
    """
    out: list[tuple[int, int, int, str]] = []
    for line_no, line in enumerate(text.splitlines()):
        for m in pattern.finditer(line):
            out.append((line_no, m.start(), m.end(), m.group("path")))
    return out


def _covers(spans, line_no: int, start: int, end: int) -> bool:
    """Does any span on ``line_no`` overlap ``[start, end)``?

    This is the POSITION-LEVEL question.  The first version of this census
    asked a DOCUMENT-LEVEL question — "is this path visible somewhere in the
    document?" — and that is BLIND to the thing it claims to measure: a path
    that appears in a visible form ELSEWHERE in the same document masks the
    blind position, so the census reports "not invisible" while the citation
    at that position is in fact invisible.  ⇒ **A DOCUMENT-LEVEL SET
    DIFFERENCE CANNOT MEASURE A POSITION-LEVEL BLIND SPOT.**
    """
    return any(sl == line_no and s < end and start < e for sl, s, e, _ in spans)


def _admitted_set(guard, resolver, pattern: re.Pattern, text: str) -> set[str]:
    """The tokens ``pattern`` produces that the resolver ADMITS.

    The guard's own ``Resolver.extract`` does exactly this for the LIVE regex
    (regex tokens + brace-expanded members, then the admission test).  This
    function is the same procedure with a DIFFERENT regex, so the pre-fix side
    is measured the same way rather than inferred.
    """
    out: set[str] = set()
    for line in text.splitlines():
        for m in pattern.finditer(line):
            tok = m.group("path")
            if resolver._admitted(tok):
                out.add(tok)
        for tok in guard._brace_members(line):
            if resolver._admitted(tok):
                out.add(tok)
    return out


def census(guard) -> dict:
    """Measure both sides of the repair over the four SSOT documents."""
    resolver = guard._resolver(REPO)
    per_doc = {}
    admitted_union: set[str] = set()
    raw_prefix_total = raw_live_total = 0
    enddot_prefix = enddot_live = 0

    for rel in DOCUMENTS:
        text = (REPO / rel).read_text(encoding="utf-8")
        raw_p = _raw_tokens(PATH_TOKEN_PREFIX, text)
        raw_l = _raw_tokens(guard.PATH_TOKEN, text)
        seen_live = set(resolver.extract(text))
        seen_prefix = _admitted_set(guard, resolver, PATH_TOKEN_PREFIX, text)
        added = sorted(seen_live - seen_prefix)
        removed = sorted(seen_prefix - seen_live)
        admitted_union |= seen_live | seen_prefix
        raw_prefix_total += len(raw_p)
        raw_live_total += len(raw_l)
        enddot_prefix += sum(1 for t in raw_p if t.endswith("."))
        enddot_live += sum(1 for t in raw_l if t.endswith("."))
        per_doc[rel] = {
            "raw_prefix": len(raw_p),
            "raw_live": len(raw_l),
            "raw_delta": len(raw_l) - len(raw_p),
            "admitted_prefix": len(seen_prefix),
            "admitted_live": len(seen_live),
            "added": added,
            "removed": removed,
        }

    # ---- the LOOSE path+period census (the blind shape) --------------------
    # POSITION-LEVEL: for every citation that sits in the blind shape, ask
    # whether a token span COVERS THAT POSITION — under the pre-fix regex
    # (what the defect could see) and under the live one (what the guard
    # sees).  The first version of this census asked a DOCUMENT-LEVEL
    # question and was therefore blind to the thing it measured.
    loose_total = 0
    invisible_total = failing_total = 0
    exposed_total = exposed_failing_total = 0
    loose_detail: dict[str, dict] = {}
    examples: list[dict] = []
    for rel in DOCUMENTS:
        text = (REPO / rel).read_text(encoding="utf-8")
        prefix_spans = _line_spans(PATH_TOKEN_PREFIX, text)
        live_spans = _line_spans(guard.PATH_TOKEN, text)
        d = {
            "loose_path_period": 0,
            "invisible_to_guard": 0,
            "of_which_failing": 0,
            "exposed_to_prefix_regex": 0,
            "of_which_failing_exposed": 0,
        }
        for line_no, line in enumerate(text.splitlines()):
            for m in LOOSE_PATH_PERIOD.finditer(line):
                path = m.group("path")
                if not resolver._admitted(path):
                    continue
                start, end = m.start(), m.end()
                live_seen = _covers(live_spans, line_no, start, end)
                prefix_seen = _covers(prefix_spans, line_no, start, end)
                bucket = resolver.classify(path)[0]
                failing = bucket in guard.FAILING_BUCKETS
                d["loose_path_period"] += 1
                if not live_seen:
                    d["invisible_to_guard"] += 1
                    if failing:
                        d["of_which_failing"] += 1
                if not prefix_seen:
                    d["exposed_to_prefix_regex"] += 1
                    if failing:
                        d["of_which_failing_exposed"] += 1
                    examples.append({
                        "document": rel,
                        "line_no": line_no,
                        "path": path,
                        "bucket": bucket,
                        "failing": failing,
                        "live_seen": live_seen,
                        "prefix_seen": prefix_seen,
                    })
        loose_detail[rel] = d
        loose_total += d["loose_path_period"]
        invisible_total += d["invisible_to_guard"]
        failing_total += d["of_which_failing"]
        exposed_total += d["exposed_to_prefix_regex"]
        exposed_failing_total += d["of_which_failing_exposed"]

    if exposed_failing_total:
        verdict = "FIRING"
    elif exposed_total:
        verdict = "LATENT"
    else:
        verdict = "NOT-EXPOSED"

    # ---- the GHOST probe ---------------------------------------------------
    ghost = {}
    for name, shape in GHOST_SHAPES.items():
        raw = _raw_tokens(guard.PATH_TOKEN, shape)
        raw_p = _raw_tokens(PATH_TOKEN_PREFIX, shape)
        ghost[name] = {
            "seen_by_live_guard": GHOST in set(resolver.extract(shape)),
            "seen_by_prefix_regex": GHOST in set(raw_p),
            "raw_tokens_live": raw,
            "raw_tokens_prefix": raw_p,
        }

    return {
        "measured_at_note": "MEASURED 2026-11-07, CPU-only, 0 GPU",
        "documents": list(DOCUMENTS),
        "documents_note": "the guard's own DOCUMENTS tuple; AGENTS.md is NOT "
                          "scanned by the guard, so it is outside this census",
        "raw_token_counts": {
            "prefix": raw_prefix_total,
            "live": raw_live_total,
            "delta": raw_live_total - raw_prefix_total,
        },
        "admitted_union_n": len(admitted_union),
        "admitted_set_changed": any(per_doc[r]["added"] or per_doc[r]["removed"]
                                    for r in per_doc),
        "per_document": per_doc,
        "end_dot_census": {
            "prefix": enddot_prefix,
            "live": enddot_live,
            "note": "0 BY CONSTRUCTION — the token's final character class is "
                    "[A-Za-z0-9_], so no emitted token can end in '.'; the "
                    "first version of this census was vacuous",
        },
        "loose_path_period_census": {
            "per_document": loose_detail,
            "total": loose_total,
            "invisible_to_guard": invisible_total,
            "of_which_failing": failing_total,
            "exposed_to_prefix_regex": exposed_total,
            "of_which_failing_exposed": exposed_failing_total,
            "examples": examples,
            "verdict": verdict,
            "measure_note": "POSITION-LEVEL: a citation in the blind shape is "
                           "INVISIBLE iff no token span covers its position "
                           "on its own line.  The first version of this "
                           "census asked a DOCUMENT-LEVEL question (is the "
                           "path visible somewhere in the document?) and was "
                           "therefore BLIND to the blind spot — a path that "
                           "also appears in a visible form elsewhere masked "
                           "it.",
            "verdict_note": "FIRING = the pre-fix regex could not see a "
                            "citation the guard would call FAILING; "
                            "LATENT = it could not see one, but that one "
                            "resolves (exposure without harm); "
                            "NOT-EXPOSED = the corpus never uses the shape.",
        },
        "ghost_probe": ghost,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/wfa269_path_token_census.json")
    args = ap.parse_args()

    guard = load_guard()
    out = census(guard)

    dest = Path(args.out)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    raw = out["raw_token_counts"]
    print("RAW tokens  prefix:", raw["prefix"], " live:", raw["live"],
          " delta:", raw["delta"])
    print("ADMITTED union n :", out["admitted_union_n"],
          "| admitted set changed:", out["admitted_set_changed"])
    for rel, d in out["per_document"].items():
        print(f"  {rel:<34} raw {d['raw_prefix']} -> {d['raw_live']} "
              f"(+{d['raw_delta']}) | admitted {d['admitted_prefix']} -> "
              f"{d['admitted_live']} | added {len(d['added'])} "
              f"removed {len(d['removed'])}")
    print("END-DOT census   :", out["end_dot_census"]["prefix"], "->",
          out["end_dot_census"]["live"], "(0 by construction)")
    lp = out["loose_path_period_census"]
    print("LOOSE path+period:", lp["total"], "| invisible to guard:",
          lp["invisible_to_guard"], "| of which failing:", lp["of_which_failing"],
          "| exposed to the pre-fix regex:", lp["exposed_to_prefix_regex"],
          "| of which failing:", lp["of_which_failing_exposed"],
          "| verdict:", lp["verdict"])
    print("GHOST probe      :")
    for name, g in out["ghost_probe"].items():
        print(f"  {name:<30} live_seen={g['seen_by_live_guard']!s:<6} "
              f"prefix_seen={g['seen_by_prefix_regex']!s:<6}")
    print()
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
