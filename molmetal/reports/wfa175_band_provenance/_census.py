"""Wf-A175 BAND branch -- band census + per-site classification.

READ-ONLY on the whole repo; writes only into this directory.
CPU-only.  Never uses os.walk (201 GB repo) -- enumerates via `git ls-files`.

POPULATION -- CORRECTED 2026-11-01 (Wf-A175 defect A4)
========================================================
The first run of this script (2026-11-01 18:57) enumerated "tracked +
untracked-and-not-ignored" only, so the whole gitignored tree was INVISIBLE
to the census.  `.gitignore:158` hides every `molmetal/reports/**/*.json` --
exactly the artefacts a band-provenance audit must read (e.g.
`molmetal/reports/wfa171_band_publication_audit/band_classification.json`).
The population is now three git groups:

    tracked                  `git ls-files --cached`
    untracked (not ignored)  `git ls-files --others --exclude-standard`
    untracked (IGNORED)      `git ls-files --others --ignored --exclude-standard`

The three calls MUST be SEPARATE.  A single
`git ls-files --cached --others --ignored --exclude-standard` returns only
17 of the 9,550 tracked files, because `--exclude-standard` also filters the
`--cached` listing (MEASURED 2026-11-01: 411,268 paths vs 420,801 for the
union of the three separate calls).

MEMORY SAFETY: the ignored tree is ~153 GB in a single .pt file, so every
file is streamed in 4 MiB chunks and never fully materialised.  A file is
treated as binary (and skipped) when its first 8 KiB contain a NUL -- the
same rule the first version used.

TWO-PHASE SCAN: phase A streams every text file and looks for the ASCII
anchor "92.7" with a C-level substring test (plus a 4-byte carry, so an
occurrence straddling a chunk boundary is still seen); only the files that
carry the anchor are re-read in phase B, where the lines are split and the
regexes are applied.  Phase B reproduces `str.splitlines()` semantics
exactly (`\\n`, `\\r`, `\\r\\n`, `\\v`, `\\f`, `\\x1c`, `\\x1d`, `\\x1e`,
`U+0085`, `U+2028`, `U+2029`; the last three are matched by their UTF-8
byte encodings).

OUTPUTS -- this script NEVER overwrites the legacy files, so the first
measurement and `_build_sites.py`'s input stay intact:
    _census_raw_v2_incl_ignored.json
    _occurrences_v1_dash_any_v2_incl_ignored.json
"""
from __future__ import annotations

import codecs
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
OUT = ROOT / "molmetal" / "reports" / "wfa175_band_provenance"
SELF_DIR = "molmetal/reports/wfa175_band_provenance"

sys.path.insert(0, str(ROOT))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

# ---------------------------------------------------------------- variants
VARIANTS = {
    # the "92.7 followed by a dash and 94" census
    "v1_dash_any": re.compile(r"92\.7\s*[-‐‑‒–—―~]\s*94"),
    # plain ascii hyphen only
    "v2_ascii_hyphen": re.compile(r"92\.7\s*-\s*94"),
    # tight, any single dash char
    "v3_tight_dash": re.compile(r"92\.7[-‐‑‒–—―~]94"),
    # wide tolerance (prose gap)
    "v4_wide_gap": re.compile(r"92\.7.{0,40}?94"),
}

# str.splitlines() boundary set, encoded as UTF-8 byte sequences.
_LINE_BOUNDARY = re.compile(
    rb"\r\n|\r|\n|\x0b|\x0c|\x1c|\x1d|\x1e|\xc2\x85|\xe2\x80\xa8|\xe2\x80\xa9"
)

ANCHOR = b"92.7"
CHUNK = 4 * 1024 * 1024
HEAD_BYTES = 8192
MAX_CARRY = 1 << 20  # 1 MiB cap on a single line held in memory


def git_ls_files(extra: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z", *extra],
        cwd=str(ROOT),
        capture_output=True,
        check=True,
    )
    return [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]


def population() -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Three SEPARATE git calls (see module docstring) -> groups + union."""
    groups = {
        "tracked": git_ls_files(["--cached"]),
        "untracked": git_ls_files(["--others", "--exclude-standard"]),
        "ignored": git_ls_files(["--others", "--ignored", "--exclude-standard"]),
    }
    owner: dict[str, str] = {}
    for name in ("tracked", "untracked", "ignored"):
        for p in groups[name]:
            owner.setdefault(p, name)
    return groups, sorted((p, g) for p, g in owner.items())


def sentence_of(line: str, start: int, end: int) -> str:
    """Full sentence (or paragraph-line) containing the match."""
    # sentence boundaries
    left = 0
    for i in range(start - 1, -1, -1):
        ch = line[i]
        if ch in ".!?" and (i + 1 >= len(line) or line[i + 1] in " \t") and not (
            i + 1 < len(line) and line[i + 1].isdigit()
        ):
            left = i + 1
            break
    right = len(line)
    for i in range(end, len(line)):
        if line[i] in ".!?" and (i + 1 >= len(line) or line[i + 1] in " \t"):
            right = i + 1
            break
    s = line[left:right].strip()
    return s or line.strip()


def phase_a(path: Path) -> dict:
    """Stream the file once: classify utf8-ness and look for the ASCII anchor.

    Returns {"kind", "bytes", "has_anchor"}.  `bytes` counts the bytes read.
    """
    out = {"kind": "utf8", "bytes": 0, "has_anchor": False}
    try:
        fh = open(path, "rb")
    except OSError:
        out["kind"] = "missing"
        return out
    with fh:
        head = fh.read(HEAD_BYTES)
        if b"\x00" in head:
            out["kind"] = "binary"
            try:
                out["bytes"] = path.stat().st_size
            except OSError:
                pass
            return out
        fh.seek(0)
        strict = codecs.getincrementaldecoder("utf-8")("strict")
        utf8_ok = True
        tail = b""
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            out["bytes"] += len(chunk)
            if utf8_ok:
                try:
                    strict.decode(chunk, False)
                except UnicodeDecodeError:
                    utf8_ok = False
            if not out["has_anchor"]:
                if ANCHOR in chunk:
                    out["has_anchor"] = True
                elif tail and ANCHOR in (tail + chunk[: len(ANCHOR) - 1]):
                    # occurrence straddling the chunk boundary
                    out["has_anchor"] = True
            tail = chunk[-(len(ANCHOR) - 1):]
        if not utf8_ok:
            out["kind"] = "non-utf8-text"
    return out


def phase_b(path: Path) -> tuple[dict, list[dict]]:
    """Split into lines and apply the regexes.  Only for anchor-carrying files."""
    counts = {name: [0, 0] for name in VARIANTS}
    matches: list[dict] = []
    truncated_lines = 0
    carry = b""
    lineno = 0
    truncated = False
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = (carry + chunk) if carry else chunk
            pos = 0
            for m in _LINE_BOUNDARY.finditer(buf):
                piece = buf[pos:m.start()]
                if truncated:
                    truncated_lines += 1
                    truncated = False
                lineno += 1
                _handle(piece, lineno, counts, matches)
                pos = m.end()
            carry = buf[pos:]
            if len(carry) > MAX_CARRY:
                carry = carry[-MAX_CARRY:]
                truncated = True
    if carry:
        if truncated:
            truncated_lines += 1
        lineno += 1
        _handle(carry, lineno, counts, matches)
    return {"counts": counts, "truncated_lines": truncated_lines}, matches


def _handle(raw: bytes, lineno: int, counts: dict, matches: list[dict]) -> None:
    if ANCHOR not in raw:
        return
    text = raw.decode("utf-8", "replace")
    for name, rx in VARIANTS.items():
        ms = list(rx.finditer(text))
        if not ms:
            continue
        counts[name][0] += 1
        counts[name][1] += len(ms)
        if name == "v1_dash_any":
            for m in ms:
                matches.append(
                    {
                        "path": None,  # filled by the caller
                        "line": lineno,
                        "matched_text": m.group(0),
                        "full_sentence": sentence_of(text, m.start(), m.end()),
                        "line_text": text.strip(),
                    }
                )


def scan_file(rel: str) -> dict:
    """Scan one file; returns kind, byte size, per-variant counts, matches."""
    path = ROOT / rel
    a = phase_a(path)
    out = {
        "kind": a["kind"],
        "bytes": a["bytes"],
        "variants": {name: [0, 0] for name in VARIANTS},
        "matches": [],
        "truncated_lines": 0,
    }
    if a["kind"] in ("missing", "binary") or not a["has_anchor"]:
        return out
    b, matches = phase_b(path)
    out["variants"] = b["counts"]
    out["truncated_lines"] = b["truncated_lines"]
    for m in matches:
        m["path"] = rel
    out["matches"] = matches
    return out


def new_stats() -> dict:
    return {k: {"files": 0, "lines": 0, "occurrences": 0} for k in VARIANTS}


def main() -> int:
    t0 = time.time()
    groups, files = population()
    pops = ("all", "nonignored", "all_excl_self", "nonignored_excl_self")
    stats = {p: new_stats() for p in pops}
    kinds = {p: {} for p in pops}
    per_file: dict[str, dict] = {}
    occurrences: list[dict] = []
    n_bytes_read = 0

    for i, (rel, group) in enumerate(files):
        r = scan_file(rel)
        n_bytes_read += r["bytes"]
        is_self = rel.startswith(SELF_DIR + "/")
        if any(nl for nl, _ in r["variants"].values()):
            # per_file lists ONLY the files that carry a band match (the
            # per-file record for all 421k candidates would be ~90 MB of JSON
            # and carries no analytic value; the population composition is in
            # `populations[].kinds`).
            per_file[rel] = {
                "group": group,
                "kind": r["kind"],
                "bytes": r["bytes"],
                "truncated_lines": r["truncated_lines"],
            }
            for v, (nl, nocc) in r["variants"].items():
                if nl:
                    per_file[rel].setdefault("variants", {})[v] = {
                        "lines": nl,
                        "occurrences": nocc,
                    }
        membership = {
            "all": True,
            "nonignored": group != "ignored",
            "all_excl_self": not is_self,
            "nonignored_excl_self": group != "ignored" and not is_self,
        }
        for p in pops:
            if not membership[p]:
                continue
            kinds[p][r["kind"]] = kinds[p].get(r["kind"], 0) + 1
            for v, (nl, nocc) in r["variants"].items():
                if nl:
                    stats[p][v]["files"] += 1
                    stats[p][v]["lines"] += nl
                    stats[p][v]["occurrences"] += nocc
        if r["matches"] and not is_self:
            # The occurrence dump EXCLUDES this branch's own directory: the
            # dump is itself a band-carrying file, so including it makes every
            # run larger than the last (a measured 9,012 -> 29,133 line
            # feedback on the "all" count).  The counts above still include the
            # branch's own files -- only the evidence dump is de-self-referenced.
            for m in r["matches"]:
                m["group"] = group
            occurrences.extend(r["matches"])
        if (i + 1) % 50000 == 0:
            print(
                f"  ... {i + 1}/{len(files)} files, {n_bytes_read / 1e9:.1f} GB read "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population_method": (
            "git ls-files --cached | git ls-files --others --exclude-standard | "
            "git ls-files --others --ignored --exclude-standard (three SEPARATE calls)"
        ),
        "population_groups": {k: len(v) for k, v in groups.items()},
        "population_union": len(files),
        "bytes_read": n_bytes_read,
        "populations": {
            p: {
                "candidates": sum(kinds[p].values()),
                "kinds": kinds[p],
                "variant_stats": stats[p],
            }
            for p in pops
        },
        "population_definitions": {
            "all": "tracked + untracked(not ignored) + untracked(IGNORED)",
            "nonignored": (
                "tracked + untracked(not ignored) -- the population the 2026-11-01 "
                "18:57 census used (the one this run corrects)"
            ),
            "all_excl_self": "all, minus " + SELF_DIR + "/ (self-reference removed)",
            "nonignored_excl_self": (
                "nonignored, minus " + SELF_DIR + "/ -- the legacy population with the "
                "branch's own artefacts removed (invariant under this report's own growth)"
            ),
        },
        "per_file": per_file,
        "per_file_note": (
            "only files carrying a band match are listed here; the population "
            "composition of all " + str(len(files)) + " candidates is in populations[].kinds"
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    atomic_write_file(
        OUT / "_census_raw_v2_incl_ignored.json",
        json.dumps(report, indent=2, ensure_ascii=False),
    )
    atomic_write_file(
        OUT / "_occurrences_v1_dash_any_v2_incl_ignored.json",
        json.dumps(occurrences, indent=2, ensure_ascii=False),
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "per_file"}, indent=2, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
