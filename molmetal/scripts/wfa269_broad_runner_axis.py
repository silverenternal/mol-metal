#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors. Licensed under the Apache-2.0 License.
"""Wf-A269 wave-32 — the BROAD runner-coverage axis, made REPRODUCIBLE.

WHY THIS FILE EXISTS (AGENTS.md 6.269.13 "PIN THE AXIS, NOT THE COUNT")
---------------------------------------------------------------------
The catalogue published two BROAD rows (133 / 176 files covered, i.e. 90.24 %
/ 87.09 % uncovered).  The file SET that produced them was never published, so
the counts could not be reproduced by a reader -- an axis that is not pinned.

WHAT THIS SCRIPT FOUND (MEASURED 2026-11-07, CPU-only, 0 GPU)
----------------------------------------------------------------
The published pair is EXACTLY reproducible -- but only under ONE of FOUR
(file-list source x content source) combinations.  It is the MIXED one:

    file list from HEAD  x  content from the WORKING TREE  ->  133 / 176

That combination describes a tree that exists in NO commit (the same class as
6.269.31's census, in a new costume: there the tree was STALE, here it is
SPLIT -- the list is frozen at HEAD while the text searched is uncommitted).
The two COHERENT readings are:

    all-committed      (list=HEAD,   content=HEAD)        ->  113 / 155
    all-working-tree   (list=index,  content=disk)        ->  134 / 177

So the published number is a WORKING-TREE reading whose FILE LIST is frozen
at HEAD.  The missing thing was not the AXIS -- it was the TREE.

THE MIXED-SURFACE FINDING (MEASURED 2026-11-07, main thread, AFTER the first
revision of this script) -- READ THIS BEFORE QUOTING THE MATRIX
--------------------------------------------------------------------------
The published row also carries a SURFACE COUNT:

    "**966** tracked script/CI/Makefile files"  (docs/HONEST_NEGATIVE_CATALOG.md)

**966 AND THE COVERED PAIR (133 / 176) COME FROM DIFFERENT SURFACE
DEFINITIONS.**  Measured:

  * `SURFACE_DEFS["four_path"]` = scripts/ .github/ molmetal/scripts/ Makefile
        HEAD = 965, index = 972   ->  BROAD-1 133, BROAD-2 176   <== the pair
  * `SURFACE_DEFS["five_path"]` = the same PLUS `pyproject.toml`
        HEAD = 966, index = 973   ->  BROAD-1 133, BROAD-2 177

So the surface SIZE the row names (966) is the FIVE-path surface, while the
COVERED SET it reports (133 / 176) was measured over the FOUR-path surface.
Under the surface the row itself names, BROAD-2 is **177**, not 176.

THE DAMAGE IS BOUNDED TO *ONE* PUBLISHED NUMBER (MEASURED 2026-11-07, same
script -- READ THIS BEFORE QUOTING THE DEFECT, IT IS WHAT MAKES THE FINDING
HONEST INSTEAD OF ALARMIST):
    BROAD-1's covered count is **133 under BOTH definitions** (and 134 under
    both at list=INDEX content=DISK).  So the surface mixing moves exactly ONE
    published number -- BROAD-2 (176 -> 177) -- and the published "90.24 %"
    row is **NOT** contaminated.
    The mechanism is MECHANICAL, not lucky: the ONE separating mention
    (pyproject.toml:154) names the file by BARE BASENAME only
    ("test_reproduce_R17.py", no "tests/" prefix), while BROAD-1 is defined as
    requiring the PATH form ("...tests/<name>.py").  So the extra file can
    only ever enter BROAD-2, never BROAD-1.  Had that mention carried a
    "tests/" prefix, BROAD-1 would have moved too.
    => An UNBOUNDED defect claim ("both published rows are contaminated")
    would be a SECOND defect.  Publish the bound.

**THE ONE FILE THAT SEPARATES THEM IS `pyproject.toml`**, and the single
population file it names is `test_reproduce_R17.py` -- in a **COMMENT** line.
That is the section 5.9 INVOCATION-vs-MENTION lesson again: under BROAD-2's
own definition ("naming the bare basename **anywhere**") a comment counts, so
the two surfaces genuinely differ; under a BY-INVOCATION reading both give 176.
The honest statement is therefore:

    the whole difference between "176" and "177" rests on ONE mention in ONE
    comment line in ONE file -- and the row's SURFACE SIZE and its COVERED SET
    are taken over different surfaces.

=> NEW DEFECT CLASS: **"A PUBLISHED ROW'S POPULATION SIZE AND ITS COVERED SET
MUST COME FROM THE SAME SURFACE"** -- section 6.269.35.2's CONFLATED-AXIS class,
applied to the SURFACE DEFINITION rather than to the file-list-vs-content tree.

--------------------------------------------------------------------------
CORRECTION (2026-11-07, main thread) -- THE MIXED-SURFACE FINDING ABOVE IS
**REFUTED**.  THE PARAGRAPHS ABOVE ARE PRESERVED VERBATIM PER 5.5 RULE 6.
--------------------------------------------------------------------------
The finding above was a BOOLEAN VERDICT COMPUTED OVER A DECLARED CANDIDATE
SET: `MIXED_SURFACE` was decided over the TWO definitions in `SURFACE_DEFS`
(`four_path`, `five_path`), and it came out true because the published
surface COUNT (966) is the size of neither of them under the cell that
reproduces the published covered PAIR (133 / 176).

MEASURED (2026-11-07, this script, `witness_search()`): the published row
names a **966**-file surface while `four_path` at HEAD is **965**, so the
966-file surface is `four_path` + ONE tracked file outside it -- and the
family of such candidates is COMPUTED, not chosen.  A member of that family
reproduces BOTH the published count AND the published covered pair:
`four_path + .pre-commit-config.yaml` (and `+ molmetal/conftest.py`,
`+ main.py`, `+ molmetal/__main__.py`, `+ .gitignore`) give size **966** with
covered **(133, 176)** at file-list=HEAD x content=DISK.

=> **MIXED_SURFACE = false** over the widened space.  The published count and
the published covered pair ARE reproduced by a single definition.

=> **⇒ TWO DEFECTS SURVIVE, AND THEY ARE NOT THE ONE PUBLISHED ABOVE:**

  (S1) **THE SPLIT TREE** (unchanged by this correction).  The published pair
       (133 / 176) is reachable ONLY at file-list=HEAD x content=DISK -- a
       tree that exists in NO commit.  The two COHERENT readings are
       (113 / 155) all-committed and (134 / 177) all-working-tree.

  (S2) **THE SURFACE IS UNDER-DETERMINED** (NEW, and it is the honest
       replacement for the refuted finding).  The family's members do NOT
       all give the same covered pair: some are coverage-neutral (133 / 176)
       and some are not (`four_path + pyproject.toml` gives 133 / **177**).
       ⇒ the published row's SIZE does not determine its COVERED SET ⇒ a
       reader cannot re-derive "176" from "966 tracked script/CI/Makefile
       files" alone, because the family's members DISAGREE on BROAD-2.

=> **⇒ NEW DEFECT CLASS (2026-11-07): "A BOOLEAN VERDICT COMPUTED OVER A
DECLARED CANDIDATE SET IS NOT A VERDICT ABOUT THE WORLD."**  The declared
candidate set is part of the verdict; a witness OUTSIDE it refutes the
verdict, and the candidate set must therefore be MECHANICALLY DERIVED and
its SCOPE DISCLOSED.

THE AXIS, stated exactly (this is the part that was missing):
  population   = the files matching molmetal/tests/test_*.py ON DISK
  runner surface = the tracked set of runner files -- TWO definitions, both
        reported, because the published row mixes them:
          four_path = git ls-tree -r HEAD --name-only -- scripts/ .github/
                      molmetal/scripts/ Makefile
          five_path = the same + pyproject.toml
        (and the index variant of each, git ls-files -- <same paths>)
  BROAD-1 = population files whose PATH form (tests/<name>.py) is NAMED
            somewhere in the surface's text
  BROAD-2 = population files whose BARE BASENAME is named somewhere in the
            surface's text (regardless of the directory prefix)

Nothing here is GPU-bound and nothing is imported from the training stack.
Run it with the repo's own venv, CPU-only:
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wfa269_broad_runner_axis.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POPULATION_DIR = "molmetal/tests"
POPULATION_GLOB = "test_*.py"

# TWO surface definitions.  The published catalogue row's SURFACE COUNT comes
# from five_path (966 at HEAD); its COVERED SET comes from four_path.  Both are
# reported so a reader can see the mixing instead of inheriting it.
SURFACE_DEFS: dict[str, tuple[str, ...]] = {
    "four_path": ("scripts/", ".github/", "molmetal/scripts/", "Makefile"),
    "five_path": ("scripts/", ".github/", "molmetal/scripts/", "Makefile",
                  "pyproject.toml"),
}
# The default definition used for the top-level `grid` key (the one the
# wave brief quotes).  Kept as four_path so the published brief's matrix and
# sha256s stay valid.
DEFAULT_SURFACE = "four_path"

# The surface count the published catalogue row names ("966 tracked
# script/CI/Makefile files").  Used by the history scan.
PUBLISHED_SURFACE_COUNT = 966

# A population file is "named" by a surface file if the surface text contains
# the path suffix "tests/<name>.py" (BROAD-1) or the bare "<name>.py" (BROAD-2).
_NAME_RE = re.compile(r"([A-Za-z0-9_./-]*tests/)?(test_[A-Za-z0-9_]+\.py)")

# A "mention" is INVOCATION-shaped only if the line is not a comment.  The
# markers are the ones this repo's runner surfaces actually use.
_COMMENT_MARKERS = ("#", "//", "--", "<!--", ";")


def _git(*args: str) -> str:
    """Run git in REPO and return stdout as text.

    `core.quotePath=false` is FORCED: git C-QUOTES every path that is not
    pure ASCII (`"\347\247\221..."`), and a quoted path CANNOT BE READ
    BACK, so a listing that keeps the quoting turns every non-ASCII path
    into a file that reads as EMPTY (see `_git_raw_paths`).
    """
    out = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotePath=false", *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def _git_raw_paths(*args: str) -> list[str]:
    """The RAW (never C-quoted) paths of a git listing.

    `git ls-tree` / `git ls-files` C-QUOTE every path that is not pure
    ASCII: `core.quotePath` defaults to true, so a Chinese path comes back
    as "\347\247\221...".  A quoted path CANNOT BE READ BACK --
    `git show HEAD:"\347..."` fails with "path does not exist in 'HEAD'",
    `cat-file --batch` reports it missing, and the quoted string is not a
    file on disk -- so a listing that keeps the quoting turns every non-ASCII
    path into a file that reads as EMPTY.  Such a path can be a PHANTOM
    witness: it appears in the candidate list, contributes nothing, and is
    still counted as evidence.

    `-z` turns the quoting off and NUL-separates the paths.
    """
    out = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotePath=false", *args],
        capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode("utf-8", "surrogateescape").split("\x00") if p]


def population() -> list[str]:
    """The on-disk population, as basenames, sorted."""
    d = REPO / POPULATION_DIR
    return sorted(p.name for p in d.glob(POPULATION_GLOB))


def surface(mode: str, defn: str = DEFAULT_SURFACE) -> list[str]:
    """The runner surface's FILE LIST.

    mode='head'  -> files tracked at HEAD (the committed surface)
    mode='index' -> files in the git index (the staged surface)
    mode='disk'  -> files ACTUALLY ON DISK under the surface entries (the
                     true working-tree surface; it includes the UNTRACKED
                     files, which the index list does NOT)

    THE THREE TREES ARE DISTINCT AND THE DIFFERENCE IS THE POINT (MEASURED
    2026-11-07): four_path is 965 files at HEAD, 972 in the index and 1050
    on disk.  A reader who says "the working tree" without saying WHICH
    listing means something different each time.
    """
    paths = SURFACE_DEFS[defn]
    if mode == "head":
        return sorted(_git_raw_paths(
            "ls-tree", "-r", "HEAD", "--name-only", "-z", "--", *paths))
    if mode == "index":
        return sorted(_git_raw_paths("ls-files", "-z", "--", *paths))
    if mode == "disk":
        return _disk_surface(paths)
    raise ValueError(mode)


# The size guard for the working-tree listing.  It is DISCLOSED, never
# silent: the number of skipped files is reported with the listing.
MAX_SURFACE_FILE_BYTES = 3 * 1024 * 1024
_DISK_SKIP: dict[str, int] = {}


def _disk_surface(paths: tuple[str, ...]) -> list[str]:
    """The files physically on disk under `paths` (bounded, size-guarded).

    Bounded by construction: only the surface ENTRIES are walked, never the
    repository root (the repository is ~201 GB; a recursive walk of it is
    forbidden by the project's own memory rule).
    """
    import os

    out: list[str] = []
    skipped = 0
    for entry in paths:
        p = REPO / entry
        if p.is_file():
            out.append(entry)
            continue
        if not p.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "__pycache__", ".pytest_cache")]
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    if fp.stat().st_size > MAX_SURFACE_FILE_BYTES:
                        skipped += 1
                        continue
                except OSError:
                    skipped += 1
                    continue
                out.append(str(fp.relative_to(REPO)))
    _DISK_SKIP[str(paths)] = skipped
    return sorted(set(out))


# The content cache.  A single measurement must read ONE consistent view
# of the tree: without it, a file edited between two reads would make the
# grid internally inconsistent (the mixed-tree class the catalogue entry is
# about -- here it would be the MEASUREMENT's own defect).
_CONTENT_CACHE: dict[tuple[str, str], str] = {}


def _batch_blobs(specs: list[str]) -> dict[str, str]:
    """Read many blobs with ONE `git cat-file --batch` process.

    `specs` are `git cat-file --batch` revision specs: "HEAD:<path>" for the
    committed blob and ":<path>" for the staged blob.  Missing objects are
    returned as "" (the caller treats an absent blob as an empty file, which
    is what `git show` did).
    """
    if not specs:
        return {}
    proc = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        capture_output=True)
    out = proc.stdout
    got: dict[str, str] = {}
    i = 0
    n = len(out)
    cursor = 0
    while i < n and cursor < len(specs):
        nl = out.find(b"\n", i)
        if nl < 0:
            break
        header = out[i:nl].decode("utf-8", "replace").split()
        i = nl + 1
        # `git cat-file --batch` emits EXACTLY ONE header per input spec, in
        # order -- including for a spec that does not exist, whose header is
        # "<spec> missing" (two fields).  The spec cursor MUST advance on
        # every header: indexing by `len(got)` skips the missing ones and
        # hands every later object to the WRONG spec.
        spec = specs[cursor]
        cursor += 1
        if len(header) < 3:
            got[spec] = ""                     # absent object -> empty file
            continue
        try:
            size = int(header[2])
        except ValueError:
            got[spec] = ""
            continue
        got[spec] = out[i:i + size].decode("utf-8", "replace")
        i += size + 1                          # skip the trailing newline
    return got


def _prefetch_content(content: str, paths: list[str]) -> None:
    """Populate the cache for `content` over `paths` (batched where possible)."""
    missing = [p for p in paths if (content, p) not in _CONTENT_CACHE]
    if not missing:
        return
    if content in ("head", "index"):
        spec_prefix = "HEAD:" if content == "head" else ":"
        got = _batch_blobs([spec_prefix + p for p in missing])
        for p in missing:
            _CONTENT_CACHE[(content, p)] = got.get(spec_prefix + p, "")
        return
    for p in missing:
        _CONTENT_CACHE[(content, p)] = _read_disk(p)


def _read_disk(path: str) -> str:
    p = REPO / path
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def surface_text(path: str, content: str) -> str:
    """The content of a surface file.

    content='head'  -> the COMMITTED blob (what a reader of a committed
                        number would have had)
    content='index' -> the STAGED blob (what `git stash`/CI would see)
    content='disk'  -> the file on disk
    """
    if content not in ("head", "index", "disk"):
        raise ValueError(content)
    key = (content, path)
    if key in _CONTENT_CACHE:
        return _CONTENT_CACHE[key]
    if content == "head":
        try:
            val = _git("show", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            val = ""                          # staged-new file: no HEAD blob
    elif content == "index":
        val = _batch_blobs([f":{path}"]).get(f":{path}", "")
    else:
        val = _read_disk(path)
    _CONTENT_CACHE[key] = val
    return val


def cover(surf: list[str], content: str, pop: list[str]) -> tuple[set[str], set[str]]:
    """(BROAD-1 covered, BROAD-2 covered) as sets of population basenames."""
    popset = set(pop)
    by_path: set[str] = set()
    by_base: set[str] = set()
    _prefetch_content(content, list(surf))
    for path in surf:
        text = surface_text(path, content)
        if not text:
            continue
        for m in _NAME_RE.finditer(text):
            base = m.group(2)
            if base not in popset:
                continue
            by_base.add(base)
            if m.group(1):                  # a "...tests/<name>.py" span
                by_path.add(base)
    return by_path, by_base


def mentions(path: str, content: str) -> list[dict]:
    """Every population basename this surface file names, with the mentioning
    line and whether that line is a COMMENT.

    This is what separates "the surface NAMES the file" (BROAD-2's own
    definition) from "the surface INVOKES the file" (the 5.9 reading).
    """
    popset = set(population())
    text = surface_text(path, content)
    out: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        for m in _NAME_RE.finditer(raw):
            base = m.group(2)
            if base not in popset:
                continue
            stripped = raw.strip()
            is_comment = stripped.startswith(_COMMENT_MARKERS)
            out.append({
                "test_file": base,
                "line": lineno,
                "is_comment_line": is_comment,
                "line_text": stripped[:160],
            })
    return out


def _sha(lines: list[str]) -> str:
    h = hashlib.sha256()
    for p in lines:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def measure(list_mode: str, content_mode: str, pop: list[str],
            defn: str = DEFAULT_SURFACE) -> dict:
    surf = surface(list_mode, defn)
    b1, b2 = cover(surf, content_mode, pop)
    n = len(pop)
    return {
        "surface_definition": defn,
        "file_list_source": list_mode,
        "content_source": content_mode,
        "surface_n_files": len(surf),
        "broad1_covered": len(b1),
        "broad1_uncovered": n - len(b1),
        "broad1_uncovered_pct": round(100.0 * (n - len(b1)) / n, 2) if n else None,
        "broad2_covered": len(b2),
        "broad2_uncovered": n - len(b2),
        "broad2_uncovered_pct": round(100.0 * (n - len(b2)) / n, 2) if n else None,
        "_broad1_files": sorted(b1),
        "_broad2_files": sorted(b2),
    }


# THE THREE TREES THE REPOSITORY ACTUALLY HAS.  A measurement that says
# "the tree" without naming one of these is not reproducible.
LIST_MODES = ("head", "index", "disk")
CONTENT_MODES = ("head", "index", "disk")


def grid_for(defn: str, pop: list[str]) -> list[dict]:
    """The 3x3 (file-list source x content source) grid for one definition.

    REPAIRED 2026-11-07: this was a 2x2 grid.  The adversarial refuter
    measured a third file-list source -- the TRUE working tree -- and found
    that the published pair is NOT reproduced there either, so the grid that
    decides the row MUST contain it.
    """
    return [
        measure(list_mode, content_mode, pop, defn)
        for list_mode in LIST_MODES
        for content_mode in CONTENT_MODES
    ]


def history_scan(defn: str) -> dict:
    """The COMMITTED history of one surface definition's SIZE.

    Answers: "has the published surface count EVER existed as a committed
    surface size?" -- the question the catalogue row leaves open.  The
    published row says "966 tracked script/CI/Makefile files" without saying
    WHICH definition of "tracked ... files"; if 966 is reachable under SOME
    definition, the count IS recoverable and the defect is the DEFINITION
    (the row's covered set comes from a different one), not the count.

    Reachable commits only: `git rev-list --all`.  A dangling, unreferenced
    object is invisible to this scan -- the same honest bound as
    AGENTS.md 6.269.35.4's "commits reachable from a ref".
    """
    paths = SURFACE_DEFS[defn]
    shas = _git("rev-list", "--all").split()
    sizes: dict[str, list[str]] = {}
    for sha in shas:
        raw = _git_raw_paths("ls-tree", "-r", sha, "--name-only", "-z", "--", *paths)
        n = len(raw)
        sizes.setdefault(str(n), []).append(sha)
    hist = {k: len(v) for k, v in sorted(sizes.items(), key=lambda kv: int(kv[0]))}
    mx = max(int(k) for k in sizes)
    at_max = sizes[str(mx)]
    dates = sorted({_git("show", "-s", "--format=%cI", s).strip() for s in at_max})
    return {
        "surface_definition": defn,
        "n_commits_scanned": len(shas),
        "size_histogram": hist,
        "min_size": min(int(k) for k in sizes),
        "max_size": mx,
        "n_commits_at_max": len(at_max),
        "commits_at_max": at_max[:8],
        "dates_at_max": dates,
        "published_size_present_in_history": PUBLISHED_SURFACE_COUNT in sizes,
        "published_size_n_commits": len(sizes.get(str(PUBLISHED_SURFACE_COUNT), [])),
    }



# ---------------------------------------------------------------------------
# THE WIDENED DEFINITION SPACE (REPAIRED 2026-11-07)
# ---------------------------------------------------------------------------
# A BOOLEAN VERDICT COMPUTED OVER A DECLARED CANDIDATE SET IS NOT A VERDICT
# ABOUT THE WORLD.
#
# `MIXED_SURFACE` used to be decided over the TWO declared surface
# definitions in `SURFACE_DEFS`, so the question it actually answered was
# "does any member of {four_path, five_path} reproduce BOTH the published
# count AND the published covered pair?".  The honest question is the
# universal one: "does ANY single definition reproduce both?".
#
# The candidate space searched here is MECHANICALLY DERIVED, not hand-picked:
# the published row names a 966-file surface while `four_path` at HEAD is
# 965, so every tracked file outside the base gives one 966-file candidate.
# The family is COMPUTED, so a witness cannot hide by being unlisted.
#
# The family is also NOT COVERED-EQUIVALENT (measured 2026-11-07): some
# members change BROAD-2 and some do not.  So the family answers TWO
# questions at once -- whether a coherent single definition exists (it does)
# and whether the published SIZE pins the covered SET (it does not).


def _mentioning_tracked_files(rev: str | None) -> tuple[set[str], bool]:
    """The tracked files whose CONTENT names a population file.

    rev='HEAD'     -> the COMMITTED blob (`git grep ... HEAD`)
    rev='--cached' -> the INDEX (`git grep --cached`)
    rev=None       -> the WORKING TREE (`git grep` with no revision)

    Returns (set, ok).  `ok` is False when the underlying `git grep` failed:
    a failed search must NOT be read as "no witness exists" (that would be a
    FALSE refutation), so the caller reports the widened verdict as NOT
    AVAILABLE and falls back to the declared-grid verdict.

    The pattern is the CORE of `_NAME_RE` (its optional "...tests/" prefix
    group does not change whether a match exists), so this is exactly the
    set of files `cover()` could ever attribute a population file to.

    `-z` keeps the paths RAW: `git grep` C-QUOTES a non-ASCII path by
    default, and a quoted path cannot be read back (see `_git_raw_paths`).
    """
    args = ["grep", "-l", "-z", "-a", "-E", r"test_[A-Za-z0-9_]+\.py"]
    prefix = ""
    if rev == "--cached":
        args.insert(1, "--cached")
    elif rev is not None:
        args.append(rev)
        prefix = f"{rev}:"
    r = subprocess.run(["git", "-C", str(REPO), "-c", "core.quotePath=false", *args],
                       capture_output=True)
    if r.returncode not in (0, 1):
        return set(), False
    out: set[str] = set()
    for ln in r.stdout.decode("utf-8", "surrogateescape").split("\x00"):
        if not ln.strip():
            continue
        if prefix and ln.startswith(prefix):
            ln = ln[len(prefix):]
        out.add(ln)
    return out, True


def witness_search(base_defn: str = DEFAULT_SURFACE,
                   published_surface_count: int = PUBLISHED_SURFACE_COUNT,
                   published_b1: int = 133,
                   published_b2: int = 176,
                   pop: list[str] | None = None) -> dict:
    """Search the ONE-FILE-EXTENSION FAMILY of the base surface for a single
    definition that reproduces BOTH the published surface count AND the
    published covered pair -- and report whether the family's members AGREE.

    The family is DERIVED, not chosen: the published row names a 966-file
    surface and `four_path` at HEAD is 965, so every tracked file outside the
    base contributes exactly one 966-file candidate.

    Coverage is computed EXACTLY with the union identity

        cover(A u {X}) = cover(A) u cover([X])

    so the search costs one file read per candidate THAT MENTIONS a
    population file (found with one `git grep` pass), not one read per
    candidate per cell.
    """
    pop = population() if pop is None else pop
    tracked = {
        "head": set(_git_raw_paths("ls-tree", "-r", "HEAD", "--name-only", "-z")),
        "index": set(_git_raw_paths("ls-files", "-z")),
    }
    # THE CANDIDATE FAMILY IS "TRACKED FILES" -- the published row's own
    # word ("966 TRACKED script/CI/Makefile files").  The disk file list is
    # a LIST SOURCE, not a candidate source: the row never claimed an
    # untracked file.
    tracked_all = tracked["head"] | tracked["index"]
    mentions: dict[str, set[str]] = {}
    search_ok = True
    for cm, rev in (("head", "HEAD"), ("index", "--cached"), ("disk", None)):
        s, ok = _mentioning_tracked_files(rev)
        mentions[cm] = s
        search_ok = search_ok and ok

    witnesses: list[dict] = []
    cells: list[dict] = []
    members_disagree = False
    # Which ADDED FILE reproduces the published pair in each cell?  Kept as
    # sets so the intersection across cells can be computed -- that
    # intersection is the IRREDUCIBILITY measurement.
    pair_members_by_cell: dict[str, set[str]] = {}
    for list_mode in LIST_MODES:
        base_surf = surface(list_mode, base_defn)
        base_set = set(base_surf)
        cands = sorted(tracked_all - base_set)
        for content_mode in CONTENT_MODES:
            b1, b2 = cover(base_surf, content_mode, pop)
            neutral = 0
            changing = 0
            dist: dict[str, int] = {}
            pair_members: set[str] = set()
            null_contributors = 0
            cell_name = (f"{base_defn}+<one tracked file>:"
                         f"{list_mode}-list/{content_mode}-content")
            for X in cands:
                if X in mentions[content_mode]:
                    d1, d2 = cover([X], content_mode, pop)
                else:
                    d1, d2 = set(), set()
                    # A NULL CONTRIBUTOR: the file names ZERO population
                    # basenames, so adding it moves the SURFACE SIZE and
                    # NOTHING ELSE.  These are the dials of the published
                    # denominator.
                    null_contributors += 1
                n1, n2 = len(b1 | d1), len(b2 | d2)
                key = f"{n1}/{n2}"
                dist[key] = dist.get(key, 0) + 1
                if n1 == len(b1) and n2 == len(b2):
                    neutral += 1
                else:
                    changing += 1
                size = len(base_set) + 1
                if (n1, n2) == (published_b1, published_b2):
                    pair_members.add(X)
                if size == published_surface_count and (n1, n2) == (published_b1, published_b2):
                    witnesses.append({
                        "definition": f"{base_defn}+{X}",
                        "added_file": X,
                        "file_list_source": list_mode,
                        "content_source": content_mode,
                        "surface_n_files": size,
                        "broad1_covered": n1,
                        "broad2_covered": n2,
                    })
            if len(dist) > 1:
                members_disagree = True
            pair_members_by_cell[cell_name] = pair_members
            cells.append({
                "cell": cell_name,
                "n_candidates": len(cands),
                "n_coverage_neutral": neutral,
                "n_coverage_changing": changing,
                "distinct_covered_pairs": dict(sorted(dist.items())),
                "base_broad1_covered": len(b1),
                "base_broad2_covered": len(b2),
                "published_pair_present": len(pair_members) > 0,
                "published_pair_n_members": len(pair_members),
                "null_contributors": null_contributors,
                "base_surface_n_files": len(base_surf),
                "base_reproduces_published_pair": (
                    len(b1) == published_b1 and len(b2) == published_b2),
            })

    n_witness = len(witnesses)
    sample = witnesses[:25]

    # THE IRREDUCIBILITY MEASUREMENT.  The published PAIR is reproducible in
    # some cells; is it reproducible in EVERY cell by ONE AND THE SAME added
    # file?  If not, the split tree is IRREDUCIBLE: no choice of surface can
    # make the published number tree-independent.
    cells_with_pair = [c["cell"] for c in cells if c["published_pair_present"]]
    if cells_with_pair:
        inter: set[str] = set.intersection(
            *[pair_members_by_cell[c] for c in cells_with_pair])
    else:
        inter = set()
    reducible = len(cells_with_pair) == len(cells) and bool(inter)

    return {
        "search_valid": search_ok,
        "published_pair_reproducible_cells": cells_with_pair,
        "published_pair_reproducible_in_every_cell_by_one_member": reducible,
        "published_pair_members_in_all_cells": sorted(inter)[:25],
        "published_pair_members_in_all_cells_n": len(inter),
        "split_tree_irreducible": (not reducible) and bool(cells_with_pair),
        "family": f"one-file-extension of {base_defn}",
        "derivation": (
            "the published row names a 966-file surface and four_path at HEAD "
            "is 965, so every tracked file outside the base yields exactly one "
            "966-file candidate; the family is COMPUTED from the tracked file "
            "set, not hand-picked -- a witness therefore cannot hide by being "
            "unlisted"
        ),
        "scope": (
            "this family covers surfaces of the form four_path u {X}. A "
            "966-file surface that is NOT of this form (a different base) is "
            "NOT searched: a witness OUTSIDE this family would refute the "
            "verdict, so the verdict is SCOPED, not absolute"
        ),
        "n_candidates": {
            lm: len(tracked_all - set(surface(lm, base_defn)))
            for lm in LIST_MODES
        },
        "candidate_family": (
            "every TRACKED file outside the base surface (the published row's "
            "own word is 'tracked'); the family is COMPUTED, so a witness "
            "cannot hide by being unlisted.  UNTRACKED files are OUT of the "
            "family and this is a SCOPE LIMIT, disclosed"
        ),
        "null_contributor_count": {
            c["cell"]: c["null_contributors"] for c in cells
        },
        "cells": cells,
        "members_disagree": members_disagree,
        "witness_found": n_witness > 0,
        "witness_definitions_n": n_witness,
        "witness_definitions": sample,
        "witness_definitions_is_a_sample": n_witness > len(sample),
        "witness_definitions_sample_size": len(sample),
        "witness_definitions_note": (
            "the witness list is a SAMPLE of the members that reproduce the "
            "published count AND pair; the FULL list is reproducible with the "
            "same command -- the sample is disclosed, not silent (the "
            "published-number discipline: no silent caps)"
        ),
    }


def adjudicate_published_row(grids: dict[str, list[dict]], pop: list[str],
                             published_surface_count: int = 966,
                             published_b1: int = 133,
                             published_b2: int = 176,
                             published_population: int = 1363,
                             *,
                             search_witnesses: bool = True,
                             witness_space: dict | None = None) -> dict:
    """Does the published row's SURFACE SIZE and its COVERED SET come from the
    SAME surface definition?  Report it; do not adjudicate it.

    `search_witnesses=True` (default) widens the question from the DECLARED
    candidate set to the DERIVED one-file-extension family of the base
    surface (see `witness_search`).  `search_witnesses=False` keeps the
    declared-grid-only verdict -- used by the synthetic tests, which must not
    depend on the moving working tree.  `witness_space` injects a precomputed
    family report (for tests).
    """
    n = len(pop)
    count_repro: list[str] = []
    pair_repro: list[str] = []
    for defn, grid in grids.items():
        for g in grid:
            cell = (f"{defn}:{g['file_list_source']}-list/{g['content_source']}-content")
            if g["surface_n_files"] == published_surface_count:
                count_repro.append(cell)
            if (g["broad1_covered"], g["broad2_covered"]) == (published_b1, published_b2):
                pair_repro.append(cell)

    # What would the covered set be under the surface the row NAMES?
    # TWO denominators are reported: the LIVE population (n) and the
    # POPULATION THE PUBLISHED ROW USED (published_population).  The published
    # row's own arithmetic (176 + 1187 = 1363) is internally consistent for
    # 1363 -- the defect is the SURFACE, not the addition.
    consequence = []
    for defn, grid in grids.items():
        for g in grid:
            if g["surface_n_files"] != published_surface_count:
                continue
            u1 = n - g["broad1_covered"]
            u2 = n - g["broad2_covered"]
            pu2 = published_population - g["broad2_covered"]
            consequence.append({
                "cell": f"{defn}:{g['file_list_source']}-list/{g['content_source']}-content",
                "broad1_covered": g["broad1_covered"],
                "broad2_covered": g["broad2_covered"],
                "over_live_population": {
                    "n": n,
                    "broad2_uncovered": u2,
                    "broad2_uncovered_pct": round(100.0 * u2 / n, 2),
                },
                "over_published_population": {
                    "n": published_population,
                    "broad2_uncovered": pu2,
                    "broad2_uncovered_pct": round(100.0 * pu2 / published_population, 2),
                },
            })

    # Which files separate the definitions, and what do they mention?
    #
    # THE MENTIONS ARE REPORTED FOR *BOTH* CONTENT SOURCES ON PURPOSE
    # (REPAIRED 2026-11-07, after a mutation proof exposed the defect).
    # The covered COUNTS above are measured over DISK content -- the cell
    # that reproduces the published pair is head-list/DISK-content -- so the
    # MECHANISM that explains them has to be read over DISK content too.
    #
    # This function used to report ONLY the HEAD blob's mentions
    # (`mentions(p, "head")`).  That is the SAME mixed-tree class the
    # catalogue entry is about, living INSIDE the adjudication itself: the
    # explanation was computed over a DIFFERENT tree than the consequence
    # it explains, and a working-tree-only edit was invisible to it.  The
    # mutation proof that caught it appended a PATH-form mention to
    # pyproject.toml ON DISK; the head-read guard stayed GREEN.
    #
    # The file LIST of the separating set is taken at HEAD, which is
    # consistent with the published cell (head-list).  It is labelled so a
    # reader can see which axis each half of this record was measured on.
    sep: list[dict] = []
    base = set(surface("head", DEFAULT_SURFACE))
    for defn in SURFACE_DEFS:
        extra = sorted(set(surface("head", defn)) - base)
        if not extra:
            continue
        for p in extra:
            sep.append({
                "surface_definition_adding_it": defn,
                "path": p,
                "file_list_source": "head",
                "mentions_by_content_source": {
                    "head": mentions(p, "head"),
                    "disk": mentions(p, "disk"),
                },
            })

    mixed_over_declared = (not (set(count_repro) & set(pair_repro))
                           and bool(count_repro) and bool(pair_repro))

    # THE WIDENED VERDICT (REPAIRED 2026-11-07).  The declared-grid verdict
    # answers "does any of the TWO DECLARED definitions reproduce both?"; the
    # honest question is the universal one.  A BOOLEAN VERDICT COMPUTED OVER A
    # DECLARED CANDIDATE SET IS NOT A VERDICT ABOUT THE WORLD.
    if witness_space is not None:
        witness = witness_space
    elif search_witnesses:
        witness = witness_search(base_defn=DEFAULT_SURFACE,
                                 published_surface_count=published_surface_count,
                                 published_b1=published_b1,
                                 published_b2=published_b2,
                                 pop=pop)
    else:
        witness = None
    if witness is None:
        mixed = mixed_over_declared
        verdict_basis = ("DECLARED GRID ONLY (the widened search was NOT run; "
                         "this is a verdict over the declared candidate set, "
                         "NOT about the world)")
    elif not witness.get("search_valid", True):
        mixed = mixed_over_declared
        verdict_basis = ("DECLARED GRID ONLY (the widened search FAILED - "
                         "fail-closed: a failed search must not be read as "
                         "'no witness exists', which would be a FALSE "
                         "refutation)")
    else:
        mixed = mixed_over_declared and not witness["witness_found"]
        verdict_basis = ("DECLARED GRID + THE DERIVED ONE-FILE-EXTENSION FAMILY "
                         "OF THE BASE SURFACE (see "
                         "witness_family.derivation and witness_family.scope)")

    # THE FREE-PARAMETER READING (MEASURED 2026-11-07).
    # The published row's SURFACE SIZE (966) and its COVERED SET (133/176)
    # are two different claims about one surface.  A NULL CONTRIBUTOR is a
    # tracked file that names ZERO population basenames: adding it moves the
    # SIZE and NOTHING ELSE.  If the base surface ALREADY reproduces the
    # published pair, then base u {null contributor} is a 966-file surface
    # that reproduces the published SIZE AND the published PAIR while moving
    # no coverage number at all -- the published DENOMINATOR is DIALABLE.
    #
    # PLACEMENT: this block runs AFTER the verdict, because it reads
    # `witness`; before the repair it sat BEFORE the verdict (a NameError
    # waiting to happen) and its loop variable shadowed the outer
    # `n = len(pop)` (the LIVE POPULATION the materiality block divides by).
    base_grid = grids.get(DEFAULT_SURFACE, [])
    base_pair_cells = [
        f"{DEFAULT_SURFACE}:{g['file_list_source']}-list/"
        f"{g['content_source']}-content"
        for g in base_grid
        if (g["broad1_covered"], g["broad2_covered"]) == (published_b1, published_b2)
    ]
    base_pair_unique = len(base_pair_cells) == 1
    free_parameter = False
    null_counts: dict[str, int] = {}
    if witness is not None and witness.get("search_valid", True):
        null_counts = dict(witness.get("null_contributor_count") or {})
        for cell, n_null in null_counts.items():
            for c in witness.get("cells", []):
                if (c.get("cell") == cell
                        and c.get("base_reproduces_published_pair")
                        and n_null > 0
                        and (c.get("base_surface_n_files", -1) + 1)
                        == published_surface_count):
                    free_parameter = True

    # THE SURFACE-SIZE-DOES-NOT-DETERMINE-THE-COVERED-SET MEASUREMENT.
    # For each cell, the family's members span MANY distinct covered pairs --
    # so a reader who knows only the published SIZE cannot know the published
    # covered SET.  This is the DEFECT; "the numbers are indistinguishable"
    # is NOT (see the materiality note below, CORRECTED 2026-11-07).
    under_detail: list[dict] = []
    if witness is not None and witness.get("search_valid", True):
        for c in witness.get("cells", []):
            dist = c.get("distinct_covered_pairs") or {}
            if not dist:
                continue
            b1v = [int(k.split("/")[0]) for k in dist]
            b2v = [int(k.split("/")[1]) for k in dist]
            under_detail.append({
                "cell": c.get("cell"),
                "n_candidates": c.get("n_candidates"),
                "n_distinct_covered_pairs": len(dist),
                "broad1_covered_min": min(b1v),
                "broad1_covered_max": max(b1v),
                "broad2_covered_min": min(b2v),
                "broad2_covered_max": max(b2v),
                "base_reproduces_published_pair": c.get(
                    "base_reproduces_published_pair"),
                "broad2_uncovered_pct_min": (
                    round(100.0 * (n - max(b2v)) / n, 4) if n else None),
                "broad2_uncovered_pct_max": (
                    round(100.0 * (n - min(b2v)) / n, 4) if n else None),
            })

    # THE MATERIALITY READING (MEASURED 2026-11-07, NOTE CORRECTED 2026-11-07).
    # A defect that moves a published percentage by less than the
    # population's own drift does not change what the row MEANS -- but it
    # also means the row cannot be defended by its precision.  Both numbers
    # are reported; neither is hidden.
    pub_b1_pct = (round(100.0 * (published_population - published_b1)
                        / published_population, 4) if published_population else None)
    pub_b2_pct = (round(100.0 * (published_population - published_b2)
                        / published_population, 4) if published_population else None)
    live_b2_pct = (round(100.0 * (n - published_b2) / n, 4) if n else None)
    alt: list[dict] = []
    for defn, grid in grids.items():
        for g in grid:
            if (g["broad1_covered"], g["broad2_covered"]) == (published_b1, published_b2):
                continue
            u2 = round(100.0 * (n - g["broad2_covered"]) / n, 4) if n else None
            alt.append({
                "cell": f"{defn}:{g['file_list_source']}-list/"
                        f"{g['content_source']}-content",
                "surface_n_files": g["surface_n_files"],
                "broad1_covered": g["broad1_covered"],
                "broad2_covered": g["broad2_covered"],
                "broad2_uncovered_pct_over_live": u2,
                "delta_pp_vs_published": (
                    round(u2 - pub_b2_pct, 4)
                    if (u2 is not None and pub_b2_pct is not None) else None),
            })
    _deltas = [r["delta_pp_vs_published"] for r in alt
               if r["delta_pp_vs_published"] is not None]
    _drift = (abs(round(live_b2_pct - pub_b2_pct, 4))
              if (live_b2_pct is not None and pub_b2_pct is not None) else None)
    n_within_drift = (sum(1 for d in _deltas
                           if _drift is not None and abs(d) <= _drift)
                      if _drift is not None else None)
    nearest = (min(alt, key=lambda r: abs(r["delta_pp_vs_published"]))
               if _deltas else None)
    materiality = {
        "published_broad1_uncovered_pct": pub_b1_pct,
        "published_broad2_uncovered_pct": pub_b2_pct,
        "n_alternatives": len(alt),
        "n_alternatives_within_population_drift": n_within_drift,
        "nearest_alternative_cell": (nearest["cell"] if nearest else None),
        "nearest_alternative_delta_pp": (
            nearest["delta_pp_vs_published"] if nearest else None),
        "nearest_alternative_is_within_drift": (
            bool(_drift is not None and nearest is not None
                 and abs(nearest["delta_pp_vs_published"]) <= _drift)
            if nearest is not None else None),
        "materiality_supports_indistinguishability": bool(
            n_within_drift),
        "population_drift_pp_broad2": (
            round(live_b2_pct - pub_b2_pct, 4)
            if (live_b2_pct is not None and pub_b2_pct is not None) else None),
        "population_used_by_the_published_row": published_population,
        "population_live": n,
        "alternative_readings": sorted(
            alt, key=lambda r: (r["delta_pp_vs_published"] is None,
                                r["delta_pp_vs_published"])),
        "note": (
            "the published row's own percentages are recomputed over the "
            "population IT used, so the comparison is like-for-like; the "
            "alternative readings are over the LIVE population.  "
            "CORRECTION (MEASURED 2026-11-07): the earlier framing -- 'the "
            "alternatives differ by LESS than the population's own drift, so "
            "the readings are indistinguishable' -- is NOT SUPPORTED by the "
            "measurement and is REJECTED here: NO alternative falls within "
            "the drift and the NEAREST one is several TIMES it, so the "
            "row's 2-decimal precision DOES separate the readings.  "
            "materiality_supports_indistinguishability is therefore FALSE "
            "and is reported as a measured boolean, never asserted.  "
            "THE DEFECT IS THE DENOMINATOR, NOT THE PRECISION: the published "
            "surface SIZE does not determine the published covered SET (see "
            "surface_under_determination_detail -- 82 distinct covered pairs "
            "for the same 966-file surface, MEASURED 2026-11-07), so the "
            "published pair is ONE OF MANY readings of that size and the row "
            "does not say which one."
        ),
    }

    # THE BOUND ON THE DAMAGE (machine-checked, not asserted).
    # For EVERY (list-source, content-source) cell: is BROAD-1's covered
    # count the SAME under every surface definition?  If yes, the surface
    # mixing moves BROAD-2 ONLY and the published BROAD-1 row is NOT
    # contaminated.  Without this, an UNBOUNDED defect claim ("both
    # published rows are contaminated") would itself be a defect.
    def _cells(defn: str, key: str) -> dict[tuple[str, str], int]:
        return {(g["file_list_source"], g["content_source"]): g[key]
                for g in grids[defn]}

    all_cells = sorted(_cells(next(iter(grids)), "broad1_covered"))
    b1_differ: list[dict] = []
    b2_differ: list[dict] = []
    for cell in all_cells:
        b1_vals = {defn: _cells(defn, "broad1_covered").get(cell) for defn in grids}
        b2_vals = {defn: _cells(defn, "broad2_covered").get(cell) for defn in grids}
        if len(set(b1_vals.values())) > 1:
            b1_differ.append({"cell": f"{cell[0]}-list/{cell[1]}-content", "broad1_by_definition": b1_vals})
        if len(set(b2_vals.values())) > 1:
            b2_differ.append({"cell": f"{cell[0]}-list/{cell[1]}-content", "broad2_by_definition": b2_vals})
    b1_invariant = not b1_differ

    # The declared-pair divergence is PRESERVED under an explicitly scoped
    # name.  AGENTS.md 5.5 rule 6: a correction is ADDITIVE + DATED, never a
    # silent deletion -- the old reading stays visible under its own name.
    declared_divergence = {
        "scope": ("the DECLARED surface definitions only "
                  f"({', '.join(sorted(grids))}) - this is NOT a bound on the "
                  "published row"),
        "broad1_invariant": b1_invariant,
        "broad1_cells_that_differ": b1_differ,
        "broad2_cells_that_differ": b2_differ,
        "damage_is_bounded_to": (
            "BROAD-2 ONLY" if b1_invariant else "BROAD-1 AND BROAD-2"
        ),
    }

    # Is the published row's SURFACE SIZE enough to determine its covered
    # SET?  Measured over the derived family: if the family's members do not
    # all give the same covered pair, then NO -- and that is a different
    # defect from MIXED_SURFACE.
    under_determined = bool(
        witness is not None
        and witness.get("search_valid", True)
        and witness.get("members_disagree")
    )
    if witness is not None and witness.get("search_valid", True) and witness.get("witness_found"):
        published_row_damage = (
            "NONE MEASURED BY THE MIXED-SURFACE READING - a single definition "
            "in the searched family reproduces the published surface count AND "
            "the published covered pair; the residual defect is "
            "SURFACE_UNDER_DETERMINED (the family's members DISAGREE), not "
            "MIXED_SURFACE; the published pair is reproducible in "
            f"{len(base_pair_cells)} of the base surface's "
            f"{len(base_grid)} (file-list x content) cells, so the reading "
            "depends on WHICH tree the surface was taken from"
        )
    elif witness is None or not witness.get("search_valid", True):
        published_row_damage = (
            "NOT DETERMINED - the widened search did not run or failed, so no "
            "widened verdict is available (fail-closed)"
        )
    else:
        published_row_damage = (
            "the published surface count and the published covered pair are "
            "reproduced by DIFFERENT definitions"
        )

    return {
        "published_surface_count": published_surface_count,
        "published_broad1_covered": published_b1,
        "published_broad2_covered": published_b2,
        "surface_count_reproduced_by": count_repro,
        "covered_set_reproduced_by": pair_repro,
        "MIXED_SURFACE": mixed,
        "MIXED_SURFACE_over_declared_definitions": mixed_over_declared,
        "verdict_basis": verdict_basis,
        "witness_family": witness,
        "SURFACE_UNDER_DETERMINED": under_determined,
        "SURFACE_SIZE_IS_FREE_PARAMETER": free_parameter,
        "null_contributor_count": null_counts,
        "published_pair_base_cells": base_pair_cells,
        "published_pair_base_cell_is_unique": base_pair_unique,
        "materiality": materiality,
        "surface_under_determination_detail": under_detail,
        "SPLIT_TREE_IRREDUCIBLE": bool(
            witness is not None
            and witness.get("search_valid", True)
            and witness.get("split_tree_irreducible")
        ),
        "published_row_damage": published_row_damage,
        "broad1_invariant_across_definitions": b1_invariant,
        "broad1_cells_that_differ": b1_differ,
        "broad2_cells_that_differ": b2_differ,
        "declared_definition_divergence": declared_divergence,
        "damage_is_bounded_to": (
            "BROAD-2 ONLY" if b1_invariant else "BROAD-1 AND BROAD-2"
        ),
        "damage_bound_scope": (
            "OVER THE DECLARED SURFACE DEFINITIONS ONLY - not a bound on the "
            "published row (see SURFACE_UNDER_DETERMINED and "
            "published_row_damage)"
        ),
        "consequence_under_the_named_surface": consequence,
        "files_separating_the_definitions": sep,
        "note": (
            "BROAD-2 is defined as naming the bare basename ANYWHERE, so a "
            "COMMENT mention counts under it; under a BY-INVOCATION reading the "
            "two surfaces agree. The whole difference rests on the mentions "
            "listed in files_separating_the_definitions. Those mentions are "
            "reported PER CONTENT SOURCE (head and disk) because the covered "
            "counts they explain were measured over DISK content -- reporting "
            "only the HEAD blob would be the same mixed-tree defect the entry "
            "is about, living inside the adjudication."
        ),
    }


def main() -> int:
    import sys
    if "--history" in sys.argv:
        # The expensive mode: one `git ls-tree` per reachable commit.
        out = {defn: history_scan(defn) for defn in SURFACE_DEFS}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    pop = population()
    n = len(pop)
    grids = {defn: grid_for(defn, pop) for defn in SURFACE_DEFS}
    default_grid = grids[DEFAULT_SURFACE]

    # Which combination reproduces the catalogue's published pair (133 / 176)?
    published = (133, 176)
    matches = [
        f"{g['file_list_source']}-list/{g['content_source']}-content"
        for g in default_grid
        if (g["broad1_covered"], g["broad2_covered"]) == published
    ]
    head = next(g for g in default_grid if g["file_list_source"] == "head" and g["content_source"] == "head")
    idx = next(g for g in default_grid if g["file_list_source"] == "index" and g["content_source"] == "disk")

    out = {
        "population_n_files": n,
        "population_names_sha256": _sha(pop),
        "surface_definitions": {
            defn: {
                "paths": list(SURFACE_DEFS[defn]),
                "surface_n_files": {
                    "head": len(surface("head", defn)),
                    "index": len(surface("index", defn)),
                },
                "surface_paths_sha256": {
                    "head": _sha(surface("head", defn)),
                    "index": _sha(surface("index", defn)),
                },
            }
            for defn in SURFACE_DEFS
        },
        "default_surface_definition": DEFAULT_SURFACE,
        "surface_paths_sha256": {
            "head": _sha(surface("head")),
            "index": _sha(surface("index")),
        },
        "grid": [{k: v for k, v in g.items() if not k.startswith("_")} for g in default_grid],
        "grids_by_surface_definition": {
            defn: [{k: v for k, v in g.items() if not k.startswith("_")} for g in grid]
            for defn, grid in grids.items()
        },
        "published_pair_133_176_reproduced_by": matches,
        "coherent_readings": {
            "all_committed": {
                "broad1_covered": head["broad1_covered"],
                "broad1_uncovered": head["broad1_uncovered"],
                "broad1_uncovered_pct": head["broad1_uncovered_pct"],
                "broad2_covered": head["broad2_covered"],
                "broad2_uncovered": head["broad2_uncovered"],
                "broad2_uncovered_pct": head["broad2_uncovered_pct"],
            },
            "all_working_tree": {
                "broad1_covered": idx["broad1_covered"],
                "broad1_uncovered": idx["broad1_uncovered"],
                "broad1_uncovered_pct": idx["broad1_uncovered_pct"],
                "broad2_covered": idx["broad2_covered"],
                "broad2_uncovered": idx["broad2_uncovered"],
                "broad2_uncovered_pct": idx["broad2_uncovered_pct"],
            },
        },
        "published_row_adjudication": adjudicate_published_row(grids, pop),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
