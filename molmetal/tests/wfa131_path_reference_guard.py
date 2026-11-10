#!/usr/bin/env python
"""(§6.242 extension) — every repo-relative path an SSOT document cites must resolve.

**The problem this module solves.**  Catalog §6.242 records that the four SSOT
documents still cite **48** repo-relative paths that do not resolve on disk
("paths under ``metrics/``, ``molmetal/``, ``scripts/``, ``configs/``,
``tests/``"), and that the Wf-A127 guard does not cover them.  §6.242 also
says the number is a *path-binding* measurement, not a verdict on each path.

**The design.**  A path-shaped citation is a claim of the form *"there is an
artifact at this location"*.  This module resolves that claim against
**values**, not against string shapes:

* the **filesystem** answers "does this exact path exist?";
* the **registry** (``registry/metrics.yaml``) answers "is this name a metric,
  and if so, what is its canonical artifact?".

That asymmetry is what lets a short-form metric id (``metrics/by_metric/m1.json``)
be exempt while a typo'd metric path (``metrics/by_metric/m22.json``) is
reported — the two strings have the same shape, but only one of them names a
thing the registry declares.

**Buckets.**  Every admitted token lands in exactly one bucket.  Only four
fail:

==========================  ============================================
bucket                      meaning
==========================  ============================================
``BROKEN_METRIC_PATH``      inside the ``metrics/`` namespace, but the
                            registry declares no such artifact there
``BROKEN_METRIC_ARTIFACT``  the registry declares the metric, but its
                            canonical ``source_json`` is not on disk
``BROKEN_ROOT_PREFIX``      a file with that basename exists, but under a
                            different directory chain
``DANGLING``                no file of that name exists anywhere
==========================  ============================================

``BROKEN_PATTERN`` is a fifth failing bucket: a glob/template citation whose
pattern expands to nothing at all (C1 measured three such in the corpus).

**What is NOT here.**  The ``TODO/(completed|pending|archive)/*.md`` family is
owned by the Wf-A127 guard and is **delegated**, not duplicated — this module
never admits a token from that family, and it makes no use of the repo's
``...`` placeholder convention.

**No filesystem walk.**  The repo is 201 GB; ``os.walk`` is forbidden by
AGENTS.md.  The path index is built from ``git ls-files`` (tracked +
untracked-but-not-ignored).  Existence is always decided by the filesystem;
the index is used only to find *where else* a basename lives.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

#: The repository root (this file lives at ``<repo>/molmetal/tests/``).
REPO = Path(__file__).resolve().parents[2]

#: The SSOT documents whose cross-references are checked.
DOCUMENTS: tuple[str, ...] = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "TODO/INDEX.md",
    "TODO/README.md",
    "docs/WF_HISTORY.md",
)

#: The family the Wf-A127 guard already owns.  Delegated, not duplicated.
OWNED_BY_WFA127 = re.compile(r"TODO/(?:completed|pending|archive)/[A-Za-z0-9_.-]+\.md")

#: A path token: repo-relative, at least one directory segment, real file
#: extension.  The lookbehind/lookahead are PATH BOUNDARIES — a token may not
#: start or end in the middle of a longer path.  (``\b`` is a *word* boundary,
#: not a *path* boundary: it splits ``.data/molmetal/x.csv`` into
#: ``data/molmetal/x.csv``, which is a different file.)
#:
#: REPAIR (2026-11-07, Wf-A269 wave-32): the lookahead used to exclude ``.``
#: as well, so a path at the END OF A SENTENCE (``see docs/x.md.``) produced
#: NO token at all: the greedy class swallowed the sentence period, the
#: backtrack that would have yielded ``docs/x.md`` was rejected by the
#: lookahead, and the scan continued — so a broken citation in that position
#: was INVISIBLE to the guard.  ``.`` is now absent from the lookahead.  The
#: token is still exactly the path: the class is greedy, so it consumes every
#: ``.`` it can, and the final character class (``[A-Za-z0-9_]``) forces the
#: period back out — the match ends BEFORE the period, never inside it.
PATH_TOKEN = re.compile(
    r"(?<![\w./\\-])"
    r"(?P<path>[A-Za-z0-9_.][A-Za-z0-9_.\-/]*[A-Za-z0-9_])"
    r"(?![\w/\\-])"
)

#: A brace group carrying a path prefix: ``molmetal/{scripts,train.py}``.
BRACE_GROUP = re.compile(r"\{([^{}]*)\}")

#: Characters that mark the rest of the token as a glob / template.
GLOB_CHARS = set("{*,?<")

#: A §5.6 safe-overwrite backup snapshot: ``x.py.bak.<tag>``.
BACKUP_SUFFIX = re.compile(r"\.bak(\.[A-Za-z0-9_.-]+)?$")

#: Top-level directories that are not repository content and must not seed
#: the root set: tooling, caches and virtualenvs.  Every *tracked* content
#: directory — ``results/``, ``tmp/``, ``logs/``, ``list/``, ``submission/``,
#: ``environments/`` — IS admitted, because a citation into one of them is
#: still a pointer a reader is expected to follow.  (MEASURED 2026-10-31:
#: admitting them adds **0** new failing paths, so the coverage is free.)
#:
#: ``paper`` and ``paper_a`` are excluded on purpose: TODO-84 forbids
#: touching ``paper/*``, so the guard does not even look there.
NON_CONTENT_TOP_LEVEL = frozenset(
    {
        ".git",
        ".venv",
        ".venv.bak.rocm7.2_1790978356",
        ".pytest_cache",
        ".pytest_tmp",
        ".ruff_cache",
        ".hypothesis",
        ".tools",
        ".ckpt",
        ".cloned",
        ".repro_out",
        ".claude",
        ".github",
        ".wf252_backups",
        "__pycache__",
        "paper",
        "paper_a",
    }
)


# --------------------------------------------------------------------- index
@lru_cache(maxsize=4)
def _git_paths_cached(repo_str: str) -> tuple[str, ...]:
    """Tracked + untracked-but-not-ignored repo paths (no filesystem walk)."""
    repo = Path(repo_str)
    out: set[str] = set()
    for args in (
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
    ):
        try:
            res = subprocess.run(
                args, cwd=repo, capture_output=True, text=True, check=False
            )
        except OSError:
            continue
        if res.returncode != 0:
            continue  # not a git repo (synthetic fixture) — no index
        out |= {p for p in res.stdout.split("\0") if p}
    return tuple(sorted(out))


def _git_paths(repo: Path) -> tuple[str, ...]:
    return _git_paths_cached(str(repo))


def _roots(repo: Path) -> set[str]:
    """The repo's top-level content directories."""
    try:
        entries = list(repo.iterdir())
    except OSError:
        return set()
    return {
        p.name for p in entries if p.is_dir() and p.name not in NON_CONTENT_TOP_LEVEL
    }


def _package_dirs(repo: Path) -> list[str]:
    """Every importable package directory (has ``__init__.py``)."""
    found: set[str] = set()
    for q in _git_paths(repo):
        if not q.endswith("__init__.py"):
            continue
        rel = Path(q).parent
        parts = rel.parts
        if len(parts) > 3 or ".git" in parts or "__pycache__" in parts:
            continue
        found.add(str(rel))
    return sorted(found)


# ------------------------------------------------------------------ registry
def _registry_lines(reg: Path) -> dict[str, str]:
    """Fallback line parser for ``registry/metrics.yaml``."""
    key2src: dict[str, str] = {}
    cur = None
    for line in reg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if m:
            cur = m.group(1)
            continue
        m2 = re.match(r"^    source_json: (.+)$", line)
        if m2 and cur:
            key2src[cur] = m2.group(1).strip()
    return key2src


def _registry(repo: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(key -> source_json, artifact basename -> source_json, mN -> source_json).

    ``registry/metrics.yaml`` is the metric SSOT.  It is parsed with PyYAML
    when importable; the line parser above is the fallback.
    """
    reg = repo / "registry/metrics.yaml"
    if not reg.exists():
        return {}, {}, {}
    key2src: dict[str, str] = {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(reg.read_text(encoding="utf-8")) or {}
        for k, v in (data.get("metrics") or {}).items():
            if isinstance(v, dict) and v.get("source_json"):
                key2src[str(k)] = str(v["source_json"])
    except Exception:
        key2src = {}
    if not key2src:
        key2src = _registry_lines(reg)
    by_base: dict[str, str] = {}
    for v in key2src.values():
        by_base.setdefault(Path(v).name, v)
    legacy: dict[str, str] = {}
    for k, v in key2src.items():
        if re.match(r"^m\d+_", k):
            legacy.setdefault(k.split("_")[0], v)
    return key2src, by_base, legacy


# ---------------------------------------------------------------- extraction
#: A basename is file-shaped when SOME dot in it is followed by a
#: letter-starting alphanumeric run.
#:
#: Requiring the extension to be the FINAL component would drop
#: ``x.py.bak.wftui.0664`` and ``…yaml.archived_2026-10-03_…`` (both real
#: files); requiring it to be the FIRST component would drop every ``...``
#: name — which would make the repo's ellipsis convention an escape hatch
#: here too, exactly the defect C2 recorded against the sibling guard (D4).
#: The ellipsis names must be ADMITTED so that they are checked like any
#: other citation.
FILE_SHAPED = re.compile(r"\.[A-Za-z][A-Za-z0-9_]*")


def _has_extension(token: str) -> bool:
    """True when the token names a file (not a bare name, not a directory)."""
    base = token.rsplit("/", 1)[-1]
    return bool(FILE_SHAPED.search(base))


def _brace_members(line: str) -> list[str]:
    """Expand ``prefix/{a,b}`` groups into their members."""
    out: list[str] = []
    for m in BRACE_GROUP.finditer(line):
        prefix = line[: m.start()]
        pm = re.search(r"(?<![\w./\\-])([A-Za-z0-9_.][A-Za-z0-9_.\-/]*)$", prefix)
        if not pm:
            continue
        for member in re.split(r"[\s,]+", m.group(1)):
            if member:
                out.append(pm.group(1) + member)
    return out


def _pattern_expansion(token: str, line: str, end: int) -> list[str] | None:
    """The literal prefixes a glob token stands for, or None if not a glob.

    ``configs/cfm_*``     -> ``["configs/cfm_"]``
    ``configs/k_{0,1,2}`` -> ``["configs/k_0", "configs/k_1", "configs/k_2"]``

    The prefix is the TOKEN, never the whole line: a line can be very long and
    is not part of the path.
    """
    if end >= len(line) or line[end] not in GLOB_CHARS:
        return None
    if line[end] == "*":
        return [token]
    if line[end] == "{":
        close = line.find("}", end)
        if close == -1:
            return None
        members = [m for m in re.split(r"[\s,]+", line[end + 1 : close]) if m]
        return [token + m for m in members] or None
    return None


# ---------------------------------------------------------------- resolution
class Resolver:
    """Everything the classifier needs about one repository, computed once."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.roots = _roots(repo)
        self.packages = _package_dirs(repo)
        self.key2src, self.src_by_base, self.legacy = _registry(repo)
        self.index = _git_paths(repo)
        self.by_base: dict[str, list[str]] = defaultdict(list)
        for q in self.index:
            self.by_base[q.rsplit("/", 1)[-1]].append(q)

    # -- extraction ------------------------------------------------------
    def extract(self, text: str) -> list[str]:
        """Every repo-relative path token in ``text``, in document order."""
        out: list[str] = []
        for line in text.splitlines():
            raw: list[str] = []
            for m in PATH_TOKEN.finditer(line):
                tok = m.group("path")
                if self._admitted(tok):
                    raw.append(tok)
            raw.extend(_brace_members(line))
            for tok in raw:
                if self._admitted(tok):
                    out.append(tok)
        return out

    def _admitted(self, tok: str) -> bool:
        if any(c in tok for c in GLOB_CHARS):
            return False
        if "/" not in tok:
            return False  # a bare filename is a NAME, not a path
        if tok.split("/", 1)[0] not in self.roots:
            return False  # not repo-rooted (library path, prose)
        if not _has_extension(tok) and not (self.repo / tok).is_dir():
            return False  # `TODO/01-13/`, `docs/metadata` — prose
        if OWNED_BY_WFA127.fullmatch(tok):
            return False  # delegated to the Wf-A127 guard
        return True

    # -- resolution ------------------------------------------------------
    def classify(self, token: str) -> tuple[str, str]:
        """Return ``(bucket, evidence)`` for one path token."""
        repo = self.repo
        if (repo / token).is_file():
            return "RESOLVED_DIRECT", token
        if (repo / token).is_dir():
            return "RESOLVED_DIR", token

        segs = token.split("/")

        # --- metrics namespace: the registry is authoritative --------------
        if "by_metric" in segs:
            base = segs[-1]
            stem = base.rsplit(".", 1)[0]
            if base in self.src_by_base:
                canon = self.src_by_base[base]
                if token == canon:
                    return "RESOLVED_METRIC", canon
                return "BROKEN_METRIC_PATH", canon
            if stem in self.legacy:
                canon = self.legacy[stem]
                if not (repo / canon).exists():
                    return "BROKEN_METRIC_ARTIFACT", canon
                return "RESOLVED_METRIC_ID", canon
            for q in self.by_base.get(base, []):
                return "BROKEN_METRIC_PATH", q
            return "BROKEN_METRIC_PATH", ""

        # --- §5.6 backup snapshot: `x.py.bak.<tag>` ------------------------
        m = BACKUP_SUFFIX.search(token)
        if m:
            inner = self.classify(token[: m.start()])
            if inner[0].startswith("RESOLVED"):
                return "RESOLVED_BACKUP", inner[1]

        # --- package directory cited as a module: `pkg.py` -----------------
        if token.endswith(".py") and (repo / token[:-3] / "__init__.py").exists():
            return "RESOLVED_PACKAGE", token[:-3]

        # --- package-relative spelling: `models/bond_head.py` --------------
        rest = "/".join(segs[1:])
        for d in self.packages:
            if d.rsplit("/", 1)[-1] == segs[0]:
                cand = f"{d}/{rest}"
                if (repo / cand).exists():
                    return "RESOLVED_PACKAGE_RELATIVE", cand

        # --- the file exists, but its directory chain differs ---------------
        base = segs[-1]
        for q in self.by_base.get(base, []):
            if q.endswith("/" + token):
                return "RESOLVED_PACKAGE_RELATIVE", q
        for q in self.by_base.get(base, []):
            return "BROKEN_ROOT_PREFIX", q
        return "DANGLING", ""

    # -- patterns --------------------------------------------------------
    def classify_pattern(self, prefixes: list[str]) -> tuple[str, str]:
        """Classify a glob citation by what its pattern expands to.

        A pattern is not a pointer to one file, so it cannot be "resolved" the
        way a path can — but a pattern that expands to NOTHING is still a dead
        end, and must be reported rather than silently dropped.

        The expansion is checked against the **filesystem** first: a single
        directory listing of the pattern's parent (never a walk).  The git
        index is only a fallback, because much of the repo's evidence tree
        (``molmetal/reports/**/*.json``, ``.data/``) is gitignored and would
        otherwise look empty.
        """
        for p in prefixes:
            if (self.repo / p).exists():
                return "RESOLVED_PATTERN", p
            parent, _, name_prefix = p.rpartition("/")
            if parent:
                directory = self.repo / parent
                if directory.is_dir():
                    try:
                        with os.scandir(directory) as entries:
                            for entry in entries:
                                if entry.name.startswith(name_prefix):
                                    return "RESOLVED_PATTERN", p
                    except OSError:
                        pass
            if any(q.startswith(p) for q in self.index):
                return "RESOLVED_PATTERN", p
        return "BROKEN_PATTERN", ""

    def extract_patterns(self, text: str) -> list[list[str]]:
        """Every glob/template citation in ``text``, as prefix lists."""
        out: list[list[str]] = []
        for line in text.splitlines():
            for m in PATH_TOKEN.finditer(line):
                tok = m.group("path")
                if ".." in tok:
                    continue  # a range / parent-dir notation, not a glob
                exp = _pattern_expansion(tok, line, m.end("path"))
                if exp is None:
                    continue
                if "/" not in tok or tok.split("/", 1)[0] not in self.roots:
                    continue
                out.append(exp)
        return out


#: The buckets that FAIL the guard.
FAILING_BUCKETS = frozenset(
    {
        "BROKEN_METRIC_PATH",
        "BROKEN_METRIC_ARTIFACT",
        "BROKEN_ROOT_PREFIX",
        "BROKEN_PATTERN",
        "DANGLING",
    }
)


def scan(repo: Path, documents: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """``{token: (bucket, evidence)}`` for every path token in the documents."""
    r = _resolver(repo)
    out: dict[str, tuple[str, str]] = {}
    for rel in documents:
        for tok in r.extract((repo / rel).read_text(encoding="utf-8")):
            out.setdefault(tok, r.classify(tok))
    return out


def scan_texts(repo: Path, texts: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Same as :func:`scan`, but over in-memory document texts.

    Used by the negative tests: a perturbed COPY of a document is scanned
    against the real repository, so the perturbation never touches the repo.
    """
    r = _resolver(repo)
    out: dict[str, tuple[str, str]] = {}
    for name, text in texts.items():
        for tok in r.extract(text):
            out.setdefault(tok, r.classify(tok))
    return out


def scan_patterns(repo: Path, texts: dict[str, str]) -> dict[str, tuple[str, str]]:
    """``{pattern: (bucket, evidence)}`` for every glob citation."""
    r = _resolver(repo)
    out: dict[str, tuple[str, str]] = {}
    for text in texts.values():
        for prefixes in r.extract_patterns(text):
            key = "|".join(prefixes)
            out.setdefault(key, r.classify_pattern(prefixes))
    return out


# --------------------------------------------------------------- compat API
_RESOLVER_CACHE: dict[str, Resolver] = {}


def _resolver(repo: Path) -> Resolver:
    key = str(repo)
    r = _RESOLVER_CACHE.get(key)
    if r is None:
        r = Resolver(repo)
        _RESOLVER_CACHE[key] = r
    return r


def classify(token: str, repo: Path) -> tuple[str, str]:
    """Module-level convenience wrapper around ``Resolver.classify``."""
    return _resolver(repo).classify(token)


def extract_tokens(text: str, repo: Path) -> list[str]:
    """Module-level convenience wrapper around ``Resolver.extract``."""
    return _resolver(repo).extract(text)
