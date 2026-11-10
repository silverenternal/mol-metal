# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Wf-A269 -- the zero-tolerance portability gate for host-bound repo paths.

THE DEFECT CLASS
----------------
A tracked file reaches a path inside this repository through an *absolute
path literal bound to one host's checkout root*.  Such a file resolves only
on a machine whose checkout happens to live at the author's path; on every
other checkout it is either a hard crash or a silent no-op.  The classic
failure mode is a test that skips itself when the path is absent -- the
suite is green on the author's box and vacuous everywhere else.

MEASURED on 2026-11-02 by the Wf-A269 audit wave: 251 tracked ``.py`` files
were in this class before the portability repair wave (the same scope file
list the repair wave worked from).

MEASURED STATE OF THIS GATE (2026-11-02, CPU-only, AFTER the repair wave)
-----------------------------------------------------------------------------
* tracked ``.py`` files scanned: **2874** (``git ls-files -- '*.py'``)
* prefilter candidates / files actually AST-analysed: 250 / 30
* allow-list entries: **8**, each with a one-line justification naming its
  SHAPE (the table was **77** before the shrink: **69** of those entries were
  STALE -- they named hits that no longer exist -- and were dropped)
* defect-class hits still present: **8 in 8 files** -- every one of them
  allow-listed, i.e. the allow-listed complement is empty
* the repair wave took the defect-class file count **251 -> 8** and introduced
  **0** new occurrences (0 new ``(file, root, target)`` triples)

The allow-list is review debt, not a shield: every entry names a
*reviewed* violation AND the SHAPE that makes it one, and
``test_allowlist_stale_entries_are_reported`` prints the entries that no
longer correspond to a live violation so they can be dropped as the repair
wave lands.  The zero-tolerance direction -- a violation that is NOT in the
allow-list -- always fails the gate.  ``--emit-allowlist`` prints the entries
the current tree needs.

THE DETECTION RULE (four conditions, all must hold)
----------------------------------------------------
A hit is recorded for a literal ``L`` in a tracked ``.py`` file when:

  (a) BINDING      -- ``L`` is a string literal whose value starts with one
                       of the forbidden host roots, and it is bound to a name
                       (``X = L``), passed to ``Path(L)``, composed with
                       ``/`` or ``os.path.join``, or interpolated into an
                       f-string.  The composed forms are resolved
                       transitively (a root bound to ``X`` and then
                       ``X / "a" / "b"`` is the same defect).
  (b) REACHABILITY -- the resolved path lands inside this repository, i.e.
                       the repo-relative remainder names a tracked file or a
                       directory that contains tracked files.  A literal that
                       points outside the checkout (external data, an
                       external tool, another repo) is NOT this class.
  (c) LIVENESS     -- the binding is *read* somewhere.  A constant assigned
                       a host path and never read -- directly or through
                       another never-read constant -- is dead code, not a
                       defect.  Liveness is transitive through the
                       composition graph.
                       KNOWN GAP: liveness is keyed by NAME, not by binding,
                       so a name re-bound to a portable value before any read
                       still looks live (a "shadowed" binding).  Such a hit is
                       reported as reachable and must be exempted by hand.
  (d) IO REACH     -- the value actually reaches a file-opening position:
                       ``open()``, ``read_text``/``read_bytes``/
                       ``write_text``/``write_bytes``, ``mkdir``/``touch``/
                       ``unlink``/``rename``/``replace``/``symlink_to``,
                       ``spec_from_file_location``, ``exec``/``eval``/
                       ``subprocess.run``/``Popen``/``check_call``/
                       ``check_output``, or ``sys.path.insert``.
                       A NAMED literal used only for a comparison, a log
                       message, a docstring, or a stat-only existence check
                       (``.exists()`` / ``.is_file()`` / ``.stat()``) is NOT
                       this class.
                       KNOWN GAP: for an UNNAMED (inline) literal the sink
                       context is never computed -- ``classify()`` hard-codes
                       the term set to ``{"INLINE"}`` -- so this exclusion is
                       NEVER EVALUATED for it.  An inline literal is therefore
                       always treated as reaching I/O.
                       KNOWN GAP (2026-11-02, Wf-A269 wave-2 verify): the
                       transitive composition walk is TRUNCATED at depth 8
                       (``terminals()`` returns the empty set past it) and
                       ``classify()`` then reports the binding as ``DEAD`` --
                       so a live, I/O-reaching root literal that is 9 or more
                       hops from its sink would be INVISIBLE to the gate.
                       MEASURED on this tree by
                       ``/tmp/wfa269k_census_corrected.py`` (which instruments
                       ``FileAnalysis.terminals`` and re-runs the real scan):
                       maximum composition depth observed = **2**, depth
                       cutoff events = **0**.  The gap is therefore currently
                       HARMLESS, and it is declared here because the gate's
                       KNOWN-GAP list must not UNDER-REPORT what it cannot see.
                       KNOWN GAP (2026-11-02, Wf-A269 wave-2 verify): a host
                       root embedded MID-STRING is invisible to ``root_of()``,
                       which requires the literal to START with a forbidden
                       root (``lit == r`` or ``lit.startswith(r + "/")``).
                       MEASURED repo-wide by
                       ``/tmp/wfa269k_census_corrected.py``: **224** string
                       constants merely CONTAIN a forbidden root -- **203**
                       of them inside docstrings, **21** in executable code.
                       Of the 21 code sites, 18 are PROSE inside message
                       strings, 2 are the two regex DETECTORS in
                       ``scripts/_wrapper_audit.py`` (which must match the
                       absolute form by design), and 1 is a real
                       ``PYTHONPATH=`` export in
                       ``scripts/_patch_i2_preflight_wrapper.py``.  No
                       wave-2-scope repair was missed, but the blind spot is
                       real and is now declared rather than implied.

The four defect classes are ``IO`` (reachable, lands on a tracked file),
``DIR_ANCHOR`` (reachable, lands on a repo directory), ``ROOT_BIND`` (a bare
checkout-root constant) and ``MISSING_TARGET`` (reachable, lands on a path
that is not in the checkout at all -- typically an output directory or the
``.venv`` interpreter).  For a NAMED binding, everything else is
``VALUE_ONLY`` (bound but never reaches I/O) or ``DEAD`` (never read) and is
NOT flagged.  For an INLINE literal the distinction does not arise: it is
always reported (see the KNOWN GAP under rule (d)).

WHAT THIS GATE COVERS
----------------------
* Every tracked ``.py`` file (``git ls-files -- '*.py'``), never
  ``os.walk`` -- this repository is 201 GB.
* The four path FORMS that matter: bare string literals, ``Path("...")``,
  ``os.path.join("...", ...)``, ``root / "a" / "b"`` chains, and f-strings
  whose prefix is an absolute path.
* A per-hit allow-list, zero tolerance on its complement.

WHAT THIS GATE DOES NOT COVER (deliberately)
---------------------------------------------
* **Shell wrappers.** ``scripts/*.sh`` are NOT scanned.  This is a
  CONCESSION, not a design decision: AGENTS.md §5.9 *mandates* the absolute
  venv path inside the launchers, and the enforcement of that rule
  (``scripts/_wrapper_audit.py``) matches the host-root ABSOLUTE form only --
  a portable ``"$REPO_ROOT"/.venv/bin/python`` wrapper is classified
  SCAFFOLD and the host-bound one REAL (MEASURED, A/B proof in
  ``molmetal/scripts/wf_a269_section_59_ab.py``).  Excluding ``.sh`` here is
  therefore a concession to a policy whose ENFORCEMENT REQUIRES the defect
  class this gate exists to find.  Measured size of the excluded half: 34
  live files / 35 in-class occurrences, of which 25 are the canonical
  ``scripts/launch_wf*.sh`` wrappers (25/25 carry the absolute
  ``.venv/bin/python`` AND the absolute ``PYTHONPATH=``; 6/25 derive a repo
  root from their own location, 19/25 do not).  Amending §5.9 is USER-gated.
  In a ``.py`` file the portable form of that interpreter path is
  ``sys.executable``.

  **CORRECTION (2026-11-02, the Wf-A269 wave-2 verify wave).**  The census
  sentence above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, but three
  of its numbers are NOT REPRODUCIBLE under any rule that could be
  reconstructed, and one of them was a DISPUTED number that this file shipped
  without qualification.  The corrected census, with the rule stated, is
  measured by ``/tmp/wfa269k_census_corrected.py`` (re-runnable; it writes
  ``/tmp/wfa269k_census_corrected.json``):

  * RULE: universe = tracked ``.sh`` files, EXCLUDING ``TODO/`` and
    ``scripts/_retired/`` (**96** live files of **113** tracked; **10**
    archived, **7** retired).  "in-class" = the file carries at least one
    forbidden-root occurrence on a line that is not a pure comment.
  * **53** live files are in-class, carrying **212** non-comment occurrences
    (**220** raw).  The superseded sentence said "34 live files / 35
    in-class occurrences".
  * The canonical ``scripts/launch_wf*.sh`` wrappers are **23**, not 25, and
    all **23** of them are in-class.  **23/23** carry the absolute
    ``.venv/bin/python`` AND an absolute ``PYTHONPATH=`` -- that sub-claim
    SURVIVES the correction unchanged.
  * **11** of the 23 derive a repo root from their own location and **12**
    do NOT.  The superseded sentence said "6/25 derive ... 19/25 do not".
    The repo's own draft had recorded this as a LIVE DISPUTE ("[R] 6/25
    derive, [L] 11/25 derive"); this file shipped the [R] side unqualified.
    The corrected measurement is **11**, which agrees with the [L] side.
* **External data and tools.** Absolute paths to datasets, binaries or
  checkouts outside this repository are a different (lower-severity) class
  and are not flagged here.
* **Docstrings and comments.** Prose that mentions a path is not a binding.
* **Non-I/O string literals.** A host path used only for a comparison, a
  log line, or a stat-only existence check is not flagged -- TRUE for a
  NAMED binding, FALSE for an inline literal (see the KNOWN GAP under rule
  (d) above).
* **Non-Python tracked files.** ``.sh``, ``.yaml``, ``.json`` and the
  markdown corpus are out of scope for this gate.
* **This file's own root constants.** The gate has to *name* the roots it
  forbids.  It therefore holds exactly three host-root literals, the
  constants that define them, and nothing else;
  ``test_gate_file_holds_no_unapproved_host_root_literal`` asserts that.
  For the same reason the allow-list stores each entry as
  ``(file, root_id, tail)`` instead of the verbatim absolute literal:
  storing the literal would make the gate a violation of its own rule.

A GATE THAT CANNOT FAIL IS WORSE THAN NO GATE
----------------------------------------------
``test_mutation_self_test_detects_a_synthetic_violation`` builds a
synthetic module containing a host-root literal that opens a tracked repo
file and asserts the detector returns it.  ``..._accepts_synthetic_non_violations``
feeds the same function the near-miss shapes and asserts they are NOT
returned.  The synthetic modules live in pytest's ``tmp_path``; nothing is
ever written into the repository.

CPU-only, no GPU, no network.  stdlib + pytest only.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The forbidden host roots.  Each is bound to its own name so that this file
# can name them without becoming a violation of its own rule: a literal that
# is the direct right-hand side of an assignment is a *binding*, and the
# binding is only flagged when the value it names reaches I/O.  Here the
# roots are only ever iterated over, so they resolve to VALUE_ONLY.
# ---------------------------------------------------------------------------
_ROOT_AUTHOR = "/home/hugo/codes/mol-metal"
_ROOT_TRITON = "/home/hugo/codes/try_triton_on_rocm"
_ROOT_DISK1 = "/mnt/disk1/Hugo_Lee/mol-metal"
_ROOTS = (_ROOT_AUTHOR, _ROOT_TRITON, _ROOT_DISK1)

IO_FUNCS = {"open"}
IO_METHODS = {
    "read_text", "read_bytes", "write_text", "write_bytes",
    "spec_from_file_location", "mkdir", "touch", "unlink", "rmdir",
    "rename", "replace", "symlink_to", "open",
}
EXEC_FUNCS = {"exec", "eval", "run", "Popen", "check_call", "check_output",
              "call"}
STAT_METHODS = {"exists", "is_file", "is_dir", "is_symlink", "stat",
                "samefile"}
STR_FUNCS = {"str", "repr", "os.fspath", "fspath", "abspath", "normpath",
             "expanduser", "realpath", "dirname", "basename"}

#: The classes that ARE the defect class and must be flagged.
DEFECT_CLASSES = ("IO", "DIR_ANCHOR", "ROOT_BIND", "MISSING_TARGET")


# ---------------------------------------------------------------------------
# Host-root resolution
# ---------------------------------------------------------------------------
def _root_list() -> tuple[str, ...]:
    """The forbidden roots, in a stable order.

    Index 0/1/2 are the three declared checkout roots; index 3 is the live
    checkout root (``str(REPO)``) when it differs from them, so the gate also
    catches a path hardcoded to wherever *this* checkout happens to live.
    """
    out: list[str] = []
    for r in _ROOTS:
        if r not in out:
            out.append(r)
    live = str(REPO)
    if live not in out:
        out.append(live)
    return tuple(out)


def root_of(lit: str) -> str | None:
    """Return the host root ``lit`` is bound to, or None."""
    for r in _root_list():
        if lit == r or lit.startswith(r + "/"):
            return r
    return None


def root_id(root: str) -> int:
    """Stable integer id of a host root (index into :func:`_root_list`)."""
    return _root_list().index(root)


# ---------------------------------------------------------------------------
# Git plumbing -- the scan universe is the git index, never a filesystem walk
# ---------------------------------------------------------------------------
def git_tracked(repo: Path) -> list[str]:
    """Every tracked path in the index (repo-relative, posix separators)."""
    res = subprocess.run(["git", "ls-files", "-z"], cwd=repo,
                         capture_output=True, text=True, check=False)
    return [p for p in res.stdout.split("\0") if p]


def git_tracked_py(repo: Path) -> list[str]:
    """Every tracked ``.py`` file."""
    res = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=repo,
                         capture_output=True, text=True, check=False)
    return sorted(p for p in res.stdout.split("\0") if p.endswith(".py"))


def git_grep_files(repo: Path, patterns: list[str]) -> list[str]:
    """Tracked ``.py`` files whose working-tree text contains any pattern.

    This is the *prefilter* for the AST scan.  It is exact rather than
    heuristic: a hit requires a string literal whose value starts with a
    host root, so the root text must occur verbatim in the file.  Files that
    do not mention any root at all cannot produce a hit, and the test
    ``test_prefilter_partition_is_exact`` pins that partition.
    """
    cmd = ["git", "grep", "-l", "-F", "-a"]
    for pat in patterns:
        cmd += ["-e", pat]
    cmd += ["--", "*.py"]
    res = subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                         check=False)
    if res.returncode not in (0, 1):
        return []
    return [l for l in res.stdout.splitlines() if l.strip()]


def prefix_map(index: set[str]) -> dict[str, int]:
    """Map every directory prefix of an indexed path to its file count."""
    out: dict[str, int] = {}
    for t in index:
        parts = t.split("/")
        for i in range(1, len(parts)):
            key = "/".join(parts[:i])
            out[key] = out.get(key, 0) + 1
    return out


def docstring_ids(tree: ast.AST) -> set[int]:
    """Ids of the nodes that hold a docstring.

    Both the docstring *statement* and its *constant* are recorded: a
    docstring is prose, and prose that mentions a path must never be
    flagged.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None:
                stmt = node.body[0]
                out.add(id(stmt))
                val = getattr(stmt, "value", None)
                if val is not None:
                    out.add(id(val))
    return out


def func_name(node: ast.AST) -> str:
    f = getattr(node, "func", None)
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
class FileAnalysis:
    """Defect-class analysis of a single source file."""

    def __init__(self, rel_path: str, src: str, index: set[str],
                 prefixes: dict[str, int]):
        self.rel = rel_path
        self.src = src
        self.index = index
        self.prefixes = prefixes
        self.tree = ast.parse(src)
        self.parents = parent_map(self.tree)
        self.docs = docstring_ids(self.tree)
        self.bindings: dict[str, dict] = {}
        self.uses: dict[str, list[tuple[str, str | None]]] = {}
        self.hits: dict[tuple, dict] = {}
        self._subtrees: dict[int, frozenset[int]] = {}
        self._collect()

    # -- subtree containment, memoised (the naive form is quadratic) ------
    def _contains(self, outer: ast.AST, inner: ast.AST) -> bool:
        key = id(outer)
        s = self._subtrees.get(key)
        if s is None:
            s = frozenset(id(n) for n in ast.walk(outer))
            self._subtrees[key] = s
        return id(inner) in s

    # -- pass A/B/C/D ------------------------------------------------------
    def _collect(self):
        # Pass A: NAME = <literal>  and  NAME = Path(<literal>)
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not names:
                continue
            lit = None
            if isinstance(node.value, ast.Constant) and \
                    isinstance(node.value.value, str):
                lit = node.value.value
            elif isinstance(node.value, ast.Call) and \
                    func_name(node.value) == "Path" and node.value.args and \
                    isinstance(node.value.args[0], ast.Constant) and \
                    isinstance(node.value.args[0].value, str):
                lit = node.value.args[0].value
            if lit is None:
                continue
            r = root_of(lit)
            if r is None:
                continue
            rel = lit[len(r):].strip("/")
            for n in names:
                self.bindings[n] = {"kind": "ROOT", "root": r,
                                    "literal": lit, "line": node.lineno,
                                    "rel": rel, "expr": node.value}
                self._record(node.lineno, n, r, rel, "LITERAL", lit)

        # Pass B: NAME = <composed path expression>  (transitive, 4 rounds)
        for _ in range(4):
            for node in ast.walk(self.tree):
                if not isinstance(node, ast.Assign):
                    continue
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if not names:
                    continue
                res = self._resolve(node.value)
                if res is None:
                    continue
                root, rel = res
                if root is None:
                    continue
                for n in names:
                    if n in self.bindings and \
                            self.bindings[n].get("kind") == "ROOT":
                        continue
                    self.bindings[n] = {"kind": "PATH", "root": root,
                                        "literal": None, "line": node.lineno,
                                        "rel": rel, "expr": node.value}
                    self._record(node.lineno, n, root, rel, "COMPOSED", None)

        # Pass C: inline literals (not the direct RHS of an assignment)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in self.docs:
                    continue
                r = root_of(node.value)
                if r is None:
                    continue
                parent = self.parents.get(id(node))
                if isinstance(parent, ast.Assign) and parent.value is node:
                    continue
                if isinstance(parent, ast.Call) and \
                        func_name(parent) == "Path":
                    gp = self.parents.get(id(parent))
                    if isinstance(gp, ast.Assign) and gp.value is parent:
                        continue
                rel = node.value[len(r):].strip("/")
                self._record(node.lineno, None, r, rel, "INLINE", node.value)
                for r2 in _root_list():
                    idx2 = node.value.find(r2)
                    if idx2 <= 0:
                        continue
                    tail = node.value[idx2 + len(r2):]
                    for delim in (" ", "\t", "\n", "'", '"', ";", "|", "&",
                                  "$", "`", "\\", ")", ",", "}"):
                        cut = tail.find(delim)
                        if cut >= 0:
                            tail = tail[:cut]
                    self._record(node.lineno, None, r2, tail.strip("/"),
                                 "EMBEDDED", node.value)

        # Pass D: the def-use graph
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self._use(node.id, node)
            if isinstance(node, ast.Attribute) and \
                    isinstance(node.value, ast.Name) and \
                    isinstance(node.value.ctx, ast.Load):
                self._use(node.value.id, node.value)

    # -- expression resolution -------------------------------------------
    def _resolve(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            r = root_of(node.value)
            if r is None:
                return None
            return r, node.value[len(r):].strip("/")
        if isinstance(node, ast.Name):
            b = self.bindings.get(node.id)
            if not b:
                return None
            return b["root"], b["rel"]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            l = self._resolve(node.left)
            r = self._resolve(node.right)
            if l is None:
                return None
            root = l[0]
            rel = l[1]
            if r is None:
                if isinstance(node.right, ast.Constant) and \
                        isinstance(node.right.value, str):
                    seg = node.right.value.strip("/")
                else:
                    return None
                return root, f"{rel}/{seg}" if rel else seg
            if r[0] is not None and r[0] != root:
                return None
            seg = r[1]
            return root, f"{rel}/{seg}" if rel and seg else (rel or seg)
        if isinstance(node, ast.Call):
            fn = func_name(node)
            if fn == "Path" and node.args:
                return self._resolve(node.args[0])
            if fn in ("str", "abspath", "normpath", "expanduser", "realpath") \
                    and node.args:
                return self._resolve(node.args[0])
            if fn == "join" and node.args:
                first = self._resolve(node.args[0])
                if first is None:
                    return None
                root, rel = first
                segs = []
                for a in node.args[1:]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        segs.append(a.value.strip("/"))
                    else:
                        r2 = self._resolve(a)
                        if r2 is None:
                            return None
                        segs.append(r2[1].strip("/"))
                tail = "/".join(s for s in segs if s)
                return root, f"{rel}/{tail}" if rel and tail else (rel or tail)
        if isinstance(node, ast.JoinedStr):
            segs = []
            root = None
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    segs.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    r2 = self._resolve(v.value)
                    if r2 is None:
                        return None
                    root = r2[0] or root
                    segs.append(r2[1])
                else:
                    return None
            joined = "".join(segs)
            if root is None:
                return None
            return root, joined.strip("/")
        return None

    # -- use graph ---------------------------------------------------------
    def _use(self, name: str, node: ast.AST):
        sink = self._sink_of(node)
        if sink.startswith("COMPOSE:"):
            self.uses.setdefault(name, []).append(
                ("COMPOSE", sink.split(":", 1)[1]))
        else:
            self.uses.setdefault(name, []).append((sink, None))

    def _sink_of(self, node: ast.AST) -> str:
        cur = node
        for _ in range(16):
            p = self.parents.get(id(cur))
            if p is None:
                return "VALUE"
            if isinstance(p, ast.keyword):
                cur = p
                continue
            if isinstance(p, (ast.List, ast.Tuple, ast.Set, ast.Dict,
                              ast.IfExp, ast.BoolOp, ast.Compare,
                              ast.NamedExpr, ast.Starred)):
                cur = p
                continue
            if isinstance(p, ast.Call):
                fn = func_name(p)
                if fn == "insert" and len(p.args) == 2:
                    if p.args[1] is cur or self._contains(cur, p.args[1]):
                        return "PATH"
                if fn in IO_FUNCS or fn in IO_METHODS or fn in EXEC_FUNCS:
                    return "IO"
                if fn in STR_FUNCS:
                    cur = p
                    continue
                return "ARG"
            if isinstance(p, ast.Attribute):
                if p.attr in IO_METHODS:
                    return "IO"
                if p.attr in STAT_METHODS:
                    return "VALUE"
                cur = p
                continue
            if isinstance(p, ast.BinOp) and isinstance(p.op, ast.Div):
                cur = p
                continue
            if isinstance(p, ast.JoinedStr):
                cur = p
                continue
            if isinstance(p, ast.FormattedValue):
                cur = p
                continue
            if isinstance(p, ast.Assign):
                for t in p.targets:
                    if isinstance(t, ast.Name):
                        return "COMPOSE:" + t.id
                return "VALUE"
            if isinstance(p, ast.Return):
                return "VALUE"
            return "VALUE"
        return "VALUE"

    def _record(self, line: int, name: str | None, root: str, rel: str,
                kind: str, literal: str | None):
        key = (line, name or "-", rel, kind)
        if key in self.hits:
            return
        self.hits[key] = {"file": self.rel, "line": line, "name": name,
                          "root": root, "rel": rel, "kind": kind,
                          "literal": literal}

    def terminals(self, name: str, depth: int = 0,
                  seen: set[str] | None = None) -> set[str]:
        seen = seen if seen is not None else set()
        if depth > 8 or name in seen:
            return set()
        seen.add(name)
        out: set[str] = set()
        for ctx, dep in self.uses.get(name, []):
            if ctx == "COMPOSE" and dep:
                out |= self.terminals(dep, depth + 1, seen)
            else:
                out.add(ctx)
        return out

    def classify(self) -> list[dict]:
        out = []
        for h in self.hits.values():
            rel = h["rel"]
            if rel == "":
                kind, n_under = "ROOT", 0
            elif rel in self.index:
                kind, n_under = "FILE", 1
            elif rel in self.prefixes:
                kind, n_under = "DIR", self.prefixes[rel]
            else:
                kind, n_under = "MISSING", 0
            if h["name"]:
                terms = self.terminals(h["name"])
            else:
                terms = {"INLINE"}
            io = bool(terms & {"IO", "PATH", "ARG", "INLINE"})
            if not terms:
                cls = "DEAD"
            elif kind == "ROOT":
                cls = "ROOT_BIND" if io else "VALUE_ONLY"
            elif kind == "FILE":
                cls = "IO" if io else "VALUE_ONLY"
            elif kind == "DIR":
                cls = "DIR_ANCHOR" if io else "VALUE_ONLY"
            else:
                cls = "MISSING_TARGET" if io else "VALUE_ONLY"
            h2 = dict(h)
            h2["class"] = cls
            h2["target_kind"] = kind
            h2["n_under"] = n_under
            h2["terms"] = sorted(terms)
            h2["abs"] = f"{h['root']}/{rel}" if rel else h["root"]
            h2["key"] = (h2["file"], root_id(h["root"]), rel)
            out.append(h2)
        return out


def scan_source(rel_path: str, src: str, index: set[str],
                prefixes: dict[str, int]) -> list[dict]:
    """Analyse one source string.  Returns the hits, classified.

    This is the single entry point the mutation self-test drives.
    """
    try:
        fa = FileAnalysis(rel_path, src, index, prefixes)
    except (SyntaxError, ValueError, RecursionError):
        return []
    return fa.classify()


def scan_lines(lines: list[str], rel_path: str = "<synthetic>",
               index: set[str] | None = None,
               prefixes: dict[str, int] | None = None) -> list[dict]:
    """Analyse a list of source lines.  The mutation self-test's entry point."""
    idx = index if index is not None else set()
    pfx = prefixes if prefixes is not None else {}
    return scan_source(rel_path, "\n".join(lines), idx, pfx)


def _has_root_literal(tree: ast.AST) -> bool:
    """True when some string literal in ``tree`` starts with a host root.

    Every hit is reached from such a literal (``root_of`` is applied to a
    ``Constant`` before any hit is recorded), so a file that fails this test
    cannot produce a hit and does not need the full analysis.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for r in _root_list():
                if node.value == r or node.value.startswith(r + "/"):
                    return True
    return False


# ---------------------------------------------------------------------------
# The allow-list.
#
# Each entry is (file, root_id, tail, justification):
#   file        -- repo-relative path of the tracked file
#   root_id     -- index into _root_list(); which checkout root the literal
#                  is bound to
#   tail        -- the repo-relative remainder of the literal, i.e. what the
#                  literal points at inside the checkout ("" for a bare root)
#   justification -- one line a reviewer can check
#
# The MATCH KEY is (file, root_id, tail).  The line number is deliberately
# NOT part of the key: a line moves on every unrelated edit and a key that
# pins a line makes the gate fail for a reason that has nothing to do with
# the defect.  The verbatim absolute literal is likewise not stored: the
# gate would then contain the very strings it forbids (see the module
# docstring).
#
# ZERO TOLERANCE ON THE COMPLEMENT: any defect-class hit whose key is not
# in this table fails the gate.  The table can therefore never hide a new
# violation -- only a violation that a reviewer has explicitly accepted.
# ---------------------------------------------------------------------------
ALLOWLIST: tuple[tuple[str, int, str, str], ...] = (
    ('molmetal/adapters/biolm_score_full_adapter.py', 0, 'references/BioLM-Score',
     "SHAPE=DEAD_CANDIDATE -- candidate #2 of the first-existing-wins tuple _REPO_ROOT_DEFAULT_CANDIDATES; candidate #1 is already repo-relative (Path(__file__).resolve().parents[2] / 'references' / 'BioLM-Score') and RESOLVES on disk, so this absolute candidate is never reached. Left alone per the wave's frozen adjudication rule 'the binding is DEAD'."),
    ('molmetal/scripts/biolm_score_oracle.py', 0, 'references/BioLM-Score',
     "SHAPE=DEAD_CANDIDATE -- candidate #2 of the first-existing-wins tuple _REPO_ROOT_DEFAULT_CANDIDATES; candidate #1 is already repo-relative (Path(__file__).resolve().parents[2] / 'references' / 'BioLM-Score') and RESOLVES on disk, so this absolute candidate is never reached. Left alone per the wave's frozen adjudication rule 'the binding is DEAD'."),
    ('molmetal/scripts/prepare_nci60_data.py', 0, 'data/MetalCytoToxDB.csv',
     "SHAPE=DEAD_CANDIDATE -- candidate #4 of the first-existing-wins list in _find_substrate(); candidate #1 is already repo-relative (PROJECT_ROOT / 'data' / 'MetalCytoToxDB.csv'), resolves on disk, and is the same file, so this absolute candidate is never reached. Left alone per the wave's frozen adjudication rule."),
    ('molmetal/tests/test_wf329w7_vina_alternative.py', 0, '.venv/bin/python',
     "SHAPE=DEAD_OVERWRITE -- the assignment PROJECT_VENV_PY = Path(<host root>/.venv/bin/python) is overwritten on the NEXT line by PROJECT_VENV_PY = PROJECT_ROOT / '.venv' / 'bin' / 'python'; all three readers see the second value. Left alone per the wave's frozen adjudication rule 'the binding is DEAD (assigned a host path and never read)'."),
    ('molmetal/tests/test_wf334u2_pipeline_master.py', 0, '',
     "SHAPE=ASSERTION_TEXT -- a string-containment assertion that the pipeline-master wrapper's TEXT mentions the project root. "
     "CORRECTION (2026-11-02, Wf-A269 wave-2 verify): the SUPERSEDED justification read verbatim -- "
     "\"SHAPE=ASSERTION_TEXT -- a string-containment assertion that the pipeline-master wrapper's TEXT mentions the project root. "
     "AGENTS.md §5.9 MANDATES that literal in scripts/*.sh, so the assertion is correct as written; rewriting it would change the test's semantics "
     "(the wave's HARD PROHIBITION). NOTE: this is the ONE assertion that would break if the §5.9 amendment is ever authorized.\" "
     "-- is WRONG on its premise, and the premise is preserved above verbatim per AGENTS.md §5.5 rule #6. "
     "AGENTS.md §5.9 mandates the VENV interpreter path inside scripts/launch_*.sh; it does NOT mandate the bare project root, and the asserted file "
     "scripts/gpu_pipeline_master.sh is not a launch_*.sh at all. The CORRECT reason to leave the literal alone: it is the master's own COMMITTED "
     "CONTENT (the master cd's to the project root), so the assertion is a text-equality check over what the master actually says; rewriting it "
     "means rewriting the master, and the §5.9 amendment that would do that is USER-gated. The DISPOSITION (leave it alone) is unchanged."),
    ('molmetal/tests/test_wf341p4_wf264_run_once.py', 0, '.venv/bin/python',
     "SHAPE=ASSERTION_TEXT -- a string-containment assertion that the wrapper's TEXT names the canonical venv interpreter. "
     "CORRECTION (2026-11-02, Wf-A269 wave-2 verify): the SUPERSEDED justification read verbatim -- "
     "\"SHAPE=ASSERTION_TEXT -- a string-containment assertion that the wrapper's TEXT names the canonical venv interpreter. "
     "AGENTS.md §5.9 MANDATES that literal in scripts/*.sh. NOTE: the assertion already accepts the portable form ('${PY:-' + '.venv/bin/python'), "
     "so it does not block the §5.9 amendment.\" -- mis-cites §5.9: the asserted file scripts/run_wf264_cfm_property_panel.sh is NOT a launch_*.sh, "
     "and §5.9 mandates the venv path in the launchers rather than in this wrapper. The corrected premise is preserved above verbatim per "
     "AGENTS.md §5.5 rule #6. The CORRECT reason to leave the literal alone: it is the wrapper's own COMMITTED CONTENT, so the assertion checks "
     "what the wrapper actually says; rewriting it means rewriting the wrapper, which is part of the USER-gated §5.9 amendment. "
     "The sub-claim that the assertion already accepts the portable form SURVIVES the correction and is unchanged."),
    ('paper/figures/fig4_homotype_vs_tanimoto.py', 1, 'molmetal/reports/wf_lambda2e_compare/pairs.csv',
     'SHAPE=PAPER_GATED -- under paper/, which AGENTS.md §5.3 (TODO-84) forbids touching without explicit user authorization. Deliberately NOT repaired; recorded here as a USER-gated residual.'),
    ('tests/test_headline_metric_loader.py', 0, '.venv/bin/python',
     "SHAPE=UNREACHABLE_ASSERTION_TEXT -- CORRECTION (2026-11-02, Wf-A269 wave-2 verify): the SUPERSEDED justification read verbatim -- "
     "\"SHAPE=PROVENANCE_ASSERTION -- asserts over the PUBLISHED metric artifact metrics/by_metric/metal_lomo_auc_wasserstein.json, whose "
     "'clean_re_measurement_command' field records the literal command a past MEASURED run actually executed. The recorded path is a fact about that run; "
     "rewriting it (or the assertion) would falsify a published record, and metrics/** is outside every agent's scope.\" "
     "-- is FACTUALLY FALSE and is preserved above verbatim per AGENTS.md §5.5 rule #6. "
     "MEASURED: the field 'clean_re_measurement_command' exists in NO metric artifact (grep over metrics/ returns 0 hits; the JSON has no such key), "
     "so the test subscripts it and raises KeyError BEFORE it reaches the assertion that holds this literal -- the literal is UNREACHABLE dead text in "
     "an already-failing test, not a provenance record. The DISPOSITION (leave it alone) is unchanged and for a different reason: the binding is dead, "
     "metrics/** is out of scope, and no live path depends on it."),
)

ALLOWLIST_KEYS = frozenset((f, rid, tail) for f, rid, tail, _ in ALLOWLIST)


# ---------------------------------------------------------------------------
# The repo-wide scan
# ---------------------------------------------------------------------------
_SCAN_CACHE: dict[str, object] = {}


def _scan(repo: Path = REPO, refresh: bool = False) -> dict:
    """Scan every tracked ``.py`` file.  Cached per process.

    The cache is keyed by the repository path, so it is only valid while the
    working tree is unchanged.  Pass ``refresh=True`` after the tree has
    been modified.
    """
    key = str(repo)
    if not refresh and key in _SCAN_CACHE:
        return _SCAN_CACHE[key]  # type: ignore[return-value]
    index = set(git_tracked(repo))
    prefixes = prefix_map(index)
    tracked = git_tracked_py(repo)
    roots = list(_root_list())
    candidates = set(git_grep_files(repo, roots))
    hits: list[dict] = []
    analyzed = 0
    for rel in tracked:
        if rel not in candidates:
            continue
        p = repo / rel
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if not _has_root_literal(tree):
            continue
        analyzed += 1
        hits.extend(scan_source(rel, src, index, prefixes))
    out = {
        "tracked_py": len(tracked),
        "candidates": len(candidates & set(tracked)),
        "analyzed": analyzed,
        "hits": hits,
        "index": index,
        "prefixes": prefixes,
    }
    _SCAN_CACHE[key] = out
    return out


def defect_hits(repo: Path = REPO, refresh: bool = False) -> list[dict]:
    """Every defect-class hit in the tracked ``.py`` corpus."""
    return [h for h in _scan(repo, refresh=refresh)["hits"]
            if h["class"] in DEFECT_CLASSES]


# ===========================================================================
# TESTS
# ===========================================================================
def pytest_skip(reason: str):
    """Minimal skip helper (stdlib only; no pytest import at module scope)."""
    try:
        import pytest
    except Exception:  # pragma: no cover
        return
    pytest.skip(reason)


def test_gate_scans_the_tracked_python_universe():
    """The scan universe is the git index -- never an os.walk of 201 GB."""
    tracked = git_tracked_py(REPO)
    assert len(tracked) > 1000, (
        f"expected the tracked .py corpus to be large, got {len(tracked)}")
    scan = _scan(REPO)
    assert scan["tracked_py"] == len(tracked)
    assert scan["analyzed"] <= scan["candidates"] <= scan["tracked_py"]


def test_prefilter_partition_is_exact():
    """The git-grep prefilter must partition the tracked .py corpus exactly.

    A file that mentions no host root at all cannot produce a hit, because
    every hit is reached from a string literal that *starts* with a host
    root.  This pins that the prefilter drops nothing.
    """
    tracked = set(git_tracked_py(REPO))
    hits = set(git_grep_files(REPO, list(_root_list())))
    cmd = ["git", "grep", "-L", "-F", "-a"]
    for pat in _root_list():
        cmd += ["-e", pat]
    cmd += ["--", "*.py"]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         check=False)
    if res.returncode not in (0, 1):
        pytest_skip("git grep -L is unavailable here")
    misses = {l for l in res.stdout.splitlines() if l.strip()}
    assert not (hits & misses), (
        "git grep -l and -L disagree on the prefilter partition")
    assert (hits | misses) >= tracked, (
        "git grep -l/-L did not cover every tracked .py file")


def test_no_unallowlisted_host_bound_path():
    """THE GATE: zero tolerance on the allow-list's complement."""
    hits = defect_hits(REPO)
    unallowlisted = [h for h in hits if h["key"] not in ALLOWLIST_KEYS]
    if unallowlisted:
        lines = [
            f"  {h['file']}:{h['line']}  [{h['class']}/{h['target_kind']}]"
            f"  {h['abs']}"
            for h in sorted(unallowlisted, key=lambda h: (h["file"], h["line"]))
        ]
        raise AssertionError(
            f"{len(unallowlisted)} host-bound path violation(s) not in the "
            f"allow-list ({len(ALLOWLIST)} reviewed entries):\n"
            + "\n".join(lines[:40])
            + ("\n  ..." if len(lines) > 40 else "")
            + "\nFix the path (derive it from __file__ / sys.executable) or "
              "add an allow-list entry with a justification."
        )


def test_allowlist_entries_are_wellformed():
    """Every allow-list entry names a tracked file and carries a reason."""
    tracked = set(git_tracked(REPO))
    seen = set()
    for entry in ALLOWLIST:
        assert len(entry) == 4, f"allow-list entry is not a 4-tuple: {entry}"
        file, rid, tail, why = entry
        assert file in tracked, (
            f"allow-list entry names an untracked file: {file}")
        assert isinstance(rid, int) and 0 <= rid < len(_root_list()), (
            f"allow-list entry has an out-of-range root id: {entry}")
        assert isinstance(tail, str), f"allow-list tail is not a string: {entry}"
        assert why and why.strip(), (
            f"allow-list entry has no justification: {file} :: {tail}")
        assert len(why.splitlines()) == 1, (
            f"allow-list justification must be one line: {file} :: {tail}")
        key = (file, rid, tail)
        assert key not in seen, f"duplicate allow-list entry: {key}"
        seen.add(key)


def test_allowlist_key_is_scoped_to_its_own_file():
    """An allow-list entry must not leak across files.

    The match key is (file, root_id, tail).  If the file component were
    dropped, one reviewed entry would silently authorise the same target in
    EVERY file in the repo -- the allow-list would become a blanket waiver.
    """
    index = set(git_tracked(REPO))
    prefixes = prefix_map(index)
    live = str(REPO)
    tracked = sorted(p for p in git_tracked_py(REPO) if p.endswith(".py"))
    target = tracked[0]
    lines = [
        'P = "%s/%s"' % (live, target),
        "with open(P) as fh:",
        "    fh.read()",
    ]
    a = scan_lines(lines, rel_path="pkg/alpha.py", index=index,
                   prefixes=prefixes)
    b = scan_lines(lines, rel_path="pkg/beta.py", index=index,
                   prefixes=prefixes)
    ka = {h["key"] for h in a if h["class"] in DEFECT_CLASSES}
    kb = {h["key"] for h in b if h["class"] in DEFECT_CLASSES}
    assert ka and kb, "the synthetic violation was not detected at all"
    assert not (ka & kb), (
        f"the allow-list key is not scoped to its file: {ka} == {kb}")
    assert all(k[0] == "pkg/alpha.py" for k in ka)
    assert all(k[0] == "pkg/beta.py" for k in kb)


def test_gate_would_fail_on_a_reintroduced_path():
    """The non-recurrence requirement: re-introduce the defect -> FAIL.

    Two distinct reintroductions are simulated:

    (a) a host-bound path in a file that has NO allow-list entry -- it must
        land outside the allow-list and therefore fail the gate;
    (b) a *different* host-bound path in a file that DOES have an
        allow-list entry -- the entry is scoped to one target, so the new
        target must also land outside the allow-list.
    """
    index = set(git_tracked(REPO))
    prefixes = prefix_map(index)
    live = str(REPO)
    tracked = sorted(p for p in git_tracked_py(REPO) if p.endswith(".py"))
    target = tracked[0]
    lines = [
        'P = "%s/%s"' % (live, target),
        "with open(P) as fh:",
        "    fh.read()",
    ]

    # (a) a file with no allow-list entry at all
    unlisted = "pkg/brand_new_module.py"
    assert unlisted not in {f for f, _, _, _ in ALLOWLIST}, (
        "the simulation file is already allow-listed; pick another name")
    hits = scan_lines(lines, rel_path=unlisted, index=index,
                      prefixes=prefixes)
    defect = [h for h in hits if h["class"] in DEFECT_CLASSES]
    assert defect, "the reintroduced violation was not detected at all"
    assert all(h["key"] not in ALLOWLIST_KEYS for h in defect), (
        "a reintroduced violation landed inside the allow-list -- the "
        "allow-list is not enforcing zero tolerance")

    # (b) a NEW target in a file that is already allow-listed
    listed = sorted({f for f, _, _, _ in ALLOWLIST})
    if listed:
        other = tracked[-1]
        assert other != target
        hits = scan_lines(
            ['P = "%s/%s"' % (live, other), "with open(P) as fh:",
             "    fh.read()"],
            rel_path=listed[0], index=index, prefixes=prefixes)
        defect = [h for h in hits if h["class"] in DEFECT_CLASSES]
        assert defect, "the reintroduced violation was not detected at all"
        assert all(h["key"] not in ALLOWLIST_KEYS for h in defect), (
            f"a NEW target in the already-allow-listed file {listed[0]} "
            "landed inside the allow-list -- the entry is not target-scoped")


def test_gate_end_to_end_on_a_scratch_repo(tmp_path):
    """End-to-end: a scratch repo that contains the defect MUST come out red.

    This exercises the whole pipeline -- ``git ls-files`` index, the
    directory prefix map, the classification, and the allow-list decision --
    on a controlled fixture.  The scratch repo lives in pytest's
    ``tmp_path``; nothing is written into this repository.
    """
    import os

    scratch = tmp_path / "scratch_repo"
    (scratch / "pkg").mkdir(parents=True)
    env = dict(os.environ,
               GIT_AUTHOR_NAME="gate", GIT_AUTHOR_EMAIL="gate@example.invalid",
               GIT_COMMITTER_NAME="gate",
               GIT_COMMITTER_EMAIL="gate@example.invalid")

    def git(*args):
        return subprocess.run(["git", *args], cwd=scratch, capture_output=True,
                              text=True, env=env, check=False)

    if git("init", "-q").returncode != 0:
        pytest_skip("git init is unavailable in this environment")
    (scratch / "pkg" / "bad.py").write_text(
        'P = "%s/pkg/data.csv"\n'
        "with open(P) as fh:\n"
        "    fh.read()\n" % str(REPO), encoding="utf-8")
    (scratch / "pkg" / "data.csv").write_text("x\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "fixture")

    hits = defect_hits(scratch)
    assert hits, (
        "MUTATION SELF-TEST FAILED: the end-to-end scan of a scratch repo "
        "containing a host-bound path found nothing")
    assert any(h["class"] == "IO" for h in hits), (
        f"the scratch-repo violation did not classify as IO: {hits}")
    assert all(h["key"] not in ALLOWLIST_KEYS for h in hits), (
        "the scratch-repo violation landed inside the allow-list -- the "
        "allow-list is not enforcing zero tolerance")

    # The same scratch repo, repaired, must come out green.
    (scratch / "pkg" / "bad.py").write_text(
        "from pathlib import Path\n"
        'P = Path(__file__).resolve().parents[1] / "pkg" / "data.csv"\n'
        "with open(P) as fh:\n"
        "    fh.read()\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "repaired")
    assert not defect_hits(scratch, refresh=True), (
        "MUTATION SELF-TEST FAILED: the repaired scratch repo is still red")


def test_gate_file_holds_no_unapproved_host_root_literal():
    """The gate may only name the three roots it forbids -- nothing else.

    This is what keeps the gate's own file honest: it cannot smuggle a
    second host path past itself, and it cannot grow an allow-list that
    stores the forbidden literals verbatim.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    approved_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and \
                isinstance(node.value, ast.Constant) and \
                isinstance(node.value.value, str) and \
                node.value.value in _ROOTS:
            approved_lines.add(node.lineno)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or \
                not isinstance(node.value, str):
            continue
        if root_of(node.value) is None:
            continue
        if node.lineno in approved_lines:
            continue
        offenders.append((node.lineno, node.value))
    assert not offenders, (
        "the gate file holds host-root literals outside its three root "
        f"constants: {offenders}")


def test_mutation_self_test_detects_a_synthetic_violation():
    """The gate must BITE.  A synthetic violation must be returned.

    The synthetic module is built at run time from the live checkout root
    so that this file never holds a forbidden literal; it is written to
    pytest's ``tmp_path`` and never into the repository.
    """
    index = set(git_tracked(REPO))
    prefixes = prefix_map(index)
    tracked = [p for p in git_tracked_py(REPO) if p.endswith(".py")]
    assert tracked, "no tracked .py file to use as a synthetic target"
    target = sorted(tracked)[0]
    live = str(REPO)
    bad = [
        f'P = "{live}/{target}"',
        "with open(P) as fh:",
        "    fh.read()",
    ]
    hits = scan_lines(bad, rel_path="<synthetic>", index=index,
                      prefixes=prefixes)
    defect = [h for h in hits if h["class"] in DEFECT_CLASSES]
    assert defect, (
        "MUTATION SELF-TEST FAILED: the detector did not flag a synthetic "
        f"host-bound path that opens the tracked file {target}")
    assert any(h["rel"] == target for h in defect), (
        f"synthetic violation was flagged but not against {target}: {defect}")
    assert any(h["class"] == "IO" for h in defect), (
        f"synthetic violation should classify as IO, got {defect}")


def test_mutation_self_test_accepts_synthetic_non_violations():
    """The gate must not over-fire.  Each near-miss shape must be clean."""
    index = set(git_tracked(REPO))
    prefixes = prefix_map(index)
    live = str(REPO)
    tracked = sorted(p for p in git_tracked_py(REPO) if p.endswith(".py"))
    target = tracked[0]
    cases = {
        "external data path": [
            'P = "/mnt/storage/external/dataset.csv"',
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "another checkout root": [
            'P = "/opt/some/other/checkout/data.csv"',
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "docstring mention": [
            '"""Historically this opened %s/%s."""' % (live, target),
            "x = 1",
        ],
        "comment mention": [
            "# once opened %s/%s" % (live, target),
            "x = 1",
        ],
        "non-I-O literal (comparison only)": [
            'P = "%s/%s"' % (live, target),
            "if P == other:",
            "    pass",
        ],
        "stat-only existence check": [
            'P = "%s/%s"' % (live, target),
            "if P.exists():",
            "    pass",
        ],
        "dead constant": [
            'P = "%s/%s"' % (live, target),
        ],
        "portable interpreter": [
            "import sys",
            "subprocess.run([sys.executable, 'x.py'])",
        ],
        "repo-relative path": [
            'P = "%s"' % target,
            "with open(P) as fh:",
            "    fh.read()",
        ],
    }
    for name, lines in cases.items():
        hits = scan_lines(lines, rel_path="<synthetic>", index=index,
                          prefixes=prefixes)
        defect = [h for h in hits if h["class"] in DEFECT_CLASSES]
        assert not defect, (
            f"MUTATION SELF-TEST FAILED: {name!r} was flagged: {defect}")


def test_detection_rule_covers_every_path_form():
    """All four path forms must be detected."""
    index = set(git_tracked(REPO))
    prefixes = prefix_map(index)
    live = str(REPO)
    tracked = sorted(p for p in git_tracked_py(REPO) if p.endswith(".py"))
    target = tracked[0]
    forms = {
        "bare string literal": [
            'P = "%s/%s"' % (live, target),
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "os.path.join": [
            "import os",
            'P = os.path.join("%s", "%s")' % (live, target),
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "Path(...)": [
            "from pathlib import Path",
            'P = Path("%s/%s")' % (live, target),
            "with open(P) as fh:",
            "    fh.read()",
        ],
        # The synthetic source is assembled at run time so that THIS file
        # never holds the forbidden literal; at run time it becomes
        #   P = f"<live root>/{target}"
        # i.e. an f-string whose PREFIX is an absolute host path.
        "f-string with an absolute prefix": [
            'P = f"%s/{target}"' % live,
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "root divided by a tail": [
            'R = "%s"' % live,
            'P = R / "%s"' % target,
            "with open(P) as fh:",
            "    fh.read()",
        ],
        "composed through two constants": [
            'R = "%s"' % live,
            'D = R / "%s"' % target,
            "with open(D) as fh:",
            "    fh.read()",
        ],
    }
    for name, lines in forms.items():
        hits = scan_lines(lines, rel_path="<synthetic>", index=index,
                          prefixes=prefixes)
        defect = [h for h in hits if h["class"] in DEFECT_CLASSES]
        assert defect, (
            f"MUTATION SELF-TEST FAILED: path form {name!r} was not detected")


def test_allowlist_stale_entries_are_reported():
    """Stale allow-list entries are review debt, not a failure.

    A stale entry means the repair wave removed a violation that had been
    reviewed; the entry should be dropped so that re-introducing the same
    path in the same file is caught again.  It is reported (not fatal)
    because the repair wave lands asynchronously and a red gate over a
    just-repaired file would be noise.
    """
    live_keys = {h["key"] for h in defect_hits(REPO)}
    stale = [k for k in ALLOWLIST_KEYS if k not in live_keys]
    print(f"\nWf-A269 gate: {len(live_keys)} live defect hit(s), "
          f"{len(ALLOWLIST_KEYS)} allow-listed key(s), "
          f"{len(stale)} stale entr(ies)")
    for k in sorted(stale):
        print(f"  STALE allow-list entry (drop it): {k}")
    assert True


if __name__ == "__main__":  # pragma: no cover
    if "--emit-allowlist" in sys.argv:
        # Maintenance mode: print the allow-list entries that the CURRENT
        # tree needs, so the table can be refreshed after a repair wave.
        # The justifications are proposals -- a reviewer must check them
        # before they are pasted back in.
        keys = {}
        for h in defect_hits(REPO):
            keys.setdefault(h["key"], h)
        for k in sorted(keys):
            print("    (%r, %d, %r," % (k[0], k[1], k[2]))
            print("     'REVIEW: see the hit in %s')," % k[0])
        print("# entries: %d" % len(keys), file=sys.stderr)
        sys.exit(0)
    hits = defect_hits(REPO)
    allowlisted = [h for h in hits if h["key"] in ALLOWLIST_KEYS]
    unallow = [h for h in hits if h["key"] not in ALLOWLIST_KEYS]
    print(f"tracked .py files : {_scan(REPO)['tracked_py']}")
    print(f"prefilter cands   : {_scan(REPO)['candidates']}")
    print(f"files analysed    : {_scan(REPO)['analyzed']}")
    print(f"defect hits       : {len(hits)}")
    print(f"allow-listed      : {len(allowlisted)}")
    print(f"NOT allow-listed  : {len(unallow)}")
    for h in sorted(unallow, key=lambda h: (h["file"], h["line"])):
        print(f"  {h['file']}:{h['line']}  [{h['class']}]  {h['abs']}")
    sys.exit(1 if unallow else 0)
