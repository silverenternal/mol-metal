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

"""Wf-A269 wave-3 -- the zero-tolerance gate for DANGLING launcher paths.

THE DEFECT CLASS
----------------
The Wf-437 simplification wave moved ten launchers out of ``scripts/`` into
``TODO/archive/wf437_simplify/`` (and a few into ``scripts/_retired/``).  It did
not update the callers.  The result is a tracked file that names a launcher
path which does not exist: a shell ``nohup bash <missing>`` dies with "No such
file or directory", a ``[ -x <missing> ] || exit 1`` guard exits 1 forever, and
a test that shells out to it exercises nothing.

THE RULE
--------
    Every ``launch_<n>.sh`` PATH reference in live code must resolve to a
    tracked file.

"PATH reference" is load-bearing: a reference carries a directory.  A bare
NAME (``launch_ok.sh``) names no path -- those are test fixtures, template
placeholders, candidate-path lists and historical rename records, and they are
OUT OF SCOPE (see COVERAGE BOUND below).

TWO SCANNERS, BECAUSE ONE IS BLIND
----------------------------------
A **token scan** (``TOKEN_RE``) finds every path-shaped *string* on a line.
It is blind to the second shape:

    LAUNCHER = REPO_ROOT / "molmetal" / "scripts" / "launch_x.sh"

Here the path is never written as one string, so no regex can see it, and a
substring rewriter cannot rewrite it either.  MEASURED on 2026-11-02 by the
Wf-A269 wave-3 adversarial review: three such composed references
(``test_wf321h_todo_119_p14_cpu_scaffold.py``,
``test_wf408a3_preflight_launcher_fixes.py``, ``test_wf415a1_wrapper_args.py``)
were dangling, invisible to the token scan, and would have survived a plan
whose whole premise was "rewrite the literal so it resolves".

So this gate runs **two** scans:

* ``_scan_tokens()``  -- regex over the raw text (catches shell, comments,
  docstrings, and any Python path that is written as one string);
* ``_scan_composed()`` -- an **AST** scan over tracked ``.py`` files that
  reconstructs ``X / "launch_*.sh"`` chains (resolving module-level constants
  such as ``SCRIPTS = ROOT / "scripts"``) and checks the reconstructed
  repo-relative path against the tracked set.

MEASURED on 2026-11-02 (CPU-only, 0 GPU) by the Wf-A269 wave-3 audit:
* the pre-repair census found **84** dangling sites in **37** tracked files
  (83 literal + 1 composed), across four classes: the target lives at
  ``scripts/`` (32), at ``TODO/archive/wf437_simplify/`` (43), at
  ``scripts/_retired/`` (2), or nowhere (6);
* the post-repair state measured by THIS gate: **0** dangling path references
  outside the allow-lists, over **726** path tokens scanned in live code plus
  **90** composed references reconstructed by the AST scan.

THE ALLOW-LISTS
---------------
A reference that does not resolve is a defect UNLESS its SHAPE says otherwise.
Two allow-lists record the shapes, each entry carrying a one-line
justification, and each is reported when it goes stale.

* ``ALLOWLIST`` -- token-scan entries.
* ``COMPOSED_ALLOWLIST`` -- AST-scan entries.

A third shape is detected **structurally** rather than listed: a composed
reference that is asserted NOT to exist (``assert not X.exists()``) is a
*negative* assertion -- the test is checking that a path is gone.  Those are
resolved by name at scan time, so a new one is caught automatically instead of
having to be allow-listed.

THE HOST-BOUND CLASS IS NOT THIS CLASS
--------------------------------------
A path anchored directly on a host checkout root
(``/home/hugo/codes/mol-metal/...``) is a *host-bound path* and is owned by
``test_wfa269_no_hardcoded_repo_paths.py``.  This gate RECORDS those and
does not fail on them; the two gates cover disjoint classes.

COVERAGE BOUND (stated, because a census is not a defect count)
----------------------------------------------------------------
The gate scans tracked ``.py``/``.sh`` outside ``TODO/``,
``molmetal/reports/`` and ``paper/``.  It does NOT cover: untracked files;
``.md``/``.yaml``/``.tex`` documents; bare launcher NAMES; paths to
non-``launch_*`` scripts that were moved by the same wave; and composed
references in **shell** (the AST scan is Python-only -- shell composition is
covered by the token scan only when the path is written as one string).
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The three host checkout roots this repo has ever been cloned under.
HOST_ROOTS = (
    "/home/hugo/codes/mol-metal",
    "/home/hugo/codes/try_triton_on_rocm",
    "/mnt/disk1/Hugo_Lee/mol-metal",
)

# Variables that stand for the repository root in live code.  A path token
# that begins with one of them is repo-relative once the prefix is removed.
ROOT_VARS = (
    "PROJECT_ROOT", "REPO_ROOT", "ROOT", "ROOT_DIR", "BASE_DIR",
    "PROJECT_DIR", "REPO_DIR", "SCRIPTS_ROOT",
)

# A path token: a maximal run of path-ish characters ending in launch_*.sh
TOKEN_RE = re.compile(r"[A-Za-z0-9_${}./\-]*launch_[A-Za-z0-9_]*\.sh")

EXCLUDED_PREFIXES = ("TODO/", "molmetal/reports/", "paper/")

# (file, token) -> the SHAPE that makes it an allowed non-defect.
ALLOWLIST: dict[tuple[str, str], str] = {
    ("molmetal/tests/test_wf_launcher_hardening_gate.py",
     "scripts/launch_child.sh"):
        "RUNTIME FIXTURE -- the test creates this file at runtime to "
        "exercise the hardening gate; it is not a repo launcher.",
    ("scripts/_launcher_hardening_gate.py",
     "scripts/launch_x.sh"):
        "DOCSTRING EXAMPLE -- a placeholder inside the gate's own "
        "docstring, not a reference to a repo launcher.",
    # --- Wf-A269 wave-3 (2026-11-02): the §5.5 rule-#6 shape ------------
    # AGENTS.md §5.5 rule #6 REQUIRES a superseded rendering to be
    # preserved VERBATIM -- and several of those renderings name a
    # launcher path that no longer resolves (that is what makes it
    # superseded).  Without this shape the wave-3 diff had to DELETE the
    # historical paths to pass this gate, which is the §5.5 rule-#6
    # violation the L1 arm found.  Every entry below is checked by
    # ``test_every_allowed_historical_rendering_carries_the_rule6_marker``:
    # the file must carry the explicit preservation marker.
    ("molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh",
     "molmetal/scripts/launch_wf299_cfm_main.sh"):
        "HISTORICAL RENDERING -- the Wf-299 candidate record's notes "
        "string quotes the pre-move path (the file was never committed "
        "under molmetal/scripts/); the rendering is preserved verbatim "
        "per AGENTS.md §5.5 rule #6.",
    ("molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py",
     "scripts/launch_wf300_cfm_main.sh"):
        "HISTORICAL RENDERING -- the comment records the path the Wf-300C "
        "assertions used to read (it resolves to nothing); preserved "
        "verbatim per AGENTS.md §5.5 rule #6.",
    ("molmetal/tests/test_wf336d3_pipeline_integrate.py",
     "molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh"):
        "HISTORICAL RENDERING -- the docstring records the path this pin "
        "used to demand; preserved verbatim per AGENTS.md §5.5 rule #6.",
    ("molmetal/tests/test_wf337u2_wf135_user_ok.py",
     "molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh"):
        "HISTORICAL RENDERING -- the pre-fix bug record names the path "
        "the pipeline master used to reference (FILE NOT FOUND); "
        "preserved verbatim per AGENTS.md §5.5 rule #6.",
    # --- Wf-A269 wave-3 L2-4 (2026-11-02) ----------------------------------
    # The L2 contract repairs in ``test_wf341p1`` each had to name the path
    # the assertion used to demand, and that path resolves to nothing (it is
    # what makes the rendering superseded).  Both entries below are the
    # §5.5 rule-#6 shape: the file carries the marker (3 occurrences).
    ("molmetal/tests/test_wf341p1_pipeline_master_integration.py",
     "scripts/launch_wf124_p3_g4.sh"):
        "HISTORICAL RENDERING -- the docstring records the path this pin "
        "used to demand (the launcher now lives in the archive, so the "
        "scripts/ path resolves to nothing); preserved verbatim per "
        "AGENTS.md §5.5 rule #6.",
    ("molmetal/tests/test_wf341p1_pipeline_master_integration.py",
     "molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh"):
        "HISTORICAL RENDERING -- the pre-fix bug record names the path this "
        "composed reference used to demand (the launcher is at scripts/); "
        "preserved verbatim per AGENTS.md §5.5 rule #6.",
    # --- Wf-A269 wave-5 (2026-11-02) ---------------------------------------
    # The D5 repair of scripts/launch_wf306a_drugood_vina_gpu.sh corrected
    # that launcher's USAGE block (the old block named only
    # WF306A_USER_OK=YES, so the documented command was refused with exit
    # 64).  AGENTS.md §5.5 rule #6 then requires the OLD rendering to be
    # preserved VERBATIM -- and that rendering names a launcher path this
    # file NEVER occupied: git MEASURED, the launcher's only committed
    # location is TODO/archive/wf437_simplify/launch_wf306a_drugood_vina_gpu.sh
    # (added in bc29aa90); it was never at molmetal/scripts/.  So the
    # preserved rendering is a dead launcher path BY CONSTRUCTION -- the
    # same shape the six entries above record.
    ("scripts/launch_wf306a_drugood_vina_gpu.sh",
     "molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh"):
        "HISTORICAL RENDERING -- the pre-D5 USAGE block names the path the "
        "launcher never occupied (it was added at "
        "TODO/archive/wf437_simplify/ and was never at molmetal/scripts/); "
        "preserved verbatim per AGENTS.md §5.5 rule #6.",
    # --- Wf-A269 wave-5 (2026-11-02), verify:t179 REFUTATION ---------------
    # The wave-5 adversarial verifier found the §5.5 rule-#6 chain broken
    # END-TO-END on this launcher: the git-recorded ORIGINAL rendering named
    # `bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh`, wave-3 re-worded
    # it to `bash scripts/...` WITHOUT preserving the original, and wave-5
    # preserved only wave-3's rendering.
    # MEASURED: `git log --all -- 'molmetal/scripts/launch_wf310c_t179_w2_gpu.sh'`
    # returns NOTHING -- that path never existed.  The launcher's only
    # committed location is the TODO/archive/wf437_simplify/ directory
    # (added in bc29aa90).  So the restored rendering is a dead launcher path
    # BY CONSTRUCTION -- the same shape as the 306a entry above.
    # (The directory is named, not the full archived file path, for the same
    # reason the 306a entry does it: the gate's own token scan reads this
    # test file, so a full dead path written here would make the gate
    # flag ITSELF.)
    ("scripts/launch_wf310c_t179_w2_gpu.sh",
     "molmetal/scripts/launch_wf310c_t179_w2_gpu.sh"):
        "HISTORICAL RENDERING -- the pre-wave-3 ORIGINAL USAGE block names "
        "the path the launcher never occupied (it was added at "
        "TODO/archive/wf437_simplify/ and was never at molmetal/scripts/); "
        "preserved verbatim per AGENTS.md §5.5 rule #6.",
    # --- Wf-A269 wave-19 (2026-11-04), verify:t310f ------------------------
    # The 11th entry, and the FIRST one whose launcher is NOT archived: the
    # wf310f launcher is LIVE at scripts/launch_wf310f_frontier_math_gpu.sh
    # (added in bc29aa90 -- the same commit that added the six wave-3
    # launchers).  MEASURED: `git log --all --` for
    # `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh` returns NOTHING,
    # so that path NEVER existed.  The launcher's USAGE block named it from
    # the start (`git log --all -S 'bash scripts/launch_wf310f'` returns
    # NOTHING, so the block was NEVER re-worded) -- which makes it a LIVE
    # defect, not a superseded rendering: the documented command did not
    # resolve.  The usage block is now re-worded to `bash scripts/...` (the
    # form every other live wrapper uses) and the original rendering is
    # preserved VERBATIM in the launcher's dated comment block.
    ("scripts/launch_wf310f_frontier_math_gpu.sh",
     "molmetal/scripts/launch_wf310f_frontier_math_gpu.sh"):
        "HISTORICAL RENDERING -- the ORIGINAL USAGE block names the path the "
        "launcher never occupied (it was added at scripts/ in bc29aa90 and "
        "was never at molmetal/scripts/); preserved verbatim per AGENTS.md "
        "§5.5 rule #6.",
}

#: The preservation marker an allow-listed HISTORICAL RENDERING must carry.
#: A file that names a dead launcher WITHOUT this marker still fails the
#: gate -- the allow-list cannot become a general hole for dead paths.
RULE6_MARKER = "\u00a75.5 rule #6"

# (file, repo-relative path) -> the SHAPE that makes an UNRESOLVED composed
# reference an allowed non-defect.  Every entry must name a shape that is
# decidable from the source, not a reason that sounds right.
COMPOSED_ALLOWLIST: dict[tuple[str, str], str] = {
    ("molmetal/tests/test_wf328w2_launcher_num_workers.py",
     "scripts/launch_wf124_p3_g4.sh"):
        "ABSENCE-SCAN LIST -- the file names the launcher in the TARGETS "
        "tuple and then filters it out through ARCHIVED_TARGETS "
        "(the launcher was archived by Wf-437); the test asserts the "
        "launcher is NOT present at scripts/, so the reference must "
        "name the path that is gone.",
    ("molmetal/tests/test_wf328w2_launcher_num_workers.py",
     "scripts/launch_wf300_cfm_main.sh"):
        "ABSENCE-SCAN LIST -- same shape as the entry above: listed in "
        "WF328W2_TARGETS and filtered out through ARCHIVED_TARGETS.",
    ("molmetal/tests/test_wf336f2_user_ok_gate.py",
     "scripts/launch_wf126_m19_lomo_gpu.sh"):
        "CANDIDATE LIST -- an explicit multi-path ABSENCE search: the "
        "test confirms that no Wf-126/Wf-176 GPU wrapper exists, per the "
        "HONEST-NEGATIVE recorded in the module docstring.",
    ("molmetal/tests/test_wf336f2_user_ok_gate.py",
     "scripts/launch_wf176_m19_lomo_gpu.sh"):
        "CANDIDATE LIST -- same shape as the entry above.",
    ("molmetal/tests/test_wf336f2_user_ok_gate.py",
     "molmetal/scripts/launch_wf126_m19_lomo_gpu.sh"):
        "CANDIDATE LIST -- same shape as the entry above.",
    ("molmetal/tests/test_wf336f2_user_ok_gate.py",
     "molmetal/scripts/launch_wf176_m19_lomo_gpu.sh"):
        "CANDIDATE LIST -- same shape as the entry above.",
}

# Names that stand for the repository root when they are built from
# ``__file__``.  A module-level constant whose value mentions ``__file__`` is
# detected structurally (see ``_root_names``); these are the conventional
# names used when the root comes from somewhere else.
EXTRA_ROOT_VARS = (
    "_REPO_ROOT", "MOLMETAL_ROOT", "REPO", "PROJECT_ROOT_DIR", "REPO_DIR",
)


def _tracked_live_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True, text=True, check=True,
    )
    names = [n for n in out.stdout.split("\0") if n]
    return sorted(
        n for n in names
        if n.endswith((".py", ".sh")) and not n.startswith(EXCLUDED_PREFIXES)
    )


def _tracked_set() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True, text=True, check=True,
    )
    return {n for n in out.stdout.split("\0") if n}


def _classify(token: str, tracked: set[str]) -> tuple[str, str]:
    """Return (kind, detail).

    kind in {OK, HOST, BARE, COMPOSED, DEAD}.
    """
    t = token.rstrip("\"'`),;:")
    for h in HOST_ROOTS:
        if t.startswith(h):
            return ("HOST", h)
    # strip a leading variable prefix: ${VAR}/ (shell) or {VAR}/ (f-string)
    t = re.sub(r"^(?:\$\{?|\{)(" + "|".join(ROOT_VARS) + r")\}?/", "", t)
    if t.startswith(("$", "{")):
        return ("COMPOSED", t)
    if "/" not in t:
        return ("BARE", t)
    cands = {t, t.replace("molmetal/scripts/../scripts/", "scripts/")}
    for c in cands:
        if c in tracked:
            return ("OK", c)
    return ("DEAD", t)


def _scan() -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]],
                    list[tuple[str, int, str]], int]:
    tracked = _tracked_set()
    dead: list[tuple[str, int, str]] = []
    host: list[tuple[str, int, str]] = []
    composed: list[tuple[str, int, str]] = []
    n_tokens = 0
    for rel in _tracked_live_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in TOKEN_RE.finditer(line):
                tok = m.group(0)
                n_tokens += 1
                kind, _detail = _classify(tok, tracked)
                if kind == "DEAD":
                    if (rel, tok) not in ALLOWLIST:
                        dead.append((rel, i, tok))
                elif kind == "HOST":
                    host.append((rel, i, tok))
                elif kind == "COMPOSED":
                    composed.append((rel, i, tok))
    return dead, host, composed, n_tokens


# ---------------------------------------------------------------------------
# The COMPOSED scan -- the AST half of the gate.
# ---------------------------------------------------------------------------
# Names that are pytest fixtures.  A path built from one of them lives in a
# pytest temporary directory, never in the repository, so it cannot be a
# dangling *repository* reference.
FIXTURE_ROOTS = frozenset({"tmp_path", "tmpdir", "tmp_path_factory"})


def _eval_consts(tree: ast.Module, file_path: Path) -> tuple[dict[str, object],
                                                           set[str]]:
    """Evaluate module-level constants that are pure path expressions.

    The evaluation namespace carries NO builtins, so a constant whose value
    calls anything other than ``Path`` / ``Path.resolve`` / ``Path.absolute``
    raises and is skipped.  An expression we cannot evaluate is one we must
    not guess at -- and a guessed-at root is exactly what hid five real
    sites in the first version of this scan (the ``.parent`` chain was
    flattened into a bogus segment list, which DELETED the root marker from
    the chain and made the site disappear).

    Evaluating the constant is the only exact answer available: the number
    of ``.parent`` hops decides whether the anchor is the repository root,
    a subdirectory, or the test file itself, and only the filesystem knows
    which.

    Returns ``(values, fixture_names)``; the second element holds the
    names that are derived from a pytest temp directory.
    """
    ns: dict[str, object] = {
        "__file__": str(file_path),
        "Path": pathlib.Path,
    }
    out: dict[str, object] = {}
    fixture_names: set[str] = set()
    assigns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    ]
    for _ in range(4):  # fixpoint: consts may depend on earlier consts
        progress = False
        for n in assigns:
            name = n.targets[0].id
            if name in out:
                continue
            if any(isinstance(x, ast.Name) and x.id in FIXTURE_ROOTS
                   for x in ast.walk(n.value)):
                # derived from a pytest temp dir -- not a repository path
                fixture_names.add(name)
                continue
            try:
                code = compile(ast.Expression(n.value), "<const>", "eval")
                val = eval(code, {"__builtins__": {}}, ns)
            except Exception:
                continue
            if not isinstance(val, (pathlib.PurePath, str)):
                continue
            out[name] = val
            ns[name] = val
            progress = True
        if not progress:
            break
    return out, fixture_names


def _composed_candidates(
    tree: ast.Module, consts: dict[str, object] | None = None
) -> list[ast.BinOp]:
    '''Every ``<expr> / "launch_*.sh"`` node in the module.

    Wf-A269 wave-3 (2026-11-02) — the outer docstring delimiter is THREE
    SINGLE QUOTES and NOT three double quotes, on purpose: the SUPERSEDED
    RENDERING below carries the old function's own double-quoted docstring,
    and nesting three double quotes inside a three-double-quote docstring
    closes the outer one early (SyntaxError at the preserved line).  Three
    single quotes keep the preserved rendering's SOURCE BYTES byte-identical,
    which is what AGENTS.md §5.5 rule #6 asks for — escaping the inner
    quotes would have changed 6 bytes of it.

    Wf-A269 wave-3 (2026-11-02) — the right operand may also be a NAME whose
    module constant is a launcher filename (``MOLMETAL_SCRIPTS_DIR /
    LAUNCHER_WF333``).  The literal-only form was a THIRD BLIND SPOT of this
    gate: the composed reference resolved to a path that does not exist and
    the gate could not see it, so the gate's "0 dangling" claim did not
    cover the NAME form.  A NAME right operand is accepted only when the
    module's own constants bind it to a string that looks like a launcher,
    so the universe stays bounded (the resolution step that follows is the
    same one the literal form used).

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
      def _composed_candidates(tree: ast.Module) -> list[ast.BinOp]:
          """Every ``<expr> / "launch_*.sh"`` node in the module."""
          out: list[ast.BinOp] = []
          for node in ast.walk(tree):
              if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                  continue
              right = node.right
              if not (isinstance(right, ast.Constant)
                      and isinstance(right.value, str)
                      and right.value.startswith("launch_")
                      and right.value.endswith(".sh")):
                  continue
              out.append(node)
          return out
    '''
    out: list[ast.BinOp] = []
    consts = consts or {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            if right.value.startswith("launch_") and right.value.endswith(".sh"):
                out.append(node)
            continue
        if isinstance(right, ast.Name):
            value = consts.get(right.id)
            if (isinstance(value, str)
                    and value.startswith("launch_")
                    and value.endswith(".sh")):
                out.append(node)
        continue
    return out


def _binding_names(tree: ast.Module) -> dict[int, set[str]]:
    """Map node id -> the names its value is bound to.

    A composed reference inside a tuple that is assigned to a name is bound
    to THAT name (the repo's own precedent keeps the archived launchers in
    a tuple and asserts, inside a ``for`` loop, that each one is gone).
    """
    out: dict[int, set[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and \
                isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            for sub in ast.walk(n.value):
                out.setdefault(id(sub), set()).add(name)
    return out


def _negated_exists_names(tree: ast.Module) -> set[str]:
    """Names used as ``assert not <name>.exists()``.

    A composed reference bound to one of these names is a NEGATIVE
    assertion -- the test asserts the path is GONE -- so it is not a
    dangling reference.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        inner = test.operand
        if not (isinstance(inner, ast.Call) and inner.args == []
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "exists"
                and isinstance(inner.func.value, ast.Name)):
            continue
        out.add(inner.func.value.id)
    return out


def _negated_container_names(tree: ast.Module, negated: set[str]) -> set[str]:
    """Container names that are iterated and then asserted NOT to exist.

    The repo's own precedent for an archived launcher
    (``molmetal/tests/test_wf327_ultracode_closeout.py``) keeps the
    archived paths in a tuple and asserts, inside a ``for`` loop, that
    each one does NOT exist.  The composed reference is an ELEMENT of
    that tuple, so the binding name is the tuple, not the loop variable.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        target = node.target
        if not (isinstance(target, ast.Name) and target.id in negated):
            continue
        it = node.iter
        if isinstance(it, ast.Name):
            out.add(it.id)
        elif isinstance(it, (ast.Tuple, ast.List)):
            for elt in it.elts:
                if isinstance(elt, ast.Name):
                    out.add(elt.id)
    return out


def _scan_composed() -> tuple[list[tuple[str, int, str]],
                              list[tuple[str, int, str]],
                              list[tuple[str, int, str]],
                              list[tuple[str, int, str]],
                              list[tuple[str, int, str]],
                              int]:
    """Return (dead, excused, fixture, unresolved, outside_repo, n_sites).

    The scan RESOLVES each composed reference by EVALUATING it against the
    module's own path constants -- never by string-matching the source.
    A site that cannot be resolved is reported as unresolved rather than
    silently skipped, and every bucket is bounded by a test so that a
    growing blind spot is visible.
    """
    tracked = _tracked_set()
    repo_real = REPO_ROOT.resolve()
    dead: list[tuple[str, int, str]] = []
    excused: list[tuple[str, int, str]] = []
    fixture: list[tuple[str, int, str]] = []
    unresolved: list[tuple[str, int, str]] = []
    outside: list[tuple[str, int, str]] = []
    n_sites = 0
    for rel in _tracked_live_files():
        if not rel.endswith(".py"):
            continue
        try:
            src = (REPO_ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        negated = _negated_exists_names(tree)
        containers = _negated_container_names(tree, negated)
        consts, fixture_names = _eval_consts(tree, REPO_ROOT / rel)
        bindings = _binding_names(tree)
        for node in _composed_candidates(tree, consts):
            n_sites += 1
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & (FIXTURE_ROOTS | fixture_names):
                # a pytest temp dir -- created by the test, not a repo path
                fixture.append((rel, node.lineno, "/".join(sorted(names))))
                continue
            ns = dict(consts)
            ns["__file__"] = str(REPO_ROOT / rel)
            ns["Path"] = pathlib.Path
            try:
                code = compile(ast.Expression(node), "<site>", "eval")
                val = eval(code, {"__builtins__": {}}, ns)
            except Exception:
                unresolved.append((rel, node.lineno, "<not statically resolvable>"))
                continue
            if not isinstance(val, pathlib.PurePath):
                unresolved.append((rel, node.lineno, "<not a path>"))
                continue
            try:
                tail = val.resolve().relative_to(repo_real).as_posix()
            except ValueError:
                outside.append((rel, node.lineno, str(val)))
                continue
            if tail in tracked:
                continue
            bound = bindings.get(id(node), set())
            if bound & negated or bound & containers:
                excused.append((rel, node.lineno, tail))
                continue
            if (rel, tail) in COMPOSED_ALLOWLIST:
                excused.append((rel, node.lineno, tail))
                continue
            dead.append((rel, node.lineno, tail))
    return dead, excused, fixture, unresolved, outside, n_sites


# ---------------------------------------------------------------------------
# Test 1 -- the gate itself
# ---------------------------------------------------------------------------
def test_no_dangling_launcher_path_literals() -> None:
    """Every ``launch_<n>.sh`` PATH literal in live code must resolve."""
    dead, _host, _composed, n = _scan()
    assert n > 100, (
        f"the token scan only saw {n} path tokens -- the universe shrank, "
        f"so this gate is vacuous"
    )
    assert not dead, (
        "dangling launcher path literals (each must resolve to a tracked "
        "file):\n  " + "\n  ".join(f"{f}:{i}: {t}" for f, i, t in dead)
    )


# ---------------------------------------------------------------------------
# Test 2 -- fail-closed on an unresolvable variable prefix
# ---------------------------------------------------------------------------
def test_composed_paths_use_a_known_root_variable() -> None:
    """A ``${VAR}/``-prefixed launcher path must use a KNOWN root variable.

    An unknown variable cannot be resolved to a repository path, so the
    existence of the target cannot be checked.  The gate fails closed
    rather than silently passing it.
    """
    _dead, _host, composed, _n = _scan()
    assert not composed, (
        "launcher paths behind an unresolvable variable prefix:\n  "
        + "\n  ".join(f"{f}:{i}: {t}" for f, i, t in composed)
    )


# ---------------------------------------------------------------------------
# Test 3 -- the allow-list is not stale
# ---------------------------------------------------------------------------
def test_allowlist_stale_entries_are_reported() -> None:
    """Print allow-list entries that no longer correspond to a live hit."""
    tracked = _tracked_set()
    live: set[tuple[str, str]] = set()
    for rel in _tracked_live_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in TOKEN_RE.finditer(text):
            tok = m.group(0)
            kind, _d = _classify(tok, tracked)
            if kind == "DEAD":
                live.add((rel, tok))
    stale = sorted(set(ALLOWLIST) - live)
    if stale:
        print("STALE ALLOW-LIST ENTRIES (drop them):")
        for key in stale:
            print(f"  {key[0]}: {key[1]} — {ALLOWLIST[key]}")
    # A stale entry is review debt, not a failure: the repo's own wave-2
    # gate reports them the same way.  The allow-list must stay SMALL
    # though -- a growing allow-list is how the class comes back.
    # Wf-A269 wave-5 (2026-11-02) — the cap moves 8 -> 9, DISCLOSED.
    # Reason: the 9th entry is the AGENTS.md §5.5 rule-#6 MANDATE for the
    # 306a launcher's pre-D5 USAGE rendering, which names a path that never
    # resolved.  The count did not grow by laxity; it grew because the
    # mandate requires a rendering that names a dead path, and the gate
    # must be able to record it.  The bound is kept (a growing allow-list is
    # how the class comes back) and every entry still has to carry a
    # reviewed SHAPE and the §5.5 rule-#6 marker, so the gate stays
    # zero-tolerance over the un-allow-listed residual.
    # Wf-A269 wave-5 (2026-11-02) — the cap moves 9 -> 10, DISCLOSED.
    # Reason: the 10th entry is the AGENTS.md §5.5 rule-#6 MANDATE for the
    # t179 launcher's pre-wave-3 ORIGINAL rendering, which names a path that
    # never resolved -- MEASURED, `git log --all --` for it returns nothing.
    # The count did not grow by laxity; it grew because the mandate requires
    # a rendering that names a dead path, and the gate must be able to
    # record it.  Both growths (8 -> 9 for 306a, 9 -> 10 for t179) have the
    # SAME root cause: the six wave-3 launchers were added at
    # TODO/archive/wf437_simplify/ in bc29aa90 and were NEVER at
    # molmetal/scripts/, so every ORIGINAL usage block naming
    # `molmetal/scripts/<launcher>.sh` is a dead path by construction.
    # The bound is kept (a growing allow-list is how the class comes back)
    # and every entry still has to carry a reviewed SHAPE and the §5.5
    # rule-#6 marker, so the gate stays zero-tolerance over the
    # un-allow-listed residual.
    # Wf-A269 wave-19 (2026-11-04) — the cap moves 10 -> 11, DISCLOSED.
    # Reason: the 11th entry is the AGENTS.md §5.5 rule-#6 MANDATE for the
    # wf310f launcher's ORIGINAL usage rendering, which names a path that
    # never resolved -- MEASURED, `git log --all --` for
    # `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh` returns NOTHING
    # (the launcher was added at scripts/ in bc29aa90).  The count did not
    # grow by laxity; it grew because the mandate requires a rendering that
    # names a dead path, and the gate must be able to record it.  The bound
    # is kept (a growing allow-list is how the class comes back) and every
    # entry still has to carry a reviewed SHAPE and the §5.5 rule-#6 marker,
    # so the gate stays zero-tolerance over the un-allow-listed residual.
    assert len(ALLOWLIST) <= 11, (
        f"the allow-list has grown to {len(ALLOWLIST)} entries — the "
        f"zero-tolerance direction is a violation that is NOT allow-listed"
    )


# ---------------------------------------------------------------------------
# Test 3b -- an allowed HISTORICAL RENDERING must carry the rule-#6 marker
# ---------------------------------------------------------------------------
def test_every_allowed_historical_rendering_carries_the_rule6_marker() -> None:
    """An allow-listed historical rendering must SAY that it is one.

    Without this, the allow-list would be a general hole for dead launcher
    paths: any file could name a missing launcher and be excused.  With it,
    the excuse is available only to a file that explicitly preserves the
    superseded rendering, per AGENTS.md §5.5 rule #6 -- and the marker is
    itself decidable from the source.
    """
    checked = 0
    for (rel, tok), reason in sorted(ALLOWLIST.items()):
        if not reason.startswith("HISTORICAL RENDERING"):
            continue
        checked += 1
        path = REPO_ROOT / rel
        assert path.exists(), f"allow-listed file is missing: {rel}"
        text = path.read_text(encoding="utf-8")
        # The marker may be WRAPPED across lines (a docstring inside a
        # function is narrow), so compare against a whitespace-normalised
        # rendering of the file.
        flat = " ".join(text.split())
        assert RULE6_MARKER in flat, (
            f"{rel} is allow-listed as a HISTORICAL RENDERING for {tok}, but "
            f"the file does not carry the {RULE6_MARKER!r} marker -- the "
            "allow-list would be a hole for any dead launcher path."
        )
    assert checked > 0, (
        "the HISTORICAL RENDERING shape is no longer exercised -- the "
        "allow-list test would be vacuous."
    )


# ---------------------------------------------------------------------------
# Test 3c -- the NAME-right-operand form must be SCANNED, not just allowed
# ---------------------------------------------------------------------------
def test_the_composed_scanner_sees_a_name_right_operand() -> None:
    """The composed scanner must see ``DIR / NAME``, not only ``DIR / "..."``.

    The scanner originally accepted ONLY a literal right operand, so a
    composed reference written as ``MOLMETAL_SCRIPTS_DIR / LAUNCHER_WF333``
    was invisible to it — and that reference resolved to a path that does
    not exist.  A gate that cannot see a shape cannot report it, so the
    "0 dangling" claim did not cover the shape.

    This test is SELF-CONTAINED: it parses a synthetic module, so it cannot
    pass merely because the repository happens to contain no such site.
    """
    source = (
        "DIR = Path('/tmp')\n"
        "LAUNCHER = 'launch_wf333_vina_4step_gpu_bench.sh'\n"
        "site = DIR / LAUNCHER\n"
        "other = DIR / 'launch_wf299_cfm_main.sh'\n"
    )
    tree = ast.parse(source)
    consts = {"DIR": pathlib.Path("/tmp"), "LAUNCHER": "launch_wf333_vina_4step_gpu_bench.sh"}
    found = _composed_candidates(tree, consts)
    names = {n.id for node in found for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert "LAUNCHER" in names, (
        "the composed scanner did not see the NAME-right-operand form "
        "(`DIR / LAUNCHER`) — the blind spot this test guards is back"
    )
    literals = {
        n.value for node in found for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "launch_wf299_cfm_main.sh" in literals, (
        "the composed scanner stopped seeing the LITERAL right operand — "
        "the broadened form must ADD a shape, never replace one"
    )


# ---------------------------------------------------------------------------
# Test 4 -- the six relocated launchers really are where the callers say
# ---------------------------------------------------------------------------
def test_relocated_launchers_resolve_at_scripts() -> None:
    """The 6 launchers wave-3 relocated must exist under ``scripts/``."""
    relocated = (
        "launch_wf264_cfm_property_panel.sh",
        "launch_wf74_v2_drugood_dann.sh",
        "launch_wf306a_drugood_vina_gpu.sh",
        "launch_wf306b_wf219_ru_gpu.sh",
        "launch_wf310c_t179_w2_gpu.sh",
        "launch_wf310c_todo220_gpu.sh",
    )
    tracked = _tracked_set()
    missing = [n for n in relocated if f"scripts/{n}" not in tracked]
    assert not missing, (
        f"relocated launchers missing from scripts/: {missing}"
    )


# ---------------------------------------------------------------------------
# Test 5 -- the COMPOSED class (the form the token scan cannot see)
# ---------------------------------------------------------------------------
def test_no_dangling_composed_launcher_references() -> None:
    """A ``X / "launch_*.sh"`` reference must resolve too.

    MEASURED (2026-11-02, Wf-A269 wave-3 adversarial review): the token
    scan alone cannot see this shape, and three such references were
    dangling and invisible to it.
    """
    dead, _excused, fixture, unresolved, outside, n = _scan_composed()
    assert n > 20, (
        f"the composed scan only saw {n} composed references -- the "
        f"universe shrank, so this half of the gate is vacuous"
    )
    assert not dead, (
        "dangling COMPOSED launcher references (each must resolve to a "
        "tracked file):\n  "
        + "\n  ".join(f"{f}:{i}: {t}" for f, i, t in dead)
    )
    # The blind spots of the scan are bounded, so that they cannot grow
    # without this test turning red.
    assert len(unresolved) <= 6, (
        f"{len(unresolved)} composed launcher references could not be "
        f"resolved statically (blind spot):\n  "
        + "\n  ".join(f"{f}:{i}: {t}" for f, i, t in unresolved)
    )
    assert len(outside) <= 2, (
        f"{len(outside)} composed launcher references resolve OUTSIDE the "
        f"repository:\n  "
        + "\n  ".join(f"{f}:{i}: {t}" for f, i, t in outside)
    )
    assert fixture, (
        "the pytest-fixture bucket is empty — the fixture classifier has "
        "stopped matching anything, so it may be hiding real sites"
    )


# ---------------------------------------------------------------------------
# Test 6 -- the composed allow-list is not stale
# ---------------------------------------------------------------------------
def test_composed_allowlist_stale_entries_are_reported() -> None:
    """Print composed allow-list entries that no longer correspond to a hit."""
    dead, excused, _f, _u, _o, _n = _scan_composed()
    live = {(f, t) for f, _i, t in excused}
    stale = sorted(set(COMPOSED_ALLOWLIST) - live)
    if stale:
        print("STALE COMPOSED ALLOW-LIST ENTRIES (drop them):")
        for key in stale:
            print(f"  {key[0]}: {key[1]} — {COMPOSED_ALLOWLIST[key]}")
    assert len(COMPOSED_ALLOWLIST) <= 8, (
        f"the composed allow-list has grown to {len(COMPOSED_ALLOWLIST)} "
        f"entries — a growing allow-list is how the class comes back"
    )
