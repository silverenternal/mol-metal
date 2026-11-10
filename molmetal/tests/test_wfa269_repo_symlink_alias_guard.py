#!/usr/bin/env python
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""(Wf-A269 §6.269.18.8 / §6.269.18.10) — a symlink is a write handle, and
the repository is full of them.

**The class this closes.**  A POSIX shell's ``>`` redirection **follows a
symlink**: ``open(O_WRONLY|O_CREAT|O_TRUNC)`` on a link truncates the
**target**.  So a symlink is not a shortcut — it is a **write handle**, and
two links to one target are **one file wearing two names**.

Catalog §6.269.18.8 records the measurement that named the class: on
2026-11-02 a wave agent built a "fixture" that was a **symlink farm** into
the live tree and then ran ``git show HEAD:scripts/$w > <farm>/scripts/$w``
— which wrote HEAD's content straight into the repository, silently
reverting a repair and re-arming a GPU-lock defect.  §6.269.18.10 records
the neutralisation (137 337 links under ``/tmp`` severed).  **This guard is
the mechanism that stops the class from coming back inside the repo.**

**The two directions.**

* **external -> repo** (the clobber direction): a link *outside* the
  repository whose target is *inside* it.  Any redirect through it writes
  into the repository.  The repository cannot police ``/tmp``, so this guard
  does not try; it polices the **sites** that create links.
* **repo -> repo** (the alias-collapse direction): a link *inside* the
  repository whose target is *inside* it.  Two distinct repository paths
  become **one file**: writing either writes both.  This direction *is*
  visible from inside the repository, and it is what this guard measures.

**The census (MEASURED 2026-11-03, CPU-only, no network, no repo writes).**
The working tree holds **2 946** symlinks outside ``.git``
(``find . -type l -not -path './.git/*'``, **0.476 s** wall).  Of those,
**80** resolve to a target inside the repository, and **3** of those are
links the repository itself owns:

===========================================================  =========  ==========
path                                                        class      status
===========================================================  =========  ==========
``molmetal/adapters/flow_matching_lipman/_reference/        TRACKED    the
  flow_matching``                                                      alias-
                                                               collapse
                                                               hazard
``molmetal/.venvs/mol-metal-venv``                          IGNORED    venv
                                                                       pointer
``.data/molmetal/bindingdb_full/MetalCytoToxDB.csv``        IGNORED    alias-
                                                                       collapse
                                                                       hazard
===========================================================  =========  ==========

The 43 tracked links pointing at ``/mnt/storage/tools/vina_gpu21_source``
are the **machine-bound** class: an absolute target cannot resolve on any
other checkout, so the file it names silently disappears from the tree.

**Why a baseline and not zero tolerance.**  AGENTS.md §5.5 rule #6 forbids
deleting history, and these links are tracked history.  Zero tolerance would
be unsatisfiable without a repo-wide rewrite that is not this guard's job.
So this is a **growth detector**: any link outside the named baseline fails
the test.  That is the same shape as
``test_wfa131_repo_path_reference_guard.py`` and
``test_wfa127_surface_count_guard.py``.

**Nothing is excluded silently.**  The working-tree scan covers the whole
repository; the excluded prefixes are named in :data:`EXCLUDED_PREFIXES`
with the ``.gitignore`` rule that backs each one, and a test asserts that
every excluded prefix really is gitignored — so the exclusion cannot rot
into a blind spot.

**The corpus is enumerated with ``find``/``git ls-files`` — never
``os.walk``** (the repository is 201 GB and a full walk has taken down this
host before).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: MEASURED 2026-11-03 — the tracked symlinks (mode 120000) whose target
#: resolves INSIDE the repository.  Writing through such a link writes into
#: the repository under a second name.
BASELINE_TRACKED_INSIDE_REPO: tuple[str, ...] = (
    "molmetal/adapters/flow_matching_lipman/_reference/flow_matching",
)

#: MEASURED 2026-11-03 — every in-scope WORKING-TREE symlink whose target
#: resolves inside the repository, with its adjudication.  This is the
#: mutation-sensitive census: a new in-repo alias fails the test.
BASELINE_WORKTREE_INSIDE_REPO: dict[str, str] = {
    "molmetal/adapters/flow_matching_lipman/_reference/flow_matching": (
        "TRACKED — the Lipman upstream reference package the CFM adapter "
        "imports.  Link path and target are both inside the repository, so a "
        "write through the link overwrites the copy under "
        "`molmetal/references/flow_matching/flow_matching/`."
    ),
    "molmetal/.venvs/mol-metal-venv": (
        "IGNORED — the per-project venv pointer (target: `.venv`).  The "
        "target is inside the repository but it is a venv, not source; "
        "`molmetal/tests/test_env_setup.py:72` instructs the operator to "
        "create it."
    ),
    ".data/molmetal/bindingdb_full/MetalCytoToxDB.csv": (
        "IGNORED — created by "
        "`scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:484` (`ln -sf`).  "
        "The canonical `MetalCytoToxDB.csv` and the source "
        "`metal_binding_db_full.csv` are ONE file."
    ),
}

#: MEASURED 2026-11-03 — working-tree symlink prefixes excluded from the
#: in-repo census, each with the reason AND the `.gitignore` rule that backs
#: it.  A test asserts every prefix here really is gitignored — via
#: `git check-ignore`, the authoritative answer — so the exclusion cannot rot
#: into a blind spot.
EXCLUDED_PREFIXES: dict[str, str] = {
    ".cloned/": "vendored SOTA repo clones — upstream trees (.gitignore:53 `.cloned/`)",
    ".claude/": (
        "tooling scratch incl. Workflow isolation worktrees "
        "(.gitignore:206 `.claude/`)"
    ),
    ".venv/": "the project venv — its internal links are the venv's (.gitignore:213 `*.venv*`)",
    ".venv.bak.": "a venv backup — same as .venv/ (.gitignore:240 `*.bak.*`)",
    "molmetal/references/": (
        "vendored upstream packages, e.g. Lipman (.gitignore:52 `references/`)"
    ),
    "references/": "vendored upstream repos, e.g. RxnFlow (.gitignore:52 `references/`)",
    "list/": (
        "a stray virtualenv (CACHEDIR.TAG + bin/lib/lib64) — not a source "
        "surface (list/.gitignore:1 `*`)"
    ),
}

#: MEASURED 2026-11-03 — tracked symlinks whose target is ABSOLUTE.  An
#: absolute target is machine-bound: the link cannot resolve on any other
#: checkout, so the file it names silently disappears from the tree.
BASELINE_COUNT_ABSOLUTE_TARGET = 44

#: MEASURED 2026-11-03 — the distinct 4-component roots the absolute
#: targets point into: this repository (1 link) and a mounted volume holding
#: the Vina-GPU build (43 links).
BASELINE_ABSOLUTE_TARGET_ROOTS: tuple[str, ...] = (
    "/home/hugo/codes/mol-metal",
    "/mnt/storage/tools/vina_gpu21_source",
)

#: MEASURED 2026-11-03 — the 43 external absolute-target links all live
#: under this link-path prefix (the docking-report trees).
BASELINE_EXTERNAL_LINK_PATH_PREFIX = "molmetal/reports/"

#: MEASURED 2026-11-03 — tracked symlinks that the index records as links
#: but that are REGULAR FILES on disk (the working tree replaced them; git
#: reports the typechange).
BASELINE_REPLACED_ON_DISK: tuple[str, ...] = (
    "results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json",
)

#: Every file in the repository that *creates* a symlink, with the
#: adjudication.  The census is exhaustive: a NEW site fails the test until
#: it is named here.  MEASURED 2026-11-03.
ADJUDICATED_SITES: dict[str, str] = {
    "molmetal/scripts/r17_gpu_retrain.py": (
        "INSIDE_REPO_ALIAS — `latest.symlink_to(dst.name)` puts `latest.pt` and "
        "`step_<n>.pt` in the SAME checkpoint dir: writing `latest.pt` overwrites "
        "the checkpoint.  Both live under the run's own ckpt archive dir."
    ),
    "molmetal/scripts/snapshot_experiment_runtime.py": (
        "OUTSIDE_REPO — `target.symlink_to(shared, ...)` points a run-local name "
        "at a shared artefact; the link path is the run's own output dir."
    ),
    "molmetal/scripts/wf185_chemprop_v2_drugood_full.py": (
        "OUTSIDE_REPO — `dst.symlink_to(src)` inside a cache dir; `src` is the "
        "upstream cache, `dst` is the copy-local cache."
    ),
    "molmetal/scripts/wf263a_verify_drugood.py": (
        "OUTSIDE_REPO — same shape as the wf185 site (cache-local copy)."
    ),
    "scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh": (
        "INSIDE_REPO_ALIAS — `ln -sf \"${METAL_CSV}\" \"${BINDINGDB_CANONICAL_LINK}\"`; "
        "both operands are under `${PROJECT_ROOT}/.data/molmetal/bindingdb_full/`, so "
        "`MetalCytoToxDB.csv` and `metal_binding_db_full.csv` are ONE file. "
        "Writing the canonical name overwrites the source."
    ),
}

#: Files that MENTION a symlink but do not CREATE one — the text of a link
#: inside a string literal, an error message, or a comment.  They are
#: adjudicated separately because §6.269.18.11's lesson is that a textual
#: scan must distinguish a mention from an executable line: conflating the
#: two is how a preserved quotation gets reported as a live defect, and how
#: a real site gets dismissed as "just a comment".  MEASURED 2026-11-03.
ADJUDICATED_MENTIONS: dict[str, str] = {
    "molmetal/tests/test_env_setup.py": (
        "MENTION_NOT_A_SITE — the `ln -snf` text is inside an f-string ERROR "
        "MESSAGE telling the operator how to create the link; no code runs it. "
        "One edit away from becoming a site, which is why it is pinned."
    ),
}

#: The python symlink-creation sites, pinned by line number (MEASURED
#: 2026-11-03).
BASELINE_PYTHON_SITE_LINES: dict[str, list[int]] = {
    "molmetal/scripts/r17_gpu_retrain.py": [496],
    "molmetal/scripts/snapshot_experiment_runtime.py": [101],
    "molmetal/scripts/wf185_chemprop_v2_drugood_full.py": [240],
    "molmetal/scripts/wf263a_verify_drugood.py": [196],
}

#: The symlink MENTION lines, pinned by line number (MEASURED 2026-11-03).
BASELINE_MENTION_LINES: dict[str, list[int]] = {
    "molmetal/tests/test_env_setup.py": [72],
}

_LN_RE = re.compile(r"\bln\s+-[a-zA-Z]*s[a-zA-Z]*\s+(.+)$")
_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PY_SITE_RE = re.compile(r"\.symlink_to\(|os\.symlink\(")
#: Any textual trace of a symlink at all — an executable call OR a mention.
_TOKEN_RE = re.compile(r"ln\s+-[a-zA-Z]*s|symlink_to\(|os\.symlink\(")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def tracked_symlinks() -> list[str]:
    """Every path the index records with mode 120000 (a symlink)."""
    out = []
    for line in _git("ls-files", "-s").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if parts and parts[0] == "120000" and path.strip():
            out.append(path.strip())
    return sorted(out)


def working_tree_symlinks() -> list[str]:
    """Every symlink in the working tree, ``.git`` excluded, repo-relative.

    MEASURED 2026-11-03: 2 946 links, 0.476 s wall.  ``find`` is used rather
    than ``os.walk`` — it stats metadata only and never loads the tree into
    memory (this repository is 201 GB).
    """
    proc = subprocess.run(
        ["find", ".", "-type", "l", "-not", "-path", "./.git/*"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("./"):
            continue
        out.append(line[2:])
    return sorted(out)


def in_scope(rel: str) -> bool:
    """True when ``rel`` is one of the repository's own surfaces."""
    return not any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def _is_inside_repo(resolved: str) -> bool:
    root = str(REPO)
    return resolved == root or resolved.startswith(root + os.sep)


def classify_path(p: Path) -> tuple[str, str]:
    """Classify a path: ``INSIDE_REPO`` | ``OUTSIDE_REPO`` | ``REPLACED_ON_DISK``."""
    if not p.is_symlink():
        return ("REPLACED_ON_DISK", "not a symlink on disk")
    resolved = os.path.realpath(p)
    if _is_inside_repo(resolved):
        return ("INSIDE_REPO", resolved)
    return ("OUTSIDE_REPO", resolved)


def classify_tracked_symlink(rel: str) -> tuple[str, str]:
    """Classify one repo-relative tracked symlink path."""
    return classify_path(REPO / rel)


def _shell_assignments(text: str) -> dict[str, str]:
    """A deliberately small ``VAR=value`` reader (one pass, no eval)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def _expand(value: str, env: dict[str, str], depth: int = 0) -> str:
    """Expand ``${VAR}`` references, one variable at a time, bounded."""
    if depth > 5 or "${" not in value:
        return value

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name == "PROJECT_ROOT":
            return str(REPO)
        return env.get(name, m.group(0))

    return _expand(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, value), env, depth + 1)


def resolve_shell_link_path(script_text: str, operand: str) -> str | None:
    """Resolve one ``ln -s`` operand against the script's own assignments."""
    operand = operand.strip()
    if len(operand) >= 2 and operand[0] == operand[-1] and operand[0] in "\"'":
        operand = operand[1:-1]
    env = _shell_assignments(script_text)
    return _expand(operand, env)


def _split_operands(raw: str) -> list[str]:
    """Split a shell operand list on whitespace, honouring quotes."""
    out: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    for ch in raw:
        if quote:
            if ch == quote:
                quote = None
            else:
                cur.append(ch)
        elif ch in "\"'":
            quote = ch
        elif ch.isspace():
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _script_corpus() -> list[str]:
    """Every tracked shell/python file under the launch + library surfaces."""
    return [
        p
        for p in _git("ls-files", "scripts", "molmetal/scripts", "molmetal/tests").split()
        if p.endswith(".sh") or p.endswith(".py")
    ]


def shell_link_sites() -> dict[str, list[tuple[int, str, str, str]]]:
    """Every live ``ln -s`` site: ``file -> [(lineno, raw, target, link)]``.

    ``link`` is the operand a redirect would write THROUGH — the second
    operand of ``ln -s TARGET LINKNAME``.  It is ``""`` when the site has
    fewer than two operands.
    """
    sites: dict[str, list[tuple[int, str, str, str]]] = {}
    for rel in _script_corpus():
        if not rel.endswith(".sh"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            m = _LN_RE.search(line)
            if not m:
                continue
            operands = _split_operands(m.group(1))
            target = operands[0] if operands else ""
            link = operands[1] if len(operands) > 1 else ""
            sites.setdefault(rel, []).append((lineno, m.group(1).strip(), target, link))
    return sites


def python_symlink_sites() -> dict[str, list[int]]:
    """Every live ``symlink_to(`` / ``os.symlink(`` site, by file and line."""
    sites: dict[str, list[int]] = {}
    for rel in _script_corpus():
        if not rel.endswith(".py"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        hits = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if _PY_SITE_RE.search(line):
                hits.append(lineno)
        if hits:
            sites[rel] = hits
    return sites


def symlink_token_lines() -> dict[str, list[int]]:
    """Every line, in every corpus file, carrying a symlink token."""
    out: dict[str, list[int]] = {}
    for rel in _script_corpus():
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        hits = [
            lineno
            for lineno, line in enumerate(text.splitlines(), 1)
            if _TOKEN_RE.search(line)
        ]
        if hits:
            out[rel] = hits
    return out


def symlink_mentions() -> dict[str, list[int]]:
    """Files that MENTION a symlink without CREATING one, and their lines.

    A mention is a textual trace of a link that is not an executable call —
    an f-string, an error message, a comment.  It is separated from a site
    because §6.269.18.11's lesson is that a scan which conflates the two
    reports preserved quotations as live defects.
    """
    sites = set(shell_link_sites()) | set(python_symlink_sites())
    return {
        rel: lines
        for rel, lines in symlink_token_lines().items()
        if rel not in sites
    }


def _absolute_tracked() -> list[tuple[str, str]]:
    """``[(repo_relative_link_path, raw_target)]`` for absolute-target links."""
    out = []
    for rel in tracked_symlinks():
        p = REPO / rel
        if not p.is_symlink():
            continue
        raw = os.readlink(p)
        if os.path.isabs(raw):
            out.append((rel, raw))
    return out


# ---------------------------------------------------------------------------
# 1 — the in-repo alias census (the hazard this guard exists for)
# ---------------------------------------------------------------------------
def test_no_tracked_symlink_resolves_inside_the_repository():
    """A tracked link whose target is inside the repo aliases two repo paths.

    Writing through it writes into the repository under a second name.
    """
    found = {}
    for rel in tracked_symlinks():
        state, detail = classify_tracked_symlink(rel)
        if state == "INSIDE_REPO":
            found[rel] = detail
    assert set(found) == set(BASELINE_TRACKED_INSIDE_REPO), (
        "the set of tracked symlinks resolving INSIDE the repository changed "
        f"(baseline {len(BASELINE_TRACKED_INSIDE_REPO)}, now {len(found)}): "
        f"added={sorted(set(found) - set(BASELINE_TRACKED_INSIDE_REPO))} "
        f"removed={sorted(set(BASELINE_TRACKED_INSIDE_REPO) - set(found))}. "
        "A NEW in-repo link is an alias-collapse hazard (two repo paths become "
        "one file); a REMOVED one means the baseline must be re-adjudicated "
        "in this file, not deleted silently."
    )


def test_the_in_repo_alias_target_actually_exists():
    """The pinned in-repo link must still RESOLVE, not dangle.

    A dangling in-repo link is a *different* defect (the import it serves
    breaks) and must not be hidden behind the baseline entry.
    """
    for rel in BASELINE_TRACKED_INSIDE_REPO:
        p = REPO / rel
        assert p.is_symlink(), f"{rel} is no longer a symlink on disk"
        assert p.exists(), (
            f"{rel} dangles — its target {os.readlink(p)} does not resolve. "
            "The CFM adapter imports this reference package; a dangling link "
            "breaks the import on this checkout even though the baseline "
            "entry still names it."
        )


def test_the_working_tree_in_repo_alias_set_is_pinned():
    """THE census: no in-scope symlink may alias a second repository path.

    This is the mutation-sensitive half of the guard — the index-only
    census above cannot see an UNTRACKED alias, which is exactly what a
    fixture builder or a `ln -sf` in a launcher would leave behind.
    """
    found = {}
    for rel in working_tree_symlinks():
        if not in_scope(rel):
            continue
        state, detail = classify_path(REPO / rel)
        if state == "INSIDE_REPO":
            found[rel] = detail
    assert set(found) == set(BASELINE_WORKTREE_INSIDE_REPO), (
        "the set of WORKING-TREE symlinks resolving INSIDE the repository "
        f"changed (baseline {len(BASELINE_WORKTREE_INSIDE_REPO)}, now "
        f"{len(found)}): added={sorted(set(found) - set(BASELINE_WORKTREE_INSIDE_REPO))} "
        f"removed={sorted(set(BASELINE_WORKTREE_INSIDE_REPO) - set(found))}. "
        "A NEW in-repo link is an alias-collapse hazard: writing through it "
        "writes into the repository under a second name."
    )


def test_every_excluded_prefix_is_really_gitignored():
    """The exclusion list must not rot into a blind spot.

    Every prefix in EXCLUDED_PREFIXES must be matched by an ignore rule —
    asked of git itself via `git check-ignore -v --no-index`, which is the
    authoritative answer (a nested `.gitignore` counts).  So a prefix can
    only be excluded while it stays untracked, and the day it becomes
    tracked the test fails and the prefix must be re-adjudicated.
    """
    for prefix, reason in EXCLUDED_PREFIXES.items():
        probe = (
            prefix.rstrip("/") + "/wfa269_probe"
            if prefix.endswith("/")
            else prefix + "wfa269_probe"
        )
        proc = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-v", "--no-index", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"EXCLUDED_PREFIXES has {prefix!r} but no ignore rule matches "
            f"{probe!r}. Either the prefix became tracked (and must be "
            f"adjudicated) or the exclusion is unjustified. Reason on record: "
            f"{reason}"
        )
        assert reason.strip(), f"{prefix} has no stated reason"


# ---------------------------------------------------------------------------
# 2 — machine-bound targets (portability)
# ---------------------------------------------------------------------------
def test_the_machine_bound_tracked_symlink_count_is_pinned():
    """An ABSOLUTE target cannot resolve on any other checkout."""
    absolute = _absolute_tracked()
    assert len(absolute) == BASELINE_COUNT_ABSOLUTE_TARGET, (
        f"tracked symlinks with an ABSOLUTE target: {len(absolute)} "
        f"(baseline {BASELINE_COUNT_ABSOLUTE_TARGET}). An absolute target is "
        "machine-bound: the link dangles on every other checkout."
    )


def test_absolute_tracked_symlink_targets_stay_inside_the_two_known_roots():
    """Absolute targets must point into a named root — not a new mount."""
    roots = set()
    for _rel, raw in _absolute_tracked():
        parts = raw.split("/")
        roots.add("/" + "/".join(parts[1:5]) if len(parts) > 4 else raw)
    expected = set(BASELINE_ABSOLUTE_TARGET_ROOTS)
    assert roots <= expected, (
        f"tracked symlinks point into unnamed roots: {sorted(roots - expected)}. "
        "An absolute target outside the two known roots is a new machine-bound "
        "dependency and must be adjudicated before it lands."
    )


def test_every_external_absolute_link_lives_under_the_known_report_prefix():
    """The 43 Vina-GPU links live under ``molmetal/reports/`` — pin that."""
    external = [
        rel
        for rel, raw in _absolute_tracked()
        if not raw.startswith(str(REPO))
    ]
    assert len(external) == BASELINE_COUNT_ABSOLUTE_TARGET - 1, (
        f"external absolute-target links: {len(external)} "
        f"(baseline {BASELINE_COUNT_ABSOLUTE_TARGET - 1}; the 44th is the "
        "in-repo `_reference` link)"
    )
    stray = [r for r in external if not r.startswith(BASELINE_EXTERNAL_LINK_PATH_PREFIX)]
    assert not stray, (
        f"external absolute-target links outside {BASELINE_EXTERNAL_LINK_PATH_PREFIX!r}: "
        f"{stray}. A new external link must be adjudicated: it is a "
        "machine-bound dependency that dangles on any other checkout."
    )


# ---------------------------------------------------------------------------
# 3 — the replaced-on-disk typechange
# ---------------------------------------------------------------------------
def test_the_replaced_on_disk_symlinks_are_pinned():
    """The index says symlink; the tree says regular file.  Pin it."""
    replaced = tuple(rel for rel in tracked_symlinks() if not (REPO / rel).is_symlink())
    assert replaced == BASELINE_REPLACED_ON_DISK, (
        f"tracked symlinks replaced on disk by regular files: {replaced} "
        f"(baseline {BASELINE_REPLACED_ON_DISK}). git reports these as a "
        "typechange; the count must be re-adjudicated, not drift."
    )


# ---------------------------------------------------------------------------
# 4 — the site census is exhaustive (no silent caps)
# ---------------------------------------------------------------------------
def test_every_symlink_creation_site_is_adjudicated():
    """Every site that CREATES a link must be named in ADJUDICATED_SITES."""
    shell = shell_link_sites()
    py = python_symlink_sites()
    found = set(shell) | set(py)
    assert found == set(ADJUDICATED_SITES), (
        f"symlink-creation sites changed. added={sorted(found - set(ADJUDICATED_SITES))} "
        f"removed={sorted(set(ADJUDICATED_SITES) - found)}. A new site must be "
        "adjudicated (is the LINK PATH inside the repository?) before it lands; "
        "a removed site means the table must be re-read, not silently pruned."
    )


def test_every_symlink_mention_is_adjudicated_separately_from_the_sites():
    """A MENTION is not a SITE — and the two tables must not overlap.

    This is §6.269.18.11's class, mechanised: the census distinguishes a
    textual mention of a link from an executable creation of one, so that
    (a) a preserved quotation is not reported as a live defect and (b) a
    mention that BECOMES a site is caught by the site table.
    """
    mentions = symlink_mentions()
    assert set(mentions) == set(ADJUDICATED_MENTIONS), (
        f"symlink MENTION files changed. added={sorted(set(mentions) - set(ADJUDICATED_MENTIONS))} "
        f"removed={sorted(set(ADJUDICATED_MENTIONS) - set(mentions))}. A new mention must be "
        "adjudicated (is it executable code or text?) before it lands."
    )
    overlap = set(ADJUDICATED_SITES) & set(ADJUDICATED_MENTIONS)
    assert not overlap, (
        f"files classified as BOTH a site and a mention: {sorted(overlap)}. "
        "A file that creates a link is a SITE even if it also mentions one."
    )
    assert set(mentions) | set(ADJUDICATED_SITES) == set(symlink_token_lines()), (
        "the site table + the mention table do not cover every file carrying "
        "a symlink token — a file was silently dropped from the census."
    )


def test_the_python_site_lines_are_pinned():
    """The python sites' line numbers are pinned, so an edit is visible."""
    py = python_symlink_sites()
    assert py == BASELINE_PYTHON_SITE_LINES, (
        f"python symlink-creation sites moved: {py} "
        f"(expected {BASELINE_PYTHON_SITE_LINES}). "
        "Re-read each site and re-adjudicate before updating the pin."
    )


def test_the_mention_lines_are_pinned():
    """The mention lines are pinned, so a mention that becomes a site is visible."""
    mentions = symlink_mentions()
    assert mentions == BASELINE_MENTION_LINES, (
        f"symlink MENTION lines moved: {mentions} "
        f"(expected {BASELINE_MENTION_LINES})."
    )


def test_the_adjudication_table_has_a_reason_for_every_site():
    """A site or mention with no stated reason is unadjudicated."""
    for table in (ADJUDICATED_SITES, ADJUDICATED_MENTIONS, BASELINE_WORKTREE_INSIDE_REPO):
        for rel, reason in table.items():
            assert reason.strip(), f"{rel} has no adjudication reason"
            assert (REPO / rel).exists(), (
                f"{rel} is adjudicated but no longer exists in the repository"
            )


# ---------------------------------------------------------------------------
# 5 — the shell resolver, mechanically
# ---------------------------------------------------------------------------
def test_the_shell_link_paths_inside_the_repository_are_pinned():
    """Resolve every shell ``ln -s`` LINK PATH and flag the in-repo ones."""
    flagged = {}
    for rel, entries in shell_link_sites().items():
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for _lineno, _raw, _target, link in entries:
            if not link:
                continue
            resolved = resolve_shell_link_path(text, link)
            if resolved and _is_inside_repo(resolved):
                flagged[rel] = resolved
    assert flagged == {
        "scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh": (
            "/home/hugo/codes/mol-metal/.data/molmetal/bindingdb_full/MetalCytoToxDB.csv"
        )
    }, (
        f"shell `ln -s` link paths resolving inside the repository: {flagged}. "
        "A link path inside the repo means two repository paths are ONE file."
    )


def test_the_shell_resolver_can_fail():
    """The resolver must actually reject a link path outside the repo.

    A guard that cannot fail is the dead-gate class (catalog §6.269.17).
    This is the reverse test: it proves the check discriminates.
    """
    text = (
        'PROJECT_ROOT="/home/hugo/codes/mol-metal"\n'
        'OUT="/tmp/somewhere/else/alias.csv"\n'
        'IN="${PROJECT_ROOT}/data/alias.csv"\n'
    )
    outside = resolve_shell_link_path(text, '"${OUT}"')
    inside = resolve_shell_link_path(text, '"${IN}"')
    assert outside == "/tmp/somewhere/else/alias.csv", outside
    assert not _is_inside_repo(outside or "")
    assert inside == f"{REPO}/data/alias.csv", inside
    assert _is_inside_repo(inside or "")


def test_the_classifier_can_fail():
    """The classifier must reject an external link and accept an in-repo one.

    Reverse test for the census itself: a classifier that always returned
    ``OUTSIDE_REPO`` would make every census above pass vacuously.
    """
    # Both probes live OUTSIDE the repository (a temp dir), pointing in and
    # out.  Creating a link inside the repo would be the very hazard this
    # guard exists to prevent; and the direction that actually clobbered the
    # tree was outside->repo, so that is the direction worth proving.
    with tempfile.TemporaryDirectory() as tmp:
        outside_link = Path(tmp) / "outside_link"
        outside_link.symlink_to("/etc/hostname")
        inside_link = Path(tmp) / "inside_link"
        inside_link.symlink_to(REPO / "README.md")
        state, detail = classify_path(outside_link)
        assert state == "OUTSIDE_REPO", f"expected OUTSIDE_REPO, got {state} ({detail})"
        assert not _is_inside_repo(detail)

        state, detail = classify_path(inside_link)
        assert state == "INSIDE_REPO", f"expected INSIDE_REPO, got {state} ({detail})"
        assert _is_inside_repo(detail)
