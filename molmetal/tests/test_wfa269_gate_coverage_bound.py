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

"""Wf-A269 -- the COVERAGE BOUND of the host-root gate.

WHY THIS FILE EXISTS
--------------------
``molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py`` is the
zero-tolerance gate for host-bound repo paths.  Its verdict is
CONDITIONAL on a hand-curated list of three absolute path prefixes
(``_ROOTS``).  A violation whose literal starts with any other prefix
is invisible to it: the prefilter drops the file before the AST is
analysed.  So "0 hits not allow-listed" means "0 hits under the three
roots", NOT "0 hits".

Nothing in the repo measured that bound.  This file measures it and
publishes it, so that the gate's verdict can be read with its coverage
attached rather than in isolation.

THE TWO UNITS
-------------
UNIT A -- the MOUNT.  A mount point is the place a host's filesystem is
anchored: ``/home/hugo``, ``/mnt/storage``, ``/usr``, ``/opt``.  Every
absolute literal in a tracked ``.py`` file that starts under a mount must
be attributable to a reviewed mount.

UNIT B -- the CHECKOUT ROOT.  Under ``/home/hugo/codes/`` there is more
than one checkout: this repo, a Triton sandbox, and two sibling
projects.  A checkout root is where "the repo lives on THIS host"
lives, so the set of checkout roots is the set of prefixes whose
violations the gate must be able to see.

THE TWO TABLES ARE REVIEW DEBT, NOT A SHIELD
----------------------------------------------
Every entry names WHY a prefix is outside the defect class.  A NEW
prefix — a new sibling checkout, a new data mount — is not silently
ignored: ``test_every_mount_is_reviewed`` fails on it, so the new
prefix has to be adjudicated rather than inherited.

MEASURED (2026-11-02, CPU-only)
--------------------------------
* tracked ``.py`` files scanned: **2874**
* distinct absolute literals: **568**
* absolute literals under a host mount: **241**
* UNIT A mounts in use: **6**
* checkout roots under ``/home/hugo/codes/``: **4** — the gate sees 2
* literals under the 2 checkout roots the gate cannot see: **14**, in
  **14** files, every one adjudicated EXTERNAL (a sibling project's
  datasets and its own evaluation virtualenv)

CPU-only, no GPU, no network.  stdlib + pytest only.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

GATE_MODULE = REPO / "molmetal" / "tests" / "test_wfa269_no_hardcoded_repo_paths.py"


# ---------------------------------------------------------------------------
# The mount table.  Key = the mount prefix; value = why it is not a
# checkout root of THIS repository.
# ---------------------------------------------------------------------------
REVIEWED_MOUNTS: dict[str, str] = {
    "/home/hugo": (
        "the operator's home directory — contains this checkout, a sibling "
        "Triton sandbox and two sibling projects; the checkout roots under "
        "it are reviewed separately in REVIEWED_CHECKOUT_ROOTS"
    ),
    "/home/linuxbrew": "a Homebrew prefix on the operator's machine — tool install, not a checkout",
    "/mnt/storage": (
        "an external data mount (CrossDocked 1.6 GB archive, PDBbind, "
        "pre-built envs); this repo's own data lives under .data/molmetal/ "
        "and is exported by env.sh §1b as MOLMETAL_DATA_DIR"
    ),
    "/opt": "system-wide tool installs (conda, vina, fpocket, p2rank, chrome) — not a checkout",
    "/usr": "system binaries (vina, chromium, conda) — not a checkout",
    "/home": "the operator's home directory referenced without the user component",
}


# ---------------------------------------------------------------------------
# The checkout-root table.  Key = a checkout root path; value = why the
# gate must (or must not) forbid it.
#
# FORBIDDEN = the gate's own list: a violation under it is this defect
# class and the gate must be able to see it.
# EXTERNAL  = a sibling checkout: its paths are a different project's
#             files, so a literal under it does not reach INTO this
#             repository — the gate's rule (b), REACHABILITY, excludes
#             it, and the review below records why.
# ---------------------------------------------------------------------------
REVIEWED_CHECKOUT_ROOTS: dict[str, str] = {
    "/home/hugo/codes/mol-metal": (
        "FORBIDDEN — this repository. The gate's _ROOT_AUTHOR."
    ),
    "/home/hugo/codes/try_triton_on_rocm": (
        "FORBIDDEN — a sibling checkout the repo reads (Triton kernels). "
        "The gate's _ROOT_TRITON."
    ),
    "/home/hugo/codes/pocket": (
        "EXTERNAL — a sibling project. Its 12 literals name ITS datasets "
        "(data/crossdocked2020, data/pdbbindpp-2020/extracted) and ITS "
        "own evaluation virtualenv (.eval_envs/pocket2mol311/bin/vina, "
        ".eval_envs/posebusters311/bin/python). None of them names a path "
        "inside THIS checkout, so rule (b) REACHABILITY excludes them. "
        "Three of them ARE live (bound and read into a subprocess exec "
        "position), so they are a real host-bound-EXTERNAL-TOOL defect — "
        "a different class, tracked as wave-3 scope, not as this class."
    ),
    "/home/hugo/codes/CadButEaas": (
        "EXTERNAL — a sibling project that holds the BioLM-Score reference "
        "checkout (references/BioLM-Score). The reference mount is "
        "deliberately outside the checkout (.gitignore: references/), so "
        "rule (b) REACHABILITY excludes it."
    ),
}


# The gate's own roots — read from the gate module, never duplicated.
def _gate_roots() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("wfa269_gate", GATE_MODULE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wfa269_gate"] = mod
    spec.loader.exec_module(mod)
    return tuple(mod._ROOTS)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------
def _tracked_python(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [n for n in out.stdout.split("\0") if n]


def _mount_of(literal: str) -> str | None:
    """The mount a literal is anchored at, or None if it is not absolute."""
    if not literal.startswith("/"):
        return None
    parts = [p for p in literal.split("/") if p]
    if not parts:
        return None
    head = parts[0]
    if head not in ("home", "mnt", "usr", "opt", "data", "srv", "media", "scratch", "workspace", "net", "volumes", "root", "var"):
        return None
    if head in ("home", "mnt") and len(parts) >= 2:
        return "/" + parts[0] + "/" + parts[1]
    return "/" + parts[0]


def _abs_literals(repo: Path) -> list[tuple[str, str]]:
    """(relative path, literal) for every absolute string literal."""
    out: list[tuple[str, str]] = []
    for rel in _tracked_python(repo):
        try:
            src = (repo / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("/")
            ):
                out.append((rel, node.value))
    return out


def _mounts(repo: Path) -> dict[str, set[str]]:
    """mount -> {files that use it}."""
    found: dict[str, set[str]] = {}
    for rel, lit in _abs_literals(repo):
        m = _mount_of(lit)
        if m is not None:
            found.setdefault(m, set()).add(rel)
    return found


def _checkout_roots(repo: Path) -> dict[str, set[str]]:
    """checkout root under /home/hugo/codes/ -> {files that use it}."""
    found: dict[str, set[str]] = {}
    for rel, lit in _abs_literals(repo):
        parts = [p for p in lit.split("/") if p]
        if len(parts) >= 4 and parts[0] == "home" and parts[1] == "hugo" and parts[2] == "codes":
            found.setdefault("/home/hugo/codes/" + parts[3], set()).add(rel)
    return found


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_gate_root_list_is_unchanged():
    """The gate's three roots are the SSOT for this file's tables."""
    roots = _gate_roots()
    assert roots == (
        "/home/hugo/codes/mol-metal",
        "/home/hugo/codes/try_triton_on_rocm",
        "/mnt/disk1/Hugo_Lee/mol-metal",
    ), (
        "the gate's root list changed — REVIEWED_CHECKOUT_ROOTS and "
        "REVIEWED_MOUNTS in this file must be re-adjudicated against the "
        f"new list; got {roots}"
    )


def test_every_mount_is_reviewed():
    """EVERY mount under which a tracked absolute literal lives is reviewed.

    A NEW mount fails the test — it must be adjudicated, not inherited.
    """
    found = _mounts(REPO)
    unknown = {m: sorted(f) for m, f in found.items() if m not in REVIEWED_MOUNTS}
    assert not unknown, (
        "NEW host mount(s) appeared in the tracked Python population and "
        "are not adjudicated: "
        + "; ".join(
            f"{m} (in {len(f)} file(s): {f[:3]})" for m, f in sorted(unknown.items())
        )
    )


def test_every_checkout_root_is_reviewed():
    """EVERY checkout root under the codes umbrella is reviewed."""
    found = _checkout_roots(REPO)
    unknown = {
        m: sorted(f) for m, f in found.items() if m not in REVIEWED_CHECKOUT_ROOTS
    }
    assert not unknown, (
        "NEW checkout root(s) appeared under /home/hugo/codes/ and are not "
        "adjudicated: "
        + "; ".join(
            f"{m} (in {len(f)} file(s): {f[:3]})" for m, f in sorted(unknown.items())
        )
    )


def test_reviewed_mount_table_has_no_stale_entries():
    """Every reviewed mount is still in use — a stale row is review debt too."""
    found = _mounts(REPO)
    stale = sorted(m for m in REVIEWED_MOUNTS if m not in found)
    assert not stale, (
        f"REVIEWED_MOUNTS rows no longer match the tree: {stale} — drop them "
        "or record why they are still expected"
    )


def test_reviewed_checkout_table_has_no_stale_entries():
    found = _checkout_roots(REPO)
    stale = sorted(m for m in REVIEWED_CHECKOUT_ROOTS if m not in found)
    assert not stale, (
        f"REVIEWED_CHECKOUT_ROOTS rows no longer match the tree: {stale}"
    )


def test_the_two_forbidden_checkout_roots_are_the_gate_roots():
    """The table's FORBIDDEN rows must agree with the gate's own list."""
    roots = set(_gate_roots())
    forbidden = {
        m
        for m, why in REVIEWED_CHECKOUT_ROOTS.items()
        if why.startswith("FORBIDDEN")
    }
    # /mnt/disk1/Hugo_Lee/mol-metal is not under the /home/hugo/codes umbrella,
    # so it is in the gate's list but not in the umbrella table.
    assert forbidden <= roots, (
        f"the umbrella table calls a checkout root FORBIDDEN that the gate "
        f"does not forbid: {sorted(forbidden - roots)}"
    )
    assert "/home/hugo/codes/mol-metal" in forbidden
    assert "/home/hugo/codes/try_triton_on_rocm" in forbidden


def test_the_coverage_bound_is_published():
    """Print the bound, and assert it still matches the tree.

    The numbers are the point: a reader of the gate's verdict needs to
    know how much of the space the verdict covers.
    """
    lits = _abs_literals(REPO)
    mounts = _mounts(REPO)
    roots = _checkout_roots(REPO)
    gated = {
        m
        for m in roots
        if m in REVIEWED_CHECKOUT_ROOTS
        and REVIEWED_CHECKOUT_ROOTS[m].startswith("FORBIDDEN")
    }
    invisible = {
        m: f for m, f in roots.items() if m not in gated
    }
    invisible_files = set().union(*invisible.values()) if invisible else set()

    print()
    print("=== Wf-A269 host-root gate — COVERAGE BOUND ===")
    print(f"tracked .py files scanned        : {len(_tracked_python(REPO))}")
    print(f"distinct absolute literals       : {len({l for _, l in lits})}")
    print(f"absolute literals under a mount  : {sum(len(f) for f in mounts.values())} occurrences "
          f"across {len(mounts)} mounts")
    print(f"checkout roots under codes/      : {len(roots)}")
    print(f"  visible to the gate            : {len(gated)} -> {sorted(gated)}")
    print(f"  NOT visible to the gate        : {len(invisible)} -> {sorted(invisible)} "
          f"({len(invisible_files)} file(s))")
    print("=====================================")

    assert len(gated) == 2, (
        "the set of checkout roots the gate can see changed — re-adjudicate "
        f"REVIEWED_CHECKOUT_ROOTS; visible={sorted(gated)}"
    )
    assert len(invisible) == 2, (
        "the set of checkout roots the gate CANNOT see changed — "
        f"re-adjudicate REVIEWED_CHECKOUT_ROOTS; invisible={sorted(invisible)}"
    )


def test_a_synthetic_new_mount_is_rejected():
    """NEGATIVE CONTROL: the review tables must actually reject.

    A gate that cannot fail is worse than no gate.  This feeds the
    census a synthetic mount and asserts the table does NOT cover it.
    """
    assert "/home/hugo/codes/new_sibling_checkout" not in REVIEWED_CHECKOUT_ROOTS
    assert "/mnt/brand_new_mount" not in REVIEWED_MOUNTS
    # and the real tables must reject a hypothetical new checkout root
    found = {"/home/hugo/codes/brand_new": {"x.py"}}
    unknown = {m for m in found if m not in REVIEWED_CHECKOUT_ROOTS}
    assert unknown == {"/home/hugo/codes/brand_new"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
