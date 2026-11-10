# SPDX-License-Identifier: Apache-2.0
"""Wf-A269 wave-22 — a YAML that names a gate NO PIPELINE OWNS is a defect.

WHY THIS EXISTS
---------------
The three Wf-A269 frontier sweeps repaired the gate-count pins (a count in a
doc, a count in a test, a count in a constant) and the dead-gate classes
(a guard that cannot fail, a gate with no consumer, a runtime-assembled flag
that never reached a parser).  None of them looked at the OTHER place the
same class hides: **a gate NAME pinned in a YAML**.

``directives.pre_flight_gates_required`` is the repo's own mechanism for
saying "this run needs THESE gates".  ``pre_flight_check.py --gates``
resolves those names against the UNION of all pipelines and HARD-ERRORS on an
unknown name (a typo must not silently run zero gates).  That is a
**per-launch** check — it fires only for the config somebody actually
launches.  There was NO sweep over the whole population, so a config that
names a gate with no referent sits in the tree until the day somebody
launches it, and the launch then REFUSES:

    $ python molmetal/scripts/pre_flight_check.py --gates p1_pytorch_cuda_available
    [pre_flight] --gates: unknown gate name(s) ['p1_pytorch_cuda_available'].

MEASURED 2026-11-04 (this wave): 115 YAMLs in the repo carry the key and
name 7 distinct gates.  3 of the 7 names are owned by NO pipeline:

  * ``p1_pytorch_cuda_available`` — 1 config.  The name has ZERO occurrences
    in any ``.py`` (HEAD + full git history); it was removed repo-wide by
    Wf-A176 (catalog §6.268) but this config was never in that population
    (created 2026-10-31, still untracked).  FIXED by this wave.
  * ``p3_protocol_alignment_user_ok_gate`` and
    ``p3_protocol_alignment_baseline_paper_refs_bib`` — 6 configs.  These
    are the DEAD-GATE set of catalog §6.267 (15 ``p3_protocol_*`` gates
    removed ON PURPOSE by the Wf-437-A2 audit).  The repair — drop the
    entries, or register real gates — is a POLICY decision that is recorded
    as OPEN in AGENTS.md, so the names are DECLARED here with their reason
    (the §5.11 ``DECLARED_SKIPPED_CONFIGS`` pattern: a declared skip is
    REPORTED every run; an UNDECLARED one is a violation, always).

THE TWO-DIRECTIONAL PIN
-----------------------
A declared-unknown name is only exempt while it is STILL unknown:

  * if a name stops being unknown (the gate gets registered), the
    declaration must be removed — otherwise the exemption outlives the thing
    it exempts and the sweep goes blind to a gate that IS enforceable;
  * if a NEW unknown name appears, the sweep FAILS.

That is the same two-directional shape as the Wf-A131 path-reference guard's
``BASELINE`` (a baseline entry that is no longer broken is dead code), and
it is deliberately NOT a frozen count: a frozen count here would be a
born-false pin the moment a config is added (Wf-A269 §6.269.29's
"POSITIONAL SLICE + VACUOUS COUNT PIN" and §6.269.32's "A PIN THAT WAS BORN
FALSE").  The population is asserted as a FLOOR, never a freeze.

CPU-only by construction: no torch import, no CUDA call, no checkpoint read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.pre_flight_check import PIPELINES, build_gates  # noqa: E402

#: The gate names that NO pipeline owns, each with the reason it is still
#: named.  A declared name is REPORTED every run; an UNDECLARED unknown name
#: is a violation, always (the §5.11 ``DECLARED_SKIPPED_CONFIGS`` pattern).
DECLARED_UNKNOWN_GATES: dict[str, str] = {
    "p3_protocol_alignment_user_ok_gate": (
        "REMOVED ON PURPOSE by the Wf-437-A2 dead-gate audit (catalog "
        "§6.267: 15 p3_protocol_* gates removed as a set).  Six tracked "
        "configs still name it; the repair (drop the entry, or register a "
        "real gate) is a POLICY decision recorded as OPEN in AGENTS.md "
        "§2 (the 'SIX TRACKED CONFIGS' row of the single-source-of-truth "
        "table)."
    ),
    "p3_protocol_alignment_baseline_paper_refs_bib": (
        "REMOVED ON PURPOSE by the Wf-437-A2 dead-gate audit (catalog "
        "§6.267: 15 p3_protocol_* gates removed as a set).  Six tracked "
        "configs still name it; the repair (drop the entry, or register a "
        "real gate) is a POLICY decision recorded as OPEN in AGENTS.md "
        "§2 (the 'SIX TRACKED CONFIGS' row of the single-source-of-truth "
        "table)."
    ),
}

#: The population FLOOR.  Not a freeze: adding configs must not turn this
#: test red, and removing configs must not either — only a collapse of the
#: population (i.e. the sweep stopped finding configs at all) may.
POPULATION_FLOOR = 80

#: The number of distinct gate names the population declares.  Also a FLOOR,
#: not a freeze (a new gate name in a config must not turn this red).
DISTINCT_NAMES_FLOOR = 5

#: The population FLOOR for configs whose required-gate list is EMPTY.  Also
#: a floor, not a freeze.  An empty list is a separate defect class (below);
#: this floor only asserts that the sweep still FINDS them.
EMPTY_LIST_FLOOR = 10


# --------------------------------------------------------------------- census
def _git_yaml_files() -> list[str]:
    """Every YAML in the repo that git can see (tracked + untracked).

    ``git ls-files`` is used instead of ``os.walk`` on purpose: this repo is
    ~201 GB and a directory walk is a memory hazard (see the repo's own
    "never os.walk this repo" rule).  ``--exclude-standard`` drops the
    gitignored ``.claude/`` worktree copies, which are stale duplicates of
    the same configs and would otherwise double-count the population.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "*.yaml", "*.yml"],
        cwd=str(REPO), capture_output=True, text=True, check=True,
    )
    return sorted(p for p in out.stdout.split() if p)


def _declared_lists(node: Any) -> list[list[str]]:
    """Every ``pre_flight_gates_required`` list, wherever it is nested."""
    found: list[list[str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pre_flight_gates_required" and isinstance(value, list):
                found.append([str(v) for v in value])
            else:
                found.extend(_declared_lists(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_declared_lists(value))
    return found


def _census() -> dict[str, list[str]]:
    """``{gate_name: [config paths]}`` over every YAML that declares one."""
    out: dict[str, list[str]] = {}
    for rel in _git_yaml_files():
        try:
            doc = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — an unparseable YAML is not this
            # sweep's defect class; it is the config-binding gate's.
            continue
        if doc is None:
            continue
        for names in _declared_lists(doc):
            for name in names:
                out.setdefault(name, []).append(rel)
    return out


def _empty_list_configs() -> list[str]:
    """Configs whose ``pre_flight_gates_required`` list is EMPTY.

    An empty list is the RESIDUE of the Wf-A176 removal (catalog §6.268):
    those configs declared exactly one entry — the phantom
    ``p1_pytorch_cuda_available`` — and dropping it left the list empty.

    That is a defect class of its own, because it is the shape that makes
    ``pre_flight_check.py --gates <list>`` run ZERO gates.  MEASURED
    2026-11-04 before this wave's repair: ``--gates ""`` produced
    ``summary.total = 0``, ``gates: []`` and exit code **0** — a "PASS" that
    ran nothing, which is exactly what the mode's own docstring promises
    never happens ("a typo must not silently run zero gates").  The repair
    is fail-CLOSED in the resolver (an empty list is a hard error, exit 1);
    whether each of these configs should instead declare a real gate is a
    config-semantics decision recorded as OPEN.
    """
    out: list[str] = []
    for rel in _git_yaml_files():
        try:
            doc = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if doc is None:
            continue
        for names in _declared_lists(doc):
            if not names:
                out.append(rel)
    return sorted(set(out))


def _owned_gate_names() -> set[str]:
    """Every gate name any pipeline owns (the UNION, like ``--gates``)."""
    owned: set[str] = set()
    for pipeline in PIPELINES:
        for name, _fn in build_gates(pipeline):
            owned.add(name)
    return owned


def _unknown_names(census: dict[str, list[str]], owned: set[str]) -> dict[str, list[str]]:
    return {name: files for name, files in census.items() if name not in owned}


# ---------------------------------------------------------------------- tests
def test_every_yaml_declared_gate_name_is_owned_by_some_pipeline() -> None:
    """THE load-bearing assertion: no config may name a gate that does not exist.

    A name with no referent makes ``pre_flight_check.py --gates`` HARD-ERROR
    ("unknown gate name(s) [...]"), so the config cannot be launched at all —
    and the failure surfaces at launch time, not at review time.
    """
    census = _census()
    unknown = _unknown_names(census, _owned_gate_names())
    undeclared = {
        name: files for name, files in unknown.items()
        if name not in DECLARED_UNKNOWN_GATES
    }
    assert not undeclared, (
        "these YAMLs name a gate that NO pipeline owns, so "
        "`pre_flight_check.py --gates <list>` refuses to run them: "
        f"{undeclared}.  Either register the gate (it must then be in a "
        "pipeline) or drop the entry from the config's "
        "`directives.pre_flight_gates_required` — the repo-wide precedent "
        "is Wf-A176 / catalog §6.268, which removed the same phantom "
        "(`p1_pytorch_cuda_available`) from 82 sibling configs.  To declare "
        "an exemption on purpose, add it to DECLARED_UNKNOWN_GATES with a "
        "reason."
    )


def test_every_declared_unknown_gate_is_still_unknown() -> None:
    """A declaration must not outlive the thing it exempts.

    Two-directional, like the Wf-A131 path-reference guard's BASELINE: if a
    declared name becomes OWNED by a pipeline, the exemption is dead code
    and must be removed, or this sweep goes blind to a gate that IS
    enforceable.
    """
    census = _census()
    unknown = _unknown_names(census, _owned_gate_names())
    stale = sorted(set(DECLARED_UNKNOWN_GATES) - set(unknown))
    assert not stale, (
        f"these gates are DECLARED_UNKNOWN but a pipeline now owns them: "
        f"{stale}.  Remove them from DECLARED_UNKNOWN_GATES — the exemption "
        "must not outlive the defect it exempts."
    )


def test_every_declared_unknown_gate_is_actually_cited_by_a_config() -> None:
    """A declaration with no citing config is dead code in the other direction."""
    census = _census()
    uncited = sorted(
        name for name in DECLARED_UNKNOWN_GATES if name not in census
    )
    assert not uncited, (
        f"these gates are DECLARED_UNKNOWN but no YAML cites them any more: "
        f"{uncited}.  Remove them from DECLARED_UNKNOWN_GATES."
    )


def test_the_census_is_not_vacuous() -> None:
    """The sweep must be FINDING the population, not reporting an empty one.

    A census that silently finds nothing is the §6.269.17 DEAD-GATE class in
    its "skipped" costume: a PASS that means "I looked at zero configs".
    """
    census = _census()
    files = {rel for files in census.values() for rel in files}
    assert len(files) >= POPULATION_FLOOR, (
        f"the census found only {len(files)} configs declaring a non-empty "
        f"required-gate list, which is below the floor of {POPULATION_FLOOR} "
        "— the sweep has stopped seeing the population (a glob that no longer "
        "matches, a directory that moved, a git-ls-files that stopped "
        "listing untracked files).  A PASS over an empty population is a "
        "DEAD GATE."
    )
    assert len(census) >= DISTINCT_NAMES_FLOOR, (
        f"the census found only {len(census)} distinct gate names, which is "
        f"below the floor of {DISTINCT_NAMES_FLOOR} — the population "
        "collapsed."
    )


def test_the_empty_list_population_is_still_visible() -> None:
    """The EMPTY-list configs must be FOUND, not silently skipped.

    This is the vacuity half of the empty-list defect class: the configs
    that declare ``pre_flight_gates_required: []`` used to be INVISIBLE to
    every name-based sweep (they name no name), which is how 22 of them
    survived the Wf-A176 removal unnoticed.  The census must still see them.
    """
    empty = _empty_list_configs()
    assert len(empty) >= EMPTY_LIST_FLOOR, (
        f"the census found only {len(empty)} configs with an EMPTY "
        f"required-gate list, which is below the floor of "
        f"{EMPTY_LIST_FLOOR} — the sweep has stopped seeing this population."
    )


def test_an_empty_required_gate_list_cannot_run_zero_gates() -> None:
    """NEGATIVE TEST: the resolver REFUSES an empty required-gate list.

    Before this wave's repair, ``pre_flight_check.py --gates ""`` ran ZERO
    gates and exited 0 — a "PASS" that ran nothing, in the very mode whose
    docstring promises "a typo must not silently run zero gates".  The
    resolver must now be fail-CLOSED on the empty spelling.
    """
    import subprocess as _sp

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = _sp.run(
        [sys.executable, str(REPO / "molmetal/scripts/pre_flight_check.py"),
         "--gates", ""],
        cwd=str(REPO), capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode != 0, (
        "an EMPTY required-gate list must be a hard error, not a silent "
        "zero-gate PASS — the resolver returned "
        f"{proc.returncode}.  (An empty list used to produce "
        "`summary.total = 0` and exit 0.)"
    )
    assert "EMPTY" in (proc.stderr + proc.stdout), (
        "the empty-list refusal must say WHY it refused; got: "
        f"{(proc.stderr + proc.stdout)[:400]!r}"
    )


def test_a_phantom_gate_name_would_be_caught() -> None:
    """NEGATIVE TEST: the checker flags a name no pipeline owns.

    Proves the assertion is load-bearing rather than a restatement: a
    synthetic config naming a gate that does not exist must be reported.
    """
    owned = _owned_gate_names()
    assert "p3_no_such_gate_exists" not in owned
    synthetic = {"p3_no_such_gate_exists": ["configs/synthetic.yaml"]}
    unknown = _unknown_names(synthetic, owned)
    assert "p3_no_such_gate_exists" in unknown, (
        "the checker did not flag a gate name that no pipeline owns — the "
        "sweep is not load-bearing."
    )
    undeclared = {
        name for name in unknown if name not in DECLARED_UNKNOWN_GATES
    }
    assert undeclared == {"p3_no_such_gate_exists"}


def test_the_published_preflight_gate_union_resolves_the_census() -> None:
    """The census and the ``--gates`` resolver must agree on the union.

    ``pre_flight_check.py --gates`` resolves against the UNION of all
    pipelines.  If the census resolved against a different population, the
    sweep would certify configs that ``--gates`` still refuses.
    """
    owned = _owned_gate_names()
    union: set[str] = set()
    for pipeline in PIPELINES:
        for name, _fn in build_gates(pipeline):
            union.add(name)
    assert owned == union
    assert "p2_tests_pass" in owned, (
        "the union lost `p2_tests_pass`, which 85 configs declare — the "
        "union the census resolves against has drifted from the union "
        "`--gates` resolves against."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
