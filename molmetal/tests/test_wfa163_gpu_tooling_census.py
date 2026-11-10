# SPDX-License-Identifier: Apache-2.0
"""Wf-A163 — the GPU-tooling census must be RE-DERIVABLE, and its shell
classifier must be shown to work.

WHAT THIS TEST TREATS AS THE CONTRACT
=====================================
The census artefact (``gpu_tooling_census.json``) is only worth quoting if
two things hold:

1. every number in its ``summary`` is RE-DERIVABLE from its own ``hits``
   array -- otherwise the summary is an assertion, not a measurement;
2. its DENOMINATOR is coherent -- a count of "N sites" is meaningless
   without the number of files that were scanned.

And the scanner itself has to be shown to work on both the POSITIVE and the
NEGATIVE control, because a classifier that never fires is not a classifier.
The shell classifier is the one that had a real bug: it counted an ``echo``
message and a trailing ``# comment`` that merely MENTION
``CUDA_VISIBLE_DEVICES`` as mask writes.  Both cases are pinned below.

CPU-only.  The census script never imports torch, and neither does this
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts import wfa163_gpu_tooling_census as tool

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa163_gpu_tooling_census"
    / "gpu_tooling_census.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("census artefact not on disk")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. THE DENOMINATOR — a count of N is meaningless without it
# ---------------------------------------------------------------------------


def test_the_denominator_is_coherent(artifact):
    d = artifact["denominator"]
    assert d["files_enumerated_in_scope"] > 0
    assert d["files_read"] == d["py_files"] + d["sh_files"]
    assert d["files_read"] <= d["files_enumerated_in_scope"]
    assert d["files_unreadable"] == 0
    assert d["parse_failures"] == []
    # every category is scanned over the whole .py set; `session_lock` also
    # covers the .sh launchers
    for cat, n in d["files_scanned_per_category"].items():
        expected = d["files_read"] if cat == "session_lock" else d["py_files"]
        assert n == expected, (cat, n, expected)
    # per-scope files sum to the enumerated total
    assert sum(v["files"] for v in d["per_scope"].values()) == d[
        "files_enumerated_in_scope"
    ]


# ---------------------------------------------------------------------------
# 2. THE SUMMARY IS RE-DERIVABLE FROM THE HITS
# ---------------------------------------------------------------------------


def test_the_summary_counts_are_re_derivable_from_the_hits(artifact):
    hits = artifact["hits"]
    s = artifact["summary"]
    assert s["hits_total"] == len(hits)

    by_kind = {}
    for h in hits:
        by_kind[h["kind"]] = by_kind.get(h["kind"], 0) + 1
    assert s["hits_by_kind"] == by_kind

    probes = [h for h in hits if h["kind"] == "cuda_probe"]
    assert s["cuda_probes_total"] == len(probes)
    assert s["cuda_probes_that_allocate_a_context"] == sum(
        1 for h in probes if h["allocates_context"] == "yes"
    )
    assert s["cuda_probes_that_respect_the_ambient_mask"] == sum(
        1 for h in probes if h["api"] in ("is_available", "device_count")
    )
    assert s["cuda_probes_at_import_time"] == sum(
        1 for h in probes if h.get("module_level")
    )


def test_the_lock_defeater_list_is_re_derivable_from_the_hits(artifact):
    hits = artifact["hits"]
    derived = sorted(
        (h["file"], h["line"])
        for h in hits
        if h.get("effect") in ("defeats_lock", "sets_nonblank_or_dynamic")
        or h.get("env_mode") == "sets_cvd_nonblank"
    )
    listed = sorted(
        (h["file"], h["line"]) for h in artifact["question_b"]["lock_defeaters"]
    )
    assert derived == listed


def test_the_drop_lock_list_is_re_derivable_from_the_hits(artifact):
    hits = artifact["hits"]
    derived = sorted(
        (h["file"], h["line"])
        for h in hits
        if h["kind"] == "child_spawn" and h.get("env_mode") == "drops_lock"
    )
    listed = sorted(
        (h["file"], h["line"]) for h in artifact["question_b"]["lock_droppers"]
    )
    assert derived == listed


# ---------------------------------------------------------------------------
# 3. THE KNOWN SITES ARE FOUND — the census is not vacuous
# ---------------------------------------------------------------------------


def test_the_census_finds_the_known_import_time_probe(artifact):
    """``molmetal/utils/device.py`` probes CUDA at IMPORT time.

    This is the site's whole reason to exist: it is the one library module
    whose import initialises CUDA, and it is imported at module level by six
    trainers/adapters.
    """
    sites = [
        (h["file"], h["line"])
        for h in artifact["hits"]
        if h["kind"] == "cuda_probe" and h.get("module_level")
    ]
    assert ("molmetal/utils/device.py", 51) in sites
    assert ("molmetal/scripts/infer_cfm.py", 37) in sites


def test_the_census_finds_a_child_spawn_that_drops_the_lock(artifact):
    """``molmetal/tests/test_four_domain_sampler.py`` builds a hand-made env
    with no ``CUDA_VISIBLE_DEVICES`` key, so its child inherits nothing."""
    sites = {
        (h["file"], h["line"])
        for h in artifact["hits"]
        if h["kind"] == "child_spawn" and h.get("env_mode") == "drops_lock"
    }
    assert ("molmetal/tests/test_four_domain_sampler.py", 419) in sites


# ---------------------------------------------------------------------------
# 4. THE SHELL CLASSIFIER — positive AND negative controls
# ---------------------------------------------------------------------------


def test_the_shell_classifier_calls_a_real_write_a_write():
    value, scope = tool._shell_cvd_assignments(
        "CUDA_VISIBLE_DEVICES=0 setsid nohup ${PY} ${SCRIPT} \\"
    )[0]
    assert scope == "command_prefix"
    assert tool._classify_shell_value(value) == "sets_nonblank_or_dynamic"


def test_the_shell_classifier_does_NOT_call_an_echo_a_write():
    """THE REGRESSION for the false positive: an ``echo`` that mentions the
    variable writes nothing."""
    assert (
        tool._shell_cvd_assignments(
            'echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES '
            '(from yaml compute.gpu=$GPU)"'
        )
        == []
    )


def test_the_shell_classifier_does_NOT_call_a_comment_a_write():
    """THE REGRESSION for the false positive: a trailing comment that
    mentions the variable writes nothing."""
    assert (
        tool._shell_cvd_assignments(
            'GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 '
            "(CUDA_VISIBLE_DEVICES=0; G6 recovery holds GPU-1)"
        )
        == []
    )


def test_the_shell_classifier_distinguishes_the_two_parameter_forms():
    """``${VAR-default}`` keeps an EMPTY ambient; ``${VAR:-default}`` replaces
    it.  The nested-brace form must be recognised too."""
    assert (
        tool._classify_shell_value("${CUDA_VISIBLE_DEVICES-${GPU}}")
        == "preserves_blank"
    )
    assert (
        tool._classify_shell_value("${CUDA_VISIBLE_DEVICES:-0}") == "defeats_lock"
    )


def test_the_shell_classifier_calls_an_empty_export_a_forcer():
    value, scope = tool._shell_cvd_assignments('export CUDA_VISIBLE_DEVICES=""')[0]
    assert scope == "export"
    assert tool._classify_shell_value(value) == "forces_blank"


# ---------------------------------------------------------------------------
# 5. NO CUDA
# ---------------------------------------------------------------------------


def test_the_census_script_does_not_import_torch():
    """STATIC: the census cannot touch a GPU even by accident."""
    import ast

    src = (REPO_ROOT / "molmetal" / "scripts" / "wfa163_gpu_tooling_census.py").read_text(
        encoding="utf-8"
    )
    roots = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert "torch" not in roots
