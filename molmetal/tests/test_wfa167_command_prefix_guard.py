"""Wf-A167 — tests for the independent command-prefix CUDA_VISIBLE_DEVICES census.

Run ONLY this file, and always with the GPU hidden::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        -m pytest molmetal/tests/test_wfa167_command_prefix_guard.py -v

Four groups of checks:

1. NEGATIVE CONTROLS — a synthetic ``.sh`` carrying all four syntactic shapes
   (a real command prefix, an occurrence inside a quoted ``log "..."`` string, a
   conditional test, an already-repaired lock-preserving form).  ONLY the real
   command prefix may be reported as a defeat.
2. VACUITY GUARD — the reported count must equal the number of listed sites.
3. CPU-ONLY GUARD — the module's import graph contains no ``torch`` and it
   never shells out to ``nvidia-smi``.
4. DRIFT CHECK — the census file's per-file counts match a fresh scan.

The bash semantics that the classifier is built on were verified empirically on
the host with plain ``bash`` (no GPU involved):

    $ FOO=bar /usr/bin/env | grep '^FOO='      -> FOO=bar        (prefix applies)
    $ if ! FOO=bar /usr/bin/env; then :; fi    -> FOO=bar        (prefix applies)
    $ setsid nohup FOO=bar /usr/bin/env
      nohup: failed to run command 'FOO=bar'    (an ARGUMENT, not an assignment)

and the default-substitution forms:

    ${VAR-default}   substitutes only when VAR is UNSET   -> an empty lock survives
    ${VAR:-default}  substitutes when VAR is unset OR EMPTY -> the empty lock is lost
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
GUARD_PATH = os.path.join(REPO_ROOT, "molmetal", "scripts", "wfa167_command_prefix_guard.py")
CENSUS_PATH = os.path.join(
    REPO_ROOT, "molmetal", "reports", "wfa167_command_prefix_guard", "census.json"
)
PATCH_PATH = os.path.join(
    REPO_ROOT, "molmetal", "reports", "wfa167_command_prefix_guard", "patch_guard_extension.diff"
)


def _load_guard():
    spec = importlib.util.spec_from_file_location("wfa167_guard_under_test", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


# --------------------------------------------------------------------------- #
# 1. NEGATIVE CONTROLS
# --------------------------------------------------------------------------- #
SYNTHETIC = """#!/usr/bin/env bash
# (i) a real command prefix -- this is the ONLY shape that may be a defeat.
CUDA_VISIBLE_DEVICES=0 "$PY" train.py

# (ii) an occurrence inside a double-quoted log string -- not an assignment.
log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER}"

# (iii) a conditional test -- a read, not an assignment.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "no ambient mask"
fi

# (iv) an already-repaired, lock-preserving prefix -- must not be a defeat.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" "$PY" train.py
"""


def test_only_the_real_command_prefix_is_flagged():
    sites = guard.scan_shell_text(SYNTHETIC, "synthetic.sh")
    defeats = [s for s in sites if s["semantic_class"] == "REAL_DEFEATER"]
    assert len(defeats) == 1, "exactly one defeat expected, got %r" % (
        [(s["line"], s["shape"], s["value_raw"]) for s in defeats],
    )
    assert defeats[0]["shape"] == "COMMAND_PREFIX"
    assert defeats[0]["line"] == 3
    assert defeats[0]["value_raw"] == "0"


def test_the_four_shapes_are_distinguished():
    sites = guard.scan_shell_text(SYNTHETIC, "synthetic.sh")
    by_line = {}
    for site in sites:
        by_line.setdefault(site["line"], []).append(site)

    # (ii) the log string
    log_sites = [s for s in sites if s["shape"] == "QUOTED_STRING"]
    assert log_sites, "the log-string occurrence must be classified QUOTED_STRING"
    assert all(s["semantic_class"] == "BENIGN" for s in log_sites)

    # (iii) the conditional test
    test_sites = [s for s in sites if s["shape"] == "TEST"]
    assert test_sites, "the conditional test must be classified TEST"
    assert all(s["semantic_class"] == "BENIGN" for s in test_sites)

    # (iv) the already-repaired lock-preserving prefix
    preserved = [
        s
        for s in sites
        if s["shape"] == "COMMAND_PREFIX" and s["semantic_class"] == "ALREADY_LOCK_PRESERVING"
    ]
    assert len(preserved) == 1, "the lock-preserving prefix must be classified as such"
    assert preserved[0]["value_raw"] == '"${CUDA_VISIBLE_DEVICES-0}"'
    assert preserved[0]["defeats_ambient_empty_mask"] is False


def test_a_comment_occurrence_is_not_an_assignment():
    sites = guard.scan_shell_text("# CUDA_VISIBLE_DEVICES=0 in a comment\n", "c.sh")
    assert sites
    assert all(s["shape"] == "COMMENT" for s in sites)
    assert all(s["semantic_class"] == "BENIGN" for s in sites)


def test_an_argument_position_assignment_is_not_an_env_assignment():
    """`cmd VAR=val` passes VAR=val as an argv element; it is NOT a prefix."""
    text = 'setsid nohup     CUDA_VISIBLE_DEVICES="${WF_CVD:-0}" \\\n  "$PY" -c "x"\n'
    sites = guard.scan_shell_text(text, "arg.sh")
    assert len(sites) == 1
    assert sites[0]["shape"] == "ARG_POSITION_ASSIGNMENT"
    assert sites[0]["semantic_class"] == "BENIGN"
    assert sites[0]["defeats_ambient_empty_mask"] is False


def test_a_prefix_after_a_control_operator_is_still_a_prefix():
    text = "foo && CUDA_VISIBLE_DEVICES=0 bar\n"
    sites = guard.scan_shell_text(text, "and.sh")
    assert len(sites) == 1
    assert sites[0]["shape"] == "COMMAND_PREFIX"
    assert sites[0]["semantic_class"] == "REAL_DEFEATER"


def test_a_prefix_after_a_keyword_is_still_a_prefix():
    for text in (
        "if ! CUDA_VISIBLE_DEVICES=0 cmd; then echo x; fi\n",
        "if CUDA_VISIBLE_DEVICES=0 cmd; then echo x; fi\n",
    ):
        sites = guard.scan_shell_text(text, "kw.sh")
        assert len(sites) == 1, text
        assert sites[0]["shape"] == "COMMAND_PREFIX", text
        assert sites[0]["semantic_class"] == "REAL_DEFEATER", text


def test_the_colon_default_form_defeats_and_the_no_colon_form_does_not():
    defeating = guard.classify_value('"${CUDA_VISIBLE_DEVICES:-0}"')
    preserving = guard.classify_value('"${CUDA_VISIBLE_DEVICES-0}"')
    assert defeating["defeats_ambient_empty_mask"] is True
    assert preserving["defeats_ambient_empty_mask"] is False
    assert preserving["effect"] == "preserves_ambient"


def test_a_self_reference_to_a_DIFFERENT_variable_still_defeats():
    """`${WF310C_CUDA_VISIBLE_DEVICES:-1}` is not a self reference."""
    verdict = guard.classify_value('"${WF310C_CUDA_VISIBLE_DEVICES:-1}"')
    assert verdict["self_reference"] is False
    assert verdict["defeats_ambient_empty_mask"] is True


def test_the_lock_preserving_form_uses_the_no_colon_default():
    form = guard.lock_preserving_form("0")
    assert form == 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"'
    assert ":-" not in form


# --------------------------------------------------------------------------- #
# 2. VACUITY GUARD
# --------------------------------------------------------------------------- #
def test_census_counts_match_the_listed_sites():
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)
    counts = census["counts"]
    by_class = counts["by_class"]
    listed = sum(by_class.values())
    assert listed == counts["sites_total"], (
        "class counts %r must sum to sites_total %r" % (by_class, counts["sites_total"])
    )
    assert counts["sites_total"] == len(census["sites"])
    assert counts["real_defeaters"] == len(census["real_defeater_sites"])
    assert census["self_check"]["counts_match_lists"] is True
    assert census["self_check"]["real_defeater_list_matches_count"] is True


# --------------------------------------------------------------------------- #
# 2b. ATTRIBUTABLE DRIFT DIAGNOSTIC
#
# The drift check compares COUNT VECTORS only.  When it goes red the reader
# was told "112 != 111" and had to re-derive the drifting file by hand.
# The census already records ``snapshot.per_file_sha256`` for every tracked
# ``*.sh``, so the attribution is read-only-provable: compare the census's
# OWN recorded shas against the live tree — NOT against git, because git
# reports the COMMITTED state while the census pinned the working tree.
#
# THE COMPARISON IS NOT WEAKENED: the same four full count vectors are still
# compared and any drift still fails the test.  Only the MESSAGE gains the
# attribution.
# --------------------------------------------------------------------------- #
def _drifting_files(census, live=None, repo_root=REPO_ROOT):
    """Return a sorted list of ``(path, census_sha256, live_sha256)``.

    ``live`` defaults to a fresh scan of the working tree.  A path present
    on only one side is reported with ``None`` for the missing side, so a
    file that was added or removed is attributed too.
    """
    recorded = (census.get("snapshot") or {}).get("per_file_sha256") or {}
    if live is None:
        live = guard.scan_repository(repo_root)["snapshot"]["per_file_sha256"]
    drift = []
    for rel in sorted(set(recorded) | set(live)):
        census_sha = recorded.get(rel)
        live_sha = live.get(rel)
        if census_sha != live_sha:
            drift.append((rel, census_sha, live_sha))
    return drift


def _drift_message(drift, limit=20):
    """Render the attribution for a failure message."""
    if not drift:
        return "no file-level sha256 drift (the census snapshot matches the live tree)"
    rows = [
        "%s: census=%s live=%s"
        % (
            rel,
            csha if csha is not None else "<absent>",
            lsha if lsha is not None else "<absent>",
        )
        for rel, csha, lsha in drift[:limit]
    ]
    if len(drift) > limit:
        rows.append("... and %d more drifting file(s)" % (len(drift) - limit))
    return "sha256 drift vs the census snapshot: " + "; ".join(rows)


def _assert_counts_match(report, census):
    """The FULL count-vector comparison + the attribution.

    The four comparisons below are the ORIGINAL assertions of
    ``test_fresh_scan_counts_match_the_census`` — unchanged, and NOT
    relaxed to a subset.  Only the failure MESSAGE gains the drifting-file
    list, so a red run names the file that moved instead of a bare count
    delta.
    """
    drift = _drifting_files(census, live=report["snapshot"]["per_file_sha256"])
    message = _drift_message(drift)
    assert report["counts"]["by_shape"] == census["counts"]["by_shape"], (
        "by_shape drifted from the census; " + message
    )
    assert report["counts"]["by_class"] == census["counts"]["by_class"], (
        "by_class drifted from the census; " + message
    )
    assert report["counts"]["sites_total"] == census["counts"]["sites_total"], (
        "sites_total drifted from the census; " + message
    )
    assert report["denominator"] == census["denominator"], (
        "denominator drifted from the census; " + message
    )


def test_fresh_scan_counts_match_the_census():
    report = guard.scan_repository(REPO_ROOT)
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)
    _assert_counts_match(report, census)


def test_every_real_defeater_carries_a_file_and_a_line():
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)
    for site in census["real_defeater_sites"]:
        assert site["file"], site
        assert isinstance(site["line"], int) and site["line"] > 0, site
        assert site["reason"], site
        assert site["lock_preserving_form"].startswith("CUDA_VISIBLE_DEVICES=")


# --------------------------------------------------------------------------- #
# 3. CPU-ONLY GUARD
# --------------------------------------------------------------------------- #
def test_the_module_imports_no_torch_and_no_gpu_runtime():
    with open(GUARD_PATH, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    assert "torch" not in imported
    assert not (imported & {"cupy", "pynvml", "tensorflow", "jax"})
    assert "subprocess" in imported

    # No argv list may contain a GPU tool.  (The module DOCSTRING mentions
    # nvidia-smi in prose; a docstring is not an invocation.)
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for element in node.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    assert "nvidia-smi" not in element.value, element.value
                    assert "nvcc" not in element.value, element.value
                    assert "nvidia" not in element.value.lower(), element.value

    # No long-lived subprocess may be spawned.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "Popen", "no long-lived subprocesses"


def test_importing_the_guard_does_not_pull_in_torch():
    code = (
        "import importlib.util, sys;"
        "spec=importlib.util.spec_from_file_location('g', %r);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print('torch' in sys.modules)" % GUARD_PATH
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout


def test_the_guard_only_shells_out_to_git():
    """Every subprocess the scanner spawns must be a `git` invocation."""
    seen = []
    original = subprocess.run

    def _recording_run(args, *a, **kw):
        seen.append(args)
        return original(args, *a, **kw)

    subprocess.run = _recording_run
    try:
        report = guard.scan_repository(REPO_ROOT)
    finally:
        subprocess.run = original
    assert report["counts"]["sites_total"] > 0
    assert seen, "the scanner is expected to enumerate tracked files via git"
    for argv in seen:
        assert argv[0] == "git", argv
        assert not any("nvidia" in str(part).lower() for part in argv), argv


# --------------------------------------------------------------------------- #
# 4. DRIFT CHECK
# --------------------------------------------------------------------------- #
def test_the_census_per_file_counts_match_a_fresh_scan():
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)
    fresh = guard.scan_repository(REPO_ROOT)
    sha = census["snapshot"]["per_file_sha256"]

    changed = []
    for rel, counts in census["per_file"].items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            changed.append(rel)
            continue
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != sha.get(rel):
            changed.append(rel)
            continue
        assert fresh["per_file"][rel] == counts, "drift in %s" % rel
    # Files whose content changed since the census are reported as drift.
    #
    # SUPERSEDED (2026-11-04, Wf-A269 stale-pins sweep — PRESERVED VERBATIM
    # per AGENTS.md §5.5 rule #6).  The old assertion demanded that a
    # content change MUST also change the classification counts:
    #     for rel in changed:
    #         assert rel not in fresh["per_file"] or fresh["per_file"][rel] != census["per_file"][rel]
    # THAT PREMISE IS FALSE: a classification-neutral edit (a log message, a
    # comment, a whitespace change) leaves the count record IDENTICAL, so the
    # old assertion reported a FALSE POSITIVE on a legitimate edit.
    # MEASURED 2026-11-04: logs/lambda_p4_sweep/run_seed_sweep.sh changed
    # content since the 2026-11-01 census (sha mismatch) and its count record
    # is identical (all-zero — the file carries no CUDA_VISIBLE_DEVICES
    # site), so the old assertion FAILED on a tree that had not drifted in
    # any way the census cares about.
    #
    # REPAIRED to the TRUE invariant.  The strict half is UNCHANGED: a
    # census file whose sha is UNCHANGED must have byte-identical counts
    # (the assertion above, "drift in %s").  The changed half now asserts
    # what is actually true — a file that is gone from disk must not still
    # be scanned, and a file that still exists must still be enumerated.
    for rel in changed:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            assert rel not in fresh["per_file"], (
                "census lists %s but it no longer exists on disk and still "
                "appears in the fresh scan" % rel
            )
            continue
        assert rel in fresh["per_file"], (
            "census lists %s but the fresh scan does not enumerate it" % rel
        )


def test_the_census_denominator_is_reproducible():
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)
    fresh = guard.scan_repository(REPO_ROOT)
    assert fresh["denominator"]["tracked_sh_files_total"] == census["denominator"]["tracked_sh_files_total"]
    assert fresh["denominator"]["files_read"] == census["denominator"]["files_read"]
    assert fresh["denominator"]["files_unreadable"] == 0


# --------------------------------------------------------------------------- #
# 4b. THE DIAGNOSTIC IS CORRECT AND THE CHECK CAN STILL FAIL
#
# MUTATION PROOF + NEGATIVE CASE.  Both operate on a COPY of the census in a
# tmp path — the published artefact is never mutated.
# --------------------------------------------------------------------------- #
def test_the_drift_diagnostic_names_the_drifting_file(tmp_path):
    """MUTATION PROOF: the check goes RED and the message NAMES the file."""
    report = guard.scan_repository(REPO_ROOT)
    live = report["snapshot"]["per_file_sha256"]
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)

    common = sorted(set(census["snapshot"]["per_file_sha256"]) & set(live))
    assert common, "the census and the live tree must share at least one file"
    rel = common[0]

    mutated = json.loads(json.dumps(census))
    # (1) the recorded sha of `rel` no longer matches the live file
    mutated["snapshot"]["per_file_sha256"] = dict(live)
    mutated["snapshot"]["per_file_sha256"][rel] = "f" * 64
    # (2) one count is changed -> the count vectors must disagree
    mutated["counts"]["sites_total"] = mutated["counts"]["sites_total"] + 1
    mutated["counts"]["by_shape"]["COMMENT"] = (
        mutated["counts"]["by_shape"]["COMMENT"] + 1
    )

    tmp_census = tmp_path / "census_mutated.json"
    with open(tmp_census, "w", encoding="utf-8") as handle:
        json.dump(mutated, handle)
    with open(tmp_census, "r", encoding="utf-8") as handle:
        from_tmp = json.load(handle)

    with pytest.raises(AssertionError) as excinfo:
        _assert_counts_match(report, from_tmp)
    message = str(excinfo.value)
    assert rel in message, message
    assert "f" * 64 in message, message
    assert live[rel] in message, message


def test_a_count_only_mutation_is_still_red_and_reports_no_sha_drift():
    """A count-only drift is RED, and the diagnostic does NOT invent drift."""
    report = guard.scan_repository(REPO_ROOT)
    live = report["snapshot"]["per_file_sha256"]
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)

    mutated = json.loads(json.dumps(census))
    mutated["snapshot"]["per_file_sha256"] = dict(live)  # no sha drift
    mutated["counts"]["sites_total"] = mutated["counts"]["sites_total"] + 1

    with pytest.raises(AssertionError) as excinfo:
        _assert_counts_match(report, mutated)
    assert "no file-level sha256 drift" in str(excinfo.value)


def test_the_drift_diagnostic_is_empty_when_nothing_drifts():
    """NEGATIVE case: nothing drifts => the diagnostic returns []."""
    report = guard.scan_repository(REPO_ROOT)
    live = report["snapshot"]["per_file_sha256"]
    with open(CENSUS_PATH, "r", encoding="utf-8") as handle:
        census = json.load(handle)

    synced = json.loads(json.dumps(census))
    synced["snapshot"]["per_file_sha256"] = dict(live)
    assert _drifting_files(synced, live=live) == []

    # a file the census knows and the live tree does not IS a drift
    absent = json.loads(json.dumps(census))
    absent["snapshot"]["per_file_sha256"] = dict(live)
    absent["snapshot"]["per_file_sha256"]["scripts/__never_tracked__.sh"] = "0" * 64
    drifted = _drifting_files(absent, live=live)
    assert [row[0] for row in drifted] == ["scripts/__never_tracked__.sh"]
    assert drifted[0][1] == "0" * 64
    assert drifted[0][2] is None


# --------------------------------------------------------------------------- #
# 5. The patch (a SPEC, not an application)
# --------------------------------------------------------------------------- #
def test_the_patch_is_a_unified_diff_for_the_a165_guard():
    assert os.path.exists(PATCH_PATH)
    with open(PATCH_PATH, "r", encoding="utf-8") as handle:
        patch = handle.read()
    assert "--- a/molmetal/scripts/wfa165_gpu_lock_guard.py" in patch
    assert "+++ b/molmetal/scripts/wfa165_gpu_lock_guard.py" in patch
    assert "@@" in patch
    # the patch must add the classification, the detection rule and the
    # denominator, and must not remove the existing rules
    assert "command_prefix_census" in patch
    assert "scan_command_prefix_census" in patch
    assert "sh_files_enumerated" in patch
    assert "arg_position" in patch
    for existing in (
        "M1_hard_override",
        "M2_empty_default_substitution",
        "_SHELL_MODS_RE",
        "_classify_shell_value",
    ):
        assert existing in patch or existing in open(
            os.path.join(REPO_ROOT, "molmetal", "scripts", "wfa165_gpu_lock_guard.py"),
            "r",
            encoding="utf-8",
        ).read()
    # no removed line may delete an existing rule
    removed = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    assert not [r for r in removed if "M1_hard_override" in r and "mechanism" in r]


def test_the_patch_applies_cleanly_to_a_scratch_copy(tmp_path):
    """`git apply --check` against a SCRATCH COPY; the working tree is untouched."""
    target = "molmetal/scripts/wfa165_gpu_lock_guard.py"
    scratch = tmp_path / "scratch"
    (scratch / "molmetal" / "scripts").mkdir(parents=True)
    src = os.path.join(REPO_ROOT, target)
    with open(src, "r", encoding="utf-8") as handle:
        original = handle.read()
    with open(scratch / target, "w", encoding="utf-8") as handle:
        handle.write(original)

    proc = subprocess.run(
        ["git", "init", "-q", "."], cwd=str(scratch), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run(
        ["git", "apply", "--check", PATCH_PATH],
        cwd=str(scratch),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "git apply --check failed: %s" % proc.stderr

    proc = subprocess.run(
        ["git", "apply", PATCH_PATH],
        cwd=str(scratch),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    with open(scratch / target, "r", encoding="utf-8") as handle:
        patched = handle.read()
    assert "command_prefix_census" in patched
    # the working tree copy must be untouched
    with open(src, "r", encoding="utf-8") as handle:
        assert handle.read() == original


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
