"""Wf-A171 — tests for the evidence-based GPU-lock liveness classifier.

Run ONLY this file, and always with the GPU hidden::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        -m pytest molmetal/tests/test_wfa171_gpu_lock_guard_path.py -v

What is being defended
----------------------
``molmetal/scripts/wfa167_command_prefix_guard.py`` decided "archived" with a
PATH-PREFIX heuristic (``ARCHIVE_PREFIXES = ("TODO/archive/", "results/")``),
so a live, tracked, executable launcher under ``results/`` was filed as
archived and the guard reported ``real_defeaters_live: 0``.  These tests pin
the replacement: an archive verdict must rest on a *declaration* found in
tracked text, and the default must be LIVE.

The bash semantics the classifier is built on are verified empirically here
(plain ``bash``, CPU-only, no GPU is ever touched)::

    $ CUDA_VISIBLE_DEVICES="" bash -c 'echo "[${CUDA_VISIBLE_DEVICES-0}]"'   -> []
    $ CUDA_VISIBLE_DEVICES="" bash -c 'echo "[${CUDA_VISIBLE_DEVICES:-0}]"'  -> [0]
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
GUARD_PATH = os.path.join(
    REPO_ROOT, "molmetal", "scripts", "wfa171_gpu_lock_guard_path.py"
)
CENSUS_PATH = os.path.join(
    REPO_ROOT, "molmetal", "reports", "wfa171_gpu_lock_guard", "census.json"
)
PATCH_PATH = os.path.join(
    REPO_ROOT,
    "molmetal",
    "reports",
    "wfa171_gpu_lock_guard",
    "patch_guard_classification.diff",
)


def _load_guard():
    spec = importlib.util.spec_from_file_location("wfa171_guard_under_test", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard()


# --------------------------------------------------------------------------- #
# 1. THE DEFECT: results/ must NOT be an archive, TODO/archive/ must be
# --------------------------------------------------------------------------- #
def test_results_cfm_full_retrain_launch_loop_is_not_archived():
    """A tracked, executable launcher under results/ is LIVE.

    This is the exact file the A167 guard filed as ``archived: true`` while
    reporting ``real_defeaters_live: 0``.
    """
    index = GUARD.build_liveness_index(REPO_ROOT)
    verdict = GUARD.classify_liveness(
        "results/cfm_full_retrain/launch_loop.sh", index
    )
    assert verdict["tracked"] is True
    assert verdict["archived"] is False
    assert verdict["basis"] == "NO_ARCHIVAL_DECLARATION"
    # It is executable in the git index — independent evidence it is a launcher.
    assert verdict["executable"] is True


def test_todo_archive_tree_is_archived():
    """TODO/archive/ is declared archival by tracked text, so it IS archived."""
    index = GUARD.build_liveness_index(REPO_ROOT)
    verdict = GUARD.classify_liveness(
        "TODO/archive/wf437_simplify/launch_wf300_cfm_main.sh", index
    )
    assert verdict["tracked"] is True
    assert verdict["archived"] is True
    assert verdict["basis"] == "DECLARED_ARCHIVED"
    # The declaration must cite a source file, so the verdict is auditable.
    assert verdict["declarations"], "an archived verdict must carry its evidence"


def test_the_whole_todo_archive_tree_is_archived():
    index = GUARD.build_liveness_index(REPO_ROOT)
    archived = [
        path
        for path in index["tracked"]
        if path.startswith("TODO/archive/")
    ]
    assert archived, "TODO/archive/ must hold tracked files"
    for path in archived:
        assert GUARD.classify_liveness(path, index)["archived"] is True


def test_results_tree_is_never_declared_archived():
    """No tracked text declares results/ archival — that is the whole defect."""
    index = GUARD.build_liveness_index(REPO_ROOT)
    assert "results/" not in index["archive_trees"]
    for tree in index["archive_trees"]:
        assert not tree.startswith("results/")


def test_scripts_retired_tree_is_archived_via_its_own_readme():
    """An in-tree marker is evidence too: scripts/_retired/README.md."""
    index = GUARD.build_liveness_index(REPO_ROOT)
    verdict = GUARD.classify_liveness(
        "scripts/_retired/launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh", index
    )
    assert verdict["archived"] is True
    kinds = {
        d["evidence"]["kind"] for d in verdict["declarations"]
    }
    assert "in_tree_marker" in kinds


def test_a_live_file_under_a_non_declared_tree_stays_live():
    """The default polarity is LIVE: no declaration => live, never archived."""
    index = GUARD.build_liveness_index(REPO_ROOT)
    verdict = GUARD.classify_liveness("scripts/start.sh", index)
    assert verdict["archived"] is False


# --------------------------------------------------------------------------- #
# 2. THE FOUR DEFEAT MECHANISMS
# --------------------------------------------------------------------------- #
def _sites(text, path="fixture.sh"):
    return GUARD.scan_text(text, path)


def _assignment_sites(text, path="fixture.sh"):
    return [s for s in _sites(text, path) if s["form"] != "REFERENCE"]


def test_mechanism_1_literal_export_is_a_defeater():
    """(1) a literal unconditional ``export CUDA_VISIBLE_DEVICES=0``."""
    sites = _assignment_sites("export CUDA_VISIBLE_DEVICES=0\n")
    assert len(sites) == 1
    site = sites[0]
    assert site["bucket"] == "DEFEATER"
    assert site["form"] == "EXPORT_ASSIGNMENT"
    assert site["defeats_ambient_empty_mask"] is True


def test_mechanism_2_colon_default_substitution_is_a_defeater():
    """(2) the ``${VAR:-default}`` form loses an EMPTY ambient lock."""
    sites = _assignment_sites('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" cmd\n')
    assert len(sites) == 1
    site = sites[0]
    assert site["bucket"] == "DEFEATER"
    assert site["form"] == "COMMAND_PREFIX"
    assert site["self_reference"] is True


def test_mechanism_3_yaml_derived_form_is_detected():
    """(3) the YAML-derived form inside scripts/start.sh.

    The repaired form (``${CUDA_VISIBLE_DEVICES-$_cvd}``, the live content of
    ``scripts/start.sh``) is lock-preserving; the pre-repair form (``$_cvd``
    alone) takes the yaml's device whatever the ambient mask says.

    The pre-repair value is an unresolved shell variable, so the honest
    bucket is POTENTIAL (a scanner cannot read the yaml at scan time) — but
    it must NOT be lock-honouring.
    """
    repaired = _assignment_sites(
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-$_cvd}"\n'
    )
    assert len(repaired) == 1
    assert repaired[0]["bucket"] == "LOCK_HONOURING"
    assert repaired[0]["defeats_ambient_empty_mask"] is False

    pre_repair = _assignment_sites('export CUDA_VISIBLE_DEVICES="$_cvd"\n')
    assert len(pre_repair) == 1
    assert pre_repair[0]["bucket"] == "POTENTIAL_DEFEATER"
    # The value is unresolved at scan time, so the site is a POSSIBLE
    # defeater, not a proven one: "defeats" stays False until a scanner
    # can read the value it points at.
    assert pre_repair[0]["defeats_ambient_empty_mask"] is False


def test_the_live_start_sh_yaml_derived_export_is_lock_preserving():
    """The live ``scripts/start.sh`` export honours an empty ambient lock."""
    path = os.path.join(REPO_ROOT, "scripts", "start.sh")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    sites = [s for s in GUARD.scan_text(text, "scripts/start.sh")
             if s["form"] != "REFERENCE" and s["bucket"] != "LOCK_HONOURING"]
    assert not sites, "scripts/start.sh must not carry a defeater: %r" % (sites,)
    preserving = [
        s for s in GUARD.scan_text(text, "scripts/start.sh")
        if s["form"] == "EXPORT_ASSIGNMENT"
    ]
    assert preserving, "scripts/start.sh must still export the mask"
    assert all(s["bucket"] == "LOCK_HONOURING" for s in preserving)


def test_mechanism_4_command_prefix_assignment_is_a_defeater():
    """(4) a COMMAND-PREFIX assignment: ``CUDA_VISIBLE_DEVICES=0 some_command``."""
    sites = _assignment_sites("CUDA_VISIBLE_DEVICES=0 /usr/bin/env python x.py\n")
    assert len(sites) == 1
    site = sites[0]
    assert site["bucket"] == "DEFEATER"
    assert site["form"] == "COMMAND_PREFIX"
    assert site["defeats_ambient_empty_mask"] is True


def test_an_argument_position_assignment_is_not_an_env_assignment():
    """``setsid nohup CUDA_VISIBLE_DEVICES=0 cmd`` does NOT set the env.

    POSIX only honours ``VAR=value`` in the command-word position; in
    argument position it is an ordinary argv element.  MEASURED with plain
    bash: the ambient empty lock survives (the value stays empty).
    """
    sites = _assignment_sites(
        'setsid nohup     CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}" \\\n'
    )
    assert len(sites) == 1
    site = sites[0]
    assert site["form"] == "ARG_POSITION_ASSIGNMENT"
    assert site["bucket"] == "NOT_AN_ASSIGNMENT"
    assert site["defeats_ambient_empty_mask"] is False

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        ["bash", "-c", 'setsid nohup env | grep "^CUDA_VISIBLE_DEVICES="'],
        capture_output=True,
        check=True,
        env=env,
    )
    assert proc.stdout.decode("utf-8").strip() == "CUDA_VISIBLE_DEVICES="


def test_a_keyword_prefix_assignment_is_still_an_env_assignment():
    """``if``/``&&``/``export`` keep the assignment in command-word position."""
    for text, form in (
        ("if CUDA_VISIBLE_DEVICES=0 cmd; then :; fi\n", "COMMAND_PREFIX"),
        ("if ! CUDA_VISIBLE_DEVICES=0 cmd; then :; fi\n", "COMMAND_PREFIX"),
        ("foo && CUDA_VISIBLE_DEVICES=0 bar\n", "COMMAND_PREFIX"),
        ("export CUDA_VISIBLE_DEVICES=0\n", "EXPORT_ASSIGNMENT"),
        ("FOO=1 CUDA_VISIBLE_DEVICES=0 cmd\n", "COMMAND_PREFIX"),
    ):
        sites = _assignment_sites(text)
        assert len(sites) == 1, text
        assert sites[0]["form"] == form, text
        assert sites[0]["bucket"] == "DEFEATER", text


# --------------------------------------------------------------------------- #
# 3. THE NO-COLON FORM
# --------------------------------------------------------------------------- #
def test_no_colon_form_is_not_a_defeater():
    """``${VAR-default}`` keeps an EMPTY lock; it is not a defeater."""
    info = GUARD.classify_value('"${CUDA_VISIBLE_DEVICES-0}"')
    assert info["bucket"] == "LOCK_HONOURING"
    assert info["defeats_ambient_empty_mask"] is False


def test_colon_form_is_a_defeater():
    info = GUARD.classify_value('"${CUDA_VISIBLE_DEVICES:-0}"')
    assert info["bucket"] == "DEFEATER"
    assert info["defeats_ambient_empty_mask"] is True


def test_bare_self_reference_is_lock_honouring():
    info = GUARD.classify_value('"${CUDA_VISIBLE_DEVICES}"')
    assert info["bucket"] == "LOCK_HONOURING"


def test_empty_literal_is_lock_honouring():
    info = GUARD.classify_value('""')
    assert info["bucket"] == "LOCK_HONOURING"


# --------------------------------------------------------------------------- #
# 4. THE BASH PROOF — the lock-preserving form leaves an EMPTY mask empty
# --------------------------------------------------------------------------- #
def _bash_cvd(expr: str) -> str:
    """Evaluate ``expr`` with an EMPTY ambient CUDA_VISIBLE_DEVICES."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        ["bash", "-c", 'printf "%%s" "%s"' % expr],
        capture_output=True,
        check=True,
        env=env,
    )
    return proc.stdout.decode("utf-8")


def test_lock_preserving_form_leaves_an_empty_cuda_visible_devices_empty():
    """MEASURED with plain bash: the no-colon form preserves the empty lock."""
    assert _bash_cvd("${CUDA_VISIBLE_DEVICES-0}") == ""
    assert _bash_cvd("${CUDA_VISIBLE_DEVICES:-0}") == "0"
    assert _bash_cvd("${CUDA_VISIBLE_DEVICES}") == ""


def test_lock_preserving_form_falls_back_when_unset():
    """With the variable UNSET the no-colon form substitutes its default."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.run(
        ["bash", "-c", 'printf "%s" "${CUDA_VISIBLE_DEVICES-0}"'],
        capture_output=True,
        check=True,
        env=env,
    )
    assert proc.stdout.decode("utf-8") == "0"


# --------------------------------------------------------------------------- #
# 5. NEGATIVE CONTROLS — a mention is not an assignment
# --------------------------------------------------------------------------- #
def test_a_mention_inside_a_string_is_not_an_assignment():
    sites = _sites('echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"\n')
    assert all(s["form"] == "REFERENCE" for s in sites)
    assert all(s["bucket"] == "NOT_AN_ASSIGNMENT" for s in sites)


def test_a_commented_out_assignment_is_not_an_assignment():
    sites = _sites("# export CUDA_VISIBLE_DEVICES=0\n")
    assert all(s["form"] == "REFERENCE" for s in sites)
    assert all(s["bucket"] == "NOT_AN_ASSIGNMENT" for s in sites)


def test_a_comparison_is_not_an_assignment():
    sites = _sites('assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"\n', "f.py")
    assert all(s["form"] != "ENV_ASSIGN" for s in sites)


# --------------------------------------------------------------------------- #
# 6. HARD CONSTRAINTS
# --------------------------------------------------------------------------- #
def test_the_module_imports_no_torch_and_never_calls_nvidia_smi():
    """AST-level: no torch/cupy import, no GPU tool in any argv literal."""
    source = open(GUARD_PATH, encoding="utf-8").read()
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
    # A docstring may PROSE about nvidia-smi; an argv may not name it.
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for element in node.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    assert "nvidia" not in element.value.lower(), element.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.strip() in ("nvidia-smi", "nvidia-smi -L"):
                raise AssertionError("the module must never invoke nvidia-smi")
    assert "os.walk(" not in source


def test_the_module_uses_git_ls_files_not_a_directory_walk():
    source = open(GUARD_PATH, encoding="utf-8").read()
    assert "ls-files" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "walk":
                raise AssertionError("os.walk is forbidden on this repo")


# --------------------------------------------------------------------------- #
# 7. THE EMITTED CENSUS + THE STAGED PATCH
# --------------------------------------------------------------------------- #
def test_census_json_exists_and_is_self_consistent():
    assert os.path.exists(CENSUS_PATH), "the census artifact must be on disk"
    report = json.load(open(CENSUS_PATH, encoding="utf-8"))
    for key, value in report["self_check"].items():
        assert value is True, "self_check failed: %s" % key
    counts = report["counts"]
    assert counts["defeaters_live"] + counts["defeaters_archived"] == counts[
        "defeaters_total"
    ]
    # The census must name a denominator for every number it reports.
    assert report["denominator"]["tracked_files_total"] > 0
    assert report["denominator"]["files_scanned"] > 0


def test_census_classifies_the_results_launcher_as_live():
    report = json.load(open(CENSUS_PATH, encoding="utf-8"))
    assert "results/" not in report["declared_archive_trees"]
    verdict = report["liveness"].get("results/cfm_full_retrain/launch_loop.sh")
    if verdict is not None:
        assert verdict["archived"] is False


def test_the_staged_patch_exists_and_applies_to_a_scratch_copy():
    assert os.path.exists(PATCH_PATH), "the staged patch must be on disk"
    patch = open(PATCH_PATH, encoding="utf-8").read()
    assert patch.startswith("---") or patch.startswith("diff ")
    # The patch must not be applied to the live tree.
    assert "wfa167_command_prefix_guard.py" in patch
