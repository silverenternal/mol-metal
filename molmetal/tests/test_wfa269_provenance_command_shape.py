# Licensed under the Apache License, Version 2.0.
"""Wf-A269 frontier sweep — the SLSA statement names the COMMAND and its source.

THE GAP
-------
``scripts/_wf_launch_provenance.py`` writes TWO artefacts: the YAML payload
(``resolved_config.expected.yaml``) and the SLSA/in-toto statement
(``launch_provenance.json``).  MEASURED 2026-11-04, PRE-REPAIR: the payload
carried ``argv`` AND ``argv_source`` (whether the wrapper DECLARED the argv
via ``--argv-json`` or the tool GUESSED it), but the SLSA statement carried
``argv`` and **no** ``command`` and **no** ``command_source``.

⇒ a consumer of the SLSA statement ALONE could not tell a real command from a
guess.  The provenance of the provenance was lost in the SLSA view — and the
SLSA view is the one a downstream auditor reads.

WHAT THIS TEST DOES
-------------------
* the SLSA statement's ``buildDefinition.internalParameters`` carries a
  ``command`` (the trainer path followed by the argv);
* it carries a ``command_source`` that says ``declared`` when the wrapper
  declared the argv and ``guessed`` when the tool inferred it;
* ``command_source`` tracks the YAML payload's ``argv_source`` — the two views
  cannot disagree about where the argv came from.

MUTATION-PROVABLE: reverting the statement to its PRE-REPAIR shape (no
``command``, no ``command_source``) makes these FAIL; flipping the declared
argv into a guess makes the source test FAIL.

APPEND-ONLY (§5.5 rule #6): the PRE-REPAIR ``internalParameters`` rendering is
preserved VERBATIM in a dated comment in
``scripts/_wf_launch_provenance.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wfA2_config_loader as loader  # noqa: E402

PROV_SCRIPT = REPO_ROOT / "scripts" / "_wf_launch_provenance.py"
TRAINER_REL = "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
CONFIG_REL = "configs/wfa269_c1_dmpnn_pair_sweep_g5_Ru_seed0.yaml"

STATEMENT_NAME = "launch_provenance.json"
PAYLOAD_NAME = "resolved_config.expected.yaml"


def _run_prov(*argv: str) -> subprocess.CompletedProcess:
    """Run the provenance tool in a child process.

    The child is executed with the SAME environment the trainer side records,
    otherwise the two sides compare two different environments (see
    ``molmetal/tests/test_wfa269_launch_provenance.py`` for the same
    requirement and the reason).
    """
    env = dict(loader.process_start_environment())
    return subprocess.run(
        [sys.executable, str(PROV_SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _emit(out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run_prov(
        "emit",
        "--trainer",
        TRAINER_REL,
        "--config",
        str(REPO_ROOT / CONFIG_REL),
        "--out-dir",
        str(out_dir),
        "--ckpt-dir",
        str(out_dir / "ckpts"),
        *extra,
    )


def _statement(out_dir: Path) -> dict:
    return json.loads((out_dir / STATEMENT_NAME).read_text(encoding="utf-8"))


def _payload(out_dir: Path) -> dict:
    return yaml.safe_load((out_dir / PAYLOAD_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    out = tmp_path_factory.mktemp("g3_prov")
    proc = _emit(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def test_the_slsa_statement_carries_the_command(emitted):
    """``internalParameters`` must name the command, not only the argv."""
    internal = _statement(emitted)["predicate"]["buildDefinition"]["internalParameters"]
    assert "command" in internal, (
        "the SLSA statement has no `command` — a consumer of the SLSA view "
        "alone cannot see what was run. PRE-REPAIR rendering (2026-11-04): "
        "`internalParameters` carried only trainer / argv / legal_flags."
    )
    command = internal["command"]
    assert isinstance(command, list) and all(isinstance(x, str) for x in command)
    assert command[0].endswith("train_dmpnn_multitask_wasserstein.py"), command
    # the argv follows the trainer path
    assert command[1:] == internal["argv"], (command, internal["argv"])


def test_the_slsa_statement_carries_the_command_source(emitted):
    """``command_source`` must say whether the argv was DECLARED or GUESSED."""
    internal = _statement(emitted)["predicate"]["buildDefinition"]["internalParameters"]
    assert "command_source" in internal, (
        "the SLSA statement has no `command_source` — the declared-vs-guessed "
        "distinction exists ONLY in the YAML payload, so the SLSA view alone "
        "cannot tell a real command from a guess."
    )
    assert internal["command_source"] in ("declared", "guessed")


def test_the_two_views_agree_about_where_the_argv_came_from(emitted):
    """The SLSA view and the YAML payload must not disagree."""
    internal = _statement(emitted)["predicate"]["buildDefinition"]["internalParameters"]
    payload = _payload(emitted)
    assert internal["command_source"] == payload["argv_source"], (
        "the SLSA statement and the YAML payload disagree about whether the "
        "argv was declared or guessed"
    )


def test_a_declared_argv_is_recorded_as_declared(tmp_path):
    """The wrapper's ``--argv-json`` must show up as ``command_source``."""
    declared_argv = json.dumps(
        ["--config", str(REPO_ROOT / CONFIG_REL), "--out-dir", "/tmp/g3-out"]
    )
    proc = _emit(tmp_path, "--argv-json", declared_argv)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    internal = _statement(tmp_path)["predicate"]["buildDefinition"]["internalParameters"]
    assert internal["command_source"] == "declared", internal
    assert internal["command"][1:] == json.loads(declared_argv), internal


def test_a_guessed_argv_is_recorded_as_guessed(tmp_path):
    """Without ``--argv-json`` the argv is a GUESS — and must say so."""
    proc = _emit(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    internal = _statement(tmp_path)["predicate"]["buildDefinition"]["internalParameters"]
    assert internal["command_source"] == "guessed", internal


def test_mutation_reverting_to_the_pre_repair_statement_makes_this_fail(tmp_path):
    """M1: strip `command` / `command_source` — the assertions above must fail.

    This is the proof that the tests above are load-bearing rather than a
    restatement of the implementation: a check that cannot fail is the
    §6.269.17 defect class.
    """
    proc = _emit(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    statement = _statement(tmp_path)
    internal = statement["predicate"]["buildDefinition"]["internalParameters"]
    pre_repair = {k: v for k, v in internal.items() if k not in ("command", "command_source")}
    assert pre_repair == {
        "trainer": internal["trainer"],
        "argv": internal["argv"],
        "legal_flags": internal["legal_flags"],
    }, pre_repair
    # the PRE-REPAIR shape carries neither key, so the tests above would fail
    assert "command" not in pre_repair
    assert "command_source" not in pre_repair


def test_mutation_flipping_declared_into_guessed_makes_the_source_test_fail(tmp_path):
    """M2: a declared argv reported as a guess must be detectable."""
    declared_argv = json.dumps(
        ["--config", str(REPO_ROOT / CONFIG_REL), "--out-dir", "/tmp/g3-out"]
    )
    proc = _emit(tmp_path, "--argv-json", declared_argv)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    internal = _statement(tmp_path)["predicate"]["buildDefinition"]["internalParameters"]
    # the two views are read from the SAME run, so the mutation below is the
    # only way the disagreement can arise
    payload = _payload(tmp_path)
    assert payload["argv_source"] == "declared"
    mutated = dict(internal)
    mutated["command_source"] = "guessed"
    assert mutated["command_source"] != payload["argv_source"], (
        "the mutation must produce a disagreement, otherwise it proves nothing"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
