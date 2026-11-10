# Licensed under the Apache License, Version 2.0.
"""Wf-A269 §6.269.19.7 item 7 — the runtime-transport gate's OWN regression tests.

A gate that has never been seen to REFUSE is not a gate (AGENTS.md §6.269.17).
These tests therefore do not assert "the gate passes"; they assert that the
gate FAILS on the two defects it exists to catch:

  (a) a transport flag the invoked trainer's parser does NOT declare — the
      launch would abort with "unrecognized arguments";
  (b) a transport site whose value is a LITERAL rather than a YAML-derived
      one — the "knob that lives only in the shell script" (§5.11).

Both are exercised against SYNTHETIC fixtures in a tmp dir: the repo's own
``scripts/start.sh`` and ``configs/`` are read-only inputs here, and nothing
outside ``tmp_path`` is written.

CPU ONLY: every invocation sets ``CUDA_VISIBLE_DEVICES=""`` before the
interpreter touches torch (AGENTS.md §5.1: this repo runs a CPU lock).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GATE_PATH = REPO_ROOT / "scripts" / "_runtime_transport_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_wfa269_rt_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


GATE = _load_gate()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
#: A trainer that MENTIONS ``"--not-a-real-flag"`` in a comment — so
#: start.sh's source probe (which greps the trainer's SOURCE for the flag
#: literal) WOULD append it — but never DECLARES it in its parser.  That is
#: exactly the shape of the real defect: the probe says "yes", argparse says
#: "unrecognized arguments".
SYNTHETIC_TRAINER = '''\
#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Synthetic trainer for the runtime-transport gate's regression tests.

This file mentions "--not-a-real-flag" and "--literal-value-flag" in a
comment so that start.sh's SOURCE PROBE would append them, but its parser
declares neither.  The gate must refuse both.
"""
import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="synthetic trainer")
    p.add_argument("--declared-flag", type=int, default=1)
    args = p.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


#: A synthetic launcher with the SAME SHAPE as scripts/start.sh: it assembles
#: the trainer command as a RUNTIME STRING and runs it through `bash -c`, so
#: no static invocation line exists for the static gates to see.
SYNTHETIC_WRAPPER = '''\
#!/usr/bin/env bash
# Synthetic runtime-string launcher for the runtime-transport gate's tests.
set -uo pipefail

LOG_FILE="/tmp/synthetic.log"

parse_yaml() {
  PY=$(cat <<'PY'
import os
from pathlib import Path

d = {}
lines = [
    f'CHECKPOINT_INTERVAL={shlex.quote(str(d.get("checkpoint_interval", 50)))}',
    f'LITERAL_VALUE_FLAG={shlex.quote("7")}',
]
with open(os.environ["YAML_ENV_FILE"], "w") as f:
    f.write("\\n".join(lines) + "\\n")
PY
)
}

YAML_ENV_FILE="/tmp/synthetic.env"
parse_yaml
. "$YAML_ENV_FILE"
rm -f "$YAML_ENV_FILE"

APPENDED_FLAGS=""
_add_flag() {
  local _flag="$1" _var="$2"
  if [[ -z "${!_var}" ]]; then
    return 0
  fi
  APPENDED_FLAGS="${APPENDED_FLAGS} ${_flag} \\"${!_var}\\""
}
{ADD_FLAG_SITES}
TRAINER_CMD="python trainer.py$APPENDED_FLAGS"
nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &
'''


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def synthetic_tree(tmp_path: Path) -> dict:
    """A config tree + a runtime-string launcher + a synthetic trainer."""
    cfg_dir = tmp_path / "configs"
    trainer = _write(cfg_dir / "trainer.py", SYNTHETIC_TRAINER)
    config = _write(
        cfg_dir / "synthetic.yaml",
        "entry:\n"
        "  script: trainer.py\n"
        "  command_template: python trainer.py --declared-flag 1\n"
        "output:\n"
        "  checkpoint_interval: 50\n",
    )
    return {"tmp": tmp_path, "trainer": trainer, "config": config, "cfg_dir": cfg_dir}


def _wrapper_with(add_flag_sites: str) -> str:
    return SYNTHETIC_WRAPPER.replace("{ADD_FLAG_SITES}", add_flag_sites)


def _audit(tree: dict, add_flag_sites: str) -> dict:
    wrapper = _write(tree["tmp"] / "synthetic_launcher.sh", _wrapper_with(add_flag_sites))
    return GATE.audit(
        config_root=tree["cfg_dir"],
        wrappers=[wrapper],
    )


# ---------------------------------------------------------------------------
# (a) a transport flag the trainer's parser does not declare
# ---------------------------------------------------------------------------
def test_undeclared_transport_flag_fails_the_gate(synthetic_tree):
    """A flag the trainer does not declare MUST make the verdict FAIL."""
    report = _audit(
        synthetic_tree,
        '_add_flag --not-a-real-flag CHECKPOINT_INTERVAL\n',
    )

    assert report["verdict"] == "FAIL", report
    kinds = {v["kind"] for v in report["violations"]}
    assert "undeclared_flag" in kinds, report["violations"]
    hit = [
        v for v in report["violations"]
        if v["kind"] == "undeclared_flag" and v["flag"] == "--not-a-real-flag"
    ]
    assert hit, report["violations"]
    assert "--not-a-real-flag" in hit[0]["detail"]


def test_undeclared_transport_flag_is_reported_per_site(synthetic_tree):
    """The violation names the transport SITE, not just the flag."""
    report = _audit(
        synthetic_tree,
        '_add_flag --not-a-real-flag CHECKPOINT_INTERVAL\n',
    )
    sites = report["sites"]
    assert any(
        s["flag"] == "--not-a-real-flag" and s["kind"] == "add_flag"
        for s in sites
    ), sites
    # the site count is an INVOCATION census, not a mention census
    assert report["n_sites"] == len(sites)
    assert report["n_mentions"], "the mention count must be reported alongside"


def test_declared_transport_flag_does_not_fail_the_gate(synthetic_tree):
    """The SAME launcher with a DECLARED flag must NOT trip the R1 check.

    This is the control: without it, the test above would pass even if the
    gate refused everything.
    """
    report = _audit(
        synthetic_tree,
        '_add_flag --declared-flag CHECKPOINT_INTERVAL\n',
    )
    undeclared = [
        v for v in report["violations"] if v["kind"] == "undeclared_flag"
    ]
    assert not undeclared, report["violations"]
    assert report["verdict"] == "PASS", report


# ---------------------------------------------------------------------------
# (b) a transport site with a LITERAL value instead of a YAML-derived one
# ---------------------------------------------------------------------------
def test_literal_value_transport_fails_the_gate(synthetic_tree):
    """A transport whose value is NOT written by the YAML->env serialisation.

    The synthetic launcher's `lines = [...]` block writes CHECKPOINT_INTERVAL
    and LITERAL_VALUE_FLAG, so a site that transports a variable the YAML does
    NOT write is exactly the "knob that lives only in the shell script"
    defect of §5.11.
    """
    report = _audit(
        synthetic_tree,
        '_add_flag --declared-flag NOT_WRITTEN_BY_THE_YAML\n',
    )

    assert report["verdict"] == "FAIL", report
    kinds = {v["kind"] for v in report["violations"]}
    assert "value_not_yaml_derived" in kinds, report["violations"]
    hit = [
        v for v in report["violations"]
        if v["kind"] == "value_not_yaml_derived"
    ]
    assert hit, report["violations"]
    assert "NOT_WRITTEN_BY_THE_YAML" in hit[0]["detail"]


def test_yaml_derived_value_does_not_fail_the_gate(synthetic_tree):
    """The control for (b): a YAML-written variable must NOT trip R2."""
    report = _audit(
        synthetic_tree,
        '_add_flag --declared-flag CHECKPOINT_INTERVAL\n',
    )
    bad = [v for v in report["violations"] if v["kind"] == "value_not_yaml_derived"]
    assert not bad, report["violations"]


def test_unconditional_transport_fails_the_gate(synthetic_tree):
    """A site with no emptiness guard emits the flag even when the YAML is silent."""
    without_guard = SYNTHETIC_WRAPPER.replace(
        '  if [[ -z "${!_var}" ]]; then\n    return 0\n  fi\n', ""
    )
    wrapper = _write(
        synthetic_tree["tmp"] / "synthetic_launcher.sh",
        without_guard.replace(
            "{ADD_FLAG_SITES}", '_add_flag --declared-flag CHECKPOINT_INTERVAL\n'
        ),
    )
    report = GATE.audit(config_root=synthetic_tree["cfg_dir"], wrappers=[wrapper])
    assert report["verdict"] == "FAIL", report
    kinds = {v["kind"] for v in report["violations"]}
    assert "value_not_conditional" in kinds, report["violations"]


# ---------------------------------------------------------------------------
# the fail-closed rule
# ---------------------------------------------------------------------------
def test_unreadable_trainer_parser_is_a_coverage_gap_not_a_skip(
    synthetic_tree, monkeypatch
):
    """A trainer whose parser cannot be captured must FAIL, never SKIP."""
    monkeypatch.setattr(GATE.LC, "_declared_flags", lambda path: (None, "boom"))
    report = _audit(
        synthetic_tree,
        '_add_flag --declared-flag CHECKPOINT_INTERVAL\n',
    )
    assert report["verdict"] == "FAIL", report
    assert report["n_coverage_gaps"] >= 1, report
    assert any(
        g["kind"] == "parser_not_captured" for g in report["coverage_gaps"]
    ), report["coverage_gaps"]


def test_no_runtime_wrapper_is_a_skip(synthetic_tree):
    """With no runtime-string launch surface the gate SKIPs (exit 2)."""
    plain = _write(
        synthetic_tree["tmp"] / "plain_launcher.sh",
        "#!/usr/bin/env bash\npython trainer.py --declared-flag 1\n",
    )
    report = GATE.audit(config_root=synthetic_tree["cfg_dir"], wrappers=[plain])
    assert report["verdict"] == "SKIP", report
    assert report["n_sites"] == 0, report


# ---------------------------------------------------------------------------
# the live repo: the gate must actually run against scripts/start.sh
# ---------------------------------------------------------------------------
def test_gate_enumerates_the_real_start_sh_transport_sites():
    """On THIS tree the gate must find the real `_add_flag` sites.

    MEASURED 2026-11-04 on this tree: 8 `_add_flag` call sites
    (scripts/start.sh:943,944,945,946,947,954,956,957), plus 2 `EVENTS_FLAG`
    assignments and 4 `_dmpnn_new_flags` flags = 14 transport sites.
    """
    report = GATE.audit()
    add_flag = [s for s in report["sites"] if s["kind"] == "add_flag"]
    assert len(add_flag) == 8, report["sites"]
    flags = {s["flag"] for s in add_flag}
    assert flags == {
        "--checkpoint-interval",
        "--resume-from",
        "--lr-scheduler",
        "--lr-warmup-epochs",
        "--lr-min-lr",
        "--lr-pct-start",
        "--weight-decay",
        "--max-norm",
    }, flags
    # every one of them carries a YAML-written value
    env_names = GATE.GATE._yaml_env_names(
        (REPO_ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    )
    for site in add_flag:
        assert site["value_var"] in env_names, site


def test_gate_runs_on_this_tree_and_prints_a_verdict(capsys):
    """The gate's stdout carries a PASS/FAIL/SKIP verdict line (AGENTS.md §4)."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = GATE.main([])
    out = buf.getvalue()
    assert "[runtime-transport]" in out
    assert out.rstrip().endswith(("PASS", "FAIL", "SKIP")), out
    assert rc in (0, 1, 2), rc


# ---------------------------------------------------------------------------
# DEFECT C (2026-11-04, Wf-A269 dead-gates sweep) — the assembly's OWN
# literal text, the command-substitution assembly, and the delegation
# escape hatch's reachability requirement.
#
# MEASURED before the repair: a runtime assembly carrying ``--epochs 99``
# produced ``n_sites 0 / n_violations 0 / n_coverage_gaps 0`` and a PASS
# verdict — the flag was invisible to this gate AND to the other three
# §5.11 gates, all of which read STATIC invocation lines only.
# ---------------------------------------------------------------------------

#: A launcher whose RUNTIME ASSEMBLY carries a LITERAL hyperparameter.
ASSEMBLY_LITERAL_WRAPPER = '''\
#!/usr/bin/env bash
# Synthetic launcher: the runtime command string carries a literal flag.
set -uo pipefail
LOG_FILE="/tmp/synthetic.log"
TRAINER_CMD="python {TRAINER} --literal-assembly-flag 7"
nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &
'''

#: A launcher whose runtime command is assembled by a COMMAND SUBSTITUTION —
#: the gate cannot enumerate what the substitution produces.
COMMAND_SUBSTITUTION_WRAPPER = '''\
#!/usr/bin/env bash
set -uo pipefail
LOG_FILE="/tmp/synthetic.log"
TRAINER_CMD="$(python -c 'print(1)')"
nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &
'''

#: A launcher whose runtime command cannot be traced, but whose FILE
#: mentions an unrelated launcher in another assignment.
UNTRACEABLE_WRAPPER = '''\
#!/usr/bin/env bash
set -uo pipefail
LOG_FILE="/tmp/synthetic.log"
OTHER_CMD="bash scripts/launch_wf124_p3_g5.sh"
TRAINER_CMD="$NOT_A_VARIABLE"
nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &
'''


def _audit_wrapper(tree: dict, template: str) -> dict:
    wrapper = _write(
        tree["tmp"] / "synthetic_launcher.sh",
        template.replace("{TRAINER}", str(tree["trainer"])),
    )
    return GATE.audit(config_root=tree["cfg_dir"], wrappers=[wrapper])


def test_assembly_literal_flag_is_a_violation(synthetic_tree):
    """A flag written as a LITERAL into the runtime command MUST be refused."""
    report = _audit_wrapper(synthetic_tree, ASSEMBLY_LITERAL_WRAPPER)
    assert report["verdict"] == "FAIL", "the assembly literal was not refused"
    flags = {v["flag"] for v in report["violations"]}
    assert "--literal-assembly-flag" in flags, (
        f"the literal flag was not reported: {report['violations']}"
    )
    kinds = {v["kind"] for v in report["violations"]}
    assert "value_not_yaml_derived" in kinds, (
        "the literal VALUE was not reported (§5.11: a knob that lives only in "
        "the shell script is a knob nobody audits)"
    )
    # The flag is NOT declared by the synthetic trainer's parser, so R1 fires
    # as well — the launch would abort with "unrecognized arguments".
    assert "undeclared_flag" in kinds


def test_assembly_literal_in_the_legal_flag_set_is_not_a_violation(synthetic_tree):
    """``--config`` selects WHICH run — it may be a literal (§5.11).

    The EXEMPTION this asserts is the R2 one (the value must be
    YAML-derived): a wrapper-legal flag is allowed to carry a literal value
    because it selects the run, not the model.  R1 is a DIFFERENT question
    ("does the trainer's parser declare it?") and the synthetic trainer
    declares no ``--config``, so R1 may still report it — that is the
    parser contract, not the YAML-derivation contract, and conflating the
    two would be the defect.
    """
    template = ASSEMBLY_LITERAL_WRAPPER.replace(
        "--literal-assembly-flag 7", "--config synthetic.yaml"
    )
    report = _audit_wrapper(synthetic_tree, template)
    r2 = [
        v
        for v in report["violations"]
        if v["kind"] == "value_not_yaml_derived" and v["flag"] == "--config"
    ]
    assert not r2, (
        f"the wrapper-legal flag set was reported as a non-YAML value: {r2}"
    )


def test_assembly_literal_scan_is_mutation_proved(synthetic_tree, monkeypatch):
    """MUTATION PROOF: with the assembly-literal scan removed, the gate PASSes."""
    report = _audit_wrapper(synthetic_tree, ASSEMBLY_LITERAL_WRAPPER)
    assert report["verdict"] == "FAIL"
    assert report["n_sites"] >= 1

    monkeypatch.setattr(GATE, "_literal_text", lambda value: "")
    mutated = _audit_wrapper(synthetic_tree, ASSEMBLY_LITERAL_WRAPPER)
    assert mutated["verdict"] != "FAIL", (
        "MUTATION NOT CAUGHT: disabling the assembly-literal scan did not "
        "restore the pre-repair PASS"
    )
    assert mutated["n_sites"] == 0


def test_command_substitution_assembly_is_a_coverage_gap(synthetic_tree):
    """An assembly the gate cannot enumerate is a GAP, not a silent PASS."""
    report = _audit_wrapper(synthetic_tree, COMMAND_SUBSTITUTION_WRAPPER)
    assert report["verdict"] == "FAIL"
    kinds = {g["kind"] for g in report["coverage_gaps"]}
    assert "assembly_not_enumerable" in kinds, (
        f"the command-substitution assembly was not reported: {report['coverage_gaps']}"
    )


def test_delegation_escape_hatch_requires_reachability(synthetic_tree):
    """An UNREACHABLE `.sh` token must not make an untraceable wrapper 'covered'.

    MUTATION PROOF: the pre-repair escape hatch scanned EVERY shell
    assignment for a ``.sh`` token.  The fixture below puts one in an
    UNRELATED assignment; the old logic would find it and declare the
    wrapper covered (a SILENT PASS), the repaired logic follows the chain
    the runtime command actually depends on and finds nothing.
    """
    template = UNTRACEABLE_WRAPPER
    report = _audit_wrapper(synthetic_tree, template)
    assert report["verdict"] == "FAIL", "the untraceable runtime command passed"
    assert report["n_coverage_gaps"] >= 1, (
        "the untraceable runtime command produced no coverage gap"
    )

    text = (synthetic_tree["tmp"] / "synthetic_launcher.sh").read_text()
    old_targets = GATE._delegated_targets(text)
    assert old_targets == ["scripts/launch_wf124_p3_g5.sh"], (
        f"the pre-repair scan no longer finds the unrelated token: {old_targets}"
    )
    assert GATE._delegation_reachable(text, "TRAINER_CMD") == [], (
        "the repaired reachability walk found a delegation the runtime "
        "command does not depend on"
    )


def test_real_r28_delegation_is_proven_reachable():
    """``scripts/_wf_r28_gpu_master.sh``'s delegation is REACHABLE, not assumed.

    The wrapper builds its runtime command from a ``for spec in ...`` loop
    over ``${T1_CMD}``..``${T5_CMD}``, so the chain runs THROUGH a for-list —
    which is not a shell assignment and was invisible to the pre-repair
    escape hatch.
    """
    wrapper = REPO_ROOT / "scripts" / "_wf_r28_gpu_master.sh"
    text = wrapper.read_text(encoding="utf-8")
    reachable = GATE._delegation_reachable(text, "cmd")
    assert "scripts/launch_wf124_p3_g5.sh" in reachable, (
        f"the r28 delegation chain is not traceable: {reachable}"
    )
    assert "scripts/launch_cfm_round7_f5f6_retrain.sh" in reachable
