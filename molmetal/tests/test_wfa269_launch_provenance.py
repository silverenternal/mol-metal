# Licensed under the Apache License, Version 2.0.
"""Launch-provenance tests (Wf-A269, 2026-11-04).

The mechanism under test: a §5.11 YAML-only launch records the configuration
it is ABOUT to run (wrapper side) and the configuration it ACTUALLY consumed
(trainer side), and a check compares the two.

The tests that matter most are the MUTATION ones.  A provenance check that
cannot FAIL is the DEAD-GATE class (honest-negative catalog §6.269.17), and a
provenance check that only ever sees matching inputs proves nothing.  So:

  * a DIFFERENT argv on the trainer side must FAIL the check;
  * a config EDITED between registration and the run must FAIL the check;
  * a trainer that OVERRIDES a config-declared value after the config was
    applied must FAIL the check (the POST-OVERRIDE MUTATION BLINDNESS
    defect — the "smallest missing RED" the Wf-A269 wave-4 design critique
    named);
  * a MISSING actual artefact must be SKIP (or FAIL under --require-actual),
    never PASS.

CPU-only: no trainer body is ever executed.  The trainer-side artefact is
produced by calling the trainer's own ``parse_args`` — the exact call
``main()`` makes before any training work happens — and then replaying the
mutations ``main()`` performs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported at MODULE level so that the child process is launched with the
# SAME environment the trainer side records — see ``_run_prov``.
from molmetal.scripts import wfA2_config_loader as loader  # noqa: E402

TRAINER_REL = "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
TRAINER = REPO_ROOT / TRAINER_REL
CONFIG_REL = "configs/wfa269_c1_dmpnn_pair_sweep_g5_Ru_seed0.yaml"
CONFIG = REPO_ROOT / CONFIG_REL

PROV_SCRIPT = REPO_ROOT / "scripts" / "_wf_launch_provenance.py"


def _run_prov(
    *argv: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run the provenance tool in a child process.

    The child MUST be executed with the SAME environment the trainer side
    RECORDS, otherwise the two sides compare two different environments and
    the check fires on a launch that did not drift.

    2026-11-04 (Wf-A269 wave-20): this used to be ``dict(os.environ)`` — the
    parent's CURRENT environment.  That carries whatever ANY earlier test in
    the same process mutated, including the CPU thread caps that
    ``molmetal/scripts/dmpnn_within_metal_auc.py`` sets at import time
    (``os.environ.setdefault(_v, "1")``) — so the wrapper side saw
    ``OMP_NUM_THREADS='1'`` while the trainer side (which records the
    environment the trainer process was EXECUTED with) saw nothing ⇒ a
    spurious MISMATCH on a launch that had not drifted.

    The CPU lock is inherited from the process-start environment, which every
    compliant invocation in this repo prefixes with ``CUDA_VISIBLE_DEVICES=""``.
    """
    env = dict(loader.process_start_environment())
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(PROV_SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _emit(
    out_dir: Path,
    config: Path = CONFIG,
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return _run_prov(
        "emit",
        "--trainer",
        TRAINER_REL,
        "--config",
        str(config),
        "--out-dir",
        str(out_dir),
        "--ckpt-dir",
        str(out_dir / "ckpts"),
        env_overrides=env_overrides,
    )


def _trainer_side_write(
    out_dir: Path,
    argv: list[str],
    *,
    rogue_mutation: str | None = None,
) -> None:
    """Produce the ACTUAL artefact the way the trainer would.

    ``train_dmpnn_multitask_wasserstein.main`` calls ``parse_args(argv)``
    before any training work, snapshots the DECLARED namespace, mutates
    ``args.ckpt_dir`` / ``args.csv`` / ``args.metals``, and then calls
    ``finalize_resolved_config``.  This helper replays exactly that, so the
    artefact is produced by the trainer's OWN code paths rather than by a
    hand-written fixture.

    ``rogue_mutation`` names a dest to overwrite AFTER the declared snapshot
    — the mutation the check must catch.  It defaults to None (no drift).

    NO trainer body runs.  No GPU.  No data.
    """
    spec = importlib.util.spec_from_file_location(
        "wfa269prov_trainer", TRAINER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wfa269prov_trainer"] = mod
    spec.loader.exec_module(mod)

    os.environ["MOLMETAL_RESOLVED_CONFIG_PATH"] = str(out_dir / "resolved_config.actual.yaml")
    old_argv = sys.argv
    sys.argv = [str(TRAINER)] + list(argv)
    try:
        args = mod.parse_args(None)
        # the DECLARED snapshot — what main() takes right after parse_args
        declared = argparse.Namespace(**vars(args))
        # main()'s post-override mutations
        args.ckpt_dir = Path(args.out_dir) / "checkpoints"
        try:
            args.csv = mod._resolve_default_csv_path(args.dataset)
        except Exception:  # noqa: BLE001 — the dataset may not be present here
            pass
        if getattr(args, "metals", None):
            args.metals = list(args.metals)
        if rogue_mutation is not None:
            setattr(args, rogue_mutation, 2)
        from molmetal.scripts.wfA2_config_loader import finalize_resolved_config

        finalize_resolved_config(
            args,
            config_path=args.config,
            argv=list(argv),
            declared=declared,
            derived_dests=("ckpt_dir", "csv", "metals"),
        )
    finally:
        sys.argv = old_argv
        os.environ.pop("MOLMETAL_RESOLVED_CONFIG_PATH", None)


def test_emit_registers_the_namespace(tmp_path: Path) -> None:
    out = tmp_path / "run"
    res = _emit(out)
    assert res.returncode == 0, res.stderr
    expected = out / "resolved_config.expected.yaml"
    assert expected.exists()
    statement = json.loads((out / "launch_provenance.json").read_text(encoding="utf-8"))
    assert statement["_type"] == "https://in-toto.io/Statement/v1"
    assert statement["predicateType"] == "https://slsa.dev/provenance/v1"
    # externalParameters must carry the whole namespace — that is the field
    # SLSA L3 says has to cover every externally-controllable influence.
    ext = statement["predicate"]["buildDefinition"]["externalParameters"]
    assert len(ext) > 30, f"externalParameters looks empty: {len(ext)}"
    assert statement["subject"][0]["digest"]["sha256"]
    # The statement must not overclaim.
    assert "does_not_cover" in statement["honest_limits"]
    # The resolver's own correctness is an honest limit, not an assumption.
    assert any(
        "RESOLVER" in s or "resolver" in s
        for s in statement["honest_limits"]["does_not_cover"]
    )


def test_trainer_side_writes_the_actual(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv)
    actual = out / "resolved_config.actual.yaml"
    assert actual.exists(), "the loader did not write the resolved-config artefact"
    assert (out / "resolved_config.actual.yaml.sha256").exists()


def test_verify_passes_when_the_run_matches(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv)
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout


def test_verify_fails_when_the_argv_drifted(tmp_path: Path) -> None:
    """MUTATION: the trainer received a DIFFERENT argv than was registered.

    This is the anti-dead-gate test.  If the check cannot detect this, the
    provenance record is a transcript, not a check.

    NOTE: the argv row is ADVISORY unless the wrapper declares its argv
    (``--argv-json``), because a guessed argv invents a flag the trainer
    never receives.  The advisory row still REPORTS the difference — the
    drift must be visible — but it cannot be the thing that fails.
    """
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    drifted = [
        "--config",
        str(CONFIG),
        "--out-dir",
        str(out),
        "--ckpt-dir",
        str(out / "ckpts"),
        "--epochs",
        "7",
    ]
    _trainer_side_write(out, drifted)
    res = _run_prov("verify", "--out-dir", str(out))
    assert "argv" in res.stdout
    assert "ADVISORY" in res.stdout, (
        "a guessed argv must be advisory — it is not what the trainer received"
    )
    assert "expected=" in res.stdout, "the argv difference must still be REPORTED"


def test_the_declared_argv_makes_the_argv_row_a_real_check(tmp_path: Path) -> None:
    """With ``--argv-json`` the wrapper's declared argv IS what the trainer
    receives, so a drift must FAIL the check."""
    out = tmp_path / "run"
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    res = _run_prov(
        "emit",
        "--trainer",
        TRAINER_REL,
        "--config",
        str(CONFIG),
        "--out-dir",
        str(out),
        "--ckpt-dir",
        str(out / "ckpts"),
        "--argv-json",
        json.dumps(argv),
    )
    assert res.returncode == 0, res.stderr
    _trainer_side_write(out, argv)
    ok = _run_prov("verify", "--out-dir", str(out))
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # ... now the trainer receives a DIFFERENT argv than was declared ...
    _trainer_side_write(out, argv + ["--epochs", "7"])
    bad = _run_prov("verify", "--out-dir", str(out))
    assert bad.returncode == 1, "a declared argv that drifted did not FAIL"
    assert "argv: MISMATCH" in bad.stdout


def test_verify_fails_when_the_config_was_edited(tmp_path: Path) -> None:
    """MUTATION: the config changed on disk between registration and the run.

    The path stays the SAME, so the argv comparison cannot fire — only the
    content changed.  That isolates the sha256 row, which is the row that
    catches "the config was edited after the run was registered".
    """
    out = tmp_path / "run"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert _emit(out, config=cfg).returncode == 0
    # ... now the config changes on disk ...
    cfg.write_text(
        CONFIG.read_text(encoding="utf-8").replace("epochs: 50", "epochs: 51"),
        encoding="utf-8",
    )
    argv = ["--config", str(cfg), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv)
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 1, "the check PASSED on an edited config"
    assert "config_sha256" in res.stdout


def test_verify_fails_when_the_trainer_overrode_a_declared_value(
    tmp_path: Path,
) -> None:
    """MUTATION — the "smallest missing RED".

    The trainer reads the config, and THEN overwrites a value the config
    declared (``args.epochs = 2``).  The run therefore used an epoch count
    the configuration never asked for.  Before the Wf-A269 repair the
    provenance record was taken INSIDE ``parse_args`` — before this
    mutation — so the check PASSED on exactly this case (the
    POST-OVERRIDE MUTATION BLINDNESS defect).

    ``epochs`` is NOT on the D-MPNN trainer's derived list
    (``ckpt_dir``, ``csv``, ``metals``), so the override is DRIFT and the
    check must FAIL.
    """
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv, rogue_mutation="epochs")
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 1, "the check PASSED on a post-override mutation"
    assert "overrides: MISMATCH" in res.stdout
    assert "epochs" in res.stdout


def test_the_trainers_own_derivations_are_allowed(tmp_path: Path) -> None:
    """The trainer's DERIVED dests (ckpt_dir / csv / metals) may differ.

    If the derived set were not recognised, the check would be red on every
    correct launch — always-red is the DEAD-GATE class by another route.
    """
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv)
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "overrides: MATCH" in res.stdout


def test_verify_reports_a_missing_actual_as_skip_not_pass(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 2, "a missing actual artefact was not reported as SKIP"
    assert "SKIP" in res.stdout
    # ... and under --require-actual it must be a FAIL, never a PASS.
    strict = _run_prov("verify", "--out-dir", str(out), "--require-actual")
    assert strict.returncode == 1, "a missing actual artefact was not a FAIL"


def test_loader_writes_nothing_without_the_env_var(tmp_path: Path) -> None:
    """The loader's write is OPT-IN.  Without the env var it must not write."""
    from molmetal.scripts import wfA2_config_loader as loader

    ns = argparse.Namespace(alpha=0.5, epochs=3)
    os.environ.pop(loader.RESOLVED_CONFIG_ENV, None)
    assert loader.write_resolved_config(ns, config_path=None, argv=[]) is None


def test_resolved_environment_excludes_the_provenance_switch() -> None:
    """The provenance env var must not make the two environments differ."""
    from molmetal.scripts import wfA2_config_loader as loader

    env = {"CUDA_VISIBLE_DEVICES": "0", loader.RESOLVED_CONFIG_ENV: "/tmp/x.yaml"}
    recorded = loader.resolved_environment(env)
    assert recorded.get("CUDA_VISIBLE_DEVICES") == "0"
    assert loader.RESOLVED_CONFIG_ENV not in recorded


def test_a_trainer_cannot_widen_its_own_derived_list() -> None:
    """The derived list is the loader's table, not the trainer's call site.

    A trainer that declared a dest the table does not allow is REFUSED, and
    the refusal is recorded in the artefact — otherwise a trainer could
    silence the check by editing its own call site.
    """
    from molmetal.scripts import wfA2_config_loader as loader

    assert "train_dmpnn_multitask_wasserstein.py" in loader.PROVENANCE_DERIVED_DESTS
    allowed = loader.PROVENANCE_DERIVED_DESTS["train_dmpnn_multitask_wasserstein.py"]
    assert "epochs" not in allowed, (
        "epochs on the derived list would make the post-override mutation "
        "test above pass silently"
    )


@pytest.mark.skipif(not TRAINER.exists(), reason="trainer not present")
def test_the_check_is_not_circular_across_processes(tmp_path: Path) -> None:
    """The expected side and the actual side must come from DIFFERENT inputs.

    The expected side is the wrapper's canonical argv; the actual side is the
    argv the trainer received.  If they were the same object the check would
    be comparing a thing with itself.
    """
    out = tmp_path / "run"
    assert _emit(out).returncode == 0
    argv = ["--config", str(CONFIG), "--out-dir", str(out), "--ckpt-dir", str(out / "ckpts")]
    _trainer_side_write(out, argv)
    import yaml

    expected = yaml.safe_load((out / "resolved_config.expected.yaml").read_text())
    actual = yaml.safe_load((out / "resolved_config.actual.yaml").read_text())
    assert expected["argv"] == actual["argv"]
    # ... and the two files are separately produced, separately hashed.
    assert (out / "resolved_config.expected.yaml.sha256").exists()
    assert (out / "resolved_config.actual.yaml.sha256").exists()
    assert (
        (out / "resolved_config.expected.yaml.sha256").read_text()
        != (out / "resolved_config.actual.yaml.sha256").read_text()
    ), "the two artefacts are byte-identical — the check compares one thing with itself"


# ---------------------------------------------------------------------------
# 2026-11-04 (Wf-A269 wave-20) — the ENVIRONMENT ROW's own guarantees.
#
# The row was the one part of the provenance check that could fire on a
# launch that had not drifted: the loader's snapshot was taken at IMPORT
# TIME, so the repo's import-time ``os.environ`` mutations decided its
# content.  These three tests pin the repaired semantics:
#   * a RUNTIME mutation must not enter the recorded row;
#   * an IMPORT-TIME mutation must not enter it either (import-order
#     independence — the mutation proof of the repair);
#   * the row must still be able to FAIL on a REAL drift (otherwise it is a
#     DEAD row — honest-negative catalog §6.269.17).
# ---------------------------------------------------------------------------


def test_the_recorded_environment_ignores_a_runtime_mutation(monkeypatch) -> None:
    """A var set AFTER the process started must not enter the recorded row.

    This is the property the snapshot exists for: a trainer mutates its own
    environment after ``parse_args`` (``apply_default_cuda_visible_devices``
    sets ``CUDA_VISIBLE_DEVICES``), and recording that mutation would fire
    the check on exactly the case a wrapper handles correctly.
    """
    monkeypatch.setenv("MOLMETAL_WFA269_RUNTIME_PROBE", "1")
    recorded = loader.resolved_environment()
    assert "MOLMETAL_WFA269_RUNTIME_PROBE" not in recorded, (
        "a var set at runtime leaked into the recorded environment — the "
        "snapshot is not the process-start environment"
    )


def test_the_recorded_environment_is_import_order_independent(tmp_path: Path) -> None:
    """An import-time mutation must not change the recorded row.

    The repo has modules that do ``os.environ.setdefault(_v, "1")`` for the
    CPU thread caps at import time (e.g.
    ``molmetal/scripts/dmpnn_within_metal_auc.py:74``).  Before the
    2026-11-04 repair the snapshot was taken at import time, so those
    mutations decided its CONTENT, and two processes running the SAME launch
    under the SAME environment recorded DIFFERENT rows.

    The loader is RE-IMPORTED with the pollution already in place: that is
    the import ORDER the defect lived in, and without the reload this test
    cannot tell a repaired loader from a pre-repair one.
    """
    before = loader.resolved_environment()
    mutator = tmp_path / "wfa269_env_mutator.py"
    mutator.write_text(
        "import os\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("wfa269_env_mutator", mutator)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert os.environ.get("OMP_NUM_THREADS") == "1", (
            "the fixture did not mutate the environment — the test proves nothing"
        )
        # Now IMPORT the loader, with the pollution ALREADY in place.  This is
        # the scenario the defect lived in: a loader that snapshots
        # ``os.environ`` AT IMPORT TIME records the pollution, so its row
        # depends on WHICH module was imported first.  A loader that reads the
        # exec-time environment (``/proc/self/environ``) does not.
        importlib.reload(loader)
        after = loader.resolved_environment()
        assert after == before, (
            "the recorded environment changed after an import-time mutation — "
            "the snapshot is import-order dependent"
        )
    finally:
        for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.pop(_name, None)
        importlib.reload(loader)


@pytest.mark.skipif(not TRAINER.exists(), reason="trainer not present")
def test_the_environment_row_can_fail_on_a_real_drift(tmp_path: Path) -> None:
    """The environment row must be able to FAIL.

    A row that can only ever be equal is a DEAD row (honest-negative catalog
    §6.269.17).  The drift here is a REAL one: the wrapper ran with a
    different ``PYTHONHASHSEED`` than the trainer did.
    """
    out = tmp_path / "run"
    res = _emit(out, env_overrides={"PYTHONHASHSEED": "12345"})
    assert res.returncode == 0, res.stderr
    argv = [
        "--config",
        str(CONFIG),
        "--out-dir",
        str(out),
        "--ckpt-dir",
        str(out / "ckpts"),
    ]
    _trainer_side_write(out, argv)
    res = _run_prov("verify", "--out-dir", str(out))
    assert res.returncode == 1, (
        f"the environment drift was NOT detected (rc={res.returncode}): {res.stdout}"
    )
    assert "environment" in res.stdout and "MISMATCH" in res.stdout, res.stdout
