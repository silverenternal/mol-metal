# SPDX-License-Identifier: Apache-2.0
"""The guidance sweep's YAML must actually reach the trainer's argparse dests.

WHY THIS EXISTS
---------------
AGENTS.md 5.11 requires hyperparameters to live in configs/*.yaml rather than
on a command line, and carries two hard-won warnings that this test enforces:

1. A YAML key that resolves to no argparse dest is a SILENT lie unless it is
   reported. ``apply_yaml_overrides`` was rewritten for exactly this. A sweep
   config whose ``n_steps`` is spelled ``steps`` would run 48 clouds at the
   default step count while the file claims otherwise, and nothing would say so.

2. Round-trips must be verified BY TYPE, not by value. A value that happens to
   equal the argparse default is indistinguishable from a key that never
   arrived -- which is how ``batch_size`` at the top level of a config went
   unnoticed for so long (memory: flat-yaml-keys-silently-ignored).

So every assertion here checks the resulting object against the value the YAML
declares, including its type, and every one of them would fail if the key were
dropped rather than merely defaulting.

RED CHECK: rename a key in the config (e.g. ``n_clouds`` -> ``n_cloud``) and
the matching test fails; restoring it returns to green.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
CFG = REPO / "configs" / "cfm_l3_guidance_sweep_2026_10_12.yaml"
PROBE = REPO / "molmetal" / "scripts" / "wf_blkc_l3_guidance_probe.py"


def _probe():
    spec = importlib.util.spec_from_file_location("l3_probe", str(PROBE))
    m = importlib.util.module_from_spec(spec)
    sys.modules["l3_probe"] = m
    spec.loader.exec_module(m)
    return m


def _ns(**over):
    # ckpt/out are None here on purpose: setting them would look like an
    # explicit CLI flag and (correctly) beat the config, so the config's own
    # ckpt/out would never be exercised.
    base = dict(
        ckpt=None, out=None, config=CFG,
        n_clouds=None, n_steps=None, method=None, seed=None,
        guidance_strength=None, arm=None,
        skip_decode=None, compare_real=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_every_declared_key_reaches_a_dest_with_its_declared_type():
    m = _probe()
    declared = {}
    for sect in ("model", "eval", "output"):
        declared.update(yaml.safe_load(CFG.read_text(encoding="utf-8"))[sect])
    got = m._merge_config(_ns())

    for key, want in declared.items():
        have = getattr(got, key)
        assert have == want, (
            f"{key}: config declares {want!r}, argparse received {have!r}"
        )
        assert type(have) is type(want), (
            f"{key}: TYPE mismatch. config says {type(want).__name__}, "
            f"argparse got {type(have).__name__}. A value that happens to "
            "equal the default proves nothing -- a dropped key looks identical."
        )


def test_store_true_is_read_as_a_bool_not_a_truthy_string():
    """A store_true flag handed the string "false" is truthy.

    That is the shape of bug AGENTS.md 5.11 warns about for list-valued flags
    (``metals: Ru Ir Os Re`` arriving as one string): a flag whose type does
    not survive the YAML round-trip. Here skip_decode must be a real False.
    """
    m = _probe()
    got = m._merge_config(_ns())
    assert got.skip_decode is False, (
        f"skip_decode became {got.skip_decode!r}; decode is the whole point "
        "of this sweep and a truthy string would silently skip it"
    )
    assert got.compare_real is True, (
        f"compare_real became {got.compare_real!r}; without the real-tmQM "
        "reference there is nothing for the generated numbers to be compared to"
    )


def test_explicit_cli_flag_beats_the_config():
    """Precedence is explicit CLI > YAML > default (AGENTS.md 5.11)."""
    m = _probe()
    got = m._merge_config(_ns(guidance_strength=0.42, n_clouds=8))
    assert got.guidance_strength == 0.42, (
        "an explicitly passed flag was overwritten by the config; a sweep "
        "would then run an arm the caller did not ask for"
    )
    assert got.n_clouds == 8, "explicit CLI --n-clouds lost to the config"
    assert got.n_steps == 50, "a key the caller did NOT pass should come from YAML"


def test_no_config_falls_back_to_the_documented_defaults():
    m = _probe()
    got = m._merge_config(_ns(config=None))
    assert got.n_clouds == 48 and got.n_steps == 50 and got.seed == 1234
    assert got.guidance_strength == 0.0
    assert got.skip_decode is False and got.compare_real is False


def test_typo_in_a_key_is_rejected_loudly(tmp_path: Path):
    """The silent-ignore class. A key that names no dest must abort."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("run:\n  n_cloud: 64\n", encoding="utf-8")   # singular typo
    r = subprocess.run(
        [sys.executable, str(PROBE), "--ckpt", "x.pt", "--out", "x.json",
         "--config", str(bad), "--n-clouds", "1"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode != 0, (
        "a YAML key resolving to no argparse dest was accepted silently -- "
        "the run would proceed while the file claims a setting it never applied"
    )
    assert "n_cloud" in (r.stderr + r.stdout), (
        "the failure did not NAME the offending key: " + (r.stderr + r.stdout)[-400:]
    )


def test_wrong_type_is_rejected(tmp_path: Path):
    bad = tmp_path / "typed.yaml"
    bad.write_text("run:\n  n_clouds: many\n", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(PROBE), "--ckpt", "x.pt", "--out", "x.json",
         "--config", str(bad)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode != 0, "n_clouds: many was coerced instead of rejected"
    assert "n_clouds" in (r.stderr + r.stdout)


def test_strength_grid_brackets_the_measured_point_and_carries_a_control():
    """The grid is derived from the one measured arm, not guessed.

    0.05 is the only strength with a number on disk
    (metrics/by_metric/cfm_l3_guidance_ab.json). A sweep must contain it, must
    contain the 0.0 control, and must bracket 0.05 on BOTH sides -- because the
    0.05 arm overshot the global minimum (1.155 vs 1.172 real) while still
    under-shooting the metal-ligand contact (1.602 vs 1.953). A grid that only
    scales outward would test one direction and assume the answer.
    """
    s = yaml.safe_load(CFG.read_text(encoding="utf-8"))["sweep"]["strengths"]
    assert isinstance(s, list), f"strengths must be a YAML LIST, got {type(s)}"
    assert s[0] == 0.0, "the first arm must be the unguided control"
    assert 0.05 in s, "the one arm with a prior measurement must be in the sweep"
    assert any(x < 0.05 for x in s), "no arm tests the overshoot direction"
    assert any(x > 0.05 for x in s), "no arm tests whether the 0.351 A metal-ligand gap closes"
    assert len(s) == len(set(s)), "duplicate strengths would spend GPU for nothing"

def test_main_body_was_not_swallowed_by_another_function():
    """A refactor here once moved ALL of main()'s body into _merge_config.

    main() then returned None immediately, and the script exited 0 printing
    nothing and writing no file. Every other test in this file still passed,
    because they exercise _merge_config directly and never call main(). That is
    the seventh instance in this repo of the same shape -- the process reports
    success while doing nothing -- and it is caught by asserting the STRUCTURE
    of main, not its return value.

    Cheap, needs no GPU and no checkpoint, and it fails the moment a function
    boundary is misplaced.
    """
    import ast

    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in fns, "main() is missing from the probe"
    main = fns["main"]

    called = {
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
    }
    for required in ("load_model", "load_clouds"):
        assert required in called, (
            f"main() does not call {required}(); its body was probably "
            f"reassigned to another function. main has {len(main.body)} "
            "statements -- check the function boundaries with ast."
        )

    # The write is the observable effect. If it is not in main(), a run can
    # exit 0 and produce nothing, which is exactly the failure being guarded.
    # (`os.replace(tmp, cli.out)` is an attribute call, so both forms are
    # collected above.)
    assert "replace" in called or "write_text" in called, (
        "main() never writes its --out file; a run would exit 0 and leave "
        "nothing on disk to check"
    )

    # And main must actually be reachable from __main__.
    entry = [
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and getattr(n.test.left, "id", "") == "__name__"
    ]
    assert entry, "the probe has no `if __name__ == '__main__'` entrypoint"
    entry_calls = {
        n.func.id
        for n in ast.walk(entry[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "main" in entry_calls, (
        "the __main__ entrypoint does not call main(); the script would "
        "import cleanly, print nothing, and exit 0"
    )


def test_decode_uses_apis_that_actually_exist():
    """The decode path was written against three symbols that do not exist.

    It imported ``YuelBondBondDecoder`` from ``lam_chem.yuelbond_decoder``
    (it lives in ``lam_chem.cfm_decoder_yuelbond``), constructed it with a
    ``device=`` kwarg it does not accept, called a ``to_smiles`` method that
    does not exist (the real entry point is ``decode(cloud)``), and imported
    ``AtomCloud`` from ``sbdd_env.atom_cloud`` (it lives in
    ``molmetal.models.bond_head``).

    None of that was caught for a day because every measurement so far passed
    ``--skip-decode``, so the broken code was never reached. A run then
    reported ``n_with_smiles: 0`` from four swallowed ImportErrors. This test
    pins the imports and the call shape so a rename cannot rot them silently.

    Cheap: no checkpoint, no GPU, no forward pass.
    """
    import importlib
    import inspect

    dec_mod = importlib.import_module(
        "molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond"
    )
    assert hasattr(dec_mod, "YuelBondBondDecoder"), (
        "YuelBondBondDecoder moved; the probe's decode() import is stale"
    )

    from molmetal.models.bond_head import AtomCloud
    assert AtomCloud.__module__ == "molmetal.models.bond_head", (
        f"AtomCloud is defined in {AtomCloud.__module__}, not molmetal.models.bond_head"
    )

    cls = dec_mod.YuelBondBondDecoder
    params = inspect.signature(cls.__init__).parameters
    for kw in ("head", "hidden_dim", "n_layers", "decision",
               "decision_threshold", "connectivity_prior",
               "connectivity_prior_min_logit"):
        assert kw in params, (
            f"YuelBondBondDecoder has no {kw!r} parameter; the probe passes it "
            f"and will raise TypeError. Actual: {list(params)}"
        )
    assert not hasattr(cls, "to_smiles"), (
        "a to_smiles method exists now -- if it is the supported entry point, "
        "the probe should be switched to it deliberately, not left on decode()"
    )
    assert hasattr(cls, "decode"), "YuelBondBondDecoder.decode() is missing"

    # The probe must name the class it can actually construct.
    src = PROBE.read_text(encoding="utf-8")
    assert "lam_chem.cfm_decoder_yuelbond import" in src, (
        "the probe does not import YuelBondBondDecoder from the module that "
        "defines it"
    )
    assert "lam_chem.yuelbond_decoder import YuelBondBondDecoder" not in src, (
        "the probe still imports YuelBondBondDecoder from a module that does "
        "not define it"
    )
    assert "res.decoded" in src, (
        "the probe never reads res.decoded, so a successful decode would be "
        "reported as no SMILES"
    )


def test_decode_swallow_cannot_hide_a_module_error():
    """Per-cloud exceptions are caught, but a decode that imports nothing
    would report n_with_smiles=0 rather than failing.

    The four broken imports above did exactly that: 4/4 rows carried an
    ``error`` field and the script still exited 0 with a DECODE: 0 / 4 line.
    An all-zero decode with a non-zero n_errors is a harness failure, not a
    model result, so the probe must say so rather than leave it to be read.
    """
    import ast

    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main = fns["main"]
    src = PROBE.read_text(encoding="utf-8")

    assert "n_errors" in src, "the decode summary does not carry n_errors"
    # Must be a REAL distinction in code, not a word that happens to appear in
    # a comment. An earlier version of this assertion tested
    # `"harness" in src.lower()` and passed on its own docstring.
    dec_fn = fns["decode"]
    main_src = ast.unparse(main)
    assert "all_clouds_errored" in main_src, (
        "main() does not branch on all_clouds_errored, so a decode in which "
        "every cloud raised still prints as '0 / N' and exits 0 -- the reading "
        "four broken imports produced"
    )
    returns = [
        n.value.value
        for n in ast.walk(main)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, int)
    ]
    assert any(v != 0 for v in returns), (
        "main() returns 0 unconditionally; a total decode failure must be a "
        "non-zero exit so a sweep loop cannot record it as a completed arm"
    )
    dec_src = ast.unparse(dec_fn)
    assert "for c in clouds" in dec_src, (
        "decode() does not iterate the sampled clouds, so it returns at most "
        "one row regardless of how many were sampled"
    )
    assert "res.decoded" in dec_src, (
        "decode() never reads res.decoded.smiles -- the result would be "
        "computed and discarded"
    )
    assert dec_src.count("except Exception") >= 2, (
        "decode() must guard both cloud construction and the decode call; one "
        "guard means a failure in the other aborts the whole sweep"
    )


def test_every_dest_ends_with_a_value_not_none(tmp_path: Path):
    """A config that sets SOME keys must not leave the rest as None.

    guidance_strength lived only as a LIST under `sweep.strengths` (a
    launcher-owned section this script deliberately does not apply), so a run
    without an explicit --guidance-strength reached the compute path with
    None and raised `TypeError: '>' not supported between instances of
    'NoneType' and 'float'`. A config gap surfaced as a crash in the guidance
    branch, far from its cause.
    """
    m = _probe()
    got = m._merge_config(_ns())
    for key in ("n_clouds", "n_steps", "method", "seed", "arm",
                "guidance_strength", "skip_decode", "compare_real"):
        assert getattr(got, key) is not None, (
            f"{key} is None after merging the config; it will crash in the "
            "compute path rather than at the merge"
        )
    assert got.guidance_strength == 0.0, (
        "guidance_strength must fall back to 0.0 (the unguided control), not "
        "to None -- None is not a strength"
    )


def test_command_template_carries_no_hyperparameter():
    """AGENTS.md 5.11: a command_template may carry only {placeholder} values.

    The first version of this config embedded `--guidance-strength 0.0` and
    `--arm strength0.0` in the template. scripts/_config_binding_gate.py failed
    it: "'entry.command_template carries --guidance-strength' resolves to no
    dest". A strength in a template is a hyperparameter that lives in a place
    the YAML cannot audit -- the same class of defect as the shell wrappers
    that originally hid use_fm_objective from five of seven CFM configs.

    Everything the run needs must therefore be reachable from `--config`.
    """
    e = yaml.safe_load(CFG.read_text(encoding="utf-8"))["entry"]
    tpl = e["command_template"]
    flags = [w for w in tpl.split() if w.startswith("--")]
    assert flags == ["--config"], (
        f"command_template carries {flags}; only --config (plus bare "
        "{placeholder} values) is permitted. Anything else is a "
        "hyperparameter smuggled past the YAML."
    )
    assert "{yaml_path}" in tpl, "the template must select the config by placeholder"
    assert e["script"].endswith(".py"), "entry.script must name the script"


def test_config_alone_is_a_complete_invocation():
    """`--config {yaml_path}` must be runnable with no other flag.

    That is what makes the template legal: if a required flag were left out of
    the YAML, the only way to run it would be to put the value on the command
    line, which is the violation above.
    """
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    run = {}
    for sect in ("model", "eval", "output"):
        run.update(data[sect])
    for key in ("ckpt", "out", "n_clouds", "n_steps", "method", "seed"):
        assert key in run, (
            f"{key} is absent, so running with --config alone would fail "
            "or silently take a default the file does not record"
        )
    ck = REPO / run["ckpt"]
    assert ck.is_file(), f"run.ckpt does not exist: {ck}"


def test_launcher_passes_only_run_selecting_flags():
    """The launcher may override ckpt/out/strength/arm -- which RUN, not WHAT.

    Anything else would be a hyperparameter arriving from outside the YAML.
    """
    launcher = REPO / "scripts" / "launch_cfm_l3_guidance_sweep.sh"
    text = launcher.read_text(encoding="utf-8")
    body = text.split("for S in $STRENGTHS; do", 1)[1]
    # Drop echo banners and comments; a `--- arm x ---` line is not a flag.
    body = "\n".join(
        ln for ln in body.splitlines()
        if not ln.lstrip().startswith(("#", "echo ", '"'))
    )
    flags = {w for w in body.split() if w.startswith("--")}
    allowed = {"--ckpt", "--config", "--out", "--guidance-strength", "--arm"}
    extra = {f for f in flags if f.rstrip('\\"') not in allowed}
    assert not extra, (
        f"the launcher passes {sorted(extra)}, which are not run-selecting "
        f"flags. Allowed: {sorted(allowed)}"
    )
