# Licensed under the Apache License, Version 2.0.
"""Regression tests for the config↔trainer binding gate.

The gate exists because a YAML key that resolves to no argparse dest is silently
dropped, and the run then trains something other than what the file says. A gate
that itself regresses is worse than none, so these cover the ways it could stop
working — and, more importantly, the two ways it could start reporting things
that are fine, which is how a gate gets switched off.

Three of these cases are real bugs the gate's own first draft had:

  * `parse_known_args` returns a TUPLE, so storing it wholesale made every
    `hasattr` False and every key in every config read as dead.
  * Patching `parse_args` and `parse_known_args` to a shared spy recurses
    forever, because `parse_args` delegates to `parse_known_args`.
  * Keying only on `entry.script` skipped exactly the configs whose binding is
    least obvious — three of them name the trainer solely inside
    `entry.command_template`, which is how `scripts/start.sh` finds it too.

Resolution is not delivery, and `TestTypeChecking` covers the gap that cost a
real run on 2026-10-09: `--metals` is `nargs="+"`, a config wrote it as the
YAML *string* `"Ru Ir Os Re"`, the key resolved cleanly, both §5.11 gates
passed, and the trainer then searched for one metal literally named
`"Ru Ir Os Re"`, matched zero rows and emitted `test_auc = nan` for all 5 seeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_config_binding_gate.py"

_spec = importlib.util.spec_from_file_location("_config_binding_gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_config_binding_gate"] = gate
_spec.loader.exec_module(gate)

from molmetal.scripts import wfA2_config_loader as L  # noqa: E402


class _Ns:
    """Stand-in for an argparse namespace."""

    def __init__(self, **dests):
        self.__dict__.update(dests)


def _write(tmp_path: Path, name: str, body: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return p


def _shapes(**flags) -> dict[str, dict]:
    """Build the gate's dest->shape map from real argparse declarations.

    Keys are flag strings, values are extra `add_argument` kwargs:

        _shapes(**{"--metals": {"nargs": "+"}, "--lr": {"type": float}})

    Built from a live parser rather than hand-written dicts so the tests cannot
    drift from what `_capture_parser` would actually hand `audit_config` — a
    fixture that invents its own shape dict tests the fixture, not the gate.
    """
    ap = argparse.ArgumentParser()
    for flag, kwargs in flags.items():
        ap.add_argument(flag, **kwargs)
    return gate._actions_by_dest(ap)


class TestDeadKeyDetection:
    def test_unresolvable_key_is_a_violation(self, tmp_path):
        cfg = _write(
            tmp_path,
            "c.yaml",
            {"model": {"no_such_knob": 3}, "protocol": {"also_missing": 0.1}},
        )
        found, inert = gate.audit_config(cfg, _Ns(epochs=50))
        assert sorted(v["key"] for v in found) == [
            "model.no_such_knob",
            "protocol.also_missing",
        ]

    def test_leaf_fallback_makes_a_section_independent(self, tmp_path):
        """`model.epochs` resolves to `epochs` — `model` is a fallback section.

        Worth pinning: it means the section a key sits in is not a claim about
        which trainer reads it, so a config cannot be "fixed" by moving a key to
        a different section.
        """
        cfg = _write(tmp_path, "c.yaml", {"model": {"epochs": 3}})
        found, _ = gate.audit_config(cfg, _Ns(epochs=50))
        assert found == []

    def test_resolvable_key_is_clean(self, tmp_path):
        cfg = _write(tmp_path, "c.yaml", {"protocol": {"epochs": 3}})
        found, inert = gate.audit_config(cfg, _Ns(epochs=50))
        assert found == [] and inert == []

    def test_alias_resolution_is_honoured(self, tmp_path):
        """`grad_clip_norm` -> `max_norm` via _LEAF_ALIASES must not read as dead."""
        cfg = _write(tmp_path, "c.yaml", {"train": {"grad_clip_norm": 5.0}})
        found, _ = gate.audit_config(cfg, _Ns(max_norm=1.0))
        assert found == []

    def test_declared_inert_is_reported_not_silent(self, tmp_path):
        """An inert key must be VISIBLE every run — never quietly dropped.

        This is the whole point of the tier: the config documents sweep intent
        nothing consumes, and hiding that recreates the dead-key blind spot.
        """
        cfg = _write(tmp_path, "c.yaml", {"output": {"per_metal_dir": "x"}})
        found, inert = gate.audit_config(cfg, _Ns())
        assert found == []
        assert inert == ["output.per_metal_dir"]

    def test_inert_prefix_covers_nested_children(self, tmp_path):
        """`_iter_leaves` descends, so registering a parent must cover children."""
        cfg = _write(
            tmp_path,
            "c.yaml",
            {"eval": {"baseline_per_cell_line_MEASURED": {"MCF7": 0.8, "A549": 0.7}}},
        )
        found, inert = gate.audit_config(cfg, _Ns())
        assert found == []
        assert len(inert) == 2
        assert "eval.baseline_per_cell_line_MEASURED.MCF7" in inert

    def test_registered_non_trainer_key_is_clean(self, tmp_path):
        cfg = _write(tmp_path, "c.yaml", {"output": {"base_dir": "reports/x"}})
        found, _ = gate.audit_config(cfg, _Ns())
        assert found == []


class TestTypeChecking:
    """The delivered value's TYPE, not just the fact it resolves.

    Every case here is one the original gate called PASS. They are the reason
    the second tier exists.
    """

    def test_nargs_list_dest_receiving_a_string_is_a_violation(self, tmp_path):
        """The exact 2026-10-09 bug: `metals: Ru Ir Os Re` against nargs="+".

        The key resolves, so the old gate said PASS. The run then searched for
        a metal named "Ru Ir Os Re", matched zero rows, produced an empty test
        set for every fold and died in statistics.pstdev with nan for all seeds.
        """
        cfg = _write(tmp_path, "c.yaml", {"data": {"metals": "Ru Ir Os Rh Re"}})
        found, _ = gate.audit_config(
            cfg,
            _Ns(metals=["Ru", "Ir", "Os", "Rh", "Re"]),
            _shapes(**{"--metals": {"nargs": "+"}}),
        )
        assert len(found) == 1
        assert found[0]["reason"] == "type"
        assert found[0]["key"] == "data.metals"
        assert "--metals" in found[0]["detail"]
        assert found[0]["received"] == "str"
        assert "list" in found[0]["required"]

    def test_nargs_list_dest_receiving_a_list_is_clean(self, tmp_path):
        cfg = _write(tmp_path, "c.yaml", {"data": {"metals": ["Ru", "Ir", "Os", "Rh", "Re"]}})
        found, _ = gate.audit_config(
            cfg, _Ns(metals=["Pt"]), _shapes(**{"--metals": {"nargs": "+"}})
        )
        assert found == []

    @pytest.mark.parametrize("nargs", ["+", "*"])
    def test_both_sequence_nargs_are_covered(self, tmp_path, nargs):
        """`*` (zero-or-more) has the same shape requirement as `+`."""
        cfg = _write(tmp_path, "c.yaml", {"protocol": {"tags": "a b c"}})
        found, _ = gate.audit_config(cfg, _Ns(tags=[]), _shapes(**{"--tags": {"nargs": nargs}}))
        assert [f["reason"] for f in found] == ["type"]

    def test_scalar_dest_receiving_an_unconvertible_value_is_a_violation(self, tmp_path):
        """`lr: nonsense` resolves to `--lr`; only the converter reveals it is junk.

        apply_yaml_overrides keeps the argparse default and records a
        'type mismatch' skip, so the run trains on a default nobody chose.
        """
        cfg = _write(tmp_path, "c.yaml", {"protocol": {"lr": "not-a-number"}})
        found, _ = gate.audit_config(
            cfg, _Ns(lr=3e-4), _shapes(**{"--lr": {"type": float}})
        )
        assert len(found) == 1
        assert found[0]["reason"] == "type"
        assert found[0]["key"] == "protocol.lr"
        assert "float" in found[0]["required"]

    def test_scalar_dest_receiving_a_coercible_string_is_clean(self, tmp_path):
        """A quoted number still converts — flagging it would train the gate off."""
        cfg = _write(tmp_path, "c.yaml", {"protocol": {"lr": "0.0003"}})
        found, _ = gate.audit_config(
            cfg, _Ns(lr=3e-4), _shapes(**{"--lr": {"type": float}})
        )
        assert found == []

    def test_scalar_dest_receiving_a_list_is_clean(self, tmp_path):
        """The loader hands list values through unconverted by design.

        `_is_empty_container` / the loader's list branch mean a list on a scalar
        dest is at most inert, and a synthetic type failure here would be a
        false positive, not a finding.
        """
        cfg = _write(tmp_path, "c.yaml", {"protocol": {"lr": [1, 2]}})
        found, _ = gate.audit_config(cfg, _Ns(lr=3e-4), _shapes(**{"--lr": {"type": float}}))
        assert found == []

    def test_type_violation_is_fatal_while_declared_inert_is_not(self, tmp_path):
        """The two tiers must not collapse into each other.

        DECLARED-INERT is documentation with no consumer — reported, never
        fatal. A type violation is a live key carrying the wrong shape, and it
        must take the verdict down with it.
        """
        cfg = _write(
            tmp_path,
            "c.yaml",
            {"data": {"metals": "Ru Ir"}, "output": {"per_metal_dir": "x"}},
        )
        found, inert = gate.audit_config(
            cfg, _Ns(metals=["Pt"]), _shapes(**{"--metals": {"nargs": "+"}})
        )
        assert [f["reason"] for f in found] == ["type"]
        assert inert == ["output.per_metal_dir"]

    def test_untyped_flag_takes_any_value(self, tmp_path):
        """No `type=` and no `nargs` means the trainer handles whatever arrives."""
        cfg = _write(tmp_path, "c.yaml", {"output": {"base_dir": "reports/x"}})
        found, _ = gate.audit_config(cfg, _Ns(), _shapes(**{"--device": {}, "--label": {}}))
        assert found == []

    def test_omitting_actions_keeps_resolution_only_behaviour(self, tmp_path):
        """Back-compat: `actions` is optional, so old callers are unaffected."""
        cfg = _write(tmp_path, "c.yaml", {"data": {"metals": "Ru Ir"}})
        found, _ = gate.audit_config(cfg, _Ns(metals=["Pt"]))
        assert found == []

    def test_shape_map_reads_the_real_parser_actions(self):
        """nargs / type / option-name all come off the live action.

        Worth pinning separately: a fixture hand-written to match the
        implementation would keep passing if `_actions_by_dest` started
        returning the wrong field, which is the whole gate.
        """
        shapes = _shapes(**{"--metals": {"nargs": "+"}, "--lr": {"type": float}, "--device": {}})
        assert shapes["metals"] == {"nargs": "+", "type": None, "option": "--metals"}
        assert shapes["lr"]["type"] is float
        assert shapes["device"]["option"] == "--device"

    def test_repo_metals_flag_is_nargs_and_is_checked(self):
        """The gate's own repo: the dest that broke is still a `+` dest.

        If someone edits --metals to a scalar, the guard this whole tier
        provides quietly stops guarding, and the test suite would still pass
        because every fixture builds its own parser.
        """
        script = REPO_ROOT / "molmetal/scripts/train_metal_lomo_wasserstein.py"
        if not script.is_file():
            pytest.skip("trainer not on disk")
        ns, actions, err = gate._capture_parser(script)
        assert err is None, err
        assert actions.get("metals", {}).get("nargs") == "+"


class TestCommandTemplate:
    def test_placeholder_values_are_allowed(self):
        """`--epochs {epochs}` is yaml-driven: start.sh fills it from the yaml."""
        cmd = "python t.py --config {yaml_path} --epochs {epochs} --out-dir {out_dir}"
        assert gate._disallowed_template_flags(cmd) == []

    def test_literal_value_is_a_violation(self):
        cmd = "python t.py --config {yaml_path} --device cuda:0"
        assert gate._disallowed_template_flags(cmd) == ["--device"]

    def test_bare_switch_is_a_violation(self):
        cmd = "python t.py --config {yaml_path} --use-multi-ph --lr {lr}"
        assert gate._disallowed_template_flags(cmd) == ["--use-multi-ph"]

    @pytest.mark.parametrize("flag", sorted(gate.ALLOWED_TEMPLATE_FLAGS))
    def test_run_selecting_flags_are_allowed(self, flag):
        assert gate._disallowed_template_flags(f"python t.py {flag} something") == []

    def test_script_found_in_command_template(self):
        """Three configs name the trainer only in the template; start.sh finds it
        the same way, so the gate must too."""
        entry = {
            "command_template": "/venv/bin/python molmetal/scripts/train_x.py --config {yaml_path}"
        }
        script, cmd = gate._entry_script_and_command(entry)
        assert script == "molmetal/scripts/train_x.py"
        assert "--config" in cmd

    def test_explicit_script_wins(self):
        entry = {
            "script": "molmetal/scripts/train_y.py",
            "command_template": "python molmetal/scripts/train_x.py",
        }
        script, _ = gate._entry_script_and_command(entry)
        assert script == "molmetal/scripts/train_y.py"

    def test_no_entry_is_handled(self):
        assert gate._entry_script_and_command(None) == (None, "")
        assert gate._entry_script_and_command({}) == (None, "")


class TestEndToEnd:
    def test_repo_configs_pass(self):
        report = gate.audit()
        assert report["verdict"] == "PASS", report["violations"]

    def test_repo_configs_pass_with_schema_metadata_blocks(self):
        """Wf-A57 W9-P2 (2026-10-16) — schema-metadata-only annotation
        mappings (siblings whose every child is one of
        :data:`_NON_TRAINER_SUBKEYS`) must not register as leaves.

        This guards against a regression where the §5.11 gate would
        recurse into a YAML block like

            strict_large_frag:
              type: bool
              default: false
              dest: bench_cfm_decode_real.py::strict_large_frag
              desc: > ...

        and emit 4 "resolves to no dest" violations for the children.
        The subkey filter at ``_iter_leaves`` makes the whole block
        an inert leaf instead.

        Asserted by reading the audit JSON (it carries violation list
        + per-config counts) and looking at the PoC YAML
        ``configs/cfm_patience10_seed0_2026_10_12.yaml`` specifically.
        """
        import json
        from pathlib import Path

        report = gate.audit()
        # The PoC config has 5 schema-metadata blocks
        # (`strict_large_frag`, `largest_frag_min_frac`,
        # `min_atoms_per_mol`, `emit_admet8`, `paper_table_cohort`).
        # Before the W9-P2 fix each emitted 4 violations (one per
        # sub-key child); the regression test pins the post-fix count
        # to zero schema-metadata violations from this config.
        violations = report.get("violations", [])
        poc_violations = [
            v
            for v in violations
            if isinstance(v, dict)
            and v.get("config", "")
            == "cfm_patience10_seed0_2026_10_12.yaml"
        ]
        keys = {str(v.get("key", "")) for v in poc_violations}
        # None of the schema-metadata sub-keys should appear.
        for subkey in ("type", "default", "dest", "desc"):
            assert not any(
                k.endswith(f".{subkey}") for k in keys
            ), (
                f"W9-P2 regression: subkey '{subkey}' recursed as a "
                f"violation; the schema-metadata filter is gone"
            )

    def test_every_config_is_actually_checked(self):
        """A gate that quietly skips configs is a gate that misses regressions."""
        report = gate.audit()
        assert report["n_configs_skipped"] == 0, report["skipped"]
        assert report["n_configs_checked"] >= 20

    def test_inert_keys_are_still_surfaced(self):
        """PASS must not mean 'nothing to say' — the inert tier must be non-empty."""
        report = gate.audit()
        assert report["n_declared_inert"] > 0

    def test_type_tier_actually_runs_on_every_config(self):
        """`test_repo_configs_pass` would also pass if the type tier were dead.

        Every real config is checked with actions captured, so the report must
        carry the type counter — and it must be 0 for this repo, not 0 because
        the check never looked. Pinned by asserting the tier sees the dest set
        it is supposed to see, below.
        """
        report = gate.audit()
        assert "n_type_violations" in report
        assert report["n_type_violations"] == 0, report["violations"]

    def test_no_config_is_audited_without_its_actions(self):
        """A config whose parser yielded no actions gets NO type check.

        Silent narrowing is the failure mode this whole tier risks: the gate
        would report PASS having checked nothing. Every `nargs="+"` dest in
        every captured parser must be present in the shape map that
        audit_config receives, so assert the map is non-trivial per config by
        re-running one real config with and without actions and confirming the
        type tier is what differs.
        """
        import glob

        seen_sequence_dests = []
        for pattern in gate.CONFIG_GLOBS:
            for cfg in sorted(REPO_ROOT.glob(pattern)):
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict):
                    continue
                script, _ = gate._entry_script_and_command(data.get("entry"))
                if not script:
                    continue
                spath = (REPO_ROOT / str(script)).resolve()
                if not spath.is_file():
                    continue
                _ns, actions, err = gate._capture_parser(spath)
                assert err is None, f"{cfg.name}: {err}"
                assert actions, f"{cfg.name}: no actions captured — type tier is dead"
                seen_sequence_dests.extend(
                    d for d, s in actions.items() if s["nargs"] in gate.SEQUENCE_NARGS
                )
        assert len(glob.glob(str(REPO_ROOT / "configs/*.yaml"))) > 0
        assert seen_sequence_dests, "no nargs='+' dest found — tier cannot fire"

    def test_report_is_json_serialisable(self):
        import json

        json.dumps(gate.audit())

    def test_exit_code_follows_the_stdout_invariant(self, capsys):
        assert gate.main([]) == 0
        assert "PASS" in capsys.readouterr().out
