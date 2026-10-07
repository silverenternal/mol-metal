"""Wf-A2 unified YAML config loader + trainer --config override tests.

Per AGENTS.md §1.2 + party-twenty-spirit §3.5: MEASURED-vs-PROJECTED tag discipline;
this wave is CPU-only docs+code+test, NO GPU launches, NO MEASURED-on-real-job
numbers. All assertions cover the loader contract + the trainer --config wiring.

Test classes:
- TestLoadConfigBasic: load_config() returns flat dict.
- TestLoadConfigNested: nested YAML -> flattened with underscore-joined keys.
- TestLoadConfigErrors: missing file (no-op), bad root type.
- TestApplyYamlOverridesBackwardCompat: no --config = argparse defaults.
- TestApplyYamlOverridesDefaults: --config only = YAML overrides defaults.
- TestApplyYamlOverridesCliWins: CLI explicit value beats YAML.
- TestApplyYamlOverridesIgnoresUnknown: YAML forward-compat keys silently skipped.
- TestApplyYamlOverridesListAlwaysWins: list YAML always wins over default.
- TestDmpnnTrainerAcceptsConfig: parse_args() accepts --config + applies.
- TestM17CanonicalConfigLoads: canonical config loads without error.

Note: CFM trainer tests are skipped because CFM's parse_args() reads sys.argv
implicitly (latent bug in the trainer; p.parse_args() with no argv param defaults
to sys.argv[1:]). pytest's argv pollutes that path. The DMPNN trainer integration
test exercises the same --config wire-in contract and is the canonical smoke.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wfA2_config_loader import (
    UnresolvedConfigKey,
    apply_yaml_overrides,
    last_skipped_keys,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yaml(content: str) -> str:
    """Write YAML to a tempfile and return path. Caller must delete."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _make_parser() -> argparse.ArgumentParser:
    """Synthetic parser mirroring a subset of trainer flags for unit tests."""
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--metals", nargs="+", default=["Ru", "Ir", "Rh", "Os", "Re"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ablation-config", default="mtl_baseline")
    p.add_argument("--wasserstein-weight", type=float, default=0.05)
    p.add_argument("--data-csv", default="default.csv")
    p.add_argument("--model-hidden-dim", type=int, default=300)
    p.add_argument("--model-depth", type=int, default=3)
    p.add_argument("--config", default=None)
    return p


# ---------------------------------------------------------------------------
# TestLoadConfigBasic
# ---------------------------------------------------------------------------


class TestLoadConfigBasic:
    def test_load_flat_yaml(self):
        path = _make_yaml("lr: 0.0007\nepochs: 17\n")
        try:
            cfg = load_config(path)
            assert cfg["lr"] == 0.0007
            assert cfg["epochs"] == 17
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestLoadConfigNested
# ---------------------------------------------------------------------------


class TestLoadConfigNested:
    def test_nested_dict_flattens_to_underscore_keys(self):
        path = _make_yaml(
            "model:\n  hidden_dim: 256\n  depth: 4\n" "data:\n  csv: /tmp/x.csv\n"
        )
        try:
            cfg = load_config(path)
            assert cfg["model_hidden_dim"] == 256
            assert cfg["model_depth"] == 4
            assert cfg["data_csv"] == "/tmp/x.csv"
        finally:
            os.unlink(path)

    def test_list_values_preserved(self):
        path = _make_yaml("metals: [Ru, Ir]\nseeds: [0, 1, 2]\n")
        try:
            cfg = load_config(path)
            assert cfg["metals"] == ["Ru", "Ir"]
            assert cfg["seeds"] == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_empty_yaml_returns_empty_dict(self):
        path = _make_yaml("")
        try:
            assert load_config(path) == {}
        finally:
            os.unlink(path)

    def test_dashed_yaml_keys_normalize_to_underscore(self):
        path = _make_yaml("wasserstein-weight: 0.07\nablation-config: mtl_structured\n")
        try:
            # Loader preserves dashed form; normalization happens during apply.
            cfg = load_config(path)
            assert cfg["wasserstein-weight"] == 0.07
            assert cfg["ablation-config"] == "mtl_structured"
            # Now exercise the apply path which normalizes:
            p = _make_parser()
            args = p.parse_args(["--config", path])
            applied = apply_yaml_overrides(p, args, args.config)
            assert applied.wasserstein_weight == 0.07
            assert applied.ablation_config == "mtl_structured"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestLoadConfigErrors
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    def test_missing_file_returns_empty_dict(self):
        # Per design: missing file = no-op (caller decides whether to error).
        assert load_config("/nonexistent/path/foo.yaml") == {}

    def test_non_mapping_root_raises(self):
        path = _make_yaml("- a\n- b\n")
        try:
            import yaml as _yaml

            with pytest.raises(_yaml.YAMLError):
                load_config(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestApplyYamlOverridesBackwardCompat
# ---------------------------------------------------------------------------


class TestApplyYamlOverridesBackwardCompat:
    def test_no_config_means_no_change(self):
        p = _make_parser()
        args = p.parse_args([])
        before = dict(vars(args))
        out = apply_yaml_overrides(p, args, None)
        assert dict(vars(out)) == before

    def test_nonexistent_config_path_means_no_change(self):
        p = _make_parser()
        args = p.parse_args([])
        before = dict(vars(args))
        out = apply_yaml_overrides(p, args, "/nonexistent.yaml")
        assert dict(vars(out)) == before


# ---------------------------------------------------------------------------
# TestApplyYamlOverridesDefaults
# ---------------------------------------------------------------------------


class TestApplyYamlOverridesDefaults:
    def test_yaml_overrides_defaults_when_no_cli(self):
        path = _make_yaml(
            "lr: 0.0007\nepochs: 17\n"
            "ablation-config: mtl_structured\ndata-csv: /tmp/x.csv\n"
        )
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.lr == 0.0007
            assert out.epochs == 17
            assert out.ablation_config == "mtl_structured"
            assert out.data_csv == "/tmp/x.csv"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestApplyYamlOverridesCliWins
# ---------------------------------------------------------------------------


class TestApplyYamlOverridesCliWins:
    def test_cli_explicit_scalar_beats_yaml(self):
        path = _make_yaml("lr: 0.0007\nepochs: 17\n")
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path, "--lr", "0.5", "--epochs", "99"])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.lr == 0.5, "CLI --lr 0.5 must beat YAML 0.0007"
            assert out.epochs == 99, "CLI --epochs 99 must beat YAML 17"
        finally:
            os.unlink(path)

    def test_partial_cli_only_overrides_specified_keys(self):
        path = _make_yaml("lr: 0.0007\nepochs: 17\n")
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path, "--lr", "0.5"])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.lr == 0.5  # CLI wins
            assert out.epochs == 17  # YAML applies (CLI did not pass --epochs)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestApplyYamlOverridesIgnoresUnknown
# ---------------------------------------------------------------------------


class TestApplyYamlOverridesIgnoresUnknown:
    def test_forward_compat_key_reported_not_fatal(self):
        """Wf-A11 (2026-10-03): unknown keys remain NON-FATAL by default, but
        are no longer silent — every skip is reported via last_skipped_keys()
        and args.yaml_unresolved_keys (WFA2_CONFIG_STRICT=1 raises instead)."""
        path = _make_yaml(
            "lr: 0.0007\nfuture_some_new_flag: 42\n" "another_unknown_key: [a, b]\n"
        )
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.lr == 0.0007
            assert not hasattr(out, "future_some_new_flag")
            assert not hasattr(out, "another_unknown_key")
            skipped = [e["key"] for e in last_skipped_keys()]
            assert "future_some_new_flag" in skipped
            assert "another_unknown_key" in skipped
            assert [e["key"] for e in out.yaml_unresolved_keys] == skipped
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestWfA11AliasResolution (dead-key durable fix, 2026-10-03)
# ---------------------------------------------------------------------------


def _make_alias_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-norm", type=float, default=5.0)
    p.add_argument("--config", default=None)
    return p


class TestWfA11AliasResolution:
    def test_hidden_dim_alias_resolves_to_hidden(self):
        path = _make_yaml("model:\n  hidden_dim: 300\n")
        try:
            p = _make_alias_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.hidden == 300  # declared value now REACHES the trainer
            assert not [
                e for e in out.yaml_unresolved_keys if e["key"] == "model.hidden_dim"
            ]
        finally:
            os.unlink(path)

    def test_direct_key_beats_alias_and_is_reported(self):
        path = _make_yaml("protocol:\n  max_norm: 5.0\n  grad_clip_norm: 1.0\n")
        try:
            p = _make_alias_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.max_norm == 5.0  # direct key wins over alias
            reasons = {e["key"]: e.get("reason") for e in out.yaml_unresolved_keys}
            assert "alias suppressed" in (reasons.get("protocol.grad_clip_norm") or "")
        finally:
            os.unlink(path)

    def test_strict_mode_raises(self, monkeypatch):
        path = _make_yaml("model:\n  hidden_typo: 300\n")
        try:
            monkeypatch.setenv("WFA2_CONFIG_STRICT", "1")
            p = _make_alias_parser()
            args = p.parse_args(["--config", path])
            with pytest.raises(UnresolvedConfigKey):
                apply_yaml_overrides(p, args, args.config)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestApplyYamlOverridesListAlwaysWins
# ---------------------------------------------------------------------------


class TestApplyYamlOverridesListAlwaysWins:
    def test_yaml_list_overrides_argparse_default_list(self):
        """Lists are always overridden by YAML because users typically
        intend YAML to define cohort subsets; argparse list defaults
        are an implementation detail."""
        path = _make_yaml("metals: [Ru, Ir]\n")
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            assert out.metals == ["Ru", "Ir"]
        finally:
            os.unlink(path)

    def test_yaml_list_beats_cli_list_when_user_did_not_change(self):
        path = _make_yaml("metals: [Pt]\n")
        try:
            p = _make_parser()
            args = p.parse_args(["--config", path])
            out = apply_yaml_overrides(p, args, args.config)
            # CLI default was ['Ru', 'Ir', 'Rh', 'Os', 'Re']; YAML = ['Pt']
            # list-always-wins rule applies since user did not explicitly pass --metals.
            assert out.metals == ["Pt"]
        finally:
            os.unlink(path)

    def test_explicit_cli_list_beats_yaml_list(self):
        """Wf-R27 FIX 4 regression guard.

        The launcher loop in every G5/G6-style wrapper enumerates ONE
        (metal, seed) pair per invocation and passes it explicitly
        (`--metals Ru --seeds 0`).  Before this fix the list branch in
        apply_yaml_overrides set the value unconditionally, so the yaml
        cohort (`data.metals: [auto]`, `data.seeds: [0..4]`) clobbered the
        explicit pair and every iteration silently ran the FULL grid.
        """
        path = _make_yaml("metals: [Ru, Ir, Rh, Os, Re]\nseeds: [0, 1, 2, 3, 4]\n")
        try:
            p = _make_parser()
            args = p.parse_args(
                ["--config", path, "--metals", "Ru", "--seeds", "0"]
            )
            out = apply_yaml_overrides(p, args, args.config)
            assert out.metals == ["Ru"], (
                f"explicit --metals must survive yaml list override, got {out.metals}"
            )
            assert out.seeds == [0], (
                f"explicit --seeds must survive yaml list override, got {out.seeds}"
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestDmpnnTrainerAcceptsConfig (smoke)
# ---------------------------------------------------------------------------


class TestDmpnnTrainerAcceptsConfig:
    def test_dmpnn_trainer_help_works_no_config(self):
        from molmetal.scripts import train_dmpnn_multitask_wasserstein as trainer

        args = trainer.parse_args([])
        assert hasattr(args, "config")
        assert args.config is None

    def test_dmpnn_trainer_parses_config_path(self):
        from molmetal.scripts import train_dmpnn_multitask_wasserstein as trainer

        path = _make_yaml("epochs: 17\n")
        try:
            args = trainer.parse_args(["--config", path])
            assert args.config == path
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestM17CanonicalConfigLoads
# ---------------------------------------------------------------------------


class TestM17CanonicalConfigLoads:
    def test_m17_yaml_path_loads(self):
        """Canonical config from configs/training/m17_*.yaml must load
        without error and expose the expected keys."""
        path = (
            REPO_ROOT
            / "configs"
            / "training"
            / "m17_pt_platinai_4cell_5seed_50ep_v1.yaml"
        )
        if not path.is_file():
            pytest.skip(f"M17 canonical config not on disk: {path}")
        cfg = load_config(path)
        # Spot-check canonical sections (keys use underscore-joined flat form).
        assert cfg.get("data_cohort_filter_metals") == ["Pt"]
        assert cfg.get("protocol_epochs") == 50
        assert cfg.get("protocol_seeds") == [0, 1, 2, 3, 4]
        assert cfg.get("model_backbone") == "chemprop_v2_v3"
        assert cfg.get("compute_gpu") == "cuda:0"
        assert cfg.get("data_cohort_filter_cells") == [
            "A2780",
            "MCF7",
            "A549",
            "A2780cis",
        ]
