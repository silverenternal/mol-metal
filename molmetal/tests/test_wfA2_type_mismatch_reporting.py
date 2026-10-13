"""§5.11 loader: a type-mismatched YAML value must be VISIBLE, not silent.

Context
-------
``apply_yaml_overrides`` coerces every YAML value through the argparse
action's own ``type=`` converter, so ``output.ckpt_dir`` arrives as a
``Path`` and ``protocol.lr`` as a ``float``. When that conversion *fails*
the loader kept the argparse default — deliberately, so one bad key cannot
abort a 30-epoch run — but it did so with a bare ``continue``:

    except (TypeError, ValueError):
        # Not convertible to the argparse type — keep the
        # default rather than crashing the launch.
        continue

That made a typo'd value indistinguishable from an absent key. Every OTHER
skip in the function is collected into ``skipped``, classified by
``_registered_non_trainer``, printed to stderr, and escalated to fatal under
``WFA2_CONFIG_STRICT=1``; this one path bypassed all of it. The run then
trained on a default its author never chose, silently.

Measured 2026-10-09 against ``train_cfm_yuelbond_wasserstein.py``:

    assemble_loss_weight: 0.25       -> 0.25   applied
    assemble_loss_weight: "0.25"     -> 0.25   applied (coerced)
    assemble_loss_weight: "nonsense" -> 0.0    SILENT   <- the bug
    assemble_loss_weight: null       -> 0.0    SILENT   <- the bug

These tests pin the fixed behaviour: keep the default (do not crash), but
record the skip with a reason so it reaches the stderr report, the strict
mode, and ``scripts/_config_binding_gate.py``.

Run::

    .venv/bin/python -m pytest molmetal/tests/test_wfA2_type_mismatch_reporting.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wfA2_config_loader as L  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--assemble-loss-weight", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--name", type=str, default="default-name")
    return p


def _write(tmp_path: Path, cfg: dict) -> str:
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(f)


def _apply(tmp_path: Path, cfg: dict):
    p = _parser()
    args = p.parse_args([])
    return p, L.apply_yaml_overrides(p, args, _write(tmp_path, cfg))


class TestGoodValuesStillApply:
    def test_float_applies(self, tmp_path):
        _, a = _apply(tmp_path, {"train": {"assemble_loss_weight": 0.25}})
        assert a.assemble_loss_weight == 0.25

    def test_coercible_string_applies(self, tmp_path):
        """`lr: "0.001"` is a common YAML quoting slip and must still work."""
        _, a = _apply(tmp_path, {"train": {"assemble_loss_weight": "0.25"}})
        assert a.assemble_loss_weight == 0.25

    def test_int_applies(self, tmp_path):
        _, a = _apply(tmp_path, {"protocol": {"epochs": 7}})
        assert a.epochs == 7

    def test_clean_load_records_no_skips(self, tmp_path):
        _apply(tmp_path, {"train": {"assemble_loss_weight": 0.25}})
        assert [s for s in L.last_skipped_keys() if "type mismatch" in str(s.get("reason"))] == []


class TestTypeMismatchIsReported:
    def test_nonsense_string_keeps_default_but_is_recorded(self, tmp_path):
        _, a = _apply(tmp_path, {"train": {"assemble_loss_weight": "nonsense"}})
        # Behaviour preserved: a bad key must not abort the run.
        assert a.assemble_loss_weight == 0.0
        # ...but it is no longer invisible.
        mism = [s for s in L.last_skipped_keys() if "type mismatch" in str(s.get("reason"))]
        assert len(mism) == 1
        assert mism[0]["key"] == "train.assemble_loss_weight"
        assert mism[0]["value"] == "nonsense"
        assert "float" in mism[0]["reason"]

    def test_explicit_null_is_recorded(self, tmp_path):
        _, a = _apply(tmp_path, {"train": {"assemble_loss_weight": None}})
        assert a.assemble_loss_weight == 0.0
        mism = [s for s in L.last_skipped_keys() if "type mismatch" in str(s.get("reason"))]
        assert len(mism) == 1

    def test_unresolved_config_key_is_raised_on_the_namespace(self, tmp_path):
        """Callers read `args.yaml_unresolved_keys`; the mismatch must be there."""
        _, a = _apply(tmp_path, {"train": {"epochs": "not-a-number"}})
        keys = [e["key"] for e in getattr(a, "yaml_unresolved_keys", [])]
        # The recorded key is the path as WRITTEN in the yaml, not a
        # canonicalised one — `epochs` is declared under `train:` here.
        assert "train.epochs" in keys

    def test_mismatch_is_not_excused_as_a_non_trainer_key(self, tmp_path):
        """The report must not let a type mismatch hide behind the
        `_NON_TRAINER_KEYS` classification — that would restore the silence
        by a different route."""
        _, _ = _apply(tmp_path, {"train": {"assemble_loss_weight": "nonsense"}})
        for entry in L.last_skipped_keys():
            if "type mismatch" in str(entry.get("reason")):
                assert not L._registered_non_trainer(entry)


class TestStrictModeEscalates:
    def test_strict_mode_raises_on_type_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WFA2_CONFIG_STRICT", "1")
        p = _parser()
        args = p.parse_args([])
        with pytest.raises(L.UnresolvedConfigKey):
            L.apply_yaml_overrides(p, args, _write(tmp_path, {"train": {"assemble_loss_weight": "x"}}))

    def test_strict_mode_still_allows_coercible_strings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WFA2_CONFIG_STRICT", "1")
        p = _parser()
        args = p.parse_args([])
        out = L.apply_yaml_overrides(p, args, _write(tmp_path, {"train": {"assemble_loss_weight": "0.25"}}))
        assert out.assemble_loss_weight == 0.25


class TestRepoConfigsAreClean:
    def test_no_repo_config_has_a_latent_type_mismatch(self):
        """If any shipped config trips this, it has been training on a default
        its author never wrote down — which is exactly what went unnoticed."""
        offenders = []
        for cfg in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
            try:
                import importlib.util
                import contextlib
                import io
                import re

                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                entry = data.get("entry") if isinstance(data, dict) else None
                script = entry.get("script") if isinstance(entry, dict) else None
                if not script and isinstance(entry, dict):
                    m = re.search(r"\btrain_[a-z0-9_]+\.py\b", str(entry.get("command_template") or ""))
                    script = "molmetal/scripts/" + m.group(0) if m else None
                if not script:
                    continue
                sp = REPO_ROOT / script
                if not sp.is_file():
                    continue
                spec = importlib.util.spec_from_file_location("tm_" + sp.stem, sp)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["tm_" + sp.stem] = mod
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    spec.loader.exec_module(mod)
                fn = getattr(mod, "_parse_args", None) or getattr(mod, "parse_args", None)
                if fn is None:
                    continue
                import inspect

                argv = ["--config", str(cfg)]
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    # SystemExit is a BaseException: a trainer whose parser
                    # requires a flag (`--out-dir` on the chemprop_v3 one)
                    # exits(2) before it can apply the config. That is not this
                    # test's business, so it must be swallowed explicitly.
                    try:
                        if len(inspect.signature(fn).parameters):
                            fn(argv)
                        else:
                            old = sys.argv
                            sys.argv = [str(sp)] + argv
                            try:
                                fn()
                            finally:
                                sys.argv = old
                    except BaseException:  # noqa: BLE001
                        continue
                for s in L.last_skipped_keys():
                    if "type mismatch" in str(s.get("reason", "")):
                        offenders.append(f"{cfg.name}:{s['key']}={s.get('value')!r}")
            except Exception:  # noqa: BLE001 - a config that won't load is not this test's business
                continue
        assert offenders == [], f"configs with a silent type mismatch: {offenders}"
