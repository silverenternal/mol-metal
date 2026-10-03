"""Wf-233 — verify the "强者默认路由倒真实测试" directive is wired.

DIRECTIVE 2026-09-28: "强者默认路由倒真实测试" — strong (production-scale)
measurements default-route to real test data. Use ``--smoke`` (or set
``data.dataset=metalcytotox_fallback`` explicitly) only for opt-in smoke
runs.

This test pins three contracts:

1. The default config (``configs/default.yaml``) routes to a real dataset
   (``qm9``) — never to a synthetic fallback (``metalcytotox_fallback``)
   that masquerades as production.
2. :func:`scripts.train.build_dataset` honours that default: when called
   without overrides it constructs a real :class:`QM9Dataset` instance.
3. Backward compatibility — :func:`scripts.train.build_dataset` still
   recognises ``metalcytotox`` (the explicit opt-in smoke flag) so the
   existing ``smoke_chemprop_v2_mt_100.py`` script keeps working.

It does NOT exercise the training loop itself — that is the Wf-233
real-test smoke (verified separately via the ``verify_real_smoke.py``
script in ``molmetal/reports/wf233/``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train import build_dataset
from utils.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _default_cfg() -> dict:
    """Load the default config and assert the file exists."""
    assert DEFAULT_CONFIG.exists(), (
        f"default config missing at {DEFAULT_CONFIG}; cannot verify "
        f"'强者默认路由倒真实测试' routing."
    )
    return load_config(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# 1. The default config itself routes to real data.
# ---------------------------------------------------------------------------
def test_default_dataset_is_real_not_synthetic_fallback():
    """The default config must not route to a synthetic fallback.

    A "强者默认路由倒真实测试" production run must consume real test
    data; routing to ``metalcytotox_fallback`` would silently replace
    real measurements with synthetic ones.
    """
    cfg = _default_cfg()
    data_cfg = cfg.get("data", {}) or {}
    dataset_name = str(data_cfg.get("dataset", "")).lower()

    assert dataset_name != "", (
        "default config has empty data.dataset — production runs "
        "would crash on the unknown-dataset branch in build_dataset()."
    )
    assert dataset_name != "metalcytotox_fallback", (
        f"default config still routes to synthetic fallback "
        f"({dataset_name!r}); production runs would consume synthetic "
        f"data instead of real test data. Restore the routing fix."
    )
    # Real datasets recognised by build_dataset (qm9 is the canonical
    # production default; metalcytotox is the opt-in smoke dataset).
    assert dataset_name in {"qm9", "metalcytotox", "metalcytotoxdb"}, (
        f"default config data.dataset={dataset_name!r} is not one of "
        f"the recognised real datasets ('qm9', 'metalcytotox')."
    )


def test_default_dataset_is_qm9_by_directive():
    """The directive pins qm9 specifically (real 3D QM9 molecules).

    ``metalcytotox`` is also real (Krasnov 2026 MetalCytoToxDB), but the
    directive's chosen production default is ``qm9`` because it carries
    explicit 3D coordinates — the flow-matching task requires real
    geometry, not the descriptor-only MetalCytoToxDB rows.
    """
    cfg = _default_cfg()
    dataset_name = str(cfg.get("data", {}).get("dataset", "")).lower()
    assert dataset_name == "qm9", (
        f"default config data.dataset={dataset_name!r}; expected 'qm9' "
        f"per the 强者默认路由倒真实测试 directive."
    )


# ---------------------------------------------------------------------------
# 2. build_dataset honours the default — no override means real dataset.
# ---------------------------------------------------------------------------
def test_build_dataset_default_constructs_real_dataset():
    """Calling ``build_dataset`` with no override must yield a real dataset.

    Skips automatically when QM9 data has not been downloaded (the test
    then has no way to verify "real" beyond inspecting the dataset
    class — which is already covered by the config tests above).
    """
    cfg = _default_cfg()
    dataset, _target_index = build_dataset(cfg)

    # The dataset class must be the real QM9Dataset (or fall through to
    # MetalCytoToxDataset when the operator explicitly opted in).
    type_name = type(dataset).__name__
    assert type_name in {"QM9Dataset", "MetalCytoToxDataset"}, (
        f"build_dataset() produced unexpected class {type_name!r}; "
        f"default routing is broken."
    )

    # And the class name must match the config it was built from — the
    # only way to silently fall through to a synthetic fallback would
    # be a mismatch between the config's declared dataset and what
    # build_dataset actually constructed.
    declared = str(cfg.get("data", {}).get("dataset", "")).lower()
    if declared == "qm9":
        assert type_name == "QM9Dataset", (
            f"default config declares qm9 but build_dataset produced "
            f"{type_name!r}."
        )
    elif declared in {"metalcytotox", "metalcytotoxdb"}:
        assert type_name == "MetalCytoToxDataset", (
            f"default config declares metalcytotox but build_dataset "
            f"produced {type_name!r}."
        )


# ---------------------------------------------------------------------------
# 3. Backward compatibility — the opt-in smoke dataset still resolves.
# ---------------------------------------------------------------------------
def test_build_dataset_smoke_override_still_works():
    """Explicit ``metalcytotox`` opt-in must still build a MetalCytoToxDB dataset.

    Operators who want a fast smoke run can still pass
    ``data.dataset=metalcytotox`` (or the ``--smoke`` flag in their
    wrapper script); this test pins that the override path keeps
    working — only the *default* has changed.
    """
    cfg = _default_cfg()
    cfg = {**cfg, "data": {**cfg.get("data", {}), "dataset": "metalcytotox"}}
    dataset, _ = build_dataset(cfg)

    type_name = type(dataset).__name__
    assert type_name == "MetalCytoToxDataset", (
        f"override data.dataset='metalcytotox' must produce a "
        f"MetalCytoToxDataset; got {type_name!r}."
    )


# ---------------------------------------------------------------------------
# 4. The directive header is present in the default config.
# ---------------------------------------------------------------------------
def test_default_config_documents_the_directive():
    """The directive must be recorded at the top of the config file.

    This is the inline-traceability equivalent of a code comment: when
    a future operator wonders *why* the default is ``qm9`` instead of a
    synthetic fallback, the answer is in the file itself.
    """
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert "DIRECTIVE 2026-09-28" in text, (
        f"default config {DEFAULT_CONFIG} is missing the "
        f"'DIRECTIVE 2026-09-28' header that documents the routing "
        f"decision. Restore the directive header."
    )
    # The Chinese directive phrase must be quoted verbatim in the file.
    assert "强者默认路由倒真实测试" in text, (
        f"default config {DEFAULT_CONFIG} is missing the directive "
        f"phrase '强者默认路由倒真实测试'. Restore it verbatim."
    )
