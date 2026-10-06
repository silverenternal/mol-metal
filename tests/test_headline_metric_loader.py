"""Regression test for ``molmetal.scripts.headline_metric_loader``.

Wf-A11 (2026-10-05) — proves the gate works against the current disqualified
headline JSON and against a fabricated-usable JSON.

Three invariants under test:

  I1. ``load_headline_metric(as_headline=False)`` returns the payload
      regardless of USABLE_AS_HEADLINE (read-only path is unblocked).
  I2. ``load_headline_metric(as_headline=True)`` against a JSON with
      ``USABLE_AS_HEADLINE: false`` and disqualifying_reasons populated
      RAISES :class:`HeadlineNotUsableError` carrying the reasons.
  I3. ``load_headline_metric(as_headline=True, allow_disqualified=True)``
      returns the payload (explicit opt-in path; the caller owns honest
      framing per AGENTS.md §5.2).

The tests use the REAL ``metrics/by_metric/metal_lomo_auc_wasserstein.json``
(I1 + I2 + I3 on the on-disk disqualified JSON), plus a tiny synthetic
JSON to prove the inverse case (``USABLE_AS_HEADLINE: true`` returns
cleanly). CPU-only, no torch, no sklearn, no network.

Run: pytest tests/test_headline_metric_loader.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE = (
    REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
)
LOADER_PY = REPO_ROOT / "molmetal" / "scripts" / "headline_metric_loader.py"


def _load_loader_module():
    """Load headline_metric_loader.py without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "headline_metric_loader", LOADER_PY
    )
    module = importlib.util.module_from_spec(spec)
    # molmetal/scripts/ has no __init__.py; add repo root so the module's
    # own absolute-path imports still resolve if any.
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loader():
    return _load_loader_module()


@pytest.fixture(scope="module")
def headline_payload():
    assert HEADLINE.exists(), (
        f"Headline JSON missing: {HEADLINE}. Run the Wf-A11 headline update."
    )
    with HEADLINE.open() as f:
        return json.load(f)


def test_headline_json_declares_not_usable(headline_payload):
    """Sanity: the on-disk JSON is disqualified (USABLE_AS_HEADLINE: false)
    with a populated disqualifying_reasons list. If this fails, somebody
    re-ran the headline with --rh-oversample=0 and updated it — drop the
    gate test (it would now correctly pass-through)."""
    assert headline_payload["USABLE_AS_HEADLINE"] is False
    reasons = headline_payload["disqualifying_reasons"]
    assert isinstance(reasons, list) and len(reasons) >= 1
    # Must mention oversample and n_folds<5
    joined = " ".join(reasons).lower()
    assert "oversample" in joined or "synthetic" in joined
    assert "n_folds" in joined or "5-metal" in joined or "2-metal" in joined


def test_headline_has_clean_re_measurement_command(headline_payload):
    """The clean re-measurement command must be present and point at the
    canonical venv python."""
    cmd = headline_payload["clean_re_measurement_command"]
    assert isinstance(cmd, str) and len(cmd) > 0
    assert "/home/hugo/codes/mol-metal/.venv/bin/python" in cmd
    assert "--rh-oversample 0" in cmd
    assert "--metals" in cmd


def test_I1_read_only_path_unblocked(loader, headline_payload):
    """as_headline=False must always return the payload."""
    out = loader.load_headline_metric(HEADLINE)
    assert out["metric"] == headline_payload["metric"]
    assert out["value"] == headline_payload["value"]


def test_I2_headline_path_raises_with_reasons(loader):
    """as_headline=True against the disqualified JSON must raise, carrying
    the disqualifying_reasons in the exception."""
    with pytest.raises(loader.HeadlineNotUsableError) as excinfo:
        loader.load_headline_metric(HEADLINE, as_headline=True)
    err = excinfo.value
    assert err.path == str(HEADLINE)
    assert isinstance(err.reasons, list) and len(err.reasons) >= 1
    assert err.value == 0.7576516768181095  # the headline value
    # The error message must surface the reasons so callers can render them.
    msg = str(err)
    assert "USABLE_AS_HEADLINE" in msg or any(
        r.split()[0] in msg for r in err.reasons
    )


def test_I3_allow_disqualified_returns_payload(loader):
    """allow_disqualified=True must return the payload (caller owns framing)."""
    out = loader.load_headline_metric(
        HEADLINE, as_headline=True, allow_disqualified=True
    )
    assert out["USABLE_AS_HEADLINE"] is False
    assert "disqualifying_reasons" in out
    assert out["value"] == 0.7576516768181095


def test_usable_json_passes_clean(loader, tmp_path):
    """A synthetic JSON with USABLE_AS_HEADLINE: true must pass through
    as_headline=True cleanly — proves the gate is not a one-way refusal."""
    synthetic = {
        "metric": "synthetic_headline_for_test",
        "value": 0.9,
        "USABLE_AS_HEADLINE": True,
        "disqualifying_reasons": [],
    }
    p = tmp_path / "synthetic_headline.json"
    p.write_text(json.dumps(synthetic))
    out = loader.load_headline_metric(p, as_headline=True)
    assert out["metric"] == "synthetic_headline_for_test"
    assert out["value"] == 0.9


def test_missing_file_raises_filenotfound(loader, tmp_path):
    """FileNotFoundError for a missing path — sanity."""
    with pytest.raises(FileNotFoundError):
        loader.load_headline_metric(tmp_path / "does_not_exist.json")


def test_loader_exports_symbols(loader):
    """Module surface — both public symbols are exported."""
    assert hasattr(loader, "load_headline_metric")
    assert hasattr(loader, "HeadlineNotUsableError")
    assert issubclass(loader.HeadlineNotUsableError, RuntimeError)
