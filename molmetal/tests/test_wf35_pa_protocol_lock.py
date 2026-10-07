"""Wf-R35 P-A — tests for configs/locks/protocol_lock.yaml + verify_protocol_lock.

CPU-only, no training path touched.  6 tests:

  1. the lock parses and carries the 5 Tier-1 metric keys
  2. every md5 in the lock matches the artifact actually on disk
  3. verify_protocol_lock PASSes for a metric whose trainer config names a
     locked dataset (m1 -> configs/dmpnn.yaml)
  4. verify_protocol_lock FAILs with a diff when the config names no dataset
     (negative case — m1 -> configs/cfm.yaml)
  5. verify_protocol_lock FAILs for an unknown metric key, without raising
  6. the 5 gates are registered on "all" and SKIP by default (opt-in
     contract, AGENTS.md §5.6/§5.7)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCK = REPO_ROOT / "configs" / "locks" / "protocol_lock.yaml"  # noqa: E402
METRIC_KEYS = (  # noqa: E402
    "m1_ru_auc",
    "m2_r2_per_cell",
    "m17_drugood_ood",
    "m19_lomo",
    "m21_nci60_9cell",
)


def _lock() -> dict:
    import yaml

    return yaml.safe_load(LOCK.read_text(encoding="utf-8"))


def test_lock_parses_with_5_tier1_metrics() -> None:
    """The lock carries exactly the 5 Tier-1 metrics it claims to lock."""
    assert LOCK.is_file(), f"lock missing: {LOCK}"
    metrics = _lock()["metrics"]
    assert sorted(metrics) == sorted(METRIC_KEYS)
    for key, entry in metrics.items():
        assert entry.get("trainer_config"), f"{key}: no trainer_config"
        assert entry.get("split") is not None, f"{key}: no split"
        assert entry.get("n_folds") is not None, f"{key}: no n_folds"
        assert entry.get("alignment_level"), f"{key}: no alignment_level"


def test_every_locked_md5_matches_disk() -> None:
    """Every accepted_dataset_md5 is MEASURED against the file on disk.

    This is the anti-drift property: if someone regenerates a cohort CSV
    without re-locking it, this test goes RED before a retrain runs.
    """
    for key, entry in _lock()["metrics"].items():
        for item in entry["accepted_dataset_md5"]:
            path = Path(item["path"])
            assert path.is_file(), f"{key}: locked dataset missing: {path}"
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            assert h.hexdigest() == item["md5"], (
                f"{key}: md5 drift on {path} — lock says {item['md5']}, "
                f"disk says {h.hexdigest()}. Re-lock or revert the dataset."
            )


def test_verify_protocol_lock_passes_m1_dmpnn() -> None:
    """Positive case: the D-MPNN config names a locked m1 cohort."""
    from molmetal.scripts.pre_flight_check import verify_protocol_lock

    result = verify_protocol_lock("m1_ru_auc", "configs/dmpnn.yaml")
    assert result["status"] == "PASS", result["evidence"]
    md5_check = next(
        c for c in result["checks"] if c["check"] == "dataset_md5"
    )
    assert md5_check["status"] == "PASS"


def test_verify_protocol_lock_fails_on_dataset_free_config() -> None:
    """Negative case: a config naming no cohort FAILs rather than vacuously passing."""
    from molmetal.scripts.pre_flight_check import verify_protocol_lock

    result = verify_protocol_lock("m1_ru_auc", "configs/cfm.yaml")
    assert result["status"] == "FAIL", result["evidence"]
    md5_check = next(
        c for c in result["checks"] if c["check"] == "dataset_md5"
    )
    assert md5_check["status"] == "FAIL"
    assert "NO dataset artifact" in md5_check["detail"]


def test_verify_protocol_lock_unknown_metric_is_fail_not_raise() -> None:
    """An unknown metric key is a FAIL result, not a crash."""
    from molmetal.scripts.pre_flight_check import verify_protocol_lock

    result = verify_protocol_lock("m99_does_not_exist", "configs/dmpnn.yaml")
    assert result["status"] == "FAIL"
    assert any(c["check"] == "lock_loads" for c in result["checks"])


def test_five_gates_registered_and_skip_by_default(monkeypatch) -> None:
    """The 5 gates are visible on `all` but SKIP unless opted in."""
    from molmetal.scripts.pre_flight_check import build_gates

    monkeypatch.delenv("WF35_PROTOCOL_LOCK", raising=False)
    names = [n for n, _ in build_gates("all") if n.startswith("p3_wf35_protocol_lock_")]
    assert len(names) == 5, f"expected 5 protocol-lock gates, got {names}"
    runners = dict(build_gates("all"))
    for name in names:
        assert runners[name]().status == "SKIP", f"{name} did not SKIP by default"

    monkeypatch.setenv("WF35_PROTOCOL_LOCK", "1")
    statuses = {n: runners[n]().status for n in names}
    assert "PASS" in statuses.values(), (
        f"opted-in run produced no PASS: {statuses}. If every metric FAILs the "
        f"lock is not tracking any real trainer config."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
