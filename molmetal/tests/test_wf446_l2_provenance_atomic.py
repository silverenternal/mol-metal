"""L2 — hp-provenance JSONL write routes through the §5.6 atomic primitive.

Pins the write path of ``gate_p1_hp_provenance_log`` in
``molmetal/scripts/pre_flight_check.py``:

  * the artefact is written via ``atomic_write_file`` (Wf-386 §5.6
    temp-file + rename), NOT a truncating ``open(w)``;
  * an unwritable target FAILs the gate (it must never raise and crash
    the pre-flight run) — proven with a real chmod-0500 dir AND with the
    primitive monkeypatched to raise;
  * the write is a whole-file replace, never an append: running the gate
    twice on the same path leaves exactly N_lines == N_configs (an
    append would accumulate stale-SHA records across runs).

CPU-only; each gate call shells out to `git` (3 probes per config), so
the config lists below are kept at 1-2 files.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PF_PATH = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


@lru_cache(maxsize=1)
def _pf():
    spec = importlib.util.spec_from_file_location("pre_flight_check_l2", PF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_write_fails_on_unwritable_parent_dir(tmp_path: Path) -> None:
    """Real constructed bad input: chmod-0500 parent -> FAIL, no raise."""
    if os.geteuid() == 0:
        pytest.skip("permission-bit FAIL proof is meaningless as root")
    pf = _pf()
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o500)
    try:
        res = pf.gate_p1_hp_provenance_log(
            "probe", "P3", "wf350a3",
            (pf.CONFIGS_DIR / "default.yaml",),
            ro_dir / "hp.jsonl",
        )
    finally:
        os.chmod(ro_dir, 0o700)  # allow pytest tmp cleanup
    assert res.status == "FAIL", (
        f"unwritable target must FAIL the gate, got {res.status}: {res.evidence}"
    )
    assert "failed to write" in res.evidence, res.evidence


@pytest.mark.parametrize("exc_type", [OSError, RuntimeError])
def test_write_failure_of_primitive_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc_type: type
) -> None:
    """Primitive raises -> gate FAILs (never raises), evidence names the write."""
    pf = _pf()

    def _boom(*args: object, **kwargs: object) -> None:
        raise exc_type("simulated write failure")

    monkeypatch.setattr(pf, "atomic_write_file", _boom)
    res = pf.gate_p1_hp_provenance_log(
        "probe", "P3", "wf350a3",
        (pf.CONFIGS_DIR / "default.yaml",),
        tmp_path / "hp.jsonl",
    )
    assert res.status == "FAIL", (
        f"raise from atomic_write_file must FAIL the gate, got {res.status}"
    )
    assert "failed to write" in res.evidence and exc_type.__name__ in res.evidence, (
        res.evidence
    )
    assert not (tmp_path / "hp.jsonl").exists(), (
        "gate must not leave a partial artefact after a write failure"
    )


def test_write_is_whole_file_replace_not_append(tmp_path: Path) -> None:
    """Run 1 (1 config) then run 2 (2 configs): file holds exactly 2 lines."""
    pf = _pf()
    jsonl_path = tmp_path / "hp.jsonl"
    configs = (
        pf.CONFIGS_DIR / "default.yaml",
        pf.CONFIGS_DIR / "cfm_full_retrain.yaml",
    )
    res1 = pf.gate_p1_hp_provenance_log(
        "probe", "P3", "wf350a3", configs[:1], jsonl_path,
    )
    assert res1.status == "PASS", res1.evidence
    res2 = pf.gate_p1_hp_provenance_log(
        "probe", "P3", "wf350a3", configs[:2], jsonl_path,
    )
    assert res2.status == "PASS", res2.evidence
    lines = [
        line for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2, (
        f"whole-file replace must leave 2 records after run 2 (got "
        f"{len(lines)}); an append-based write would leave 3 stale lines"
    )
