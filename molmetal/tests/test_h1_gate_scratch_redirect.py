"""H1 / HOLE-8 follow-up — commit-path gate writes must land in the scratch root.

Pins the redirect in ``molmetal/scripts/pre_flight_check.py``:

  * default ``out_dir`` of ``gate_wf385_scaffold_loss_wire_in_verified`` and
    ``gate_cfm_live_decode_canary`` live under ``results/gate_scratch/``
    (gitignored; L199), NOT under the source-adjacent ``molmetal/reports/``;
  * both gates FAIL (not raise) on an unusable target dir;
  * ``_build_reproducibility_manifest`` prefers the scratch hp-provenance
    JSONL and falls back to the legacy ``molmetal/reports/`` path.

CPU-only; no subprocess smoke (the FAIL cases short-circuit before any
subprocess, the PASS side is exercised by the live gate run in the wave
closeout).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PF_PATH = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


@lru_cache(maxsize=1)
def _pf():
    spec = importlib.util.spec_from_file_location("pre_flight_check_h1", PF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_defaults_are_under_scratch_root() -> None:
    pf = _pf()
    assert pf.GATE_SCRATCH_ROOT == pf.REPO_ROOT / "results" / "gate_scratch"
    for fn in (
        pf.gate_wf385_scaffold_loss_wire_in_verified,
        pf.gate_cfm_live_decode_canary,
    ):
        default = inspect.signature(fn).parameters["out_dir"].default
        assert default == pf.GATE_SCRATCH_ROOT / "wf385a3_scaffold_wire_in", (
            f"{fn.__name__}: default out_dir still points at "
            f"{default} (must be under GATE_SCRATCH_ROOT)"
        )


def test_wf385_gate_fails_on_unusable_out_dir(tmp_path: Path) -> None:
    pf = _pf()
    blocker = tmp_path / "blocker"
    blocker.write_text("regular file", encoding="utf-8")
    res = pf.gate_wf385_scaffold_loss_wire_in_verified(
        "probe", "P3", "wf385a3",
        pf.SCRIPT_DIR / "wf385a3_scaffold_wire_in_gate.py",
        out_dir=blocker / "sub",
    )
    assert res.status == "FAIL"
    assert "cannot create out-dir" in res.evidence


def test_hp_provenance_gate_fails_on_unusable_target(tmp_path: Path) -> None:
    pf = _pf()
    blocker = tmp_path / "blocker"
    blocker.write_text("regular file", encoding="utf-8")
    res = pf.gate_p1_hp_provenance_log(
        "probe", "P3", "wf350a3",
        (pf.CONFIGS_DIR / "default.yaml",),
        blocker / "sub" / "hp.jsonl",
    )
    assert res.status == "FAIL"
    assert "cannot create JSONL parent dir" in res.evidence


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "config_path": f"config-{i}.yaml",
                "current_sha": "0" * 40,
                "recent_commits": [],
                "uncommitted_changes": False,
            }) + "\n")


def test_manifest_prefers_scratch_over_legacy(tmp_path: Path, monkeypatch) -> None:
    pf = _pf()
    scratch = tmp_path / "scratch"
    legacy_root = tmp_path / "molmetal"
    _write_jsonl(scratch / "wf350a3_hp_provenance.jsonl", 1)
    _write_jsonl(legacy_root / "reports" / "wf350a3_hp_provenance.jsonl", 2)
    monkeypatch.setattr(pf, "GATE_SCRATCH_ROOT", scratch)
    monkeypatch.setattr(pf, "MOLMETAL_ROOT", legacy_root)
    manifest = pf._build_reproducibility_manifest()
    assert len(manifest["hp_provenance"]) == 1, (
        "scratch JSONL must win over the legacy molmetal/reports/ copy"
    )


def test_manifest_falls_back_to_legacy_when_scratch_absent(
    tmp_path: Path, monkeypatch,
) -> None:
    pf = _pf()
    scratch = tmp_path / "scratch"  # never created
    legacy_root = tmp_path / "molmetal"
    _write_jsonl(legacy_root / "reports" / "wf350a3_hp_provenance.jsonl", 2)
    monkeypatch.setattr(pf, "GATE_SCRATCH_ROOT", scratch)
    monkeypatch.setattr(pf, "MOLMETAL_ROOT", legacy_root)
    manifest = pf._build_reproducibility_manifest()
    assert len(manifest["hp_provenance"]) == 2, (
        "legacy molmetal/reports/ JSONL must still be read when the "
        "scratch copy does not exist"
    )
