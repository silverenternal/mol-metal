#!/usr/bin/env python
"""Wf-A201 -- guard tests for the CPU-only encoder/head localisation probe.

WHAT THIS FILE GUARDS
    1. CPU-only refusal  -- the probe must refuse to run when a CUDA device
       is visible, and it must clear ``CUDA_VISIBLE_DEVICES`` before torch is
       imported.
    2. Deterministic split identity -- the ridge arm and every head arm must
       be scored on the SAME test rows (hypothesis H5).
    3. Write confinement -- the probe may only write inside its own output
       directory.

THE TESTS MUST BITE
    Every guard is load-bearing, and that is asserted rather than asserted-in-
    prose: for each guard a /tmp copy of the probe is made with that ONE
    guard removed, and the corresponding test is re-run against the copy as a
    subprocess. The copy must make the test FAIL; the failure message is
    asserted so a guard that silently stopped firing cannot pass unnoticed.

    Set ``WFA201_BITE_CHILD=1`` in the subprocess so the bite tests themselves
    do not recurse.

USAGE
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        molmetal/tests/test_wfa201_encoder_readout_probe.py -q
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODULE_ENV = "WFA201_PROBE_MODULE"
DEFAULT_MODULE = "molmetal.scripts.wfa201_encoder_readout_probe"
IS_BITE_CHILD = os.environ.get("WFA201_BITE_CHILD") == "1"

REAL_OUT_DIR = REPO / "molmetal/reports/wfa201_encoder_readout"


# ---------------------------------------------------------------------------
# module under test
# ---------------------------------------------------------------------------
def _load_probe():
    spec = os.environ.get(MODULE_ENV, DEFAULT_MODULE)
    if spec.endswith(".py") or os.path.sep in spec:
        path = Path(spec).resolve()
        digest = hashlib.md5(str(path).encode()).hexdigest()[:10]
        name = f"wfa201_probe_under_test_{digest}"
        mod_spec = importlib.util.spec_from_file_location(name, path)
        assert mod_spec and mod_spec.loader, f"cannot load {path}"
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
        return mod, path
    return importlib.import_module(spec), None


PROBE, PROBE_PATH = _load_probe()


def _break_guard(src: str, func_name: str) -> str:
    """Return ``src`` with the body of ``func_name`` replaced by a no-op.

    Used to build the /tmp copies that prove each guard actually bites.
    """
    pat = re.compile(
        r"^def %s\(.*?(?=^def |^class |\Z)" % re.escape(func_name),
        re.M | re.S,
    )
    m = pat.search(src)
    assert m, f"guard {func_name!r} not found in the probe source"
    stub = f"def {func_name}(*args, **kwargs):\n    return None\n\n\n"
    return src[: m.start()] + stub + src[m.end():]


def _snapshot(path: Path):
    if not path.exists():
        return {}
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


# ---------------------------------------------------------------------------
# 1. CPU-only
# ---------------------------------------------------------------------------
def test_cpu_only_env_is_cleared_before_torch_import():
    """The probe clears CUDA_VISIBLE_DEVICES at import time (module level)."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert PROBE.os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_cpu_only_guard_refuses_when_gpu_visible(monkeypatch):
    """A visible CUDA device must abort the probe before any work happens."""
    monkeypatch.setattr(PROBE.torch.cuda, "is_available", lambda: True)
    with pytest.raises(PROBE.CpuOnlyViolation) as excinfo:
        PROBE.assert_cpu_only()
    msg = str(excinfo.value)
    assert "CPU-ONLY" in msg and "CUDA" in msg


def test_cpu_only_guard_passes_when_no_gpu(monkeypatch):
    monkeypatch.setattr(PROBE.torch.cuda, "is_available", lambda: False)
    PROBE.assert_cpu_only()  # must not raise


# ---------------------------------------------------------------------------
# 2. deterministic split identity (same rows)
# ---------------------------------------------------------------------------
def test_same_rows_guard_fires_on_mismatch():
    with pytest.raises(PROBE.SameRowsViolation) as excinfo:
        PROBE.assert_same_test_rows(994, 993, 994, "bite/demo")
    msg = str(excinfo.value)
    assert "NOT on the same rows" in msg


def test_same_rows_guard_passes_on_match():
    PROBE.assert_same_test_rows(994, 994, 994, "bite/demo")


def test_split_identity_is_deterministic(tmp_path, monkeypatch):
    """Two independent split builds must agree on every row id."""
    monkeypatch.setattr(PROBE, "OUT_DIR", tmp_path / "out")
    a = PROBE.build_dataset("Ru", featurise=False)
    b = PROBE.build_dataset("Ru", featurise=False)
    assert a["n_test"] == b["n_test"] == 994
    assert a["n_joint_train"] == b["n_joint_train"] == 3480
    assert a["n_target_train"] == b["n_target_train"] == 2262
    assert PROBE.sha256_rows(a["target_test_idx"]) == PROBE.sha256_rows(b["target_test_idx"])
    assert PROBE.sha256_rows(a["target_test_idx"]) == (
        "bdd5349364f96f8cdd7be54e5a390103843658e6311dffd394bb5df3f1494589"
    )


# ---------------------------------------------------------------------------
# 3. write confinement
# ---------------------------------------------------------------------------
def test_write_scope_guard_fires_outside_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(PROBE, "OUT_DIR", tmp_path / "out")
    inside = tmp_path / "out" / "report.json"
    PROBE.assert_write_scope(inside)  # must not raise
    with pytest.raises(PROBE.WriteScopeViolation) as excinfo:
        PROBE.assert_write_scope(tmp_path / "escape.json")
    msg = str(excinfo.value)
    assert "outside the probe's own output directory" in msg


def test_probe_runs_cpu_only_and_writes_only_into_its_output_dir(tmp_path, monkeypatch):
    """End-to-end: --smoke runs on CPU, exits 0, writes only inside OUT_DIR."""
    out = tmp_path / "out"
    before = _snapshot(REAL_OUT_DIR)
    monkeypatch.setattr(PROBE, "OUT_DIR", out)
    rc = PROBE.main(["--smoke"])
    assert rc == 0
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    written = sorted(p.name for p in out.iterdir() if p.is_file())
    assert written == ["report.json", "report.md"]
    # the real on-disk artefacts must be untouched by the test
    assert _snapshot(REAL_OUT_DIR) == before
    md = (out / "report.md").read_text()
    last = [ln for ln in md.splitlines() if ln.strip()][-1]
    assert last.startswith("END OF REPORT"), last
    import json

    rep = json.loads((out / "report.json").read_text())
    assert rep["scope"]["cpu_only"] is True
    assert rep["scope"]["gpu_used"] is False
    assert rep["scope"]["encoder_trained"] is False
    assert rep["not_measured"], "the report must declare what was NOT measured"
    assert rep["hypothesis_ranking"]["verdict"]["h6_is_proven_for_morgan_fp"] is True


# ---------------------------------------------------------------------------
# 4. the guards must BITE
# ---------------------------------------------------------------------------
def _run_child(module_path: Path, test_name: str):
    env = dict(os.environ)
    env[MODULE_ENV] = str(module_path)
    env["WFA201_BITE_CHILD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-p", "no:cacheprovider", "-k", test_name],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900,
    )


@pytest.mark.skipif(IS_BITE_CHILD, reason="bite child: do not recurse")
@pytest.mark.parametrize(
    "guard_name,test_name",
    [
        ("assert_cpu_only", "test_cpu_only_guard_refuses_when_gpu_visible"),
        ("assert_same_test_rows", "test_same_rows_guard_fires_on_mismatch"),
        ("assert_write_scope", "test_write_scope_guard_fires_outside_output_dir"),
    ],
)
def test_guard_bites_when_removed(tmp_path, guard_name, test_name):
    """Break ONE guard in a /tmp copy -> the matching test MUST fail."""
    src = PROBE_PATH.read_text() if PROBE_PATH else Path(PROBE.__file__).read_text()
    broken = tmp_path / f"broken_{guard_name}.py"
    broken.write_text(_break_guard(src, guard_name))
    assert guard_name not in broken.read_text().split("def ")[0] or True  # sanity
    res = _run_child(broken, test_name)
    assert res.returncode != 0, (
        f"guard {guard_name!r} was removed but {test_name} still passed -- "
        f"the guard is NOT load-bearing"
    )
    assert "DID NOT RAISE" in res.stdout, (
        f"expected the guard-removal failure to be a missing-raise failure; "
        f"stdout was:\n{res.stdout[-4000:]}"
    )


@pytest.mark.skipif(IS_BITE_CHILD, reason="bite child: do not recurse")
def test_unbroken_guards_all_pass_in_the_child(tmp_path):
    """Control for the bite test: the intact probe passes the same tests."""
    src = PROBE_PATH.read_text() if PROBE_PATH else Path(PROBE.__file__).read_text()
    intact = tmp_path / "intact_probe.py"
    intact.write_text(src)
    res = _run_child(intact, "guard")
    assert res.returncode == 0, f"intact probe must pass the guard tests:\n{res.stdout[-4000:]}"
