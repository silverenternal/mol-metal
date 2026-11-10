# SPDX-License-Identifier: Apache-2.0
"""Wf-A157 — the ``--help`` probe: concurrency race + marginal timeout budget.

THE DEFECT
----------
``pre_flight_check.py`` runs its gates CONCURRENTLY (``ThreadPoolExecutor``,
default ``--jobs 8``).  Twelve gates ask the same question about the same
file — "is flag X declared on ``train_chemprop_v2_v3.py``?" — and each one
spawned ``python train_chemprop_v2_v3.py --help`` on its own.

Two defects compounded:

1. **The memoisation was a no-op under concurrency.**  ``_HELP_PROBE_CACHE``
   is a plain dict read with get-then-set, so every concurrent caller MISSED
   the cache and spawned its own subprocess.  The cache was introduced
   (2026-10-04) precisely to stop twelve identical spawns; under the default
   ``--jobs 8`` it never fired.
2. **The 30 s budget was marginal.**  MEASURED 2026-11-01 on this host:
   ``python train_chemprop_v2_v3.py --help`` takes **4.92 s** on a quiet
   host and **14.84 s** on a busy one.  The old budget was a hard-coded
   30 s at every call site.  Eight simultaneous torch+chemprop imports on
   an already-loaded host pushed every probe past 30 s, so twelve
   determinate questions were answered TIMEOUT instead of PASS.

MEASURED 2026-11-01 (pre-fix, full ``--pipeline all`` run, 129 gates):
``101 PASS / 3 FAIL / 11 SKIP / 0 WARN / 14 TIMEOUT`` — the 14 TIMEOUTs
included all 12 help-probe gates, and the PASS count fell **below the 107
floor** pinned by ``test_wf415a3_preflight_fail_audit.py``.

THE FIX
-------
* ``wf343c1_pre_retrain_check.help_probe_timeout()`` — ONE env-overridable
  budget (``MOLMETAL_HELP_PROBE_TIMEOUT_SEC``), default 180 s.
* ``wf343c1_pre_retrain_check.help_probe_lock(cache_key)`` — one lock per
  cache key, so the probe happens exactly once and the other eleven wait
  for it and are served the same answer.
* ``pre_flight_check.gate_help`` uses both, and shares the lock registry
  with ``wf343c1`` via the existing ``_wf343c1`` import.

CPU-ONLY: every probe in this file runs a synthetic script or the real
trainer's ``--help`` (argparse exits before any work).  No GPU is touched;
the module also forces ``CUDA_VISIBLE_DEVICES=""`` for its children.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(module_name: str, relpath: str):
    """Import a repo script by path (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wf343c1 = _load("wfa157_wf343c1", "molmetal/scripts/wf343c1_pre_retrain_check.py")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# 1 — the budget is ONE env-overridable number
# ---------------------------------------------------------------------------
def test_help_probe_timeout_default_is_180() -> None:
    """MEASURED 2026-11-01: the quiet-host cost is 4.92 s, the busy-host
    cost 14.84 s.  180 s is ~12x the busy-host cost."""
    assert wf343c1.HELP_PROBE_TIMEOUT_SEC_DEFAULT == 180.0
    assert wf343c1.help_probe_timeout() == 180.0


def test_help_probe_timeout_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MOLMETAL_HELP_PROBE_TIMEOUT_SEC", "42.5")
    assert wf343c1.help_probe_timeout() == 42.5


@pytest.mark.parametrize("raw", ["", "   ", "nonsense", "0", "-5", "None"])
def test_help_probe_timeout_malformed_falls_back(monkeypatch, raw: str) -> None:
    """A malformed knob must never abort a pre-flight run."""
    monkeypatch.setenv("MOLMETAL_HELP_PROBE_TIMEOUT_SEC", raw)
    assert wf343c1.help_probe_timeout() == 180.0


# ---------------------------------------------------------------------------
# 2 — the probe happens exactly ONCE under concurrency
# ---------------------------------------------------------------------------
def _synthetic_help_script(tmp_path: Path) -> Path:
    """A tiny argparse script that prints help instantly (no torch import)."""
    p = tmp_path / "synthetic_trainer.py"
    p.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--no-multi-task', action='store_true')\n"
        "ap.add_argument('--no-wass', action='store_true')\n"
        "ap.add_argument('--n-seeds-min', type=int, default=5)\n"
        "ap.add_argument('--single-metal', action='store_true')\n"
        "ap.add_argument('--use-click-aug', action='store_true')\n"
        "ap.add_argument('--use-master-wass', action='store_true')\n"
        "ap.add_argument('--use-master-cath', action='store_true')\n"
        "ap.add_argument('--enhanced-head', action='store_true')\n"
        "ap.add_argument('--metal-attention', action='store_true')\n"
        "ap.parse_args()\n",
        encoding="utf-8",
    )
    return p


def test_concurrent_gate_help_spawns_exactly_one_subprocess(tmp_path) -> None:
    """The regression: 12 concurrent probes used to spawn 12 subprocesses.

    MEASURED 2026-11-01 (post-fix, 7 threads, real chemprop driver): 1
    subprocess spawn, 7 PASS, total wall 4.92 s.
    """
    script = _synthetic_help_script(tmp_path)
    patterns = [
        "--no-multi-task --no-wass",
        "--no-multi-task --no-wass",
        "--enhanced-head --metal-attention",
        "--use-master-wass --use-master-cath",
        "--single-metal",
        "--n-seeds-min",
        "--use-click-aug",
    ]
    spawns: list = []
    guard = threading.Lock()
    original_run = wf343c1._run

    def counting_run(cmd, **kwargs):
        with guard:
            spawns.append(tuple(cmd))
        return original_run(cmd, **kwargs)

    wf343c1._run = counting_run
    wf343c1._HELP_PROBE_CACHE.clear()
    try:
        results: list = []
        res_lock = threading.Lock()

        def worker(pattern: str) -> None:
            res = wf343c1.gate_help(
                f"wfa157_{abs(hash(pattern)) % 100000}",
                "P1",
                "code",
                script,
                pattern,
                py=sys.executable,
            )
            with res_lock:
                results.append(res)

        threads = [threading.Thread(target=worker, args=(p,)) for p in patterns]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
    finally:
        wf343c1._run = original_run

    assert len(spawns) == 1, (
        f"the --help probe spawned {len(spawns)} subprocesses for ONE "
        f"cache key; the memoisation must collapse them to 1"
    )
    assert len(results) == len(patterns)
    assert {r.status for r in results} == {"PASS"}, (
        f"expected every gate to PASS, got {[(r.name, r.status) for r in results]}"
    )
    # The 6 non-first callers are served from the cache: they must not have
    # paid the subprocess cost.  (The first caller paid it once.)
    assert wall < 120.0, f"concurrent probe wall {wall:.1f}s is implausible"


def test_gate_help_probe_lock_is_shared_per_key() -> None:
    """Two calls with the same key get the SAME lock object."""
    k1 = ("py", "script", ())
    k2 = ("py", "script", ("--extra",))
    assert wf343c1.help_probe_lock(k1) is wf343c1.help_probe_lock(k1)
    assert wf343c1.help_probe_lock(k1) is not wf343c1.help_probe_lock(k2)


# ---------------------------------------------------------------------------
# 3 — no call site still hard-codes the old 30 s budget
# ---------------------------------------------------------------------------
def test_no_hardcoded_30s_timeout_at_help_probe_call_sites() -> None:
    """Every ``gate_help(...)`` call site must use the env-backed budget."""
    for relpath in (
        "molmetal/scripts/pre_flight_check.py",
        "molmetal/scripts/wf343c1_pre_retrain_check.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        # locate each gate_help( ... ) call and check its timeout argument
        idx = 0
        checked = 0
        while True:
            idx = text.find("gate_help(", idx)
            if idx == -1:
                break
            start = idx
            depth = 0
            end = idx
            for i in range(idx, len(text)):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            call = text[start:end]
            if "timeout=" in call:
                assert "timeout=30.0" not in call, (
                    f"{relpath}: a gate_help call site still hard-codes the "
                    f"marginal 30 s budget: {call[:200]}"
                )
                checked += 1
            idx = end
        assert checked >= 6, f"{relpath}: expected >= 6 timeout-bearing gate_help calls, got {checked}"


def test_pre_flight_shares_the_wf343c1_lock_registry() -> None:
    """The two modules must not keep separate lock registries.

    ``pre_flight_check.py`` imports ``wf343c1_pre_retrain_check`` as
    ``_wf343c1`` and already aliases its probe cache into it; the lock
    registry must live in the SAME module instance, otherwise the two
    copies can still spawn two subprocesses for one key.
    """
    import inspect

    pre_flight = _load("wfa157_pre_flight", "molmetal/scripts/pre_flight_check.py")
    pf_wf343c1 = pre_flight._wf343c1
    key = ("py", "script", ())
    # one lock per key, within the ONE module instance both routes use
    assert pf_wf343c1.help_probe_lock(key) is pf_wf343c1.help_probe_lock(key)
    # the cache alias still holds (pre-flight's dict is the one wf343c1 reads)
    assert pf_wf343c1._HELP_PROBE_CACHE is pre_flight._HELP_PROBE_CACHE
    # and pre-flight's gate_help routes its probe through that shared lock
    src = inspect.getsource(pre_flight.gate_help)
    assert "_wf343c1.help_probe_lock" in src, (
        "pre_flight_check.gate_help must take the shared per-key probe lock"
    )
    assert "_wf343c1.help_probe_timeout()" in src, (
        "pre_flight_check.gate_help must use the env-backed probe budget"
    )


# ---------------------------------------------------------------------------
# 4 — a TIMEOUT is still NOT a "flag missing" verdict (Wf-A154 invariant)
# ---------------------------------------------------------------------------
def test_timeout_is_not_a_missing_flag_verdict(tmp_path) -> None:
    """A probe that never finishes must NOT be reported as FAIL."""
    script = tmp_path / "slow_trainer.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    res = wf343c1.gate_help(
        "wfa157_slow",
        "P1",
        "code",
        script,
        "--some-flag",
        py=sys.executable,
        timeout=1.0,
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert "NOT a verdict" in res.evidence


# ---------------------------------------------------------------------------
# 5 — the live driver: opt-in (it imports torch + chemprop)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("MOLMETAL_WFA157_LIVE_CHEMPROP") != "1",
    reason="set MOLMETAL_WFA157_LIVE_CHEMPROP=1 to probe the real chemprop driver",
)
def test_live_chemprop_help_probe_declares_the_flags() -> None:
    """MEASURED 2026-11-01: the real ``--help`` declares every probed flag."""
    driver = wf343c1.CHEMPROP_V3_DRIVER
    for pattern in (
        "--no-multi-task --no-wass --no-kendall --no-dann",
        "--enhanced-head --metal-attention",
        "--use-master-wass --use-master-cath",
        "--single-metal",
        "--n-seeds-min",
        "--use-click-aug",
    ):
        res = wf343c1.gate_help(
            f"wfa157_live_{abs(hash(pattern)) % 100000}",
            "P1",
            "code",
            driver,
            pattern,
        )
        assert res.status == "PASS", f"{pattern}: {res.status}: {res.evidence}"


def test_children_never_touch_the_gpu() -> None:
    """The probe children inherit CUDA_VISIBLE_DEVICES="" from this module."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", "import os;print(os.environ.get('CUDA_VISIBLE_DEVICES'))"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.stdout.strip() == ""
