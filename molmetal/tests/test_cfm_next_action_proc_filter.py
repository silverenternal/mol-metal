# SPDX-License-Identifier: Apache-2.0
"""``_proc`` must not match the resolver's own tooling.

THE DEFECT THIS PINS
--------------------
``_proc(pattern)`` shells out to ``pgrep -f``, which matches ANY command line
containing the pattern -- including the resolver's OWN arguments, the cron tick
that invoked it, and any sweep launcher carrying the run name in a path.

MEASURED consequence: the L3 guidance sweep runs as

    wf_blkc_l3_guidance_probe.py --ckpt molmetal/reports/cfm_patience10_seed0/...

so a probe for ``cfm_patience10_seed0`` matched the sweep launcher and the
resolver reported a USER-STOPPED training run as "training in flight". The
user had stopped it; the resolver insisted it was running.

The fix filters ``pgrep`` PIDs through ``ps`` and drops any line carrying an
``_EXCLUDE_MARKERS`` substring (bench_*, probe, aggregate, _ab.py, dump, panel,
evaluate, audit, sweep_aggregate, cfm_next_action). That list is a heuristic,
so it is pinned from BOTH directions here: a real trainer must still be
detected, and the tooling that merely mentions the name must not be.

RED CHECK: delete the ``_EXCLUDE_MARKERS`` filter from ``_proc`` and
``test_bench_and_probe_lines_are_excluded`` fails.

Split out from ``test_cfm_next_action_resolver.py``2026-10-12 when that file
was rewritten for the DEAD-run detection and this coverage was overwritten
alongside it. Nothing was lost from the source -- only this test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESOLVER = REPO / "scripts" / "cfm_next_action.py"


def _resolver():
    if "_cfm_next_action_proc_test" in sys.modules:
        return sys.modules["_cfm_next_action_proc_test"]
    spec = importlib.util.spec_from_file_location(
        "_cfm_next_action_proc_test", str(RESOLVER)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_cfm_next_action_proc_test"] = m
    spec.loader.exec_module(m)
    return m


def test_exclude_markers_exist_and_cover_the_known_offenders():
    marks = getattr(_resolver(), "_EXCLUDE_MARKERS", ())
    assert marks, "the exclusion list vanished -- pgrep -f will self-match again"
    for needed in ("probe", "cfm_next_action", "_ab.py", "bench_"):
        assert needed in marks, f"{needed!r} missing from _EXCLUDE_MARKERS"


def test_bench_and_probe_lines_are_excluded(monkeypatch):
    """The measured false positive: a sweep launcher naming the ckpt path."""
    r = _resolver()

    # ``_proc`` calls subprocess.run TWICE: once for pgrep (PIDs) and once for
    # ps (full command lines). Patching only the first returns REAL ps output
    # -- empty -- so the filter never sees the offending line and the test
    # passes while proving nothing. An earlier version of this test had exactly
    # that defect: it went green with _EXCLUDE_MARKERS emptied.
    def _fake_out(cmd, *args, **kwargs):
        text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "pgrep" in text:
            out = "111\n222\n"
        elif text.startswith("ps"):
            # ONLY the probe. The measured situation is that the user stopped
            # the trainer and a sweep carrying the checkpoint path in --ckpt is
            # the only thing still running; an earlier version of this fixture
            # also listed the trainer, which made _proc correctly return True
            # and the test fail for the wrong reason.
            out = (
                "python molmetal/scripts/wf_blkc_l3_guidance_probe.py --ckpt "
                "molmetal/reports/cfm_patience10_seed0/checkpoints/x.pt\n"
            )
        else:
            out = ""
        return type("O", (), {"stdout": out, "returncode": 0})()

    monkeypatch.setattr(r.subprocess, "run", _fake_out)
    assert r._proc("cfm_patience10_seed0") is False, (
        "a guidance probe carrying the checkpoint path in its command line "
        "must not be read as the trainer being in flight"
    )


def test_a_real_trainer_line_is_still_detected(monkeypatch):
    r = _resolver()
    def _fake_both(cmd, *args, **kwargs):
        text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        out = (
            "111\n"
            if "pgrep" in text
            else "python molmetal/scripts/train_x.py\n"
        )
        return type("O", (), {"stdout": out, "returncode": 0})()

    monkeypatch.setattr(r.subprocess, "run", _fake_both)
    assert r._proc("train_x") is True, (
        "the exclusion filter must not swallow a genuine training process -- "
        "that would report a live run as stopped and hide real work"
    )


def test_empty_pgrep_output_is_not_in_flight():
    assert _resolver()._proc("definitely_no_such_process_xyz_12345") is False