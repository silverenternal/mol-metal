# SPDX-License-Identifier: Apache-2.0
"""A dead experiment must not be reported as a running one.

THE DEFECT THIS PINS
--------------------
``scripts/cfm_next_action.py`` decided "is this experiment still running?" by
asking whether a DONE marker was present. A crashed run and a live run both
lack that marker, so the two are indistinguishable.

MEASURED consequence on 2026-10-12: the pad_groups distributional test died of
CUDA OOM at 04:57:04, writing a full traceback into
``$CLAUDE_JOB_DIR/tmp/pad_dist.txt``. The resolver nevertheless reported
BLK-C-PERF as "in flight" for the next THIRTEEN HOURS, and every recurring
tick in that window burned a turn re-reading an advisory about a process that
no longer existed. The grouped arm had already finished (n_out 37, decoded 37,
single_frag 0.8919, n_frags 1.108); only the padded arm was missing.

The tick rules separately require reporting a dead experiment's exit code and
output tail rather than silently relaunching -- a requirement this resolver
could not satisfy for its own subject.

THE OTHER DIRECTION MATTERS MORE
--------------------------------
A live run must NOT be called dead. A false DEAD sends someone to debug a
process that is working. So the positive case (a traceback) and the negative
case (in-progress output with no marker) are both pinned.

RED CHECK: make ``_dead_run_marker`` always return ``None`` and every
``detects_*`` test fails; make it always return a dict and every
``does_not_*`` test fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESOLVER = REPO / "scripts" / "cfm_next_action.py"


def _resolver():
    if "_cfm_next_action_test" in sys.modules:
        return sys.modules["_cfm_next_action_test"]
    spec = importlib.util.spec_from_file_location(
        "_cfm_next_action_test", str(RESOLVER)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_cfm_next_action_test"] = m
    spec.loader.exec_module(m)
    return m


# The real log from the 2026-10-12 OOM, trimmed but structurally faithful:
# allocator lines, then the traceback header, then ~40 lines of frames, then
# the final exception. The timestamps sit at the TOP, far above the marker.
REAL_OOM_LOG = """[molmetal] applied general env vars
[tmqm-clouds] metals=['Ir', 'Os', 'Pt', 'Re', 'Rh', 'Ru'] 192 cloud(s)
clouds 192
SEED  path      n_out  decoded  single_frag  n_frags
0     grouped     37       37      0.8919     1.108   [1892.7s]
[W1012 04:57:04.160054747 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0 while trying to allocate 945815552 bytes (free: 435159040, total: 33766572032).
Traceback (most recent call last):
  File "/home/hugo/.claude/jobs/f12dd2b2/tmp/cfm_pad_groups_dist.py", line 70, in <module>
    n_out, d_ok, sfr, nfm = conn(path, seed)
  File "/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py", line 4767, in evaluate_generation_validity
    clouds = generate_clouds_from_noise(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 902.00 MiB.
"""


def test_detects_the_real_oom_log():
    m = _resolver()._dead_run_marker(REAL_OOM_LOG)
    assert m is not None, (
        "the resolver must recognise the pad_dist.txt OOM as terminal; it did "
        "not, and reported the run as in flight for thirteen hours"
    )
    assert "memory" in m["what"].lower()


def test_timestamp_comes_from_the_allocator_line_at_the_top():
    """The traceback is ~40 lines; a window near the exception finds nothing."""
    m = _resolver()._dead_run_marker(REAL_OOM_LOG)
    assert m["when"] == "04:57:04", (
        f"expected the allocator timestamp 04:57:04, got {m['when']!r}. A "
        "python traceback puts the timestamp ~40 lines ABOVE the exception, so "
        "the scan must start at the traceback header, not near the marker."
    )


def test_does_not_flag_a_live_run_that_merely_lacks_a_done_marker():
    live = """[molmetal] applied general env vars
clouds 192
SEED  path      n_out  decoded  single_frag  n_frags
0     grouped     37       37      0.8919     1.108   [1892.7s]
1     padded      12        9      0.7500     1.400   [41.2s]
"""
    assert _resolver()._dead_run_marker(live) is None, (
        "an in-progress run with no DONE marker must NOT be called dead; a "
        "false DEAD sends someone to debug a process that is working"
    )


def test_does_not_flag_an_empty_or_absent_log():
    assert _resolver()._dead_run_marker("") is None
    assert _resolver()._dead_run_marker("clouds 192\n") is None


def test_detects_a_plain_python_exception_without_a_cuda_marker():
    txt = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "ValueError: cannot convert float NaN to integer\n"
    )
    m = _resolver()._dead_run_marker(txt)
    assert m is not None, "a plain python traceback is terminal too"
    assert "traceback" in m["what"]


def test_done_marker_wins_over_dead_detection():
    """A completed run whose log happens to contain the word Error is not dead.

    The resolver checks DEAD before DONE, so an over-eager pattern could
    strand a finished experiment forever -- the mirror image of the original
    bug.
    """
    done_with_noise = (
        "[molmetal] applied general env vars\n"
        "error_rate on the last batch was 0.0\n"
        "DONE\n"
    )
    assert _resolver()._dead_run_marker(done_with_noise) is None, (
        "informational 'error' text in a completed log must not read as a crash"
    )


def test_withdrawn_speedup_is_never_quoted_as_current():
    """22.8x belonged to the implementation that needed 92 GiB and OOM'd.

    After the memory fix (bucketing) the runnable speedup is 4.78x. The
    resolver kept printing "22.8x speedup is measured and unaffected" as a
    standing note on the DEAD branch long after the McNemar run had produced
    a different number, so every tick repeated a figure that had been
    withdrawn.

    Every MENTION of 22.8 must sit inside a window that also carries a
    withdrawal marker. An earlier version of this test indexed lines with
    ``list.index(line)``, which returns the FIRST match and so read the wrong
    window whenever a line repeated -- the test then failed for a reason
    unrelated to what it was checking.
    """
    lines = RESOLVER.read_text(encoding="utf-8").splitlines()
    markers = ("WITHDRAWN", "withdrawn", "Do NOT quote", "belonged to")
    hits = [i for i, ln in enumerate(lines) if "22.8" in ln]
    assert hits, "the figure is gone entirely -- fine, but check the edit"
    for i in hits:
        window = "\n".join(lines[max(0, i - 8): i + 9])
        assert any(m in window for m in markers), (
            f"line {i + 1} mentions 22.8x with no withdrawal marker nearby: "
            f"{lines[i].strip()!r}"
        )


def test_settled_mcnemar_moves_perf_out_of_the_dead_branch():
    """Once the A/B has a verdict file, PERF is SETTLED -- not DEAD forever."""
    import os

    f = (REPO / "metrics" / "by_metric" / "cfm_pad_groups_mcnemar.json")
    if not f.is_file():
        import pytest
        pytest.skip("McNemar verdict not on disk yet")
    env = dict(os.environ)
    try:
        out = _resolver().resolve()
    finally:
        os.environ.clear()
        os.environ.update(env)
    ids_closed = [c["id"] for c in out.get("closed", [])]
    ids_dead = [d["id"] for d in out.get("dead", [])]
    perf_closed = any("BLK-C-PERF" in i for i in ids_closed)
    perf_dead = any("BLK-C-PERF" in i for i in ids_dead)
    assert perf_closed and not perf_dead, (
        f"with a McNemar verdict on disk PERF should be CLOSED, not DEAD: "
        f"closed={ids_closed} dead={ids_dead}"
    )


def test_real_pad_dist_log_on_this_host_is_recognised_if_present():
    """If the artefact is still on disk, the resolver must still flag it."""
    import os

    job = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "pad_dist.txt"
    if not job.is_file():
        import pytest

        pytest.skip(f"pad_dist.txt not present at {job}")
    txt = job.read_text(encoding="utf-8", errors="replace")
    if "Traceback" not in txt:
        import pytest

        pytest.skip("pad_dist.txt on this host is not a crashed run")
    assert _resolver()._dead_run_marker(txt) is not None