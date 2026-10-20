#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Resolve the next actionable CFM step from cfm_todo.json + live process state.

WHY THIS EXISTS
---------------
Background experiments finish at unpredictable times. A human-facing "is it
done yet?" prompt every few minutes wastes a session's worth of tokens on
polling, and polling is exactly what AGENTS.md 5.8 tells us not to do. This
script answers the only question worth asking -- "given what is finished, what
is the next unblocked step?" -- and answers it from disk, so the answer costs
one cheap process launch rather than one agent turn.

It READS ONLY. It never launches training, never commits, never edits a
metric JSON. Deciding is cheap; acting stays with the agent (and with the user,
for anything that spends GPU or rewrites a record).

USAGE
    python scripts/cfm_next_action.py          # human-readable
    python scripts/cfm_next_action.py --json   # machine-readable
    python scripts/cfm_next_action.py --quiet  # exit code only (0 = work ready)

EXIT CODES
    0  a next step is ready and unblocked
    1  nothing ready -- waiting on a live experiment
    2  the state on disk is unreadable (say so; do not guess)
"""

from __future__ import annotations

import argparse
import re
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Wf-A100 (2026-10-20) — ONE copy of the McNemar settlement text.  It used to
#: exist only inside the loop over the ephemeral equiv*.txt logs, which is why
#: the settlement disappeared with /tmp.  Both the permanent-artefact check and
#: the legacy log branch read this constant, so the two can never disagree
#: about what was measured.
_MCNEMAR_SETTLED_TEXT = (
    "SETTLED by the McNemar A/B "
    "(metrics/by_metric/cfm_pad_groups_mcnemar.json): "
    "4.78x speedup, McNemar exact p=0.0769 with 63 of 192 "
    "clouds flipping -- net difference NOT established, but "
    "the churn is far too large to call the two paths "
    "equivalent on one seed. The 22.8x figure is WITHDRAWN: "
    "it came from the 92 GiB single-batch implementation "
    "that could not run."
)
TODO = REPO / "cfm_todo.json"


#: Command-line markers of a process that is READING a checkpoint rather than
#: TRAINING one. A training invocation is ``train_cfm_yuelbond_wasserstein.py``
#: (or ``scripts/launch_*.sh``); everything below measures, decodes, scores or
#: sweeps and must never be mistaken for a run in progress.
_EXCLUDE_MARKERS = (
    "bench_",
    "probe",
    "aggregate",
    "_ab.py",
    "dump",
    "panel",
    "evaluate",
    "audit",
    "sweep_aggregate",
    "cfm_next_action",
)


def _proc(pattern: str, exclude_substrings: tuple = ()) -> bool:
    """True if a live TRAINER process matches ``pattern``. Never raises.

    2026-10-12. ``pgrep -f <pattern>`` matches ANY command line containing the
    pattern, and this repo is full of paths that unrelated processes
    legitimately carry. Concretely: the BLK-C-L3 guidance sweep launches
    ``--ckpt molmetal/reports/cfm_patience10_seed0/checkpoints/...``, so
    ``_proc("cfm_patience10_seed0")`` matched the sweep and reported the
    patience=10 run as "training in flight, last epoch 27" -- a run the user
    stopped. ``pgrep`` excludes only ITSELF, never the caller.

    A hit whose command line carries any of ``_EXCLUDE_MARKERS`` is therefore
    a MEASUREMENT reading a checkpoint, not a training run, and does not
    count.
    """
    try:
        r = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return False
    except Exception:
        return False

    pids = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    if not pids:
        return False
    markers = tuple(_EXCLUDE_MARKERS) + tuple(exclude_substrings)
    try:
        out = subprocess.run(
            ["ps", "-o", "args=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=10,
        ).stdout or ""
    except Exception:
        return False
    for line in out.splitlines():
        low = line.lower()
        if not low.strip():
            continue
        if any(m in low for m in markers):
            continue
        return True
    return False


def _run_done(run_dir: str) -> bool:
    """True if a run wrote its terminal 'done' event."""
    p = REPO / "molmetal" / "reports" / run_dir / "events.jsonl"
    if not p.is_file():
        return False
    try:
        with p.open(encoding="utf-8", errors="replace") as fh:
            return any('"event":"done"' in line for line in fh)
    except OSError:
        return False


def _epochs(run_dir: str):
    """(last epoch, n epochs recorded) or None."""
    p = REPO / "molmetal" / "reports" / run_dir / "events.jsonl"
    if not p.is_file():
        return None
    try:
        with p.open(encoding="utf-8", errors="replace") as fh:
            eps = [
                json.loads(ln).get("epoch")
                for ln in fh
                if '"event":"train_epoch"' in ln
            ]
        eps = [e for e in eps if isinstance(e, int)]
        return (max(eps), len(eps)) if eps else None
    except Exception:
        return None


def _dead_run_marker(txt: str) -> Optional[dict]:
    """Detect a TERMINAL failure in a run log, and say when it happened.

    WHY THIS EXISTS (2026-10-12). The pad_groups distributional test died of
    CUDA OOM at 04:57, writing a full traceback into its own output file. The
    resolver checked only for a "DONE" marker, so it classified that corpse as
    "in flight" and every recurring tick for the next THIRTEEN HOURS reported
    BLK-C-PERF as still running. A missing DONE marker means "not finished",
    which is true of both a live run and a dead one; conflating them is what
    let a crashed experiment masquerade as a slow one.

    The tick rules separately require reporting a dead experiment's exit code
    and output tail instead of relaunching silently -- which this resolver
    could not do for its own subject.

    Returns None when nothing terminal is found, so a genuinely running job is
    unaffected. The patterns are specific enough that a live run's progress
    lines do not match them: a real traceback ends with `Traceback (most
    recent call last):`, and torch/cuda failures name a concrete exception.
    """
    if not txt:
        return None
    lines = txt.splitlines()
    # Look for the LAST terminal marker, and take the timestamp from the
    # nearest bracketed prefix above it when the log has one.
    marker = None
    for i, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            marker = (i, "python traceback in the output")
        elif "OutOfMemoryError" in line or "CUDA out of memory" in line:
            marker = (i, "CUDA out of memory")
        elif line.startswith("Killed") or "Segmentation fault" in line:
            marker = (i, line.strip()[:80])
    if marker is None:
        return None
    idx, what = marker
    # Take the EARLIEST timestamp in the traceback region, not one within a
    # few lines of the marker. A python traceback is ~40 lines long and the
    # allocator's `[W1012 04:57:04]` sits at its TOP, above the
    # `Traceback (most recent call last):` line -- so a window anchored on the
    # final exception line never reaches it and reports "unknown time" for a
    # failure whose minute is right there in the log.
    start = 0
    for j, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            start = max(0, j - 12)
            break
    when = "unknown time"
    for j in range(start, min(idx + 1, len(lines))):
        # Two log dialects carry a timestamp: a bracketed `MM-DD HH:MM:SS`
        # (python logging) and the CUDACachingAllocator's `[W1012 04:57:04]`
        # form. Both are scanned because the allocator line is the one that
        # marks when the GPU actually ran out.
        m = re.search(r"\[(\d{2}-\d{2} \d{2}:\d{2}:\d{2})", lines[j])
        if m:
            when = m.group(1)
            break
        m = re.search(r"\[W\d+ (\d{2}:\d{2}:\d{2})", lines[j])
        if m:
            when = m.group(1)
            break
        if re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", lines[j]):
            when = lines[j].split()[0].strip("[]")
            break
    return {"what": what, "when": when}


def resolve() -> dict:
    if not TODO.is_file():
        return {"status": "unreadable", "why": f"{TODO} missing"}
    try:
        d = json.loads(TODO.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "unreadable", "why": f"cfm_todo.json: {e}"}

    blockers = {b.get("id"): b for b in d.get("blockers", [])}
    waiting: list[dict] = []
    ready: list[dict] = []
    closed: list[dict] = []
    dead: list[dict] = []

    # --- BLK-C-M1: patience=10 run -------------------------------------------
    m1_dir = "cfm_patience10_seed0"
    m1_live = _proc("cfm_patience10_seed0")
    # A run that is not live but already wrote epochs was STOPPED, not killed.
    # The two need different responses and conflating them is how a script
    # relaunches work the user deliberately ended.

    m1_done = _run_done(m1_dir)
    # A run that is not live but already wrote epochs was STOPPED, not killed.
    # The two need different responses and conflating them is how a script
    # relaunches work the user deliberately ended.
    _ep0 = _epochs(m1_dir)
    m1_stopped_by_user = (not m1_live) and (not m1_done) and bool(_ep0)
    if m1_done:
        ready.append({
            "id": "BLK-C-M1",
            "action": "READ the patience=10 result and settle the outcome",
            "read": f"molmetal/reports/{m1_dir}/events.jsonl (the 'done' event)",
            "decides": (
                "If seed0 reaches ~0.97, most past A/B differences were early-stop "
                "artefacts and BLK-C-M7 matters. If it stalls near 0.93, seed2's "
                "0.9703 is a real signal. Either way update cfm_todo.json and the "
                "M1 blocker."
            ),
            "then": "BLK-C-M7 triage becomes meaningful; report which arms need re-running.",
        })
    elif m1_live:
        ep = _epochs(m1_dir)
        waiting.append({
            "id": "BLK-C-M1",
            "why": "patience=10 training in flight",
            "progress": f"last epoch {ep[0]} ({ep[1]} recorded)" if ep else "no epoch yet",
        })
    elif m1_stopped_by_user:
        # 2026-10-12 -- the user directive changed the order of work: stop
        # training, benchmark the existing checkpoint on QED/SA/ADMET, and only
        # then design the loss / architecture changes. A stopped run is NOT a
        # dead run and must not be relaunched by this script.
        #
        # It is filed under `closed`, NOT `ready`. It used to be filed under
        # `ready`, which made every recurring tick report READY forever with an
        # action that is by definition "do nothing" -- the exact behaviour this
        # script's own comment further down calls out: "Reporting READY for
        # 'wait for file X' ... trains the reader to ignore READY." Two
        # consecutive ticks on 2026-10-12 produced that identical no-op before
        # it was noticed.
        closed.append({
            "id": "BLK-C-M1",
            "outcome": "CLOSED BY USER DIRECTIVE 2026-10-12 -- terminal, not pending.",
            "action": "NONE. There is nothing to do here and there never will be.",
            "detail": (
                "The patience=10 run was stopped at epoch 27 with gate 0.9532 and "
                "no plateau, at the user's direction, so the checkpoint could be "
                "benchmarked first. The checkpoint is retained and is the highest "
                "gate value recorded for any seed (0.9532 vs seed0's "
                "patience-truncated 0.9194). Do not relaunch it; do not report it "
                "as work waiting."
            ),
            "checkpoint_kept": (
                "molmetal/reports/cfm_patience10_seed0/checkpoints/"
                "cfm_yuelbond_wasserstein_seed0_last.pt"
            ),
        })
    else:
        ready.append({
            "id": "BLK-C-M1",
            "action": "RELAUNCH configs/cfm_patience10_seed0_2026_10_12.yaml — the run died",
            "check_first": "grep for 'done' in the events.jsonl; it may have finished cleanly",
            "gates": "pre_flight_check.py --pipeline launch must be rc=0 first",
        })

    # --- pad_groups equivalence A/B ------------------------------------------
    # 2026-10-12 — this pointed at equiv.txt, which was the FIRST equivalence
    # run: it died during tmQM loading, printed 2 lines, and a `grep -E
    # "clouds|..."` matched the substring inside "[tmqm-clouds]" and returned 0,
    # so it looked successful. It has since been superseded by equiv2.txt, run
    # from a NAMED script so `pgrep` cannot self-match on a file path. Both are
    # checked; the newest one with a VERDICT wins.
    job = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp"
    ab_outs = [job / "equiv2.txt", job / "equiv.txt"]
    # Wf-A100 (2026-10-20) — the settlement does NOT depend on those /tmp logs.
    # The McNemar verdict is a committed artefact, but it used to be reachable
    # only from inside the loop below, so once /tmp was cleared (or the host
    # changed) a SETTLED blocker fell through to nothing at all: neither
    # closed nor dead.  test_cfm_next_action_resolver caught exactly that.
    # Check the permanent artefact first; the loop then has nothing to read.
    _mcnemar_permanent = (
        REPO / "metrics" / "by_metric" / "cfm_pad_groups_mcnemar.json"
    )
    if _mcnemar_permanent.is_file():
        closed.append({
            "id": "BLK-C-PERF (pad_groups A/B)",
            "outcome": _MCNEMAR_SETTLED_TEXT,
        })
        ab_outs = []
    for ab_out in ab_outs:
        if not ab_out.is_file():
            continue
        try:
            txt = ab_out.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "VERDICT:" in txt:
            line = next(
                (ln for ln in txt.splitlines() if ln.startswith(("VERDICT", "grouped", "compared"))),
                "",
            )
            # 2026-10-12 — the coordinate-equality criterion this file used to
            # apply is IMPOSSIBLE, so a DIVERGES verdict from equiv*.txt is a
            # FALSE NEGATIVE and must not be read as "padding is wrong".
            # padded draws randn(192,72,3) = 41472 values; grouped draws
            # randn(b,n_atoms,3) per group over 52 groups ~= 21000. Same seed,
            # different shapes, unrelated samples. A max coordinate difference of
            # 7.58 Angstrom measures the NOISE DRAW, not the batching.
            # The pad_groups verdict therefore comes from the DISTRIBUTIONAL test
            # (pad_dist.txt), which compares n_out / decoded / single_fragment
            # / n_frags over the same clouds.
            dist = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "pad_dist.txt"
            dist_txt = (
                dist.read_text(encoding="utf-8", errors="replace")
                if dist.is_file()
                else ""
            )
            dist_done = "DONE" in dist_txt
            # 2026-10-12 FIX. A crashed run and a running run are the same
            # thing to a "is there a DONE marker" test: both lack it. So on
            # 2026-10-12 this resolver reported BLK-C-PERF as "in flight" for
            # THIRTEEN HOURS after the distributional test had died of CUDA
            # OOM at 04:57 with a full traceback in its own log.
            #
            # Every tick in that window burned a turn re-reading an advisory
            # about a process that no longer existed, and the tick rules
            # separately require reporting a dead experiment's exit code --
            # a requirement this resolver could not meet for its own subject.
            #
            # A traceback or an Error line is a TERMINAL state, so it is
            # detected before the WAITING branch and reported as DEAD.
            #
            # 2026-10-12 (later) — the McNemar A/B has since been RUN, so the
            # dead legacy log no longer describes the state of this blocker.
            # The new verdict lives in
            # metrics/by_metric/cfm_pad_groups_mcnemar.json and it WITHDRAWS
            # the 22.8x figure: that number came from the same single
            # global-max batch that needed 92 GiB and OOM'd twice, and the
            # memory fix (bucketing) traded the rest of the speed away. The
            # runnable speedup is 4.78x. This branch is kept only so a stale
            # pad_dist.txt still reports as DEAD rather than WAITING, but it
            # must NOT restate the withdrawn number as current.
            _mcnemar_file = (
                REPO / "metrics" / "by_metric" / "cfm_pad_groups_mcnemar.json"
            )
            _settled = _mcnemar_file.is_file()
            _dead = _dead_run_marker(dist_txt)
            if _settled:
                closed.append({
                    "id": f"BLK-C-PERF ({ab_out.name})",
                    "outcome": _MCNEMAR_SETTLED_TEXT,
                })
                break
            if _dead and not dist_done:
                dead.append({
                    "id": f"BLK-C-PERF ({ab_out.name})",
                    "died": _dead["when"],
                    "why": _dead["what"],
                    "progress": f"{dist_txt.count(chr(10))} lines, no DONE marker",
                    "exit_hint": (
                        "check the tail of "
                        f"{dist} for the traceback; it is a terminal state, "
                        "NOT an experiment still running"
                    ),
                    "note": (
                        "The legacy pad_dist.txt crashed; pad_groups itself "
                        "needed 92 GiB in its single-batch form. Do NOT quote "
                        "22.8x from equiv2.txt -- that number belongs to the "
                        "implementation that OOM'd and has been withdrawn."
                    ),
                })
                break
            # If the verdict still depends on a running experiment, that is
            # WAITING, not READY. Reporting READY for "wait for file X" makes
            # every tick re-read the same advisory and burns a turn to say
            # nothing -- the exact behaviour the recurring tick is meant to
            # avoid. It also trains the reader to ignore READY.
            if not dist_done:
                waiting.append({
                    "id": f"BLK-C-PERF ({ab_out.name})",
                    "why": "pad_groups distributional test in flight",
                    "progress": (
                        f"{dist_txt.count(chr(10))} lines, no DONE marker"
                        if dist_txt
                        else "output file not created yet"
                    ),
                    "note": (
                        "Do NOT quote 22.8x: that figure came from the "
                        "single global-max batch that needed 92 GiB and OOM'd, "
                        "and it has been WITHDRAWN. The runnable speedup is "
                        "4.78x. The coordinate test's DIVERGES is a FALSE "
                        "NEGATIVE (mis-specified criterion) and is not evidence "
                        "against padding either."
                    ),
                })
                break
            ready.append({
                "id": f"BLK-C-PERF ({ab_out.name})",
                "action": (
                    f"Settle pad_groups from the DISTRIBUTIONAL test ({dist.name})."
                    if dist_done else
                    f"pad_dist.txt has no DONE line yet; the coordinate test's "
                    f"DIVERGES is a false negative (see comment) and is NOT evidence "
                    f"against padding. Wait for pad_dist.txt."
                ),
                "measured": line.strip(),
                "coordinate_verdict_status": (
                    "FALSE NEGATIVE -- mis-specified criterion, superseded"
                ),
                "speedup_measured": (
                    "4.78x (1958.0s -> 409.7s, 192 clouds) AFTER the memory "
                    "fix. The 22.8x figure is WITHDRAWN -- it belonged to the "
                    "92 GiB single-batch implementation that could not run."
                ),
                "then": (
                    "Record in metrics/by_metric/cfm_pad_groups_ab.json using the "
                    "distributional numbers, not the coordinate comparison. Keep "
                    "pad_groups default OFF until then."
                ),
            })
            break

    # --- anything with no live experiment and no pending result ---------------
    for b in d.get("next_actions", []):
        bid = str(b.get("action", ""))
        if "patience10" in bid or "BLK-C-M7" in bid or "entropy_reg_012" in bid:
            continue  # covered above or gated on M1
        blocking = str(b.get("blocking", "none")).lower()
        if blocking == "none" and not ready:
            ready.append({
                "id": "next_actions",
                "action": b.get("action"),
                "why": b.get("why"),
                "priority": b.get("priority"),
            })
        break

    out = {
        "status": "ready" if ready else "waiting",
        "blocker_status": {k: v.get("status", "?")[:80] for k, v in blockers.items()},
        "ready": ready,
        "waiting": waiting,
        # Terminal items, reported once and never acted on. Listing them keeps
        # the resolver honest about WHY they are absent from `ready`: a reader
        # who expected BLK-C-M1 to appear should be able to see that it closed
        # rather than having it silently vanish.
        "closed": closed,
        # 2026-10-12. DEAD experiments, kept separate from `closed` because
        # they did NOT resolve anything -- they stopped. Reporting them as
        # closed would be the same error as reporting them as running.
        "dead": dead,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", dest="quiet")
    a = ap.parse_args()

    r = resolve()
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    elif a.quiet:
        pass
    else:
        print(f"cfm_next_action: {r['status'].upper()}")
        for c in r.get("closed", []):
            print(f"  CLOSED  {c['id']}: {c.get('outcome', 'closed')}")
        for x in r.get("dead", []):
            # Printed FIRST among the non-ready states and never summarised
            # away: a dead run that reports as "waiting" costs a tick per
            # invocation, forever, and looks exactly like patience.
            print(f"  DEAD    {x['id']}: {x.get('why')} (at {x.get('died')})")
            if x.get("exit_hint"):
                print(f"          {x['exit_hint']}")
            if x.get("note"):
                print(f"          note: {x['note']}")
        for w in r.get("waiting", []):
            prog = f" ({w['progress']})" if w.get("progress") else ""
            print(f"  WAITING {w['id']}: {w['why']}{prog}")
        for x in r.get("ready", []):
            print(f"  READY   {x['id']}: {x['action']}")
            if x.get("measured"):
                print(f"          measured: {x['measured']}")
            if x.get("decides"):
                print(f"          decides: {x['decides']}")
            if x.get("then"):
                print(f"          then: {x['then']}")
            if x.get("why"):
                print(f"          why: {x['why']}")
        if r["status"] == "unreadable":
            print(f"  !! {r['why']}")
    return 0 if r["status"] == "ready" else (1 if r["status"] == "waiting" else 2)


if __name__ == "__main__":
    raise SystemExit(main())