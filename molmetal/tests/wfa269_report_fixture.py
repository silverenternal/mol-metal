"""Wf-A269 wave-16 / wave-17 — a REALISTIC pre-flight report for the
shared gate-verdict consumer.

WHY THIS MODULE EXISTS
----------------------
``molmetal/scripts/wf_launch_gate_verdict.py`` is FAIL-CLOSED on a
pre-flight report (Wf-A269 wave-16, 2026-11-03; wave-17 added the payload
binding).  It refuses a report that carries

  * no ``summary`` block, or a ``summary`` that measures ZERO gates,
  * no ``last_refresh`` stamp, or one that is stale / unparseable / in the
    future,
  * no ``provenance`` block,
  * a ``provenance.git_sha`` that is not the live HEAD,
  * a ``provenance.gate_source_sha256`` that is not the live gate source's,
  * a ``provenance.payload_sha256`` that does not match the report's own
    content.

Every one of those refusals is CORRECT and intended.  But a synthetic test
fixture that simply omits the fields would then be refused for a reason
that has nothing to do with the behaviour under test — so this module
stands in for what the PRODUCER would really have written:

  * the live HEAD (``git rev-parse HEAD``),
  * the live gate source's SHA-256,
  * a fresh ISO-8601 UTC ``last_refresh`` stamp,
  * the payload hash, computed by the SHARED canonical function imported
    from the consumer — never re-implemented here, so the fixture cannot
    drift away from the definition it is testing.

The payload hash covers the report WITHOUT its provenance block, exactly as
the consumer's ``canonical_payload_sha256`` does, so the fixture and the
consumer agree by construction rather than by coincidence.

MEASURED, not assumed: on 2026-11-03 the round-trip was verified on a real
report produced by ``pre_flight_check.py --gates
p3_wf343c4_hyperparams_config_complete`` — the hash the producer wrote into
the file equals the hash recomputed from the file's parsed content.

CPU-only, read-only: no GPU, no torch, no network, and no repo file is
written by this module.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from molmetal.scripts.wf_launch_gate_verdict import canonical_payload_sha256

#: The repo root — this module lives at ``<repo>/molmetal/tests/``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The producer whose hash the report is bound to.  Relative to REPO_ROOT,
#: which is the form the producer itself records.
GATE_SOURCE_REL = "molmetal/scripts/pre_flight_check.py"


def live_head_sha() -> str:
    """The current commit SHA, or ``""`` when git cannot be reached."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def live_gate_source_sha256() -> str:
    """SHA-256 of the live gate source, or ``""`` when it cannot be read."""
    try:
        data = (REPO_ROOT / GATE_SOURCE_REL).read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()


def stamp_live_provenance(report: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp ``report`` with the freshness + provenance a real report carries.

    Mutates and returns ``report``.  The order matters: ``last_refresh`` is
    part of the payload, so it is stamped BEFORE the payload hash is
    computed; the provenance block is excluded from the payload, so adding it
    afterwards does not invalidate the hash.
    """
    now = datetime.now(timezone.utc).isoformat()
    report["last_refresh"] = now
    report["provenance"] = {
        "level": "L1",
        "git_sha": live_head_sha(),
        "git_dirty": None,
        "run_id": "fixture",
        "started_on": now,
        "finished_on": now,
        "gate_source": GATE_SOURCE_REL,
        "gate_source_sha256": live_gate_source_sha256(),
    }
    report["provenance"]["payload_sha256"] = canonical_payload_sha256(report)
    return report
