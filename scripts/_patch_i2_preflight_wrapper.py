#!/usr/bin/env python3
"""BLK-I2 one-shot: add pre_flight_check.py wrapper to 8 launchers.

Per AGENTS.md BLK-I2 fix_steps: every scripts/launch_*.sh that actually
launches a long-running trainer starts with `if ! .venv/bin/python
molmetal/scripts/pre_flight_check.py; then exit 1; fi`.

Scope (per BLK-I1 + this wave):
  PATCH (8): launchers that ALREADY launch a trainer and were BLK-I1-patched
             (custom 2 + helper 5) + BLK-C2 (1) — these are the load-bearing
             trainer launchers.
  SKIP  (5): pre-existing 3 (launch_all_training + launch_item4_chain_post_wf131
             + launch_chemprop_v3_p14_post_cfm_round7 — out of BLK-I1 scope per
             the BLK-I1 closeout §"Inventory of patched launchers" + the
             "launchers with pre-existing setsid/nohup" row), the 1 CPU-only
             smoke (launch_variance_puct_smoke — pre_flight is for fresh
             launches, this one IS a smoke), and the 1 currently-running
             trainer (launch_wf_fix3lines_cfm_100ep_qed — BLK-C4 is in flight
             on GPU 0; editing its launcher mid-run would change a file the
             trainer is NOT reading but the user is observing).

This script is idempotent: a re-run finds the preflight guard already
present and skips. Per AGENTS.md §5.6 the writes go through
`atomic_write_file` (per-path process-local lock + temp file +
os.replace).

Per AGENTS.md §5.9 single-venv: the inserted line invokes
`.venv/bin/python` directly. The existing launchers in scope use
`${PY}` whose default is `.venv/bin/python`; we follow the same
convention.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# 8 launchers to patch (rationale: see module docstring)
TARGETS = [
    "launch_cfm_round7_f5f6_retrain.sh",
    "launch_cfm_tmqm_all6_retrain.sh",
    "launch_dmpnn_clean5_lomo.sh",
    "launch_dmpnn_clean5v3_loso.sh",
    "launch_dmpnn_clean5v4_loso_d1d3.sh",
    "launch_wf_fix3lines_bace.sh",
    "launch_wf_fix3lines_cfm.sh",
    "launch_wf_fix3lines_cfm_100ep.sh",
]

PREFLIGHT_BLOCK = """
# BLK-I2 (2026-10-09): pre-flight gate per AGENTS.md §1.4.
# Per `todo.json` BLK-I2 fix_steps: every trainer-launching
# scripts/launch_*.sh runs `pre_flight_check.py` first; rc != 0
# aborts before the trainer is invoked. Captures stdout to
# logs/preflight_<script>_<ts>.log so the gate evidence is on disk.
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_LOG="$LOG_DIR/preflight_$(basename "$0" .sh)_$TS.log"
if ! PYTHONPATH=/home/hugo/codes/mol-metal ${PY:-.venv/bin/python} \
    molmetal/scripts/pre_flight_check.py \
    --pipeline launch \
    >"$PREFLIGHT_LOG" 2>&1; then
  echo "FAIL: pre_flight_check (BLK-I2 gate); see $PREFLIGHT_LOG" >&2
  exit 1
fi
echo "[BLK-I2] pre_flight_check PASS (log: $PREFLIGHT_LOG)"
"""


def patch_one(path: Path) -> tuple[bool, str]:
    """Insert the preflight block into `path` if not already present.

    Returns (changed, summary_string).
    """
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    original = path.read_text()
    if "BLK-I2" in original and "pre_flight_check" in original:
        return (False, "already patched (BLK-I2 marker present)")

    # Insert AFTER the last consecutive comment/shebang block at the top,
    # or after `set -e` if present, whichever comes first. We pick the
    # first `set -e` line as the anchor; if none, we anchor after the
    # last top-of-file comment.
    lines = original.splitlines(keepends=True)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("set -e") or line.strip().startswith("set -euo"):
            insert_idx = i + 1
            break
    if insert_idx is None:
        # No set -e — find the end of the first contiguous comment block
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped == "":
                continue
            insert_idx = i
            break
        if insert_idx is None:
            insert_idx = 0

    new_lines = lines[:insert_idx] + [PREFLIGHT_BLOCK + "\n"] + lines[insert_idx:]
    new_content = "".join(new_lines)
    atomic_write_file(path=path, content=new_content)

    # Preserve the executable bit (atomic_write_file on a new file
    # defaults to umask; for an existing file the mode is preserved on
    # Linux but the chmod here is belt-and-braces).
    st = path.stat()
    if not (st.st_mode & 0o111):
        os.chmod(path, st.st_mode | 0o755)

    return (True, f"inserted at line {insert_idx + 1} ({len(PREFLIGHT_BLOCK.splitlines())} line block)")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for t in TARGETS:
            print(t)
        return 0

    print(f"[BLK-I2 patcher] target count: {len(TARGETS)}")
    n_changed = 0
    n_skipped = 0
    for name in TARGETS:
        p = SCRIPTS / name
        if not p.exists():
            print(f"  SKIP  {name} (not found)")
            n_skipped += 1
            continue
        changed, summary = patch_one(p)
        marker = "PATCH" if changed else "NOOP"
        print(f"  {marker:5s} {name:60s}  {summary}")
        if changed:
            n_changed += 1

    print(f"[BLK-I2 patcher] done: {n_changed} patched, {n_skipped} skipped, {len(TARGETS) - n_changed - n_skipped} noop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
