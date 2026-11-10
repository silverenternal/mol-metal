"""Wf-A165 — `scripts/start.sh` must honour an EXPLICIT ambient CUDA_VISIBLE_DEVICES.

DEFECT (Wf-A163 GPU-tooling census, branch B).  `scripts/start.sh` derived
the GPU mask from the yaml `compute.gpu` field and exported it
UNCONDITIONALLY, overwriting any ambient mask:

    _cvd="${GPU#cuda:}"
    _cvd="${_cvd%%[^0-9]*}"
    if [[ -n "$_cvd" ]]; then
      export CUDA_VISIBLE_DEVICES="$_cvd"
      echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from yaml compute.gpu=$GPU)"
    ...

So a session lock of `CUDA_VISIBLE_DEVICES=""` (no GPU) was silently
replaced by the yaml's device index: a yaml-only launch RE-ENABLED a GPU
the operator had switched off.  The downstream resolver
(`molmetal/molmetal_lam/sbdd_env/training_utils.py` ->
`apply_default_cuda_visible_devices`) already honours an explicitly-empty
env var ("env > default GPU 0"), so the launcher was the only leak.

THE RULE these tests pin:

  * ambient UNSET      -> the yaml decides the device.  Unchanged behaviour;
                          the yaml stays the device SSOT (AGENTS.md §5.11).
  * ambient SET to ""  -> the empty mask wins (the no-GPU lock).  An empty
                          mask is NOT the same thing as an unset variable.
  * ambient SET to "N" -> the caller's explicit device wins over the yaml.

HOW THIS IS PROVEN.  The real blocks are EXTRACTED from the real
`scripts/start.sh` (by their sentinel markers) and EXECUTED in a sandboxed
`bash -c` with a stubbed `$GPU`.  `scripts/start.sh` itself is NEVER run —
it launches a trainer — and nothing in this file calls CUDA: the shell
blocks only manipulate strings.  No GPU memory is allocated.

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        molmetal/tests/test_wfa165_start_shell.py -q
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
START_SH = PROJECT_ROOT / "scripts" / "start.sh"

# Sentinel markers written by Wf-A165 into scripts/start.sh.  They make the
# extraction of "the real block" deterministic instead of regex-fragile.
CAPTURE_START = "# --- Wf-A165: ambient CUDA_VISIBLE_DEVICES capture (start of block) ---"
CAPTURE_END = "# --- Wf-A165: ambient CUDA_VISIBLE_DEVICES capture (end of block) ---"
MASK_START = "# --- Wf-A165: yaml-derived GPU mask (start of block) ---"
MASK_END = "# --- Wf-A165: yaml-derived GPU mask (end of block) ---"

# The PRE-REPAIR shape, verbatim from `git show HEAD:scripts/start.sh` before
# the Wf-A165 repair.  It is the NEGATIVE CONTROL: the rule below must flag
# it, so a re-introduction of the defect fails this file.
PRE_REPAIR_BLOCK = r'''# "cuda:0" → "0"; ignore anything non-numeric ("auto", "cpu", …) so a
# bad yaml value cannot silently pin the wrong device.
_cvd="${GPU#cuda:}"
_cvd="${_cvd%%[^0-9]*}"
if [[ -n "$_cvd" ]]; then
  export CUDA_VISIBLE_DEVICES="$_cvd"
  echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from yaml compute.gpu=$GPU)"
else
  echo "[start.sh][WARN] yaml compute.gpu='$GPU' is not a cuda:N index; leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
'''


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def _extract(src: str, start_marker: str, end_marker: str) -> str:
    """Return the source between two sentinel markers (exclusive)."""
    lines = src.splitlines()
    if start_marker not in lines:
        raise AssertionError(f"sentinel marker not found: {start_marker!r}")
    if end_marker not in lines:
        raise AssertionError(f"sentinel marker not found: {end_marker!r}")
    i = lines.index(start_marker)
    j = lines.index(end_marker)
    assert i < j, f"markers out of order: {start_marker!r} at {i+1}, {end_marker!r} at {j+1}"
    return "\n".join(lines[i + 1 : j])


def _current_impl() -> str:
    """The real, current implementation: capture block + mask block."""
    src = START_SH.read_text(encoding="utf-8")
    capture = _extract(src, CAPTURE_START, CAPTURE_END)
    mask = _extract(src, MASK_START, MASK_END)
    return capture + "\n" + mask


def _conditional_depth_at(lines: List[str], idx: int) -> int:
    """How many `if`/`for`/`while`/`case` constructs enclose line `idx`.

    0 == the statement is unconditional (top level of the block).  Only the
    first word of a line is inspected, which is exactly how the Wf-A165
    blocks are written.
    """
    depth = 0
    for line in lines[:idx]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first = stripped.split()[0]
        if first in ("if", "for", "while", "until", "case"):
            depth += 1
        elif first in ("fi", "done", "esac"):
            depth -= 1
    return depth


# --------------------------------------------------------------------------
# sandboxed execution
# --------------------------------------------------------------------------
def _run_block(
    block_src: str, gpu: str, ambient: Optional[str]
) -> Tuple[str, bool, str]:
    """Execute `block_src` under bash with a stubbed `$GPU` and the given
    ambient CUDA_VISIBLE_DEVICES.

    `ambient=None` means "the variable is NOT in the environment at all";
    `ambient=""` means "the variable is present and empty" — the two are
    different, and the defect is exactly about that difference.

    Returns (stdout, mask_is_set, mask_value).
    """
    script = (
        "set -euo pipefail\n"
        + block_src
        + "\n"
        + 'if [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]]; then\n'
        + "  printf '__SET__=1\\n__VALUE__=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
        + "else\n"
        + "  printf '__SET__=0\\n__VALUE__=\\n'\n"
        + "fi\n"
    )
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["GPU"] = gpu
    if ambient is not None:
        env["CUDA_VISIBLE_DEVICES"] = ambient
    proc = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"sandboxed block failed rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    is_set = "__SET__=1" in proc.stdout
    value = ""
    for line in proc.stdout.splitlines():
        if line.startswith("__VALUE__="):
            value = line[len("__VALUE__=") :]
    return proc.stdout, is_set, value


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def ambient_violations(block_src: str, gpu: str = "cuda:0") -> List[str]:
    """The Wf-A165 rule, expressed as a check.

    An EXPLICITLY-set ambient CUDA_VISIBLE_DEVICES must survive the block
    unchanged — including the empty mask.  Returns a list of violations
    (empty == the block honours the rule).
    """
    violations: List[str] = []
    for ambient in ("", "1"):
        _, is_set, value = _run_block(block_src, gpu=gpu, ambient=ambient)
        if not is_set:
            violations.append(
                f"ambient={ambient!r}: CUDA_VISIBLE_DEVICES became UNSET "
                f"(the caller's explicit mask was dropped)"
            )
        elif value != ambient:
            violations.append(
                f"ambient={ambient!r}: CUDA_VISIBLE_DEVICES became {value!r} "
                f"(the yaml overwrote the caller's explicit mask)"
            )
    return violations


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_sentinel_markers_present():
    """The extraction contract: the markers Wf-A165 wrote must still be there."""
    src = START_SH.read_text(encoding="utf-8")
    for marker in (CAPTURE_START, CAPTURE_END, MASK_START, MASK_END):
        assert marker in src, f"missing sentinel marker in scripts/start.sh: {marker}"


def test_ambient_unset_yaml_decides_the_device():
    """Case (a): the caller said nothing -> the yaml is the SSOT (unchanged)."""
    out, is_set, value = _run_block(_current_impl(), gpu="cuda:0", ambient=None)
    assert is_set, "the yaml-derived mask must still be exported"
    assert value == "0", f"yaml compute.gpu=cuda:0 must give mask '0', got {value!r}"
    assert "(from yaml compute.gpu=cuda:0)" in out, out


def test_ambient_unset_yaml_decides_a_non_zero_device():
    """The yaml still decides WHICH device when the caller is silent."""
    _, is_set, value = _run_block(_current_impl(), gpu="cuda:3", ambient=None)
    assert is_set and value == "3", f"expected '3', got {value!r} (set={is_set})"


def test_ambient_empty_lock_is_honoured():
    """Case (b): `CUDA_VISIBLE_DEVICES=""` is the no-GPU lock -> it must win."""
    out, is_set, value = _run_block(_current_impl(), gpu="cuda:0", ambient="")
    assert is_set, "the empty mask must stay SET (an unset variable is a different thing)"
    assert value == "", f"the no-GPU lock must survive, got {value!r}"
    assert "ambient mask honoured" in out, out
    assert "cuda:0 NOT applied" in out, out


def test_ambient_explicit_device_is_honoured():
    """Case (c): `CUDA_VISIBLE_DEVICES=1` is an explicit choice -> it must win."""
    out, is_set, value = _run_block(_current_impl(), gpu="cuda:0", ambient="1")
    assert is_set and value == "1", f"expected '1', got {value!r} (set={is_set})"
    assert "ambient mask honoured" in out, out


def test_default_path_is_byte_for_byte_unchanged():
    """The pre-repair default path and the current one must agree EXACTLY."""
    new_out, new_set, new_val = _run_block(_current_impl(), gpu="cuda:0", ambient=None)
    old_out, old_set, old_val = _run_block(PRE_REPAIR_BLOCK, gpu="cuda:0", ambient=None)
    assert new_out == old_out, (
        "stdout of the default path changed:\n"
        f"PRE-REPAIR:\n{old_out}\nCURRENT:\n{new_out}"
    )
    assert (new_set, new_val) == (old_set, old_val)


def test_negative_control_pre_repair_shape_is_flagged():
    """NEGATIVE CONTROL: the rule must flag the pre-repair shape, and must
    NOT flag the current implementation."""
    violations = ambient_violations(PRE_REPAIR_BLOCK)
    assert violations, "the pre-repair shape must be flagged by the rule"
    assert ambient_violations(_current_impl()) == [], (
        "the current implementation violates the rule: "
        f"{ambient_violations(_current_impl())}"
    )


def test_no_unconditional_export_in_the_file():
    """Whole-file property: every `export CUDA_VISIBLE_DEVICES` in
    scripts/start.sh lives inside the guarded block, and the ambient
    presence test precedes it."""
    lines = START_SH.read_text(encoding="utf-8").splitlines()
    i = lines.index(MASK_START)
    j = lines.index(MASK_END)
    exports = [n for n, line in enumerate(lines) if "export CUDA_VISIBLE_DEVICES" in line]
    assert exports, "no export of CUDA_VISIBLE_DEVICES found in scripts/start.sh"
    for n in exports:
        assert i < n < j, (
            f"scripts/start.sh:{n + 1} exports CUDA_VISIBLE_DEVICES outside the "
            f"guarded block ({lines[n].strip()!r})"
        )
    presence = [
        n
        for n, line in enumerate(lines)
        if "${CUDA_VISIBLE_DEVICES+set}" in line or "-v CUDA_VISIBLE_DEVICES" in line
    ]
    assert presence, "no ambient-presence test found in scripts/start.sh"
    assert presence[0] < exports[0], (
        "the ambient-presence test must come BEFORE the first export "
        "(scripts/start.sh:{0})".format(exports[0] + 1)
    )
    # every export must sit inside a conditional branch — a top-level export
    # would be unconditional, i.e. it would overwrite the ambient mask.
    block_lines = lines[i + 1 : j]
    for n in exports:
        depth = _conditional_depth_at(block_lines, n - (i + 1))
        assert depth >= 1, (
            "scripts/start.sh:{0} exports CUDA_VISIBLE_DEVICES unconditionally "
            "(conditional depth 0)".format(n + 1)
        )


def test_non_numeric_gpu_still_warns_and_pins_nothing():
    """A bad yaml value must still not silently pin a device (old behaviour)."""
    out, is_set, value = _run_block(_current_impl(), gpu="auto", ambient=None)
    assert not is_set, f"a non-numeric compute.gpu must not pin a device (got {value!r})"
    assert "is not a cuda:N index" in out, out
