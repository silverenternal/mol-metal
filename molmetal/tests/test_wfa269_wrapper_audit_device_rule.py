# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Wf-A269 wave-5 -- the repaired DUPLICATE classifier of scripts/_wrapper_audit.py.

Owner: Hugo Lee
Date: 2026-11-02
Parent: Wf-A269 wave-4 adjudication (defect D-class, §5.9 gate regression)
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

THE DEFECT
----------
``scripts/_wrapper_audit.py`` classified a wrapper as DUPLICATE when the RAW
TEXT of the file contained more than one ``--device <word>`` match.  Two
MEASURED consequences, both from the 2026-11-02 wave-4 hardening:

1. FALSE POSITIVE.  ``scripts/launch_wf310c_todo220_gpu.sh`` carries TWO
   ``--device`` flags -- but they belong to TWO DISTINCT trainers
   (``molmetal/scripts/wf219_temporal_split_driver.py`` at the Stage-A
   invocation and ``molmetal/scripts/nci60_9cell_chemprop_driver.py`` at the
   Stage-B invocation), and both carry the SAME device value.  The wave-4
   hardening then added PROSE ``--device cuda`` inside the wrapper's own
   DRY_RUN.log / plan.json heredocs, so the raw count went 2 -> 6 while the
   number of real trainer invocations stayed at 2.  The gate therefore
   reported exactly one violation, on a wrapper that violates nothing.

2. BLIND SPOT.  The regex ``--device\\s+\\w+`` cannot see the
   ``--device=<value>`` spelling (the ``=`` is not whitespace), so a wrapper
   carrying BOTH ``--device=cuda:0`` and ``--device=cuda:1`` -- the actual
   violation the rule exists to catch -- was INVISIBLE to the gate.  Six
   wrappers on disk use the ``=`` spelling.

THE REPAIRED RULE
------------------
A wrapper violates the canonical "one resolver, one device" pattern (repo
commit 87b026ee) when it names MORE THAN ONE DISTINCT GPU DEVICE:

  * both spellings are matched (``--device X`` and ``--device=X``);
  * values are normalised (quotes / trailing punctuation stripped; a shell
    variable reference is an opaque symbolic device);
  * a ``cpu`` value is exempt (every wrapper on disk that carries one uses it
    in an explicitly-CPU branch).

This module is the regression net for that rule.  It asserts BOTH directions:
the true violation is still caught, and the false positive is gone.

CPU-ONLY.  No GPU, no training, no sampling.  < 1 s wall.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_wrapper_audit():
    """Import scripts/_wrapper_audit.py by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location(
        "_wrapper_audit_wfa269", SCRIPTS_DIR / "_wrapper_audit.py"
    )
    assert spec and spec.loader, "scripts/_wrapper_audit.py not importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = _load_wrapper_audit()


# ---------------------------------------------------------------------------
# Helpers — every fixture is a MINIMAL wrapper; the rule only reads the
# --device spellings, so the rest of the canonical surface is irrelevant here.
# ---------------------------------------------------------------------------
def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "launch_wf999_probe.sh"
    path.write_text(body, encoding="utf-8")
    return path


def _wrapper_with(*device_flags: str) -> str:
    """A wrapper whose single trainer invocation carries the given --device flags."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'PY="/home/hugo/codes/mol-metal/.venv/bin/python"',
        '"${PY}" molmetal/scripts/probe_trainer.py \\',
    ]
    for flag in device_flags:
        lines.append(f"    {flag} \\")
    lines.append('    --out "${OUT}"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 1. The TRUE violation must still be caught (space spelling).
# ---------------------------------------------------------------------------
def test_two_distinct_gpu_devices_are_duplicates(tmp_path: Path) -> None:
    """The canonical violation: one wrapper naming two different GPU devices."""
    path = _write(tmp_path, _wrapper_with("--device cuda:0", "--device cuda:1"))
    assert _audit.has_multiple_device_flags(path) is True


# ---------------------------------------------------------------------------
# 2. The `=` spelling -- the blind spot the old regex could not see.
# ---------------------------------------------------------------------------
def test_equals_spelling_two_distinct_gpu_devices_are_duplicates(
    tmp_path: Path,
) -> None:
    """`--device=cuda:0` + `--device=cuda:1` — invisible to the OLD rule."""
    path = _write(tmp_path, _wrapper_with("--device=cuda:0", "--device=cuda:1"))
    assert _audit.has_multiple_device_flags(path) is True


def test_equals_spelling_mixed_with_space_spelling_is_duplicates(
    tmp_path: Path,
) -> None:
    """The two spellings are compared against each other, not separately."""
    path = _write(tmp_path, _wrapper_with("--device=cuda:0", "--device cuda:1"))
    assert _audit.has_multiple_device_flags(path) is True


# ---------------------------------------------------------------------------
# 3. The FALSE POSITIVE must be gone: two DISTINCT trainers, one device.
# ---------------------------------------------------------------------------
def test_two_distinct_trainers_same_device_are_not_duplicates(tmp_path: Path) -> None:
    """The todo220 class: two trainers, same device value -> NOT a duplicate."""
    body = (
        "#!/usr/bin/env bash\n"
        'PY="/home/hugo/codes/mol-metal/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/trainer_a.py --device cuda --out "${A}"\n'
        '"${PY}" molmetal/scripts/trainer_b.py --device cuda --out "${B}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is False


# ---------------------------------------------------------------------------
# 4. PROSE inside a heredoc must not change the verdict.
# ---------------------------------------------------------------------------
def test_prose_inside_heredoc_does_not_change_the_verdict(tmp_path: Path) -> None:
    """A DRY_RUN.log / plan.json heredoc that mentions --device is prose."""
    body = (
        "#!/usr/bin/env bash\n"
        "cat <<EOF\n"
        "[DRY-RUN] stage_a: 5 metals x 5 seeds, --device cuda\n"
        "[DRY-RUN] stage_b: 9 cells x 3 seeds, --device cuda\n"
        "EOF\n"
        '"${PY}" molmetal/scripts/trainer_a.py --device cuda --out "${A}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is False


# ---------------------------------------------------------------------------
# 5. The CPU-fallback class: cpu + one GPU device is NOT a duplicate.
# ---------------------------------------------------------------------------
def test_cpu_fallback_is_not_a_duplicate(tmp_path: Path) -> None:
    """`if [ "${gpu}" != "cpu" ]` fallback -> one GPU target, not two."""
    body = (
        "#!/usr/bin/env bash\n"
        'if [ "${gpu}" != "cpu" ]; then\n'
        "    args+=( --device=cuda )\n"
        "else\n"
        "    args+=( --device cpu )\n"
        "fi\n"
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is False


# ---------------------------------------------------------------------------
# 6. The same shell variable is the same device, quoted or not.
# ---------------------------------------------------------------------------
def test_quoted_and_unquoted_variable_are_the_same_device(tmp_path: Path) -> None:
    """`--device "${GPU}"` and `--device ${GPU}` are ONE symbolic device."""
    path = _write(tmp_path, _wrapper_with('--device "${GPU}"', "--device ${GPU}"))
    assert _audit.has_multiple_device_flags(path) is False


def test_two_different_variables_are_two_devices(tmp_path: Path) -> None:
    """`${GPU}` and `${OTHER_GPU}` are two distinct symbolic devices."""
    path = _write(tmp_path, _wrapper_with("--device ${GPU}", "--device ${OTHER_GPU}"))
    assert _audit.has_multiple_device_flags(path) is True


# ---------------------------------------------------------------------------
# 6b. Wf-A269 wave-19 (2026-11-04) — the RESOLUTION class.
#
# The rule compared the captured values as OPAQUE tokens, so a shell
# variable was a device name in its own right.  MEASURED on
# scripts/launch_wf310b_g6_final.sh: tokens were
# ['var:GPU', 'cuda:0', 'var:GPU'] while the file assigns GPU="cuda:0" --
# ONE device, reported as TWO.
#
# The rule now RESOLVES a variable from the file's own plain-literal
# assignment.  These four tests lock the resolution AND its fail-closed
# edge: an UNRESOLVABLE variable must keep its opaque token, or the rule
# would go blind exactly where it cannot decide.
# ---------------------------------------------------------------------------
def test_a_resolvable_variable_and_its_literal_are_one_device(
    tmp_path: Path,
) -> None:
    """`GPU="cuda:0"` + `--device "${GPU}"` + `--device cuda:0` = ONE device."""
    body = (
        "#!/usr/bin/env bash\n"
        'GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0\n'
        '"${PY}" molmetal/scripts/probe_trainer.py \\\n'
        '    --device "${GPU}" \\\n'
        "    --device cuda:0\n"
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is False, (
        "a variable that RESOLVES to the same literal must not be a second "
        "device"
    )


def test_two_resolvable_variables_with_different_literals_are_two_devices(
    tmp_path: Path,
) -> None:
    """`GPU="cuda:0"` + `OTHER="cuda:1"` really are two devices."""
    body = (
        "#!/usr/bin/env bash\n"
        'GPU="cuda:0"\n'
        'OTHER="cuda:1"\n'
        '"${PY}" molmetal/scripts/probe_trainer.py \\\n'
        '    --device "${GPU}" \\\n'
        '    --device "${OTHER}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is True


def test_a_variable_assigned_two_literals_is_still_two_devices(
    tmp_path: Path,
) -> None:
    """A variable assigned cuda:0 in one branch and cuda:1 in another."""
    body = (
        "#!/usr/bin/env bash\n"
        'GPU="cuda:0"\n'
        'if [ -n "${WAVE_B}" ]; then\n'
        '  GPU="cuda:1"\n'
        "fi\n"
        '"${PY}" molmetal/scripts/probe_trainer.py \\\n'
        '    --device "${GPU}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is True


def test_an_unresolvable_variable_keeps_its_opaque_token(tmp_path: Path) -> None:
    """No plain-literal assignment -> the token stays `var:NAME` (fail-closed)."""
    body = (
        "#!/usr/bin/env bash\n"
        'GPU="${CUDA_VISIBLE_DEVICES-0}"  # a substitution, not a literal\n'
        '"${PY}" molmetal/scripts/probe_trainer.py \\\n'
        '    --device "${GPU}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit._resolved_device_flag_values(path) == ["var:GPU"], (
        "a substitution is not a stated literal — the variable must stay "
        "unresolved"
    )
    assert _audit.has_multiple_device_flags(path) is False


# ---------------------------------------------------------------------------
# 7. The real wrapper that produced the false positive.
# ---------------------------------------------------------------------------
def test_real_todo220_launcher_is_not_a_duplicate() -> None:
    """MEASURED on-disk state: the wave-4 hardening's one violation is gone."""
    path = SCRIPTS_DIR / "launch_wf310c_todo220_gpu.sh"
    assert path.exists(), f"{path} not on disk"
    values = _audit.device_flag_values(path)
    assert values, "the wrapper names no --device flag at all"
    assert len({v for v in values if v != "cpu"}) == 1, (
        f"expected exactly one distinct GPU device, got {sorted(set(values))}"
    )
    assert _audit.has_multiple_device_flags(path) is False


def test_no_on_disk_wrapper_names_two_gpu_devices() -> None:
    """Repo-wide invariant: the "one resolver, one device" pattern holds."""
    offenders = [
        wrapper.name
        for wrapper in sorted(SCRIPTS_DIR.glob("launch_wf*.sh"))
        if _audit.has_multiple_device_flags(wrapper)
    ]
    assert offenders == [], (
        f"wrappers naming more than one distinct GPU device: {offenders}"
    )


# ---------------------------------------------------------------------------
# 7b. The BACKTICK class (measured 2026-11-02 by the main thread).
# ---------------------------------------------------------------------------
def test_markdown_quoted_device_in_a_comment_is_the_same_device(
    tmp_path: Path,
) -> None:
    """A comment quoting the device flag in backticks names ONE device.

    MEASURED DEFECT: the value group's delimiter set excluded whitespace,
    quotes, comma, semicolon, `)`, `]` and `}` but NOT the backtick, so a
    backtick-quoted `--device cuda` captured as the DIFFERENT token "cuda`".
    `_normalize_device_value` does not strip a trailing backtick, so the one
    device plus its own prose echo counted as TWO distinct devices -> a
    FALSE DUPLICATE on a wrapper that violates nothing.
    """
    body = (
        "#!/usr/bin/env bash\n"
        "# the other wrappers all spell it `--device cuda`, so this one is odd\n"
        '"${PY}" molmetal/scripts/trainer_a.py --device cuda --out "${A}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.device_flag_values(path) == ["cuda", "cuda"]
    assert _audit.has_multiple_device_flags(path) is False


def test_double_backtick_quoted_device_is_the_same_device(tmp_path: Path) -> None:
    """The double-backtick prose form (the repo's docstring style) is safe too."""
    body = (
        "#!/usr/bin/env bash\n"
        "# ``--device cuda`` is the device the trainer runs on\n"
        '"${PY}" molmetal/scripts/trainer_a.py --device cuda --out "${A}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.device_flag_values(path) == ["cuda", "cuda"]
    assert _audit.has_multiple_device_flags(path) is False


def test_backtick_quoting_does_not_mask_a_real_violation(tmp_path: Path) -> None:
    """The fix must not weaken the rule: two REAL distinct devices still flag."""
    body = (
        "#!/usr/bin/env bash\n"
        "# prose quoting `--device cuda`\n"
        '"${PY}" trainer_a.py --device cuda:0 --out "${A}"\n'
        '"${PY}" trainer_b.py --device cuda:1 --out "${B}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.has_multiple_device_flags(path) is True


# ---------------------------------------------------------------------------
# 8. The normalisation unit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ('"cuda:0"', "cuda:0"),
        ("'cuda:0'", "cuda:0"),
        ("${GPU}", "var:GPU"),
        ('"${GPU}"', "var:GPU"),
        ("$GPU", "var:GPU"),
        ("cuda,", "cuda"),
        ("cuda\"", "cuda"),
    ],
)
def test_normalize_device_value(raw: str, expected: str) -> None:
    """The value normaliser strips quotes / punctuation and symbols variables."""
    assert _audit._normalize_device_value(raw) == expected


def test_device_flag_values_are_collected_in_file_order(tmp_path: Path) -> None:
    """The collector returns every value, normalised, in file order."""
    body = (
        "#!/usr/bin/env bash\n"
        '"${PY}" trainer_a.py --device cuda:0 --out "${A}"\n'
        '"${PY}" trainer_b.py --device=cuda:1 --out "${B}"\n'
    )
    path = _write(tmp_path, body)
    assert _audit.device_flag_values(path) == ["cuda:0", "cuda:1"]
