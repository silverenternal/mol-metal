# SPDX-License-Identifier: Apache-2.0
"""`resume_from` is a MODE; the checkpoint file goes in `resume_from_path`.

2026-10-21 — the C7 full-corpus resume died 99 s into startup with

    --resume-from must be one of auto/none/path; got
    'molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/..._seed0_last.pt'.

The value was a perfectly good checkpoint path; the key was the wrong one.  The
message named neither the mistake nor the second key, and because the check sits
after the corpus build and the model construction, the cost of the ambiguity is
a full startup rather than a parse error.

These tests pin the hint, and pin that a real mode never gets one — a hint on
`auto` or `none` would be noise on every legitimate run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _load_trainer():
    import importlib.util
    import sys

    if "_t_resume_mode" in sys.modules:
        return sys.modules["_t_resume_mode"]
    spec = importlib.util.spec_from_file_location("_t_resume_mode", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_resume_mode"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "value",
    [
        "molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt",
        "checkpoints/last.pt",
        "./ckpt.pt",
        "C:\\runs\\seed0.pt",
        "something.ckpt",
    ],
)
def test_a_path_in_the_mode_key_is_told_about_the_second_key(value):
    message = _load_trainer().resume_mode_error(value)
    assert "resume_from_path" in message
    assert "MODE" in message
    # The original diagnosis must survive — the caller still needs to know
    # which values ARE legal.
    assert "auto/none/path" in message


@pytest.mark.parametrize("value", ["typo", "resume", "pathh"])
def test_a_non_path_typo_gets_no_path_hint(value):
    message = _load_trainer().resume_mode_error(value)
    assert "resume_from_path" not in message
    assert "auto/none/path" in message


def test_the_hint_quotes_the_offending_value():
    value = "molmetal/reports/x.pt"
    assert repr(value) in _load_trainer().resume_mode_error(value)
