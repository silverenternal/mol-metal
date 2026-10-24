"""Wf-A108i — the launcher must not supply a value the config did not declare.

MEASURED (§6.197).  `configs/c7_fullgen_2026_10_20.yaml` declares
`train.lr_scheduler: cosine` and `train.lr_warmup_epochs: 5`.  Every one of that
run's 18 `lr_step` events records `"phase": "constant"` at 3.0e-4, because
`scripts/start.sh` read `protocol.scheduler.type` — a key the yaml does not have
— defaulted it to `"none"`, and passed `--lr-scheduler none` on the command
line, which wins over the yaml under the documented precedence rule.  The same
script reported `epochs=50` for an 18-epoch config.

The rule these tests pin: for each launcher-supplied knob, take
`protocol.<key>` if declared, else the trainer's own yaml key `train.<key>` if
declared, else emit NOTHING and let the trainer's argparse default stand.

These spawn the real launcher in `--dry-run` and read the rendered command, so
they exercise the bash and the embedded Python together rather than a copy of
the logic.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
START = PROJECT_ROOT / "scripts" / "start.sh"
CFM_TRAIN_ONLY = PROJECT_ROOT / "configs" / "c7_fullgen_2026_10_20.yaml"
CFM_BOTH = PROJECT_ROOT / "configs" / "c8_fullgen_b64_2026_10_24.yaml"

pytestmark = [
    pytest.mark.skipif(not START.is_file(), reason="scripts/start.sh not present"),
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None,
        reason="start.sh pre-flight requires a GPU host",
    ),
]


def _render(cfg: Path) -> str:
    """Run `start.sh <cfg> --dry-run --no-tui` and return its stdout."""
    env = dict(os.environ)
    env["USER_OK"] = "1"
    proc = subprocess.run(
        ["bash", str(START), str(cfg), "--dry-run", "--no-tui"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def _command(stdout: str) -> str:
    m = re.search(r"^\[start\.sh\] trainer command: (.*)$", stdout, re.MULTILINE)
    assert m, stdout[-3000:]
    return m.group(1)


def test_train_lr_scheduler_reaches_the_trainer() -> None:
    """c7 declares train.lr_scheduler: cosine and nothing under protocol.scheduler.

    Before the fix this rendered `--lr-scheduler "none"`; the run then trained at
    a constant LR while its config said cosine.
    """
    cmd = _command(_render(CFM_TRAIN_ONLY))
    assert '--lr-scheduler "cosine"' in cmd, cmd


def test_protocol_scheduler_still_wins() -> None:
    """c8 declares both, at `none`; the launcher must forward the declared value."""
    cmd = _command(_render(CFM_BOTH))
    assert '--lr-scheduler "none"' in cmd, cmd


def test_epochs_line_reads_train_when_protocol_omits_it() -> None:
    """c7 declares train.epochs: 18 and no protocol.epochs.

    The display line used to say 50, the launcher's default.
    """
    out = _render(CFM_TRAIN_ONLY)
    m = re.search(r"^\s+epochs=(\S+)$", out, re.MULTILINE)
    assert m, out[-3000:]
    assert m.group(1) == "18", m.group(1)


def test_train_resume_from_reaches_the_trainer() -> None:
    """c7 declares train.resume_from: none; `auto` is the launcher's default.

    Emitting `auto` over `none` would let a stale checkpoint in the output
    directory silently become the starting point of a "cold" run.
    """
    cmd = _command(_render(CFM_TRAIN_ONLY))
    assert '--resume-from "none"' in cmd, cmd


def test_undeclared_scheduler_emits_no_flag(tmp_path: Path) -> None:
    """With neither key declared the launcher must emit nothing at all.

    A temporary copy of c8 with every scheduler key removed stands in for the
    142 configs in this repo that declare no schedule.  The trainer's own
    argparse default is the correct outcome; a launcher default is not.
    """
    import yaml

    data = yaml.safe_load(CFM_BOTH.read_text())
    data["train"].pop("lr_scheduler", None)
    data["train"].pop("lr_warmup_epochs", None)
    data["protocol"].pop("scheduler", None)
    # Keep the launcher's pre-flight happy: it requires these four sections.
    tmp_cfg = PROJECT_ROOT / "configs" / f"_tmp_wfa108i_{tmp_path.name}.yaml"
    try:
        tmp_cfg.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        cmd = _command(_render(tmp_cfg))
        assert "--lr-scheduler" not in cmd, cmd
        assert "--lr-warmup-epochs" not in cmd, cmd
    finally:
        tmp_cfg.unlink(missing_ok=True)
