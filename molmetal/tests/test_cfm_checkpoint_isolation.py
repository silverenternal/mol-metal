"""A run's checkpoints must live with that run.

2026-10-04 — `--ckpt-dir` defaulted to the repo-global `molmetal/checkpoints`
regardless of `--out-dir`.  MEASURED consequence: the CFM module's own
liveness test runs the trainer for ONE 32-sample epoch with
`--out-dir <pytest tmp>` and no `--ckpt-dir`, so it wrote a 1-epoch/32-sample
checkpoint over the shipped `cfm_yuelbond_wasserstein.pt` and
`..._seed0.pt`.  Both files on disk carried
`out_dir=/tmp/pytest-of-hugo/...` in their saved args, and the Wf-124 P2
50-epoch weights they previously held are not in git and have no surviving
copy — every `metrics/by_metric/cfm_decode_real.json` claim cites that path.

The second half is why a test-only fix is not enough: `--resume-from`
defaults to `auto`, which scans `ckpt_dir`.  A smoke writing into the shared
directory therefore poisons the NEXT real launch, which silently resumes from
the smoke's weights.  The D-MPNN trainer already fixed this class in Wf-A10;
these guards keep the CFM side fixed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)
SHARED_DIR = "molmetal/checkpoints"


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("_t_ckpt_iso", str(TRAINER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_ckpt_iso"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ckpt_dir_has_no_shared_default(trainer, monkeypatch):
    """The parser default must be None, and resolve under --out-dir."""
    # This trainer's parse_args() takes no argv argument — it reads sys.argv.
    monkeypatch.setattr(sys, "argv", ["train_cfm_yuelbond_wasserstein.py"])
    args = trainer.parse_args()
    assert args.ckpt_dir is None, (
        f"--ckpt-dir defaults to {args.ckpt_dir!r}.  A non-None default is a "
        "SHARED directory: any smoke run that omits the flag writes there, and "
        "--resume-from auto then reads it back on the next real launch."
    )


def test_resolution_follows_out_dir():
    """Source guard: the resolved path must derive from out_dir.

    Asserted on the source because the resolution is inline in `main()` with
    no callable boundary — the same reason
    `test_cfm_bond_target_truth.py` asserts on source.
    """
    src = TRAINER_PATH.read_text()
    assert "ckpt_dir = out_dir / \"checkpoints\"" in src, (
        "the unset --ckpt-dir path no longer derives from --out-dir; runs "
        "would share a checkpoint directory again"
    )
    assert f'default="{SHARED_DIR}"' not in src, (
        f"--ckpt-dir is pinned back to the repo-global {SHARED_DIR!r}"
    )


def test_no_cfm_config_pins_the_shared_directory():
    """Configs must not point output.ckpt_dir back at the shared dir.

    start.sh passes `output.ckpt_dir` straight through as `--ckpt-dir`, so a
    yaml value overrides the trainer default and re-opens the whole class.
    """
    offenders = []
    for cfg in sorted((PROJECT_ROOT / "configs").glob("cfm*.yaml")):
        for i, line in enumerate(cfg.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("ckpt_dir:"):
                continue
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value == SHARED_DIR:
                offenders.append(f"{cfg.name}:{i}")
    assert not offenders, (
        f"config(s) pin a run's checkpoints to the shared {SHARED_DIR}: "
        f"{offenders}. Point them at the run's own directory instead — "
        "start.sh forwards this value as --ckpt-dir."
    )


def test_the_liveness_probe_passes_its_own_ckpt_dir():
    """The test that caused this must not rely on the default for safety."""
    src = (
        PROJECT_ROOT
        / "molmetal"
        / "tests"
        / "test_cfm_loss_gradient_liveness.py"
    ).read_text()
    assert '"--ckpt-dir"' in src, (
        "test_cfm_loss_gradient_liveness executes real training; it must "
        "pass --ckpt-dir (belt and braces with the trainer default) so a "
        "future default change cannot silently reintroduce the overwrite"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_no_test_writes_into_the_shared_checkpoint_dir(tmp_path: Path) -> None:
    """A test that runs real training must isolate its ckpt_dir.

    2026-10-05 — a 1-epoch / 32-sample checkpoint was found sitting at
    ``molmetal/checkpoints/cfm_yuelbond_wasserstein_seed0.pt``, the default
    that ``de_novo_loop.py`` and three other scripts load.  Its saved args
    read ``out_dir=/tmp/pytest-of-hugo/pytest-5024/...`` — it was a test
    artifact wearing a production filename, so anything using the default
    would have silently loaded a toy model.

    The fix (2026-10-04, f7fec1f360) added ``--ckpt-dir`` to the gradient
    probe.  This test pins it, because the failure is silent: the directory
    is untracked, so nothing in git history records that it happened.
    """
    probe = (
        PROJECT_ROOT / "molmetal" / "tests" / "test_cfm_loss_gradient_liveness.py"
    ).read_text()
    assert '"--ckpt-dir"' in probe, (
        "the gradient probe runs a real training epoch; without --ckpt-dir it "
        "writes into molmetal/checkpoints and overwrites whatever production "
        "checkpoint carries that name"
    )
    assert '"--resume-from"' in probe, (
        "the probe must not inherit a previous run's weights via --resume-from auto"
    )
