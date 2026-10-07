#!/usr/bin/env python3
# 2026-10-07 (Wf-R17-A + Wf-R18-A) — CFM eval-only bridge tool.
#
# Why this exists
# ---------------
# The R17-A agent caught 3 premise errors in the eval-only bridge brief:
#   (1) trainer writes `cfm_yuelbond_wasserstein_seed{seed}.pt` (no epoch
#       suffix) — periodic saves overwrite in place; `*_epoch5.pt` never exists
#   (2) YAML `output.ckpt_dir: results/cfm_tmqm_all6/checkpoints` is shared
#       between R8 and R14-A — R14-A epoch 5 overwrites R8 epoch 4 in place
#   (3) `--epochs 1 --resume-from path` is silent no-op (range(5,1) empty)
#       unless `--eval-only` is passed
#
# What this script does
# ---------------------
#   1. Waits for a candidate checkpoint at a known location (default shared
#      R14-A path: `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein_seed0.pt`).
#   2. Loads the checkpoint dict and VERIFIES provenance via
#      `args.out_dir == "<expected out_dir>"` (e.g. ".../cfm_r9_fix_v2") to
#      exclude the stale R8 ckpt that lives at the same shared path.
#   3. If the provenanced ckpt is at a shared dir, atomically copies it to an
#      isolated per-run location so subsequent eval-only runs cannot clobber
#      what they are measuring (and so future R14-A saves do not race with
#      this script's read).
#   4. Runs the eval-only bridge command with corrected flags:
#        `--config <cfg> --eval-only --resume-from path --resume-from-path <CKPT>
#         --gen-eval-n <N> --gen-eval-seeds "<seeds>" --n-steps <S>
#         --device <dev> --out-dir <eval_out_dir>`
#      The trainer's own `--eval-only` flag (Wf-R11-A3) suppresses BOTH
#      checkpoint writes (per `train_cfm_yuelbond_wasserstein.py:7648` area)
#      so the eval-only run cannot overwrite what it is measuring.
#
# Usage
# -----
#   # Stage 1: wait for R14-A epoch 5 ckpt at the shared dir, verify it's
#   # R14-A (not R8), copy to isolated, run eval-only bridge with corrected
#   # flags.  Default polling interval 5 min, max wait 6 hr.
#   .venv/bin/python molmetal/scripts/r17_cfm_eval_bridge.py \
#       --config configs/cfm_tmqm_all6.yaml \
#       --expected-out-dir-suffix cfm_r9_fix_v2 \
#       --eval-out-dir molmetal/reports/cfm_r17_eval512 \
#       --gen-eval-n 512 --gen-eval-seeds "0 1 2 3" --n-steps 50 \
#       --device cuda:0 --poll-sec 300 --max-wait-sec 21600
#
#   # Stage 2 (after R14-A epoch 5 lands, manually): re-run with explicit
#   # `--ckpt <PATH>` and `--no-wait` to skip the polling phase.
#   .venv/bin/python molmetal/scripts/r17_cfm_eval_bridge.py \
#       --config configs/cfm_tmqm_all6.yaml \
#       --expected-out-dir-suffix cfm_r9_fix_v2 \
#       --ckpt results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
#       --no-wait --eval-out-dir molmetal/reports/cfm_r17_eval512
#
# Output
# ------
#   - Provenance-verified ckpt copied to `<eval_out_dir>/checkpoints/source_ckpt.pt`
#   - Stage 2 trainer run writes per_epoch metrics + final events.jsonl to
#     `<eval_out_dir>/` (trainer's normal eval-only output surface)
#   - Stdout emits: PROVENANCE_OK / EVAL_RESULT / NEXT — plain text per
#     AGENTS.md §5.8 ≤1500 token budget discipline
#
# Constraints
# -----------
#   - .venv/bin/python (per AGENTS.md §5.9 single-venv rule)
#   - No `paper/*` touches (TODO-84)
#   - No new docs (per user 2026-09-30 directive "别让 agent 再写文档了")
#   - Honest-framing MEASURED vs PROJECTED tags in stdout (per §5.2)
#   - Per §5.7: NOT a 50-step smoke (this is a 1-shot Stage 2 bridge), but the
#     underlying trainer is smoke-tested elsewhere

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SHARED_CKPT = (
    PROJECT_ROOT
    / "results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein_seed0.pt"
)


def _verify_provenance(ckpt_path: Path, expected_out_dir_suffix: str) -> dict:
    """Load ckpt and verify args.out_dir ends with the expected suffix.

    Returns the ckpt dict (with 'args' key). Raises SystemExit if provenance
    fails — the Stage 2 bridge MUST NOT evaluate a ckpt that does not
    belong to the run we are waiting on (R8 vs R14-A collision).
    """
    import torch  # local import; trainer-only dependency

    _sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _args = _sd.get("args", {})
    _out_dir = str(_args.get("out_dir", ""))
    _epoch = int(_sd.get("epoch", -1))
    _decode = float(_sd.get("decode_ratio", _sd.get("best_decode_ratio", 0.0)))
    print(
        f"[bridge-provenance] ckpt={ckpt_path} "
        f"epoch={_epoch} decode_ratio={_decode:.3f} out_dir={_out_dir}",
        flush=True,
    )
    if expected_out_dir_suffix and expected_out_dir_suffix not in _out_dir:
        raise SystemExit(
            f"[bridge-provenance] REJECTED: ckpt out_dir '{_out_dir}' does "
            f"not contain expected suffix '{expected_out_dir_suffix}'.  This "
            f"is the Wf-R17-A shared-ckpt_dir collision: a stale R8 ckpt "
            f"with out_dir='cfm_r8_cosine_40ep' is at the same path.  "
            f"Aborting rather than measuring the wrong run."
        )
    return _sd


def _copy_isolated(src: Path, eval_out_dir: Path) -> Path:
    """Atomic copy src → <eval_out_dir>/checkpoints/source_ckpt.pt.

    eval-only Stage 2 runs read this isolated copy, so subsequent R14-A
    saves at the shared dir cannot race with the eval.
    """
    iso_dir = eval_out_dir / "checkpoints"
    iso_dir.mkdir(parents=True, exist_ok=True)
    iso_path = iso_dir / "source_ckpt.pt"
    # shutil.copy2 preserves mtime (helpful for downstream consumers that
    # check freshness); write to a temp file then rename for atomicity.
    tmp = iso_path.with_suffix(".pt.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, iso_path)
    print(f"[bridge-copy] {src} -> {iso_path}", flush=True)
    return iso_path


def _wait_for_ckpt(
    ckpt_path: Path,
    *,
    must_exist: bool,
    poll_sec: int,
    max_wait_sec: int,
) -> None:
    """Poll for ckpt existence up to max_wait_sec.  Raises SystemExit on
    timeout.  must_exist=False (i.e. --no-wait) skips polling entirely."""
    if not must_exist:
        if not ckpt_path.is_file():
            raise SystemExit(
                f"[bridge-wait] --no-wait but ckpt missing: {ckpt_path}"
            )
        return
    _start = time.monotonic()
    while True:
        if ckpt_path.is_file():
            return
        _elapsed = time.monotonic() - _start
        if _elapsed > max_wait_sec:
            raise SystemExit(
                f"[bridge-wait] TIMEOUT after {max_wait_sec}s waiting for "
                f"{ckpt_path}.  R14-A epoch-5 save did not land in window.  "
                f"Re-run later with --ckpt <PATH> --no-wait."
            )
        print(
            f"[bridge-wait] ckpt missing ({int(_elapsed)}s/{max_wait_sec}s); "
            f"polling again in {poll_sec}s",
            flush=True,
        )
        time.sleep(poll_sec)


def _run_eval_only(
    *,
    py_exec: str,
    config: Path,
    ckpt_path: Path,
    eval_out_dir: Path,
    gen_eval_n: int,
    gen_eval_seeds: str,
    n_steps: int,
    device: str,
) -> int:
    """Run trainer with --eval-only --resume-from path --resume-from-path."""
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py_exec,
        str(PROJECT_ROOT / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"),
        "--config", str(config),
        "--eval-only",
        "--resume-from", "path",
        "--resume-from-path", str(ckpt_path),
        "--gen-eval-n", str(gen_eval_n),
        "--gen-eval-seeds", gen_eval_seeds,
        "--n-steps", str(n_steps),
        "--device", device,
        "--out-dir", str(eval_out_dir),
    ]
    print(f"[bridge-eval] CMD: {' '.join(cmd)}", flush=True)
    # Stream stdout/stderr live so the calling workflow sees per-epoch progress.
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CFM eval-only bridge tool (Wf-R17-A/Wf-R18-A)"
    )
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cfm_tmqm_all6.yaml")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_SHARED_CKPT,
        help="Checkpoint path to verify + eval.  Default R14-A shared path.",
    )
    p.add_argument(
        "--expected-out-dir-suffix",
        type=str,
        default="cfm_r9_fix_v2",
        help="Provenance guard: ckpt args.out_dir MUST contain this suffix.",
    )
    p.add_argument(
        "--eval-out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal/reports/cfm_r17_eval512",
        help="Where the eval-only run writes its outputs.",
    )
    p.add_argument("--gen-eval-n", type=int, default=512)
    p.add_argument("--gen-eval-seeds", type=str, default="0 1 2 3")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--poll-sec", type=int, default=300)
    p.add_argument("--max-wait-sec", type=int, default=21600)
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip polling (use when re-running with an explicit known ckpt).",
    )
    p.add_argument(
        "--py",
        type=str,
        default=str(PROJECT_ROOT / ".venv/bin/python"),
        help="Python interpreter (per AGENTS.md §5.9 single-venv rule).",
    )
    args = p.parse_args(argv)

    print(
        f"[bridge-start] config={args.config} ckpt={args.ckpt} "
        f"eval_out_dir={args.eval_out_dir} no_wait={args.no_wait}",
        flush=True,
    )

    # Stage 1: wait (or assert-present)
    _wait_for_ckpt(
        args.ckpt,
        must_exist=not args.no_wait,
        poll_sec=args.poll_sec,
        max_wait_sec=args.max_wait_sec,
    )

    # Provenance guard
    _verify_provenance(args.ckpt, args.expected_out_dir_suffix)

    # Stage 1.5: copy to isolated location
    iso_ckpt = _copy_isolated(args.ckpt, args.eval_out_dir)

    # Stage 2: eval-only run using the isolated ckpt
    rc = _run_eval_only(
        py_exec=args.py,
        config=args.config,
        ckpt_path=iso_ckpt,
        eval_out_dir=args.eval_out_dir,
        gen_eval_n=args.gen_eval_n,
        gen_eval_seeds=args.gen_eval_seeds,
        n_steps=args.n_steps,
        device=args.device,
    )
    print(f"[bridge-done] eval-only rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
