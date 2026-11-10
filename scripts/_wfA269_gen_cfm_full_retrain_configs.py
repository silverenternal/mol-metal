#!/usr/bin/env python
"""Wf-A269 (§5.11 repair) — generate the per-(pocket, seed) CFM retrain configs.

WHY THIS EXISTS
---------------
``results/cfm_full_retrain/launch_loop.sh`` used to pass ``--epochs 50
--hidden-dim 128 --n-layers 3 --pocket-id <pocket> --seed <seed>`` on the
trainer invocation line.  AGENTS.md §5.11 permits a launch wrapper to pass
ONLY ``--config`` / ``--out-dir`` / ``--ckpt-dir`` / ``--help`` — flags that
select WHICH run, never WHAT to train.  Three of the five flags described
the model (they belong in the YAML); the other two varied per invocation
(so a per-run value cannot live in a shared config).

The compliant shape is therefore a STATIC YAML family: one file per
(pocket, seed) cell of the sweep, each carrying its own ``seed`` and
``pocket_id``.  Both keys resolve to the CFM trainer's own argparse dests
(``--seed`` and ``--pocket-id``), so NO trainer flag is added — §5.7's
50-step CPU smoke is not triggered, because nothing new was wired in.

WHAT THIS SCRIPT DOES
---------------------
It copies ``configs/cfm_full_retrain.yaml`` (comments preserved verbatim —
this is a TEXT transform, not a yaml.dump) and applies three exact,
asserted replacements:

  1. inserts a provenance header + ``seed:`` + ``pocket_id:`` (top level,
     so the leaf name IS the argparse dest — no alias fallback needed);
  2. repoints ``output.base_dir`` / ``output.ckpt_dir`` at this run's own
     directory (a shared ckpt_dir would let two pockets with the same seed
     overwrite the same ``cfm_yuelbond_wasserstein_seed{N}.pt``);
  3. rewrites ``entry.command_template`` to the §5.11-legal flag set.

Every replacement is a fail-closed exact-string match: if the base file
ever changes shape, this script RAISES instead of writing a wrong config.

RUN
---
    cd /home/hugo/codes/mol-metal
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/_wfA269_gen_cfm_full_retrain_configs.py

Idempotent: re-running rewrites the same bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

BASE = REPO_ROOT / "configs" / "cfm_full_retrain.yaml"
OUT_DIR = REPO_ROOT / "configs"

#: The 5 pockets the launch loop enumerates.  Identical to
#: ``molmetal/molmetal_lam/reward/pocket_closure_bias.py::SUPPORTED_POCKETS``
#: (MEASURED 2026-11-04: the tuple is exactly ("1h36","3eml","3ny8","4aaw","4rlu"),
#: every pocket the loop names is a KNOWN pocket, so ``pocket_id`` never hits
#: the "unknown pocket_id" branch.
POCKETS: tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Where the repaired wrapper puts this run's artefacts.  Identical to the
#: out-dir the pre-repair wrapper used (``results/cfm_full_retrain/<pocket>/
#: seed<seed>``), so the run's output LOCATION is unchanged by the repair.
RUN_ROOT = "results/cfm_full_retrain"

BASE_OLD = "  base_dir: results/cfm_full_retrain_v2"
BASE_NEW_TMPL = "  base_dir: {run_dir}"
CKPT_OLD = "  ckpt_dir: results/cfm_full_retrain_v2/checkpoints"
CKPT_NEW_TMPL = "  ckpt_dir: {run_dir}/checkpoints"
#: The repaired wrapper appends every run to ONE master log, at
#: ``results/cfm_full_retrain/master.log`` (the pre-repair wrapper did the
#: same).  The base config points at the ``_v2`` tree, which the repaired
#: loop never writes to, so the derived configs are repointed at the log the
#: loop actually produces.
LOG_OLD = "  log_master: results/cfm_full_retrain_v2/master.log"
LOG_NEW = "  log_master: results/cfm_full_retrain/master.log"

OLD_TEMPLATE = (
    "  command_template: /home/hugo/codes/mol-metal/.venv/bin/python "
    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py --config {yaml_path} "
    "--seed {seed} --out-dir {out_dir}"
)
NEW_TEMPLATE = (
    "  command_template: /home/hugo/codes/mol-metal/.venv/bin/python "
    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py --config {yaml_path} "
    "--out-dir {out_dir} --ckpt-dir {ckpt_dir}"
)

HEADER_TMPL = """\
# =============================================================================
# Wf-A269 (2026-11-04) — §5.11 REPAIR: per-(pocket, seed) config, GENERATED.
#
# DO NOT HAND-EDIT: regenerate with
#   CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/_wfA269_gen_cfm_full_retrain_configs.py
#
# Base: configs/cfm_full_retrain.yaml — this file is that config plus the two
# per-run identity keys below, the per-run output paths, and the §5.11-legal
# entry.command_template.  The base's comments are preserved VERBATIM.
#
# WHY the two keys exist: AGENTS.md §5.11 lets a wrapper pass only --config /
# --out-dir / --ckpt-dir / --help, so a value that VARIES per invocation cannot
# be passed as a flag.  The pre-repair wrapper passed --pocket-id and --seed on
# the invocation line (results/cfm_full_retrain/launch_loop.sh, pre-repair).
#
# Both keys resolve to the CFM trainer's own argparse dests (no new flag, so
# §5.7's 50-step CPU smoke is NOT triggered):
#   seed      -> --seed       (int)  train_cfm_yuelbond_wasserstein.py:15227
#   pocket_id -> --pocket-id  (str)  train_cfm_yuelbond_wasserstein.py:15221
#
# pocket: {pocket}   seed: {seed}
# =============================================================================
seed: {seed}
pocket_id: {pocket}
"""


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Fail-closed exact-string replacement."""
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"[wfA269-gen] FATAL: expected exactly 1 occurrence of {what}, "
            f"found {n}.  The base config changed shape; fix the generator "
            f"before regenerating."
        )
    return text.replace(old, new)


def build(base_text: str, pocket: str, seed: int) -> str:
    run_dir = f"{RUN_ROOT}/{pocket}/seed{seed}"

    # 1. provenance header + the two per-run identity keys, in front of `data:`.
    anchor = "\ndata:\n"
    if base_text.count(anchor) != 1:
        raise SystemExit(
            "[wfA269-gen] FATAL: the base config has no unique top-level "
            "`data:` anchor; refusing to write a derived config."
        )
    header = HEADER_TMPL.format(pocket=pocket, seed=seed)
    text = base_text.replace(anchor, "\n" + header + "\ndata:\n", 1)

    # 2. per-run output paths.
    text = _replace_once(text, BASE_OLD, BASE_NEW_TMPL.format(run_dir=run_dir),
                         "output.base_dir")
    text = _replace_once(text, CKPT_OLD, CKPT_NEW_TMPL.format(run_dir=run_dir),
                         "output.ckpt_dir")
    text = _replace_once(text, LOG_OLD, LOG_NEW, "output.log_master")

    # 3. §5.11-legal command_template.
    text = _replace_once(text, OLD_TEMPLATE, NEW_TEMPLATE,
                         "entry.command_template")
    return text


def main() -> int:
    import yaml

    base_text = BASE.read_text(encoding="utf-8")
    written = []
    for pocket in POCKETS:
        for seed in SEEDS:
            text = build(base_text, pocket, seed)
            # Fail closed BEFORE writing: the file must parse and must carry
            # the values this generator claims.
            data = yaml.safe_load(text)
            assert isinstance(data, dict), "derived config did not parse to a map"
            assert data["seed"] == seed and isinstance(data["seed"], int), (
                f"seed did not round-trip for {pocket}/{seed}"
            )
            assert data["pocket_id"] == pocket and isinstance(data["pocket_id"], str), (
                f"pocket_id did not round-trip for {pocket}/{seed}"
            )
            assert data["protocol"]["epochs"] == 50
            assert data["model"]["hidden_dim"] == 128
            assert data["model"]["n_layers"] == 3
            assert data["output"]["base_dir"] == f"{RUN_ROOT}/{pocket}/seed{seed}"
            out = OUT_DIR / f"cfm_full_retrain_{pocket}_seed{seed}.yaml"
            atomic_write_file(out, text)
            written.append(out)
    print(f"[wfA269-gen] wrote {len(written)} config(s) into {OUT_DIR}")
    for p in written:
        print(f"[wfA269-gen]   {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
