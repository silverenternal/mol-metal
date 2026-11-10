#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Generate the AGENTS.md §5.11 static launch configs for Wf-A269 cohort c1.

WHAT THIS IS
------------
Cohort ``c1_dmpnn_pair_sweep`` owns five launchers that all drive the SAME
trainer (``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``) over a
(metal, seed) sweep:

    scripts/launch_wf124_p3_g5.sh          5 metals x 3 seeds           = 15 tuples
    scripts/launch_wf124_p3_g6.sh          4 arms   x 5 metals x 3 seeds = 60 tuples
    scripts/launch_wf310b_g6_final.sh      4 arms   x 5 metals x 3 seeds = 60 tuples
    scripts/retry_g5_missing_pairs.sh     2 pairs                          =  2 tuples
    scripts/recover_g6_failed_tuples.sh   4 arms   x 5 metals x 3 seeds = 60 tuples

WHY ONE STATIC FILE PER TUPLE
-----------------------------
The trainer sweeps the cohort INTERNALLY --

    molmetal/scripts/train_dmpnn_multitask_wasserstein.py:4352
        for metal in args.metals:
    molmetal/scripts/train_dmpnn_multitask_wasserstein.py:4356
        for seed in args.seeds:

-- while every launcher starts ONE trainer PROCESS per (metal, seed) tuple,
each with its own ``--out-dir`` / ``--ckpt-dir``.  AGENTS.md §5.11 (user
directive 2026-10-09, "以后要求能且只能用yaml配置启动训练和推理，no cli参数")
lets a launcher pass only ``--config`` / ``--out-dir`` / ``--ckpt-dir`` /
``--help`` to the trainer, and this wave's contract forbids generating a YAML
at launch time ("the config is a file on disk").  The only shape that
satisfies both is one STATIC file per tuple, each pinning exactly one metal
and one seed, so the trainer's internal sweep runs exactly once per process
-- the same number of processes, the same number of trainings and the same
output layout as before the migration.

A single config carrying the whole grid would train every (metal, seed) pair
inside EVERY process (15x the work per process).  A config written at launch
time would put the cohort back into a shell variable -- the knob nobody
audits.  Both are rejected.

HOW THE VALUES ARE DERIVED (nothing here is hand-typed)
-------------------------------------------------------
For every tuple this script

  1. rebuilds the PRE-MIGRATION argv exactly as the launcher spelled it
     (``--config configs/dmpnn.yaml --metals M --seeds S --csv ... --data-csv
     ... --out-dir ... --ckpt-dir ...`` plus that launcher's COMMON_ARGS and,
     for the G6 family, the arm's ablation lever);
  2. evaluates it the way the trainer does -- ``parse_args(argv)`` followed by
     ``apply_yaml_overrides(parser, ns, "configs/dmpnn.yaml", argv=argv)`` --
     which yields the EFFECTIVE pre-migration namespace
     (explicit CLI flag > YAML > argparse default, AGENTS.md §5.11);
  3. writes a YAML whose every leaf carries that effective value, in the
     trainer's own dest names and in the loader's eight leaf sections;
  4. re-reads the written YAML through the trainer's parser and compares the
     two namespaces DEST BY DEST.  A single differing dest aborts the run.

So the configs cannot drift from the pre-migration behaviour: they are
generated from it and then checked against it.

USAGE
-----
    .venv/bin/python molmetal/reports/wfa269_yaml_migration/wfa269_c1_generate_configs.py
    .venv/bin/python molmetal/reports/wfa269_yaml_migration/wfa269_c1_generate_configs.py --check

``--check`` regenerates nothing and only verifies the files already on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path("/home/hugo/codes/mol-metal")
TRAINER_REL = "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
TRAINER = REPO / TRAINER_REL
BASE_CONFIG_REL = "configs/dmpnn.yaml"
BASE_CONFIG = REPO / BASE_CONFIG_REL
CSV = "/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

METALS = ["Ru", "Ir", "Os", "Rh", "Re"]
SEEDS = [0, 1, 2]
ARMS = {
    "no_multi_task": ["--single-task"],
    "no_wass": ["--wasserstein-weight", "0.0"],
    "no_kendall": ["--no-kendall"],
    "no_dann": ["--no-dann"],
}

G5_OUT = "results/wf124_p3_50epoch_full_sweep_g5"
G6_OUT = "results/wf124_p3_per_innovation_ablation_g6"

# ---------------------------------------------------------------------------
# The pre-migration COMMON_ARGS of each launcher, verbatim from
# molmetal/reports/wfa269_yaml_migration/<wrapper>.before.sh (HEAD).
# ---------------------------------------------------------------------------
COMMON_G5 = [
    "--epochs", "50",
    "--batch-size", "256",
    "--hidden", "300",
    "--depth", "3",
    "--lr", "1e-3",
    "--alpha", "0.5",
    "--wasserstein-weight", "0.05",
    "--use-l2-guard",
    "--ablation-config", "mtl_baseline",
    "--w2-gradient-mode", "sliced",
    "--n-mc", "256",
    "--device", "cuda:0",
    "--cuda-visible-devices", "0",
    "--cpu-num-workers", "8",
    "--data-csv", CSV,
]
COMMON_G6 = [
    "--epochs", "50",
    "--batch-size", "256",
    "--hidden", "300",
    "--depth", "3",
    "--lr", "1e-3",
    "--alpha", "0.5",
    "--w2-gradient-mode", "sliced",
    "--n-mc", "256",
    "--use-l2-guard",
    "--device", "cuda:0",
    "--cuda-visible-devices", "0",
    "--cpu-num-workers", "8",
    "--data-csv", CSV,
]
COMMON_310B = [
    "--epochs", "50",
    "--batch-size", "64",
    "--hidden", "300",
    "--depth", "3",
    "--lr", "1e-3",
    "--alpha", "0.5",
    "--w2-gradient-mode", "sliced",
    "--n-mc", "256",
    "--use-l2-guard",
    "--device", "cuda:0",
    "--cuda-visible-devices", "0",
    "--cpu-num-workers", "8",
    "--data-csv", CSV,
]
COMMON_RETRY = [
    "--epochs", "50",
    "--batch-size", "256",
    "--hidden", "300",
    "--depth", "3",
    "--lr", "1e-3",
    "--alpha", "0.5",
    "--wasserstein-weight", "0.05",
    "--use-l2-guard",
    "--ablation-config", "mtl_baseline",
    "--w2-gradient-mode", "sliced",
    "--n-mc", "256",
    # MEASURED 2026-11-03 from the pre-migration wrapper
    # (molmetal/reports/wfa269_yaml_migration/retry_g5_missing_pairs.before.sh):
    # the retry launcher's COMMON_ARGS does NOT carry --device, but its
    # trainer invocation DOES carry `--device cuda:0` (the per-pair GPU is
    # selected by the CUDA_VISIBLE_DEVICES command prefix, the device INDEX
    # by this flag).  It also does NOT carry --cuda-visible-devices, so that
    # dest keeps its argparse default ("" = inherit the shell env).
    "--device", "cuda:0",
    "--cpu-num-workers", "8",
    "--data-csv", CSV,
]
COMMON_RECOVER = [
    "--epochs", "50",
    "--batch-size", "256",
    "--hidden", "300",
    "--depth", "3",
    "--lr", "1e-3",
    "--alpha", "0.5",
    "--w2-gradient-mode", "sliced",
    "--n-mc", "256",
    "--use-l2-guard",
    "--device", "cuda:0",
    "--cuda-visible-devices", "0",
    "--cpu-num-workers", "8",
    "--data-csv", CSV,
]


# ---------------------------------------------------------------------------
# launcher specs: (tag, wrapper, common args, out_base, arm flags, extra)
# ---------------------------------------------------------------------------
def _specs() -> List[Dict[str, Any]]:
    return [
        {
            "tag": "g5",
            "wrapper": "scripts/launch_wf124_p3_g5.sh",
            "common": COMMON_G5,
            "out_base": G5_OUT,
            "arms": [None],
            "with_csv": True,
            "with_cuda_visible": True,
        },
        {
            "tag": "g6",
            "wrapper": "scripts/launch_wf124_p3_g6.sh",
            "common": COMMON_G6,
            "out_base": G6_OUT,
            "arms": list(ARMS),
            "with_csv": True,
            "with_cuda_visible": True,
        },
        {
            "tag": "wf310b",
            "wrapper": "scripts/launch_wf310b_g6_final.sh",
            "common": COMMON_310B,
            "out_base": G6_OUT,
            "arms": list(ARMS),
            "with_csv": True,
            "with_cuda_visible": True,
        },
        {
            "tag": "retry_g5",
            "wrapper": "scripts/retry_g5_missing_pairs.sh",
            "common": COMMON_RETRY,
            "out_base": G5_OUT,
            "arms": [None],
            "with_csv": True,
            "with_cuda_visible": False,
            "pairs": [("Ru", 2), ("Ir", 0)],
        },
        {
            "tag": "recover_g6",
            "wrapper": "scripts/recover_g6_failed_tuples.sh",
            "common": COMMON_RECOVER,
            "out_base": G6_OUT,
            "arms": list(ARMS),
            "with_csv": False,
            "with_cuda_visible": True,
        },
    ]


def _tuples(spec: Dict[str, Any]) -> List[Tuple[str | None, str, int]]:
    """(arm, metal, seed) for every tuple this launcher starts a process for."""
    if spec.get("pairs"):
        return [(None, m, s) for m, s in spec["pairs"]]
    out = []
    for arm in spec["arms"]:
        for metal in METALS:
            for seed in SEEDS:
                out.append((arm, metal, seed))
    return out


def _out_dir(spec: Dict[str, Any], arm: str | None, metal: str, seed: int) -> str:
    if arm is None:
        return f"{spec['out_base']}/{metal}/seed{seed}"
    return f"{spec['out_base']}/{arm}/{metal}/seed{seed}"


def _config_name(spec: Dict[str, Any], arm: str | None, metal: str, seed: int) -> str:
    parts = ["wfa269_c1_dmpnn_pair_sweep", spec["tag"]]
    if arm is not None:
        parts.append(arm)
    parts += [metal, f"seed{seed}"]
    return "_".join(parts) + ".yaml"


def _old_argv(spec: Dict[str, Any], arm: str | None, metal: str, seed: int) -> List[str]:
    """The pre-migration argv, exactly as the launcher spelled it."""
    argv = ["--config", BASE_CONFIG_REL, "--metals", metal, "--seeds", str(seed)]
    if spec["with_csv"]:
        argv += ["--csv", CSV]
    argv += ["--data-csv", CSV]
    argv += ["--out-dir", _out_dir(spec, arm, metal, seed)]
    argv += ["--ckpt-dir", _out_dir(spec, arm, metal, seed) + "/ckpts"]
    if arm is not None:
        argv += ARMS[arm]
    argv += list(spec["common"])
    return argv


# ---------------------------------------------------------------------------
# trainer parser / loader plumbing
# ---------------------------------------------------------------------------
def _capture_parser():
    name = "wfa269_c1_probe"
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts/_yaml_equivalence_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.capture_parser(TRAINER)


def _apply(parser, ns, cfg_rel: str, argv: List[str]):
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    return apply_yaml_overrides(parser, ns, str(REPO / cfg_rel), argv=list(argv))


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------
def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in ":#{}[],&*?|-<>=!%@`") or text.strip() != text:
        return '"' + text.replace('"', '\\"') + '"'
    if text.lower() in ("true", "false", "null", "yes", "no", "on", "off", "~"):
        return '"' + text + '"'
    try:
        float(text)
    except ValueError:
        pass
    else:
        return '"' + text + '"'
    return text


def _yaml_list(values: List[Any]) -> str:
    if not values:
        return "[]"
    return "\n" + "\n".join(f"    - {_yaml_scalar(v)}" for v in values)


def _render(cfg: Dict[str, Any], header: List[str]) -> str:
    out: List[str] = []
    out.extend(line if line.startswith("#") else f"# {line}" for line in header)
    for section, body in cfg.items():
        out.append("")
        if not isinstance(body, dict):
            # a top-level scalar leaf (the `csv` dest)
            out.append(f"{section}: {_yaml_scalar(body)}")
            continue
        out.append(f"{section}:")
        for key, value in body.items():
            if isinstance(value, list):
                out.append(f"  {key}:{_yaml_list(value)}")
            else:
                out.append(f"  {key}: {_yaml_scalar(value)}")
    return "\n".join(out) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# the per-tuple config
# ---------------------------------------------------------------------------
def _build_config(ns, spec, arm, metal, seed) -> Dict[str, Any]:
    """Map the effective pre-migration namespace onto the loader's sections.

    Every value comes from ``ns`` (the effective old namespace), never from a
    literal typed here.
    """

    def v(dest):
        return getattr(ns, dest)

    out_dir = _out_dir(spec, arm, metal, seed)
    ckpt_dir = out_dir + "/ckpts"

    cfg: Dict[str, Any] = {}
    cfg["model"] = {
        "hidden": v("hidden"),
        "depth": v("depth"),
        "use_triton_smiles_featurize": v("use_triton_smiles_featurize"),
        "use_triton_relu_mlp": v("use_triton_relu_mlp"),
        "use_triton_bond_head": v("use_triton_bond_head"),
        "pic50_ridge_readout": v("pic50_ridge_readout"),
    }
    train = {
        "epochs": v("epochs"),
        "batch_size": v("batch_size"),
        "lr": v("lr"),
        "weight_decay": v("weight_decay"),
        "max_norm": v("max_norm"),
        "ema_decay": v("ema_decay"),
        "use_ema": v("use_ema"),
        "lr_scheduler": v("lr_scheduler"),
        "lr_min_lr": v("lr_min_lr"),
        "lr_warmup_epochs": v("lr_warmup_epochs"),
        "alpha": v("alpha"),
        "single_task": v("single_task"),
        "kendall": v("kendall"),
        "dann": v("dann"),
    }
    if v("ablation_config") is not None:
        train["ablation_config"] = v("ablation_config")
    cfg["train"] = train
    cfg["data"] = {
        "csv": str(v("data_csv")),
        "metals": [metal],
        "seeds": [seed],
        "metal_coord_term": v("metal_coord_term"),
        "metal_coord_weight": v("metal_coord_weight"),
        "active_pic50_threshold": v("active_pic50_threshold"),
    }
    eval_block = {
        "wasserstein_weight": v("wasserstein_weight"),
        "use_wasserstein_2": v("use_wasserstein_2"),
        "use_wasserstein_w1": v("use_wasserstein_w1"),
        "w2_gradient_mode": v("w2_gradient_mode"),
        "use_triton_wasserstein_sliced": v("use_triton_wasserstein_sliced"),
        "use_ot_cfm": v("use_ot_cfm"),
        "triton_ot": v("triton_ot"),
        "use_l2_guard": v("use_l2_guard"),
        "n_mc": v("n_mc"),
    }
    cfg["eval"] = eval_block
    compute = {
        "device": v("device"),
        "tf32": v("tf32"),
        "torch_compile": v("torch_compile"),
        "cpu_num_workers": v("cpu_num_workers"),
        "pin_memory": v("pin_memory"),
    }
    if v("cuda_visible_devices") is not None:
        compute["cuda_visible_devices"] = str(v("cuda_visible_devices"))
    cfg["compute"] = compute
    cfg["protocol"] = {"seeds": [seed]}
    cfg["output"] = {
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "resume_from": str(v("resume_from")),
        "checkpoint_interval": v("checkpoint_interval"),
    }
    if v("csv") is not None:
        # TOP-LEVEL `csv:` -> the `csv` dest.  Under `data:` the leaf `csv`
        # resolves to `data_csv` instead, so the two flags need two keys.
        cfg["csv"] = str(v("csv"))
    cfg["directives"] = {
        "kill_then_adapt": True,
        "time_budget": "48h",
        "user_ok_required": True,
        "user_ok_granted": (
            "48h budget + RTX PRO 6000 Blackwell + project root results/ (2026-09-28)"
        ),
    }
    cfg["entry"] = {
        "script": TRAINER_REL,
        "command_template": (
            "/home/hugo/codes/mol-metal/.venv/bin/python "
            f"{TRAINER_REL} --config {{yaml_path}} --out-dir {{out_dir}}"
        ),
    }
    return cfg


def _header(spec, arm, metal, seed, argv: List[str]) -> List[str]:
    arm_txt = arm if arm else "base (no ablation arm)"
    return [
        f"configs/wfa269_c1_dmpnn_pair_sweep_{spec['tag']}"
        f"{'_' + arm if arm else ''}_{metal}_seed{seed}.yaml — AGENTS.md §5.11",
        f"# launch config for {spec['wrapper']} — tuple metal={metal} seed={seed}, arm={arm_txt}.",
        "#",
        "# WHY ONE STATIC FILE PER TUPLE: the trainer sweeps `args.metals` x",
        "# `args.seeds` INTERNALLY (train_dmpnn_multitask_wasserstein.py:4352",
        "# `for metal in args.metals:` / :4356 `for seed in args.seeds:`) while this",
        "# launcher starts ONE trainer PROCESS per (metal, seed) tuple, each with",
        "# its own --out-dir.  §5.11 allows the launcher to pass only --config /",
        "# --out-dir / --ckpt-dir / --help, and this wave's contract forbids",
        "# generating a YAML at launch time — so the tuple coordinates are pinned",
        "# in a STATIC file, one per tuple.  One config for the whole grid would",
        "# train every pair inside every process; a launch-time-written config",
        "# would put the cohort back into a shell variable.",
        "#",
        "# VALUES: every leaf below is the EFFECTIVE pre-migration value —",
        "# configs/dmpnn.yaml overridden by the launcher's own command line",
        "# (explicit CLI flag > YAML > argparse default, AGENTS.md §5.11).",
        "# Generated + verified dest-by-dest by",
        "# molmetal/reports/wfa269_yaml_migration/wfa269_c1_generate_configs.py",
        "# (MEASURED 2026-11-03); nothing here is hand-typed.",
        "#",
        "# PRE-MIGRATION ARGV (molmetal/reports/wfa269_yaml_migration/"
        f"{Path(spec['wrapper']).stem}.before.sh):",
        "#   " + " ".join(argv),
        "#",
        "# SECTION SCHEMA: only protocol | model | flow | data | compute | eval |",
        "# output | train carry leaves; a leaf under any other top-level section",
        "# is SILENTLY IGNORED by molmetal/scripts/wfA2_config_loader.py.",
        "# DEST-NAME TRAPS: `data.csv` resolves to the dest `data_csv` (the flat",
        "# two-component leaf wins) — the `csv` dest needs the TOP-LEVEL `csv:`",
        "# key.  `--use-kendall`/`--no-kendall` share dest `kendall`;",
        "# `--use-dann`/`--no-dann` share dest `dann`; there is no `no_l2_guard`",
        "# dest; `--cuda-visible-devices` declares no type=, so it is the STRING",
        "# \"0\" (quoted on purpose).",
        "# LIST dests (`metals`, `seeds`) MUST be YAML sequences — a bare string",
        "# would match zero rows and yield test_auc = nan on every seed",
        "# (AGENTS.md §5.11, MEASURED 2026-10-09).",
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the files already on disk instead of writing them",
    )
    args = ap.parse_args(argv)

    parser = _capture_parser()
    n_written = 0
    n_checked = 0
    problems: List[str] = []

    for spec in _specs():
        for arm, metal, seed in _tuples(spec):
            old_argv = _old_argv(spec, arm, metal, seed)
            # the effective pre-migration namespace, exactly as the trainer
            # evaluates it
            ns = parser.parse_args(list(old_argv))
            ns = _apply(parser, ns, BASE_CONFIG_REL, old_argv)

            cfg = _build_config(ns, spec, arm, metal, seed)
            name = _config_name(spec, arm, metal, seed)
            path = REPO / "configs" / name
            text = _render(cfg, _header(spec, arm, metal, seed, old_argv))

            if not args.check:
                _write_atomic(path, text)
                n_written += 1

            # ---- verify: the written config reproduces the old namespace ----
            new_ns = parser.parse_args(["--config", str(path)])
            new_ns = _apply(parser, new_ns, f"configs/{name}", ["--config", str(path)])
            for dest in sorted(vars(ns)):
                if dest == "config" or dest.startswith("yaml_"):
                    continue
                old_value = getattr(ns, dest)
                new_value = getattr(new_ns, dest, "<MISSING>")
                if type(old_value) is not type(new_value) or old_value != new_value:
                    problems.append(
                        f"{name}: dest {dest}: old={old_value!r} ({type(old_value).__name__}) "
                        f"new={new_value!r} ({type(new_value).__name__})"
                    )
            n_checked += 1

    print(f"[wfa269-c1] configs written: {n_written}, verified: {n_checked}")
    if problems:
        print(f"[wfa269-c1] {len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print(f"  {p}")
        return 1
    print("[wfa269-c1] every config reproduces its pre-migration namespace dest by dest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
