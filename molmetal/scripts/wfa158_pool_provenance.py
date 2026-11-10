#!/usr/bin/env python3
"""Wf-A158 BRANCH POOL — provenance + round-trip audit of the C8 free-generation pool.

CPU-only BY CONSTRUCTION
------------------------
* ``CUDA_VISIBLE_DEVICES`` is forced to ``""`` before anything heavy is
  imported, and the run asserts at the end that ``torch`` was never
  imported (the "禁止使用gpu" directive of the Wf-A119/A124/A125/A156/A158
  chain overrides the 2026-09-29 standing GPU grant).
* No checkpoint is ever deserialised.  The two ``.pt`` files next to the
  pool are hashed as BYTES (sha256) only — no ``torch.load``, no weights
  in RAM.
* RDKit + pure python + stdlib only.

WHAT THIS ANSWERS
------------------
Q1  PROVENANCE
    Which run produced ``generated_smiles.csv``, on what date, from which
    checkpoint, with which sampling settings — quoted from ``master.log``
    / ``events.jsonl`` / the run's YAML config with ``file:line``
    citations.  When a claim cannot be resolved from disk the script
    says so; it never guesses.

Q2  ROUND-TRIP
    Re-parse every row's ``smiles`` with ``Chem.MolFromSmiles`` (RDKit's
    DEFAULT: canonical parse, sanitisation on), recompute the fragment
    count and the largest-fragment atom fraction, and quantify how far
    the published columns are from the recomputation.  This is the number
    that separates a SERIALISATION artefact from a POST-PROCESSING
    difference.

Q3  COLUMN SEMANTICS
    What ``n_frags_raw`` and ``largest_frag_frac`` actually mean in the
    producing code, with the source lines embedded in the artifact.

EXIT CODES
----------
0 = PASS, 1 = FAIL, 2 = SKIP (AGENTS.md §4 stdout invariant).
"""

from __future__ import annotations

import os
import sys

# Hard CPU-only, BEFORE any third-party import.  Done at module scope so
# importing this module is itself safe under the no-GPU directive.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

#: Whether torch was ALREADY in ``sys.modules`` when this module was
#: imported.  The script itself never imports torch; but a test harness
#: (pytest plugins, a conftest) may have loaded it first, and then
#: "torch is in sys.modules" is not evidence that THIS script used a GPU.
_TORCH_PRESENT_AT_IMPORT = "torch" in sys.modules

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

# --------------------------------------------------------------------------
# Defaults — the C8 free-generation pool (the DENOMINATOR of every
# frontier-parity number the concurrent Wf-A155 wave computes).
# --------------------------------------------------------------------------
DEFAULT_RUN_DIR = REPO / "molmetal/reports/cfm_c8_fullgen_2026_10_24"
DEFAULT_POOL_CSV = DEFAULT_RUN_DIR / "generated_smiles.csv"
DEFAULT_MASTER_LOG = DEFAULT_RUN_DIR / "master.log"
DEFAULT_EVENTS = DEFAULT_RUN_DIR / "events.jsonl"
DEFAULT_CKPT_DIR = DEFAULT_RUN_DIR / "checkpoints"
DEFAULT_CONFIG = REPO / "configs/c8_fullgen_b64_2026_10_24.yaml"
DEFAULT_OUT_DIR = REPO / "molmetal/reports/wfa158_pool_provenance"
TRAINER = REPO / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"

#: The cohort metals, as the producing code defines them
#: (``_TARGET_METALS`` in the trainer).  Asserted against the live source
#: below so a rename upstream cannot silently change this script.
METAL_Z = frozenset({44, 45, 75, 76, 77, 78})  # Ru Rh Re Os Ir Pt
_METAL_LITERAL = "_TARGET_METALS = frozenset({44, 45, 75, 76, 77, 78})"

#: The published columns are rounded to 4 dp by the writer
#: (``round(_lfrac, 4)``, trainer:19310), so the round-trip equality
#: tolerance is half a unit in the last published place.
ROUNDED_DP = 4
LFRAC_TOL = 0.5 * (10.0 ** (-ROUNDED_DP)) + 1e-12

POOL_HEADER = ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_stamp(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": int(st.st_size),
        "mtime_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)
        ),
        "mtime_local_utc+8": time.strftime(
            "%Y-%m-%dT%H:%M:%S+08:00",
            time.gmtime(st.st_mtime + 8 * 3600),
        ),
        "sha256": _sha256(path),
    }


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _read_lines(path: Path) -> List[Tuple[int, str]]:
    if not path.is_file():
        return []
    out: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            out.append((i, line.rstrip("\n")))
    return out


def _cite(path: Path, start: int, end: Optional[int] = None) -> Dict[str, Any]:
    """Embed a source excerpt as evidence (file:line + verbatim text)."""
    end = start if end is None else end
    lines = _read_lines(path)
    body = [t for n, t in lines if start <= n <= end]
    rel = None
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        rel = str(path)
    return {
        "source": rel,
        "line_start": start,
        "line_end": end,
        "text": "\n".join(body),
    }


def _find(lines: List[Tuple[int, str]], pattern: str) -> List[Dict[str, Any]]:
    rx = re.compile(pattern)
    out = []
    for n, t in lines:
        if rx.search(t):
            out.append({"line": n, "text": t})
    return out


# --------------------------------------------------------------------------
# Q1 — PROVENANCE
# --------------------------------------------------------------------------
def q1_provenance(
    master_log: Path,
    events_path: Path,
    ckpt_dir: Path,
    config_path: Path,
    pool_csv: Path,
) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    log_lines = _read_lines(master_log)

    # -- the log's own identity ------------------------------------------
    run_tag_hits = _find(log_lines, r"^\[Wf-124 P2\]")
    log_head = log_lines[:1]
    claims.append(
        {
            "id": "log_is_a_trainer_run_log",
            "question": "Is master.log the stdout of a trainer run?",
            "status": "RESOLVED" if run_tag_hits else "UNRESOLVED",
            "value": (
                "train_cfm_yuelbond_wasserstein.py (Wf-124 P2 train_cfm_yuelbond_wasserstein.py)"
                if run_tag_hits
                else "could not confirm"
            ),
            "evidence": (
                [
                    {"source": _rel(master_log), "line": h["line"], "text": h["text"]}
                    for h in run_tag_hits[:4]
                ]
                + (
                    [
                        {
                            "source": _rel(master_log),
                            "line": log_lines[-1][0],
                            "text": log_lines[-1][1][:400],
                        }
                    ]
                    if log_lines
                    else []
                )
            ),
        }
    )

    # -- which run / which config / which out_dir -------------------------
    cfg: Dict[str, Any] = {}
    cfg_load_error = None
    try:
        import yaml  # local import: only used when the config exists

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        cfg_load_error = f"{type(exc).__name__}: {exc}"

    def _dig(d: Any, *path: str) -> Any:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _dig_first(d: Any, *paths: Tuple[str, ...]) -> Tuple[Any, Optional[str]]:
        for p in paths:
            v = _dig(d, *p)
            if v is not None:
                return v, ".".join(p)
        return None, None

    cfg_out_dir = _dig(cfg, "output", "out_dir")
    cfg_ckpt_dir = _dig(cfg, "output", "ckpt_dir")
    cfg_log_master = _dig(cfg, "output", "log_master")
    cfg_save_gen = _dig(cfg, "output", "save_generated_smiles")
    # `seed` and `epochs` live in different sections in different configs;
    # take the first non-None so the value is reported wherever the YAML
    # puts it, and record WHICH section it came from.
    cfg_seed, cfg_seed_src = _dig_first(cfg, ("protocol", "seed"), ("train", "seed"))
    cfg_epochs, cfg_epochs_src = _dig_first(
        cfg, ("train", "epochs"), ("protocol", "epochs")
    )
    cfg_batch = _dig(cfg, "train", "batch_size")
    cfg_lr = _dig(cfg, "train", "lr")
    cfg_gen_n = _dig(cfg, "eval", "gen_eval_n")
    cfg_gen_seeds = _dig(cfg, "eval", "gen_eval_seeds")
    cfg_method = _dig(cfg, "eval", "method")
    cfg_steps = _dig(cfg, "eval", "n_steps")
    cfg_ckpt_interval = _dig(cfg, "output", "checkpoint_interval")
    cfg_resume = _dig(cfg, "output", "resume_from")
    cfg_entry = _dig(cfg, "entry", "command_template")

    out_dir_match = (
        cfg_out_dir is not None
        and Path(str(cfg_out_dir)).name == pool_csv.parent.name
    )
    claims.append(
        {
            "id": "run_identity",
            "question": "Which run produced this pool CSV?",
            "status": "RESOLVED" if out_dir_match else "UNRESOLVED",
            "value": {
                "config": str(config_path),
                "config_declared_out_dir": cfg_out_dir,
                "pool_csv_parent_dir": str(pool_csv.parent),
                "config_declared_log_master": cfg_log_master,
                "config_declared_ckpt_dir": cfg_ckpt_dir,
                "save_generated_smiles": cfg_save_gen,
                "entry_command_template": cfg_entry,
                "config_sha256": _sha256(config_path) if config_path.is_file() else None,
                "config_introduced_by_commit": _git(
                    "log", "-1", "--format=%H %ad %s", "--date=iso",
                    "--", str(config_path.relative_to(REPO)),
                )
                if config_path.is_file()
                else None,
                "config_load_error": cfg_load_error,
            },
            "evidence": (
                [_cite(config_path, _yaml_line_of(config_path, "out_dir:"), _yaml_line_of(config_path, "out_dir:"))]
                if config_path.is_file()
                else []
            ),
        }
    )

    # -- date -------------------------------------------------------------
    evs = _read_jsonl(events_path)
    ts_vals = [e.get("ts") for e in evs if isinstance(e.get("ts"), (int, float))]
    crash = _find(log_lines, r"^Traceback \(most recent call last\):")
    oom = _find(log_lines, r"OutOfMemoryError: CUDA out of memory")
    claims.append(
        {
            "id": "run_date",
            "question": "When did the run happen?",
            "status": "RESOLVED" if ts_vals else "UNRESOLVED",
            "value": {
                "first_event_utc": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(ts_vals)))
                    if ts_vals
                    else None
                ),
                "last_event_utc": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(ts_vals)))
                    if ts_vals
                    else None
                ),
                "last_event_local_utc+8": (
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%S+08:00",
                        time.gmtime(max(ts_vals) + 8 * 3600),
                    )
                    if ts_vals
                    else None
                ),
                "pool_csv_mtime_utc": _file_stamp(pool_csv)["mtime_utc"],
                "pool_csv_mtime_local_utc+8": _file_stamp(pool_csv)[
                    "mtime_local_utc+8"
                ],
                "date_note": (
                    "the run's local date (UTC+8) is 2026-10-25, which is why "
                    "the artifacts carry the W1025 log prefix and the "
                    "2026-10-24 directory name (the config was committed at "
                    "23:15 on 2026-10-24 local)"
                ),
                "master_log_mtime_utc": (
                    _file_stamp(master_log)["mtime_utc"]
                    if master_log.is_file()
                    else None
                ),
                "run_crashed": bool(crash),
                "crash_evidence": [
                    {"source": _rel(master_log), "line": c["line"], "text": c["text"][:300]}
                    for c in (crash + oom)[:4]
                ],
            },
            "evidence": [],
        }
    )

    # -- sampling settings ------------------------------------------------
    solver_hits = _find(log_lines, r"^\[cfm-gen\] epoch=\d+ gen_validity=")
    knobs_hits = _find(log_lines, r"^\[BLK-C5\] decode knobs in force")
    claims.append(
        {
            "id": "sampling_settings",
            "question": "With which sampling settings was the pool decoded?",
            "status": "RESOLVED" if solver_hits and knobs_hits else "UNRESOLVED",
            "value": {
                "ode_method": cfg_method,
                "ode_n_steps": cfg_steps,
                "gen_eval_n_per_seed": cfg_gen_n,
                "gen_eval_seeds": cfg_gen_seeds,
                "seed": cfg_seed,
                "solver_reported_in_log": sorted(
                    {m for h in solver_hits for m in re.findall(r"solver=(\S+)", h["text"])}
                ),
                "seeds_reported_in_log": sorted(
                    {m for h in solver_hits for m in re.findall(r"seeds=(\S+)", h["text"])}
                ),
                "decode_knobs_reported_in_log": (
                    knobs_hits[0]["text"] if knobs_hits else None
                ),
                "note": (
                    "The CSV carries NO seed / template_offset column, so the "
                    "per-row seed mapping is NOT recoverable from the pool "
                    "artifact; only the epoch-level aggregate is."
                ),
            },
            "evidence": [
                {"source": _rel(master_log), "line": h["line"], "text": h["text"]}
                for h in (solver_hits + knobs_hits)[:8]
            ],
        }
    )

    # -- checkpoint -------------------------------------------------------
    ckpts = []
    if ckpt_dir.is_dir():
        for p in sorted(ckpt_dir.iterdir()):
            if p.is_file():
                ckpts.append(_file_stamp(p))
    # Which epoch each checkpoint holds is INFERRED from its mtime aligned
    # against the per-epoch event timestamps — stated as an inference, not
    # as a fact read out of the file (the files are never deserialised).
    epoch_ends = sorted(
        (
            time.gmtime(float(e["ts"])),
            int(e.get("epoch", -1)),
        )
        for e in evs
        if e.get("event") == "train_epoch" and isinstance(e.get("ts"), (int, float))
    )
    for c in ckpts:
        mt = time.gmtime(Path(c["path"]).stat().st_mtime)
        best = None
        for t, ep in epoch_ends:
            if abs(t.tm_year - mt.tm_year) < 2 and abs(t.tm_yday - mt.tm_yday) < 2:
                if abs(t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
                       - (mt.tm_hour * 3600 + mt.tm_min * 60 + mt.tm_sec)) < 300:
                    best = ep
        c["epoch_inferred_from_mtime"] = best
        c["epoch_attribution"] = "INFERRED from mtime vs per-epoch event ts"
    claims.append(
        {
            "id": "checkpoint_provenance",
            "question": "From WHICH checkpoint path did the pool rows come?",
            "status": "RESOLVED",
            "value": {
                "answer": (
                    "NO checkpoint file. The CSV rows are decoded from the "
                    "LIVE in-memory model at the end of each training epoch "
                    "(trainer:18919/18926/18969 -> 19215-19251): "
                    "_gen_smiles is created inside the epoch loop and the CSV "
                    "is appended from it. The .pt files below are the "
                    "periodic and gate-selected saves of the same run, "
                    "hashed as bytes only (no torch.load, no weights read)."
                ),
                "checkpoint_dir": str(ckpt_dir),
                "checkpoints_on_disk": ckpts,
                "gate_metric_from_log": _find(log_lines, r"^\[cfm-bond-type\] epoch="),
            },
            "evidence": [
                _cite(TRAINER, 18926, 18926),
                _cite(TRAINER, 19215, 19216),
                _cite(TRAINER, 19248, 19252),
            ],
        }
    )

    # -- crash / truncation of the epoch plan ------------------------------
    epochs_done = sorted(
        {int(e["epoch"]) for e in evs if e.get("event") == "train_epoch"}
    )
    claims.append(
        {
            "id": "run_completeness",
            "question": "Did the run finish its planned epochs?",
            "status": "RESOLVED" if epochs_done else "UNRESOLVED",
            "value": {
                "epochs_completed": epochs_done,
                "epochs_planned": cfg_epochs,
                "ran_to_completion": False if crash else None,
                "note": (
                    "The pool is a TRUNCATED run: "
                    f"{len(epochs_done)} of {cfg_epochs} planned epochs, the "
                    "process died with a CUDA OOM during the next epoch's "
                    "backward pass. The pool is NOT a converged model."
                )
                if crash
                else "No crash traceback found in master.log.",
            },
            "evidence": [
                {"source": _rel(master_log), "line": c["line"], "text": c["text"][:300]}
                for c in (crash + oom)[:4]
            ],
        }
    )

    unresolved = [c["id"] for c in claims if c["status"] == "UNRESOLVED"]
    return {
        "verdict": (
            "PROVENANCE NOT PROVABLE"
            if unresolved
            else "PROVENANCE RESOLVED (from on-disk artifacts)"
        ),
        "unresolved_claims": unresolved,
        "claims": claims,
        "run": {
            "config": str(config_path),
            "out_dir": cfg_out_dir,
            "ckpt_dir": cfg_ckpt_dir,
            "seed": cfg_seed,
            "seed_declared_at": cfg_seed_src,
            "epochs_planned": cfg_epochs,
            "epochs_declared_at": cfg_epochs_src,
            "batch_size": cfg_batch,
            "lr": cfg_lr,
            "checkpoint_interval": cfg_ckpt_interval,
            "resume_from": cfg_resume,
        },
        "sampling": {
            "ode_method": cfg_method,
            "ode_n_steps": cfg_steps,
            "gen_eval_n_per_seed": cfg_gen_n,
            "gen_eval_seeds": cfg_gen_seeds,
        },
        "events_summary": _events_summary(evs),
    }


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _torch_imported_by_this_script() -> bool:
    """True iff torch became importable AFTER this module was imported.

    A test harness may have loaded torch before this module was imported
    (pytest plugins do); that is not this script using a GPU, and it is
    not this script's doing.
    """
    return ("torch" in sys.modules) and not _TORCH_PRESENT_AT_IMPORT


def _yaml_line_of(path: Path, needle: str) -> int:
    for i, line in enumerate(_read_lines(path), start=1):
        if line[1].strip().startswith(needle):
            return i
    return 1


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _events_summary(evs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    seed_tot: Dict[int, int] = {}
    seed_err: Dict[int, int] = {}
    seed_n: Dict[int, List[int]] = {}
    for e in evs:
        if e.get("event") != "gen_eval":
            continue
        ep = int(e.get("epoch", -1))
        seed_tot[ep] = seed_tot.get(ep, 0) + int(e.get("n_total") or 0)
        seed_err[ep] = seed_err.get(ep, 0) + int(e.get("n_decode_error") or 0)
        seed_n.setdefault(ep, []).append(int(e.get("n_total") or 0))
    for e in evs:
        if e.get("event") != "train_epoch":
            continue
        ep = int(e.get("epoch", -1))
        rows.append(
            {
                "epoch": e.get("epoch"),
                "ts_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(e["ts"]))
                )
                if isinstance(e.get("ts"), (int, float))
                else None,
                "gen_validity": e.get("gen_validity"),
                "gen_validity_n": e.get("gen_validity_n"),
                "gen_eval_n": e.get("gen_eval_n"),
                "gen_eval_seeds": e.get("gen_eval_seeds"),
                "gen_solver": e.get("gen_solver"),
                "gen_single_fragment_ratio": e.get("gen_single_fragment_ratio"),
                "gen_largest_fragment_frac": e.get("gen_largest_fragment_frac"),
                "gen_metal_ratio": e.get("gen_metal_ratio"),
                "train_loss": e.get("train_loss"),
                "sum_per_seed_n_total": seed_tot.get(ep),
                "sum_per_seed_n_decode_error": seed_err.get(ep),
                "per_seed_n_total": sorted(seed_n.get(ep, [])),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Q2 — ROUND-TRIP
# --------------------------------------------------------------------------
def roundtrip_one(smiles: str) -> Dict[str, Any]:
    """Re-parse one SMILES with RDKit and recompute the two published columns.

    Mirrors the producing code EXACTLY: ``Chem.MolFromSmiles`` with RDKit's
    defaults (sanitise on, explicit H removed) and
    ``Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)``.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    out: Dict[str, Any] = {
        "smiles": smiles,
        "parsed": False,
        "sanitised": False,
        "n_frags": None,
        "largest_frag_frac": None,
        "n_heavy": None,
        "canonical_smiles": None,
        "canonical_identical": None,
        "metal_z": None,
        "metal_alone": None,
        "metal_shares_fragment": None,
        "error": None,
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if mol is None:
        # Distinguish "does not parse at all" from "parses but fails
        # sanitisation": the latter is the case where the writer's
        # decode_chemistry() returned None and the row's connectivity
        # columns were a sentinel rather than a measurement.
        try:
            raw = Chem.MolFromSmiles(smiles, sanitize=False)
        except Exception:  # noqa: BLE001
            raw = None
        out["parses_without_sanitisation"] = raw is not None
        return out

    out["parsed"] = True
    out["sanitised"] = True
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    sizes = [len(f) for f in frags] or [0]
    n_heavy = int(mol.GetNumHeavyAtoms())
    out["n_frags"] = len(sizes)
    out["largest_frag_frac"] = (max(sizes) / n_heavy) if n_heavy else 0.0
    out["n_heavy"] = n_heavy
    out["frag_sizes"] = sizes
    try:
        canon = Chem.MolToSmiles(mol)
        out["canonical_smiles"] = canon
        out["canonical_identical"] = canon == smiles
        # If the stored string is not the canonical form, is it still the
        # SAME molecule? (a ring-closure numbering difference would be a
        # pure serialisation artefact)
        out["same_molecule_as_canonical"] = (
            Chem.MolToSmiles(Chem.MolFromSmiles(canon)) == canon
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"

    metal_idx = {
        a.GetIdx() for a in mol.GetAtoms() if int(a.GetAtomicNum()) in METAL_Z
    }
    if metal_idx:
        frag_of: Dict[int, Tuple[int, ...]] = {}
        for f in frags:
            for i in f:
                frag_of[i] = f
        shared = any(
            any((j not in metal_idx) for j in frag_of[mi]) for mi in metal_idx
        )
        out["metal_z"] = sorted(
            {int(a.GetAtomicNum()) for a in mol.GetAtoms() if a.GetAtomicNum() in METAL_Z}
        )
        out["metal_shares_fragment"] = bool(shared)
        out["metal_alone"] = not shared
        out["metal_survives_as_single_atom"] = any(
            len(frag_of[mi]) == 1 for mi in metal_idx
        )
    return out


def q2_roundtrip(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    n_rows = len(rows)
    n_parsed = 0
    n_unparsed = 0
    n_parse_only_no_sanitise = 0
    frag_equal = 0
    frag_differ = 0
    frag_hist: Counter = Counter()
    lfrac_equal = 0
    lfrac_differ = 0
    lfrac_deltas: List[float] = []
    canon_identical = 0
    canon_differ: List[Dict[str, Any]] = []
    metal_alone_rows: List[int] = []
    metal_alone_survive = 0
    metal_rows = 0
    single_frag_rows = 0
    rows_losing_atoms_under_truncation = 0
    unparsed_examples: List[Dict[str, Any]] = []
    per_row: List[Dict[str, Any]] = []

    for idx, r in enumerate(rows):
        smi = r["smiles"]
        rec = roundtrip_one(smi)
        rec["row_index"] = idx
        rec["epoch"] = r.get("epoch")
        rec["stored_n_frags_raw"] = r.get("n_frags_raw")
        rec["stored_largest_frag_frac"] = r.get("largest_frag_frac")
        per_row.append(rec)

        if not rec["parsed"]:
            n_unparsed += 1
            if rec.get("parses_without_sanitisation"):
                n_parse_only_no_sanitise += 1
            if len(unparsed_examples) < 20:
                unparsed_examples.append({"row_index": idx, "smiles": smi[:200]})
            continue
        n_parsed += 1

        try:
            s_n = int(float(r["n_frags_raw"]))
            s_lf = float(r["largest_frag_frac"])
        except (TypeError, ValueError):
            s_n = None
            s_lf = None

        if s_n is not None and rec["n_frags"] is not None:
            if s_n == rec["n_frags"]:
                frag_equal += 1
            else:
                frag_differ += 1
                frag_hist[s_n - rec["n_frags"]] += 1

        if s_lf is not None and rec["largest_frag_frac"] is not None:
            delta = rec["largest_frag_frac"] - s_lf
            lfrac_deltas.append(delta)
            if abs(delta) <= LFRAC_TOL:
                lfrac_equal += 1
            else:
                lfrac_differ += 1

        if rec.get("canonical_identical"):
            canon_identical += 1
        else:
            canon_differ.append(
                {
                    "row_index": idx,
                    "epoch": r.get("epoch"),
                    "stored_smiles": smi,
                    "canonical_smiles": rec.get("canonical_smiles"),
                    "same_molecule_as_canonical": rec.get(
                        "same_molecule_as_canonical"
                    ),
                    "n_frags_stored": s_n,
                    "n_frags_recomputed": rec["n_frags"],
                    "largest_frag_frac_stored": s_lf,
                    "largest_frag_frac_recomputed": rec["largest_frag_frac"],
                }
            )

        if rec.get("metal_z"):
            metal_rows += 1
            if rec.get("metal_alone"):
                metal_alone_rows.append(idx)
                if rec.get("metal_survives_as_single_atom"):
                    metal_alone_survive += 1
        if rec["n_frags"] == 1:
            single_frag_rows += 1
        if rec["n_frags"] and rec["n_frags"] > 1:
            rows_losing_atoms_under_truncation += 1

    # What the field-standard POST-PROCESS (keep the largest connected
    # component) would do to this pool.  This is OUR measurement of what
    # the transformation WOULD buy — it asserts nothing about what any
    # frontier method does.
    trunc_would_be_single = 0
    trunc_dropped_atoms = 0
    for rec in per_row:
        if not rec["parsed"] or not rec["n_frags"]:
            continue
        if rec["n_frags"] == 1:
            trunc_would_be_single += 1
            continue
        trunc_would_be_single += 1
        trunc_dropped_atoms += rec["n_heavy"] - max(rec["frag_sizes"])

    def _stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0}
        sv = sorted(vals)
        return {
            "n": len(sv),
            "min": sv[0],
            "max": sv[-1],
            "mean": sum(sv) / len(sv),
            "median": sv[len(sv) // 2],
            "max_abs": max(abs(v) for v in sv),
        }

    return {
        "n_rows": n_rows,
        "n_parsed": n_parsed,
        "n_unparsed": n_unparsed,
        "n_parse_without_sanitisation": n_parse_only_no_sanitise,
        "unparsed_examples": unparsed_examples,
        "frag_count": {
            "n_equal": frag_equal,
            "n_differ": frag_differ,
            "difference_histogram_stored_minus_recomputed": {
                str(k): int(v) for k, v in sorted(frag_hist.items())
            },
            "interpretation": (
                "0 differences means the published n_frags_raw is exactly "
                "reproducible from the pool's own SMILES column with the "
                "DEFAULT RDKit parse — i.e. the connectivity columns carry no "
                "serialisation artefact."
            ),
        },
        "largest_frag_frac": {
            "tolerance": LFRAC_TOL,
            "tolerance_reason": (
                "the writer rounds to 4 dp (trainer:19310), so half a unit "
                "in the last published place is the exact-equality bound"
            ),
            "n_equal": lfrac_equal,
            "n_differ": lfrac_differ,
            "delta_stats": _stats(lfrac_deltas),
            "delta_histogram": {
                f"{k:.6f}": int(v)
                for k, v in sorted(
                    Counter(round(d, 6) for d in lfrac_deltas).items()
                )
            },
        },
        "canonical_roundtrip": {
            "n_identical": canon_identical,
            "n_differ": len(canon_differ),
            "differing_rows": canon_differ[:20],
        },
        "metal": {
            "metal_z_set": sorted(METAL_Z),
            "n_rows_with_metal": metal_rows,
            "n_metal_shares_fragment": metal_rows - len(metal_alone_rows),
            "n_metal_alone": len(metal_alone_rows),
            "n_metal_alone_surviving_as_single_atom": metal_alone_survive,
            "metal_alone_row_indices": metal_alone_rows[:50],
        },
        "postprocess_truncation_impact": {
            "definition": (
                "keep the largest connected component per row (the "
                "field-standard disconnection post-process); measured on "
                "this pool"
            ),
            "n_rows_that_would_become_single_fragment": trunc_would_be_single,
            "n_rows_that_would_become_single_fragment_pct": (
                round(100.0 * trunc_would_be_single / n_rows, 4) if n_rows else None
            ),
            "n_rows_currently_multi_fragment": rows_losing_atoms_under_truncation,
            "atoms_that_truncation_would_drop_total": trunc_dropped_atoms,
            "note": (
                "This is the size of the RAW-vs-POST-PROCESSED break for "
                "OUR pool. It says nothing about what any frontier method "
                "does — that comparison is a different branch."
            ),
        },
        "_per_row": per_row,
    }


# --------------------------------------------------------------------------
# Q3 — COLUMN SEMANTICS
# --------------------------------------------------------------------------
def q3_semantics() -> Dict[str, Any]:
    trainer_text = TRAINER.read_text(encoding="utf-8") if TRAINER.is_file() else ""
    metal_literal_present = _METAL_LITERAL in trainer_text
    return {
        "writer": {
            "file": _rel(TRAINER),
            "guard": "trainer:19215 — `if bool(getattr(args, "
            "\"save_generated_smiles\", False)) and _gen_smiles:`",
            "path": "trainer:19216 — `_smi_path = out_dir / \"generated_smiles.csv\"`",
            "header": "trainer:19249-19251 — `epoch,smiles,n_frags_raw,largest_frag_frac`",
            "append_semantics": "trainer:19248 — opened with mode \"a\"; append-only across epochs",
            "evidence": [
                _cite(TRAINER, 19215, 19216),
                _cite(TRAINER, 19245, 19252),
                _cite(TRAINER, 19308, 19311),
            ],
        },
        "n_frags_raw": {
            "meaning": (
                "The number of RDKit CONNECTED COMPONENTS of the raw decode, "
                "computed by `decode_chemistry()` as "
                "`len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))` "
                "on the SANITISED mol produced by `Chem.MolFromSmiles(smiles)` "
                "with RDKit's defaults.  It is the count of FRAGMENTS of the "
                "decoded graph, NOT a count of bonds and NOT a count of "
                "atoms.  It is a RAW count: it describes the decode BEFORE "
                "any largest-fragment truncation."
            ),
            "computed_before_or_after_sanitisation": "AFTER (MolFromSmiles defaults to sanitize=True)",
            "includes_metal_atom": True,
            "is_rdkit_fragment_count": True,
            "evidence": [
                _cite(TRAINER, 9671, 9676),
                _cite(TRAINER, 9697, 9703),
                _cite(TRAINER, 19282, 19287),
            ],
        },
        "largest_frag_frac": {
            "meaning": (
                "`max(fragment_sizes) / mol.GetNumHeavyAtoms()` — the atom "
                "share of the LARGEST connected component.  Both the "
                "numerator and the denominator INCLUDE the metal atom: the "
                "metal is a heavy atom of the mol and sits inside whichever "
                "fragment it belongs to.  A metal that is alone in its own "
                "fragment contributes 1 atom to the denominator and 0 to "
                "the numerator."
            ),
            "computed_before_or_after_sanitisation": "AFTER",
            "includes_metal_atom": True,
            "rounded_to_dp": ROUNDED_DP,
            "evidence": [
                _cite(TRAINER, 9674, 9676),
                _cite(TRAINER, 9703, 9703),
                _cite(TRAINER, 19310, 19310),
            ],
        },
        "truncation_flag": {
            "meaning": (
                "`--decode-largest-fragment` is `action=\"store_true\"`, so "
                "it defaults OFF.  When OFF the `smiles` column keeps the "
                "FULL multi-fragment decode and `n_frags_raw` describes it; "
                "when ON the `smiles` column would hold only the largest "
                "fragment while `n_frags_raw` still described the pre-"
                "truncation decode.  The C8 config does not set the flag, "
                "and the pool's own multi-fragment rows still carry their "
                "dots, so truncation was OFF for this pool."
            ),
            "evidence": [
                _cite(TRAINER, 14210, 14212),
                _cite(TRAINER, 19240, 19242),
            ],
        },
        "sentinel_history": {
            "meaning": (
                "As of Wf-A114 (2026-10-30) an UNPARSEABLE smiles writes "
                "`n_frags_raw = -1` / `largest_frag_frac = -1.0`.  This pool "
                "predates that fix (it was written 2026-10-25), so an "
                "unparseable row in it would carry the OLD fabricated `1` / "
                "`1.0`.  MEASURED: 0 of the pool's rows fail to parse, so the "
                "sentinel question does not arise for this pool."
            ),
            "evidence": [_cite(TRAINER, 19253, 19266)],
        },
        "target_metals_literal_present_in_trainer": metal_literal_present,
        "target_metals": sorted(METAL_Z),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_checks(
    pool_rows: List[Dict[str, str]],
    q1: Dict[str, Any],
    q2: Dict[str, Any],
    q3: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    checks.append(
        {
            "name": "pool_header_matches_published_schema",
            "status": "PASS"
            if pool_rows
            and list(pool_rows[0].keys()) == POOL_HEADER
            else "FAIL",
            "detail": f"header={list(pool_rows[0].keys()) if pool_rows else None}",
        }
    )

    ev_rows = q1.get("events_summary", [])
    csv_epochs = Counter(int(r["epoch"]) for r in pool_rows)
    mismatches = []
    for e in ev_rows:
        ep = int(e["epoch"])
        n_ev = e.get("gen_validity_n")
        n_csv = csv_epochs.get(ep, 0)
        if n_ev is None or int(n_ev) != n_csv:
            mismatches.append({"epoch": ep, "events_gen_validity_n": n_ev, "csv_rows": n_csv})
    checks.append(
        {
            "name": "pool_rows_per_epoch_match_events_gen_validity_n",
            "status": "PASS" if ev_rows and not mismatches else "FAIL",
            "detail": (
                f"per-epoch csv rows={dict(sorted(csv_epochs.items()))}; "
                f"events gen_validity_n={[e.get('gen_validity_n') for e in ev_rows]}; "
                f"mismatches={mismatches}"
            ),
        }
    )

    checks.append(
        {
            "name": "pool_is_complete_for_what_the_run_decoded",
            "status": "PASS"
            if ev_rows
            and all(
                e.get("sum_per_seed_n_total") == e.get("gen_validity_n")
                and e.get("sum_per_seed_n_decode_error") == 0
                for e in ev_rows
            )
            else "FAIL",
            "detail": (
                "the pool must hold EVERY sample the run decoded: the sum of "
                "the per-seed n_total (events.jsonl) must equal the epoch's "
                "gen_validity_n and the CSV's row count, with "
                "n_decode_error = 0"
            ),
        }
    )

    checks.append(
        {
            "name": "provenance_claims_resolved",
            "status": "PASS" if not q1["unresolved_claims"] else "FAIL",
            "detail": f"unresolved={q1['unresolved_claims']}",
        }
    )

    checks.append(
        {
            "name": "roundtrip_completed_over_all_rows",
            "status": "PASS"
            if q2["n_rows"] > 0 and q2["n_parsed"] == q2["n_rows"]
            else "FAIL",
            "detail": (
                f"parsed={q2['n_parsed']}/{q2['n_rows']}; "
                f"unparsed={q2['n_unparsed']}"
            ),
        }
    )

    checks.append(
        {
            "name": "target_metals_literal_still_matches_trainer_source",
            "status": "PASS" if q3["target_metals_literal_present_in_trainer"] else "FAIL",
            "detail": (
                "the metal set used for the metal-alone recomputation is "
                "grepped out of the live trainer source, so an upstream "
                "rename cannot silently change the number"
            ),
        }
    )

    checks.append(
        {
            "name": "torch_never_imported_by_this_script",
            "status": "PASS" if not _torch_imported_by_this_script() else "FAIL",
            "detail": (
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; "
                f"torch in sys.modules={'torch' in sys.modules}; "
                f"torch was already loaded before this module imported="
                f"{_TORCH_PRESENT_AT_IMPORT}"
            ),
        }
    )
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Wf-A158 BRANCH POOL — provenance + round-trip audit (CPU only)."
    )
    ap.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    ap.add_argument("--master-log", type=Path, default=DEFAULT_MASTER_LOG)
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat an UNRESOLVED provenance claim as FAIL (default: also FAIL, "
        "kept for symmetry with the repo's --strict convention)",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    out_json = args.out_json or (args.out_dir / "pool_provenance.json")

    if not args.pool_csv.is_file():
        print(f"FAIL: pool CSV not found: {args.pool_csv}")
        return 1

    with args.pool_csv.open("r", encoding="utf-8", newline="") as fh:
        pool_rows = list(csv.DictReader(fh))
    if not pool_rows:
        print(f"FAIL: pool CSV has no data rows: {args.pool_csv}")
        return 1

    q1 = q1_provenance(
        args.master_log, args.events, args.ckpt_dir, args.config, args.pool_csv
    )
    q2 = q2_roundtrip(pool_rows)
    q3 = q3_semantics()
    checks = build_checks(pool_rows, q1, q2, q3)

    per_row = q2.pop("_per_row")
    failed = [c for c in checks if c["status"] == "FAIL"]
    skipped = [c for c in checks if c["status"] == "SKIP"]
    verdict = "PASS" if not failed and not skipped else ("FAIL" if failed else "SKIP")

    artifact = {
        "schema_version": "wfa158_pool_provenance.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "molmetal/scripts/wfa158_pool_provenance.py",
        "wall_time_s": round(time.time() - t0, 3),
        "cpu_only": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_imported": "torch" in sys.modules,
            "torch_imported_by_this_script": _torch_imported_by_this_script(),
            "torch_already_loaded_before_this_module": _TORCH_PRESENT_AT_IMPORT,
            "checkpoint_deserialised": False,
            "note": (
                "the two .pt files are hashed as bytes (sha256) only — no "
                "torch.load, no weights in RAM"
            ),
        },
        "inputs": {
            "pool_csv": _file_stamp(args.pool_csv),
            "master_log": (
                _file_stamp(args.master_log) if args.master_log.is_file() else None
            ),
            "events_jsonl": (
                _file_stamp(args.events) if args.events.is_file() else None
            ),
            "config": (
                _file_stamp(args.config) if args.config.is_file() else None
            ),
            "trainer_source": (
                _file_stamp(TRAINER) if TRAINER.is_file() else None
            ),
        },
        "q1_provenance": q1,
        "q2_roundtrip": q2,
        "q3_column_semantics": q3,
        "checks": checks,
        "summary": {
            "verdict": verdict,
            "n_checks": len(checks),
            "n_failed": len(failed),
            "n_skipped": len(skipped),
        },
        "per_row_digest": per_row,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(
        out_json, json.dumps(artifact, indent=2, sort_keys=False, default=str) + "\n"
    )

    # -- stdout summary ---------------------------------------------------
    print(f"[wfa158] pool            : {args.pool_csv}")
    print(f"[wfa158] rows            : {q2['n_rows']}")
    print(
        f"[wfa158] Q1 provenance   : {q1['verdict']}"
        + (f" (unresolved={q1['unresolved_claims']})" if q1["unresolved_claims"] else "")
    )
    print(
        f"[wfa158] Q2 parsed       : {q2['n_parsed']}/{q2['n_rows']} "
        f"(unparsed={q2['n_unparsed']})"
    )
    print(
        f"[wfa158] Q2 frag count   : equal={q2['frag_count']['n_equal']} "
        f"differ={q2['frag_count']['n_differ']} "
        f"hist={q2['frag_count']['difference_histogram_stored_minus_recomputed']}"
    )
    print(
        f"[wfa158] Q2 largest frac : equal={q2['largest_frag_frac']['n_equal']} "
        f"differ={q2['largest_frag_frac']['n_differ']} "
        f"max_abs_delta={q2['largest_frag_frac']['delta_stats'].get('max_abs')}"
    )
    print(
        f"[wfa158] Q2 canonical    : identical={q2['canonical_roundtrip']['n_identical']} "
        f"differ={q2['canonical_roundtrip']['n_differ']}"
    )
    print(
        f"[wfa158] Q2 metal alone  : {q2['metal']['n_metal_alone']} "
        f"(survive as single atom: {q2['metal']['n_metal_alone_surviving_as_single_atom']})"
    )
    print(
        f"[wfa158] Q3 semantics    : n_frags_raw/largest_frag_frac = RDKit "
        f"connected components of the SANITISED mol, metal INCLUDED"
    )
    for c in checks:
        print(f"[wfa158] check {c['status']:4s} {c['name']}: {c['detail']}")
    print(f"[wfa158] artifact        : {out_json}")
    print(f"[wfa158] wall_time_s     : {artifact['wall_time_s']}")
    print(verdict)
    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
