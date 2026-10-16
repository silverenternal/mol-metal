"""Build the canonical D-MPNN checkpoint manifest.

=====================================================================
molmetal/checkpoints/ holds 105 D-MPNN ``.pt`` files across 4 incompatible
architecture classes (pic50-root, mtw-encoder-dann, mtw-encoder-nodann,
mtw-v3-inner). Without a manifest, callers must ``torch.load`` every file
to learn its (metal, seed, hidden, depth, metric, ...). This script walks
every ``dmpnn_*.pt`` (excluding ``.before_retrain``), infers each
checkpoint's shape from its ``state_dict`` prefix topology, and emits a
single JSON keyed by relative path so downstream code can discover ckpt
metadata WITHOUT loading weights.

**Outputs**
* ``molmetal/checkpoints/ckpt_manifest.json`` — one record per ckpt with:
  ``metal``, ``seed``, ``arch_class``, ``hidden``, ``depth``, ``best_epoch``,
  ``val_auc``, ``val_rmse``, ``test_auc``, ``test_rmse``, ``has_optimizer_state``,
  ``n_state_dict_keys``, ``size_bytes``, ``sha256``, ``mtime``, ``load_ok``,
  ``error`` (zero unless ``load_ok`` is False).
* stdout summary: counts by arch class, load failures (if any).

**Arch class taxonomy** (derived from state_dict TOP-LEVEL prefixes — NOT
filename echoes):
  * ``pic50_root``            : {atom_emb, bond_emb, edge_init, gru_update, readout}
  * ``mtw_encoder_dann``      : {cls_heads, domain_head, encoder, reg_heads}
  * ``mtw_encoder_nodann``    : {cls_heads, encoder, reg_heads}
  * ``mtw_v3_inner``          : {domain_head, inner}
  * ``unknown``               : any other prefix set (counted + reported)

**Hardware**: CPU only (SHA256 + JSON).
**Wall**: ~10-30 s for 105 ckpts (one torch.load per ckpt, ``map_location='cpu'``,
``weights_only=False``).

**Honest framing**
* Every numeric value is MEASURED on disk; nothing is projected.
* ``arch_class`` is inferred from state_dict prefixes, not from filename —
  this is the entire point of the manifest (filename ``dmpnn_mtw_ru_seed0.pt``
  may carry ``mtw-encoder-dann`` OR ``mtw-encoder-nodann`` OR ``mtw-v3-inner``
  depending on which wave wrote the file).
* One corrupt checkpoint cannot lose the other 104 — every file is wrapped
  in try/except.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_ROOT = PROJECT_ROOT / "molmetal" / "checkpoints"
MANIFEST_PATH = CKPT_ROOT / "ckpt_manifest.json"

# ---------------------------------------------------------------------------
# Arch class taxonomy (derived from observed state_dict top-level prefixes)
# ---------------------------------------------------------------------------
ARCH_CLASSES: Dict[Tuple[str, ...], str] = {
    ("atom_emb", "bond_emb", "edge_init", "gru_update", "readout"): "pic50_root",
    ("cls_heads", "domain_head", "encoder", "reg_heads"): "mtw_encoder_dann",
    ("cls_heads", "encoder", "reg_heads"): "mtw_encoder_nodann",
    ("domain_head", "inner"): "mtw_v3_inner",
}


def _classify_arch(state_dict_keys: List[str]) -> str:
    """Map state_dict top-level prefixes to a named arch_class."""
    if not state_dict_keys:
        return "unknown"
    prefixes = tuple(sorted(set(k.split(".", 1)[0] for k in state_dict_keys)))
    return ARCH_CLASSES.get(prefixes, f"unknown:{','.join(prefixes)}")


def _extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Pull the state_dict out of either a wrapped ckpt or a bare sd."""
    if not isinstance(obj, dict):
        return None
    for key in ("model_state_dict", "state_dict", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    # Maybe the dict IS the state_dict
    sample = next(iter(obj.values()), None)
    if isinstance(sample, torch.Tensor):
        return obj
    return None


def _infer_hidden(sd: Dict[str, Any]) -> Optional[int]:
    """Pick the atom-embedding weight out HiddenDim from any common prefix."""
    for k, v in sd.items():
        if k.endswith("atom_emb.weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[0])
    for k, v in sd.items():
        if k.endswith("W_i.weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[0])
    return None


def _infer_depth(sd: Dict[str, Any]) -> Optional[int]:
    """Count GRU update blocks: keys like ``gru_update.W_z.weight`` (1 set
    per round of message passing). For v3-inner models, fall back to counting
    distinct ``message_passing_step`` indices."""
    gru_keys = [k for k in sd.keys() if "gru_update" in k and k.endswith(".weight")]
    if gru_keys:
        return len(gru_keys)
    mp_keys = [k for k in sd.keys() if "message_passing" in k and k.endswith(".weight")]
    if mp_keys:
        return len(mp_keys)
    return None


def _coerce_metric(value: Any) -> Optional[float]:
    """Best-effort coerce metric values to float (NaN -> None)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item() if value.numel() == 1 else None
    try:
        f = float(value)
        import math
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _record_for(path: Path) -> Dict[str, Any]:
    """Build the per-ckpt manifest record. NEVER raises — corrupt ckpts are
    captured as ``load_ok=False`` + ``error`` message."""
    try:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    stat = path.stat()
    record: Dict[str, Any] = {
        "path": rel,
        "size_bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "sha256": _sha256(path),
        "load_ok": False,
        "error": None,
        "metal": None,
        "seed": None,
        "arch_class": None,
        "hidden": None,
        "depth": None,
        "best_epoch": None,
        "val_auc": None,
        "val_rmse": None,
        "test_auc": None,
        "test_rmse": None,
        "has_optimizer_state": False,
        "n_state_dict_keys": None,
    }
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(obj)
        if sd is None:
            record["error"] = "no state_dict key"
            return record
        sd_keys = list(sd.keys())
        record["n_state_dict_keys"] = len(sd_keys)
        record["arch_class"] = _classify_arch(sd_keys)
        record["hidden"] = _infer_hidden(sd)
        record["depth"] = _infer_depth(sd)
        if isinstance(obj, dict):
            for src, dst in (("metal", "metal"), ("seed", "seed"),
                              ("best_epoch", "best_epoch")):
                if src in obj:
                    val = obj[src]
                    if isinstance(val, torch.Tensor):
                        val = val.item() if val.numel() == 1 else None
                    record[dst] = val
            record["val_auc"] = _coerce_metric(obj.get("val_auc"))
            record["val_rmse"] = _coerce_metric(obj.get("val_rmse"))
            record["test_auc"] = _coerce_metric(obj.get("test_auc"))
            record["test_rmse"] = _coerce_metric(obj.get("test_rmse"))
            if "best_epoch" not in obj and "epoch" in obj:
                val = obj["epoch"]
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else None
                record["best_epoch"] = val
            record["has_optimizer_state"] = "optimizer_state_dict" in obj
        record["load_ok"] = True
    except Exception as exc:  # noqa: BLE001 — must catch corrupt ckpts
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return record


def discover_dmpnn_ckpts(root: Path) -> List[Path]:
    """Find every ``dmpnn_*.pt`` under ``root`` (recursive), skipping
    ``.before_retrain`` backup files."""
    return sorted(
        p for p in root.rglob("dmpnn_*.pt")
        if p.is_file() and not p.name.endswith(".before_retrain")
    )


def build_manifest(root: Path = CKPT_ROOT) -> Dict[str, Any]:
    """Walk all dmpnn ckpts and build the manifest dict."""
    paths = discover_dmpnn_ckpts(root)
    records: Dict[str, Any] = {}
    class_counts: Dict[str, int] = {}
    failures: List[Tuple[str, str]] = []
    t0 = time.time()
    for i, p in enumerate(paths, start=1):
        rec = _record_for(p)
        rel = rec["path"]
        records[rel] = rec
        cls = rec.get("arch_class") or "unknown"
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if not rec["load_ok"]:
            failures.append((rel, rec["error"] or ""))
        if i % 25 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {rel}  ({rec.get('arch_class')})", file=sys.stderr)
    elapsed = time.time() - t0
    manifest: Dict[str, Any] = {
        "version": 1,
        "generated_at": int(time.time()),
        "elapsed_sec": round(elapsed, 3),
        "ckpt_root": str(root),
        "n_ckpts": len(records),
        "n_load_failures": len(failures),
        "arch_class_counts": class_counts,
        "load_failures": [{"path": p, "error": e} for p, e in failures],
        "checkpoints": records,
    }
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser_args = argv if argv is not None and argv else sys.argv[1:]
    out = MANIFEST_PATH
    if parser_args:
        # Honour --out for ad-hoc redirection (test fixture use)
        if "--out" in parser_args:
            i = parser_args.index("--out")
            out = Path(parser_args[i + 1])
        if "--ckpt-root" in parser_args:
            i = parser_args.index("--ckpt-root")
            root = Path(parser_args[i + 1])
        else:
            root = CKPT_ROOT
    else:
        root = CKPT_ROOT
    manifest = build_manifest(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    try:
        out_label = str(out.relative_to(PROJECT_ROOT))
    except ValueError:
        out_label = str(out)
    print(f"Wrote {out_label}: "
          f"{manifest['n_ckpts']} ckpts, "
          f"{manifest['n_load_failures']} load failures, "
          f"{manifest['elapsed_sec']:.1f} s")
    print("arch_class_counts:", json.dumps(manifest["arch_class_counts"], indent=0))
    if manifest["load_failures"]:
        print("FAILURES:")
        for f in manifest["load_failures"]:
            print(f"  {f['path']}: {f['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())