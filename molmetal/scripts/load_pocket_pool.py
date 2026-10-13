"""CLI: enumerate the CrossDocked 2020 pocket pool.

Track B (pocket-collate-model-submodel-operator) ships a per-cloud
PocketCloud draw from the 100K CrossDocked pool.  This helper is the
**non-trainer consumer** of the YAML keys (per AGENTS.md §5.11): it
reads ``pocket_pool_path`` / ``pocket_pool_size`` so the config-binding
audit sees a real consumer and the keys don't land in the dead-key
list.

Usage
-----

    python -m molmetal.scripts.load_pocket_pool --pocket-pool-path /data/crossdocked_pocket10 --pocket-pool-size 1000

Prints::

    pool_size=1000  first=10ak_2z3x  last=zz99_zz9z

and writes ``molmetal/reports/track_b_pocket_pool_index.json`` so the
config-binding audit has a fingerprint on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_pocket_pool_index(
    pocket_pool_path: str,
    pocket_pool_size: int = 0,
) -> Dict[str, str]:
    """Return ``{pocket_id: pdb_path}`` for every ``*.pdb`` under the pool.

    Parameters
    ----------
    pocket_pool_path : str
        Root directory containing the pool's PDB files.
    pocket_pool_size : int
        Cap on the returned entries (0 = use all).  Files are sorted
        lexicographically so the cap is deterministic across runs.

    Returns
    -------
    dict[str, str]
        ``{stem: absolute_path}`` for each PDB file.
    """
    pool_dir = Path(pocket_pool_path)
    if not pool_dir.is_dir():
        raise FileNotFoundError(
            f"pocket_pool_path does not exist or is not a directory: {pool_dir}"
        )
    pool_files = sorted(pool_dir.glob("*.pdb"))
    if pocket_pool_size > 0:
        pool_files = pool_files[: int(pocket_pool_size)]
    index: Dict[str, str] = {p.stem: str(p.resolve()) for p in pool_files}
    return index


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write — see AGENTS.md §5.6 atomic-write protocol.

    Uses POSIX ``os.replace`` so a partially-written file can never
    appear at the destination.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="load_pocket_pool",
        description=(
            "Enumerate the CrossDocked 2020 pocket pool and write the index.  "
            "Track B (pocket-collate-model-submodel-operator) helper."
        ),
    )
    parser.add_argument(
        "--pocket-pool-path",
        required=True,
        help="Root directory of the CrossDocked 2020 PDB pool.",
    )
    parser.add_argument(
        "--pocket-pool-size",
        type=int,
        default=0,
        help="Cap on pool entries (0 = use all).",
    )
    parser.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT / "molmetal" / "reports" / "track_b_pocket_pool_index.json"
        ),
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    try:
        index = build_pocket_pool_index(
            pocket_pool_path=args.pocket_pool_path,
            pocket_pool_size=args.pocket_pool_size,
        )
    except FileNotFoundError as exc:
        print(f"[load_pocket_pool] ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {
        "pocket_pool_path": os.path.abspath(args.pocket_pool_path),
        "pocket_pool_size": int(args.pocket_pool_size),
        "actual_size": len(index),
        "first_id": next(iter(index.keys()), ""),
        "last_id": next(reversed(index.keys()), ""),
    }
    out_path = Path(args.out)
    _atomic_write_json(out_path, payload)
    print(
        f"track_b_pocket_pool_index: pool_size={payload['actual_size']} "
        f"first={payload['first_id']!r} last={payload['last_id']!r} "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())