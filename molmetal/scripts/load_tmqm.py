"""CLI: inspect the tmQM pre-training corpus (TODO F2 / P1).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.load_tmqm                  # Pt/Ru/Ir subset
    python -m molmetal.scripts.load_tmqm --all-metals     # all 30 d-block
    python -m molmetal.scripts.load_tmqm --refresh        # re-parse the .gz

Prints n_total, per-metal counts and the coordination-number distribution,
and writes the same numbers to
``molmetal/reports/f2_tmqm_stats.json`` for the report / figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.tmqm import (  # noqa: E402
    PAPER_METALS,
    load_tmqm,
    summarize,
)

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect the tmQM corpus.")
    p.add_argument(
        "--metals",
        nargs="*",
        default=list(PAPER_METALS),
        help="Metal centres to keep (default: Pt Ru Ir)",
    )
    p.add_argument(
        "--all-metals",
        action="store_true",
        help="Ignore --metals and report on all 30 d-block metals",
    )
    p.add_argument(
        "--keep-missing-smiles",
        action="store_true",
        help="Do not drop entries whose SMILES field is empty",
    )
    p.add_argument("--refresh", action="store_true", help="Force re-parse of the .gz shards")
    p.add_argument("--out", default=None, help="Output JSON path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metals = None if args.all_metals else list(args.metals)

    t0 = time.time()
    full = load_tmqm(
        metals=None,
        require_smiles=not args.keep_missing_smiles,
        refresh=args.refresh,
    )
    load_s = time.time() - t0
    print(f"[load_tmqm] parsed corpus in {load_s:.1f}s")
    print(f"[load_tmqm] n_total (all metals, smiles-filtered) = {len(full):,}")

    raw = load_tmqm(metals=None, require_smiles=False, refresh=False)
    print(f"[load_tmqm] n_total (all metals, incl. empty SMILES) = {len(raw):,}")
    print(f"[load_tmqm] dropped for empty SMILES = {len(raw) - len(full):,}")

    df = full if metals is None else full[full["metal"].isin(metals)].reset_index(drop=True)
    stats = summarize(df)

    label = "ALL" if metals is None else "+".join(metals)
    print(f"\n[load_tmqm] subset = {label}  n = {stats.n_total:,}")
    print(f"{'metal':<8}{'n':>8}{'mean_CN':>10}{'mean_BO':>10}   coord-number histogram")
    for metal, n in sorted(stats.per_metal.items(), key=lambda kv: -kv[1]):
        hist = stats.coord_distribution[metal]
        top = ", ".join(
            f"{cn}:{c} ({100.0 * c / n:.0f}%)"
            for cn, c in sorted(hist.items(), key=lambda kv: -kv[1])[:5]
        )
        print(
            f"{metal:<8}{n:>8,}{stats.coord_mean[metal]:>10.2f}"
            f"{stats.bo_mean[metal]:>10.2f}   {top}"
        )
    print(f"\n[load_tmqm] entries missing Wiberg BO = {stats.n_missing_bo:,}")

    payload = {
        "n_total_all_metals_with_smiles": int(len(full)),
        "n_total_all_metals_raw": int(len(raw)),
        "subset": label,
        **stats.to_dict(),
    }
    out = Path(args.out) if args.out else REPORTS_DIR / "f2_tmqm_stats.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[load_tmqm] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
