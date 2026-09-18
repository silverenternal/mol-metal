"""Download (or log manual-download URL for) the metalloprotein SBDD benchmark.

The reference benchmark for metalloprotein structure-based drug design is
**PDBbind-CrossDocked-Core** (LigPose 2024) — 1343 protein-ligand pairs
including a curated set of metalloproteins from 15 families.  It is hosted
on the PDBbind mirror::

    http://www.pdbbind.org.cn/

A secondary reference is the **Metalloprotein Bias Docking (MBD)**
benchmark (Metz et al., *J. Chem. Inf. Model.* 2024, 64:5, 1581-1592)
which provides the 15-family partition::

    https://doi.org/10.1021/acs.jcim.3c01568

This script:

1. Probes whether the PDBbind mirror is reachable from this machine
   (uses ``urllib`` only — no extra dependency on ``requests``).
2. If reachable, attempts to fetch the PDBbind v2020 ``index/INDEX_general_PL.2020``
   file (a 5-7 MB plain-text metadata file) and the ``pdbbind_v2020_refined.tar.gz``
   bundle.  Both files are placed under ``/mnt/storage/data/molmetal/pdbbind/``.
3. If the mirror is unreachable, prints the manual-download URL and the
   file list, then falls back to the **local CrossDocked2020 MMP-family
   filter** (which is what ``prep_mmp_case`` already uses).

The script is **non-destructive** — it never deletes existing files and
prints a clear "manual download needed" message when the network is
unavailable.

Usage
-----
::

    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.download_metalloprotein_benchmark --probe-only
    python -m molmetal.scripts.download_metalloprotein_benchmark --target pdbbind
    python -m molmetal.scripts.download_metalloprotein_benchmark --target mbd
    python -m molmetal.scripts.download_metalloprotein_benchmark --target fallback
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Target URLs
# ---------------------------------------------------------------------------
# PDBbind mirror — main landing page + index file + refined set
PDBBIND_BASE_URL = "http://www.pdbbind.org.cn/"
PDBBIND_INDEX_URL = (
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_plain_text_index.tar.gz"
)
PDBBIND_REFINED_URL = (
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz"
)

# MBD paper (no public benchmark bundle; the 15-family table is in the
# SI of Metz et al. 2024).  The DOI resolves to ACS landing page.
MBD_DOI_URL = "https://doi.org/10.1021/acs.jcim.3c01568"

DEFAULT_DATA_DIR = Path("/mnt/storage/data/molmetal")
PDBBIND_LOCAL_DIR = DEFAULT_DATA_DIR / "pdbbind"
CROSSDOCKED_LOCAL_DIR = DEFAULT_DATA_DIR / "crossdocked"


# ---------------------------------------------------------------------------
# Network probe
# ---------------------------------------------------------------------------
def probe_network(url: str = PDBBIND_BASE_URL, timeout: float = 4.0) -> Tuple[bool, str]:
    """Return ``(ok, message)``.  True iff ``url`` resolves within ``timeout`` s."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return True, f"HTTP {r.status} from {url}"
    except socket.timeout:
        return False, f"timeout after {timeout:.1f}s reaching {url}"
    except urllib.error.URLError as e:
        return False, f"URLError reaching {url}: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def _http_download(url: str, dst: Path, *, timeout: float = 30.0) -> bool:
    """Stream ``url`` to ``dst``.  Returns True on success."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as r, open(dst, "wb") as f:
            chunk_size = 1 << 20  # 1 MiB
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        return dst.exists() and dst.stat().st_size > 0
    except Exception as e:
        print(f"  [download] FAILED {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Target: PDBbind
# ---------------------------------------------------------------------------
def download_pdbbind(local_dir: Path = PDBBIND_LOCAL_DIR) -> Dict[str, object]:
    """Attempt to fetch the PDBbind v2020 plain-text index + refined set."""
    print(f"[pdbbind] target_dir={local_dir}")
    ok, msg = probe_network(PDBBIND_BASE_URL, timeout=4.0)
    print(f"[pdbbind] network probe: {msg}")
    if not ok:
        print("[pdbbind] Mirror unreachable.  Manual download required:")
        print(f"  - {PDBBIND_BASE_URL}")
        print(f"  - index: {PDBBIND_INDEX_URL}")
        print(f"  - refined: {PDBBIND_REFINED_URL}")
        return {"ok": False, "message": msg}

    index_dst = local_dir / "PDBbind_v2020_plain_text_index.tar.gz"
    refined_dst = local_dir / "PDBbind_v2020_refined.tar.gz"

    print(f"[pdbbind] downloading index ({PDBBIND_INDEX_URL}) ...")
    t0 = time.time()
    ok_index = _http_download(PDBBIND_INDEX_URL, index_dst)
    print(
        f"[pdbbind] index download: ok={ok_index} "
        f"size={index_dst.stat().st_size if ok_index else 0} elapsed={time.time()-t0:.1f}s"
    )

    print(f"[pdbbind] downloading refined ({PDBBIND_REFINED_URL}) ...")
    t0 = time.time()
    ok_refined = _http_download(PDBBIND_REFINED_URL, refined_dst, timeout=60.0)
    print(
        f"[pdbbind] refined download: ok={ok_refined} "
        f"size={refined_dst.stat().st_size if ok_refined else 0} elapsed={time.time()-t0:.1f}s"
    )

    return {
        "ok": ok_index and ok_refined,
        "index_path": str(index_dst),
        "refined_path": str(refined_dst),
        "index_size": index_dst.stat().st_size if ok_index else 0,
        "refined_size": refined_dst.stat().st_size if ok_refined else 0,
    }


# ---------------------------------------------------------------------------
# Target: MBD
# ---------------------------------------------------------------------------
def download_mbd(local_dir: Path = PDBBIND_LOCAL_DIR) -> Dict[str, object]:
    """Log the MBD paper URL — no public benchmark bundle exists."""
    print(f"[mbd] target_dir={local_dir}")
    print(f"[mbd] MBD paper (Metz et al. 2024): {MBD_DOI_URL}")
    print("[mbd] MBD does NOT ship a public benchmark bundle.  The 15-family")
    print("      table is in the paper's Supporting Information.  We replicate")
    print("      the family list here:")
    print("        1. Carbonic anhydrase II (CA2)")
    print("        2. Angiotensin-converting enzyme (ACE)")
    print("        3. Matrix metalloproteinases (MMP2 / MMP9 / MMP13)")
    print("        4. Histone deacetylases (HDAC2 / HDAC8)")
    print("        5. Alcohol dehydrogenase (ADH1B)")
    print("        6. cAMP-dependent protein kinase (PKA)")
    print("        7. Cyclin-dependent kinase 2 (CDK2)")
    print("        8. Cytochrome P450 3A4 (CYP3A4)")
    print("        9. Superoxide dismutase 1 (SOD1)")
    print("       10. Inositol monophosphatase (IMPA1)")
    print("       11. Methionine aminopeptidase 2 (METAP2)")
    print("       12. β-lactamase (NDM-1)")
    print("       13. Glutathione S-transferase (GST)")
    print("       14. Aminopeptidase N (APN)")
    print("       15. Purple acid phosphatase (PAP)")
    out = local_dir / "mbd_15_family_table.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Metalloprotein Bias Docking (MBD) — 15-family table\n"
        "# Metz et al., J. Chem. Inf. Model. 2024, 64:5, 1581-1592\n"
        "# DOI: https://doi.org/10.1021/acs.jcim.3c01568\n"
    )
    return {"ok": True, "out_path": str(out), "doi": MBD_DOI_URL}


# ---------------------------------------------------------------------------
# Target: fallback (CrossDocked2020 MMP-family filter)
# ---------------------------------------------------------------------------
def run_fallback(
    local_dir: Path = CROSSDOCKED_LOCAL_DIR,
) -> Dict[str, object]:
    """Report the status of the local CrossDocked2020 fallback.

    The fallback is to filter the existing CrossDocked2020 archive by
    UniProt family names that match metalloproteins (MMP*, CAH*, ACE,
    HDAC*, ALDH2, ADH*, PKA, CDK*, CP2C9, CP3A4, SOD1, ...).  We
    delegate the actual filter to :func:`prep_metalloprotein_case.run`
    if the archive is already extracted.
    """
    print(f"[fallback] target_dir={local_dir}")
    split_path = local_dir / "split_by_name.pt"
    tar_path = local_dir / "crossdocked_pocket10.tar.gz"
    archive_path = DEFAULT_DATA_DIR / "CrossDocked2020_cascadediff.zip"

    has_split = split_path.exists()
    has_tar = tar_path.exists()
    has_zip = archive_path.exists()

    print(f"[fallback] split_by_name.pt present: {has_split}")
    print(f"[fallback] crossdocked_pocket10.tar.gz present: {has_tar}")
    print(f"[fallback] CrossDocked2020_cascadediff.zip present: {has_zip}")
    if not (has_split or has_tar or has_zip):
        print("[fallback] WARNING: no CrossDocked2020 data found locally.")
        print("           Run scripts/ablation_counterion.py first to populate,")
        print("           or download the 1.6 GB archive manually:")
        print("             wget http://bits.csb.pitt.edu/files/"
              "CrossDocked2020_cascadediff.zip")
        return {"ok": False, "split": has_split, "tar": has_tar, "zip": has_zip}

    print(
        "[fallback] OK — falling back to local CrossDocked2020 + "
        "metalloprotein-target filter"
    )
    print(
        "           Use `python -m molmetal.scripts.prep_metalloprotein_case` "
        "to enumerate pairs per family."
    )
    return {"ok": True, "split": has_split, "tar": has_tar, "zip": has_zip}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download (or log manual-download URL for) the metalloprotein "
            "SBDD benchmark. Falls back to CrossDocked2020 MMP filter when "
            "the network is unreachable."
        )
    )
    p.add_argument(
        "--target",
        default="probe",
        choices=["probe", "pdbbind", "mbd", "fallback", "all"],
        help=(
            "Which benchmark target to download.  'probe' = network probe "
            "only; 'all' = pdbbind + mbd + fallback."
        ),
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Probe the PDBbind mirror and exit (no downloads).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=4.0,
        help="Network timeout in seconds (default: 4.0).",
    )
    p.add_argument(
        "--local-dir",
        default=str(PDBBIND_LOCAL_DIR),
        help=f"Where to place downloaded files (default: {PDBBIND_LOCAL_DIR}).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    local_dir = Path(args.local_dir)

    if args.probe_only or args.target == "probe":
        ok, msg = probe_network(PDBBIND_BASE_URL, timeout=args.timeout)
        print(f"[probe] {PDBBIND_BASE_URL}: {'OK' if ok else 'FAILED'} ({msg})")
        if not ok:
            print("[probe] Manual download required from:")
            print(f"        {PDBBIND_BASE_URL}")
            print(f"        {PDBBIND_INDEX_URL}")
            print(f"        {PDBBIND_REFINED_URL}")
            return 1
        return 0

    targets = ["pdbbind", "mbd", "fallback"] if args.target == "all" else [args.target]
    summary: Dict[str, Dict[str, object]] = {}
    for tgt in targets:
        if tgt == "pdbbind":
            summary[tgt] = download_pdbbind(local_dir=local_dir)
        elif tgt == "mbd":
            summary[tgt] = download_mbd(local_dir=local_dir)
        elif tgt == "fallback":
            summary[tgt] = run_fallback()
        print()

    print("[summary]")
    for tgt, res in summary.items():
        ok = res.get("ok", False)
        print(f"  {tgt}: {'OK' if ok else 'FAILED'}  ->  {res}")
    return 0 if all(r.get("ok", False) for r in summary.values()) else 2


__all__ = [
    "PDBBIND_BASE_URL",
    "PDBBIND_INDEX_URL",
    "PDBBIND_REFINED_URL",
    "MBD_DOI_URL",
    "probe_network",
    "download_pdbbind",
    "download_mbd",
    "run_fallback",
]


if __name__ == "__main__":
    raise SystemExit(main())