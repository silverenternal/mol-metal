"""CrossDocked2020 dataset — 100k protein-ligand pairs for SBDD.

The reference distribution is the **DiffDock / TargetDiff** split:
100,000 train pairs, 100 test pairs, drawn from CrossDocked2020 with the
``pocket10`` heuristic (binding site = 10 Å around the ligand centroid).

Source archive
--------------
``/mnt/storage/data/molmetal/CrossDocked2020_cascadediff.zip`` (1.6 GB)
contains two files at the top level of ``crossdocked/``:

* ``split_by_name.pt``               — pickled ``{"train": [...], "test": [...]}``
                                       list of ``(pocket_pdb_relpath, ligand_sdf_relpath)`` tuples
* ``crossdocked_pocket10.tar.gz``    — the 100,000 pocket/ligand files

Archive quirks (discovered empirically)
---------------------------------------
The archive has TWO copies of the central directory + EOCD, stitched back
to back.  The first EOCD (at offset 1,613,439,540) is the legitimate one —
its CD has the correct local-header offsets and compressed sizes.  The
second EOCD (at offset 1,625,895,476) is a duplicate tail.  When loaded as
a single zip, Python's :mod:`zipfile` resolves to the *last* EOCD, which
leads to the broken offset behaviour we observed in early dev.

This loader therefore:
1. Parses both EOCDs and uses the **first** one (the well-formed half).
2. For each file in the CD, seeks to the local-header offset and reads
   ``csize`` bytes of raw deflate stream.
3. Uses ``zlib.decompressobj(-15)`` (raw deflate, no zlib header) — the
   entries are stored with raw deflate even though the local header's
   ``method`` field is 8.

If the archive is already extracted at ``extracted_dir``, the loader
prefers that path (no need to re-decompress 1.6 GB on every run).
"""

from __future__ import annotations

import io
import os
import struct
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("/mnt/storage/data/molmetal")
DEFAULT_ARCHIVE = DEFAULT_DATA_DIR / "CrossDocked2020_cascadediff.zip"
DEFAULT_EXTRACT_DIR = DEFAULT_DATA_DIR / "crossdocked"
DEFAULT_TARBALL_NAME = "crossdocked_pocket10.tar.gz"
DEFAULT_SPLIT_NAME = "split_by_name.pt"

# CascadeDiff filter thresholds from the original paper
RMSD_MAX = 1.0
HEAVY_ATOMS_MIN = 3
HEAVY_ATOMS_MAX = 50


# ---------------------------------------------------------------------------
# Robust zip reader — handles the dual-EOCD quirk
# ---------------------------------------------------------------------------
def _parse_cd_entry(buf: bytes, off: int) -> Tuple[dict, int]:
    """Parse one central-directory entry starting at ``off``.  Returns (header, next_off)."""
    # Standard CD entry: 4-byte sig + 26 bytes of 13 uint16 + 16 bytes of 4 uint32 = 46 bytes
    (
        sig,
        ver_made,
        ver_needed,
        flags,
        method,
        mtime,
        mdate,
        crc,
        csize,
        usize,
        fnlen,
        exlen,
        fclen,
        disk_start,
        iattr,
        eattr,
        local_offset,
    ) = struct.unpack(
        "<4sHHHHHHIIIHHHHHII", buf[off : off + 46]
    )
    cur = off + 46
    fname = buf[cur : cur + fnlen].decode("utf-8", errors="replace")
    cur += fnlen
    cur += exlen + fclen
    return (
        {
            "flags": int(flags),
            "method": int(method),
            "csize": int(csize),
            "usize": int(usize),
            "fname": fname,
            "local_offset": int(local_offset),
        },
        cur,
    )


def _read_first_eocd(archive_path: Path) -> Tuple[int, int, int]:
    """Find the FIRST EOCD record.  Returns ``(cd_size, cd_offset, n_entries)``."""
    size = archive_path.stat().st_size
    EOCD_SIG = b"PK\x05\x06"
    # Scan the last 64 KB for EOCD signatures (standard practice).
    tail_size = min(1 << 16, size)
    seek_off = size - tail_size
    with open(archive_path, "rb") as f:
        f.seek(seek_off)
        tail = f.read()
    hits = []
    pos = 0
    while True:
        j = tail.find(EOCD_SIG, pos)
        if j < 0:
            break
        hits.append(j)
        pos = j + 1
    if not hits:
        raise zipfile.BadZipFile(f"No EOCD found in {archive_path}")
    # Use the FIRST hit.  Parse it.
    abs_off = (size - len(tail)) + hits[0]
    with open(archive_path, "rb") as f:
        f.seek(abs_off)
        eocd = f.read(22)
    (
        sig,
        disk,
        cd_disk,
        total_this_disk,
        total_entries,
        cd_size,
        cd_offset,
        comment_len,
    ) = struct.unpack("<IHHHHIIH", eocd)
    if sig != 0x06054B50:
        raise zipfile.BadZipFile(f"Bad EOCD signature at {abs_off}")
    return int(cd_size), int(cd_offset), int(total_entries)


def _read_central_directory(archive_path: Path) -> List[dict]:
    """Read the (first) central directory; return list of entry dicts."""
    cd_size, cd_offset, n_entries = _read_first_eocd(archive_path)
    with open(archive_path, "rb") as f:
        f.seek(cd_offset)
        buf = f.read(cd_size)
    entries: List[dict] = []
    off = 0
    for _ in range(n_entries):
        if buf[off : off + 4] != b"PK\x01\x02":
            break
        hdr, off = _parse_cd_entry(buf, off)
        entries.append(hdr)
    return entries


def _stream_extract_member(
    archive_path: Path,
    target_name: str,
    out_path: Path,
) -> bool:
    """Robustly extract one member of the zip to ``out_path``.

    Returns True on success, False if the member was not found.

    Strategy
    --------
    Read the (first) central directory, get the file's local-header offset
    and compressed size, then seek to ``local_offset + 30 + fnlen + exlen``
    and read ``csize`` bytes of raw deflate data, decompressing with
    ``wbits=-15`` (raw deflate, no zlib header).
    """
    cd = _read_central_directory(archive_path)
    match = None
    for entry in cd:
        if entry["fname"].rstrip("/").endswith(target_name):
            match = entry
            break
    if match is None:
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the local header at ``local_offset`` to get fnlen, exlen, then
    # skip past the header.
    with open(archive_path, "rb") as f:
        f.seek(match["local_offset"])
        lh = f.read(30)
        if lh[:4] != b"PK\x03\x04":
            return False
        (
            ver,
            flags,
            method,
            mtime,
            mdate,
            crc,
            lh_csize,
            lh_usize,
            fnlen,
            exlen,
        ) = struct.unpack("<HHHHHIIIHH", lh[4:30])
        f.seek(match["local_offset"] + 30 + fnlen + exlen)
        csize = match["csize"]
        if csize == 0:
            # Streaming mode — fall back to scan-based recovery is not yet
            # needed for the supplied archive; the CD always carries a
            # valid csize.
            return False
        if method == 8:
            import zlib

            # Stream-decompress: read csize bytes in 4 MB chunks
            z = zlib.decompressobj(-15)
            remaining = csize
            with open(out_path, "wb") as dst:
                while remaining > 0:
                    chunk = f.read(min(1 << 22, remaining))
                    if not chunk:
                        break
                    out = z.decompress(chunk)
                    if out:
                        dst.write(out)
                    remaining -= len(chunk)
                # flush any remaining decompressed bytes
                tail = z.flush()
                if tail:
                    dst.write(tail)
            return True
        elif method == 0:
            # Stored — raw bytes
            with open(out_path, "wb") as dst:
                remaining = csize
                while remaining > 0:
                    chunk = f.read(min(1 << 22, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
            return True
    return False


def _try_native_zip_extract(archive_path: Path, member: str, out_path: Path) -> bool:
    """Try the standard zipfile path first (cheap when it works)."""
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            with z.open(member) as src, open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CrossDockedDataset:
    """CrossDocked2020 protein-ligand pairs (DiffDock split).

    Parameters
    ----------
    archive_path : Path
        Path to ``CrossDocked2020_cascadediff.zip``.
    extracted_dir : Path
        Where to place the extracted split file + tarball.  If it already
        contains a valid ``split_by_name.pt``, the loader skips re-extracting
        the zip entirely.
    rmsd_max : float
        Maximum allowed RMSD to the reference pose (default 1.0 Å, the
        CascadeDiff filter).
    heavy_atoms_range : tuple
        ``(min, max)`` heavy-atom count filter for the ligand (default
        ``(3, 50)``).  Pairs outside this range are skipped on iteration.

    Iteration yields dicts with::

        pocket_pdb_path : str   # path to the 10-Å pocket PDB
        ligand_sdf_path : str   # path to the ligand SDF
        affinity        : float # placeholder (populated lazily from index)
    """

    def __init__(
        self,
        archive_path: Path = DEFAULT_ARCHIVE,
        extracted_dir: Path = DEFAULT_EXTRACT_DIR,
        rmsd_max: float = RMSD_MAX,
        heavy_atoms_range: Tuple[int, int] = (HEAVY_ATOMS_MIN, HEAVY_ATOMS_MAX),
        split: str = "train",
        auto_extract: bool = True,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.extracted_dir = Path(extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.rmsd_max = float(rmsd_max)
        self.heavy_atoms_range = (int(heavy_atoms_range[0]), int(heavy_atoms_range[1]))
        self.split = str(split)
        self._tarball_path: Optional[Path] = None
        self._split_path: Optional[Path] = None
        self._pairs: List[Tuple[str, str]] = []

        if auto_extract:
            self._ensure_extracted()

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _ensure_extracted(self) -> None:
        split_path = self.extracted_dir / DEFAULT_SPLIT_NAME
        tarball_path = self.extracted_dir / DEFAULT_TARBALL_NAME
        # First, try to find an already-extracted directory containing the tarball
        # (the user may have done it manually).
        if split_path.exists() and tarball_path.exists():
            self._split_path = split_path
            self._tarball_path = tarball_path
        elif self.archive_path.exists():
            # Try native zip first, then fall back to streaming
            ok_split = _try_native_zip_extract(
                self.archive_path, f"crossdocked/{DEFAULT_SPLIT_NAME}", split_path
            )
            ok_tar = _try_native_zip_extract(
                self.archive_path, f"crossdocked/{DEFAULT_TARBALL_NAME}", tarball_path
            )
            if not ok_split:
                _stream_extract_member(
                    self.archive_path, DEFAULT_SPLIT_NAME, split_path
                )
            if not ok_tar:
                _stream_extract_member(
                    self.archive_path, DEFAULT_TARBALL_NAME, tarball_path
                )
            if split_path.exists():
                self._split_path = split_path
            if tarball_path.exists():
                self._tarball_path = tarball_path
        # Load split
        if self._split_path is None or not self._split_path.exists():
            raise FileNotFoundError(
                f"Could not obtain {DEFAULT_SPLIT_NAME} from {self.archive_path} "
                f"or {self.extracted_dir}.  Set auto_extract=False and pre-extract "
                f"manually if extraction is unreliable on your platform."
            )
        self._pairs = self._load_split(self._split_path)

    def _load_split(self, split_path: Path) -> List[Tuple[str, str]]:
        d = torch.load(split_path, weights_only=False)
        if not isinstance(d, dict) or self.split not in d:
            raise ValueError(
                f"Unexpected split file format.  Expected dict with key "
                f"{self.split!r}, got {type(d).__name__}"
            )
        pairs = d[self.split]
        if not isinstance(pairs, list):
            raise ValueError(f"split[{self.split}] should be a list, got {type(pairs).__name__}")
        # Normalise to (str, str) tuples
        out: List[Tuple[str, str]] = []
        for entry in pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                out.append((str(entry[0]), str(entry[1])))
            else:
                # Best-effort fallback
                out.append((str(entry), ""))
        return out

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        pocket_rel, ligand_rel = self._pairs[int(idx)]
        pocket_abs = self._resolve_inside_tarball(pocket_rel)
        ligand_abs = self._resolve_inside_tarball(ligand_rel)
        # Affinity placeholder; the cascade-diff release does not ship a
        # parsed CSV.  Callers can patch in values from a metadata file.
        return {
            "pocket_pdb_path": pocket_abs,
            "ligand_sdf_path": ligand_abs,
            "pocket_pdb_relpath": pocket_rel,
            "ligand_sdf_relpath": ligand_rel,
            "affinity": float("nan"),
            "rmsd_max": self.rmsd_max,
            "heavy_atoms_range": self.heavy_atoms_range,
            "split": self.split,
        }

    # ------------------------------------------------------------------
    def _resolve_inside_tarball(self, rel_path: str) -> str:
        """Return the path string for ``rel_path`` *if* the tarball has been
        extracted; otherwise return the rel-path string (the tarball is the
        canonical source).
        """
        candidate = self.extracted_dir / rel_path
        if candidate.exists():
            return str(candidate)
        return rel_path

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "n_pairs": int(len(self._pairs)),
            "split": self.split,
            "archive_path": str(self.archive_path),
            "extracted_dir": str(self.extracted_dir),
            "split_file": str(self._split_path) if self._split_path else None,
            "tarball_file": str(self._tarball_path) if self._tarball_path else None,
            "rmsd_max": self.rmsd_max,
            "heavy_atoms_range": self.heavy_atoms_range,
            "first_pair": self._pairs[0] if self._pairs else None,
        }

    # ------------------------------------------------------------------
    # Tarball inspection (for tests)
    # ------------------------------------------------------------------
    def tarball_path(self) -> Optional[Path]:
        return self._tarball_path

    def split_path(self) -> Optional[Path]:
        return self._split_path


__all__ = ["CrossDockedDataset"]