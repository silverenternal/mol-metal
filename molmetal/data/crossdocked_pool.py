"""CrossDocked 2020 pocket pool — 100K canonical benchmark loader.

2026-10-13 — Wf-A14 plan: closes Gap 1 (pocket pool too small).
The local ``.data/molmetal/crossdocked/`` directory holds only 9.7 MB
(mostly empty placeholders). The real CrossDocked 2020 100K benchmark
(the same one used by Pocket2Mol / TargetDiff / DecompDiff / DiffBP /
RxnFlow) lives in the sibling project
``/home/hugo/codes/pocket/data/crossdocked2020/`` and exposes a
pre-built DiffDock split via ``manifests/{train,test}_manifest.csv``.

This module is the canonical access surface:

    from molmetal.data.crossdocked_pool import list_pocket_pool
    train = list_pocket_pool(split="train")  # ≥ 90k PocketEntry rows
    test  = list_pocket_pool(split="test")   # exactly 100 rows (DiffDock)

A :class:`PocketEntry` is a frozen dataclass with the absolute paths to
the pocket PDB + docked ligand SDF, the target id (CSD-style refcode
prefix), and the SHA256 of both files (so reviewers can verify the
pool against the public manifest).

Backward compat
---------------
If the sibling project is missing, the module raises
:class:`FileNotFoundError` with a clear remediation message — the
error never propagates as a bare ``None`` that downstream code
silently ignores (the failure mode that originally hid this gap).
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger("molmetal.data.crossdocked_pool")


# ---------------------------------------------------------------------------
# Default path resolution
# ---------------------------------------------------------------------------
def _default_root() -> Path:
    """Return the canonical CrossDocked 2020 root, with overrides.

    Precedence (highest to lowest):

    1. ``CROSSDOCKED_2020_ROOT`` environment variable (set by the
       launch wrapper or a workflow).
    2. The sibling project location
       ``/home/hugo/codes/pocket/data/crossdocked2020`` (the canonical
       location on this host).
    3. ``molmetal/.data/crossdocked2020`` (fallback for hosts that
       mirror the dataset inside the repo).
    """
    env = os.environ.get("CROSSDOCKED_2020_ROOT")
    if env:
        return Path(env)
    canonical = Path("/home/hugo/codes/pocket/data/crossdocked2020")
    if canonical.is_dir():
        return canonical
    return Path(__file__).resolve().parents[2] / ".data" / "crossdocked2020"


CROSSDOCKED_2020_ROOT: Path = _default_root()
TRAIN_MANIFEST: Path = CROSSDOCKED_2020_ROOT / "manifests" / "train_manifest.csv"
TEST_MANIFEST: Path = CROSSDOCKED_2020_ROOT / "manifests" / "test_manifest.csv"
POCKET10_SUBDIR: Path = CROSSDOCKED_2020_ROOT / "pocket10" / "crossdocked_pocket10"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PocketEntry:
    """One row of the CrossDocked 2020 manifest.

    Attributes
    ----------
    target_id : str
        CSD-style refcode prefix (e.g. ``DYR_STAAU_2_158_0``) — the
        unique pocket identifier shared by all docking poses for that
        binding site.
    pocket_path : str
        Absolute path to the 10-Å pocket PDB on disk.
    pocket_sha256 : str
        SHA256 of the pocket PDB (per the manifest's claim; we trust
        the manifest because the upstream download is deterministic).
    ligand_path : str
        Absolute path to the docked ligand SDF on disk.
    ligand_sha256 : str
        SHA256 of the ligand SDF.
    split_role : str
        ``"train"`` or ``"test"`` per the DiffDock split.
    pocket_definition_provenance : str
        Free-form provenance string (e.g.
        ``"public_benchmark_prepared_pocket"``).
    native_ligand_allowed_for_generation : bool
        Whether the native ligand may be used as a generation seed
        (False for both train + test in the standard release).
    """

    target_id: str
    pocket_path: str
    pocket_sha256: str
    ligand_path: str
    ligand_sha256: str
    split_role: str
    pocket_definition_provenance: str
    native_ligand_allowed_for_generation: bool

    def pocket_exists(self) -> bool:
        return Path(self.pocket_path).is_file()

    def ligand_exists(self) -> bool:
        return Path(self.ligand_path).is_file()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_pocket_pool(
    split: str = "train",
    max_n: Optional[int] = None,
    require_paths_exist: bool = False,
) -> List[PocketEntry]:
    """Return the canonical CrossDocked 2020 pocket pool for ``split``.

    Parameters
    ----------
    split
        ``"train"`` (default — 100K) or ``"test"`` (100, DiffDock).
    max_n
        Cap on returned rows.  ``None`` returns the full pool.  Used
        by the R21 GPU budget (capped to ~10k for the first pass).
    require_paths_exist
        When ``True``, filter out rows whose ``pocket_path`` or
        ``ligand_path`` is missing on disk.  Default ``False`` so the
        manifest itself is the contract (path existence is a separate
        check — see :func:`verify_pool_paths`).

    Returns
    -------
    list[PocketEntry]
        The pool as a deterministic list (the manifest is already
        sorted by target_id in the upstream build).

    Raises
    ------
    FileNotFoundError
        When the root or the manifest is missing.  The error names the
        resolved root so a reviewer can see what path was tried.
    """
    split = str(split).strip().lower()
    if split not in ("train", "test"):
        raise ValueError(
            f"split must be 'train' or 'test', got {split!r}"
        )
    manifest = TRAIN_MANIFEST if split == "train" else TEST_MANIFEST
    if not CROSSDOCKED_2020_ROOT.is_dir():
        raise FileNotFoundError(
            f"CrossDocked 2020 root not found at {CROSSDOCKED_2020_ROOT}. "
            f"Set CROSSDOCKED_2020_ROOT env var, or point the canonical "
            f"sibling project to this path.  Tried (in order): env var, "
            f"/home/hugo/codes/pocket/data/crossdocked2020, "
            f"{CROSSDOCKED_2020_ROOT}."
        )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"CrossDocked 2020 {split} manifest not found at {manifest}. "
            f"Re-run the upstream prepare step, or check that the "
            f"manifests/ subdirectory is intact."
        )
    out: List[PocketEntry] = []
    with manifest.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Wf-A30-2 B1 (2026-10-14) — schema-drift guard.
        # Every CrossDocked 2020 manifest must carry these canonical
        # columns; if any are missing the upstream prepare step has
        # drifted (e.g. re-export from a newer version that renamed
        # ligand_path -> reference_ligand_path). The legacy names
        # ``ligand_path`` / ``ligand_sha256`` are accepted by alias so
        # older manifests from Wf-124 / Wf-181 (G4/G5/G6 launchers)
        # keep working. Fail loud rather than silently skipping every
        # row with KeyError, which was the prior behaviour and masked
        # the upstream schema drift until the per-pocket pool came
        # back empty.
        expected_cols = {
            "target_id",
            "pocket_path",
            "pocket_sha256",
            "reference_ligand_path",
            "reference_ligand_sha256",
        }
        # Per-column legacy alias map.  When the upstream prepare step
        # renamed ``ligand_path`` -> ``reference_ligand_path`` (Wf-32
        # schema bump), the two columns can be substituted one-for-one;
        # other columns have no legacy equivalent.
        legacy_alias_map = {
            "reference_ligand_path": "ligand_path",
            "reference_ligand_sha256": "ligand_sha256",
        }
        actual_cols = set(reader.fieldnames or [])
        missing = expected_cols - actual_cols
        if missing:
            # Allow legacy alias only for the two columns that were
            # renamed in Wf-32 schema bump.
            unaliasable = [m for m in missing if m not in legacy_alias_map]
            if unaliasable:
                raise ValueError(
                    f"[crossdocked_pool] manifest schema drift: missing "
                    f"cols={sorted(missing)} (no legacy alias available for "
                    f"{sorted(unaliasable)}); re-run the upstream prepare "
                    f"step"
                )
            LOG.info(
                "[crossdocked_pool] %s manifest uses legacy column "
                "names %s; aliasing to reference_ligand_*",
                split, sorted(legacy_alias_map[m] for m in missing),
            )
        for row in reader:
            try:
                native_ok = str(
                    row.get("native_ligand_allowed_for_generation", "false")
                ).strip().lower() in ("true", "1", "yes")
                # Translate legacy column names if the alias path
                # accepted them above.
                if "reference_ligand_path" not in row and "ligand_path" in row:
                    row["reference_ligand_path"] = row["ligand_path"]
                if "reference_ligand_sha256" not in row and "ligand_sha256" in row:
                    row["reference_ligand_sha256"] = row["ligand_sha256"]
                entry = PocketEntry(
                    target_id=str(row["target_id"]).strip(),
                    pocket_path=str(row["pocket_path"]).strip(),
                    pocket_sha256=str(row["pocket_sha256"]).strip(),
                    ligand_path=str(row["reference_ligand_path"]).strip(),
                    ligand_sha256=str(
                        row.get("reference_ligand_sha256", "")
                    ).strip(),
                    split_role=str(row["split_role"]).strip(),
                    pocket_definition_provenance=str(
                        row.get("pocket_definition_provenance", "")
                    ).strip(),
                    native_ligand_allowed_for_generation=bool(native_ok),
                )
            except KeyError as exc:
                LOG.warning(
                    "[crossdocked_pool] %s manifest row missing %s; skip",
                    split, exc,
                )
                continue
            if require_paths_exist and not (
                entry.pocket_exists() and entry.ligand_exists()
            ):
                continue
            out.append(entry)
            if max_n is not None and len(out) >= int(max_n):
                break
    LOG.info(
        "[crossdocked_pool] loaded %d %s pockets from %s",
        len(out), split, manifest,
    )
    return out


def verify_pool_paths(
    pool: List[PocketEntry],
    sample_n: int = 1000,
    seed: int = 0,
) -> dict:
    """Sample-check that the manifest paths actually exist on disk.

    The manifest itself does not guarantee that the 100K pockets are
    present on this host; this function samples ``sample_n`` rows
    (deterministic with ``seed``) and reports how many of those
    have both the PDB and the SDF on disk.

    Returns
    -------
    dict
        ``{"sampled": int, "pocket_ok": int, "ligand_ok": int,
        "both_ok": int, "first_missing": Optional[str]}``.
    """
    import random

    if not pool:
        return {
            "sampled": 0, "pocket_ok": 0, "ligand_ok": 0,
            "both_ok": 0, "first_missing": None,
        }
    rng = random.Random(int(seed))
    sample = rng.sample(pool, k=min(int(sample_n), len(pool)))
    pocket_ok = sum(1 for e in sample if e.pocket_exists())
    ligand_ok = sum(1 for e in sample if e.ligand_exists())
    both_ok = sum(
        1 for e in sample if e.pocket_exists() and e.ligand_exists()
    )
    first_missing = next(
        (
            e.target_id
            for e in sample
            if not (e.pocket_exists() and e.ligand_exists())
        ),
        None,
    )
    return {
        "sampled": len(sample),
        "pocket_ok": pocket_ok,
        "ligand_ok": ligand_ok,
        "both_ok": both_ok,
        "first_missing": first_missing,
    }


# ---------------------------------------------------------------------------
# Convenience: dataset-style iteration for the CFM trainer
# ---------------------------------------------------------------------------
def iter_pocket_targets(
    split: str = "train",
    max_n: Optional[int] = None,
) -> List[str]:
    """Return just the ``target_id`` list — handy for sampling pockets
    per-cloud without holding the full PocketEntry in memory.

    The CFM trainer can do ``np.random.choice(iter_pocket_targets(...))``
    at cloud-build time to attach a per-cloud pocket id, then call
    :func:`molmetal.data.tmqm.load_pocket_from_pdb` for that target.

    Parameters
    ----------
    split
        ``"train"`` or ``"test"``.
    max_n
        Cap on returned target ids.  ``None`` = all.
    """
    return [
        e.target_id
        for e in list_pocket_pool(split=split, max_n=max_n)
    ]


__all__ = [
    "PocketEntry",
    "CROSSDOCKED_2020_ROOT",
    "TRAIN_MANIFEST",
    "TEST_MANIFEST",
    "POCKET10_SUBDIR",
    "list_pocket_pool",
    "iter_pocket_targets",
    "verify_pool_paths",
]
