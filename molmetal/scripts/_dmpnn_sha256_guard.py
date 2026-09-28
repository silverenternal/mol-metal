"""Wf-120 Phase 3 — D-MPNN pIC50 SHA256 overwrite guard helpers.

=====================================================================
Pure-stdlib SHA256 utilities for ``train_dmpnn_pic50_real.py``
=====================================================================

This module is deliberately **dependency-free** (only ``hashlib``,
``pathlib``, ``typing``). It exists so that:

  * ``train_dmpnn_pic50_real.py`` can import the guard without pulling
    ``pandas`` / ``torch`` / ``rdkit`` into the test path;
  * unit tests can exercise the guard via ``import
    molmetal.scripts._dmpnn_sha256_guard`` without side-effects;
  * the SHA256-audit script (``dmpnn_sha256_audit.py``) and the 5-seed
    driver (``train_dmpnn_pic5_5seed.py``) share the same canonical
    helper.

**Two-phase guard**
-------------------
Wf-120 P3 uses a *two-phase* guard that defends against the silent
copy-paste bug observed in Wf-120 P0 (Rh seed0 ckpt == Rh seed1 ckpt):

  1. **PRE-train** (``pre_train_sha256_overwrite_guard``) — invoked
     *before* training starts. If the target ``(metal, seed)`` ckpt
     already exists AND its SHA256 byte-matches a *different* (metal',
     seed') sibling, refuse to start training. This catches the case
     where a previous run left a buggy duplicate on disk.

  2. **POST-save** (``post_save_sha256_overwrite_guard``) — invoked
     *after* ``torch.save`` writes the new ckpt. Computes the just-
     written file's SHA256 and verifies it does NOT byte-match any
     sibling. If it does:

       * the just-written file is removed (rolling back the bad save),
       * a ``RuntimeError`` is raised with a loud message,
       * the SHA256 is logged to stdout before the raise.

This is the smoking-gun check that would have caught the Rh seed0=seed1
bug — the post-save SHA256 of a re-trained ``rh_seed1.pt`` would have
matched the on-disk ``rh_seed0.pt``.

**Hardware**: CPU only (SHA256 + path operations).

**Audit log**
------------
Each invocation logs to stdout a single line of the form::

    [wf120-dmpnn-sha256-guard] <filename_or_tag> sha256=<hex> ...

so the audit trail can be captured by the wrapping CLI driver.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# Tag emitted on every guard log line — easy to grep for in CI logs.
GUARD_TAG = "[wf120-dmpnn-sha256-guard]"


# ---------------------------------------------------------------------------
# Core SHA256 helper
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Compute SHA256 of ``path``, streaming 1 MiB chunks.

    Identical contract to ``molmetal.scripts.dmpnn_sha256_audit.sha256_of_file``
    — duplicated here so this module has zero project-internal imports
    (keeps the test surface small).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Filename parser (shared with audit)
# ---------------------------------------------------------------------------
_FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")


def parse_dmpnn_filename(p: Path) -> Optional[Tuple[str, int]]:
    """Parse ``dmpnn_pic50_{metal}_seed{N}.pt`` → ``(metal, seed)`` or None.

    Mirrors ``molmetal.scripts.dmpnn_sha256_audit.parse_filename`` for the
    subset of cases the guard cares about.
    """
    stem = p.stem  # dmpnn_pic50_{metal}_seed{N}
    parts = stem.split("_")
    if len(parts) < 4 or not parts[-1].startswith("seed"):
        return None
    seed_str = parts[-1][4:]
    if not seed_str.isdigit():
        return None
    metal = "_".join(parts[2:-1])
    metal_cap = metal.capitalize()
    if metal_cap not in _FIVE_METALS:
        return None
    return metal_cap, int(seed_str)


def target_path(metal: str, seed: int, ckpt_dir: Path) -> Path:
    """Return ``ckpt_dir / dmpnn_pic50_{metal.lower()}_seed{seed}.pt``."""
    return ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"


# ---------------------------------------------------------------------------
# Pre-train guard (Wf-120 P1.5 — already existed; re-homed here)
# ---------------------------------------------------------------------------
def pre_train_sha256_overwrite_guard(
    metal: str,
    seed: int,
    ckpt_dir: Path,
) -> None:
    """Refuse to train if an existing target ckpt's SHA256 matches a sibling.

    Walks every existing ``dmpnn_pic50_*.pt`` under ``ckpt_dir``, hashes
    them, and verifies that the **target** ``(metal, seed)`` ckpt — if it
    already exists — has a SHA256 that does NOT byte-match any other
    (metal', seed') ckpt. Raises ``RuntimeError`` with a loud message
    instead of silently overwriting a known-bad duplicate.

    Note: this guard does NOT block first-time training. If the target
    ckpt does not yet exist, the function returns silently.

    Parameters
    ----------
    metal : str
        Metal symbol (e.g. ``"Rh"``) — used to build the target filename.
    seed : int
        Random seed (e.g. ``0``) — used to build the target filename.
    ckpt_dir : Path
        Directory holding ``dmpnn_pic50_*.pt`` files.

    Raises
    ------
    RuntimeError
        If the existing target ckpt's SHA256 matches a sibling ckpt.
    """
    target = target_path(metal, seed, ckpt_dir)
    if not target.exists():
        return  # first-time training; no pre-existing duplicate to check.

    target_sha = sha256_of_file(target)
    siblings = sorted(p for p in ckpt_dir.glob("dmpnn_pic50_*.pt") if p != target)
    for sib in siblings:
        if sha256_of_file(sib) == target_sha:
            raise RuntimeError(
                f"{GUARD_TAG} refusing to overwrite {target.name}: its "
                f"SHA256 matches sibling {sib.name}. This is the silent-"
                f"copy bug signature — delete one of the duplicate files "
                f"manually before retraining."
            )


# ---------------------------------------------------------------------------
# Post-save guard (Wf-120 P3 — NEW)
# ---------------------------------------------------------------------------
def post_save_sha256_overwrite_guard(
    metal: str,
    seed: int,
    ckpt_dir: Path,
    *,
    remove_on_fail: bool = True,
) -> str:
    """Wf-120 Phase 3 — POST-save SHA256 overwrite guard.

    Invoked immediately after ``torch.save`` writes a new
    ``dmpnn_pic50_{metal.lower()}_seed{seed}.pt`` file. Verifies that the
    just-written file's SHA256 does NOT byte-match any other D-MPNN pIC50
    ckpt already on disk. This is the smoking-gun check that would have
    caught the Wf-120 P0 silent-copy bug (Rh seed0 ckpt == Rh seed1 ckpt).

    Behaviour
    ---------
    1. Compute ``new_sha = sha256_of_file(target)``.
    2. Log ``new_sha`` to stdout for the audit trail.
    3. For every *sibling* ``dmpnn_pic50_*.pt`` (≠ target), compute its
       SHA256. If any sibling SHA256 equals ``new_sha``:
         * if ``remove_on_fail`` is True (default), unlink the just-
           written file (rolling back the bad save);
         * log the collision to stdout;
         * raise ``RuntimeError`` with a loud, descriptive message.
    4. Return ``new_sha`` on success.

    Parameters
    ----------
    metal : str
        Metal symbol used to build the target filename.
    seed : int
        Random seed used to build the target filename.
    ckpt_dir : Path
        Directory holding D-MPNN pIC50 ckpts.
    remove_on_fail : bool, default True
        If True, remove the just-written file when a collision is
        detected (rollback). Set False in tests that want to inspect
        the file post-raise.

    Returns
    -------
    str
        The hex SHA256 of the just-written ckpt.

    Raises
    ------
    RuntimeError
        If the just-written file's SHA256 matches any sibling ckpt.
    FileNotFoundError
        If the target ckpt does not exist (caller forgot to save).
    """
    target = target_path(metal, seed, ckpt_dir)
    if not target.exists():
        raise FileNotFoundError(
            f"{GUARD_TAG} post-save guard called but {target.name} does "
            f"not exist — caller forgot to torch.save first"
        )

    new_sha = sha256_of_file(target)
    siblings = sorted(p for p in ckpt_dir.glob("dmpnn_pic50_*.pt") if p != target)

    # Always log SHA to stdout — required for audit trail.
    print(
        f"{GUARD_TAG} {target.name} sha256={new_sha} "
        f"siblings_scanned={len(siblings)}",
        file=sys.stdout,
        flush=True,
    )

    for sib in siblings:
        sib_sha = sha256_of_file(sib)
        if sib_sha == new_sha:
            if remove_on_fail:
                try:
                    target.unlink()
                    print(
                        f"{GUARD_TAG} removed {target.name} after SHA256 "
                        f"collision with {sib.name} (rollback)",
                        file=sys.stdout,
                        flush=True,
                    )
                except OSError as exc:
                    # Rollback failure is itself a loud problem.
                    print(
                        f"{GUARD_TAG} WARNING failed to remove {target.name} "
                        f"after collision: {exc!r}",
                        file=sys.stdout,
                        flush=True,
                    )
            raise RuntimeError(
                f"{GUARD_TAG} refusing to keep {target.name}: just-written "
                f"SHA256 {new_sha[:12]}... matches sibling {sib.name}. "
                f"This is the silent-copy bug signature — the new "
                f"training produced a byte-identical ckpt to an existing "
                f"one. Investigate seed/init reproducibility."
            )

    return new_sha


# ---------------------------------------------------------------------------
# Per-seed audit helper — used by dmpnn_sha256_audit.py extension
# ---------------------------------------------------------------------------
def find_duplicate_shas(ckpt_paths: Iterable[Path]) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return ``[(sha, (filename1, filename2, ...)), ...]`` for every duplicate.

    A duplicate is any SHA256 mapped to more than one filename. The
    result is sorted by SHA hex for stable test output.

    This is the same logic as
    ``molmetal.scripts.dmpnn_sha256_audit.audit`` but exposed as a
    pure helper that does not depend on the audit script's argparse
    surface.
    """
    sha_to_files: dict[str, List[str]] = {}
    for p in ckpt_paths:
        if not p.exists():
            continue
        # Only consider dmpnn_pic50_*.pt files (defensive against mixed dirs).
        if not (p.is_file() and p.name.startswith("dmpnn_pic50_") and p.suffix == ".pt"):
            continue
        sha = sha256_of_file(p)
        sha_to_files.setdefault(sha, []).append(p.name)

    dupes: List[Tuple[str, Tuple[str, ...]]] = []
    for sha, files in sha_to_files.items():
        if len(files) > 1:
            dupes.append((sha, tuple(sorted(files))))
    dupes.sort(key=lambda item: item[0])
    return dupes


def scan_all_ckpts(ckpt_dir: Path) -> dict:
    """Scan every ``dmpnn_pic50_*.pt`` under ``ckpt_dir``.

    Returns a dict with:

      * ``ckpt_dir`` — absolute path of the scanned dir;
      * ``n_ckpts`` — number of files scanned;
      * ``rows`` — list of ``{filename, metal, seed, sha256, size_bytes}``;
      * ``duplicates`` — list of ``(sha, (file1, file2, ...))`` for SHAs
        that appear on disk more than once;
      * ``status`` — ``"PASS"`` if no duplicates else ``"FAIL"``;
      * ``guard_would_pass`` — True iff the post-save guard would NOT
        raise on the current on-disk state. This is the Wf-120 P3
        "simulated guard" check used by the extension in
        ``dmpnn_sha256_audit.py``.
    """
    if not ckpt_dir.is_dir():
        return {
            "ckpt_dir": str(ckpt_dir),
            "n_ckpts": 0,
            "rows": [],
            "duplicates": [],
            "status": "FAIL",
            "guard_would_pass": False,
            "reason": f"ckpt_dir not found: {ckpt_dir}",
        }

    paths = sorted(ckpt_dir.glob("dmpnn_pic50_*.pt"))
    rows: List[dict] = []
    for p in paths:
        parsed = parse_dmpnn_filename(p)
        if parsed is None:
            rows.append(
                {
                    "filename": p.name,
                    "metal": None,
                    "seed": None,
                    "sha256": sha256_of_file(p),
                    "size_bytes": p.stat().st_size,
                    "parse_error": True,
                }
            )
            continue
        metal, seed = parsed
        rows.append(
            {
                "filename": p.name,
                "metal": metal,
                "seed": seed,
                "sha256": sha256_of_file(p),
                "size_bytes": p.stat().st_size,
                "parse_error": False,
            }
        )

    duplicates = find_duplicate_shas(paths)
    n_dupes = len(duplicates)
    return {
        "ckpt_dir": str(ckpt_dir),
        "n_ckpts": len(rows),
        "rows": rows,
        "duplicates": [
            {"sha256": sha, "filenames": list(files)} for sha, files in duplicates
        ],
        "n_duplicates": n_dupes,
        "status": "PASS" if n_dupes == 0 else "FAIL",
        # P3 extension: would the post-save guard raise on this state?
        "guard_would_pass": n_dupes == 0,
    }


__all__ = [
    "GUARD_TAG",
    "sha256_of_file",
    "parse_dmpnn_filename",
    "target_path",
    "pre_train_sha256_overwrite_guard",
    "post_save_sha256_overwrite_guard",
    "find_duplicate_shas",
    "scan_all_ckpts",
]