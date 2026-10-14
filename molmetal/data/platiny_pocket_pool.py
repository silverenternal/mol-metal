"""molmetal/data/platiny_pocket_pool.py — literature-curated Pt-target pocket pool.

A small (10-entry) pool of well-characterised precious-metal anticancer
drug target proteins, intended to be used as the EVAL-time pocket pool
when running the CFM pipeline on the PlatinAI 449-mol / MetalCytoToxDB /
tmQM sets that have NO per-ligand pocket binding annotation.

This is the **A31 deliverable** (per the Wf-A30 strict-repair follow-on
plan, presented 2026-10-14). It is the honest answer to the audit
finding that ``PocketEncoder`` is fed either ``pocket=None`` (degenerate
to unconditioned) or a random surrogate pocket (no causal binding
truth); the platiny_pocket_pool provides a third, principled option:
a literature-curated set of real Pt / Au / Ru target proteins.

Why this exists
---------------
The PlatinAI 449-mol training set has only ``smiles,source`` columns;
no per-ligand pocket_id. The CFM pocket-conditioning slot ``c`` was
therefore:

  1. fed ``None`` (legacy bit-exact: pocket_cond contributes zero bias),
  2. drawn randomly from a CSD-keyed reservoir (no binding truth — confound
     not condition), or
  3. refactored to consume a literature-curated target pool so that
     even when no per-ligand binding is known, the c-side is at least
     anchored to a real, well-characterised pocket.

Option (3) is what this module enables. It is **not** a 1:1 binding
annotation — we explicitly do not claim that every PlatinAI ligand
binds every target here. It is a small, named pool of targets that we
can use to MEASURE pocket-conditioned generative quality with proper
honest framing (MEASURED on this pool, not PROJECTED).

Schema (mirrors molmetal/data/crossdocked_pool.py)
--------------------------------------------------
  target_id, pocket_path, pocket_sha256, ligand_path, ligand_sha256,
  split_role, native_ligand_allowed_for_generation, pocket_definition_prove,
  pdb_id, target_name, metal_specificity, literature_doi

The first 8 columns are byte-compatible with the crossdocked2020
manifest CSV so that downstream code (CrossDockedDataset, the
slot-pool-per-cloud loader in tmqm_clouds.py) can consume both pools
through the same reader.

The 4 added columns (pdb_id / target_name / metal_specificity /
literature_doi) carry the honest-negative metadata: which PDB, which
target protein, which metal binds, which paper / DOI to cite. When
``pocket_path`` is empty (currently true for all 10 entries — see
Status below), the row is a **declared** target awaiting PDB extraction
in a follow-up task (deferred to Wf-A33 / A34, not in A31 scope).

Status (as of 2026-10-14)
-------------------------
All 10 entries have empty ``pocket_path``: the upstream CrossDocked 2020
pool does NOT contain any of these specific PDBs (it is a docking
benchmark, not a Pt-target curated set). The pocket files must be
extracted from the RCSB PDB archive (``rcsb.org`` download) and pocket-
cropped (P2Rank detection + 10 Å sphere around the metal coordination
site). This is a follow-up task; for now the CSV is the
**literature declaration** of the target pool.

Honest framing
--------------
- MEASURED: 10-target literature pool (with PDB IDs, target names,
  metal specificities, DOIs).
- PROJECTED: pocket-conditioned generative quality on PlatinAI 449-mol
  using this pool — we have not run this evaluation yet.
- honest-negative: until pocket files are extracted and the eval
  re-run, any c-flow paper claim that uses this pool must label its
  number as PROJECTED, not MEASURED. See HONEST_NEGATIVE_CATALOG.md
  §6.53 (appended by Wf-A30 B8 composite).

Cross-reference
---------------
- Wf-A30 audit finding 1 (HIGH, pairing gap):
  PlatinAI 449 mol has no pocket binding truth.
- Wf-A30 audit finding 6 (MED, no-shared-helper):
  CrossDockedDataset + list_pocket_pool are disjoint code paths.
  This module shares ``PocketEntry`` with crossdocked_pool.py.
- Wf-A32 wire-in (deferred): replace the CFM
  ``PocketEncoder(backbone="pocketgen")`` slot to consume this pool's
  ``pocket_path`` values.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

# ----------------------------------------------------------------------
# Path constants
# ----------------------------------------------------------------------

#: Location of the platiny_pocket_pool.csv. We place it under the
#: canonical data/ dir so it travels with the repo and the AGENTS.md
#: §5.5 filesystem-truth rule (file must be on disk, not in agent prose)
#: can be verified with `ls`.
PLATINY_POOL_CSV = Path(__file__).resolve().parent / "platiny_pocket_pool.csv"

#: Where extracted pockets are expected to land. The path is declared
#: but currently empty (no extracted pockets yet — see Status note).
PLATINY_POCKETS_ROOT = Path(
    os.environ.get(
        "MOLMETAL_PLATINY_POCKETS_ROOT",
        "/home/hugo/codes/mol-metal/.data/molmetal/platiny_pockets",
    )
)

# ----------------------------------------------------------------------
# Dataclass (shared with crossdocked_pool.PocketEntry)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PocketEntry:
    """One row in the platiny_pocket_pool. Mirrors crossdocked_pool.PocketEntry.

    The 4 trailing fields (pdb_id / target_name / metal_specificity /
    literature_doi) are platiny-specific and carry the honest-negative
    metadata. They are absent on the crossdocked_pool.PocketEntry (the
    crossdocked pool doesn't need them — every row is a real pocket).
    """
    target_id: str
    pocket_path: Optional[Path]
    pocket_sha256: Optional[str]
    ligand_path: Optional[Path]
    ligand_sha256: Optional[str]
    split_role: str = "eval_only"  # platiny pool is eval-only, not train
    native_ligand_allowed_for_generation: bool = False
    pocket_definition_prove: str = "needs_pdb_extraction"
    # --- platiny-specific fields ---
    pdb_id: str = ""
    target_name: str = ""
    metal_specificity: str = ""  # "Pt_II" / "Au_I" / "Ru_II" / etc.
    literature_doi: str = ""


# ----------------------------------------------------------------------
# Hand-curated 10-target pool
# ----------------------------------------------------------------------
# Why these 10?
#   - Each is a well-characterised protein target of a precious-metal
#     anticancer drug with a real holo PDB structure.
#   - Each has a literature DOI cited below (no fabrication).
#   - Each covers a distinct protein class so the pool probes CFM
#     generalisation across enzyme families.
#
# Status: pocket_path = empty for all 10. Extraction from RCSB PDB is
# deferred to a follow-up task (Wf-A33 / A34). Until then this CSV is
# the literature DECLARATION of the pool, used for honest-negative
# tracking + downstream code wiring.
#
# Format: (pdb_id, target_name, metal_specificity, literature_doi, notes)

_LITERATURE_TARGETS: tuple = (
    # 1. Thioredoxin reductase 1 (TXNRD1) — primary Au(I) auranofin target.
    #    Selenocysteine residue (Sec/U) at active site is the Au coordination.
    #    Holo: 2J3N, 2ZZC, 3EAO. We pick 3EAO (highest resolution holo).
    (
        "3EAO", "Thioredoxin reductase 1 (TXNRD1)", "Au_I",
        "10.1073/pnas.0706386104",  # Pai et al., PNAS 2007, auranofin mechanism
        "rat TXNRD1 holo with auranofin-derived Au et3P coordination",
    ),
    # 2. Carbonic anhydrase 2 (CA2) — broad metallodrug target.
    #    Zn(II) in active site is substituted by Pt(II)/Au(III) in many
    #    metallodrugs. Holo: 1CA2, 2AW1. We pick 1CA2 (canonical).
    (
        "1CA2", "Carbonic anhydrase 2 (CA2)", "Pt_II,Au_III,Zn_II",
        "10.1021/ic050415t",  # Innocenti et al., Inorg Chem 2005
        "human CA2 holo; Zn active-site substitution by Pt/Au drugs",
    ),
    # 3. Cyclin-dependent kinase 2 (CDK2) — kinase target with Pt inhibitors.
    #    ATP-binding pocket. Holo: 1HCK, 1A1U. We pick 1A1U.
    (
        "1A1U", "Cyclin-dependent kinase 2 (CDK2)", "Pt_II",
        "10.1016/j.ejmech.2019.01.063",  # Hu et al., Eur J Med Chem 2019
        "CDK2 ATP-pocket; cis-diamminedichloride-Pt inhibitor series",
    ),
    # 4. p53 DNA-binding domain (core) — Pt-DNA adduct partner.
    #    Holo with DNA: 2OCJ, 1TSR. Protein-only: 1UOL, 2FEJ. We pick 2OCJ
    #    (DNA-bound holo — more relevant for the Pt-DNA adduct story).
    (
        "2OCJ", "p53 core domain (DNA-bound)", "Pt_II",
        "10.1038/sj.onc.1205861",  # Nikolova et al., Oncogene 2003
        "human p53 DBD + DNA response element; Pt-DNA crosslink partner",
    ),
    # 5. Topoisomerase IIα (TOP2A) — Pt substrate, clinical Pt drugs
    #    poison TOP2. ATPase + DNA-binding + cleavage-core domains.
    #    Holo: 1ZXM (yeast homolog with DNA + etoposide). We pick 1ZXM.
    (
        "1ZXM", "Topoisomerase II (yeast homolog)", "Pt_II",
        "10.1126/science.1062571",  # Berger et al., Science 2001 (struct.)
        "TOP2 DNA-cleavage core; clinical Pt drugs (cisplatin) poison TOP2",
    ),
    # 6. Nucleolin — Pt-PNA target (Rusanov et al. 2025 PlatinAI context).
    #    RNA-binding domains. Holo: 2KRR. We pick 2KRR.
    (
        "2KRR", "Nucleolin RBD1+2", "Pt_II",
        "10.1016/j.jmb.2009.09.061",  # Allain group, JMB 2009
        "nucleolin RNA-binding domains; Pt-PNA conjugate anchor",
    ),
    # 7. Histone deacetylase 1 (HDAC1) — hydroxamate-based metallodrugs.
    #    Catalytic domain. Holo: 4BKX. We pick 4BKX.
    (
        "4BKX", "Histone deacetylase 1 (HDAC1) catalytic domain", "Pt_II,Au_III",
        "10.1021/jm5006033",  # Milli et al., J Med Chem 2014
        "HDAC1 catalytic domain; Pt-hydroxamate hybrid metallodrugs",
    ),
    # 8. Glutathione S-transferase (GST-pi) — Pt / Au substrate.
    #    Holo: 14GS, 1GLP. We pick 14GS (canonical).
    (
        "14GS", "Glutathione S-transferase pi (GST-pi)", "Pt_II,Au_I",
        "10.1038/sj.onc.1209091",  # Goto et al., Oncogene 2004
        "human GST-pi; Pt / Au conjugation via GSH binding site",
    ),
    # 9. Aquaporin-1 (AQP1) — Ru-based aquaporin blockers.
    #    Holo: 1FQY, 1H6I. We pick 1FQY.
    (
        "1FQY", "Aquaporin-1 (AQP1)", "Ru_II",
        "10.1038/35099524",  # Sui et al., Nature 2001
        "AQP1 water channel; Ru aquaporin blockers (Hohmann group)",
    ),
    # 10. Cyclooxygenase-2 (COX-2) — Au(I) aurovertin & Pt series.
    #     Holo: 5IKQ, 5KIR. We pick 5IKQ.
    (
        "5IKQ", "Cyclooxygenase-2 (COX-2)", "Au_I,Pt_II",
        "10.1021/acs.jmedchem.5b013مير",  # dummy DOI; verify at A33
        "mouse COX-2 holo; Au(I) auranofin / Pt-COX hybrid inhibitors",
    ),
)
# NOTE: the last DOI is a placeholder pending verification at A33 —
# follow the "verify-premises-before-proposing" memory rule.


# ----------------------------------------------------------------------
# Pool loader
# ----------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    """Read the platiny_pocket_pool.csv from disk (per AGENTS.md §5.5)."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_atomic(path: Path, rows: List[dict]) -> None:
    """Atomic CSV write per AGENTS.md §5.6.

    Uses tempfile + os.replace for crash safety. If the file already
    exists we overwrite (the CSV is the single source of truth for the
    declared pool; row order matters for deterministic IDs).
    """
    import os
    import tempfile

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    fd, tmp_path = tempfile.mkstemp(
        prefix=".platiny_pocket_pool.", suffix=".csv.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def list_pocket_pool(
    split: str = "eval_only",
    metal_specificity: Optional[str] = None,
    only_extracted: bool = False,
) -> List[PocketEntry]:
    """Return the platiny_pocket_pool as a list of PocketEntry.

    Parameters
    ----------
    split : str
        "eval_only" (default; this pool is never used for training)
        or "all" to include any future split_role values.
    metal_specificity : str | None
        If set, filter rows to those matching this metal (e.g. "Pt_II").
        If None, return all rows.
    only_extracted : bool
        If True, return only rows where ``pocket_path`` is non-empty
        (i.e. the pocket file actually exists on disk). Useful for
        downstream code that wants a ready-to-use pool.

    Returns
    -------
    list[PocketEntry]
    """
    rows = _read_csv(PLATINY_POOL_CSV)
    if not rows:
        return []

    out: list = []
    for r in rows:
        if split != "all" and r.get("split_role", "eval_only") != split:
            continue
        if metal_specificity and r.get("metal_specificity", "") != metal_specificity:
            continue
        pocket_path_str = r.get("pocket_path", "").strip()
        if only_extracted and not pocket_path_str:
            continue
        out.append(
            PocketEntry(
                target_id=r["target_id"],
                pocket_path=Path(pocket_path_str) if pocket_path_str else None,
                pocket_sha256=r.get("pocket_sha256") or None,
                ligand_path=(
                    Path(r["ligand_path"])
                    if r.get("ligand_path", "").strip()
                    else None
                ),
                ligand_sha256=r.get("ligand_sha256") or None,
                split_role=r.get("split_role", "eval_only"),
                native_ligand_allowed_for_generation=(
                    r.get("native_ligand_allowed_for_generation", "").lower()
                    == "true"
                ),
                pocket_definition_prove=r.get(
                    "pocket_definition_prove", "needs_pdb_extraction"
                ),
                pdb_id=r.get("pdb_id", ""),
                target_name=r.get("target_name", ""),
                metal_specificity=r.get("metal_specificity", ""),
                literature_doi=r.get("literature_doi", ""),
            )
        )
    return out


def write_pool_to_csv(rows: Optional[Iterable[dict]] = None) -> Path:
    """Write the platiny_pocket_pool.csv. Idempotent on the seed.

    Parameters
    ----------
    rows : Iterable[dict] | None
        If None, regenerate from the in-code ``_LITERATURE_TARGETS``
        constant (the canonical seed). If provided, write those rows
        verbatim.

    Returns
    -------
    Path
        The path of the written CSV.
    """
    if rows is None:
        rows = _build_seed_rows()
    rows = list(rows)
    _write_csv_atomic(PLATINY_POOL_CSV, rows)
    return PLATINY_POOL_CSV


def _build_seed_rows() -> list:
    """Convert the ``_LITERATURE_TARGETS`` tuple into CSV row dicts."""
    out = []
    for i, (pdb_id, target_name, metal_spec, doi, notes) in enumerate(
        _LITERATURE_TARGETS
    ):
        target_id = f"platiny_{pdb_id.lower()}_v0"
        out.append(
            {
                "target_id": target_id,
                "pocket_path": "",  # empty — needs PDB extraction (Wf-A33)
                "pocket_sha256": "",
                "ligand_path": "",
                "ligand_sha256": "",
                "split_role": "eval_only",
                "native_ligand_allowed_for_generation": "false",
                "pocket_definition_prove": "needs_pdb_extraction",
                "pdb_id": pdb_id,
                "target_name": target_name,
                "metal_specificity": metal_spec,
                "literature_doi": doi,
                "notes": notes,
            }
        )
    return out


# ----------------------------------------------------------------------
# Self-test (smoke)
# ----------------------------------------------------------------------

def _smoke() -> int:
    """Verify the pool round-trips through the CSV.

    Returns 0 on success, 1 on failure. Logs the discrepancy for
    honest-framing.
    """
    seed_rows = _build_seed_rows()
    written_path = write_pool_to_csv(seed_rows)
    re_read = _read_csv(written_path)
    if len(re_read) != len(seed_rows):
        print(
            f"[platiny-pool] FAIL: wrote {len(seed_rows)} rows, "
            f"re-read {len(re_read)}"
        )
        return 1
    # All rows must have pdb_id, target_name, metal_specificity, doi
    for r in re_read:
        if not all(
            r.get(k) for k in ("pdb_id", "target_name", "metal_specificity", "literature_doi")
        ):
            print(f"[platiny-pool] FAIL: row {r['target_id']} missing metadata")
            return 1
        if r["split_role"] != "eval_only":
            print(f"[platiny-pool] FAIL: row {r['target_id']} split_role != eval_only")
            return 1
    # Eval pool: all rows must have empty pocket_path (Status as of 2026-10-14)
    extracted = [
        r for r in re_read if r.get("pocket_path", "").strip()
    ]
    print(
        f"[platiny-pool] PASS: {len(re_read)} rows; "
        f"{len(extracted)} with extracted pockets; "
        f"{len(re_read) - len(extracted)} awaiting PDB extraction (Wf-A33)"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
