"""Greedy MaxMin diversity sampling over Morgan-ECFP4 fingerprints.

Used by ``molmetal/scripts/r10_cfg_real_crossdocked.py`` to build a
500-mol training pool from large metallodrug corpora
(PlatinAI_MBFinder + MetalCytoToxDB + tmQM) without re-training on
near-duplicates.  Same algorithm as the standard "pick the mol that is
farthest from everything already picked" greedy selection used in
MOSES / GuacaMol benchmarking (Polykovskiy 2020, Brown 2019).

Public surface
--------------
- :func:`morgan_fp`           — single-mol ECFP4 fingerprint (uint8 np.ndarray)
- :func:`morgan_fps`          — batch fingerprinting with RDKit-parseable filter
- :func:`greedy_maxmin_diversity` — n-mol selection by MaxMin distance
- :func:`load_platinai_smiles`     — load + RDKit-parse PlatinAI_MBFinder
- :func:`load_metal_cytotox_smiles`— load + RDKit-parse MetalCytoToxDB
- :func:`load_tmqm_smiles`         — load + RDKit-parse tmQM
- :func:`build_combined_pool`      — union of the 3 corpora
- :func:`cache_csv`                — write a metadata CSV (smiles, source) to disk
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    _HAS_RDKIT = True
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - RDKit is required, surface ImportError
    _HAS_RDKIT = False
    Chem = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """ECFP4-style Morgan fingerprint packed into a 1-D ``np.uint8`` bit array.

    Returns ``None`` if RDKit cannot parse the SMILES (including empty
    strings, whitespace, or unparseable tokens).  Uses
    :func:`rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect` so the output
    is bit-vector (sparse-by-default fixed length) and Jaccard-comparable.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for morgan_fp")
    if not smiles or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_fps(smiles_list: Iterable[str], radius: int = 2, n_bits: int = 2048):
    """Vectorised :func:`morgan_fp` over a list of SMILES.

    Returns a tuple ``(fps, parsed_indices)``:
    * ``fps``          : ``np.ndarray`` of shape ``(k, n_bits)`` where ``k``
      is the number of parseable SMILES in the input (drops bad ones).
    * ``parsed_indices``: original indices in ``smiles_list`` that survived
      parsing — used by the caller to look up metadata for the selected
      ``k`` mols.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for morgan_fps")
    parsed_indices: list = []
    rows: list = []
    for i, smi in enumerate(smiles_list):
        smi_str = (smi or "").strip()
        if not smi_str:
            continue
        fp = morgan_fp(smi_str, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        parsed_indices.append(i)
        rows.append(fp)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.uint8), []
    return np.stack(rows, axis=0), parsed_indices


# ---------------------------------------------------------------------------
# Greedy MaxMin selection
# ---------------------------------------------------------------------------
def _pairwise_tanimoto(fps: np.ndarray) -> np.ndarray:
    """Dense ``(n, n)`` Tanimoto distance matrix from a packed bit-vector matrix.

    Tanimoto distance = 1 - Tanimoto similarity.  Bit-vectors are
    ``np.uint8`` so we use bit-counting via ``popcount`` for speed.  For
    typical n=1500 candidate pools this is sub-second on a single core.
    """
    if fps.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # Treat each row as a 1-D byte string of {0, 1} bits and let numpy
    # unpack into uint64 words for fast popcount.
    n_bits = fps.shape[1]
    # round up to nearest multiple of 8 for byte packing
    n_bytes = (n_bits + 7) // 8
    packed = np.packbits(fps, axis=1)  # (n, n_bytes) uint8
    # Convert to uint64 words for vectorised popcount via np.unpackbits+sum
    # (np doesn't have popcount until 2.0; we use bitwise dot instead).
    a_int = packed.astype(np.uint64)
    # Treat bytes as set bits per byte — Tanimoto on packed bytes is a
    # bit-for-bit equivalent when both vectors are 0/1 bit fields
    # because the dot product on packed bytes equals the bit dot product.
    # sum(x & y) / (sum(x) + sum(y) - sum(x & y))
    # We compute via outer products in chunks to keep memory bounded.
    intersection = np.zeros((fps.shape[0], fps.shape[0]), dtype=np.float32)
    union_pop = fps.sum(axis=1).astype(np.float32)
    # Blocked matmul keeps memory O(B * n) for block size B.
    block = 256
    for i in range(0, fps.shape[0], block):
        end = min(i + block, fps.shape[0])
        # bitwise AND on uint8 = element-wise on bit positions
        # We use float32 dot product on the original bit matrix (fastest path
        # because numpy BLAS does it in one call).
        intersection[i:end] = fps[i:end].astype(np.float32) @ fps.T.astype(np.float32)
    sim = np.zeros_like(intersection)
    for i in range(intersection.shape[0]):
        denom = union_pop[i] + union_pop - intersection[i]
        sim[i] = np.where(denom > 0, intersection[i] / np.maximum(denom, 1e-9), 0.0)
    return 1.0 - sim  # distance


def greedy_maxmin_diversity(
    smiles_list: Sequence[str],
    n: int = 500,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 42,
):
    """Greedy MaxMin selection over Morgan-ECFP(radius) fingerprints.

    Algorithm
    ---------
    1. RDKit-parse each SMILES; drop unparseable ones.
    2. Compute ``(k, n_bits)`` fingerprint matrix.
    3. Compute the dense Tanimoto *distance* matrix.
    4. Seed with the parseable mol whose fingerprint is most "central"
       (smallest mean distance to all others — proxy for chemical centre).
    5. Iteratively pick the mol whose minimum distance to the already-
       selected set is largest (the MaxMin criterion).

    Parameters
    ----------
    smiles_list : Sequence[str]
        Pool of candidate SMILES.
    n : int
        Number of mols to select (``len(selected) == n`` when feasible).
    radius : int
        Morgan fingerprint radius (default 2 = ECFP4).
    n_bits : int
        Fingerprint length in bits (default 2048 — standard for ECFP4).
    seed : int
        RNG seed for the central-mol tie-break (default 42).

    Returns
    -------
    selected_indices : list[int]
        Indices into ``smiles_list`` (the original ordering, including
        unparseable ones are skipped) for the ``n`` selected mols.  The
        caller can re-derive the SMILES via ``smiles_list[i]``.
    selected_smiles : list[str]
        Convenience: the canonicalised SMILES strings for the selected
        mols (RDKit-canonicalised so duplicate SMILES collapse to one
        canonical form).
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for greedy_maxmin_diversity")
    if n <= 0:
        return [], []
    fps, parsed = morgan_fps(smiles_list, radius=radius, n_bits=n_bits)
    if len(parsed) == 0:
        return [], []
    if len(parsed) <= n:
        # All fit — return as-is (canonicalised).
        canonicals = _canonicalise([smiles_list[i] for i in parsed])
        return list(parsed), canonicals
    rng = np.random.default_rng(seed)
    dist = _pairwise_tanimoto(fps)
    # Step 4 — central seed.
    mean_dist = dist.mean(axis=1)
    seed_idx = int(rng.choice(np.flatnonzero(mean_dist == mean_dist.max())))
    selected = [seed_idx]
    # Track min-distance to selected set for each remaining mol.
    min_d = dist[seed_idx].copy()
    for _ in range(n - 1):
        # Mask out already-selected.
        for s in selected:
            min_d[s] = -np.inf
        next_idx = int(np.argmax(min_d))
        selected.append(next_idx)
        # Update min distances.
        np.minimum(min_d, dist[next_idx], out=min_d)
    canonicals = _canonicalise([smiles_list[i] for i in selected])
    return [parsed[i] for i in selected], canonicals


def _canonicalise(smiles_list: Iterable[str]) -> list:
    """RDKit canonicalise, drop entries that fail to parse."""
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out.append(Chem.MolToSmiles(mol))
    return out


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
PLATINAI_DEFAULT = Path("/mnt/storage/data/molmetal/PlatinAI_MBFinder_dataset.xlsx")
METAL_CYTOTOX_DEFAULT = Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv")
TMQM_DEFAULT_DIR = Path("/mnt/storage/data/molmetal/tmQM")


def load_platinai_smiles(path: Optional[Path] = None, max_rows: Optional[int] = None) -> list:
    """Load PlatinAI_MBFinder SMILES (Pt complexes).  Best-effort CSV fallback."""
    path = Path(path) if path else PLATINAI_DEFAULT
    if not path.is_file():
        return []
    smiles_col = None
    rows = []
    try:
        import pandas as pd
        df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)
        for cand in ("SMILES", "smiles", "SMILES_Ligand", "SMILES_Ligands"):
            if cand in df.columns:
                smiles_col = cand
                break
        if smiles_col is None:
            # Fallback — first string-typed column.
            for col in df.columns:
                if df[col].dtype == object:
                    smiles_col = col
                    break
        if smiles_col is None:
            return []
        for s in df[smiles_col].dropna().astype(str).tolist():
            rows.append(s)
            if max_rows is not None and len(rows) >= max_rows:
                break
    except Exception:
        # CSV fallback if openpyxl is not installed.
        try:
            import csv
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    for cand in ("SMILES", "smiles", "SMILES_Ligand", "SMILES_Ligands"):
                        if cand in r and r[cand]:
                            rows.append(r[cand])
                            break
                    if max_rows is not None and len(rows) >= max_rows:
                        break
        except Exception:
            return []
    return rows


def load_metal_cytotox_smiles(
    path: Optional[Path] = None, max_rows: Optional[int] = None
) -> list:
    """Load MetalCytoToxDB SMILES (Ru/Ir/Rh/Os/Re organometallics)."""
    path = Path(path) if path else METAL_CYTOTOX_DEFAULT
    if not path.is_file():
        return []
    rows = []
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        col = "SMILES_Ligands" if "SMILES_Ligands" in df.columns else df.columns[0]
        for s in df[col].dropna().astype(str).tolist():
            rows.append(s)
            if max_rows is not None and len(rows) >= max_rows:
                break
    except Exception:
        try:
            import csv
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    val = r.get("SMILES_Ligands") or r.get("SMILES") or next(iter(r.values()), None)
                    if val:
                        rows.append(val)
                    if max_rows is not None and len(rows) >= max_rows:
                        break
        except Exception:
            return []
    return rows


def load_tmqm_smiles(
    path: Optional[Path] = None, max_rows: Optional[int] = None
) -> list:
    """Load tmQM SMILES (transition-metal complexes).

    Tries the project's own loader first (:mod:`molmetal.data.tmqm`); falls
    back to a direct read of ``tmQM_y.csv`` which is ``';'``-separated
    with the SMILES column usually in the trailing field.
    """
    rows: list = []
    try:
        from molmetal.data.tmqm import tmqm_dataset

        try:
            ds = tmqm_dataset()
        except Exception:
            ds = None
        if ds is not None:
            for item in ds:
                smi = item.get("smiles") if isinstance(item, dict) else None
                if smi:
                    rows.append(smi)
                    if max_rows is not None and len(rows) >= max_rows:
                        return rows
            if rows:
                return rows
    except Exception:
        pass
    # Fallback — read tmQM_y.csv directly.
    tmqm_dir = Path(path) if path else TMQM_DEFAULT_DIR
    candidates = list(tmqm_dir.glob("tmQM_y*"))
    if not candidates:
        candidates = list(tmqm_dir.glob("*.csv"))
    for csv_path in candidates:
        try:
            import csv
            with csv_path.open(newline="") as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader, None)
                smi_idx = None
                if header:
                    for i, name in enumerate(header):
                        if str(name).strip().lower() == "smiles":
                            smi_idx = i
                            break
                    if smi_idx is None:
                        smi_idx = len(header) - 1
                else:
                    smi_idx = -1
                for r in reader:
                    if not r:
                        continue
                    val = r[smi_idx] if 0 <= smi_idx < len(r) else r[-1]
                    if val:
                        rows.append(val.strip())
                    if max_rows is not None and len(rows) >= max_rows:
                        return rows
            if rows:
                return rows
        except Exception:
            continue
    return rows


def build_combined_pool(
    platinai_path: Optional[Path] = None,
    cytotox_path: Optional[Path] = None,
    tmqm_dir: Optional[Path] = None,
    platinai_n: int = 500,
    cytotox_n: int = 500,
    tmqm_n: int = 500,
    n: int = 500,
    radius: int = 2,
    seed: int = 42,
):
    """Build a diversity-sampled pool of ``n`` mols from the 3 corpora.

    Strategy
    --------
    1. Pull up to ``platinai_n`` from PlatinAI_MBFinder (Pt-only).
    2. Pull up to ``cytotox_n`` from MetalCytoToxDB (multi-metal).
    3. Pull up to ``tmqm_n`` from tmQM (transition-metal).
    4. Concat into a candidate pool, then greedy-MaxMin-sample to ``n``.
    """
    pool: list = []
    sources: list = []
    for loader, label, n_src in (
        (load_platinai_smiles, "platinai", platinai_n),
        (load_metal_cytotox_smiles, "metal_cytotox", cytotox_n),
        (load_tmqm_smiles, "tmqm", tmqm_n),
    ):
        kwargs = {"max_rows": n_src}
        if label == "platinai":
            kwargs["path"] = platinai_path
        elif label == "metal_cytotox":
            kwargs["path"] = cytotox_path
        else:
            kwargs["path"] = tmqm_dir
        rows = loader(**kwargs)
        pool.extend(rows)
        sources.extend([label] * len(rows))
    selected_orig_idx, selected_smiles = greedy_maxmin_diversity(
        pool, n=n, radius=radius, seed=seed
    )
    selected_sources = [sources[i] for i in selected_orig_idx]
    return selected_smiles, selected_sources


def cache_csv(smiles: Sequence[str], sources: Sequence[str], path: Path) -> Path:
    """Persist ``(smiles, source)`` as a 2-column CSV at ``path``.

    Creates parent directories as needed.  Overwrites any existing file
    at ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "source"])
        for s, src in zip(smiles, sources):
            w.writerow([s, src])
    return path


# ---------------------------------------------------------------------------
# WF-T30 P4.2 — patent / known-Pt-drug similarity axis
#
# Honest framing (per TODO/pending/30_pitfall_reinforce_plan.md Rank-2):
# Tanimoto similarity to known drugs is a CHEMICAL-NOVELTY proxy, not a
# substitute for a patent search.  A generated molecule with Tanimoto
# 0.4 to cisplatin is structurally novel in fingerprint space, but that
# says nothing about composition-of-matter claims, formulation patents,
# or method-of-use IP.  The output of these functions is intended for
# novelty triage only; patent freedom-to-operate (FTO) requires a proper
# prior-art search (SureChEMBL / PatentScope / etc.).
# ---------------------------------------------------------------------------
DEFAULT_KNOWN_PT_DRUGS_CSV = Path(
    "/home/hugo/codes/try_triton_on_rocm/molmetal/data/known_pt_drugs.csv"
)


def load_known_pt_drugs(csv_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load the known-Pt-drug reference table.

    Returns a list of dicts with keys ``name``, ``smiles``,
    ``fda_approved_year``, ``primary_indication``.  Drops rows whose
    SMILES cannot be RDKit-parsed.  Returns an empty list when the
    CSV is missing (graceful fallback — caller should treat empty
    as "no patent axis" rather than raising).
    """
    path = Path(csv_path) if csv_path else DEFAULT_KNOWN_PT_DRUGS_CSV
    if not path.is_file():
        return []
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            smi = (r.get("smiles") or "").strip()
            if not smi:
                continue
            rows.append({
                "name": (r.get("name") or "").strip(),
                "smiles": smi,
                "fda_approved_year": (r.get("fda_approved_year") or "").strip(),
                "primary_indication": (r.get("primary_indication") or "").strip(),
            })
    return rows


def _tanimoto_sim_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary bit-vectors (np.uint8).

    Returns ``1.0`` for two empty bit-vectors (the standard convention
    used in :mod:`molmetal.validation.training_novelty`).
    """
    a = np.asarray(a, dtype=np.uint8).reshape(-1)
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    if a.size != b.size:
        raise ValueError(
            f"fingerprint length mismatch: a={a.size} vs b={b.size}"
        )
    inter = int(np.bitwise_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    if union <= 0:
        return 1.0
    return float(inter) / float(union)


def metric_max_sim_known_pt_drugs(
    smiles: str,
    drug_db_path: Optional[Path] = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[str, Any]:
    """Return the max Tanimoto similarity of ``smiles`` to the known Pt drug DB.

    Honest framing: this is a CHEMICAL-NOVELTY proxy, not a substitute
    for a patent search.  A generated molecule with Tanimoto 0.4 to
    cisplatin is structurally novel in fingerprint space, but that
    says nothing about composition-of-matter claims, formulation
    patents, or method-of-use IP.  See
    ``molmetal/reports/wf_t30_patent_axis/final.md``.

    Parameters
    ----------
    smiles : str
        Generated-mol SMILES (RDKit-parseable).
    drug_db_path : Path | None
        Override path to the known-drug CSV.  Defaults to
        ``molmetal/data/known_pt_drugs.csv``.

    Returns
    -------
    dict with keys:
        * ``max_sim``           — float in [0, 1]; 0.0 on parse failure
                                   or empty DB.
        * ``closest_drug``      — str; name of the closest known drug,
                                   or ``""`` if no DB / parse failure.
        * ``any_drug_above_0.7``— bool; True iff at least one DB entry
                                   has Tanimoto > 0.7 (HIGH novelty
                                   risk).
        * ``any_drug_above_0.4``— bool; True iff at least one DB entry
                                   has Tanimoto > 0.4 (MODERATE
                                   novelty risk; conservative default
                                   threshold).
        * ``per_drug``          — list of ``(name, sim)`` for audit
                                   (caller can drop in JSON).
    """
    if not _HAS_RDKIT:
        return dict(max_sim=0.0, closest_drug="",
                    any_drug_above_0_7=False, any_drug_above_0_4=False,
                    per_drug=[])
    drugs = load_known_pt_drugs(drug_db_path)
    if not drugs:
        return dict(max_sim=0.0, closest_drug="",
                    any_drug_above_0_7=False, any_drug_above_0_4=False,
                    per_drug=[])
    fp = morgan_fp(smiles, radius=radius, n_bits=n_bits)
    if fp is None:
        # Parse failure: 0.0 max sim, but the caller may still want
        # the audit trail of the DB being scanned.  We return the
        # closest_drug="" so downstream risk flags default to False.
        return dict(max_sim=0.0, closest_drug="",
                    any_drug_above_0_7=False, any_drug_above_0_4=False,
                    per_drug=[])
    per_drug: List[Tuple[str, float]] = []
    max_sim = 0.0
    closest = ""
    for d in drugs:
        fp_d = morgan_fp(d["smiles"], radius=radius, n_bits=n_bits)
        if fp_d is None:
            continue
        s = _tanimoto_sim_uint8(fp, fp_d)
        per_drug.append((d["name"], s))
        if s > max_sim:
            max_sim = s
            closest = d["name"]
    return dict(
        max_sim=float(max_sim),
        closest_drug=str(closest),
        any_drug_above_0_7=bool(max_sim > 0.7),
        any_drug_above_0_4=bool(max_sim > 0.4),
        per_drug=per_drug,
    )


def scaffold_in_known_pt_drugs(
    smiles: str,
    drug_db_path: Optional[Path] = None,
) -> bool:
    """Return ``True`` iff ``smiles`` shares a Bemis-Murcko scaffold with
    any known Pt drug.

    Uses :func:`molmetal.validation.training_novelty.scaffold_smiles`
    (line 24-27) for canonical scaffold SMILES and a simple
    set-membership comparison.  Returns ``False`` on parse failure
    or empty DB.

    Honest framing: this is a *topological* scaffold match — it does
    not check coordination geometry, oxidation state, or other
    3D-aware constraints.  Patent FTO still requires a chemistry-
    aware prior-art search.

    Notes on acyclic Pt complexes
    -----------------------------
    All 7 known Pt drugs in this DB are acyclic Pt complexes — their
    Bemis-Murcko scaffold is the empty string.  An empty-scaffold
    query is therefore ambiguous: it could be cisplatin, an alkane,
    water, etc.  We resolve the ambiguity by requiring the *query*
    SMILES to contain Pt before declaring a scaffold match against an
    empty-scaffold drug entry.  This makes the test
    "scaffold_in_known_pt_drugs(cisplatin) == True" and
    "scaffold_in_known_pt_drugs(hexane) == False" both pass — and
    "scaffold_in_known_pt_drugs(some_other_Pt_complex) == True"
    when the other Pt complex is also acyclic.  Cyclic drugs
    (satraplatin's tetramethyl-cyclohexadiene ring) match via the
    standard non-empty scaffold path.
    """
    if not _HAS_RDKIT:
        return False
    drugs = load_known_pt_drugs(drug_db_path)
    if not drugs:
        return False
    try:
        # Local import to avoid the heavy torch / GPU dependency in
        # callers that only want the patent axis.
        from molmetal.validation.training_novelty import scaffold_smiles
    except Exception:
        return False
    try:
        scaf_q = scaffold_smiles(smiles)
    except Exception:
        return False
    # Pre-compute Pt presence for the acyclic-match disambiguation.
    try:
        q_mol = Chem.MolFromSmiles(smiles)
        q_has_pt = q_mol is not None and any(
            a.GetAtomicNum() == 78 for a in q_mol.GetAtoms()
        )
    except Exception:
        q_has_pt = False
    for d in drugs:
        try:
            scaf_d = scaffold_smiles(d["smiles"])
        except Exception:
            continue
        if scaf_d and scaf_d == scaf_q:
            return True
        # Acyclic-drug path: only flag a match when the query is
        # also an acyclic Pt complex (cisplatin-like) — i.e. when
        # both scaffolds are empty AND the query contains Pt.
        if (not scaf_d) and (not scaf_q) and q_has_pt:
            return True
    return False
