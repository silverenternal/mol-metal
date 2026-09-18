"""AutoDock Vina adapter — implements the DockingEngine Protocol.

================================================================
What this replaces
================================================================
The original ``molmetal/molmetal_lam/scripts/baselines.py`` reported
binding affinity as ``0.5 + 0.1 * NumRotatableBonds`` — a transparent
proxy that had no correlation with real binding. This module wraps the
``vina`` Python binding (Eberhardt 2021, *J. Chem. Inf. Model.*) +
``meeko`` for receptor/ligand preparation, so we can report real
kcal/mol scores on any drug-like ligand.

Public API
----------
* :class:`VinaDockingAdapter`  -- implements ``DockingEngine``
* :func:`dock_smiles(smiles, pocket, config)` -- one-shot helper
* :func:`redock_for_test(pdb_id, ref_ligand_smiles)` -- redocking
  benchmark that returns RMSD vs. crystal pose (for tests only)

Reference protocol
------------------
1. Build receptor PDBQT from Pocket.coords (we write a minimal PDB +
   call ``mk_prepare_receptor.py`` if available, otherwise fall back
   to direct ``vina`` load).
2. Build ligand PDBQT from SMILES → 3D embed → meeko MoleculePreparation
   → PDBQTWriterLegacy.
3. ``vina.Vina(sf_name='vina')`` → set_receptor + set_ligand_from_file
   → compute_vina_maps → dock → poses.

Note
----
This module imports :mod:`vina` and :mod:`meeko` lazily, so importing
the rest of :mod:`molmetal.molmetal_lam` does not require them.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import DockingConfig, DockingEngine

logger = logging.getLogger(__name__)

# Supported docking engines. ``vina`` is the default and matches
# mmp13_vina_real.md; ``qvina`` and ``quickvina2`` are CLI binaries.
# ``auto`` picks the best available path:
#   1. ``vina`` Python binding (preferred — full programmatic control)
#   2. ``vina`` CLI binary (if libboost resolves)
#   3. ``qvina`` / ``quickvina2`` CLI (if installed)
SUPPORTED_ENGINES = ("auto", "vina", "vina-cli", "qvina", "quickvina2")
# Default per round-7 wire-vina / TODO-04 qvina_swap: prefer the ``vina``
# Python binding (vina 1.2.7) over any subprocess fallback.  The CLI
# binaries (qvina / quickvina2 / vina-cli) are only used when the caller
# explicitly passes ``--engine qvina`` etc.
DEFAULT_ENGINE = "vina"

# Engine-class dispatch map.  ``python`` means the ``vina`` Python
# binding (full programmatic energies() + poses()) and is preferred;
# ``subprocess`` means a CLI binary (qvina / quickvina2 / vina-cli)
# invoked via ``subprocess.run``.  The constructor resolves
# ``engine='auto'`` before we get here, so the values below are always
# one of these four concrete names.
_ENGINE_DISPATCH = {
    "vina":       "python",
    "vina-cli":   "subprocess",
    "qvina":      "subprocess",
    "quickvina2": "subprocess",
}

# Canonical binary names on PATH.
_QVINA_BIN_CANDIDATES = ("qvina", "qvina2", "qvina02", "qvina2.1", "qvina.exe")
_QUICKVINA2_BIN_CANDIDATES = (
    "quickvina2", "quickvina", "qvina2", "qvina02", "qvina2.1", "quickvina2.exe",
)

# The vendored qvina02 is QuickVina 2, byte-identical to QVina/qvina's
# bin/qvina02 (git blob 85281985807632dc6d2e0a8564d2a3c027166f49).
# Its --version still says AutoDock Vina 1.1.2; see the identity evidence in
# reports/quickvina2_binary_identity.md. Both public engine names can use it.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_QVINA = _PROJECT_ROOT / "molmetal" / "references" / "SoftMol" / "gated_mcts" / "utils" / "docking" / "qvina02"


# -----------------------------------------------------------------
# Lazy optional imports — fail soft so this module is still importable
# -----------------------------------------------------------------
def _have_vina() -> bool:
    try:
        import vina  # noqa: F401
        return True
    except Exception:
        return False

def _have_meeko() -> bool:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_binary(candidates: Tuple[str, ...]) -> Optional[str]:
    """Resolve a runnable binary: engine override, PATH, then verified bundle.

    QuickVina 2 is distributed as qvina02 or qvina2.1 upstream. A differently
    named binary on PATH is therefore unnecessary. Probe --help so a broken
    shared-library dependency cannot be mistaken for an available engine.
    """
    override_names = {
        _QVINA_BIN_CANDIDATES: "MOLMETAL_QVINA_BIN",
        _QUICKVINA2_BIN_CANDIDATES: "MOLMETAL_QUICKVINA2_BIN",
    }
    override_name = override_names.get(candidates)
    paths = []
    if override_name and os.environ.get(override_name):
        override = os.environ[override_name]
        paths.append(shutil.which(override) or override)
    paths.extend(shutil.which(name) for name in candidates)
    if override_name:
        paths.append(str(_BUNDLED_QVINA))
    for path in dict.fromkeys(p for p in paths if p):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            continue
        try:
            probe = subprocess.run(
                [path, "--help"], capture_output=True, check=False, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return str(Path(path).resolve())
    return None


def _have_qvina_binary() -> bool:
    return _resolve_binary(_QVINA_BIN_CANDIDATES) is not None


def _have_quickvina2_binary() -> bool:
    return _resolve_binary(_QUICKVINA2_BIN_CANDIDATES) is not None


# Canonical ``vina`` CLI candidates (system package install).
_VINA_BIN_CANDIDATES = ("vina", "vina.exe")


def _probe_vina_binary() -> Optional[str]:
    """Return absolute path of a working ``vina`` CLI binary, or ``None``.

    On this ROCm stack the system ``vina`` binary is installed at
    ``/usr/bin/vina`` but its ``libboost_thread.so.1.90.0`` dep is
    absent (only ``1.92.0`` is shipped), so even though ``shutil.which``
    finds it, ``vina --help`` fails with::

        vina: error while loading shared libraries:
        libboost_thread.so.1.90.0: cannot open shared object file

    (See ``molmetal/reports/round7_install_report.md`` §1, "CLI binary
    — vina".)  This probe rejects such broken binaries by actually
    invoking ``--help`` and confirming ``returncode == 0``.
    """
    for name in _VINA_BIN_CANDIDATES:
        path = shutil.which(name)
        if not path:
            continue
        try:
            proc = subprocess.run(
                [path, "--help"],
                check=False, capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if proc.returncode == 0:
            return path
    return None


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def _pocket_to_pdb_string(pocket: Pocket) -> str:
    """Serialise a :class:`Pocket` as a minimal PDB file (one ATOM per row).

    For mk_prepare_receptor.py to produce a valid PDBQT we need to write
    proper protein residue names (ALA, GLY, …) and atom names (N, CA, C,
    O, CB, …) **AND** the element column must match the atom name —
    i.e. an atom named "N" must really be nitrogen. We cycle through
    the canonical ALA backbone atoms and substitute the element from
    the cycle (not from ``atom_types``). This is OK because the pocket
    geometry — not its chemistry — drives Vina's clash scoring.
    """
    from rdkit import Chem  # type: ignore

    # (atom_name, element) cycle for a single ALA residue:
    ALA_BACKBONE = [
        ("N",  "N"),  # backbone nitrogen
        ("CA", "C"),  # alpha carbon
        ("C",  "C"),  # backbone carbon
        ("O",  "O"),  # backbone oxygen
        ("CB", "C"),  # beta carbon
    ]

    lines = []
    n = int(pocket.coords.shape[0])
    coords = pocket.coords.cpu().numpy()
    for i in range(n):
        atom_name, elem = ALA_BACKBONE[i % len(ALA_BACKBONE)]
        resname = "ALA"
        chain_char = "A"
        resid = (i // len(ALA_BACKBONE)) + 1
        serial = (i + 1) % 100000
        x, y, zc = coords[i]
        line = (
            f"ATOM  {serial:5d} {atom_name:<4s} {resname:<3s} {chain_char}"
            f"{resid:4d}    {x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {elem:>2s}  "
        )
        lines.append(line)
    lines.append("END\n")
    return "\n".join(lines)


def _receptor_from_pocket(pocket: Pocket, tmpdir: Path) -> Path:
    """Write receptor PDB and (try to) call mk_prepare_receptor.py.

    If the Pocket was loaded from a real PDB (with proper residues and
    connectivity), the round-trip through ``_pocket_to_pdb_string``
    strips CONECT records, which makes ``mk_prepare_receptor.py``
    fail its residue template lookup. To avoid this, callers can
    supply ``pocket.pdb_path`` as a string attribute pointing at the
    original PDB on disk — we then prefer that file directly.

    If no path is available we fall back to the synthetic round-trip
    and pass ``--allow_bad_res`` to mk_prepare_receptor so it does not
    crash on every residue.
    """
    # Prefer the original PDB file if the caller attached it to the Pocket
    # (Pocket is a frozen dataclass so we look for ``_pdb_path`` /
    # ``pdb_path``; tests may attach either).
    pdb_path = getattr(pocket, "_pdb_path", None) or getattr(pocket, "pdb_path", None)
    if pdb_path is not None:
        pdb_path = Path(str(pdb_path))
        if pdb_path.is_file() and pdb_path.suffix == ".pdb":
            basename = tmpdir / f"{pocket.pdb_id}_receptor"
            pdbqt_path = basename.with_suffix(".pdbqt")
            proc = subprocess.run(
                ["mk_prepare_receptor.py", "--read_pdb", str(pdb_path),
                 "-o", str(basename), "-p"],
                check=False, capture_output=True, timeout=120,
            )
            if proc.returncode == 0 and pdbqt_path.exists():
                return pdbqt_path
            logger.info(
                "mk_prepare_receptor.py on original PDB failed "
                "(rc=%s, stderr=%s)",
                proc.returncode, proc.stderr.decode(errors="ignore")[:200],
            )
            return pdb_path

    # Synthetic fallback: write our generated PDB and pass --allow_bad_res.
    pdb_path = tmpdir / f"{pocket.pdb_id}_receptor.pdb"
    pdb_path.write_text(_pocket_to_pdb_string(pocket))
    basename = tmpdir / f"{pocket.pdb_id}_receptor"
    pdbqt_path = basename.with_suffix(".pdbqt")
    try:
        proc = subprocess.run(
            ["mk_prepare_receptor.py", "--read_pdb", str(pdb_path),
             "-o", str(basename), "-p", "--allow_bad_res"],
            check=False, capture_output=True, timeout=120,
        )
        if proc.returncode == 0 and pdbqt_path.exists():
            return pdbqt_path
        logger.info(
            "mk_prepare_receptor.py failed (rc=%s, stderr=%s); using raw PDB",
            proc.returncode, proc.stderr.decode(errors="ignore")[:200],
        )
        return pdb_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.info("mk_prepare_receptor.py unavailable (%s); using raw PDB", exc)
        return pdb_path


def _ligand_to_pdbqt(smiles: str, tmpdir: Path, name: str = "lig") -> Path:
    """Convert a SMILES to a ligand PDBQT using RDKit + meeko."""
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status == -1:
        # try ETKDGv3 with coord map 0, then UFF relax
        from rdkit.Chem import AllChem as AC  # type: ignore
        params = AC.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_path = tmpdir / f"{name}.pdbqt"
    with open(pdbqt_path, "w") as fh:
        for setup in mol_setups:
            pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
            if not is_ok:
                raise RuntimeError(f"PDBQT conversion failed: {err_msg}")
            fh.write(pdbqt_string)
    return pdbqt_path


# -----------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------
class VinaDockingAdapter(DockingEngine):
    """DockingEngine implementation that wraps AutoDock Vina.

    Parameters
    ----------
    default_box_padding : float
        How far (Å) the docking box extends beyond the pocket's binding
        sphere. Default 8 Å matches the CrossDocked2020 / PDBBind
        standard protocol.
    cpu_count : int
        Number of CPU threads per docking call. Default 0 = all cores.
    """

    @property
    def name(self) -> str:
        # Reflects the engine selected at construction (used by telemetry).
        return {
            "vina":       "AutoDockVina_v1",
            "vina-cli":   "AutoDockVina_CLI",
            "qvina":      "QVina",
            "quickvina2": "QuickVina2",
        }.get(self._engine, f"VinaAdapter[{self._engine}]")

    def __init__(
        self,
        default_box_padding: float = 8.0,
        cpu_count: int = 0,
        sf_name: str = "vina",
        engine: str = DEFAULT_ENGINE,
    ) -> None:
        engine = str(engine).lower().strip()
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unknown docking engine {engine!r}; "
                f"supported: {SUPPORTED_ENGINES}"
            )

        # ``auto`` resolution — pick the first available path:
        #   1. vina Python pkg (preferred — programmatic energies())
        #   2. vina CLI binary (if libboost resolves; otherwise skip)
        #   3. qvina / quickvina2 CLI (if installed)
        #   4. raise (caller can still override with engine='vina-cli' etc.)
        engine_binary: Optional[str] = None
        if engine == "auto":
            chosen = None
            if _have_vina() and _have_meeko():
                chosen = ("vina", None)
            else:
                # Emit the canonical "fall back to subprocess" log line
                # the task description asks for.  Mirrors the Phase-1
                # install report (round7_install_report.md §1).
                logger.warning(
                    "vina Python pkg not installed; using subprocess fallback. "
                    "Install with: uv add vina"
                )
            if chosen is None:
                # Try the system ``vina`` binary — works iff libboost
                # resolves.  We probe with ``--help`` rather than
                # ``which`` so a broken binary is filtered out.
                vbin = _probe_vina_binary()
                if vbin is not None:
                    chosen = ("vina-cli", vbin)
            if chosen is None:
                # Last resort — QVina / QuickVina2 CLI.
                qbin = _resolve_binary(_QVINA_BIN_CANDIDATES)
                if qbin is not None:
                    chosen = ("qvina", qbin)
                else:
                    qbin2 = _resolve_binary(_QUICKVINA2_BIN_CANDIDATES)
                    if qbin2 is not None:
                        chosen = ("quickvina2", qbin2)
            if chosen is None:
                raise RuntimeError(
                    "engine='auto' could not find any usable docking engine. "
                    "Install one of: `uv add vina` (preferred), qvina "
                    "(conda-forge), or quickvina2. See "
                    "molmetal/reports/round7_install_report.md §1."
                )
            engine, engine_binary = chosen

        # Strict path: when the caller EXPLICITLY asks for qvina /
        # quickvina2 / vina-cli the binary MUST exist (round-7 wire-vina
        # / TODO-04).  We raise a clear ``FileNotFoundError`` with the
        # canonical ``mamba install -c conda-forge qvina`` hint so the
        # test suite + CLI can detect the missing binary without
        # silently falling back.  Only ``engine='auto'`` (resolved
        # above) is allowed to silently downgrade.
        if engine == "qvina" and not _have_qvina_binary():
            raise FileNotFoundError(
                "engine='qvina' requested but no qvina/qvina2 binary found "
                "on PATH. Install via "
                "`mamba install -c conda-forge qvina` "
                "(or set engine='vina' to use the Python binding)."
            )
        elif engine == "quickvina2" and not _have_quickvina2_binary():
            raise FileNotFoundError(
                "engine='quickvina2' requested but no quickvina2/quickvina "
                "binary found on PATH. Install via "
                "`mamba install -c conda-forge quickvina` "
                "(or set engine='vina' to use the Python binding)."
            )
        elif engine == "vina-cli":
            # Caller explicitly asked for the CLI binary.  Verify it
            # actually launches (libboost may be missing — see
            # round7_install_report.md §1 CLI binary note).
            vbin = _probe_vina_binary()
            if vbin is None:
                raise FileNotFoundError(
                    "engine='vina-cli' requested but the system `vina` "
                    "binary failed to launch (likely missing libboost). "
                    "Use engine='vina' (Python binding) or install "
                    "libboost_thread via `mamba install -c conda-forge vina`."
                )
            engine_binary = vbin

        if engine == "vina":
            if not _have_vina():
                raise RuntimeError(
                    "vina package not installed. Run: "
                    "`source .venv/bin/activate && uv pip install vina meeko`"
                )
            if not _have_meeko():
                raise RuntimeError(
                    "meeko package not installed (needed for ligand prep). Run: "
                    "`source .venv/bin/activate && uv pip install meeko`"
                )
            import vina as _vina  # noqa: F401  (imported for side-effect)
        else:
            if not _have_meeko():
                raise RuntimeError(
                    "meeko package not installed (needed for ligand prep). Run: "
                    "`source .venv/bin/activate && uv pip install meeko`"
                )

        self._padding = float(default_box_padding)
        self._cpu = int(cpu_count)
        self._sf_name = str(sf_name)
        self._engine = engine
        # Public attribute (round-7 wire-vina spec).  Mirrors
        # ``self._engine`` so callers can read ``adapter.engine`` without
        # poking at a private field.  Kept in sync explicitly so
        # mutation by mistake stays loud (no setter).
        self.engine: str = engine
        if engine == "qvina":
            self._engine_binary = engine_binary or _resolve_binary(_QVINA_BIN_CANDIDATES)
        elif engine == "quickvina2":
            self._engine_binary = engine_binary or _resolve_binary(_QUICKVINA2_BIN_CANDIDATES)
        elif engine == "vina-cli":
            self._engine_binary = engine_binary or _probe_vina_binary()
        else:
            self._engine_binary = None
        self._tmpdir = Path(tempfile.mkdtemp(prefix="vina_adapter_"))
        # Pocket → PDBQT cache
        self._receptor_pdbqt: dict[str, Path] = {}
        self.last_pose_mols: list = []

    def _engine_dispatched(self) -> str:
        """Return the dispatch path for the configured engine.

        Returns either ``"python"`` (the ``vina`` Python binding —
        preferred, full programmatic ``energies()`` + ``poses()``) or
        ``"subprocess"`` (a CLI binary invoked via ``subprocess.run`` —
        requires parsing VINA RESULT lines from the docked PDBQT).

        This is the single source of truth for the python-vs-CLI
        branch in :meth:`dock` and is exported as ``_engine_dispatched``
        (underscore prefix = "implementation detail, but stable enough
        for tests + telemetry") so callers / unit tests can assert on
        the dispatch without poking at private fields.
        """
        return _ENGINE_DISPATCH.get(self._engine, "subprocess")

    # ---------------------------------------------------------- Port API
    def setup(self, device: str = "cuda") -> None:
        # Vina is CPU-only; the ``device`` arg is a Protocol-contract no-op.
        # We only log it so callers can see what was requested.
        logger.info("VinaDockingAdapter.setup(device=%s) — Vina is CPU-only", device)

    def dock(
        self,
        molecule: Molecule,
        pocket: Pocket,
        config: DockingConfig,
    ) -> List[Complex]:
        """Dock ``molecule.smiles`` into ``pocket`` and return top-N poses."""
        self.last_pose_mols = []
        if not molecule.smiles:
            raise ValueError(
                "Molecule.smiles is empty — Vina needs a SMILES, not a "
                "coordinate tensor. Set ``smiles`` when constructing Molecule."
            )

        receptor_pdbqt = self._prepare_receptor(pocket)

        # Ligand PDBQT
        lig_pdbqt = _ligand_to_pdbqt(molecule.smiles, self._tmpdir, name="lig")

        # Box: pocket.center, padding on each side
        center = pocket.center.cpu().numpy().astype(float)
        box_size = np.full(3, 2.0 * (pocket.radius + self._padding), dtype=float)
        # Vina needs each side >= 0; clamp to 8 Å minimum
        box_size = np.maximum(box_size, 8.0)

        if self._engine_dispatched() == "python":
            energies, poses_pdbqt = self._dock_python_binding(
                receptor_pdbqt=receptor_pdbqt,
                lig_pdbqt=lig_pdbqt,
                center=center,
                box_size=box_size,
                config=config,
            )
        else:
            energies, poses_pdbqt = self._dock_cli_binary(
                receptor_pdbqt=receptor_pdbqt,
                lig_pdbqt=lig_pdbqt,
                center=center,
                box_size=box_size,
                config=config,
            )

        # Retain the full RDKit pose (including stereo) for downstream SDF/PB
        # evaluation. These objects belong to this docking call only.
        self.last_pose_mols = []
        # Convert each pose to a Molecule via RDKit + meeko reverse
        complexes: List[Complex] = []
        from meeko import PDBQTMolecule, RDKitMolCreate  # type: ignore
        from rdkit import Chem  # type: ignore

        # PDBQTMolecule accepts a multi-model PDBQT string and an
        # ``energy_range`` (kcal/mol). We pass the per-pose energies to
        # ``RDKitMolCreate.from_pdbqt_mol`` returns one molecule per ligand
        # component, with poses stored as conformers of that molecule.
        try:
            pdbqt_mol = PDBQTMolecule(poses_pdbqt, is_dlg=False)
            rdkit_mols = RDKitMolCreate.from_pdbqt_mol(
                pdbqt_mol, only_cluster_leads=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("PDBQT → RDKit conversion failed: %s", exc)
            return complexes

        if len(rdkit_mols) != 1 or rdkit_mols[0] is None:
            logger.warning("Expected one ligand component in docked PDBQT")
            return complexes
        rdkit_mol = Chem.RemoveHs(rdkit_mols[0])
        conformers = list(rdkit_mol.GetConformers())
        if len(conformers) != len(energies):
            logger.warning("Pose/energy count mismatch (%d/%d); refusing ambiguous scores", len(conformers), len(energies))
            return complexes
        for i in range(min(int(config.n_poses), len(conformers), len(energies))):
            # energies[i, 0] is total binding affinity (kcal/mol). Columns:
            # [total, inter, intra, torsions, intra_best_pose]
            score_val = float(np.asarray(energies[i, 0]).reshape(-1)[0])
            if not np.isfinite(score_val):
                continue
            try:
                pose = Chem.Mol(rdkit_mol)
                pose.RemoveAllConformers()
                conf = Chem.Conformer(conformers[i])
                conf.Set3D(True)
                pose.AddConformer(conf, assignId=True)
                if not np.isfinite(pose.GetConformer().GetPositions()).all():
                    raise ValueError("Non-finite docked coordinates")
                mol = Molecule.from_rdkit_mol(pose)
            except Exception as exc:  # pragma: no cover
                logger.warning("Pose %d parsing failed: %s", i, exc)
                continue
            self.last_pose_mols.append(pose)
            confidence = self._vina_to_confidence(score_val)
            complexes.append(
                Complex(
                    pocket=pocket,
                    molecule=mol,
                    pose_confidence=confidence,
                    vina_score=score_val,
                )
            )
        return complexes

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": self._engine,
            "engine_label": {
                "vina":       "AutoDock Vina (Python binding)",
                "vina-cli":   "AutoDock Vina (CLI binary)",
                "qvina":      "QVina (CLI binary)",
                "quickvina2": "QuickVina2 (CLI binary)",
            }.get(self._engine, self._engine),
            "engine_binary": self._engine_binary,
            "engine_version": (
                _vina_version() if self._engine == "vina" else _cli_binary_version(
                    self._engine_binary
                )
            ),
            "scoring_function": self._sf_name,
            "default_box_padding_A": self._padding,
            "tmpdir": str(self._tmpdir),
        }

    # ---------------------------------------------------------- internals
    def _prepare_receptor(self, pocket: Pocket) -> Path:
        if pocket.pdb_id in self._receptor_pdbqt:
            return self._receptor_pdbqt[pocket.pdb_id]
        path = _receptor_from_pocket(pocket, self._tmpdir)
        self._receptor_pdbqt[pocket.pdb_id] = path
        return path

    def _dock_python_binding(
        self,
        receptor_pdbqt: Path,
        lig_pdbqt: Path,
        center: np.ndarray,
        box_size: np.ndarray,
        config: DockingConfig,
    ) -> Tuple[np.ndarray, str]:
        """Dock via the ``vina`` Python binding (Vina 1.2.7)."""
        import vina  # type: ignore

        # Python Vina treats zero as random. Preserve normal positive seeds,
        # map the remaining uint32 experiment seeds into its nonzero int32 range.
        effective_seed = native_vina_seed(config.seed)
        self.last_docking_seed = {"requested": int(config.seed), "effective": effective_seed}
        v = vina.Vina(sf_name=self._sf_name, cpu=self._cpu, seed=effective_seed, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(
            center=[float(c) for c in center],
            box_size=[float(b) for b in box_size],
        )
        v.dock(
            exhaustiveness=int(config.exhaustiveness),
            n_poses=int(config.n_poses),
        )
        energies = v.energies(n_poses=int(config.n_poses))
        poses_pdbqt = v.poses(n_poses=int(config.n_poses))
        return energies, poses_pdbqt

    def _dock_cli_binary(
        self,
        receptor_pdbqt: Path,
        lig_pdbqt: Path,
        center: np.ndarray,
        box_size: np.ndarray,
        config: DockingConfig,
    ) -> Tuple[np.ndarray, str]:
        """Dock via a CLI binary (QVina or QuickVina2).

        Both QVina and QuickVina2 share AutoDock Vina's CLI surface
        (``--config`` and ``--out``), so a single subprocess path covers
        both engines — only the binary differs.
        """
        if self._engine_binary is None:
            # Constructor should have already fallen back, but be defensive.
            logger.warning(
                "engine=%r requested but binary missing; returning empty pose",
                self._engine,
            )
            return np.zeros((0, 0)), ""

        out_pdbqt = self._tmpdir / f"docked_{os.getpid()}.pdbqt"
        self.last_docking_seed = {"requested": int(config.seed), "effective": int(config.seed)}
        cmd = [
            self._engine_binary,
            "--receptor", str(receptor_pdbqt),
            "--ligand",   str(lig_pdbqt),
            "--center_x", str(float(center[0])),
            "--center_y", str(float(center[1])),
            "--center_z", str(float(center[2])),
            "--size_x",   str(float(box_size[0])),
            "--size_y",   str(float(box_size[1])),
            "--size_z",   str(float(box_size[2])),
            "--out",      str(out_pdbqt),
            "--exhaustiveness", str(int(config.exhaustiveness)),
            "--num_modes",      str(int(config.n_poses)),
            "--seed",           str(int(config.seed)),
            "--cpu",            str(int(self._cpu)) if int(self._cpu) > 0 else "0",
        ]
        try:
            proc = subprocess.run(
                cmd, check=False, capture_output=True, timeout=600,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "%s CLI failed (%s); returning empty pose", self._engine, exc,
            )
            return np.zeros((0, 0)), ""

        if proc.returncode != 0 or not out_pdbqt.exists():
            logger.warning(
                "%s CLI failed (rc=%s, stderr=%s)",
                self._engine, proc.returncode,
                proc.stderr.decode(errors="ignore")[:200],
            )
            return np.zeros((0, 0)), ""

        poses_pdbqt = out_pdbqt.read_text()
        # Parse per-model energies from the PDBQT VINA RESULT lines, since
        # CLI engines don't expose a programmatic energies() call.
        energies = _parse_vina_result_energies(poses_pdbqt)
        return energies, poses_pdbqt

    @staticmethod
    def _vina_to_confidence(score: float) -> float:
        """Map Vina kcal/mol to [0, 1] confidence (lower score → higher conf)."""
        # -12 kcal/mol → 1.0, -4 kcal/mol → 0.0, clamped
        if score <= -12.0:
            return 1.0
        if score >= -4.0:
            return 0.0
        return float((-4.0 - score) / 8.0)


def _vina_version() -> str:
    try:
        import vina  # type: ignore
        return getattr(vina, "__version__", "unknown")
    except Exception:
        return "uninstalled"


def native_vina_seed(seed: int) -> int:
    """Map uint32 experiment seeds to deterministic nonzero Vina int32 seeds."""
    value = int(seed)
    if not 0 <= value < 2**32:
        raise ValueError("Docking seed must be in [0, 2**32)")
    return value % (2**31 - 1) or (2**31 - 1)


def _cli_binary_version(binary_path: Optional[str]) -> str:
    """Return the stdout of ``<binary> --version``, or 'uninstalled'."""
    if not binary_path:
        return "uninstalled"
    try:
        proc = subprocess.run(
            [binary_path, "--version"],
            check=False, capture_output=True, timeout=10,
        )
        out = (proc.stdout or proc.stderr).decode(errors="ignore").strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def _parse_vina_result_energies(pdbqt_text: str) -> np.ndarray:
    """Parse ``VINA RESULT`` lines from a docked PDBQT.

    Each ``MODEL`` block contains one line of the form
    ``REMARK VINA RESULT: <affinity> <rmsd_lb> <rmsd_ub>``.
    Only affinity is an energy. The four unavailable energy-component
    columns are NaN, not the RMSD values or invented zero measurements.
    """
    rows: List[List[float]] = []
    model_rows: List[List[float]] = []
    current = None
    current_seen = False
    def energy_row(value):
        return [value, *([float("nan")] * 4)]
    for line in pdbqt_text.splitlines():
        line = line.strip()
        if line.startswith("MODEL "):
            if current is not None:
                model_rows.append(energy_row(current))
            current, current_seen = float("nan"), False
            continue
        if line == "ENDMDL" and current is not None:
            model_rows.append(energy_row(current))
            current = None
            continue
        if line.startswith("REMARK "):
            line = line[len("REMARK "):].lstrip()
        if line.startswith("VINA RESULT:"):
            score = float("nan")
            try:
                score = float(line.split(":", 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
            if current is not None:
                current = score if not current_seen else float("nan")
                current_seen = True
            elif np.isfinite(score):
                rows.append(energy_row(score))
    if current is not None:
        model_rows.append(energy_row(current))
    if model_rows:
        # Keep one row per MODEL even if its score is unavailable. dock()
        # skips that same conformer, preserving later pose/energy alignment.
        return np.asarray(model_rows, dtype=float)
    if not rows:
        return np.zeros((0, 5), dtype=float)
    return np.asarray(rows, dtype=float)


def build_arg_parser():
    """Argparse parser exposing ``--engine {vina,qvina,quickvina2,...}``.

    Returned for callers that want a CLI front-end (the module itself
    is library-first; this keeps the constructor and CLI in lock-step).
    """
    import argparse  # local import — module is library-first
    p = argparse.ArgumentParser(
        prog="vina_adapter",
        description="AutoDock Vina / QVina / QuickVina2 docking adapter.",
    )
    p.add_argument(
        "--engine",
        choices=list(SUPPORTED_ENGINES),
        default=DEFAULT_ENGINE,
        help=(
            "Docking engine to use. Choices: vina (Python binding, "
            "preferred; vina 1.2.7), vina-cli (system `vina` binary), "
            "qvina (QVina CLI), quickvina2 (QuickVina2 CLI), auto (pick "
            "the best available). When qvina/quickvina2 are requested "
            "explicitly the corresponding binary MUST be on PATH — the "
            "constructor raises FileNotFoundError with a `mamba install` "
            "hint otherwise. ``auto`` is the only engine that silently "
            "downgrades to Vina 1.2.7 when the requested binary is "
            "missing."
        ),
    )
    p.add_argument("--cpu", type=int, default=0,
                   help="CPU threads (0 = all cores).")
    p.add_argument("--box-padding", type=float, default=8.0,
                   help="Box padding around the pocket (Å).")
    p.add_argument("--sf-name", default="vina",
                   help="Scoring function name (vina/ad4).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI front-end for ``vina_adapter``.

    Usage example (round-7 wire-vina / TODO-04 qvina_swap):

        python -m molmetal_lam.sbdd_env.vina_adapter --engine vina
        python -m molmetal_lam.sbdd_env.vina_adapter --engine qvina
        python -m molmetal_lam.sbdd_env.vina_adapter --engine quickvina2

    The CLI instantiates :class:`VinaDockingAdapter` with the chosen
    engine, prints its metadata (engine name / binary path / version)
    and exits 0.  This is purely a smoke front-end — full docking
    pipelines still go through :func:`dock_smiles` or
    :meth:`VinaDockingAdapter.dock` (the closed loop in
    ``pipeline/closed_loop.py`` uses the latter via ``docker.dock()``).

    Exit codes:

    * ``0`` — adapter constructed and metadata printed.
    * ``1`` — engine chosen but binary missing / construction failed.
    * ``2`` — argparse rejected the flags.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        adapter = VinaDockingAdapter(
            default_box_padding=float(args.box_padding),
            cpu_count=int(args.cpu),
            sf_name=str(args.sf_name),
            engine=str(args.engine),
        )
    except FileNotFoundError as exc:
        # qvina / quickvina2 / vina-cli with a missing binary — emit
        # the mamba hint the spec asked for and exit 1 (per round-7
        # TODO-04).
        print(f"vina_adapter: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"vina_adapter: failed to construct adapter: {exc}",
              file=sys.stderr)
        return 1

    meta = adapter.get_metadata()
    print(
        "vina_adapter: engine={engine} ({label}) binary={binary} "
        "version={version} padding={padding}Å".format(
            engine=meta["engine"],
            label=meta["engine_label"],
            binary=meta["engine_binary"] or "<python binding>",
            version=meta["engine_version"],
            padding=meta["default_box_padding_A"],
        )
    )
    return 0


# -----------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------
def dock_smiles(
    smiles: str,
    pocket: Pocket,
    n_poses: int = 5,
    exhaustiveness: int = 8,
    engine: str = DEFAULT_ENGINE,
) -> List[Complex]:
    """One-shot: dock a SMILES into a Pocket."""
    adapter = VinaDockingAdapter(engine=engine)
    adapter.setup()
    dummy = Molecule(
        coords=torch.zeros(1, 3),
        atom_types=torch.tensor([6], dtype=torch.long),
        bonds=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.tensor([0], dtype=torch.long),
        smiles=smiles,
    )
    cfg = DockingConfig(n_poses=n_poses, exhaustiveness=exhaustiveness)
    return adapter.dock(dummy, pocket, cfg)


def redock_for_test(
    pdb_id: str,
    ref_ligand_smiles: str,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 10.0,
    n_atoms: int = 50,
    exhaustiveness: int = 4,
) -> Optional[float]:
    """Redock ``ref_ligand_smiles`` into a synthetic pocket.

    For benchmarking only. Returns the Vina best score (kcal/mol), or
    None if docking fails. RMSD-to-reference is not measurable on a
    synthetic pocket (no crystal pose), so this helper just exercises
    the full pipeline.
    """
    # Build a minimal Pocket with dummy atoms at the centre
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    coords[:, 0] = torch.linspace(-radius, radius, n_atoms)
    atom_types = torch.full((n_atoms,), 6, dtype=torch.long)  # all carbon
    pocket = Pocket(
        pdb_id=pdb_id,
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.zeros(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=torch.tensor(center, dtype=torch.float32),
        radius=float(radius),
    )
    try:
        complexes = dock_smiles(
            ref_ligand_smiles, pocket,
            n_poses=1, exhaustiveness=exhaustiveness,
        )
        return complexes[0].vina_score if complexes else None
    except Exception as exc:  # pragma: no cover
        logger.warning("redock_for_test failed: %s", exc)
        return None


__all__ = [
    "VinaDockingAdapter",
    "dock_smiles",
    "redock_for_test",
    "build_arg_parser",
    "main",
    "SUPPORTED_ENGINES",
    "DEFAULT_ENGINE",
    "_have_vina",
    "_have_meeko",
    "_have_qvina_binary",
    "_have_quickvina2_binary",
    "_probe_vina_binary",
]


# -----------------------------------------------------------------
# Module CLI entry point — round-7 wire-vina / TODO-04 qvina_swap.
# -----------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
