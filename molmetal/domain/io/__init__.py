"""I/O bridges between domain dataclasses and external formats.

  - rdkit_io:    SMILES / SDF / RDKit Mol  ⇄  Molecule
  - pdb_loader:  PDB file                  ⇄  Pocket

Heavy imports (RDKit, BioPython) live inside the helper functions so that
``from molmetal.domain import Pocket, Molecule, Complex`` stays cheap.
"""
