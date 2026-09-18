# T4 — MetalLigandGenerator port + adapter complete

**Date:** 2026-09-11
**Branch:** local TODO
**Scope:** wire the metal-specific generator port to a concrete adapter that
consumes :mod:`molmetal.data.metal_smiles` (multi-component form) and emits
``List[Complex]`` end-to-end.

## What was added

| Path | Role |
| --- | --- |
| `molmetal/ports/generators.py` | New :class:`MetalLigandGenerator` Protocol (PEP 544, ``@runtime_checkable``) + :class:`MetalLigandConfig` dataclass. Method shape: ``generate(pocket, metal, ligand_smiles, config) -> List[Complex]``; attrs ``name``, ``setup``, ``get_metadata``. |
| `molmetal/molmetal_lam/sbdd_env/metal_generator_adapter.py` | :class:`MetalLigandAdapter` (frozen-shape stub satisfying the Protocol). Wraps :func:`molmetal.data.metal_smiles.reconstruct_multi_component`; optional RDKit ETKDGv3 + MMFF94/UFF 3-D embedding; emits 0-atom placeholder :class:`Molecule` carrying only the SMILES when embedding is disabled (default), letting the downstream Vina adapter re-embed. Also exports :func:`reconstruct_complex_smiles` for direct callers. |
| `molmetal/tests/test_metal_generator_adapter.py` | 14 pytest cases organised in three classes (T4 requirement met): **Protocol compliance** (4 tests — ``isinstance(adapter, MetalLigandGenerator)``, attrs, setup, config dataclass), **multi-component round-trip** (5 tests — ``str``/``list`` helper inputs, ``List[Complex]`` return, RDKit round-trip with 5 fragments for cisplatin, empty-input → ``[]``), **metal token injection** (5 tests — Pt/Ru/Ir token present, oxidation-state-driven coordination, end-to-end Pt atom in Complex). |

## Test result

```
14 passed in 1.39s
```

## Design choices

1. **Multi-component form only.** The Protocol emits ``L1.L2....Ln.[M]`` (RDKit round-trips 100% on MetalCytoToxDB), not the legacy bracket form which splits into disconnected fragments for Pt/Pd/Au.
2. **Metal token injection inside ``generate``.** Callers pass ``metal`` separately from ``ligand_smiles`` so the adapter can drive coordination geometry from the metal's oxidation state. The Protocol mirrors :class:`MoleculeGenerator` so the existing orchestration layer can drop it in for metalloprotein targets.
3. **No RDKit at import time.** Reconstruction is pure-string. RDKit is lazy-loaded in ``setup()`` and only used when ``embed_3d=True``; otherwise the adapter returns 0-atom placeholder :class:`Molecule` carrying the SMILES (Vina re-embeds downstream).
4. **Stateless.** Each call reconstructs from scratch — no cached metal tokens or ligand pools — matching the existing Pocket2Mol/FlowDock stub adapters.

## Notes for downstream

* The ``[OH2]`` water placeholders for under-coordinated complexes are intentional and survive RDKit parsing.
* Bare halides (``Cl`` / ``Br``) are kept by the upstream heuristic for cisplatin-like complexes even though ``count_donors`` returns 0 for them — this matches the existing behaviour in :mod:`metal_smiles` and is not changed here.
* A future "live" MetalLigandAdapter could swap the SMILES-reconstruction step for a learned metal-conditional generator (cf. `Metal-Organic VAE` literature); the Protocol is ready for that swap.
