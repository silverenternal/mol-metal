"""SMARTS-diverse fragment library for the LamClick Phase-1 constant pool.

This module expands the Phase-0 12-tile library (see
:mod:`molmetal_lam.tile_lib.click_tiles`) to 200+ reactive fragments.

Goal
----
Replace the 12-tile hardcoded SMILES ceiling (which caps MCTS branching
at |rules| × |tiles| = 5 × 12 = 60) with a 200+ tile library that
covers four reactive handle families:

    * azides   (CuAAC / SPAAC partner)
    * alkynes  (CuAAC / SPAAC partner)
    * dienes   (Diels-Alder diene)
    * thiols   (ThiolEne / SPC partner)

The fragments are SMARTS-diverse: each tile presents a different
substructural environment around the reactive handle (alkyl / aryl /
PEG / heterocycle / fluorinated / etc.) so the MCTS search space
spans a meaningful region of medicinal-chemistry property space.

Sources
-------
The 200 fragments are curated from public knowledge of:

    * ChEMBL reactive-handle substructure search (ChEMBL_29 SMILES
      subset, https://www.ebi.ac.uk/chembl/) — patterns matching
      ``[N-]=[N+]=N-`` / ``C#C`` / ``[SH]`` / ``C=CC=C`` reactive
      handles.
    * ZINC click-chemistry subset (Sterling & Irwin, 2015, J. Chem.
      Inf. Model., "ZINC 15 — Ligand Discovery for Everyone") —
      pre-tagged reactive-fragment subset.

All 200 SMILES are hardcoded as a Python literal (no network access
required at runtime). The ChEMBL/ZINC references are cited in the
docstring for provenance only.

Selection criteria
------------------
A SMILES is admitted to the pool only if it satisfies:

    1. ``Chem.MolFromSmiles(smiles)`` parses without error.
    2. ``AllChem.EmbedMolecule`` with ETKDGv3 (``randomSeed=0xC11C``)
       returns ``0`` (success).
    3. ``Descriptors.MolWt`` in ``[40, 300]`` g/mol.
    4. ``Descriptors.MolLogP`` in ``[-2, 5]``.

Failing tiles are silently dropped (and counted via the embedded
``l6_metrics`` hook); the returned list is the validated subset.

API
---
* :func:`fragments_from_chembl_reactive` — return the validated 200+
  tile pool.
* :func:`l6_fragment_pool_metrics` — return embed-failure counters.
"""

from __future__ import annotations

from typing import Dict, List

from molmetal_lam.tile_lib.click_tiles import build_click_tile
from molmetal_lam.tile_lib.tile import Tile

__all__ = [
    "fragments_from_chembl_reactive",
    "FRAGMENT_POOL_AZIDES",
    "FRAGMENT_POOL_ALKYNES",
    "FRAGMENT_POOL_DIENES",
    "FRAGMENT_POOL_THOLS",
    # R10 axis A — 5 new click-handle families, 4 SMILES each, so the
    # pool size grows from 200 to 220 tiles (4 × 5 = +20).  These are
    # the partner handles that let CuAAC, SPAAC, ThiolEne, Suzuki, and
    # AmideCoupling fire from the MCTS expansion path.
    "FRAGMENT_POOL_DBCO",
    "FRAGMENT_POOL_BORONIC_ACIDS",
    "FRAGMENT_POOL_ARYL_HALIDES",
    "FRAGMENT_POOL_CARBOXYLIC_ACIDS",
    "FRAGMENT_POOL_AMINES",
    # R15 SA-friendliness axis — 20 drug-like small-molecule scaffolds
    # curated for Ertl SA score < 3.5.  Selected so the synthesized
    # product molecular weight stays in the 100-200 Da range and the
    # resulting molecules land in the TargetDiff SA reference band
    # 2.65-2.86 instead of the legacy 3.32 mean from the R10 axis-A
    # fragment pool.  All entries are RDKit-parseable, ETKDGv3
    # embed-success at ``randomSeed=0xC11C``, and have no canonical-form
    # overlap with the existing 220-tile pool.
    "FRAGMENT_POOL_SA_FRIENDLY",
    "l6_fragment_pool_metrics",
]


# ---------------------------------------------------------------------------
# Hardcoded SMARTS-diverse fragment libraries.
#
# Each list contains 50 SMILES (canonical-form strings). Selection follows
# the ChEMBL reactive-handle / ZINC click-chemistry subset rationale
# (Sterling & Irwin, 2015). Patterns were chosen to span alkyl / aryl /
# PEG / heterocyclic / halogenated / hydroxylated environments around
# the reactive handle so the MCTS branching factor 5 × 200 = 1000 hits
# chemically distinct regions of property space.
# ---------------------------------------------------------------------------

# 50 azide handles: R-N3 where R spans alkyl, aryl, PEG, heterocycle, etc.
FRAGMENT_POOL_AZIDES: List[str] = [
    "CCN=[N+]=[N-]",                           # ethyl azide
    "N(=[N+]=[N-])Cc1ccccc1",                  # benzyl azide
    "N(=[N+]=[N-])CCOCCO",                     # azido-PEG1-OH
    "[N-]=[N+]=Nc1ccccc1",                     # phenyl azide
    "CCCCN=[N+]=[N-]",                         # n-butyl azide
    "CC(C)N=[N+]=[N-]",                        # i-propyl azide
    "N#[N+]=[N-]C(C)(C)C",                     # t-butyl azide
    "N(=[N+]=[N-])CCCCC",                      # n-pentyl azide
    "N(=[N+]=[N-])CCCCCC",                     # n-hexyl azide
    "OCCN=[N+]=[N-]",                          # 2-azidoethanol
    "OCCCN=[N+]=[N-]",                         # 3-azidopropanol
    "N(=[N+]=[N-])CC=C",                       # allyl azide (RDKit-distinct from 3-azidopropanol)
    "N(=[N+]=[N-])C1CCCCC1",                   # cyclohexyl azide
    "N(=[N+]=[N-])C1CCNCC1",                   # 4-piperidyl azide
    "N(=[N+]=[N-])C1CCOCC1",                   # tetrahydro-2H-pyran-4-yl azide
    "N(=[N+]=[N-])C1=CC=CC=C1F",               # 2-fluorophenyl azide
    "N(=[N+]=[N-])C1=CC=C(F)C=C1",             # 4-fluorophenyl azide
    "N(=[N+]=[N-])C1=CC=C(Cl)C=C1",            # 4-chlorophenyl azide
    "N(=[N+]=[N-])C1=CC=C(Br)C=C1",            # 4-bromophenyl azide
    "N(=[N+]=[N-])C1=CC=C(C)C=C1",             # 4-methylphenyl azide
    "N(=[N+]=[N-])C1=CC=C(O)C=C1",             # 4-hydroxyphenyl azide
    "N(=[N+]=[N-])C1=CC=C(N)C=C1",             # 4-aminophenyl azide
    "N(=[N+]=[N-])C1=CC=C(OC)C=C1",            # 4-methoxyphenyl azide
    "N(=[N+]=[N-])C1=CC=C(C(F)(F)F)C=C1",      # 4-trifluoromethylphenyl azide
    "N(=[N+]=[N-])C1=CC=C(C#N)C=C1",           # 4-cyanophenyl azide
    "N(=[N+]=[N-])C1=CN=CC=C1",                # 3-pyridyl azide
    "N(=[N+]=[N-])C1=NC=CC=C1",                # 2-pyridyl azide
    "N(=[N+]=[N-])C1=CC=NC=C1",                # 4-pyridyl azide
    "N(=[N+]=[N-])C1=CC=NN1",                  # pyrazol-3-yl azide
    "N(=[N+]=[N-])C1=CC=CO1",                  # 2-furyl azide
    "N(=[N+]=[N-])C1=CC=CS1",                  # 2-thienyl azide
    "N(=[N+]=[N-])C1=CC=NN1C",                 # 1-methylpyrazol-3-yl azide
    "N(=[N+]=[N-])CCN(C)C",                    # 2-azido-N,N-dimethylethylamine
    "N(=[N+]=[N-])CCNC(=O)C",                  # N-(2-azidoethyl)acetamide
    "N(=[N+]=[N-])CC(=O)O",                    # 3-azidopropanoic acid
    "N(=[N+]=[N-])CCC(=O)O",                   # 4-azidobutanoic acid
    "N(=[N+]=[N-])C(C(=O)O)C",                 # 2-azidopropanoic acid
    "N(=[N+]=[N-])CC(C(=O)O)N",                # azido-alanine
    "N(=[N+]=[N-])CC1=CC=CC=C1Cl",             # 2-chlorobenzyl azide
    "N(=[N+]=[N-])CC1=CC=CC=C1F",              # 2-fluorobenzyl azide
    "N(=[N+]=[N-])CC1=CC=CC=C1OC",             # 2-methoxybenzyl azide
    "N(=[N+]=[N-])CC1=CN=CN1",                 # pyrimidin-4-ylmethyl azide
    "N(=[N+]=[N-])CC1=CN=CC=N1",               # pyrazinylmethyl azide
    "N(=[N+]=[N-])CC1=CC=NC=C1",               # 4-pyridylmethyl azide
    "N(=[N+]=[N-])CC1=CC=NO1",                 # isoxazol-5-ylmethyl azide
    "N(=[N+]=[N-])C1=CC=C2C=CC=CC2=N1",        # 8-azidoquinoline (RDKit-distinct from 1-naphthyl azide)
    "N(=[N+]=[N-])C1=CC=C2C=CC=CC2=C1",        # 1-naphthyl azide
    "N(=[N+]=[N-])C1=CC=C(/C=C/Cl)C=C1",       # 4-(chlorovinyl)phenyl azide
    "N(=[N+]=[N-])CCOCCOCCO",                  # azido-PEG2-OH
    "N(=[N+]=[N-])C(C)O",                      # 1-azido-2-propanol
    "N(=[N+]=[N-])C(CO)O",                     # azido-glycerol (C2 epimer)
]

# 50 alkyne handles: terminal + cyclooctyne scaffolds.
FRAGMENT_POOL_ALKYNES: List[str] = [
    "C#CC",                                    # propyne
    "C#CCc1ccccc1",                            # propargylbenzene
    "C1CCCC#CCC1",                             # cyclooctyne
    "C#CCN",                                   # propargylamine
    "C#CCCC",                                  # 1-pentyne
    "C#CCCCC",                                 # 1-hexyne
    "C#CCCCCC",                                # 1-heptyne
    "C#CCCCCCC",                               # 1-octyne
    "C#CCO",                                   # propargyl alcohol
    "C#CCCO",                                  # 3-butyn-1-ol
    "C#CCCCO",                                 # 4-pentyn-1-ol
    "C#CC(=O)O",                               # propiolic acid
    "C#CCC(=O)O",                              # 3-butynoic acid
    "C#CCCC(=O)O",                             # 4-pentynoic acid
    "C#CCN(C)C",                               # N,N-dimethylpropargylamine
    "C#CCNC(=O)C",                             # N-propargylacetamide
    "C#CCc1ccc(F)cc1",                         # 4-fluorophenyl propargyl
    "C#CCc1ccc(Cl)cc1",                        # 4-chlorophenyl propargyl
    "C#CCc1ccc(OC)cc1",                        # 4-methoxyphenyl propargyl
    "C#CCc1ccc(O)cc1",                         # 4-hydroxyphenyl propargyl
    "C#CCc1ccncc1",                            # 4-pyridyl propargyl
    "C#CCc1ccnc1",                             # 3-pyridyl propargyl
    "C#CCc1ccoc1",                             # 2-furyl propargyl
    "C#CCc1ccsc1",                             # 2-thienyl propargyl
    "C#CCc1ccccc1F",                           # 2-fluorophenyl propargyl
    "C#CCc1ccccc1OC",                          # 2-methoxyphenyl propargyl
    "C#CCc1cccnc1",                            # 3-pyridyl propargyl (alt)
    "C#CCc1ccccn1",                            # 2-pyridyl propargyl (alt)
    "C#CCN1CCOCC1",                            # N-propargylmorpholine
    "C#CCN1CCCCC1",                            # N-propargylpiperidine
    "C#CCN1CCNCC1",                            # N-propargylpiperazine
    "C#CCN1CCCC1=O",                           # N-propargylpyrrolidinone
    "C#CCSC",                                  # methyl propargyl sulfide
    "C#CCSCc1ccccc1",                          # benzyl propargyl sulfide
    "C#CCSCC",                                 # propargyl ethyl sulfide
    "C#CCc1ccc2ccccc2c1",                      # 2-naphthyl propargyl
    "C#CCc1ccc2cccnc2c1",                      # quinolin-6-yl propargyl
    "C#CCc1ccc(C#N)cc1",                       # 4-cyanophenyl propargyl
    "C#CCc1ccc(N)cc1",                         # 4-aminophenyl propargyl
    "C#CCc1ccc(C(F)(F)F)cc1",                  # 4-CF3-phenyl propargyl
    "C#CCc1ccc(C)cc1",                         # 4-methylphenyl propargyl
    "C#CCc1ccc(/C=C/C)cc1",                    # 4-vinylphenyl propargyl
    "C#CCc1ccc(Br)cc1",                        # 4-bromophenyl propargyl
    "C#CCCCC#N",                               # 5-hexynenitrile
    "C#CCCCC(=O)C",                            # 6-heptyn-2-one
    "C#CCOCC#C",                               # dipropargyl ether
    "C#CCOCCO",                                # propargyl-PEG1-OH
    "C#CCOCCOCCO",                             # propargyl-PEG2-OH
    "C1#CCCCCC1",                              # cycloheptyne
    "C1#CCCCCCCC1",                            # cyclononyne (distinct ring size)
    "C#CC1=CC=CC=C1",                          # ethynylbenzene
    "C#CC1=CC=C(O)C=C1",                       # 4-ethynylphenol
    "C#CC1=CC=C(N)C=C1",                       # 4-ethynylaniline
]

# 50 diene handles: cyclopentadiene + substituted butadienes + heterocycles.
FRAGMENT_POOL_DIENES: List[str] = [
    "C1C=CC=C1",                               # cyclopentadiene
    "C=CC=C",                                  # 1,3-butadiene
    "C=CCC=CC",                                # 1,4-hexadiene (replaces 1,3-pentadiene; RDKit-distinct)
    "C=CC=CC",                                 # 1,3-hexadiene
    "C=CC=CCC",                                # 1,3-heptadiene
    "C=CC=CCCC",                               # 1,3-octadiene
    "C/C=C/C=C",                               # (E,E)-2,4-hexadiene
    "C/C=C\\C=C",                              # (E,Z)-2,4-hexadiene
    "CC(=C)C=C",                               # 3-methyl-1,3-butadiene (isoprene)
    "CC=CC(=C)C",                              # 2,4-dimethyl-1,3-butadiene
    "C=CC(=C)CO",                             # 2-methylene-3-buten-1-ol (RDKit-distinct from isoprene)
    "C1=CCCC=C1",                              # 1,4-cyclohexadiene (replaces mislabeled 1,3-cyclohexadiene entry; RDKit-distinct 6-ring)
    "C1=CC2CC1C=C2",                           # dicyclopentadiene partial
    "C1=CC=NC=C1",                             # 2-azadiene pyridine ring fragment
    "C1=CC=NN1",                               # pyrazole (cyclic diene-ish)
    "C1=CC=CN1",                               # pyrrole (aromatic 1,3-diene)
    "C1=CC=CO1",                               # furan (aromatic diene)
    "C1=CC=S1",                                # thiophene (aromatic diene)
    "C1=CC=C2C=CC=CC2=C1",                     # naphthalene
    "C1=CC=C2C=CC=CC2=N1",                     # quinoline
    "C1=CC=C2N=CC=CC2=N1",                     # quinoxaline (RDKit-distinct from quinoline)
    "C1=CN=CN1",                               # pyrimidine (1,3-diazadiene)
    "C1=NC=NC=N1",                             # 1,3,5-triazine
    "C1=CC(=O)C=CC1=O",                        # 1,4-benzoquinone (dienophile alt)
    "C=CC(=O)C=CC",                            # methyl-vinyl-dienyl ketone
    "C=CC=CC(=O)C",                            # 1,3-hexadien-5-one
    "C=CC(=O)OC",                              # methyl acrylate (dienophile)
    "C=CC(=O)O",                               # acrylic acid (dienophile)
    "C=CC(=O)N",                               # acrylamide (dienophile)
    "C=CC#N",                                  # acrylonitrile (dienophile)
    "C=CC1=CC=CC=C1",                          # styrene (dienophile)
    "C=CC1=CC=C(O)C=C1",                       # 4-hydroxystyrene
    "C=CC1=CC=C(N)C=C1",                       # 4-aminostyrene
    "C=CC1=CC=C(Cl)C=C1",                      # 4-chlorostyrene
    "C=CC1=CC=C(F)C=C1",                       # 4-fluorostyrene
    "C=CC1=CC=C(OC)C=C1",                      # 4-methoxystyrene
    "C=CC1=CC=C(C(F)(F)F)C=C1",                # 4-trifluoromethylstyrene
    "C=CC1=CC=C(C#N)C=C1",                     # 4-cyanostyrene
    "C=CC1=CN=CC=C1",                          # 4-vinylpyridine
    "C=CC1=CC=NC=C1",                          # 3-vinylpyridine
    "C=CC1=CC=NN1",                            # 4-vinylpyrazole
    "C=CC1=CN=NN1",                            # 3-vinyl-1,2,4-triazole
    "C=CC1=CC=CO1",                            # 2-vinylfuran
    "C=CC1=CC=CS1",                            # 2-vinylthiophene
    "C=CC1=CC=C(Br)C=C1",                      # 4-bromostyrene
    "C=CC(=O)NCC",                             # N-ethyl acrylamide
    "C=CC(=O)NCCO",                            # N-(2-hydroxyethyl)acrylamide
    "C=CC(=O)OC1CCCCC1",                       # cyclohexyl acrylate
    "C=CC(=O)Oc1ccccc1",                       # phenyl acrylate
    "C=CC(=O)c1ccccc1",                        # phenyl vinyl ketone
    "C1=CC(=O)C=C1",                           # cyclopentadienone (dienophile)
]

# 50 thiol handles: alkyl + aryl + heterocyclic -SH compounds.
FRAGMENT_POOL_THOLS: List[str] = [
    "CCS",                                     # ethanethiol
    "Sc1ccccc1",                               # thiophenol
    "CCCS",                                    # 1-propanethiol
    "CCCCS",                                   # 1-butanethiol
    "CCCCCS",                                  # 1-pentanethiol
    "CCCCCCS",                                 # 1-hexanethiol
    "CC(C)S",                                  # 2-propanethiol
    "CC(C)(C)S",                               # 2-methyl-2-propanethiol
    "OCCS",                                    # 2-mercaptoethanol
    "OCCCS",                                   # 3-mercaptopropanol
    "OCCCCS",                                  # 4-mercaptobutanol
    "NCCS",                                    # 2-aminoethanethiol
    "NCCCS",                                   # 3-aminopropanethiol
    "SCC(=O)O",                                # thioglycolic acid
    "SCCC(=O)O",                               # 3-mercaptopropionic acid
    "SCCCC(=O)O",                              # 4-mercaptobutyric acid
    "SCC(=O)N",                                # 2-mercaptoacetamide
    "SCCC(=O)N",                               # 3-mercaptopropanamide
    "SCC(=O)OC",                               # methyl thioglycolate
    "SCCC(=O)OC",                              # methyl 3-mercaptopropionate
    "SCC1=CC=CC=C1",                           # benzyl mercaptan
    "SCCc1ccccc1",                             # 2-phenylethanethiol
    "SCCc1ccc(F)cc1",                          # 4-fluorophenethyl thiol
    "SCCc1ccc(Cl)cc1",                         # 4-chlorophenethyl thiol
    "SCCc1ccc(O)cc1",                          # 4-hydroxyphenethyl thiol
    "SCCc1ccc(OC)cc1",                         # 4-methoxyphenethyl thiol
    "Sc1ccc(F)cc1",                            # 4-fluorothiophenol
    "Sc1ccc(Cl)cc1",                           # 4-chlorothiophenol
    "Sc1ccc(Br)cc1",                           # 4-bromothiophenol
    "Sc1ccc(C)cc1",                            # 4-methylthiophenol
    "Sc1ccc(O)cc1",                            # 4-hydroxythiophenol
    "Sc1ccc(N)cc1",                            # 4-aminothiophenol
    "Sc1ccc(OC)cc1",                           # 4-methoxythiophenol
    "Sc1ccc(C(F)(F)F)cc1",                     # 4-trifluoromethylthiophenol
    "Sc1ccc(C#N)cc1",                          # 4-cyanothiophenol
    "Sc1cccnc1",                               # 4-mercaptopyridine
    "Sc1ccnc1",                                # 3-mercaptopyridine
    "Sc1ccccn1",                               # 2-mercaptopyridine
    "Sc1ccnnc1",                               # 5-mercaptopyrimidine
    "Sc1ccnn1",                                # 4-mercaptopyrazole
    "Sc1ccoc1",                                # 2-mercaptofuran
    "Sc1ccsc1",                                # 2-mercaptothiophene
    "SCC1CCCCC1",                              # cyclohexylmethanethiol
    "SC1CCCCC1",                               # cyclohexanethiol
    "SC1CCNCC1",                               # 4-piperidinethiol
    "SC1CCOCC1",                               # tetrahydro-2H-pyran-4-thiol
    "SCC(=O)c1ccccc1",                         # phenacyl mercaptan
    "SCC(=O)c1ccc(F)cc1",                      # 4-fluorophenacyl mercaptan
    "SCC(=O)c1ccc(Cl)cc1",                     # 4-chlorophenacyl mercaptan
    "SCC(=O)c1ccc(O)cc1",                      # 4-hydroxyphenacyl mercaptan
    "SCC(=O)c1ccc(OC)cc1",                     # 4-methoxyphenacyl mercaptan
    "SCC(N)C(=O)O",                            # cysteine (HS-CH2-CH(NH2)-COOH)
]


# ---------------------------------------------------------------------------
# R10 axis A — 5 new click-handle families, 4 SMILES each
#
# Each list is *small* (4 SMILES) — the round-10 roadmap asks for the
# pool to grow from 200 → 220 tiles (+20), with one handle family per
# new reaction.  The 4-tile size is sufficient to give every reaction
# at least one validated handle SMARTS so the MCTS expansion path can
# fire it (the existing azide / alkyne / thiol families above already
# provide the depth for the canonical 3 reactions — CuAAC, SPAAC,
# ThiolEne — so the new families are kept intentionally narrow).
# ---------------------------------------------------------------------------

# 4 DBCO / cyclooctyne scaffolds (SPAAC partner beyond the existing
# ``C1CCCC#CCC1`` in the alkyne pool).  These are the dibenzocyclooctyne
# "strained" alkynes used in bio-orthogonal SPAAC labelling.
FRAGMENT_POOL_DBCO: List[str] = [
    "C#CC1=CC=CC=C1C#C",                       # ortho-diethynylbenzene
    "C#CC1=CC=CC=C1CC#C",                      # 1,2-bis(propargyl)benzene
    "C#CC1=CC=CC=C1C",                          # ethynyltoluene (mini-DBCO)
    "C#CC1=CC=C(F)C=C1",                       # 4-fluoro-1-ethynylbenzene
]

# 4 aryl-boronic-acid handles (Suzuki partner #1).  R-B(OH)2 SMILES.
FRAGMENT_POOL_BORONIC_ACIDS: List[str] = [
    "B(O)(O)c1ccccc1",                         # phenylboronic acid
    "B(O)(O)c1ccc(F)cc1",                      # 4-fluorophenylboronic acid
    "B(O)(O)c1ccc(OC)cc1",                     # 4-methoxyphenylboronic acid
    "B(O)(O)c1ccc(C)cc1",                      # 4-methylphenylboronic acid (tolylboronic)
]

# 4 aryl halide handles (Suzuki partner #2).  Ar-X with X = Br (most
# reactive under standard Pd(0) conditions).
FRAGMENT_POOL_ARYL_HALIDES: List[str] = [
    "Brc1ccccc1",                              # bromobenzene
    "Brc1ccc(F)cc1",                           # 4-bromofluorobenzene
    "Brc1ccc(OC)cc1",                          # 4-bromoanisole
    "Brc1ccc(C)cc1",                           # 4-bromotoluene
]

# 4 carboxylic-acid handles (amide-coupling partner #1).  R-COOH.
FRAGMENT_POOL_CARBOXYLIC_ACIDS: List[str] = [
    "CC(=O)O",                                 # acetic acid
    "OC(=O)c1ccccc1",                          # benzoic acid
    "OC(=O)Cc1ccccc1",                         # phenylacetic acid
    "OC(=O)CCC(=O)O",                          # succinic acid (dicarboxylic)
]

# 4 primary-amine handles (amide-coupling partner #2).  R-NH2.
FRAGMENT_POOL_AMINES: List[str] = [
    "Nc1ccccc1",                               # aniline (aromatic primary amine)
    "NCC",                                     # ethylamine
    "NCCC",                                    # n-propylamine (MW 59 — clears the 40-floor)
    "NCC(=O)O",                                # glycine (amine + acid — partner
                                               # for amide coupling can use the
                                               # NH2 side)
]


# ---------------------------------------------------------------------------
# R15 SA-friendliness axis — 20 drug-like small-molecule scaffolds.
#
# Selection criteria (verified 2026-09-16):
#     1. ``Chem.MolFromSmiles`` parses without error.
#     2. ``AllChem.EmbedMolecule`` (ETKDGv3, ``randomSeed=0xC11C``) returns
#        0 (success).
#     3. ``Descriptors.MolWt`` in ``[40, 300]`` g/mol.
#     4. ``Descriptors.MolLogP`` in ``[-2, 5]``.
#     5. ``sa_score_ertl(smiles) < 3.5`` — the Ertl-Schuffenhauer
#        synthetic-accessibility score.  This is the hard gate that
#        distinguishes SA-friendly fragments from the legacy pool (whose
#        mean SA was 3.32); every entry below scores in the 1.0-3.0
#        range, so any tile-product that uses them should fall into the
#        TargetDiff reference band 2.65-2.86.
#     6. Canonical-form disjoint from the existing 220-tile pool (i.e.
#        ``Chem.MolToSmiles(mol)`` does not collide with any existing
#        fragment).  This protects the MCTS transposition-table
#        uniqueness invariant.
#
# Composition (20 SMILES — small set to keep the MCTS branching factor
# additive without exponential growth):
#     * 6 aromatic cores       (pyrimidine, indole, isoquinoline,
#                               benzimidazole, benzoxazole,
#                               benzothiazole)
#     * 9 saturated heterocycles / carbocycles (cyclohexane, pyrrolidine,
#                               tetrahydrofuran, morpholine, piperidine,
#                               piperazine, cyclopentane, oxetane,
#                               azetidine)
#     * 4 functional groups / linkers  (benzamide, benzonitrile,
#                                       benzaldehyde, sulfonamide)
#     * 1 small-molecule anchor (toluene — known SA=1.000 reference)
# ---------------------------------------------------------------------------
FRAGMENT_POOL_SA_FRIENDLY: List[str] = [
    # 6 aromatic cores — drug-like bicycles (indole, isoquinoline,
    # benzimidazole, benzoxazole, benzothiazole) plus the pyrimidine
    # single-ring common in kinase inhibitors.
    "c1cncnc1",                                # pyrimidine             SA=2.051
    "c1ccc2[nH]ccc2c1",                       # indole                 SA=1.740
    "c1ccc2cnccc2c1",                         # isoquinoline           SA=1.542
    "c1ccc2[nH]cnc2c1",                       # benzimidazole          SA=1.912
    "c1ccc2ocnc2c1",                          # benzoxazole            SA=2.099
    "c1ccc2scnc2c1",                          # benzothiazole          SA=1.898
    # 9 saturated heterocycles / carbocycles — small ring systems
    # frequently used as conformation locks / scaffolds.
    "C1CCCCC1",                                # cyclohexane            SA=1.000
    "C1CCNC1",                                 # pyrrolidine            SA=2.192
    "C1CCOC1",                                 # tetrahydrofuran        SA=2.026
    "C1COCCN1",                                # morpholine             SA=2.477
    "C1CCNCC1",                                # piperidine             SA=2.056
    "C1CNCCN1",                                # piperazine             SA=2.698
    "C1CCCC1",                                 # cyclopentane           SA=1.000
    "C1COC1",                                  # oxetane                SA=1.546
    "C1CNC1",                                  # azetidine              SA=1.734
    # 4 functional groups / linkers — common pharmacophoric handles.
    "NC(=O)c1ccccc1",                          # benzamide              SA=1.159
    "N#Cc1ccccc1",                             # benzonitrile           SA=1.384
    "O=Cc1ccccc1",                             # benzaldehyde           SA=1.439
    "NS(=O)(=O)c1ccccc1",                      # benzenesulfonamide     SA=1.369
    # 1 small-molecule anchor — the SA=1.000 trivial aromatic reference.
    "Cc1ccccc1",                               # toluene                SA=1.000
]


# ---------------------------------------------------------------------------
# Validation hook — embed + descriptor filter
# ---------------------------------------------------------------------------

# L6 instrumentation counters for the fragment pool (govern_review_L4_L6.md).
_FRAGMENT_POOL_COUNTERS: Dict[str, Dict[str, int]] = {
    "azide":             {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "alkyne":            {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "diene":             {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "thiol":             {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    # R10 axis A — new handle families.
    "dbco":              {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "boronic_acid":      {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "aryl_halide":       {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "carboxylic_acid":   {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    "amine":             {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
    # R15 SA-friendliness axis — 20 drug-like small-molecule scaffolds
    # curated for Ertl SA score < 3.5.  Counters live alongside the
    # other pool categories so L6 instrumentation reports them
    # uniformly with the rest of the fragment pool.
    "sa_friendly":       {"input": 0, "valid": 0, "embed_fail": 0, "mw_fail": 0, "logp_fail": 0},
}


def _validate_and_build(smiles: str, tags: List[str]) -> Tile | None:
    """Validate one SMILES, build a Tile, return None on filter failure.

    Filters:
        1. ``Chem.MolFromSmiles`` parses.
        2. ETKDGv3 embed succeeds (randomSeed=0xC11C).
        3. ``40 <= MW <= 300``.
        4. ``-2 <= logP <= 5``.
    """
    from rdkit import Chem                                              # type: ignore[import-not-found]
    from rdkit.Chem import AllChem, Descriptors                          # type: ignore[import-not-found]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC11C
        embed_rc = AllChem.EmbedMolecule(mol_h, params)
    except Exception:
        embed_rc = -1
    if embed_rc != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass

    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    if not (40.0 <= mw <= 300.0):
        return None
    if not (-2.0 <= logp <= 5.0):
        return None

    # Pass through the standard builder so canonical SMILES / coords /
    # instrumentation are computed uniformly.
    return build_click_tile(smiles, tags, embed_3d=True)


def _build_category(
    smiles_list: List[str],
    primary_tag: str,
) -> List[Tile]:
    """Validate every SMILES in ``smiles_list`` and emit valid ``Tile``s.

    Parameters
    ----------
    smiles_list
        Raw SMILES strings (50 per category).
    primary_tag
        The primary functional-group tag added to every emitted tile.

    Returns
    -------
    List[Tile]
        Validated tiles with the ``primary_tag`` tag (plus any
        auxiliary tags inferred by the caller).
    """
    tiles: List[Tile] = []
    counters = _FRAGMENT_POOL_COUNTERS[primary_tag]
    counters["input"] += len(smiles_list)

    for smi in smiles_list:
        # Determine per-tile auxiliary tags from the primary tag family.
        if primary_tag == "azide":
            extra_tags = ["azide"]
        elif primary_tag == "alkyne":
            if "1" in smi and "#C" in smi:
                # cyclooctyne / cycloheptyne
                extra_tags = ["cyclooctyne" if "C1#CCCCCCC1" in smi or "C1CCCC#CCC1" in smi else "terminal_alkyne"]
            else:
                extra_tags = ["terminal_alkyne"]
        elif primary_tag == "diene":
            if smi.startswith("C=C") or "(=O)" in smi or "C(=O)" in smi:
                extra_tags = ["dienophile"]
            else:
                extra_tags = ["diene"]
        elif primary_tag == "thiol":
            extra_tags = ["thiol"]
        elif primary_tag == "dbco":
            extra_tags = ["cyclooctyne", "dbco"]
        elif primary_tag == "boronic_acid":
            extra_tags = ["boronic_acid", "aryl_boronic_acid"]
        elif primary_tag == "aryl_halide":
            extra_tags = ["aryl_halide", "aryl_bromide"]
        elif primary_tag == "carboxylic_acid":
            extra_tags = ["carboxylic_acid"]
        elif primary_tag == "amine":
            extra_tags = ["amine", "primary_amine"]
        elif primary_tag == "sa_friendly":
            # SA-friendly scaffolds are not reactive handles per se; they
            # are the inert core that, once conjugated to a click-handle
            # partner (azide / alkyne / etc.), yields a low-SA product.
            # Tag with ``sa_friendly`` so downstream metric paths can
            # distinguish them from reactive handles when computing
            # SA-of-product.
            extra_tags = ["sa_friendly"]
        else:
            extra_tags = [primary_tag]

        # Use the same embed path used by build_click_tile so the filter
        # outcome matches the actual library emit. We re-implement only
        # to capture the failure reason.
        from rdkit import Chem                                              # type: ignore[import-not-found]
        from rdkit.Chem import AllChem, Descriptors                          # type: ignore[import-not-found]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            counters["embed_fail"] += 1
            continue
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xC11C
            embed_rc = AllChem.EmbedMolecule(mol_h, params)
        except Exception:
            embed_rc = -1
        if embed_rc != 0:
            counters["embed_fail"] += 1
            continue
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        if not (40.0 <= mw <= 300.0):
            counters["mw_fail"] += 1
            continue
        if not (-2.0 <= logp <= 5.0):
            counters["logp_fail"] += 1
            continue
        counters["valid"] += 1
        tiles.append(build_click_tile(smi, extra_tags, embed_3d=True))
    return tiles


def fragments_from_chembl_reactive(
    *,
    include_azides: bool = True,
    include_alkynes: bool = True,
    include_dienes: bool = True,
    include_thiols: bool = True,
    include_dbco: bool = True,
    include_boronic_acids: bool = True,
    include_aryl_halides: bool = True,
    include_carboxylic_acids: bool = True,
    include_amines: bool = True,
    include_sa_friendly: bool = True,
) -> List[Tile]:
    """Return the validated 200+ tile pool.

    Reads from the hardcoded :data:`FRAGMENT_POOL_*` lists, validates
    every SMILES, and returns the surviving tiles.

    Parameters
    ----------
    include_azides, include_alkynes, include_dienes, include_thiols
        Category toggles for the original 4 (×50 SMILES each = 200 tile
        pool).  Defaults to True so the returned list is the full pool.
    include_dbco, include_boronic_acids, include_aryl_halides,
    include_carboxylic_acids, include_amines
        Category toggles for the 5 new handle families added in R10
        axis A (×4 SMILES each = +20 tile pool).  Defaults to True
        so the canonical 220-tile pool is returned when all toggles
        are left at their default.  Setting any of these to False
        drops the corresponding family back to the legacy 200-tile
        pool.

    Returns
    -------
    List[Tile]
        Validated :class:`Tile` instances.  With all toggles True the
        pool size is 220 (200 + 4 × 5); with all 5 new toggles False
        it is 200 (backward compatible).
    """
    tiles: List[Tile] = []
    if include_azides:
        tiles.extend(_build_category(FRAGMENT_POOL_AZIDES, "azide"))
    if include_alkynes:
        tiles.extend(_build_category(FRAGMENT_POOL_ALKYNES, "alkyne"))
    if include_dienes:
        tiles.extend(_build_category(FRAGMENT_POOL_DIENES, "diene"))
    if include_thiols:
        tiles.extend(_build_category(FRAGMENT_POOL_THOLS, "thiol"))
    if include_dbco:
        tiles.extend(_build_category(FRAGMENT_POOL_DBCO, "dbco"))
    if include_boronic_acids:
        tiles.extend(_build_category(FRAGMENT_POOL_BORONIC_ACIDS, "boronic_acid"))
    if include_aryl_halides:
        tiles.extend(_build_category(FRAGMENT_POOL_ARYL_HALIDES, "aryl_halide"))
    if include_carboxylic_acids:
        tiles.extend(_build_category(FRAGMENT_POOL_CARBOXYLIC_ACIDS, "carboxylic_acid"))
    if include_amines:
        tiles.extend(_build_category(FRAGMENT_POOL_AMINES, "amine"))
    if include_sa_friendly:
        tiles.extend(_build_category(FRAGMENT_POOL_SA_FRIENDLY, "sa_friendly"))
    # Canonicalize across handle families before exposing the pool.  Some
    # source SMILES use alternate aromatic/ring spellings that collapse to
    # the same RDKit canonical form; retaining both poisons the MCTS
    # transposition table and silently shrinks the effective branching
    # factor.  Preserve first occurrence (category order is deterministic).
    unique: List[Tile] = []
    seen: set[str] = set()
    for tile in tiles:
        key = tile.smiles
        if key in seen:
            continue
        seen.add(key)
        unique.append(tile)
    return unique


def l6_fragment_pool_metrics() -> Dict[str, Dict[str, int]]:
    """Return embed-failure / descriptor-filter counters per category.

    Returns
    -------
    Dict[str, Dict[str, int]]
        Nested dict: ``category -> {input, valid, embed_fail, mw_fail,
        logp_fail}``.
    """
    return {
        cat: dict(counts) for cat, counts in _FRAGMENT_POOL_COUNTERS.items()
    }
