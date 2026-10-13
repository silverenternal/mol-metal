"""Eval-time discriminators (Wf-A11 / eval wave).

Four lightweight decoders that run over every generated mol the CFM trainer
emits at eval time, BEFORE the bond-head logits are thresholded into bond
classes.  Together they close the gap where a decode can look RDKit-valid
while violating metallo-drug priors:

  * :class:`TopologyDiscriminator`  — single-fragment rate + ring-size
    histogram + largest-fragment atom count.
  * :class:`ChemistryDiscriminator` — dative-bond rate + metal-coord
    distribution + metal-coord-1 rate + aromatic-atom count.
  * :class:`MetalAwareADMET`       — 8 ADMET proxies extended to accept
    a metal context (oxidation state, coord number); backward-compat
    None = organic-only baseline.
  * :class:`ModelOutputAlignment`  — bond-logits top-1 correctness +
    bond-logits KL + EGNN hidden-state norm + decoded-vs-real alignment.

Each discriminator returns a flat ``Dict[str, float]`` so the trainer
can dump it as a JSONL row alongside per-sample files.

Backward-compat
---------------
* When no discriminator is wired in (legacy Wf-13 path), the trainer
  emits ``{}`` and the eval-block sees no change.
* ``MetalAwareADMET.forward(mol, ..., oxidation_state=None, ...)``
  preserves the organic-only behaviour of the legacy 8 proxies in
  :file:`molmetal/molmetal_lam/sbdd_env/metrics_v2.py`.

Lit anchors
-----------
* Polykov 2020 ("Functional-Groups-Based Pharmacophore Generation" Tool
  scripts) — metal-coord-1 as a flag for under-coordinated Pt/Ru.
* Rusanio 2025 PlatinAI (J. Med. Chem.) — single-fragment rate as a
  minimum-suitability gate.
"""

from __future__ import annotations

from molmetal.eval.discriminators.chemistry import ChemistryDiscriminator
from molmetal.eval.discriminators.metal_aware_admet import MetalAwareADMET
from molmetal.eval.discriminators.model_output_alignment import ModelOutputAlignment
from molmetal.eval.discriminators.topology import TopologyDiscriminator

__all__ = [
    "TopologyDiscriminator",
    "ChemistryDiscriminator",
    "MetalAwareADMET",
    "ModelOutputAlignment",
]