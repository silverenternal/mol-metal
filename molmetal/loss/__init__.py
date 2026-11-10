"""molmetal/loss — modular loss-layer package.

Per Wf-A13 (2026-10-13) — this package replaces inline loss-term blocks in
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` (the CFM trainer) with
configurable loss classes. The original behavior is bit-exact for every class
whose weight defaults to 0.0 (cf. AGENTS.md §3.3 audit-ledger — bit-exact-
default-pass-through; ~~AGENTS.md §5.11 backward-compat rule~~ was the
superseded pointer, corrected 2026-10-31).

Layout rationale
-----------------
  * :mod:`chemistry_losses` — ContinuousBOLoss / MoleculeLoss / DiversityLoss /
    QEDLoss / DiversityRegularizer. The last wraps the existing inline
    diversity_term block (3142-3269 in the trainer) so its many sub-flags
    (mmd_weight / repulsion_weight / scaffold_conditioning / top_p /
    top_k / temperature) keep their bit-exact pre-Wf-A13 semantics.
  * :mod:`aggregation_losses` — AssembleTerm (arithmetic_mean formulation,
    L6/L-A7 fix) + SoftValenceTerm (decoupled from training_mode per D-NEW-3).
  * :mod:`distillation_losses` — ShortcutDistillLoss placeholder (KL student
    vs teacher; teacher_logits=None ⇒ returns zero).
  * :mod:`regularization_losses` — EntropyTerm / ConsistencyTerm / SO3Term /
    EDMTerm placeholders that close 4 of L8's 8 DEAD terms.
  * :mod:`prior_losses` — ClashPenalty standalone loss version of the model-
    side patch. Wired independently so ``loss.clash_penalty_weight_override``
    can dial loss-side weight separately from model-side.

D-NEW-1 name-collision note
---------------------------
This ``__init__`` does NOT re-export a top-level ``sa_loss`` symbol. The
trainer's local ``sa_loss`` (line 3287 of the v2 trainer) is renamed to
``_sa_local`` at the Wf-A13 edit site, but a NEW loss class called
``sa_loss`` would still shadow the trainer's local if imported into the
``train_one_epoch`` namespace. Any future sa-aware loss class must therefore
live under a different name (e.g. ``SAMoleculePenalty``).

Backward-compat
---------------
Every class returns ``torch.zeros((), device=device)`` when its weight is 0
or its ``enabled`` flag is False. The 7 shipped CFM YAMLs do not declare
any of the new ``loss.*`` keys, so the 12-term sum is bit-exact identical to
pre-Wf-A13: only the 5 LIVE terms aggregate, and each new class's
``.compute`` is an exact zero on the 0-identity path.
"""

from molmetal.loss.aggregation_losses import AssembleTerm, SoftValenceTerm
from molmetal.loss.chemistry_losses import (
    ContinuousBOLoss,
    DiversityLoss,
    DiversityRegularizer,
    MoleculeLoss,
    NaturalChargeLoss,
    QEDLoss,
)
from molmetal.loss.distillation_losses import ShortcutDistillLoss
from molmetal.loss.fragment_losses import SingleFragmentLoss
from molmetal.loss.ring_template import RingTemplatePrior, RingTemplatePriorConfig
from molmetal.loss.prior_losses import ClashPenalty
from molmetal.loss.regularization_losses import (
    ConsistencyTerm,
    EDMTerm,
    EnergyConservation,
    EntropyTerm,
    FmOdeTerm,
    SBOO3Term,
    SO3Term,
)
from molmetal.loss.property_losses import MolecularPropertyTargetLoss
from molmetal.loss.stererochemistry_losses import (
    FragmentConnectivityLoss,
    StereoCentreSuppressionLoss,
)

__all__ = [
    # chemistry_losses.py
    "ContinuousBOLoss",
    "DiversityLoss",
    "DiversityRegularizer",
    "MoleculeLoss",
    "NaturalChargeLoss",
    "QEDLoss",
    # aggregation_losses.py
    "AssembleTerm",
    "SoftValenceTerm",
    # distillation_losses.py
    "ShortcutDistillLoss",
    # fragment_losses.py (Wf-A14, A.6 fix)
    "SingleFragmentLoss",
    # ring_template.py (Wf-A78 / OPT-5)
    "RingTemplatePrior",
    "RingTemplatePriorConfig",
    # prior_losses.py
    "ClashPenalty",
    # property_losses.py (Wf-A14, A.4 fix)
    "MolecularPropertyTargetLoss",
    # regularization_losses.py
    "ConsistencyTerm",
    "EDMTerm",
    "EnergyConservation",
    "EntropyTerm",
    "FmOdeTerm",
    "SBOO3Term",
    "SO3Term",
    # stererochemistry_losses.py (Wf-A14, A.3 fix)
    "FragmentConnectivityLoss",
    "StereoCentreSuppressionLoss",
]
