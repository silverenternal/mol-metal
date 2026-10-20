"""Mol-Metal eval package.

Wf-A30-1 (2026-10-14) A7 — exposes :func:`ckpt_is_smoke_stub`, the
canonical helper for refusing the smoke-stub ``.pt`` artefacts that
the Wf-124 P3 / Wf-299 3-wave CPU smoke produced.  Reading one of
those checkpoints through a real training or decoding path silently
produces pass-through output that looks like a successful
measurement but carries no trained signal — the helper exists so a
single ``is True / is False`` call covers all eval modules
(:mod:`cross_dataset`, :mod:`linear_probe`, future additions) with
one definition.

Wf-A35-1 (2026-10-14) C3 — also exposes :func:`bond_decision_settings`,
the single read-site for the ``(bond_decision, bond_decision_threshold,
class_logit_bias)`` triplet. The metrics side
(:func:`molmetal.scripts.train_cfm_yuelbond_wasserstein.evaluate_bond_accuracy`)
and the decoder side
(:class:`molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond.YuelBondBondDecoder`)
both read through this helper so a flag rename is one diff.
"""

from __future__ import annotations

from pathlib import Path

from molmetal.eval.bond_decision import bond_decision_settings
from molmetal.eval.bond_metric_schema import (
    BOND_AP_EXISTENCE,
    BOND_AP_NO_BOND_LEGACY,
    BOND_AP_UNDECLARED,
    MixedBondApDefinition,
    bond_ap_definition,
    bond_ap_legend,
    require_single_definition,
    stamp_counts,
)


def ckpt_is_smoke_stub(path: "Path") -> bool:
    """Return True when ``path`` is the smoke-stub ``.pt`` artefact.

    The smoke-stub convention was introduced by Wf-A30-1 (2026-10-14):
    when a checkpoint file ends in ``.smoke_stub.pt``, downstream
    readers must refuse to treat its weights as a trained model.  The
    marker is purely lexical (the suffix on the file stem) so the
    rule stays in source control rather than relying on file size or
    modification time.

    Parameters
    ----------
    path : pathlib.Path
        Candidate checkpoint path.  A non-``.pt`` file, a path with
        no recognised stem suffix, or a non-existent path all return
        ``False`` — ``ckpt_is_smoke_stub`` is purely a name check and
        does not touch the filesystem.

    Returns
    -------
    bool
        ``True`` iff ``path.suffix == ".pt"`` AND ``path.stem``
        ends in ``".smoke_stub"``.
    """
    suffix = path.suffix.lower()
    stem = path.stem
    return suffix == ".pt" and stem.endswith(".smoke_stub")


__all__ = [
    "ckpt_is_smoke_stub",
    "bond_decision_settings",
    "BOND_AP_EXISTENCE",
    "BOND_AP_NO_BOND_LEGACY",
    "BOND_AP_UNDECLARED",
    "MixedBondApDefinition",
    "bond_ap_definition",
    "bond_ap_legend",
    "require_single_definition",
    "stamp_counts",
]
