#!/usr/bin/env python3
"""Size-adaptive D-MPNN width policy for within-metal training.

=====================================================================
Why this exists
=====================================================================
#184 measured the capacity effect of ``hidden=128 -> 512`` on the Table-1
within-metal column, device-matched (both legs ``cuda:0`` + ``--deterministic``,
``model.hidden`` the only free variable, 5 split seeds, folds verified identical):

    metal  n      capacity effect (h512 - h128)   seeds positive   p_Holm
    Ru    4793   +0.0183                          5/5              0.0042
    Ir    1281   +0.0828                          5/5              0.0017
    Rh     334   -0.0027                          1/5              1.0000
    Os     333   -0.0029                          2/5              1.0000
    Re     250   +0.0114                          4/5              0.1139

Source: ``metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json``.

The two cohorts above n=1000 gain significantly; the three below n=334 do not,
and their point estimates straddle zero. So a uniform width choice is wrong in
both directions: 512 wastes compute (and adds seed variance) on small cohorts,
128 leaves 43-48% of the Ru/Ir gap on the table.

This module turns that measured pattern into an explicit, auditable decision so
nobody has to remember it. It is a POLICY, not a law: the cut sits in the empty
interval between 334 and 1281, and it was chosen after seeing these five points.
It has not been validated on a held-out cohort.

Usage
-----
::

    from molmetal.width_policy import select_hidden
    hidden, decision = select_hidden(n_unique=4793, policy="size-adaptive")

The trainer exposes this as ``--width-policy {fixed,size-adaptive}``; the default
is ``fixed`` so every already-published column stays reproducible from its
config unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Cohorts at or above this size get the wide model. The cut is measured, not
#: tuned: it lies in the empty interval between the largest null cohort (334)
#: and the smallest significant one (1281).
MIN_N_FOR_WIDE = 1000

WIDE_HIDDEN = 512
NARROW_HIDDEN = 128

POLICIES = ("fixed", "size-adaptive")


def select_hidden(
    n_unique: int,
    *,
    policy: str = "fixed",
    fixed_hidden: int = NARROW_HIDDEN,
) -> Tuple[int, Dict[str, object]]:
    """Return ``(hidden, decision)`` for one metal's cohort.

    ``fixed`` returns ``fixed_hidden`` unchanged. ``size-adaptive`` returns
    ``WIDE_HIDDEN`` when ``n_unique >= MIN_N_FOR_WIDE`` and ``NARROW_HIDDEN``
    otherwise. The decision dict is meant to be written into the run artifact so
    the width actually used is never implicit.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown width policy {policy!r}; expected one of {POLICIES}")
    n = int(n_unique)

    if policy == "fixed":
        return int(fixed_hidden), {
            "policy": "fixed",
            "n_unique": n,
            "hidden": int(fixed_hidden),
        }

    hidden = WIDE_HIDDEN if n >= MIN_N_FOR_WIDE else NARROW_HIDDEN
    return hidden, {
        "policy": "size-adaptive",
        "rule": f"hidden={WIDE_HIDDEN} iff n_unique >= {MIN_N_FOR_WIDE}, else {NARROW_HIDDEN}",
        "n_unique": n,
        "hidden": hidden,
        "basis": "metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json",
        "validated_on": ["Ru", "Ir", "Rh", "Os", "Re"],
        "caveat": (
            "Post-hoc cut on 5 metals, not a validated design rule. Both cohorts above the "
            "cut gained significantly (Holm 0.0042 / 0.0017); the three below it did not "
            "(-0.0027 / -0.0029 / +0.0114 ns)."
        ),
    }
