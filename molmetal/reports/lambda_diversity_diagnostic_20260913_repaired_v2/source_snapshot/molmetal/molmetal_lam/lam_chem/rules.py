"""Public-facing click-chemistry rule registry (R10 axis A).

This module re-exports the **5 canonical click-chemistry reactions** that
the MCTS expansion path consumes — one rule per row of the round-10
roadmap table:

    1. CuAAC         — copper-catalysed azide + terminal alkyne → triazole
    2. SPAAC         — strain-promoted azide + cyclooctyne → triazole
    3. thiol-ene     — photo-initiated thiol + alkene → thioether
    4. Suzuki        — Pd-catalysed aryl-boronic acid + aryl halide → biaryl
    5. amide coupling — carboxylic acid + amine → amide

The names are exposed in the *user-facing* spelling requested by the
governance brief (mix of camelCase / lower-case-with-dash / lower-case-
with-space), and each name maps to a concrete
:class:`molmetal_lam.reactions.beta_reductions.ReactionRule` instance
that the MCTS expansion path can fire via the canonical ``reduce((a, b))``
interface.

The 5 rules are a strict subset of the 7 registered in
:data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES` — the
extra two (``SPC`` and ``DielsAlder``) are kept in the lower-level
registry for backwards compatibility but are *not* part of the
public-facing R10 axis-A click set.

Each name exposes a ``ReactionRule`` singleton (instantiated once at
module import so the search hot path can grab the rule by attribute
access without paying dataclass construction cost per expansion).
"""

from __future__ import annotations

from typing import Dict

from molmetal_lam.reactions.beta_reductions import (
    AmideCoupling,
    CuAAC,
    REACTION_RULES,
    SPAAC,
    Suzuki,
    ThiolEne,
)

__all__ = [
    "CuAAC",
    "SPAAC",
    "ThiolEne",
    "Suzuki",
    "AmideCoupling",
    "CLICK_REACTIONS",
    "list_reactions",
]


#: Canonical round-10 click set — one rule per reaction class.
#: Re-exposed at module level so callers can do
#: ``from molmetal_lam.lam_chem.rules import CuAAC`` and get a ready-to-fire
#: :class:`ReactionRule` singleton.
CuAAC = REACTION_RULES["CuAAC"]
SPAAC = REACTION_RULES["SPAAC"]
ThiolEne = REACTION_RULES["ThiolEne"]
Suzuki = REACTION_RULES["Suzuki"]
AmideCoupling = REACTION_RULES["AmideCoupling"]


#: Public registry of the 5 click reactions, indexed by the *exact* names
#: listed in the R10 axis-A brief.  Lower-case / dashed / spaced spellings
#: are all supported so callers using the metrics-harness string convention
#: (``"thiol-ene"``, ``"amide coupling"``) get the same rule instance as
#: callers using the canonical chemistry name (``"ThiolEne"``,
#: ``"AmideCoupling"``).
CLICK_REACTIONS: Dict[str, object] = {
    # canonical camelCase
    "CuAAC":         CuAAC,
    "SPAAC":         SPAAC,
    "ThiolEne":      ThiolEne,
    "Suzuki":        Suzuki,
    "AmideCoupling": AmideCoupling,
    # lower-case-with-dash variant (metrics-harness convention)
    "cuaac":         CuAAC,
    "spaac":         SPAAC,
    "thiol-ene":     ThiolEne,
    "suzuki":        Suzuki,
    "amide coupling": AmideCoupling,
    # lower-case-with-space variant
    "amide":         AmideCoupling,
    "amide_coupling": AmideCoupling,
}


def list_reactions() -> list[str]:
    """Return the canonical (camelCase) name list of the 5 click reactions."""
    return ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"]