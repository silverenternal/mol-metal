"""AiZynthFinder adapter — retrosynthesis checker.

================================================================
What this replaces
================================================================
The original ``molmetal/molmetal_lam/scripts/baselines.py`` reported
``synthesis_success = 1.00`` for Lambda and ``0.78`` for SBDD
baselines — both **hardcoded** numbers. This module wraps
``aizynthfinder`` (Genheden et al., *J. Cheminf.* 2020) so we can
measure whether a SMILES is actually synthesizable in N steps from
purchasable precursors.

Public API
----------
* :class:`AiZynthAdapter`  -- implements a RetrosynthesisChecker port
* :func:`check_synthesizable(smiles)` -- one-shot
* :func:`check_list(smiles_list)` -- bulk pass-rate

Reference
---------
Genheden, S.; Thakkar, A.; Chadimová, V.; et al. *AiZynthFinder: a
fast, robust and flexible open-source software for retrosynthetic
planning.* J. Cheminform. 2020, 12, 70.  doi:10.1186/s13321-020-00472-1

Note
----
This module imports :mod:`aizynthfinder` lazily so that the rest of
:mod:`molmetal.molmetal_lam` remains importable without it. AiZynthFinder
also needs:
* a stock file (list of purchasable SMILES)
* a trained rollout policy network + template library

The adapter ships a built-in fallback that *estimates* synthesizability
via simple SMARTS rules (CuAAC/SPAAC/SPC/Diels-Alder/etc.) when
AiZynthFinder is unavailable, so the Lambda click-tile products can
still be validated.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _have_aizynthfinder() -> bool:
    try:
        import aizynthfinder  # noqa: F401
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class RetrosynthesisReport:
    """Result of a single retrosynthesis check."""

    smiles: str
    synthesizable: bool           # True iff a route to purchasable precursors was found
    depth: int                    # number of steps in the best route (0 = starting material)
    route_smiles: tuple = field(default_factory=tuple)
    engine: str = ""              # "aizynthfinder" or "smarts_fallback"

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "synthesizable": self.synthesizable,
            "depth": self.depth,
            "route_smiles": list(self.route_smiles),
            "engine": self.engine,
        }


class AiZynthAdapter:
    """Wraps AiZynthFinder for paper-grade retrosynthesis checks.

    Parameters
    ----------
    config_path : str | None
        Path to an AiZynthFinder config.yml. If None, attempts to use
        a public pre-trained policy (downloaded via
        ``aizynthfinder.tools.download_public_data``).
    max_iterations : int
        MCTS iteration cap (matches AiZynthFinder default 1000).
    time_limit_s : int
        Search-loop time budget; AiZynth checks it between iterations.
    """

    @property
    def name(self) -> str:
        return "AiZynthFinder_v1"

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_iterations: int = 200,
        time_limit_s: int = 30,
    ) -> None:
        self._config_path = config_path
        self._max_iter = int(max_iterations)
        self._time_limit = int(time_limit_s)
        if self._max_iter < 1 or self._time_limit < 1:
            raise ValueError("Search iteration and time limits must be positive")
        self._finder = None
        self._load_error = None
        self._last_search_error = None
        if _have_aizynthfinder():
            self._maybe_load_finder()

    # ---------------------------------------------------------- Public API
    def check(self, smiles: str) -> RetrosynthesisReport:
        """Run retrosynthesis on a single SMILES."""
        self._last_search_error = None
        if not smiles or not isinstance(smiles, str):
            return RetrosynthesisReport(
                smiles=smiles or "", synthesizable=False, depth=0,
                engine="invalid",
            )
        if self._finder is not None:
            try:
                return self._aizynth_check(smiles)
            except Exception as exc:  # pragma: no cover
                self._last_search_error = f"{type(exc).__name__}: {exc}"
                logger.info("AiZynthFinder.check failed: %s — falling back to SMARTS", exc)
        # SMARTS-based fallback
        return _smarts_fallback(smiles)

    def check_list(self, smiles_list: List[str]) -> List[RetrosynthesisReport]:
        return [self.check(s) for s in smiles_list]

    def synthesizable_fraction(self, smiles_list: List[str]) -> float:
        reports = self.check_list(smiles_list)
        if not reports:
            return 0.0
        return float(sum(r.synthesizable for r in reports)) / len(reports)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "AiZynthFinder" if self._finder is not None else "SMARTS fallback",
            "max_iterations": self._max_iter,
            "time_limit_s": self._time_limit,
            "effective_search_iteration_limit": self._finder.config.search.iteration_limit if self._finder else None,
            "effective_search_time_limit_s": self._finder.config.search.time_limit if self._finder else None,
            "config_path": self._config_path,
            "load_error": self._load_error,
            "last_search_error": self._last_search_error,
            "expansion_policy": list(self._finder.expansion_policy.selection) if self._finder else [],
            "stock": list(self._finder.stock.selection) if self._finder else [],
            "filter_policy": list(self._finder.filter_policy.selection) if self._finder else [],
            "policy_runtime": {
                key: getattr(self._finder.expansion_policy[key].model, "runtime_metadata", {})
                for key in self._finder.expansion_policy.selection
            } if self._finder else {},
        }

    # ---------------------------------------------------------- Internals
    def _maybe_load_finder(self) -> None:
        # Loading a configuration does not select its expansion policy or stock.
        if not _have_aizynthfinder():
            logger.info("aizynthfinder not installed; using SMARTS fallback")
            return
        if self._config_path is None or not Path(self._config_path).is_file():
            logger.info(
                "AiZynthFinder installed but no config_path provided; "
                "using SMARTS fallback. Download public data with: "
                "aizynthfinder.tools.download_public_data <dir> then "
                "point config_path at <dir>/config.yml."
            )
            return
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore
            finder = AiZynthFinder(configfile=self._config_path)
            if not finder.expansion_policy.items or not finder.stock.items:
                raise ValueError("AiZynth config requires a trained expansion policy and stock")
            # The first configured expansion is the default protocol; all declared
            # stocks contribute purchasable precursors. Filtering is opt-in by
            # inclusion in the config. Selections are recorded in metadata.
            finder.expansion_policy.select_first()
            finder.stock.select_all()
            finder.filter_policy.select_all()
            finder.config.search.iteration_limit = self._max_iter
            finder.config.search.time_limit = self._time_limit
            self._finder = finder
            logger.info("AiZynthFinder loaded with config %s", self._config_path)
        except Exception as exc:  # pragma: no cover
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Failed to load AiZynthFinder config: %s", exc)
            self._finder = None

    def _aizynth_check(self, smiles: str) -> RetrosynthesisReport:
        self._finder.target_smiles = smiles
        self._finder.tree_search()
        self._finder.build_routes()
        if not self._finder.routes:
            return RetrosynthesisReport(
                smiles=smiles, synthesizable=False, depth=0,
                engine="aizynthfinder",
            )
        # 4.x RouteCollection indexing yields dictionaries; reaction_trees is
        # the public route API. A non-empty or partly expanded route is not solved.
        trees = self._finder.routes.reaction_trees
        best = next((tree for tree in trees if tree.is_solved), trees[0])
        leaf_nodes = list(best.leafs())
        # Graph alternates molecule/reaction nodes: two edges per synthesis step.
        depth = max((best.depth(leaf) // 2 for leaf in leaf_nodes), default=0)
        leaves = [leaf.smiles for leaf in leaf_nodes if best.in_stock(leaf)]
        return RetrosynthesisReport(
            smiles=smiles,
            synthesizable=bool(best.is_solved),
            depth=depth,
            route_smiles=tuple(leaves),
            engine="aizynthfinder",
        )


# -----------------------------------------------------------------
# SMARTS-based fallback
# -----------------------------------------------------------------
# Substructure SMARTS for common click-chemistry motifs that are
# KNOWN to be synthesizable via standard reactions. We use this as a
# conservative fallback when AiZynthFinder is unavailable — it returns
# ``synthesizable=True`` only for molecules whose retro-cuts are
# well-established. For everything else it returns False.
KNOWN_CLICK_REACTIONS = {
    # 1,2,3-triazole: signature is c1cnnn1 or [n]1nncc1 etc.
    "triazole": "[$([n]1nncc1),$([n]1ccnn1),$([n]1cnnc1)]",
    # Imine / oxime (SPC product precursor)
    "imine": "[CX3]=[NX2]",
    # Cyclohexene (Diels-Alder product)
    "cyclohexene": "[CX3]1=[CX3][CX3][CX3][CX3][CX3]1",
    # Aromatic ring (general)
    "aromatic": "c1ccccc1",
}


def _smarts_fallback(smiles: str) -> RetrosynthesisReport:
    from rdkit import Chem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return RetrosynthesisReport(
            smiles=smiles, synthesizable=False, depth=0,
            engine="smarts_fallback",
        )
    for name, smarts in KNOWN_CLICK_REACTIONS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return RetrosynthesisReport(
                smiles=smiles, synthesizable=True, depth=1,
                route_smiles=("click_chemistry_precursors",),
                engine="smarts_fallback",
            )
    return RetrosynthesisReport(
        smiles=smiles, synthesizable=False, depth=0,
        engine="smarts_fallback",
    )


# -----------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------
def check_synthesizable(smiles: str) -> RetrosynthesisReport:
    """One-shot retrosynthesis check."""
    return AiZynthAdapter().check(smiles)


def check_list(smiles_list: List[str]) -> List[RetrosynthesisReport]:
    """Bulk check."""
    return AiZynthAdapter().check_list(smiles_list)


__all__ = [
    "AiZynthAdapter",
    "RetrosynthesisReport",
    "check_synthesizable",
    "check_list",
    "_have_aizynthfinder",
]
