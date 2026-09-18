"""tile_lib — Click-chemistry tile library for the LamClick framework.

Layer 9 of the MLC architecture (see ``molmetal_lam/__init__.py``):
the constant pool used by the proof-search to construct drug candidates
out of small reactive fragments.

Public API
----------
* :class:`Tile`             — frozen dataclass for a single tile
* :func:`build_tile_library` — Phase-0 12-tile library
* :func:`build_tile_dict`    — same library as ``{tile_id: Tile}``
* :data:`AZIDES`             — list of 4 azide tiles
* :data:`ALKYNES`            — list of 4 alkyne tiles
* :data:`PARTNERS`           — list of 4 partner tiles
* :data:`STANDARD_12`        — the full 12-tile library
"""

from molmetal_lam.tile_lib.tile import Tile
from molmetal_lam.tile_lib.click_tiles import (
    ALKYNE_TILES,
    ALL_CLICK_HANDLES,
    AZIDE_PARTNER_TILES,
    AZIDE_TILES,
    BORONIC_PARTNER_TILES,
    BROMIDE_PARTNER_TILES,
    PARTNER_TILES,
    PARTNER_TILES_V2,
    STANDARD_12_TILES,
    build_all_click_tiles,
    build_click_tile,
)
from molmetal_lam.tile_lib.library import (
    ALKYNES,
    AZIDES,
    PARTNERS,
    STANDARD_12,
    build_tile_dict,
    build_tile_library,
)

__all__ = [
    "Tile",
    "build_tile_library",
    "build_tile_dict",
    "AZIDES",
    "ALKYNES",
    "PARTNERS",
    "STANDARD_12",
    # Functions
    "build_click_tile",
    "build_all_click_tiles",
    "AZIDE_TILES",
    "ALKYNE_TILES",
    "PARTNER_TILES",
    "STANDARD_12_TILES",
    # V2 partner tiles (WF-Partner-Tiles-PathA)
    "PARTNER_TILES_V2",
    "AZIDE_PARTNER_TILES",
    "BORONIC_PARTNER_TILES",
    "BROMIDE_PARTNER_TILES",
    "ALL_CLICK_HANDLES",
]
