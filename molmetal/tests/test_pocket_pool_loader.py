"""Track B regression test — pocket-pool loader heterogeneity.

Per the design (Track B Gap 1 / 4) the loader must attach a
``pocket_id`` per-row so the per-cloud PocketCloud draw produces
genuinely heterogeneous pockets (not one pocket shared across the
whole cohort).  Two regression cases:

1. ``test_pool_path_none_is_legacy`` — ``load_tmqm(pocket_pool_path=None)``
   yields a frame with **no** ``pocket_id`` column (legacy behaviour).
2. ``test_pool_draw_is_heterogeneous`` — when ``pocket_pool_path``
   points at a directory of stub PDBs, the resulting ``pocket_id``
   column contains **at least two distinct ids** across the cohort
   (per-cloud heterogeneity).
3. ``test_pool_size_cap_holds`` — ``pocket_pool_size`` truncates the
   pool entries consulted.

The stub PDBs do not require RDKit — we only assert the loader's
filesystem + DataFrame contract.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.tmqm import _attach_pocket_id_column  # noqa: E402


_STUB_PDB = (
    "HEADER    STUB                                                  \n"
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
    "END                                                               \n"
)


@pytest.fixture
def pool_dir(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    for i in range(8):
        (d / f"stub_{i:03d}.pdb").write_text(_STUB_PDB, encoding="utf-8")
    return d


def test_pool_path_none_is_legacy():
    """``pocket_pool_path=None`` ⇒ no ``pocket_id`` column."""
    import pandas as pd
    df = pd.DataFrame({"csd_code": ["a", "b", "c"], "smiles": ["C", "O", "N"]})
    out = _attach_pocket_id_column(df, pocket_pool_path=None, pocket_pool_size=0)
    assert "pocket_id" not in out.columns
    assert len(out) == 3


def test_pool_path_missing_dir_is_legacy():
    """``pocket_pool_path`` pointing at a non-existent directory ⇒ no column."""
    import pandas as pd
    df = pd.DataFrame({"csd_code": ["a", "b"], "smiles": ["C", "O"]})
    out = _attach_pocket_id_column(
        df, pocket_pool_path="/no/such/dir", pocket_pool_size=0,
    )
    assert "pocket_id" not in out.columns


def test_pool_path_empty_dir_is_legacy(pool_dir):
    """``pocket_pool_path`` pointing at a directory with no PDBs ⇒ legacy."""
    import pandas as pd
    empty = pool_dir.parent / "empty_pool"
    empty.mkdir()
    df = pd.DataFrame({"csd_code": ["a", "b"], "smiles": ["C", "O"]})
    out = _attach_pocket_id_column(
        df, pocket_pool_path=str(empty), pocket_pool_size=0,
    )
    assert "pocket_id" not in out.columns


def test_pool_draw_is_heterogeneous(pool_dir):
    """Pool with 8 PDBs ⇒ ``pocket_id`` column with ≥ 2 distinct values."""
    import pandas as pd
    df = pd.DataFrame({
        "csd_code": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "smiles": ["C", "O", "N", "S", "P", "F", "Cl", "Br"],
    })
    out = _attach_pocket_id_column(
        df, pocket_pool_path=str(pool_dir), pocket_pool_size=0,
    )
    assert "pocket_id" in out.columns
    assert out["pocket_id"].nunique() >= 2


def test_pool_size_cap_holds(pool_dir):
    """``pocket_pool_size=2`` ⇒ only the first 2 pool entries are consulted."""
    import pandas as pd
    df = pd.DataFrame({
        "csd_code": [f"csd_{i:04d}" for i in range(64)],
        "smiles": ["C"] * 64,
    })
    out = _attach_pocket_id_column(
        df, pocket_pool_path=str(pool_dir), pocket_pool_size=2,
    )
    assert out["pocket_id"].nunique() <= 2


def test_pool_draw_is_deterministic(pool_dir):
    """Same ``csd_code`` ⇒ same ``pocket_id`` across two runs."""
    import pandas as pd
    df = pd.DataFrame({"csd_code": ["a", "b", "c"], "smiles": ["C", "O", "N"]})
    a = _attach_pocket_id_column(df, pocket_pool_path=str(pool_dir), pocket_pool_size=0)
    b = _attach_pocket_id_column(df, pocket_pool_path=str(pool_dir), pocket_pool_size=0)
    assert a["pocket_id"].tolist() == b["pocket_id"].tolist()