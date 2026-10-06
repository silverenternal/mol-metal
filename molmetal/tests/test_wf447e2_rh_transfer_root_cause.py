"""Wf-447-E2 — lock in the D-MPNN Rh-transfer root cause.

Root cause (MEASURED, see module docstring of the companion note):
the LOMO D-MPNN is **metal-blind**, and the Rh fold's 108 real rows are
62 % ligand-scaffold-unseen relative to the 5 352-row train pool.

These tests are *characterisation* tests: they assert the properties that
make Rh transfer impossible, so that a future change to the fold assembly
or to the featuriser cannot silently re-break it.  If one of these starts
failing, the fix has either landed (good — update the test) or the
representation changed (must re-derive the conclusion).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "molmetal" / "data" / "metal_lomo.csv"
SYNTH_DIR = REPO / "molmetal" / "reports" / "wf125_p1_rh_oversample"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV)


# ---------------------------------------------------------------------------
# Property 1 — the LOMO model receives no metal identity
# ---------------------------------------------------------------------------
def test_dmpnn_multitask_model_has_no_metal_embedding() -> None:
    """``DMPNNMultiTaskModel`` (the class the LOMO trainer instantiates)
    has no metal embedding, so a leave-one-METAL-out fold cannot condition
    on which metal was held out.  ``V4MultiTaskDMPNN`` *does* have one
    (``metal_embed``) but is not what ``train_metal_lomo_wasserstein``
    uses.
    """
    src = (REPO / "molmetal" / "baselines" / "dmpnn_multitask.py").read_text()
    assert "metal_embed" not in src, (
        "DMPNNMultiTaskModel gained a metal embedding — re-derive the Rh "
        "transfer conclusion before assuming the metal-blindness still holds"
    )


def test_lomo_trainer_never_passes_metal_ids() -> None:
    """The trainer calls ``model(atom_f, bond_f, e_src, e_dst)`` — four graph
    args, no metal ids.  This is the call that makes the fold metal-blind.
    """
    src = (REPO / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py").read_text()
    assert "metal_ids" not in src
    assert "model(atom_f, bond_f, e_src, e_dst)" in src


# ---------------------------------------------------------------------------
# Property 2 — the Rh fold is small and mostly unseen
# ---------------------------------------------------------------------------
def test_rh_fold_is_108_rows(df: pd.DataFrame) -> None:
    rh = df[df.Metal == "Rh"]
    assert len(rh) == 108
    assert int(rh.active.sum()) == 35  # 32.4 % positive


def test_real_data_carries_no_metal_token(df: pd.DataFrame) -> None:
    """No real row's SMILES contains a bracketed metal token — the metal is
    only ever in the ``Metal`` column, which the model never sees.  This is
    why ``_strip_metal_atoms`` is a no-op on real data and only bites on the
    synthetic cohort.
    """
    for metal in ("Rh", "Ru", "Ir", "Re", "Os"):
        sub = df[df.Metal == metal]
        assert not sub.SMILES_Ligands.str.contains(rf"\[{metal}\]", regex=True).any()


def test_rh_metal_strip_is_a_noop(df: pd.DataFrame) -> None:
    from molmetal.data.featurize import _strip_metal_atoms

    changed = sum(1 for s in df.SMILES_Ligands if _strip_metal_atoms(s)[1] > 0)
    assert changed == 0, (
        "real SMILES started carrying metal tokens — the metal-blindness "
        "analysis must be redone"
    )


# ---------------------------------------------------------------------------
# Property 3 — the synthetic cohort is off-distribution after stripping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fname", ["synthetic_rh_cohort.json", "synthetic_rh_cohort_active.json"])
def test_synth_collapses_to_fragments_when_metal_stripped(fname: str) -> None:
    """Stripping ``[Rh]`` from the synthetic cohort leaves *disconnected*
    ligand fragments (6 fragments / 0.8 rings on average), whereas real Rh
    is a connected complex (3.7 fragments / 4.7 rings).  This is the
    distribution shift that makes the oversample uninformative.
    """
    from molmetal.data.featurize import _count_fragments, _strip_metal_atoms
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    data = json.loads((SYNTH_DIR / fname).read_text())
    frags, rings = [], []
    for row in data["per_compound"]:
        stripped, _ = _strip_metal_atoms(row["smiles"])
        mol = Chem.MolFromSmiles(stripped)
        if mol is None:
            continue
        frags.append(_count_fragments(stripped))
        rings.append(Chem.GetSSSR(mol))
    mean_frag = sum(frags) / len(frags)
    assert mean_frag > 4.0, (
        f"mean fragment count dropped to {mean_frag:.1f} — the synthetic "
        "cohort is no longer fragmented; the KS-shift argument must be redone"
    )


def test_synth_active_labels_are_not_measured() -> None:
    """The cohort's own metadata states the Active=1 labels are a Tanimoto
    transfer, not measured IC50 (median max-sim ~0.18).
    """
    stats = json.loads((SYNTH_DIR / "synthetic_rh_cohort_active.json").read_text())["stats"]
    assert stats["label_is_measured"] is False
    assert stats["max_sim_to_real_active"]["median"] < 0.25


def test_synth_labels_contradict_real_pic50_distribution(df: pd.DataFrame) -> None:
    """Synthetic rows are hard-coded to pIC50 7.0 (active) / 5.0 (inactive);
    real Rh spans 3.30-6.30 with median 4.60.  The active archetype sits
    *above the entire real Rh range*, so the regression head is trained on a
    target the real fold can never produce.
    """
    src = (REPO / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py").read_text()
    assert 'row["pIC50"] = 7.0' in src
    rh = df[df.Metal == "Rh"].pIC50
    assert rh.max() < 7.0
    assert 7.0 - rh.max() > 0.5, "real Rh now reaches the synthetic active pIC50"
