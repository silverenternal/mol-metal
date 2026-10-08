"""Wf-R44 — decode -> panel link: persist decoded SMILES + feed the panel.

The CFM generation eval decodes a SMILES per sample and used to throw the
string away, so no generated cohort ever existed to score with
``cfm_property_panel.build_panel`` and m4 SA / m5 QED / m7 diversity stayed
UNMEASURED.  These tests pin the two halves of the fix, CPU-only:

  1. ``evaluate_generation_validity`` now collects the decoded SMILES into
     ``smiles_out`` (the flag alone is FLAG_ONLY without this).
  2. A CSV written in the trainer's ``generated_smiles.csv`` shape
     (``epoch,smiles``) is read by the panel's own ``read_smiles_csv`` and
     scored by ``build_panel`` -- the consumer end of the chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PANEL_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"


def _load_panel():
    spec = importlib.util.spec_from_file_location(
        "cfm_property_panel_r44", str(PANEL_PATH)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    import importlib.util as ilu

    path = (
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py"
    )
    spec = ilu.spec_from_file_location("cfm_trainer_r44", str(path))
    assert spec and spec.loader
    mod = ilu.module_from_spec(spec)
    # Register before exec: the trainer's dataclasses resolve their own
    # __module__ through sys.modules at class-creation time.
    sys.modules["cfm_trainer_r44"] = mod
    # Executing the trainer module only defines functions/classes; it does not
    # train because nothing calls main() at import.
    spec.loader.exec_module(mod)
    return mod


def test_signature_has_smiles_out(trainer):
    """The collector param must exist, or the flag is FLAG_ONLY."""
    import inspect

    sig = inspect.signature(trainer.evaluate_generation_validity)
    assert "smiles_out" in sig.parameters
    # Optional so every existing caller is unchanged.
    assert sig.parameters["smiles_out"].default is None


def test_decoded_smiles_are_collected(trainer, monkeypatch):
    """A decode that yields a SMILES must append it to smiles_out."""

    class _Decoded:
        smiles = "Cl[Pt](N)(N)Cl"
        sanitized = True

    class _Res:
        decoded = _Decoded()

    class _FakeDecoder:
        def decode(self, cloud):
            return _Res()

    # Generate one cloud per template without a model.
    monkeypatch.setattr(
        trainer,
        "generate_clouds_from_noise",
        lambda model, pool, device, *, method, n_steps, seed, **kw: list(pool),
        raising=True,
    )
    smiles_out: list[str] = []
    n_ok, n_tot = trainer.evaluate_generation_validity(
        model=object(),
        decoder=_FakeDecoder(),
        templates=[object(), object()],
        device="cpu",
        n_samples=2,
        seed=0,
        smiles_out=smiles_out,
    )
    assert n_tot == 2
    assert smiles_out == ["Cl[Pt](N)(N)Cl", "Cl[Pt](N)(N)Cl"]


def test_smiles_out_none_is_a_noop(trainer, monkeypatch):
    """Without smiles_out the call is bit-for-bit the old behaviour."""
    monkeypatch.setattr(
        trainer,
        "generate_clouds_from_noise",
        lambda model, pool, device, *, method, n_steps, seed, **kw: list(pool),
        raising=True,
    )
    # No smiles_out passed -> must not raise.
    n_ok, n_tot = trainer.evaluate_generation_validity(
        model=object(),
        decoder=object(),
        templates=[object()],
        device="cpu",
        n_samples=1,
        seed=0,
    )
    assert n_tot == 1


def test_generated_smiles_csv_feeds_panel(tmp_path):
    """The trainer's generated_smiles.csv shape is panel-readable and scored.

    This is the consumer half of the link: a two-column epoch,smiles CSV
    (exactly what --save-generated-smiles writes) must be read by the panel's
    read_smiles_csv and produce a real SA/QED panel.
    """
    panel = _load_panel()
    csv_path = tmp_path / "generated_smiles.csv"
    csv_path.write_text(
        "epoch,smiles\n"
        "1,CCO\n"
        "1,c1ccccc1\n"
        "2,CC(=O)Oc1ccccc1C(=O)O\n",
        encoding="utf-8",
    )
    smiles = panel.read_smiles_csv(csv_path)
    assert smiles == ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    built = panel.build_panel(smiles)
    assert built["n_input"] == 3
    assert built["n_parsed"] == 3
    assert built["aggregate"]["sa"]["mean"] > 0.0
    assert 0.0 <= built["aggregate"]["qed"]["mean"] <= 1.0
