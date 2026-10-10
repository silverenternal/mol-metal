# SPDX-License-Identifier: Apache-2.0
"""The decode-path largest-fragment step, and the disclosure it carries.

Added 2026-10-10 after the BLK-C11 A/B came back null and the literature
located the real intervention: the field-standard post-process for
disconnection (`--all_frags` in the EDM/DiffSBDD family) operates on the
DECODED molecule, not on a training loss.

MEASURED leverage check, on 776 already-decoded molecules, no retraining:

    multi-fragment          85.1%
    metal centre detached   16.4%   ->   0.0%  after truncation
    sanitize-passing        100%    ->   100%   (buys nothing)
    heavy atoms retained     80.9%  (costs 19.1% of all heavy atoms)

So the step is implemented and available, but the honest reading is that it is
NOT a quality win here: our validity was never fragmentation-limited, because
RDKit accepts multi-fragment SMILES. What it does buy is a metal centre that is
actually bonded to something, at a real price in discarded atoms.

That price is why the disclosure is per-row data rather than a docstring. If
anyone ever turns this on, the CSV and `_gen_chem` must say how much of the
reported number came from dropping atoms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


@pytest.fixture(scope="module")
def parser_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_flag_exists_and_defaults_off(parser_module):
    """Default OFF: no shipped run changes behaviour silently."""
    sys.argv = ["t"]
    args = parser_module.parse_args()
    assert hasattr(args, "decode_largest_fragment"), (
        "--decode-largest-fragment is not on the parser; the step would be "
        "unreachable from a config (the FLAG_ONLY shape, AGENTS.md §5.7)"
    )
    assert args.decode_largest_fragment is False


def test_flag_round_trips(parser_module):
    sys.argv = ["t", "--decode-largest-fragment"]
    assert parser_module.parse_args().decode_largest_fragment is True


def test_saved_smiles_csv_declares_the_disclosure_columns():
    """The header must carry the raw fragment stats on every row.

    Without them the file looks like a set of clean molecules and the reader
    cannot tell that 19% of the atoms were discarded to produce it.
    """
    src = TRAINER.read_text(encoding="utf-8")
    assert '"n_frags_raw"' in src and '"largest_frag_frac"' in src, (
        "the per-row disclosure columns were removed from generated_smiles.csv"
    )
    for key in (
        "n_truncated_to_largest_fragment",
        "atoms_dropped_by_largest_fragment",
        "largest_fragment_disclosure",
    ):
        assert key in src, f"_gen_chem lost {key} — the truncation cost is untracked"


def test_disclosure_names_the_discard_not_the_repair():
    """The wording must not imply the fragments were reconnected."""
    src = TRAINER.read_text(encoding="utf-8")
    assert "discards atoms" in src or "DISCARDS atoms" in src, (
        "the disclosure must state that atoms are discarded rather than "
        "reconnected — that is the field's own warning about this step"
    )


def test_largest_fragment_behaviour_on_a_hand_built_case():
    """The truncation itself, independent of the trainer."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    # Two disconnected pieces; the metal sits on the smaller one, so the
    # largest-component rule must be what decides, not "keep the metal".
    m = Chem.MolFromSmiles("CC[Pt](Cl)Cl.CCCCCCC")
    frs = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    assert len(frs) == 2
    biggest = max(frs, key=lambda f: f.GetNumHeavyAtoms())
    kept = Chem.MolToSmiles(biggest)
    assert "Pt" not in kept, (
        "keeping the largest component must be allowed to drop the metal "
        "fragment — otherwise the rule is really 'keep the metal', which is a "
        "different and unstated policy"
    )
