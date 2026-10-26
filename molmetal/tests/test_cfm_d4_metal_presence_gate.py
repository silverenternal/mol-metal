"""Round-3 D4 — the assembled graph must still carry the metal it was seeded
with, and the failure has to be visible when it does not.

Why this file exists.  ``metal_ratio`` on the Round-2 run read 1.000 at every
epoch, and the Round-3 brief recorded that as "confirmed metal presence is
correct".  It was not evidence.  ``metal_ratio`` asked a question the corpus
cannot answer wrongly: "is one of {Ru,Rh,Re,Os,Ir,Pt} present somewhere in
this SMILES?"  The seeds are all metal complexes, the decoder is scored on
those seeds, so the metric is saturated by construction.  A check that reads
1.000 no matter what the model does cannot detect its own failure — which is
exactly the failure mode that let the all-BOND_NO_BOND target run unnoticed
for weeks (see ``_bond_target_for_clean_cloud``).

D4 asks the sharper question: does the assembled graph contain the metal THIS
sample was seeded with.  Four properties are pinned here:

1. **Seed-scoped, not corpus-scoped.**  A Pt-seeded sample that decodes to an
   organic molecule has lost its metal and must score 0, even though
   ``has_metal`` against the corpus-wide target set would also be False here —
   the test uses a CROSS-METAL case (Pt seed, Ru decode) where the weak check
   says True and only the seed-scoped check says False.  That case is the
   whole point: it is the only situation the old check misses.
2. **A metal-less composition is not a failure.**  An all-carbon sample has no
   seed metal, so it cannot lose one.  Scoring it as missing would make the
   gate fire on a corpus it does not apply to; the verdict there is SKIP.
3. **The gate is diagnostic, never a block.**  Above 0.10 it is WARN, above
   0.50 GATE_FAIL — and neither changes the return value of the eval, which
   stays ``(n_ok, n_total)``.  A gate that silently stopped counting decodes
   would corrupt ``gen_validity`` on its way to being useful.
4. **Legacy rows still summarise.**  Rows produced before D4 lack the new
   keys.  They must aggregate without raising and must not manufacture a
   healthy-looking ``n_metal_seeded=0`` PASS out of "never asked".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("rdkit", reason="D4 metal-presence gate needs RDKit")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load(
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py",
        "_t_cfm_d4_metal_presence",
    )


def _row(*, present, n_fragments=1, n_heavy=20, has_metal=True, zs=(78,)):
    return {
        "n_fragments": n_fragments,
        "n_heavy": n_heavy,
        "largest_fragment_frac": 1.0,
        "has_metal": has_metal,
        "seed_metal_present": present,
        "seed_metal_z": list(zs),
    }


# ---------------------------------------------------------------------------
# 1. Seed-scoped presence.  The cross-metal case is the one that matters.
# ---------------------------------------------------------------------------


def test_cross_metal_swap_is_reported_missing(trainer):
    """A Pt-seeded sample that decodes to Ru is NOT metal-present.

    Asked the OLD way (no seed passed), the corpus-wide check is True — a Ru
    IS in _TARGET_METALS — so the pre-D4 metric passes this sample and
    ``metal_ratio`` stays saturated.  Asked the SEED-scoped way, the same
    SMILES is a failure, because the metal it was seeded with (Pt) is gone.
    That difference is the entire reason this gate exists.
    """
    smi = "[Ru](N)(N)N"

    old = trainer.decode_chemistry(smi)
    assert old is not None
    assert old["has_metal"] is True, "the old corpus-wide check passes the swap"
    assert old["seed_metal_present"] is None, "…without ever asking the question"

    new = trainer.decode_chemistry(smi, frozenset({78}))
    assert new is not None
    assert new["has_metal"] is False, (
        "seeded with Pt and decoded as Ru, has_metal must narrow to False"
    )
    assert new["seed_metal_present"] is False, "but the SEED was Pt"
    assert new["seed_metal_z"] == [78]


def test_metal_absent_entirely_is_missing(trainer):
    d = trainer.decode_chemistry("c1cc[nH]c1", frozenset({78}))
    assert d is not None
    assert d["seed_metal_present"] is False


def test_seed_metal_present_when_kept(trainer):
    d = trainer.decode_chemistry("[Pt](N)(N)N", frozenset({78}))
    assert d is not None
    assert d["seed_metal_present"] is True
    assert d["seed_metal_z"] == [78]


def test_unseeded_call_keeps_legacy_semantics(trainer):
    """No seed given -> the old corpus-wide question, and no seed verdict.

    Pre-D4 callers must get the pre-D4 answer.  ``seed_metal_present`` is
    ``None`` (not False) because "not asked" and "asked and absent" are
    different facts, and conflating them would let an unseeded row be counted
    as a metal loss by the summariser.
    """
    d = trainer.decode_chemistry("[Pt](N)(N)N")
    assert d is not None
    assert d["has_metal"] is True
    assert d["seed_metal_present"] is None


# ---------------------------------------------------------------------------
# 2. Seed extraction.  The generated cloud is its own seed record.
# ---------------------------------------------------------------------------


def test_seed_metal_z_reads_composition(trainer):
    import torch

    cloud = trainer.AtomCloud(
        positions=torch.zeros(4, 3),
        atomic_numbers=torch.tensor([78, 6, 6, 7]),
    )
    assert trainer.seed_metal_z(cloud) == frozenset({78})


def test_seed_metal_z_empty_for_ligand_only(trainer):
    import torch

    cloud = trainer.AtomCloud(
        positions=torch.zeros(3, 3), atomic_numbers=torch.tensor([6, 6, 6])
    )
    assert trainer.seed_metal_z(cloud) == frozenset()


# ---------------------------------------------------------------------------
# 3. The gate thresholds.  WARN and GATE_FAIL are labels, not behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, "PASS"),
        (0.05, "PASS"),
        (0.10, "PASS"),  # boundary is strict: > 0.10 warns
        (0.1001, "WARN"),
        (0.25, "WARN"),
        (0.50, "WARN"),  # boundary is strict: > 0.50 fails
        (0.5001, "GATE_FAIL"),
        (1.0, "GATE_FAIL"),
    ],
)
def test_verdict_thresholds(trainer, rate, expected):
    assert trainer.metal_presence_verdict(rate, 10) == expected


def test_verdict_is_skip_when_nothing_seeded(trainer):
    """A corpus with no metal must not report a healthy 0.0.

    ``SKIP`` is the honest answer to "0.0 of 0 samples are missing"; ``PASS``
    would be a number that means nothing.
    """
    assert trainer.metal_presence_verdict(0.0, 0) == "SKIP"


# ---------------------------------------------------------------------------
# 4. Summarisation.  Counts, denominators, and legacy rows.
# ---------------------------------------------------------------------------


def test_summarise_counts_only_seeded_rows(trainer):
    rows = [_row(present=True) for _ in range(8)]
    rows += [_row(present=False) for _ in range(2)]
    s = trainer.summarise_decode_chemistry(rows)
    assert s["n_metal_seeded"] == 10
    assert s["n_metal_missing"] == 2
    assert s["metal_missing_rate"] == pytest.approx(0.2)
    assert s["metal_presence_gate"] == "WARN"


def test_metal_ratio_can_be_one_while_samples_are_missing(trainer):
    """The whole reason D4 exists, asserted directly.

    10 samples, all containing SOME metal (so ``metal_ratio`` == 1.000, the
    value the Round-2 run reported as confirmation), 2 of which lost the
    metal they were seeded with.  The saturated metric cannot see it; the
    new one must.
    """
    rows = [_row(present=True) for _ in range(8)]
    rows += [
        _row(present=False, has_metal=True, zs=(44,)) for _ in range(2)
    ]
    s = trainer.summarise_decode_chemistry(rows)
    assert s["metal_ratio"] == 1.0
    assert s["n_metal_missing"] == 2
    assert s["metal_presence_gate"] == "WARN"


def test_unseeded_rows_are_excluded_from_denominator(trainer):
    """A metal-less sample cannot lose a metal.

    Two seeded rows, both fine, plus one ``None`` verdict (never asked).
    Counting it in the denominator would report 0.0 over 3 rather than over
    the 2 that could actually fail.
    """
    rows = [_row(present=True), _row(present=True), _row(present=None)]
    s = trainer.summarise_decode_chemistry(rows)
    assert s["n_metal_seeded"] == 2
    assert s["n_metal_missing"] == 0
    assert s["metal_missing_rate"] == 0.0
    assert s["metal_presence_gate"] == "PASS"


def test_legacy_rows_do_not_crash_or_fake_a_pass(trainer):
    """Pre-D4 rows lack the new keys entirely.

    They must aggregate without raising, and must land on SKIP — reporting
    PASS off zero seeded rows would be a green light from an unasked question.
    """
    legacy = [
        {
            "n_fragments": 1,
            "n_heavy": 20,
            "largest_fragment_frac": 1.0,
            "has_metal": True,
        }
        for _ in range(5)
    ]
    s = trainer.summarise_decode_chemistry(legacy)
    assert s["n_chemistry_scored"] == 5
    assert s["n_metal_seeded"] == 0
    assert s["metal_presence_gate"] == "SKIP"


def test_all_lost_is_gate_fail(trainer):
    rows = [_row(present=False) for _ in range(4)]
    s = trainer.summarise_decode_chemistry(rows)
    assert s["metal_missing_rate"] == 1.0
    assert s["metal_presence_gate"] == "GATE_FAIL"


def test_no_parseable_rows_is_skip_not_fail(trainer):
    s = trainer.summarise_decode_chemistry([None, None])
    assert s["n_chemistry_scored"] == 0
    assert s["metal_presence_gate"] == "SKIP"


# ---------------------------------------------------------------------------
# 5. The gate is DIAGNOSTIC.  It must not be wired into the eval's result.
# ---------------------------------------------------------------------------


def test_gate_does_not_alter_the_eval_return_shape(trainer):
    """``evaluate_generation_validity`` still returns ``(n_ok, n_total)``.

    The gate is a label attached to the chemistry dict.  If it ever became a
    filter — dropping metal-less samples from ``n_total``, or zeroing
    ``n_ok`` — it would quietly redefine ``gen_validity`` while appearing to
    add a safety check.
    """
    import inspect

    sig = inspect.signature(trainer.evaluate_generation_validity)
    assert list(sig.parameters) == [
        "model",
        "decoder",
        "templates",
        "device",
        "method",
        "n_steps",
        "n_samples",
        "seed",
        "template_offset",
        "noise_sigma",
        # Wf-R43 m4 — the SA observation switch.  Listed here because this
        # assertion is the tripwire that catches a kwarg added to the eval
        # without the caller being updated.
        "sa_target",
        "chemistry_out",
        "chemistry_rows_out",
        # Wf-R44 — the decoded-SMILES collector, so the generation eval can
        # hand its output to a property panel.  Listed here for the same
        # reason as sa_target: this assertion is the tripwire that catches a
        # kwarg added to the eval without the caller being updated.
        # Wf-A100 (2026-10-20) — the tripwire fired: five kwargs had been
        # added since this list was written (Wf-A75 guidance, Wf-A81's
        # rejection collector, and the pad-group path) and nobody updated the
        # list, so this test was RED and the tripwire had stopped being
        # readable.  All five are live features with live callers; the list
        # now matches the signature again, IN SIGNATURE ORDER, and keeps
        # catching the next one.
        "bond_length_guidance_weight",
        "smiles_out",
        "rejected_smiles_out",
        "per_cloud_out",
        "pad_groups",
        "pad_group_bucket",
        # Wf-A112 (MOD5) — the two kwargs that carry the CN request and the
        # classifier-free-guidance scale into the generation eval.  The
        # tripwire fired again on 2026-10-26, which is what it is for: the
        # caller WAS updated in the same change, this list was not.
        "cn_cfg_scale",
        "cn_request_override",
    ]
    src = inspect.getsource(trainer.evaluate_generation_validity)
    # The function body must never branch on the gate verdict.
    assert "metal_presence_gate" not in src, (
        "evaluate_generation_validity must not branch on metal_presence_gate; "
        "the gate is diagnostic and lives in the caller"
    )
    assert "GATE_FAIL" not in src
    assert src.count("return n_ok, len(clouds)") == 1


def test_greps_confirm_the_metric_is_wired(trainer):
    """The names must exist in the trainer source, not only in this test.

    A gate that exists but is never printed or emitted is the FLAG_ONLY
    failure mode AGENTS.md §5.7 is about, so assert the wiring textually.
    """
    src = (PROJECT_ROOT / "molmetal" / "scripts" /
           "train_cfm_yuelbond_wasserstein.py").read_text()
    assert "metal_missing" in src
    assert "n_metal_missing=" in src
    assert "metal_presence_gate" in src
    # And it must reach events.jsonl, not only stdout.
    assert "metal_presence_gate=str(" in src
