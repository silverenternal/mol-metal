# SPDX-License-Identifier: Apache-2.0
"""``per_cloud_out`` must be index-aligned with ``clouds``, failures included.

WHY THIS EXISTS (BLK-C-PERF)
----------------------------
The pad_groups A/B is PAIRED by construction: same 192 val clouds, same noise
seed, only the batching differs. So the whole question -- does padding change
the result -- lives in the DISCORDANT CELLS, and McNemar on per-cloud pass
vectors is the licensed test.

That requires a per-cloud outcome vector. The obvious source,
``smiles_out``, CANNOT provide one: it appends only on success
(``if smiles_out is not None and smi``), so failed decodes are compressed out
and its indices stop corresponding to ``clouds``. Zipping two arms'
``smiles_out`` lists compares DIFFERENT clouds and would report a difference
that is purely an artefact of which clouds happened to succeed.

``per_cloud_out`` is the aligned vector: one bool per cloud, in cloud order,
with failures recorded as ``False``.

THE TWO PLACES IT CAN BREAK
---------------------------
1. A decode that RAISES hits ``except: ... continue``, skipping any append
   placed after the decode. The vector would then be short and every cloud
   after the first failure would be mis-paired.
2. A success that does not flip the placeholder to True leaves every entry
   False -- a vector that is aligned but uniformly wrong.

RED CHECK: move the append to AFTER the decode (case 1) or drop the
``per_cloud_out[-1] = True`` line (case 2) and the corresponding test fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _trainer():
    if "_per_cloud_trainer" in sys.modules:
        return sys.modules["_per_cloud_trainer"]
    spec = importlib.util.spec_from_file_location("_per_cloud_trainer", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_per_cloud_trainer"] = m
    spec.loader.exec_module(m)
    return m


class _FakeDecoded:
    def __init__(self, smiles: str, sanitized: bool = True):
        self.smiles = smiles
        self.sanitized = sanitized
        self.error = ""


class _FakeResult:
    def __init__(self, smiles: str):
        self.decoded = _FakeDecoded(smiles)


class _ScriptedDecoder:
    """Returns one scripted outcome per cloud; ``None`` means 'raise'."""

    def __init__(self, outcomes: List[Optional[str]]):
        self.outcomes = outcomes
        self.n = 0

    def decode(self, cloud):
        out = self.outcomes[self.n]
        self.n += 1
        if out is None:
            raise RuntimeError("simulated decode failure")
        return _FakeResult(out)


class _Cloud:
    """A cloud is consumed twice before the decode loop: once for the noise
    seed's per-cloud atom count, and once by the decoder. Only
    ``atomic_numbers`` is read here, so a list of ints is enough and the test
    needs no tensors on disk."""

    def __init__(self, i: int):
        import torch

        self.i = i
        # a real tensor: the noise draw reads `.shape` and indexes it.
        self.atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)


class _FakeModel:
    """The decode loop calls ``model.eval()`` / ``model.training`` but nothing
    else -- no module, no parameters. A real nn.Module would work too; this
    keeps the test from needing a checkpoint on disk."""

    training = True

    def eval(self):
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode


def _run(outcomes: List[Optional[str]]):
    """Call the decode loop DIRECTLY.

    Going through ``evaluate_generation_validity`` runs a real 50-step ODE
    solve before it ever reaches the decode -- minutes per case, and it makes
    the alignment property untestable in any reasonable budget. The loop was
    extracted for exactly this reason.
    """
    t = _trainer()
    clouds = [_Cloud(i) for i in range(len(outcomes))]
    per_cloud: List[bool] = []
    smiles: List[str] = []
    rows: List[Any] = []
    counts: dict = {}
    t.decode_clouds_and_classify(
        _ScriptedDecoder(outcomes), clouds, smiles, None, per_cloud, rows,
        None, counts,
    )
    return per_cloud, smiles


def test_vector_length_matches_cloud_count_even_with_failures():
    per_cloud, _ = _run(["CCO", None, "c1ccccc1", None])
    assert len(per_cloud) == 4, (
        f"per_cloud_out must have one entry per cloud; got {len(per_cloud)} "
        "for 4 clouds. A decode that RAISES hits `except: continue`, so any "
        "append placed after the decode is skipped and the vector goes short."
    )


def test_failures_are_recorded_false_and_successes_true():
    per_cloud, _ = _run(["CCO", None, "c1ccccc1", "CC"])
    assert per_cloud == [True, False, True, True], (
        f"expected [True, False, True, True], got {per_cloud}"
    )


def test_index_alignment_survives_a_raise_in_the_middle():
    """The failure that actually happened on2026-10-12."""
    per_cloud, _ = _run([None, "CCO", None, "c1ccccc1", "CC"])
    assert per_cloud == [False, True, False, True, True], (
        "a raise in the middle must not shift every later cloud by one; the "
        f"vector came back {per_cloud}"
    )


def test_smiles_out_is_shorter_and_therefore_unpairable():
    """This is WHY per_cloud_out had to be added -- pin the asymmetry."""
    _, smiles = _run(["CCO", None, "c1ccccc1", None])
    assert len(smiles) == 2, (
        f"smiles_out is expected to skip failures and so be shorter; got "
        f"{len(smiles)}. If this ever becomes index-aligned, the "
        "survivor-conditioning argument for per_cloud_out should be revisited."
    )


def test_all_failures_gives_all_false_not_empty():
    per_cloud, _ = _run([None, None, None])
    assert per_cloud == [False, False, False]


def test_parameter_is_optional_and_existing_callers_unaffected():
    """Omitting it must produce no vector and no error."""
    t = _trainer()
    clouds = [_Cloud(0), _Cloud(1)]
    smiles: List[str] = []
    rows: List[Any] = []
    counts: dict = {}
    t.decode_clouds_and_classify(
        _ScriptedDecoder(["CCO", "CC"]), clouds, smiles, None, None, rows,
        None, counts,
    )
    assert smiles == ["CCO", "CC"], "smiles_out must still work without the new arg"
    assert counts.get("ok") == 2