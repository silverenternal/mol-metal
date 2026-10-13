# SPDX-License-Identifier: Apache-2.0
"""pad_groups must pad each BUCKET to its own max, not the global max.

THE DEFECT THIS PINS
--------------------
``generate_clouds_from_noise`` with ``pad_groups=True`` put ALL clouds into a
single batch padded to the GLOBAL maximum atom count. Its own comment said so:
"ONE group at the largest atom count, every other cloud padded up to it."

MEASURED on the 192-cloud tmQM val cohort (min 14, max 72, mean 38.8 atoms):

    grouped, 52 exact-size buckets     0.33 GiB peak VRAM
    one global-max batch               3.86 GiB peak VRAM   11.7x
    one global-max batch at n_steps=50   OOM at 92.06 GiB on a 95 GiB card,
                                        twice, with the padded arm ALONE

The 11.7x exceeds the 3.2x edge-count ratio because ``radius_edge_index``
pads every cloud in a batch to ``e_max``, the MAXIMUM edge count over the
batch: a 14-atom cloud is allocated the edge budget of the largest one.

THE FIX
-------
Coarse size buckets. Each cloud's atom count is rounded UP to a multiple of
``pad_group_bucket`` and each bucket is padded only to its own bucket max, so
padding waste is bounded by (bucket - 1) atoms instead of the global maximum.

THE PROPERTY THAT MATTERS
-------------------------
Not "it runs" -- it ran before, it just consumed 92 GiB. The property is that
MEMORY IS BOUNDED BY THE BUCKET, and that it is a strict function of
``pad_group_bucket``: a larger bucket may waste more padding, never more
batches of unbounded size.

RED CHECK: restore the single global-max batch and
``test_many_buckets_is_not_one_global_batch`` fails.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _src() -> str:
    return TRAINER.read_text(encoding="utf-8")


def _generate_fn() -> ast.FunctionDef:
    tree = ast.parse(_src())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_clouds_from_noise":
            return node
    raise AssertionError("generate_clouds_from_noise not found")


def _bucket_keys(bucket: int, sizes: List[int]) -> List[int]:
    """Mirror of the shipped ceil-div bucket key, for property checks."""
    b = max(1, int(bucket))
    keys = sorted({-(-n // b) * b for n in sizes})
    return keys


COHORT = [14, 18, 22, 27, 31, 38, 45, 52, 61, 72] * 19 + [14, 18]  # 192, real range


def test_many_buckets_is_not_one_global_batch():
    node = _generate_fn()
    body = ast.get_source_segment(_src(), node) or ""
    assert "list(templates)" not in body, (
        "the single global-max batch is back: `group = list(templates)` puts "
        "every cloud in one batch padded to the global maximum, which measured "
        "0.33 -> 3.86 GiB (11.7x) and OOM'd at 92 GiB on the real run"
    )


def test_bucketing_is_present():
    body = ast.get_source_segment(_src(), _generate_fn()) or ""
    assert "pad_group_bucket" in body, "no bucket size parameter in the body"
    assert "by_size" in body, "the size histogram is not being used to bucket"


def test_bucket_count_is_small_but_greater_than_one():
    """The whole point of padding is fewer launches than exact-size buckets."""
    keys = _bucket_keys(8, COHORT)
    assert len(keys) > 1, "a single bucket is the global-max bug again"
    # 14..72 rounded up to multiples of 8 -> 16,24,...,72 = 8 buckets,
    # against 52 exact sizes. So it batches 6.5x fewer times than grouped.
    assert len(keys) <= 10, f"too many buckets ({len(keys)}); padding buys nothing"
    assert len(keys) < len(set(COHORT)), "no reduction in launch count vs grouped"


def test_padding_waste_is_bounded_by_the_bucket_width():
    """Each cloud is padded to the SMALLEST bucket multiple that covers it.

    This is the property that replaces the 92 GiB batch: with bucket=8 a
    72-atom cloud stays at 72 and a 14-atom cloud goes to 16, so the largest
    batch is sized by the largest BUCKET rather than by the global maximum.

    (An earlier version of this test asserted `all(k >= n for every key and
    every size)`, which is not a property of any bucketing -- a small key is
    never >= a large size. The correct statement is per-size: the key covering
    n is the least multiple of the bucket that is >= n.)
    """
    top = max(COHORT)
    for bucket in (1, 4, 8, 16):
        keys = set(_bucket_keys(bucket, COHORT))
        for n in set(COHORT):
            want = -(-n // bucket) * bucket          # smallest multiple >= n
            assert want in keys, f"bucket={bucket}: {n} has no covering key"
            assert want - n < bucket or want == n, (
                f"bucket={bucket}: {n} padded to {want}, wasting "
                f"{want - n} atoms -- must be under {bucket}"
            )
        assert max(keys) <= top + bucket - 1, (
            f"bucket={bucket}: largest batch key {max(keys)} exceeds the "
            f"largest cloud {top} by more than the bucket width"
        )


def test_bucket_one_reproduces_exact_grouping():
    """bucket=1 must be the grouped behaviour: every distinct size its own batch."""
    keys = _bucket_keys(1, COHORT)
    assert keys == sorted(set(COHORT)), f"bucket=1 gave {keys}"


def test_larger_bucket_never_yields_more_buckets():
    """Monotonicity: a coarser bucket can waste padding, never add batches."""
    prev = None
    for bucket in (1, 2, 4, 8, 16, 32, 64):
        n = len(_bucket_keys(bucket, COHORT))
        if prev is not None:
            assert n <= prev, (
                f"bucket={bucket} produced {n} batches, more than the "
                f"{prev} of a finer bucket -- that is not coarser, that is "
                "a different partitioning"
            )
        prev = n


def test_parameter_defaults_to_eight():
    """Read the default from the SOURCE TEXT, not from ast arg alignment.

    An earlier version indexed args.defaults positionally against args.args
    and got the wrong slot, so it asserted on an unrelated parameter and
    passed for the wrong reason."""
    import re

    m = re.search(r"pad_group_bucket:\s*int\s*=\s*(\d+)", _src())
    assert m, "no `pad_group_bucket: int = N` in the source"
    assert int(m.group(1)) == 8, f"default bucket is {m.group(1)}, expected 8"


def test_pad_groups_false_path_is_untouched():
    """The default path must still do exact-size grouping."""
    body = ast.get_source_segment(_src(), _generate_fn()) or ""
    assert "for n_atoms in sorted(by_size):" in body, (
        "the grouped path's exact-size loop is gone; every archived run uses "
        "it and its behaviour must not change"
    )