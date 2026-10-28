# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T73 — a cohort FILTER's default must be "off".

``--tmqm-coord-number`` is applied as an exact-equality filter on tmQM's
``coord_number`` column (``tmqm_clouds.py:1034``), not as a target.  Its default
used to be ``4``, so a config that forgot the key kept only four-coordinate
complexes.

MEASURED (metrics/by_metric/wfa113_t73_cohort_filter.json, tmqm_parsed.csv):

    filter=4   Pt 83.4%  Ru 0.34%  Ir 3.8%  Rh 11.8%  Os 0.37%  Re 0.31%
    no filter  Pt 24.8%  Ru 29.3%  Ir 15.7% Rh 13.9%  Os 3.9%   Re 12.5%

A "five-metal" config inheriting the old default would have trained on one
metal.  Every one of the 14 ``dataset: tmqm`` configs on disk overrides the key,
so nothing measured changed -- which is exactly why the default was never
caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_coord_number_default_is_no_filter():
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    old = sys.argv
    try:
        sys.argv = ["probe"]
        args = tr.parse_args()
    finally:
        sys.argv = old
    assert int(args.tmqm_coord_number) == -1, (
        "the coordination-number default must be -1 (no filter); a positive "
        "default is an exact-equality filter that keeps only that CN, which "
        "for 4 is 83.4% Pt and 0.34% Ru"
    )


def test_every_tmqm_config_still_declares_the_filter():
    """A config may narrow the cohort, but it must say so.

    This is the guard the default could not provide: with the default now
    'off', a config that intends CN-4 has to spell it out, and a reader can
    see it.
    """
    import re

    cfg_dir = PROJECT_ROOT / "configs"
    offenders = []
    for p in sorted(cfg_dir.glob("*.yaml")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if not re.search(r"^\s*dataset:\s*tmqm\s*$", text, re.M):
            continue
        if not re.search(r"^\s*tmqm_coord_number:", text, re.M):
            offenders.append(p.name)
    assert not offenders, (
        "dataset: tmqm configs without an explicit tmqm_coord_number: "
        f"{offenders}"
    )
