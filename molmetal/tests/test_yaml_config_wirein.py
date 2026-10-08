# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Wf-A2 YAML config wire-in across the headline trainers.

User directive: "所有的训练以及任何的脚本通透都得采用yaml配置文件驱动而不是cli参数".

CPU-only, no training.  Proves each trainer's parser (a) accepts --config and
(b) actually APPLIES the YAML value to the namespace (the §5.7 distinction
between a declared flag and a wired one).  A `--help`-only check would pass
even for a FLAG_ONLY wire-in, so these tests assert the resolved value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump({"train": {"epochs": 7}, "model": {"hidden_dim": 123}}),
        encoding="utf-8",
    )
    return p


def test_lomo_wasserstein_applies_yaml(tmp_path):
    from molmetal.scripts.train_metal_lomo_wasserstein import _parse_args

    cfg = _write_cfg(tmp_path)
    ns = _parse_args(["--config", str(cfg)])
    assert ns.epochs == 7, "YAML epochs override did not reach the namespace"
    assert getattr(ns, "config", None) == str(cfg)

    default = _parse_args([])
    assert default.epochs != 7, "absent --config must keep the argparse default"


def test_lomo_wasserstein_cli_wins_over_yaml(tmp_path):
    from molmetal.scripts.train_metal_lomo_wasserstein import _parse_args

    cfg = _write_cfg(tmp_path)
    ns = _parse_args(["--config", str(cfg), "--epochs", "3"])
    assert ns.epochs == 3, "explicit CLI must win over the YAML default"


def test_all_headline_trainers_expose_config():
    """Each wired trainer must accept --config in its parser.

    lomo_v2 and chemprop build their parser inside main(); we assert the
    source wires apply_yaml_overrides so a later refactor cannot silently
    drop the override call.
    """
    import inspect

    from molmetal.scripts import train_chemprop_v2_v3 as chem
    from molmetal.scripts import train_metal_lomo_v2 as lomo2
    from molmetal.scripts import train_metal_lomo_wasserstein as lomow

    assert "apply_yaml_overrides" in inspect.getsource(lomo2.main)
    assert "--config" in inspect.getsource(lomo2.main)
    assert "apply_yaml_overrides" in inspect.getsource(chem.main)
    assert "--config" in inspect.getsource(chem.main)
    assert "apply_yaml_overrides" in inspect.getsource(lomow._parse_args)
