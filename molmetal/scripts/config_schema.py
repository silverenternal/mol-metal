# -*- coding: utf-8 -*-
"""Tree-structured config schema for Mol-Metal.

Mirror of OmegaFold's `omegafold/config.py` ``_make_config`` /
``make_config`` recursive-Namespace pattern.  A nested dict is
recursively converted to :class:`argparse.Namespace` so consumers can
use ``cfg.plm.node`` dot-syntax rather than nested ``cfg["plm"]["node"]``
lookups.

Closely matches the canonical OmegaFold implementation
(``omegafold/config.py:32-40`` for ``_make_config`` + 43-111 for
``make_config``).

The Mol-Metal ``make_config(model_idx)`` here is a thin scaffold so
importing ``molmetal.config_schema`` does not pull in torch / RDKit /
the heavy ``flow_matching_lipman`` adapter chain.  Per-metric /
per-launcher ``make_config(model_idx=1)`` factories are added in
follow-on waves (Wf-440+ — see :mod:`molmetal.baselines.train_cfm`
and :mod:`molmetal.baselines.train_dmpnn_multitask_wasserstein`).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict


def _make_config(input_dict: Dict[str, Any]) -> argparse.Namespace:
    """Recursively go through dictionary, converting nested dicts to Namespaces.

    Mirrors ``omegafold.config._make_config`` (``omegafold/config.py:32-40``).
    """
    new_dict: Dict[str, Any] = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            new_dict[k] = _make_config(v)
        else:
            new_dict[k] = v
    return argparse.Namespace(**new_dict)


def make_config(model_idx: int = 1) -> argparse.Namespace:
    """Return the canonical Mol-Metal config namespace.

    Mirrors ``omegafold.config.make_config`` (``omegafold/config.py:43-111``).
    ``model_idx`` is a stub — Mol-Metal has 1 active baseline family at
    the time of writing (Wf-440+ will add the chemprop_v2/v3 + CFM
    family indices).

    Returns
    -------
    argparse.Namespace
        Tree-structured Namespace with ``plm`` + ``struct`` + ``flat``
        sub-namespaces; access via ``cfg.plm.node`` dot-syntax.
    """
    if model_idx not in (1, 2):
        raise ValueError(f"model_idx must be 1 or 2 (got {model_idx!r})")
    cfg: Dict[str, Any] = dict(
        # High-level flat config (single Namespace level)
        node_dim=256,
        edge_dim=128,
        hidden_dim=128,
        n_layers=3,
        c=16,
        geo_num_blocks=50,
        gating=True,
        attn_c=32,
        attn_n_head=8,
        transition_multiplier=4,
        activation="ReLU",
        # Nested struct sub-tree
        struct=dict(
            node_dim=384,
            edge_dim=128,
            num_cycle=8,
            num_transition=3,
            num_head=12,
            num_point_qk=4,
            num_point_v=8,
            num_scalar_qk=16,
            num_scalar_v=16,
            num_channel=128,
            num_residual_block=2,
            hidden_dim=128,
            num_bins=50,
        ),
        # Nested data sub-tree
        data=dict(
            csv_path="molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv",
            scaffold_split_seed=261,
            n_scaffolds_min=2,
            min_train=200,
        ),
    )
    cfg["struct_embedder"] = model_idx == 2
    return _make_config(cfg)


__all__ = ["_make_config", "make_config"]


if __name__ == "__main__":
    cfg = make_config(model_idx=1)
    print(f"node_dim={cfg.node_dim}  hidden_dim={cfg.hidden_dim}")
    print(
        f"struct.num_cycle={cfg.struct.num_cycle}  struct.num_head={cfg.struct.num_head}"
    )
    print(f"data.csv_path={cfg.data.csv_path}")
