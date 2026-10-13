# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Manifest surface — END5 fix.

Generates a 12-field reproducibility manifest from a config + checkpoint:
git_commit, git_branch, git_status_short, python_version, torch_version,
cuda_version, yaml_sections, rng_seeds, dataset_version, ckpt_path + step
+ best_metric, environment. Output YAML + JSON + paper-appendix text.
"""
from __future__ import annotations

from molmetal.manifest.manifest_generator import ManifestGenerator
from molmetal.manifest.manifest_writer import ManifestWriter

__all__ = ["ManifestGenerator", "ManifestWriter"]