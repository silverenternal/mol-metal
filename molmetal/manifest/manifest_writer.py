# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""ManifestWriter — emits a ManifestGenerator payload as YAML + JSON + paper-appendix."""
from __future__ import annotations

import json
import os
from typing import Dict

import yaml


class ManifestWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write_yaml(self, manifest: Dict, filename: str = "manifest.yaml") -> str:
        path = os.path.join(self.out_dir, filename)
        with open(path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        return path

    def write_json(self, manifest: Dict, filename: str = "manifest.json") -> str:
        path = os.path.join(self.out_dir, filename)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return path

    def write_paper_appendix(self, manifest: Dict, filename: str = "manifest_appendix.md") -> str:
        path = os.path.join(self.out_dir, filename)
        lines = ["# Reproducibility Manifest", ""]
        for k, v in manifest.items():
            lines.append(f"## {k}")
            lines.append("```")
            lines.append(str(v))
            lines.append("```")
            lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def write_all(self, manifest: Dict) -> Dict[str, str]:
        return {
            "yaml": self.write_yaml(manifest),
            "json": self.write_json(manifest),
            "appendix": self.write_paper_appendix(manifest),
        }


__all__ = ["ManifestWriter"]