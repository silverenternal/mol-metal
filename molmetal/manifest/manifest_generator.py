# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""ManifestGenerator — END5 fix.

12-field reproducibility manifest over git + Python + torch + CUDA +
yaml sections + RNG seeds + dataset + environment. Pools gate:
all provenance must be either a measured file path or a documented external
dependency (e.g. CUDA build version).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional


class ManifestGenerator:
    YAML_SECTIONS = ["data", "model", "flow", "protocol", "operator", "loss", "egnn", "compute"]

    def __init__(self, project_root: str, config_path: str, ckpt_path: Optional[str] = None) -> None:
        self.project_root = project_root
        self.config_path = config_path
        self.ckpt_path = ckpt_path

    def _git(self, args: List[str]) -> str:
        try:
            return subprocess.check_output(
                ["git"] + args,
                cwd=self.project_root,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except Exception:
            return "HONEST-NEGATIVE: git unavailable"

    def generate(self) -> Dict:
        out: Dict = {}
        out["git_commit"] = self._git(["rev-parse", "HEAD"]).decode() if isinstance(self._git(["rev-parse", "HEAD"]), bytes) else self._git(["rev-parse", "HEAD"])
        out["git_branch"] = self._git(["rev-parse", "--abbrev-ref", "HEAD"])
        out["git_status_short"] = self._git(["status", "--short"])
        out["python_version"] = sys.version.split()[0]
        try:
            import torch  # noqa: PLC0415
            out["torch_version"] = torch.__version__
            out["cuda_version"] = torch.version.cuda or "no-cuda-build"
            out["cuda_available"] = bool(torch.cuda.is_available())
        except ImportError:
            out["torch_version"] = "HONEST-NEGATIVE: torch not installed"
            out["cuda_version"] = "HONEST-NEGATIVE"
            out["cuda_available"] = False
        out["yaml_sections"] = self._yaml_sections()
        out["rng_seeds"] = self._rng_seeds()
        out["dataset_version"] = self._dataset_version()
        out["ckpt"] = self._ckpt()
        out["environment"] = self._environment()
        out["platform"] = platform.platform()
        return out

    def _yaml_sections(self) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        try:
            import yaml  # noqa: PLC0415
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            for sec in self.YAML_SECTIONS:
                sections[sec] = "present" if sec in cfg else "absent"
        except Exception as e:
            sections["error"] = f"HONEST-NEGATIVE: yaml parse failed: {e}"
        return sections

    def _rng_seeds(self) -> Dict[str, Optional[int]]:
        # Best-effort: many training scripts set seeds via argparse defaults.
        # We report the YAML-declared seed if present, else honest-negative.
        seeds = {"numpy": None, "torch": None, "random": None}
        try:
            import yaml  # noqa: PLC0415
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f) or {}
            # cfm configs declare seeds via --seed CLI, not via config; declare HONEST-NEGATIVE
            seeds["yaml_declared_seed"] = cfg.get("seed", "HONEST-NEGATIVE: seed is CLI-only")
        except Exception as e:
            seeds["error"] = str(e)
        return seeds

    def _dataset_version(self) -> str:
        try:
            import yaml  # noqa: PLC0415
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return str(cfg.get("data", {}).get("dataset", "HONEST-NEGATIVE"))
        except Exception:
            return "HONEST-NEGATIVE"

    def _ckpt(self) -> Dict[str, str]:
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            return {"path": "HONEST-NEGATIVE: no ckpt path provided", "exists": "false"}
        return {"path": self.ckpt_path, "exists": "true", "size_bytes": str(os.path.getsize(self.ckpt_path))}

    def _environment(self) -> str:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            return out
        except Exception:
            return "HONEST-NEGATIVE: nvidia-smi unavailable or no GPU"


__all__ = ["ManifestGenerator"]