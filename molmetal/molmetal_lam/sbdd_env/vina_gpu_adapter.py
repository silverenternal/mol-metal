"""QuickVina2-GPU 2.1 OpenCL adapter with explicit execution evidence.

The official AMD build selects the first visible AMD GPU. This adapter supports
that device only and checks the engine's actual device log. GPU computing lanes
and Monte Carlo search depth are independent of native Vina exhaustiveness.
Ligand preparation, output refinement and RDKit conversion retain the engine's
CPU chemistry. No native Vina fallback is attempted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from molmetal.domain import Molecule, Pocket
from molmetal.ports import DockingConfig
from .vina_adapter import VinaDockingAdapter, _have_meeko, _parse_vina_result_energies, native_vina_seed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuickVinaGPUAdapter(VinaDockingAdapter):
    """Reuse the measured-pose chemistry path, replacing only docking execution.

    ``kernel_dir`` contains the official ``OpenCL/`` source tree (or the two
    compiled kernel binaries for a cache-only build). Output directories are
    unique per invocation, including kernel caches; shared source is read-only.
    Callers must supply a prepared receptor via ``set_prepared_receptor`` or the
    inherited receptor cache. Reconstructing a receptor from pocket atom types
    would lose the real residue chemistry and is deliberately unsupported here.
    """

    def __init__(self, binary_path: str | Path, *, kernel_dir: str | Path | None = None,
                 gpu_threads: int = 1000, search_depth: int = 1,
                 opencl_device: str = "opencl:0", expected_device: str = "gfx1101",
                 default_box_padding: float = 0., output_dir: str | Path | None = None,
                 timeout_s: float = 600., trace_library: str | Path | None = None):
        if isinstance(gpu_threads, bool) or int(gpu_threads) != gpu_threads or gpu_threads < 1000:
            raise ValueError("QuickVina2-GPU computing lanes (gpu_threads) must be an integer >= 1000")
        if isinstance(search_depth, bool) or int(search_depth) != search_depth or search_depth < 1:
            raise ValueError("GPU search_depth must be a positive integer")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(default_box_padding) or default_box_padding < 0:
            raise ValueError("default_box_padding must be finite and nonnegative")
        if not expected_device.strip():
            raise ValueError("An expected OpenCL GPU device name is required")
        binary = Path(binary_path).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"Executable QuickVina2-GPU binary required: {binary}")
        self._kernel_dir = Path(kernel_dir).expanduser().resolve() if kernel_dir else binary.parent
        if not (self._kernel_dir / "OpenCL").is_dir() and not all(
                (self._kernel_dir / name).is_file() for name in ("Kernel1_Opt.bin", "Kernel2_Opt.bin")):
            raise FileNotFoundError(f"OpenCL source tree or both kernel binaries required: {self._kernel_dir}")
        if not _have_meeko():
            raise RuntimeError("QuickVina2-GPU pose chemistry requires meeko and RDKit")
        self._trace_library = Path(trace_library).expanduser().resolve() if trace_library else None
        if self._trace_library is not None and not self._trace_library.is_file():
            raise FileNotFoundError(self._trace_library)
        self._engine = self.engine = "quickvina2-gpu"
        self._engine_binary = str(binary)
        self._binary_sha256 = _sha256(binary)
        kernel_inputs = sorted(path for path in (self._kernel_dir / "OpenCL").rglob("*")
                               if path.is_file() and path.suffix in (".cl", ".h"))
        if not kernel_inputs:
            kernel_inputs = [self._kernel_dir / name for name in ("Kernel1_Opt.bin", "Kernel2_Opt.bin")]
        self._kernel_input_hashes = {str(path.relative_to(self._kernel_dir)): _sha256(path) for path in kernel_inputs}
        self._kernel_inputs_sha256 = hashlib.sha256(json.dumps(self._kernel_input_hashes, sort_keys=True).encode()).hexdigest()
        self._gpu_threads, self._search_depth = int(gpu_threads), int(search_depth)
        self._expected_device, self._timeout_s = expected_device, float(timeout_s)
        self._padding, self._cpu, self._sf_name = float(default_box_padding), 1, "vina"
        if output_dir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="quickvina_gpu_"))
        else:
            self._tmpdir = Path(output_dir).expanduser().resolve()
            self._tmpdir.mkdir(parents=True, exist_ok=True)
        self._receptor_pdbqt: dict[str, Path] = {}
        self.last_pose_mols: list = []
        self.last_run_metadata: dict = {}
        self.last_docking_seed: dict = {}
        self.setup(opencl_device)

    @property
    def name(self) -> str:
        return "QuickVina2_GPU_2.1_OpenCL"

    def setup(self, device: str = "opencl:0") -> None:
        if device != "opencl:0":
            raise ValueError("This AMD build supports only explicit opencl:0; CPU/CUDA fallback is unavailable")
        self._device = device

    def set_prepared_receptor(self, pocket: Pocket, preparation: dict) -> None:
        if preparation.get("passed") is not True:
            raise ValueError("Measured docking requires passed receptor preparation")
        receptor = Path(preparation["pdbqt"]).resolve()
        if not receptor.is_file():
            raise FileNotFoundError(receptor)
        self._receptor_pdbqt[pocket.pdb_id] = receptor

    def _prepare_receptor(self, pocket: Pocket) -> Path:
        path = self._receptor_pdbqt.get(pocket.pdb_id)
        if path is None or not Path(path).is_file():
            raise ValueError("Supply a chemically prepared receptor PDBQT before GPU docking")
        return Path(path).resolve()

    def dock(self, molecule: Molecule, pocket: Pocket, config: DockingConfig):
        self.last_run_metadata = {}
        self.last_docking_seed = {}
        self.last_pose_mols = []
        if isinstance(config.n_poses, bool) or int(config.n_poses) != config.n_poses or config.n_poses < 1:
            raise ValueError("n_poses must be a positive integer")
        # Base dock preserves the Meeko graph, conformer, stereo and charge,
        # and rejects ambiguous energy/pose counts. Dispatch is subprocess.
        return super().dock(molecule, pocket, config)

    def _dock_cli_binary(self, receptor_pdbqt, lig_pdbqt, center, box_size, config):
        if not np.isfinite(center).all() or not np.isfinite(box_size).all() or np.any(box_size <= 0):
            raise ValueError("Docking box must have finite coordinates and positive dimensions")
        run_dir = Path(tempfile.mkdtemp(prefix="dock_", dir=self._tmpdir))
        source = self._kernel_dir / "OpenCL"
        if source.is_dir():
            (run_dir / "OpenCL").symlink_to(source, target_is_directory=True)
        for name in ("Kernel1_Opt.bin", "Kernel2_Opt.bin"):
            if (self._kernel_dir / name).is_file():
                shutil.copyfile(self._kernel_dir / name, run_dir / name)
        ligand = run_dir / "ligand.pdbqt"
        shutil.copyfile(lig_pdbqt, ligand)
        output = run_dir / "docked.pdbqt"
        effective_seed = native_vina_seed(config.seed)
        self.last_docking_seed = {"requested": int(config.seed), "effective": effective_seed,
                                  "mapping": "native_vina_seed; explicit nonzero signed int32"}
        command = [self._engine_binary, "--receptor", str(Path(receptor_pdbqt).resolve()),
                   "--ligand", str(ligand), "--out", str(output),
                   "--thread", str(self._gpu_threads), "--search_depth", str(self._search_depth),
                   "--num_modes", str(int(config.n_poses)), "--seed", str(effective_seed),
                   "--opencl_binary_path", str(run_dir)]
        for axis, c, size in zip("xyz", center, box_size):
            command += [f"--center_{axis}", str(float(c)), f"--size_{axis}", str(float(size))]
        env = dict(os.environ)
        env.update(GPU_DEVICE_ORDINAL="0", HIP_VISIBLE_DEVICES="0")
        if self._trace_library is not None:
            env["LD_PRELOAD"] = str(self._trace_library) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        log_path = run_dir / "engine.log"
        self.last_run_metadata = {
            "status": "running", "device_requested": self._device,
            "expected_device": self._expected_device, "gpu_verified": False,
            "command": command, "cwd": str(run_dir), "log_path": str(log_path),
            "output_pdbqt": str(output), "binary_sha256": self._binary_sha256,
            "kernel_inputs_sha256": self._kernel_inputs_sha256,
            "receptor_sha256": _sha256(Path(receptor_pdbqt)), "ligand_sha256": _sha256(ligand),
            "seed": dict(self.last_docking_seed), "gpu_threads": self._gpu_threads,
            "search_depth": self._search_depth, "requested_native_exhaustiveness_ignored": config.exhaustiveness,
            "gpu_budget_native_exhaustiveness_equivalent": None,
            "device_environment": {k: env[k] for k in ("GPU_DEVICE_ORDINAL", "HIP_VISIBLE_DEVICES")},
            "trace_library": str(self._trace_library) if self._trace_library else None,
            "trace_library_sha256": _sha256(self._trace_library) if self._trace_library else None,
        }
        started = time.monotonic()
        try:
            with log_path.open("w") as log:
                proc = subprocess.run(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=self._timeout_s, check=False)
            self.last_run_metadata["returncode"] = proc.returncode
            log_text = log_path.read_text(errors="replace")
            devices = re.findall(r"GPU Device:\s*(.+?)(?=Platform|\r|\n|$)", log_text)
            self.last_run_metadata["device_observed"] = [name.strip() for name in devices]
            self.last_run_metadata["device_log"] = [line for line in log_text.splitlines()
                                                    if "GPU Platform" in line or "GPU Device" in line]
            # stdout's progress bar has no terminating newline and may precede
            # the stderr trace on the same line in the merged subprocess log.
            traces = re.findall(r"OPENCL_TRACE [^\r\n]*", log_text)
            self.last_run_metadata["kernel_trace"] = traces
            if proc.returncode != 0:
                raise RuntimeError(f"QuickVina2-GPU exited {proc.returncode}; see {log_path}")
            if "GPU Platform: AMD" not in log_text or [name.strip() for name in devices] != [self._expected_device]:
                raise RuntimeError(f"Expected AMD OpenCL GPU {self._expected_device}, observed {devices}; see {log_path}")
            logged_seed = re.findall(r"Using random seed:\s*(-?\d+)", log_text)
            if not logged_seed or any(int(seed) != effective_seed for seed in logged_seed):
                raise RuntimeError(f"Engine seed log does not match configured seed; see {log_path}")
            if self._trace_library is not None:
                for kernel in ("kernel1", "kernel2"):
                    if not any(f"kernel={kernel} device={self._expected_device} enqueue=0 wait=0 status=0" in t for t in traces):
                        raise RuntimeError(f"No successful {kernel} OpenCL execution trace; see {log_path}")
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError(f"QuickVina2-GPU produced no pose; see {log_path}")
            poses = output.read_text()
            energies = _parse_vina_result_energies(poses)
            if not len(energies) or not np.isfinite(energies[:, 0]).any():
                raise RuntimeError(f"QuickVina2-GPU produced no finite pose score; see {log_path}")
            self.last_run_metadata.update(status="completed", gpu_verified=True,
                                          gpu_evidence="kernel_trace" if self._trace_library else "engine_device_log",
                                          output_sha256=_sha256(output))
            return energies, poses
        except Exception as exc:
            self.last_run_metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.last_run_metadata["elapsed_wall_s"] = time.monotonic() - started
            self.last_run_metadata["log_sha256"] = _sha256(log_path) if log_path.exists() else None

    def get_metadata(self) -> dict:
        return {"name": self.name, "engine": self.engine, "engine_version": "QuickVina2-GPU 2.1",
                "engine_binary": self._engine_binary, "binary_sha256": self._binary_sha256,
                "kernel_inputs_sha256": self._kernel_inputs_sha256, "kernel_input_hashes": self._kernel_input_hashes,
                "device": self._device, "expected_device": self._expected_device,
                "gpu_threads": self._gpu_threads, "search_depth": self._search_depth,
                "budget_note": "GPU lanes and Monte Carlo depth; no native exhaustiveness equivalence",
                "native_exhaustiveness_used": False, "scoring_function": self._sf_name,
                "cpu_stages": ["RDKit/Meeko ligand and receptor chemistry", "native final pose refinement"],
                "gpu_stages": ["OpenCL grid calculation", "OpenCL parallel Monte Carlo search"],
                "kernel_dir": str(self._kernel_dir), "tmpdir": str(self._tmpdir),
                "trace_library": str(self._trace_library) if self._trace_library else None,
                "timing_note": "Python monotonic wall time; OpenCL profiling timestamps may be invalid on this driver; tracing adds synchronization",
                "last_run": dict(self.last_run_metadata)}
