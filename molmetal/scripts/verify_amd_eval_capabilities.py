"""Record actual AMD execution evidence for installed evaluation backends."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def torch_admet() -> dict:
    import torch
    from molmetal.validation.admet_runner import _get_admet_model, _predict_admet_ai

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("ROCm PyTorch GPU unavailable")
    prop = torch.cuda.get_device_properties(0)
    record = {
        "torch_version": torch.__version__, "hip_version": torch.version.hip,
        "visible_devices": torch.cuda.device_count(), "device_index": 0,
        "device_name": prop.name, "gcn_arch": prop.gcnArchName,
        "total_memory_bytes": prop.total_memory,
        "visibility": {key: os.environ.get(key) for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
    }
    model = _get_admet_model()
    if model is None:
        raise RuntimeError("ADMET-AI model loading failed")
    executions = []
    handles = []

    def observe(module, args):
        executions.append(str(next(module.parameters()).device))

    for ensemble in model.model_lists:
        for network in ensemble:
            handles.append(network.register_forward_pre_hook(observe))
    try:
        scores = _predict_admet_ai("CCO")
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if scores is None:
        raise RuntimeError("Direct learned ADMET-AI inference failed; no RDKit fallback counted")
    record.update(
        inference_path="molmetal.validation.admet_runner._predict_admet_ai",
        smiles="CCO", n_properties=len(scores),
        neural_forward_parameter_devices=executions,
        neural_forward_calls=len(executions),
        examples={key: scores[key] for key in ("hERG", "AMES", "BBB_Martins") if key in scores},
        working=bool(executions) and all(d.startswith("cuda") for d in executions)
                and all(math.isfinite(value) for value in scores.values()),
        scope="actual ROCm learned ADMET inference; not GPU docking",
    )
    return record


def openmm_opencl() -> dict:
    import openmm as mm
    from openmm import unit

    platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    platform = mm.Platform.getPlatformByName("OpenCL")
    system = mm.System()
    system.addParticle(12)
    force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", 100)
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform,
                         {"Precision": "mixed", "OpenCLPlatformIndex": "0", "DeviceIndex": "0"})
    context.setPositions([[0.1, 0, 0]])
    context.setVelocities([[0, 0, 0]])
    before = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    integrator.step(5)
    after = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    properties = {name: platform.getPropertyValue(context, name)
                  for name in ("DeviceName", "DeviceIndex", "OpenCLPlatformName", "OpenCLPlatformIndex", "Precision")}
    return {
        "openmm_version": mm.__version__, "available_platforms": platforms,
        "plugin_load_failures": list(mm.Platform.getPluginLoadFailures()),
        "selected_platform": context.getPlatform().getName(), "context_properties": properties,
        "system": "one 12 amu particle, harmonic external potential k=100, x0=0.1 nm",
        "steps": 5, "dt_ps": 0.001, "potential_before_kj_mol": before,
        "potential_after_kj_mol": after,
        "working": "AMD" in properties["OpenCLPlatformName"] and properties["DeviceName"] == "gfx1101"
                   and math.isfinite(after) and 0 < after < before,
        "scope": "AMD OpenCL dynamics/energy smoke only; no protein force-field or docking validation",
    }


def gnina_probe() -> dict:
    names = ("gnina", "autodock_gpu", "autodock_gpu_128wi", "Vina-GPU", "QuickVina2-GPU-2-1")
    found = {name: shutil.which(name) for name in names}
    result = {"binaries_on_path": found,
              "vendored_quickvina_gpu_path": str(ROOT / "molmetal/references/SynFlowNet/bin/QuickVina2-GPU-2-1"),
              "vendored_quickvina_gpu_exists": (ROOT / "molmetal/references/SynFlowNet/bin/QuickVina2-GPU-2-1").is_file(),
              "working_amd_gpu_docking": False}
    if found["gnina"]:
        receptor = ROOT / "molmetal/reports/crossdocked_test001_receptor/receptor.pdbqt"
        from molmetal.scripts.prepare_crossdocked_receptor import manifest_pair
        _, ligand = manifest_pair(ROOT / "molmetal/data/crossdocked100_manifest.csv", "test_001")
        command = [found["gnina"], "--receptor", str(receptor), "--ligand", str(ligand),
                   "--score_only", "--cpu", "1", "--seed", "42"]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
        result.update(command=command, returncode=proc.returncode,
                      stdout=proc.stdout[-10000:], stderr=proc.stderr[-10000:],
                      note="gnina.real is a CUDA-linked build; successful CPU fallback is not AMD GPU support")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["torch_admet", "openmm_opencl", "gnina_probe"])
    parser.add_argument("--out", type=Path, default=ROOT / "molmetal/reports/amd_eval_capabilities.json")
    args = parser.parse_args()
    if args.worker:
        try:
            record = globals()[args.worker]()
        except Exception as exc:
            record = {"working": False, "error": f"{type(exc).__name__}: {exc}"}
        args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return 0
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "installed capabilities only; no new dependency installation", "checks": {}}
    with tempfile.TemporaryDirectory(prefix="amd-capabilities-") as directory:
        for name in ("torch_admet", "openmm_opencl", "gnina_probe"):
            output = Path(directory) / f"{name}.json"
            command = [sys.executable, "-m", "molmetal.scripts.verify_amd_eval_capabilities",
                       "--worker", name, "--out", str(output)]
            try:
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                record = json.loads(output.read_text()) if output.exists() else {"working": False}
                record.update(worker_returncode=process.returncode, worker_stderr=process.stderr[-5000:])
            except subprocess.TimeoutExpired:
                record = {"working": False, "error": "Worker timed out after 120 seconds"}
            report["checks"][name] = record
            print(f"{name}: completed", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
