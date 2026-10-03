#!/usr/bin/env python3
"""check_blackwell_env.py — R18 Blackwell GPU env probe + JSON dump.

Read-only. Parses nvidia-smi, finds any kanzi_venv python process, and
probes candidate venvs for torch+CUDA+sm_120 (Blackwell) support. Outputs
a structured JSON summary (default: molmetal/reports/wf_blackwell_retrain/
gpu_env_check.json). Recommends GPU 1 (RTX 5090) for R18 work when kanzi
is actively using GPU 0.
"""

from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

REPORT_DIR = Path("/home/hugo/codes/mol-metal/molmetal/reports/wf_blackwell_retrain")
CANDIDATE_VENVS = [
    Path("/home/hugo/codes/mol-metal/.venv/bin/python"),
    Path("/home/hugo/codes/mol-metal/.venv/bin/python"),
    Path("/home/hugo/.venv-flowmol311/bin/python"),
]
KANZI_NEEDLES = ("kanzi_venv/bin/python", "sweep_kanzi")


def _run(cmd, timeout=30):
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    except Exception as e:
        return -1, "", f"{type(e).__name__}: {e}"


def _parse_smi():
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,"
            "memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if rc != 0:
        return gpus, err.strip()
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8:
            continue
        gpus.append(
            {
                "index": int(p[0]),
                "name": p[1],
                "vram_total_mb": int(p[2]),
                "vram_used_mb": int(p[3]),
                "vram_free_mb": int(p[4]),
                "gpu_util_pct": int(p[5]),
                "temperature_c": int(p[6]),
                "power_draw_w": float(p[7]),
            }
        )
    return gpus, None


def _detect_kanzi():
    """Find the actual kanzi python interpreter (not the zsh wrapper)."""
    rc, out, _ = _run(["ps", "-eo", "pid,user,pcpu,pmem,etime,stat,args"])
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        cmd = parts[6]
        if not any(n in cmd for n in KANZI_NEEDLES):
            continue
        argv0 = cmd.split(None, 1)[0]
        if not argv0.endswith("kanzi_venv/bin/python"):  # skip zsh wrapper
            continue
        return {
            "pid": int(parts[0]),
            "user": parts[1],
            "pcpu": float(parts[2]),
            "etime": parts[4],
            "stat": parts[5],
            "cmdline_short": cmd[:200],
            "cwd": os.readlink(f"/proc/{parts[0]}/cwd")
            if os.path.exists(f"/proc/{parts[0]}/cwd")
            else None,
            "is_running": parts[5].startswith("R"),
        }
    return None


def _per_process_vram(pid):
    rc, out, _ = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return None
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and int(p[0]) == pid:
            return int(p[1])
    return None


def _probe_torch(py):
    if not py.exists():
        return {"venv": str(py), "ok": False, "reason": "missing"}
    code = (
        "import torch; ok=torch.cuda.is_available(); "
        "n=torch.cuda.get_device_name(0) if ok else ''; "
        "c=torch.cuda.get_device_capability(0) if ok else None; "
        "print('|'.join([torch.__version__, torch.version.cuda or '', "
        "str(ok), n, str(c), str(c==(12,0)), "
        "str(torch.cuda.device_count() if ok else 0)]))"
    )
    rc, out, err = _run([str(py), "-c", code], timeout=30)
    if rc != 0:
        return {
            "venv": str(py),
            "ok": False,
            "reason": "probe failed",
            "stderr": err.strip()[:200],
        }
    f = out.strip().split("|")
    if len(f) < 7:
        return {"venv": str(py), "ok": False, "reason": "parse", "raw": out}
    return {
        "venv": str(py),
        "ok": f[2] == "True" and f[5] == "True",
        "torch": f[0],
        "cuda_runtime": f[1],
        "device_name": f[3],
        "compute_capability": f[4],
        "sm120_blackwell": f[5] == "True",
        "device_count": int(f[6]),
    }


def _actions(gpus, kanzi, kanzi_vram, tp):
    a = []
    if not gpus:
        return ["ERROR: no GPUs visible — check nvidia-smi / driver"]
    if not any("Blackwell" in g["name"] for g in gpus):
        a.append("WARN: no Blackwell GPU visible")
    if not tp.get("ok"):
        a.append(
            "ERROR: no torch+CUDA+sm_120 venv — source env.sh "
            "(routes uv to /home/hugo/codes/mol-metal/.venv)"
        )
    else:
        a.append(
            f"OK: torch {tp['torch']} + CUDA {tp['cuda_runtime']} "
            f"+ sm_120 on {tp['device_name']}"
        )
    if kanzi is None:
        a.append("OK: no kanzi_venv process occupying GPU")
    else:
        vram = (kanzi_vram or 0) / 1024.0
        util = next((g["gpu_util_pct"] for g in gpus if g["index"] == 0), None)
        if kanzi.get("is_running") or (util is not None and util >= 1):
            a.append(
                f"INFO: kanzi PID {kanzi['pid']} actively running "
                f"(stat={kanzi['stat']}, CPU={kanzi['pcpu']:.1f}%, "
                f"GPU0 util={util}%); R18 should run on GPU 1 (RTX 5090) "
                f"to avoid contention"
            )
        elif vram > 5.0:
            a.append(
                f"RECOMMEND: kanzi PID {kanzi['pid']} holds {vram:.1f} GB "
                f"on GPU 0 but util=0% (idle) — close kanzi or run R18 on GPU 1"
            )
        else:
            a.append(
                f"OK: kanzi PID {kanzi['pid']} lightly holds {vram:.1f} GB on GPU 0"
            )
    return a


def main():
    out_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT_DIR / "gpu_env_check.json"
    )
    gpus, smi_err = _parse_smi()
    kanzi = _detect_kanzi()
    kanzi_vram = _per_process_vram(kanzi["pid"]) if kanzi else None
    tp = {"ok": False, "reason": "no candidate probed"}
    for c in CANDIDATE_VENVS:
        r = _probe_torch(c)
        if r.get("ok"):
            tp = r
            break
        if tp.get("reason") == "no candidate probed":
            tp = r
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "nvidia_smi_rc": 0 if smi_err is None else -1,
            "nvidia_smi_error": smi_err,
        },
        "gpu_count": len(gpus),
        "gpu_names": [g["name"] for g in gpus],
        "vram_total_gb": round(sum(g["vram_total_mb"] for g in gpus) / 1024.0, 1),
        "vram_used_gb": round(sum(g["vram_used_mb"] for g in gpus) / 1024.0, 1),
        "vram_free_gb": round(sum(g["vram_free_mb"] for g in gpus) / 1024.0, 1),
        "gpu_util_pct": [g["gpu_util_pct"] for g in gpus],
        "gpu_details": [
            {
                "index": g["index"],
                "name": g["name"],
                "vram_total_gb": round(g["vram_total_mb"] / 1024.0, 1),
                "vram_used_gb": round(g["vram_used_mb"] / 1024.0, 1),
                "vram_free_gb": round(g["vram_free_mb"] / 1024.0, 1),
                "util_pct": g["gpu_util_pct"],
                "temp_c": g["temperature_c"],
                "power_w": g["power_draw_w"],
            }
            for g in gpus
        ],
        "kanzi_process_pid": kanzi["pid"] if kanzi else None,
        "kanzi_vram_gb": round((kanzi_vram or 0) / 1024.0, 2) if kanzi else None,
        "kanzi_pcpu": kanzi["pcpu"] if kanzi else None,
        "kanzi_stat": kanzi["stat"] if kanzi else None,
        "kanzi_etime": kanzi["etime"] if kanzi else None,
        "kanzi_cwd": kanzi["cwd"] if kanzi else None,
        "mol_metal_vram_gb": (
            round((gpus[0]["vram_used_mb"] - (kanzi_vram or 0)) / 1024.0, 2)
            if gpus and kanzi_vram is not None
            else (round(gpus[0]["vram_used_mb"] / 1024.0, 2) if gpus else None)
        ),
        "torch_probe": tp,
        "action_plan": _actions(gpus, kanzi, kanzi_vram, tp),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(json.dumps(summary["action_plan"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
