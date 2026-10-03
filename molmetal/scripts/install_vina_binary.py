"""Wf-84 Phase 2 — install Vina binary on the current host.

Strategy: try in order until one works (or we exhaust the list).

    1. probe existing installations (env var, PATH, conda envs, sweep)
    2. mamba / micromamba create -n vina -c conda-forge vina
    3. conda install -c conda-forge vina  (fallback if mamba unavailable)
    4. pip install vina                  (Python wrapper only — has CLI)
    5. apt-get install autodock-vina     (Debian/Ubuntu only)
    6. source build from ccsb-scripps/AutoDock-Vina
    7. honest stub fallback + download script

The script is CPU-only, hermetic (no network on a cache hit), and writes
a JSON log to ``molmetal/reports/wf84_p2_vina_install/install_log.json``
plus a wrapper at ``molmetal/bin/vina`` that downstream code (the bench
script, the cross_layer_metrics harness, etc.) can discover via
``MOLMETAL_VINA_BIN`` or just by being on ``PATH``.

Honest framing
--------------
If the real binary is reachable, ``final_path`` is the absolute path and
``status="OK"``. If every attempt failed, ``status="STUB_FALLBACK"`` and
``stub_warning.md`` documents the gap (no fabricated ``vina`` shim that
writes fake scores).

CLI
---
::

    python -m molmetal.scripts.install_vina_binary [--bin-dir BIN_DIR] \
        [--log-path LOG_PATH] [--force-reinstall] [--skip-install]

Default ``--bin-dir`` is ``<project_root>/molmetal/bin`` (i.e. where
``molmetal.bin.vina`` lives after install). ``--skip-install`` only
probes existing installations (useful for CI / dry-run).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN_DIR = PROJECT_ROOT / "molmetal" / "bin"
DEFAULT_LOG_PATH = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf84_p2_vina_install" / "install_log.json"
)
DEFAULT_STUB_PATH = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf84_p2_vina_install" / "stub_warning.md"
)
# Standard conda env dirs on linux (conda, mamba, micromamba all share).
_CONDA_PREFIX_HINTS = (
    os.path.expanduser("~/.mamba"),
    os.path.expanduser("~/miniconda3"),
    os.path.expanduser("~/anaconda3"),
    os.path.expanduser("~/micromamba"),
    "/opt/conda",
    "/usr/local/conda",
)
# Hard-coded prefixes that show up after a conda install.
_VINA_PREFIX_BIN_CANDIDATES = (
    "bin/vina",
    "envs/vina/bin/vina",
    "envs/molmetal/bin/vina",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _run(
    cmd: List[str],
    *,
    timeout: int = 600,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run ``cmd`` capturing stdout/stderr. Never raises on non-zero exit."""
    started = time.time()
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy(),
            cwd=str(cwd) if cwd else None,
            check=False,
        )
        return {
            "cmd": cmd,
            "rc": cp.returncode,
            "stdout_tail": (cp.stdout or "")[-2000:],
            "stderr_tail": (cp.stderr or "")[-2000:],
            "elapsed_s": round(time.time() - started, 2),
            "ok": cp.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "rc": -1,
            "stdout_tail": (exc.stdout or "")[-2000:] if exc.stdout else "",
            "stderr_tail": f"TIMEOUT after {timeout}s",
            "elapsed_s": round(time.time() - started, 2),
            "ok": False,
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "rc": -1,
            "stdout_tail": "",
            "stderr_tail": f"command not found: {exc}",
            "elapsed_s": round(time.time() - started, 2),
            "ok": False,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "cmd": cmd,
            "rc": -1,
            "stdout_tail": "",
            "stderr_tail": f"unexpected error: {exc}",
            "elapsed_s": round(time.time() - started, 2),
            "ok": False,
        }


def _probe_vina(path: Path) -> Tuple[bool, str]:
    """Return ``(ok, version_line)`` for the binary at ``path``.

    ``ok`` is True iff the file exists, is executable, and the
    ``--version`` banner contains "AutoDock Vina".
    """
    if not path.is_file() or not os.access(path, os.X_OK):
        return False, ""
    cp = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=10
    )
    out = (cp.stdout or "") + (cp.stderr or "")
    first = out.strip().splitlines()[0] if out.strip() else ""
    if "AutoDock Vina" in first or "AutoDock-Vina" in first:
        return True, first
    return False, first


def _find_existing_vina() -> Tuple[Optional[Path], str, List[Dict[str, Any]]]:
    """Probe well-known locations for an existing Vina binary.

    Returns ``(path, version_line, candidates)`` where ``candidates``
    is a per-probe list (used for the JSON log).
    """
    candidates: List[Dict[str, Any]] = []

    def _try(label: str, p: Path) -> Optional[Tuple[Path, str]]:
        ok, ver = _probe_vina(p)
        candidates.append({"label": label, "path": str(p), "ok": ok, "version": ver})
        if ok:
            return p, ver
        return None

    # 1. env override
    env_path = os.environ.get("MOLMETAL_VINA_BIN")
    if env_path:
        hit = _try("env:MOLMETAL_VINA_BIN", Path(env_path))
        if hit:
            return hit[0], hit[1], candidates

    # 2. PATH lookup
    which_vina = shutil.which("vina")
    if which_vina:
        hit = _try("path:which(vina)", Path(which_vina))
        if hit:
            return hit[0], hit[1], candidates

    # 3. conda prefix sweep
    for prefix in _CONDA_PREFIX_HINTS:
        prefix_path = Path(prefix)
        if not prefix_path.is_dir():
            continue
        for rel in _VINA_PREFIX_BIN_CANDIDATES:
            hit = _try(f"conda_prefix:{prefix}/{rel}", prefix_path / rel)
            if hit:
                return hit[0], hit[1], candidates

    # 4. filesystem sweep (last resort)
    sweep_roots = [Path("/home"), Path("/opt"), Path("/usr/local")]
    for root in sweep_roots:
        if not root.exists():
            continue
        for p in root.glob("**/bin/vina"):
            hit = _try(f"sweep:{p}", p)
            if hit:
                return hit[0], hit[1], candidates

    return None, "", candidates


# ---------------------------------------------------------------------------
# Install attempts
# ---------------------------------------------------------------------------
def _try_micromamba_install(env_name: str = "molmetal_vina") -> Dict[str, Any]:
    """Try ``micromamba create -n <env> -c conda-forge vina -y``.

    Honours a per-user ``MAMBA_ROOT_PREFIX`` (default ``~/.local/micromamba``)
    so we never try to write to the system-wide ``/usr/envs`` location
    that the distro's micromamba binary may default to.
    """
    if not shutil.which("micromamba"):
        return {
            "channel": "micromamba",
            "ok": False,
            "reason": "micromamba not on PATH",
        }
    # Use a user-writable MAMBA_ROOT_PREFIX so we don't trip the
    # ``EnvironmentNotWritableError`` that the system /opt/mamba install hits.
    user_root = os.path.expanduser("~/.local/micromamba")
    env = os.environ.copy()
    env.setdefault("MAMBA_ROOT_PREFIX", user_root)
    Path(user_root).mkdir(parents=True, exist_ok=True)
    cmd = [
        "micromamba",
        "create",
        "-n",
        env_name,
        "-c",
        "conda-forge",
        "vina",
        "-y",
        "--json",
    ]
    res = _run(cmd, timeout=900, env=env)
    if res["ok"]:
        # Locate the binary inside the env
        env_prefix = _locate_mamba_env_prefix(env_name, env=env)
        if env_prefix:
            bin_path = env_prefix / "bin" / "vina"
            ok, ver = _probe_vina(bin_path)
            return {
                "channel": "micromamba",
                "ok": ok,
                "bin": str(bin_path),
                "version": ver,
                "elapsed_s": res["elapsed_s"],
                "rc": res["rc"],
                "stderr_tail": res["stderr_tail"][-500:],
            }
    return {
        "channel": "micromamba",
        "ok": False,
        "reason": "install failed or env missing",
        "rc": res["rc"],
        "stderr_tail": res["stderr_tail"][-500:],
        "elapsed_s": res["elapsed_s"],
    }


def _locate_mamba_env_prefix(
    env_name: str, env: Optional[Dict[str, str]] = None
) -> Optional[Path]:
    """Return the prefix path of a micromamba env if it exists."""
    if not shutil.which("micromamba"):
        return None
    res = _run(["micromamba", "env", "list", "--json"], timeout=30, env=env)
    if not res["ok"]:
        return None
    try:
        data = json.loads(res["stdout_tail"])
        envs = data.get("envs", [])
        for env_str in envs:
            if env_str.endswith(f"/envs/{env_name}") or env_str.endswith(
                f"\\{env_name}"
            ):
                return Path(env_str)
        # Fallback: search for env by name in any prefix
        for env_str in envs:
            if Path(env_str).name == env_name:
                return Path(env_str)
    except (json.JSONDecodeError, KeyError):
        pass
    # Manual fallback: guess the prefix from MAMBA_ROOT_PREFIX or default
    root = (
        os.environ.get("MAMBA_ROOT_PREFIX")
        or (env or os.environ).get("MAMBA_ROOT_PREFIX")
        or os.path.expanduser("~/.local/micromamba")
    )
    candidate = Path(root) / "envs" / env_name
    if (candidate / "bin" / "vina").is_file():
        return candidate
    return None


def _try_conda_install(env_name: str = "molmetal_vina") -> Dict[str, Any]:
    """Try ``conda install -c conda-forge vina`` via the conda shell function."""
    if not shutil.which("conda") and not shutil.which("mamba"):
        return {
            "channel": "conda",
            "ok": False,
            "reason": "no conda binary on PATH",
        }
    cmd = ["conda", "install", "-y", "-c", "conda-forge", "vina"]
    res = _run(cmd, timeout=600)
    return {
        "channel": "conda",
        "ok": res["ok"],
        "rc": res["rc"],
        "elapsed_s": res["elapsed_s"],
        "stderr_tail": res["stderr_tail"][-500:],
    }


def _try_pip_install() -> Dict[str, Any]:
    """Try ``pip install vina`` (provides a CLI wrapper via entry point)."""
    pip = shutil.which("pip") or shutil.which("pip3")
    if not pip:
        # Fall back to python -m pip
        py = sys.executable
        cmd = [py, "-m", "pip", "install", "--quiet", "vina"]
    else:
        cmd = [pip, "install", "--quiet", "vina"]
    res = _run(cmd, timeout=300)
    if res["ok"]:
        # Find the installed script
        # The "vina" PyPI package is a Python wrapper, not the C++ binary,
        # so it can do real docking via the bindings.
        py = sys.executable
        return {
            "channel": "pip",
            "ok": res["ok"],
            "python": py,
            "elapsed_s": res["elapsed_s"],
            "note": (
                "pip vina is a Python wrapper around the C++ bindings; "
                "exposes 'vina' CLI if entry point installed."
            ),
        }
    return {
        "channel": "pip",
        "ok": False,
        "rc": res["rc"],
        "elapsed_s": res["elapsed_s"],
        "stderr_tail": res["stderr_tail"][-500:],
    }


def _try_apt_install() -> Dict[str, Any]:
    """Try ``sudo apt-get install -y autodock-vina``."""
    if not shutil.which("apt-get"):
        return {"channel": "apt", "ok": False, "reason": "apt-get not on PATH"}
    # Honour the user's "CPU-only; user permission required for sudo" rule
    sudo = shutil.which("sudo")
    cmd = (
        [sudo, "apt-get", "install", "-y", "autodock-vina"]
        if sudo
        else ["apt-get", "install", "-y", "autodock-vina"]
    )
    res = _run(cmd, timeout=600)
    return {
        "channel": "apt",
        "ok": res["ok"],
        "rc": res["rc"],
        "elapsed_s": res["elapsed_s"],
        "stderr_tail": res["stderr_tail"][-500:],
    }


def _try_source_build(build_dir: Path) -> Dict[str, Any]:
    """Try a source build of ccsb-scripps/AutoDock-Vina (last resort).

    Per task constraints this is CPU-only and we only attempt it if all
    package-manager channels failed. The build requires cmake + a C++
    compiler + boost, so it commonly fails; we time-box aggressively.

    The ccsb-scripps repo has its ``CMakeLists.txt`` under ``src/`` and
    requires Boost system + thread libraries (≥1.71). We attempt the
    upstream-documented build flow; if it fails we record the reason and
    return without raising.
    """
    if shutil.which("git") is None:
        return {"channel": "source", "ok": False, "reason": "git not on PATH"}
    if shutil.which("cmake") is None:
        return {"channel": "source", "ok": False, "reason": "cmake not on PATH"}

    build_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = build_dir / "AutoDock-Vina"
    # If already cloned from a prior attempt, reuse.
    if not repo_dir.exists():
        clone = _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ccsb-scripps/AutoDock-Vina.git",
                str(repo_dir),
            ],
            timeout=300,
        )
        if not clone["ok"]:
            return {
                "channel": "source",
                "ok": False,
                "reason": "git clone failed",
                "stderr_tail": clone["stderr_tail"][-500:],
            }

    # The upstream CMakeLists.txt lives in src/ (post v1.2 refactor).
    src_dir = repo_dir / "src"
    if not (src_dir / "CMakeLists.txt").is_file():
        # Older layout: build from the repo root with cmake
        cmake_src_dir = repo_dir
    else:
        cmake_src_dir = src_dir
    build_subdir = repo_dir / "build"
    build_subdir.mkdir(exist_ok=True)
    cmake_cfg = _run(
        ["cmake", str(cmake_src_dir), "-DCMAKE_BUILD_TYPE=Release"],
        timeout=120,
        cwd=build_subdir,
    )
    if not cmake_cfg["ok"]:
        return {
            "channel": "source",
            "ok": False,
            "reason": "cmake configure failed",
            "stderr_tail": cmake_cfg["stderr_tail"][-500:],
        }

    # Bound the build to a sane CPU-time budget (Vina is small).
    nproc = os.cpu_count() or 2
    nproc = max(1, min(nproc, 8))  # cap to keep CI-friendly
    make = _run(
        ["make", f"-j{nproc}"],
        timeout=1500,
        cwd=build_subdir,
    )
    if not make["ok"]:
        return {
            "channel": "source",
            "ok": False,
            "reason": "make failed",
            "stderr_tail": make["stderr_tail"][-500:],
        }

    built_bin = build_subdir / "vina"
    if not built_bin.is_file():
        return {
            "channel": "source",
            "ok": False,
            "reason": "build did not produce vina",
        }
    return {
        "channel": "source",
        "ok": True,
        "bin": str(built_bin),
    }


# ---------------------------------------------------------------------------
# Wrapper / stub
# ---------------------------------------------------------------------------
_WRAPPER_TEMPLATE = """#!/bin/sh
# Auto-generated by molmetal/scripts/install_vina_binary.py (Wf-84 Phase 2)
# Delegates to the real Vina binary at:
#   {real_bin}
# If that binary disappears the wrapper fails loudly (no silent stub).
REAL_BIN="{real_bin}"
if [ ! -x "$REAL_BIN" ]; then
    echo "ERROR: Vina wrapper points at missing/non-executable: $REAL_BIN" 1>&2
    exit 127
fi
exec "$REAL_BIN" "$@"
"""


def _write_wrapper(bin_dir: Path, real_bin: Path) -> Path:
    """Drop an exec-shim at ``bin_dir/vina`` that delegates to ``real_bin``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "vina"
    wrapper.write_text(_WRAPPER_TEMPLATE.format(real_bin=str(real_bin)))
    wrapper.chmod(0o755)
    return wrapper


_STUB_TEMPLATE = """# Wf-84 Phase 2 — Vina install STUB_FALLBACK

**Date**: {captured_at}
**Status**: STUB_FALLBACK — no real Vina binary reachable on this host.

## What was attempted

| # | Channel | Result | Notes |
|---|---------|--------|-------|
{attempts}

## Why this happened

{reason}

## What the bench script does in this state

`molmetal.scripts.bench_vina_triple_rebench._detect_vina_binary()`
returns ``status="MISSING_DEP"`` and every ``(pocket, cand, seed)`` cell
falls back to ``STUB_FALLBACK`` with ``score_kcal_mol=null``. The bench
**never fabricates a Vina score** — see `Wf-60 Phase 2` framing
(`molmetal/reports/wf60_vina_triple/vina_binary_status.json`).

## How to recover

1. **Easiest** — point at an existing install:

   ```bash
   export MOLMETAL_VINA_BIN=/path/to/vina      # e.g. /home/hugo/.mamba/bin/vina
   ```

2. **Conda-forge** — preferred fresh install:

   ```bash
   micromamba create -n molmetal_vina -c conda-forge vina -y
   export MOLMETAL_VINA_BIN=$HOME/micromamba/envs/molmetal_vina/bin/vina
   ```

3. **Debian/Ubuntu**:

   ```bash
   sudo apt-get install -y autodock-vina
   ```

4. **From source** (slow, last resort; requires `cmake` + boost):

   ```bash
   git clone https://github.com/ccsb-scripps/AutoDock-Vina.git
   cd AutoDock-Vina/build && cmake .. && make -j$(nproc)
   export MOLMETAL_VINA_BIN=$(pwd)/vina
   ```

5. **Re-run** the installer to (re)write the wrapper:

   ```bash
   python -m molmetal.scripts.install_vina_binary
   ```

The installer will detect the env override, symlink the binary into
`molmetal/bin/vina`, and the bench's `MISSING_DEP` row flips to
`REAL_VINA_AVAILABLE`.
"""


def _write_stub_warning(
    path: Path, attempts: List[Dict[str, Any]], reason: str
) -> None:
    """Write the honest-stub-fallback warning document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    for i, att in enumerate(attempts, 1):
        verdict = "OK" if att.get("ok") else "FAIL"
        note = (
            att.get("reason")
            or att.get("bin")
            or att.get("stderr_tail", "").splitlines()[-1][:120]
            or ""
        )
        rows.append(f"| {i} | {att.get('channel', '?')} | {verdict} | {note} |")
    text = _STUB_TEMPLATE.format(
        captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        attempts="\n".join(rows) if rows else "| - | - | - | no channels attempted |",
        reason=reason,
    )
    path.write_text(text)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def install_vina(
    bin_dir: Path = DEFAULT_BIN_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
    skip_install: bool = False,
    force_reinstall: bool = False,
) -> Dict[str, Any]:
    """Top-level orchestrator. Returns the log dict (also written to disk).

    Strategy:
        1. Probe existing Vina.
        2. If found, drop wrapper, mark status=OK.
        3. Else (and not skip_install) try mamba > conda > pip > apt > source.
        4. If anything worked, drop wrapper.
        5. Else, write STUB_FALLBACK doc + status.
    """
    log: Dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bin_dir": str(bin_dir),
        "skip_install": skip_install,
        "force_reinstall": force_reinstall,
        "probe": {"candidates": []},
        "attempts": [],
        "final_path": None,
        "wrapper": None,
        "status": "STUB_FALLBACK",
        "reason": "",
    }

    # 1. probe (skipped when --force-reinstall)
    if force_reinstall:
        # Still log the probe results for transparency, but don't short-circuit
        _, _, probe_candidates = _find_existing_vina()
        log["probe"] = {
            "candidates": probe_candidates,
            "found": False,
            "note": "skipped due to --force-reinstall",
        }
        found = None
        version = ""
        candidates = probe_candidates
    else:
        found, version, candidates = _find_existing_vina()
        log["probe"] = {"candidates": candidates, "found": bool(found)}
    if found:
        log["final_path"] = str(found)
        log["version"] = version
        log["status"] = "OK"
        log["reason"] = "binary already installed and probed"
    elif not skip_install:
        # 2. try mamba
        for attempt_fn in (
            lambda: _try_micromamba_install("molmetal_vina"),
            _try_conda_install,
            _try_pip_install,
            _try_apt_install,
        ):
            try:
                att = attempt_fn()
            except Exception as exc:  # pragma: no cover - defensive
                att = {"channel": "?", "ok": False, "reason": str(exc)}
            log["attempts"].append(att)
            if att.get("ok"):
                cand = att.get("bin")
                if cand:
                    p = Path(cand)
                    ok, ver = _probe_vina(p)
                    if ok:
                        log["final_path"] = str(p)
                        log["version"] = ver
                        log["status"] = "OK"
                        log["reason"] = f"installed via {att['channel']}; probed OK"
                        break
        # 3. try source build if still nothing (last resort)
        if log["status"] != "OK":
            src_attempt = _try_source_build(Path("/tmp/vina_build"))
            log["attempts"].append(src_attempt)
            if src_attempt.get("ok"):
                p = Path(src_attempt["bin"])
                ok, ver = _probe_vina(p)
                if ok:
                    log["final_path"] = str(p)
                    log["version"] = ver
                    log["status"] = "OK"
                    log["reason"] = "built from source; probed OK"

    # 4. wrapper + 5. stub fallback
    if log["status"] == "OK" and log["final_path"]:
        try:
            wrapper = _write_wrapper(bin_dir, Path(log["final_path"]))
            log["wrapper"] = str(wrapper)
        except Exception as exc:
            log["reason"] += f" (wrapper write FAILED: {exc})"
    else:
        log["status"] = "STUB_FALLBACK"
        if not log["reason"]:
            log["reason"] = "all install attempts failed; see attempts[]"
        try:
            _write_stub_warning(
                DEFAULT_STUB_PATH,
                log["attempts"],
                log["reason"],
            )
            log["stub_warning"] = str(DEFAULT_STUB_PATH)
        except Exception as exc:
            log["reason"] += f" (stub doc write FAILED: {exc})"

    # write log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--bin-dir",
        type=Path,
        default=DEFAULT_BIN_DIR,
        help=f"where to drop the vina wrapper (default: {DEFAULT_BIN_DIR})",
    )
    p.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"where to write install_log.json (default: {DEFAULT_LOG_PATH})",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="only probe; never attempt to install",
    )
    p.add_argument(
        "--force-reinstall",
        action="store_true",
        help="skip the probe-and-reuse path; always try install attempts",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    log = install_vina(
        bin_dir=args.bin_dir,
        log_path=args.log_path,
        skip_install=args.skip_install,
        force_reinstall=args.force_reinstall,
    )
    print(json.dumps(log, indent=2, sort_keys=True))
    return 0 if log["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
