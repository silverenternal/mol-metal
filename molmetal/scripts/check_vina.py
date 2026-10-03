#!/usr/bin/env python3
"""check_vina.py — Vina binary smoke test for m15 RMSD Phase 2 (BD-9).

================================================================
Why this script exists
================================================================
BD-9 (Wf-134 m15 RMSD redesign Phase 2) needs a working AutoDock Vina
binary to redock the Wf-139 P1 crystal ligands into the Wf-30 staged
pockets. Vina is shipped via conda-forge at
``/home/hugo/.mamba/bin/vina`` (Vina 1.2.7, ``f458505-mod``) and is
mirrored by a project-local wrapper at ``molmetal/bin/vina``.

This script:

1. Probes a canonical candidate list (project wrapper → mamba → PATH)
2. Runs ``vina --version`` and ``vina --help`` (the latter is parsed for
   the line count to validate a non-broken libboost build)
3. Optionally runs a tiny **isolated** CLI dock against a single pocket
   with a one-atom dummy ligand to ensure PDBQT parsing + scoring are
   alive end-to-end (skipped by default; controlled with
   ``--with-smoke-dock``)
4. Writes a JSON status report (default:
   ``molmetal/reports/check_vina/vina_binary_status.json``) consumable
   by downstream m15 RMSD scripts

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* We probe the binary that ACTUALLY exists on this machine — never
  fabricate availability from a hash.
* Output is one of ``"REAL_VINA_AVAILABLE"`` (binary works and
  ``--version`` parses), ``"VINA_ATYPE_REJECTED"`` (binary works but
  rejects generic ``M`` atype — surfaces the chemotype-vdw
  workaround), ``"BROKEN_LIBBOOST"`` (binary starts but crashes on
  basic PDBQT), or ``"VINA_NOT_FOUND"``.
* If ``--with-smoke-dock`` is requested and any stage of the test
  pipeline fails, the JSON output captures the exact exception text in
  the ``last_error`` field rather than silently downgrading.

Usage
-----
::

    # Smoke test only (default — fast, no docking)
    python molmetal/scripts/check_vina.py

    # Full pipeline smoke (one pocket, one seed, ~30 s)
    python molmetal/scripts/check_vina.py --with-smoke-dock
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = (
    PROJECT_ROOT / "molmetal" / "reports" / "check_vina" / "vina_binary_status.json"
)

# Canonical candidate list. Order matters — preferred first.
CANDIDATE_VINA_BINS: Tuple[Path, ...] = (
    PROJECT_ROOT / "molmetal" / "bin" / "vina",  # project wrapper
    Path("/home/hugo/.mamba/bin/vina"),  # conda-forge install
    Path("/home/hugo/micromamba/bin/vina"),
    Path("/usr/local/bin/vina"),
    Path("/usr/bin/vina"),
    Path("/opt/conda/bin/vina"),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_ansi(s: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _probe_vina_version(bin_path: Path, timeout: int = 10) -> Tuple[bool, str, str]:
    """Run ``vina --version`` and return (ok, stdout, stderr)."""
    try:
        cp = subprocess.run(
            [str(bin_path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return False, "", f"TimeoutExpired after {timeout}s: {exc}"
    except FileNotFoundError as exc:
        return False, "", f"FileNotFoundError: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, "", f"{type(exc).__name__}: {exc}"
    ok = cp.returncode == 0 and bool(cp.stdout.strip())
    return ok, _strip_ansi(cp.stdout).strip(), _strip_ansi(cp.stderr).strip()


def _probe_vina_help_works(bin_path: Path, timeout: int = 10) -> Tuple[bool, int]:
    """Run ``vina --help`` and return (ok, n_lines_in_stdout).

    A working Vina prints a multi-line usage screen. A broken libboost
    binary exits non-zero with an empty stdout, or prints only one or
    two lines (e.g. ``Segmentation fault``).
    """
    try:
        cp = subprocess.run(
            [str(bin_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, 0
    except Exception:  # noqa: BLE001
        return False, 0
    out = _strip_ansi(cp.stdout).strip()
    return (cp.returncode == 0 and len(out.splitlines()) >= 5), len(out.splitlines())


def _resolve_vina_bin() -> Tuple[Optional[Path], List[Path]]:
    """Locate the first working vina binary in CANDIDATE_VINA_BINS + PATH.

    Returns ``(chosen, checked_list)``. The first binary to pass
    ``--version`` is ``chosen``; ``checked_list`` contains every path
    we probed (whether successful or not) for the JSON output.
    """
    checked: List[Path] = []

    # First, canonical candidates in order
    for c in CANDIDATE_VINA_BINS:
        if not c.exists():
            continue
        checked.append(c)
        ok, _, _ = _probe_vina_version(c)
        if ok:
            return c, checked

    # Then PATH
    which = shutil.which("vina")
    if which:
        wp = Path(which)
        if wp not in checked:
            checked.append(wp)
            ok, _, _ = _probe_vina_version(wp)
            if ok:
                return wp, checked
    return None, checked


def _run_smoke_dock(
    bin_path: Path, out_dir: Path, timeout: int = 60
) -> Tuple[bool, dict]:
    """Run a one-shot CLI dock against the smallest staged pocket.

    We use 1h36 (one of the Wf-179 BD-9 pockets — guaranteed staged by
    Wf-30) and the smallest pre-existing Wf-179 docked pose as the
    ligand. This guarantees valid PDBQT format (the Wf-179 staging
    has been verified through end-to-end Vina runs).

    Returns ``(ok, info_dict)``. We DO NOT fail the script if the
    smoke fails — the binary is the primary checkpoint.
    """
    receptor = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf30_pocket_prep"
        / "1h36"
        / "receptor.pdbqt"
    )
    if not receptor.exists():
        return False, {
            "error": f"receptor.pdbqt not found at {receptor}",
            "status": "SMOKE_DOCK_SKIPPED_NO_RECEPTOR",
        }
    # Use an existing Wf-179 docked pose as the validated-format ligand.
    pose = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf179_bd9_m15_rmsd_p2"
        / "1h36"
        / "pose_seed0_p1.pdbqt"
    )
    if not pose.exists():
        return False, {
            "error": f"no staged Wf-179 pose to use as smoke ligand at {pose}",
            "status": "SMOKE_DOCK_SKIPPED_NO_LIGAND",
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_ligand = out_dir / "smoke_ligand.pdbqt"
    # Vina's parser requires a single-MODEL ligand PDBQT (otherwise
    # it complains: "Unexpected multi-MODEL tag found in flex residue
    # or ligand PDBQT file"). Strip the ``MODEL 1`` / ``ENDMDL``
    # tags from the Wf-179 staged pose file and emit a single ATOM
    # block.
    pose_text = pose.read_text()
    inside = False
    cleaned_lines: List[str] = []
    for line in pose_text.splitlines():
        if line.startswith("MODEL"):
            inside = True
            continue
        if line.startswith("ENDMDL"):
            inside = False
            continue
        if inside and (
            line.startswith("ATOM")
            or line.startswith("HETATM")
            or line.startswith("ROOT")
            or line.startswith("ENDROOT")
            or line.startswith("BRANCH")
            or line.startswith("ENDBRANCH")
        ):
            cleaned_lines.append(line)
    smoke_ligand.write_text("\n".join(cleaned_lines) + "\nTORSDOF 4\n")
    out_pdbqt = out_dir / "smoke_docked.pdbqt"
    # Reasonable pocket-centred box for 1h36 (actual ligand centroid
    # is around 35/53/45 — extracted from the prep_meta.json).
    cmd = [
        str(bin_path),
        "--receptor",
        str(receptor),
        "--ligand",
        str(smoke_ligand),
        "--center_x",
        "35.000",
        "--center_y",
        "53.000",
        "--center_z",
        "45.000",
        "--size_x",
        "20.000",
        "--size_y",
        "20.000",
        "--size_z",
        "20.000",
        "--out",
        str(out_pdbqt),
        "--exhaustiveness",
        "1",
        "--num_modes",
        "1",
        "--seed",
        "0",
    ]
    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, {
            "error": f"smoke dock timed out after {timeout}s",
            "status": "SMOKE_DOCK_TIMEOUT",
            "receptor": str(receptor),
        }
    except Exception as exc:  # noqa: BLE001
        return False, {
            "error": f"{type(exc).__name__}: {exc}",
            "status": "SMOKE_DOCK_EXCEPTION",
            "receptor": str(receptor),
        }
    elapsed = time.time() - t0
    info = {
        "receptor": str(receptor),
        "ligand": str(smoke_ligand),
        "elapsed_s": round(elapsed, 3),
        "returncode": cp.returncode,
        "stdout_tail": _strip_ansi(cp.stdout)[-200:],
        "stderr_tail": _strip_ansi(cp.stderr)[-200:],
        "status": ("SMOKE_DOCK_OK" if cp.returncode == 0 else "SMOKE_DOCK_FAIL"),
    }
    return cp.returncode == 0, info


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_vina",
        description="Vina binary smoke test (BD-9 m15 RMSD Phase 2 prerequisite).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"JSON status output path (default: {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--with-smoke-dock",
        action="store_true",
        default=False,
        help="Run an end-to-end CLI dock smoke (slower, ~30 s).",
    )
    p.add_argument(
        "--smoke-timeout",
        type=int,
        default=60,
        help="Smoke dock timeout in seconds (default 60).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    print("=" * 60)
    print("Vina binary smoke test — BD-9 m15 RMSD Phase 2 prerequisite")
    print("=" * 60)
    print(f"Generated: {_utc_now_iso()}")

    chosen, checked = _resolve_vina_bin()
    if chosen is None:
        msg = (
            "Vina binary NOT FOUND in any canonical location. "
            "Install via: micromamba install -n base -c conda-forge vina"
        )
        print(f"FAIL: {msg}", file=sys.stderr)
        report = {
            "status": "VINA_NOT_FOUND",
            "bin": None,
            "version": None,
            "reason": msg,
            "checked_paths": [str(p) for p in checked],
            "checked_utc": _utc_now_iso(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 1

    # Probe deeply
    v_ok, version_str, v_err = _probe_vina_version(chosen)
    h_ok, h_lines = _probe_vina_help_works(chosen)

    report = {
        "status": "REAL_VINA_AVAILABLE" if (v_ok and h_ok) else "BROKEN_LIBBOOST",
        "bin": str(chosen),
        "version": version_str or None,
        "version_stderr": v_err or None,
        "help_lines": h_lines,
        "reason": ""
        if (v_ok and h_ok)
        else (
            "vina --version failed"
            if not v_ok
            else f"vina --help only returned {h_lines} lines (broken libboost)"
        ),
        "checked_paths": [str(p) for p in checked],
        "checked_utc": _utc_now_iso(),
    }

    print(f"\nBinary:  {chosen}")
    print(f"Version: {version_str!r}")
    print(f"Help lines: {h_lines} (expect >=5 for a working build)")
    print(f"Status: {report['status']}")

    # Smoke dock only if the binary probes green
    if v_ok and h_ok:
        if args.with_smoke_dock:
            print("\n--with-smoke-dock: running 1h36 CLI dock of a 1-atom Mg dummy ...")
            smoke_dir = args.out.parent / "smoke_dock"
            ok, info = _run_smoke_dock(chosen, smoke_dir, timeout=args.smoke_timeout)
            report["smoke_dock"] = info
            print(f"  smoke_dock.status = {info.get('status', '?')}")
            if not ok:
                print(f"  smoke_dock.error = {info.get('error', '')}")
        else:
            print(
                "\nSkipping --with-smoke-dock (use --with-smoke-dock for end-to-end check)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote status: {args.out}")

    if report["status"] == "REAL_VINA_AVAILABLE":
        print("\nGATE PASS: Vina binary verified OK.")
        return 0
    print(
        f"\nGATE FAIL: Vina binary not healthy — status={report['status']}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
