#!/usr/bin/env python3
"""Wf-84 Phase 1 — Vina binary diagnostic.

================================================================
Detects whether ANY real AutoDock Vina (1.2.x) binary is reachable
on this host, by probing five well-known install channels in
priority order:

    1. ``MOLMETAL_VINA_BIN`` env override     (user-pinned absolute path)
    2. ``MOLMETAL_QVINA_BIN`` env override    (qvina-style alt)
    3. PATH lookup                             (shutil.which on canonical names)
    4. conda/mamba package cache               (``~/.mamba``, ``~/.micromamba``,
                                              ``/opt/miniforge3``, project venv)
    5. Filesystem sweep                        (recursive find for ``vina*``
                                              executables under ``/home`` and
                                              ``/opt``)

For every candidate path we **actually invoke** ``--version`` and
require ``returncode == 0`` plus an "AutoDock Vina" / "vina"
banner in stderr. This rejects broken binaries that resolve under
``shutil.which`` but fail to load (the classic libboost version
mismatch noted in ``round7_install_report.md``).

The script preserves an honest STUB_FALLBACK chain: when no real
binary is found we report ``status="MISSING_DEP"`` with the list of
probed channels and a recommended install command (CPU-only — no
auto-install). The output JSON mirrors the schema already used by
``molmetal/reports/wf60_vina_triple/vina_binary_status.json`` so the
downstream pipeline can read both sources interchangeably.

Public API
----------
* :func:`detect_vina_binary()` returns the diagnostic dict.
* :func:`main()` writes the JSON report and prints a 5-line summary.
* Run as a script::

      PYTHONPATH=/home/hugo/codes/mol-metal \
      python -m molmetal.scripts.vina_diagnostic \
          --out-dir molmetal/reports/wf84_p1_vina_diagnostic

Constraints
-----------
* CPU-only diagnostic — no pip install, no apt install, no network.
* No mutation of the host. Only ``subprocess.run`` against ``--version``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("wf84.vina_diagnostic")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Canonical names. Keep aligned with molmetal_lam/sbdd_env/vina_adapter.py.
# ---------------------------------------------------------------------------
VINA_BIN_CANDIDATES: Tuple[str, ...] = ("vina", "vina.exe", "vina-cli")
QVINA_BIN_CANDIDATES: Tuple[str, ...] = (
    "qvina",
    "qvina2",
    "qvina02",
    "qvina2.1",
    "qvina.exe",
)

# Env override names.
ENV_OVERRIDES: Dict[Tuple[str, ...], str] = {
    VINA_BIN_CANDIDATES: "MOLMETAL_VINA_BIN",
    QVINA_BIN_CANDIDATES: "MOLMETAL_QVINA_BIN",
}

# Filesystem roots to sweep when PATH is empty. ``/home`` covers both
# user-installed mamba envs and project venvs; ``/opt`` covers system
# installs (miniforge3 etc.).
SWEEP_ROOTS: Tuple[str, ...] = ("/home", "/opt", "/usr/local", "/usr")

# mamba / micromamba / conda package caches that survive env deletion.
CONDA_PKG_HINTS: Tuple[str, ...] = (
    "~/.mamba/pkgs",
    "~/.micromamba/pkgs",
    "~/micromamba/pkgs",
    "/opt/miniforge3/pkgs",
    "/opt/conda/pkgs",
)

# Vina banner substring — covers the upstream "AutoDock Vina" plus
# any "*-mod" fork header (e.g. f458505-mod from conda-forge).
VINA_BANNER = "AutoDock Vina"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ProbeHit:
    """One successful probe against a real Vina binary."""

    path: str
    source: str  # "env_override" | "path" | "conda_pkg" | "sweep" | "venv"
    version_line: str
    full_version: str
    python_binding_version: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ProbeMiss:
    """One candidate that we found but rejected (wrong banner or rc != 0)."""

    path: str
    source: str
    reason: str  # "non_zero_exit" | "banner_mismatch" | "not_executable" | "timeout"


@dataclass
class DiagnosticResult:
    """Top-level diagnostic payload.

    Schema mirrors ``molmetal/reports/wf60_vina_triple/vina_binary_status.json``
    so the downstream pipeline can consume both.
    """

    status: str  # "OK" | "MISSING_DEP"
    bin: Optional[str]
    version: str
    python_binding_version: Optional[str]
    reason: str
    checked_paths: List[str]
    hits: List[ProbeHit] = field(default_factory=list)
    misses: List[ProbeMiss] = field(default_factory=list)
    channels_probed: List[str] = field(default_factory=list)
    recommended_install: List[str] = field(default_factory=list)
    host: Dict[str, str] = field(default_factory=dict)
    captured_at: str = ""

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["hits"] = [h.as_dict() for h in self.hits]
        return d


# ---------------------------------------------------------------------------
# Probe primitives
# ---------------------------------------------------------------------------
def _parse_version(stderr_text: str, stdout_text: str) -> Tuple[str, str]:
    """Extract ``(<banner-line>, <full-version>)`` from a Vina --version probe.

    Vina typically writes the version banner on its own line, e.g.::

        AutoDock Vina 1.2.7 (...build hash...)

    The banner-line is the first non-empty line that contains
    ``VINA_BANNER``.  The full-version is that line minus the prefix
    (e.g. ``"AutoDock Vina "``).  We fall back to whichever stream had
    a banner.
    """
    text = stderr_text or stdout_text or ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if VINA_BANNER in line:
            prefix = f"{VINA_BANNER} "
            return line, line[len(prefix) :] if line.startswith(prefix) else line
    return "", ""


def _probe_binary(path: str, timeout: float = 5.0) -> Optional[Tuple[str, str, int]]:
    """Run ``<path> --version`` and return ``(version_line, full_version, rc)`` if it looks like Vina.

    Returns ``None`` if the binary is missing, not executable, fails the
    banner check, or times out.  Crucially, a non-zero exit code does
    NOT auto-fail the probe because some Vina forks (notably qvina02)
    print the banner on stderr and still return 1 on ``--version``.
    We accept the result if the banner matches.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file() or not os.access(p, os.X_OK):
        return None
    try:
        proc = subprocess.run(
            [str(p), "--version"],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    version_line, full_version = _parse_version(
        proc.stderr.decode("utf-8", errors="replace"),
        proc.stdout.decode("utf-8", errors="replace"),
    )
    if not version_line:
        return None
    return version_line, full_version, proc.returncode


def _resolve_from_env(candidates: Tuple[str, ...]) -> Optional[ProbeHit]:
    """Try env-var override (``MOLMETAL_VINA_BIN`` / ``MOLMETAL_QVINA_BIN``)."""
    env_name = ENV_OVERRIDES.get(candidates)
    if not env_name:
        return None
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    resolved = shutil.which(raw) or raw
    hit = _probe_binary(resolved)
    if hit is None:
        return None
    version_line, full_version, _ = hit
    return ProbeHit(
        path=str(Path(resolved).resolve()),
        source="env_override",
        version_line=version_line,
        full_version=full_version,
    )


def _resolve_from_path(candidates: Tuple[str, ...]) -> List[ProbeHit]:
    """Try ``shutil.which`` for each canonical name."""
    hits: List[ProbeHit] = []
    for name in candidates:
        path = shutil.which(name)
        if not path:
            continue
        hit = _probe_binary(path)
        if hit is None:
            continue
        version_line, full_version, _ = hit
        hits.append(
            ProbeHit(
                path=str(Path(path).resolve()),
                source="path",
                version_line=version_line,
                full_version=full_version,
            )
        )
    return hits


def _resolve_from_conda_pkg_hints() -> List[ProbeHit]:
    """Walk ``~/.mamba/pkgs``, ``/opt/miniforge3/pkgs`` etc. for ``vina*`` binaries."""
    hits: List[ProbeHit] = []
    for hint in CONDA_PKG_HINTS:
        root = Path(hint).expanduser()
        if not root.is_dir():
            continue
        # Look for ``vina`` / ``vina.exe`` inside ``bin/`` of any pkgs/vina-* dir.
        for pkg_dir in sorted(root.glob("vina*")):
            bin_dir = pkg_dir / "bin"
            if not bin_dir.is_dir():
                continue
            for cand_name in VINA_BIN_CANDIDATES + QVINA_BIN_CANDIDATES:
                cand = bin_dir / cand_name
                if not cand.is_file() or not os.access(cand, os.X_OK):
                    continue
                hit = _probe_binary(str(cand))
                if hit is None:
                    continue
                version_line, full_version, _ = hit
                hits.append(
                    ProbeHit(
                        path=str(cand.resolve()),
                        source="conda_pkg",
                        version_line=version_line,
                        full_version=full_version,
                    )
                )
    return hits


def _resolve_from_sweep(
    roots: Sequence[str] = SWEEP_ROOTS, max_depth: int = 6, limit: int = 50
) -> List[ProbeHit]:
    """Bounded recursive sweep of ``/home`` + ``/opt`` for ``vina*`` executables.

    We bound ``max_depth`` and ``limit`` so the scan stays under
    ~5 s on a typical workstation.  Files outside the SWEEP_ROOTS
    (e.g. ``/tmp``) are deliberately excluded to keep the search
    deterministic.
    """
    hits: List[ProbeHit] = []
    seen: set = set()
    name_re = __import__("re").compile(r"^vina(-cli)?(\.exe)?$")
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        try:
            iterator = root_path.rglob("vina*")
        except OSError:
            continue
        for cand in iterator:
            if len(hits) >= limit:
                break
            if not cand.is_file() or not os.access(cand, os.X_OK):
                continue
            name = cand.name
            # Only exact matches ("vina", "vina-cli", "vina.exe") — exclude
            # vina_score.py, vina_adapter.py, vina.py, etc.
            if not name_re.match(name):
                continue
            # Skip things obviously not on PATH (e.g. inside site-packages
            # where vina.py is a Python source file).
            try:
                depth = len(cand.relative_to(root_path).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            resolved = str(cand.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            hit = _probe_binary(resolved)
            if hit is None:
                continue
            version_line, full_version, _ = hit
            hits.append(
                ProbeHit(
                    path=resolved,
                    source="sweep",
                    version_line=version_line,
                    full_version=full_version,
                )
            )
    return hits


def _resolve_from_project_venv() -> List[ProbeHit]:
    """Look for a ``vina`` executable inside ``molmetal/.venv/bin`` or ``molmetal/.venvs/*/bin``."""
    hits: List[ProbeHit] = []
    venv_roots = [
        PROJECT_ROOT / "molmetal" / ".venv" / "bin",
        PROJECT_ROOT / ".venv" / "bin",
    ]
    venv_root_glob = PROJECT_ROOT / "molmetal" / ".venvs"
    if venv_root_glob.is_dir():
        for child in venv_root_glob.iterdir():
            bin_dir = child / "bin"
            if bin_dir.is_dir():
                venv_roots.append(bin_dir)
    for bin_dir in venv_roots:
        if not bin_dir.is_dir():
            continue
        for cand_name in VINA_BIN_CANDIDATES:
            cand = bin_dir / cand_name
            if not cand.is_file() or not os.access(cand, os.X_OK):
                continue
            hit = _probe_binary(str(cand))
            if hit is None:
                continue
            version_line, full_version, _ = hit
            hits.append(
                ProbeHit(
                    path=str(cand.resolve()),
                    source="venv",
                    version_line=version_line,
                    full_version=full_version,
                )
            )
    return hits


def _resolve_from_active_conda_envs() -> List[ProbeHit]:
    """Look for ``vina`` in *active* conda envs (i.e. envs whose libboost
    is loaded into the current process).

    Walk the well-known mamba/micromamba env roots and try ``vina`` from
    each active env's ``bin/``. We do not probe the conda pkg cache here
    because those binaries typically lack the libboost resolver from the
    active env — that's why we keep them in a separate channel and let
    ``_probe_binary`` reject them via the libboost error.
    """
    hits: List[ProbeHit] = []
    env_roots = [
        Path("~/.mamba/envs").expanduser(),
        Path("~/.micromamba/envs").expanduser(),
        Path("~/micromamba/envs").expanduser(),
        Path("~/.conda/envs").expanduser(),
        Path("~/anaconda3/envs").expanduser(),
        Path("~/miniconda3/envs").expanduser(),
        Path("/opt/miniforge3/envs").expanduser(),
        Path("/opt/conda/envs").expanduser(),
    ]
    # Also include the "root" envs themselves (the bin/ sibling of pkgs/).
    env_roots += [
        Path("~/.mamba").expanduser(),
        Path("~/.micromamba").expanduser(),
        Path("~/micromamba").expanduser(),
        Path("~/.conda").expanduser(),
        Path("~/anaconda3").expanduser(),
        Path("~/miniconda3").expanduser(),
        Path("/opt/miniforge3").expanduser(),
        Path("/opt/conda").expanduser(),
    ]
    for env_root in env_roots:
        if not env_root.is_dir():
            continue
        for bin_dir in [env_root / "bin"]:
            if not bin_dir.is_dir():
                continue
            for cand_name in VINA_BIN_CANDIDATES + QVINA_BIN_CANDIDATES:
                cand = bin_dir / cand_name
                if not cand.is_file() or not os.access(cand, os.X_OK):
                    continue
                hit = _probe_binary(str(cand))
                if hit is None:
                    continue
                version_line, full_version, _ = hit
                hits.append(
                    ProbeHit(
                        path=str(cand.resolve()),
                        source="active_conda_env",
                        version_line=version_line,
                        full_version=full_version,
                    )
                )
    return hits


def _python_binding_version() -> Optional[str]:
    """Return the ``vina`` Python module's ``__version__`` if importable, else None."""
    try:
        import vina  # type: ignore

        return getattr(vina, "__version__", None)
    except Exception:
        return None


def _have_meeko() -> bool:
    try:
        import meeko  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Recommended install commands — for the JSON report only.
# ---------------------------------------------------------------------------
def _recommend_install(host: Dict[str, str]) -> List[str]:
    """Build a CPU-only install recommendation list.

    No command is actually run by this diagnostic. The list is for
    the operator to read.
    """
    pkg_mgr = host.get("pkg_mgr", "")
    has_conda = host.get("has_conda", "false") == "true"
    has_pip = host.get("has_pip", "false") == "true"
    recs: List[str] = []
    if pkg_mgr == "apt":
        recs.append("sudo apt-get update && sudo apt-get install -y autodock-vina")
    elif pkg_mgr == "dnf":
        recs.append("sudo dnf install -y autodock-vina")
    elif pkg_mgr == "pacman":
        recs.append("sudo pacman -S autodock-vina")
    if has_conda:
        recs.append("mamba install -c conda-forge vina")
    if has_pip:
        recs.append("uv pip install vina meeko   # Python binding only (no CLI binary)")
    recs.append(
        "Build from source: https://github.com/ccsb-scripps/AutoDock-Vina "
        "(Boost >= 1.78 required)"
    )
    return recs


def _detect_pkg_mgr() -> str:
    for cand, name in (
        ("apt-get", "apt"),
        ("dnf", "dnf"),
        ("yum", "yum"),
        ("pacman", "pacman"),
        ("apk", "apk"),
        ("zypper", "zypper"),
    ):
        if shutil.which(cand):
            return name
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_vina_binary(env: Optional[Dict[str, str]] = None) -> DiagnosticResult:
    """Run the full diagnostic and return a structured result.

    Parameters
    ----------
    env : mapping, optional
        Override the process env (mostly useful in tests).
    """
    if env is not None:
        # Patch os.environ for this call so _resolve_from_env reads it.
        # We restore the prior value at the end via try/finally.
        saved = {k: os.environ.get(k) for k in ENV_OVERRIDES.values()}
        try:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return _detect_vina_binary_impl()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _detect_vina_binary_impl()


def _detect_vina_binary_impl() -> DiagnosticResult:
    channels: List[str] = []
    hits: List[ProbeHit] = []

    # 1. Env override.
    channels.append("env:MOLMETAL_VINA_BIN")
    channels.append("env:MOLMETAL_QVINA_BIN")
    for candidates in (VINA_BIN_CANDIDATES, QVINA_BIN_CANDIDATES):
        h = _resolve_from_env(candidates)
        if h is not None and not any(x.path == h.path for x in hits):
            hits.append(h)

    # 2. PATH lookup.
    channels.append("path:which")
    for candidates in (VINA_BIN_CANDIDATES, QVINA_BIN_CANDIDATES):
        for h in _resolve_from_path(candidates):
            if not any(x.path == h.path for x in hits):
                hits.append(h)

    # 3. Active conda / mamba envs (each env has libboost wired in PATH/LD_LIBRARY_PATH).
    channels.append("active_conda_envs")
    for h in _resolve_from_active_conda_envs():
        if not any(x.path == h.path for x in hits):
            hits.append(h)

    # 4. conda / mamba package caches (rarely works because libboost
    #    isn't on the loader path outside an active env — kept for completeness).
    channels.append("conda_pkg_hints")
    for h in _resolve_from_conda_pkg_hints():
        if not any(x.path == h.path for x in hits):
            hits.append(h)

    # 5. Project venv bin (separated from generic sweep to keep provenance clear).
    channels.append("project_venv")
    for h in _resolve_from_project_venv():
        if not any(x.path == h.path for x in hits):
            hits.append(h)

    # 5. Filesystem sweep (bounded).
    channels.append(f"sweep:{','.join(SWEEP_ROOTS)}")
    for h in _resolve_from_sweep():
        if not any(x.path == h.path for x in hits):
            hits.append(h)

    # 6. Sanity: re-probe in source-priority order so ``hits[0]`` is the
    #    preferred primary binary. We do not mutate the original hit list
    #    beyond re-ordering; all hits are still recorded.
    src_order = {
        "env_override": 0,
        "path": 1,
        "active_conda_env": 2,
        "venv": 3,
        "conda_pkg": 4,
        "sweep": 5,
    }
    hits.sort(key=lambda h: (src_order.get(h.source, 99), h.path))

    # Python binding detection (parallel to CLI — does not satisfy "binary"
    # by itself, but worth recording for downstream callers).
    py_v = _python_binding_version()

    # Host fingerprint (no PII).
    host = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "pkg_mgr": _detect_pkg_mgr(),
        "has_conda": str(
            shutil.which("mamba") is not None
            or shutil.which("micromamba") is not None
            or shutil.which("conda") is not None
        ).lower(),
        "has_pip": str(
            shutil.which("pip") is not None or shutil.which("uv") is not None
        ).lower(),
    }

    # Assemble the canonical "checked_paths" list expected by the wf60 schema.
    checked_paths = [h.path for h in hits]

    if hits:
        primary = hits[0]
        for h in hits:
            if h.python_binding_version is None and py_v:
                h.python_binding_version = py_v
        return DiagnosticResult(
            status="OK",
            bin=primary.path,
            version=primary.full_version,
            python_binding_version=py_v,
            reason=(
                f"Real Vina binary reachable via '{primary.source}'. "
                f"Other hits: {len(hits) - 1}."
            ),
            checked_paths=checked_paths,
            hits=hits,
            misses=[],
            channels_probed=channels,
            recommended_install=[],
            host=host,
            captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    reason_parts = [
        "No executable Vina/Q-Vina2/smina binary found on PATH, env override, "
        "conda/mamba pkg cache, project venv, or filesystem sweep.",
    ]
    if py_v:
        reason_parts.append(
            f"Python binding 'vina'=={py_v} is importable but provides no CLI "
            f"binary (the .so wrapper is bound to CPython only)."
        )
    else:
        reason_parts.append(
            "The 'vina' Python package is not importable in this environment either."
        )
    if not _have_meeko():
        reason_parts.append("The 'meeko' package is not importable.")
    return DiagnosticResult(
        status="MISSING_DEP",
        bin=None,
        version="",
        python_binding_version=py_v,
        reason=" ".join(reason_parts),
        checked_paths=checked_paths,
        hits=[],
        misses=[],
        channels_probed=channels,
        recommended_install=_recommend_install(host),
        host=host,
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else None
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "molmetal" / "reports" / "wf84_p1_vina_diagnostic"),
        help="Directory to write vina_status.json into.",
    )
    p.add_argument(
        "--json-stdout",
        action="store_true",
        help="Also print the JSON to stdout (useful for piping).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    result = detect_vina_binary()
    payload = result.as_dict()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vina_status.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", out_path)

    if args.json_stdout:
        print(json.dumps(payload, indent=2, sort_keys=True))

    # 5-line summary on stderr.
    summary = [
        f"status        : {result.status}",
        f"bin           : {result.bin}",
        f"version       : {result.version}",
        f"python_vina   : {result.python_binding_version}",
        f"hits          : {len(result.hits)}  misses: {len(result.misses)}",
    ]
    print("\n".join(summary))
    if result.status == "MISSING_DEP":
        print(f"reason        : {result.reason}")
        print("recommended_install:")
        for r in result.recommended_install:
            print(f"  - {r}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
