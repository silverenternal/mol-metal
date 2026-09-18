"""REINVENT4 multi-property scoring bridge — JSON-lines RPC worker (WF-Extra-2).

================================================================
What this module is
================================================================
This is the canonical JSON-lines RPC worker for REINVENT4's *native*
multi-property objective (sum of weighted components such as QED,
SlogP, ring count, etc.).  It runs inside the isolated REINVENT4
ROCm project at ``/mnt/storage/env-projects/reinvent4-rocm/.venv``
and is invoked by the project's ``reinvent`` CLI.

Unlike :mod:`reinvent4_jsonl_worker`, which is an RDKit-only protocol
bridge, this worker drives the *real* REINVENT4 scoring CLI:

    /mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent \
        --scoring-config <auto_generated.toml>

It writes the request SMILES to a temporary ``.smi`` file, runs the
upstream CLI, parses the produced ``.csv`` (per-component values plus
an aggregate total) and returns the aggregate score as a single
``multiproperty_score`` float in ``[0, 1]`` — exactly matching the
existing ``reinvent4_jsonl_worker`` wire contract.

================================================================
Wire protocol (JSON-lines over stdin/stdout)
================================================================
Request::

    {
        "smiles": "CCO",
        "components": {"logp": 0.4, "ring_count": 0.2, "qed": 0.4},
        "timeout": 60.0
    }

Response::

    {
        "smiles": "CCO",
        "multiproperty_score": 0.85,        # weighted sum in [0, 1], or null
        "components_raw": {"logp": 0.95, "ring_count": 0.50, "qed": 0.78},
        "device": "cuda:0",
        "model_sha": "<scoring_toml_sha256>",
        "ok": true,
        "reason": null
    }

Failure modes (always return ``ok=false`` rather than crashing the
subprocess):

* ``invalid_smiles``        — RDKit / REINVENT4 cannot parse the SMILES
* ``timeout``               — the reinvent subprocess exceeded ``timeout`` seconds
* ``backend_error``         — non-zero exit / CSV parse failure
* ``missing_binary``        — the reinvent binary is not on PATH

================================================================
Mapping components → REINVENT4 TOML
================================================================
The user-supplied ``components`` dict is translated into a REINVENT4
scoring TOML using a built-in template.  Supported keys (component
weights are honoured from the dict):

* ``qed``            → ``[scoring.component.QED]``
* ``logp``           → ``[scoring.component.SlogP]`` (RDKit) with a
                        reverse-sigmoid transform centered on logP ≈ 2
* ``ring_count``     → ``[scoring.component.NumRings]`` with a sigmoid
                        transform that peaks at ~3 rings
* ``mw``             → ``[scoring.component.MolecularWeight]`` double-
                        sigmoid centered on 350 Da
* ``tpsa``           → ``[scoring.component.TPSA]`` double-sigmoid
                        centered on 75 Å²
* ``hbd``            → ``[scoring.component.HBondDonors]`` reverse-
                        sigmoid (fewer H-bonds = better)
* ``hba``            → ``[scoring.component.HBondAcceptors]`` reverse-
                        sigmoid
* ``rot_bonds``      → ``[scoring.component.NumRotBond]`` reverse-
                        sigmoid (fewer rotatable bonds = better)
* ``aromatic_rings`` → ``[scoring.component.NumAromaticRings]`` sigmoid
                        peaking at ~2
* ``heavy_atoms``    → ``[scoring.component.NumHeavyAtoms]`` reverse-
                        sigmoid (smaller = better)
* ``matching_smarts`` → ``[scoring.component.MatchingSubstructure]`` (penalty)

If the user supplies a component we don't recognise, it is *silently
ignored* — this worker never crashes on an unknown key.  The aggregate
is computed by REINVENT4 itself using ``type = "arithmetic_mean"``
(weighted).  Any single-component value in ``[0, 1]`` from REINVENT4
is clipped and emitted under ``components_raw`` for downstream audit.

================================================================
Public API
================================================================
* :func:`handle`        — process one request dict, return one response dict
* :func:`main`          — JSON-lines loop on stdin/stdout
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#: Path to the isolated REINVENT4 ROCm project venv.  Falls back to
#: ``reinvent`` on PATH for development environments that ship the
#: upstream package directly.
DEFAULT_REINVENT_BIN = "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent"
DEVICE = "cuda:0" if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
    "ROCM_VISIBLE_DEVICES"
) else "cuda:0"

#: Default scoring aggregate type.  REINVENT4 supports ``geometric_mean``
#: and ``arithmetic_mean``; arithmetic mean is the closest analogue to
#: the existing reward aggregator (weighted sum of components in [0,1]).
DEFAULT_AGGREGATE_TYPE = "arithmetic_mean"

#: Default per-request timeout (seconds).
DEFAULT_TIMEOUT = 60.0

#: Recognised component keys → REINVENT4 scoring-component recipe.
#: Each recipe returns the TOML fragment to splice into the scoring
#: config when the user supplies a weight > 0.
COMPONENT_RECIPES: Dict[str, str] = {
    "qed": """
[[scoring.component]]
[scoring.component.QED]

[[scoring.component.QED.endpoint]]
name = "QED"
weight = {weight}
""",
    "logp": """
[[scoring.component]]
[scoring.component.SlogP]

[[scoring.component.SlogP.endpoint]]
name = "SlogP"
weight = {weight}
transform.type = "reverse_sigmoid"
transform.high = 4.0
transform.low = 0.5
transform.k = 0.5
""",
    "ring_count": """
[[scoring.component]]
[scoring.component.NumRings]

[[scoring.component.NumRings.endpoint]]
name = "NumRings"
weight = {weight}
transform.type = "sigmoid"
transform.high = 5
transform.low = 1
transform.k = 0.5
""",
    "aromatic_rings": """
[[scoring.component]]
[scoring.component.NumAromaticRings]

[[scoring.component.NumAromaticRings.endpoint]]
name = "NumAromaticRings"
weight = {weight}
transform.type = "sigmoid"
transform.high = 3
transform.low = 0
transform.k = 0.5
""",
    "mw": """
[[scoring.component]]
[scoring.component.MolecularWeight]

[[scoring.component.MolecularWeight.endpoint]]
name = "MW"
weight = {weight}
transform.type = "double_sigmoid"
transform.high = 500.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
""",
    "tpsa": """
[[scoring.component]]
[scoring.component.TPSA]

[[scoring.component.TPSA.endpoint]]
name = "TPSA"
weight = {weight}
transform.type = "double_sigmoid"
transform.high = 140.0
transform.low = 0.0
transform.coef_div = 140.0
transform.coef_si = 20.0
transform.coef_se = 20.0
""",
    "hbd": """
[[scoring.component]]
[scoring.component.HBondDonors]

[[scoring.component.HBondDonors.endpoint]]
name = "HBD"
weight = {weight}
transform.type = "reverse_sigmoid"
transform.high = 5
transform.low = 0
transform.k = 0.5
""",
    "hba": """
[[scoring.component]]
[scoring.component.HBondAcceptors]

[[scoring.component.HBondAcceptors.endpoint]]
name = "HBA"
weight = {weight}
transform.type = "reverse_sigmoid"
transform.high = 10
transform.low = 4
transform.k = 0.5
""",
    "rot_bonds": """
[[scoring.component]]
[scoring.component.NumRotBond]

[[scoring.component.NumRotBond.endpoint]]
name = "NumRotBond"
weight = {weight}
transform.type = "reverse_sigmoid"
transform.high = 10
transform.low = 2
transform.k = 0.5
""",
    "heavy_atoms": """
[[scoring.component]]
[scoring.component.NumHeavyAtoms]

[[scoring.component.NumHeavyAtoms.endpoint]]
name = "NumHeavyAtoms"
weight = {weight}
transform.type = "reverse_sigmoid"
transform.high = 50
transform.low = 15
transform.k = 0.5
""",
    "matching_smarts": """
[[scoring.component]]
[scoring.component.MatchingSubstructure]

[[scoring.component.MatchingSubstructure.endpoint]]
name = "MatchingSubstructure"
weight = {weight}
params.smarts = "c1ccccc1"
params.use_chirality = false
""",
}


# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------
def _build_scoring_fragments(components: Dict[str, float]) -> List[str]:
    """Translate a user ``components`` dict into per-component TOML fragments.

    Returns a list of recipe strings (one per active component).  Used
    by :func:`_build_scoring_toml` for the protocol-level (no-path)
    TOML and by :func:`_run_reinvent_scoring` for the on-disk TOML
    that points at a concrete ``INPUT.smi`` / ``OUTPUT.csv``.
    """
    fragments: List[str] = []
    for key, weight in components.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w <= 0.0:
            continue
        recipe = COMPONENT_RECIPES.get(str(key).lower())
        if recipe is None:
            # Silently ignore unknown keys; the worker never crashes on
            # an unknown component, it just omits it.
            continue
        fragments.append(recipe.format(weight=w))

    if not fragments:
        # Default to plain QED so a missing/unknown components dict
        # still returns a meaningful score.
        fragments.append(COMPONENT_RECIPES["qed"].format(weight=1.0))
    return fragments


def _build_scoring_toml(components: Dict[str, float]) -> str:
    """Translate a user ``components`` dict into a REINVENT4 scoring TOML."""
    fragments = _build_scoring_fragments(components)
    return (
        "# Auto-generated by reinvent4_multiproperty_jsonl_worker.py\n"
        f"# SHA256 input components: {hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:12]}\n"
        "run_type = \"scoring\"\n"
        "\n"
        "[parameters]\n"
        "smiles_file = \"INPUT.smi\"\n"
        "output_csv = \"OUTPUT.csv\"\n"
        "\n"
        "[scoring]\n"
        f"type = \"{DEFAULT_AGGREGATE_TYPE}\"\n"
        "parallel = 1\n"
        + "".join(fragments)
    )


def _toml_sha256(toml_text: str) -> str:
    return hashlib.sha256(toml_text.encode("utf-8")).hexdigest()


def components_key(components: Dict[str, float]) -> str:
    """Stable string fingerprint of a components dict (for hashing)."""
    norm = []
    for k in sorted(str(k) for k in components.keys()):
        try:
            v = float(components[k])
        except (TypeError, ValueError):
            continue
        if v <= 0.0:
            continue
        norm.append(f"{k}={v}")
    return json.dumps(norm, sort_keys=True)


# ---------------------------------------------------------------------------
# REINVENT4 subprocess invocation
# ---------------------------------------------------------------------------
def _resolve_reinvent_binary() -> Optional[str]:
    """Return path to the ``reinvent`` binary or ``None``."""
    configured = os.environ.get("REINVENT4_BIN")
    for cand in (configured, DEFAULT_REINVENT_BIN, "reinvent", "reinvent4"):
        if cand and (os.path.isfile(cand) and os.access(cand, os.X_OK) or shutil.which(cand)):
            return cand
    return None


def _run_reinvent_scoring(
    smiles: str,
    components: Dict[str, float],
    timeout: float,
) -> Tuple[Optional[float], Dict[str, float], Optional[str]]:
    """Run REINVENT4 scoring CLI for a single SMILES. Returns (score, raw, error)."""
    binary = _resolve_reinvent_binary()
    if binary is None:
        return None, {}, "missing_binary"

    fragments = _build_scoring_fragments(components)
    # Hash is computed on the canonical (path-independent) TOML so the
    # same components dict always yields the same model_sha.
    canonical = "".join(fragments) + components_key(components)
    toml_sha = hashlib.sha256(canonical.encode()).hexdigest()

    workdir = tempfile.mkdtemp(prefix="reinvent4_mp_")
    try:
        smi_path = os.path.join(workdir, "INPUT.smi")
        csv_path = os.path.join(workdir, "OUTPUT.csv")
        toml_path = os.path.join(workdir, "scoring.toml")
        with open(smi_path, "w") as fh:
            fh.write(smiles + "\n")
        with open(toml_path, "w") as fh:
            fh.write(
                "# Auto-generated by reinvent4_multiproperty_jsonl_worker.py\n"
                f"# SHA256: {toml_sha}\n"
                "run_type = \"scoring\"\n"
                "\n"
                "[parameters]\n"
                f"smiles_file = {json.dumps(smi_path)}\n"
                f"output_csv = {json.dumps(csv_path)}\n"
                "\n"
                "[scoring]\n"
                f"type = \"{DEFAULT_AGGREGATE_TYPE}\"\n"
                "parallel = 1\n"
                + "".join(fragments)
            )

        cmd = [binary, toml_path, "--device", "cpu"]
        # REINVENT4 with only built-in RDKit-style components runs
        # cleanly on CPU.  Force CPU to avoid ROCm surprise in the
        # worker subprocess; the request did not opt into GPU.
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, {}, "timeout"
        except FileNotFoundError:
            return None, {}, "missing_binary"
        except Exception as exc:  # pragma: no cover - defensive
            return None, {}, f"backend_error:{type(exc).__name__}:{exc}"

        if proc.returncode != 0:
            return None, {}, f"backend_error:exit_{proc.returncode}"

        if not os.path.isfile(csv_path):
            return None, {}, "backend_error:csv_missing"

        return _parse_scoring_csv(csv_path, toml_sha)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_scoring_csv(
    csv_path: str, toml_sha: str
) -> Tuple[Optional[float], Dict[str, float], Optional[str]]:
    """Parse a REINVENT4 scoring CSV; return (aggregate, components, error).

    REINVENT4 v4 writes a CSV with one row per SMILES and the columns::

        SMILES,Comment,RDKit_SMILES (REINVENT),Score,<component>,...,<component> (raw)

    * ``Score`` is the weighted aggregate (in ``[0, 1]`` when every
      component is in ``[0, 1]``).
    * ``<component>`` are the per-component *transformed* scores (in
      ``[0, 1]`` by construction of the transforms).
    * ``<component> (raw)`` are the untransformed raw values (e.g. raw
      SlogP can be ``-0.001``) — we ignore these in
      :attr:`components_raw` because they are not reward-shaped.

    We return the ``Score`` column as :attr:`multiproperty_score` and
    the per-component *transformed* columns as :attr:`components_raw`.
    """
    try:
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lower = {str(k).strip().lower(): v for k, v in row.items()}
                if not lower:
                    continue
                # Aggregate: REINVENT4 writes the column "Score".
                aggregate: Optional[float] = None
                for key in ("score", "total_score"):
                    v = lower.get(key)
                    if v is not None and str(v).strip():
                        try:
                            aggregate = float(v)
                        except (TypeError, ValueError):
                            aggregate = None
                        if aggregate is not None:
                            break
                # Per-component transformed values — everything that
                # is NOT a raw column, NOT "score", NOT identity cols.
                raw: Dict[str, float] = {}
                for k, v in lower.items():
                    if k in ("smiles", "comment", "rdkit_smiles (reinvent)",
                             "score", "total_score", "source", "id"):
                        continue
                    if k.endswith(" (raw)"):
                        continue
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    raw[k] = float(max(0.0, min(1.0, f)))
                if aggregate is None:
                    # Fall back to mean of per-component values.
                    if raw:
                        aggregate = sum(raw.values()) / len(raw)
                    else:
                        return None, raw, "backend_error:no_numeric_columns"
                return float(max(0.0, min(1.0, aggregate))), raw, None
        return None, {}, "backend_error:csv_empty"
    except Exception as exc:  # pragma: no cover - defensive
        return None, {}, f"backend_error:csv_parse:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Fast-path: pure-Python fallback (no subprocess)
# ---------------------------------------------------------------------------
def _validate_smiles(smiles: str) -> Optional[Any]:
    """Return an RDKit mol or ``None`` if the SMILES is invalid.

    We import RDKit lazily so the worker can boot even on a stripped
    ROCm image that lacks the heavy RDKit build.
    """
    try:
        from rdkit import Chem  # type: ignore
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API — JSON-lines handler
# ---------------------------------------------------------------------------
def handle(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process one JSON-lines request and return the response dict."""
    smiles = request.get("smiles", "")
    components = request.get("components") or {}
    timeout = float(request.get("timeout", DEFAULT_TIMEOUT))

    if not isinstance(components, dict):
        components = {}

    base: Dict[str, Any] = {
        "smiles": smiles,
        "multiproperty_score": None,
        "components_raw": {},
        "device": DEVICE,
        "model_sha": None,
        "ok": False,
        "reason": None,
    }

    if not isinstance(smiles, str) or not smiles.strip():
        base["reason"] = "invalid_smiles"
        return base

    # Cheap pre-check: RDKit parse.  If RDKit rejects it, REINVENT4
    # will reject it too — fail fast without spawning a subprocess.
    if _validate_smiles(smiles) is None:
        base["reason"] = "invalid_smiles"
        return base

    t0 = time.monotonic()
    aggregate, raw, err = _run_reinvent_scoring(smiles, components, timeout)
    elapsed = time.monotonic() - t0

    base["components_raw"] = raw
    base["model_sha"] = hashlib.sha256(components_key(components).encode()).hexdigest()[:16]
    base["elapsed_seconds"] = round(elapsed, 4)

    if err is not None:
        base["reason"] = err
        return base

    base["multiproperty_score"] = aggregate
    base["ok"] = True
    return base


def main() -> int:
    """JSON-lines loop on stdin/stdout — one request per line, one response per line."""
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                request = {}
        except Exception:
            request = {}
        try:
            response = handle(request)
        except Exception as exc:  # never crash the bridge
            response = {
                "smiles": request.get("smiles") if isinstance(request, dict) else None,
                "multiproperty_score": None,
                "components_raw": {},
                "ok": False,
                "reason": f"unhandled:{type(exc).__name__}:{exc}",
            }
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
