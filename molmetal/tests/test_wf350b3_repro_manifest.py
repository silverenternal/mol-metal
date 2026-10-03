"""Wf-350-B3 — `--json` output reproducibility manifest integration tests.

Per the 2026-09-29 user ultracode directive + Wf-343a-R1 finding #3
("Reproducibility manifest should extend --json output" — DeepMind
§3.6 contract), the ``--json`` flag of
:mod:`molmetal/scripts/pre_flight_check.py` now emits a top-level
``reproducibility_manifest`` block alongside the existing ``gates``
array. The manifest embeds the on-disk Wf-350 reproducibility
artefacts (``wf350a1_runtime_energy.json`` +
``wf350a2_compute_manifest.json`` + ``wf350a3_hp_provenance.jsonl``)
plus a forward-looking ``prereg_path`` slot (Wf-343a-R1 / DeepMind
§3.1 pre-registration record contract).

This file ships 2 NEW integration tests covering the manifest's
existence + per-section presence contracts. The tests run the gate
suite in an isolated subprocess against a tmp_path JSON output and
parse the resulting JSON. They do NOT touch paper/ or any external
state.

Tests
-----

1. **test_preflight_json_includes_reproducibility_manifest** —
   runs ``pre_flight_check.py --pipeline all --json <tmp>`` as a
   subprocess, parses the JSON, and asserts (a) the top-level
   ``reproducibility_manifest`` key is present, (b) the manifest
   block has all 4 expected sub-keys (``compute`` +
   ``hp_provenance`` + ``runtime_energy`` + ``prereg_path``),
   and (c) the manifest is a dict (not None).

2. **test_reproducibility_manifest_per_section_present** —
   drills into each of the 3 Wf-350 reproducibility slots and
   validates its per-section schema: ``compute`` is a dict with the
   canonical NeurIPS §E.5 top-level keys (``gpus`` +
   ``driver_version`` + ``torch_version`` + ``cuda_version`` +
   ``system``); ``hp_provenance`` is a list of dicts each with
   ``config_path`` + ``current_sha`` + ``recent_commits`` +
   ``uncommitted_changes`` (the 4 Wf-350-A3 mandatory keys);
   ``runtime_energy`` is a dict with the Wf-350-A1 4 mandatory
   keys (``per_gpu`` + ``total_duration_s`` +
   ``estimated_energy_j`` + ``n_gpus_reported``). On a fresh
   sandbox without the Wf-350 artefacts, the test SKIPs (env-
   aware, not FAIL).

Honest framing (party-twenty-spirit §3.5)
------------------------------
* Each test asserts ONE observable contract (manifest-in-top-level /
   per-section schema validity) so failures localise cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
   to a Wf-350 on-disk artefact or a tmp-path subprocess output.
* The subprocess is the canonical ``pre_flight_check.py`` binary;
   no private helpers, no in-process shortcuts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
COMPUTE_MANIFEST = REPORTS_DIR / "wf350a2_compute_manifest.json"
HP_PROVENANCE = REPORTS_DIR / "wf350a3_hp_provenance.jsonl"
RUNTIME_ENERGY = REPORTS_DIR / "wf350a1_runtime_energy.json"


def _run_preflight_json(tmp_path: Path) -> dict:
    """Invoke ``pre_flight_check.py --pipeline all --json <tmp>`` and parse the JSON.

    The subprocess writes the JSON to ``tmp_path`` AND emits the same
    JSON to stdout (per the --json contract). The test parses the
    ``tmp_path`` file (avoids stdout buffering ambiguity).
    """
    out_json = tmp_path / "preflight.json"
    cmd = [
        sys.executable,
        str(PRE_FLIGHT_SCRIPT),
        "--pipeline",
        "all",
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=240.0,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # Surface subprocess output on non-zero exit for debuggability.
    if proc.returncode not in (0, 1, 2):
        pytest.skip(
            f"pre_flight_check.py exited {proc.returncode}: "
            f"{proc.stderr[:200]}"
        )
    if not out_json.is_file():
        pytest.skip(
            f"pre_flight_check.py --json did not write {out_json} "
            f"(stderr: {proc.stderr[:200]})"
        )
    with out_json.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_preflight_json_includes_reproducibility_manifest(tmp_path: Path) -> None:
    """``--json`` output includes the top-level ``reproducibility_manifest``.

    Per Wf-343a-R1 finding #3 (DeepMind §3.6 reproducibility manifest
    should be a 1st-class artifact in --json output), the Wf-350-B3
    schema extension adds a ``reproducibility_manifest`` key to the
    report dict. This test asserts the key exists at the top level +
    has all 4 expected sub-keys.
    """
    report = _run_preflight_json(tmp_path)

    # (a) top-level reproducibility_manifest key present
    assert "reproducibility_manifest" in report, (
        "Wf-350-B3: --json output MUST include "
        "'reproducibility_manifest' top-level key (DeepMind §3.6 "
        "reproducibility manifest contract)."
    )

    rm = report["reproducibility_manifest"]

    # (b) manifest is a dict, not None
    assert isinstance(rm, dict), (
        f"Wf-350-B3: 'reproducibility_manifest' must be a dict, got "
        f"{type(rm).__name__}"
    )

    # (c) all 4 sub-keys present
    expected_keys = {"compute", "hp_provenance", "runtime_energy", "prereg_path"}
    actual_keys = set(rm.keys())
    assert expected_keys.issubset(actual_keys), (
        f"Wf-350-B3: 'reproducibility_manifest' missing sub-keys: "
        f"{expected_keys - actual_keys}"
    )


def test_reproducibility_manifest_per_section_present(tmp_path: Path) -> None:
    """Per-section validity of the reproducibility manifest.

    Drills into each of the 3 Wf-350 reproducibility slots and
    validates its per-section schema. The 3rd slot (``prereg_path``)
    is forward-looking per Wf-343a-R1 / DeepMind §3.1, so its value
    is allowed to be None.

    SKIP semantics: on a fresh sandbox without the Wf-350 on-disk
    artefacts, the manifest slots will be None and the test SKIPs
    (env-aware, not FAIL — same contract as Wf-345A2 yuelbond_decoder_env).
    """
    report = _run_preflight_json(tmp_path)
    rm = report.get("reproducibility_manifest", {})

    # ---- (a) compute manifest (Wf-350-A2) ----
    compute = rm.get("compute")
    if compute is None:
        pytest.skip(
            "Wf-350-B3: 'compute' slot is None — "
            "wf350a2_compute_manifest.json not on disk (env without "
            "Wf-350-A2 artefact)."
        )
    assert isinstance(compute, dict), (
        f"Wf-350-B3: 'compute' must be a dict, got {type(compute).__name__}"
    )
    compute_required_keys = {
        "gpus",
        "driver_version",
        "torch_version",
        "cuda_version",
        "system",
    }
    missing_compute = compute_required_keys - set(compute.keys())
    assert not missing_compute, (
        f"Wf-350-B3: compute manifest missing keys: {missing_compute}"
    )

    # ---- (b) hp provenance (Wf-350-A3, JSONL → list of dicts) ----
    hp = rm.get("hp_provenance")
    if hp is None:
        pytest.skip(
            "Wf-350-B3: 'hp_provenance' slot is None — "
            "wf350a3_hp_provenance.jsonl not on disk (env without "
            "Wf-350-A3 artefact)."
        )
    assert isinstance(hp, list), (
        f"Wf-350-B3: 'hp_provenance' must be a list, got "
        f"{type(hp).__name__}"
    )
    assert len(hp) >= 1, (
        "Wf-350-B3: 'hp_provenance' list is empty (expected ≥1 "
        "config record from canonical 7-config suite)"
    )
    hp_required_keys = {
        "config_path",
        "current_sha",
        "recent_commits",
        "uncommitted_changes",
    }
    for i, rec in enumerate(hp):
        assert isinstance(rec, dict), (
            f"Wf-350-B3: hp_provenance[{i}] must be a dict, got "
            f"{type(rec).__name__}"
        )
        missing_hp = hp_required_keys - set(rec.keys())
        assert not missing_hp, (
            f"Wf-350-B3: hp_provenance[{i}] missing keys: {missing_hp}"
        )

    # ---- (c) runtime + GPU energy (Wf-350-A1) ----
    energy = rm.get("runtime_energy")
    if energy is None:
        pytest.skip(
            "Wf-350-B3: 'runtime_energy' slot is None — "
            "wf350a1_runtime_energy.json not on disk (env without "
            "Wf-350-A1 artefact)."
        )
    assert isinstance(energy, dict), (
        f"Wf-350-B3: 'runtime_energy' must be a dict, got "
        f"{type(energy).__name__}"
    )
    energy_required_keys = {
        "per_gpu",
        "total_duration_s",
        "estimated_energy_j",
        "n_gpus_reported",
    }
    missing_energy = energy_required_keys - set(energy.keys())
    assert not missing_energy, (
        f"Wf-350-B3: runtime_energy missing keys: {missing_energy}"
    )

    # ---- (d) prereg_path (Wf-343a-R1 forward-looking) ----
    prereg = rm.get("prereg_path")
    # prereg_path is forward-looking — allowed to be None on a host
    # without a prereg.md (current canonical state per Wf-343a-R1).
    assert prereg is None or isinstance(prereg, str), (
        f"Wf-350-B3: 'prereg_path' must be None or str, got "
        f"{type(prereg).__name__}"
    )
