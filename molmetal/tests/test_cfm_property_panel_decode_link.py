# Licensed under the Apache License, Version 2.0.
"""Tests for Wf-A54 — cfm_property_panel.py ``--ckpt-path`` decode→panel link.

Why this file exists
--------------------
Per AGENTS.md §5.2 honest-framing + memory ``cfm-property-panel-measures-no-model.md``:
the pre-Wf-A54 cfm_property_panel.py accepted only an external SMILES CSV; a CFM
checkpoint could not be fed in directly.  This was the "m4/m5/m7 unmeasured" gap.
Wf-A54 adds ``--ckpt-path`` to close the loop; this test file locks the new
surface (mutual exclusion, requirement, HONEST-NEGATIVE paths) so a future
refactor cannot silently regress.

All tests are CPU-only; the panel never loads the trainer into GPU.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"


def _load_module():
    """Load cfm_property_panel.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "cfm_property_panel", str(SCRIPT_PATH)
    )
    assert spec and spec.loader, "cfm_property_panel.py must exist and be importable"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smoke_stub_paths() -> list[Path]:
    ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints"
    return sorted(ckpt_dir.glob("cfm_yuelbond_wasserstein*.smoke_stub.pt"))


# ---------------------------------------------------------------------------
# CLI surface — new args, mutual exclusion, requirement
# ---------------------------------------------------------------------------
def test_cli_help_lists_new_args():
    """--ckpt-path, --n-samples, --decode-seed appear in --help output."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr}"
    out = proc.stdout
    assert "--ckpt-path" in out, f"--ckpt-path missing from --help: {out}"
    assert "--n-samples" in out, f"--n-samples missing from --help: {out}"
    assert "--decode-seed" in out, f"--decode-seed missing from --help: {out}"
    assert "--smiles-csv" in out, f"--smiles-csv still in --help: {out}"


def test_mutual_exclusion_rejects_both_args(tmp_path):
    """Setting BOTH --smiles-csv and --ckpt-path is a SystemExit."""
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--smiles-csv", str(tmp_path / "fake.csv"),
            "--ckpt-path", str(tmp_path / "fake.pt"),
            "--out-json", str(tmp_path / "out.json"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "mutually exclusive" in combined, (
        f"Expected mutually-exclusive rejection; got rc={proc.returncode} "
        f"combined={combined!r}"
    )


def test_requirement_rejects_neither_arg(tmp_path):
    """Setting NEITHER --smiles-csv nor --ckpt-path is a SystemExit."""
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--out-json", str(tmp_path / "out.json"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "required" in combined, (
        f"Expected required-arg rejection; got rc={proc.returncode} "
        f"combined={combined!r}"
    )


# ---------------------------------------------------------------------------
# --ckpt-path HONEST-NEGATIVE surfaces
# ---------------------------------------------------------------------------
def test_ckpt_path_missing_file_emits_honest_negative(tmp_path):
    """A non-existent --ckpt-path emits the HONEST-NEGATIVE print.

    Per Wf-A54 fix design: every failure mode in ``_decode_smiles_from_ckpt``
    prints a [wfA54-decode] HONEST-NEGATIVE line and returns an empty list, so
    the panel still writes a valid (empty-cohort) JSON.
    """
    out_json = tmp_path / "out.json"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--ckpt-path", str(tmp_path / "does_not_exist.pt"),
            "--out-json", str(out_json),
            "--n-samples", "4",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"Missing-file should NOT crash; got rc={proc.returncode} stderr={proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "[wfA54-decode] HONEST-NEGATIVE" in combined, (
        f"Missing the [wfA54-decode] HONEST-NEGATIVE marker; got {combined!r}"
    )
    # Panel still writes a valid JSON with empty cohort.
    assert out_json.is_file(), f"out-json not written: {out_json}"
    payload = json.loads(out_json.read_text())
    assert payload["n_input"] == 0
    assert payload["n_parsed"] == 0


def test_smoke_stub_emits_shape_mismatch_honest_negative(tmp_path):
    """The on-disk smoke-stub (z_table_len=4) emits a shape-mismatch
    HONEST-NEGATIVE rather than crashing.

    Per Wf-A54 fix design: smoke-stub ckpts use a 4-entry z_table that does
    not match the canonical 13-entry ``pt`` preset; ``load_state_dict`` raises
    RuntimeError on shape mismatch (strict=False does NOT suppress it) and
    the helper catches that and prints the precise HONEST-NEGATIVE.
    """
    stubs = _smoke_stub_paths()
    if not stubs:
        pytest.skip(
            "no smoke-stub ckpts under molmetal/checkpoints/ — fix-test "
            "only meaningful when stubs are on disk"
        )
    out_json = tmp_path / "out.json"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--ckpt-path", str(stubs[0]),
            "--out-json", str(out_json),
            "--n-samples", "4",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"Smoke-stub should NOT crash; got rc={proc.returncode} stderr={proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "[wfA54-decode] HONEST-NEGATIVE" in combined, (
        f"Missing the HONEST-NEGATIVE marker; got {combined!r}"
    )
    assert out_json.is_file(), f"out-json not written: {out_json}"
    payload = json.loads(out_json.read_text())
    # Empty cohort when decode fails — never a fabricated number.
    assert payload["n_input"] == 0
    assert payload["n_parsed"] == 0


# ---------------------------------------------------------------------------
# Helper direct — fast unit-level coverage without subprocess overhead
# ---------------------------------------------------------------------------
def test_helper_direct_missing_chkpt_returns_empty_list():
    """``_decode_smiles_from_ckpt`` returns ``[]`` for a non-existent ckpt."""
    mod = _load_module()
    result = mod._decode_smiles_from_ckpt(
        Path("/tmp/this_does_not_exist.pt"),
        n_samples=4,
        seed=0,
    )
    assert result == [], (
        f"Helper must return empty list for missing ckpt, got {result!r}"
    )


def test_helper_direct_non_dict_state_returns_empty_list(tmp_path):
    """``_decode_smiles_from_ckpt`` returns ``[]`` when ckpt top-level is not
    a dict with model_state_dict."""
    import torch  # noqa: WPS433

    # Save a tensor-only ckpt (no model_state_dict key).
    fake = tmp_path / "bad.pt"
    torch.save({"not_model_state_dict": {}}, fake)
    mod = _load_module()
    result = mod._decode_smiles_from_ckpt(fake, n_samples=4, seed=0)
    assert result == [], (
        f"Helper must return empty list for non-state-dict ckpt, got {result!r}"
    )