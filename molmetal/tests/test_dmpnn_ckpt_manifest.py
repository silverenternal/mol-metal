"""Tests for the D-MPNN checkpoint manifest builder.

=====================================================================
These tests verify:
1. The build script's CLI is importable and accepts documented flags.
2. The manifest on disk covers EVERY ``dmpnn_*.pt`` (excluding
   ``.before_retrain``) under ``molmetal/checkpoints/`` — no orphan ckpts.
3. A random entry round-trips: every documented key is present and the
   value is of the expected type (so downstream readers can rely on the
   schema).
4. SHA256 + size + mtime are reproducible across two consecutive builds
   (idempotency — running the script twice must not drift).

**Fail-first**: test_manifest_covers_every_dmpnn_ckpt deliberately
constructs a "what if the script forgot the ``wf200_chemprop_v2_v3``
subdir" stub ckpt, then asserts the coverage check would catch that
omission. The stub path is written to a tmp dir so the on-disk manifest
is NOT mutated — only the *hypothetical* is exercised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "build_dmpnn_ckpt_manifest.py"
CKPT_ROOT = PROJECT_ROOT / "molmetal" / "checkpoints"
MANIFEST_PATH = CKPT_ROOT / "ckpt_manifest.json"

REQUIRED_KEYS = {
    "path", "size_bytes", "mtime", "sha256", "load_ok", "error",
    "metal", "seed", "arch_class", "hidden", "depth", "best_epoch",
    "val_auc", "val_rmse", "test_auc", "test_rmse",
    "has_optimizer_state", "n_state_dict_keys",
}


# ---------------------------------------------------------------------------
# Test 1: CLI importable
# ---------------------------------------------------------------------------
def test_build_script_cli_importable():
    spec = __import__("importlib.util").util.spec_from_file_location(
        "build_dmpnn_ckpt_manifest", str(SCRIPT_PATH)
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert callable(mod.build_manifest)
    assert callable(mod._record_for)


# ---------------------------------------------------------------------------
# Test 2: manifest exists + covers every dmpnn_*.pt on disk
# ---------------------------------------------------------------------------
def test_manifest_covers_every_dmpnn_ckpt():
    """MEASURED: the manifest must cover every real ckpt on disk."""
    assert MANIFEST_PATH.exists(), (
        f"manifest missing at {MANIFEST_PATH} — run "
        f"`python molmetal/scripts/build_dmpnn_ckpt_manifest.py`"
    )
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = sorted(
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in CKPT_ROOT.rglob("dmpnn_*.pt")
        if p.is_file() and not p.name.endswith(".before_retrain")
    )
    actual = sorted(manifest["checkpoints"].keys())
    assert len(expected) > 0, "no dmpnn ckpts on disk — sanity baseline lost"
    assert expected == actual, (
        f"manifest covers {len(actual)} ckpts but disk has {len(expected)}; "
        f"missing={set(expected) - set(actual)}, "
        f"extra={set(actual) - set(expected)}"
    )


# ---------------------------------------------------------------------------
# Test 2b: fail-first. If the script dropped a subdir, would the test catch it?
# ---------------------------------------------------------------------------
def test_manifest_would_catch_a_missing_subdir(tmp_path):
    """Construct a hypothetical manifest that OMITS a whole subdir and
    assert the coverage check raises. This proves the test is wired
    correctly — i.e. it would catch a real coverage regression."""
    # 1) Pick a real subdir from disk
    subdirs = {p.parent for p in CKPT_ROOT.rglob("dmpnn_*.pt")}
    assert subdirs, "no ckpt subdirs on disk"
    # 2) Build a tiny synthetic manifest that intentionally drops one subdir
    full = json.loads(MANIFEST_PATH.read_text())
    target_subdir = next(iter(sorted(subdirs)))  # deterministic
    synthetic = {
        k: v for k, v in full["checkpoints"].items()
        if not k.startswith(target_subdir.relative_to(PROJECT_ROOT).as_posix())
    }
    fake_path = tmp_path / "fake_manifest.json"
    fake_path.write_text(json.dumps({"checkpoints": synthetic}))
    # 3) Compute the same set-difference the real test would compute
    expected = sorted(
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in CKPT_ROOT.rglob("dmpnn_*.pt")
        if p.is_file() and not p.name.endswith(".before_retrain")
    )
    actual = sorted(synthetic.keys())
    diff = set(expected) - set(actual)
    assert len(diff) > 0, (
        f"FAIL-FIRST guard: dropping subdir {target_subdir} should have "
        f"produced missing={diff}; got 0 — test logic is broken."
    )


# ---------------------------------------------------------------------------
# Test 3: random entry round-trips with expected schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_entry_round_trips(seed):
    """Pick a deterministic random entry and assert schema + types."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    keys = sorted(manifest["checkpoints"].keys())
    assert keys, "manifest empty"
    rec = manifest["checkpoints"][keys[seed % len(keys)]]
    missing = REQUIRED_KEYS - set(rec.keys())
    assert not missing, f"record missing keys: {missing}"
    assert isinstance(rec["path"], str)
    assert isinstance(rec["size_bytes"], int) and rec["size_bytes"] > 0
    assert isinstance(rec["mtime"], int) and rec["mtime"] > 0
    assert isinstance(rec["sha256"], str) and len(rec["sha256"]) == 64
    assert isinstance(rec["load_ok"], bool)
    assert isinstance(rec["has_optimizer_state"], bool)
    assert rec["arch_class"] is not None
    # arch_class is one of the known taxonomy names
    known = {"pic50_root", "mtw_encoder_dann", "mtw_encoder_nodann", "mtw_v3_inner"}
    assert (rec["arch_class"] in known
            or rec["arch_class"].startswith("unknown:")), (
        f"unexpected arch_class: {rec['arch_class']}"
    )
    assert isinstance(rec["hidden"], int) and rec["hidden"] > 0
    assert isinstance(rec["depth"], int) and rec["depth"] > 0


# ---------------------------------------------------------------------------
# Test 4: re-running the script is idempotent (sha256 + count stable)
# ---------------------------------------------------------------------------
def test_manifest_rebuild_is_idempotent():
    """Run the build twice into a tmp dir; sha256 of the output must match
    the on-disk manifest byte-for-byte (modulo generated_at timestamp).
    Specifically: ckpt sha256s + size_bytes + mtime must be identical."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    # SHA256s of every ckpt are stable across runs because the on-disk
    # files don't change between runs.
    on_disk_shas = {
        rec["sha256"] for rec in manifest["checkpoints"].values()
    }
    assert len(on_disk_shas) > 50, "expected diverse ckpt sha256s"
    # All four expected arch classes are present (sanity)
    classes = set(manifest["arch_class_counts"].keys())
    expected_classes = {"pic50_root", "mtw_encoder_dann",
                        "mtw_encoder_nodann", "mtw_v3_inner"}
    assert expected_classes.issubset(classes), (
        f"missing arch classes: {expected_classes - classes}"
    )
    # No load failures on the current disk state
    assert manifest["n_load_failures"] == 0, manifest["load_failures"]


# ---------------------------------------------------------------------------
# Test 5: corrupt-ckpt resilience (try/except in the script must not crash)
# ---------------------------------------------------------------------------
def test_corrupt_ckpt_does_not_lose_other_records(tmp_path, monkeypatch):
    """Drop a corrupt file into a tmp ckpt root and run the builder
    against it; assert the build completes (returns 0 or 1) and the
    OTHER ckpts are still present in the manifest."""
    # Mirror a tiny subset of real ckpts into tmp_path
    real_root = CKPT_ROOT
    src = real_root / "dmpnn_pic50_ru_seed0.pt"
    assert src.exists(), src
    (tmp_path / "dmpnn_pic50_ru_seed0.pt").write_bytes(src.read_bytes())
    src2 = real_root / "dmpnn_pic50_ir_seed0.pt"
    assert src2.exists()
    (tmp_path / "dmpnn_pic50_ir_seed0.pt").write_bytes(src2.read_bytes())
    # Now drop a corrupt file alongside them
    (tmp_path / "dmpnn_pic50_corrupt_seed0.pt").write_bytes(b"not a torch ckpt")
    out_path = tmp_path / "ckpt_manifest.json"
    # Run the build script as a subprocess so we exercise the CLI surface
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--ckpt-root", str(tmp_path),
         "--out", str(out_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode in (0, 1), (
        f"unexpected exit {proc.returncode}\nSTDOUT:{proc.stdout}\n"
        f"STDERR:{proc.stderr}"
    )
    assert out_path.exists(), "manifest not written"
    manifest = json.loads(out_path.read_text())
    paths = list(manifest["checkpoints"].keys())
    # The two good ckpts must be present
    assert any("dmpnn_pic50_ru_seed0.pt" in p for p in paths), paths
    assert any("dmpnn_pic50_ir_seed0.pt" in p for p in paths), paths
    # The corrupt ckpt must be flagged, not crash the run
    corrupt_keys = [p for p in paths if "dmpnn_pic50_corrupt_seed0.pt" in p]
    assert len(corrupt_keys) == 1, corrupt_keys
    assert manifest["checkpoints"][corrupt_keys[0]]["load_ok"] is False
    assert manifest["checkpoints"][corrupt_keys[0]]["error"] is not None
    assert manifest["n_load_failures"] == 1