"""Regression tests: D-MPNN SHA256 audit must cover all D-MPNN ckpt families.

Background (Wf ultracode audit finding): ``dmpnn_sha256_audit.py``'s
``discover_dmpnn_ckpts`` only globs ``dmpnn_pic50_*.pt``. The audit's
intended surface — every D-MPNN checkpoint the project emits — also
includes ``dmpnn_mtw_*.pt`` (multi-task Wasserstein, ~80 files on disk)
and ``dmpnn_dann_*.pt`` (DANN domain-adversarial, 1 file). Under the
narrow glob these are invisible to the SHA256 diversity check, so a
silent copy-paste bug across the mtw or dann families would slip past
the guard that originally caught the Wf-120 rh_seed1==rh_seed0 issue.

These tests assert all three filename prefixes are discovered. They
fail against the narrow glob (proving the gap) and pass once the glob
is widened to cover the full D-MPNN family.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_empty(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"")
    return p


def test_discover_dmpnn_ckpts_includes_pic50_mtw_and_dann(tmp_path: Path):
    from molmetal.scripts.dmpnn_sha256_audit import discover_dmpnn_ckpts

    pic50 = _write_empty(tmp_path, "dmpnn_pic50_ru_seed0.pt")
    mtw = _write_empty(tmp_path, "dmpnn_mtw_ir_seed0.pt")
    dann = _write_empty(tmp_path, "dmpnn_dann_drugood_seed0.pt")
    # Decoy file from a non-D-MPNN family must still be filtered out.
    _write_empty(tmp_path, "affinity_platinai_seed0.pt")
    _write_empty(tmp_path, "README.md")

    found = {p.name for p in discover_dmpnn_ckpts(tmp_path)}
    assert pic50.name in found, f"pic50 ckpt missing: {pic50.name}"
    assert mtw.name in found, f"mtw ckpt missing: {mtw.name}"
    assert dann.name in found, f"dann ckpt missing: {dann.name}"
    assert "affinity_platinai_seed0.pt" not in found, (
        "non-D-MPNN file leaked into discovery"
    )


def test_discover_dmpnn_ckpts_handles_real_on_disk_layout(tmp_path: Path):
    """The on-disk layout has ``dmpnn_mtw_{metal}_seed{N}.pt`` and
    ``dmpnn_dann_{cohort}_seed{N}.pt`` shapes; both must be picked up."""
    from molmetal.scripts.dmpnn_sha256_audit import discover_dmpnn_ckpts

    for stem in [
        "dmpnn_mtw_ru_seed0",
        "dmpnn_mtw_ru_seed1",
        "dmpnn_mtw_ir_seed2",
        "dmpnn_mtw_os_seed0",
        "dmpnn_dann_drugood_seed0",
    ]:
        _write_empty(tmp_path, stem + ".pt")

    found = sorted(p.name for p in discover_dmpnn_ckpts(tmp_path))
    assert found == sorted(
        [
            "dmpnn_mtw_ru_seed0.pt",
            "dmpnn_mtw_ru_seed1.pt",
            "dmpnn_mtw_ir_seed2.pt",
            "dmpnn_mtw_os_seed0.pt",
            "dmpnn_dann_drugood_seed0.pt",
        ]
    ), found


def test_parse_filename_handles_mtw_and_dann():
    """``parse_filename`` must return ``(cohort, seed)`` for mtw + dann shapes
    so downstream audit reporting labels them correctly."""
    from molmetal.scripts.dmpnn_sha256_audit import parse_filename

    # mtw shape
    parsed = parse_filename(Path("dmpnn_mtw_ir_seed0.pt"))
    assert parsed is not None
    metal, seed = parsed
    assert metal.lower() == "ir"
    assert seed == 0

    # dann shape — DANN cohorts are not in FIVE_METALS, so parse_filename
    # is allowed to return None for them (the discovery filter still picks
    # them up). The contract here is that mtw parses; dann is allowed to
    # be None because it does not share the FIVE_METALS capitalisation
    # contract.
    dann = parse_filename(Path("dmpnn_dann_drugood_seed0.pt"))
    # Either is None (cohort not in FIVE_METALS) or ("Drugood", 0).
    if dann is not None:
        _, dann_seed = dann
        assert dann_seed == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
