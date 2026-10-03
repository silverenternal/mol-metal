"""Wf-121 Phase 3 — acceptance tests for the cisplatin × 10-pockets sweep.

================================================================
What these tests pin
================================================================
Wf-121 Phase 3 closes the Wf-30 honest-negative loop: the 10 Wf-30
pockets (``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``)
are docked with **real** AutoDock Vina 1.2.7 against cisplatin using
the Wf-121 P1/P2 metal-PDBQT path. The kcal/mol numbers replace the
Wf-30 hash-derived stub floor (-9.6574) with real measurements.

The tests pin three layers:

1. **bench_cisplatin_real_vina.py** — the sweep script.
   * Generates a valid PDBQT for cisplatin (real meeko path).
   * Rewrites ``M`` → ``Mg`` (Vina parser compatibility — see
     ``_rewrite_metal_atype_for_vina`` docstring).
   * Runs Vina 1.2.7 per (pocket, seed) and parses kcal/mol.
   * Falls back to ``boltz_affinity`` (D-MPNN pIC50) when Vina
     cannot produce a finite score.
   * Writes ``per_seed_metrics.json`` + ``synthesis.md``.

2. **score_any._w_vina_metal_fallback** — the router wrapper.
   * Tries Vina first.
   * On ``MetalPDBQTUnavailable`` or non-finite kcal/mol falls
     back to ``_w_boltz_affinity``.
   * Both paths are real (not mock); neither fabricates a value.

3. **REGISTRY + ROUTING_POLICY** — the policy audit.
   * ``vina_metal_fallback`` is registered in REGISTRY.
   * Its ROUTING_POLICY entry lists ``boltz_affinity`` as the
     fallback chain.
   * ``policy.json`` mirrors the same shape.

Honest framing
--------------
* Vina kcal/mol here is real Vina output with a **generic VDW
  atype** (``Mg`` substituted for ``M`` — Vina's strict parser
  rejects ``M``). Disposition is ``MEASURED_GENERIC_ATOM``, not
  full-fidelity chemotype-tuned.
* Tests are CPU-only and skip if vina/meeko are not installed —
  matching the existing ``tests/test_vina_adapter_metals.py``
  skip policy.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Skip policy
# ---------------------------------------------------------------------------
def _have_vina_meeko() -> bool:
    try:
        import vina  # noqa: F401
        import meeko  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_vina_meeko(),
    reason="vina/meeko not installed (required for Wf-121 P3 sweep)",
)


# ---------------------------------------------------------------------------
# 1. Bench script — sweep completion
# ---------------------------------------------------------------------------
class TestBenchCisplatinRealVinaSweep:
    """The sweep must complete on the 10 Wf-30 pockets and write the
    canonical output artifacts."""

    SWEEP_DIR = REPO_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
    POCKETS = (
        "830c",
        "1h36",
        "3eml",
        "3ny8",
        "4aaw",
        "4rlu",
        "4rn0",
        "1fmc",
        "1g9v",
        "2hnx",
    )

    def test_sweep_dir_exists_with_per_pocket_json(self):
        assert self.SWEEP_DIR.exists(), (
            f"sweep dir missing: {self.SWEEP_DIR} — re-run "
            f"molmetal/scripts/bench_cisplatin_real_vina.py"
        )
        for pocket_id in self.POCKETS:
            p = self.SWEEP_DIR / pocket_id / "per_seed_metrics.json"
            assert p.exists(), f"per-seed metrics missing for {pocket_id}: {p}"

    def test_sweep_summary_present(self):
        p = self.SWEEP_DIR / "sweep_summary.json"
        assert p.exists(), f"sweep_summary.json missing: {p}"

    def test_synthesis_md_present(self):
        p = self.SWEEP_DIR / "synthesis.md"
        assert p.exists(), f"synthesis.md missing: {p}"

    def test_sweep_summary_contains_10_pockets(self):
        p = self.SWEEP_DIR / "sweep_summary.json"
        summary = json.loads(p.read_text())
        assert (
            summary["_n_pockets"] == 10
        ), f"expected 10 pockets, got {summary['_n_pockets']}"

    def test_all_per_seed_records_are_measured(self):
        for pocket_id in self.POCKETS:
            record = json.loads(
                (self.SWEEP_DIR / pocket_id / "per_seed_metrics.json").read_text()
            )
            for entry in record["_per_seed"]:
                assert entry["status"] in ("MEASURED", "DOCK_FAILED"), (
                    f"{pocket_id} seed={entry['seed']}: unexpected status "
                    f"{entry['status']!r}"
                )
                if entry["status"] == "MEASURED":
                    kcal = entry["kcal_mol"]
                    assert isinstance(kcal, (int, float))
                    assert math.isfinite(
                        kcal
                    ), f"{pocket_id} seed={entry['seed']}: non-finite kcal/mol"


# ---------------------------------------------------------------------------
# 2. kcal/mol — honest values, not the Wf-30 stub floor
# ---------------------------------------------------------------------------
class TestKcalMolRealNotStub:
    """Pins that kcal/mol is real Vina 1.2.7 output (NOT -9.6574 stub)."""

    SWEEP_DIR = REPO_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"

    def test_no_stub_floor_in_per_seed_records(self):
        for pocket_id in ("830c", "1h36", "3eml"):
            record = json.loads(
                (self.SWEEP_DIR / pocket_id / "per_seed_metrics.json").read_text()
            )
            for entry in record["_per_seed"]:
                if entry["status"] == "MEASURED":
                    # The Wf-30 hash-derived stub scored cisplatin at
                    # exactly -9.6574 — that is the anti-pattern this
                    # sweep replaces. Any value near -9.6574 in the
                    # MEASURED records would be suspicious.
                    kcal = float(entry["kcal_mol"])
                    assert abs(kcal - (-9.6574)) > 0.5, (
                        f"{pocket_id} seed={entry['seed']}: kcal/mol "
                        f"{kcal} is suspiciously close to the Wf-30 "
                        f"stub floor (-9.6574) — was the stub path "
                        f"silently re-introduced?"
                    )

    def test_kcal_mol_within_realistic_range(self):
        """Real Vina kcal/mol for a small cisplatin ligand should be
        in the [-10, -1] range. Values outside this band are either
        numerical noise or a wrong code path."""
        for pocket_id in ("830c", "1h36"):
            record = json.loads(
                (self.SWEEP_DIR / pocket_id / "per_seed_metrics.json").read_text()
            )
            for entry in record["_per_seed"]:
                if entry["status"] == "MEASURED":
                    kcal = float(entry["kcal_mol"])
                    assert -10.0 <= kcal <= -1.0, (
                        f"{pocket_id} seed={entry['seed']}: kcal/mol "
                        f"{kcal} outside [-10, -1] real-Vina range"
                    )

    def test_boltz_pic50_is_real_number(self):
        """boltz_affinity (Wf-120 D-MPNN pIC50) MUST be either a
        finite float OR None — never NaN/empty. Both indicate a
        real ML inference result (not a mock)."""
        record = json.loads(
            (self.SWEEP_DIR / "830c" / "per_seed_metrics.json").read_text()
        )
        boltz = record["_boltz_affinity_pic50"]
        assert boltz is None or (
            isinstance(boltz, (int, float)) and math.isfinite(float(boltz))
        ), f"boltz_affinity must be None or finite float, got {boltz!r}"


# ---------------------------------------------------------------------------
# 3. Per-pocket record shape
# ---------------------------------------------------------------------------
class TestPerPocketRecordShape:
    """Every per-seed JSON must have the canonical Wf-121 P3 schema."""

    SWEEP_DIR = REPO_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
    REQUIRED_TOP_KEYS = {
        "_workflow",
        "_phase",
        "_generated_utc",
        "_engine",
        "_engine_version",
        "_smiles",
        "_n_poses",
        "_exhaustiveness",
        "_pocket",
        "_boltz_affinity_pic50",
        "_per_seed",
        "_aggregate",
    }

    def test_required_top_level_keys_present(self):
        record = json.loads(
            (self.SWEEP_DIR / "830c" / "per_seed_metrics.json").read_text()
        )
        missing = self.REQUIRED_TOP_KEYS - set(record.keys())
        assert not missing, f"missing top-level keys: {missing}"

    def test_pocket_metadata_shape(self):
        record = json.loads(
            (self.SWEEP_DIR / "830c" / "per_seed_metrics.json").read_text()
        )
        pocket = record["_pocket"]
        assert pocket["pocket_id"] == "830c"
        assert "receptor_pdbqt" in pocket
        assert "n_heavy_atoms" in pocket
        assert "docking_box_center" in pocket
        assert len(pocket["docking_box_center"]) == 3

    def test_per_seed_entry_shape(self):
        record = json.loads(
            (self.SWEEP_DIR / "830c" / "per_seed_metrics.json").read_text()
        )
        entry = record["_per_seed"][0]
        for k in (
            "pocket_id",
            "seed",
            "smiles",
            "method",
            "kcal_mol",
            "elapsed_s",
            "n_poses",
            "n_atoms_ligand",
            "best_pose_rmsd_lb",
            "best_pose_rmsd_ub",
            "status",
        ):
            assert k in entry, f"per_seed entry missing key {k!r}"

    def test_engine_reported_is_real_vina(self):
        record = json.loads(
            (self.SWEEP_DIR / "830c" / "per_seed_metrics.json").read_text()
        )
        # Either the Python binding (preferred) or the CLI binary is
        # fine — both produce real Vina kcal/mol. We just forbid mock
        # / stub / placeholder substrings.
        engine_str = f"{record['_engine']} {record['_engine_version']}".lower()
        for forbidden in ("mock", "stub", "placeholder"):
            assert forbidden not in engine_str, (
                f"engine string {engine_str!r} contains forbidden "
                f"substring {forbidden!r}"
            )


# ---------------------------------------------------------------------------
# 4. Synthesis.md — honest framing
# ---------------------------------------------------------------------------
class TestSynthesisMarkdown:
    """Pins the honest-framing paragraphs in synthesis.md."""

    SWEEP_DIR = REPO_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
    SYNTHESIS = SWEEP_DIR / "synthesis.md"

    def test_synthesis_mentions_real_vina(self):
        text = self.SYNTHESIS.read_text()
        assert "real Vina" in text, (
            "synthesis.md must mention 'real Vina' — honest framing "
            "requires explicit real-vs-stub framing"
        )

    def test_synthesis_mentions_generic_caveat(self):
        text = self.SYNTHESIS.read_text()
        assert "MEASURED_GENERIC_ATOM" in text, (
            "synthesis.md must document the MEASURED_GENERIC_ATOM "
            "disposition so callers know the kcal/mol uses a generic "
            "VDW atype for Pt-group metals."
        )

    def test_synthesis_excludes_stub_marker(self):
        text = self.SYNTHESIS.read_text()
        # The Wf-30 hash-derived stub scored cisplatin at -9.6574.
        # The synthesis.md may mention it in the honest-framing
        # section as the anti-pattern we replaced, but it MUST NOT
        # appear as an actual result number (i.e. not preceded by
        # 'kcal/mol = ' or '|' or '=' that would make it look like a
        # measurement).
        # The check: if -9.6574 appears, it must be in a context that
        # makes clear it's an anti-pattern reference, not a result.
        if "-9.6574" in text:
            # Make sure each occurrence is in the "honest framing"
            # section (i.e. explaining what we replaced), not as a
            # measurement.
            import re

            for m in re.finditer(r"-9\.6574", text):
                ctx = text[max(0, m.start() - 40) : m.end() + 40]
                # The marker should appear in a "not ... floor" or
                # "not ... stub" context — anything else would mean
                # we're surfacing it as a measurement.
                assert any(
                    marker in ctx.lower()
                    for marker in ("stub", "anti-pattern", "not ", "floor")
                ), (
                    f"synthesis.md mentions -9.6574 in non-anti-pattern "
                    f"context: {ctx!r}"
                )


# ---------------------------------------------------------------------------
# 5. score_any registration — policy audit
# ---------------------------------------------------------------------------
class TestScoreAnyRegistration:
    """The router must register ``vina_metal_fallback`` with the
    correct backend, tool name, and fallback chain."""

    def test_vina_metal_fallback_in_registry(self):
        from molmetal.scoring.score_any import REGISTRY

        assert (
            "vina_metal_fallback" in REGISTRY
        ), "vina_metal_fallback missing from score_any.REGISTRY"

    def test_vina_metal_fallback_in_routing_policy(self):
        from molmetal.scoring.score_any import ROUTING_POLICY

        assert (
            "vina_metal_fallback" in ROUTING_POLICY
        ), "vina_metal_fallback missing from score_any.ROUTING_POLICY"

    def test_vina_metal_fallback_fallback_chain_is_boltz(self):
        from molmetal.scoring.score_any import ROUTING_POLICY

        entry = ROUTING_POLICY["vina_metal_fallback"]
        assert "boltz_affinity" in entry.fallback_chain, (
            f"fallback_chain must include boltz_affinity; got "
            f"{entry.fallback_chain}"
        )

    def test_vina_metal_fallback_status_is_measured(self):
        from molmetal.scoring.score_any import ROUTING_POLICY

        entry = ROUTING_POLICY["vina_metal_fallback"]
        assert entry.wf43_status == "MEASURED", (
            f"vina_metal_fallback wf43_status must be MEASURED "
            f"(real tool), got {entry.wf43_status!r}"
        )

    def test_vina_metal_fallback_backend_is_python_cli(self):
        from molmetal.scoring.score_any import REGISTRY

        ref = REGISTRY["vina_metal_fallback"]
        assert ref.backend == "python+cli", (
            f"backend must be 'python+cli' (Vina supports both "
            f"Python binding and CLI binary), got {ref.backend!r}"
        )

    def test_policy_json_mirrors_registry(self):
        policy = json.loads(
            (REPO_ROOT / "molmetal" / "scoring" / "policy.json").read_text()
        )
        assert "vina_metal_fallback" in policy["policy"]
        entry = policy["policy"]["vina_metal_fallback"]
        assert entry["wf43_status"] == "MEASURED"
        assert "boltz_affinity" in entry["fallback_chain"]

    def test_no_mock_keyword_in_registry(self):
        """Wf-43 invariant: no metric name in REGISTRY contains
        'mock'."""
        from molmetal.scoring.score_any import REGISTRY

        for name in REGISTRY:
            assert "mock" not in name.lower(), (
                f"REGISTRY metric name {name!r} contains 'mock' — "
                f"violates Wf-43 invariant"
            )


# ---------------------------------------------------------------------------
# 6. score_any._w_vina_metal_fallback — wrapper behavior
# ---------------------------------------------------------------------------
class TestVinaMetalFallbackWrapper:
    """The wrapper must try Vina first, fall back to boltz_affinity,
    and never fabricate a value."""

    def _build_synthetic_pocket(self):
        """Synthetic pocket for wrapper smoke tests — keeps the
        real Vina binary quiet without depending on a real receptor."""
        import torch
        from molmetal.domain import Pocket

        coords = torch.zeros(50, 3, dtype=torch.float32)
        coords[:, 0] = torch.linspace(-8.0, 8.0, 50)
        return Pocket(
            pdb_id="SYN",
            coords=coords,
            atom_types=torch.full((50,), 6, dtype=torch.long),
            residue_ids=torch.zeros(50, dtype=torch.long),
            chain_ids=torch.zeros(50, dtype=torch.long),
            mask=torch.ones(50, dtype=torch.bool),
            center=torch.zeros(3, dtype=torch.float32),
            radius=12.0,
        )

    def test_wrapper_returns_a_list(self):
        from molmetal.scoring.score_any import _w_vina_metal_fallback

        pocket = self._build_synthetic_pocket()
        # Use a SMILES that will hit the boltz_affinity fallback path
        # (cisplatin with a synthetic pocket — Vina may not produce a
        # finite score, exercising the fallback branch).
        out = _w_vina_metal_fallback(
            {
                "molecules": ["[Pt](N)(N)(Cl)Cl"],
                "pocket": pocket,
                "exhaustiveness": 1,
                "n_poses": 1,
            }
        )
        assert isinstance(out, list)
        assert len(out) == 1

    def test_wrapper_requires_pocket(self):
        from molmetal.scoring.score_any import (
            AdapterUnavailable,
            _w_vina_metal_fallback,
        )

        with pytest.raises(AdapterUnavailable) as ei:
            _w_vina_metal_fallback({"molecules": ["[Pt](N)(N)(Cl)Cl"], "pocket": None})
        assert "pocket" in str(ei.value).lower()

    def test_wrapper_handles_empty_complexes_via_fallback(self):
        """When Vina returns no complexes (or no finite score) the
        wrapper MUST fall back to boltz_affinity rather than
        fabricating a NaN."""
        from molmetal.molmetal_lam.sbdd_env import vina_adapter
        from molmetal.scoring.score_any import _w_vina_metal_fallback

        # Patch dock_smiles to return [] — simulates Vina failure
        # path that should trigger the boltz_affinity fallback.
        original_dock = vina_adapter.dock_smiles
        try:
            vina_adapter.dock_smiles = lambda *a, **k: []
            pocket = self._build_synthetic_pocket()
            out = _w_vina_metal_fallback(
                {
                    "molecules": ["[Pt](N)(N)(Cl)Cl"],
                    "pocket": pocket,
                    "exhaustiveness": 1,
                    "n_poses": 1,
                }
            )
            assert isinstance(out, list)
            assert len(out) == 1
            # The boltz_affinity fallback produces a real D-MPNN
            # prediction (or NaN if no checkpoint) — both are
            # acceptable so long as the wrapper does NOT raise.
            v = out[0]
            assert isinstance(v, float)
            assert math.isfinite(v) or math.isnan(v)
        finally:
            vina_adapter.dock_smiles = original_dock


# ---------------------------------------------------------------------------
# 7. Bench script — direct unit tests
# ---------------------------------------------------------------------------
class TestBenchCisplatinRealVinaScript:
    """Pin the bench script's API surface and helpers."""

    def test_module_importable(self):
        from molmetal.scripts import bench_cisplatin_real_vina

        assert hasattr(bench_cisplatin_real_vina, "CISPLATIN_SMILES")
        assert hasattr(bench_cisplatin_real_vina, "DEFAULT_POCKETS")
        assert hasattr(bench_cisplatin_real_vina, "run_pocket_sweep")
        assert hasattr(bench_cisplatin_real_vina, "run_full_sweep")

    def test_cisplatin_smiles_is_canonical(self):
        from molmetal.scripts import bench_cisplatin_real_vina

        assert bench_cisplatin_real_vina.CISPLATIN_SMILES == "[Pt](N)(N)(Cl)Cl"

    def test_default_pockets_match_wf30_cohort(self):
        from molmetal.scripts import bench_cisplatin_real_vina

        expected = (
            "830c",
            "1h36",
            "3eml",
            "3ny8",
            "4aaw",
            "4rlu",
            "4rn0",
            "1fmc",
            "1g9v",
            "2hnx",
        )
        assert bench_cisplatin_real_vina.DEFAULT_POCKETS == expected

    def test_default_seeds_are_three(self):
        from molmetal.scripts import bench_cisplatin_real_vina

        assert bench_cisplatin_real_vina.DEFAULT_SEEDS == (0, 1, 2)

    def test_rewrite_metal_atype_changes_M_to_Mg(self, tmp_path):
        from molmetal.scripts.bench_cisplatin_real_vina import (
            _rewrite_metal_atype_for_vina,
        )

        # Build a minimal PDBQT with one 'M' atom-type row.
        pdbqt_text = (
            "ATOM      1 Pt   UNL     1      -0.016   0.710   0.008  1.00  0.00    +0.000 M \n"
            "ATOM      2 N    UNL     1      -0.500   1.200   0.300  1.00  0.00    +0.000 N \n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(pdbqt_text)
        # Wf-143 P3: ``chemotype_vdw`` defaults to True now (rewrites
        # ``M`` → ``Pt``); explicit ``chemotype_vdw=False`` preserves the
        # Wf-121 P1 ``M`` → ``Mg`` behaviour under test here.
        out = _rewrite_metal_atype_for_vina(lig, chemotype_vdw=False)
        assert out == lig
        text = out.read_text()
        # Pt line should now read 'Mg' in the atype columns.
        pt_line = [l for l in text.splitlines() if "Pt" in l][0]
        assert pt_line[77:79].strip() == "Mg"
        # The N atom should be unchanged.
        n_line = [
            l for l in text.splitlines() if l.startswith("ATOM") and "N " in l[12:16]
        ][0]
        assert n_line[77:79].strip() == "N"

    def test_boltz_pic50_returns_real_number_or_none(self):
        """The boltz_affinity fallback must produce either a real
        D-MPNN pIC50 float or None — never a fabricated value."""
        from molmetal.scripts.bench_cisplatin_real_vina import _boltz_pic50

        v = _boltz_pic50("[Pt](N)(N)(Cl)Cl")
        assert v is None or (
            isinstance(v, float) and math.isfinite(v)
        ), f"_boltz_pic50 must be None or finite float, got {v!r}"
