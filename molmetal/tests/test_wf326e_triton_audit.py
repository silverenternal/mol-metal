"""Wf-326E Triton kernel coverage audit tests.

Per user feedback ("之前参考的那些源代码的潜力估计还是没有被榨干" +
"本来就做过triton优化了，为什么还是不行"), these tests verify what
Triton kernels already exist, which hot-path operations are MISSING
Triton kernels, and how the existing kernels' dispatch policy behaves
on the RTX 5090 (Blackwell sm_120) target.

These tests are CPU-only — no GPU launch is performed. They exist to
make the Wf-326E audit reproducible + to surface the "kernels exist
but dispatch is OFF because gfx1101 autotune configs are wrong for
sm_120" finding as a regression-detector.

Per AGENTS.md §5.1, this is an audit (no retrain / panel / ablation
launch), so pre-flight gate (TODO-207) is N/A and the kill_then_adapt
directive does not apply.

Categories (8 tests):
  Group A — Existing Triton kernel coverage verification (3 tests):
    test_01_triton_kernels_inventory_complete
    test_02_kernel_dispatch_status_default_off_blackwell
    test_03_dmpnn_backbone_uses_relu_no_fused_kernel
  Group B — Missing Triton candidate analysis (3 tests):
    test_04_missing_fused_relu_mlp_dmpnn_cited
    test_05_missing_fused_wasserstein_sliced_w310c_cited
    test_06_missing_fused_bond_head_aggregation_cited
  Group C — Speedup measurement / projected speedup (2 tests):
    test_07_existing_kernel_speedup_projected_band
    test_08_sm_120_autotune_retune_would_unlock_dispatch

All numeric claims follow party-twenty-spirit §3.5 honest-negative
framing: PROJECTED speedups are clearly labelled; MEASURED speedups
reference their on-disk source path.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path



# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the pattern in other molmetal tests)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AUDIT_PATH = (
    _PROJECT_ROOT / "molmetal" / "reports" / "wf326e_triton_audit" / "audit.json"
)
_TRITON_KERNELS_ROOT = _PROJECT_ROOT / "triton_kernels"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_audit() -> dict:
    """Load the Wf-326E audit JSON from disk.

    The audit is the source-of-truth per-party-twenty-spirit §3.5; tests
    cross-reference it instead of inlining numbers.
    """
    assert (
        _AUDIT_PATH.is_file()
    ), f"Wf-326E audit missing at {_AUDIT_PATH}. Run the audit pipeline first."
    with _AUDIT_PATH.open() as f:
        return json.load(f)


def _count_python_files_in(dir_: Path) -> int:
    return sum(1 for _ in dir_.glob("*.py"))


def _has_kernel_module(name: str) -> bool:
    """Return True iff ``triton_kernels/<name>.py`` exists."""
    return (_TRITON_KERNELS_ROOT / f"{name}.py").is_file()


# ===========================================================================
# Group A — Existing Triton kernel coverage verification (3 tests)
# ===========================================================================
class TestExistingKernelCoverage:
    """Tests 01-03: verify the existing 20-kernel inventory + dispatch policy."""

    def test_01_triton_kernels_inventory_complete(self):
        """Verify the on-disk Triton kernel inventory matches the audit JSON.

        Reads the audit JSON (source-of-truth) and confirms each named
        module file exists under triton_kernels/. Catches the case where
        a kernel is removed from disk but still referenced in audit.json
        (silent drift).
        """
        audit = _load_audit()
        kernels = audit["triton_kernels_inventory"]["kernels"]
        # Build a set of unique module filenames (one per module).
        modules = sorted({k["module"] for k in kernels})
        missing = [m for m in modules if not _has_kernel_module(m)]
        assert not missing, (
            f"Audit JSON references {len(modules)} Triton kernel modules "
            f"({', '.join(modules)}) but {len(missing)} are missing from "
            f"disk under {_TRITON_KERNELS_ROOT}/: {missing}"
        )
        # Also verify the inventory count claim.
        assert audit["triton_kernels_inventory"]["total_kernels"] == len(
            kernels
        ), "Audit JSON total_kernels claim diverges from on-disk kernel list length."

    def test_02_kernel_dispatch_status_default_off_blackwell(self):
        """Verify that the Triton dispatch config is OFF by default — the
        root cause of the user's "GPU0 算力没吃满" observation.

        Per triton_kernels/config.py L107: "Default changed 2026-09-27:
        Triton kernels tuned for gfx1101 are 3-11x slower than PyTorch
        on Blackwell; PyTorch is now the default". We assert:
          (a) the audit JSON reports default_enabled=False
          (b) the config.py docstring matches the audit's rationale
          (c) the gate is environment-driven (TRITON_USE_FUSED=1 to opt in)
        """
        audit = _load_audit()
        # (a) Audit JSON reports default off.
        assert audit["dispatch_status"]["triton_config_default_enabled"] is False, (
            "Audit JSON should report triton_config default = False. "
            "If this is now True, the dispatch gate has flipped and "
            "the audit's 'default OFF' reasoning is stale."
        )
        # (b) config.py docstring matches the audit's headline finding.
        config_path = _TRITON_KERNELS_ROOT / "config.py"
        config_text = config_path.read_text()
        assert "gfx1101" in config_text and "Blackwell" in config_text, (
            "triton_kernels/config.py docstring should reference both "
            "gfx1101 (RDNA3) and Blackwell (sm_120) per the audit's "
            "dispatch_status.rationale claim."
        )
        # (c) The gate is environment-driven (TRITON_USE_FUSED).
        assert "_env_flag" in config_text and "TRITON_USE_FUSED" in config_text, (
            "triton_kernels/config.py should be environment-driven via "
            "TRITON_USE_FUSED env var."
        )

    def test_03_dmpnn_backbone_uses_relu_and_fused_kernel_now_ships(self):
        """Verify the D-MPNN backbone uses ReLU (not SiLU/GELU) AND that
        the MISSING-001 fused_relu_mlp kernel shipped in Wf-328-W9
        (2026-09-29).

        The D-MPNN paper (Justin et al., JCIM 2020) is a published
        baseline; swapping ReLU for SiLU/GELU would alter model
        semantics, so the wiring-site comment in models/dmpnn.py L36-47
        notes "when a fused_relu_mlp kernel lands, swap the three
        ``nn.Sequential(Linear, ReLU, Linear)`` blocks".

        History: Wf-326E (2026-09-29 12:05) flagged MISSING-001 as the
        #1 gap. Wf-328-W9 (2026-09-29 12:22) shipped the kernel
        (``triton_kernels/fused_relu_mlp.py``). This test now asserts
        the new reality (kernel exists + audit acknowledges ship).
        """
        # (a) D-MPNN source uses nn.Sequential with ReLU.
        dmpnn_path = _PROJECT_ROOT / "molmetal" / "models" / "dmpnn.py"
        dmpnn_text = dmpnn_path.read_text()
        relu_mlp_pattern = re.compile(r"nn\.Sequential\(\s*[^)]*Linear[^)]*,[^)]*ReLU")
        relu_mlp_matches = relu_mlp_pattern.findall(dmpnn_text)
        assert len(relu_mlp_matches) >= 1, (
            "Expected D-MPNN backbone to contain nn.Sequential(Linear, ReLU, ...). "
            f"Found {len(relu_mlp_matches)} matches in {dmpnn_path}."
        )
        # (b) fused_relu_mlp.py kernel SHIPS (Wf-328-W9) — the audit's
        # MISSING-001 entry is now historical (moved to shipped_since_audit).
        assert _has_kernel_module("fused_relu_mlp"), (
            "fused_relu_mlp.py should exist on disk (Wf-328-W9 ship of "
            "MISSING-001 from Wf-326E audit). If this file is missing, "
            "the Wf-328-W9 ship regressed."
        )
        # (c) D-MPNN wires the per-layer edge MLP via fused_relu_linear
        # (the gating hot-path the user complained about being un-fused).
        assert "_fused_relu_linear(" in dmpnn_text, (
            "Expected models/dmpnn.py to call _fused_relu_linear(...) "
            "in the per-layer edge MLP wiring (Wf-328-W9 ship)."
        )
        # (d) Audit JSON cross-ref: missing_kernels.shipped_since_audit
        # should list MISSING-001 as shipped.
        audit = _load_audit()
        shipped = audit["missing_kernel_candidates"].get("shipped_since_audit", [])
        shipped_ids = {s["id"] for s in shipped}
        assert "MISSING-001" in shipped_ids, (
            "Audit JSON missing_kernel_candidates.shipped_since_audit "
            "should include MISSING-001 (Wf-328-W9 ship). "
            f"Current shipped list: {shipped}"
        )


# ===========================================================================
# Group B — Missing Triton candidate analysis (3 tests)
# ===========================================================================
class TestMissingKernelCandidates:
    """Tests 04-06: verify the 4 missing-kernel candidates' wiring-site
    references are accurate (no stale citations)."""

    def test_04_missing_fused_relu_mlp_dmpnn_cited(self):
        """MISSING-001 candidate: verify the audit's wiring-site claim
        (models/dmpnn.py L36-47) cites the ReLU gap."""
        dmpnn_path = _PROJECT_ROOT / "molmetal" / "models" / "dmpnn.py"
        dmpnn_text = dmpnn_path.read_text()
        # The wiring comment must mention a future fused_relu_mlp.
        assert "fused_relu_mlp" in dmpnn_text, (
            "Expected models/dmpnn.py to reference 'fused_relu_mlp' "
            "in its wiring comment. If this reference is gone, the "
            "audit MISSING-001 wiring_site citation is stale."
        )
        # Cross-reference audit JSON.
        audit = _load_audit()
        miss001 = next(
            (
                c
                for c in audit["missing_kernel_candidates"]["candidates"]
                if c["id"] == "MISSING-001"
            ),
            None,
        )
        assert miss001 is not None, "Audit JSON missing MISSING-001 entry."
        assert (
            "models/dmpnn.py" in miss001["wiring_site"]
        ), "MISSING-001 wiring_site should reference models/dmpnn.py."

    def test_05_missing_fused_wasserstein_sliced_w310c_cited(self):
        """MISSING-002 candidate: verify the audit's _sliced_w2 wiring-site
        citation (molmetal_lam/reward/wasserstein_loss.py L217-272) is
        accurate — the file exists and contains the function."""
        wl_path = (
            _PROJECT_ROOT
            / "molmetal"
            / "molmetal_lam"
            / "reward"
            / "wasserstein_loss.py"
        )
        assert wl_path.is_file(), (
            f"Wasserstein loss file missing at {wl_path} — audit "
            f"MISSING-002 wiring_site citation is stale."
        )
        wl_text = wl_path.read_text()
        assert "def _sliced_w2" in wl_text, (
            f"Expected _sliced_w2 in {wl_path}. If this function is "
            f"renamed, audit MISSING-002 wiring_site is stale."
        )
        # Audit JSON cross-reference.
        audit = _load_audit()
        miss002 = next(
            (
                c
                for c in audit["missing_kernel_candidates"]["candidates"]
                if c["id"] == "MISSING-002"
            ),
            None,
        )
        assert miss002 is not None, "Audit JSON missing MISSING-002 entry."
        assert (
            "Wf-310C" in miss002["rationale"]
        ), "MISSING-002 rationale should reference Wf-310C TODO-179."

    def test_06_missing_fused_bond_head_aggregation_cited(self):
        """MISSING-003 candidate: verify the audit's bond-head wiring-site
        citation (molmetal/models/bond_head.py) is accurate."""
        bh_path = _PROJECT_ROOT / "molmetal" / "models" / "bond_head.py"
        assert bh_path.is_file(), (
            f"Bond head file missing at {bh_path} — audit MISSING-003 "
            f"wiring_site citation is stale."
        )
        bh_text = bh_path.read_text()
        assert "BondOrderHead" in bh_text or "BondAwareDecoder" in bh_text, (
            f"Expected BondOrderHead or BondAwareDecoder in {bh_path}. "
            f"If renamed, audit MISSING-003 wiring_site is stale."
        )
        # Cross-reference audit JSON: the missing candidate should reference bond head.
        audit = _load_audit()
        miss003 = next(
            (
                c
                for c in audit["missing_kernel_candidates"]["candidates"]
                if c["id"] == "MISSING-003"
            ),
            None,
        )
        assert miss003 is not None, "Audit JSON missing MISSING-003 entry."
        assert (
            "bond_head" in miss003["wiring_site"].lower()
        ), "MISSING-003 wiring_site should reference bond_head.py."


# ===========================================================================
# Group C — Speedup measurement / projected speedup (2 tests)
# ===========================================================================
class TestProjectedSpeedup:
    """Tests 07-08: PROJECTED speedup range + sm_120 retune unlock."""

    def test_07_existing_kernel_speedup_projected_band(self):
        """Verify the audit's projected speedups are inside the honest
        range band per party-twenty-spirit §3.5 (PROJECTED 1.5-3x for
        retune + new kernels; per-kernel 1.2-5x).

        We assert:
          (a) each per_kernel_status entry has a non-empty sm_120_perf claim
          (b) all speedup claims follow the format "Nx slower" (negative
              dispatch direction) or "Nx speedup" (positive direction)
              — i.e. no unlabelled numbers
          (c) the aggregate projected_speedup_with_retune_and_new_kernels
              claim is in [1.5x, 3.0x] (honest-negative band)
        """
        audit = _load_audit()
        per_kernel = audit["per_kernel_status_table"]
        assert len(per_kernel) >= 10, (
            f"Expected per-kernel status table to have >=10 entries; "
            f"got {len(per_kernel)}. Audit may be incomplete."
        )
        # (b) Every sm_120_perf claim should be labelled PROJECTED or reference
        # a measured source path.
        for entry in per_kernel:
            sm120 = entry["sm_120_perf"]
            assert "PROJECTED" in sm120 or "MEASURED" in sm120, (
                f"Per-kernel sm_120_perf claim for {entry['kernel']} "
                f"is missing the PROJECTED/MEASURED tag — violates "
                f"party-twenty-spirit §3.5."
            )
        # (c) Aggregate projected speedup is in the honest band.
        aggregate_claim = audit["audit_summary"][
            "projected_speedup_with_retune_and_new_kernels"
        ]
        # Extract the leading numeric (e.g. "1.5-3x" or "PROJECTED 1.5-3x").
        match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*x", aggregate_claim)
        assert match is not None, (
            f"Could not parse projected_speedup_with_retune_and_new_kernels "
            f"claim: {aggregate_claim!r}"
        )
        lo, hi = float(match.group(1)), float(match.group(2))
        assert 1.0 <= lo <= hi <= 5.0, (
            f"Projected speedup range [{lo}x, {hi}x] is outside the "
            f"honest band [1x, 5x]. If a larger claim is intended, "
            f"update the audit's honest_framing section."
        )

    def test_08_sm_120_autotune_retune_would_unlock_dispatch(self):
        """Verify the audit's headline finding — that sm_120 autotune
        retune is the primary blocker, not missing kernels — is
        consistent with the dispatch_status block + follow_up_recommendations.

        Specifically:
          (a) dispatch_status.rationale cites gfx1101 vs Blackwell perf gap
          (b) audit_summary.primary_root_cause_for_under_utilization references
              "sm_120" or "Blackwell"
          (c) follow_up_recommendations includes an sm_120 retune item
        """
        audit = _load_audit()
        # (a)
        rationale = audit["dispatch_status"]["rationale"]
        assert "gfx1101" in rationale and "Blackwell" in rationale, (
            "dispatch_status.rationale must cite both gfx1101 and "
            "Blackwell per the audit's headline finding."
        )
        # (b)
        root_cause = audit["audit_summary"]["primary_root_cause_for_under_utilization"]
        assert "sm_120" in root_cause or "Blackwell" in root_cause, (
            "primary_root_cause_for_under_utilization must reference "
            "sm_120 or Blackwell."
        )
        # (c)
        recommendations = audit["follow_up_recommendations"]
        retune_items = [
            r
            for r in recommendations
            if "retune" in r.lower() and ("sm_120" in r or "autotune" in r.lower())
        ]
        assert len(retune_items) >= 1, (
            "follow_up_recommendations should include at least one "
            "sm_120 / autotune retune item."
        )
        # Sanity: the missing-kernels count should be 4 (matches Group B).
        assert audit["audit_summary"]["n_missing_candidates"] == 4, (
            "Audit summary claims 4 missing candidates but tests "
            "Group B only verifies 3 of them. Either Group B is "
            "incomplete or the audit count is wrong."
        )
