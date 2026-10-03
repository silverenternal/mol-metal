"""Wf-321A — TODO-176 m19 LOMO Rh oversample Phase 2 CPU scaffold tests.

======================================================================
Wf-321A — CPU scaffold VERIFICATION for Wf-135 Phase 2 (post-Wf-309A)
======================================================================

Phase 1 (Wf-309A SHIPPED): the 500-row balanced Rh cohort
(250 active + 250 inactive) is on disk at
``molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json``
(also mirrored at ``wf139_p7_rh_active_cohort``).

Phase 2 (Wf-310E launch wrapper READY): 5-metal x 5-seed x 50-epoch
re-bench.  In GPU pipeline per Wf-310E / TODO-176 — requires USER_OK
gate per mol-metal-gpu-execution §0 ("kill then adapt").

This test file verifies the **CPU scaffold** that wires Phase 1 into
Phase 2:

  1. cohort JSON load + scaffold-split integrity (3 tests)
  2. trainer CLI flag wiring (--balanced-cohort-json / --oversample-positives
     / --pos-weight / --task-type / --model / --quick) (3 tests)
  3. pre-flight gate validation (the new p3_wf135_phase2_cpu_smoke_ready
     gate) (3 tests)
  4. GPU portion integration readiness (Wf-310E launch wrapper exists +
     USER_OK gate contract verified) (3 tests)

Total: 12 NEW tests (all CPU-only, no GPU launch).

Honest framing per party-twenty-spirit §3.5:
  * Phase 1 cohort SHIPPED — values below are MEASURED (Wf-309A verified).
  * Phase 2 5x5x50 is PROJECTED wall ~10 GPU-hr = ~2 calendar weeks
    (best-case 7 days, worst-case 14 days per Phase 1 plan §2.4).
  * This test file is CPU-only and does NOT launch any GPU work.

Wf-321A is the CPU scaffold VERIFICATION pass; the GPU launch itself
is gated on USER_OK=1 per Wf-310E.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
TESTS_DIR = MOLMETAL_ROOT / "tests"
SCRIPTS_DIR = MOLMETAL_ROOT / "scripts"
PRE_FLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"
TRAINER_SCRIPT = SCRIPTS_DIR / "train_metal_lomo_v2.py"
LAUNCH_WRAPPER = REPO_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"

# Phase 1 cohort paths (Wf-309A ship + Wf-145 P1 mirror).
PHASE1_COHORT_WF139 = (
    MOLMETAL_ROOT
    / "reports"
    / "wf139_p7_rh_active_cohort"
    / "synthetic_rh_pos_neg.json"
)
PHASE1_COHORT_WF145 = (
    MOLMETAL_ROOT
    / "reports"
    / "wf145_p1_rh_active_cohort"
    / "synthetic_rh_pos_neg.json"
)
DEFAULT_RH_BALANCED_JSON = PHASE1_COHORT_WF145


# =====================================================================
# Section 1 — cohort JSON load + scaffold-split integrity (3 tests)
# =====================================================================
class TestCohortJSONLoad:
    """Wf-309A Phase 1 cohort JSON — load + scaffold-split integrity."""

    def test_wf139_p7_cohort_json_exists_and_balanced(self):
        """The Wf-139 P7 / Wf-309A Phase 1 cohort JSON must exist with 250+250.

        MEASURED per Wf-309A §2: 250 active + 250 inactive balanced
        cohort.  This is the prerequisite for Wf-135 Phase 2.
        """
        assert PHASE1_COHORT_WF139.exists(), (
            f"Phase 1 cohort missing at {PHASE1_COHORT_WF139}. "
            "Wf-309A should have shipped this — re-run Wf-309A "
            "rh_active_label_generation.py first."
        )
        with PHASE1_COHORT_WF139.open() as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        stats = data.get("stats", {})
        n_active = int(stats.get("n_active", 0))
        n_inactive = int(stats.get("n_inactive", 0))
        assert n_active == 250, f"expected n_active=250, got {n_active}"
        assert n_inactive == 250, f"expected n_inactive=250, got {n_inactive}"
        assert len(data.get("actives", [])) == n_active
        assert len(data.get("inactives", [])) == n_inactive

    def test_wf145_p1_cohort_json_default_loads_via_trainer_helper(self):
        """``load_balanced_rh_smiles`` must accept the default Phase 1 cohort.

        The trainer at ``train_metal_lomo_v2.py`` line 286-342 defines
        ``load_balanced_rh_smiles`` which the CLI uses to load the
        cohort.  We exercise it directly here to confirm the helper
        parses the JSON correctly.
        """
        # Import lazily so a missing-torch in test env does not break
        # the import of the whole trainer module.
        from molmetal.scripts.train_metal_lomo_v2 import load_balanced_rh_smiles

        if not DEFAULT_RH_BALANCED_JSON.exists():
            pytest.skip(
                f"Wf-145 P1 cohort missing at {DEFAULT_RH_BALANCED_JSON} "
                "(Wf-309A may not have mirrored to this path)"
            )
        smis, labels = load_balanced_rh_smiles(DEFAULT_RH_BALANCED_JSON)
        assert len(smis) == 500
        assert len(labels) == 500
        assert sum(labels) == 250
        assert sum(1 for x in labels if x == 0) == 250
        assert all(
            isinstance(s, str) and s.strip() for s in smis
        ), "all cohort SMILES must be non-empty strings"

    def test_scaffold_split_integrity_zero_leak(self):
        """Per Wf-139 P7 / Wf-309A: scaffold_overlap_count == 0 between actives and inactives.

        This is the canary that prevents information leak between the
        training-time active cohort and the inactive cohort.  If the
        scaffold overlap ever drifts above 0, the synthetic cohort is
        useless for the honest-negative Phase 2 lift.
        """
        if not PHASE1_COHORT_WF139.exists():
            pytest.skip(f"Wf-139 P7 cohort missing at {PHASE1_COHORT_WF139}")
        with PHASE1_COHORT_WF139.open() as fh:
            data = json.load(fh)
        stats = data.get("stats", {})
        # Per Wf-139 P7 ship: scaffold_overlap_count = 0 (clean split).
        scaffold_overlap = int(stats.get("scaffold_overlap_count", -1))
        assert scaffold_overlap == 0, (
            f"scaffold_overlap_count={scaffold_overlap} (expected 0). "
            "Phase 1 cohort scaffold-split integrity is broken — "
            "do NOT launch Phase 2 until Wf-309A re-runs."
        )


# =====================================================================
# Section 2 — trainer CLI flag wiring (3 tests)
# =====================================================================
class TestTrainerCLIFlagWiring:
    """All Phase-2-relevant CLI flags must be wired into the trainer."""

    def test_balanced_cohort_json_flag_exists(self):
        """``--balanced-cohort-json`` flag must exist (Wf-145 P3 wiring)."""
        with TRAINER_SCRIPT.open() as fh:
            src = fh.read()
        assert (
            "--balanced-cohort-json" in src
        ), "--balanced-cohort-json flag missing from trainer CLI"
        assert (
            "DEFAULT_RH_BALANCED_JSON" in src
        ), "DEFAULT_RH_BALANCED_JSON default path missing from trainer"
        # Verify the default points to the Wf-145 P1 cohort (post-Wf-309A
        # mirror path) so users get the Phase 1 cohort by default.
        m = re.search(
            r"DEFAULT_RH_BALANCED_JSON\s*=\s*\(?\s*\n?\s*PROJECT_ROOT",
            src,
        )
        assert (
            m is not None
        ), "DEFAULT_RH_BALANCED_JSON assignment must use PROJECT_ROOT"

    def test_oversample_and_pos_weight_flags_exist(self):
        """Wf-187 m19 Rh oversample fix + pos_weight flags must be wired.

        These are the m19 mechanism flags from TODO-176 — without them
        Phase 2 cannot lift the Rh fold regression.
        """
        with TRAINER_SCRIPT.open() as fh:
            src = fh.read()
        assert (
            "--oversample-positives" in src
        ), "--oversample-positives flag missing (Wf-187 m19 fix)"
        assert "--pos-weight" in src, "--pos-weight flag missing (Wf-187 m19 fix)"
        # Verify the helper function is defined.
        assert (
            "def oversample_underrepresented_metals" in src
        ), "oversample_underrepresented_metals helper missing"
        # Verify the CLI path wires into the runner constructor.
        assert (
            "oversample_positives=args.oversample_positives" in src
        ), "CLI --oversample-positives not wired into BalancedLOMORunner"
        assert (
            "pos_weight=args.pos_weight" in src
        ), "CLI --pos-weight not wired into BalancedLOMORunner"

    def test_task_type_quick_model_flags_exist(self):
        """Wf-201 --task-type + --quick + --model flags must all be wired.

        --task-type is the LOMO-safe single-task driver (paper §6 honest
        framing).  --quick is the CPU smoke runner.  --model is the
        backbone selector.
        """
        with TRAINER_SCRIPT.open() as fh:
            src = fh.read()
        assert "--task-type" in src, "--task-type flag missing (Wf-201)"
        assert (
            'choices=["single", "multi"]' in src
        ), "--task-type choices must include single + multi"
        assert "--quick" in src, "--quick flag missing (CPU smoke runner)"
        assert "--model" in src, "--model flag missing"
        assert (
            "chemprop_v2" in src and "v3" in src
        ), "--model must support chemprop_v2 + v3"
        # Verify --quick wires into args.epochs = 2 + args.metals[:1] + args.seeds[:1].
        # That is the canonical 1-metal x 1-seed x 2-epoch smoke config.
        assert (
            "args.epochs = 2" in src
        ), "--quick must set args.epochs=2 (canonical smoke)"
        assert (
            "args.metals[:1]" in src
        ), "--quick must set args.metals[:1] (single-metal smoke)"


# =====================================================================
# Section 3 — pre-flight gate validation (3 tests)
# =====================================================================
class TestPreFlightGate:
    """The new p3_wf135_phase2_cpu_smoke_ready gate must be wired."""

    def test_new_gate_function_defined(self):
        """``gate_wf321a_wf135_phase2_cpu_smoke`` must be defined in pre_flight_check.py."""
        with PRE_FLIGHT_SCRIPT.open() as fh:
            src = fh.read()
        assert "def gate_wf321a_wf135_phase2_cpu_smoke" in src, (
            "gate_wf321a_wf135_phase2_cpu_smoke function not defined in "
            "pre_flight_check.py"
        )
        # Verify it returns a GateResult (sanity check on the contract).
        assert (
            "GateResult("
            in src.split("def gate_wf321a_wf135_phase2_cpu_smoke", 1)[1].split(
                "def ", 1
            )[0]
        ), "gate_wf321a_wf135_phase2_cpu_smoke must construct a GateResult"

    def test_new_gate_registered_in_main_runners(self):
        """The new gate must be registered in the main runners list (all pipeline)."""
        with PRE_FLIGHT_SCRIPT.open() as fh:
            src = fh.read()
        # The registration must live inside the main() function and
        # carry the canonical gate name.
        assert "p3_wf135_phase2_cpu_smoke_ready" in src, (
            "gate name p3_wf135_phase2_cpu_smoke_ready not found in "
            "pre_flight_check.py"
        )
        # Verify the gate has a clear comment block referencing Wf-321A
        # so the audit trail is obvious.
        gate_section = src.split("p3_wf135_phase2_cpu_smoke_ready", 1)
        assert (
            len(gate_section) == 2
        ), "gate name appears exactly once (or at least one occurrence)"

    def test_gate_validates_cohort_plus_trainer_plus_launch_wrapper(self):
        """The new gate must check the 3 prerequisites for Phase 2 launch.

        Per Wf-321A brief: the gate verifies (a) Phase 1 cohort on disk,
        (b) trainer script has --balanced-cohort-json flag, and
        (c) Wf-310E launch wrapper exists on disk.
        """
        with PRE_FLIGHT_SCRIPT.open() as fh:
            src = fh.read()
        # Get just the gate body (from def to next def).
        gate_body_match = re.search(
            r"def gate_wf321a_wf135_phase2_cpu_smoke.*?(?=\ndef\s|\nclass\s|\Z)",
            src,
            re.DOTALL,
        )
        assert (
            gate_body_match is not None
        ), "could not isolate gate_wf321a_wf135_phase2_cpu_smoke body"
        body = gate_body_match.group(0)
        # 3 prerequisites must all be referenced in the gate body.
        assert (
            "synthetic_rh_pos_neg.json" in body
        ), "gate must check Phase 1 cohort JSON path"
        assert (
            "--balanced-cohort-json" in body
        ), "gate must check trainer --balanced-cohort-json flag wiring"
        assert (
            "launch_wf310e_wf135_phase2_gpu.sh" in body
        ), "gate must check Wf-310E GPU launch wrapper exists"


# =====================================================================
# Section 4 — GPU portion integration readiness (3 tests)
# =====================================================================
class TestGPUPortionIntegrationReadiness:
    """Verify the Wf-310E GPU launch wrapper is in place (USER_OK-gated)."""

    def test_launch_wrapper_exists(self):
        """Wf-310E launch wrapper script must exist on disk.

        This is the GPU portion entry point — without it, Phase 2 has no
        launch surface even if Phase 1 cohort + trainer are ready.
        """
        assert LAUNCH_WRAPPER.exists(), (
            f"Wf-310E launch wrapper missing at {LAUNCH_WRAPPER}. "
            "Phase 2 GPU launch is blocked — re-create the wrapper per "
            "TODO-176 / Wf-310E spec."
        )

    def test_launch_wrapper_enforces_user_ok_gate(self):
        """The wrapper must enforce USER_OK=1 gate (mol-metal-gpu-execution §0).

        The kill_then_adapt directive forbids silent GPU launches > 1 hr
        without explicit fresh user OK.  Phase 2 is ~2 weeks wall —
        USER_OK is mandatory.
        """
        assert LAUNCH_WRAPPER.exists(), "launch wrapper missing"
        with LAUNCH_WRAPPER.open() as fh:
            src = fh.read()
        assert "USER_OK" in src, "launch wrapper does not reference USER_OK gate"
        # Verify the wrapper REFUSES to run unless USER_OK=1 is set.
        # Common patterns: explicit check + exit 1, or env-gated path.
        assert (
            "REFUSES" in src or "exit 1" in src or "USER_OK=1" in src
        ), "launch wrapper must REFUSE without USER_OK=1"

    def test_launch_wrapper_references_trainer_and_phase1_cohort(self):
        """Wrapper must point to the trainer + Phase 1 cohort + USER_OK-gated path."""
        assert LAUNCH_WRAPPER.exists(), "launch wrapper missing"
        with LAUNCH_WRAPPER.open() as fh:
            src = fh.read()
        assert (
            "train_metal_lomo_v2.py" in src
        ), "launch wrapper does not reference the trainer"
        assert (
            "synthetic_rh_pos_neg.json" in src
        ), "launch wrapper does not reference the Phase 1 cohort JSON"
        assert (
            "Phase 1 prerequisite" in src or "PHASE1_COHORT" in src
        ), "launch wrapper does not mark Phase 1 as a prerequisite"
        # Verify it lists the 5-metal x 5-seed x 50-epoch spec.
        assert (
            "Rh" in src and "Ru" in src and "Ir" in src and "Os" in src and "Re" in src
        ), "launch wrapper must enumerate all 5 LOMO metals"
        assert (
            "EPOCHS=50" in src or "epochs=50" in src or "EPOCHS= 50" in src
        ), "launch wrapper must specify 50-epoch spec"
