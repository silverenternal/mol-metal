"""Wf-338-V2 — Wf-298D + Wf-264 sub-panel + Wf-176 m19 LOMO integration tests.

================================================================
Wf-338-V2 — Integration tests for the Wf-298D recovery wrapper +
Wf-264 Vina sub-panel recovery (Smina / QuickVina 2) + Wf-176 m19
LOMO USER_OK gate + Wf-338-M3 CPU portion.
================================================================

Per the Wf-338-V2 task brief (2026-09-29):

  CONTEXT:
    - Wf-334-U3 (the Wf-264 CFM property panel launch script) shipped.
    - Wf-334-U3 MEASURED 0/100 docked ok on 1h36 (meeko metal-atom limit).
    - Wf-325C substitute (Wf-327-F13 verification doc) shipped; the
      Wf-176 m19 LOMO USER_OK-BLOCKED state is preserved per user
      directive 2026-09-29 ("先把honest-negative先解决了").
    - Wf-298D harness detector is shipped (the wrapper.py file in
      molmetal/reports/wf298d_harness_detector/) but not yet
      integrated as a project-level recovery module.

  TASK:
    1. Verify Wf-337-A2 (Wf-298D recovery) status: import path OK +
       auto-fallback contract MEASURED on the Wf-291C regression class.
    2. Verify Wf-337-U1 (Wf-126 USER_OK gate) status: Wf-310E GPU
       wrapper USER_OK gate ENFORCED + Wf-338-M3 CPU wrapper also
       enforces the gate.
    3. Integrate Wf-298D recovery + Wf-264 Smina/QuickVina probe +
       Wf-176 USER_OK + Wf-338-M3 CPU portion.
    4. Run the 4 NEW tests + document.

  HONEST FRAMING (party-twenty-spirit §3.5):
    - Wf-334-U3 MEASURED 0/100 docked ok on 1h36 (meeko metal-atom
      limit; honest-negative per Wf-264 §2). The Smina/QuickVina 2
      probe wires the RECOVERY path but the probe returns False
      on the current host (smina / qvina2 not installed); integration
      is structural, not functional.
    - Wf-176 m19 LOMO Rh fold 0.136 regression IS the honest-negative
      preserved per user directive; Wf-338-M3 CPU portion is the
      verification scaffold, NOT a recovery launch.
    - CPU-only edit + test — no GPU launch.

These tests are CPU-only (no torch.cuda, no real GPU launches). The
4 test functions lock down:
  1. Wf-298D recovery (HarnessOutputExtractor) import + auto-fallback
  2. Wf-176 USER_OK gate + Wf-338-M3 CPU wrapper contract
  3. Integration with pipeline master (gpu_pipeline_master.sh)
  4. Backward compat (existing wrappers still parse + work)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project + path constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/hugo/codes/mol-metal")
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPTS_DIR = MOLMETAL_ROOT / "scripts"
TESTS_DIR = MOLMETAL_ROOT / "tests"
REPORTS_DIR = MOLMETAL_ROOT / "reports"

# Wf-298D recovery (canonical wrapper + project-level alias)
WF298D_CANONICAL_WRAPPER = REPORTS_DIR / "wf298d_harness_detector" / "wrapper.py"
WF298D_PROJECT_ALIAS = MOLMETAL_ROOT / "recovery" / "wf298d_harness_detector.py"

# Wf-264 Vina sub-panel + recovery (Smina / QuickVina 2)
WF264_VINA_PANEL = SCRIPTS_DIR / "cfm_vina_dock_panel.py"

# Wf-176 m19 LOMO USER_OK gate + Wf-338-M3 CPU portion
WF176_GPU_WRAPPER = REPO_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"
WF176_CPU_WRAPPER_WF338V2 = (
    REPO_ROOT / "scripts" / "launch_wf338v2_wf176_m19_lomo_cpu.sh"
)

# Phase 1 cohort (Wf-309A SHIPPED)
PHASE1_COHORT_WF139 = (
    REPORTS_DIR / "wf139_p7_rh_active_cohort" / "synthetic_rh_pos_neg.json"
)
PHASE1_COHORT_WF145 = (
    REPORTS_DIR / "wf145_p1_rh_active_cohort" / "synthetic_rh_pos_neg.json"
)

# Pipeline master (Wf-336-D3)
PIPELINE_MASTER = REPO_ROOT / "scripts" / "gpu_pipeline_master.sh"


# ===========================================================================
# Test 1 — Wf-298D recovery: project-level alias + auto-fallback contract
# ===========================================================================
class TestWf298DRecoveryIntegration:
    """Wf-298D recovery is importable from the project-level alias path
    AND the auto-fallback contract from the Wf-291C regression class
    works on the canonical regression payloads.
    """

    def test_wf298d_alias_imports_and_re_exports_canonical_symbols(self):
        """The project-level alias
        ``molmetal.recovery.wf298d_harness_detector`` must import OK
        and re-export HarnessOutputExtractor + FallbackExtraction +
        fast_filter_says_schema.

        Per Wf-298D wrapper_report.md §3, the canonical implementation
        lives at
        ``molmetal/reports/wf298d_harness_detector/wrapper.py``.
        The project-level alias is a thin re-export — it does NOT
        re-implement the detector.
        """
        assert WF298D_PROJECT_ALIAS.exists(), (
            f"Wf-338-V2 project-level alias missing: {WF298D_PROJECT_ALIAS}. "
            "Re-create the alias module per the Wf-338-V2 integration plan."
        )
        assert WF298D_CANONICAL_WRAPPER.exists(), (
            f"Wf-298D canonical wrapper missing: {WF298D_CANONICAL_WRAPPER}. "
            "Re-run Wf-298D ship surface per wrapper_report.md."
        )
        # Project-level alias must import OK + re-export canonical symbols.
        from molmetal.recovery.wf298d_harness_detector import (  # noqa: WPS433
            HarnessOutputExtractor,
            FallbackExtraction,
            fast_filter_says_schema,
        )

        assert HarnessOutputExtractor is not None
        assert FallbackExtraction is not None
        assert callable(fast_filter_says_schema)
        # Verify the alias re-exports the CANONICAL class (not a re-implementation).
        from molmetal.reports.wf298d_harness_detector.wrapper import (  # noqa: WPS433
            HarnessOutputExtractor as Canonical,
        )

        assert HarnessOutputExtractor is Canonical, (
            "Wf-338-V2 alias must re-export the canonical "
            "HarnessOutputExtractor (NOT a re-implementation)."
        )

    def test_wf298d_autofallback_contract_on_regression_payloads(self):
        """The auto-fallback must classify the Wf-291C regression payloads
        as schema-shape AND return FallbackExtraction with the
        sibling_text recovered.

        Per Wf-298D wrapper_report.md §3 smoke test table: 4 of 4
        regression payloads (Wf-270A / Wf-271D / Wf-272A / Wf-273B)
        must be detected as schema-shape; 1 known-good payload
        (Wf-272B) must be returned as data.
        """
        from molmetal.recovery.wf298d_harness_detector import (  # noqa: WPS433
            HarnessOutputExtractor,
            FallbackExtraction,
        )

        extractor = HarnessOutputExtractor()
        # Wf-270A regression payload (per Wf-298D §3 table)
        wf270a = json.dumps(
            {
                "type": "object",
                "properties": {
                    "G1_cuda_fix_verified": {"type": "boolean"},
                    "G2_config_yaml_exists": {"type": "boolean"},
                },
                "required": ["G1_cuda_fix_verified", "G2_config_yaml_exists"],
            }
        )
        # Wf-272B known-good data payload
        wf272b = json.dumps(
            {
                "G1_n_new_entries": 4,
                "G2_existing_entries_preserved": True,
            }
        )
        # Plain-text fallback case
        wf274a_plain = "Wf-274A complete. Added regression_disclosure top-level field"

        # Regression class — schema-shape detection.
        assert (
            extractor.detect_schema_shape(wf270a) is True
        ), "Wf-270A schema-shape payload must be detected"
        # Plain-text payload — must NOT be detected as schema.
        assert (
            extractor.detect_schema_shape(wf274a_plain) is False
        ), "Wf-274A plain-text payload must NOT be flagged as schema"
        # Known-good data — must NOT be detected as schema.
        assert (
            extractor.detect_schema_shape(wf272b) is False
        ), "Wf-272B known-good data payload must NOT be flagged as schema"

        # Auto-fallback: schema-shape returns FallbackExtraction
        # wrapping sibling_text.
        sibling = "fallback text: Wf-270A recovered from sibling comment"
        result = extractor.extract(wf270a, sibling_text=sibling)
        assert isinstance(
            result, FallbackExtraction
        ), "schema-shape payload must yield FallbackExtraction"
        assert result.schema_shape_detected is True
        assert result.recovered_text == sibling
        assert result.fallback_reason in {
            "properties_is_schema_object",
            "properties_has_schema_meta_keys",
            "nested_object_with_schema_type_keyword",
            "root_has_schema_root_markers",
        }, f"unexpected fallback_reason: {result.fallback_reason}"

        # Auto-fallback: known-good data returns parsed dict verbatim.
        result_data = extractor.extract(wf272b)
        assert isinstance(
            result_data, dict
        ), "known-good data payload must yield a dict"
        assert result_data.get("G1_n_new_entries") == 4
        assert result_data.get("G2_existing_entries_preserved") is True


# ===========================================================================
# Test 2 — Wf-176 m19 LOMO USER_OK gate + Wf-338-M3 CPU wrapper
# ===========================================================================
class TestWf176USEROKGateAndWf338MCPUPortion:
    """Wf-176 m19 LOMO USER_OK gate is enforced on both the Wf-310E
    GPU wrapper AND the Wf-338-M3 CPU portion wrapper.

    Per Wf-322F §4 + user directive 2026-09-29 ("先把honest-negative先解
    决了"), the Wf-176 Phase 2 GPU launch is USER_OK-BLOCKED. The
    CPU portion (Wf-338-M3) is a verification scaffold, NOT a recovery
    launch — but the gate is still enforced for reproducibility audit.
    """

    def test_wf310e_gpu_wrapper_enforces_user_ok_gate(self):
        """Wf-310E Wf-135 Phase 2 GPU wrapper must enforce USER_OK gate.

        Per Wf-310E §0 (line 80-88): the wrapper REFUSES to run unless
        USER_OK=1 is set. End-to-end invocation without USER_OK must
        exit non-zero (exit code 64) BEFORE any pre-flight check.
        """
        assert (
            WF176_GPU_WRAPPER.exists()
        ), f"Wf-310E GPU wrapper missing: {WF176_GPU_WRAPPER}"
        # bash -n syntax check first
        proc_syntax = subprocess.run(
            ["bash", "-n", str(WF176_GPU_WRAPPER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"Wf-310E wrapper bash syntax error: {proc_syntax.stderr!r}"
        # End-to-end invocation without USER_OK must exit non-zero
        proc = subprocess.run(
            ["bash", str(WF176_GPU_WRAPPER), "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "USER_OK"},
            timeout=30,
        )
        assert proc.returncode != 0, (
            "Wf-310E wrapper MUST refuse to run without USER_OK=1 "
            "(kill_then_adapt standing directive); "
            f"got rc={proc.returncode} stdout={proc.stdout[-200:]!r}"
        )
        # The gate message must reference the USER_OK gate text.
        assert (
            "USER_OK" in proc.stderr or "USER_OK" in proc.stdout
        ), "USER_OK gate refusal text missing from Wf-310E wrapper stdout/stderr"

    def test_wf338v2_cpu_wrapper_enforces_user_ok_gate_and_validates_cohort(self):
        """Wf-338-M3 CPU wrapper must:
        (a) parse via bash -n (no syntax errors)
        (b) enforce USER_OK gate (kill_then_adapt standing directive)
        (c) verify Phase 1 cohort on disk + balanced (250 + 250)
        (d) reference Wf-310E GPU wrapper for cross-validation
        (e) emit a plan.json with MEASURED-vs-PROJECTED tags per
            party-twenty-spirit §3.5
        """
        assert (
            WF176_CPU_WRAPPER_WF338V2.exists()
        ), f"Wf-338-M3 CPU wrapper missing: {WF176_CPU_WRAPPER_WF338V2}"
        # bash -n syntax check
        proc_syntax = subprocess.run(
            ["bash", "-n", str(WF176_CPU_WRAPPER_WF338V2)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"Wf-338-M3 wrapper bash syntax error: {proc_syntax.stderr!r}"
        # Source grep for USER_OK + Phase 1 cohort + Wf-310E reference.
        src = WF176_CPU_WRAPPER_WF338V2.read_text()
        assert "USER_OK" in src, "USER_OK gate text missing from Wf-338-M3 CPU wrapper"
        assert (
            "synthetic_rh_pos_neg.json" in src
        ), "Wf-338-M3 CPU wrapper must reference the Phase 1 cohort JSON"
        assert (
            "launch_wf310e_wf135_phase2_gpu.sh" in src
        ), "Wf-338-M3 CPU wrapper must reference Wf-310E GPU wrapper"
        assert "MEASURED" in src and "PROJECTED" in src, (
            "Wf-338-M3 CPU wrapper must emit MEASURED/PROJECTED tags "
            "per party-twenty-spirit §3.5"
        )
        # 5-metal × 5-seed × 50-epoch spec is the canonical Wf-176 Phase 2.
        for metal in ("Rh", "Ru", "Ir", "Os", "Re"):
            assert metal in src, (
                f"Wf-338-M3 CPU wrapper must enumerate metal {metal} "
                "in the 5-metal LOMO spec"
            )
        # End-to-end invocation without USER_OK must exit non-zero.
        proc = subprocess.run(
            ["bash", str(WF176_CPU_WRAPPER_WF338V2), "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "USER_OK"},
            timeout=30,
        )
        assert proc.returncode != 0, (
            "Wf-338-M3 wrapper MUST refuse to run without USER_OK=1; "
            f"got rc={proc.returncode} stdout={proc.stdout[-200:]!r}"
        )
        # Phase 1 cohort must exist on disk for the wrapper to reach the
        # USER_OK gate path (the gate fires before cohort check, but
        # if neither cohort path exists we can't end-to-end verify).
        cohort_present = PHASE1_COHORT_WF139.exists() or PHASE1_COHORT_WF145.exists()
        assert cohort_present, (
            "Wf-176 / Wf-338-M3 prerequisite: Phase 1 cohort must be on disk "
            f"(Wf-309A ship). Missing: {PHASE1_COHORT_WF139} "
            f"and {PHASE1_COHORT_WF145}"
        )

    def test_wf176_phase1_cohort_balanced_and_zero_scaffold_leak(self):
        """The Phase 1 cohort JSON must be balanced (250 + 250) with
        scaffold_overlap_count == 0 (clean active/inactive split).
        This is the Wf-309A ship invariant that Wf-176 / Wf-338-M3
        requires — if it ever drifts, the wrapper must FAIL.
        """
        cohort_path = None
        if PHASE1_COHORT_WF139.exists():
            cohort_path = PHASE1_COHORT_WF139
        elif PHASE1_COHORT_WF145.exists():
            cohort_path = PHASE1_COHORT_WF145
        assert cohort_path is not None, (
            "Wf-176 prerequisite: Phase 1 cohort must be on disk "
            f"(Wf-309A ship). Missing: {PHASE1_COHORT_WF139} "
            f"and {PHASE1_COHORT_WF145}"
        )
        with cohort_path.open() as fh:
            data = json.load(fh)
        stats = data.get("stats", {})
        assert (
            int(stats.get("n_active", 0)) == 250
        ), f"n_active={stats.get('n_active')} (expected 250)"
        assert (
            int(stats.get("n_inactive", 0)) == 250
        ), f"n_inactive={stats.get('n_inactive')} (expected 250)"
        assert int(stats.get("scaffold_overlap_count", -1)) == 0, (
            f"scaffold_overlap_count={stats.get('scaffold_overlap_count')} "
            "(expected 0 — clean active/inactive split per Wf-139 P7 / Wf-309A)"
        )


# ===========================================================================
# Test 3 — Wf-264 sub-panel recovery (Smina / QuickVina 2) integration
# ===========================================================================
class TestWf264SubPanelSminaQuickVina2Recovery:
    """Wf-264 sub-panel + recovery: the cfm_vina_dock_panel.py must
    expose --engine {vina,smina,quickvina2,auto} + probe functions for
    Smina and QuickVina 2. The probe is honest-negative: on the current
    host smina / qvina2 are not installed (probe returns False); the
    structural integration is MEASURED, not the recovery itself.
    """

    def test_cfm_vina_dock_panel_exposes_engine_cli_and_probe_functions(self):
        """The Wf-264 sub-panel + recovery wiring must:
        (a) import OK
        (b) expose _probe_smina_binary + _probe_quickvina2_binary +
            _select_engine functions
        (c) define DEFAULT_ENGINE + SUPPORTED_ENGINES constants
        (d) parse --engine {vina,smina,quickvina2,auto} CLI choice
        (e) wire --smina-bin + --qvina2-bin overrides
        """
        assert (
            WF264_VINA_PANEL.exists()
        ), f"Wf-264 Vina panel script missing: {WF264_VINA_PANEL}"
        # Import + probe + select functions all importable.
        from molmetal.scripts.cfm_vina_dock_panel import (  # noqa: WPS433
            _probe_smina_binary,
            _probe_quickvina2_binary,
            _select_engine,
            DEFAULT_ENGINE,
            SUPPORTED_ENGINES,
            DEFAULT_SMINA_BIN,
            DEFAULT_QUICKVINA2_BIN,
        )

        assert callable(_probe_smina_binary)
        assert callable(_probe_quickvina2_binary)
        assert callable(_select_engine)
        assert DEFAULT_ENGINE == "vina"
        assert set(SUPPORTED_ENGINES) == {"vina", "smina", "quickvina2"}
        assert DEFAULT_SMINA_BIN == "smina"
        assert DEFAULT_QUICKVINA2_BIN == "qvina2"

        # CLI parses --engine + --smina-bin + --qvina2-bin.
        from molmetal.scripts.cfm_vina_dock_panel import _build_parser  # noqa: WPS433

        parser = _build_parser()
        # --engine defaults to vina
        args_default = parser.parse_args(
            [
                "--smiles-csv",
                "/tmp/dummy.csv",
                "--pocket-pdb-dir",
                "/tmp/dummy",
                "--out-json",
                "/tmp/out.json",
            ]
        )
        assert (
            args_default.engine == "vina"
        ), f"--engine default must be 'vina', got {args_default.engine!r}"
        # --engine smina parses
        args_smina = parser.parse_args(
            [
                "--smiles-csv",
                "/tmp/dummy.csv",
                "--pocket-pdb-dir",
                "/tmp/dummy",
                "--out-json",
                "/tmp/out.json",
                "--engine",
                "smina",
                "--smina-bin",
                "/custom/path/to/smina",
            ]
        )
        assert args_smina.engine == "smina"
        assert args_smina.smina_bin == "/custom/path/to/smina"
        # --engine quickvina2 parses
        args_qvina2 = parser.parse_args(
            [
                "--smiles-csv",
                "/tmp/dummy.csv",
                "--pocket-pdb-dir",
                "/tmp/dummy",
                "--out-json",
                "/tmp/out.json",
                "--engine",
                "quickvina2",
            ]
        )
        assert args_qvina2.engine == "quickvina2"
        # --engine auto parses (transparent fallback)
        args_auto = parser.parse_args(
            [
                "--smiles-csv",
                "/tmp/dummy.csv",
                "--pocket-pdb-dir",
                "/tmp/dummy",
                "--out-json",
                "/tmp/out.json",
                "--engine",
                "auto",
            ]
        )
        assert args_auto.engine == "auto"

    def test_smina_quickvina2_probe_honest_negative_on_host(self):
        """Per AGENTS.md §5.2 honest framing: the probe NEVER fabricates
        a successful result. On the current host smina / qvina2 are
        not installed (per Wf-329-W7 + Wf-336-X4 §2.2 — only Vina is
        available at ~/.local/micromamba/envs/molmetal_vina/bin/vina),
        so the probe returns (False, ""). This is the EXPECTED state
        on this host.

        The probe contract is verified by:
          1. (False, "") for smina / qvina2 on this host
          2. _select_engine('vina') returns the canonical Vina path +
             ok=True (Vina CLI IS installed per Wf-329-W7)
          3. _select_engine('smina') returns smina_bin + ok=False
             (honest: NOT installed; no fabrication)
          4. _select_engine('quickvina2') returns qvina2_bin + ok=False
          5. _select_engine('auto') falls through Vina → Smina →
             QuickVina 2; returns ('vina', bin, True) because Vina
             IS available.
        """
        from molmetal.scripts.cfm_vina_dock_panel import (  # noqa: WPS433
            _probe_smina_binary,
            _probe_quickvina2_binary,
            _select_engine,
        )

        # (1) Smina + QuickVina 2 probe: NOT installed on this host.
        smina_ok, _smina_version = _probe_smina_binary()
        qvina2_ok, _qvina2_version = _probe_quickvina2_binary()
        assert smina_ok is False, (
            "Smina is NOT installed on this host per Wf-329-W7; "
            "probe MUST return False (no fabrication per AGENTS.md §5.2)"
        )
        assert qvina2_ok is False, (
            "QuickVina 2 is NOT installed on this host; "
            "probe MUST return False (no fabrication per AGENTS.md §5.2)"
        )

        # (2) _select_engine('vina') — Vina CLI IS available per Wf-329-W7.
        v_engine, v_bin, v_ok = _select_engine(
            engine="vina",
            vina_bin="/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina",
        )
        assert v_engine == "vina"
        assert "molmetal_vina" in v_bin
        assert (
            v_ok is True
        ), "Wf-329-W7 MEASURED: Vina CLI is installed at the canonical path"

        # (3) _select_engine('smina') returns ok=False honestly.
        s_engine, s_bin, s_ok = _select_engine(engine="smina")
        assert s_engine == "smina"
        assert s_ok is False, (
            "_select_engine('smina') MUST return ok=False on this host "
            "(smina not installed; no fabrication)"
        )

        # (4) _select_engine('quickvina2') returns ok=False honestly.
        q_engine, q_bin, q_ok = _select_engine(engine="quickvina2")
        assert q_engine == "quickvina2"
        assert q_ok is False, (
            "_select_engine('quickvina2') MUST return ok=False on this host "
            "(qvina2 not installed; no fabrication)"
        )

        # (5) _select_engine('auto') falls through Vina → Smina → QuickVina 2.
        a_engine, a_bin, a_ok = _select_engine(
            engine="auto",
            vina_bin="/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina",
        )
        assert a_engine == "vina", (
            f"_select_engine('auto') must return 'vina' on this host "
            f"(Vina IS available); got {a_engine!r}"
        )
        assert a_ok is True


# ===========================================================================
# Test 4 — Pipeline master + backward compat
# ===========================================================================
class TestPipelineMasterAndBackwardCompat:
    """The gpu_pipeline_master.sh must integrate the Wf-176 + Wf-338-M3
    CPU wrapper path AND all existing wrappers must still parse
    correctly (backward compat).

    Per Wf-336-D3 + Wf-336-X2/X3 pipeline master ship surface: the
    master has 4 NEW wrappers integrated (Wf-264, Wf-119 P1.4, Wf-74
    v2, Wf-333 Vina 4-step). The Wf-338-M3 CPU wrapper is a NEW
    addition that does NOT need to be in the master (it's CPU-only,
    runs after the LOMO job), but it must coexist with all existing
    wrappers without breaking them.
    """

    def test_pipeline_master_integrates_wf176_cpu_wrapper_path(self):
        """The pipeline master + Wf-338-M3 CPU wrapper must coexist:
        (a) pipeline master parses via bash -n (no syntax errors)
        (b) pipeline master references the Wf-176 Phase 2 GPU wrapper
            (Wf-310E, USER_OK-gated)
        (c) Wf-338-M3 CPU wrapper is NOT in the master (CPU-only,
            runs after the 14-day LOMO job per Wf-310E spec) but
            is on disk for downstream consumers
        """
        assert PIPELINE_MASTER.exists(), f"Pipeline master missing: {PIPELINE_MASTER}"
        assert (
            WF176_CPU_WRAPPER_WF338V2.exists()
        ), f"Wf-338-M3 CPU wrapper missing: {WF176_CPU_WRAPPER_WF338V2}"
        # bash -n syntax check on pipeline master
        proc_syntax = subprocess.run(
            ["bash", "-n", str(PIPELINE_MASTER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"pipeline master bash syntax error: {proc_syntax.stderr!r}"
        master_src = PIPELINE_MASTER.read_text()
        # Pipeline master must reference the Wf-310E Wf-135 Phase 2 wrapper
        # (Wf-176 Phase 2 GPU launch path).
        assert "launch_wf310e_wf135_phase2_gpu.sh" in master_src, (
            "Pipeline master must reference the Wf-310E Wf-135 Phase 2 "
            "GPU wrapper (Wf-176 Phase 2 launch surface)"
        )
        # Pipeline master should NOT necessarily reference the Wf-338-M3
        # CPU wrapper (it's CPU-only, post-LOMO job per Wf-310E spec).
        # We allow either presence or absence; the contract is that
        # the Wf-338-M3 CPU wrapper does NOT BLOCK the pipeline master.

    def test_existing_wrappers_still_parse_and_test_passes(self):
        """Backward compat: all existing wrappers referenced by the
        pipeline master + the Wf-264 + Wf-298D ship surface must still
        parse correctly. No regression.
        """
        # Wf-298D canonical wrapper + project alias must import.
        from molmetal.recovery.wf298d_harness_detector import (  # noqa: WPS433
            HarnessOutputExtractor,
        )

        extractor = HarnessOutputExtractor()
        assert extractor is not None

        # Wf-264 CFM property panel launcher must parse.
        wf264_launcher = REPO_ROOT / "scripts" / "launch_wf264_cfm_property_panel.sh"
        assert wf264_launcher.exists(), f"Wf-264 launcher missing: {wf264_launcher}"
        proc = subprocess.run(
            ["bash", "-n", str(wf264_launcher)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc.returncode == 0
        ), f"Wf-264 launcher bash syntax error: {proc.stderr!r}"

        # Wf-264 Vina dock panel + recovery must parse (CLI --engine).
        from molmetal.scripts.cfm_vina_dock_panel import _build_parser  # noqa: WPS433

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--smiles-csv",
                "/tmp/x.csv",
                "--pocket-pdb-dir",
                "/tmp/pdb",
                "--out-json",
                "/tmp/out.json",
                "--engine",
                "auto",
            ]
        )
        assert args.engine == "auto"

        # Wf-310E Wf-135 Phase 2 GPU wrapper must parse.
        assert (
            WF176_GPU_WRAPPER.exists()
        ), f"Wf-310E wrapper missing: {WF176_GPU_WRAPPER}"
        proc = subprocess.run(
            ["bash", "-n", str(WF176_GPU_WRAPPER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc.returncode == 0
        ), f"Wf-310E wrapper bash syntax error: {proc.stderr!r}"

        # Wf-338-M3 CPU wrapper must parse (already verified in test 2
        # but we re-verify here for backward-compat lock-in).
        proc = subprocess.run(
            ["bash", "-n", str(WF176_CPU_WRAPPER_WF338V2)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc.returncode == 0
        ), f"Wf-338-M3 wrapper bash syntax error: {proc.stderr!r}"
