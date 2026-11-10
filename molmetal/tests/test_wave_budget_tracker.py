"""Wf-262 / TODO-208 — unit tests for ``wave_budget_tracker.py``.

Covers the §7.1 + §C contract documented in
``TODO/completed/207_unified_pre_flight_checklist_v2.md``:

  1. CLI: argparse recognises the contract flags
     (``--budget-hr``, ``--waves``, ``--report-json`` + extras).
  2. ``--waves`` parsing: handles well-formed input and rejects
     malformed tokens (usage error).
  3. Cumulative computation: when sum of estimated_hr is within the
     budget, every wave is ``PLANNED`` (or ``COMPLETE`` if actual_hr is
     loaded).
  4. Over-budget: when sum of estimated_hr exceeds the budget, later
     waves are flagged ``DEFERRED``; the boundary wave (whose start fits
     but whose end overshoots) is flagged ``RUNNING``.
  5. Actual wall-time mining: a synthetic log with ``WALL_SEC=...`` is
     picked up from the configured reports-dir glob; an absent wave
     has ``actual_hr = None``.
  6. JSON schema: the emitted report has all contract fields
     (``budget_hr``, ``cumulative_wave_hr``, ``remaining_hr``, ``waves``,
     ``deferred_waves``, ``honest_framing``, ``wave_source``,
     ``auto_discovered``, ``n_configs_scanned``, ``n_actual_hr_loaded``,
     ``schema_version``, ``ts``).
  7. Auto-discovery: when ``--waves`` is omitted, the wave list is
     derived from ``configs/*.yaml`` + canonical §0.3 baseline.
  8. Subprocess smoke via the contract ``.venv/bin/python``.

All tests are CPU-only (no torch, no GPU); they construct ephemeral
reports/log dirs under ``tmp_path`` so the real ``molmetal/reports/``
tree is untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``wave_budget_tracker`` importable as a module.
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import wave_budget_tracker  # noqa: E402

VENV_PY = sys.executable

CONTRACT_FIELDS = (
    "schema_version",
    "ts",
    "budget_hr",
    "cumulative_wave_hr",
    "remaining_hr",
    "waves",
    "deferred_waves",
    "honest_framing",
    "wave_source",
    "auto_discovered",
    "n_configs_scanned",
    "n_actual_hr_loaded",
    "reports_globs",
)


# ---------------------------------------------------------------------------
# Test 1 — CLI exposes the contract flags (G2 ≥ 3)
# ---------------------------------------------------------------------------


def test_cli_exposes_contract_flags():
    """The 3 contract flags (--budget-hr, --waves, --report-json) must
    all be present in the argparse parser."""
    ns = wave_budget_tracker.parse_argv(
        [
            "--budget-hr",
            "8.5",
            "--waves",
            "Wave-A:2.0,Wave-B:3.0",
            "--report-json",
            "/tmp/foo.json",
        ]
    )
    assert ns.budget_hr == 8.5
    assert ns.waves == "Wave-A:2.0,Wave-B:3.0"
    assert str(ns.report_json) == "/tmp/foo.json"
    # G2 target: at least 3 flags exposed (extras like --auto-discover
    # + --no-auto-discover count as 2 distinct flags).
    extras_visible = (
        "--reports-dir",
        "--auto-discover",
        "--no-auto-discover",
        "--wave-glob",
        "--no-default-glob",
        "--verbose",
    )
    assert all(e.startswith("--") for e in extras_visible)


# ---------------------------------------------------------------------------
# Test 2 — Backward-compat: --report-json default points to wf262 path
# ---------------------------------------------------------------------------


def test_report_json_default_is_wf262_path():
    """Per 207 v2 §7.1, --report-json default must be
    ``molmetal/reports/wf262_wave_budget/wave_budget_report.json``."""
    ns = wave_budget_tracker.parse_argv(["--budget-hr", "10", "--waves", "Wave-1:1.0"])
    assert ns.report_json == wave_budget_tracker.DEFAULT_REPORT_JSON
    expected = (
        wave_budget_tracker.REPORTS_DIR
        / "wf262_wave_budget"
        / "wave_budget_report.json"
    )
    assert ns.report_json == expected


# ---------------------------------------------------------------------------
# Test 3 — --waves parsing (well-formed + malformed)
# ---------------------------------------------------------------------------


def test_parse_waves_arg_well_formed():
    out = wave_budget_tracker.parse_waves_arg("Wave-1:4.6,Wave-2:5.0,Wave-3:1.0")
    assert len(out) == 3
    assert out[0][0] == "Wave-1"
    assert out[0][1] == 4.6
    assert out[1][0] == "Wave-2"
    assert out[1][1] == 5.0
    assert out[2][0] == "Wave-3"
    assert out[2][1] == 1.0
    # Every wave carries the CLI provenance tag.
    assert all(src == "from --waves CLI" for _n, _h, src in out)


def test_parse_waves_arg_malformed_raises():
    with pytest.raises(ValueError, match="invalid --waves token"):
        wave_budget_tracker.parse_waves_arg("Wave-1:4.6,not-a-kv,Wave-3:1.0")
    with pytest.raises(ValueError, match="hr must be non-negative"):
        wave_budget_tracker.parse_waves_arg("Wave-1:-1.0")


# ---------------------------------------------------------------------------
# Test 4 — Cumulative within budget (no DEFERRED)
# ---------------------------------------------------------------------------


def test_cumulative_within_budget(tmp_path: Path):
    """When the sum of estimated_hr is within the budget, every wave
    is ``PLANNED`` (or ``COMPLETE`` if actual_hr loaded) and the
    report has no DEFERRED waves."""
    report_path = tmp_path / "report_within.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-1:3.0,Wave-2:2.0,Wave-3:1.0",
            "--report-json",
            str(report_path),
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["budget_hr"] == 10.0
    assert report["cumulative_wave_hr"] == 6.0
    assert report["remaining_hr"] == 4.0
    assert report["deferred_waves"] == []
    statuses = [w["status"] for w in report["waves"]]
    assert statuses == ["PLANNED", "PLANNED", "PLANNED"]
    # Honest framing acknowledges the buffer.
    assert "buffer" in report["honest_framing"]


# ---------------------------------------------------------------------------
# Test 5 — Cumulative over budget marks DEFERRED (boundary wave = RUNNING)
# ---------------------------------------------------------------------------


def test_cumulative_over_budget_marks_deferred(tmp_path: Path):
    """When the sum of estimated_hr exceeds the budget, later waves
    are flagged DEFERRED.  The boundary wave (start fits, end
    overshoots) is flagged RUNNING."""
    report_path = tmp_path / "report_over.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-1:5.0,Wave-2:4.0,Wave-3:3.0,Wave-4:2.0",
            "--report-json",
            str(report_path),
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Cumulative: 5 + 4 + 3 + 2 = 14; remaining = -4
    assert report["cumulative_wave_hr"] == 14.0
    assert report["remaining_hr"] == -4.0
    # Statuses: Wave-1 PLANNED (5 fits), Wave-2 RUNNING (5+4=9 fits end),
    #           Wave-3 RUNNING (5+4+3=12 > 10 but starts at 9 ≤ 10),
    #           Wave-4 DEFERRED (cumulative 12 > 10 already)
    statuses = [w["status"] for w in report["waves"]]
    assert statuses[0] == "PLANNED"
    assert statuses[1] == "PLANNED"  # cumulative 9 ≤ 10
    assert statuses[2] == "RUNNING"  # cumulative 9 ≤ 10 < 12
    assert statuses[3] == "DEFERRED"
    # Deferred list contains Wave-4.
    assert "Wave-4" in report["deferred_waves"]
    # Honest framing mentions "exceeded by" + "DEFERRED".
    assert "exceeded" in report["honest_framing"]
    assert "DEFERRED" in report["honest_framing"]


# ---------------------------------------------------------------------------
# Test 6 — Actual wall-time loaded from log if present
# ---------------------------------------------------------------------------


def test_actual_hr_loaded_from_log_if_present(tmp_path: Path):
    """When a per-wave ``log`` file with ``WALL_SEC=...`` exists in the
    configured reports-dir glob, the wave's ``actual_hr`` is loaded
    and its status is COMPLETE."""
    # Construct: reports_dir/wf_wave_test/wf_wave_test/log with WALL_SEC=3600 (1h)
    fake_reports = tmp_path / "reports"
    wave_log_dir = fake_reports / "wf_wave_test" / "wf_wave_test"
    wave_log_dir.mkdir(parents=True)
    log_path = wave_log_dir / "log"
    log_path.write_text("WALL_SEC=3600\n")

    # Wave name "wf_wave_test" so the substring matcher picks up
    # the on-disk wf_wave_test directory.  Glob uses ** to descend
    # into sub-directories (Python ``glob`` only honours ``**`` when
    # ``recursive=True`` is passed; see the mine_actual_wall_hr docstring).
    report_path = tmp_path / "report_actual.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "5.0",
            "--waves",
            "wf_wave_test:2.0,Wave-Y:1.0",
            "--report-json",
            str(report_path),
            "--reports-dir",
            str(fake_reports),
            "--no-default-glob",
            "--wave-glob",
            str(fake_reports / "wf*" / "**" / "log"),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    # Wave-X should pick up actual_hr=1.0h; Wave-Y has no log → None.
    wave_x = report["waves"][0]
    wave_y = report["waves"][1]
    assert wave_x["actual_hr"] is not None
    assert wave_x["actual_hr"] == pytest.approx(1.0, abs=1e-3)
    assert wave_x["status"] == "COMPLETE"
    assert wave_x["actual_source"] is not None
    assert "wf_wave_test" in wave_x["actual_source"]
    # Wave-Y: no log → actual_hr=None, status=PLANNED.
    assert wave_y["actual_hr"] is None
    assert wave_y["status"] == "PLANNED"
    assert wave_y["actual_source"] is None
    # n_actual_hr_loaded reflects the single successful load.
    assert report["n_actual_hr_loaded"] >= 1


def test_actual_hr_none_when_log_absent(tmp_path: Path):
    """When no log exists for a wave, ``actual_hr`` stays None and
    status is PLANNED (when within budget)."""
    report_path = tmp_path / "report_no_log.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-A:1.0",
            "--report-json",
            str(report_path),
            "--reports-dir",
            str(tmp_path / "empty_reports"),
            "--no-default-glob",
            "--wave-glob",
            str(tmp_path / "empty_reports" / "wf*" / "**" / "log"),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["waves"][0]["actual_hr"] is None
    assert report["waves"][0]["status"] == "PLANNED"
    assert report["waves"][0]["actual_source"] is None


# ---------------------------------------------------------------------------
# Test 7 — JSON schema: all contract fields present
# ---------------------------------------------------------------------------


def test_report_json_schema_complete(tmp_path: Path):
    """The emitted JSON must contain all 13 contract fields."""
    report_path = tmp_path / "schema.json"
    wave_budget_tracker.main(
        [
            "--budget-hr",
            "8.0",
            "--waves",
            "Wave-1:2.0,Wave-2:3.0",
            "--report-json",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text())
    for k in CONTRACT_FIELDS:
        assert k in report, f"contract field '{k}' missing from report"
    # Per-wave schema: each wave must carry {wave_name, estimated_hr,
    # actual_hr, status, actual_source, source}.
    for w in report["waves"]:
        assert "wave_name" in w
        assert "estimated_hr" in w
        assert "actual_hr" in w
        assert "status" in w
        assert "actual_source" in w
        assert "source" in w
        assert w["status"] in {"PLANNED", "RUNNING", "COMPLETE", "DEFERRED"}
    # Type spot-checks.
    assert isinstance(report["budget_hr"], float)
    assert isinstance(report["cumulative_wave_hr"], float)
    assert isinstance(report["remaining_hr"], float)
    assert isinstance(report["deferred_waves"], list)
    assert isinstance(report["auto_discovered"], bool)
    assert isinstance(report["n_configs_scanned"], int)
    assert isinstance(report["n_actual_hr_loaded"], int)
    assert report["schema_version"] == "262.v1"
    # wave_source is the CLI path (not auto-derived).
    assert report["wave_source"] == "from --waves CLI"
    assert report["auto_discovered"] is False


# ---------------------------------------------------------------------------
# Test 8 — Auto-discovery from configs/*.yaml
# ---------------------------------------------------------------------------


def test_auto_discover_when_waves_omitted(tmp_path: Path):
    """When --waves is omitted, the wave list is auto-derived from
    ``configs/*.yaml`` + canonical §0.3 baseline.  Wave source is
    flagged accordingly."""
    report_path = tmp_path / "auto_report.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "100.0",  # large budget so nothing defers
            "--report-json",
            str(report_path),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["auto_discovered"] is True
    assert "auto-derived" in report["wave_source"]
    # The auto-discovered list MUST contain at least one wave from
    # DEFAULT_AUTO_WAVES (the canonical §0.3 baseline).
    wave_names = [w["wave_name"] for w in report["waves"]]
    canonical_first = wave_budget_tracker.DEFAULT_AUTO_WAVES[0][0]
    assert canonical_first in wave_names
    # n_configs_scanned is non-negative int.
    assert isinstance(report["n_configs_scanned"], int)
    assert report["n_configs_scanned"] >= 0


def test_no_auto_discover_without_waves_errors(tmp_path: Path):
    """When --no-auto-discover is set and --waves is omitted, the
    script exits with usage error (rc=1)."""
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--no-auto-discover",
            "--report-json",
            str(tmp_path / "should_not_exist.json"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "should_not_exist.json").exists()


# ---------------------------------------------------------------------------
# Test 9 — compute_wave_status pure-function contract
# ---------------------------------------------------------------------------


def test_compute_wave_status_actual_takes_precedence():
    """When ``actual_hr`` is loaded, the wave is COMPLETE and the
    cumulative counter advances by the actual (not the estimate)."""
    waves = [
        ("Wave-A", 5.0, "test"),
        ("Wave-B", 5.0, "test"),
    ]
    actual = {"Wave-A": (4.0, "/tmp/log_a"), "Wave-B": (None, None)}
    out = wave_budget_tracker.compute_wave_status(10.0, waves, actual)
    assert out[0]["status"] == "COMPLETE"
    assert out[0]["actual_hr"] == 4.0
    assert out[1]["status"] == "PLANNED"  # cumulative 4+5=9 fits
    assert out[1]["actual_hr"] is None


def test_compute_wave_status_running_boundary():
    """Boundary wave (cumulative ≤ budget < cumulative + estimated) is
    flagged RUNNING, not DEFERRED."""
    waves = [
        ("Wave-A", 6.0, "test"),
        ("Wave-B", 6.0, "test"),
    ]
    out = wave_budget_tracker.compute_wave_status(10.0, waves, {})
    assert out[0]["status"] == "PLANNED"  # cumulative 6 ≤ 10
    # Wave-B: cumulative=6 ≤ 10 < 12 → RUNNING
    assert out[1]["status"] == "RUNNING"


# ---------------------------------------------------------------------------
# Test 10 — Subprocess smoke via the project-venv interpreter
# ---------------------------------------------------------------------------


def test_subprocess_via_venv(tmp_path: Path):
    """The contract interpreter ``/home/hugo/codes/mol-metal/.venv/bin/python``
    must be able to run ``--help`` and a typical CLI invocation."""
    if not Path(VENV_PY).exists():
        pytest.skip(f"project-venv interpreter not present at {VENV_PY}")
    # --help smoke (exit 0).
    rc = subprocess.run(
        [VENV_PY, str(wave_budget_tracker.__file__), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    ).returncode
    assert rc == 0, f"--help failed: {rc}"
    # End-to-end smoke: emit a JSON report with all flags exercised.
    report_path = tmp_path / "subproc_report.json"
    rc2 = subprocess.run(
        [
            VENV_PY,
            str(wave_budget_tracker.__file__),
            "--budget-hr",
            "12.0",
            "--waves",
            "Wave-1:4.0,Wave-2:3.0,Wave-3:5.0",
            "--report-json",
            str(report_path),
            "--verbose",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).returncode
    assert rc2 == 0, f"subprocess run failed (rc={rc2}): {rc2}"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["budget_hr"] == 12.0
    # Cumulative = 4+3+5 = 12 → exactly consumed.
    assert report["cumulative_wave_hr"] == 12.0
    assert report["remaining_hr"] == 0.0
    # "exactly consumed" framing.
    assert "exactly consumed" in report["honest_framing"]


# ---------------------------------------------------------------------------
# Test 11 — _parse_log_for_wall_hr helpers
# ---------------------------------------------------------------------------


def test_parse_log_wall_sec(tmp_path: Path):
    log = tmp_path / "log"
    log.write_text("WALL_SEC=7200\n")
    hr = wave_budget_tracker._parse_log_for_wall_hr(log)
    assert hr == pytest.approx(2.0, abs=1e-6)


def test_parse_log_json_key(tmp_path: Path):
    log = tmp_path / "log"
    log.write_text('{"wall_sec": 1800, "epoch": 49}\n')
    hr = wave_budget_tracker._parse_log_for_wall_hr(log)
    assert hr == pytest.approx(0.5, abs=1e-6)


def test_parse_log_hms(tmp_path: Path):
    log = tmp_path / "log"
    log.write_text("total time: 01:30:00\n")
    hr = wave_budget_tracker._parse_log_for_wall_hr(log)
    assert hr == pytest.approx(1.5, abs=1e-6)


def test_parse_log_returns_none_for_garbage(tmp_path: Path):
    log = tmp_path / "log"
    log.write_text("no timing info here\n")
    hr = wave_budget_tracker._parse_log_for_wall_hr(log)
    assert hr is None


def test_parse_log_returns_none_for_missing(tmp_path: Path):
    hr = wave_budget_tracker._parse_log_for_wall_hr(tmp_path / "nope.log")
    assert hr is None


def test_no_enforce_budget_skips_deferred_label():
    """When the user opts out of the budget cap, no wave is DEFERRED.

    Per Wf-262 §0.5 honest-gap fix: user 2026-09-28 said "时间预算没关系
    的" — the 10h budget is a tracker, not a hard blocker.  When the
    user passes ``--no-enforce-budget``, all waves stay PLANNED even if
    their cumulative projected wall-time exceeds the budget; the honest
    framing still reports the over-budget figure.
    """
    waves = [
        ("Wave-A", 1.8, None),
        ("Wave-B", 5.0, None),
        ("Wave-C", 7.5, None),  # cumulative crosses 10h here
        ("Wave-D", 1.0, None),
    ]
    out = wave_budget_tracker.compute_wave_status(
        budget_hr=10.0,
        waves=waves,
        actual_hr_by_name={},
        enforce_budget=False,
    )
    assert [d["status"] for d in out] == ["PLANNED"] * 4
    assert all(d["status"] != "DEFERRED" for d in out)


# ---------------------------------------------------------------------------
# Test 12 — Wf-268C: TODO-180 G4 MEASURED actual_hr by metal
#              (per 207 v2 §7.1.2)
# ---------------------------------------------------------------------------


def test_todo_180_g4_actuals_table_constant():
    """The hardcoded TODO-180 G4 actuals table must carry all 5 metal
    entries with the Wf-268C-specified MEASURED hours (Ru 28.9, Ir 7.7,
    Os 1.9, Rh 1.7, Re 1.6).  Per 207 v2 §7.1.2."""
    expected = {
        "Ru": 28.9,
        "Ir": 7.7,
        "Os": 1.9,
        "Rh": 1.7,
        "Re": 1.6,
    }
    actual = wave_budget_tracker.TODO_180_G4_ACTUAL_HR_BY_METAL
    assert set(actual.keys()) == set(expected.keys())
    for metal, hr in expected.items():
        assert actual[metal] == pytest.approx(hr, abs=1e-3), (
            f"TODO_180_G4_ACTUAL_HR_BY_METAL[{metal}] = {actual[metal]} "
            f"!= expected {hr}"
        )


def test_mine_actual_wall_hr_todo_180_g4_per_metal():
    """When a wave name carries a per-metal token (Ru/Ir/Os/Rh/Re),
    ``mine_actual_wall_hr`` MUST return the MEASURED TODO-180 G4 actual
    before scanning the file system (per 207 v2 §7.1.2)."""
    for metal, hr in wave_budget_tracker.TODO_180_G4_ACTUAL_HR_BY_METAL.items():
        actual_hr, actual_src = wave_budget_tracker.mine_actual_wall_hr(
            f"Wave-X:{metal}_seed0_30ep", globs=[]
        )
        assert actual_hr == pytest.approx(hr, abs=1e-3)
        assert actual_src is not None
        assert metal in actual_src
        assert "TODO_180_G4_ACTUAL_HR_BY_METAL" in actual_src
        assert "207 v2 §7.1.2" in actual_src


def test_mine_actual_wall_hr_todo_180_g4_aggregate():
    """When a wave name matches the multi-metal G4 cohort
    (canonical ``Wave-3:chemprop_v3_G4_14_pairs``), the lookup returns
    the SUM of the per-metal actuals (Ru+Ir+Os+Rh+Re = 41.8 hr)."""
    actual_hr, actual_src = wave_budget_tracker.mine_actual_wall_hr(
        "Wave-3:chemprop_v3_G4_14_pairs", globs=[]
    )
    expected_total = sum(wave_budget_tracker.TODO_180_G4_ACTUAL_HR_BY_METAL.values())
    assert actual_hr == pytest.approx(expected_total, abs=1e-3)
    assert actual_src is not None
    assert "sum" in actual_src.lower()
    assert "207 v2 §7.1.2" in actual_src


def test_todo_180_g4_actuals_referenced_in_docstring():
    """The script docstring must reference 207 v2 §7.1.2 for the
    TODO-180 G4 actuals provenance."""
    src = wave_budget_tracker.__doc__ or ""
    assert "§7.1.2" in src
    assert "TODO-180 G4" in src
    assert "MEASURED" in src


def test_default_reports_glob_prefers_wf_log():
    """Wf-268C: the default glob must prefer ``molmetal/reports/wf*/**/log``
    (logs only) over the previous broad ``wf*`` match.  Backward compat:
    callers using ``--wave-glob`` are unaffected."""
    g = wave_budget_tracker.DEFAULT_REPORTS_GLOB
    assert "wf*" in g
    assert g.endswith("/log") or g.endswith(
        str(Path("**") / "log")
    ), f"DEFAULT_REPORTS_GLOB should end with /log path, got: {g}"
    assert "**" in g  # recursive glob to descend into wf* sub-dirs


def test_wave_3_g4_marks_complete_with_measured_actual(tmp_path: Path):
    """End-to-end: when running the canonical Wave-3 chemprop_v3_G4_14_pairs
    wave, the MEASURED TODO-180 G4 actuals (41.8 hr aggregate) must be
    picked up and status flagged COMPLETE.  We invoke ``compute_wave_status``
    directly (the auto-discovered canonical wave name carries colons which
    the ``--waves`` CLI parser rejects; see ``parse_waves_arg`` contract)."""
    waves = [
        ("Wave-3:chemprop_v3_G4_14_pairs", 1.8, "TODO-180 G4 canonical"),
    ]
    actual_hr, actual_src = wave_budget_tracker.mine_actual_wall_hr(
        "Wave-3:chemprop_v3_G4_14_pairs", globs=[]
    )
    assert actual_hr == pytest.approx(41.8, abs=1e-3)
    assert "TODO_180_G4_ACTUAL_HR_BY_METAL" in actual_src
    actual_hr_by_name = {"Wave-3:chemprop_v3_G4_14_pairs": (actual_hr, actual_src)}
    out = wave_budget_tracker.compute_wave_status(
        budget_hr=50.0,
        waves=waves,
        actual_hr_by_name=actual_hr_by_name,
    )
    assert out[0]["status"] == "COMPLETE"
    assert out[0]["actual_hr"] == pytest.approx(41.8, abs=1e-3)


def test_g4_per_metal_end_to_end_via_cli(tmp_path: Path):
    """End-to-end CLI smoke: when a --waves wave name carries the metal
    token (e.g. ``Wf268c_Ru:50.0``), the MEASURED TODO-180 G4 actual for
    Ru (28.9 hr) is loaded and the wave is marked COMPLETE.  Note: the
    CLI format is ``name:NUMERIC_HR`` per the ``WAVE_KV_RE`` contract;
    the wave name carries the metal token (e.g. ``Ru``) which the
    word-boundary matcher recognises."""
    report_path = tmp_path / "g4_cli.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "50.0",
            "--waves",
            "Wf268c_Ru:50.0",
            "--report-json",
            str(report_path),
            "--no-default-glob",
            "--wave-glob",
            str(tmp_path / "empty" / "**" / "log"),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    wave = report["waves"][0]
    assert wave["actual_hr"] == pytest.approx(28.9, abs=1e-3)
    assert wave["status"] == "COMPLETE"
    assert "TODO_180_G4_ACTUAL_HR_BY_METAL[Ru]" in wave["actual_source"]


def test_no_enforce_budget_honest_framing_acknowledges_opt_out():
    """Summarise() emits the user-opt-out string when over budget."""
    wave_dicts = [
        {
            "wave_name": "W1",
            "estimated_hr": 6.0,
            "actual_hr": None,
            "status": "PLANNED",
            "actual_source": None,
            "source": None,
        },
        {
            "wave_name": "W2",
            "estimated_hr": 8.0,
            "actual_hr": None,
            "status": "PLANNED",
            "actual_source": None,
            "source": None,
        },
    ]
    summary = wave_budget_tracker.summarise(
        budget_hr=10.0,
        wave_dicts=wave_dicts,
        enforce_budget=False,
    )
    assert summary["deferred_waves"] == []
    assert summary["remaining_hr"] < 0
    assert "--no-enforce-budget" in summary["honest_framing"]
    assert "user opted out" in summary["honest_framing"]


# ---------------------------------------------------------------------------
# Test 13 — Wf-285C: --no-enforce-budget CLI flag round-trip
# ---------------------------------------------------------------------------


def test_no_enforce_budget_flag(tmp_path: Path):
    """Wf-285C: ``--no-enforce-budget`` CLI flag must suppress DEFERRED
    labeling when the cumulative projected wall-time exceeds the
    budget.  When the flag is set, all over-budget waves stay PLANNED
    (the user's opt-out is respected)."""
    report_path = tmp_path / "no_enforce_report.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-1:5.0,Wave-2:4.0,Wave-3:3.0,Wave-4:2.0",
            "--report-json",
            str(report_path),
            "--no-enforce-budget",
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Cumulative: 5 + 4 + 3 + 2 = 14; remaining = -4 (over budget).
    assert report["cumulative_wave_hr"] == 14.0
    assert report["remaining_hr"] == -4.0
    # NO wave is DEFERRED — every one stays PLANNED.
    statuses = [w["status"] for w in report["waves"]]
    assert statuses == ["PLANNED", "PLANNED", "PLANNED", "PLANNED"]
    assert report["deferred_waves"] == []
    # Honest framing acknowledges the user opt-out + over-budget figure.
    assert "--no-enforce-budget" in report["honest_framing"]
    assert "user opted out" in report["honest_framing"]
    assert "exceeded" in report["honest_framing"]


def test_budget_enforced_when_flag_off(tmp_path: Path):
    """Wf-285C: without ``--no-enforce-budget`` (default = enforced),
    over-budget waves ARE flagged DEFERRED (RUNNING for boundary)."""
    report_path = tmp_path / "enforce_report.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-1:5.0,Wave-2:4.0,Wave-3:3.0,Wave-4:2.0",
            "--report-json",
            str(report_path),
            # Note: NO --no-enforce-budget here → enforcement ON (default).
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Cumulative: 14; remaining = -4 (over budget).
    assert report["cumulative_wave_hr"] == 14.0
    assert report["remaining_hr"] == -4.0
    # DEFERRED labeling IS in effect: Wave-4 must be DEFERRED.
    statuses = [w["status"] for w in report["waves"]]
    assert "DEFERRED" in statuses
    assert "Wave-4" in report["deferred_waves"]
    # Honest framing mentions "exceeded by" + "DEFERRED" (not opt-out).
    assert "exceeded" in report["honest_framing"]
    assert "DEFERRED" in report["honest_framing"]
    assert "--no-enforce-budget" not in report["honest_framing"]
    assert "user opted out" not in report["honest_framing"]


def test_no_enforce_budget_default_false():
    """Wf-285C: the ``--no-enforce-budget`` flag defaults to ``False`` —
    i.e. budget enforcement is ON unless the user explicitly opts out."""
    ns = wave_budget_tracker.parse_argv(
        ["--budget-hr", "10.0", "--waves", "Wave-A:2.0"]
    )
    assert ns.no_enforce_budget is False
    # And passing the flag explicitly sets it to True.
    ns2 = wave_budget_tracker.parse_argv(
        ["--budget-hr", "10.0", "--waves", "Wave-A:2.0", "--no-enforce-budget"]
    )
    assert ns2.no_enforce_budget is True


# ---------------------------------------------------------------------------
# Test 14 — Wf-272I: --cfm-yuelbond-required CLI flag round-trip
#              (LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF YuelBond-conditionality
#              enforcement gate; per Wf-269C real-cohort MEASURED
#              --no-yuelbond → decode_ratio 0.000, Δ=-1.000 vs full=1.000)
# ---------------------------------------------------------------------------


def test_cfm_yuelbond_required_flag():
    """Wf-272I: ``--cfm-yuelbond-required`` CLI flag round-trips via
    argparse.  When set, ``args.cfm_yuelbond_required`` is True; default
    is False (backward compat)."""
    ns_default = wave_budget_tracker.parse_argv(
        ["--budget-hr", "10.0", "--waves", "Wave-A:2.0"]
    )
    assert ns_default.cfm_yuelbond_required is False
    ns_set = wave_budget_tracker.parse_argv(
        [
            "--budget-hr",
            "10.0",
            "--waves",
            "Wave-A:2.0",
            "--cfm-yuelbond-required",
        ]
    )
    assert ns_set.cfm_yuelbond_required is True


def test_cfm_yuelbond_required_enforces(tmp_path: Path):
    """Wf-272I: when ``--cfm-yuelbond-required`` is set, a CFM wave whose
    name does NOT carry ``yuelbond`` is forcibly DEFERRED, regardless of
    budget fit.  Per LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF Wf-269C
    MEASURED ``--no-yuelbond`` → decode_ratio 0.000 on real cohort."""
    report_path = tmp_path / "cfm_yb_required_report.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "100.0",  # large budget so non-CFM waves would all PLANNED
            "--waves",
            "chemprop_v2_smoke:1.0,cfm_full_retrain:2.0,cfm_per_innovation_ablation:3.0,chemprop_v3_lomo:1.0",
            "--report-json",
            str(report_path),
            "--cfm-yuelbond-required",
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Build a name→status map for easy lookup.
    statuses = {w["wave_name"]: w["status"] for w in report["waves"]}
    # chemprop waves (no "cfm" in name) stay PLANNED.
    assert statuses["chemprop_v2_smoke"] == "PLANNED"
    assert statuses["chemprop_v3_lomo"] == "PLANNED"
    # CFM waves WITHOUT yuelbond in name are forcibly DEFERRED.
    assert statuses["cfm_full_retrain"] == "DEFERRED"
    assert statuses["cfm_per_innovation_ablation"] == "DEFERRED"
    # Deferred list captures the CFM-without-YuelBond waves.
    deferred = report["deferred_waves"]
    assert "cfm_full_retrain" in deferred
    assert "cfm_per_innovation_ablation" in deferred


def test_cfm_yuelbond_not_required_default(tmp_path: Path):
    """Wf-272I: when ``--cfm-yuelbond-required`` is NOT set (default off,
    backward compat), CFM waves WITHOUT ``yuelbond`` in the name are
    NOT forcibly DEFERRED by the YuelBond gate — they are subject only
    to the normal budget enforcement.  With a large enough budget they
    stay PLANNED."""
    report_path = tmp_path / "default_no_yb_gate.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "100.0",  # large budget so nothing defers on budget
            "--waves",
            "cfm_full_retrain:2.0,cfm_per_innovation_ablation:3.0",
            "--report-json",
            str(report_path),
            # No --cfm-yuelbond-required flag.
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    statuses = [w["status"] for w in report["waves"]]
    # Both waves PLANNED (large budget + default YuelBond gate off).
    assert statuses == ["PLANNED", "PLANNED"]
    assert report["deferred_waves"] == []
    # Sanity-check the helper directly (the heuristic function).
    assert (
        wave_budget_tracker.is_cfm_wave_missing_yuelbond("Wave-cfm_full_retrain")
        is True
    )


def test_cfm_yuelbond_required_with_yuelbond_present(tmp_path: Path):
    """Wf-272I: when ``--cfm-yuelbond-required`` is set, a CFM wave that
    DOES carry ``yuelbond`` in its name is NOT deferred by the gate — it
    stays PLANNED (assuming normal budget fit).  The gate only fires on
    CFM waves MISSING ``yuelbond``."""
    report_path = tmp_path / "cfm_with_yb.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr",
            "100.0",
            "--waves",
            "cfm_full_retrain_yuelbond:2.0,cfm_YuelBond_wasserstein:3.0,chemprop_v2_yuelbond_probe:1.0",
            "--report-json",
            str(report_path),
            "--cfm-yuelbond-required",
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    statuses = {w["wave_name"]: w["status"] for w in report["waves"]}
    # All three waves carry "yuelbond" — the gate must NOT defer them.
    assert statuses["cfm_full_retrain_yuelbond"] == "PLANNED"
    assert statuses["cfm_YuelBond_wasserstein"] == "PLANNED"
    # The third wave has "yuelbond" but no "cfm" — gate never fires; PLANNED.
    assert statuses["chemprop_v2_yuelbond_probe"] == "PLANNED"
    assert report["deferred_waves"] == []
    # Sanity-check helper directly (positive case: present).
    assert (
        wave_budget_tracker.is_cfm_wave_missing_yuelbond("cfm_full_retrain_yuelbond")
        is False
    )
    # Non-CFM wave (no "cfm" in name) is never gated — helper returns False.
    assert (
        wave_budget_tracker.is_cfm_wave_missing_yuelbond("chemprop_v2_smoke") is False
    )
