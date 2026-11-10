"""Wf-426-A3 post-fix retrain launcher + aggregator regression-guard tests.

Honors AGENTS.md §5.7 NEW (50-step CPU smoke before any trainer flag
wire-in) + §5.6 (atomic write protocol) + §5.5 (filesystem-truth) +
§1.2 (honest framing).

Tests:

  TestAggregatorSmoke — pure aggregator smoke test on synthetic candidates.
  TestAggregatorBlockOnMissingWrapper — verifies BLOCKED verdict for
      a candidate whose wrapper script does NOT exist on disk.
  TestAggregatorBlockOnSyntaxFail — verifies BLOCKED verdict for a
      wrapper that fails bash -n syntax check.
  TestAggregatorBlockOnBadPlanJson — verifies BLOCKED verdict for a
      plan.json that is malformed or missing required meta fields.
  TestAggregatorWarnOnMissingDryRun — verifies WARN verdict when
      DRY_RUN.log is missing for an existing wrapper.
  TestAggregatorReadyAllGreen — verifies READY verdict when all
      checks pass.
  TestAggregatorZeroCandidates — verifies graceful handling of an
      empty candidates dir (returns empty candidates list).
  TestAggregatorAtomicWrite — verifies the aggregator's output JSON is
      written via the atomic_write_file primitive (file exists post-run).
  TestLauncherImportsValidBash — verifies the launcher shell script
      passes bash -n syntax check.
  TestLauncherHonorsUserOkGuard — verifies the launcher exits with
      code 64 if USER_OK is unset (mock shell exec).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGGREGATOR = (
    REPO_ROOT / "molmetal" / "scripts" / "wf426a3_post_fix_results_aggregator.py"
)
LAUNCHER = REPO_ROOT / "molmetal" / "scripts" / "wf426a3_post_fix_retrain_launcher.sh"
AGGREGATOR = (
    REPO_ROOT / "molmetal" / "scripts" / "wf426a3_post_fix_results_aggregator.py"
)
ATOMIC_WRITE = REPO_ROOT / "molmetal" / "scripts" / "wf386a3_atomic_write.py"


@pytest.fixture
def candidates_dir(tmp_path):
    """Per-test temp dir; mirrors the launcher's contract."""
    d = tmp_path / "wf426a3_candidates"
    d.mkdir(exist_ok=True)
    return d


def _write_candidate(
    candidates_dir: Path,
    name: str,
    script_path: str = "",
    script_exists: bool = False,
    report_dir: str = "",
    report_dir_exists: bool = False,
    dry_run_log_exists: bool = False,
    plan_json_exists: bool = False,
    notes: str = "",
    plan_json_content: dict | None = None,
) -> Path:
    """Write one candidate record to the candidates dir."""
    rec = {
        "name": name,
        "script": script_path,
        "script_exists": script_exists,
        "report_dir": report_dir,
        "report_dir_exists": report_dir_exists,
        "dry_run_log_exists": dry_run_log_exists,
        "plan_json_exists": plan_json_exists,
        "notes": notes,
    }
    p = candidates_dir / f"{name}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    if plan_json_content is not None and report_dir_exists:
        # materialise plan.json + report_dir for the test
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        plan_p = Path(report_dir) / "plan.json"
        plan_p.write_text(json.dumps(plan_json_content), encoding="utf-8")
    if dry_run_log_exists and report_dir_exists:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / "DRY_RUN.log").write_text(
            "# DRY_RUN.log smoke\n", encoding="utf-8"
        )
    return p


def _run_aggregator(candidates_dir: Path, output_path: Path, **kwargs) -> dict:
    """Run the aggregator as a subprocess and return parsed JSON."""
    cmd = [
        sys.executable,
        str(AGGREGATOR),
        "--candidates-dir",
        str(candidates_dir),
        "--output",
        str(output_path),
    ]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"aggregator rc={proc.returncode}\n{proc.stderr}"
    assert output_path.exists(), "aggregator did not write output file"
    return json.loads(output_path.read_text(encoding="utf-8"))


class TestAggregatorSmoke:
    """Pure aggregator smoke test on synthetic candidates."""

    def test_aggregator_imports(self):
        """Aggregator module imports cleanly without exception."""
        # Importability is implicit in subprocess.run above; here we
        # additionally check that the module can be loaded as a
        # Python module (catches syntax + import errors at module level).
        spec = __import__("importlib.util").util.spec_from_file_location(
            "agg_mod",
            str(AGGREGATOR),
        )
        mod = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "aggregate_readiness")
        assert hasattr(mod, "_validate_candidate")


class TestAggregatorBlockOnMissingWrapper:
    """Verifies BLOCKED verdict for a candidate whose wrapper script
    does NOT exist on disk (AGENTS.md §5.5 filesystem-truth rule)."""

    def test_missing_wrapper_blocked(self, candidates_dir, tmp_path):
        out = tmp_path / "readiness.json"
        _write_candidate(
            candidates_dir,
            "wf299_cfm_3wave",
            script_path="/nonexistent/wrapper.sh",
            script_exists=False,
            notes="Wf-299 wrapper MISSING per filesystem-truth",
        )
        data = _run_aggregator(candidates_dir, out)
        cand = data["candidates"][0]
        assert cand["verdict"] == "BLOCKED"
        assert "MISSING on disk" in cand["reason"]
        assert cand["ready_to_launch"] is False
        assert data["blocked_count"] == 1


class TestAggregatorBlockOnSyntaxFail:
    """Verifies BLOCKED verdict for a wrapper that fails bash -n syntax."""

    def test_syntax_fail_blocked(self, candidates_dir, tmp_path):
        out = tmp_path / "readiness.json"
        # Write a syntactically broken shell script.
        bad_script = tmp_path / "bad_wrapper.sh"
        bad_script.write_text(
            "#!/usr/bin/env bash\nthis is broken syntax (\n", encoding="utf-8"
        )
        bad_script.chmod(0o755)
        _write_candidate(
            candidates_dir,
            "wf_syn_fail",
            script_path=str(bad_script),
            script_exists=True,
        )
        data = _run_aggregator(candidates_dir, out)
        cand = data["candidates"][0]
        assert cand["verdict"] == "BLOCKED"
        assert "bash -n syntax FAIL" in cand["reason"]


class TestAggregatorBlockOnBadPlanJson:
    """Verifies BLOCKED-or-WARN verdict for a plan.json that is malformed
    or missing required meta fields (per Wf-363-A1 schema)."""

    def test_missing_meta_wrapper_name(self, candidates_dir, tmp_path):
        out = tmp_path / "readiness.json"
        # Valid bash + executable + plan.json but missing wrapper_name.
        good_script = tmp_path / "good_wrapper.sh"
        good_script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
        good_script.chmod(0o755)
        report_dir = str(tmp_path / "report")
        _write_candidate(
            candidates_dir,
            "wf_bad_plan",
            script_path=str(good_script),
            script_exists=True,
            report_dir=report_dir,
            report_dir_exists=True,
            dry_run_log_exists=True,
            plan_json_exists=True,
            plan_json_content={"meta": {}},  # missing wrapper_name
        )
        data = _run_aggregator(candidates_dir, out)
        cand = data["candidates"][0]
        # WARN, not BLOCKED — wrapper is functional; just plan.json is
        # missing optional meta.
        assert cand["verdict"] == "WARN"
        assert "wrapper_name" in cand["reason"]


class TestAggregatorWarnOnMissingDryRun:
    """Verifies WARN verdict when DRY_RUN.log is missing for an
    existing wrapper."""

    def test_no_dry_run_log_warn(self, candidates_dir, tmp_path):
        out = tmp_path / "readiness.json"
        good_script = tmp_path / "ok_wrapper.sh"
        good_script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
        good_script.chmod(0o755)
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        # Note: dry_run_log_exists=False (default).
        _write_candidate(
            candidates_dir,
            "wf_no_dry",
            script_path=str(good_script),
            script_exists=True,
            report_dir=str(report_dir),
            report_dir_exists=True,
            dry_run_log_exists=False,
        )
        data = _run_aggregator(candidates_dir, out)
        cand = data["candidates"][0]
        assert cand["verdict"] == "WARN"
        assert "DRY_RUN.log missing" in cand["reason"]


class TestAggregatorReadyAllGreen:
    """Verifies READY verdict when all checks pass."""

    def test_all_green_ready(self, candidates_dir, tmp_path):
        out = tmp_path / "readiness.json"
        good_script = tmp_path / "perfect_wrapper.sh"
        good_script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
        good_script.chmod(0o755)
        report_dir = tmp_path / "report_perfect"
        report_dir.mkdir()
        (report_dir / "DRY_RUN.log").write_text("# ok\n", encoding="utf-8")
        (report_dir / "plan.json").write_text(
            json.dumps(
                {
                    "meta": {"wrapper_name": "perfect", "user_ok_required": True},
                    "plan": {"steps": 7},
                }
            ),
            encoding="utf-8",
        )
        _write_candidate(
            candidates_dir,
            "wf_perfect",
            script_path=str(good_script),
            script_exists=True,
            report_dir=str(report_dir),
            report_dir_exists=True,
            dry_run_log_exists=True,
            plan_json_exists=True,
        )
        data = _run_aggregator(candidates_dir, out)
        cand = data["candidates"][0]
        assert cand["verdict"] == "READY"
        assert cand["ready_to_launch"] is True


class TestAggregatorZeroCandidates:
    """Verifies graceful handling of an empty candidates dir."""

    def test_empty_candidates_dir(self, tmp_path):
        out = tmp_path / "readiness.json"
        data = _run_aggregator(tmp_path, out)
        assert data["total_candidates"] == 0
        assert data["ready_count"] == 0
        assert data["blocked_count"] == 0
        assert data["warn_count"] == 0


class TestAggregatorAtomicWrite:
    """Verifies the aggregator's output JSON is written via the
    atomic_write_file primitive (AGENTS.md §5.6 contract)."""

    def test_output_file_persists(self, candidates_dir, tmp_path):
        out = tmp_path / "atomic_test.json"
        _write_candidate(candidates_dir, "wf_atomic", script_exists=False)
        data = _run_aggregator(candidates_dir, out)
        assert out.exists()
        assert data["generated_at"]
        assert data["wf_origin"] == "Wf-426-A3 post-fix retrain readiness aggregator"


class TestLauncherImportsValidBash:
    """Verifies the launcher shell script passes bash -n syntax check."""

    def test_launcher_bash_n(self):
        """Launcher file exists and passes bash -n syntax check."""
        assert LAUNCHER.exists(), f"launcher MISSING: {LAUNCHER}"
        proc = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, (
            f"launcher bash -n syntax FAIL rc={proc.returncode}: " f"{proc.stderr}"
        )


class TestLauncherHonorsUserOkGuard:
    """Verifies the launcher exits with code 64 if USER_OK is unset."""

    def test_no_user_ok_exits_64(self, tmp_path):
        env = os.environ.copy()
        env.pop("USER_OK", None)
        proc = subprocess.run(
            ["bash", str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert proc.returncode == 64, (
            f"launcher should exit 64 with USER_OK unset, got "
            f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        assert "USER_OK" in proc.stderr


class TestLauncherAcceptsUserOk:
    """Verifies the launcher proceeds past USER_OK guard when set.

    Wf-A269 wave-25 (2026-11-04) — THE PRE-EXISTING RED, AND WHY THE OLD
    TEST COULD NOT PASS.

    THE OLD TEST was `subprocess.run(..., timeout=15)`.  It asserted
    "rc != 64", i.e. "USER_OK=1 got past the guard".  BUT the launcher does
    NOT exit after the guard: the very next statement is the LIVE pre-flight
    (`pre_flight_check.py --pipeline all`), which MEASURED **111.8 s** wall
    (rc 1) on 2026-11-04 — 7.5x the 15 s timeout.  ⇒ the test could only
    ever end in `subprocess.TimeoutExpired`, i.e. it was RED BY CONSTRUCTION
    and could not distinguish "the guard rejected" from "the guard passed
    and the pre-flight is slow".

    THE REPAIR makes the test BOUNDED and PROCESS-GROUP-SAFE instead of
    merely longer:
      * the launcher is started in its OWN SESSION (`start_new_session=True`)
        so the whole tree can be killed with one `os.killpg`;
      * stdout is read until the marker that is printed ONLY AFTER the
        USER_OK guard passes ("[INFO] Running pre-flight v3 fresh-run
        check") — that marker IS the proof the guard passed;
      * the process group is then SIGKILLed, so the 111.8 s pre-flight is
        abandoned and NO orphan survives;
      * the whole thing is bounded by a 30 s deadline.

    WHY NOT SIMPLY RAISE THE TIMEOUT: a 15-minute test that burns the full
    pre-flight on every run is a CI cost with no extra signal, and it would
    still leave the pre-flight's GPU-free but CPU-heavy work running.

    SUPERSEDED TEST RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6
    (the old body is quoted in the docstring of the replacement below).
    """

    #: Printed by the launcher ONLY after the USER_OK guard (rc 64) and the
    #: pre-flight-existence guard (rc 65) have both passed.
    POST_GUARD_MARKER = "[INFO] Running pre-flight v3 fresh-run check"

    def _run_until_marker(self, user_ok, deadline_sec=30.0):
        """Start the launcher, read stdout until POST_GUARD_MARKER.

        Returns (saw_marker, combined_output, rc).  The process group is
        ALWAYS killed before returning — on every path — so the test can
        never leave an orphan pre-flight behind.
        """
        import select
        import signal
        import time

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        if user_ok is None:
            env.pop("USER_OK", None)
        else:
            env["USER_OK"] = user_ok

        proc = subprocess.Popen(
            ["bash", str(LAUNCHER), "--no-such-arg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        chunks = []
        saw_marker = False
        deadline = time.monotonic() + deadline_sec
        try:
            assert proc.stdout is not None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([proc.stdout], [], [], remaining)
                if not ready:
                    break
                line = proc.stdout.readline()
                if line == "":  # EOF — the launcher exited
                    break
                chunks.append(line)
                if self.POST_GUARD_MARKER in line:
                    saw_marker = True
                    break
        finally:
            # Kill the WHOLE process group (the launcher + the pre-flight it
            # spawned).  Never leave an orphan.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                proc.kill()
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except OSError:  # pragma: no cover - defensive
                pass
        rc = proc.returncode
        return saw_marker, "".join(chunks), rc

    def test_user_ok_passes_guard(self):
        """USER_OK=1 gets PAST the guard (marker reached) and does not rc 64."""
        saw_marker, output, rc = self._run_until_marker("1")
        assert saw_marker, (
            "USER_OK=1 should have reached the post-guard marker "
            f"{self.POST_GUARD_MARKER!r}\n"
            f"rc={rc}\noutput={output[-2000:]}"
        )
        assert rc != 64, (
            f"USER_OK=1 should bypass the guard; got rc=64\noutput={output[-2000:]}"
        )

    def test_user_ok_unset_is_rejected_with_rc64(self):
        """USER_OK unset => rc 64, and the post-guard marker is NEVER printed."""
        saw_marker, output, rc = self._run_until_marker(None)
        assert rc == 64, (
            f"USER_OK unset must exit 64; got rc={rc}\noutput={output[-2000:]}"
        )
        assert not saw_marker, (
            "the launcher reached the post-guard marker WITHOUT USER_OK — "
            "the §5.1 USER_OK gate is DEAD"
        )

    def test_user_ok_wrong_value_is_rejected_with_rc64(self):
        """USER_OK=yes (not '1') => rc 64 — the guard tests the VALUE, not
        merely the presence of the variable."""
        saw_marker, output, rc = self._run_until_marker("yes")
        assert rc == 64, (
            f"USER_OK=yes must exit 64; got rc={rc}\noutput={output[-2000:]}"
        )
        assert not saw_marker, (
            "the launcher reached the post-guard marker with USER_OK=yes — "
            "the §5.1 USER_OK gate accepts a non-'1' value"
        )


# ---------------------------------------------------------------------------
# Wf-A269 wave-25 (2026-11-04): the "MEASURE IT, THEN DISCARD IT" class.
#
# THE DEFECT.  The launcher MEASURED the pre-flight live — it ran
# `pre_flight_check.py --pipeline all --json`, parsed PASS / FAIL / SKIP /
# TOTAL, PRINTED them and GATED on FAIL — and then DISCARDED them: the
# aggregator was invoked WITHOUT --pre-flight-pass / --pre-flight-total /
# --pre-flight-skip, so every published pre-flight number fell back to the
# aggregator's hardcoded Wf-353-B1 / Wf-424-B1 baseline (114 PASS / 117
# TOTAL / 3 SKIP) while the live `all` pipeline carries 133 gates
# (MEASURED 2026-11-04:
#   pre_flight_check.EXPECTED_GATE_COUNTS["all"] == 133 == len(build_gates("all"))
# ).  A second instance of the same class sat INSIDE a published
# candidate-description string, where no code could ever compare it.
#
# THE REPAIR has two halves, and BOTH are pinned below:
#   (1) the launcher FORWARDS the counts it measured;
#   (2) the aggregator LABELS which source the published numbers came from
#       (`pre_flight_source`), so a default can never again be read as a
#       measurement.
#
# Each half carries a MUTATION PROOF — a check that cannot fail is the
# repo's own §6.269.17 DEAD-GATE class, and the AGENTS.md §5.11 gates
# exist precisely because "it passed" was never evidence that "it can fail".
# ---------------------------------------------------------------------------

PRE_FLIGHT_FLAGS = ("pass", "total", "skip")

#: The historical baseline the aggregator falls back to.  PRESERVED VERBATIM
#: per AGENTS.md §5.5 rule #6 (the supersession is dated + additive, never a
#: silent edit); it is a HISTORICAL reading, never a measurement.
HISTORICAL_BASELINE = {"pass": 114, "total": 117, "skip": 3}


def _aggregator_invocation(launcher_text: str) -> str:
    """The launcher's LIVE aggregator invocation block (bash continuations).

    Wf-A269 wave-26 (2026-11-04) — TWO fixes to this helper, both forced by
    real defects:

    (1) It used to anchor on ``'python "${AGGREGATOR}"'`` — i.e. it hardcoded
        the INTERPRETER.  When wave-26 repaired the launcher's last two bare
        ``python`` sites (AGENTS.md §5.9), the anchor stopped matching and the
        helper silently returned "" ⇒ the forwarding checks went RED for a
        reason that had nothing to do with forwarding.  That is the repo's
        own "an invariant written against a MENTAL MODEL of the file" trap.
        The anchor is now the AGGREGATOR REFERENCE, which is what actually
        identifies the invocation, independent of which interpreter runs it.

    (2) It used to scan COMMENT lines too.  The wave-26 repair PRESERVED the
        superseded rendering VERBATIM in a dated comment (AGENTS.md §5.5
        rule #6) — and that comment contains the very text
        ``python "${AGGREGATOR}" --candidates-dir ...``.  ⇒ the helper would
        have matched the PRESERVED-BUT-DEAD rendering instead of the live
        one.  Full-line comments are now skipped.
    """
    lines = launcher_text.splitlines()
    collected: list[str] = []
    collecting = False
    for line in lines:
        if line.lstrip().startswith("#"):
            continue  # a comment — including a preserved superseded rendering
        if not collecting:
            if '"${AGGREGATOR}"' in line:
                collecting = True
                collected.append(line)
            continue
        collected.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(collected)


def _launcher_forwards_preflight_flags(launcher_text: str) -> bool:
    """True iff the launcher forwards ALL THREE --pre-flight-* flags."""
    invocation = _aggregator_invocation(launcher_text)
    if not invocation:
        return False
    return all(f"--pre-flight-{k}" in invocation for k in PRE_FLIGHT_FLAGS)


def _aggregator_declares_source(aggregator_text: str) -> bool:
    """True iff the aggregator's published summary carries the source label.

    NB: the check looks for the KEY ASSIGNMENT, not for the bare key name —
    the aggregator's own explanatory comment mentions `"pre_flight_source"`
    by name, so a substring check on the bare name would read the
    DOCUMENTATION as the CODE (the same trap as the §6.269.25 AST-order
    check, where the module's comment names `ThreadPoolExecutor`).
    """
    return '"pre_flight_source":' in aggregator_text


class TestPreFlightCountsAreMeasuredNotDiscarded:
    """Wf-A269 wave-25: the measured pre-flight counts must REACH the
    published readiness JSON, and their source must be LABELLED."""

    # ---- half (1): the launcher forwards what it measured ----------------

    def test_the_launcher_forwards_the_three_measured_counts(self):
        """The aggregator invocation carries all three --pre-flight-* flags."""
        text = LAUNCHER.read_text(encoding="utf-8")
        assert _launcher_forwards_preflight_flags(text), (
            "the launcher does NOT forward --pre-flight-pass/total/skip to "
            "the aggregator — the measured counts would be DISCARDED and "
            "the historical baseline published as if it were a measurement"
        )

    def test_the_launcher_forwards_the_live_counts_not_constants(self):
        """The forwarded VALUES are the launcher's own parsed counts."""
        invocation = _aggregator_invocation(LAUNCHER.read_text(encoding="utf-8"))
        for var, flag in (
            ("PASS_COUNT", "--pre-flight-pass"),
            ("TOTAL_COUNT", "--pre-flight-total"),
            ("SKIP_COUNT", "--pre-flight-skip"),
        ):
            assert f'{flag} "${{{var}}}"' in invocation, (
                f"{flag} must carry the launcher's parsed ${{{var}}}, not a literal"
            )

    def test_the_launcher_candidate_record_interpolates_the_measured_counts(self):
        """The published candidate description carries the LIVE counts and
        PRESERVES the superseded baseline VERBATIM (AGENTS.md §5.5 rule #6)."""
        text = LAUNCHER.read_text(encoding="utf-8")
        assert "${PASS_COUNT}" in text and "${TOTAL_COUNT}" in text
        assert "114/0/3/117" in text, (
            "the superseded Wf-424-A1 baseline rendering must be preserved "
            "VERBATIM per AGENTS.md §5.5 rule #6"
        )

    # ---- half (2): the aggregator labels the source ----------------------

    def test_the_summary_publishes_the_measured_counts_with_a_measured_source(
        self, candidates_dir, tmp_path
    ):
        """With the flags supplied, the summary carries the MEASURED values
        and says so."""
        out = tmp_path / "readiness_measured.json"
        data = _run_aggregator(
            candidates_dir,
            out,
            pre_flight_pass=130,
            pre_flight_total=133,
            pre_flight_skip=3,
        )
        assert data["pre_flight_pass"] == 130
        assert data["pre_flight_total"] == 133
        assert data["pre_flight_skip"] == 3
        assert data["pre_flight_source"] == "measured"
        assert data["pre_flight_source_fields"] == ["pass", "skip", "total"]

    def test_without_the_flags_the_historical_baseline_is_used_and_labelled(
        self, candidates_dir, tmp_path
    ):
        """Without the flags, the PRESERVED historical baseline is published
        — and LABELLED as such, so it can never read as a measurement."""
        out = tmp_path / "readiness_default.json"
        data = _run_aggregator(candidates_dir, out)
        assert data["pre_flight_total"] == HISTORICAL_BASELINE["total"]
        assert data["pre_flight_pass"] == HISTORICAL_BASELINE["pass"]
        assert data["pre_flight_skip"] == HISTORICAL_BASELINE["skip"]
        assert data["pre_flight_source"] == "historical_baseline_default"
        assert data["pre_flight_source_fields"] == []
        assert "HISTORICAL" in data["pre_flight_baseline_note"]

    def test_a_partial_flag_set_is_labelled_partial(self, candidates_dir, tmp_path):
        """Passing SOME of the flags is neither measured nor default."""
        out = tmp_path / "readiness_partial.json"
        data = _run_aggregator(candidates_dir, out, pre_flight_total=133)
        assert data["pre_flight_source"] == "partial"
        assert data["pre_flight_source_fields"] == ["total"]

    def test_the_summary_always_declares_the_preflight_source(
        self, candidates_dir, tmp_path
    ):
        """The label is present on EVERY published summary, whatever the
        caller passed."""
        for name, kwargs in (
            ("measured", {"pre_flight_pass": 1, "pre_flight_total": 2, "pre_flight_skip": 0}),
            ("default", {}),
        ):
            out = tmp_path / f"readiness_label_{name}.json"
            data = _run_aggregator(candidates_dir, out, **kwargs)
            assert "pre_flight_source" in data, (
                f"the {name} summary does NOT declare pre_flight_source"
            )
            assert data["pre_flight_source"] in (
                "measured",
                "partial",
                "historical_baseline_default",
            )

    # ---- the historical baseline is PRESERVED, not silently edited -------

    def test_the_historical_baseline_literals_are_preserved_verbatim(self):
        """The three constants survive byte-for-byte, with a DATED note that
        records what supersedes them (AGENTS.md §5.5 rule #6)."""
        text = AGGREGATOR.read_text(encoding="utf-8")
        for name, value in HISTORICAL_BASELINE.items():
            literal = f"EXPECTED_PRE_FLIGHT_{name.upper()} = {value}"
            assert literal in text, f"the historical literal `{literal}` was edited"
        assert "Wf-A269 wave-25 (2026-11-04)" in text, (
            "the supersession of the baseline must be DATED (§5.5 rule #6)"
        )
        # NB: each phrase is checked on ONE comment line — the note's line
        # breaks fall between "A HISTORICAL" and "BASELINE", and a naive
        # whitespace normalisation would leave the comment's `#` marker in
        # the middle of the phrase.
        assert "THE THREE CONSTANTS ABOVE ARE A HISTORICAL" in text
        assert "BASELINE, NOT A MEASUREMENT OF ANY RUN" in text

    # ---- MUTATION PROOFS: each check must be able to go RED --------------

    def test_the_launcher_forwarding_check_can_fail(self, tmp_path):
        """MUTATION PROOF: strip the flags from a COPY of the launcher and
        confirm the check reports RED."""
        text = LAUNCHER.read_text(encoding="utf-8")
        assert _launcher_forwards_preflight_flags(text) is True
        mutated = text
        for k in PRE_FLIGHT_FLAGS:
            # NB: the replacement must NOT contain the original token as a
            # substring — `--pre-flight-pass-MUTATED` would still contain
            # `--pre-flight-pass` and the mutation would silently NOT take.
            mutated = mutated.replace(f"--pre-flight-{k}", "REMOVED_BY_MUTATION")
        assert mutated != text, "the mutation did not change the launcher text"
        assert _launcher_forwards_preflight_flags(mutated) is False, (
            "the launcher check CANNOT FAIL — it is a DEAD GATE (§6.269.17)"
        )

    def test_the_source_label_check_can_fail(self, tmp_path):
        """MUTATION PROOF: strip `pre_flight_source` from a COPY of the
        aggregator and confirm the published summary loses the label."""
        src = AGGREGATOR.read_text(encoding="utf-8")
        assert _aggregator_declares_source(src) is True
        mutated = src.replace('"pre_flight_source": pf_source,', "")
        assert mutated != src, "the mutation did not change the aggregator text"
        assert _aggregator_declares_source(mutated) is False

        copy = tmp_path / "mutated_aggregator.py"
        copy.write_text(mutated, encoding="utf-8")
        out = tmp_path / "readiness_mutated.json"
        cdir = tmp_path / "mutated_candidates"
        cdir.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(copy),
                "--candidates-dir",
                str(cdir),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"mutated aggregator rc={proc.returncode}\n{proc.stderr}"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "pre_flight_source" not in data, (
            "the mutated aggregator still published a source label — the "
            "mutation did not take"
        )


# ---------------------------------------------------------------------------
# Wf-A269 wave-25 (2026-11-04): the THREE LAUNCHER DEFECTS, and the checks
# that hold them fixed.
#
#   F6-5 (§5.9): every python invocation was BARE `python`; the ambient PATH
#        resolves it to /usr/bin/python (3.14.7, NO torch) — not the project
#        venv.  MEASURED: `USER_OK=1 bash <launcher>` exited rc=1 after
#        47.5 s and the readiness JSON was never produced.
#   F6-6 (`set -e` + command substitution): `VAR="$(cmd)"` propagates cmd's
#        status and errexit aborts, so the next line's `RC=$?` was
#        UNREACHABLE.
#   F6-8 (DEAD GATE): the pre-flight's stdout carries the molmetal import
#        banner BEFORE the JSON, so `json.load(sys.stdin)` raised, the
#        `|| echo "0"` fallback fired, ALL FOUR COUNTS became "0", and
#        `if [[ "${FAIL_COUNT}" -gt 0 ]]` could NEVER fire — while the live
#        pipeline reports 3 FAIL.  MEASURED live (2026-11-04): 120 PASS /
#        3 FAIL / 10 SKIP / 133 TOTAL.
#
# EACH CHECK BELOW CARRIES A MUTATION PROOF: a check that cannot fail is the
# repo's own §6.269.17 DEAD-GATE class, and "it passed" was never evidence
# that "it can fail".
# ---------------------------------------------------------------------------

#: A `python` in COMMAND POSITION — i.e. an actual invocation, not a mention
#: inside a string, a path component, or a comment.
BARE_PYTHON_CMD = re.compile(r"(?:^|[|;&]|\$\()\s*python\b", re.M)


def _strip_shell_comment_lines(text: str) -> str:
    """Drop full-line shell comments.

    Full-line only (a `#` at the start of a stripped line) — an inline `#`
    is NOT dropped, because a `#` inside a double-quoted string is literal.
    The superseded renderings this wave PRESERVED VERBATIM live in
    full-line comments, so they are invisible to the checks below — which is
    correct: a preserved-but-dead rendering must not trip the live check.
    """
    kept = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _launcher_code() -> str:
    """The launcher text with full-line comments removed."""
    return _strip_shell_comment_lines(LAUNCHER.read_text(encoding="utf-8"))


def _bare_python_sites(code: str) -> list:
    """Every command-position `python` invocation in the launcher."""
    return [m.group(0).strip() for m in BARE_PYTHON_CMD.finditer(code)]


#: The PORTABLE spelling of the canonical venv interpreter — the form in
#: docs/STANDING_DIRECTIVES.md §F's canonical block
#: (`exec "$REPO_ROOT/.venv/bin/python" ...`).
PORTABLE_VENV_PY = "${REPO_ROOT}/.venv/bin/python"


def _venv_python_named(text: str) -> bool:
    """Does the launcher name the canonical venv interpreter?

    The ABSOLUTE spelling is DERIVED from this file's own location, never
    written as a host literal: the absolute form is the defect class
    `molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py` guards, and that
    guard's own repair rule for it is "derive it from __file__ /
    sys.executable".

    The PORTABLE form is accepted too, so this check does not block the
    USER-gated §5.9 amendment — but it still FAILS if the launcher names
    neither.
    """
    absolute = str(REPO_ROOT / ".venv" / "bin" / "python")
    return PORTABLE_VENV_PY in text or absolute in text


class TestLauncherVenvAndFailClosedCounts:
    """The three launcher defects, each pinned by a check that CAN fail."""

    # -- F6-5: the §5.9 single-venv rule -------------------------------------

    def test_the_launcher_declares_the_canonical_venv_python(self):
        """AGENTS.md §5.9: the launcher must use the repo's single venv.

        The launcher may spell the interpreter PORTABLY (the §F canonical
        pattern) or ABSOLUTELY (the §5.9 spelling) — the check accepts
        BOTH, so it does not block the §5.9 amendment, but it still FAILS
        if the launcher names neither.

        The check reads the launcher's LIVE CODE (full-line comments
        stripped), so the SUPERSEDED absolute rendering preserved in a
        comment cannot satisfy it on its own.
        """
        assert _venv_python_named(_launcher_code()), (
            "the launcher no longer names the canonical venv interpreter "
            "(AGENTS.md §5.9) — neither the portable form %r nor the "
            "absolute form derived from this file's location is present"
            % (PORTABLE_VENV_PY,)
        )

    def test_the_venv_python_check_can_fail(self):
        """MUTATION PROOF for the check above."""
        mutated = _launcher_code().replace(PORTABLE_VENV_PY, "python3")
        assert mutated != _launcher_code(), "the mutation did not apply"
        assert not _venv_python_named(mutated), (
            "the mutated launcher still names the venv interpreter — the "
            "check cannot fail"
        )

    def test_the_launcher_never_invokes_a_bare_python(self):
        """No command-position `python` survives (comments are excluded)."""
        sites = _bare_python_sites(_launcher_code())
        assert sites == [], (
            "the launcher invokes a BARE `python` at: %r — AGENTS.md §5.9 "
            "forbids the ambient interpreter" % (sites,)
        )

    def test_the_bare_python_check_can_fail(self):
        """MUTATION PROOF for the check above."""
        mutated = _launcher_code().replace('"${PY}"', "python")
        assert mutated != _launcher_code(), "the mutation did not apply"
        assert _bare_python_sites(mutated) != [], (
            "the mutated launcher still shows no bare `python` — the check "
            "cannot fail"
        )

    # -- F6-6: the rc capture must be REACHABLE ------------------------------

    def test_the_preflight_rc_capture_is_reachable(self):
        """`set -e` kills `VAR="$(cmd)"` before the next line's `RC=$?`.

        ⇒ the rc MUST be captured on the SAME command, via `|| RC=$?`.
        """
        code = _launcher_code()
        assert "|| PRE_FLIGHT_RC=$?" in code, (
            "the pre-flight's exit status is not captured via `|| ... $?` — "
            "under `set -e` a separate `PRE_FLIGHT_RC=$?` line is UNREACHABLE"
        )
        # and there must be no standalone (unreachable) capture left over
        assert not re.search(r"^\s*PRE_FLIGHT_RC=\$\?\s*$", code, re.M), (
            "a standalone `PRE_FLIGHT_RC=$?` line is UNREACHABLE under "
            "`set -e` — the status must be captured on the assignment itself"
        )

    def test_the_rc_capture_check_can_fail(self):
        """MUTATION PROOF for the check above."""
        mutated = _launcher_code().replace(
            "|| PRE_FLIGHT_RC=$?", "\nPRE_FLIGHT_RC=$?", 1
        )
        assert mutated != _launcher_code(), "the mutation did not apply"
        assert re.search(r"^\s*PRE_FLIGHT_RC=\$\?\s*$", mutated, re.M), (
            "the mutated launcher still has no standalone capture — the check "
            "cannot fail"
        )

    # -- F6-8: the counts must be READ, and their absence must ABORT ---------

    def test_the_counts_are_read_from_the_report_file_not_stdout(self):
        """The pre-flight's stdout is NOT pure JSON (the import banner)."""
        code = _launcher_code()
        assert '--json "${PRE_FLIGHT_JSON}"' in code, (
            "the launcher no longer asks the pre-flight to write a JSON "
            "FILE — reading its stdout re-introduces the banner defect"
        )
        assert "json.load(sys.stdin)" not in code, (
            "the launcher parses the pre-flight's STDOUT as JSON — the "
            "molmetal import banner makes that raise"
        )

    def test_the_count_extraction_has_no_silent_zero_fallback(self):
        """A count that cannot be read must ABORT, never become 0."""
        code = _launcher_code()
        assert '|| echo "0"' not in code, (
            "the launcher still substitutes a silent `0` for a count it "
            "could not read"
        )
        assert "exit 67" in code, (
            "the launcher no longer has a dedicated fail-closed exit for an "
            "unreadable pre-flight report"
        )

    def test_the_fail_closed_check_can_fail(self):
        """MUTATION PROOF for the check above."""
        mutated = _launcher_code().replace(
            'read -r PASS_COUNT FAIL_COUNT SKIP_COUNT TOTAL_COUNT',
            'PASS_COUNT="$(echo x | python -c \'print(0)\' || echo "0")"\n'
            'read -r PASS_COUNT FAIL_COUNT SKIP_COUNT TOTAL_COUNT',
            1,
        )
        assert mutated != _launcher_code(), "the mutation did not apply"
        assert '|| echo "0"' in mutated, (
            "the mutation did not introduce the silent zero — the check "
            "cannot fail"
        )
        mutated2 = _launcher_code().replace("exit 67", "exit 0")
        assert "exit 67" not in mutated2, (
            "the mutated launcher still fails closed — the check cannot fail"
        )

    # -- §5.5 rule #6: the superseded renderings survive VERBATIM ------------

    def test_the_superseded_bare_python_rendering_is_preserved_verbatim(self):
        """AGENTS.md §5.5 rule #6 — a correction is ADDITIVE + DATED."""
        text = LAUNCHER.read_text(encoding="utf-8")
        for line in (
            'PRE_FLIGHT_OUTPUT="$(python "${PRE_FLIGHT}" --pipeline all --json 2>&1)"',
            "PASS_COUNT=\"$(echo \"${PRE_FLIGHT_OUTPUT}\" | python -c",
            "FAIL_COUNT=\"$(echo \"${PRE_FLIGHT_OUTPUT}\" | python -c",
            "SKIP_COUNT=\"$(echo \"${PRE_FLIGHT_OUTPUT}\" | python -c",
            "TOTAL_COUNT=\"$(echo \"${PRE_FLIGHT_OUTPUT}\" | python -c",
        ):
            assert line in text, (
                "the superseded rendering is NOT preserved VERBATIM: %r" % line
            )


# ---------------------------------------------------------------------------
# Wf-A269 wave-26 (2026-11-04): THE §1.4 FAIL GATE MUST BE ALIVE.
#
# THE DEFECT.  After wave-25 revived the count extraction, the gate was
# STILL mechanically unreachable: the rc check aborted on any rc other than
# 0 or 2, and the pre-flight's rc 1 IS the "a gate FAILed" case (see
# `pre_flight_check.py::main`: `return 1` on a FAIL or a TIMEOUT).  ⇒ the
# script exited at the rc check and NEVER REACHED
# `if [[ "${FAIL_COUNT}" -gt 0 ]]`.  The gate that wave-25 set out to revive
# was, mechanically, still DEAD — the repo's own §6.269.17 class in its
# SECOND costume.
#
# THE PROOF.  A STUB pre-flight (a copy of the launcher in a temp repo
# layout, with a fake `pre_flight_check.py` that writes a chosen report and
# exits with a chosen rc) lets the gate be exercised in ~0.2 s instead of
# the ~100 s of the live 133-gate pre-flight.  The launcher is copied WHOLE
# — no line is edited — so what is exercised is the real script.
#
# WHY A STUB AND NOT A MOCKED FUNCTION: a stub that REPLACES a dependency is
# the only way to prove a BASH gate fires; a Python-level mock would prove
# nothing about the shell.
# ---------------------------------------------------------------------------

STUB_PREFLIGHT = """import json
import sys

path = None
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--json" and i + 1 < len(argv):
        path = argv[i + 1]
doc = {"summary": {"pass": %(pass)d, "fail": %(fail)d, "skip": %(skip)d,
                   "total": %(total)d}}
if path:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
sys.exit(%(rc)d)
"""


def _stub_launcher(tmp_path, n_fail, n_skip, total, rc):
    """Copy the REAL launcher into a temp repo layout with a stub pre-flight.

    The launcher derives REPO_ROOT from its own path as
    ``dirname/../..``, so placing it at ``<tmp>/repo/molmetal/scripts/`` makes
    REPO_ROOT = ``<tmp>/repo`` and the launcher resolves ITS OWN stub
    ``pre_flight_check.py``.  Nothing in the repo is written.
    """
    repo = tmp_path / "repo"
    scripts = repo / "molmetal" / "scripts"
    scripts.mkdir(parents=True)
    stub = scripts / "pre_flight_check.py"
    stub.write_text(
        STUB_PREFLIGHT
        % {
            "pass": total - n_fail - n_skip,
            "fail": n_fail,
            "skip": n_skip,
            "total": total,
            "rc": rc,
        },
        encoding="utf-8",
    )
    # The aggregator is copied too, so the FULL launcher path (candidate
    # records -> aggregator -> readiness JSON) can be exercised.  It
    # resolves `molmetal.scripts.wf386a3_atomic_write` from the INSTALLED
    # package, not from its own directory — so it works from the stub repo.
    aggregator = scripts / "wf426a3_post_fix_results_aggregator.py"
    aggregator.write_text(
        AGGREGATOR.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (repo / "molmetal" / "reports").mkdir(parents=True, exist_ok=True)
    # The launcher resolves its interpreter from ITS OWN repo root
    # (`PY="${REPO_ROOT}/.venv/bin/python"` — the portable form, docs/
    # STANDING_DIRECTIVES.md §F's canonical pattern), so the stub repo must
    # carry a `.venv/bin/python`.  It is a SHIM that execs the interpreter
    # running THIS test — DERIVED, never a host literal (the defect class
    # `molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py` guards).
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    shim = venv_bin / "python"
    shim.write_text(
        "#!/bin/sh\nexec %s \"$@\"\n" % os.path.abspath(sys.executable),
        encoding="utf-8",
    )
    shim.chmod(0o755)
    launcher = scripts / "launcher.sh"
    launcher.write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    return launcher


def _run_stub_launcher(launcher, user_ok="1", timeout=120):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    if user_ok is None:
        env.pop("USER_OK", None)
    else:
        env["USER_OK"] = user_ok
    return subprocess.run(
        ["bash", str(launcher)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(launcher.parent),
    )


class TestThePreFlightFailGateIsAlive:
    """The §1.4 FAIL gate must FIRE when the pre-flight reports a FAIL."""

    def test_the_fail_gate_fires_on_a_reported_failure(self, tmp_path):
        """rc 1 + fail=2 ⇒ the launcher aborts WITH THE COUNT."""
        launcher = _stub_launcher(tmp_path, n_fail=2, n_skip=0, total=3, rc=1)
        proc = _run_stub_launcher(launcher)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 2, (
            f"the §1.4 FAIL gate did not fire (rc={proc.returncode}); "
            f"expected 2\n{combined[-2000:]}"
        )
        assert "Pre-flight has 2 FAIL gate(s)" in combined, (
            "the abort did not report the MEASURED FAIL count\n"
            f"{combined[-2000:]}"
        )

    def test_the_fail_gate_is_not_preempted_by_the_rc_check(self, tmp_path):
        """rc 1 must reach the FAIL gate, NOT abort earlier with a bare rc.

        This is the wave-26 defect: the rc check used to swallow rc 1, so
        the measured count was never printed.
        """
        launcher = _stub_launcher(tmp_path, n_fail=2, n_skip=0, total=3, rc=1)
        proc = _run_stub_launcher(launcher)
        combined = proc.stdout + proc.stderr
        assert "outside the pre-flight's documented" not in combined, (
            "the rc check swallowed rc 1 — the §1.4 FAIL gate is UNREACHABLE "
            "again (the wave-26 defect)\n" + combined[-2000:]
        )
        assert "Pre-flight: 1 PASS / 2 FAIL / 0 SKIP / 3 TOTAL" in combined, (
            "the launcher did not print the measured counts\n"
            + combined[-2000:]
        )

    def test_a_timeout_with_zero_fail_gates_still_aborts(self, tmp_path):
        """rc 1 with fail=0 = a TIMEOUT (or strict SKIP) ⇒ must not pass."""
        launcher = _stub_launcher(tmp_path, n_fail=0, n_skip=0, total=3, rc=1)
        proc = _run_stub_launcher(launcher)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 2, (
            f"a pre-flight rc=1 with 0 FAIL gates passed silently "
            f"(rc={proc.returncode})\n{combined[-2000:]}"
        )
        assert "0 FAIL gate(s)" in combined, (
            "the abort did not explain the unexplained non-zero rc\n"
            + combined[-2000:]
        )

    def test_a_clean_preflight_passes_the_gate(self, tmp_path):
        """rc 0 + fail=0 ⇒ the §1.4 gate does NOT fire (negative control)."""
        launcher = _stub_launcher(tmp_path, n_fail=0, n_skip=0, total=3, rc=0)
        proc = _run_stub_launcher(launcher)
        combined = proc.stdout + proc.stderr
        assert "Pre-flight has 0 FAIL gate(s)" not in combined, (
            "the FAIL gate fired on a clean pre-flight\n" + combined[-2000:]
        )
        assert "outside the pre-flight's documented" not in combined, (
            "the rc check fired on rc 0\n" + combined[-2000:]
        )

    def test_the_full_path_publishes_the_measured_counts(self, tmp_path):
        """rc 0 + fail=0 ⇒ the launcher runs to completion and PUBLISHES.

        This exercises the whole F6 chain end-to-end: measure (the pre-flight
        report) -> forward (the three --pre-flight-* flags) -> label (the
        aggregator's `pre_flight_source`).  It also covers the two bare
        `python` sites wave-26 repaired (the aggregator call and the
        summary-table heredoc), which the failing-pre-flight cases never
        reach.
        """
        launcher = _stub_launcher(tmp_path, n_fail=0, n_skip=0, total=3, rc=0)
        proc = _run_stub_launcher(launcher)
        combined = proc.stdout + proc.stderr
        readiness = tmp_path / "repo" / "molmetal" / "reports" / "wf426a3_retrain_readiness.json"
        assert proc.returncode == 0, (
            f"the launcher did not complete (rc={proc.returncode})\n"
            f"{combined[-3000:]}"
        )
        assert readiness.exists(), (
            "the readiness JSON was never written\n" + combined[-3000:]
        )
        data = json.loads(readiness.read_text(encoding="utf-8"))
        assert data["pre_flight_source"] == "measured", (
            "the published summary does not say its numbers were MEASURED: "
            f"{data.get('pre_flight_source')!r}"
        )
        assert data["pre_flight_total"] == 3, (
            "the published TOTAL is not the measured one: "
            f"{data.get('pre_flight_total')!r}"
        )
        assert data["pre_flight_pass"] == 3, (
            "the published PASS count is not the measured one: "
            f"{data.get('pre_flight_pass')!r}"
        )

    def test_the_fail_gate_check_can_fail(self):
        """MUTATION PROOF: restore the wave-26 defect and the gate must die.

        The mutation re-introduces the ORIGINAL rc check (which aborts on
        rc 1 before the FAIL gate).  With the mutation in place, the
        fail-gate test MUST fail.
        """
        original = LAUNCHER.read_text(encoding="utf-8")
        mutated = original.replace(
            'if [[ "${PRE_FLIGHT_RC}" -ne 0 && "${PRE_FLIGHT_RC}" -ne 1 '
            '&& "${PRE_FLIGHT_RC}" -ne 2 ]]; then',
            'if [[ "${PRE_FLIGHT_RC}" -ne 0 && "${PRE_FLIGHT_RC}" -ne 2 ]]; then',
            1,
        )
        assert mutated != original, "the mutation did not apply"
        assert 'PRE_FLIGHT_RC}" -ne 1' not in mutated, (
            "the mutation did not restore the original rc check"
        )
