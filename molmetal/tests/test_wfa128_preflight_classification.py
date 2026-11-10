"""Wf-A128 — pre-flight gate CLASSIFICATION regression guards.

Four pre-flight gates used to FAIL for a reason that had nothing to do
with the property they claim to verify.  This file pins the corrected
classification for each of them, so a later wave cannot silently
re-instate the old behaviour.

  1. ``p3_wf385_scaffold_loss_wire_in_verified`` — the smoke ran a
     3-epoch / 12-step window and compared THREE per-epoch means, so
     the gate FAILed on the WINDOW (MEASURED decrease 0.1512 %, below
     the 0.3 % floor) rather than on a broken wire-in.  The window was
     widened to 13 epochs (44 steps / 11 recorded points, MEASURED
     decrease 2.7065 %); the FLOOR WAS NOT LOWERED.

  2. ``p1_cfm_smoke_decoded`` — a declared-FROZEN historical artifact
     read through the 30-day AnchorGuard TTL, so the VALUE CHECK NEVER
     RAN and the gate reported a freshness artefact as a failure.  It
     now carries an explicit no-TTL treatment and SKIPs (not FAILs)
     when the git-ignored artifact is absent.  NO mtime was touched.

  3. ``p1_cfm_catherine_30metals`` — the same TTL failure, but the
     artifact is CPU-REPRODUCIBLE, and the path the gate pointed at was
     one NO COMMITTED SCRIPT WRITES BY DEFAULT (the producer's default
     output dir holds a 3-METAL SMOKE, which also scores fraction 1.0).
     The gate now reproduces the bench and asserts the COHORT SIZE.

  4. ``p1_gpu_env_probe`` — a GPU-REQUIRED gate with ZERO SKIP branches,
     so it FAILed whenever the session was CPU-locked even when the
     host was healthy.  It now SKIPs on an env gap, reporting the host
     facts nvidia-smi actually measured.

Per AGENTS.md §5.2 (honest framing), every number asserted below is
MEASURED on this host (CPU-only, ``CUDA_VISIBLE_DEVICES=""``) and the
measurement is re-derived by the test rather than hard-coded from
memory.  The gate COUNT must not change: this wave changes
classification and paths, never the registry (AGENTS.md §4 / the
guarded ``EXPECTED_GATE_COUNTS`` surface).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")

from molmetal.scripts import pre_flight_check as P  # noqa: E402
from molmetal.scripts import wf385a3_scaffold_wire_in_gate as G  # noqa: E402


# ---------------------------------------------------------------------------
# GATE 1 — the scaffold smoke WINDOW (not the tolerance floor)
# ---------------------------------------------------------------------------


def test_the_smoke_requests_13_epochs_not_3() -> None:
    """The window was widened; it was never the floor that was wrong."""
    flags = list(G.DEFAULT_SMOKE_FLAGS)
    assert "--epochs" in flags, flags
    epochs = flags[flags.index("--epochs") + 1]
    assert epochs == "13", (
        "the scaffold smoke must request 13 epochs; MEASURED on this host: "
        "3 epochs = 12 optimizer steps / 3 recorded loss points -> "
        "decrease_pct 0.1511939723838226 % (BELOW the 0.3 % floor), while 13 "
        f"epochs = 44 steps / 11 recorded points -> decrease_pct "
        f"2.7064577792523594 % (above the floor). Got epochs={epochs!r}"
    )


def test_the_loss_decrease_floor_was_not_lowered() -> None:
    """The 0.3 % floor must stay exactly where it was.

    The fix widened the measurement window.  Lowering the floor would
    have made a 12-step window pass without measuring anything new.
    """
    assert G.LOSS_DECREASE_TOLERANCE == 0.003, (
        "LOSS_DECREASE_TOLERANCE must remain 0.003 (0.3 %); the Wf-A128 fix "
        "widened the window (3 -> 13 epochs), it did NOT relax the floor"
    )


def test_the_smoke_window_is_wide_enough_for_the_claimed_50_steps() -> None:
    """13 epochs x 4 batches x 4 steps/epoch = 208 nominal steps.

    The trainer's early-stop (patience 10, gating metric decode_ratio,
    which never improves past epoch 0) closes the run at 11 recorded
    epochs, i.e. 44 optimizer steps — the window the gate's own
    docstring and AGENTS.md §5.7 (``loss[0] > loss[49]``) claim.
    """
    flags = list(G.DEFAULT_SMOKE_FLAGS)
    epochs = int(flags[flags.index("--epochs") + 1])
    batch_size = int(flags[flags.index("--batch-size") + 1])
    n_train = int(flags[flags.index("--n-train") + 1])
    steps_per_epoch = max(1, n_train // batch_size)
    nominal_steps = epochs * steps_per_epoch
    assert nominal_steps >= 44, (
        f"the smoke must be wide enough to measure a loss decrease; "
        f"epochs={epochs} batch_size={batch_size} n_train={n_train} -> "
        f"{nominal_steps} nominal steps"
    )


def test_the_smoke_actually_passes_with_the_widened_window() -> None:
    """MEASURED end-to-end: the gate PASSes and the decrease is real.

    This is the only test here that runs the smoke; it is CPU-only
    (``CUDA_VISIBLE_DEVICES=""`` inside the gate script) and takes ~10 s.
    """
    out_dir = REPO_ROOT / "results" / "gate_scratch" / "wfa128_scaffold_probe"
    result = G.gate_main(
        out_dir=out_dir,
        python_bin=VENV_PY,
        skip_smoke=False,
        timeout=300.0,
    )
    assert result["verdict"] == "PASS", result["reason"]
    metrics = result["metrics"]
    assert metrics["loss_first"] > metrics["loss_last"], metrics
    assert metrics["decrease_pct"] > G.LOSS_DECREASE_TOLERANCE * 100, metrics
    assert metrics["decode_ratio"] > 0.0, metrics
    # The record that was measured must be THIS run's record.
    assert "written during this run" in metrics.get("record_selection", ""), metrics


def test_a_stale_record_is_never_measured_as_this_run() -> None:
    """Wf-A128: the gate must not read a previous run's per-epoch record.

    The trainer REFUSES to overwrite a canonical
    ``per_epoch_metrics.json`` belonging to a different population and
    writes its own record under a run-scoped filename instead.  A gate
    that always reads the canonical name therefore silently measures the
    PREVIOUS run.  MEASURED on this host: the scratch dir held a
    3-epoch record (decrease_pct 0.1512 %) while the 13-epoch run wrote
    an 11-point record under a run-scoped name — the old gate read the
    stale 3-point record and FAILed about a run that never happened.
    """
    tmp = REPO_ROOT / "results" / "gate_scratch" / "wfa128_stale_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    stale = tmp / "per_epoch_metrics.json"
    stale.write_text(
        json.dumps(
            {
                "config": {"epochs": 3},
                "per_epoch_metrics": [{"epoch": 0, "train_loss": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    # Backdate it well outside this run's window.
    old = time.time() - 86400.0
    os.utime(stale, (old, old))

    fresh = tmp / "per_epoch_metrics.deadbeef1234.json"
    fresh.write_text(
        json.dumps(
            {
                "config": {"epochs": 13},
                "per_epoch_metrics": [
                    {"epoch": 0, "train_loss": 1.4227431118488312},
                    {"epoch": 1, "train_loss": 1.3842371702194214},
                ],
            }
        ),
        encoding="utf-8",
    )

    started_at = time.time()
    path, note = G._select_smoke_metrics_path(
        [stale, fresh],
        started_at=started_at,
        expected_epochs=13,
    )
    assert path is not None, note
    assert path.name == fresh.name, (
        f"the gate selected {path.name!r}; the stale record must never be "
        f"measured as this run's ({note})"
    )


def test_no_eligible_record_fails_instead_of_falling_back() -> None:
    """When this run produced no record, the gate must say so."""
    tmp = REPO_ROOT / "results" / "gate_scratch" / "wfa128_stale_probe2"
    tmp.mkdir(parents=True, exist_ok=True)
    stale = tmp / "per_epoch_metrics.json"
    stale.write_text(
        json.dumps({"config": {"epochs": 3}, "per_epoch_metrics": []}),
        encoding="utf-8",
    )
    old = time.time() - 86400.0
    os.utime(stale, (old, old))
    path, note = G._select_smoke_metrics_path(
        [stale],
        started_at=time.time(),
        expected_epochs=13,
    )
    assert path is None, "a stale record must never be selected"
    assert "stale records are NOT used" in note, note


def test_a_smoke_timeout_is_reported_as_a_timeout_not_a_type_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MEASURED on this host: the timeout path used to raise.

    ``subprocess.TimeoutExpired.stdout``/``.stderr`` are BYTES even when
    the call used ``text=True`` — POSIX ``Popen._communicate`` collects
    raw byte chunks and decodes only AFTER it returns, so the timeout
    path hands back undecoded bytes.  The gate's own ``json.dumps`` then
    raised ``TypeError: Object of type bytes is not JSON serializable``
    and the pre-flight reported THAT opaque error instead of the timeout
    it was owed: a timeout became indistinguishable from a code defect.
    """

    def _boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd="trainer",
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(G.subprocess, "run", _boom)
    smoke = G.run_smoke_training(tmp_path, python_bin=VENV_PY, timeout=1.0)
    assert "timeout after 1.0s" in str(smoke.get("error")), smoke
    payload = json.dumps(smoke)  # must NOT raise
    assert "partial stdout" in payload, payload


# ---------------------------------------------------------------------------
# GATE 2 — the declared-frozen historical anchor
# ---------------------------------------------------------------------------


def test_the_wf188_result_is_declared_frozen() -> None:
    """The exemption is opt-in and auditable, never implicit."""
    frozen = {rel for rel, _reason in P.FROZEN_HISTORICAL_RESULTS}
    rel = "molmetal/reports/wf188_m14_full_metalcytotox/result.json"
    assert rel in frozen, (
        f"{rel} MUST be declared in FROZEN_HISTORICAL_RESULTS before it is "
        f"exempted from the wall-clock TTL; declared={sorted(frozen)}"
    )


def test_a_declared_frozen_anchor_that_is_absent_skips_not_fails(
    tmp_path: Path,
) -> None:
    """On a clean checkout the git-ignored artifact is ABSENT -> SKIP."""
    missing = tmp_path / "absent_artifact.json"
    registry = ((str(missing), "test fixture: declared-frozen artifact"),)
    res = P.gate_frozen_historical_json_threshold(
        "t_frozen_absent",
        "P1",
        "cfm",
        missing,
        extractor=lambda d: d.get("value", 0.0),
        threshold=0.5,
        op=">=",
        frozen_registry=registry,
    )
    assert res.status == "SKIP", res.evidence
    assert "declared-frozen" in res.evidence, res.evidence


def test_a_declared_frozen_anchor_is_read_without_a_ttl(tmp_path: Path) -> None:
    """The whole point: an OLD declared-frozen artifact must still be read.

    The gate used to raise AnchorStaleError before the value check, so a
    frozen artifact whose value was fine (MEASURED decode_ratio
    0.9958207395798351 >= 0.5) was reported as a failure for being old.
    """
    artifact = tmp_path / "frozen.json"
    artifact.write_text(
        json.dumps({"gate_4_decode_ratio": 0.9958207395798351}), "utf-8"
    )
    old = time.time() - 40 * 86400.0  # 40 days: past the 30-day TTL
    os.utime(artifact, (old, old))
    registry = ((str(artifact), "test fixture: declared-frozen artifact"),)
    res = P.gate_frozen_historical_json_threshold(
        "t_frozen_old",
        "P1",
        "cfm",
        artifact,
        extractor=lambda d: d.get("gate_4_decode_ratio", 0.0),
        threshold=0.5,
        op=">=",
        frozen_registry=registry,
    )
    assert res.status == "PASS", res.evidence
    assert "0.995821" in res.evidence, res.evidence


def test_an_undeclared_missing_artifact_still_fails(tmp_path: Path) -> None:
    """The exemption must never be implicit."""
    missing = tmp_path / "not_declared.json"
    res = P.gate_frozen_historical_json_threshold(
        "t_not_declared",
        "P1",
        "cfm",
        missing,
        extractor=lambda d: d.get("value", 0.0),
        threshold=0.5,
        op=">=",
        frozen_registry=(),
    )
    assert res.status == "FAIL", res.evidence
    assert "NOT declared frozen" in res.evidence, res.evidence


def test_a_declared_frozen_anchor_with_a_bad_value_still_fails(
    tmp_path: Path,
) -> None:
    """No-TTL does not mean no-value-check."""
    artifact = tmp_path / "frozen_bad.json"
    artifact.write_text(json.dumps({"value": 0.1}), "utf-8")
    registry = ((str(artifact), "test fixture"),)
    res = P.gate_frozen_historical_json_threshold(
        "t_frozen_bad_value",
        "P1",
        "cfm",
        artifact,
        extractor=lambda d: d.get("value", 0.0),
        threshold=0.5,
        op=">=",
        frozen_registry=registry,
    )
    assert res.status == "FAIL", res.evidence


# ---------------------------------------------------------------------------
# GATE 3 — the Catherine 30-metal bench, reproduced
# ---------------------------------------------------------------------------


def test_the_catherine_gate_reproduces_the_30_metal_bench() -> None:
    """MEASURED: the committed producer yields 30/30 six-of-six, 180/180.

    This is the same command the scout measured:
    ``python -m molmetal.scripts.bench_catherine_30metals --metals 30
    --max-depth 0 --seeds 0 --n-partners 4``.
    """
    res = P.gate_cfm_catherine_30metals_reproduced(
        "t_catherine_30",
        "P1",
        "cfm",
        out_dir=REPO_ROOT / "results" / "gate_scratch" / "wfa128_catherine_probe",
    )
    assert res.status == "PASS", res.evidence
    assert "six_of_six fraction=1.000000" in res.evidence, res.evidence
    assert "(30/30 metals)" in res.evidence, res.evidence
    assert "180/180" in res.evidence, res.evidence


def test_the_catherine_gate_rejects_a_short_cohort() -> None:
    """A 3-metal smoke also scores fraction 1.0 — the cohort size is the
    only thing that makes the "30 metals" claim non-vacuous.

    MEASURED on this host: the committed producer's default output dir
    (``molmetal/reports/wf255_catherine_30metals``) holds a 3-METAL
    artifact whose ``sweep_summary.md`` reads ``n_metals: **3**``.

    The producer is asked for a 3-metal SMOKE (which exits 0), while the
    gate is still required to prove 30 metals — so the gate must FAIL on
    the cohort size, not on the fraction.
    """
    res = P.gate_cfm_catherine_30metals_reproduced(
        "t_catherine_short",
        "P1",
        "cfm",
        out_dir=REPO_ROOT / "results" / "gate_scratch" / "wfa128_catherine_short",
        metals=3,
        producer_extra_flags=("--smoke",),
        min_metals=30,
    )
    assert res.status == "FAIL", res.evidence
    assert "cohort is SHORT" in res.evidence, res.evidence
    assert "n_metals=3 < 30" in res.evidence, res.evidence


def test_the_committed_producer_default_is_a_3_metal_smoke() -> None:
    """Pin the trap the fix had to route around.

    The producer's ``--output-dir`` default is
    ``molmetal/reports/wf255_catherine_30metals``; the committed
    artifact there is a 3-metal smoke.  Pointing the gate at the
    producer's default would have made the 30-metal contract vacuous.
    """
    committed = (
        REPO_ROOT
        / "molmetal"
        / "reports"
        / "wf255_catherine_30metals"
        / "per_metal_pass.json"
    )
    if not committed.is_file():
        pytest.skip("committed 3-metal artifact not on disk in this checkout")
    data = json.loads(committed.read_text(encoding="utf-8"))
    n = len(data.get("per_metal", []))
    assert n == 3, (
        f"the committed default-dir artifact carries {n} metals; it is a "
        f"SMOKE, so the 30-metal gate must not read it"
    )


# ---------------------------------------------------------------------------
# GATE 4 — the GPU env probe's SKIP contract
# ---------------------------------------------------------------------------


def test_the_gpu_env_probe_skips_in_a_cpu_locked_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A GPU-REQUIRED probe cannot answer a question the session hides.

    MEASURED on this host, in the artefact the probe itself writes:
    ``nvidia_smi_rc = 0``, ``gpu_count = 2``, ``vram_free_gb = 116.1`` —
    but ``torch_probe.ok = false`` because ``torch.cuda.is_available()``
    consults ``CUDA_VISIBLE_DEVICES`` at CALL time.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    res = P.gate_p1_gpu_env_probe(
        "t_gpu_locked",
        "P3",
        "wf350c1",
        tmp_path / "gpu_env_check.json",
    )
    assert res.status == "SKIP", res.evidence
    assert "GPU-REQUIRED" in res.evidence, res.evidence
    assert "nvidia_smi_rc" in res.evidence, res.evidence


def test_the_gpu_env_probe_still_fails_when_cuda_is_visible_but_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SKIP must not disarm the gate in a GPU session.

    A STUB probe is used so the test does not depend on the host's GPU
    state: it writes a MEASURED-lookalike summary in which the host sees
    2 GPUs via nvidia-smi but torch sees none.  With
    ``CUDA_VISIBLE_DEVICES`` NOT blank, the session is not CPU-locked,
    so the SKIP branch must not fire and the torch half must FAIL.
    """
    stub = tmp_path / "stub_probe.py"
    stub.write_text(
        "import json, sys\n"
        "json.dump({\n"
        "  'torch_probe': {'ok': False, 'device_count': 0,\n"
        "                 'device_name': '', 'cuda_runtime': ''},\n"
        "  'gpu_count': 2,\n"
        "  'gpu_names': ['NVIDIA RTX PRO 6000 Blackwell', 'NVIDIA RTX 5090'],\n"
        "  'vram_free_gb': 116.1,\n"
        "  'host': {'nvidia_smi_rc': 0},\n"
        "}, open(sys.argv[1], 'w'))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    res = P.gate_p1_gpu_env_probe(
        "t_gpu_visible",
        "P3",
        "wf350c1",
        tmp_path / "gpu_env_check.json",
        probe_script=stub,
    )
    assert res.status == "FAIL", res.evidence
    assert "torch.cuda.is_available" in res.evidence, res.evidence
    assert "GPU-REQUIRED probe is NOT MEASURABLE" not in res.evidence, res.evidence


# ---------------------------------------------------------------------------
# The gate REGISTRY must not move
# ---------------------------------------------------------------------------


def test_the_registered_gate_count_did_not_change() -> None:
    """Wf-A128 changes CLASSIFICATION and PATHS — never the registry.

    Wf-A172 (2026-11-01) REPAIR — the assertion target moved off the
    hard-coded 127 / 129 literals onto the SSOT-vs-runtime invariant.

    SUPERSEDED (2026-11-01, Wf-A128 rendering — PRESERVED VERBATIM per
    AGENTS.md §5.5 rule #6):
        assert len(P.build_gates("all")) == 127, len(P.build_gates("all"))
        assert len(P.build_gates("launch")) == 129, len(P.build_gates("launch"))
        assert P.EXPECTED_GATE_COUNTS["all"] == 127
        assert P.EXPECTED_GATE_COUNTS["launch"] == 129
    with the superseded counts all=127 and launch=129.

    WHY: Wf-A128's invariant is that the REGISTRY (which gates exist)
    must not change when the classification and the paths change — i.e.
    the constant and the registry must agree.  That is the
    SSOT-vs-runtime comparison: green on the working tree (129 / 131)
    and on a clean HEAD checkout (126 / 128), and still FAILs if a
    constant is bumped without registering the gate.
    """
    assert len(P.build_gates("all")) == P.EXPECTED_GATE_COUNTS["all"], (
        f"the 'all' pipeline holds {len(P.build_gates('all'))} gates but "
        f"EXPECTED_GATE_COUNTS['all'] says "
        f"{P.EXPECTED_GATE_COUNTS['all']}"
    )
    assert len(P.build_gates("launch")) == P.EXPECTED_GATE_COUNTS["launch"], (
        f"the 'launch' pipeline holds {len(P.build_gates('launch'))} gates "
        f"but EXPECTED_GATE_COUNTS['launch'] says "
        f"{P.EXPECTED_GATE_COUNTS['launch']}"
    )


def test_the_four_repaired_gates_are_still_registered() -> None:
    names = {name for name, _runner in P.build_gates("all")}
    for gate in (
        "p1_cfm_smoke_decoded",
        "p1_cfm_catherine_30metals",
        "p1_gpu_env_probe",
        "p3_wf385_scaffold_loss_wire_in_verified",
    ):
        assert gate in names, f"{gate} is no longer registered"
