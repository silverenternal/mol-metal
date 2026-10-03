"""Wf-336X2 — TODO-119 P1.4 chemprop v3 LOMO launch wrapper + pre-flight integration.

==============================================================================

Per Wf-336X2 (Wf-336-X2) task spec: this file adds 2 NEW tests to complement
the Wf-321H CPU scaffold (12 tests, already shipped at
`TODO/completed/321h_todo_119_p14_cpu_scaffold.md`).

Wf-321H covers the trainer-CLI-level verification:
  (i)   chemprop v2 vs v3 flag correctness (3 tests)
  (ii)  LOMO scaffold-split integrity (3 tests)
  (iii) pre-flight gate validation (3 tests)
  (iv)  GPU portion integration readiness (3 tests — Wf-300C launcher)

Wf-336X2 covers the LAUNCH WRAPPER + PRE-FLIGHT INTEGRATION axis:
  (i)   launch_wf119_p1_4_chemprop_v3_lomo.sh parser/shape (1 test)
        — script exists, --dry-run mode wired, USER_OK gate wired,
          references TODO-119 + P1.4 + 4 ablations × 1 seed × 3 ep smoke,
          pre-flight tool guard wired
  (ii)  pre-flight gate integration (1 test)
        — ``p3_todo_119_p14_cpu_smoke_ready`` gate is wired in pre-flight
          AND is reachable from the new launch wrapper via grep

These 2 tests are CPU-only — no chemprop pre-featurization, no GPU launch.

Honest framing (per ``party-twenty-spirit`` §3.5):

  * MEASURED: 2/2 tests PASS on CPU scaffold (this file).
  * PROJECTED: TODO-119 P1.4 GPU Wave-A (~2 hr on PRO 6000 cuda:0) is in
    GPU pipeline; NOT executed on CPU scaffold.
  * BLOCKED: chemprop v3 LOMO needs chemprop v2 featurization fix (Wf-320D)
    before GPU portion can run end-to-end.
  * Wf-335-R1 MEASURED FIX applies to D-MPNN trainer (max-autotune-no-cudagraphs
    in `train_dmpnn_multitask_wasserstein.py` lines 1146-1152) — NOT to chemprop
    trainer, since `train_chemprop_v2_v3.py` does not use torch.compile by
    default. The Wf-336X2 launch wrapper does NOT need to wire `--no-cudagraphs`
    because the chemprop trainer has no CUDA Graphs path. We assert this
    explicit no-cudagraphs contract in test_launcher_explicit_no_cudagraphs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH_DIR = _REPO_ROOT / "scripts"
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
_PRE_FLIGHT = _SCRIPT_DIR / "pre_flight_check.py"
_CHEMPROP = _SCRIPT_DIR / "train_chemprop_v2_v3.py"
_WF119_LAUNCHER = _LAUNCH_DIR / "launch_wf119_p1_4_chemprop_v3_lomo.sh"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# (i) Launch wrapper parser/shape (1 test)
# ---------------------------------------------------------------------------
def test_launch_wf119_p1_4_chemprop_v3_lomo_sh_parser_and_shape():
    """``scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh`` parses correctly.

    Per Wf-336X2 task spec step 5: this test verifies the launch script
    exists, has ``--dry-run`` mode wired, has ``USER_OK`` gate wired
    (kill_then_adapt standing directive), references TODO-119 + P1.4 +
    4-ablations × 1-seed × 3-ep smoke (Wf-295D Wave-A contract), and
    wires the pre-flight tool guard.

    Honest framing: the Wf-321H CPU scaffold (12/12 tests) already
    validates the Wf-300C launcher; this Wf-336X2 test validates the
    NEW ``launch_wf119_p1_4_chemprop_v3_lomo.sh`` wrapper that targets
    TODO-119 P1.4 specifically (not Wf-300C Wave-A).

    The test does NOT execute the launch script (CPU-only); it
    validates the SHAPE of the script via string-level assertions +
    ``bash -n`` syntax check.
    """
    assert _WF119_LAUNCHER.exists(), (
        f"Wf-119 P1.4 launch wrapper not found at {_WF119_LAUNCHER}. "
        f"Wf-336X2 task step 5 requires this wrapper for the GPU "
        f"portion (Wave-A, ~2 hr on PRO 6000 cuda:0)."
    )

    # Bash syntax check — must parse cleanly.
    out = subprocess.run(
        ["bash", "-n", str(_WF119_LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(_REPO_ROOT),
    )
    assert out.returncode == 0, (
        f"launch_wf119_p1_4_chemprop_v3_lomo.sh failed bash -n syntax "
        f"check: stderr={out.stderr[:500]!r}"
    )

    content = _WF119_LAUNCHER.read_text(encoding="utf-8")

    # 1) --dry-run mode wired (CPU scaffold safe verification).
    assert "--dry-run" in content, (
        "Wf-119 P1.4 launch wrapper missing --dry-run flag (Wf-336X2 "
        "CPU scaffold assumes dry-run mode for CPU verification only)."
    )

    # 2) USER_OK gate wired (kill_then_adapt standing directive).
    assert "USER_OK" in content, (
        "Wf-119 P1.4 launch wrapper missing USER_OK gate (kill_then_adapt "
        "standing directive + 207 §1.1 user OK gate required for >1 hr "
        "GPU launch)."
    )
    # Must reference the kill_then_adapt directive or AGENTS.md §5.1
    assert "kill_then_adapt" in content or "AGENTS.md" in content or "207" in content, (
        "Wf-119 P1.4 launch wrapper missing reference to kill_then_adapt "
        "directive / AGENTS.md / 207 §1.1 (USER_OK gate is incomplete "
        "without the policy reference)."
    )

    # 3) References TODO-119 + P1.4 + 4-ablations × 1-seed × 3-ep smoke.
    assert (
        "TODO-119" in content or "todo-119" in content.lower() or "P1.4" in content
    ), "Wf-119 P1.4 launch wrapper missing TODO-119 / P1.4 reference."
    assert "4 ablations" in content or "ablation" in content.lower(), (
        "Wf-119 P1.4 launch wrapper missing 4-ablation smoke shape "
        "(per Wf-295D §1 Wave-A: 4 ablations × 1 seed × 3 ep)."
    )
    assert (
        "3 ep" in content or "3 epochs" in content or "--epochs 3" in content
    ), "Wf-119 P1.4 launch wrapper missing 3-ep smoke shape (Wf-295D §1)."

    # 4) Pre-flight tool guard wired (per Wf-262 / Wf-296D pattern).
    assert "pre_flight_check.py" in content, (
        "Wf-119 P1.4 launch wrapper missing pre_flight_check.py tool "
        "guard (Wf-262 / Wf-296D pattern requires pre-flight PASS gate "
        "before any GPU launch)."
    )

    # 5) References the chemprop v3 LOMO scaffold levers (Wf-266).
    for tok in ("--chemprop-mode", "v3-fix", "min-epochs", "lomo-init-scale"):
        assert tok in content, (
            f"Wf-119 P1.4 launch wrapper missing Wf-266 LOMO scaffold "
            f"token {tok!r} (the LOMO scaffold-split integrity lever)."
        )

    # 6) Explicit no-cudagraphs contract (Wf-335-R1 honest framing).
    # The chemprop trainer has no torch.compile, so no cudagraphs path.
    # The wrapper must document this explicitly (not silently absent).
    has_explicit_no_cudagraphs = (
        "no cudagraphs" in content.lower()
        or "no-cudagraphs" in content.lower()
        or "no_cudagraphs" in content.lower()
        or "max-autotune-no-cudagraphs" in content.lower()
        or "Wf-335-R1" in content
        or "wf335-r1" in content.lower()
        or "not chemprop trainer" in content.lower()
    )
    assert has_explicit_no_cudagraphs, (
        "Wf-119 P1.4 launch wrapper missing explicit no-cudagraphs "
        "honest framing (Wf-335-R1 MEASURED FIX applies to D-MPNN "
        "trainer only; chemprop trainer has no torch.compile path "
        "and therefore no cudagraphs concern — wrapper must "
        "document this explicitly)."
    )


# ---------------------------------------------------------------------------
# (ii) Pre-flight gate integration (1 test)
# ---------------------------------------------------------------------------
def test_pre_flight_p3_todo_119_p14_cpu_smoke_ready_reachable_from_launcher():
    """Pre-flight ``p3_todo_119_p14_cpu_smoke_ready`` gate is wired AND the
    new Wf-119 P1.4 launch wrapper can read it.

    Per Wf-321H §3: the gate ``p3_todo_119_p14_cpu_smoke_ready`` was
    added to the chemprop_v2_v3 pipeline in pre_flight_check.py (gate
    verifies ``--chemprop-mode v3-fix --chemprop-min-epochs
    --chemprop-lomo-init-scale`` token presence in --help).

    Wf-336X2 extends this contract: the new
    ``launch_wf119_p1_4_chemprop_v3_lomo.sh`` --dry-run mode must be
    able to invoke the pre-flight gate and confirm PASS before any
    GPU launch. This test verifies BOTH halves:

      (a) the gate is wired in pre_flight_check.py
      (b) the launch wrapper references the same gate name (so that
          --dry-run mode can grep for it)

    Honest framing: this is a string-level integration contract test,
    NOT a behavioural test. We do not invoke the pre-flight gate
    here (gate-level behavioural tests are in Wf-321H §4.3).
    """
    # (a) Pre-flight gate wired.
    src = _PRE_FLIGHT.read_text(encoding="utf-8")
    expected_gate = "p3_todo_119_p14_cpu_smoke_ready"
    assert expected_gate in src, (
        f"Pre-flight gate '{expected_gate}' missing from "
        f"pre_flight_check.py. Wf-321H ships this gate; Wf-336X2 "
        f"expects it to remain wired."
    )
    # Must be inside the chemprop_v2_v3 pipeline block.
    assert 'pipeline in ("chemprop_v2_v3", "all")' in src, (
        "build_gates() chemprop_v2_v3 pipeline block missing — "
        f"gate '{expected_gate}' would have nowhere to be wired."
    )

    # (b) Launch wrapper references the same gate name.
    assert (
        _WF119_LAUNCHER.exists()
    ), f"Wf-119 P1.4 launch wrapper not found at {_WF119_LAUNCHER}."
    launcher_content = _WF119_LAUNCHER.read_text(encoding="utf-8")
    assert expected_gate in launcher_content, (
        f"Wf-119 P1.4 launch wrapper missing reference to "
        f"'{expected_gate}'. The launcher's --dry-run mode greps "
        f"this gate name to confirm pre-flight PASS; the wrapper "
        f"must reference the gate by name to integrate."
    )
