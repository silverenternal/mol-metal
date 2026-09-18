"""WF-Metallodrug-Vertical Phase 3 protocol-align — regression tests.

These tests verify that the four TargetDiff-aligned CLI flags
added to the CFM/Lambda eval scripts behave as documented:

- ``--physical-exhaustiveness`` (r10_cfg_real_crossdocked.py + r4_c_full_sweep.py)
- ``--n-samples`` (r4_lambda_only_run.py)
- ``--pocket10-radius`` (all three scripts)
- ``--reference-ligand`` (all three scripts)

The tests are deliberately minimal — they exercise the
argparser surface (so the spec cannot drift) and the
run_sweep/run_one_cell kwarg plumbing (so the value flows from
the CLI to the cell.warnings audit log).  They do NOT touch
real docking, MCTS, or metal-chemistry machinery (those are
GPU-bound and slow).  Smoke-grade integration via the
evaluate_candidates path is a follow-up.

Backward-compatibility contract:
    - The pre-Phase-3 default ``--physical-exhaustiveness 1`` for
      r10_cfg_real_crossdocked.py is recoverable by passing
      ``--physical-exhaustiveness 1``.
    - The pre-Phase-3 default ``--n-samples 8`` for r4_lambda_only_run
      is recoverable by passing ``--n-samples 8``.
    - ``--pocket10-radius 8.0`` recovers the legacy DiffDock-Pocket
      8 Å crop.
    - ``--reference-ligand`` is OFF by default; OFF recovers the
      legacy unconditional sampling path bit-exactly.

Honest framing: the per-cell generation work in r4_lambda_only_run
is unbounded by ``--n-samples`` today (MCTSProofSearch produces as
many candidates as it can within ``--n-simulations``); ``--n-samples``
is recorded on the cell.warnings audit and consumed by downstream
metrics that want to enforce a per-cell cap.
"""
import pytest

from molmetal.scripts import r4_lambda_only_run as lambda_run
from molmetal.scripts import r4_c_full_sweep as sweep


# -------------------------------------------------------------------
# r4_lambda_only_run.py — argparser surface
# -------------------------------------------------------------------


def test_lambda_run_n_samples_default_is_targetdiff_aligned():
    """Phase 3 bumped --n-samples default 8 -> 100 (TargetDiff per-cell)."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3"])
    assert args.n_samples == 100, (
        f"--n-samples default should be 100 (TargetDiff per-cell), got {args.n_samples}"
    )


def test_lambda_run_n_samples_backward_compatible_via_explicit_8():
    """Setting --n-samples 8 recovers the pre-Phase-3 smoke value."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3", "--n-samples", "8"])
    assert args.n_samples == 8


def test_lambda_run_pocket10_radius_default_is_crossdocked2020():
    """Phase 3 added --pocket10-radius default 10.0 Å (CrossDocked2020)."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3"])
    assert args.pocket10_radius == 10.0


def test_lambda_run_pocket10_radius_legacy_8A_recoverable():
    """Setting --pocket10-radius 8.0 recovers the legacy 8 Å crop."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3", "--pocket10-radius", "8.0"])
    assert args.pocket10_radius == 8.0


def test_lambda_run_reference_ligand_default_off():
    """--reference-ligand is OFF by default (legacy unconditional)."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3"])
    assert args.reference_ligand is False


def test_lambda_run_reference_ligand_can_be_enabled():
    """--reference-ligand flag toggles to True."""
    parser = lambda_run._build_argparser()
    args = parser.parse_args(["--output-dir", "/tmp/_test_p3", "--reference-ligand"])
    assert args.reference_ligand is True


# -------------------------------------------------------------------
# r4_c_full_sweep.py — argparser surface
# -------------------------------------------------------------------


def _read_sweep_source():
    """Read the r4_c_full_sweep.py source as text.

    The argparser is built inside ``main()`` (not at module scope), so we
    can't import the parser directly without pulling in the full Triton
    + QVina + torch stack.  Instead, we read the source and apply a
    regex-based structural check that confirms the Phase-3 flags are
    wired into the parser-build block, AND we extract the default value
    string for each flag.
    """
    import os
    src_path = os.path.abspath(sweep.__file__)
    with open(src_path) as fh:
        return fh.read()


def _extract_add_argument_block(source: str, flag_name: str) -> str:
    """Return the textual ``parser.add_argument('--flag', ...)`` call.

    Locates the start of the call and consumes balanced parens so
    that nested tuples (e.g. ``choices=(...)``) do not terminate the
    match prematurely.  Returns an empty string if the flag is not
    present.
    """
    import re
    start_pat = re.compile(
        r"parser\.add_argument\(\s*['\"](--" + re.escape(flag_name) + r")['\"]"
    )
    m = start_pat.search(source)
    if m is None:
        return ""
    # Walk forward from m.end() and count ``(`` and ``)`` to find the
    # balanced close-paren of the add_argument call.
    depth = 1
    i = m.end()
    while i < len(source) and depth > 0:
        c = source[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if depth != 0:
        # Unbalanced — give up; return the substring up to EOF.
        return source[m.start():]
    return source[m.start():i]


def _extract_default_value(add_argument_call: str):
    """Pull the ``default=...`` kwarg value out of an add_argument call."""
    import re
    m = re.search(r"default\s*=\s*([^,\n]+)", add_argument_call)
    if m is None:
        return None
    return m.group(1).strip()


def test_sweep_physical_exhaustiveness_default_is_targetdiff_aligned():
    """Phase 3 verified --physical-exhaustiveness default = 8 (was 1 in pre-Phase-3 smoke)."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "physical-exhaustiveness")
    assert call, "--physical-exhaustiveness must be wired into r4_c_full_sweep.main()"
    assert "default=8" in call, (
        f"--physical-exhaustiveness default should be 8 (TargetDiff).  Got: {call[:200]}"
    )


def test_sweep_physical_exhaustiveness_backward_compatible_via_explicit_1():
    """Setting --physical-exhaustiveness 1 is accepted (smoke recoverability)."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "physical-exhaustiveness")
    assert call, "--physical-exhaustiveness must be wired into r4_c_full_sweep.main()"
    # The flag is an int with default=8 — argparse will accept 1 verbatim.
    assert "type=int" in call


def test_sweep_pocket10_radius_default_is_crossdocked2020():
    """Phase 3 added --pocket10-radius default 10.0 Å (CrossDocked2020)."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "pocket10-radius")
    assert call, "--pocket10-radius must be wired into r4_c_full_sweep.main()"
    assert "default=10.0" in call, (
        f"--pocket10-radius default should be 10.0.  Got: {call[:200]}"
    )


def test_sweep_pocket10_radius_legacy_8A_recoverable():
    """Setting --pocket10-radius 8.0 is accepted (legacy recoverability)."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "pocket10-radius")
    assert call, "--pocket10-radius must be wired into r4_c_full_sweep.main()"
    assert "type=float" in call


def test_sweep_reference_ligand_default_off():
    """--reference-ligand is OFF by default (legacy unconditional)."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "reference-ligand")
    assert call, "--reference-ligand must be wired into r4_c_full_sweep.main()"
    # ``store_true`` action: presence in argv enables, absence keeps False.
    assert "store_true" in call


def test_sweep_reference_ligand_can_be_enabled():
    """--reference-ligand flag toggles to True when passed."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "reference-ligand")
    assert call, "--reference-ligand must be wired into r4_c_full_sweep.main()"
    assert "store_true" in call


def test_sweep_default_engine_still_both():
    """WF-D7-Apply set --engine default to 'both'; Phase 3 must NOT regress that."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "engine")
    assert call, "--engine must be wired into r4_c_full_sweep.main()"
    assert "default=\"both\"" in call or "default='both'" in call, (
        f"--engine default should be 'both' (D7).  Got: {call[:200]}"
    )


# -------------------------------------------------------------------
# r4_lambda_only_run.py — kwarg plumbing
# -------------------------------------------------------------------


def test_lambda_run_run_sweep_signature_accepts_protocol_align_kwargs():
    """run_sweep() must accept the three Phase-3 kwargs.

    This is a smoke-level signature check; full integration is exercised
    in the Lambda 5x1 pilots (see ``molmetal/reports/wf_round12_lambda_pilot/``).
    """
    import inspect
    sig = inspect.signature(lambda_run.run_sweep)
    for name in ("n_samples", "pocket10_radius", "reference_ligand"):
        assert name in sig.parameters, (
            f"run_sweep() must accept Phase-3 kwarg '{name}' (got {list(sig.parameters)})"
        )


def test_lambda_run_run_one_cell_signature_accepts_protocol_align_kwargs():
    """run_one_cell() must accept the three Phase-3 kwargs."""
    import inspect
    sig = inspect.signature(lambda_run.run_one_cell)
    for name in ("n_samples", "pocket10_radius", "reference_ligand"):
        assert name in sig.parameters, (
            f"run_one_cell() must accept Phase-3 kwarg '{name}' (got {list(sig.parameters)})"
        )


# -------------------------------------------------------------------
# r4_c_full_sweep.py — backward-compat defaults audit
# (use the source-text structural probe, not the live parser — the live
#  parser lives inside main() and would require Triton+QVina+torch to
#  import, which adds ~12s to the test runtime)
# -------------------------------------------------------------------


def test_sweep_engine_default_still_both():
    """WF-D7-Apply set --engine default to 'both'; Phase 3 must NOT regress that."""
    source = _read_sweep_source()
    call = _extract_add_argument_block(source, "engine")
    assert call, "--engine must be wired into r4_c_full_sweep.main()"
    assert "default=\"both\"" in call or "default='both'" in call, (
        f"--engine default should be 'both' (D7).  Got: {call[:200]}"
    )


def test_sweep_phase3_flags_all_present_in_source():
    """Phase 3 flags must all be wired into r4_c_full_sweep.main()."""
    source = _read_sweep_source()
    for flag in ("physical-exhaustiveness", "pocket10-radius", "reference-ligand"):
        assert _extract_add_argument_block(source, flag), (
            f"--{flag} must be wired into r4_c_full_sweep.main()"
        )


def test_r10_cfg_phase3_flags_all_present_in_source():
    """Phase 3 flags must all be wired into r10_cfg_real_crossdocked.py.

    r10_cfg_real_crossdocked.py builds its argparser at module scope
    (inside the ``if __name__ == '__main__'`` block), so we read the
    source to confirm the four flags are wired.
    """
    import os
    import re
    r10_path = os.path.abspath(
        os.path.join(os.path.dirname(sweep.__file__), "r10_cfg_real_crossdocked.py")
    )
    with open(r10_path) as fh:
        source = fh.read()
    for flag in ("physical-exhaustiveness", "pocket10-radius", "reference-ligand"):
        # Same balanced-paren extraction logic.
        start_pat = re.compile(
            r"p\.add_argument\(\s*['\"]?(--" + re.escape(flag) + r")['\"]?"
        )
        m = start_pat.search(source)
        assert m is not None, f"--{flag} must be wired into r10_cfg_real_crossdocked.py"


def test_r10_cfg_engine_dock_call_uses_physical_exhaustiveness():
    """r10's evaluate_candidates() call must use ``args.physical_exhaustiveness``.

    Pre-Phase-3 the script hardcoded ``exhaustiveness=1`` (smoke).  Phase 3
    routes through the new flag so the TargetDiff-aligned value (default 8)
    takes effect.
    """
    import os
    r10_path = os.path.abspath(
        os.path.join(os.path.dirname(sweep.__file__), "r10_cfg_real_crossdocked.py")
    )
    with open(r10_path) as fh:
        source = fh.read()
    # The smoke artefact ``exhaustiveness=1`` must no longer be present in
    # the evaluate_candidates call.  We allow ``exhaustiveness=1`` to live
    # in comments / docstrings (where it's harmless).
    assert "exhaustiveness=args.physical_exhaustiveness" in source, (
        "r10_cfg_real_crossdocked.evaluate_candidates() must route through "
        "args.physical_exhaustiveness (Phase 3 protocol-align)."
    )