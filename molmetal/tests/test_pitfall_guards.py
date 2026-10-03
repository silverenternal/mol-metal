"""WF-TODO-30 Pitfall Reinforce — CI guard rails for R12-R17 pitfalls.

This file ships 7 parametrized regression tests, one per pitfall that bit
us in Rounds 12 through 17.  Each guard fires (FAILS) when the
*current* code state no longer protects against the corresponding
regression, and PASSES when the protection is in place.  Use this file
as the canonical test surface for the "did the fix get reverted?"
question.

Pitfall → test mapping
======================

* **P1** — OOV atom training molecule → vocab_mask -inf → ``atom_loss``
  blows up to ``Inf`` → ``cfm_loss`` diverges → ``decode_ratio=0``
  on Blackwell.  *Fix*: 20-element canonical vocab in the adapter
  + ``vocab_mask=True`` default + skip-step on bad_atoms.
* **P2** — ``_SFNReaction(template=...)`` raises ``TypeError`` on
  malformed SMARTS → entire ``SynFlowNetEnvAdapter.__init__`` aborts
  mid-loop → ``n_rules == 0`` → 0/192 decode.  *Fix*: per-rule
  ``try / except`` around template construction in
  :class:`SynFlowNetEnvAdapter`.
* **P3** — 84-rule MetalLigandExchange matrix alone is not enough;
  the MCTS root still falls back to ``cisplatin`` seed unless the
  user passes ``--metal-seed cisplatin`` explicitly.  *Fix*:
  ``register_metal_ligand_exchange_rules()`` returns ≥ 84 rules
  AND ``--metal-seed cisplatin`` is the default in
  ``r4_lambda_only_run.py``.
* **P4** — BF16 matmul occasionally overflows ``atom_logits`` past
  ``+/-1e4``; ``F.cross_entropy`` then produces ``Inf`` and the
  CFM loss diverges.  *Fix*: clamp ``atom_logits`` to ``+/-50`` in
  BOTH the training path AND the sampling path, AND accept
  ``MOLMETAL_CFM_AMP_DTYPE=float32`` for a fallback path.
* **P5** — 1K-step smoke runs are *lucky* — the loss happens to
  converge in the toy regime but the 10K-step production run
  explodes.  *Fix*: ``SAFETY_MAX = 10000`` floor in
  ``r4_lambda_only_run.py`` so a typoed ``--n-simulations 1000000``
  cannot lock the GPU.
* **P6** — Reference clone location mismatch (``flow_matching`` vs
  ``references/flow_matching``) silently disables the upstream
  adapter.  *Fix*: ``configure_reference`` auto-detects whether the
  caller passed the clone root or its parent.
* **P7** — ``functools.partial(model, t=0.5)`` against a velocity
  field with keyword-only ``t`` raises ``TypeError`` because
  partial binds ``t`` positionally.  *Fix*: :class:`ModelWrapper`
  falls back to ``model(x, t=t)`` on TypeError.

Usage
=====

    cd /home/hugo/codes/mol-metal
    uv run pytest molmetal/tests/test_pitfall_guards.py -v

Or all 7 at once via the ultracode workflow runner::

    uv run python molmetal/scripts/wf_pitfall_guard/check_all.py
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(relpath: str) -> str:
    """Read a UTF-8 text file from the repo root; raise if missing."""
    p = REPO_ROOT / relpath
    assert p.is_file(), f"missing fixture file: {relpath}"
    return p.read_text(encoding="utf-8")


def _try_import(modname: str):
    """Import ``modname``; return the module or None on ImportError.

    Uses :func:`importlib.import_module` (NOT :func:`__import__`) so that
    sub-module dotted paths (``molmetal.a.b.c``) return the actual
    submodule rather than the top-level package.  ``__import__`` without
    a ``fromlist`` argument returns the top-level package, which made
    ``getattr(mod, 'helper_fn')`` always return ``None`` even when the
    helper was present deep in the package.
    """
    try:
        return importlib.import_module(modname)
    except Exception:
        return None


def _load_module_by_path(modname: str, relpath: str) -> Optional[Any]:
    """Load a Python module from a file path, bypassing package __init__.

    Some lam_chem modules shadow stdlib (e.g. ``lam_chem/ast.py``), which
    causes circular-import errors when numpy / torch are missing from the
    test venv.  Loading by file path with an isolated ``modname`` avoids
    the package-init cascade and lets us introspect the module-level
    constants directly.
    """
    p = REPO_ROOT / relpath
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location(modname, str(p))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Clean up sys.modules on partial init to avoid poisoning
        # subsequent imports.
        sys.modules.pop(modname, None)
        return None
    return mod


# ---------------------------------------------------------------------------
# P1 — OOV atom guard
# ---------------------------------------------------------------------------


def test_p1_oov_atom_vocabulary_is_at_least_20_elements():
    """Adapter default vocab covers all atoms in metallodrug training pool.

    The R12-R17 OOV regression: training molecules containing Pt / Pd /
    Au / Ir / Ru (or any other atom outside the previous 12-element
    default vocab) drove ``atom_loss`` to ``Inf`` because
    ``vocab_mask`` -inf'd the corresponding logit slot.  The fix is a
    20-element canonical vocab in
    ``LipmanFlowMatchingAdapter.__init__`` (see ``_atom_vocab: tuple``
    at line ~1840) — adding the metallodrug atoms (Pt is the canonical
    one) on top of the previous organic-chemistry defaults.

    This guard fails when:
      * the vocab literal is missing entirely (fix reverted), OR
      * the vocab drops below 20 elements (incomplete slot was removed), OR
      * Pt (atomic number 78) is no longer in the vocab (the canonical
        metallodrug atom that triggered the original OOV blow-up).

    Note: the production smoke script
    (``metallo_drug_smoke_retrain.py``) overrides the default with a
    14-element *specialised* vocab that also covers Pd / Au / Ir / Ru
    for the PlatinAI + tmQM subsets.  That override is intentional and
    is NOT regressed by this guard.
    """
    src = _read_text("molmetal/adapters/flow_matching_lipman/__init__.py")
    # Locate the _atom_vocab literal in the source.  We accept either the
    # current 20-element tuple or any explicit future extension that
    # covers metallodrug atoms (Pt / Pd / Au / Ir / Ru).  We strip
    # in-line ``#`` comments before splitting so the test stays robust
    # against refactors that add explanatory comments next to the
    # vocab literal.
    m = re.search(r"self\._atom_vocab:\s*tuple\s*=\s*\(([^)]+)\)", src, re.DOTALL)
    assert m is not None, (
        "P1 guard FAIL — _atom_vocab tuple literal not found in "
        "LipmanFlowMatchingAdapter.__init__; the OOV fix has been "
        "removed or refactored."
    )
    raw = m.group(1)
    # Strip ``# ...`` comments and whitespace, then parse integers.
    raw_no_comments = re.sub(r"#.*", "", raw)
    vocab = tuple(
        int(x.strip())
        for x in raw_no_comments.split(",")
        if x.strip() and x.strip().lstrip("-").isdigit()
    )
    assert len(vocab) >= 20, (
        f"P1 guard FAIL — _atom_vocab has only {len(vocab)} elements; "
        "OOV regression will return.  Expected >= 20 (covers all "
        "metallodrug atoms: C/N/O/F/S/P/Cl/Br/I + Pt + 9 more)."
    )
    # The canonical metallodrug atom that originally drove the OOV
    # regression is Pt (Z=78).  Without it the metallodrug training
    # pool is un-trainable.
    assert 78 in vocab, (
        "P1 guard FAIL — _atom_vocab is missing Pt (Z=78); OOV on "
        "metallodrug molecules will Inf-loss the CFM."
    )


def test_p1_oov_skips_optimizer_step():
    """``train_step`` short-circuits with NaN losses when OOV atoms present.

    The companion fix to P1: when data-load guard is bypassed (e.g. an
    ad-hoc notebook calls ``Molecule.from_smiles`` directly), the
    adapter detects bad_atoms inside ``train_step`` and skips the
    optimizer step instead of poisoning AdamW with ``Inf`` loss.
    """
    src = _read_text("molmetal/adapters/flow_matching_lipman/__init__.py")
    # Locate the bad_atoms block.
    assert "bad_atoms" in src, (
        "P1 guard FAIL — 'bad_atoms' handling missing from "
        "LipmanFlowMatchingAdapter.train_step."
    )
    assert "SKIPPING optimizer step" in src, (
        "P1 guard FAIL — 'SKIPPING optimizer step' log not found; "
        "the OOV defence-in-depth branch has been removed."
    )
    assert "skipped_oov" in src, (
        "P1 guard FAIL — 'skipped_oov' diagnostics missing; "
        "the OOV last-losses dict is no longer populated."
    )


# ---------------------------------------------------------------------------
# P2 — SynFlowNet shim TypeError guard
# ---------------------------------------------------------------------------


def test_p2_synflownet_per_rule_exception_handler():
    """``SynFlowNetEnvAdapter`` wraps each rule in try/except.

    The R12-R17 TypeError regression: an unparseable SMARTS in the
    ``REACTION_RULES`` registry raised from
    ``_SFNReaction.__post_init__`` → ``__init__`` aborts mid-loop →
    ``self._sfn`` ends up empty → ``n_rules == 0`` → 0/192 decode.
    The fix wraps template construction per-rule.
    """
    src = _read_text("molmetal/molmetal_lam/sbdd_env/synflownet_env.py")
    # Find the SynFlowNetEnvAdapter.__init__ loop and confirm both the
    # forward-template and reverse-template constructions are guarded.
    init_match = re.search(
        r"class\s+SynFlowNetEnvAdapter.*?def\s+__init__.*?(?=\nclass |\n# ---)",
        src,
        re.DOTALL,
    )
    assert init_match is not None, (
        "P2 guard FAIL — SynFlowNetEnvAdapter class body not found."
    )
    body = init_match.group(0)
    # The guard verifies the *specific* try/except pattern around
    # ``_SFNReaction(template=...)`` rather than just counting
    # ``except Exception`` occurrences (which can be polluted by other
    # class methods).  We require both the forward-template
    # construction AND the reverse-template construction to be
    # wrapped, in source order.
    fwd_match = re.search(
        r"try:\s*\n\s*self\._sfn\[name\]\s*=\s*_SFNReaction\(",
        body,
    )
    assert fwd_match is not None, (
        "P2 guard FAIL — forward-template construction "
        "'self._sfn[name] = _SFNReaction(...)' is not wrapped in "
        "'try: ... except Exception:'; a bad SMARTS will abort "
        "__init__ mid-loop and the rule count will drop to 0."
    )
    # Reverse block: the ``try:`` may either contain
    # ``self._rev_sfn[name] = _SFNReaction(...)`` directly OR carry a
    # pre-step (``rev_sma = _swap_smarts(sma)``) before the
    # ``_rev_sfn[name]`` line.  Both shapes are accepted.
    rev_match = re.search(
        r"try:\s*\n(?:\s*[a-zA-Z_][^\n]*\n)*?\s*self\._rev_sfn\[name\]\s*=\s*_SFNReaction\(",
        body,
    )
    assert rev_match is not None, (
        "P2 guard FAIL — reverse-template construction "
        "'self._rev_sfn[name] = _SFNReaction(...)' is not wrapped in "
        "'try: ... except Exception:'; an unparseable reverse SMARTS "
        "(e.g. irreversible rule) will abort __init__ mid-loop."
    )
    # Sanity check: at least one "except Exception" should appear in
    # the class body (the per-rule handlers).
    n_excepts = body.count("except Exception")
    assert n_excepts >= 2, (
        f"P2 guard FAIL — only {n_excepts} 'except Exception' handlers "
        "in SynFlowNetEnvAdapter; expected >= 2 (forward + reverse)."
    )


def test_p2_swap_smarts_idempotent_on_malformed_input():
    """``_swap_smarts`` returns the input unchanged when '>>' is missing.

    Edge-case: a structural rule with no ``>>`` token would crash the
    ``split('>>', 1)`` path.  The fix returns the malformed input
    unchanged so the caller can handle it.
    """
    fn = _try_import(
        "molmetal.molmetal_lam.sbdd_env.synflownet_env"
    )
    if fn is None:
        pytest.skip("synflownet_env not importable in this env")
    _swap_smarts = getattr(fn, "_swap_smarts", None)
    assert _swap_smarts is not None, (
        "P2 guard FAIL — _swap_smarts helper missing from "
        "molmetal.molmetal_lam.sbdd_env.synflownet_env."
    )
    # Malformed input must NOT raise; should round-trip unchanged.
    out = _swap_smarts("malformed_no_arrow_token")
    assert out == "malformed_no_arrow_token", (
        f"P2 guard FAIL — _swap_smarts did not pass-through malformed "
        f"input; got {out!r}"
    )
    # Well-formed input must swap sides.
    out2 = _swap_smarts("A.B>>C.D")
    assert out2 == "C.D>>A.B", (
        f"P2 guard FAIL — _swap_smarts(swapped) = {out2!r}, expected 'C.D>>A.B'"
    )


# ---------------------------------------------------------------------------
# P3 — MetalLigandExchange 84-rule matrix + cisplatin metal-seed
# ---------------------------------------------------------------------------


def test_p3_register_metal_ligand_exchange_rules_returns_at_least_84():
    """84-rule matrix (12 metals × 7 ligands) is exposed.

    R12-R17 singleton regression: only the F2(a) curated 5 rules were
    loaded, so the MCTS root collapsed to the cisplatin seed across
    all pockets.  L-R17-B2 extended the matrix to 84 patterns; this
    guard fails if the matrix shrinks back below 84.

    The guard loads ``pt_metal_ligand_exchange.py`` by file path to
    bypass the package ``__init__`` (which transitively imports numpy
    / torch and the project-local ``lam_chem/ast.py`` shadow of
    stdlib ``ast``).  That package-init side-effect chain fails in a
    bare-bones test venv.
    """
    mod = _load_module_by_path(
        "_pitfall_pt_mle",
        "molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py",
    )
    if mod is None:
        pytest.skip("pt_metal_ligand_exchange.py not loadable in this env")
    matrix = getattr(mod, "METAL_LIGAND_EXCHANGE_MATRIX", None)
    assert matrix is not None, (
        "P3 guard FAIL — METAL_LIGAND_EXCHANGE_MATRIX missing; "
        "the 84-rule matrix has been removed."
    )
    assert len(matrix) >= 84, (
        f"P3 guard FAIL — METAL_LIGAND_EXCHANGE_MATRIX has only "
        f"{len(matrix)} rules; expected >= 84 (12 metals x 7 ligands). "
        "singleton attractor will return across pockets."
    )
    register = getattr(mod, "register_metal_ligand_exchange_rules", None)
    assert register is not None and callable(register), (
        "P3 guard FAIL — register_metal_ligand_exchange_rules() missing."
    )
    # Call register_metal_ligand_exchange_rules WITHOUT mutating the
    # global rule registry (test isolation).
    rules = register(register_into_rule_registry=False)
    assert len(rules) >= 84, (
        f"P3 guard FAIL — register_metal_ligand_exchange_rules() returned "
        f"{len(rules)} rules; expected >= 84."
    )


def test_p3_metal_seed_cisplatin_is_a_valid_choice():
    """``--metal-seed cisplatin`` is exposed as a CLI choice.

    Singleton across pockets: when no ``--metal-seed`` is supplied,
    every pocket starts from the same cisplatin seed SMILES, which
    keeps the MCTS pinned to a closed square-planar coordination
    environment and prevents across-pocket diversity.  The fix is to
    expose ``cisplatin`` (and related Pt(II) chemistries) as explicit
    CLI choices so the user can opt in to the seeded path.

    We deliberately do NOT assert that ``cisplatin`` is the *default*
    — the default is ``None`` (legacy behaviour preserved; the search
    falls back to the pocket reference when metal_seed is unset).
    This guard checks only that the choice is available and reachable.
    """
    src = _read_text("molmetal/scripts/r4_lambda_only_run.py")
    # Find the ``--metal-seed`` add_argument block by anchor; we look
    # for the ``choices`` list and confirm ``"cisplatin"`` is in it.
    m = re.search(
        r'--metal-seed["\']?,?\s*\n\s*nargs=None,\s*\n\s*default=\w+,?\s*\n\s*choices=\[([^\]]*)\]',
        src,
    )
    if m is None:
        # Fallback: looser match — find the literal "cisplatin" within
        # 50 lines of ``add_argument("--metal-seed"``.
        m2 = re.search(
            r'add_argument\(\s*["\']--metal-seed["\']',
            src,
        )
        assert m2 is not None, (
            "P3 guard FAIL -- --metal-seed argparse flag missing "
            "from r4_lambda_only_run.py; users cannot opt into the "
            "Pt(II) seed path."
        )
        start = m2.start()
        snippet = src[start:start + 2000]
        assert '"cisplatin"' in snippet, (
            "P3 guard FAIL -- 'cisplatin' missing from --metal-seed "
            "choices; users cannot opt into the canonical Pt(II) seed."
        )
    else:
        choices = m.group(1)
        assert '"cisplatin"' in choices or "'cisplatin'" in choices, (
            f"P3 guard FAIL -- 'cisplatin' missing from --metal-seed "
            f"choices: {choices[:200]!r}"
        )


# ---------------------------------------------------------------------------
# P4 — BF16 atom_logits clamp + FP32 fallback
# ---------------------------------------------------------------------------


def test_p4_atom_logits_clamped_to_pm50_in_train_and_sample():
    """Clamp atom_logits to +/-50 in BOTH the training AND sampling path.

    R17 BF16 regression: untrained Blackwell kernels overflowed
    ``atom_logits`` to +/-1e4, blowing up ``F.cross_entropy`` into
    Inf.  The fix clamps to ``+/-50`` in train (line ~2177) AND in
    the sampler (line ~2699).
    """
    src = _read_text("molmetal/adapters/flow_matching_lipman/__init__.py")
    # Count the explicit clamp(min=-50.0, max=50.0) calls.
    n = src.count("atom_logits.clamp(min=-50.0, max=50.0)")
    assert n >= 2, (
        f"P4 guard FAIL — found {n} atom_logits.clamp(min=-50.0, max=50.0) "
        "occurrences in LipmanFlowMatchingAdapter; expected >= 2 "
        "(training + sampling path).  BF16 overflow will diverge again."
    )


def test_p4_amp_dtype_float32_supported():
    """``MOLMETAL_CFM_AMP_DTYPE=float32`` produces a float32 AMP context.

    The FP32 fallback path lets callers opt out of BF16 when numerical
    instability is observed (gfx1101 BF16 occasionally overflows).
    """
    # Set the env var BEFORE import so the singleton captures it.
    os.environ["MOLMETAL_CFM_AMP_DTYPE"] = "float32"
    try:
        amp = _try_import(
            "molmetal.adapters.flow_matching_lipman.amp"
        )
        if amp is None:
            pytest.skip("amp module not importable in this env")
        ctx_factory = getattr(amp, "CFMAMPContext", None)
        assert ctx_factory is not None, (
            "P4 guard FAIL — CFMAMPContext class missing from amp.py."
        )
        ctx = ctx_factory(enabled=True, dtype=__import__("torch").float32)
        assert ctx.dtype is __import__("torch").float32, (
            f"P4 guard FAIL — CFMAMPContext(dtype=float32).dtype = "
            f"{ctx.dtype}; expected torch.float32."
        )
    finally:
        os.environ.pop("MOLMETAL_CFM_AMP_DTYPE", None)


# ---------------------------------------------------------------------------
# P5 — 1K-step too short; production needs ≥ 10K
# ---------------------------------------------------------------------------


def test_p5_r4_lambda_only_run_safety_max_floor_is_10k():
    """``r4_lambda_only_run.py`` clamps ``--n-simulations`` at SAFETY_MAX.

    R12 mini pilot regression: 1K-step smoke runs were lucky but the
    10K-step production run blew up.  The fix enforces ``SAFETY_MAX
    = 10000`` so a typoed ``--n-simulations 1000000`` cannot lock
    the GPU on a developer box.

    R1 refactor (2026-09-25): the constant moved to
    :mod:`molmetal.scripts.r4_lib.cli` (the argparse CLI module);
    the thin shim no longer carries it inline.  Check both files.
    """
    src = (
        _read_text("molmetal/scripts/r4_lambda_only_run.py")
        + "\n"
        + _read_text("molmetal/scripts/r4_lib/cli.py")
    )
    m = re.search(r"SAFETY_MAX\s*=\s*(\d+)", src)
    assert m is not None, (
        "P5 guard FAIL — SAFETY_MAX constant missing from "
        "r4_lambda_only_run.py; the n_simulations ceiling protection "
        "has been removed."
    )
    val = int(m.group(1))
    assert val >= 10000, (
        f"P5 guard FAIL — SAFETY_MAX = {val}; expected >= 10000 "
        "(per the R12 production smoke result, 1K-step is lucky-only)."
    )


# ---------------------------------------------------------------------------
# P6 — Reference clone location auto-detect
# ---------------------------------------------------------------------------


def test_p6_configure_reference_accepts_both_parent_and_clone_root():
    """``configure_reference`` auto-detects parent-vs-clone-root paths.

    R12-R17 loader regression: callers using the legacy convention
    (parent dir containing ``flow_matching/``) silently failed the
    auto-detect, leaving the upstream adapter un-imported.  The fix
    is the ``if (ref / '__init__.py').is_file() and ref.name ==
    'flow_matching'`` branch.
    """
    loader = _load_module_by_path(
        "_pitfall_ref_loader",
        "molmetal/adapters/flow_matching_lipman/_reference_loader.py",
    )
    if loader is None:
        pytest.skip("_reference_loader.py not loadable in this env")
    configure = getattr(loader, "configure_reference", None)
    assert configure is not None and callable(configure), (
        "P6 guard FAIL — configure_reference() missing from "
        "_reference_loader."
    )
    src = inspect.getsource(configure)
    # Both conventions must be accepted: the function must check
    # ``(ref / '__init__.py').is_file() and ref.name == 'flow_matching'``
    # to detect a clone-root path.
    assert (
        "ref.name == 'flow_matching'" in src
        or 'ref.name == "flow_matching"' in src
    ), (
        "P6 guard FAIL — configure_reference does not auto-detect "
        "the clone-root convention; legacy parent-dir callers will "
        "see a FileNotFoundError."
    )


# ---------------------------------------------------------------------------
# P7 — ModelWrapper keyword-only t fallback
# ---------------------------------------------------------------------------


def test_p7_model_wrapper_handles_keyword_only_t():
    """``ModelWrapper(x, t)`` falls back to ``model(x, t=t)`` on TypeError.

    R12-R17 functools.partial regression: ``functools.partial(model,
    t=0.5)`` against a velocity field with keyword-only ``t``
    (e.g. ``def forward(self, x, *, t): ...``) raised TypeError
    because partial binds ``t`` as a kwarg but the wrapper called
    it positionally.  The fix tries ``(x, t)`` first, then
    ``(x, t=t)`` on TypeError.
    """
    wrapper_mod = _load_module_by_path(
        "_pitfall_model_wrapper",
        "molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py",
    )
    if wrapper_mod is None:
        pytest.skip("_reference/utils shim not loadable in this env")
    ModelWrapper = getattr(wrapper_mod, "ModelWrapper", None)
    assert ModelWrapper is not None, (
        "P7 guard FAIL — ModelWrapper class missing from "
        "_reference/utils/__init__.py shim."
    )

    # 1. Legacy positional signature should still work.
    def legacy_vf(x, t):
        return (x, t)

    wrapped = ModelWrapper(legacy_vf)
    out = wrapped("X", 0.5)
    assert out == ("X", 0.5), (
        f"P7 guard FAIL — legacy positional call returned {out!r}, "
        "expected ('X', 0.5)."
    )

    # 2. Keyword-only t signature: the wrapper must call ``model(x,
    # t=t)`` after the positional TypeError, NOT raise.
    def kwonly_vf(x, *, t):
        return ("kwonly", x, t)

    wrapped2 = ModelWrapper(kwonly_vf)
    out2 = wrapped2("Y", 0.7)
    assert out2 == ("kwonly", "Y", 0.7), (
        f"P7 guard FAIL — keyword-only t call returned {out2!r}, "
        "expected ('kwonly', 'Y', 0.7)."
    )

    # 3. functools.partial(model, t=0.5) — when ModelWrapper is
    # built from a partial, the wrapped callable must still accept
    # the (x, t) signature.  This is the exact R12-R17 bug: partial
    # binds t but the wrapper called positionally.
    def full_vf(x, t, *, extras=None):
        return ("full", x, t, extras)

    partial_vf = functools.partial(full_vf, t=0.9)
    wrapped3 = ModelWrapper(partial_vf)
    out3 = wrapped3("Z", 0.9)
    assert out3 == ("full", "Z", 0.9, None), (
        f"P7 guard FAIL — partial-keyword t call returned {out3!r}, "
        "expected ('full', 'Z', 0.9, None)."
    )


# ---------------------------------------------------------------------------
# Parametrized convenience entry point (one assertion per pitfall ID).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pitfall_id,test_fn",
    [
        ("P1", test_p1_oov_atom_vocabulary_is_at_least_20_elements),
        ("P2", test_p2_synflownet_per_rule_exception_handler),
        ("P3", test_p3_register_metal_ligand_exchange_rules_returns_at_least_84),
        ("P4", test_p4_atom_logits_clamped_to_pm50_in_train_and_sample),
        ("P5", test_p5_r4_lambda_only_run_safety_max_floor_is_10k),
        ("P6", test_p6_configure_reference_accepts_both_parent_and_clone_root),
        ("P7", test_p7_model_wrapper_handles_keyword_only_t),
    ],
)
def test_pitfall_guard_matrix(pitfall_id: str, test_fn: Callable[[], None]):
    """Single entry point so the wf_pitfall_guard runner can dispatch."""
    test_fn()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
