"""R17 / C4 — legacy compatibility regression tests.

R17 shipped a wave of small, surgical fixes (TODO-33 vocab expansion,
defensive upstream wrappers, smoke-retrain CLI flags, flat-clone
auto-detect) that have been folded into the codebase across several
unrelated commits.  Without an explicit guard, any of these can
silently regress when somebody refactors the relevant module.  This file
locks them in place: six parametrised checks that simply grep the
relevant source file for a fingerprint that ONLY exists when the R17
fix is in place.  If the fix is removed, the keyword disappears, the
test fails, and the next reader knows exactly which invariant was
violated.

Why source-grep rather than behavioural?
  - The R17 fixes are "shaped" (a try/except branch, a CLI flag, a vocab
    tuple expansion) rather than "semantic" (a metric, a curve) — they
    are best characterised by their literal form in source.
  - Behavioural tests for each fix already exist elsewhere (see
    test_atom_training_contract.py for the vocab M15 regression pair);
    the value here is a *fast, isolated* canary that fails within ms
    when somebody deletes a line during a clean-up.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Locator helpers — all paths are repo-relative to keep the test resilient
# to where the runner was invoked from (pytest is typically run from the
# repo root, but CI runners can vary).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOLMETAL = _REPO_ROOT / "molmetal"
_ADAPTER = _MOLMETAL / "adapters" / "flow_matching_lipman"
_SCRIPTS = _MOLMETAL / "scripts"


def _read(p: Path) -> str:
    """Read *p* as UTF-8 text; surface a clear error if the file vanished."""
    assert p.is_file(), f"R17 regression target missing: {p}"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The 6 R17 invariants we are guarding.
# ---------------------------------------------------------------------------
# Each entry is a (id, file, keyword, description) tuple.  Keeping them as
# data (rather than a hand-written test_*) makes the table easy to extend
# when new R17-style fixes land, and ensures every invariant follows the
# same verification recipe.
_R17_INVARIANTS = [
    (
        "atom_vocab_has_20_elements",
        _ADAPTER / "__init__.py",
        # R17 / TODO-33 — canonical 20-element vocab (1, 5, 6, 7, 8, 9, 14,
        # 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78).  The
        # _atom_vocab tuple MUST be assigned (not just mentioned in a
        # comment) so the model knows the expanded support.  20 comma-
        # separated integers separated by ', ' are unique to this vocab.
        ("1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, " "39, 40, 53, 72, 78"),
        "Canonical 20-element _atom_vocab tuple (TODO-33 / M12 fix)",
    ),
    (
        "solver_has_defensive_typeerror",
        _ADAPTER / "_reference" / "solver" / "__init__.py",
        # R17 — defensive try/except TypeError around upstream sampler
        # call.  The block is uniquely characterised by the comment
        # "Upstream rejected the wrapper at sample-time" — only present
        # when the R17 fallback path is wired in.
        "Upstream rejected the wrapper at sample-time",
        "Solver shim falls back when upstream sampler raises TypeError",
    ),
    (
        "utils_modelwrapper_has_4try_fallback",
        _ADAPTER / "_reference" / "utils" / "__init__.py",
        # R17 — ModelWrapper tries 4 call signatures in sequence.  The
        # 4th (last-resort) line is the most diagnostic — it has no
        # surrounding try/except so its presence is a load-bearing
        # signal.
        "        return self.model(x, t, **extras)",
        "ModelWrapper 4-try fallback covers positional + kw + extras",
    ),
    (
        "smoke_retrain_has_lr_warmup_flag",
        _SCRIPTS / "metallo_drug_smoke_retrain.py",
        # R17 — `--lr-warmup-steps` CLI flag (and the ``lr_warmup_steps``
        # attribute on the parsed namespace).  Underscore-to-dash is the
        # argparse convention; either spelling is a fingerprint.
        "--lr-warmup-steps",
        "metallo_drug_smoke_retrain accepts --lr-warmup-steps",
    ),
    (
        "smoke_retrain_has_decode_smoke_every_flag",
        _SCRIPTS / "metallo_drug_smoke_retrain.py",
        # R17 — `--decode-smoke-every` CLI flag introduced so the
        # mid-training decode smoke is configurable.  The argparse
        # declaration is the load-bearing line.
        "--decode-smoke-every",
        "metallo_drug_smoke_retrain accepts --decode-smoke-every",
    ),
    (
        "reference_loader_has_flat_clone_autodetect",
        _ADAPTER / "_reference_loader.py",
        # R17 — auto-detect whether the user passed the parent dir
        # (legacy) or the clone root directly (canonical).  The
        # discriminant is the literal pair of branches that checks for
        # ``__init__.py`` AND ``ref.name == "flow_matching"``.
        'ref.name == "flow_matching"',
        "_reference_loader auto-detects flat vs nested clone root",
    ),
]


@pytest.mark.parametrize(
    "file_path,keyword,description",
    [(item[1], item[2], item[3]) for item in _R17_INVARIANTS],
    ids=[item[0] for item in _R17_INVARIANTS],
)
def test_r17_invariant_present(file_path: Path, keyword: str, description: str) -> None:
    """Each R17 invariant must remain literally present in source."""
    source = _read(file_path)
    assert keyword in source, (
        f"R17 regression: {description}. "
        f"Expected to find {keyword!r} in {file_path.relative_to(_REPO_ROOT)}, "
        f"but it is missing.  Was the R17 fix reverted?"
    )
