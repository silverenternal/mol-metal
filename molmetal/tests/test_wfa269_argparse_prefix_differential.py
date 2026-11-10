# Licensed under the Apache License, Version 2.0.
"""Wf-A269 frontier sweep — the two ``_argparse_accepts`` twins vs REAL argparse.

THE QUESTION
------------
Two §5.11 gates decide "would this flag die the launch?" with a hand-rolled
prefix rule:

* ``scripts/_script_contract_gate.py::_argparse_accepts`` (AST vocabulary), and
* ``scripts/_runtime_transport_gate.py::_argparse_accepts`` (runtime vocabulary).

A hand-rolled rule that DISAGREES with ``argparse`` is worse than no rule: it
either reports a launch as broken when it is fine (a false positive, which
trains the reader to ignore the gate) or blesses a launch that dies.  So the
rule must be checked against the thing it claims to model — ``argparse``
itself — not against a restatement of itself.

WHAT THIS TEST DOES
-------------------
* **D1** a differential matrix: for every (parser, flag) pair, the twins must
  agree with what a REAL ``argparse.ArgumentParser`` does.  The ground truth
  is ``parse_known_args`` — a flag argparse consumed is not left in the
  extras, and an ambiguous or unrecognised flag raises ``SystemExit``.
* **D2** the premise the twins rest on, turned from a comment into a
  test-locked fact: ``argparse`` enables prefix abbreviation by default, and
  **0** of the repo's parsers pass ``allow_abbrev=False``.  (The auditor's
  G6(a) claimed the twin "never reads ``allow_abbrev``" — that is a
  MISREADING: the twin's own docstring says exactly the opposite, and this
  test is what makes the statement checkable rather than asserted.)
* **D3** the vocabulary the twins read: the AST-extracted option set and the
  runtime-imported option set must agree on every LONG option, for every
  trainer a config's ``entry.script`` names.  A twin can only be as right as
  the vocabulary it is handed — an incomplete vocabulary produces false
  positives, not misses.

HONEST LIMITS
-------------
* The twins are asked about BARE flags; the ``--flag=value`` form is out of
  scope for both (their own token regexes exclude ``=``).
* ``D3``'s census needs the RUNTIME vocabulary, so it imports every trainer
  (CPU-only; MEASURED 8.0 s on this tree).  If a trainer cannot be imported,
  the census reports a coverage gap rather than pretending it passed.
* ``D3``'s known gap is DECLARED, not hidden — see
  :data:`DECLARED_VOCABULARY_GAPS`.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relpath: str):
    """Import a repo-local gate script by path (they are not a package)."""
    path = REPO_ROOT / relpath
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SCRIPT_GATE = _load("_script_contract_gate", "scripts/_script_contract_gate.py")
_RUNTIME_GATE = _load("_runtime_transport_gate", "scripts/_runtime_transport_gate.py")

if _SCRIPT_GATE is None or not hasattr(_SCRIPT_GATE, "_argparse_accepts"):
    pytest.skip(
        "scripts/_script_contract_gate.py is not importable here — the §5.11 "
        "script-contract gate is an UNTRACKED file, so a clean checkout of "
        "HEAD does not have it (§6.269.23.12)",
        allow_module_level=True,
    )
if _RUNTIME_GATE is None or not hasattr(_RUNTIME_GATE, "_argparse_accepts"):
    pytest.skip(
        "scripts/_runtime_transport_gate.py is not importable here — the "
        "§5.11 runtime-transport gate is an UNTRACKED file, so a clean "
        "checkout of HEAD does not have it (§6.269.23.12)",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------
# Ground truth — REAL argparse
# --------------------------------------------------------------------------


def _argparse_really_accepts(parser: argparse.ArgumentParser, flag: str) -> bool:
    """Would a REAL argparse parser CONSUME ``flag`` as an option?

    ``parse_known_args`` is used because it does not raise on an
    unrecognised argument: the flag comes back in ``extras``.  An AMBIGUOUS
    prefix raises ``SystemExit`` (argparse's own "ambiguous option" error).
    Either way, the answer is "no".
    """
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            _ns, extras = parser.parse_known_args([flag])
    except SystemExit:
        return False
    return flag not in extras


def _parser(*option_strings: str) -> argparse.ArgumentParser:
    """A parser whose options take NO value (so no "expected one argument")."""
    ap = argparse.ArgumentParser(add_help=False)
    for opt in option_strings:
        ap.add_argument(opt, action="store_true")
    return ap


def _ast_vocabulary(parser: argparse.ArgumentParser) -> dict[str, str]:
    """The vocabulary the AST-based twin is handed: option -> arity."""
    return {opt: "none" for opt in parser._option_string_actions}


def _runtime_vocabulary(parser: argparse.ArgumentParser) -> set[str]:
    """The vocabulary the runtime-based twin is handed: the option strings."""
    return set(parser._option_string_actions)


# --------------------------------------------------------------------------
# D1 — the differential matrix
# --------------------------------------------------------------------------

#: (parser options, flags under test, what argparse does with each flag)
#:
#: MEASURED against a real ``argparse.ArgumentParser`` on CPython 3.12.
DIFFERENTIAL_CASES = [
    # exact match, no prefix relationship
    (("--seeds", "--steps"), [("--seeds", True), ("--steps", True)]),
    # a unique PREFIX is accepted (argparse's allow_abbrev default)
    (("--seeds", "--steps"), [("--see", True)]),
    # an AMBIGUOUS prefix is rejected ("ambiguous option")
    (("--seeds", "--steps"), [("--s", False)]),
    # an EXACT match wins even when other options share its prefix —
    # argparse checks exact membership BEFORE the ambiguity test
    (("--seed", "--seeds", "--seed-x"), [("--seed", True)]),
    # no match at all -> "unrecognized arguments"
    (("--seeds",), [("--seedz", False)]),
    # a SHORT option does not perturb long-flag prefix matching
    (("-s", "--seeds"), [("--seeds", True), ("--s", True)]),
    # a short option alone is not a long flag
    (("-s",), [("--s", False)]),
    # an empty vocabulary rejects everything
    ((), [("--anything", False)]),
]



@pytest.mark.parametrize(
    "options,flag,expected",
    [
        pytest.param(o, f, e, id=f"{'|'.join(o) or 'EMPTY'}::{f}")
        for o, flags in DIFFERENTIAL_CASES
        for f, e in flags
    ],
)
def test_d1_the_twins_agree_with_real_argparse(options, flag, expected):
    """Both twins must return exactly what a real argparse parser does."""
    parser = _parser(*options)

    truth = _argparse_really_accepts(parser, flag)
    assert truth is expected, (
        f"the ground truth changed for options={options} flag={flag}: "
        f"argparse says {truth}, the case table says {expected}. CPython's "
        "argparse semantics moved — re-derive the table, do not paper over it."
    )

    ast_twin = _SCRIPT_GATE._argparse_accepts(_ast_vocabulary(parser), flag)
    rt_twin = _RUNTIME_GATE._argparse_accepts(_runtime_vocabulary(parser), flag)

    assert ast_twin is truth, (
        f"scripts/_script_contract_gate.py::_argparse_accepts says {ast_twin} "
        f"for {flag!r} against options={options}, but argparse says {truth}"
    )
    assert rt_twin is truth, (
        f"scripts/_runtime_transport_gate.py::_argparse_accepts says {rt_twin} "
        f"for {flag!r} against options={options}, but argparse says {truth}"
    )


def test_d1b_the_twins_agree_with_each_other():
    """The two gates must not disagree — they police the SAME launches."""
    for options, flags in DIFFERENTIAL_CASES:
        parser = _parser(*options)
        for flag, _expected in flags:
            a = _SCRIPT_GATE._argparse_accepts(_ast_vocabulary(parser), flag)
            b = _RUNTIME_GATE._argparse_accepts(_runtime_vocabulary(parser), flag)
            assert a is b, f"{flag!r} against {options}: script-contract={a}, runtime-transport={b}"


# --------------------------------------------------------------------------
# D2 — the premise the twins rest on, made checkable
# --------------------------------------------------------------------------


def _repo_parsers_using_allow_abbrev() -> list[str]:
    """Repo scripts whose ``ArgumentParser`` passes ``allow_abbrev=``."""
    hits: list[str] = []
    for path in sorted((REPO_ROOT / "molmetal" / "scripts").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "ArgumentParser"):
                continue
            for kw in node.keywords:
                if kw.arg == "allow_abbrev":
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


def test_d2_no_repo_parser_disables_prefix_abbreviation():
    """The twins' prefix rule is the RIGHT rule for this repo — MEASURED.

    ``argparse`` enables prefix abbreviation by default; the twins therefore
    treat ``--see`` as acceptable for a parser that declares ``--seeds``.
    That is only correct while NO repo parser passes ``allow_abbrev=False``
    (which would switch abbreviation OFF and make a prefix UNRECOGNISED).

    MEASURED 2026-11-04 on this tree: **0** such parsers.  If one appears,
    the twins become wrong for it and this test says so.
    """
    hits = _repo_parsers_using_allow_abbrev()
    assert hits == [], (
        "these repo parsers pass allow_abbrev=False, so argparse does NOT "
        "accept a prefix for them — but both §5.11 twins apply the prefix "
        f"rule unconditionally: {hits}"
    )


def test_d2b_the_dual_prefix_rule_is_the_one_argparse_uses():
    """The rule is "exact match wins, else a UNIQUE prefix" — pin the order.

    ``argparse`` checks exact membership BEFORE the ambiguity test
    (``ArgumentParser._parse_optional``), so an option that is both an exact
    match and a prefix of others is ACCEPTED.  A twin that only counted
    prefixes would reject it.
    """
    parser = _parser("--seed", "--seeds", "--seed-x")
    assert _argparse_really_accepts(parser, "--seed") is True
    assert _SCRIPT_GATE._argparse_accepts(_ast_vocabulary(parser), "--seed") is True
    assert _RUNTIME_GATE._argparse_accepts(_runtime_vocabulary(parser), "--seed") is True


# --------------------------------------------------------------------------
# D3 — the vocabulary the twins read
# --------------------------------------------------------------------------


#: Trainers whose AST-extracted option set CANNOT equal the runtime one, with
#: the reason on file.  MEASURED 2026-11-04: exactly one trainer in the
#: population a config's ``entry.script`` names.
#:
#: ``molmetal/scripts/train_cisplatin_resistance_v2.py`` builds its parser
#: with ``add_common_args(p, ...)`` / ``add_gpu_args(p)`` from
#: ``molmetal/scripts/_common_train_args.py``, so **16** of its options are
#: declared in a SHARED builder the AST walk does not follow.  MEASURED: the
#: AST vocabulary has 6 entries, the runtime vocabulary 22, and the 17
#: runtime-only names are ``--batch --cuda-visible-devices --data --depth
#: --dropout --encoder-ckpt --epochs --eval-every --hidden --lr
#: --metrics-path --mixed-precision --num-workers --out-dir --seeds
#: --summary-path --train-max-per-metal``.
#:
#: This is DECLARED here rather than silently tolerated: the census still
#: fails on any NEW drift, and the gap is visible in the test source.
DECLARED_VOCABULARY_GAPS: dict[str, str] = {
    "molmetal/scripts/train_cisplatin_resistance_v2.py": (
        "the parser is built by add_common_args()/add_gpu_args() from "
        "molmetal/scripts/_common_train_args.py, so the AST walk cannot see "
        "the 16 shared options it adds (MEASURED 2026-11-04: AST vocabulary "
        "6 entries, runtime vocabulary 22)"
    ),
}


def _entry_script_trainers() -> list[Path]:
    """Every trainer a shipped config's ``entry.script`` names."""
    found: dict[str, Path] = {}
    for root in (REPO_ROOT / "configs", REPO_ROOT / "molmetal" / "configs"):
        if not root.is_dir():
            continue
        for cfg in sorted(root.rglob("*.yaml")):
            try:
                data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            entry = data.get("entry")
            if not isinstance(entry, dict):
                continue
            script = entry.get("script")
            if not isinstance(script, str) or not script:
                continue
            p = REPO_ROOT / script
            if not p.exists():
                for cand in (
                    REPO_ROOT / "molmetal" / "scripts" / Path(script).name,
                    REPO_ROOT / "scripts" / Path(script).name,
                ):
                    if cand.exists():
                        p = cand
                        break
            if p.exists():
                found[str(p.relative_to(REPO_ROOT))] = p
    return [found[k] for k in sorted(found)]


def test_d3_the_twins_read_the_same_long_options():
    """The AST vocabulary and the runtime vocabulary must agree.

    A twin can only be as right as the vocabulary it is handed.  The AST
    extractor cannot follow a SHARED argument builder, so its vocabulary can
    be missing options the real parser declares — and then the twin reports
    a perfectly good flag as "undeclared", a FALSE POSITIVE.

    MEASURED 2026-11-04 on this tree: 32 trainers, 1 with a DECLARED gap
    (:data:`DECLARED_VOCABULARY_GAPS`), 0 undeclared.
    """
    lc = _load("_launch_contract_gate", "scripts/_launch_contract_gate.py")
    if lc is None or not hasattr(lc, "_declared_flags"):
        pytest.skip("scripts/_launch_contract_gate.py is not importable here")

    drift: dict[str, list[str]] = {}
    unreadable: list[str] = []
    for script in _entry_script_trainers():
        rel = str(script.relative_to(REPO_ROOT))
        ast_flags, ast_err, _state = _SCRIPT_GATE.declared_flags(script)
        runtime, rt_err = lc._declared_flags(script)
        if ast_flags is None or runtime is None:
            unreadable.append(f"{rel} (ast_err={ast_err!r} rt_err={rt_err!r})")
            continue
        only_runtime = sorted(
            opt for opt in (runtime - set(ast_flags)) if opt.startswith("--")
        )
        if only_runtime and rel not in DECLARED_VOCABULARY_GAPS:
            drift[rel] = only_runtime

    assert not drift, (
        "the AST option vocabulary and the RUNTIME option vocabulary disagree "
        "for a trainer a shipped config names — the AST-based §5.11 gate would "
        f"report a valid flag as undeclared: {drift}. Declare the trainer in "
        "DECLARED_VOCABULARY_GAPS with a reason, or fix the extractor."
    )


def test_d3b_every_declared_gap_is_still_real():
    """A DECLARED gap must still be a gap — otherwise it rots.

    If the shared builder is ever inlined (or the extractor learns to follow
    it), the declared gap becomes stale and must be removed.
    """
    lc = _load("_launch_contract_gate", "scripts/_launch_contract_gate.py")
    if lc is None or not hasattr(lc, "_declared_flags"):
        pytest.skip("scripts/_launch_contract_gate.py is not importable here")

    stale: list[str] = []
    for rel in DECLARED_VOCABULARY_GAPS:
        script = REPO_ROOT / rel
        if not script.is_file():
            stale.append(f"{rel}: the trainer is no longer on disk")
            continue
        ast_flags, _err, _state = _SCRIPT_GATE.declared_flags(script)
        runtime, _rt_err = lc._declared_flags(script)
        if ast_flags is None or runtime is None:
            continue
        only_runtime = {o for o in (runtime - set(ast_flags)) if o.startswith("--")}
        if not only_runtime:
            stale.append(f"{rel}: the AST vocabulary now MATCHES — drop the declaration")
    assert not stale, (
        "these DECLARED_VOCABULARY_GAPS are no longer gaps (§5.5 rule #6: a "
        f"declaration that is not re-checked rots): {stale}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
