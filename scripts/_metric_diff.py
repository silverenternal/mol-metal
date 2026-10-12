#!/usr/bin/env python3
"""Tolerance-based structural diff for metric JSON artifacts.

Paper A's reproducibility driver (``scripts/reproduce_paper_a.py``) regenerates a
metric into a scratch file and must decide whether it matches the on-disk
artifact. Byte comparison is the wrong tool: the scorers are thread-nondeterministic
(XGBoost/LightGBM reduce in a thread-dependent order) and some artifacts round
floats to a fixed number of digits. So we compare *structure* and *values with a
tolerance*.

Rules
-----
* Dicts must have the same key set (unless a key is listed in ``ignore``).
* Lists must have the same length.
* Floats are equal when ``abs(a - b) <= abs_tol + rel_tol * abs(b)``.
* ``bool`` is never equal to ``int`` (``True == 1`` in Python; here it is not).
* Everything else compares with ``==``.

``ignore`` entries are dotted paths from the root, e.g.
``"verdict.recommendation"`` or ``"arms.tmqm.per_seed"``. A path that names a
list index (``"per_seed[0]"``) is also accepted.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-9


def _norm_ignore(ignore: Iterable[str]) -> Sequence[str]:
    return tuple(ignore)


def _is_ignored(path: str, ignore: Sequence[str]) -> bool:
    if not path:
        return False
    for pat in ignore:
        if path == pat or path.startswith(pat + "."):
            return True
    return False


def _join(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def _num_equal(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) <= abs_tol + rel_tol * abs(b)


def structural_diff(
    expected: Any,
    actual: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    ignore: Iterable[str] = (),
    path: str = "",
) -> List[str]:
    """Return a list of human-readable differences (empty list == match)."""
    ignore = _norm_ignore(ignore)
    diffs: List[str] = []

    if _is_ignored(path, ignore):
        return diffs

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            child = _join(path, key)
            if _is_ignored(child, ignore):
                continue
            if key not in expected:
                diffs.append(f"+{child}: unexpected key (value {actual[key]!r})")
            elif key not in actual:
                diffs.append(f"-{child}: missing key (expected {expected[key]!r})")
            else:
                diffs += structural_diff(
                    expected[key], actual[key],
                    abs_tol=abs_tol, rel_tol=rel_tol, ignore=ignore, path=child,
                )
        return diffs

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"~{path}: length {len(expected)} != {len(actual)}")
            return diffs
        for i, (e, a) in enumerate(zip(expected, actual)):
            child = f"{path}[{i}]"
            if _is_ignored(child, ignore):
                continue
            diffs += structural_diff(
                e, a, abs_tol=abs_tol, rel_tol=rel_tol, ignore=ignore, path=child
            )
        return diffs

    # bool must not silently equal int (True == 1 in Python)
    if isinstance(expected, bool) or isinstance(actual, bool):
        if not (isinstance(expected, bool) and isinstance(actual, bool)):
            diffs.append(f"!{path}: type bool vs {type(actual).__name__}")
        elif expected != actual:
            diffs.append(f"!{path}: {expected!r} != {actual!r}")
        return diffs

    if isinstance(expected, float) and isinstance(actual, float):
        if not _num_equal(expected, actual, abs_tol, rel_tol):
            diffs.append(
                f"!{path}: {expected!r} != {actual!r} "
                f"(|d|={abs(expected - actual):.3e} > tol {abs_tol:.1e})"
            )
        return diffs

    if type(expected) is not type(actual):
        diffs.append(
            f"!{path}: type {type(expected).__name__} != {type(actual).__name__}"
        )
        return diffs

    if expected != actual:
        diffs.append(f"!{path}: {expected!r} != {actual!r}")
    return diffs
