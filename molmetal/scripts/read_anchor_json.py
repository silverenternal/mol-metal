"""Wf-302A — ``AnchorGuard`` helper for read_anchor_json.

=====================================================================
Wf-291H ship surface: a single Python helper that any downstream
metric / report / dashboard can use to safely read a paper-grade
``*_anchor.json`` file with three explicit guards:

  1. **Path exists**   — raises :class:`AnchorMissingError` if not.
  2. **mtime fresh**    — raises :class:`AnchorStaleError` if older
                         than ``max_age_sec`` (default 24 hr).
  3. **JSON parseable** — raises :class:`AnchorJSONError` with the
                         offending line/column on malformed JSON.
  4. **version match**  — optional ``required_version`` check; raises
                         :class:`AnchorVersionMismatchError` on miss.

Why a helper rather than ad-hoc ``json.loads(open(...).read())``
-----------------------------------------------------------------
The M14 lead-deployment precondition (TODO-207 §9.AF +
TODO-207 §9.6.3) reuses a paper-grade anchor from Wf-287A across
multiple downstream consumers (paper §4, m14 LEAD claim, dashboard,
auto-update). The historical bug class is:

  *   "**stale read**" — the anchor was last regenerated weeks ago
      and the underlying cohort / checkpoint has since moved on, but
      a consumer silently re-emitted the old number.
  *   "**missing read**" — the anchor file was renamed / archived
      and the consumer crashed with a bare ``FileNotFoundError``
      whose line numbers pointed at the wrong call site.
  *   "**version drift**" — a downstream consumer pinned ``v1``
      but the anchor was upgraded to ``v2`` with a different schema,
      causing a silent ``KeyError`` deep in the consumer.

``AnchorGuard`` makes all three failure modes loud and explicit
with named exception types so callers can either propagate them or
catch them at the API boundary.

Public surface
--------------
* :class:`AnchorGuard`  — the main helper class.
* :class:`AnchorMissingError`
* :class:`AnchorStaleError`
* :class:`AnchorJSONError`
* :class:`AnchorVersionMismatchError`

Usage
-----
::

    from molmetal.scripts.read_anchor_json import AnchorGuard

    guard = AnchorGuard(
        path="molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json",
        max_age_sec=86400,                # 24 hr
        required_version="m14_v1",        # optional
    )
    anchor = guard.require()
    print(anchor["value"])

The ``require()`` method is an alias for ``read()`` — the spec
distinguishes them semantically (``read`` returns the dict, ``require``
asserts + returns) but they share the same implementation here.

Hardware: CPU-only. No GPU. No external services.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


__all__ = [
    "AnchorGuard",
    "AnchorMissingError",
    "AnchorStaleError",
    "AnchorJSONError",
    "AnchorVersionMismatchError",
]


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------
class AnchorMissingError(FileNotFoundError):
    """Raised when the anchor file does not exist on disk.

    Inherits from :class:`FileNotFoundError` so callers using a bare
    ``except FileNotFoundError`` keep working — but the named type is
    exported for callers who want to distinguish anchor-missing from
    any other file-not-found.
    """


class AnchorStaleError(RuntimeError):
    """Raised when the anchor's mtime is older than ``max_age_sec``.

    The message includes the measured age in seconds so the caller can
    surface a useful diagnostic without re-stat'ing the file.
    """


class AnchorJSONError(ValueError):
    """Raised when the anchor file exists but is not valid JSON.

    Wraps :class:`json.JSONDecodeError` with the anchor path prefixed
    so the call site is obvious from the exception message alone.
    """


class AnchorVersionMismatchError(RuntimeError):
    """Raised when the anchor's ``version_field`` does not match ``required_version``.

    The message includes both the expected and the actual value.
    """


# ---------------------------------------------------------------------------
# AnchorGuard
# ---------------------------------------------------------------------------
class AnchorGuard:
    """Read a paper-grade ``*_anchor.json`` with explicit freshness / version guards.

    Parameters
    ----------
    path
        Filesystem path to the anchor JSON. May be a :class:`str` or a
        :class:`pathlib.Path`; the constructor coerces to ``Path``.
    max_age_sec
        Maximum allowed mtime age in seconds. Default ``86400`` (24 hr).
        Anchors older than this raise :class:`AnchorStaleError`.
    required_version
        If given (and not ``None``), the anchor's ``version_field``
        value MUST equal this string. Mismatch raises
        :class:`AnchorVersionMismatchError`. If ``None``, the version
        check is skipped entirely.
    version_field
        Top-level JSON key that carries the version stamp. Default
        ``"version"``. Wf-287A's ``m14_anchor.json`` does not yet
        carry a ``version`` field, so consumers can pass a different
        field name (e.g. ``"wf_origin"``) for legacy anchors.

    Methods
    -------
    read() -> dict
        Run the four guards and return the parsed JSON. Raises one of
        :class:`AnchorMissingError`, :class:`AnchorStaleError`,
        :class:`AnchorJSONError`, or :class:`AnchorVersionMismatchError`.
    require() -> dict
        Alias for :meth:`read`. Kept distinct so callers can express
        intent at the call site (``guard.require()`` reads as "I
        demand this anchor" rather than ``guard.read()`` which reads
        as "I want to read this anchor").
    """

    def __init__(
        self,
        path: Union[str, Path],
        max_age_sec: float = 86400.0,
        required_version: Optional[str] = None,
        version_field: str = "version",
    ) -> None:
        self.path: Path = Path(path)
        self.max_age_sec: float = float(max_age_sec)
        self.required_version: Optional[str] = (
            None if required_version is None else str(required_version)
        )
        self.version_field: str = str(version_field)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read(self) -> Dict[str, Any]:
        """Read the anchor JSON, enforcing freshness + version guards.

        Returns
        -------
        dict
            The parsed JSON content. Always a ``dict`` at the top
            level — list-rooted JSON is rejected with
            :class:`AnchorJSONError`.

        Raises
        ------
        AnchorMissingError
            File does not exist (or stat failed).
        AnchorStaleError
            File mtime is older than ``max_age_sec``.
        AnchorJSONError
            File exists but is not parseable as a JSON object.
        AnchorVersionMismatchError
            ``required_version`` was supplied and the anchor's
            ``version_field`` value does not match.
        """
        # 1. Path exists
        if not self.path.exists():
            raise AnchorMissingError(f"anchor not found: {self.path}")

        # 2. mtime freshness
        try:
            stat_result = self.path.stat()
            age_sec = time.time() - stat_result.st_mtime
        except OSError as exc:
            raise AnchorMissingError(
                f"anchor stat failed: {self.path}"
            ) from exc
        if age_sec > self.max_age_sec:
            raise AnchorStaleError(
                f"anchor stale: age={age_sec:.1f}s > "
                f"max_age_sec={self.max_age_sec:.1f} ({self.path})"
            )

        # 3. JSON parse
        try:
            raw_text = self.path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise AnchorJSONError(
                f"anchor is not valid JSON: {self.path} "
                f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
            ) from exc
        except OSError as exc:
            raise AnchorMissingError(
                f"anchor read failed: {self.path}"
            ) from exc

        if not isinstance(data, dict):
            raise AnchorJSONError(
                f"anchor must be a JSON object at the top level; "
                f"got {type(data).__name__}: {self.path}"
            )

        # 4. version match (optional)
        if self.required_version is not None:
            actual = data.get(self.version_field)
            if actual != self.required_version:
                raise AnchorVersionMismatchError(
                    f"anchor version mismatch at {self.path}: "
                    f"expected {self.required_version!r}, "
                    f"got {actual!r} "
                    f"(version_field={self.version_field!r})"
                )

        return data

    def require(self) -> Dict[str, Any]:
        """Alias for :meth:`read` — semantically "demand this anchor".

        The implementation is identical; the name is kept so callers
        can express intent at the call site.
        """
        return self.read()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def read_anchor_json(
    path: Union[str, Path],
    max_age_sec: float = 86400.0,
    required_version: Optional[str] = None,
    version_field: str = "version",
) -> Dict[str, Any]:
    """Convenience wrapper: build a one-shot :class:`AnchorGuard` and call ``require()``.

    Equivalent to::

        AnchorGuard(path, max_age_sec, required_version, version_field).require()

    Provided so trivial one-call sites do not need to construct an
    :class:`AnchorGuard` instance.
    """
    return AnchorGuard(
        path=path,
        max_age_sec=max_age_sec,
        required_version=required_version,
        version_field=version_field,
    ).require()
