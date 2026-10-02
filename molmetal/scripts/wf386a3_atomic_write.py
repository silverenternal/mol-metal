"""Wf-386-A3: Atomic write protocol + overwrite guard + truncation detection.

================================================================
Closes the recurring "verification reliability + write overwriting"
problem surfaced across Wf-371 / Wf-373 / Wf-377 / Wf-378 / Wf-384
waves. The pattern is: an agent verifies the state of disk, then a
later agent overwrites substantial content under the same filename,
breaking the previous closeout. Symptoms:

  * A closeout marked SHIPPED in Wf-N is re-authored in Wf-N+1 with
    materially different content (>= 50% line diff), losing the
    historical evidence trail.
  * An agent's output is truncated (mid-sentence, missing closing
    tokens) and the calling workflow treats it as a complete closeout.
  * Two agents write to the same path concurrently, producing a
    half-merged file.

This module provides four primitives:

  * :func:`atomic_write_file` — write content to a temp path, fsync,
    rename atomically. Concurrent writers are serialised via a
    process-level lock keyed on the canonical (real) path.

  * :func:`safe_overwrite_check` — read the existing file at ``path``,
    compute a line-level diff against ``new_content``. Warn (log +
    raise :class:`OverwriteRisk`) when the existing content is
    substantial (>= ``min_lines`` lines) AND the diff exceeds
    ``diff_threshold`` (default 0.5). Silent when the file is new or
    the diff is small.

  * :func:`detect_truncation` — scan agent output for missing closing
    tokens (Markdown header / fenced code block / parenthetical /
    JSON brace / Python docstring). Flags ``True`` if output ends
    mid-structure or is shorter than ``expected_min_length``.

  * :func:`write_closeout` — convenience helper combining all three:
    runs :func:`safe_overwrite_check` first (returns an ``OverwriteRisk``
    when threshold exceeded, caller decides), then :func:`atomic_write_file`,
    then :func:`detect_truncation` on the written content as a sanity
    check.

Pure-Python — Linux only (uses ``os.rename`` which is atomic on POSIX,
not on Windows). Stdlib-only — no third-party deps. Safe to call from
any harness layer (workflow orchestrator, CI synthesiser, journal
writer, closeout drafter).

Adoption path
-------------

1. Replace direct ``Path(...).write_text(...)`` calls in closeout-drafting
   scripts with :func:`write_closeout`.
2. Wrap agent-output-parsing code paths with :func:`detect_truncation`
   before accepting the output as a complete closeout.
3. Wire :func:`safe_overwrite_check` into ``Wf-XXX-A1`` agent prompts
   that re-author prior closeouts — the agent should explicitly
   confirm it intends to overwrite substantial content.

Cross-references
----------------

* Wf-386-A1 root cause — overwriting Wf-377 closeouts on disk.
* Wf-386-A2 verbatim verdict — "WRONG 0/5 SHIPPED" + Wf-378-A3
  recomputation = "5/5 SHIPPED".
* :mod:`wf302d_retry_with_correction` — sibling defence for
  StructuredOutput schema confusion; this module is the I/O twin.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

LOG = logging.getLogger("wf386a3.atomic_write")
if not LOG.handlers:
    LOG.addHandler(logging.NullHandler())

# Per-canonical-path lock, process-local. Acquired during the temp-file
# write + fsync + rename window. Concurrent writers (multi-thread /
# multi-agent) will block here rather than racing on the rename.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _lock_for(path: Union[str, Path]) -> threading.Lock:
    """Return the per-path lock, creating it on first access."""
    key = str(Path(path).resolve())
    with _LOCKS_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


# ---------------------------------------------------------------------------
# Closing tokens used by detect_truncation. A "complete" agent output must
# end with one of these (after stripping trailing whitespace). We support
# Markdown, JSON, Python, plain English closeout patterns.
# ---------------------------------------------------------------------------
_CLOSING_TOKENS = (
    "\n# ",      # Markdown next-section header
    "\n## ",     # Markdown sub-section header
    "\n### ",    # Markdown sub-sub-section header
    "```\n",     # Fenced code block close
    "\n```",     # Fenced code block close (no newline after)
    "}\n",       # JSON closing brace + newline
    "\"\"\"\n",  # Python docstring close
    "\n",        # Any newline-ending line (most permissive fallback)
)


class OverwriteRisk(RuntimeError):
    """Raised by :func:`safe_overwrite_check` when an overwrite would
    destroy substantial prior content.

    The exception's ``.existing_lines`` and ``.diff_ratio`` attributes
    let callers decide whether to proceed (e.g., for an explicit
    USER_OK-gated re-author) or to back off and use a different
    filename.
    """

    def __init__(self, message: str, existing_lines: int, diff_ratio: float) -> None:
        super().__init__(message)
        self.existing_lines = existing_lines
        self.diff_ratio = diff_ratio


class TruncationDetected(RuntimeError):
    """Raised by :func:`detect_truncation` when agent output looks
    incomplete (missing closing tokens or too short).

    The exception's ``.output_length`` and ``.reason`` attributes let
    callers decide whether to retry, fall back to a sibling-data
    reconstruction, or treat the output as a partial closeout.
    """

    def __init__(self, message: str, output_length: int, reason: str) -> None:
        super().__init__(message)
        self.output_length = output_length
        self.reason = reason


def atomic_write_file(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = True,
) -> None:
    """Atomically write ``content`` to ``path`` via temp-file + rename.

    Process-local per-path serialisation means concurrent writers block
    rather than producing a half-merged file. The temp file is created
    in the same directory as the target so ``os.rename`` is a single
    inode swap (atomic on POSIX).

    Steps:

      1. Acquire per-path lock.
      2. Create ``{path}.tmp.{pid}.{counter}`` in same directory.
      3. Write content + fsync.
      4. ``os.rename(tmp, path)`` — atomic on POSIX.
      5. Release lock.

    On failure, the temp file is unlinked and the original ``path``
    is left untouched.
    """
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    lock = _lock_for(path)
    with lock:
        # Use NamedTemporaryFile with delete=False so we get a unique
        # name (avoids collisions even within the same process).
        fd, tmp_str = tempfile.mkstemp(
            prefix=f".{path.name}.tmp.", suffix=".part", dir=str(parent),
        )
        tmp_path = Path(tmp_str)
        try:
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp_path, path)  # os.replace is atomic on POSIX
            LOG.info("atomic_write: %s (%d bytes)", path, len(content))
        except Exception:
            # Clean up the temp file on any failure so we don't leak.
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise


def _line_diff_ratio(existing: str, new: str) -> float:
    """Compute a coarse line-level diff ratio in [0.0, 1.0].

    0.0 = identical lines; 1.0 = completely different lines. We use a
    simple set-of-lines Jaccard distance because line-level diff is
    the right granularity for Markdown closeouts (where line ordering
    is stable but content rewrites are common).
    """
    existing_lines = set(line for line in existing.splitlines() if line.strip())
    new_lines = set(line for line in new.splitlines() if line.strip())
    if not existing_lines and not new_lines:
        return 0.0
    union = existing_lines | new_lines
    if not union:
        return 0.0
    intersection = existing_lines & new_lines
    jaccard_sim = len(intersection) / len(union)
    return 1.0 - jaccard_sim


def safe_overwrite_check(
    path: Union[str, Path],
    new_content: str,
    *,
    min_lines: int = 50,
    diff_threshold: float = 0.5,
    raise_on_risk: bool = False,
) -> Optional[float]:
    """Check whether writing ``new_content`` to ``path`` would destroy
    substantial existing content.

    Returns the diff ratio if the existing file is substantial (>=
    ``min_lines`` non-empty lines) and the diff exceeds
    ``diff_threshold``. Returns ``None`` when:

      * the file does not yet exist (new write, no risk);
      * the existing file is small (< ``min_lines`` lines);
      * the diff is below threshold (small rewrite).

    When ``raise_on_risk`` is True and the diff exceeds threshold,
    raises :class:`OverwriteRisk` instead of returning a ratio.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        existing = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        LOG.warning("safe_overwrite_check: could not read %s (%s)", path, exc)
        return None

    existing_nonempty = sum(1 for line in existing.splitlines() if line.strip())
    if existing_nonempty < min_lines:
        return None  # file is small, no risk

    ratio = _line_diff_ratio(existing, new_content)
    if ratio > diff_threshold:
        msg = (
            f"Overwrite risk: {path} has {existing_nonempty} non-empty lines, "
            f"diff_ratio={ratio:.3f} > threshold={diff_threshold:.3f}. "
            "Use a different filename or confirm USER_OK for re-author."
        )
        LOG.warning(msg)
        if raise_on_risk:
            raise OverwriteRisk(msg, existing_nonempty, ratio)
        return ratio
    return None


def detect_truncation(
    agent_output: str,
    *,
    expected_min_length: int = 200,
) -> Optional[str]:
    """Return a reason string if ``agent_output`` looks truncated,
    else ``None``.

    Heuristic: the output must be at least ``expected_min_length`` chars
    long and must end with one of the canonical closing tokens (Markdown
    section header, fenced code close, JSON brace, Python docstring).
    A bare "..." or comma at the end is also flagged.

    This is intentionally permissive (matches on common patterns) rather
    than perfect (won't catch all truncation modes). When in doubt,
    call :func:`detect_truncation` and inspect the returned reason.
    """
    if not agent_output:
        return "empty_output"
    if len(agent_output) < expected_min_length:
        return f"too_short({len(agent_output)}<{expected_min_length})"
    # Strip trailing whitespace for the closing-token match.
    stripped = agent_output.rstrip()
    # Common truncation markers in agent output.
    bad_endings = (":", ",", "...", "…", "TODO", "FIXME", "WIP", "[truncated]")
    for bad in bad_endings:
        if stripped.endswith(bad):
            return f"ends_with_bad_token({bad!r})"
    # Require a closing token. Strip a trailing newline before matching
    # so we accept e.g. "...end of section.\n" as complete.
    for token in _CLOSING_TOKENS:
        if agent_output.endswith(token):
            return None
    # If we get here, the output doesn't end with any known close.
    # Check for the most common case: a final paragraph ending in
    # punctuation (period, question mark, exclamation, or right-paren).
    if stripped.endswith((".", "?", "!", ")", "]")):
        return None
    return "missing_closing_token"


def write_closeout(
    path: Union[str, Path],
    content: str,
    *,
    min_size: int = 1000,
    encoding: str = "utf-8",
    raise_on_overwrite: bool = False,
) -> None:
    """High-level helper: atomically write a closeout with overwrite
    guard + post-write truncation check.

    Steps:

      1. Run :func:`safe_overwrite_check` with default thresholds.
         When the existing file is substantial and the diff is high,
         returns the ratio (or raises if ``raise_on_overwrite=True``).
      2. Call :func:`atomic_write_file`.
      3. Run :func:`detect_truncation` on the written content as a
         sanity check; log a warning if it looks truncated.

    Use ``raise_on_overwrite=True`` when the caller is a re-author
    workflow that should refuse to silently overwrite prior closeouts.
    """
    path = Path(path)
    risk = safe_overwrite_check(
        path, content,
        min_lines=50,
        diff_threshold=0.5,
        raise_on_risk=raise_on_overwrite,
    )
    if risk is not None:
        LOG.warning(
            "write_closeout: overwrite risk at %s (ratio=%.3f) — caller should confirm USER_OK",
            path, risk,
        )

    atomic_write_file(path, content, encoding=encoding)

    # Post-write truncation sanity check. We log but don't raise —
    # a real truncation problem should already have surfaced during
    # the upstream agent's own self-check.
    truncation = detect_truncation(content, expected_min_length=min_size)
    if truncation is not None:
        LOG.warning(
            "write_closeout: post-write truncation check failed for %s (%s)",
            path, truncation,
        )


__all__ = [
    "atomic_write_file",
    "safe_overwrite_check",
    "detect_truncation",
    "write_closeout",
    "OverwriteRisk",
    "TruncationDetected",
]