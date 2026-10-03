"""Import-health regression guard for Wf-416-A1-migrated scripts.

Wf-A10 FIX (2026-10-03) — why this file exists
----------------------------------------------
The Wf-416-A1 atomic-write migration rewrote direct ``Path.write_text``
calls across a dozen ``molmetal/scripts/*.py`` files.  It reported success
and shipped, but it silently corrupted five of them:

  * four files had the injected helper spliced **between a ``@dataclass``
    decorator and the class it decorated** — the decorator then landed on
    the plain function and the module died at import with
    ``AttributeError: 'function' object has no attribute '__mro__'``, while
    the class silently lost its generated ``__init__``::

        @dataclass                              # ← was decorating NCIPreciousDataset
        def _wf416_atomic_write_text(...): ...  # ← now the decorated target
        class NCIPreciousDataset: ...           # ← no longer a dataclass

    (``train_nci60_10cell.py``, ``train_metal_lomo.py``,
    ``wf146_chemprop_v2_lomo_v2.py``, ``generate_section4.py``)

  * one file had the injected import block dropped **into the middle of a
    function body**, orphaning the function's tail into the ``except
    ImportError:`` clause — ``SyntaxError: 'return' outside function``
    (``m6_wf320h_vina_phase2.py``).

All five had been reported SHIPPED.  ``train_nci60_10cell.py`` is the
script behind the m21 Tier-1 LEAD metric (0.8007 AUC); ``train_metal_lomo.py``
is the LOMO trainer.  Neither could be imported, so neither baseline could
be reproduced.

The defect class is invisible to a syntax check on the *source* (a
``compile()`` succeeds for the decorator case) and invisible to the
migration's own tests, because **nothing ever imported the migrated
scripts**.  This is the same failure mode AGENTS.md §5.7 names for trainer
flags: a SCAFFOLD whose unit tests pass while the integration point is
never exercised.

What this guard does
--------------------
Any script that carries the migration marker opts itself into an
import-health check, so a future migration (or hand edit) that breaks an
import fails here instead of shipping.  Two layers:

  1. ``test_no_dataclass_clobber`` — a fast static check for the exact
     splice signature, giving a precise diagnosis rather than a bare
     ImportError.
  2. ``test_marked_script_imports`` — one subprocess import per marked
     script.  This runs in a subprocess because decorator/annotation
     errors only surface on import, not on ``compile()``.

Adding a script to the guarded set requires no edit here: carry the marker
and it is covered.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"

#: Any script carrying one of these opts into the import-health guard.
MIGRATION_MARKERS = ("_wf416_atomic_write_text", "_WF416_HAS_ATOMIC")

#: The splice signature that broke four files — GENERALISED (Wf-A16).
#:
#: This used to be pinned to the exact Wf-416 helper name
#: (``def _wf416_atomic_write_text``). That made the guard marker-specific, and
#: it duly missed the SAME defect arriving from a different wave:
#: ``molmetal/scripts/r4_c_full_sweep.py`` had the Wf-424-A4 migration splice its
#: helper between ``@dataclass`` and ``class PocketResult``, which killed the
#: import and stopped three test modules from even being COLLECTED. A guard that
#: keys on one wave's identifier protects against one wave's bug, not the bug.
#:
#: So the pattern is now the DEFECT SHAPE, not the tool that caused it: any
#: ``@dataclass`` decorator immediately followed by a ``def``. `\b` after
#: ``@dataclass`` is load-bearing — it stops the rule firing on the legitimate
#: ``@dataclass_transform()``, which really does decorate a function (`_` is a
#: word character, so there is no boundary between ``dataclass`` and
#: ``_transform``).
_DECORATOR_CLOBBER = re.compile(
    r"^[ \t]*@dataclass\b[^\n]*\n" r"[ \t]*(?:async[ \t]+)?def[ \t]",
    re.MULTILINE,
)


#: Every script, not just the marked ones — the Wf-424 instance was unmarked.
def _all_scripts() -> list[Path]:
    """Every ``molmetal/scripts/*.py``, sorted for stable ids."""
    return sorted(SCRIPTS_DIR.glob("*.py"))


def _marked_scripts() -> list[Path]:
    """Scripts carrying the Wf-416-A1 migration marker, sorted for stable ids."""
    out: list[Path] = []
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover — unreadable file
            continue
        if any(m in text for m in MIGRATION_MARKERS):
            out.append(path)
    return out


MARKED = _marked_scripts()
MARKED_IDS = [p.name for p in MARKED]


def test_guard_is_not_vacuous() -> None:
    """If discovery silently finds nothing, the suite would pass green and mean nothing."""
    assert MARKED, (
        f"no scripts under {SCRIPTS_DIR} carry any of {MIGRATION_MARKERS!r} — "
        "either the migration was reverted or this guard has stopped discovering "
        "the population it is supposed to protect"
    )


def test_no_dataclass_clobber() -> None:
    """No script — marked or not — may have ``@dataclass`` spliced onto a ``def``.

    Wf-A16: this used to iterate ``MARKED``, i.e. only scripts carrying the
    Wf-416 marker. That is exactly why it missed ``r4_c_full_sweep.py``, whose
    clobber came from Wf-424-A4 and therefore was never in the population the
    guard looked at. The population is now every script.
    """
    offenders: list[str] = []
    for path in _all_scripts():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _DECORATOR_CLOBBER.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line_no}")
    assert not offenders, (
        "a @dataclass decorator is sitting on a function (which means a class "
        "below it lost its decorator and the module cannot import — "
        "AttributeError: 'function' object has no attribute '__mro__'), or the "
        "module will not import at all. See this file's docstring. Offenders: "
        + ", ".join(offenders)
    )


def test_clobber_pattern_is_not_vacuous() -> None:
    """A pattern that matches nothing proves nothing — fire it on a plant.

    Wf-A16: the previous guard was specific enough to be safe and too specific
    to be useful. This pins the OTHER direction: the regex must still catch the
    defect shape, and must NOT catch the legitimate ``@dataclass_transform()``
    (which decorates functions on purpose).
    """
    planted = "@dataclass\ndef _wf424_whatever(x): ...\n"
    assert _DECORATOR_CLOBBER.search(planted), (
        "the clobber regex no longer fires on a planted clobber — the guard has "
        "gone vacuous and would pass on real corruption"
    )
    legitimate = "@dataclass_transform()\ndef create_model(cls): ...\n"
    assert not _DECORATOR_CLOBBER.search(legitimate), (
        "the clobber regex fires on @dataclass_transform(), which legitimately "
        "decorates functions — the guard would now flag correct code"
    )


@pytest.mark.parametrize("script", MARKED, ids=MARKED_IDS)
def test_marked_script_imports(script: Path) -> None:
    """Each marked script must import cleanly in a fresh interpreter.

    Import-time failures (decorator on a function, orphaned ``return``,
    bad annotations) are exactly what the migration shipped, so a
    subprocess import is the assertion that actually catches them.
    """
    module = f"molmetal.scripts.{script.stem}"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib, sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            f"importlib.import_module({module!r})",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"importing {module} failed (rc={proc.returncode}).\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}"
    )
