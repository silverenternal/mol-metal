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

#: The exact splice signature that broke four files.
_DECORATOR_CLOBBER = re.compile(
    r"^[ \t]*@dataclass[^\n]*\n"
    r"[ \t]*def _wf416_atomic_write_text\b",
    re.MULTILINE,
)


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
    """No marked script may have ``@dataclass`` spliced onto the injected helper."""
    offenders: list[str] = []
    for path in MARKED:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _DECORATOR_CLOBBER.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line_no}")
    assert not offenders, (
        "the Wf-416-A1 helper is decorated with @dataclass, which means a "
        "class below it lost its decorator and the module cannot import — "
        "see this file's docstring for the failure mode. Offenders: "
        + ", ".join(offenders)
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
