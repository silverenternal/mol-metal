"""Wf-A128 (2026-10-31) — MolDiff attribution guard.

arXiv:2305.07508 is "MolDiff: Addressing the Atom-Bond Inconsistency
Problem in 3D Molecule Diffusion Generation" by **Xingang Peng, Jiaqi
Guan, Qiang Liu, Jianzhu Ma** (submitted 11 May 2023).

VERIFIED 2026-10-31 against primary sources:

  * the arXiv abs page and the PDF's own title block name the four
    authors above;
  * the venue is **ICML 2023** — the publisher's own entry
    ``https://proceedings.mlr.press/v202/peng23b.html`` carries the
    title above, the four authors, and the booktitle "Proceedings of
    the 40th International Conference on Machine Learning" (PMLR
    v202).  The arXiv abs page names no venue, which is why this had
    to be checked against the PDF / publisher rather than the
    abstract page.

The repo attributed the paper to "Vignac et al. 2023" in five shipped
places.  Vignac is a different paper (DiGress / MiDi, arXiv:2303.04791),
so the AUTHOR attribution was wrong; the "ICML 2023" venue it also
carried was correct and is kept.

This file is the guard for that correction:

  * every in-scope citation site names Peng, Guan, Liu & Ma,
  * every in-scope citation site keeps the (correct) arXiv id,
  * no in-scope site names a WRONG venue for the paper,
  * the MiDi citation (which IS Vignac) is untouched — a blanket
    "Vignac -> Peng" replacement would have broken it,
  * a repo-wide line scan finds no NEW "MolDiff + Vignac" attribution.

The last outstanding site was ``docs/HONEST_NEGATIVE_CATALOG.md``,
which the main thread owns (this agent must not write to ``docs/``).
It was listed in ``_KNOWN_OUTSTANDING`` so the scan still failed if the
set GREW.  The main thread corrected that site on 2026-10-31
(catalog line 4705-4706: ``Vignac et al. 2023`` -> ``Peng, Guan, Liu &
Ma 2023``), so the outstanding set is now EMPTY and the repo-wide scan
demands zero mis-pairings.

SELF-EXEMPTION (2026-10-31, Wf-A132).  This guard's own prose quotes
the mis-pairing it exists to detect (its docstring, its comment
headers, and the message text it prints).  Those are a DESCRIPTION of
the defect, not a mis-attribution, so this file is excluded from the
repo-wide scan by ``_SELF_PATH`` — a named, one-entry exemption, not a
silent hole.  ``test_self_exemption_is_exactly_one_file`` pins that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The five sites this wave corrected (path relative to the repo root).
IN_SCOPE_SITES = (
    "molmetal/adapters/flow_matching_lipman/__init__.py",
    "molmetal/ports/__init__.py",
    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    "molmetal/tests/test_wfa75_bond_length_guidance.py",
)

#: The MiDi citation — Vignac is CORRECT there and must not change.
MIDI_SITE = "molmetal/adapters/flow_matching_lipman/vector_channels.py"

#: Sites that still carry the wrong attribution and are owned by another
#: thread.  The scan below fails if this set GROWS — it does not fail
#: merely because these are still wrong.
#:
#: EMPTY as of 2026-10-31 (Wf-A132): the last entry,
#: ``docs/HONEST_NEGATIVE_CATALOG.md``, was corrected by the main thread
#: (catalog lines 4705-4706).  The set is kept (rather than deleted) so
#: that a future site can be registered without losing the
#: "the set must not grow" property.
_KNOWN_OUTSTANDING: frozenset[str] = frozenset()

#: This guard's own file.  It necessarily contains the mis-pairing in
#: its prose and in the message text it prints, so the repo-wide scan
#: skips it.  Named explicitly (one entry) and pinned by
#: ``test_self_exemption_is_exactly_one_file``.
_SELF_PATH = "molmetal/tests/test_wfa128_moldiff_attribution.py"

_CORRECT_AUTHORS = "Peng, Guan, Liu & Ma"
_ARXIV_ID = "2305.07508"


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _lines_with_both(text: str, *needles: str) -> list[int]:
    """1-indexed line numbers whose text contains ALL of ``needles``."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if all(n in line for n in needles):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# 1 — every in-scope site names the correct authors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", IN_SCOPE_SITES)
def test_moldiff_citation_names_correct_authors(rel: str) -> None:
    text = _read(rel)
    assert _CORRECT_AUTHORS in text, (
        f"{rel}: the MolDiff citation must name {_CORRECT_AUTHORS} "
        f"(the real authors of arXiv:2305.07508), not Vignac."
    )


# ---------------------------------------------------------------------------
# 2 — the arXiv id survives the correction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", IN_SCOPE_SITES)
def test_moldiff_citation_keeps_arxiv_id(rel: str) -> None:
    assert _ARXIV_ID in _read(rel), (
        f"{rel}: the MolDiff citation must keep the (correct) arXiv id "
        f"{_ARXIV_ID}."
    )


# ---------------------------------------------------------------------------
# 3 — no in-scope site mis-attributes MolDiff to Vignac
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", IN_SCOPE_SITES)
def test_moldiff_citation_not_attributed_to_vignac(rel: str) -> None:
    text = _read(rel)
    bad = _lines_with_both(text, "MolDiff", "Vignac")
    assert not bad, (
        f"{rel}: line(s) {bad} still attribute MolDiff to Vignac.  "
        f"MolDiff (arXiv:2305.07508) is Peng, Guan, Liu & Ma; Vignac is "
        f"MiDi/DiGress (arXiv:2303.04791)."
    )


# ---------------------------------------------------------------------------
# 4 — the venue claim
# ---------------------------------------------------------------------------
#: Venues the paper could plausibly be mis-credited with.  ICML 2023 is
#: the CORRECT one (PMLR v202, peng23b); every other token is wrong.
_WRONG_VENUES = (
    "ICLR",
    "ICCV",
    "ECCV",
    "NeurIPS",
    "AAAI",
    "CVPR",
    "KDD",
    "WWW",
)


@pytest.mark.parametrize("rel", IN_SCOPE_SITES)
def test_moldiff_citation_claims_no_wrong_venue(rel: str) -> None:
    """MolDiff is an ICML 2023 paper (PMLR v202 ``peng23b``).  No site
    may credit it with a different venue."""
    text = _read(rel)
    for venue in _WRONG_VENUES:
        bad = _lines_with_both(text, "MolDiff", venue)
        assert not bad, (
            f"{rel}: line(s) {bad} credit MolDiff with {venue}.  The paper "
            f"is ICML 2023 (PMLR v202 peng23b)."
        )


@pytest.mark.parametrize("rel", IN_SCOPE_SITES)
def test_moldiff_venue_is_icml_2023_when_named(rel: str) -> None:
    """Where the citation is stated in prose and a venue is named, that
    venue must be ICML 2023 — the one VERIFIED against the publisher's
    entry on 2026-10-31."""
    text = _read(rel)
    lines = text.splitlines()
    idx = [i for i, line in enumerate(lines) if "MolDiff" in line]
    assert idx, f"{rel}: no MolDiff citation found at all"
    for i in idx:
        window = lines[max(0, i - 6): min(len(lines), i + 7)]
        blob = "\n".join(window)
        for venue in ("ICML",) + _WRONG_VENUES:
            if venue not in blob:
                continue
            if venue != "ICML":
                pytest.fail(
                    f"{rel}: the MolDiff citation at line {i + 1} names the "
                    f"wrong venue {venue}; the paper is ICML 2023."
                )
            assert "ICML 2023" in blob, (
                f"{rel}: the MolDiff citation at line {i + 1} says ICML "
                f"without the year; write 'ICML 2023'."
            )


# ---------------------------------------------------------------------------
# 5 — the MiDi citation is untouched (the anti-blanket-replacement guard)
# ---------------------------------------------------------------------------
def test_midi_citation_still_credits_vignac() -> None:
    text = _read(MIDI_SITE)
    assert "MiDi" in text and "Vignac" in text and "2303.04791" in text, (
        "the MiDi citation (Vignac et al., arXiv:2303.04791) must NOT be "
        "touched — Vignac really is a MiDi/DiGress author."
    )


# ---------------------------------------------------------------------------
# 6 — repo-wide scan: no NEW "MolDiff + Vignac" attribution
# ---------------------------------------------------------------------------
def _tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_repo_wide_no_new_moldiff_vignac_attribution() -> None:
    flagged: list[str] = []
    for rel in _tracked_files():
        if rel.startswith(".claude/worktrees/"):
            continue
        if rel == _SELF_PATH:
            continue
        p = PROJECT_ROOT / rel
        try:
            if p.stat().st_size > 4_000_000:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _lines_with_both(text, "MolDiff", "Vignac") or _lines_with_both(
            text, "2305.07508", "Vignac"
        ):
            flagged.append(rel)
    unexpected = sorted(set(flagged) - _KNOWN_OUTSTANDING)
    assert not unexpected, (
        f"file(s) {unexpected} attribute MolDiff (arXiv:2305.07508) to "
        f"Vignac.  The correct authors are {_CORRECT_AUTHORS}.  If one of "
        f"these is a site this wave is not allowed to touch, add it to "
        f"_KNOWN_OUTSTANDING with a note — do not delete the assertion."
    )


# ---------------------------------------------------------------------------
# 7 — the self-exemption is exactly one file, and it is this file
# ---------------------------------------------------------------------------
def test_self_exemption_is_exactly_one_file() -> None:
    """The guard may not exempt itself from a growing hole: the exemption
    is a single, named path, and it must be THIS file."""
    assert isinstance(_SELF_PATH, str) and _SELF_PATH.endswith(
        "test_wfa128_moldiff_attribution.py"
    ), f"the self-exemption must name this guard file, got {_SELF_PATH!r}"
    assert (PROJECT_ROOT / _SELF_PATH).is_file(), (
        f"the self-exemption path {_SELF_PATH!r} does not exist — an "
        f"exemption for a file that is not there is a hole, not a guard."
    )
    # The exemption must not be a prefix/wildcard: it is compared with
    # equality against a repo-relative path.
    assert "*" not in _SELF_PATH and "?" not in _SELF_PATH


def test_known_outstanding_set_did_not_grow() -> None:
    """``_KNOWN_OUTSTANDING`` is the register of sites this wave was not
    allowed to touch.  It was emptied on 2026-10-31 when the main thread
    corrected the catalog; it must not silently re-grow."""
    assert _KNOWN_OUTSTANDING == frozenset(), (
        f"_KNOWN_OUTSTANDING grew to {sorted(_KNOWN_OUTSTANDING)}.  A new "
        f"mis-attribution site must be FIXED, not registered as outstanding."
    )
