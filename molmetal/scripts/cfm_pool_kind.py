# SPDX-License-Identifier: Apache-2.0
"""ONE owner for the CFM pool-kind rule (CPU-only, header + at most ONE row).

WHY THIS MODULE EXISTS
-----------------------
A CFM run leaves a ``generated_smiles.csv`` pool behind, and two DIFFERENT
code paths can write one:

  * ``molmetal/scripts/wf_cfm_decode_dump_smiles.py`` -- in its FIRST era
    (before 2026-10-12) it loaded REAL tmQM geometries and ran only the bond
    head.  That is a RECONSTRUCTION of real chemistry.  In its SECOND era
    (2026-10-12 onwards) it calls the trainer's own generation path
    (``evaluate_generation_validity`` -> ``generate_clouds_from_noise``, the
    real ODE integrator): that is FREE GENERATION.
  * ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` writes
    ``n_frags_raw``: the trainer's FREE-GENERATION path.

The two are not comparable.  A reconstruction pool inherits the properties of
the real chemistry it started from; calling its connectivity rate a
"fragmentation rate of the generator" is false.  v1.0 of the C7 audit compared
the two and called the gap "training scale"; it was a pipeline difference.

THE RULE IS NO LONGER HEADER-ONLY, AND HERE IS WHY
--------------------------------------------------
One script wrote ONE header in TWO eras.  ``wf_cfm_decode_dump_smiles.py``
emitted the same ``epoch,seed,smiles,sanitized,n_fragments,
largest_fragment_frac,metal_z`` header before and after 2026-10-12, but the
rows mean different things: era A rows carry the bond head's own
``sanitized`` flag and the real cloud's ``metal_z``; era B rows come from the
ODE sampler, whose ``decode_chemistry`` dict has NO ``sanitized`` key at all
(so it is structurally always 0) and whose ``metal_z`` is empty unless the
generated molecule happens to contain a matched metal.

So the header alone cannot decide a ``metal_z`` pool.  Reading the header and
calling every ``metal_z`` pool a reconstruction is what mis-bucketed
``molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv`` (66 rows),
which the project's own ``metrics/by_metric/cfm_free_generation_panel_seed0.json``
(2026-10-12: 192 attempted, 66 valid, gen_validity 0.344) shows is FREE
GENERATION.  The rule therefore reads the header AND AT MOST ONE data row.

THE RULE, AS CODE
------------------
1. an explicit ``pool_kind`` column is AUTHORITATIVE -- its value
   (``"generation"`` / ``"reconstruction"``) wins outright; anything else is
   unknown (the writer stamps the column, so a future pool cannot be
   ambiguous);
2. ``n_frags_raw`` in the header (and no ``metal_z``) -> generation
   (the trainer's free-generation path, one writer, unambiguous);
3. ``metal_z`` in the header (and no ``n_frags_raw``) -> NOT decidable from
   the header.  ``classify_pool_header`` returns ``kind="unknown"`` with a
   reason that says the CONTENT must be read; ``classify_pool`` reads AT MOST
   ONE data row and then decides:
     * ``sanitized == 1`` AND ``metal_z`` non-empty -> reconstruction
       (era A: the bond head on real geometries);
     * ``sanitized == 0`` AND ``metal_z`` empty      -> generation
       (era B: the ODE sampler);
   anything else -- the two signals disagree, a signal is missing, or there
   is no data row -- is UNKNOWN, with a reason that NAMES the disagreement.
   A pool is never silently bucketed;
4. BOTH markers in the header -> unknown ("both markers");
5. neither marker -> unknown.

WHY UNKNOWN IS A REAL OUTCOME (it is not a bug)
------------------------------------------------
Case 4 used to be silently resolved in favour of "reconstruction" ("checked
first, conservative").  It is not conservative to invent a provenance: a
header carrying both markers means the pool's writer changed, and the only
honest reading is that the evidence is contradictory.  Same for the content
disagreement in case 3.

THE REASON STRING IS THE EVIDENCE
---------------------------------
The reason is the ONLY place the deciding signal survives, so it always says
WHICH signal decided it (``marker: ...`` / ``content: sanitized ... + metal_z
...`` / the explicit ``pool_kind`` column) and, for an unknown pool, it names
the contradiction.

UNTIL THIS MODULE, THE RULE HAD THREE OWNERS
---------------------------------------------
``c7_scale_matched_pools.py`` held the marker constants and the classifier;
``cfm_fragmentation_audit.py`` held a SECOND copy of the per-kind SEMANTICS
(``MEASUREMENT_MEANING`` / ``VERDICT_SCOPE``); and ``cfm_fragmentation_gate.py``
was pool-kind BLIND -- it scored a pool without ever recording which pipeline
produced it.  One rule, one owner, one import.

UNKNOWN IS FIRST-CLASS, NEVER SILENTLY BUCKETED
------------------------------------------------
``classify_pool`` NEVER raises.  A missing file, an unreadable file and an
empty file are all ``kind="unknown"`` with a ``reason`` that says why, and a
malformed CSV is reported the same way (``csv.Error`` is NOT an ``OSError``,
so it used to escape this function -- while the docstring promised it never
raises -- and both C7 mains called it uncaught).  The refusal to use an
unclassified pool is a SEPARATE, EXPLICIT act: ``require_known_kind`` raises
``UnknownPoolKindError``.  A consumer that wants "refuse rather than bucket"
says so; a consumer that wants "report and move on" gets the reason string.

USAGE
-----
    from molmetal.scripts.cfm_pool_kind import classify_pool, require_known_kind

    kind = classify_pool(csv_path)
    if kind.kind == "unknown":
        ...  # kind.reason says why

    kind = require_known_kind(csv_path)   # raises UnknownPoolKindError

CPU-only.  Reads at most the header and ONE data row per call -- the body of a
pool is never touched (the c7 pool holds 12,668 rows).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

#: Written by wf_cfm_decode_dump_smiles.py -- in era A a bond-head
#: reconstruction of a real tmQM geometry.  Its PRESENCE no longer decides the
#: kind (see the module docstring); the CONTENT does.
RECONSTRUCTION_MARKER = "metal_z"

#: Written ONLY by the trainer's free-generation path.
GENERATION_MARKER = "n_frags_raw"

#: The explicit, AUTHORITATIVE column the decode-dump writer stamps.  When it
#: is present its value decides the kind outright.
POOL_KIND_COLUMN = "pool_kind"

#: The marker reported when the kind was decided by the CONTENT of the first
#: data row (``sanitized`` + ``metal_z``) rather than by a header marker.
CONTENT_MARKER = "sanitized+metal_z"

KIND_RECONSTRUCTION = "reconstruction"
KIND_GENERATION = "generation"
KIND_UNKNOWN = "unknown"

#: Every kind this module can return.
ALL_KINDS = (KIND_RECONSTRUCTION, KIND_GENERATION, KIND_UNKNOWN)

#: One-line statement of the rule, for payloads and docs.
POOL_KIND_RULE = (
    f"read from the CSV HEADER (and, for a {RECONSTRUCTION_MARKER!r} pool, the "
    f"CONTENT of its first data row), not the run name: an explicit "
    f"{POOL_KIND_COLUMN!r} column is authoritative; {GENERATION_MARKER!r} in the "
    f"header is the trainer's free-generation path; a {RECONSTRUCTION_MARKER!r} "
    f"header is decided by its content (sanitized == 1 and metal_z non-empty -> "
    f"reconstruction; sanitized == 0 and metal_z empty -> generation); both "
    f"markers, a missing row or disagreeing signals -> unknown"
)

#: What the re-derived ratio MEANS, per pool kind.  A reconstruction pool's
#: number is the decoder's fidelity to real chemistry; calling it a
#: fragmentation rate of the generator would be false.
MEASUREMENT_MEANING = {
    KIND_GENERATION: (
        "the sampler's free generation: the fraction of the model's own "
        "output that is ONE connected molecule -- true of BOTH free-generation "
        "writers (the trainer's own path and the 2026-10-12+ ODE path of "
        "wf_cfm_decode_dump_smiles.py)"
    ),
    KIND_RECONSTRUCTION: (
        "era A: the bond head applied to REAL tmQM geometries (the "
        "pre-2026-10-12 decode-dump path): the DECODER's fidelity to real "
        "chemistry, NOT a fragmentation rate of the generator"
    ),
    KIND_UNKNOWN: (
        "the pool's CSV does not establish which code path wrote it (no "
        "marker, contradictory markers, or a missing / disagreeing content "
        "signal); treat the numbers as unclassified"
    ),
}

#: Which artefact's number the verdict is about.
VERDICT_SCOPE = {
    KIND_GENERATION: "generator_fragmentation",
    KIND_RECONSTRUCTION: "decoder_fidelity_on_real_geometries",
    KIND_UNKNOWN: "unclassified",
}

#: Values of ``metal_z`` that count as "no metal was recorded".
_EMPTY_METAL_Z = frozenset({"", "nan", "none", "null", "-"})


@dataclass(frozen=True)
class PoolKind:
    """The classification of one pool, with the evidence that produced it.

    ``header`` is the header row AS READ (a tuple), so a reader can see
    exactly which columns were available.  ``marker`` names the signal that
    DECIDED the kind -- a header marker, the explicit ``pool_kind`` column, or
    :data:`CONTENT_MARKER` -- and is ``None`` when nothing decided it.
    ``reason`` is a human-readable sentence that names the evidence; for an
    unknown pool it says which markers were looked for and not found, or which
    two signals disagreed.
    """

    kind: str
    marker: Optional[str]
    header: Tuple[str, ...]
    reason: str
    path: Optional[Path] = None

    @property
    def is_known(self) -> bool:
        return self.kind != KIND_UNKNOWN

    def to_dict(self) -> dict:
        """JSON-safe view (the path becomes a string)."""
        return {
            "kind": self.kind,
            "marker": self.marker,
            "header": list(self.header),
            "reason": self.reason,
            "path": str(self.path) if self.path is not None else None,
        }


class UnknownPoolKindError(RuntimeError):
    """Raised by :func:`require_known_kind` when a pool cannot be classified.

    Carries the path, the header that was read and the reason, so a consumer
    can report WHY it refused instead of just failing.
    """

    def __init__(
        self,
        path: Optional[Path],
        header: Sequence[str],
        reason: str,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.header = tuple(header)
        self.reason = reason
        super().__init__(
            f"unknown pool kind: {self.path} ({reason}); "
            f"header={list(self.header)}"
        )


def _parse_sanitized(value: Optional[str]) -> Optional[int]:
    """Parse a CSV ``sanitized`` cell into 1 / 0 / None (unparseable).

    CSV fields are strings, so "1", "1.0" and " 1" all count as 1 and
    anything unparseable is NOT 1 (it is also not 0 -- it is unknown).
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number == 1.0:
        return 1
    if number == 0.0:
        return 0
    return None


def _metal_z_is_empty(value: Optional[str]) -> bool:
    """True when a ``metal_z`` cell records no metal at all."""
    return (value or "").strip().lower() in _EMPTY_METAL_Z


def _unknown(
    path: Optional[Path],
    header: Sequence[str],
    reason: str,
) -> PoolKind:
    return PoolKind(
        kind=KIND_UNKNOWN,
        marker=None,
        header=tuple(header),
        reason=reason,
        path=path,
    )


def classify_pool_header(header: Sequence[str]) -> PoolKind:
    """Classify a pool from its HEADER row alone (pure -- no I/O).

    ``header`` may be any sequence of column names; a string is NOT accepted
    as a sequence of columns (a bare string would classify itself char by
    char), so it is rejected explicitly.

    This answers only what the HEADER can answer.  A header that carries
    ``metal_z`` (without ``n_frags_raw``) is NOT decidable -- one script wrote
    one header in two eras -- so the answer is ``kind="unknown"`` with a
    reason that says the CONTENT must be read; :func:`classify_pool` does
    that.  The same holds for the explicit ``pool_kind`` column, whose VALUE
    is only in the data.
    """
    if isinstance(header, str):
        raise TypeError(
            "classify_pool_header expects a sequence of column names, not a str"
        )
    cols = tuple(header)
    if POOL_KIND_COLUMN in cols:
        return _unknown(
            None,
            cols,
            f"header carries the explicit {POOL_KIND_COLUMN!r} column: its "
            f"VALUE must be read from the first data row",
        )
    has_reconstruction = RECONSTRUCTION_MARKER in cols
    has_generation = GENERATION_MARKER in cols
    if has_reconstruction and has_generation:
        return _unknown(
            None,
            cols,
            f"header carries BOTH {RECONSTRUCTION_MARKER!r} and "
            f"{GENERATION_MARKER!r}: the two markers disagree, refusing to "
            f"bucket",
        )
    if has_generation:
        return PoolKind(
            kind=KIND_GENERATION,
            marker=GENERATION_MARKER,
            header=cols,
            reason=(
                f"marker: header carries {GENERATION_MARKER!r}: the trainer's "
                f"free-generation path"
            ),
        )
    if has_reconstruction:
        return _unknown(
            None,
            cols,
            f"marker: header carries {RECONSTRUCTION_MARKER!r}: NOT decidable "
            f"from the header, the CONTENT must be read (sanitized + "
            f"{RECONSTRUCTION_MARKER} of the first data row)",
        )
    return _unknown(
        None,
        cols,
        f"header carries neither {RECONSTRUCTION_MARKER!r} nor "
        f"{GENERATION_MARKER!r}",
    )


def _classify_from_pool_kind_column(
    path: Path,
    cols: Tuple[str, ...],
    row: Optional[Sequence[str]],
) -> PoolKind:
    """The explicit ``pool_kind`` column is authoritative (rule 1)."""
    index = cols.index(POOL_KIND_COLUMN)
    if row is None:
        return _unknown(
            path,
            cols,
            f"explicit {POOL_KIND_COLUMN!r} column: the pool has no data row "
            f"to read its value from",
        )
    value = row[index] if index < len(row) else ""
    text = (value or "").strip().lower()
    if text == KIND_GENERATION:
        return PoolKind(
            kind=KIND_GENERATION,
            marker=POOL_KIND_COLUMN,
            header=cols,
            reason=(
                f"explicit {POOL_KIND_COLUMN!r} column says "
                f"{KIND_GENERATION!r} (authoritative)"
            ),
            path=path,
        )
    if text == KIND_RECONSTRUCTION:
        return PoolKind(
            kind=KIND_RECONSTRUCTION,
            marker=POOL_KIND_COLUMN,
            header=cols,
            reason=(
                f"explicit {POOL_KIND_COLUMN!r} column says "
                f"{KIND_RECONSTRUCTION!r} (authoritative)"
            ),
            path=path,
        )
    return _unknown(
        path,
        cols,
        f"explicit {POOL_KIND_COLUMN!r} column says {text!r}, which is "
        f"neither {KIND_GENERATION!r} nor {KIND_RECONSTRUCTION!r}",
    )


def _classify_from_content(
    path: Path,
    cols: Tuple[str, ...],
    row: Optional[Sequence[str]],
) -> PoolKind:
    """Decide a ``metal_z`` pool from the CONTENT of its first data row."""
    if row is None:
        return _unknown(
            path,
            cols,
            f"content: header carries {RECONSTRUCTION_MARKER!r} but the pool "
            f"has NO data row, so sanitized + {RECONSTRUCTION_MARKER} cannot "
            f"be read",
        )
    sanitized_index = (
        cols.index("sanitized") if "sanitized" in cols else None
    )
    metal_index = cols.index(RECONSTRUCTION_MARKER)
    sanitized_raw = (
        row[sanitized_index]
        if sanitized_index is not None and sanitized_index < len(row)
        else ""
    )
    metal_raw = row[metal_index] if metal_index < len(row) else ""
    sanitized = _parse_sanitized(sanitized_raw)
    metal_empty = _metal_z_is_empty(metal_raw)
    metal_text = (metal_raw or "").strip()

    if sanitized == 1 and not metal_empty:
        return PoolKind(
            kind=KIND_RECONSTRUCTION,
            marker=CONTENT_MARKER,
            header=cols,
            reason=(
                f"content: sanitized == 1 and {RECONSTRUCTION_MARKER} = "
                f"{metal_text!r} (non-empty) -> era A, the bond head applied "
                f"to a REAL tmQM geometry"
            ),
            path=path,
        )
    if sanitized == 0 and metal_empty:
        return PoolKind(
            kind=KIND_GENERATION,
            marker=CONTENT_MARKER,
            header=cols,
            reason=(
                f"content: sanitized == 0 and {RECONSTRUCTION_MARKER} is "
                f"empty -> era B, the ODE sampler's free generation"
            ),
            path=path,
        )
    # The two signals disagree (or one of them is missing / unparseable):
    # name the disagreement, never bucket.
    if sanitized is None:
        sanitized_desc = (
            f"sanitized = {sanitized_raw!r} (absent or not 0/1)"
        )
    else:
        sanitized_desc = f"sanitized = {sanitized}"
    if metal_empty:
        metal_desc = f"{RECONSTRUCTION_MARKER} is empty"
    else:
        metal_desc = f"{RECONSTRUCTION_MARKER} = {metal_text!r} (non-empty)"
    return _unknown(
        path,
        cols,
        f"content: the two signals disagree -- {sanitized_desc} but "
        f"{metal_desc}; refusing to bucket",
    )


def classify_pool(csv_path: Path) -> PoolKind:
    """Classify the pool at ``csv_path``.  NEVER raises.

    Reads the header and AT MOST ONE data row (the rule is no longer
    header-only: one script wrote one header in two eras -- see the module
    docstring).  The body of the pool is never touched.

    Failure modes are reported, not raised:

    * unreadable (missing / permission / a directory): ``kind="unknown"``,
      ``reason="unreadable: ..."``;
    * a malformed CSV (e.g. ``csv.Error: field larger than field limit
      (131072)`` -- a pool whose header row carries a very long cell): the
      SAME unknown kind, ``reason="unreadable: ..."``.  ``csv.Error`` is NOT an
      ``OSError``, so it used to escape this function -- while the docstring
      promised it never raises -- and both C7 mains called it uncaught;
    * empty file: ``kind="unknown"``, ``reason="empty file: no header row"``;
    * a header with neither marker: ``kind="unknown"``, reason naming both;
    * a ``metal_z`` pool whose content is missing or self-contradictory:
      ``kind="unknown"``, reason naming the disagreement.
    """
    path = Path(csv_path)
    header: Tuple[str, ...] = ()
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader, []))
            if not header:
                return _unknown(path, (), "empty file: no header row")
            # The data row is read ONLY when the header cannot decide on its
            # own (an explicit pool_kind column, or a metal_z pool).
            row: Optional[Sequence[str]] = None
            if POOL_KIND_COLUMN in header or (
                RECONSTRUCTION_MARKER in header
                and GENERATION_MARKER not in header
            ):
                row = next(reader, None)
    except OSError as exc:
        return _unknown(path, header, f"unreadable: {exc.strerror or exc}")
    except csv.Error as exc:
        return _unknown(path, header, f"unreadable: csv.Error: {exc}")

    if POOL_KIND_COLUMN in header:
        return _classify_from_pool_kind_column(path, header, row)
    if RECONSTRUCTION_MARKER in header and GENERATION_MARKER in header:
        return _unknown(
            path,
            header,
            f"header carries BOTH {RECONSTRUCTION_MARKER!r} and "
            f"{GENERATION_MARKER!r}: the two markers disagree, refusing to "
            f"bucket",
        )
    if GENERATION_MARKER in header:
        return PoolKind(
            kind=KIND_GENERATION,
            marker=GENERATION_MARKER,
            header=header,
            reason=(
                f"marker: header carries {GENERATION_MARKER!r}: the trainer's "
                f"free-generation path"
            ),
            path=path,
        )
    if RECONSTRUCTION_MARKER in header:
        return _classify_from_content(path, header, row)
    return _unknown(
        path,
        header,
        f"header carries neither {RECONSTRUCTION_MARKER!r} nor "
        f"{GENERATION_MARKER!r}",
    )


def require_known_kind(csv_path: Path) -> PoolKind:
    """Classify the pool, or REFUSE.

    This is the explicit refusal act.  ``classify_pool`` never raises; this
    does, with :class:`UnknownPoolKindError`.
    """
    kind = classify_pool(csv_path)
    if kind.kind == KIND_UNKNOWN:
        raise UnknownPoolKindError(kind.path, kind.header, kind.reason)
    return kind
