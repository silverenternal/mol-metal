#!/usr/bin/env python
"""Wf-A163 branch N -- published-artifact NARRATIVE-TRUTH census.

GENERALISES the Wf-A155 finding (pool-specific CLAIMS embedded as string
literals inside a ``tag: MEASURED`` artifact) from ONE tool to the WHOLE
published tree.

THE CLASS
---------
An artifact is a machine record (numbers, keys) PLUS prose (strings).  A
number rendered in a prose string is a CLAIM.  If that claim has no
counterpart anywhere in the artifact's own machine fields, the prose is
UNBACKED: it was written by a human or by a tool constant, not derived
from the run that the artifact describes.  When the artifact is regenerated
on a different population, an unbacked prose number becomes a LIE, because
nothing recomputes it.

This scanner is deliberately INDEPENDENT of any producer's own guard: it
knows nothing about which fields are "narrative" and needs no whitelist.
It re-derives, for every artifact, the set of numbers the artifact's own
machine fields can back, and then checks every prose string against it.

RULES
-----
R1  NARRATIVE_NUMBER_NO_MACHINE_COUNTERPART
    An integer token in a prose string that appears nowhere in the
    artifact's own machine fields (numbers -- including their
    percent/scaled and rounded renderings -- keys, or any OTHER string).
R2  SELF_CONTRADICTING_PROSE_POOL
    A prose string names a pool / generated-file path that the artifact's
    own ``source_pool`` does not name, AND that pool name is not
    legitimately cross-referenced by another machine field of the same
    artifact.
R3  SELF_CONTRADICTING_PROSE_COUNT
    A prose string claims a population size ("N rows", "N samples",
    "pool of N") that contradicts the artifact's own denominator fields.
R4  SELF_CONTRADICTING_PROSE_DATE
    A prose string names a date that is not the artifact's own ``date``
    (reported as UNSURE by default: cross-referencing another run's date
    is legitimate).

CLASSIFICATION
--------------
TRUE_POSITIVE  the prose contradicts the artifact's own fields.
BENIGN         a date, a version, an identifier, a cross-reference, or a
               number that IS in a machine field but rendered differently.
UNSURE         needs a human read.

No os.walk: the tree is 201 GB.  Enumeration is Path.iterdir() + glob.

CPU-ONLY.  No torch, no CUDA, no decode, no trainer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "molmetal" / "scripts"))

try:  # AGENTS.md 5.6 -- atomic write protocol
    from wf386a3_atomic_write import atomic_write_file
except Exception:  # pragma: no cover - defensive, fallback is a plain write
    def atomic_write_file(path, content, **kw):  # type: ignore
        Path(path).write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------

ARTIFACT_DIRS = (
    ("metrics/by_metric", "by_metric"),
    ("metrics/paper_tables", "paper_tables"),
)


def enumerate_artifacts(root: Path) -> List[Path]:
    """Every JSON under the two published artifact directories.

    NEVER os.walk (201 GB repo).  Path.iterdir() + glob only.
    """
    out: List[Path] = []
    for rel, _tag in ARTIFACT_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix == ".json":
                out.append(p)
    return out


# --------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------

def flatten(obj: Any, path: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], Any]]:
    out: List[Tuple[Tuple[Any, ...], Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, path + (i,)))
    else:
        out.append((path, obj))
    return out


def path_str(path: Sequence[Any]) -> str:
    parts = []
    for p in path:
        parts.append(f"[{p}]" if isinstance(p, int) else str(p))
    return "/" + "/".join(parts)


def collect_keys(obj: Any) -> List[str]:
    keys: List[str] = []

    def rec(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                keys.append(str(k))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)

    rec(obj)
    return keys


# --------------------------------------------------------------------------
# number extraction
# --------------------------------------------------------------------------

# An integer (possibly comma-grouped) or a decimal.  Left/right guards keep
# us out of identifiers ("wfa155") and out of a trailing decimal point.
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")
# ISO-ish dates: their digits are dates, not counts.
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Identifiers that contain a digit are NAMES, not counts.
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*")
# A ratio "529/2112" is two counts; a path "a/b" is one name.
SLASH_RE = re.compile(r"[\w.\-]+/[\w.\-]+")
# Pool / generated-file names.
POOL_RE = re.compile(
    r"(?:cfm_[a-z0-9]+_[a-z0-9_]*\d{4}_\d{2}_\d{2})",
    re.IGNORECASE,
)

# Prose words that introduce a POPULATION CLAIM.
ROWCOUNT_RE = re.compile(
    r"(?P<n>\d{1,3}(?:,\d{3})+|\d+)\s+(?:rows|samples|molecules|compounds|structures|entries)\b",
    re.IGNORECASE,
)
POOLOF_RE = re.compile(
    r"pool of\s+(?P<n>\d{1,3}(?:,\d{3})+|\d+)",
    re.IGNORECASE,
)

# The UNIT VOCABULARY of a COUNT CLAIM.  A number standing immediately in
# front of one of these words is a CLAIM about a population / a size; a
# number anywhere else in prose is usually a cross-reference (a year, a
# seed index, a table number) and is not a claim this artifact must back.
UNIT_WORDS = (
    "rows", "row", "samples", "sample", "molecules", "molecule", "compounds",
    "compound", "structures", "structure", "entries", "entry", "cases", "case",
    "seeds", "seed", "folds", "fold", "epochs", "epoch", "runs", "run",
    "points", "point", "atoms", "atom", "fragments", "fragment", "ligands",
    "ligand", "poses", "pose", "files", "file", "directories", "directory",
    "configs", "config", "trials", "trial", "replicates", "replicate",
    "decodes", "decode", "generators", "generator", "checkpoints", "checkpoint",
    "examples", "example", "bins", "bin", "buckets", "bucket", "classes",
    "class", "panels", "panel", "walks", "walk", "iterations", "iteration",
    "steps", "step", "updates", "update", "participants", "subjects",
    "patients", "candidates", "hits", "hit", "errors", "error", "warnings",
    "warning", "tests", "test", "pages", "page", "lines", "line", "words",
    "word", "tokens", "token", "bytes", "byte", "kb", "mb", "gb",
    "percent", "percentage", "decoys", "metal", "metals", "elements",
    "element",
)
UNIT_RE = re.compile(
    r"(?<![\w/.-])(?:%s)\b" % "|".join(sorted(set(UNIT_WORDS), key=len, reverse=True)),
    re.IGNORECASE,
)
PCT_RE = re.compile(r"%")
# "A/B" numeric ratio and "A of B" -- both sides are counts.
RATIO_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\s*/\s*(\d{1,3}(?:,\d{3})+|\d+)(?![\w.])")
NOF_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\s+of\s+(\d{1,3}(?:,\d{3})+|\d+)(?![\w.])")


def _spans(text: str, rx: re.Pattern) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in rx.finditer(text)]


def _in_spans(a: int, b: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(s <= a and b <= e for s, e in spans)


def _slash_is_path(text: str, a: int, b: int) -> bool:
    """True if the slash-run covering [a,b) looks like a path, not a ratio."""
    m = SLASH_RE.search(text)
    if not m:
        return False
    for mm in SLASH_RE.finditer(text):
        if mm.start() <= a and b <= mm.end():
            segs = mm.group(0).split("/")
            if all(any(c.isalpha() for c in s) or "." in s for s in segs):
                return True
    return False


def extract_int_tokens(text: str) -> List[Dict[str, Any]]:
    """Integer tokens in `text` that are COUNTS (not dates, not identifiers).

    Returns a list of dicts: token, value, start, end, followed_by_pct.
    """
    date_spans = _spans(text, DATE_RE)
    ident_spans = _spans(text, IDENT_RE)
    out: List[Dict[str, Any]] = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        # decimals are not INTEGER tokens
        if m.group(2) is not None:
            continue
        s, e = m.start(), m.end()
        if _in_spans(s, e, date_spans):
            continue
        if _in_spans(s, e, ident_spans):
            continue
        if _slash_is_path(text, s, e):
            continue
        try:
            val = int(tok.replace(",", ""))
        except ValueError:
            continue
        tail = text[e:e + 3]
        out.append(
            {
                "token": tok,
                "value": val,
                "start": s,
                "end": e,
                "followed_by_pct": bool(re.match(r"\s*%", tail)),
            }
        )
    return out


def extract_float_tokens(text: str) -> List[Dict[str, Any]]:
    date_spans = _spans(text, DATE_RE)
    ident_spans = _spans(text, IDENT_RE)
    out: List[Dict[str, Any]] = []
    for m in NUM_RE.finditer(text):
        if m.group(2) is None:
            continue
        tok = m.group(0)
        s, e = m.start(), m.end()
        if _in_spans(s, e, date_spans) or _in_spans(s, e, ident_spans):
            continue
        if _slash_is_path(text, s, e):
            continue
        try:
            val = float(tok)
        except ValueError:
            continue
        out.append({"token": tok, "value": val, "start": s, "end": e})
    return out


# --------------------------------------------------------------------------
# backing
# --------------------------------------------------------------------------

def _standalone(token: str, text: str) -> bool:
    try:
        pat = re.compile(r"(?<![0-9.,])" + re.escape(token) + r"(?![0-9.,])")
    except re.error:  # pragma: no cover - defensive
        return token in text
    return bool(pat.search(text))


def _number_backings(value: float) -> List[float]:
    """Every way a machine float can legitimately RENDER as an integer.

    Deliberately CONSERVATIVE.  A scaled form (value*100) is only accepted
    for a token that is rendered as a PERCENT in the prose; otherwise every
    float in the artifact would "back" an arbitrary integer and the
    scanner would stop seeing anything.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return []  # NaN / inf back nothing
    out = [value]
    if abs(value - round(value)) < 1e-12:
        out.append(float(round(value)))
    return out


def _int_backed(
    val: int,
    token: str,
    followed_by_pct: bool,
    machine_numbers: Sequence[float],
) -> bool:
    """Is this integer token backed by a machine NUMBER of the artifact?"""
    for m in machine_numbers:
        if m != m:  # NaN
            continue
        if abs(m - val) <= 1e-9:
            return True
        # a rounded rendering of a machine number
        if abs(m - val) <= 0.5:
            return True
        # a PERCENT rendering of a machine number ("8.51 %" <- 0.0850963)
        if followed_by_pct and abs(m * 100.0 - val) <= 0.5:
            return True
    return False


def _float_matches(token_value: float, decimals: int, machine_numbers: Sequence[float]) -> bool:
    tol = 0.5 * (10.0 ** (-decimals)) if decimals else 0.0
    for m in machine_numbers:
        if abs(m - token_value) <= tol + 1e-12:
            return True
        if abs(m * 100.0 - token_value) <= tol + 1e-12:
            return True
        if abs(m * 1000.0 - token_value) <= tol + 1e-12:
            return True
        if abs(m - token_value) <= 0.5 * (10.0 ** (-max(decimals, 0))) + 1e-12:
            return True
    return False


# --------------------------------------------------------------------------
# artifact-level analysis
# --------------------------------------------------------------------------

SOURCE_POOL_KEYS = ("source_pool", "pool", "pool_path", "csv", "csv_path",
                     "input_csv", "source_csv", "generated_csv")


def _own_source_pool(obj: Any) -> List[str]:
    """The pool name(s) this artifact claims as ITS OWN population."""
    found: List[str] = []
    if not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        kl = str(k).lower()
        if kl not in SOURCE_POOL_KEYS:
            continue
        if isinstance(v, str):
            found.append(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    found.append(vv)
    return found


def _pool_names(text: str) -> List[str]:
    return POOL_RE.findall(text)


def _denominator_values(obj: Any) -> Dict[str, int]:
    """Denominator-ish machine numbers of the artifact."""
    out: Dict[str, int] = {}

    def rec(o: Any, path: Tuple[Any, ...]) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, path + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                rec(v, path + (i,))
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            leaf = path[-1] if path else ""
            lk = str(leaf).lower()
            if any(
                k in lk
                for k in ("n_rows", "n_samples", "n_total", "n_analyzed",
                          "n_data_rows", "n_examples", "n_instances",
                          "n_compounds", "n_molecules", "denominator",
                          "n_pool", "n_records")
            ):
                try:
                    out[path_str(path)] = int(o)
                except (TypeError, ValueError):
                    pass

    rec(obj, ())
    return out


GENERIC_WORDS = {
    # English function words: they must never act as a discriminator.
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "there", "here", "when", "where", "while",
    "who", "which", "what", "why", "how", "for", "from", "into", "onto",
    "over", "under", "with", "without", "within", "about", "above", "below",
    "only", "just", "also", "very", "more", "most", "some", "such", "own",
    "same", "other", "another", "each", "every", "both", "all", "any",
    "none", "no", "not", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "its", "it", "in", "on", "at",
    "by", "as", "so", "we", "our", "you", "your", "they", "their", "them",
    "he", "she", "his", "her", "up", "out", "off", "down", "again", "per",
    # count-unit nouns that are never discriminators.
    "rows", "row", "samples", "sample", "molecules", "molecule", "compounds",
    "structures", "entries", "entry", "cases", "case", "seeds", "seed",
    "folds", "fold", "epochs", "epoch", "runs", "run", "points", "point",
    "atoms", "atom", "fragments", "fragment", "ligands", "ligand", "files",
    "file", "n", "count", "counts", "total", "num", "number", "size",
    "examples", "example", "bins", "bin", "classes", "class", "trials",
    "trial", "iterations", "iteration", "steps", "step", "percent",
    "percentage", "pool", "pools", "decode", "decodes", "hits", "hit",
    "errors", "error", "warnings", "warning", "tests", "test", "mean",
    "median", "min", "max", "std", "value", "values",
    # file extensions / generic path tokens
    "csv", "json", "md", "txt", "pt", "npz", "yaml", "yml", "png", "pdf",
    "html", "htm", "tar", "gz", "log",
}


def _key_words(key: str) -> set:
    return {
        w for w in re.split(r"[^A-Za-z0-9]+", str(key))
        if w and not w.isdigit()
    }


def _distinctive_key_map(obj: Any) -> Dict[str, List[Tuple[str, int]]]:
    """key-word -> [(key, int value)] for words that identify ONE key only.

    A word that appears in several keys ("rows") is not a discriminator and
    is dropped; a word that appears in exactly one key ("halide") is.
    """
    word_map: Dict[str, List[Tuple[str, int]]] = {}
    def rec(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        if v != v or v in (float("inf"), float("-inf")):
                            iv = None
                        else:
                            iv = int(v)
                    except (TypeError, ValueError, OverflowError):
                        iv = None
                    if iv is not None:
                        for w in _key_words(k):
                            word_map.setdefault(w.lower(), []).append((str(k), iv))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)
    rec(obj)
    return {
        w: hits for w, hits in word_map.items()
        if w not in GENERIC_WORDS and len({k for k, _v in hits}) == 1
    }


def _key_word_sets(obj: Any) -> List[Tuple[str, frozenset, int]]:
    """(key, non-generic word set, int value) for every int-valued key.

    A key qualifies as a SIBLING candidate only when it carries at least
    TWO non-generic words: a one-word key ("n_train") matches far too many
    prose windows to be a discriminator.
    """
    out: List[Tuple[str, frozenset, int]] = []

    def rec(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        if v != v or v in (float("inf"), float("-inf")):
                            iv = None
                        else:
                            iv = int(v)
                    except (TypeError, ValueError, OverflowError):
                        iv = None
                    if iv is not None:
                        words = frozenset(
                            w.lower()
                            for w in _key_words(k)
                            if w.lower() not in GENERIC_WORDS and len(w) >= 3
                        )
                        if len(words) >= 2:
                            out.append((str(k), words, iv))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)

    rec(obj)
    return out


CLAUSE_BREAK_RE = re.compile(r"[.;,)\]}|]|\s--\s|\s-\s")


def _clause_window(text: str, start: int, limit: int = 90) -> str:
    """The noun phrase that follows a number, up to the next clause break."""
    tail = text[start:start + limit]
    m = CLAUSE_BREAK_RE.search(tail)
    if m:
        tail = tail[: m.start()]
    return tail.lower()


def _pair_values(text: str) -> Dict[int, int]:
    """token offset -> the value of its partner in an "A of B" / "A/B" pair."""
    out: Dict[int, int] = {}
    for rx in (RATIO_RE, NOF_RE):
        for m in rx.finditer(text):
            try:
                a = int(m.group(1).replace(",", ""))
                b = int(m.group(2).replace(",", ""))
            except ValueError:
                continue
            out[m.start(1)] = b
            out[m.start(2)] = a
    return out


def _in_range(text: str, pos: int) -> bool:
    """True if the token at `pos` is one end of a range ("92.7-94 %")."""
    before = text[max(0, pos - 4):pos]
    after = text[pos:pos + 6]
    if re.search(r"[\d.]\s*[-–—]\s*$", before):
        return True
    if re.match(r"\s*[-–—]\s*[\d.]", after):
        return True
    return False


def _count_flagged_here(flags: List[Dict[str, Any]], pstr: str, token: str) -> bool:
    return any(
        f["rule"] == "SELF_CONTRADICTING_PROSE_COUNT"
        and f["path"] == pstr
        and str(f["token"]) == str(token)
        for f in flags
    )


_ARTIFACT_REF_RE = re.compile(
    r"[\w./\-]*\.(?:json|csv|md|npz|pt|yaml)\b|metrics/by_metric/|\bjson\b",
    re.IGNORECASE,
)
_THRESHOLD_PATH_RE = re.compile(
    r"gate|threshold|design|protocol|config|hyperparam|requirement", re.IGNORECASE
)


def _names_another_artifact(clause: str) -> bool:
    """True if the clause names a DIFFERENT artifact / file."""
    return bool(_ARTIFACT_REF_RE.search(clause))


def _is_threshold_field(path: str) -> bool:
    """True if the field is a protocol gate / design threshold."""
    return bool(_THRESHOLD_PATH_RE.search(path))


def analyze_artifact(path: Path, root: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except Exception as exc:  # unreadable / not JSON
        return {
            "path": str(path.relative_to(root)),
            "error": f"{type(exc).__name__}: {exc}",
            "flags": [],
            "n_flags": 0,
        }

    leaves = flatten(obj)
    machine_numbers: List[float] = []
    machine_strings: List[str] = []
    for _p, v in leaves:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            machine_numbers.append(float(v))
        elif isinstance(v, str):
            machine_strings.append(v)
    machine_keys = collect_keys(obj)

    # integers that any machine number can back (exact / integral values only)
    machine_int_backings: set = set()
    for m in machine_numbers:
        for b in _number_backings(m):
            if abs(b - round(b)) < 1e-9:
                machine_int_backings.add(int(round(b)))

    # all pool names that appear ANYWHERE as a machine value (cross-refs).
    # NOTE: the string under test must be EXCLUDED, otherwise a prose string
    # that misattributes the pool would "legitimately" cross-reference
    # itself.
    machine_pool_names: set = set()
    for s in machine_strings:
        for nm in _pool_names(s):
            machine_pool_names.add(nm)

    own_pool_texts = _own_source_pool(obj)
    own_pool_names: set = set()
    for t in own_pool_texts:
        own_pool_names.update(_pool_names(t))
        # also the directory component of a path
        for seg in re.split(r"[\\/]", t):
            if seg:
                own_pool_names.update(_pool_names(seg))

    denominators = _denominator_values(obj)
    denom_values = set(denominators.values())
    if denom_values:
        denom_values.add(max(denom_values))
    key_word_sets = _key_word_sets(obj)

    flags: List[Dict[str, Any]] = []
    loose_unbacked = 0

    for p, v in leaves:
        if not isinstance(v, str):
            continue
        text = v
        pstr = path_str(p)

        # ---------------- R1: unbacked integer token ----------------
        claims = claim_positions(text)
        for tok in extract_int_tokens(text):
            val = tok["value"]
            backed_by = []
            if _int_backed(
                val, tok["token"], tok["followed_by_pct"], machine_numbers
            ):
                backed_by.append("MACHINE_NUMBER")
            if any(_standalone(tok["token"], ks) for ks in machine_keys):
                backed_by.append("MACHINE_KEY")
            if any(
                _standalone(tok["token"], ms) for ms in machine_strings if ms != text
            ):
                backed_by.append("OTHER_STRING")
            if backed_by:
                continue
            is_claim = tok["start"] in claims
            if not is_claim:
                loose_unbacked += 1
                continue
            flags.append(
                {
                    "rule": "NARRATIVE_NUMBER_NO_MACHINE_COUNTERPART",
                    "path": pstr,
                    "token": tok["token"],
                    "value": val,
                    "excerpt": _excerpt(text, tok["start"]),
                    "classification": "UNSURE",
                    "reason": "no machine field of this artifact backs this number",
                }
            )

        # ---------------- R1b: unbacked PERCENT claim ----------------
        # "8.51 %" is the canonical rendering of a machine float 0.0851.
        for ftok in extract_float_tokens(text):
            tail = text[ftok["end"]:ftok["end"] + 3]
            if not re.match(r"\s*%", tail):
                continue
            decimals = len(ftok["token"].split(".")[1]) if "." in ftok["token"] else 0
            if _float_matches(ftok["value"], decimals, machine_numbers):
                continue
            if any(ftok["token"] in ms for ms in machine_strings if ms != text):
                continue
            flags.append(
                {
                    "rule": "NARRATIVE_PERCENT_NO_MACHINE_COUNTERPART",
                    "path": pstr,
                    "token": ftok["token"],
                    "value": ftok["value"],
                    "excerpt": _excerpt(text, ftok["start"]),
                    "classification": "UNSURE",
                    "reason": (
                        "a percentage is rendered in prose but no machine float of "
                        "this artifact renders to it (checked value, value*100, "
                        "value*1000, and rounded forms)"
                    ),
                }
            )

        # ---------------- R5: semantic sibling mismatch ----------------
        # The prose says "The 60 rows with a free halide counterion"; the
        # artifact carries a machine key "n_with_free_halide_counterion".
        # If the prose number disagrees with the machine field that the
        # prose itself names -- i.e. EVERY non-generic word of the key
        # occurs in the prose window -- one of the two is wrong.
        if key_word_sets:
            pair_values = _pair_values(text)
            for tok in extract_int_tokens(text):
                if tok["start"] not in claims:
                    continue
                if _in_range(text, tok["start"]):
                    continue
                if _count_flagged_here(flags, pstr, tok["token"]):
                    continue
                window = _clause_window(text, tok["end"])
                window = re.sub(r"[\w./\-]*\.(?:csv|json|md|txt|pt|npz|yaml)\b", " ", window)
                window = re.sub(r"\b[a-z][a-z0-9_]*_\d[a-z0-9_\-]*\b", " ", window)
                wwords = {w for w in re.split(r"[^A-Za-z0-9]+", window) if w}
                hits = [
                    (hk, words, hv)
                    for hk, words, hv in key_word_sets
                    if words and words <= wwords
                ]
                if not hits:
                    continue
                sibling_values = {hv for _hk, _w, hv in hits}
                if tok["value"] in sibling_values:
                    continue
                # "44 of 12668" -- the sibling may hold EITHER side
                if pair_values.get(tok["start"]) in sibling_values:
                    continue
                flags.append(
                    {
                        "rule": "SELF_CONTRADICTING_PROSE_SIBLING",
                        "path": pstr,
                        "token": tok["token"],
                        "value": tok["value"],
                        "excerpt": _excerpt(text, tok["start"]),
                        "classification": "UNSURE",
                        "reason": (
                            "the prose names a discriminator that the artifact "
                            f"reports as {sorted(sibling_values)}, not "
                            f"{tok['value']}"
                        ),
                        "sibling_keys": sorted({hk for hk, _w, _v in hits}),
                    }
                )

        # ---------------- R2: pool contradiction ----------------
        if own_pool_names:
            other_pool_names: set = set()
            for ms in machine_strings:
                if ms == text:
                    continue
                other_pool_names.update(_pool_names(ms))
            for nm in set(_pool_names(text)):
                if nm in own_pool_names:
                    continue
                if nm in other_pool_names:
                    continue  # legitimate cross-reference to another artifact
                flags.append(
                    {
                        "rule": "SELF_CONTRADICTING_PROSE_POOL",
                        "path": pstr,
                        "token": nm,
                        "value": nm,
                        "excerpt": _excerpt(text, text.find(nm)),
                        "classification": "TRUE_POSITIVE",
                        "reason": (
                            "prose names pool "
                            f"{nm!r}, but this artifact's own source_pool names "
                            f"{sorted(own_pool_names)!r} and no other machine field "
                            "of this artifact references it"
                        ),
                    }
                )

        # ---------------- R3: population-count contradiction ----------------
        if denom_values:
            for m in list(ROWCOUNT_RE.finditer(text)) + list(POOLOF_RE.finditer(text)):
                raw = m.group("n")
                try:
                    n = int(raw.replace(",", ""))
                except ValueError:
                    continue
                if n in denom_values:
                    continue
                if _int_backed(n, raw, False, machine_numbers):
                    continue  # the artifact DOES report this count elsewhere
                clause = _clause_window(text, max(0, m.start("n") - 60))
                if _names_another_artifact(clause):
                    continue  # a cross-reference to another artifact's population
                if _is_threshold_field(pstr):
                    continue  # a protocol gate / threshold, not a measurement
                flags.append(
                    {
                        "rule": "SELF_CONTRADICTING_PROSE_COUNT",
                        "path": pstr,
                        "token": raw,
                        "value": n,
                        "excerpt": _excerpt(text, m.start("n")),
                        "classification": "TRUE_POSITIVE",
                        "reason": (
                            f"prose claims a population of {n} but this artifact's "
                            f"own denominators are {sorted(denom_values)}"
                        ),
                    }
                )

        # ---------------- R4: date contradiction ----------------
        own_date = None
        if isinstance(obj, dict) and isinstance(obj.get("date"), str):
            own_date = obj["date"]
        if own_date:
            for m in DATE_RE.finditer(text):
                if m.group(0) != own_date:
                    if m.group(0) in machine_strings:
                        continue
                    flags.append(
                        {
                            "rule": "SELF_CONTRADICTING_PROSE_DATE",
                            "path": pstr,
                            "token": m.group(0),
                            "value": m.group(0),
                            "excerpt": _excerpt(text, m.start()),
                            "classification": "UNSURE",
                            "reason": (
                                f"prose names date {m.group(0)} but the artifact's own "
                                f"date field is {own_date} (may be a legitimate "
                                "cross-reference to another run)"
                            ),
                        }
                    )

    # de-duplicate: one flag per (rule, path, token)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for f in flags:
        key = (f["rule"], f["path"], str(f["token"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    flags = deduped
    flags.sort(key=lambda f: (f["rule"], str(f["path"]), str(f["token"])))
    return {
        "path": str(path.relative_to(root)),
        "n_machine_numbers": len(machine_numbers),
        "n_machine_strings": len(machine_strings),
        "n_machine_keys": len(machine_keys),
        "n_strings": sum(1 for _p, v in leaves if isinstance(v, str)),
        "n_loose_unbacked_integers": loose_unbacked,
        "n_flags": len(flags),
        "flags": flags,
    }


def _excerpt(text: str, pos: int) -> str:
    if pos is None or pos < 0:
        pos = 0
    a = max(0, pos - 60)
    b = min(len(text), pos + 90)
    return text[a:b].replace("\n", " ")


def claim_positions(text: str) -> set:
    """Offsets of integer tokens that stand in a COUNT-CLAIM position.

    A claim position is a number that stands immediately in front of a unit
    word ("2112 rows"), in front of a percent sign ("8.51 %"), on either
    side of a ratio ("529/2112") or of an "N of M" construction ("44 of
    12668").  A number anywhere else in prose is normally a
    cross-reference (a year, a seed index, a table number, a citation),
    and demanding that the artifact back it would make the scanner cry
    wolf.
    """
    pos: set = set()
    unit_spans = [(m.start(), m.end()) for m in UNIT_RE.finditer(text)]
    for m in NUM_RE.finditer(text):
        s, e = m.start(), m.end()
        for us, ue in unit_spans:
            if e <= us and us - e <= 3:
                pos.add(s)
                break
    for m in PCT_RE.finditer(text):
        # the number immediately before the percent sign
        best = None
        for mm in NUM_RE.finditer(text):
            if mm.end() <= m.start() and m.start() - mm.end() <= 2:
                if best is None or mm.end() > best.end():
                    best = mm
        if best is not None:
            pos.add(best.start())
    for rx in (RATIO_RE, NOF_RE):
        for m in rx.finditer(text):
            for g in (1, 2):
                pos.add(m.start(g))
    return pos


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# counterfactual: the PRE-repair c7 artifact
# --------------------------------------------------------------------------
#
# The pre-repair c7 artifact is NOT on disk: it was overwritten by the
# Wf-A162 repair and never committed (both files are untracked).  What IS
# preserved is (a) the pre-repair c8 artifact, whose narrative strings ARE
# the pool-specific literals the old tool emitted, and (b) the Wf-A162
# report's own table of the seven literal sites
# (molmetal/reports/wfa162_c7_pool_ligand_connected_rate/report.md, §3).
#
# The counterfactual below is a MECHANICAL substitution: the repaired c7
# artifact (all machine fields) with the pre-repair narrative strings
# (copied verbatim from the preserved pre-repair c8 artifact) put back at
# the corresponding sites.  It is a COUNTERFACTUAL, NOT the original: the
# original pre-repair c7 JSON no longer exists anywhere on disk.

COUNTERFACTUAL_SITES = (
    (("purpose",), ("purpose",)),
    (("not_comparable_to", "end_to_end_median_headline"),
     ("not_comparable_to", "8_51_percent_headline")),
    (("not_comparable_to", "other_pools_single_pool_counts"),
     ("not_comparable_to", "25_05_percent_single_pool_fragment_count")),
    (("comparability", "breaks", 0, "ours"), ("comparability", "breaks", 0, "ours")),
    (("caveats", 2), ("caveats", 2)),
    (("caveats", 4), ("caveats", 4)),
    (("comparability", "statement"), ("comparability", "statement")),
)

# Keys the Wf-A162 REPAIR ADDED, measured by diffing the key sets of the
# preserved pre-repair c8 artifact against the repaired c8 artifact
# (metrics/by_metric/wfa155_ligand_connected_rate.json).  The pre-repair
# tool did not emit them, so a faithful pre-repair c7 must not carry them.
COUNTERFACTUAL_DROP_TOP = ("end_to_end_headline", "published_cross_check",
                           "narrative_self_consistency")
COUNTERFACTUAL_DROP_COMPARABILITY = ("frontier_denominator", "n_breaks")


def _get(d: Any, path: Sequence[Any]) -> Any:
    cur = d
    for part in path:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _set(d: Any, path: Sequence[Any], value: Any) -> None:
    cur = d
    for part in path[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    last = path[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def build_counterfactual(repaired_c7: Path, pre_repair_c8: Path) -> Dict[str, Any]:
    new = json.loads(repaired_c7.read_text(encoding="utf-8"))
    pre = json.loads(pre_repair_c8.read_text(encoding="utf-8"))
    reverted = []
    for new_path, pre_path in COUNTERFACTUAL_SITES:
        try:
            pre_text = _get(pre, pre_path)
        except Exception:
            continue
        try:
            _get(new, new_path)
        except Exception:
            continue
        _set(new, new_path, pre_text)
        reverted.append(
            {
                "site": ".".join(map(str, new_path)),
                "pre_repair_source": ".".join(map(str, pre_path)),
                "text": pre_text,
            }
        )
    # restore the PRE-repair key names under not_comparable_to
    nc = new.get("not_comparable_to")
    if isinstance(nc, dict):
        for old_key in ("8_51_percent_headline", "25_05_percent_single_pool_fragment_count"):
            try:
                nc[old_key] = pre["not_comparable_to"][old_key]
            except Exception:
                continue
            for new_key in ("end_to_end_median_headline", "other_pools_single_pool_counts"):
                nc.pop(new_key, None)
    # drop the keys the REPAIR added (measured pre/post c8 key-set diff)
    for k in COUNTERFACTUAL_DROP_TOP:
        new.pop(k, None)
    comp = new.get("comparability")
    if isinstance(comp, dict):
        for k in COUNTERFACTUAL_DROP_COMPARABILITY:
            comp.pop(k, None)
    notice = (
        "COUNTERFACTUAL -- NOT the original artifact. The pre-repair c7 JSON "
        "does not exist on disk (it was overwritten by the Wf-A162 repair and "
        "was never committed). This file = the REPAIRED c7 artifact with the "
        "pre-repair narrative strings restored verbatim from the preserved "
        "pre-repair c8 artifact "
        "(wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json), at the "
        "sites listed in wfa162 report.md section 3, and with the keys the "
        "repair ADDED removed (measured by diffing the pre-repair c8 key set "
        "against the repaired c8 key set). Machine fields are the c7 run's "
        "own MEASURED values and are unchanged."
    )
    return new, {"notice": notice, "reverted_sites": reverted}


# --------------------------------------------------------------------------
# TOOL-LAYER scan
# --------------------------------------------------------------------------
#
# The artifact-level census above can only see the RESIDUE of the Wf-A155
# defect class (a prose number the artifact does not record).  The defect
# itself lives one level up, in the PRODUCER: a tool that accepts an
# arbitrary --pool yet writes pool-specific literals into its output.  The
# artifact it emits for the DEFAULT pool is self-consistent (so the
# artifact census sees nothing), but the tool is ARMED: point it at
# another pool and it emits false statements in a `tag: MEASURED` artifact.
#
# Mechanical rule: a script under molmetal/scripts/ that (a) exposes a
# --pool argument AND (b) contains a pool-specific literal AFTER its first
# top-level `def` (i.e. in code, not in the module docstring) is ARMED.

TOOL_LITERAL_MARKERS = (
    "2112",
    "2,112",
    "cfm_c8_fullgen_2026_10_24",
    "cfm_c7_fullgen_2026_10_20",
    "704 rows",
    "529/2112",
    "2082/2112",
    "30/2112",
)


def scan_tools(root: Path) -> Dict[str, Any]:
    scripts_dir = root / "molmetal/scripts"
    armed: List[Dict[str, Any]] = []
    checked = 0
    self_name = Path(__file__).name
    for p in sorted(scripts_dir.glob("*.py")):
        checked += 1
        if p.name == self_name:
            continue  # this scanner names the markers on purpose
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if '"--pool"' not in text and "'--pool'" not in text:
            continue
        lines = text.split("\n")
        doc_end = 0
        try:
            import ast
            tree = ast.parse(text)
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                doc_end = tree.body[0].end_lineno or 0
        except Exception:
            doc_end = 0
        hits = []
        for i, line in enumerate(lines, 1):
            for marker in TOOL_LITERAL_MARKERS:
                idx = line.find(marker)
                if idx < 0:
                    continue
                # is the marker inside a string literal?  (quote parity)
                before = line[:idx]
                q = before.count('"') + before.count("'")
                in_string = bool(q % 2 == 1)
                # a pool PATH constant is a legitimate default INPUT, not a
                # prose claim; a prose claim is the number in a sentence.
                # the enclosing string literal of the marker
                lq = line.rfind('"', 0, idx)
                rq = line.find('"', idx)
                enclosing = line[lq + 1:rq] if (lq >= 0 and rq > 0) else ""
                is_path_default = bool(
                    re.search(r"DEFAULT_|_REL\b|POOL_DIR|RUN_DIR|_CSV\b", line)
                    or enclosing.strip().startswith(("molmetal/", "reports/", "/home/", "./"))
                    or ".csv" in enclosing
                    or "generated_smiles.csv" in enclosing
                    or line[idx + len(marker):idx + len(marker) + 1] == "/"
                    or (
                        marker.lower().startswith("cfm_")
                        and " " not in enclosing.strip()
                    )
                )
                hits.append(
                    {
                        "line": i,
                        "marker": marker,
                        "in_code": i > doc_end,
                        "in_string_literal": in_string,
                        "is_path_default": is_path_default,
                        "text": line.strip()[:160],
                    }
                )
        code_hits = [
            h for h in hits
            if h["in_code"] and h["in_string_literal"] and not h["is_path_default"]
        ]
        if not code_hits:
            continue
        armed.append(
            {
                "script": str(p.relative_to(root)),
                "has_pool_argument": True,
                "n_literal_sites": len(code_hits),
                "literal_sites": code_hits,
            }
        )
    return {
        "scripts_checked": checked,
        "armed_scripts": len(armed),
        "armed": armed,
    }


def run(root: Path, paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    paths = paths if paths is not None else enumerate_artifacts(root)
    artifacts = [analyze_artifact(p, root) for p in paths]
    per_rule: Dict[str, int] = {}
    per_class: Dict[str, int] = {}
    flagged_artifacts = 0
    for a in artifacts:
        if a.get("error"):
            continue
        if a["n_flags"]:
            flagged_artifacts += 1
        for f in a["flags"]:
            per_rule[f["rule"]] = per_rule.get(f["rule"], 0) + 1
            per_class[f["classification"]] = per_class.get(f["classification"], 0) + 1
    ok = [a for a in artifacts if not a.get("error")]
    tools = scan_tools(root)
    return {
        "tool": "wfa163_narrative_truth_scan",
        "wave": "Wf-A163 branch N",
        "date": "2026-11-01",
        "cpu_only": True,
        "gpu": "0 (CPU-only; no torch import, no decode, no trainer)",
        "method": {
            "enumeration": (
                "Path.iterdir() over metrics/by_metric/ and metrics/paper_tables/ "
                "-- NEVER os.walk (201 GB repo)"
            ),
            "rules": [
                "R1 NARRATIVE_NUMBER_NO_MACHINE_COUNTERPART: integer token in a "
                "prose string (dates and identifiers excluded) with no counterpart "
                "in the artifact's own machine numbers / keys / other strings",
                "R2 SELF_CONTRADICTING_PROSE_POOL: prose names a pool the artifact's "
                "own source_pool does not name and no other machine field "
                "cross-references",
                "R3 SELF_CONTRADICTING_PROSE_COUNT: prose claims a population size "
                "contradicting the artifact's own denominator fields",
                "R4 SELF_CONTRADICTING_PROSE_DATE: prose names a date other than the "
                "artifact's own date (UNSURE by default)",
            ],
            "independence": (
                "no whitelist of narrative fields; the scanner re-derives the "
                "backing set from each artifact itself"
            ),
            "tool_layer": (
                "a script under molmetal/scripts/ that exposes a --pool "
                "argument AND carries a pool-specific literal in code (after "
                "its first top-level def) is ARMED: its artifact for the "
                "default pool is self-consistent, but a re-run on another "
                "pool emits false statements"
            ),
        },
        "denominator": {
            "artifacts_enumerated": len(paths),
            "artifacts_parsed": len(ok),
            "artifacts_unreadable": len(artifacts) - len(ok),
            "artifact_dirs": [d for d, _t in ARTIFACT_DIRS],
        },
        "counts": {
            "artifacts_with_flags": flagged_artifacts,
            "artifacts_clean": len(ok) - flagged_artifacts,
            "total_flags": sum(per_rule.values()),
            "flags_by_rule": per_rule,
            "flags_by_classification": per_class,
        },
        "tool_layer_scan": tools,
        "artifacts": artifacts,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_REPO_ROOT))
    ap.add_argument(
        "--out",
        default=str(
            _REPO_ROOT
            / "molmetal/reports/wfa163_narrative_truth_scan/narrative_truth_scan.json"
        ),
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--counterfactual",
        action="store_true",
        help="build + scan the COUNTERFACTUAL pre-repair c7 artifact "
        "(repaired c7 machine fields + pre-repair narrative literals "
        "restored from the preserved pre-repair c8 file)",
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="scan these explicit JSON files instead of the published tree "
        "(used for the Wf-A162 cross-check on the two repaired artifacts)",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if args.counterfactual:
        rep = root / "molmetal/reports/wfa162_c7_pool_ligand_connected_rate"
        cf_path = (
            root
            / "molmetal/reports/wfa163_narrative_truth_scan"
            / "counterfactual_pre_repair_c7.json"
        )
        cf, meta = build_counterfactual(
            rep / "ligand_connected_rate_c7.json",
            rep / "wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json",
        )
        atomic_write_file(cf_path, json.dumps(cf, indent=2, ensure_ascii=False) + "\n")
        meta_path = cf_path.with_name("counterfactual_pre_repair_c7.meta.json")
        atomic_write_file(
            meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )
        rep_cf = run(root, [cf_path])
        rep_cf["counterfactual"] = {
            "path": str(cf_path.relative_to(root)),
            "meta_path": str(meta_path.relative_to(root)),
            "notice": meta["notice"],
            "reverted_sites": meta["reverted_sites"],
            "counts": rep_cf["counts"],
            "flags": rep_cf["artifacts"][0]["flags"],
        }
        atomic_write_file(
            Path(args.out),
            json.dumps(rep_cf, indent=2, ensure_ascii=False) + "\n",
        )
        if not args.quiet:
            print(f"counterfactual written: {cf_path}")
            print(f"counterfactual flags: {rep_cf['counts']['total_flags']} "
                  f"{rep_cf['counts']['flags_by_rule']}")
            for f in rep_cf["artifacts"][0]["flags"]:
                print(f"  {f['rule']} token={f['token']} path={f['path']}")
        return 0
    explicit = None
    if args.files:
        explicit = []
        for f in args.files:
            p = Path(f)
            if not p.is_absolute():
                p = root / p
            explicit.append(p)
    report = run(root, explicit)
    out = Path(args.out)
    atomic_write_file(out, json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False) + "\n")
    if not args.quiet:
        c = report["counts"]
        d = report["denominator"]
        print(f"scanned {d['artifacts_parsed']} artifacts "
              f"({d['artifacts_unreadable']} unreadable) under {d['artifact_dirs']}")
        print(f"artifacts with >=1 flag: {c['artifacts_with_flags']} / {d['artifacts_parsed']}")
        print(f"total flags: {c['total_flags']}  by rule: {c['flags_by_rule']}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
