#!/usr/bin/env python3
"""Wf-A199 -> Wf-A204 -- repo-wide producer<->artefact parity census for dated
rule-#6 records.

WHAT THIS IS
------------
A MEASUREMENT.  It fixes nothing.  It enumerates every JSON artefact under
``molmetal/reports/**`` and ``metrics/by_metric/**``, collects the JSON KEYS
that name a dated AGENTS.md §5.5 rule-#6 record, and then asks -- for each such
key -- whether a PRODUCER (a ``.py`` script, i.e. anything that can regenerate
the artefact) contains that key name.  A dated rule-#6 record that lives in a
published artefact but in NO producer is a parity gap: a re-run of the producer
silently DROPS the dated history, which is the AGENTS.md §5.5 rule-#6
violation described in the honest-negative catalogue 6.268.4.

A HAND-AUTHORED artefact (one with no producer at all) is NOT a parity defect
and this census says so explicitly in the classification.

THE COVERAGE BOUND (Wf-A204)
---------------------------
A count without a bound is not a measurement (honest-negative catalogue
6.244 / 6.268 O12).  This tool therefore PUBLISHES, in ``report.json`` under
``coverage_bound``, the exact rules it ran on:

  * the CORPUS       -- which files were scanned, how they were enumerated,
                        how many bytes, and a manifest digest so the counts
                        are reproducible across runs;
  * the MATCH RULES  -- ``MATCH_RULES`` below, the ONE place they are defined;
  * the DESCENT RULE -- when the walk stops descending into a matched key;
  * the PATH-SCOPING RULE -- how a producer is attached to an artefact, and
                        how a producer's value site is matched to a record;
  * the VALUE-PARITY RULE -- what "equal" / "differs" / "not-comparable" mean
                        and what happens to a producer value that is a runtime
                        expression (it gets its OWN bucket; it is never
                        silently absorbed into "PRODUCER-EMITTED").

KEY-NAME MATCHING (the documented pattern set)
----------------------------------------------
A JSON key name is a "dated rule-#6 record candidate" when it satisfies ANY of:

  R1  it contains a token from SUPERSESSION_TOKENS
      (supersede*/superseded/demoted/demotion/retract*/historical/obsolete/
      deprecated/legacy), compared case-insensitively -- so ``SUPERSEDED_BY``,
      ``HISTORICAL`` and ``RETRACTED_2026_10_11`` all match;
  R2  it contains the token ``closed`` AND a date-shaped token
      (``YYYY-MM-DD`` or ``YYYY_MM_DD``) -- the task brief's ``*_closed_2026_*``;
  R3  it contains a date-shaped token AND a supersession/demotion WORD
      (substring, not token) -- the task brief's "date token + supersession word";
  R4  it contains a date-shaped token.  ADDED 2026-11-02 by Wf-A204.

WHY R4 EXISTS (the D1 defect, MEASURED)
----------------------------------------
R1/R2/R3 are a SUPERSESSION-SEMANTICS filter.  This tool's own ``TIER_ENUM``
defines the DATED tier as "the key name carries a date-shaped token: a rule-#6
record proper" -- but under R1/R2/R3 a date-carrying key with NO supersession
word never entered the population, so it could never be tier-DATED and could
never be a parity gap.  On the Wf-A199 corpus that single omission moved the
published gap count from 2 to 65 (same corpora, same descent rule, same
producer set; the population widened from "R1/R2/R3" to "any date token").
R1/R2/R3 are KEPT verbatim; R4 is ADDED as a named rule.

DESCENT
-------
A DATED RECORD RENDERING's children are its payload, not separate records: the
walk does not descend into it.  A key matched only by R4 (a date-shaped token
with no supersession semantics) is a NAMESPACE, so the walk DOES descend into
it and a dated record nested inside it is still collected.  A key matched only
by R1 (a supersession word with no date) is likewise a container.

PRODUCER CORPUS
---------------
Every ``*.py`` file in the repository, obtained from the gitignore-aware union

    git ls-files                                  (tracked)
    git ls-files --others --exclude-standard      (untracked, not ignored)
    git ls-files --others --ignored --exclude-standard

MINUS the exclusions listed in ``EXCLUDED_PRODUCER_COMPONENTS`` (documented in
the report) and MINUS this script itself (it names pattern tokens, not
artefacts; including it would let the census poison its own result).

CLASSIFICATION
--------------
  PRODUCER-EMITTED                 the key name is an exact Python string
                                   constant in the artefact's own producer AND
                                   a value comparison COMPLETED and was equal;
  PRODUCER-EMITTED-BUT-VALUE-DIFFERS
                                   the key name is in the artefact's own
                                   producer, a value comparison completed and
                                   differed -- both values are quoted;
  PRODUCER-EMITTED-NOT-COMPARABLE  the key name is in the artefact's own
                                   producer but NO value comparison could be
                                   completed (the producer's value node is a
                                   runtime expression -- an ``ast.Name``, a
                                   ``dict(NAME)`` call, an f-string, ...).
                                   ADDED 2026-11-02 by Wf-A204: before,
                                   "not comparable" was ABSORBED into
                                   "PRODUCER-EMITTED", which made the
                                   published "VALUE-DIFFERS = 0" vacuous;
  PRODUCER-BUILDS-NAME-DYNAMICALLY  the key name is NOT a literal in any
                                   producer, but a producer's source contains
                                   every alphabetic token of the name in order
                                   within one statement -- i.e. the producer
                                   could build the name at run time.  ALSO
                                   covers a key emitted into an OPEN-ENDED
                                   namespace (a dict comprehension whose key
                                   is not a constant);
  HAND-AUTHORED (no producer)      the artefact has NO producer at all -- the
                                   record was written by hand, so there is
                                   nothing to be in parity with.  NOT a defect;
  PRODUCER-MISSING-KEY             the artefact HAS a producer, the key name
                                   is in NO producer, and no dynamic build is
                                   detectable -- i.e. the defect signature.

The last class is an addition to the four classes named in the task brief; it
is required because a candidate that is none of the four cannot be labelled
HAND-AUTHORED without asserting something false (a producer DOES exist).

CLI
---
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \\
        molmetal/scripts/wfa199_parity_census.py \\
        --out-dir molmetal/reports/wfa199_parity_census

CPU-only, no GPU, no network, read-only with respect to the repository.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF_REL = "molmetal/scripts/wfa199_parity_census.py"

ARTEFACT_ROOTS = ("molmetal/reports", "metrics/by_metric")
SELF_ARTEFACT_DIR = "molmetal/reports/wfa199_parity_census"

#: Wf-A199 (2026-11-02) -- the PRE-REPAIR measurement of this SAME census, taken
#: on the SAME corpus BEFORE the emitter
#: `molmetal/scripts/wf_a155_ligand_connected_rate.py` was repaired to emit the
#: dated rule-#6 record it had been silently DROPPING.  AGENTS.md §5.5 rule #6
#: preserves a superseded rendering; this constant is where the pre-repair
#: count survives, because the pre-repair run's own report lived only in /tmp
#: (/tmp/wfa199_out) and was never published.  MEASURED 2026-11-02 02:28 CST by
#: this same tool on this same corpus.
SUPERSEDED_DATED_PARITY_GAPS_PRE_REPAIR_2026_11_02_WFA199 = 2

#: The IDENTITY of each of those two pre-repair gaps, preserved with the
#: count so the superseded measurement survives VERBATIM (AGENTS.md §5.5 rule
#: #6).  MEASURED by the pre-repair run of this same tool, same corpus.
SUPERSEDED_PARITY_GAP_IDENTITIES_PRE_REPAIR_2026_11_02_WFA199 = (
    (
        "metrics/by_metric/cfm_qed_fragment_aware.json",
        "historical_claim_2026_10_11",
    ),
    (
        "molmetal/reports/wfa162_c7_pool_ligand_connected_rate/"
        "ligand_connected_rate_c7.json",
        "superseded_frontier_2026_11_02_wfa194",
    ),
)

#: Wf-A204 (2026-11-02) -- the WHOLE published Wf-A199 report's counts,
#: preserved VERBATIM (AGENTS.md §5.5 rule #6: a superseded rendering is
#: preserved, the correction is additive and dated).  These are the numbers
#: the Wf-A199 report.json carried under schema
#: ``wfa199-parity-census/1``; the live census publishes its own numbers
#: under schema ``wfa199-parity-census/2``.  MEASURED 2026-11-02 02:37:25 CST
#: by the Wf-A199 run (report sha256
#: 2a081c3a78c3446eab282bbb3f9b313d78d2b50298cfe686175d635a3205dfa0).
SUPERSEDED_COUNTS_V1_2026_11_02_WFA199 = {
    "dated_record_keys": 572,
    "distinct_key_names": 143,
    "distinct_artefacts_with_dated_records": 122,
    "tier_DATED": 79,
    "tier_UNDATED": 493,
    "candidates_key_absent_from_every_producer": 143,
    "candidates_tier_DATED": 12,
    "candidates_tier_UNDATED": 131,
    "parity_gaps_dated_producer_missing_key": 2,
    "undated_producer_missing_key_not_rule6": 24,
    "parse_failures": 30,
}

SUPERSEDED_CLASSIFICATION_COUNTS_V1_2026_11_02_WFA199 = {
    "classification_counts_all_dated_records": {
        "PRODUCER-EMITTED": 72,
        "PRODUCER-EMITTED-BUT-VALUE-DIFFERS": 26,
        "PRODUCER-BUILDS-NAME-DYNAMICALLY": 0,
        "HAND-AUTHORED (no producer)": 405,
        "PRODUCER-MISSING-KEY": 69,
    },
    "classification_counts_candidates": {
        "PRODUCER-EMITTED": 0,
        "PRODUCER-EMITTED-BUT-VALUE-DIFFERS": 0,
        "PRODUCER-BUILDS-NAME-DYNAMICALLY": 0,
        "HAND-AUTHORED (no producer)": 117,
        "PRODUCER-MISSING-KEY": 26,
    },
    "classification_counts_candidates_DATED": {
        "PRODUCER-EMITTED": 0,
        "PRODUCER-EMITTED-BUT-VALUE-DIFFERS": 0,
        "PRODUCER-BUILDS-NAME-DYNAMICALLY": 0,
        "HAND-AUTHORED (no producer)": 10,
        "PRODUCER-MISSING-KEY": 2,
    },
}

#: The Wf-A199 report's own gap identities, preserved VERBATIM.
SUPERSEDED_PARITY_GAPS_VERBATIM_V1_2026_11_02_WFA199 = (
    (
        "metrics/by_metric/cfm_qed_fragment_aware.json",
        "historical_claim_2026_10_11",
    ),
    (
        "molmetal/reports/wfa175_frontier_comparability/refutation/refutation.json",
        "superseded_live_counter_evidence_3_verbatim_2026_11_02_wfa200_b",
    ),
)

EXCLUDED_PRODUCER_COMPONENTS = (
    # directory components that mark third-party / non-producer trees
    ".venv",              # virtualenvs (incl. .venv.bak.*)
    "references",         # vendored third-party trees (top level + molmetal/)
    ".cloned",            # cloned third-party repos
    "list",               # a stray virtualenv
    "node_modules",
    "site-packages",
    "worktrees",          # sibling waves' worktrees -- not this tree's producers
    ".cache",
    ".data",
    "tests",              # the brief's exclusion (molmetal/tests); also .claude skills' tests/
)

# Top-level artefact fields that may name the producing script.
PRODUCER_FIELD_NAMES = (
    "script",
    "source_script",
    "producer",
    "produced_by",
    "tool",
    "generated_by",
    "generator",
    "entry_point",
    "entrypoint",
    "command",
)

# Filename shapes of the sibling producer scripts the brief names.
SIBLING_PRODUCER_RE = re.compile(r"^(?:_?(?:write|build|make|emit|render)[a-z0-9_]*)\.py$")

DATE_TOKEN_RE = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}")
TOKEN_SPLIT_RE = re.compile(r"[A-Za-z0-9]+")
SUPERSESSION_WORD_RE = re.compile(
    r"(?i)supersed|demot|retract|historic|obsolet|deprecat|legacy|closed"
)

# A key whose NAME is a file path is a MAP KEY over files, not a dated
# record -- e.g. `q3.hits_per_file.TODO/04_history/23_weak_to_strong_plan_
# superseded_2026-09-26.md`.  Such a key is excluded from the population.
PATH_LIKE_RE = re.compile(r"^[^/\\]*(?:/[^/\\]*)*$")
FILE_EXT_RE = re.compile(
    r"(?i)\.(md|json|jsonl|csv|tsv|yaml|yml|py|sh|txt|log|tex|bib|pdf|png|svg|pt|pth|npy|npz)$"
)

SUPERSESSION_TOKENS = frozenset(
    {
        "superseded",
        "supersede",
        "supersedes",
        "supersession",
        "supersessions",
        "demoted",
        "demote",
        "demotion",
        "demotions",
        "historical",
        "historically",
        "retracted",
        "retract",
        "retraction",
        "obsolete",
        "deprecated",
        "legacy",
    }
)

# --------------------------------------------------------------------------
# THE MATCH RULES -- ONE PLACE (Wf-A204)
# --------------------------------------------------------------------------
# Every rule id, its name and its published definition live HERE, and the
# report's `coverage_bound.match_rules` block is rendered from THIS structure.
# The report can therefore never drift from the code.
MATCH_RULES: tuple[dict, ...] = (
    {
        "id": "R1",
        "name": "supersession_token",
        "added_by": "Wf-A199",
        "definition": (
            "the key name contains one of the SUPERSESSION_TOKENS as a "
            "case-insensitive TOKEN: "
            + ", ".join(sorted(SUPERSESSION_TOKENS))
        ),
        "population_effect": (
            "includes UNDATED keys (a supersession word with no date token)"
        ),
    },
    {
        "id": "R2",
        "name": "closed_plus_date",
        "added_by": "Wf-A199",
        "definition": (
            "the key name contains the token 'closed' AND a date-shaped token "
            "(YYYY-MM-DD / YYYY_MM_DD)"
        ),
        "population_effect": "a DATED key",
    },
    {
        "id": "R3",
        "name": "date_plus_supersession_word",
        "added_by": "Wf-A199",
        "definition": (
            "the key name contains a date-shaped token AND a "
            "supersession/demotion WORD as a SUBSTRING: "
            + SUPERSESSION_WORD_RE.pattern
        ),
        "population_effect": "a DATED key",
    },
    {
        "id": "R4",
        "name": "date_token",
        "added_by": "Wf-A204",
        "definition": (
            "the key name contains a date-shaped token (YYYY-MM-DD / "
            "YYYY_MM_DD).  This IS the tool's own DATED tier predicate (see "
            "TIER_ENUM): a rule-#6 record proper.  ADDED 2026-11-02 because "
            "R1/R2/R3 are a SUPERSESSION-SEMANTICS filter and under them a "
            "date-carrying key with no supersession word never entered the "
            "population, so it could never be tier-DATED and could never be a "
            "parity gap."
        ),
        "population_effect": "a DATED key",
    },
)

#: The DESCENT rule, published verbatim in the report's coverage bound.
DESCENT_RULE = (
    "the walk descends into a matched key UNLESS the matched key is a DATED "
    "RECORD RENDERING -- a key that carries BOTH a date-shaped token AND a "
    "supersession/demotion word (R2/R3, or R1 together with a date).  A "
    "rendering's children are its payload (e.g. `why_superseded`, "
    "`corrected_claim`) and are not separate records.  A key matched ONLY by "
    "R4 (a date-shaped token with no supersession semantics) is a NAMESPACE, "
    "not a rendering, so the walk descends into it and a dated record nested "
    "inside it is still collected.  A key matched ONLY by R1 (a supersession "
    "word with no date) is likewise a container.  The Wf-A199 rule ('the walk "
    "does not descend into a matched key') stopped at EVERY matched key, "
    "including date-keyed namespaces, and therefore hid the records nested "
    "inside them."
)

#: The PATH-SCOPING rule for the producer<->artefact association.
PATH_SCOPING_RULE = (
    "a producer is attached to an artefact by FOUR pieces of evidence, in "
    "decreasing strength: (1) the artefact DECLARES its producer (a `script` "
    "/ `source_script` / `producer` / `produced_by` / `tool` / `generated_by` "
    "/ `generator` / `entry_point` / `entrypoint` / `command` field naming an "
    "existing .py); (2) a producer whose source carries the artefact's FULL "
    "repo-relative path (or the path with the leading `molmetal/` stripped) as "
    "a string constant; (3) a sibling producer script in the artefact's own "
    "directory (`_write*.py` / `_build*.py` / `_make*.py` / `_emit*.py` / "
    "`_render*.py`); (4) any producer in the artefact's own `tools/` "
    "subdirectory.  Rules (1) and (2) are STRONG (the producer demonstrably "
    "writes or claims to write THIS artefact); rules (3) and (4) are WEAK (a "
    "sibling script may write a different file).  A bare BASENAME match is "
    "NOT evidence: many artefacts in this repo are called `refutation.json`."
)

#: The PATH-SCOPING rule for the VALUE match.
VALUE_PATH_SCOPING_RULE = (
    "the value comparison is path-scoped.  A producer dict-literal site "
    "matches a record only when the site's CONTAINER PATH (the chain of "
    "literal dict keys enclosing the site in the producer source) is a "
    "SUFFIX of the record's ARTEFACT CONTAINER PATH (the chain of parent "
    "keys of the record in the artefact JSON), or when the site's container "
    "path is EMPTY (a module-level dict literal, which can correspond to any "
    "artefact container).  Sites in a different container are NOT compared "
    "and are counted as `path_scoped_out_sites`."
)

#: The VALUE-PARITY rule, published verbatim in the report's coverage bound.
VALUE_PARITY_RULE = (
    "a producer value is taken from the artefact's OWN producer (a generic key "
    "name such as `superseded_by` lives in many unrelated producers, and only "
    "the artefact's own producer can reproduce THAT artefact).  A site whose "
    "value node is a Python LITERAL yields a comparison: `equal` when the "
    "JSON-serialisable forms match, `differs` otherwise.  A site whose value "
    "node is a RUNTIME EXPRESSION (an `ast.Name`, a `dict(NAME)` call, an "
    "f-string, a call, ...) yields `not-comparable` -- UNLESS the expression "
    "resolves to a MODULE-LEVEL CONSTANT of the same producer, in which case "
    "the constant's value is compared and the resolution is recorded as "
    "`module_constant`.  A record for which NO comparison could be completed "
    "is classified `PRODUCER-EMITTED-NOT-COMPARABLE` -- it is NEVER absorbed "
    "into `PRODUCER-EMITTED`.  A producer key that the ARTEFACT lacks is "
    "reported in its own list, `producer_emits_key_artefact_lacks`."
)

CLASSIFICATION_ENUM = (
    "PRODUCER-EMITTED",
    "PRODUCER-EMITTED-BUT-VALUE-DIFFERS",
    "PRODUCER-EMITTED-NOT-COMPARABLE",
    "PRODUCER-BUILDS-NAME-DYNAMICALLY",
    "HAND-AUTHORED (no producer)",
    "PRODUCER-MISSING-KEY",
)

TIER_ENUM = (
    "DATED",    # the key name carries a date-shaped token: a rule-#6 record proper
    "UNDATED",  # the key name only carries a supersession/demotion word
)

#: How a PRODUCER-MISSING-KEY record relates to its artefact's producer.
GAP_CLASS_ENUM = (
    # a producer is CONFIRMED to write this artefact and cannot reproduce the
    # key -> a re-run of that producer DROPS the dated record.
    "PARITY-DEFECT (producer-would-drop)",
    # a producer is only WEAKLY associated (a sibling script, a tools/ dir) ->
    # a re-run MIGHT drop it, but the writer is not confirmed.
    "PRODUCER-UNVERIFIED (writer not confirmed)",
    # the artefact has NO producer at all -> hand-authored; NOT a defect.
    "HAND-AUTHORED (no producer)",
)

MAX_VALUE_CHARS = 4000


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------
def git_paths(args: list[str], pathspec: str) -> list[str]:
    """`git ls-files -z -- <pathspec>` -> list of paths (NUL-separated)."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotepath=false"] + args + ["--", pathspec],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [x for x in out.split("\0") if x]


def _ls(args: list[str], pathspec: str) -> list[str]:
    return git_paths(["ls-files", "-z"] + args, pathspec)


def list_json_artefacts() -> list[str]:
    """gitignore-aware enumeration of every JSON artefact in scope."""
    found: set[str] = set()
    for args in (
        [],
        ["--others", "--exclude-standard"],
        ["--others", "--ignored", "--exclude-standard"],
    ):
        for root in ARTEFACT_ROOTS:
            for p in _ls(args, root):
                if p.endswith(".json"):
                    found.add(p)
    found = {p for p in found if not p.startswith(SELF_ARTEFACT_DIR + "/")}
    return sorted(found)


def list_producers() -> tuple[list[str], dict[str, int]]:
    """Every .py in the repo except the documented exclusions.

    A path is excluded when ANY of its directory components is in
    ``EXCLUDED_PRODUCER_COMPONENTS`` (so ``molmetal/references/...`` and
    ``references/...`` are both caught), plus the census script itself.
    """
    found: set[str] = set()
    for args in (
        [],
        ["--others", "--exclude-standard"],
        ["--others", "--ignored", "--exclude-standard"],
    ):
        for p in _ls(args, "*.py"):
            found.add(p)
    excluded: dict[str, int] = {}
    kept: list[str] = []
    for p in sorted(found):
        if p == SELF_REL:
            excluded["<the census script itself>"] = (
                excluded.get("<the census script itself>", 0) + 1
            )
            continue
        hit = None
        for comp in p.split("/")[:-1]:
            if comp in EXCLUDED_PRODUCER_COMPONENTS or comp.startswith(".venv"):
                hit = comp
                break
        if hit:
            excluded[hit] = excluded.get(hit, 0) + 1
            continue
        kept.append(p)
    return kept, excluded


# --------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------
def _abs(p: str | Path) -> Path:
    """Resolve a repo-relative (or already absolute) path."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def _display_path(p: Path) -> str:
    """Repo-relative when the path is inside the repo, else the absolute path."""
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def _path_variants(p: Path) -> list[str]:
    """Every string form under which `p` may be named by a producer."""
    out: list[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    s = str(p)
    add(s)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(REPO).as_posix()
        except ValueError:
            rel = None
        if rel:
            add(rel)
            add("./" + rel)
            if rel.startswith("molmetal/"):
                add(rel[len("molmetal/"):])
        else:
            add("./" + s)
    else:
        # a RELATIVE path (e.g. a producer's string constant): never resolve
        # it against the CWD -- it is a repo-relative name.
        add("./" + s)
        if s.startswith("molmetal/"):
            add(s[len("molmetal/"):])
    return out


def _norm_name(s: str) -> str:
    """Case/format-insensitive normalisation of a key name."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _stem(s: str) -> str:
    """The key name with its date token removed."""
    return _norm_name(DATE_TOKEN_RE.sub("", s))


# --------------------------------------------------------------------------
# key-name matching
# --------------------------------------------------------------------------
def match_reasons(key: str) -> list[str]:
    """Return the list of rule ids (R1/R2/R3/R4) a key name satisfies."""
    reasons: list[str] = []
    tokens = [t.lower() for t in TOKEN_SPLIT_RE.findall(key)]
    hits = sorted(set(tokens) & SUPERSESSION_TOKENS)
    if hits:
        reasons.append("R1_supersession_token:" + ",".join(hits))
    if "closed" in tokens and DATE_TOKEN_RE.search(key):
        reasons.append("R2_closed_plus_date")
    if DATE_TOKEN_RE.search(key) and SUPERSESSION_WORD_RE.search(key):
        reasons.append("R3_date_plus_supersession_word")
    if DATE_TOKEN_RE.search(key):
        reasons.append("R4_date_token")
    return reasons


def is_record_rendering(key: str) -> bool:
    """True when the key names a DATED RECORD RENDERING.

    A rendering carries BOTH a date-shaped token and a supersession/demotion
    word.  Its children are its payload (see ``DESCENT_RULE``).
    """
    if not DATE_TOKEN_RE.search(key):
        return False
    return bool(SUPERSESSION_WORD_RE.search(key))


def is_path_like_key(key: str) -> bool:
    """True when the key NAME is a file path rather than a record name.

    ``TODO/04_history/23_weak_to_strong_plan_superseded_2026-09-26.md`` is a
    MAP KEY over files.  It carries a supersession word and a date token, but
    it is not a dated rule-#6 record: it names a document, and its value is a
    hit count.  Such keys are excluded from the population.
    """
    if "/" not in key and "\\" not in key:
        return False
    return bool(FILE_EXT_RE.search(key))


def collect_dated_records(obj) -> list[dict]:
    """Walk a parsed JSON document and collect dated-record keys.

    Implements ``DESCENT_RULE``: descends into a matched key unless the
    matched key is a DATED RECORD RENDERING.
    """
    out: list[dict] = []

    def walk(node, path: str, container: list[str]):
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    continue
                if is_path_like_key(k):
                    continue
                reasons = match_reasons(k)
                if reasons:
                    out.append(
                        {
                            "key_path": path + k,
                            "container_path": list(container),
                            "key": k,
                            "value": v,
                            "match_reasons": reasons,
                        }
                    )
                    if is_record_rendering(k):
                        continue
                walk(v, path + k + ".", container + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]." % (path, i), container)

    walk(obj, "", [])
    return out


# --------------------------------------------------------------------------
# producer analysis
# --------------------------------------------------------------------------
class ProducerInfo:
    __slots__ = (
        "path",
        "strings",
        "dict_entries",
        "keyword_names",
        "module_constants",
        "templates",
        "dynamic_namespaces",
        "keyword_entries",
        "text",
        "lines",
    )

    def __init__(self, path: str):
        self.path = path
        #: exact string constant -> [lineno, ...]
        self.strings: dict[str, list[int]] = {}
        #: exact key name -> [entry, ...]  (a dict literal's literal entries)
        self.dict_entries: dict[str, list[dict]] = {}
        #: keyword-argument name -> [entry, ...]  (a NAME the producer emits)
        self.keyword_entries: dict[str, list[dict]] = {}
        #: module-level constant name -> literal value
        self.module_constants: dict[str, object] = {}
        # list of (lineno, compiled regex) for every f-string / concatenation
        # in the file: the regexes match the STRING VALUES such an expression
        # could evaluate to, with the interpolated slots left open.
        self.templates: list[tuple[int, re.Pattern]] = []
        # list of (lineno, statement text) for every dict comprehension whose
        # key is not a constant: an OPEN-ENDED key namespace.
        self.dynamic_namespaces: list[tuple[int, str]] = []
        self.text: str = ""
        self.lines: list[str] = []


def _template_segments(node) -> list[tuple[str, str | None]]:
    """Flatten an expression into ('lit', text) / ('var', None) segments.

    Only the string-producing shapes are handled: string constants, f-strings
    (ast.JoinedStr) and ``+`` concatenations of those.  Anything else becomes
    a single opaque slot.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return [("lit", node.value)]
        return []
    if isinstance(node, ast.JoinedStr):
        out: list[tuple[str, str | None]] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(("lit", v.value))
            else:
                out.append(("var", None))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _template_segments(node.left) + _template_segments(node.right)
    if isinstance(node, ast.FormattedValue):
        return [("var", None)]
    return [("var", None)]


def _template_regex(node) -> re.Pattern | None:
    segs = _template_segments(node)
    if not segs:
        return None
    # A template with no literal text (e.g. a bare f"{x}") would match EVERY
    # key name, so require at least 3 literal characters somewhere in it.
    literal_chars = sum(len(t) for kind, t in segs if kind == "lit")
    if literal_chars < 3:
        return None
    body = []
    for kind, text in segs:
        if kind == "lit":
            body.append(re.escape(text))
        else:
            body.append(r"[^'\"\n]*")
    try:
        return re.compile("".join(body))
    except re.error:
        return None


def _parent_map(tree) -> dict:
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _container_path(node, parents) -> list[str]:
    """The chain of literal dict keys enclosing `node` in the source."""
    chain: list[str] = []
    cur = node
    while True:
        p = parents.get(cur)
        if p is None:
            break
        if isinstance(p, ast.Dict):
            found = None
            for k, v in zip(p.keys, p.values):
                if v is cur:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        found = k.value
                    break
            if found is None:
                break
            chain.append(found)
        cur = p
    chain.reverse()
    return chain


def _const_name_of(value_node) -> str | None:
    """The module-level constant a value node refers to, if any.

    Handles a bare ``NAME`` and the ``dict(NAME)`` idiom used by the
    rule-#6 emitters.
    """
    if isinstance(value_node, ast.Name):
        return value_node.id
    if (
        isinstance(value_node, ast.Call)
        and isinstance(value_node.func, ast.Name)
        and value_node.func.id == "dict"
        and len(value_node.args) == 1
        and isinstance(value_node.args[0], ast.Name)
    ):
        return value_node.args[0].id
    return None


def scan_producer(path: str) -> ProducerInfo | None:
    p = _abs(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    info = ProducerInfo(path if not os.path.isabs(str(path)) else str(p))
    info.text = text
    info.lines = text.splitlines()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return info  # text-only producer; literals unavailable

    # module-level constants FIRST: the value resolution needs them.
    for node in tree.body:
        targets = []
        value_node = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        if value_node is None:
            continue
        try:
            val = ast.literal_eval(value_node)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                info.module_constants[t.id] = val

    parents = _parent_map(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            info.strings.setdefault(node.value, []).append(node.lineno)
        elif isinstance(node, ast.Dict):
            container = _container_path(node, parents)
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                entry = {
                    "line": getattr(k, "lineno", node.lineno),
                    "container_path": list(container),
                    "evaluable": True,
                    "value": None,
                    "const_name": None,
                }
                try:
                    entry["value"] = ast.literal_eval(v)
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    entry["evaluable"] = False
                    entry["const_name"] = _const_name_of(v)
                info.dict_entries.setdefault(k.value, []).append(entry)
        elif isinstance(node, ast.Call):
            for kw in node.keywords or []:
                if not kw.arg:
                    continue
                entry = {
                    "line": getattr(kw, "lineno", node.lineno),
                    "container_path": [],
                    "evaluable": True,
                    "value": None,
                    "const_name": None,
                }
                try:
                    entry["value"] = ast.literal_eval(kw.value)
                except (
                    ValueError,
                    SyntaxError,
                    TypeError,
                    MemoryError,
                    RecursionError,
                ):
                    entry["evaluable"] = False
                    entry["const_name"] = _const_name_of(kw.value)
                info.keyword_entries.setdefault(kw.arg, []).append(entry)
        elif isinstance(node, (ast.JoinedStr, ast.BinOp)):
            rx = _template_regex(node)
            if rx is not None:
                info.templates.append((getattr(node, "lineno", 0), rx))
        elif isinstance(node, ast.DictComp):
            # an OPEN-ENDED key namespace: the comprehension's key is not a
            # constant, so the producer can emit ANY name into it.
            lo = getattr(node, "lineno", 0)
            hi = getattr(node, "end_lineno", lo) or lo
            snippet = "\n".join(info.lines[lo - 1: hi])
            info.dynamic_namespaces.append((lo, snippet))
    return info


# --------------------------------------------------------------------------
# classification helpers
# --------------------------------------------------------------------------
def jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def value_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def compare_values(artefact_value, producer_value) -> str:
    """One of 'equal' / 'differs' / 'not-comparable'."""
    if producer_value is None:
        return "not-comparable"
    try:
        if jsonable(artefact_value) == jsonable(producer_value):
            return "equal"
    except Exception:
        return "not-comparable"
    return "differs"


def _path_matches(producer_container: list[str], artefact_container: list[str]) -> bool:
    """The VALUE_PATH_SCOPING_RULE predicate."""
    if not producer_container:
        return True
    if len(producer_container) > len(artefact_container):
        return False
    return producer_container == artefact_container[-len(producer_container):]


def compare_record_value(
    info: ProducerInfo,
    key: str,
    artefact_value,
    artefact_container: list[str],
) -> tuple[list[dict], int, int]:
    """Path-scoped value comparison of one record against one producer.

    Returns ``(rows, path_scoped_out_sites, total_sites)``.
    """
    rows: list[dict] = []
    scoped_out = 0
    total = 0
    for entry in info.dict_entries.get(key, []):
        total += 1
        if not _path_matches(entry["container_path"], artefact_container):
            scoped_out += 1
            continue
        if entry["evaluable"]:
            rows.append(
                {
                    "producer": info.path,
                    "line": entry["line"],
                    "comparison": compare_values(artefact_value, entry["value"]),
                    "producer_value": entry["value"],
                    "resolution": "dict_literal",
                    "producer_container_path": entry["container_path"],
                }
            )
            continue
        rows.append(
            {
                "producer": info.path,
                "line": entry["line"],
                "comparison": "not-comparable",
                "producer_value": None,
                "resolution": "dict_literal_unevaluable",
                "producer_container_path": entry["container_path"],
            }
        )
        cn = entry["const_name"]
        if cn and cn in info.module_constants:
            cval = info.module_constants[cn]
            rows.append(
                {
                    "producer": info.path,
                    "line": entry["line"],
                    "comparison": compare_values(artefact_value, cval),
                    "producer_value": cval,
                    "resolution": "module_constant",
                    "constant": cn,
                    "producer_container_path": entry["container_path"],
                }
            )
    # A keyword-argument name is a NAME the producer emits (e.g.
    # `build(..., note_supplement_2026_11_01_wfa175_frontier=...)`).
    for entry in info.keyword_entries.get(key, []):
        total += 1
        if entry["evaluable"]:
            rows.append(
                {
                    "producer": info.path,
                    "line": entry["line"],
                    "comparison": compare_values(artefact_value, entry["value"]),
                    "producer_value": entry["value"],
                    "resolution": "keyword_argument",
                    "producer_container_path": [],
                }
            )
            continue
        rows.append(
            {
                "producer": info.path,
                "line": entry["line"],
                "comparison": "not-comparable",
                "producer_value": None,
                "resolution": "keyword_argument_unevaluable",
                "producer_container_path": [],
            }
        )
        cn = entry["const_name"]
        if cn and cn in info.module_constants:
            cval = info.module_constants[cn]
            rows.append(
                {
                    "producer": info.path,
                    "line": entry["line"],
                    "comparison": compare_values(artefact_value, cval),
                    "producer_value": cval,
                    "resolution": "module_constant",
                    "constant": cn,
                    "producer_container_path": [],
                }
            )
    return rows, scoped_out, total


def assert_parity_clean(report: dict) -> None:
    """THE GUARD.  Raises when the census found a parity defect.

    The message names every offending key, so a failure is actionable.
    """
    gaps = report.get("parity_gaps", [])
    inverse = report.get("producer_emits_key_artefact_lacks", [])
    if not gaps and not inverse:
        return
    gap_txt = "; ".join(
        "%s :: %s [%s]" % (g["artefact"], g["key"], g.get("gap_class", "?"))
        for g in gaps
    )
    inv_txt = "; ".join(
        "%s :: %s (producer %s)" % (i["artefact"], i["key"], i["producer"])
        for i in inverse
    )
    parts = []
    if gap_txt:
        parts.append("producer-missing dated keys: " + gap_txt)
    if inv_txt:
        parts.append("producer emits a dated key the artefact lacks: " + inv_txt)
    raise AssertionError(
        "producer<->artefact parity broken: " + " | ".join(parts)
    )


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------
def run_census(
    verbose: bool = True,
    artefact_paths: list[str] | None = None,
    producer_paths: list[str] | None = None,
) -> dict:
    """Run the census.

    ``artefact_paths`` / ``producer_paths`` default to the repo-wide corpora.
    They exist as a TEST SEAM: a caller may pass an explicit corpus (absolute
    or repo-relative paths) so that the guard can be exercised against a
    /tmp mini-corpus without touching the repository.
    """
    t0 = time.time()
    if artefact_paths is None:
        artefact_paths = list_json_artefacts()
    if producer_paths is None:
        producers_rel, excluded = list_producers()
    else:
        producers_rel, excluded = list(producer_paths), {}
    artefacts_abs = [_abs(p) for p in artefact_paths]
    producers_abs = [_abs(p) for p in producers_rel]
    if verbose:
        print(
            "[wfa199] artefacts=%d producers=%d" % (len(artefacts_abs), len(producers_abs)),
            flush=True,
        )

    producer_infos: list[ProducerInfo] = []
    for rel, ap in zip(producers_rel, producers_abs):
        info = scan_producer(ap)
        if info is not None:
            info.path = _display_path(ap)
            producer_infos.append(info)
    if verbose:
        print(
            "[wfa199] producer AST scan done (%d files, %.1fs)"
            % (len(producer_infos), time.time() - t0),
            flush=True,
        )

    # key name -> producer file:line evidence (exact string constants only)
    name_hits: dict[str, list[dict]] = {}
    # key name -> producer file:line evidence (KEYWORD-ARGUMENT names: a NAME
    # the producer emits, e.g. `build(..., note_supplement_2026_11_01=...)`)
    keyword_hits: dict[str, list[dict]] = {}
    # normalised name -> the RAW names seen in producers (case/format variants)
    norm_names: dict[str, set[str]] = {}
    # date-stripped stem -> the RAW names seen in producers
    stem_names: dict[str, set[str]] = {}
    # artefact path (several string forms) -> producers that name it
    json_path_hits: dict[str, list[str]] = {}
    for info in producer_infos:
        for const, lines in info.strings.items():
            if const in name_hits:
                for ln in lines:
                    name_hits[const].append({"producer": info.path, "line": ln})
            else:
                name_hits[const] = [
                    {"producer": info.path, "line": ln} for ln in lines
                ]
            nn = _norm_name(const)
            if nn:
                norm_names.setdefault(nn, set()).add(const)
            st = _stem(const)
            if len(st) >= 4:
                stem_names.setdefault(st, set()).add(const)
            if const.endswith(".json"):
                for cand in _path_variants(Path(const)):
                    if info.path not in json_path_hits.setdefault(cand, []):
                        json_path_hits[cand].append(info.path)

    info_by_path = {i.path: i for i in producer_infos}

    for info in producer_infos:
        for kw_name, entries in info.keyword_entries.items():
            keyword_hits.setdefault(kw_name, []).extend(
                {"producer": info.path, "line": e["line"]} for e in entries
            )

    def artefact_producers(rel_abs: Path, doc: dict) -> dict:
        """The producer(s) that CAN regenerate this artefact.

        See ``PATH_SCOPING_RULE``.  Returns the producer list plus, per
        producer, WHICH rule attached it (so a reader can tell a confirmed
        writer from a sibling script).
        """
        prods: list[str] = []
        assoc: dict[str, dict] = {}

        def add(cand: str, rule: str):
            cand = cand.strip()
            if not cand.endswith(".py"):
                return
            if cand.startswith(str(REPO)):
                cand = cand[len(str(REPO)) + 1:]
            resolved = None
            for base in (REPO, rel_abs.parent):
                p = Path(cand)
                if p.is_absolute():
                    if p.is_file():
                        resolved = p
                        break
                    continue
                cand_abs = (base / cand)
                if cand_abs.is_file():
                    resolved = cand_abs
                    break
            if resolved is None:
                return
            disp = _display_path(resolved)
            if disp not in prods:
                prods.append(disp)
                assoc[disp] = {"rule": rule, "strong": rule in ("declared", "path_constant")}
            elif rule in ("declared", "path_constant"):
                assoc[disp]["rule"] = rule
                assoc[disp]["strong"] = True

        for field in PRODUCER_FIELD_NAMES:
            val = doc.get(field)
            for s in (val if isinstance(val, list) else [val]):
                if isinstance(s, str):
                    for piece in s.split():
                        if piece.endswith(".py"):
                            add(piece, "declared")
        parent = rel_abs.parent
        for sib in sorted(parent.glob("*.py")):
            if SIBLING_PRODUCER_RE.match(sib.name):
                add(str(sib), "sibling")
        tools = parent / "tools"
        if tools.is_dir():
            for sib in sorted(tools.glob("*.py")):
                add(str(sib), "tools_dir")
        for prod in json_path_hits.get(_display_path(rel_abs), []):
            if prod not in prods:
                prods.append(prod)
                assoc[prod] = {"rule": "path_constant", "strong": True}
        for variant in _path_variants(rel_abs):
            for prod in json_path_hits.get(variant, []):
                if prod not in prods:
                    prods.append(prod)
                    assoc[prod] = {"rule": "path_constant", "strong": True}
        strong = sorted(p for p in prods if assoc.get(p, {}).get("strong"))
        return {
            "producers": sorted(prods),
            "association": assoc,
            "strong_producers": strong,
        }

    records: list[dict] = []
    parse_failures: list[dict] = []
    artefact_bytes = 0
    artefacts_scanned = 0
    artefacts_parsed = 0
    non_dict_artefacts = 0
    artefact_meta: list[dict] = []
    for rel_abs in artefacts_abs:
        artefacts_scanned += 1
        rel = _display_path(rel_abs)
        try:
            artefact_bytes += rel_abs.stat().st_size
            with rel_abs.open("r", encoding="utf-8", errors="replace") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            parse_failures.append({"artefact": rel, "error": type(exc).__name__})
            continue
        artefacts_parsed += 1
        if not isinstance(doc, dict):
            non_dict_artefacts += 1
            continue
        prods_info = artefact_producers(rel_abs, doc)
        names: set[str] = set()

        def _collect_names(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(k, str):
                        names.add(k)
                    _collect_names(v)
            elif isinstance(node, list):
                for v in node:
                    _collect_names(v)

        _collect_names(doc)
        artefact_meta.append(
            {
                "artefact": rel,
                "strong_producers": prods_info["strong_producers"],
                "producers": prods_info["producers"],
                "key_names": names,
            }
        )
        for rec in collect_dated_records(doc):
            rec["artefact"] = rel
            rec["artefact_producers"] = prods_info["producers"]
            rec["artefact_producer_association"] = prods_info["association"]
            rec["artefact_strong_producers"] = prods_info["strong_producers"]
            records.append(rec)
        del doc
    if verbose:
        print(
            "[wfa199] artefacts parsed=%d dated-record keys=%d (%.1fs)"
            % (artefacts_parsed, len(records), time.time() - t0),
            flush=True,
        )

    # ---- dynamic-build detection, only for keys with no literal hit ------
    # A key name is DYNAMICALLY buildable when some producer contains an
    # f-string (or a concatenation of string literals) whose literal parts and
    # interpolated slots can spell exactly that name.  The templates were
    # extracted from each producer's AST; the interpolated slots are matched
    # by `[^'"\n]*`, so the match must reproduce the WHOLE key name.
    keys_needing_dynamic = sorted(
        {
            r["key"]
            for r in records
            if not name_hits.get(r["key"]) and not keyword_hits.get(r["key"])
        }
    )
    dynamic_hits: dict[str, list[dict]] = {}
    for info in producer_infos:
        for lineno, rx in info.templates:
            for k in keys_needing_dynamic:
                if k in dynamic_hits and len(dynamic_hits[k]) >= 20:
                    continue
                if rx.fullmatch(k):
                    dynamic_hits.setdefault(k, []).append(
                        {"producer": info.path, "line": lineno, "matched": k}
                    )
    if verbose:
        print(
            "[wfa199] dynamic detection done (%d key names, %d dynamic, %.1fs)"
            % (
                len(keys_needing_dynamic),
                len(dynamic_hits),
                time.time() - t0,
            ),
            flush=True,
        )

    candidates: list[dict] = []
    all_classified: list[dict] = []
    for rec in records:
        key = rec["key"]
        rel = rec["artefact"]
        prods = rec["artefact_producers"]
        container = rec.get("container_path") or []
        tier = "DATED" if DATE_TOKEN_RE.search(key) else "UNDATED"
        hits = name_hits.get(key, [])
        kw_hits = keyword_hits.get(key, [])
        # hits restricted to the artefact's OWN producer(s) -- a generic key name
        # such as `superseded_by` lives in many unrelated producers, and only
        # the artefact's own producer can reproduce THAT artefact.
        own_hits = [h for h in hits if h["producer"] in prods]
        own_kw_hits = [h for h in kw_hits if h["producer"] in prods]
        evidence: dict = {
            "artefact_producers": prods,
            "artefact_producer_association": rec["artefact_producer_association"],
            "artefact_strong_producers": rec["artefact_strong_producers"],
            "literal_producer_hits": hits[:20],
            "n_literal_producer_hits": len(hits),
            "keyword_argument_producer_hits": kw_hits[:20],
            "n_keyword_argument_producer_hits": len(kw_hits),
            "own_producer_literal_hits": own_hits[:20],
            "n_own_producer_literal_hits": len(own_hits),
            "own_producer_keyword_argument_hits": own_kw_hits[:20],
        }

        if own_hits or own_kw_hits:
            comparisons: list[dict] = []
            scoped_out = 0
            total_sites = 0
            seen_producers: set[str] = set()
            for h in list(own_hits) + list(own_kw_hits):
                if h["producer"] in seen_producers:
                    continue
                seen_producers.add(h["producer"])
                info = info_by_path.get(h["producer"])
                if info is None:
                    continue
                rows, so, ts = compare_record_value(
                    info, key, rec["value"], container
                )
                comparisons.extend(rows)
                scoped_out += so
                total_sites += ts
                if len(comparisons) >= 12:
                    break
            differing = [c for c in comparisons if c["comparison"] == "differs"]
            equal = [c for c in comparisons if c["comparison"] == "equal"]
            evidence["value_comparisons"] = comparisons[:12]
            evidence["value_comparisons_total"] = len(comparisons)
            evidence["value_path_scoped_out_sites"] = scoped_out
            evidence["value_sites_total"] = total_sites
            const_hits = []
            for c in comparisons:
                if c.get("constant"):
                    const_hits.append(
                        {"producer": c["producer"], "constant": c["constant"]}
                    )
            for h in list(own_hits) + list(own_kw_hits):
                info = info_by_path.get(h["producer"])
                if info is None:
                    continue
                for cname, cval in info.module_constants.items():
                    if jsonable(cval) == jsonable(rec["value"]):
                        const_hits.append(
                            {"producer": h["producer"], "constant": cname}
                        )
                        break
            if const_hits:
                evidence["value_parity_constant"] = const_hits[:8]
            if differing and not equal:
                classification = "PRODUCER-EMITTED-BUT-VALUE-DIFFERS"
            elif equal:
                classification = "PRODUCER-EMITTED"
            else:
                # NO value comparison could be completed.  Wf-A204: this is
                # its OWN bucket; it used to be absorbed into
                # "PRODUCER-EMITTED", which made the published
                # "VALUE-DIFFERS = 0" vacuous.
                classification = "PRODUCER-EMITTED-NOT-COMPARABLE"
        else:
            # no literal in the artefact's own producer: could the name be built
            # at run time by that producer?
            own_dyn = [
                d for d in dynamic_hits.get(key, []) if d["producer"] in prods
            ]
            any_dyn = dynamic_hits.get(key, [])
            evidence["dynamic_candidates"] = any_dyn[:20]
            evidence["own_producer_dynamic_candidates"] = own_dyn[:20]
            # an OPEN-ENDED namespace: the producer builds a dict whose key is
            # not a constant, so it can emit ANY name into a namespace whose
            # name appears in the same statement.
            container_tail = container[-1] if container else ""
            ns_hits = []
            if container_tail:
                for prod in prods:
                    info = info_by_path.get(prod)
                    if info is None:
                        continue
                    for line, snippet in info.dynamic_namespaces:
                        if container_tail in snippet:
                            ns_hits.append(
                                {"producer": prod, "line": line, "namespace": container_tail}
                            )
            evidence["dynamic_namespace_candidates"] = ns_hits[:20]
            if own_dyn or ns_hits:
                classification = "PRODUCER-BUILDS-NAME-DYNAMICALLY"
            elif not prods:
                classification = "HAND-AUTHORED (no producer)"
            else:
                classification = "PRODUCER-MISSING-KEY"

        val = rec["value"]
        serialized = json.dumps(jsonable(val), ensure_ascii=False, sort_keys=True)
        entry = {
            "artefact": rel,
            "key": key,
            "key_path": rec["key_path"],
            "container_path": container,
            "tier": tier,
            "match_reasons": rec["match_reasons"],
            "classification": classification,
            "value_sha256": value_digest(val),
            "value_preview": serialized[:MAX_VALUE_CHARS],
            "value_truncated": len(serialized) > MAX_VALUE_CHARS,
            "evidence": evidence,
        }
        # ---- the gap sub-classification (Wf-A204) ----
        if classification == "PRODUCER-MISSING-KEY":
            strong = rec.get("artefact_strong_producers") or []
            if not prods:
                gap_class = "HAND-AUTHORED (no producer)"
            elif not strong:
                gap_class = "PRODUCER-UNVERIFIED (writer not confirmed)"
            else:
                gap_class = "PARITY-DEFECT (producer-would-drop)"
            is_scalar = isinstance(val, (str, int, float, bool)) or val is None
            variants = sorted(
                n
                for n in norm_names.get(_norm_name(key), set())
                if n != key
            )
            stems = sorted(
                n
                for n in stem_names.get(_stem(key), set())
                if n != key and _stem(n) == _stem(key)
            )
            entry["gap_class"] = gap_class
            entry["gap_flags"] = {
                "artefact_has_producer": bool(prods),
                "producer_writes_artefact_confirmed": bool(strong),
                "is_pointer_or_label": is_scalar,
                "value_is_structured_record": (not is_scalar) and isinstance(
                    val, (dict, list)
                ),
                "producer_emits_under_a_different_name": bool(variants or stems),
                "producer_emits_under_different_name_variants": variants[:8],
                "producer_emits_under_different_name_stems": stems[:8],
                "re_run_would_drop_the_record": gap_class.startswith("PARITY-DEFECT"),
            }
        all_classified.append(entry)
        if not hits and not kw_hits:
            candidates.append(entry)

    counts: dict[str, int] = {c: 0 for c in CLASSIFICATION_ENUM}
    for e in all_classified:
        counts[e["classification"]] = counts.get(e["classification"], 0) + 1

    candidate_counts: dict[str, int] = {c: 0 for c in CLASSIFICATION_ENUM}
    for e in candidates:
        candidate_counts[e["classification"]] = candidate_counts.get(e["classification"], 0) + 1

    # THE DEFECT SURFACE: dated records whose own producer cannot reproduce the
    # key name at all.  An UNDATED key (e.g. the chemistry counter
    # `ok_repaired:kekulize_demoted_all`) is not a rule-#6 record, so it is
    # reported separately and is NOT a parity gap.
    parity_gaps = [
        e
        for e in candidates
        if e["classification"] == "PRODUCER-MISSING-KEY" and e["tier"] == "DATED"
    ]
    undated_missing = [
        e
        for e in candidates
        if e["classification"] == "PRODUCER-MISSING-KEY" and e["tier"] == "UNDATED"
    ]

    # ---- the INVERSE class: the producer emits a dated key the artefact
    # lacks (the artefact is STALE with respect to its own producer) ----
    inverse_findings: list[dict] = []
    seen_inverse: set[tuple[str, str, str]] = set()
    for meta in artefact_meta:
        rel = meta["artefact"]
        prods = meta["strong_producers"]
        if not prods:
            continue
        names = meta["key_names"]
        for prod in prods:
            info = info_by_path.get(prod)
            if info is None:
                continue
            for key, entries in info.dict_entries.items():
                if not DATE_TOKEN_RE.search(key):
                    continue
                if key in names:
                    continue
                sig = (rel, key, prod)
                if sig in seen_inverse:
                    continue
                seen_inverse.add(sig)
                e0 = entries[0]
                inverse_findings.append(
                    {
                        "artefact": rel,
                        "key": key,
                        "producer": prod,
                        "line": e0["line"],
                        "producer_container_path": e0["container_path"],
                        "reason": (
                            "the producer emits this dated key but the artefact "
                            "does not carry it -- the artefact is STALE with "
                            "respect to its own producer"
                        ),
                    }
                )
    inverse_findings.sort(key=lambda f: (f["artefact"], f["key"], f["producer"]))

    # ---- the corpus manifest digest (Wf-A204) ----
    manifest_lines = []
    for rel_abs in artefacts_abs:
        try:
            size = rel_abs.stat().st_size
        except OSError:
            size = -1
        manifest_lines.append("%s\t%d" % (_display_path(rel_abs), size))
    manifest_lines.sort()
    corpus_manifest_sha256 = hashlib.sha256(
        "\n".join(manifest_lines).encode("utf-8")
    ).hexdigest()
    producer_manifest_lines = sorted(_display_path(p) for p in producers_abs)
    producer_manifest_sha256 = hashlib.sha256(
        "\n".join(producer_manifest_lines).encode("utf-8")
    ).hexdigest()

    report = {
        "schema": "wfa199-parity-census/2",
        "wave": "Wf-A204",
        "generated_by": SELF_REL,
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "purpose": (
            "MEASUREMENT of the producer<->artefact parity defect class "
            "(honest-negative catalogue 6.268.4): dated AGENTS.md 5.5 rule-#6 "
            "records that live in a published artefact but in NO producer."
        ),
        "coverage_bound": {
            "why": (
                "a count without a bound is not a measurement (honest-negative "
                "catalogue 6.244 / 6.268 O12).  Every number below was produced "
                "under exactly these rules; the rules are published so a reader "
                "can reproduce the count or disagree with the bound."
            ),
            "units": {
                "population_record": "one (artefact, JSON key path) pair",
                "candidate": (
                    "a population record whose KEY NAME is absent as an exact "
                    "Python string constant from EVERY producer in the corpus"
                ),
                "parity_gap": (
                    "a candidate that is classified PRODUCER-MISSING-KEY and is "
                    "in the DATED tier"
                ),
                "identity": "the (artefact, key name) pair",
            },
            "corpus": {
                "artefact_roots": list(ARTEFACT_ROOTS),
                "artefact_file_selection": (
                    "git ls-files UNION git ls-files --others --exclude-standard "
                    "UNION git ls-files --others --ignored --exclude-standard, "
                    "filtered to *.json"
                ),
                "excluded_artefact_dirs": [
                    SELF_ARTEFACT_DIR + "/ (the census's own output)"
                ],
                "producer_selection": "every *.py in the repo, same gitignore-aware union",
                "excluded_producer_components": list(EXCLUDED_PRODUCER_COMPONENTS)
                + [SELF_REL + " (the census script itself)"],
                "excluded_producer_file_counts": excluded,
                "n_producers": len(producers_abs),
                "n_artefacts_scanned": artefacts_scanned,
                "n_artefacts_parsed": artefacts_parsed,
                "n_artefacts_non_dict": non_dict_artefacts,
                "artefact_bytes_total": artefact_bytes,
                "artefact_corpus_manifest_sha256": corpus_manifest_sha256,
                "producer_corpus_manifest_sha256": producer_manifest_sha256,
                "manifest_definition": (
                    "sha256 over the sorted lines '<repo-relative path>\\t<byte "
                    "size>' for the artefact corpus, and over the sorted "
                    "repo-relative producer paths for the producer corpus.  A "
                    "file landing, vanishing or changing size changes the "
                    "digest, so the counts are reproducible across runs."
                ),
            },
            "match_rules": [dict(r) for r in MATCH_RULES],
            "match_rule_ids": [r["id"] for r in MATCH_RULES],
            "tier_rule": (
                "DATED = the key name carries a date-shaped token (a rule-#6 "
                "record proper).  UNDATED = the key name carries a "
                "supersession/demotion word but no date."
            ),
            "descent_rule": DESCENT_RULE,
            "path_scoping_rule": PATH_SCOPING_RULE,
            "value_path_scoping_rule": VALUE_PATH_SCOPING_RULE,
            "value_parity_rule": VALUE_PARITY_RULE,
            "literal_hit_definition": (
                "the key name is an exact Python string constant in the producer "
                "(ast.Constant); prose mentions do not count"
            ),
            "dynamic_hit_definition": (
                "an f-string / concatenation in the producer whose literal parts "
                "and interpolated slots can spell the WHOLE key name, OR the "
                "producer builds an open-ended dict namespace (a dict "
                "comprehension with a non-constant key) whose statement names "
                "the record's container"
            ),
            "gap_class_enum": list(GAP_CLASS_ENUM),
        },
        "scope": {
            "artefact_roots": list(ARTEFACT_ROOTS),
            "artefact_file_selection": (
                "git ls-files UNION git ls-files --others --exclude-standard "
                "UNION git ls-files --others --ignored --exclude-standard, "
                "filtered to *.json"
            ),
            "excluded_artefact_dirs": [
                SELF_ARTEFACT_DIR + "/ (the census's own output)"
            ],
            "producer_selection": "every *.py in the repo, same gitignore-aware union",
            "excluded_producer_components": list(EXCLUDED_PRODUCER_COMPONENTS)
            + [SELF_REL + " (the census script itself)"],
            "excluded_producer_file_counts": excluded,
            "n_producers": len(producers_abs),
            "n_artefacts_scanned": artefacts_scanned,
            "n_artefacts_parsed": artefacts_parsed,
            "n_artefacts_non_dict": non_dict_artefacts,
            "artefact_bytes_total": artefact_bytes,
            "artefact_corpus_manifest_sha256": corpus_manifest_sha256,
            "producer_corpus_manifest_sha256": producer_manifest_sha256,
        },
        "key_patterns": {
            "R1_supersession_token": sorted(SUPERSESSION_TOKENS),
            "R2_closed_plus_date": (
                "token 'closed' AND a date-shaped token (YYYY-MM-DD / YYYY_MM_DD)"
            ),
            "R3_date_plus_supersession_word": (
                "date-shaped token AND a supersession/demotion word as a SUBSTRING "
                "(supersed|demot|retract|historic|obsolet|deprecat|legacy|closed)"
            ),
            "R4_date_token": (
                "date-shaped token (YYYY-MM-DD / YYYY_MM_DD) -- the tool's own "
                "DATED tier predicate.  ADDED 2026-11-02 by Wf-A204."
            ),
            "no_descent_rule": DESCENT_RULE,
            "literal_hit_definition": (
                "the key name is an exact Python string constant in the producer "
                "(ast.Constant); prose mentions do not count"
            ),
            "dynamic_hit_definition": (
                "every alphabetic token of the key occurs in the producer source "
                "and the digit runs can be matched with \\d+ inside one statement, "
                "so the producer could build the name at run time"
            ),
        },
        "schema_changes_v1_to_v2": [
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "R4_date_token added to the match rule set (the population "
                    "widened from 'R1/R2/R3' to 'any date-shaped token')."
                ),
                "why": (
                    "the tool's own DATED tier is 'the key name carries a "
                    "date-shaped token'; under R1/R2/R3 a date-only key never "
                    "entered the population, so it could never be tier-DATED.  "
                    "The published gap count (2) was computed on a population "
                    "32x smaller than the tool's own DATED tier."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "descent rule changed: the walk now stops only at a DATED "
                    "RECORD RENDERING (a date token AND a supersession word), "
                    "not at every matched key."
                ),
                "why": (
                    "the Wf-A199 rule stopped at date-keyed NAMESPACES too and "
                    "therefore hid the dated records nested inside them."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "new classification PRODUCER-EMITTED-NOT-COMPARABLE; a "
                    "producer value that is a runtime expression is no longer "
                    "absorbed into PRODUCER-EMITTED."
                ),
                "why": (
                    "the published 'PRODUCER-EMITTED-BUT-VALUE-DIFFERS = 0' was "
                    "vacuous on the tier it was stated for: 35 of the 39 DATED "
                    "records the tool called PRODUCER-EMITTED were never value "
                    "compared."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "the value comparison is now PATH-SCOPED (a producer site "
                    "must sit in a container path that is a suffix of the "
                    "record's artefact container path)."
                ),
                "why": (
                    "the Wf-A199 comparison was by key NAME only, so the same "
                    "key name at 2-3 producer sites produced an ambiguous "
                    "verdict."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "new list `producer_emits_key_artefact_lacks`: the INVERSE "
                    "class (the producer emits a dated key the artefact does "
                    "not carry -- the artefact is stale w.r.t. its producer)."
                ),
                "why": (
                    "the one PROVABLE producer/artefact difference found by the "
                    "Wf-A204 audit sits in that class, and the census had no "
                    "bucket for it."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "the field `all_dated_records` no longer holds UNDATED "
                    "records: the full population moved to "
                    "`all_population_records` and `all_dated_records` is now "
                    "the DATED tier only."
                ),
                "why": (
                    "the v1 field name said 'dated' and held 493 UNDATED "
                    "records of 572, so a reader could quote the wrong number."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "change": (
                    "new `coverage_bound` block publishing the corpus, the "
                    "match rules, the descent rule, the path-scoping rule and "
                    "the value-parity rule, plus two corpus manifest digests."
                ),
                "why": (
                    "the v1 report published counts with no bound, so the "
                    "counts were neither reproducible nor interpretable "
                    "(honest-negative catalogue 6.244 / 6.268 O12)."
                ),
            },
        ],
        "superseded_measurements": [
            {
                "date": "2026-11-02",
                "wave": "Wf-A199",
                "stage": "PRE-REPAIR",
                "dated_parity_gaps": (
                    SUPERSEDED_DATED_PARITY_GAPS_PRE_REPAIR_2026_11_02_WFA199
                ),
                "gap_identities": [
                    {"artefact": a, "key": k}
                    for a, k in SUPERSEDED_PARITY_GAP_IDENTITIES_PRE_REPAIR_2026_11_02_WFA199
                ],
                "note": (
                    "the same tool on the same corpus, run BEFORE "
                    "molmetal/scripts/wf_a155_ligand_connected_rate.py was "
                    "repaired to emit "
                    "`comparability.superseded_frontier_2026_11_02_wfa194`.  "
                    "The pre-repair run's own report lived only in /tmp "
                    "(/tmp/wfa199_out) and was never published; only this "
                    "count and this identity list survive, as a DATED "
                    "record.  The LIVE state is this report's own "
                    "`parity_gaps` list."
                ),
            },
            {
                "date": "2026-11-02",
                "wave": "Wf-A204",
                "stage": "PUBLISHED-V1-COUNTS-PRESERVED-VERBATIM",
                "note": (
                    "the WHOLE count block of the published Wf-A199 report "
                    "(schema wfa199-parity-census/1, generated 2026-11-02 "
                    "02:37:25 CST, report sha256 "
                    "2a081c3a78c3446eab282bbb3f9b313d78d2b50298cfe686175d635a"
                    "3205dfa0) is preserved VERBATIM below.  AGENTS.md §5.5 rule "
                    "#6: a superseded rendering is preserved; the correction is "
                    "additive and dated.  The v1 report itself is preserved "
                    "beside this one as `report.v1.2026_11_02.json`."
                ),
                "v1_counts_verbatim": dict(SUPERSEDED_COUNTS_V1_2026_11_02_WFA199),
                "v1_classification_counts_verbatim": {
                    k: dict(v)
                    for k, v in SUPERSEDED_CLASSIFICATION_COUNTS_V1_2026_11_02_WFA199.items()
                },
                "v1_parity_gap_identities_verbatim": [
                    {"artefact": a, "key": k}
                    for a, k in SUPERSEDED_PARITY_GAPS_VERBATIM_V1_2026_11_02_WFA199
                ],
                "v1_verbatim_note": (
                    "v1's '2 gaps' is the count under v1's OWN R1/R2/R3 "
                    "population, which excluded every date-only rule-#6 key "
                    "BEFORE the DATED tier test.  It is preserved here "
                    "unmodified; it is not the live count."
                ),
            },
        ],
        "classification_enum": list(CLASSIFICATION_ENUM),
        "tier_enum": list(TIER_ENUM),
        "counts": {
            "population_records_total": len(all_classified),
            "population_records_DATED_tier": sum(
                1 for e in all_classified if e["tier"] == "DATED"
            ),
            "population_records_UNDATED_tier": sum(
                1 for e in all_classified if e["tier"] == "UNDATED"
            ),
            "distinct_key_names": len({e["key"] for e in all_classified}),
            "distinct_artefacts_with_records": len(
                {e["artefact"] for e in all_classified}
            ),
            "candidates_total": len(candidates),
            "candidates_DATED_tier": sum(1 for e in candidates if e["tier"] == "DATED"),
            "candidates_UNDATED_tier": sum(
                1 for e in candidates if e["tier"] == "UNDATED"
            ),
            "parity_gaps_DATED_producer_missing_key": len(parity_gaps),
            "parity_gaps_UNDATED_producer_missing_key": len(undated_missing),
            "producer_emits_key_artefact_lacks": len(inverse_findings),
            "parse_failures": len(parse_failures),
        },
        "counts_v1_names_preserved_verbatim": dict(
            SUPERSEDED_COUNTS_V1_2026_11_02_WFA199
        ),
        "classification_counts_all_population_records": counts,
        "classification_counts_all_dated_records": {
            c: sum(
                1
                for e in all_classified
                if e["tier"] == "DATED" and e["classification"] == c
            )
            for c in CLASSIFICATION_ENUM
        },
        "classification_counts_candidates": candidate_counts,
        "classification_counts_candidates_DATED": {
            c: sum(
                1
                for e in candidates
                if e["tier"] == "DATED" and e["classification"] == c
            )
            for c in CLASSIFICATION_ENUM
        },
        "gap_class_counts_DATED_parity_gaps": {
            g: sum(1 for e in parity_gaps if e.get("gap_class") == g)
            for g in GAP_CLASS_ENUM
        },
        "parse_failures": parse_failures,
        "candidates": candidates,
        "parity_gaps": parity_gaps,
        "producer_emits_key_artefact_lacks": inverse_findings,
        "undated_missing_keys_not_rule6_records": undated_missing,
        "all_population_records": all_classified,
        "all_dated_records": [e for e in all_classified if e["tier"] == "DATED"],
    }
    report["_runtime_seconds"] = round(time.time() - t0, 2)
    return report


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------
def render_md(report: dict) -> str:
    L: list[str] = []
    A = L.append
    scope = report["scope"]
    cb = report["coverage_bound"]
    counts = report["counts"]
    A("# Wf-A204 -- producer<->artefact parity census (dated rule-#6 records)")
    A("")
    A("**This is a MEASUREMENT. It repairs nothing.**")
    A("")
    A("- wave: `%s`" % report["wave"])
    A("- generated by: `%s`" % report["generated_by"])
    A("- generated at: %s" % report["generated_at_local"])
    A("- runtime: %s s, CPU-only, no GPU, no network" % report["_runtime_seconds"])
    A("- schema: `%s`" % report["schema"])
    A("")
    A("## 0. The verdict on the published \"2 gaps\" claim")
    A("")
    A(
        "**ILL-POSED as published.**  The number `2` is arithmetically correct "
        "under the Wf-A199 tool's OWN R1/R2/R3 population (the same tool, same "
        "corpus, same descent rule reproduces the same two identities), but the "
        "report presented it as the census of dated rule-#6 records and "
        "published no bound.  R1/R2/R3 are a SUPERSESSION-SEMANTICS filter: "
        "they require a supersession word, so a date-carrying key with no "
        "supersession word never entered the population and could never be "
        "tier-DATED -- even though the tool's own `TIER_ENUM` defines DATED as "
        "\"the key name carries a date-shaped token: a rule-#6 record proper\". "
        "  Under the coverage bound published in §1 below, the true count is "
        "**%d** dated parity gaps (and **%d** records in the whole DATED "
        "population)."
        % (
            counts["parity_gaps_DATED_producer_missing_key"],
            counts["population_records_DATED_tier"],
        )
    )
    A("")
    A(
        "The pre-repair count `2` and its two identities are PRESERVED VERBATIM "
        "in `report.json` under `superseded_measurements` (AGENTS.md §5.5 rule "
        "#6) and in the tool as "
        "`SUPERSEDED_DATED_PARITY_GAPS_PRE_REPAIR_2026_11_02_WFA199`."
    )
    A("")
    A("## 1. The coverage bound (a count without a bound is not a measurement)")
    A("")
    A("### 1.1 Corpus")
    A("")
    A("| item | value |")
    A("|---|---|")
    A(
        "| artefact roots | %s |"
        % ", ".join("`%s`" % r for r in cb["corpus"]["artefact_roots"])
    )
    A("| artefact enumeration | %s |" % cb["corpus"]["artefact_file_selection"])
    A(
        "| artefacts scanned / parsed / non-dict | %d / %d / %d |"
        % (
            cb["corpus"]["n_artefacts_scanned"],
            cb["corpus"]["n_artefacts_parsed"],
            cb["corpus"]["n_artefacts_non_dict"],
        )
    )
    A("| artefact bytes read | %d |" % cb["corpus"]["artefact_bytes_total"])
    A(
        "| producers (every .py, minus exclusions) | %d |"
        % cb["corpus"]["n_producers"]
    )
    A(
        "| artefact corpus manifest sha256 | `%s` |"
        % cb["corpus"]["artefact_corpus_manifest_sha256"]
    )
    A(
        "| producer corpus manifest sha256 | `%s` |"
        % cb["corpus"]["producer_corpus_manifest_sha256"]
    )
    A("| manifest definition | %s |" % cb["corpus"]["manifest_definition"])
    A(
        "| excluded producer trees | %s |"
        % ", ".join(
            "`%s` (%d files)" % (k, v)
            for k, v in sorted(cb["corpus"]["excluded_producer_file_counts"].items())
        )
    )
    A("")
    A("### 1.2 Match rules (the ONE place they are defined: `MATCH_RULES`)")
    A("")
    A("| id | name | added by | effect |")
    A("|---|---|---|---|")
    for r in cb["match_rules"]:
        A(
            "| **%s** | `%s` | %s | %s |"
            % (r["id"], r["name"], r["added_by"], r["population_effect"])
        )
        A("| | %s | | |" % r["definition"])
    A("")
    A("**Tier rule.** %s" % cb["tier_rule"])
    A("")
    A("### 1.3 Descent rule")
    A("")
    A(cb["descent_rule"])
    A("")
    A("### 1.4 Path-scoping rule (producer <-> artefact)")
    A("")
    A(cb["path_scoping_rule"])
    A("")
    A("### 1.5 Path-scoping rule (value match)")
    A("")
    A(cb["value_path_scoping_rule"])
    A("")
    A("### 1.6 Value-parity rule")
    A("")
    A(cb["value_parity_rule"])
    A("")
    A("### 1.7 Units")
    A("")
    for k, v in cb["units"].items():
        A("- **%s** -- %s" % (k, v))
    A("")
    A("## 2. Counts")
    A("")
    A(
        "**Read the row label.**  `population_*` counts EVERY record in the "
        "population (all tiers).  `candidates_*` counts only the records whose "
        "key name is absent from EVERY producer.  `parity_gaps_*` is the "
        "defect surface."
    )
    A("")
    A("| metric | value |")
    A("|---|---|")
    for k, v in counts.items():
        A("| %s | %d |" % (k, v))
    A("")
    A("### 2.1 Classification -- ALL POPULATION records (every tier)")
    A("")
    A("| classification | count |")
    A("|---|---|")
    for k, v in report["classification_counts_all_population_records"].items():
        A("| %s | %d |" % (k, v))
    A("")
    A("### 2.2 Classification -- ALL DATED-tier records only")
    A("")
    A("| classification | count |")
    A("|---|---|")
    for k, v in report["classification_counts_all_dated_records"].items():
        A("| %s | %d |" % (k, v))
    A("")
    A("### 2.3 Classification -- CANDIDATES only (key absent from every producer)")
    A("")
    A("| classification | all candidates | DATED tier only |")
    A("|---|---|---|")
    for k, v in report["classification_counts_candidates"].items():
        A(
            "| %s | %d | %d |"
            % (k, v, report["classification_counts_candidates_DATED"].get(k, 0))
        )
    A("")
    A("### 2.4 Gap class of the DATED parity gaps")
    A("")
    A("| gap class | count |")
    A("|---|---|")
    for k, v in report["gap_class_counts_DATED_parity_gaps"].items():
        A("| %s | %d |" % (k, v))
    A("")
    A(
        "`HAND-AUTHORED (no producer)` is **not** a parity defect: an artefact "
        "with no producer has nothing to be in parity with -- report it, do "
        "NOT \"fix\" it.  `PRODUCER-UNVERIFIED (writer not confirmed)` means a "
        "producer script sits next to the artefact but the writer was not "
        "confirmed.  `PARITY-DEFECT (producer-would-drop)` means the producer "
        "IS confirmed to write the artefact and cannot reproduce the key -- a "
        "re-run DROPS the dated record."
    )
    A("")
    A("## 3. Parity gaps (DATED tier, PRODUCER-MISSING-KEY -- the defect surface)")
    A("")
    gaps = report["parity_gaps"]
    if not gaps:
        A("None.")
    else:
        A("| artefact | key | gap class | pointer/label? | emitted under another name? |")
        A("|---|---|---|---|---|")
        for g in gaps:
            flags = g.get("gap_flags", {})
            A(
                "| `%s` | `%s` | %s | %s | %s |"
                % (
                    g["artefact"],
                    g["key"],
                    g.get("gap_class", "?"),
                    "yes" if flags.get("is_pointer_or_label") else "no",
                    "yes" if flags.get("producer_emits_under_a_different_name")
                    else "no",
                )
            )
    A("")
    A(
        "The UNDATED tier contributes %d further `PRODUCER-MISSING-KEY` rows; they are "
        "listed in `report.json` under "
        "`undated_missing_keys_not_rule6_records` and are **not** rule-#6 records."
        % counts["parity_gaps_UNDATED_producer_missing_key"]
    )
    A("")
    A("## 4. The INVERSE class -- the producer emits a dated key the artefact lacks")
    A("")
    inv = report["producer_emits_key_artefact_lacks"]
    if not inv:
        A("None.")
    else:
        A("| artefact | key | producer |")
        A("|---|---|---|")
        for i in inv:
            A("| `%s` | `%s` | `%s` |" % (i["artefact"], i["key"], i["producer"]))
        A("")
        A(
            "These are NOT parity gaps (nothing is dropped -- the artefact is "
            "STALE with respect to its own producer: a re-run would ADD the "
            "record).  They are the class the Wf-A199 census had no bucket "
            "for, and it is where the one PROVABLE producer/artefact "
            "difference of the Wf-A204 audit lives."
        )
    A("")
    A("## 5. The gap detail")
    A("")
    for g in gaps:
        flags = g.get("gap_flags", {})
        A("### `%s` -- `%s`" % (g["artefact"], g["key"]))
        A("")
        A("- key path: `%s`" % g["key_path"])
        A("- container path: `%s`" % (g.get("container_path") or []))
        A("- tier: **%s**" % g["tier"])
        A("- matched by: %s" % ", ".join(g["match_reasons"]))
        A("- classification: **%s**" % g["classification"])
        A("- gap class: **%s**" % g.get("gap_class", "?"))
        A(
            "- artefact's producer(s): %s"
            % (
                ", ".join("`%s`" % p for p in g["evidence"]["artefact_producers"])
                or "(none found)"
            )
        )
        A(
            "- confirmed writer(s): %s"
            % (
                ", ".join(
                    "`%s`" % p
                    for p in g["evidence"].get("artefact_strong_producers", [])
                )
                or "(none -- the writer is not confirmed)"
            )
        )
        A("- pointer/label (scalar value, not a rendering): **%s**" % (
            "yes" if flags.get("is_pointer_or_label") else "no"))
        A(
            "- the producer emits it under another name: **%s**%s"
            % (
                "yes" if flags.get("producer_emits_under_a_different_name") else "no",
                (
                    " (`%s`)"
                    % ", ".join(
                        "`%s`" % n
                        for n in (flags.get("producer_emits_under_different_name_variants")
                                  or [])
                        + (flags.get("producer_emits_under_different_name_stems") or [])
                    )
                )
                if flags.get("producer_emits_under_a_different_name")
                else "",
            )
        )
        A("- value sha256: `%s`" % g["value_sha256"])
        A("")
    A("## 6. Superseded measurements (AGENTS.md §5.5 rule #6 -- preserved verbatim)")
    A("")
    for sm in report["superseded_measurements"]:
        A(
            "- **%s / %s / %s** -- dated_parity_gaps: %s"
            % (
                sm.get("date", "?"),
                sm.get("wave", "?"),
                sm.get("stage", "?"),
                sm.get("dated_parity_gaps", "(see the verbatim block below)"),
            )
        )
        if "v1_counts_verbatim" in sm:
            A("")
            A("  The v1 count block, preserved VERBATIM:")
            A("")
            A("  | key | value |")
            A("  |---|---|")
            for k, v in sm["v1_counts_verbatim"].items():
                A("  | %s | %d |" % (k, v))
    A("")
    A("## 7. Full candidate list")
    A("")
    A(
        "%d candidate(s): a (artefact, key) pair whose key name is an exact string "
        "constant in NO producer .py." % len(report["candidates"])
    )
    A("")
    for c in report["candidates"]:
        A("### %s -- `%s`" % (c["artefact"], c["key"]))
        A("")
        A("- key path: `%s`" % c["key_path"])
        A("- tier: **%s**" % c["tier"])
        A("- matched by: %s" % ", ".join(c["match_reasons"]))
        A("- classification: **%s**" % c["classification"])
        if c.get("gap_class"):
            A("- gap class: **%s**" % c["gap_class"])
        A(
            "- artefact's producer(s): %s"
            % (
                ", ".join("`%s`" % p for p in c["evidence"]["artefact_producers"])
                or "(none found)"
            )
        )
        A("- value sha256: `%s`" % c["value_sha256"])
        dyn = c["evidence"].get("dynamic_candidates") or []
        if dyn:
            A(
                "- dynamic-build candidates: %s"
                % ", ".join("`%s:%d`" % (d["producer"], d["line"]) for d in dyn[:6])
            )
        A("")
    A("## 8. JSON parse failures (artefacts that could not be read)")
    A("")
    if not report["parse_failures"]:
        A("None.")
    else:
        for f in report["parse_failures"]:
            A("- `%s` -- %s" % (f["artefact"], f["error"]))
    A("")
    A("## 9. What this census did NOT do")
    A("")
    A("- it repaired nothing outside its own scope;")
    A(
        "- it registered no pre-flight gate and did not touch "
        "`pre_flight_check.EXPECTED_GATE_COUNTS`;"
    )
    A("- it did not modify `molmetal/reports/run_ledger.jsonl` or any other ledger;")
    A(
        "- it did not modify `paper/**`, `docs/**`, `TODO/**`, `AGENTS.md`, "
        "`registry/**` or `metrics/**`."
    )
    A(
        "- it did not touch the emitter "
        "`molmetal/scripts/wf_a155_ligand_connected_rate.py` (another wave owns "
        "it) nor any metric file."
    )
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "molmetal" / "reports" / "wfa199_parity_census"),
        help="directory for report.json + report.md",
    )
    ap.add_argument("--json-name", default="report.json")
    ap.add_argument("--md-name", default="md-report.tmp")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    report = run_census(verbose=not args.quiet)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / args.json_name
    md_path = out_dir / "report.md"

    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False)
    if json_path.exists():
        safe_overwrite_check(json_path, payload)
    detect_truncation(payload)
    atomic_write_file(json_path, payload)

    md = render_md(report)
    if md_path.exists():
        safe_overwrite_check(md_path, md)
    detect_truncation(md)
    atomic_write_file(md_path, md)

    print("[wfa199] wrote %s (%d B)" % (json_path, json_path.stat().st_size))
    print("[wfa199] wrote %s (%d B)" % (md_path, md_path.stat().st_size))
    print(
        "[wfa199] population=%d (DATED %d / UNDATED %d) candidates=%d "
        "(DATED %d / UNDATED %d) parity_gaps=%d inverse=%d"
        % (
            report["counts"]["population_records_total"],
            report["counts"]["population_records_DATED_tier"],
            report["counts"]["population_records_UNDATED_tier"],
            report["counts"]["candidates_total"],
            report["counts"]["candidates_DATED_tier"],
            report["counts"]["candidates_UNDATED_tier"],
            report["counts"]["parity_gaps_DATED_producer_missing_key"],
            report["counts"]["producer_emits_key_artefact_lacks"],
        )
    )
    for cls in CLASSIFICATION_ENUM:
        print(
            "[wfa199]   %-40s population=%d dated=%d candidates=%d candidates_dated=%d"
            % (
                cls,
                report["classification_counts_all_population_records"].get(cls, 0),
                report["classification_counts_all_dated_records"].get(cls, 0),
                report["classification_counts_candidates"].get(cls, 0),
                report["classification_counts_candidates_DATED"].get(cls, 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
