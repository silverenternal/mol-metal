# SPDX-License-Identifier: Apache-2.0
"""Wf-A160 — independent verification of the EF-TALFM bond-threshold rule.

WHY THIS FILE EXISTS
--------------------
``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::radii_scaled_factor``
implements the bond-candidacy threshold as ``lambda * math.pi`` and its
docstring asserts that this is "exactly ``lambda * pi``, as quoted" from
EF-TALFM (arXiv 2609.08333).

Wf-A156's frontier branch fetched the paper's LaTeX source and reported that
the paper's rule is ``tau_ij <- lambda_POLICY * (r_cov(z_i) + r_cov(z_j))``
— i.e. the ``pi`` is a SUBSCRIPT on ``lambda`` (the reconstruction policy),
NOT a multiplicative constant.  That is a large enough claim (it changes the
threshold by a factor of pi) that it must not rest on ONE agent's report.

This file re-derives the verdict from the paper's own source, independently,
and records the provenance (the e-print's sha256) so the verdict is
MEASURED-with-a-path rather than asserted.

WHAT IT DOES
------------
1. Fetches ``https://arxiv.org/e-print/2609.08333`` (or reuses a cached
   copy whose sha256 matches ``EXPECTED_EPRINT_SHA256``).
2. Extracts it into an empty scratch directory and locates the threshold
   rule in ``src/appendix.tex``.
3. Records:
   * the rule line(s) verbatim,
   * the ``ChoosePolicy`` line that defines what ``pi`` is,
   * whether ANY ``lambda * pi`` product appears anywhere in the source,
   * the derivation of the appendix letter and the algorithm number,
   * whether the all-atom branch uses Open Babel for connectivity.

CPU-ONLY: this file downloads a LaTeX tarball and greps text.  It never
imports torch, never touches a GPU, and never trains or decodes.

HONEST FRAMING
--------------
MEASURED: every string and count below is read out of the fetched source.
PROJECTED: nothing.  This file makes no claim about what the threshold does
to any decoded pool — that is UNMEASURED under the 0-GPU mandate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

#: The e-print's sha256, MEASURED 2026-11-01 by an independent fetch.
#: A future fetch that does not match this hash is reported as a DRIFT and
#: the verdict is NOT re-derived (the paper may have been revised).
EXPECTED_EPRINT_SHA256 = (
    "e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224"
)

EPRINT_URL = "https://arxiv.org/e-print/2609.08333"

#: The rule as it appears in the paper, and the pattern that recognises it.
RULE_LINE_RE = re.compile(
    r"Compute bond threshold\s*\$\\tau_\{ij\}\s*\\gets\s*\\lambda_\\pi\s*"
    r"\\big\(r\^\{\\mathrm\{cov\}\}\(\\hat z_i\)\s*\+\s*"
    r"r\^\{\\mathrm\{cov\}\}\(\\hat z_j\)\\big\)"
)

POLICY_LINE_RE = re.compile(
    r"Select reconstruction policy\s*\$\\pi\s*\\gets\s*"
    r"\\texttt\{ChoosePolicy\}"
)

#: Any product of the constant pi with a lambda, in any TeX spelling.
#:
#: The negative lookahead after ``\\lambda`` is LOAD-BEARING: the paper's own
#: usage is ``\lambda_\pi`` (pi as a SUBSCRIPT), which must NOT be counted
#: as a product.  A subscripted lambda is excluded.
LAMBDA_TIMES_PI_RE = re.compile(
    r"\\lambda(?!_)[^%\n]{0,40}?\\pi|\\pi[^%\n]{0,40}?\\lambda(?!_)",
    re.IGNORECASE,
)

#: pi as a *subscript* on lambda (the paper's actual usage).
LAMBDA_SUBSCRIPT_PI_RE = re.compile(r"\\lambda_\{?\\pi\}?")

#: pi as a *policy* being selected.
POLICY_DEF_RE = re.compile(r"\\pi\s*\\gets\s*\\texttt\{ChoosePolicy\}")

OPENBABEL_RE = re.compile(r"Open Babel", re.IGNORECASE)

#: ``\section[short]{long}`` — the paper uses the optional short form, so a
#: regex that stops at the first brace MISSES those sections (and the
#: appendix-letter derivation silently shifts).  The LONG title (the last
#: braced group) is the one the paper prints.
SECTION_RE = re.compile(
    r"\s*\\(?P<kind>subsub|sub|)section\*?(?:\[[^\]]*\])?\{(?P<title>.+?)\}"
)

_SECTION_LEVELS = {"": 1, "sub": 2, "subsub": 3}


def _section_title_from_line(line: str) -> Optional[Dict[str, object]]:
    """Return ``{level, title}`` for a ``\\section``/``\\subsection`` line."""
    m = SECTION_RE.match(line)
    if not m:
        return None
    level = _SECTION_LEVELS[m.group("kind")]
    title = m.group("title").replace("\\\\", " ").strip()
    return {"level": level, "title": title}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_eprint(cache_dir: Path, *, offline: bool = False) -> Dict[str, object]:
    """Fetch (or reuse) the e-print tarball; return its provenance.

    The tarball is written to ``cache_dir`` and its sha256 recorded.  The
    tarball is never extracted in place — extraction goes to a fresh
    temp directory (AGENTS.md: do not run interpreters over downloaded
    archives).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball = cache_dir / "2609.08333.tar.gz"

    if offline:
        if not tarball.exists():
            raise SystemExit(f"offline mode but no cached tarball at {tarball}")
        fetched = False
    else:
        cmd = [
            "curl", "-sL", "-A", "mol-metal-audit/1.0", "-o", str(tarball),
            EPRINT_URL,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise SystemExit(f"curl failed rc={proc.returncode}: {proc.stderr[:300]}")
        fetched = True

    digest = sha256_file(tarball)
    return {
        "url": EPRINT_URL,
        "path": str(tarball),
        "bytes": tarball.stat().st_size,
        "sha256": digest,
        "sha256_matches_expected": digest == EXPECTED_EPRINT_SHA256,
        "fetched_this_run": fetched,
    }


def extract_to_scratch(tarball: Path, scratch: Path) -> Path:
    """Extract the tarball into ``scratch`` (which must be empty)."""
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as tf:
        # Python 3.12+: filter="data".  Older: extractall with a guard.
        try:
            tf.extractall(scratch, filter="data")
        except TypeError:  # pragma: no cover - older tarfile
            tf.extractall(scratch)
    return scratch


def _tex_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.tex"))


def locate_rule(root: Path) -> Dict[str, object]:
    """Locate the threshold rule and its context inside the extracted source."""
    rule_hits: List[Dict[str, object]] = []
    policy_hits: List[Dict[str, object]] = []
    lambda_pi_subscript_hits: List[Dict[str, object]] = []
    lambda_times_pi_hits: List[Dict[str, object]] = []
    openbabel_hits: List[Dict[str, object]] = []
    algorithm_starts: List[int] = []
    section_starts: List[Dict[str, object]] = []

    for tex in _tex_files(root):
        rel = str(tex.relative_to(root))
        lines = tex.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, start=1):
            if RULE_LINE_RE.search(line):
                rule_hits.append({"file": rel, "line": i, "text": line.strip()})
            if POLICY_LINE_RE.search(line):
                policy_hits.append({"file": rel, "line": i, "text": line.strip()})
            if LAMBDA_SUBSCRIPT_PI_RE.search(line):
                lambda_pi_subscript_hits.append(
                    {"file": rel, "line": i, "text": line.strip()[:200]}
                )
            if LAMBDA_TIMES_PI_RE.search(line):
                lambda_times_pi_hits.append(
                    {"file": rel, "line": i, "text": line.strip()[:200]}
                )
            if OPENBABEL_RE.search(line):
                openbabel_hits.append(
                    {"file": rel, "line": i, "text": line.strip()[:240]}
                )
            if "\\begin{algorithm}" in line:
                algorithm_starts.append(len(algorithm_starts) + 1)
            sec = _section_title_from_line(line)
            if sec:
                section_starts.append(
                    {
                        "file": rel,
                        "line": i,
                        "level": sec["level"],
                        "title": sec["title"],
                    }
                )

    return {
        "rule_hits": rule_hits,
        "policy_hits": policy_hits,
        "lambda_pi_subscript_hits": lambda_pi_subscript_hits,
        "lambda_times_pi_hits": lambda_times_pi_hits,
        "openbabel_hits": openbabel_hits,
        "algorithm_count": len(algorithm_starts),
        "section_starts": section_starts,
    }


def derive_reference_numbers(root: Path, rule_hit: Dict[str, object]) -> Dict[str, object]:
    """Derive the appendix letter / section / algorithm number of the rule.

    The paper has exactly ONE ``\\appendix`` (main.tex) and ONE appendix file,
    so the top-level ``\\section``s inside it are Appendix A, B, C ... in
    document order.  Algorithms are numbered in document order across the
    WHOLE source (the main text has none).

    NOTE: the appendix file uses the ``\\section[short]{long}`` form, so a
    parser that stops at the first brace misses those sections and shifts
    every letter by one.  :func:`_section_title_from_line` handles it.
    """
    appendix_file = "src/appendix.tex"
    hit_line = int(rule_hit["line"])
    hits: List[Dict[str, object]] = []

    appendix_path = root / appendix_file
    if not appendix_path.exists():
        appendix_path = next(
            (p for p in root.rglob("appendix.tex")), appendix_path
        )
    sections = _section_starts_in(appendix_path)
    top = [s for s in sections if s["level"] == 1]

    containing = None
    for s in top:
        if s["line"] <= hit_line:
            containing = s
    if containing is None and top:
        containing = top[0]
    letter = chr(ord("A") + top.index(containing)).upper() if containing else "?"

    # walk down the levels: each level's index is the count of same-level
    # headings between the parent heading and the rule line
    path_parts = [letter]
    titles: List[str] = []
    if containing:
        titles.append(str(containing["title"]))
        lower = int(containing["line"])
        for level in (2, 3):
            same = [
                s
                for s in sections
                if s["level"] == level and lower < s["line"] <= hit_line
            ]
            if not same:
                break
            chosen = same[-1]
            path_parts.append(str(len([s for s in same if s["line"] <= chosen["line"]])))
            titles.append(str(chosen["title"]))
            lower = int(chosen["line"])

    # The float is PHYSICALLY placed under the LAST heading before it, but
    # the paper INTRODUCES it from the reconstruction subsection via
    # ``\Cref{algo:reconstruction}``.  Both references are reported; the
    # second is the one a reader follows.
    semantic_parts = list(path_parts)
    semantic_titles = list(titles)
    cref_sections = [
        s
        for s in sections
        if _references_algorithm(root / appendix_file, int(s["line"]))
    ]
    if cref_sections:
        sem = cref_sections[-1]
        sem_top = None
        for t in top:
            if int(t["line"]) <= int(sem["line"]):
                sem_top = t
        sem_idx = top.index(sem_top) if sem_top is not None else 0
        sem_letter = chr(ord("A") + sem_idx).upper()
        sem_parts = [sem_letter]
        sem_titles: List[str] = []
        parent = sem_top
        for level in (2, 3):
            same = [
                s
                for s in sections
                if s["level"] == level and parent is not None
                and int(parent["line"]) < s["line"] <= int(sem["line"])
            ]
            if not same:
                break
            chosen = same[-1]
            sem_parts.append(
                str(len([s for s in same if s["line"] <= chosen["line"]]))
            )
            sem_titles.append(str(chosen["title"]))
            parent = chosen
        semantic_parts = sem_parts
        semantic_titles = sem_titles

    # algorithm number + caption: count \begin{algorithm} in document order
    algo_no = 0
    caption = ""
    for tex in _tex_files(root):
        lines = tex.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, start=1):
            if "\\begin{algorithm}" in line:
                algo_no += 1
                for j in range(i, min(i + 6, len(lines))):
                    if "\\caption" in lines[j]:
                        caption = lines[j].strip()
                        break
            if (
                tex.name == Path(str(rule_hit["file"])).name
                and i == hit_line
            ):
                return {
                    "appendix_section_by_containment": "Appendix "
                    + ".".join(path_parts),
                    "appendix_section_introducing_the_algorithm": "Appendix "
                    + ".".join(semantic_parts),
                    "algorithm_number": algo_no,
                    "algorithm_caption": caption,
                    "appendix_top_sections": [s["title"] for s in top],
                    "heading_titles_by_containment": titles,
                    "heading_titles_by_reference": semantic_titles,
                }
    return {
        "appendix_section_by_containment": "Appendix " + ".".join(path_parts),
        "appendix_section_introducing_the_algorithm": "Appendix "
        + ".".join(semantic_parts),
        "algorithm_number": algo_no,
        "algorithm_caption": caption,
        "appendix_top_sections": [s["title"] for s in top],
        "heading_titles_by_containment": titles,
        "heading_titles_by_reference": semantic_titles,
    }


def _references_algorithm(path: Path, from_line: int) -> bool:
    """True if the heading AT ``from_line`` CROSS-REFERENCES the algorithm.

    Only ``\\Cref{...}`` / ``\\cref{...}`` count — the algorithm's own
    ``\\caption{...}\\label{algo:reconstruction}`` must NOT, or the float's
    own caption would make its enclosing heading look like the referrer.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = from_line - 1
    if idx < 0 or idx >= len(lines):
        return False
    window = "\n".join(lines[idx : idx + 40])
    return bool(re.search(r"\\[Cc]ref\{algo:reconstruction\}", window))


def _section_starts_in(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        sec = _section_title_from_line(line)
        if sec:
            out.append({"line": i, "level": sec["level"], "title": sec["title"]})
    return out


def verify(*, cache_dir: Path, offline: bool = False) -> Dict[str, object]:
    prov = fetch_eprint(cache_dir, offline=offline)
    if not prov["sha256_matches_expected"]:
        return {
            "verdict": "DRIFT",
            "reason": (
                "the e-print sha256 differs from the MEASURED "
                f"{EXPECTED_EPRINT_SHA256}; the paper may have been revised, "
                "so the verdict is NOT re-derived from a changed source"
            ),
            "provenance": prov,
        }

    with tempfile.TemporaryDirectory(prefix="wfa160_src_") as td:
        root = extract_to_scratch(Path(str(prov["path"])), Path(td))
        found = locate_rule(root)
        if not found["rule_hits"]:
            return {
                "verdict": "NOT-FOUND",
                "reason": "the tau_ij rule pattern did not match any .tex line",
                "provenance": prov,
                "scan": found,
            }
        rule_hit = found["rule_hits"][0]
        refs = derive_reference_numbers(root, rule_hit)
        scan = {
            k: v
            for k, v in found.items()
            if k not in ("rule_hits", "policy_hits")
        }

    verdict = (
        "THE-PI-IS-A-SUBSCRIPT"
        if found["lambda_pi_subscript_hits"] and not found["lambda_times_pi_hits"]
        else "INCONCLUSIVE"
    )
    return {
        "verdict": verdict,
        "provenance": prov,
        "rule_line": rule_hit,
        "policy_line": (found["policy_hits"][0] if found["policy_hits"] else None),
        "scan": scan,
        "reference_numbers": refs,
        "repo_claim_under_test": (
            "molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::"
            "radii_scaled_factor returns lambda * math.pi and its docstring "
            "says the factor is 'exactly lambda * pi, as quoted'"
        ),
        "repo_claim_verdict": (
            "REFUTED — the paper quotes a lambda with pi as a SUBSCRIPT "
            "(the reconstruction policy), and NO lambda*pi product appears "
            "anywhere in the source"
            if verdict == "THE-PI-IS-A-SUBSCRIPT"
            else "INCONCLUSIVE"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        default="/tmp/wfa160_arxiv_cache",
        help="where the e-print tarball is cached",
    )
    ap.add_argument("--offline", action="store_true", help="reuse the cached tarball")
    ap.add_argument("--out", default="", help="write the JSON verdict here")
    args = ap.parse_args()

    result = verify(
        cache_dir=Path(args.cache_dir), offline=bool(args.offline)
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("verdict") == "THE-PI-IS-A-SUBSCRIPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
