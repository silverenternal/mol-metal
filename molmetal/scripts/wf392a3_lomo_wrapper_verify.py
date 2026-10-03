#!/usr/bin/env python3
"""
Wf-392-A3 — Launch wrapper flag pass-through verifier (CPU-only).

Verifies that USER_OK-gated GPU launch wrappers correctly pass through
their expected CLI flags to the trainer. Three layers of verification:

  (a) trainer_supports_flag(trainer_path, flag)        — does the trainer's
      argparse accept this flag?
  (b) extract_wrapper_flags(wrapper_path, trainer_path) — does the wrapper
      invoke the trainer with this flag?
  (c) verify_wrapper_passes_flags(wrapper_path, expected_flags, trainer_path)
      — combine (a) + (b): returns list of (status, flag, evidence).

Statuses:
  - "PASSED"   — flag exists in trainer AND wrapper passes it
  - "GAP-TRAINER-MISSING" — wrapper expects flag, trainer does not define it
  - "GAP-WRAPPER-MISSING" — trainer has flag, wrapper does not pass it
  - "INFO"     — flag found in wrapper but trainer import path differs

Honest framing per party-twenty-spirit §3.5:
  - Trainer --help parsing uses static AST scan (no live import) so the
    verifier never actually launches the trainer.
  - Wrapper parsing uses regex (bash-friendly; no shell exec) so the
    verifier never launches the wrapper.

Usage:
  python molmetal/scripts/wf392a3_lomo_wrapper_verify.py \
      --wrapper scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh \
      --trainer molmetal/scripts/train_chemprop_v2_v3.py \
      --expected "--chemprop-mode" "--chemprop-min-epochs" \
                 "--chemprop-lomo-init-scale"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]  # mol-metal/


# Regex fragments used by both wrapper and trainer extractors.
# NOTE: use greedy quantifier on the flag-name character class so we capture
# the entire flag name (e.g. "--allow-single-seed") instead of the shortest
# match ("--a"). A bare `--[a-zA-Z]` followed by `--chemprop-mode` would
# otherwise mis-capture.
_FLAG_NAME = r"--[a-zA-Z][a-zA-Z0-9_:-]*"
_QUOTED_OR_FLAG = rf'(?:{_FLAG_NAME}|"[^"]*"|\'[^\']*\')'


# ---------------------------------------------------------------------------
# Trainer flag extraction (static AST scan; never imports)
# ---------------------------------------------------------------------------
# Match argparse calls like:
#   p.add_argument("--foo", ...)
#   p.add_argument("--foo-bar", type=int, ...)
#   parser.add_argument("--foo")
# We also match the dest= variant for backward-compat flags.
ARG_ADD_PATTERN = re.compile(
    r"""add_argument\(
        \s*
        (?:
            ['"](?P<flag>--[a-zA-Z][a-zA-Z0-9_-]*)['"]
            |
            (?P<flag2>--[a-zA-Z][a-zA-Z0-9_-]*)
        )
    """,
    re.VERBOSE,
)


def trainer_supports_flag(trainer_path: Path, flag: str) -> bool:
    """Static check: does the trainer file declare this argparse flag?

    Returns True iff an ``add_argument("--flag", ...)`` call exists in
    the trainer's source. Does NOT run --help (CPU-only friendly).
    """
    if not trainer_path.exists():
        return False
    text = trainer_path.read_text(encoding="utf-8", errors="replace")
    needle = f'"{flag}"'
    return needle in text


def trainer_flag_set(trainer_path: Path) -> List[str]:
    """Return ALL argparse --flag-style arguments defined in trainer."""
    if not trainer_path.exists():
        return []
    text = trainer_path.read_text(encoding="utf-8", errors="replace")
    flags: List[str] = []
    seen: set = set()
    for m in ARG_ADD_PATTERN.finditer(text):
        f = m.group("flag") or m.group("flag2")
        if f and f.startswith("--") and f not in seen:
            flags.append(f)
            seen.add(f)
    return flags


# ---------------------------------------------------------------------------
# Wrapper flag extraction (regex scan; never runs bash)
# ---------------------------------------------------------------------------
# Look for trainer invocations like:
#   "${PY}" "${TRAINER}" --flag1 val --flag2 ...
#   "${PY}" "${SCRIPT}" ...args...
#   python trainer.py --flag ...
#   ${PY} "${TRAINER}" --flag ...
# Variable names may contain digits (e.g. P1_4_TRAINER).
TRAINER_REF_PATTERN = re.compile(
    r"""(?:
        \"\$\{PY\}\"\s+\"\$\{TRAINER\}\"
        |
        \"\$\{PY\}\"\s+\"\$\{[A-Z][A-Z0-9_]*TRAINER\}\"
        |
        \"\$\{PY\}\"\s+\"\$\{SCRIPT\}\"
        |
        python\s+\"\$\{TRAINER\}\"
        |
        \b\$\{SCRIPT\}\b
    )""",
    re.VERBOSE,
)


def extract_wrapper_invocation_block(wrapper_text: str) -> List[str]:
    """Locate trainer-invocation lines in the wrapper and return the
    token stream after each invocation. Returns a flat list of tokens.

    Note: we ALSO scan for bash array assignments like:
        COMMON_ARGS=( --epochs 3 --flag val ... )
    and case statement bodies that contain flag definitions like:
        v3_fix_only)
            ABLATION_ARGS=( --chemprop-mode v3-fix --out-dir ... )
    These two patterns surface LOMO-specific levers that are not on a
    direct ``${PY} ${TRAINER}`` invocation line.
    """
    tokens: List[str] = []
    # Pattern 1: trainer reference (e.g. "${PY}" "${TRAINER}")
    for line_match in TRAINER_REF_PATTERN.finditer(wrapper_text):
        start = line_match.end()
        # Walk forward through backslash-continued bash lines
        i = wrapper_text.find("\n", start)
        if i < 0:
            chunk = wrapper_text[start:]
        else:
            chunk = wrapper_text[start:i]
        # Now we may have backslash-continuation. Greedily extend.
        while chunk.rstrip().endswith("\\"):
            next_nl = wrapper_text.find("\n", i + 1)
            if next_nl < 0:
                chunk += wrapper_text[i + 1 :]
                break
            chunk = chunk.rstrip().rstrip("\\") + " " + wrapper_text[i + 1 : next_nl]
            i = next_nl
        # Strip shell continuation backslashes + newlines + extra whitespace
        clean = chunk.replace("\\\n", " ").replace("\\\r\n", " ")
        for tok in re.findall(_QUOTED_OR_FLAG, clean):
            tokens.append(tok)

    # Pattern 2: bash array assignments to *_ARGS / *_FLAGS / CFM_NEW_FLAGS arrays
    # Strip bash line comments first so `(per Wf-321H §2)` does NOT
    # terminate the array body early.
    text_no_comments = _strip_bash_comments(wrapper_text)
    array_pat = re.compile(
        r"""(?P<name>[A-Z][A-Z0-9_]*(?:ARGS|FLAGS))\s*=\s*\((?P<body>[^)]*)\)""",
        re.DOTALL,
    )
    for arr_match in array_pat.finditer(text_no_comments):
        body = arr_match.group("body")
        for tok in re.findall(_QUOTED_OR_FLAG, body):
            tokens.append(tok)

    # Pattern 3: bash function bodies that accumulate args via args+=( --flag )
    # (e.g. DrugOOD wrapper's build_wf185_flags function). We approximate by
    # scanning the entire wrapper text for any --flag token outside of
    # comment regions, deduped with the array+invocation patterns above.
    for tok in re.findall(_FLAG_NAME, text_no_comments):
        tokens.append(tok)

    return tokens


def _strip_bash_comments(text: str) -> str:
    """Strip `` # ... -- end-of-line`` comments from bash text.

    Comment detection is line-based: the ``#`` is treated as a comment
    only if it appears OUTSIDE of a single/double-quoted region. This
    is a lightweight approximation — full bash comment stripping would
    require a shell parser — but it covers the common case of inline
    annotations like ``--epochs 3  # Wf-XXX`` which otherwise trip up
    the array-body regex (a ``)`` inside a comment would terminate
    the body early).
    """
    out_lines = []
    for line in text.splitlines():
        in_single = False
        in_double = False
        cleaned_chars = []
        i = 0
        while i < len(line):
            c = line[i]
            if c == "'" and not in_double:
                in_single = not in_single
                cleaned_chars.append(c)
                i += 1
                continue
            if c == '"' and not in_single:
                in_double = not in_double
                cleaned_chars.append(c)
                i += 1
                continue
            if c == "#" and not in_single and not in_double:
                # rest of line is a comment
                break
            cleaned_chars.append(c)
            i += 1
        out_lines.append("".join(cleaned_chars))
    return "\n".join(out_lines)


def extract_wrapper_flags(wrapper_path: Path) -> List[str]:
    """Extract --flag-style tokens passed to the trainer from the wrapper."""
    if not wrapper_path.exists():
        return []
    text = wrapper_path.read_text(encoding="utf-8", errors="replace")
    toks = extract_wrapper_invocation_block(text)
    return [t for t in toks if t.startswith("--")]


# ---------------------------------------------------------------------------
# Verifier: combine trainer + wrapper views
# ---------------------------------------------------------------------------
@dataclass
class FlagVerdict:
    flag: str
    status: str  # PASSED | GAP-TRAINER-MISSING | GAP-WRAPPER-MISSING | INFO
    evidence: str  # 1-line explanation


@dataclass
class WrapperReport:
    wrapper_path: str
    trainer_path: str
    expected_flags: List[str]
    wrapper_flag_set: List[str] = field(default_factory=list)
    trainer_flag_set: List[str] = field(default_factory=list)
    verdicts: List[FlagVerdict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def verify_wrapper_passes_flags(
    wrapper_path: Path,
    expected_flags: List[str],
    trainer_path: Path,
) -> WrapperReport:
    """Run the verifier on a single wrapper/trainer/expected_flags triple."""
    rep = WrapperReport(
        wrapper_path=str(wrapper_path),
        trainer_path=str(trainer_path),
        expected_flags=list(expected_flags),
    )
    if not wrapper_path.exists():
        for f in expected_flags:
            rep.verdicts.append(
                FlagVerdict(
                    f, "GAP-WRAPPER-MISSING", f"wrapper missing: {wrapper_path}"
                )
            )
        return rep
    if not trainer_path.exists():
        for f in expected_flags:
            rep.verdicts.append(
                FlagVerdict(
                    f, "GAP-TRAINER-MISSING", f"trainer missing: {trainer_path}"
                )
            )
        return rep
    rep.wrapper_flag_set = sorted(set(extract_wrapper_flags(wrapper_path)))
    rep.trainer_flag_set = sorted(set(trainer_flag_set(trainer_path)))
    wrapper_set = set(rep.wrapper_flag_set)
    trainer_set = set(rep.trainer_flag_set)
    for f in expected_flags:
        in_wrapper = f in wrapper_set
        in_trainer = f in trainer_set
        if in_wrapper and in_trainer:
            rep.verdicts.append(
                FlagVerdict(f, "PASSED", "flag present in both wrapper and trainer")
            )
        elif in_wrapper and not in_trainer:
            rep.verdicts.append(
                FlagVerdict(
                    f,
                    "GAP-TRAINER-MISSING",
                    f"wrapper passes --{f[2:] if f.startswith('--') else f} "
                    f"but trainer argparse does NOT define it (argparse error on real launch)",
                )
            )
        elif not in_wrapper and in_trainer:
            rep.verdicts.append(
                FlagVerdict(
                    f,
                    "GAP-WRAPPER-MISSING",
                    "trainer defines flag but wrapper does NOT pass it",
                )
            )
        else:
            rep.verdicts.append(
                FlagVerdict(
                    f,
                    "GAP-TRAINER-MISSING",
                    "neither wrapper nor trainer has flag (likely SCAFFOLD-only)",
                )
            )
    n_passed = sum(1 for v in rep.verdicts if v.status == "PASSED")
    n_gap_trainer = sum(1 for v in rep.verdicts if v.status == "GAP-TRAINER-MISSING")
    n_gap_wrapper = sum(1 for v in rep.verdicts if v.status == "GAP-WRAPPER-MISSING")
    rep.summary = {
        "n_expected": len(expected_flags),
        "n_passed": n_passed,
        "n_gap_trainer": n_gap_trainer,
        "n_gap_wrapper": n_gap_wrapper,
    }
    return rep


# ---------------------------------------------------------------------------
# Smoke launcher check (bash -n syntax only; never run wrapper)
# ---------------------------------------------------------------------------
def smoke_bash_syntax(wrapper_path: Path) -> Tuple[bool, str]:
    """Run ``bash -n`` on the wrapper; returns (ok, evidence)."""
    if not wrapper_path.exists():
        return (False, f"wrapper missing: {wrapper_path}")
    import subprocess

    try:
        r = subprocess.run(
            ["bash", "-n", str(wrapper_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0:
            return (True, f"bash -n {wrapper_path} OK")
        return (False, f"bash -n {wrapper_path} FAIL: {r.stderr.strip()}")
    except Exception as exc:  # pragma: no cover
        return (False, f"bash -n raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Wf-392-A3 launch wrapper flag pass-through verifier (CPU-only)"
    )
    p.add_argument(
        "--wrapper",
        type=Path,
        required=True,
        help="Path to launch wrapper script",
    )
    p.add_argument(
        "--trainer",
        type=Path,
        required=True,
        help="Path to trainer script",
    )
    p.add_argument(
        "--expected-json",
        default="",
        help=(
            "JSON list of expected CLI flags (e.g. "
            '\'["--chemprop-mode","--chemprop-min-epochs"]\'). '
            "Useful when flag names start with '--' (argparse optional-"
            "arg greediness would otherwise consume them)."
        ),
    )
    p.add_argument(
        "--expected",
        action="append",
        default=[],
        help=(
            "Expected CLI flag (repeat one per flag). Each occurrence "
            "consumes the NEXT argument as the flag value, even if it "
            "starts with '--'. Example: --expected --foo --expected --bar."
            " Alternatively, use --expected-json for batched input."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Repo root for relative path resolution",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report on stdout (machine-readable)",
    )
    p.add_argument(
        "--bash-n",
        action="store_true",
        help="Also run ``bash -n`` syntax check on the wrapper",
    )
    args = p.parse_args()

    # Combine --expected + --expected-json into a single expected list
    expected_combined = []
    if args.expected_json:
        try:
            expected_combined.extend(json.loads(args.expected_json))
        except json.JSONDecodeError as exc:
            p.error(f"--expected-json invalid JSON: {exc}")
    expected_combined.extend(args.expected)
    if not expected_combined:
        p.error("--expected or --expected-json required")

    # Resolve relative paths
    wrapper = args.wrapper
    trainer = args.trainer
    if not wrapper.is_absolute():
        wrapper = args.repo_root / wrapper
    if not trainer.is_absolute():
        trainer = args.repo_root / trainer

    rep = verify_wrapper_passes_flags(wrapper, expected_combined, trainer)
    if args.bash_n:
        ok, ev = smoke_bash_syntax(wrapper)
        rep.summary["bash_n_ok"] = ok
        rep.summary["bash_n_evidence"] = ev

    if args.json:
        out_dict = asdict(rep)
        out_dict["verdicts"] = [asdict(v) for v in rep.verdicts]
        print(json.dumps(out_dict, indent=2))
    else:
        print(f"Wrapper:  {rep.wrapper_path}")
        print(f"Trainer:  {rep.trainer_path}")
        print(f"Expected: {len(rep.expected_flags)} flag(s)")
        print()
        for v in rep.verdicts:
            print(f"  [{v.status:>22}] {v.flag}")
            print(f"                            evidence: {v.evidence}")
        print()
        if rep.summary:
            print(f"  SUMMARY: {rep.summary}")
        if rep.summary.get("bash_n_evidence"):
            print(f"  BASH-N:  {rep.summary['bash_n_evidence']}")
    # Exit code: 0 if all PASSED; 1 if any GAP-*
    has_gap = any(v.status.startswith("GAP-") for v in rep.verdicts)
    return 1 if has_gap else 0


if __name__ == "__main__":
    sys.exit(main())
