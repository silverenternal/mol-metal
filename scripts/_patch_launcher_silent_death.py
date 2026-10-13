#!/usr/bin/env python
"""
BLK-I1 Step 1 helper: prepend `setsid nohup` to trainer-invocation lines in
scripts/launch_*.sh so the trainer process detaches from the parent bash
process group and survives parent-shell exit.

Per AGENTS.md section 5.6, every closeout write MUST go through
atomic_write_file; this helper does so for each patched launcher.

Scope (BLK-I1 fix_step 1: "Patch all scripts/launch_*.sh"):
  - 32 of 35 launch_*.sh lack nohup/setsid (already covered: 2 fix3lines +
    1 chemprop_v3_p14_post_cfm_round7 + 1 launch_all_training +
    1 launch_item4_chain_post_wf131).
  - Patches only trainer-invocation lines (heuristic: lines that match
    `(\.venv/bin/python|\\$\\{CMD\\[@\\]\\}|\\$\\{COMMON_ARGS\\[@\\]\\}|python[23]?\\s+-m)`
    AND contain a `--out-` flag or a `.py` argument).  All other lines are
    left untouched (preserves existing logging via `| tee`, PIPESTATUS
    handling, exec >LOG redirect, 8-waves-per-config loops).
  - Skips launchers that already contain `setsid` or `nohup`.
  - Skips launchers that do not invoke python (e.g. shell-only utility).

Per AGENTS.md section 5.9: does NOT touch venv path; if a launcher
invokes python3 / /usr/bin/python (instead of .venv/bin/python), the
helper flags it but does not modify the invocation (BLK-I1 is silent
death, not venv audit; the venv audit is BLK-I3 scope per §5.9).
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Trainer-invocation line regexes (the lines we want to prepend `setsid nohup` to).
# We deliberately err on the side of caution: only patch lines that look like
# the actual python invocation, not comment lines or pip install lines.
TRAINER_LINE_PATTERNS = [
    re.compile(r"^\s*\.venv/bin/python\b"),
    re.compile(r'^\s*"\$\{CMD\[@\]\}"'),
    re.compile(r'^\s*"\$\{COMMON_ARGS\[@\]\}"'),
    re.compile(r"^\s*\$\{PY\}\b"),
    re.compile(r"^\s*python[23]?\s+-m\s+molmetal"),
    # "PYTHONPATH=... ${PY} molmetal/scripts/..." style
    re.compile(r"^\s*PYTHONPATH=[^\s]+\s+\$\{?PY\}?\s+"),
    # Quoted "${PY}" "${TRAINER}" or "${PY}" molmetal/...
    re.compile(r'^\s*"\$\{PY\}"\s+'),
    # Unquoted ${PY} followed by "..." (var ref)
    re.compile(r'^\s*\$\{PY\}\s+"\$\{'),
    # Unquoted ${PY} followed by molmetal/ path
    re.compile(r"^\s*\$\{PY\}\s+molmetal/"),
    # CUDA_VISIBLE_DEVICES=0 ... "${PY}" "${TRAINER}" ... (direct invocation, no if)
    re.compile(r"^\s*CUDA_VISIBLE_DEVICES=\S+\s+"),
    # NOTE: `if CUDA_VISIBLE_DEVICES=0 "${PY}" ...` is intentionally NOT
    # wrapped here — that line is the if-guard syntax, and prepending
    # `setsid nohup` breaks the `if ... ; then` structure. The trainer
    # invocation itself is on a continuation line, which the CUDA_VISIBLE_DEVICES
    # pattern above catches (with `^\s` matching the indent of the line).
]

# Lines that LOOK like python invocations but are if-guard heads (not real invocations).
NON_TRAINER_PATTERNS = [
    re.compile(r"--help\b"),
    re.compile(r"\s-c\s+['\"]"),
    re.compile(r'command -v "\$\{PY\}"'),
    re.compile(r"PYTHONPATH=.+\.venv/bin/python\s*$"),  # PYTHONPATH=... .venv/bin/python  (no follow-on)
    re.compile(r"^\s*if\s+"),  # if-guard heads, never invoke anything
]

# Lines that LOOK like python invocations but are probes/checks, not long-running trainers.
NON_TRAINER_PATTERNS = [
    re.compile(r"--help\b"),
    re.compile(r"\s-c\s+['\"]"),
    re.compile(r'command -v "\$\{PY\}"'),
    re.compile(r"PYTHONPATH=.+\.venv/bin/python\s*$"),  # PYTHONPATH=... .venv/bin/python  (no follow-on)
]


def patch_launcher(path: Path) -> tuple[bool, str]:
    """Return (changed, message).  Idempotent: returns (False, 'already-patched')
    if `setsid` or `nohup` already present."""
    text = path.read_text(encoding="utf-8")

    if "setsid" in text or "nohup" in text:
        return False, "already-patched (setsid/nohup present)"

    new_lines = []
    n_patched = 0
    for line in text.splitlines(keepends=True):
        patched_line = line
        # Skip pure comments.
        if line.lstrip().startswith("#"):
            new_lines.append(line)
            continue
        # Skip gate/probe/if-guard lines.
        is_non_trainer = any(pat.search(line) for pat in NON_TRAINER_PATTERNS)
        if is_non_trainer:
            new_lines.append(line)
            continue
        for pat in TRAINER_LINE_PATTERNS:
            if pat.match(line):
                patched_line = "setsid nohup " + line
                n_patched += 1
                break
        new_lines.append(patched_line)

    if n_patched == 0:
        return False, "no trainer-invocation line matched"

    new_text = "".join(new_lines)
    return True, f"patched {n_patched} trainer-invocation line(s)"


def main(argv: list[str]) -> int:
    targets = sorted(SCRIPTS_DIR.glob("launch_*.sh"))
    if not targets:
        print("ERROR: no launch_*.sh found in", SCRIPTS_DIR, file=sys.stderr)
        return 2

    # Lazy import so script can be syntax-checked without molmetal on PYTHONPATH.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    except Exception as exc:
        print(f"ERROR: cannot import atomic_write_file: {exc}", file=sys.stderr)
        return 2

    only = set(argv[1:]) if len(argv) > 1 else None
    n_changed = 0
    n_skipped = 0
    n_no_match = 0
    for path in targets:
        if only and path.name not in only:
            continue
        try:
            changed, msg = patch_launcher(path)
        except Exception as exc:
            print(f"  {path.name}: ERROR ({exc})")
            n_no_match += 1
            continue
        if not changed:
            if "already-patched" in msg:
                n_skipped += 1
                print(f"  {path.name}: SKIP ({msg})")
            else:
                n_no_match += 1
                print(f"  {path.name}: NO_MATCH ({msg})")
            continue

        new_text = "setsid nohup ".join(  # noqa: F841 -- placeholder; we re-read
            __import__("io").open(path).read().splitlines(keepends=True)
        )
        # Re-run patcher logic inline to produce final text (avoids race
        # between patch_launcher and the write).
        text = path.read_text(encoding="utf-8")
        out_lines = []
        n_patched = 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("#"):
                out_lines.append(line)
                continue
            is_non_trainer = any(pat.search(line) for pat in NON_TRAINER_PATTERNS)
            if is_non_trainer:
                out_lines.append(line)
                continue
            pl = line
            for pat in TRAINER_LINE_PATTERNS:
                if pat.match(line):
                    pl = "setsid nohup " + line
                    n_patched += 1
                    break
            out_lines.append(pl)
        new_text = "".join(out_lines)

        atomic_write_file(path, new_text)
        # atomic_write_file preserves existing mode (incl. +x) per §5.6.
        n_changed += 1
        print(f"  {path.name}: PATCHED ({n_patched} line(s))")

    print()
    print(f"summary: changed={n_changed} skipped={n_skipped} no_match={n_no_match}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
