"""check_submission_ready.py — arXiv submission pre-flight (F4 backlog item).

Walks the 9-section checklist in ``paper/SUBMISSION_CHECKLIST.md`` and
reports PASS/FAIL per item.  Expensive gates (full pipeline, evaluator
suite, pitfall guards) are reported as DEFER — caller must invoke the
upstream scripts in §3 of the checklist and confirm PASS manually.

Usage::

    cd /home/hugo/codes/mol-metal
    python3 molmetal/scripts/check_submission_ready.py            # text
    python3 molmetal/scripts/check_submission_ready.py --json     # JSON

Exit codes:
    0 — all auto-checked items PASS (deferred items allowed)
    1 — at least one auto-checked item FAILed
    2 — the checker itself could not run (missing repo root, etc.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper"
METRICS = REPO_ROOT / "metrics"

# ---------------------------------------------------------------- checks -----

def chk_pdf() -> tuple[bool, str]:
    p = PAPER / "main.pdf"
    if not p.exists():
        return False, "paper/main.pdf missing"
    age_days = (datetime.now(timezone.utc).timestamp() - p.stat().st_mtime) / 86400
    sz = p.stat().st_size
    ok = sz >= 100_000 and age_days <= 14
    return ok, f"{sz/1024:.0f} KB, age {age_days:.1f}d"


def chk_main_tex_clean() -> tuple[bool, str]:
    p = PAPER / "main.tex"
    if not p.exists():
        return False, "paper/main.tex missing"
    src = p.read_text(errors="replace")
    n_todo = len(re.findall(r"\\TODO\{", src)) + src.count("TODO: author list")
    return (n_todo == 0), f"{n_todo} TODO placeholders remain"


def chk_undefined_refs() -> tuple[bool, str]:
    log = PAPER / "main.log"
    if not log.exists():
        return False, "main.log missing — run pdflatex 4-pass first"
    txt = log.read_text(errors="replace")
    n_undef = len(re.findall(r"Warning: Reference .* undefined", txt))
    return n_undef == 0, f"{n_undef} undefined-reference warnings"


def chk_author_populated() -> tuple[bool, str]:
    p = PAPER / "main.tex"
    src = p.read_text(errors="replace")
    bad = "TODO: author list" in src
    has_corr = bool(re.search(r"\\correspondingauthor\{[^}]+\}", src))
    ok = (not bad) and has_corr
    return ok, f"TODO={'yes' if bad else 'no'}, corresponding={'yes' if has_corr else 'no'}"


def chk_appendices() -> tuple[bool, str]:
    ap = PAPER / "appendices"
    if not ap.is_dir():
        return False, "paper/appendices/ missing"
    need = {"closure_theorem.tex", "betanf_semantics.tex"}
    have = {p.name for p in ap.glob("*.tex")}
    missing = need - have
    return not missing, f"have {sorted(have & need)}, missing {sorted(missing) or 'none'}"


def chk_supplementary() -> tuple[bool, str]:
    supp = PAPER / "supp"
    single = PAPER / "supplementary.pdf"
    has_dir = supp.is_dir() and any(supp.glob("*.pdf"))
    has_single = single.exists() and single.stat().st_size >= 200_000
    ok = has_dir or has_single
    where = "supp/" if has_dir else ("supplementary.pdf" if has_single else "—")
    return ok, f"at {where}"


def chk_cover_letter() -> tuple[bool, str]:
    p = PAPER / "cover_letter.txt"
    if not p.exists():
        return False, "paper/cover_letter.txt missing"
    n = len(p.read_text(errors="replace"))
    return 200 <= n <= 3500, f"{n} chars"


def chk_metrics_count() -> tuple[bool, str]:
    bm = METRICS / "by_metric"
    if not bm.is_dir():
        return False, "metrics/by_metric/ missing"
    jsons = list(bm.glob("*.json"))
    ok = len(jsons) >= 20
    return ok, f"{len(jsons)} metric JSONs"


def chk_metric_json_shape() -> tuple[bool, str]:
    bm = METRICS / "by_metric"
    bad = []
    for p in bm.glob("*.json"):
        try:
            d = json.loads(p.read_text(errors="replace"))
        except json.JSONDecodeError:
            bad.append(p.name + "(invalid JSON)")
            continue
        if not isinstance(d, dict) or "value" not in d:
            bad.append(p.name)
    return not bad, f"{len(bad)} malformed" if bad else "all have 'value'"


def chk_limitations_section() -> tuple[bool, str]:
    p = PAPER / "sections" / "06_limitations.tex"
    if not p.exists():
        return False, "paper/sections/06_limitations.tex missing"
    n = sum(1 for _ in p.read_text(errors="replace").splitlines())
    return n >= 80, f"{n} lines"


def chk_future_work() -> tuple[bool, str]:
    p = PAPER / "sections" / "07_future.tex"
    return (p.exists(), "present" if p.exists() else "missing")


def chk_overclaim() -> tuple[bool, str]:
    main = (PAPER / "main.tex").read_text(errors="replace")
    sections_dir = PAPER / "sections"
    sec_src = "\n".join((sections_dir / f).read_text(errors="replace")
                        for f in ("04_results.tex",) if (sections_dir / f).exists())
    blob = main + "\n" + sec_src
    flagged = []
    for phrase in (r"\bSOTA\b", r"\bstate-of-the-art\b", r"\bunprecedented\b"):
        for m in re.finditer(phrase, blob, re.IGNORECASE):
            ctx = blob[max(0, m.start()-40):m.end()+40].replace("\n", " ")
            if "qualified" not in ctx.lower() and "benchmark" not in ctx.lower():
                flagged.append(m.group(0))
    return (not flagged, f"flagged: {sorted(set(flagged)) or 'none'}")


def chk_traceability() -> tuple[bool, str]:
    p = PAPER / "traceability.csv"
    return (p.exists(), "present" if p.exists() else "missing")


# Deferred gates (run the upstream scripts and confirm externally).
DEFERRED = [
    ("3.1 full pipeline", "bash molmetal/scripts/wf_full_pipeline.sh"),
    ("3.2 14 evaluators", "python3 molmetal/scripts/eval_all_evaluators.py"),
    ("3.3 7 pitfall guards", "python3 molmetal/scripts/wf_pitfall_guard/check_all.py"),
    ("3.4 seed-0 reproduce", "run twice, sha256 the ckpt"),
]

CHECKS = [
    ("1.1 main.pdf fresh", chk_pdf),
    ("1.2 main.tex clean", chk_main_tex_clean),
    ("1.9 pdflatex clean", chk_undefined_refs),
    ("2.1+2.2 author populated", chk_author_populated),
    ("1.5 appendices present", chk_appendices),
    ("1.6 supplementary PDF", chk_supplementary),
    ("1.7 cover letter", chk_cover_letter),
    ("1.8 traceability.csv", chk_traceability),
    ("4.1 metrics count ≥ 20", chk_metrics_count),
    ("4.4 metric JSON shape", chk_metric_json_shape),
    ("5.1 limitations §6", chk_limitations_section),
    ("5.2 future work §7", chk_future_work),
    ("5.3 no overclaim", chk_overclaim),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    results = []
    n_pass = n_fail = 0
    for label, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"checker error: {type(exc).__name__}: {exc}"
        results.append({"id": label, "status": "PASS" if ok else "FAIL", "detail": detail})
        n_pass += int(ok)
        n_fail += int(not ok)

    deferred = [{"id": d[0], "cmd": d[1], "status": "DEFER"} for d in DEFERRED]

    if args.json:
        print(json.dumps({"auto": results, "deferred": deferred,
                          "summary": {"pass": n_pass, "fail": n_fail}}, indent=2))
    else:
        print("arXiv pre-flight — auto-checked items")
        print("=" * 60)
        for r in results:
            tag = "PASS" if r["status"] == "PASS" else "FAIL"
            print(f"  [{tag}] {r['id']:30s}  {r['detail']}")
        print()
        print("Deferred (must be invoked manually before submission):")
        for d in deferred:
            print(f"  [DEFER] {d['id']:24s}  {d['cmd']}")
        print()
        print(f"summary: {n_pass} PASS, {n_fail} FAIL, {len(deferred)} DEFER")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())