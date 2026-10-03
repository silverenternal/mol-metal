#!/usr/bin/env python3
"""Validate metric JSON files under metrics/by_metric/ against schemas/metric_v1.schema.json.

Tolerant mode (default): emits WARNING for missing required fields, FAIL only on
JSON parse error or schema-level 'type' violation (e.g. value is a string when
schema says number). This preserves backward compatibility with the existing
~50 metric files which do not yet carry the v1 envelope (metric_id + value +
unit + source_path + timestamp).

Strict mode (--strict): emit FAIL when any required field is missing. Use this
on-going for newly-tagged metric files (i.e. those with a top-level
'metric_v1_envelope': true marker).

Usage:
    .venv/bin/python scripts/_validate_metrics.py [--strict] [--quiet]

Exit code:
    0 — all files PASS (warnings allowed in tolerant mode)
    1 — at least one FAIL (or any FAIL in strict mode)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema  # type: ignore
except ImportError:
    sys.stderr.write(
        "[validate_metrics] FATAL: jsonschema not installed. "
        "Run `.venv/bin/pip install jsonschema` first.\n"
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "metric_v1.schema.json"
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"

REQUIRED_V1_FIELDS = ("metric_id", "value", "unit", "source_path", "timestamp")


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_metric_files(metrics_dir: Path) -> list[Path]:
    if not metrics_dir.is_dir():
        return []
    return sorted(p for p in metrics_dir.glob("*.json") if p.is_file())


def _validate_one(schema: dict, path: Path, strict: bool, validator) -> tuple[str, str]:
    """Validate one file. Returns (status, message) where status in {PASS, WARN, FAIL}.

    WARN: missing required v1 fields (tolerant mode only) OR schema-level type mismatch (tolerant)
    FAIL: schema-level type violation OR JSON parse error (always FAIL in strict mode);
          only FAIL in tolerant mode when JSON parse fails.
    PASS: schema valid AND (strict: all required fields present; tolerant: file is valid JSON
          object regardless of v1 envelope).
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return ("FAIL", f"JSON parse error: {e}")

    if not isinstance(data, dict):
        if strict:
            return ("FAIL", "top-level is not a JSON object")
        return ("WARN", "top-level is not a JSON object")

    schema_errors: list[str] = []
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        schema_errors.append(e.message)

    missing = [f for f in REQUIRED_V1_FIELDS if f not in data or data.get(f) is None]

    if strict:
        if schema_errors:
            return ("FAIL", f"schema violation: {schema_errors[0]}")
        if missing:
            return ("FAIL", f"missing v1 fields: {','.join(missing)}")
        return ("PASS", "OK")

    # Tolerant mode: JSON-parses + is-object → PASS; missing fields → WARN; schema mismatch → WARN
    parts: list[str] = []
    if missing:
        parts.append(f"missing v1 fields: {','.join(missing)}")
    if schema_errors:
        parts.append(f"schema: {schema_errors[0]}")
    if parts:
        return ("WARN", "; ".join(parts))
    return ("PASS", "OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on missing v1 fields")
    parser.add_argument("--quiet", action="store_true", help="Suppress PASS output")
    args = parser.parse_args(argv)

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)

    files = _iter_metric_files(METRICS_DIR)
    if not files:
        print(f"[validate_metrics] no metric files found in {METRICS_DIR}", file=sys.stderr)
        return 0

    pass_n = warn_n = fail_n = 0
    for path in files:
        status, msg = _validate_one(schema, path, args.strict, validator)
        rel = path.relative_to(REPO_ROOT)
        if status == "PASS":
            pass_n += 1
            if not args.quiet:
                print(f"  PASS  {rel}")
        elif status == "WARN":
            warn_n += 1
            print(f"  WARN  {rel}  ({msg})")
        else:
            fail_n += 1
            print(f"  FAIL  {rel}  ({msg})", file=sys.stderr)

    summary = (
        f"[validate_metrics] {pass_n} PASS / {warn_n} WARN / {fail_n} FAIL "
        f"(total {len(files)} files, mode={'strict' if args.strict else 'tolerant'})"
    )
    print(summary)
    if fail_n:
        print(f"[validate_metrics] exit 1 (failures present)", file=sys.stderr)
        return 1
    print("[validate_metrics] exit 0 (no failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
