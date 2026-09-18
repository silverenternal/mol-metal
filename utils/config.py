"""Lightweight YAML config loader.

Tries to use PyYAML; falls back to a minimal hand-rolled parser that supports
the subset of YAML used in `configs/default.yaml` (nested mappings via
indentation, scalar values as strings/numbers/booleans/null, comments).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # pyyaml is the preferred loader
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - pyyaml missing
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _coerce(value: str) -> Any:
    """Coerce a raw string scalar into int / float / bool / None / str."""
    s = value.strip()
    if s == "" or s.lower() in ("null", "~"):
        return None
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # Numbers (int first, then float).
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Strip surrounding quotes if present.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _mini_yaml_load(text: str) -> Any:
    """Very small indentation-based YAML loader.

    Supports a top-level mapping with nested mappings via 2-space indent.
    Lists are not used by the project's configs and are not parsed.
    """
    root: dict = {}
    stack: list = [(-1, root)]  # list of (indent, container)

    for raw_line in text.splitlines():
        # Strip comments and trailing whitespace.
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("-"):
            # Lists are not supported in the mini parser.
            raise NotImplementedError(
                "The fallback YAML parser does not support list items; "
                "install pyyaml (`uv pip install pyyaml`)."
            )

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            raise ValueError(f"Malformed YAML line: {raw_line!r}")
        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()

        # Pop stack until current indent is greater than parent's indent.
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            # Mapping header.
            new_dict: dict = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _coerce(value)

    return root


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load a YAML config and return it as a nested dict.

    Parameters
    ----------
    path : path-like, optional
        Path to the YAML file. Defaults to `configs/default.yaml`.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    RuntimeError
        If pyyaml is missing and the fallback parser can't handle the file.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")

    if _HAS_YAML:
        loaded = yaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Top-level YAML in {cfg_path} must be a mapping, got "
                f"{type(loaded).__name__}"
            )
        return loaded

    # Fallback: hand-rolled parser.
    try:
        loaded = _mini_yaml_load(text)
    except NotImplementedError as exc:
        raise RuntimeError(
            "pyyaml is not installed and the fallback parser cannot handle "
            "this config. Install pyyaml with `uv pip install pyyaml`."
        ) from exc
    return loaded


__all__ = ["load_config"]
