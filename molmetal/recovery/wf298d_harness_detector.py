"""Wf-298D harness-layer StructuredOutput auto-fallback (project-level alias).

Wf-338-V2: thin project-level import path for the Wf-298D harness
detector wrapper. The canonical implementation lives at
``molmetal/reports/wf298d_harness_detector/wrapper.py`` (per the
Wf-298D ship report at
``molmetal/reports/wf298d_harness_detector/wrapper_report.md``).
This module is a project-level alias so downstream consumers
(``scripts/wf320b_harness_autofallback.py``,
``scripts/wf302d_retry_with_correction.py``, the Wf-298D recovery
integration in Wf-338-V2) can import a single, stable path:

    from molmetal.recovery.wf298d_harness_detector import (
        HarnessOutputExtractor,
        FallbackExtraction,
        fast_filter_says_schema,
    )

The Wf-298D ship surface (see wrapper_report.md §3) defines the
detector in terms of three diagnostic signals:

  1. Nested object carries schema meta-keys (type / properties /
     required / items / enum / $ref / format / description).
  2. Nested object's ``type`` value is a JSON-Schema type keyword
     (string/integer/number/boolean/array/object/null) AND the nested
     object has <= 3 keys (schema nodes are short; data nodes are not).
  3. Hybrid empty-string pattern from Wf-271D (>= 1/2 of top-level
     values are empty strings AND the object carries schema root
     markers).

This module re-exports the canonical symbols without modification
so the wrapper_report.md contract remains the source of truth and
any future Wf-298D evolution (Wf-298E prompt-text-only briefs /
Wf-298F retry-with-explicit-correction) can flow through one
canonical import.

Wf-291C parent root-cause investigation:
``molmetal/reports/wf291c_pipeline_root_cause/root_cause.md``
"""
from __future__ import annotations

# Re-export from the canonical Wf-298D ship surface.
# Importing from a sub-module path keeps the canonical implementation
# as the single source of truth and avoids drift.
from molmetal.reports.wf298d_harness_detector.wrapper import (
    HarnessOutputExtractor,
    FallbackExtraction,
    fast_filter_says_schema,
)

__all__ = [
    "HarnessOutputExtractor",
    "FallbackExtraction",
    "fast_filter_says_schema",
]
