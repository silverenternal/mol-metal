# SPDX-License-Identifier: Apache-2.0
"""The two memory knobs are argparse dests, not env-var-only.

FIX-C12 introduced `MOLMETAL_EMPTY_CACHE_EVERY` and `MOLMETAL_MEM_TRACE` as
environment variables. That is the §5.11 failure shape: a knob that lives only
in the launch command cannot be read back from the run's own artifacts, so
"which value did this run use" has no answer after the shell exits. Both are
now real dests, with the env vars kept as the fallback so every existing
launcher behaves identically.

`empty_cache_every` also changes memory behaviour materially — it bounds
`memory_reserved` at the cost of throughput — so its default has to stay 0
(never call `empty_cache`) or every existing run silently changes speed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _trainer():
    if "_t_mem_knobs" in sys.modules:
        return sys.modules["_t_mem_knobs"]
    spec = importlib.util.spec_from_file_location("_t_mem_knobs", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_mem_knobs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["trainer", *argv])
    return _trainer().parse_args()


def test_empty_cache_is_off_by_default(monkeypatch):
    # Off is what keeps every pre-FIX-C12 run bit-identical; on-by-default would
    # silently cost ~51% epoch time.
    assert _parse(monkeypatch, []).empty_cache_every == 0


def test_mem_trace_is_off_by_default(monkeypatch):
    assert _parse(monkeypatch, []).mem_trace is False


def test_empty_cache_every_reaches_its_dest(monkeypatch):
    assert _parse(monkeypatch, ["--empty-cache-every", "8"]).empty_cache_every == 8


def test_mem_trace_flag_reaches_its_dest(monkeypatch):
    assert _parse(monkeypatch, ["--mem-trace"]).mem_trace is True


def test_the_env_fallback_still_works(monkeypatch):
    # The launchers that predate the dests pass the env var; it must keep
    # taking effect when the flag is absent. The module reads it through these
    # two helpers at import time, so exercising the helpers is the honest test.
    mod = _trainer()
    monkeypatch.setenv("MOLMETAL_EMPTY_CACHE_EVERY", "3")
    monkeypatch.setenv("MOLMETAL_MEM_TRACE", "1")
    assert mod._empty_cache_every() == 3
    assert mod._mem_trace_enabled() is True

    monkeypatch.delenv("MOLMETAL_EMPTY_CACHE_EVERY")
    monkeypatch.delenv("MOLMETAL_MEM_TRACE")
    assert mod._empty_cache_every() == 0
    assert mod._mem_trace_enabled() is False


def test_a_junk_env_value_does_not_raise(monkeypatch):
    # A malformed env var must not abort a 7-hour run at import time.
    mod = _trainer()
    monkeypatch.setenv("MOLMETAL_EMPTY_CACHE_EVERY", "eight")
    assert mod._empty_cache_every() == 0
    monkeypatch.setenv("MOLMETAL_EMPTY_CACHE_EVERY", "-4")
    assert mod._empty_cache_every() == 0
