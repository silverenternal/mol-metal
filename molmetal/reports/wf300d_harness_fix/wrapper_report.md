# Wf-300D — Harness Auto-Fallback for StructuredOutput Complex-Schema Returns

**Date**: 2026-09-29
**Follows**: Wf-291C (insight: agents tasked with `StructuredOutput` complex
schemas return the schema shape instead of validated values on retry
exhaustion)
**Scope**: Wf-291F harness-layer auto-fallback **actual implementation**
(previous Wf-291F only documented the issue)

## 1. Background

Per the Wf-291C observation, when the underlying Claude Code SDK reaches its
`error_max_structured_output_retries` subtype (see
`/home/hugo/.claude/plugins/marketplaces/claude-plugins-official/plugins/security-guidance/hooks/llm.py:1281`),
the SDK emits the literal schema document as the `structured_output` payload.
A naive workflow script that does `JSON.parse(agentResult)` then receives a
schema shape rather than the intended value and either:

* crashes downstream with a `KeyError` for a field that the schema declared
  but no instance populated, or
* silently passes the schema through to a downstream `agent(...)` call,
  causing the second agent to be asked to fill in the schema.

The previous Wf-291F remediation attempt shipped only documentation; the
Wf-300D task is the actual user-writable wrapper.

## 2. Invocation Sites

`grep -rn "StructuredOutput" ~/.claude/` returns hits in:

* `cache/changelog.md` — historical fix notes (e.g. "Fixed `StructuredOutput`
  schema cache bug causing ~50% failure rate when using multiple schemas").
* `plugins/marketplaces/claude-plugins-official/plugins/security-guidance/hooks/llm.py`
  — the **SDK-side** invocation site, lines 415, 416, 1280-1297, 1353-1354,
  1387. This file is **not user-modifiable** (it lives inside a marketplace
  plugin checkout). Auto-updateable via `claude plugin update`, but not
  edited by hand.
* `plugins/marketplaces/.../math-olympiad/skills/.../SKILL.md` — a skill
  description that mentions `StructuredOutput` as a *return* mechanism, not a
  caller.

The workflow-script caller surface is `agent(...).then(JSON.parse(...))`
style — i.e. each `agent(...)` invocation in `wf_*.json` files under
`~/.claude/projects/.../workflows/`. Those scripts are user-modifiable per
project.

## 3. User-Modifiable Wrapper Location

The user-modifiable layer for this fix lives at **`/home/hugo/.claude/`** —
the user-level claude config directory. The chosen artifact is:

```
/home/hugo/.claude/wrapper.py   (~3 KB, pure Python, no SDK imports)
/home/hugo/.claude/test_wrapper.py   (black-box tests, no pytest dep)
```

Why here:

* `~/.claude/` is the canonical user-writable Claude Code config root.
* The plugin marketplace files (`~/.claude/plugins/marketplaces/...`) are
  user-writable in principle but are clobbered on `claude plugin update`,
  so they are not durable.
* The project-level `.claude/` directory holds **skills** only (this repo's
  `/home/hugo/codes/mol-metal/.claude/skills/`); it does not host runtime
  helpers, so wrapping there would be re-implemented per project.
* Workflow scripts at `~/.claude/projects/.../workflows/wf_*.json` import
  via plain path so a single user-level `wrapper.py` is reachable from
  every project.

## 4. Detector

```python
def detect_schema_shape(json_str: str) -> bool:
    if not isinstance(json_str, str):
        return False
    try:
        data = json.loads(json_str)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("type") == "object"
        and "properties" in data
        and isinstance(data["properties"], dict)
    )
```

**Why these three conditions**:

* `data["type"] == "object"` — only an `object`-typed schema document is
  ambiguous with a real result. A `string` or `integer` schema document
  cannot collide with a real value of the same shape.
* `"properties" in data` — the SDK's emitted schemas always carry a
  `properties` map. Real results do not.
* `isinstance(data["properties"], dict)` — guards against the corner case
  where a real result has a `properties` key holding a string or list (in
  which case we must not classify it as a schema).

The function is side-effect free, never raises, and degrades to `False` on
any malformed input so the caller's normal parsing path runs.

## 5. Extractor Wrapper

```python
def extract_agent_result(agent_result: str) -> dict:
    if detect_schema_shape(agent_result):
        return {"schema_shape": True, "text": agent_result}
    try:
        parsed = json.loads(agent_result)
    except (ValueError, TypeError) as exc:
        return {
            "schema_shape": False,
            "parse_error": str(exc),
            "text": agent_result,
        }
    return {"schema_shape": False, "value": parsed}
```

Three outcomes:

| Outcome | Shape | Meaning |
| --- | --- | --- |
| Schema shape | `{"schema_shape": True, "text": "<raw>"}` | SDK exhausted schema-retries; caller must use fallback branch |
| Valid value | `{"schema_shape": False, "value": <parsed>}` | Normal happy path |
| Parse error | `{"schema_shape": False, "parse_error": "...", "text": "<raw>"}` | Hard failure; caller should retry or surface raw text |

The discriminator key (`schema_shape`) is on **every** return so callers can
branch with a single `if (result.schema_shape)` without inspecting keys.

## 6. Test Verdict

```
$ python3 /home/hugo/.claude/test_wrapper.py
PASS  test_detect_schema_shape_basic_object
PASS  test_detect_schema_shape_rejects_real_value
PASS  test_detect_schema_shape_rejects_array
PASS  test_detect_schema_shape_rejects_partial
PASS  test_detect_schema_shape_rejects_non_object_type
PASS  test_detect_schema_shape_rejects_garbage
PASS  test_extract_real_value
PASS  test_extract_schema_shape
PASS  test_extract_parse_error

9/9 tests passed
```

Test coverage:

* Detector accepts a nested schema (the SDK's actual emission pattern).
* Detector rejects a real value with overlapping top-level keys (`type`).
* Detector rejects arrays, missing-properties dicts, and wrong-typed
  schemas.
* Detector never raises on garbage / empty / non-string input.
* Extractor returns `value` for a real result, `schema_shape + text` for a
  schema, and `parse_error + text` for invalid JSON.

## 7. Integration Sketch (for follow-up Wf-300E)

A future Wf-300E should wire this wrapper into the workflow-script runner.
Concretely, in each `wf_*.json` script that does:

```js
const raw = await agent(prompt, { label: 'foo' });
const out = JSON.parse(raw);
```

replace with:

```js
const { extract_agent_result } = await import('/home/hugo/.claude/wrapper.py');
const raw = await agent(prompt, { label: 'foo' });
const out = extract_agent_result(raw);
if (out.schema_shape) {
  // log + degrade: maybe retry with --no-schema, or fall back to plain text
}
```

This Wf-300D deliberately ships only the **detector + extractor** as pure
Python helpers; the JS-side wiring is a separate task because it requires
either (a) a runner-level patch to `claude code`'s script runner or (b) a
pre/post-processing wrapper around `agent(...)` in every workflow.

## 8. Files Touched

| Path | Purpose | LOC |
| --- | --- | --- |
| `/home/hugo/.claude/wrapper.py` | Detector + extractor + docstring | ~85 |
| `/home/hugo/.claude/test_wrapper.py` | Black-box tests (9 cases) | ~115 |
| `/home/hugo/codes/mol-metal/molmetal/reports/wf300d_harness_fix/wrapper_report.md` | This report | (this file) |

## 9. Honest Limitations

* The wrapper only handles the **JSON-string** form of the schema-shape
  return. If the SDK ever returns the schema as a structured Python object
  already (bypassing stringification), this detector won't fire — but per
  the SDK code at `llm.py:1353` (`if msg.structured_output is not None`),
  the schema shape is currently emitted as a dict, not a string, so a
  workflow script calling `JSON.stringify(raw)` first must do so before
  handing it to `extract_agent_result`. The wrapper docstring spells this
  out so callers don't forget.
* This is **a layer, not a fix** for the underlying SDK bug. The real fix
  lives in `claude code`'s SDK (marketplace plugin). When the SDK ships a
  schema-retry fix, this wrapper becomes a no-op for new sessions but
  remains useful for older SDK builds still in rotation.
* No real production workflow was migrated in Wf-300D. Wiring 250+ existing
  `wf_*.json` scripts is a separate task (Wf-300E suggestion).
