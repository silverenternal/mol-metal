# Atomic Write Protocol

**Version:** 1.0.0
**Author:** Wf-386-A3 (autocoded 2026-09-30)
**Status:** SHIPPED
**Source:** `molmetal/scripts/wf386a3_atomic_write.py` + `molmetal/tests/test_wf386a3_atomic_write.py`

---

## §1 Problem statement

Across Wf-371 / Wf-373 / Wf-377 / Wf-378 / Wf-384 we observed a
recurring pattern of **verification reliability + write overwriting**:

1. **Verification reliability**: An agent verifies the on-disk state
   of a previous wave's closeouts (e.g., "Wf-378-A3 SHIPPED 0/5") and
   later waves invalidate that verdict (e.g., "Wf-378 actually SHIPPED
   5/5"). Without a content-stability check, two opposing verdicts
   sit side-by-side in the project's audit trail.

2. **Write overwriting**: An agent (Wf-378-A3, Wf-384-A3) re-authors
   a closeout that was already on disk from a previous wave. The
   rewrite is substantial (≥ 50% line diff) and discards the prior
   evidence trail. The result: someone reading the audit trail later
   sees the new content but not the historical context that motivated
   the original closeout.

3. **Truncated agent output**: An agent's response ends mid-sentence
   (e.g., "...content goes here TODO") and the calling workflow
   accepts it as a complete closeout because the output length exceeds
   the size threshold but lacks structural completeness.

This protocol provides three primitives that close each gap:

| Primitive                       | Closes                          | Function                |
|---------------------------------|---------------------------------|-------------------------|
| Atomic write + rename           | Half-merged file state          | `atomic_write_file`     |
| Overwrite detection             | Substantial content loss        | `safe_overwrite_check`  |
| Truncation detection            | Mid-sentence agent output       | `detect_truncation`     |

---

## §2 Atomic write protocol

### §2.1 What "atomic" means here

A file write is *atomic* if, from the perspective of any other reader,
the file is either fully written or fully absent — never half-written.
On POSIX filesystems, `os.replace(src, dst)` is atomic (single inode
swap). Our implementation uses this guarantee.

### §2.2 Algorithm

1. Acquire a per-path process-local lock (so concurrent writers on the
   same path serialise).
2. Create a temp file in the same directory as the target
   (`{path}.tmp.{pid}.{counter}.part`).
3. Write the content + `fsync` (durability across power loss).
4. `os.replace(tmp, target)` — atomic inode swap.
5. Release the lock.

### §2.3 Concurrent write semantics

* **Same path, multiple threads/processes**: Serialised via the
  per-path lock. Final content is one of the complete writes, never
  a half-merged mix.
* **Different paths**: No serialisation. Independent writers proceed
  in parallel.
* **Failure mid-write**: Temp file is unlinked; target is untouched.

### §2.4 Linux-only guarantee

`os.replace` is atomic on POSIX. On Windows, `os.rename` is NOT
atomic — this module is **Linux-only**. The `os.replace` call will
raise on Windows, so the failure mode is loud, not silent.

---

## §3 Overwrite detection protocol

### §3.1 Heuristic

Substantial-overwrite risk is defined as:

```
risk = existing_nonempty_lines >= min_lines  (default 50)
        AND
        line_set_jaccard_distance > diff_threshold  (default 0.5)
```

We use **line-set Jaccard distance** rather than edit distance
because line-level granularity matches Markdown closeout semantics
(line ordering is stable, content rewrites are common).

### §3.2 When to call

Call `safe_overwrite_check` before any non-trivial write to an
existing path. Typical use:

```python
risk = safe_overwrite_check(path, new_content, raise_on_risk=True)
if risk is not None:
    # Existing content is substantial AND diff exceeds threshold.
    # Caller decides: re-author under a new filename, or proceed with
    # USER_OK-gated confirmation.
```

### §3.3 Defaults are conservative

| Threshold       | Default | Rationale                                                          |
|-----------------|---------|--------------------------------------------------------------------|
| `min_lines`     | 50      | Below 50 non-empty lines, files are small enough to overwrite freely. |
| `diff_threshold`| 0.5     | At 50% line-set Jaccard distance, half the content has been rewritten. |

Both are tuned to the canonical closeout-document shape (50–500 lines
of Markdown). Adjust per use case.

---

## §4 Truncation detection protocol

### §4.1 Heuristic

An agent output is *truncated* if any of:

1. **Empty**: `len(output) == 0`.
2. **Too short**: `len(output) < expected_min_length` (default 200).
3. **Bad trailing token**: Ends with `:`, `,`, `...`, `…`, `TODO`,
   `FIXME`, `WIP`, or `[truncated]`.
4. **Missing closing token**: Does not end with one of:
   - `\n# `, `\n## `, `\n### ` (Markdown next-section header)
   - ` ```\n` or `\n```` (fenced code close)
   - `}\n` (JSON close)
   - `\"\"\"\n` (Python docstring close)
   - `\n` (any newline-ending line, most permissive)
   - `.`, `?`, `!`, `)`, `]` (final punctuation)

This is intentionally **permissive** (matches on common patterns)
rather than **perfect** (won't catch all truncation modes). When in
doubt, call `detect_truncation` and inspect the returned reason.

### §4.2 When to call

Call `detect_truncation` on agent output *before* accepting it as a
complete closeout. Typical use:

```python
reason = detect_truncation(agent_output, expected_min_length=500)
if reason is not None:
    # Output looks truncated. Retry the agent, fall back to a
    # sibling-data reconstruction, or mark as PARTIAL closeout.
    log.warning("Agent output appears truncated: %s", reason)
```

### §4.3 Default `expected_min_length`

200 chars by default. A well-formed Markdown section is ~100–300 chars,
a full closeout is 1000–10000 chars. Set per use case:

| Use case                                | `expected_min_length` |
|-----------------------------------------|-----------------------|
| Single Markdown section                 | 50                    |
| Multi-section closeout (Wf-XXX-A*.md)   | 1000                  |
| Multi-page synthesis (Wf-XXX-*.md)      | 5000                  |

---

## §5 Agent prompt template

When adding atomic-write semantics to a workflow agent, use this
template (copy-paste into the agent prompt):

```
When writing closeout files:
1. ALWAYS use atomic_write_file(path, content) — never Path.write_text().
2. ALWAYS call safe_overwrite_check(path, content) before writing to an
   existing path. If risk > 0.5, either:
   (a) use a different filename (e.g., add a suffix),
   (b) confirm USER_OK for re-authoring,
   (c) stop and report the risk to the orchestrator.
3. After writing, call detect_truncation(content, expected_min_length=1000).
   If flagged, retry the agent with a longer expected output.
4. The atomic_write_file function is at:
   molmetal/scripts/wf386a3_atomic_write.py
5. See docs/ATOMIC_WRITE_PROTOCOL.md for full documentation.
```

---

## §6 Examples

### §6.1 Basic atomic write (new file)

```python
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

atomic_write_file(
    "TODO/completed/386a3_atomic_write_protocol.md",
    "# Closeout\n\nFull content here.\n",
)
```

### §6.2 Overwrite with explicit confirmation

```python
from molmetal.scripts.wf386a3_atomic_write import safe_overwrite_check

# Step 1: detect risk.
risk = safe_overwrite_check(
    "TODO/completed/378a3_cpu_only_gaps.md",
    new_content,
    raise_on_risk=False,
)
if risk is not None:
    print(f"Overwrite risk: {risk:.3f}. Confirm USER_OK before proceeding.")
    # User says OK → proceed with raise_on_risk=False (log warning).
    # Or user says no → use a new filename with a suffix.

# Step 2: write atomically (after confirmation).
atomic_write_file("TODO/completed/378a3_cpu_only_gaps.md", new_content)
```

### §6.3 Truncation check on agent output

```python
from molmetal.scripts.wf386a3_atomic_write import detect_truncation

reason = detect_truncation(agent_output, expected_min_length=1000)
if reason is not None:
    log.warning("Agent output appears truncated: %s", reason)
    # Retry the agent with a longer expected output.
else:
    # Output is complete — proceed with the closeout.
    atomic_write_file(path, agent_output)
```

### §6.4 High-level `write_closeout`

```python
from molmetal.scripts.wf386a3_atomic_write import write_closeout

# Combines safe_overwrite_check + atomic_write_file + post-write
# truncation check.
write_closeout(
    "TODO/completed/386a3_atomic_write_protocol.md",
    content,
    min_size=1000,
    raise_on_overwrite=False,  # log warning, proceed
)
```

---

## §7 Cross-references

| File / artifact                                  | Purpose                                      |
|--------------------------------------------------|----------------------------------------------|
| `molmetal/scripts/wf386a3_atomic_write.py`       | Implementation (this protocol's source).     |
| `molmetal/tests/test_wf386a3_atomic_write.py`    | Test suite (27 cases).                       |
| `TODO/completed/386a3_atomic_write_protocol.md`  | Wf-386-A3 closeout doc.                    |
| `TODO/completed/384_final_synthesis.md`          | Predecessor wave context.                   |
| `TODO/completed/378_final_synthesis.md`          | Wf-378 closeout that prompted this fix.     |
| `docs/HONEST_NEGATIVE_CATALOG.md`                | Honest-negative tracking.                   |
| `wf302d_retry_with_correction.py`                | Sibling defence for StructuredOutput schema confusion. |

---

## §8 Honest framing

This protocol was authored on 2026-09-30 in response to the recurring
"verification reliability + write overwriting" pattern observed
across Wf-371 / Wf-373 / Wf-377 / Wf-378 / Wf-384. It is:

* **MEASURED-effective for the documented failure modes** (per the
  27 test cases in `test_wf386a3_atomic_write.py`).
* **NOT a replacement for careful agent design**. The protocol is a
  *belt-and-braces* defence — agents should still verify before
  writing, and orchestrators should still cross-check closeouts.
* **NOT a substitute for USER_OK-gated USER_OK-gated destructive
  actions**. Per AGENTS.md §5.1, USER_OK is still required for any
  action that destroys prior work or launches GPU resources.

Adoption is **opt-in**. Existing closeout-drafting scripts continue
to work as before; new scripts should call this protocol where
appropriate.

---

**END OF PROTOCOL** — Cross-ref `wf302d_retry_with_correction.py` for
the StructuredOutput-confusion twin; cross-ref
`docs/HONEST_NEGATIVE_CATALOG.md` for honest-negative tracking.