# Wf-A16 — close-out (2026-10-03)

## §0 TL;DR

Wf-A16 had five deliverables.  Four landed in code; the fifth — a register
document that **seventeen test modules already referenced by name** — had never
been written, so those tests pointed at a file that did not exist.  That gap is
closed here (the register itself, plus this close-out).

## §1 What landed (evidence is the point, per AGENTS.md §5.5)

| # | Deliverable | On-disk evidence |
|---:|---|---|
| 1 | **Gate path split** — `p3_wf343c4_anchor_freshness_24h` and `p3_wf343c2_per_wave_runtime_check` moved OFF the commit path and onto the launch path | `grep -n '"all": 103' molmetal/scripts/pre_flight_check.py` |
| 2 | **`--gates NAME[,NAME...]` explicit-name mode** so a launch config can enforce exactly the gates it declares, ignoring `--pipeline` | `molmetal/scripts/pre_flight_check.py:12603` |
| 3 | **Import-health guard generalised** — detection keys on the defect SHAPE (`@dataclass` immediately followed by `def`), not on one wave's helper name; population widened from "marked scripts" to every script | `ls -la molmetal/tests/test_script_import_health.py` |
| 4 | **17 uncollectable test modules converted to loud SKIPs** (module-level import of a symbol that does not exist => collection ERROR under `--strict-markers`) | `grep -rl 'Wf-A16: feature not implemented' molmetal/tests/*.py \| wc -l` -> `17` |
| 5 | **The register** (this wave's missing deliverable) | see §2 |

### Raw `ls -la` evidence

```
$ -rw-r--r-- 1 hugo hugo 9971 Oct  3 22:13 TODO/completed/wfa16_unimplemented_feature_register.md
$ -rw-r--r-- 1 hugo hugo 8277 Oct  3 18:06 molmetal/tests/test_script_import_health.py
$ -rw-r--r-- 1 hugo hugo 23003 Oct  3 20:33 molmetal/tests/test_joint_3stage_mcts.py
-rw-r--r-- 1 hugo hugo 36189 Oct  3 20:33 molmetal/tests/test_score_any_gpu_routing.py
```

```
$ grep -n '"all": 103' molmetal/scripts/pre_flight_check.py
4682:    "all": 103,  # Wf-A16 (2026-10-03): 105 -> 103, MEASURED via
4724:    "launch": 105,
$ grep -c '_DECORATOR_CLOBBER' molmetal/tests/test_script_import_health.py
4
$ grep -rl 'Wf-A16: feature not implemented' molmetal/tests/*.py | wc -l
17
```

## §2 The register, and a correction about it

`TODO/completed/wfa16_unimplemented_feature_register.md` now exists
(17 entries, 8 distinct modules).  It was
built by parsing each test module's **own skip message** — not from a wave brief
and not from memory — so every symbol, module path and filename in it is
MEASURED.

**Correction recorded rather than smoothed over:** an earlier turn of this
session listed this file as missing *and* stated it was "referenced in skip
messages".  That was right about the reference and wrong about nothing else —
but the *reason* it was missing had not been established.  It is now: the wave
shipped the SKIP conversion (deliverable 4) and never wrote the register those
SKIPs point at (deliverable 5).  The code and the doc disagreed, and the code
was the half that shipped.

## §3 Why the gate split matters beyond this wave

Moving the two gates was the right call for a different reason than "they were
slow": both FAILed on conditions **no legitimate action could repair** — the
anchor gate on wall-clock alone (nothing refreshes anchors on a schedule) and
the runtime gate on a grid/budget the caller never declared.  A gate that cannot
be satisfied is not a gate; it is a tax that teaches people to bypass.

The same defect was found and fixed one wave later in
`df8155d45b` (the two gates' PASS/FAIL/SKIP semantics), which is the natural
continuation of this work.

## §4 Honest framing

* **MEASURED**: all counts and paths above were produced by the `grep`/`ls`
  commands shown, on the committed tree, 2026-10-03. No number here is projected.
* **NOT claimed**: that the 17 registered features are important, or that any
  should be implemented. The register records absence; it is not a plan, and no
  decision has been made about it.
* **Still open**: the 17 tests still SKIP. A SKIP is not a pass, and the suite
  still cannot exercise those wire-ins.
* One entry (`_make_compiled_encoder_forward`) appears in two test modules; the
  register lists 17 entries across 8 modules, not 17 distinct features.

## §5 Cross-references

* `TODO/completed/wfa16_unimplemented_feature_register.md` — the register (NEW).
* `TODO/completed/385a1_wf380_cli_flag_audit.md` — the earlier audit that found
  17 NEW CLI flags declared but never wired; the same class as §1 item 4.
* AGENTS.md §5.5 (filesystem is the source of truth), §5.7 (trainer flag
  wire-in smoke-required rule).
* `df8155d45b` — the follow-on commit that repaired the two gates' semantics.
