# Wf-427-B1 ultracode docs sync + pre-flight verify

## VERDICT
- **4 SHIPPED closeouts verified on disk per AGENTS.md §5.5**:
  - Wf-427-A1 kcal sign fix (10,838 B mtime 2026-10-01 10:17 CST)
  - Wf-427-A2 lambda module fix (9,945 B mtime 2026-10-01 10:35 CST)
  - Wf-427-A3 B6 wire-in fix (17,266 B mtime 2026-10-01 10:25 CST)
  - Wf-427-A4 B5 schema adapter (10,024 B mtime 2026-10-01 10:14 CST)

- **PRE-FLIGHT v3 fresh-run MEASURED at Wf-427-B1 timestamp 2026-10-01T02:43:49+0000 UTC**:
  - **113 PASS / 0 FAIL / 4 SKIP / 117 TOTAL wall 192.48 s**
  - per /tmp/wf427b1_preflight.json (38,833 bytes)
  - **NEW BASELINE since Wf-411-A3 close** — p2_tests_pass cascade FAIL that blocked Wf-413..Wf-419 GPU launches is FULLY CLOSED via Wf-424-A1 use_dann kwarg fix
  - 4 SKIPs preserved BY-DESIGN:
    1. p3_wf343c4_yuelbond_decoder_env (env gap per Wf-345-A2)
    2. p3_wf343c4_wave_budget_remaining_nonneg (-183.04h overrun per Wf-262 §7.1)
    3+4. data_2_1_6_bindingdb_external_honest_negative (duplicate registration across chemprop_v2_v3 + frontier pipelines per Wf-422-A1)

## §5.5 filesystem-truth verification

```
TODO/completed/427a1_kcal_sign_fix.md        10,838 B  2026-10-01 10:17
TODO/completed/427a2_lambda_module_fix.md    9,945 B  2026-10-01 10:35
TODO/completed/427a3_b6_wirein_fix.md       17,266 B  2026-10-01 10:25
TODO/completed/427a4_b5_schema_adapter.md   10,024 B  2026-10-01 10:14
```

## Forward-looking USER_OK-gated sub-flows

Per AGENTS.md §5.1 + Wf-368-B1 + Wf-370-A3 + Wf-415-B1 + Wf-426-B1 prior syncs: 6+ USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) BLOCKED on user OK; m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL preserved per Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verdicts.

## CHRONOLOGY
Wf-427-B1 ships AFTER Wf-296..427-A4 ultracode waves; §22.123 INDEX entry + §6 AGENTS.md row appended per existing pattern.
