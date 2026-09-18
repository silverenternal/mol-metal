# QuickVina 2 binary identity and discovery — 2026-09-13

QuickVina 2 is available locally. The executable at
`molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02` is byte-identical
to the official QVina repository's `bin/qvina02`. Missing a command named
`quickvina2` on PATH was an incorrect environment-blocker diagnosis.

## Identity evidence

The official repository [README](https://github.com/QVina/qvina/blob/f4bb3b1073a0d50bb2f1fdd14d38594f937602ee/README.md)
identifies QuickVina 2 and QuickVina-W as separate tools. Its `bin/qvina02`
is QuickVina 2; `bin/qvina2.1` is another version, and `bin/qvina-w` is the
distinct blind-docking engine. This verification establishes the bundled
QuickVina 2 binary; it does not relabel it as version 2.1 or QuickVina-W.

Official revision checked: `f4bb3b1073a0d50bb2f1fdd14d38594f937602ee`.

| Evidence | Result |
|---|---|
| Official `bin/qvina02` Git blob SHA-1 | `85281985807632dc6d2e0a8564d2a3c027166f49` |
| Local Git blob SHA-1 (including Git blob header) | `85281985807632dc6d2e0a8564d2a3c027166f49` |
| SHA-256, both local and downloaded official binary | `f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0` |
| File size | 3,317,424 bytes |
| `file` | statically linked x86-64 ELF |
| `--version` | `AutoDock Vina 1.1.2 (May 11, 2011)` — inherited version string |
| `--help` | exits 0; receptor/ligand/box/exhaustiveness/CPU options present |
| Actual docking stdout | cites “Fast, Accurate, and Reliable Molecular Docking with QuickVina 2” |

The identical second copy in `molmetal/references/MLM-Scaling/utils/docking/qvina02`
has the same SHA-256 but lacks executable permission; discovery uses the
executable SoftMol copy. SoftMol's README explicitly documents
`chmod +x gated_mcts/utils/docking/qvina02`, and its `docking_utils.py` invokes
that file for PARP1 and other targets. Its source notes attribution to
`https://github.com/SeulLee05/MOOD/blob/main/scorer/docking.py`.

Verification commands:

```bash
molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02 --version
molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02 --help
sha256sum molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02
curl -L --fail --max-time 60 -sS https://raw.githubusercontent.com/QVina/qvina/f4bb3b1073a0d50bb2f1fdd14d38594f937602ee/bin/qvina02 -o /tmp/qvina02-official-verify
cmp /tmp/qvina02-official-verify molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02
```

The first download timed out with a partial file; resuming with `curl -C -`
completed it. Only the completed file was used for the matching SHA-256 and
byte comparison. No installation or rebuild was needed.

## Adapter change and validation

`vina_adapter.py` now recognizes upstream names `qvina02` and `qvina2.1`
and allows both engine selectors (`qvina`, `quickvina2`) to discover the
verified bundled QuickVina 2 executable. Resolution order is:

1. `MOLMETAL_QVINA_BIN` or `MOLMETAL_QUICKVINA2_BIN`, independently per engine.
2. A recognized executable on PATH.
3. The bundled SoftMol `qvina02`.

Each candidate must be executable and complete `--help` successfully;
broken libraries, permission failures and timeouts do not count as available.
Explicit overrides and PATH installs take precedence over the bundled copy.
The two selectors may identify the same physical engine. Comparing them
against each other is not an independent engine-parity experiment.

```bash
HIP_VISIBLE_DEVICES=0 uv run pytest -q molmetal/molmetal_lam/tests/test_qvina_swap.py
```

Result: **15 passed**, no skips. Coverage includes missing binaries on any
host, both bundled aliases, upstream PATH names, override precedence,
independent overrides, non-executable files, broken executable fallback,
and the default Vina path.

An additional real subprocess smoke used the adapter-resolved binary for
each engine name, an RDKit/Meeko-prepared `CCO` ligand, and the vendored
`parp1.pdbqt` receptor. The native SoftMol PARP1 box was used: center
`(26.413, 11.282, 27.238)`, size `(18.521, 17.479, 19.995)` Å, with
`--exhaustiveness 1 --num_modes 1 --cpu 1 --seed 42`.

| Selector | Return code | Output pose |
|---|---:|---|
| `qvina` | 0 | `REMARK VINA RESULT: -2.6 0.000 0.000` |
| `quickvina2` | 0 | `REMARK VINA RESULT: -2.6 0.000 0.000` |

This confirms the discovered executable actually docks and writes a pose;
it is a smoke test, not an affinity benchmark or Vina/QuickVina parity study.
The adapter's existing energy parser separately needs to handle the
`REMARK VINA RESULT` prefix. That parsing issue was reported to the parent
agent and left outside this discovery-only change.
