# AiZynth CPU runtime

This is a separate Python 3.12 uv project for AiZynthFinder 4.4.1. Its
RDKit 2023.9.6 / NumPy 1.26.4 requirements conflict with the ROCm project's
newer chemistry stack, so do not add it to the root `pyproject.toml`.

`uv.lock` is the source of exact versions. The matching
`requirements.lock.txt` is generated with:

```bash
uv export --project environments/aizynth --frozen --no-hashes --no-emit-project \
  --format requirements-txt --output-file environments/aizynth/requirements.lock.txt
```

Use the isolated cached runtime:

```bash
bash environments/aizynth/run.sh python -c 'from aizynthfinder.aizynthfinder import AiZynthFinder'
bash environments/aizynth/run.sh python molmetal/scripts/configure_aizynth_public.py
bash environments/aizynth/run.sh python molmetal/scripts/aizynth_learned_smoke.py \
  --config /mnt/storage/data/molmetal/aizynth_public/config.yml \
  --output molmetal/reports/aizynth_real_backend_20260913/gate_smoke.json
```

The wrapper uses `uv run --isolated --no-project --offline`, with every
dependency version supplied by the exported lock. It does not place the
root `.venv` on `sys.path`; this isolation was explicitly checked. Offline
mode applies to uv package resolution only; the downloader itself can use
the network. `runtime_versions.json` records the validated installed versions.
Cached wheel archives have not been independently checked against publisher
hashes by this wrapper. A normal hash-checked clean installation is
`uv sync --project environments/aizynth --frozen`; the first attempt was
interrupted after slow redundant downloads of packages already cached.

The download helper obtains URLs from the installed package's official
`FILES_TO_DOWNLOAD` mapping. It writes atomic assets, a manifest with local
SHA256, and `config.pending.yml` when any asset remains unavailable. It
never claims that an incomplete configuration is ready. Actual model/stock
compatibility requires the separate smoke test.

As of 2026-09-13, genuine learned retrosynthesis works with explicitly
labelled official **legacy v3 USPTO** weights/templates and ZINC stock.
Current upstream Zenodo assets still time out; the historical official
Figshare policy was downloaded, whole-file MD5-verified, and converted with
preserved weights. Use these audited project configurations:

- `molmetal/configs/aizynth_legacy_v3_cpu.yml`: CPU ONNX.
- `molmetal/configs/aizynth_legacy_v3_rocm.yml`: the same policy through a
  persistent ROCm worker in the root environment; official chemistry/MCTS
  remain in this isolated CPU environment.

The root API is `build_synthesis_gate("aizynthfinder_isolated", config_path)`.
It validates an actual backend, batches candidate checks, and rejects fallback
reports. Process deadlines reclaim the launcher and all GPU descendants.

See `molmetal/reports/aizynth_real_backend_20260913/report.md` for full
provenance, CPU/GPU route evidence, 28-input probability/top-k comparisons,
and command lines. These are functional smoke tests; there is no claim of
current-policy parity, identical MCTS trajectories or end-to-end speedup.
