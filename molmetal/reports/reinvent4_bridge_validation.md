# REINVENT4 JSON-lines bridge validation

The upstream REINVENT4 `reinvent` command is a TOML-configured RL entry
point. It does not implement the line-oriented `{"op":"score",...}` protocol
used by the MolMetal subprocess adapter. Launching the upstream command as a
long-lived JSON-lines worker therefore cannot produce scores reliably.

`molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py` is an isolated
protocol bridge for environment validation. It imports RDKit only, reads one
JSON object per stdin line, and writes one JSON response per stdout line. It
does not import REINVENT4 or PyTorch and can be launched from a separate
Python environment:

```bash
uv run python molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py <<'EOF'
{"op":"ping"}
{"op":"score","smiles":["CCO"]}
EOF
```

The returned values are deterministic RDKit proxies for QED, SA, binding,
and novelty. They validate process isolation and wire compatibility only;
they must not be reported as REINVENT4 learned-plugin scores. The real
REINVENT4 binary remains environment-blocked and should be configured via
`REINVENT4_BIN` after installation in a separate environment.
