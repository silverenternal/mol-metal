#!/usr/bin/env bash
# Restore the verified server recovery release from an already-downloaded directory.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/release-assets" >&2
  exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
assets=$(cd "$1" && pwd -P)
for cmd in sha256sum tar unzip zstd; do
  command -v "$cmd" >/dev/null || { echo "Missing required tool: $cmd" >&2; exit 2; }
done
(cd "$assets" && sha256sum -c "$repo/data/recovery/assets.sha256")
python_bin="$repo/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin=$(command -v python3) || { echo "Python 3 is required" >&2; exit 2; }
fi
# Refuse to overwrite a changed local input. Each manifest is inside the
# hash-verified asset directory, and every archive member is named there.
"$python_bin" - "$repo" "$assets" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,assets=map(Path,sys.argv[1:])
for manifest in ('external-registry-data.json','training-inputs.json',
                 'training-data-extras.json','source-extras.json','referenced-evidence.json',
                 'referenced-evidence-extra.json','referenced-evidence-patterns.json'):
    for entry in json.loads((assets/manifest).read_text())['files']:
        rel=Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise SystemExit(f'Unsafe archive path in {manifest}: {rel}')
        p=root/rel
        if p.is_symlink() or any((root/parent).is_symlink() for parent in rel.parents if str(parent) != '.'):
            raise SystemExit(f'Refusing archive path through a symlink: {rel}')
        if not p.exists(): continue
        if not p.is_file():
            raise SystemExit(f'Refusing to overwrite non-regular file: {rel}')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=entry['sha256']:
            raise SystemExit(f'Refusing to overwrite changed file: {rel}')
PY
cd "$repo"
for archive in external-registry-data training-inputs training-data-extras source-extras referenced-evidence referenced-evidence-extra referenced-evidence-patterns; do
  tar -xzf "$assets/$archive.tar.gz" -C "$repo"
done
mkdir -p .data/molmetal/bindingdb_full
binding=.data/molmetal/bindingdb_full/BindingDB_All.tsv
binding_sha=66bb5f955bd88d2970636764603f8c26c0e010a58b46276ac27d85656a9aa8fb
if [[ -e "$binding" ]]; then
  printf '%s  %s\n' "$binding_sha" "$binding" | sha256sum -c -
else
  unzip -q "$assets/BindingDB_All_202609_tsv.zip" -d .data/molmetal/bindingdb_full
  printf '%s  %s\n' "$binding_sha" "$binding" | sha256sum -c -
fi
chembl=.data/molmetal/chembl_29/chembl_29_sqlite/chembl_29.db
chembl_sha=3111c31f645e9e74c5ea0ad57317791b4cf044cd12e815cb52ff638ae5c52fb0
if [[ -e "$chembl" ]]; then
  printf '%s  %s\n' "$chembl_sha" "$chembl" | sha256sum -c -
else
  cat "$assets/chembl_29.tar.zst.part-aa" "$assets/chembl_29.tar.zst.part-ab" "$assets/chembl_29.tar.zst.part-ac" | zstd -dc | tar -xf - -C "$repo"
  printf '%s  %s\n' "$chembl_sha" "$chembl" | sha256sum -c -
fi
ensure_link() {
  local target=$1 link=$2
  if [[ -L "$link" ]]; then
    [[ $(readlink "$link") == "$target" ]] || { echo "Unexpected symlink: $link" >&2; exit 1; }
  elif [[ -e "$link" ]]; then
    echo "Refusing to replace existing path: $link" >&2; exit 1
  else
    ln -s "$target" "$link"
  fi
}
ensure_link metal_binding_db_full.csv .data/molmetal/bindingdb_full/MetalCytoToxDB.csv
ensure_link cfm_yuelbond_wasserstein_seed0.pt results/wfa108h_b32_chunk2_vector_channels/ckpt/cfm_yuelbond_wasserstein_seed0_last.pt
echo 'Recovery assets restored and checked.'
