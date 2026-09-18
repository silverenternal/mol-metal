"""Build an auditable CrossDocked train-only index, then evaluate report candidates.

Run as `uv run --no-sync python -m molmetal.scripts.training_set_novelty ...`.
Large per-SDF inventory/SQLite/fingerprint assets belong outside the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time

import numpy as np
from rdkit import Chem, RDLogger, rdBase

from molmetal.validation.training_novelty import (
    canonical_graph, morgan_fingerprints, nearest_tanimoto, scaffold_smiles,
)


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_ligand(task):
    root, partition, receptor, ligand = task
    RDLogger.DisableLog('rdApp.*')
    record = dict(partition=partition, receptor=receptor, ligand=ligand)
    try:
        raw = (Path(root) / ligand).read_bytes()
        record['sha256'] = hashlib.sha256(raw).hexdigest()
        mols = list(Chem.ForwardSDMolSupplier(io.BytesIO(raw), sanitize=True, removeHs=False))
        record['n_sdf_records'] = len(mols)
        if len(mols) != 1:
            raise ValueError(f'expected exactly one SDF record; got {len(mols)}')
        record['canonical_smiles'] = canonical_graph(mols[0])
        record['scaffold'] = scaffold_smiles(record['canonical_smiles'])
        record['status'] = 'valid'
    except FileNotFoundError:
        record['status'] = 'missing'
    except Exception as exc:
        record.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
    return record


def build(args):
    import torch
    started = time.monotonic()
    out = Path(args.index)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'index.sqlite').exists():
        raise FileExistsError('choose a fresh index directory; refusing to mix index versions')
    split = torch.load(args.split, weights_only=False, map_location='cpu')
    train_paths = {lig for _, lig in split['train']}
    test_paths = {lig for _, lig in split['test']}
    if train_paths & test_paths:
        raise ValueError('train/test ligand paths overlap')
    conn = sqlite3.connect(out / 'index.sqlite')
    conn.executescript('''
      CREATE TABLE structures(id INTEGER PRIMARY KEY, smiles TEXT UNIQUE, scaffold TEXT);
      CREATE TABLE sources(partition TEXT, receptor TEXT, ligand TEXT, sha256 TEXT,
                           status TEXT, structure_id INTEGER, error TEXT);
      CREATE INDEX source_structure ON sources(structure_id);
    ''')
    counts = {part: Counter(valid=0, missing=0, invalid=0) for part in ('train', 'test')}
    train_ids = {}
    test_graphs = []
    inventory = out / 'source_inventory.jsonl'
    tasks = ((args.root, part, rec, lig) for part in ('train', 'test') for rec, lig in split[part])
    with inventory.open('w') as stream, multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for i, record in enumerate(pool.imap(read_ligand, tasks, chunksize=64), 1):
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            part = record['partition']
            counts[part][record['status']] += 1
            structure_id = None
            if record['status'] == 'valid':
                smiles = record['canonical_smiles']
                if part == 'train':
                    if smiles not in train_ids:
                        structure_id = len(train_ids)
                        train_ids[smiles] = structure_id
                        conn.execute('INSERT INTO structures VALUES(?,?,?)',
                                     (structure_id, smiles, record['scaffold']))
                    structure_id = train_ids[smiles]
                else:
                    test_graphs.append(smiles)
                    # Reference an existing train structure only; never add test chemistry.
                    structure_id = train_ids.get(smiles)
            conn.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',
                         (part, record['receptor'], record['ligand'], record.get('sha256'),
                          record['status'], structure_id, record.get('error')))
            if i % 10000 == 0:
                conn.commit()
                print(json.dumps(dict(processed=i, unique_train=len(train_ids),
                                      elapsed_s=round(time.monotonic()-started, 1))), flush=True)
    conn.commit()
    structures = list(conn.execute('SELECT id,smiles,scaffold FROM structures ORDER BY id'))
    fps = morgan_fingerprints([row[1] for row in structures])
    np.save(out / 'morgan_r2_2048.npy', fps, allow_pickle=False)
    train_scaffolds = {row[2] for row in structures}
    test_unique = set(test_graphs)
    manifest = dict(
        scope='CrossDocked split_by_name.pt train partition only',
        source_split=str(Path(args.split).resolve()), split_sha256=sha256(args.split),
        ligand_root=str(Path(args.root).resolve()), rdkit_version=rdBase.rdkitVersion,
        canonicalization='RDKit sanitized SDF; exactly one record required; remove atom maps and ordinary explicit H; preserve stereochemistry, isotopes, charge and fragments; no salt stripping, tautomer normalization or neutralization',
        fingerprint='Morgan radius=2 bits=2048 includeChirality=False (CPU RDKit)',
        scaffold='Bemis-Murcko atom/bond scaffold, includeChirality=False; acyclic empty scaffold explicitly labelled',
        split_counts={key: len(split[key]) for key in ('train', 'test')},
        file_counts={key: dict(value) for key, value in counts.items()},
        train_unique_structures=len(train_ids),
        train_duplicate_valid_instances=counts['train']['valid']-len(train_ids),
        train_unique_scaffolds=len(train_scaffolds),
        train_acyclic_unique_structures=sum(not row[2] for row in structures),
        train_test_ligand_path_overlap=0,
        test_unique_structures=len(test_unique),
        test_instances_exactly_in_train=sum(s in train_ids for s in test_graphs),
        test_unique_structures_exactly_in_train=len(test_unique & train_ids.keys()),
        build_seconds=time.monotonic()-started,
    )
    conn.close()
    manifest['artifacts'] = {name: dict(path=str((out/name).resolve()), sha256=sha256(out/name))
                             for name in ('source_inventory.jsonl', 'index.sqlite', 'morgan_r2_2048.npy')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def evaluate(args):
    import torch
    started = time.monotonic()
    index = Path(args.index)
    manifest = json.loads((index / 'manifest.json').read_text())
    for name, artifact in manifest['artifacts'].items():
        if sha256(index / name) != artifact['sha256']:
            raise ValueError(f'index artifact hash mismatch: {name}')
    conn = sqlite3.connect(f'file:{index.resolve()}/index.sqlite?mode=ro', uri=True)
    structures = list(conn.execute('SELECT id,smiles,scaffold FROM structures ORDER BY id'))
    exact_ids = {row[1]: row[0] for row in structures}
    scaffold_ids = {}
    for sid, _, scaffold in structures:
        scaffold_ids.setdefault(scaffold, sid)
    source = json.loads(Path(args.source).read_text())
    candidates = []
    for job_index, job in enumerate(source['per_pocket']):
        reference_in_train = None
        if job.get('ligand_path'):
            try:
                relative = str(Path(job['ligand_path']).relative_to(manifest['ligand_root']))
                reference = conn.execute(
                    "SELECT status,structure_id FROM sources WHERE partition='test' AND ligand=?", (relative,)
                ).fetchone()
                if reference and reference[0] == 'valid':
                    reference_in_train = reference[1] is not None
            except ValueError:
                pass
        rows = job.get('all_candidates', job.get('candidates', []))
        for rank, row in enumerate(rows):
            if not row.get('is_generated', not row.get('is_seed', False)):
                continue
            record = dict(job_index=job_index, pocket_id=job.get('pocket_id'), seed=job.get('seed'),
                          candidate_index=rank, smiles=row.get('smiles'),
                          reference_ligand_exact_graph_in_train=reference_in_train)
            try:
                record['canonical_smiles'] = canonical_graph(Chem.MolFromSmiles(row['smiles']))
                record['status'] = 'valid'
            except Exception as exc:
                record.update(status='invalid', error=str(exc))
            candidates.append(record)
    unique = sorted({row['canonical_smiles'] for row in candidates if row['status'] == 'valid'})
    fps = np.load(index / 'morgan_r2_2048.npy', mmap_mode='r')
    scores, neighbors = nearest_tanimoto(morgan_fingerprints(unique), fps,
                                        device=args.device, chunk_size=args.chunk_size)
    metrics = {}
    for smiles, score, neighbor in zip(unique, scores, neighbors):
        scaffold = scaffold_smiles(smiles)
        nearest = structures[int(neighbor)] if neighbor >= 0 else None
        provenance = list(conn.execute(
            "SELECT ligand,sha256 FROM sources WHERE partition='train' AND structure_id=? ORDER BY ligand",
            (int(neighbor),))) if nearest else []
        metrics[smiles] = dict(
            canonical_smiles=smiles, scaffold=scaffold, acyclic=not bool(scaffold),
            exact_graph_in_train=smiles in exact_ids,
            scaffold_in_train=scaffold in scaffold_ids,
            novel_nonempty_scaffold=bool(scaffold) and scaffold not in scaffold_ids,
            nearest_tanimoto=float(score) if nearest else None,
            nearest_train_structure_id=int(neighbor) if nearest else None,
            nearest_train_smiles=nearest[1] if nearest else None,
            nearest_source_count=len(provenance),
            nearest_sources=[dict(ligand=ligand, sha256=digest) for ligand, digest in provenance],
        )
    for candidate in candidates:
        if candidate['status'] == 'valid':
            candidate.update(metrics[candidate['canonical_smiles']])
    valid = [r for r in candidates if r['status'] == 'valid']
    def aggregate(rows):
        return dict(n=len(rows), n_exact_graph_overlap=sum(r['exact_graph_in_train'] for r in rows),
                    n_scaffold_overlap=sum(r['scaffold_in_train'] for r in rows),
                    n_novel_nonempty_scaffold=sum(r['novel_nonempty_scaffold'] for r in rows),
                    mean_nearest_tanimoto=float(np.mean([r['nearest_tanimoto'] for r in rows])) if rows and len(structures) else None)
    device = torch.device(args.device)
    report = dict(
        scope='Novelty against indexed CrossDocked training chemistry; not all pretrained model training data or chemical universe',
        source_json=str(Path(args.source).resolve()), source_sha256=sha256(args.source),
        candidate_scope='all_candidates when present, otherwise saved candidates; generated only; includes undocked instances',
        index_manifest=manifest, index_manifest_sha256=sha256(index / 'manifest.json'),
        n_input_candidates=len(candidates), n_invalid_candidates=len(candidates)-len(valid),
        unique_structures=aggregate(list(metrics.values())), instances=aggregate(valid),
        reference_overlap_subgroups={label: aggregate([
            row for row in valid if row['reference_ligand_exact_graph_in_train'] is flag
        ]) for label, flag in [('shared_reference_graph', True), ('nonshared_reference_graph', False),
                              ('unknown_reference_graph', None)]},
        device=str(device), torch_version=torch.__version__, hip_version=torch.version.hip,
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        gpu_arch=getattr(torch.cuda.get_device_properties(device), 'gcnArchName', None) if device.type == 'cuda' else None,
        similarity_backend='torch chunked binary fingerprint intersection/union; deterministic first-index ties',
        chunk_size=args.chunk_size, evaluation_seconds=time.monotonic()-started,
        unique_results=list(metrics.values()), candidates=candidates,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k,v in report.items() if k not in ('candidates', 'index_manifest', 'unique_results')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--split', required=True)
    builder.add_argument('--root', required=True)
    builder.add_argument('--index', required=True)
    builder.add_argument('--workers', type=int, default=4)
    evaluator = sub.add_parser('evaluate')
    evaluator.add_argument('--index', required=True)
    evaluator.add_argument('--source', required=True)
    evaluator.add_argument('--output', required=True)
    evaluator.add_argument('--device', default='cuda:0')
    evaluator.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()
    (build if args.command == 'build' else evaluate)(args)


if __name__ == '__main__':
    main()
