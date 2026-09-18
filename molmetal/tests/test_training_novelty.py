from types import SimpleNamespace
import json
import sqlite3

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molmetal.validation.training_novelty import (
    canonical_graph, morgan_fingerprints, nearest_tanimoto, scaffold_smiles,
)


def test_canonicalization_preserves_chemistry_and_normalizes_maps_and_h():
    assert canonical_graph(Chem.MolFromSmiles('[CH3:8][OH:2]')) == 'CO'
    assert canonical_graph(Chem.AddHs(Chem.MolFromSmiles('CO'))) == 'CO'
    assert canonical_graph(Chem.MolFromSmiles('C[C@H](O)F')) != canonical_graph(Chem.MolFromSmiles('C[C@@H](O)F'))
    assert canonical_graph(Chem.MolFromSmiles('[13CH3]O')) != 'CO'
    assert '.' in canonical_graph(Chem.MolFromSmiles('C[NH3+].[Cl-]'))
    with pytest.raises(ValueError):
        canonical_graph(Chem.MolFromSmiles('*C'))
    assert scaffold_smiles('CCO') == ''


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_nearest_matches_rdkit_and_preserves_first_tie(device):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    train = ['CCO', 'CCN', 'c1ccccc1', 'CCO', 'CC(=O)O']
    query = ['CCO', 'CCCl', 'c1ccncc1']
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    refs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train]
    expected = np.array([DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), refs) for s in query])
    values, indices = nearest_tanimoto(morgan_fingerprints(query), morgan_fingerprints(train),
                                       device=device, chunk_size=2, query_chunk_size=1)
    np.testing.assert_allclose(values, expected.max(1), atol=1e-7)
    np.testing.assert_array_equal(indices, expected.argmax(1))
    assert indices[0] == 0


def test_empty_zero_invalid_fingerprint_cases():
    empty = np.empty((0, 2048), dtype=np.uint8)
    zero = np.zeros((1, 2048), dtype=np.uint8)
    values, indices = nearest_tanimoto(zero, empty, device='cpu')
    assert np.isnan(values[0]) and indices[0] == -1
    values, indices = nearest_tanimoto(empty, zero, device='cpu')
    assert len(values) == len(indices) == 0
    values, indices = nearest_tanimoto(zero, zero, device='cpu')
    assert values[0] == 1 and indices[0] == 0
    with pytest.raises(ValueError, match='binary'):
        nearest_tanimoto(np.ones((1, 8))*2, np.ones((1, 8)), device='cpu')


def test_index_dedup_missing_invalid_and_test_exclusion(tmp_path):
    import torch
    from molmetal.scripts.training_set_novelty import build, evaluate
    for name, smiles in [('a.sdf', 'CCO'), ('b.sdf', 'OCC'), ('test.sdf', 'c1ccccc1')]:
        writer = Chem.SDWriter(str(tmp_path / name))
        writer.write(Chem.MolFromSmiles(smiles))
        writer.close()
    (tmp_path / 'bad.sdf').write_text('invalid\n$$$$\n')
    split = tmp_path / 'split.pt'
    torch.save(dict(train=[('r', 'a.sdf'), ('r', 'b.sdf'), ('r', 'bad.sdf'), ('r', 'missing.sdf')],
                    test=[('r', 'test.sdf')]), split)
    index = tmp_path / 'index'
    build(SimpleNamespace(index=str(index), split=str(split), root=str(tmp_path), workers=1))
    manifest = json.loads((index / 'manifest.json').read_text())
    assert manifest['file_counts']['train'] == dict(valid=2, invalid=1, missing=1)
    assert manifest['train_unique_structures'] == 1
    assert manifest['train_duplicate_valid_instances'] == 1
    with sqlite3.connect(index / 'index.sqlite') as conn:
        assert conn.execute('SELECT smiles FROM structures').fetchall() == [('CCO',)]
        assert conn.execute("SELECT structure_id FROM sources WHERE partition='test'").fetchone() == (None,)
    source = tmp_path / 'candidates.json'
    source.write_text(json.dumps(dict(per_pocket=[dict(pocket_id='test', seed=4,
        all_candidates=[dict(smiles='CCO', is_generated=True), dict(smiles='bad', is_generated=True),
                        dict(smiles='CCO', is_seed=True), dict(smiles='CCO', is_generated=True)],
        candidates=[])])))
    output = tmp_path / 'novelty.json'
    evaluate(SimpleNamespace(index=str(index), source=str(source), output=str(output), device='cpu', chunk_size=1))
    report = json.loads(output.read_text())
    assert report['n_input_candidates'] == 3
    assert report['n_invalid_candidates'] == 1
    assert report['instances']['n_exact_graph_overlap'] == 2
    assert report['unique_structures']['n'] == 1
    assert report['unique_results'][0]['nearest_source_count'] == 2
