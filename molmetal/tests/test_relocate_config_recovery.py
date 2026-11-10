"""Recovery config relocation preserves non-path values and never overwrites."""
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.relocate_config import OLD_ROOT, relocate


def test_relocate_only_whole_root_in_nested_values():
    value = {'train': {'lr': 0.001, 'batch_size': 32}, 'paths': [
        OLD_ROOT + '/data/x.csv', 'python ' + OLD_ROOT + '/run.py',
        OLD_ROOT + '-other/file', '/prefix' + OLD_ROOT + '/file',
    ]}
    actual, count = relocate(value, OLD_ROOT, '/new/repo')
    assert count == 2
    assert actual['train'] == value['train']
    assert actual['paths'] == ['/new/repo/data/x.csv', 'python /new/repo/run.py',
                               OLD_ROOT + '-other/file', '/prefix' + OLD_ROOT + '/file']


def test_cli_refuses_overwrite_and_writes_valid_yaml(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / 'source.yaml'
    dest = tmp_path / 'dest.yaml'
    source.write_text('train:\n  lr: 0.001\ndata:\n  csv: ' + OLD_ROOT + '/x.csv\n')
    cmd = [sys.executable, str(repo / 'scripts/relocate_config.py'), str(source),
           str(dest), '--new-root', str(tmp_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    result = yaml.safe_load(dest.read_text())
    assert result['train']['lr'] == 0.001
    assert result['data']['csv'] == str(tmp_path / 'x.csv')
    before = dest.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert dest.read_bytes() == before
