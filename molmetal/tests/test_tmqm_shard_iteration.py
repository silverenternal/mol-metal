"""Every tmQM shard must be read, not just the first.

2026-10-04 — `iter_tmqm_blocks` ended its per-file loop with a bare ``return``
instead of ``break``.  In a generator, ``return`` exits the WHOLE generator, so
the loop never advanced past the first path.  MEASURED consequence:

    xyz blocks yielded          34,970   (only tmQM_X1.xyz.gz)
    corpus on disk             108,541
    six-metal complexes reachable   6,872   (should be 33,429)

So 73,571 complexes (67.8%) were invisible to every tmQM consumer, and a caller
asking for more than ~6,872 got 6,872 with no warning.  This is the reason no
CFM run could ever train on the large corpus: not the `--n-train` default
alone, but the data path itself could not deliver more than a fifth of the
available data.

The signature is a count, so it is directly testable.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from molmetal.data import tmqm_clouds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMQM_DIR = PROJECT_ROOT / ".data" / "molmetal" / "tmQM"
XYZ_SHARDS = sorted(TMQM_DIR.glob("tmQM_X*.xyz.gz"))

needs_corpus = pytest.mark.skipif(
    len(XYZ_SHARDS) < 2, reason="tmQM XYZ shards not present"
)


def _write_shard(path: Path, codes) -> None:
    """Minimal valid tmQM XYZ shard: count first, CSD_code on the comment."""
    lines = []
    for code in codes:
        lines += ["2", f"CSD_code = {code} | x", "C 0.0 0.0 0.0", "O 1.2 0.0 0.0"]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


def test_reads_every_shard_it_is_given(tmp_path):
    """The synthetic guard: three shards in, three shards' worth of blocks out.

    Deterministic and corpus-free, so it fails loudly wherever it runs.
    """
    paths = []
    for i in (1, 2, 3):
        p = tmp_path / f"tmQM_X{i}.xyz.gz"
        _write_shard(p, [f"CODE{i}A", f"CODE{i}B"])
        paths.append(p)

    got = [code for code, _, _ in tmqm_clouds.iter_tmqm_blocks(paths)]
    assert got == [
        "CODE1A", "CODE1B", "CODE2A", "CODE2B", "CODE3A", "CODE3B",
    ], (
        f"iter_tmqm_blocks returned {got}.  A bare `return` at the end of a "
        "file's read loop exits the generator and silently drops every later "
        "shard; it must be `break`."
    )


@needs_corpus
def test_real_corpus_yields_every_block():
    """MEASURED against disk: the yield must match the shard totals."""
    total = 0
    for shard in XYZ_SHARDS:
        n = sum(1 for _ in tmqm_clouds.iter_tmqm_blocks([shard]))
        assert n > 0, f"{shard.name} yielded nothing"
        total += n
    # Streamed together, the count must be the SUM, not the first shard's.
    together = sum(1 for _ in tmqm_clouds.iter_tmqm_blocks(XYZ_SHARDS))
    assert together == total, (
        f"streaming all {len(XYZ_SHARDS)} shards yielded {together} blocks but "
        f"they hold {total} between them — {total - together} complexes are "
        "being skipped because the iterator stops after the first file"
    )


@needs_corpus
def test_six_metal_pool_is_the_whole_corpus_not_one_shard():
    """The consequence guard: the trainable pool must be ~33k, not ~6.8k."""
    meta = tmqm_clouds.load_tmqm_metadata(TMQM_DIR / "tmqm_parsed.csv")
    metals = {"Pt", "Ru", "Ir", "Rh", "Os", "Re"}
    keep = {
        c for c, row in meta.items()
        if str(row.get("metal", "")).strip() in metals
    }
    assert len(keep) > 30_000, (
        f"only {len(keep)} six-metal complexes in the metadata; expected the "
        "full corpus"
    )
    codes = {c for c, _, _ in tmqm_clouds.iter_tmqm_blocks(XYZ_SHARDS)}
    reachable = len(codes & keep)
    assert reachable == len(keep), (
        f"{len(keep) - reachable} of {len(keep)} six-metal complexes have a "
        "metadata row but no reachable geometry — the shard iterator is "
        "dropping shards again"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
