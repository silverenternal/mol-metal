#!/usr/bin/env python3
"""Retrieve the official v3 USPTO model via verified S3 byte ranges.

The bucket/object is the actual Location returned by official Figshare file
23086454, listed by MolecularAI/aizynthfinder tag v3.0.0. This is the same
object through AWS's HTTPS virtual-hosted endpoint, not a third-party mirror.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import requests

URL = "https://pfigshare-u-files.s3.eu-west-1.amazonaws.com/23086454/full_uspto_03_05_19_rollout_policy.hdf5"
URLS = [URL, URL + "?download=1",
        "https://s3.eu-west-1.amazonaws.com/pfigshare-u-files/23086454/full_uspto_03_05_19_rollout_policy.hdf5"]
SIZE = 300068076
CHUNK = 8 * 1024 * 1024
DEST = Path("/mnt/storage/data/molmetal/aizynth_public/legacy_v3/uspto_model.hdf5")
EXPECTED_SHA256 = "49c3e54b280106fdc9412939b4feaa940ef6ea426461565679f69a1c826450b3"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file():
        with DEST.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == EXPECTED_SHA256:
                print("Official legacy model already present with verified SHA256")
                return
    temp = DEST.with_suffix(".part")
    fd = os.open(temp, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, SIZE)
    chunk_dir = DEST.parent / "model_chunks"
    chunk_dir.mkdir(exist_ok=True)

    def chunk(start):
        end = min(start + CHUNK, SIZE) - 1
        journal = chunk_dir / f"{start}.json"
        if journal.is_file():
            row = json.loads(journal.read_text())
            data = os.pread(fd, end - start + 1, start)
            if row.get("sha256") == hashlib.sha256(data).hexdigest():
                return row
        for attempt in range(9):
            try:
                response = requests.get(URLS[attempt % len(URLS)], headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 25))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Server did not return the exact requested byte range")
                data = response.content
                if len(data) != end - start + 1:
                    raise ValueError("Range payload size mismatch")
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], start + written)
                row = {"start": start, "end": end, "etag": response.headers.get("ETag"),
                       "sha256": hashlib.sha256(data).hexdigest()}
                journal.write_text(json.dumps(row) + "\n")
                print(f"downloaded {start}-{end}", flush=True)
                return row
            except Exception:
                if attempt == 8:
                    raise

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(chunk, range(0, SIZE, CHUNK)))
        if len({row["etag"] for row in rows}) != 1 or rows[0]["etag"] is None:
            raise ValueError("Asset ETag changed between range requests")
        os.fsync(fd)
    finally:
        os.close(fd)
    with temp.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Downloaded model differs from the audited official source SHA256")
    etag = rows[0]["etag"].strip('"')
    md5_verified = False
    if len(etag) == 32 and all(c in "0123456789abcdef" for c in etag.lower()):
        with temp.open("rb") as handle:
            if hashlib.file_digest(handle, "md5").hexdigest() != etag:
                raise ValueError("Downloaded asset does not match the S3 object MD5 ETag")
        md5_verified = True
    temp.replace(DEST)
    manifest = {"status": "downloaded", "url": URL,
                "official_figshare_url": "https://ndownloader.figshare.com/files/23086454",
                "size": SIZE, "sha256": digest, "s3_md5_etag_verified": md5_verified, "chunks": rows}
    DEST.with_name("model_range_download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "chunks"}))


if __name__ == "__main__":
    main()
