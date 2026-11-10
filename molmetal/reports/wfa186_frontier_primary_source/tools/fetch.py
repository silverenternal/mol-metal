#!/usr/bin/env python3
"""Wf-A186 live re-fetch of the three primary-source e-prints.

CPU-ONLY. Uses the host proxy (http://127.0.0.1:7890) and follows the
/e-print/<id> -> /src/<id> redirect itself.  Writes:
    cache/<id>.tar.gz        the raw e-print bytes
    cache/<id>.json          the fetch receipt (url, final_url, status, bytes,
                             sha256, file_names, per-file byte counts, sha256)
    cache/<id>/              the extracted tarball

Run with:  CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -I tools/fetch.py
"""

import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CACHE = os.path.join(BASE, "cache")

PROXY = "http://127.0.0.1:7890"
IDS = ["2305.01140v1", "2603.27113v1", "2609.08333v4"]


def _opener():
    op = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    )
    op.addheaders = [
        ("User-Agent", "Mozilla/5.0 (wfa186-frontier-primary-source-verifier)"),
        ("Accept", "*/*"),
    ]
    return op


def fetch_one(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    op = _opener()
    started = time.time()
    resp = op.open(url, timeout=300)
    status = getattr(resp, "status", None)
    final_url = resp.geturl()
    raw = resp.read()
    elapsed = time.time() - started
    sha = hashlib.sha256(raw).hexdigest()

    tar_path = os.path.join(CACHE, f"{arxiv_id}.tar.gz")
    with open(tar_path, "wb") as fh:
        fh.write(raw)

    file_names = []
    file_bytes = {}
    file_sha = {}
    extract_dir = os.path.join(CACHE, arxiv_id)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            file_names.append(name)
            data = tf.extractfile(member).read()
            file_bytes[name] = len(data)
            file_sha[name] = hashlib.sha256(data).hexdigest()
            out = os.path.join(extract_dir, name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(data)

    receipt = {
        "arxiv": arxiv_id,
        "url": url,
        "final_url": final_url,
        "status": status,
        "bytes": len(raw),
        "sha256": sha,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": round(elapsed, 2),
        "tarball_path": os.path.relpath(tar_path, BASE),
        "extract_dir": os.path.relpath(extract_dir, BASE),
        "file_names": sorted(file_names),
        "file_bytes": file_bytes,
        "file_sha256": file_sha,
        "n_files": len(file_names),
    }
    with open(os.path.join(CACHE, f"{arxiv_id}.json"), "w") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return receipt


def main():
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for aid in IDS:
        try:
            out.append(fetch_one(aid))
            print(f"OK   {aid}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {aid}: {exc}", file=sys.stderr)
            out.append({"arxiv": aid, "error": repr(exc)})
    with open(os.path.join(CACHE, "_fetch_index.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    for r in out:
        print(json.dumps({k: r.get(k) for k in ("arxiv", "status", "bytes", "sha256",
                                                "final_url", "n_files", "retrieved_utc",
                                                "error")}))


if __name__ == "__main__":
    main()
