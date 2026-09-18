#!/usr/bin/env python3
"""Download official AiZynth 4.4.1 public assets with atomic, audited writes.

Run with the isolated project: uv run --project environments/aizynth python ...
No package installation or main environment mutation happens in this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import requests
import yaml
from aizynthfinder.tools.download_public_data import FILES_TO_DOWNLOAD


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path("/mnt/storage/data/molmetal/aizynth_public"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    keys = ["policy_model_onnx", "template_file", "stock"]

    def download(key):
        spec = FILES_TO_DOWNLOAD[key]
        dest = args.asset_dir / spec["filename"]
        row = {"key": key, **spec, "path": str(dest.resolve()), "attempts": []}
        temp = dest.with_suffix(dest.suffix + ".part")
        try:
            if not dest.exists():
                variants = [spec["url"]]
                if "zenodo.org/record/" in spec["url"]:
                    variants += [spec["url"].replace("/record/", "/records/"),
                                 spec["url"].replace("/record/", "/records/") + "?download=1"]
                for attempt in range(max(1, args.attempts)):
                    url = variants[attempt % len(variants)]
                    try:
                        with requests.get(url, stream=True, timeout=(10, args.timeout)) as response:
                            response.raise_for_status()
                            row["resolved_url"] = response.url.split("?")[0]
                            ctype = response.headers.get("Content-Type", "")
                            if "text/html" in ctype or response.status_code != 200:
                                raise ValueError(f"Unexpected asset response: {response.status_code} {ctype}")
                            expected = int(response.headers.get("Content-Length", "0"))
                            with temp.open("wb") as handle:
                                for chunk in response.iter_content(1024 * 1024):
                                    handle.write(chunk)
                            if not temp.stat().st_size or (expected and temp.stat().st_size != expected):
                                raise ValueError("Asset response size mismatch")
                            temp.replace(dest)
                        row["attempts"].append({"url": url, "status": "downloaded"})
                        break
                    except Exception as exc:
                        row["attempts"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                        if attempt + 1 >= max(1, args.attempts):
                            raise
            row.update(status="downloaded", size_bytes=dest.stat().st_size, sha256=sha256(dest))
        except Exception as exc:
            row.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            temp.unlink(missing_ok=True)
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(download, keys))
    ready = all(row["status"] == "downloaded" for row in rows)
    paths = {row["key"]: row["path"] for row in rows}
    configuration = {
        "expansion": {"uspto": [paths["policy_model_onnx"], paths["template_file"]]},
        "stock": {"zinc": paths["stock"]},
        "search": {"iteration_limit": 20, "time_limit": 10, "return_first": True},
    }
    config_path = args.asset_dir / ("config.yml" if ready else "config.pending.yml")
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aizynthfinder_version": importlib.metadata.version("aizynthfinder"),
        "source": "aizynthfinder.tools.download_public_data.FILES_TO_DOWNLOAD",
        "publisher_checksum_verified": False,
        "checksum_scope": "local SHA256; compatibility is verified separately by real model load/search",
        "status": "assets_downloaded" if ready else "blocked_asset_download",
        "config_path": str(config_path.resolve()), "assets": rows,
    }
    (args.asset_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
