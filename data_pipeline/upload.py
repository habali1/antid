#!/usr/bin/env python3
"""upload.py — push a cleaned image tree to a GCS or S3 bucket.

Layout in the bucket mirrors the local one:
    {bucket}/{species_slug}/{image_id}.jpg

The backend is inferred from the bucket URI scheme (gs:// or s3://) or the
STORAGE_BUCKET env var, and can be forced with --backend. If DATABASE_URL is
set, image rows' storage_path can be (re)written to the cloud URI.

Examples
--------
  STORAGE_BUCKET=gs://antid-training python upload.py --src ./data/clean
  python upload.py --src ./data/clean --bucket s3://antid-training --backend s3
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from common import StorageClient, iter_image_files  # noqa: E402


def run(args: argparse.Namespace) -> int:
    bucket = args.bucket or os.environ.get("STORAGE_BUCKET")
    if not bucket:
        print("error: provide --bucket or set STORAGE_BUCKET", file=sys.stderr)
        return 2

    src = Path(args.src)
    files = list(iter_image_files(src))
    if not files:
        print(f"No images found under {src}", file=sys.stderr)
        return 1

    client = StorageClient(bucket, args.backend)
    backend = client.backend  # 'gcs' or 's3' (resolved from scheme/env)

    uploaded = 0
    for path in files:
        key = str(path.relative_to(src)).replace(os.sep, "/")
        if args.prefix:
            key = f"{args.prefix.strip('/')}/{key}"
        if args.dry_run:
            print(f"  would upload {path}  ->  {client.uri(key)}")
            uploaded += 1
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        uri = client.upload_bytes(key, path.read_bytes(), content_type)
        if args.verbose:
            print(f"  {path}  ->  {uri}")
        uploaded += 1

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {uploaded} images to {bucket}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Local cleaned image directory")
    p.add_argument("--bucket", default=None, help="gs:// or s3:// bucket (or STORAGE_BUCKET env)")
    p.add_argument("--backend", choices=["gcs", "s3", "auto"], default="auto")
    p.add_argument("--prefix", default=None, help="Optional key prefix inside the bucket")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
