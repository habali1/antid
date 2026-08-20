#!/usr/bin/env python3
"""clean.py — filter a raw image tree into a clean training set.

Removes, in this order:
  1. Corrupt / unreadable images (PIL verify + load).
  2. Exact duplicates (MD5 of decoded-and-re-encoded bytes is overkill;
     we hash the raw file bytes, which catches re-downloads and copies).
  3. Images whose SHORTER dimension is below --min-dim (default 200px).
And flags (does not remove) images that carry no EXIF, for optional audit.

By default it COPIES survivors into --output, leaving the raw tree intact.
Pass --in-place to instead delete rejects from --input directly.

Examples
--------
  python clean.py --input ./data/raw --output ./data/clean
  python clean.py --input ./data/raw --in-place --min-dim 256
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from common import iter_image_files, md5_of_file  # noqa: E402


def inspect(path: Path, min_dim: int):
    """Return (status, width, height, has_exif).

    status in {'ok', 'corrupt', 'too_small'}.
    """
    from PIL import Image  # lazy
    try:
        with Image.open(path) as im:
            im.verify()  # cheap integrity check; invalidates the handle
        with Image.open(path) as im:
            im.load()    # full decode
            width, height = im.size
            has_exif = bool(getattr(im, "_getexif", lambda: None)())
    except Exception:  # noqa: BLE001
        return "corrupt", None, None, False
    if min(width, height) < min_dim:
        return "too_small", width, height, has_exif
    return "ok", width, height, has_exif


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    if not args.in_place and output_dir is None:
        print("error: provide --output DIR or --in-place", file=sys.stderr)
        return 2
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, Path] = {}
    report_rows: list[dict] = []
    counts = {"ok": 0, "corrupt": 0, "too_small": 0, "duplicate": 0, "no_exif": 0}

    for path in iter_image_files(input_dir):
        rel = path.relative_to(input_dir)
        status, w, h, has_exif = inspect(path, args.min_dim)

        if status == "ok":
            digest = md5_of_file(path)
            if digest in seen_hashes:
                status = "duplicate"
            else:
                seen_hashes[digest] = path

        kept = status == "ok"
        if status == "ok" and not has_exif:
            counts["no_exif"] += 1
        counts[status] = counts.get(status, 0) + 1

        report_rows.append({
            "path": str(rel), "status": status,
            "width": w or "", "height": h or "",
            "has_exif": int(bool(has_exif)), "kept": int(kept),
        })

        if kept and output_dir is not None and not args.dry_run:
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        elif not kept and args.in_place and not args.dry_run:
            path.unlink(missing_ok=True)

    # Report.
    report_path = Path(args.report) if args.report else (
        (output_dir or input_dir) / "clean_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report_rows[0].keys())
                                if report_rows else
                                ["path", "status", "width", "height", "has_exif", "kept"])
        writer.writeheader()
        writer.writerows(report_rows)

    total = len(report_rows)
    print(f"Scanned {total} images:")
    print(f"  kept (ok)     : {counts['ok']}")
    print(f"  corrupt       : {counts['corrupt']}")
    print(f"  too_small     : {counts['too_small']} (< {args.min_dim}px shorter side)")
    print(f"  duplicate     : {counts['duplicate']}")
    print(f"  flagged no-EXIF (kept): {counts['no_exif']}")
    print(f"Report: {report_path}")
    if output_dir is not None and not args.dry_run:
        print(f"Clean set: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Raw image directory (recursively scanned)")
    p.add_argument("--output", default=None, help="Destination for survivors (copy mode)")
    p.add_argument("--in-place", action="store_true",
                   help="Delete rejects from --input instead of copying survivors")
    p.add_argument("--min-dim", type=int, default=200,
                   help="Minimum shorter-side pixels (default 200)")
    p.add_argument("--report", default=None, help="CSV report path")
    p.add_argument("--dry-run", action="store_true",
                   help="Report only; copy/delete nothing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
