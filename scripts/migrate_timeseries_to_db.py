#!/usr/bin/env python3
"""
migrate_timeseries_to_db.py — One-time migration to TimeseriesDB.

Pass 1: Scan all .INFO files in Goa_Separated_crops, write timeseries arrays
        to /home/kim/Projects/mir/data/timeseries.db on the NVMe.

Pass 2: Strip timeseries list fields from companion .json files in goa-small
        (and goa-stems if --stems is given), rewriting them as scalar-only JSON.
        padding_mask is always preserved.

Usage:
    python scripts/migrate_timeseries_to_db.py --dry-run
    python scripts/migrate_timeseries_to_db.py
    python scripts/migrate_timeseries_to_db.py --skip-pass1   # JSON strip only
    python scripts/migrate_timeseries_to_db.py --skip-pass2   # DB build only
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.timeseries_db import TimeseriesDB, is_timeseries_field

CROPS_DIR  = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
SMALL_DIR  = Path("/run/media/kim/Lehto/goa-small")
STEMS_DIR  = Path("/run/media/kim/Lehto/goa-stems")


# ---------------------------------------------------------------------------
# Pass 1: INFO → DB
# ---------------------------------------------------------------------------

def build_db_from_info(db: TimeseriesDB, dry: bool) -> None:
    info_files = sorted(CROPS_DIR.rglob("*.INFO"))
    total = len(info_files)
    print(f"Pass 1: scanning {total:,} .INFO files → {db.path}")

    already = db.count()
    if already:
        print(f"  DB already has {already:,} entries — will skip existing keys")

    existing_keys = set(db.all_keys()) if already else set()

    def _iter_new():
        for i, info_path in enumerate(info_files):
            key = info_path.stem
            if key in existing_keys:
                continue
            try:
                with open(info_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            ts = {k: v for k, v in data.items() if is_timeseries_field(k, v)}
            if ts:
                yield key, ts
            if (i + 1) % 5000 == 0:
                pct = 100 * (i + 1) // total
                print(f"  {i+1:,}/{total:,} ({pct}%) scanned", flush=True)

    if dry:
        n = sum(1 for _ in _iter_new())
        print(f"  DRY RUN: would insert {n:,} entries")
    else:
        written = db.bulk_put(_iter_new())
        print(f"  Wrote {written:,} entries  (total now: {db.count():,})")


# ---------------------------------------------------------------------------
# Pass 2: Strip list fields from companion JSONs
# ---------------------------------------------------------------------------

def strip_json_dir(target_dir: Path, dry: bool) -> tuple[int, int, int]:
    """Strip timeseries from all .json companion files under target_dir.

    Returns (files_checked, files_rewritten, bytes_saved).
    """
    json_files = list(target_dir.rglob("*.json"))
    total = len(json_files)
    print(f"\nPass 2: stripping {total:,} .json files in {target_dir.name} ...")

    checked = rewritten = bytes_saved = 0

    for i, json_path in enumerate(json_files):
        checked += 1
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Identify fields to strip (lists other than padding_mask)
        to_strip = [k for k, v in data.items() if is_timeseries_field(k, v)]
        if not to_strip:
            continue

        old_size = json_path.stat().st_size
        stripped = {k: v for k, v in data.items() if k not in to_strip}

        if not dry:
            # Atomic write via temp file
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=json_path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(stripped, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp_path, json_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        new_size = len(json.dumps(stripped, ensure_ascii=False, separators=(",", ":")))
        bytes_saved += old_size - new_size
        rewritten += 1

        if (i + 1) % 10000 == 0:
            pct = 100 * (i + 1) // total
            print(f"  {i+1:,}/{total:,} ({pct}%)  rewritten: {rewritten:,}  "
                  f"saved: {bytes_saved/1e9:.2f} GB", flush=True)

    return checked, rewritten, bytes_saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-pass1", action="store_true",
                        help="Skip DB build (Pass 1)")
    parser.add_argument("--skip-pass2", action="store_true",
                        help="Skip JSON stripping (Pass 2)")
    parser.add_argument("--stems",      action="store_true",
                        help="Also strip goa-stems companion JSONs (Pass 2)")
    parser.add_argument("--db-path",    default=None,
                        help="Override DB path")
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        print("*** DRY RUN — no changes will be made ***\n")

    db_path = Path(args.db_path) if args.db_path else TimeseriesDB.__init__.__defaults__[0]
    db = TimeseriesDB.open(db_path)
    print(f"TimeseriesDB: {db.path}")

    # ------------------------------------------------------------------
    if not args.skip_pass1:
        build_db_from_info(db, dry)

    # ------------------------------------------------------------------
    total_saved = 0
    if not args.skip_pass2:
        dirs = [SMALL_DIR]
        if args.stems:
            dirs.append(STEMS_DIR)

        for d in dirs:
            checked, rewritten, saved = strip_json_dir(d, dry)
            total_saved += saved
            verb = "Would rewrite" if dry else "Rewrote"
            print(f"  {verb} {rewritten:,} / {checked:,} files  "
                  f"({saved / 1e9:.2f} GB saved)")

    print(f"\n{'DRY RUN ' if dry else ''}Done.")
    print(f"  DB entries : {db.count():,}")
    if not args.skip_pass2:
        print(f"  Space freed: {total_saved / 1e9:.2f} GB")
    db.close()


if __name__ == "__main__":
    main()
