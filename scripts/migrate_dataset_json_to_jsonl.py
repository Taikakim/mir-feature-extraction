#!/usr/bin/env python3
"""Migrate a legacy single-line dataset.json to dataset.jsonl + dataset.meta.json.

The old format (one giant dict, ``_meta`` + per-track entries all on one JSON
line) is unsearchable with grep/jq. DataStore now reads/writes
``dataset.jsonl`` (one record per line, keyed by ``_key``) plus a small
``dataset.meta.json`` sidecar for the generation metadata. See
src/core/data_store.py for the format.

This script converts one directory's dataset.json in place:
  1. Load the old dataset.json (`_meta` + per-track dict).
  2. Write dataset.jsonl + dataset.meta.json via DataStore.flush()
     (same atomic-write path DataStore uses everywhere else).
  3. Verify: reload the new files and diff entry-for-entry against the old.
  4. On success, rename dataset.json -> dataset.json.bak (never deleted
     outright — the caller can remove the .bak once satisfied).

Usage:
    python scripts/migrate_dataset_json_to_jsonl.py /path/to/output/dir [...]
    python scripts/migrate_dataset_json_to_jsonl.py --dry-run /path/to/output/dir
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.data_store import DataStore  # noqa: E402


def migrate(root: Path, dry_run: bool = False) -> bool:
    old_path = root / "dataset.json"
    if not old_path.exists():
        print(f"  SKIP {root}: no dataset.json")
        return False

    jsonl_path = root / "dataset.jsonl"
    if jsonl_path.exists():
        print(f"  SKIP {root}: dataset.jsonl already exists")
        return False

    print(f"  loading {old_path} ({old_path.stat().st_size / 1e6:.1f} MB)...")
    with open(old_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("_meta", {})
    root_str = meta.get("root")
    store_root = Path(root_str) if root_str else root
    data = {k: v for k, v in raw.items() if k != "_meta"}
    print(f"  {len(data)} entries (old _meta: {meta})")

    if dry_run:
        print(f"  DRY-RUN: would write {jsonl_path} + {jsonl_path.with_name('dataset.meta.json')}")
        return True

    store = DataStore(path=jsonl_path, data=data, root=store_root)
    store.flush()
    print(f"  wrote {jsonl_path}")

    # Verify: reload and diff entry-for-entry against the source dict.
    reloaded = DataStore.load(jsonl_path)
    if len(reloaded) != len(data):
        raise RuntimeError(
            f"VERIFY FAILED for {root}: {len(reloaded)} reloaded vs {len(data)} source entries"
        )
    if reloaded.data != data:
        # find first mismatch for a useful error
        for key in data:
            if key not in reloaded.data or reloaded.data[key] != data[key]:
                raise RuntimeError(
                    f"VERIFY FAILED for {root}: mismatch at key {key!r}\n"
                    f"  source:   {data[key]!r}\n"
                    f"  reloaded: {reloaded.data.get(key)!r}"
                )
        raise RuntimeError(f"VERIFY FAILED for {root}: dicts differ but no single key found")
    print(f"  VERIFIED: {len(reloaded)} entries round-trip identical")

    backup_path = old_path.with_suffix(".json.bak")
    old_path.rename(backup_path)
    print(f"  renamed {old_path.name} -> {backup_path.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="Directories containing dataset.json")
    ap.add_argument("--dry-run", action="store_true", help="Load and report, but don't write anything")
    args = ap.parse_args()

    ok = 0
    for root in args.roots:
        print(f"[{root}]")
        try:
            if migrate(root, dry_run=args.dry_run):
                ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
    print(f"\n{ok}/{len(args.roots)} migrated")


if __name__ == "__main__":
    main()
