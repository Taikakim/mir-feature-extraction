"""
Repair script: recompute rms_energy_bass/body/mid for crops where inline
extraction wrote -60.0 due to missing mono conversion (bug fixed in
multiband_rms.py).

Detection heuristic: bass=-60 AND body=-60 AND mid=-60 AND air≠-60.
Genuinely silent crops would have all four at -60; those are skipped.

Usage:
    python scripts/repair_multiband_rms.py /path/to/crops [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def is_affected(info: dict) -> bool:
    return (
        info.get('rms_energy_bass') == -60.0 and
        info.get('rms_energy_body') == -60.0 and
        info.get('rms_energy_mid') == -60.0 and
        info.get('rms_energy_air', -60.0) != -60.0
    )


def repair(crops_root: Path, dry_run: bool):
    from spectral.multiband_rms import analyze_multiband_rms
    from core.json_handler import safe_update

    affected = []
    for p in crops_root.rglob('*.INFO'):
        try:
            d = json.loads(p.read_text())
            if is_affected(d):
                affected.append(p)
        except Exception:
            pass

    print(f"Affected INFO files: {len(affected)}")
    if dry_run:
        print("(dry-run — no changes written)")
        return

    fixed = 0
    failed = 0
    for info_path in affected:
        # Find the audio file: same stem as INFO, drop .INFO extension
        audio_path = info_path.with_suffix('')
        if not audio_path.exists():
            # Try common extensions
            for ext in ('.flac', '.mp3', '.wav'):
                candidate = info_path.parent / (info_path.stem + ext)
                if candidate.exists():
                    audio_path = candidate
                    break
            else:
                print(f"  SKIP (no audio): {info_path.name}")
                failed += 1
                continue

        try:
            results = analyze_multiband_rms(audio_path)
            safe_update(info_path, results)
            fixed += 1
            if fixed % 500 == 0:
                print(f"  {fixed}/{len(affected)} fixed...")
        except Exception as e:
            print(f"  FAIL {info_path.name}: {e}")
            failed += 1

    print(f"Done — fixed: {fixed}, failed: {failed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('crops_root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repair(args.crops_root, args.dry_run)
