#!/usr/bin/env python3
"""
Script to fix the get_info_path() bug in batch analysis functions.

Bug: Modules were calling get_info_path(folder) which resolves to parent's .INFO
Fix: Call get_info_path(stems['full_mix']) to get the correct track's .INFO

Files to fix:
- src/rhythm/onsets.py
- src/rhythm/syncopation.py
- src/rhythm/complexity.py
- src/rhythm/per_stem_rhythm.py
- src/spectral/spectral_features.py
- src/spectral/multiband_rms.py
- src/harmonic/chroma.py
- src/harmonic/per_stem_harmonic.py
- src/timbral/audiobox_aesthetics.py
"""

import re
from pathlib import Path

FILES_TO_FIX = [
    "src/rhythm/onsets.py",
    "src/rhythm/syncopation.py",
    "src/rhythm/complexity.py",
    "src/rhythm/per_stem_rhythm.py",
    "src/spectral/spectral_features.py",
    "src/spectral/multiband_rms.py",
    "src/harmonic/chroma.py",
    "src/harmonic/per_stem_harmonic.py",
    "src/timbral/audiobox_aesthetics.py",
]

def fix_file(filepath):
    """Fix get_info_path bug in a file."""
    path = Path(filepath)
    if not path.exists():
        print(f"  SKIP: {filepath} not found")
        return False

    content = path.read_text()

    # Pattern to find and fix the batch function
    # Look for the section where we check if already processed BEFORE getting stems
    pattern = r'(        # Check if already processed\n        info_path = get_info_path\(folder\)\n.*?except Exception:\n                pass\n\n        # Find full_mix file\n        stems = get_stem_files\(folder, include_full_mix=True\)\n        if \'full_mix\' not in stems:\n            logger\.warning\(f"No full_mix found in \{folder\.name\}"\)\n            stats\[\'failed\'\] \+= 1\n            continue)'

    # This is complex, let's do a simpler approach
    # Find: info_path = get_info_path(folder) that comes BEFORE stems = get_stem_files
    # Replace with moving the get_stem_files up and changing to get_info_path(stems['full_mix'])

    # Simpler: just swap the two sections
    old_pattern = """        # Check if already processed
        info_path = get_info_path(folder)
        if info_path.exists() and not overwrite:
            try:
                import json
                with open(info_path, 'r') as f:
                    data = json.load(f)"""

    if old_pattern in content:
        # Find the full section to replace
        # This is getting complex - let me just do manual replacements
        lines = content.split('\n')
        new_lines = []
        i = 0
        fixed = False

        while i < len(lines):
            line = lines[i]

            # Look for "# Check if already processed" followed by "info_path = get_info_path(folder)"
            if i + 1 < len(lines) and \
               "# Check if already processed" in line and \
               "info_path = get_info_path(folder)" in lines[i + 1]:

                # Found the bug pattern
                # Need to:
                # 1. Skip the "Check if already processed" section
                # 2. Find the "Find full_mix file" section
                # 3. Swap them

                # Collect "Check if already processed" section
                check_section = [line]  # "# Check if already processed"
                i += 1
                while i < len(lines):
                    check_section.append(lines[i])
                    if "except Exception:" in lines[i]:
                        # Continue until we find the "pass" and empty line
                        i += 1
                        check_section.append(lines[i])  # "pass" line
                        i += 1
                        if i < len(lines) and lines[i].strip() == "":
                            check_section.append(lines[i])  # empty line
                        break
                    i += 1

                # Now collect "Find full_mix file" section
                i += 1
                find_section = []
                while i < len(lines):
                    find_section.append(lines[i])
                    if "continue" in lines[i] and "stats['failed']" in lines[i - 1]:
                        break
                    i += 1

                # Now output in correct order:
                # 1. Find full_mix section first
                new_lines.extend(find_section)
                new_lines.append("")  # empty line

                # 2. Fix the check section - change get_info_path(folder) to get_info_path(stems['full_mix'])
                for check_line in check_section:
                    if "info_path = get_info_path(folder)" in check_line:
                        check_line = check_line.replace("get_info_path(folder)", "get_info_path(stems['full_mix'])")
                    new_lines.append(check_line)

                fixed = True
            else:
                new_lines.append(line)

            i += 1

        if fixed:
            new_content = '\n'.join(new_lines)
            path.write_text(new_content)
            print(f"  FIXED: {filepath}")
            return True
        else:
            print(f"  SKIP: Pattern not found in {filepath}")
            return False
    else:
        print(f"  SKIP: Already fixed or different pattern in {filepath}")
        return False

def main():
    print("Fixing get_info_path() bug in batch analysis functions...")
    print()

    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if fix_file(filepath):
            fixed_count += 1

    print()
    print(f"Fixed {fixed_count}/{len(FILES_TO_FIX)} files")

if __name__ == "__main__":
    main()
