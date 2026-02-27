#!/usr/bin/env python3
"""Rebuild feature_explorer.html from the vara (known-good) template.

Data is loaded from external JS files rebuilt by statistical_analysis.py:
  feature_explorer_data.js      -- DATA, TRACKS, FEATURES arrays
  feature_explorer_classes.js   -- CLASSES dict
  feature_explorer_captions.js  -- CAPTIONS dict (lazy-loaded on demand)

Usage:
  python plots/rebuild_explorer.py
"""
import os, sys, shutil

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE  = os.path.join(PLOTS_DIR, 'feature_explorer-vara.html')
DST       = os.path.join(PLOTS_DIR, 'feature_explorer.html')

if not os.path.exists(TEMPLATE):
    print(f'ERROR: Template not found: {TEMPLATE}', file=sys.stderr)
    sys.exit(1)

size_before = os.path.getsize(DST) if os.path.exists(DST) else 0
shutil.copy2(TEMPLATE, DST)
size_after = os.path.getsize(DST)
print(f'Done. {size_after:,} chars ({size_after - size_before:+,} vs previous)')
