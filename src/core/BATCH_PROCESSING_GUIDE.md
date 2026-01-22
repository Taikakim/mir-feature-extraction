# Batch Processing Guide

How to implement safe, resumable, parallel batch processing with file locking.

## Core Components

### 1. File Locking (`file_locks.py`)

Prevents multiple processes from processing the same file simultaneously.

```python
from core.file_locks import FileLock

# Context manager (recommended)
with FileLock(folder) as lock:
    if lock.acquired:
        process_folder(folder)
    else:
        logger.info(f"Skipping {folder.name} - locked by another process")
```

### 2. Resume Capability (`batch_utils.py`)

Skips already-processed files unless --overwrite flag is used.

```python
from core.batch_utils import should_process_folder

if not should_process_folder(folder, required_features=['lufs', 'lra'], overwrite=False):
    logger.info(f"Skipping {folder.name} - already has features")
    continue
```

### 3. Combined Pattern

```python
from core.file_locks import FileLock
from core.batch_utils import should_process_folder

for folder in folders:
    # Check if needs processing (resume support)
    if not should_process_folder(folder, required_features=['lufs'], overwrite=args.overwrite):
        continue

    # Acquire lock (parallel safety)
    with FileLock(folder) as lock:
        if not lock.acquired:
            continue

        # Process folder (safe - locked and needed)
        process_folder(folder)
```

## Usage Patterns

### Pattern 1: Simple Batch with Resume

For CPU-bound tasks without model caching:

```python
def batch_analyze_loudness(root_directory: str | Path, overwrite: bool = False):
    from core.file_utils import find_organized_folders
    from core.file_locks import FileLock
    from core.batch_utils import should_process_folder, print_batch_summary

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped_complete': 0,
        'skipped_locked': 0,
        'failed': 0,
        'errors': []
    }

    required_features = ['lufs', 'lra', 'lufs_drums', 'lra_drums',
                         'lufs_bass', 'lra_bass', 'lufs_other', 'lra_other']

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Resume: Skip if already has features
        if not overwrite and has_features(folder, required_features):
            stats['skipped_complete'] += 1
            logger.info(f"  Skipping - features already exist")
            continue

        # Lock: Prevent parallel conflicts
        with FileLock(folder) as lock:
            if not lock.acquired:
                stats['skipped_locked'] += 1
                logger.info(f"  Skipping - locked by another process")
                continue

            # Process
            try:
                results = analyze_folder_loudness(folder)
                stats['success'] += 1
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(f"{folder.name}: {e}")
                logger.error(f"  Failed: {e}")

    print_batch_summary(stats, "Loudness Analysis")
    return stats
```

### Pattern 2: Batch with Model Caching

For GPU/model-based tasks (like Essentia):

```python
def batch_analyze_essentia_optimized(root_directory: str | Path, overwrite: bool = False):
    from core.file_utils import find_organized_folders
    from core.file_locks import FileLock
    from core.batch_utils import has_features, print_batch_summary
    from classification.essentia_features_optimized import EssentiaAnalyzer

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped_complete': 0,
        'skipped_locked': 0,
        'failed': 0
    }

    required_features = ['danceability', 'atonality']

    # OPTIMIZATION: Load models ONCE
    logger.info("Loading Essentia models...")
    analyzer = EssentiaAnalyzer()

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Resume: Skip if already complete
        if not overwrite and has_features(folder, required_features):
            stats['skipped_complete'] += 1
            logger.info(f"  Skipping - features already exist")
            continue

        # Lock: Prevent conflicts
        with FileLock(folder) as lock:
            if not lock.acquired:
                stats['skipped_locked'] += 1
                logger.info(f"  Skipping - locked by another process")
                continue

            # Process using cached model
            try:
                results = analyzer.analyze_file(get_full_mix(folder))
                save_to_info(folder, results)
                stats['success'] += 1
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"  Failed: {e}")

    print_batch_summary(stats, "Essentia Analysis")
    return stats
```

### Pattern 3: BatchProcessor Helper Class

Using the helper class for simplified code:

```python
from core.batch_utils import BatchProcessor

def batch_analyze_with_helper(root_directory: str | Path, overwrite: bool = False):
    from core.file_utils import find_organized_folders

    folders = find_organized_folders(root_directory)

    # Create processor with resume support
    processor = BatchProcessor(
        required_features=['lufs', 'lra'],
        lock_timeout=3600  # 1 hour
    )

    # Process batch (handles locking, resume, errors automatically)
    stats = processor.process_batch(
        folders=folders,
        process_func=analyze_folder_loudness,
        overwrite=overwrite,
        skip_errors=True
    )

    return stats
```

### Pattern 4: Parallel Processing (Future)

When implementing parallelization:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.file_locks import FileLock
from core.batch_utils import filter_folders_to_process

def process_folder_safe(folder: Path) -> dict:
    """Worker function for parallel processing."""

    # Acquire lock
    with FileLock(folder) as lock:
        if not lock.acquired:
            return {'skipped': True, 'reason': 'locked'}

        # Process
        try:
            results = analyze_folder(folder)
            return {'success': True, 'results': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}


def batch_analyze_parallel(root_directory: str | Path,
                           overwrite: bool = False,
                           max_workers: int = 20):
    from core.file_utils import find_organized_folders

    folders = find_organized_folders(root_directory)

    # Filter to only folders that need processing
    folders_to_process, stats = filter_folders_to_process(
        folders,
        required_features=['lufs', 'lra'],
        overwrite=overwrite,
        check_lock=False  # Will check in worker
    )

    logger.info(f"Processing {len(folders_to_process)} folders with {max_workers} workers")

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_folder_safe, folder): folder
            for folder in folders_to_process
        }

        # Collect results
        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
                if result.get('success'):
                    stats['success'] += 1
                elif result.get('skipped'):
                    stats['skipped_locked'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Worker failed for {folder.name}: {e}")

    return stats
```

## Utility Scripts

### Check Processing Progress

```bash
# See how many files are complete
python src/core/batch_utils.py progress dataset/ --features lufs lra bpm

# Output:
# Processing Progress:
#   Total folders:  10000
#   Complete:       3542 (35.4%)
#   Incomplete:     6458
#   Locked:         23
```

### Find Folders Needing Processing

```bash
# List folders that still need processing
python src/core/batch_utils.py check dataset/ --features lufs lra

# Output:
# Folders to Process: 6458
#   Total:          10000
#   Ready:          6458
#   Skipped (done): 3542
#   Skipped (lock): 0
```

### Show Missing Features

```bash
# See which specific features are missing
python src/core/batch_utils.py missing dataset/ --features lufs lra bpm danceability

# Output:
# Artist - Album - Track 1:
#   Missing: danceability
#
# Artist - Album - Track 2:
#   Missing: lufs, lra, bpm, danceability
```

### Clean Up Dead Locks

```bash
# Remove stale lock files (after crashes)
python src/core/file_locks.py cleanup dataset/ --timeout 3600

# Output:
# Scanning for dead locks in: dataset/
# Found 15 lock files
# Removed dead lock: Track1.lock (process no longer exists)
# Removed dead lock: Track2.lock (age > 1 hour)
# Removed 2 dead lock files
```

### List Active Locks

```bash
# See which files are currently being processed
python src/core/file_locks.py list dataset/

# Output:
# Found 3 lock files:
#   [ACTIVE] Track123.lock (age: 12.3 min)
#   [ACTIVE] Track456.lock (age: 5.1 min)
#   [DEAD] Track789.lock (age: 125.7 min)
```

## Best Practices

### 1. Always Use Both Locking and Resume

```python
# ✓ GOOD: Safe parallel processing with resume
if not overwrite and has_features(folder, required_features):
    continue

with FileLock(folder) as lock:
    if not lock.acquired:
        continue
    process_folder(folder)

# ✗ BAD: Missing lock (conflicts in parallel)
if not overwrite and has_features(folder, required_features):
    continue
process_folder(folder)  # Danger!

# ✗ BAD: Missing resume check (wastes time)
with FileLock(folder) as lock:
    if not lock.acquired:
        continue
    process_folder(folder)  # Reprocesses everything!
```

### 2. Set Appropriate Lock Timeouts

```python
# Short tasks (< 5 minutes per file)
lock = FileLock(folder, timeout=600)  # 10 minutes

# Medium tasks (5-30 minutes per file)
lock = FileLock(folder, timeout=3600)  # 1 hour

# Long tasks (30+ minutes per file)
lock = FileLock(folder, timeout=7200)  # 2 hours
```

### 3. Clean Up Dead Locks Periodically

```bash
# Before starting large batch processing
python src/core/file_locks.py cleanup dataset/

# Then start processing
python src/preprocessing/loudness.py dataset/ --batch
```

### 4. Monitor Progress During Long Runs

```bash
# Terminal 1: Run processing
python src/timbral/audio_commons.py dataset/ --batch

# Terminal 2: Monitor progress
watch -n 60 'python src/core/batch_utils.py progress dataset/ --features brightness roughness'

# Updates every 60 seconds with completion percentage
```

### 5. Handle Interruptions Gracefully

Processing can be stopped (Ctrl+C) and resumed safely:

```bash
# Start processing
python src/preprocessing/loudness.py dataset/ --batch

# (interrupt with Ctrl+C after 1000 files)

# Resume later - skips completed files automatically
python src/preprocessing/loudness.py dataset/ --batch

# Output:
# Processing 1/10000: Track001
#   Skipping - features already exist
# Processing 2/10000: Track002
#   Skipping - features already exist
# ...
# Processing 1001/10000: Track1001
#   Processing...  (resumes from where it left off)
```

### 6. Run Multiple Instances in Parallel

Safe to run multiple instances (different machines or screens):

```bash
# Terminal 1 (or Machine 1)
python src/preprocessing/loudness.py dataset/ --batch

# Terminal 2 (or Machine 2)
python src/preprocessing/loudness.py dataset/ --batch

# Both instances coordinate via locks - no conflicts!
# Output shows:
#   Processing 45/10000: Track045
#     Skipping - locked by another process  (other instance is processing)
#   Processing 46/10000: Track046
#     Processing...  (this instance processes)
```

## Integration Checklist

When adding locking/resume to a batch function:

- [ ] Import required modules
  ```python
  from core.file_locks import FileLock
  from core.batch_utils import has_features, print_batch_summary
  ```

- [ ] Define required features list
  ```python
  required_features = ['lufs', 'lra', 'bpm']
  ```

- [ ] Add overwrite parameter
  ```python
  def batch_analyze(root_directory, overwrite: bool = False):
  ```

- [ ] Check features before processing
  ```python
  if not overwrite and has_features(folder, required_features):
      stats['skipped_complete'] += 1
      continue
  ```

- [ ] Wrap processing in FileLock
  ```python
  with FileLock(folder) as lock:
      if not lock.acquired:
          stats['skipped_locked'] += 1
          continue
      # ... process folder
  ```

- [ ] Track statistics properly
  ```python
  stats = {
      'total': len(folders),
      'success': 0,
      'skipped_complete': 0,
      'skipped_locked': 0,
      'failed': 0,
      'errors': []
  }
  ```

- [ ] Print summary at end
  ```python
  print_batch_summary(stats, "Operation Name")
  ```

## Example: Complete Implementation

See `src/classification/essentia_features_optimized.py` for a reference implementation that includes:
- ✓ Model caching
- ✓ File locking
- ✓ Resume capability
- ✓ Progress tracking
- ✓ Error handling
- ✓ Proper statistics

This serves as a template for optimizing other modules.

---

**Note:** Parallel processing will be added after all modules have locking and resume support.
