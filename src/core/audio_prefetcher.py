"""
Audio Prefetcher for MIR Pipeline

Loads audio files into RAM in a background thread to hide HDD I/O latency.
Useful when processing files from slow storage (HDD) while GPU does computation.

Usage:
    from core.audio_prefetcher import AudioPrefetcher

    with AudioPrefetcher(file_list, buffer_size=4) as prefetcher:
        for path, audio_data in prefetcher:
            # audio_data is (samples, sample_rate) tuple
            process(audio_data)
"""

import logging
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import List, Tuple, Optional, Iterator, Union
import numpy as np

logger = logging.getLogger(__name__)


class AudioPrefetcher:
    """
    Prefetches audio files into RAM using a background thread.

    While the main thread processes file N, the background thread
    loads files N+1, N+2, etc. into a bounded queue.

    Args:
        file_list: List of audio file paths to process
        buffer_size: Number of files to keep loaded ahead (default: 4)
        target_sr: Target sample rate for resampling (None = keep original)
        mono: Convert to mono (default: True)
    """

    def __init__(
        self,
        file_list: List[Path],
        buffer_size: int = 4,
        target_sr: Optional[int] = None,
        mono: bool = True,
    ):
        self.file_list = [Path(f) for f in file_list]
        self.buffer_size = buffer_size
        self.target_sr = target_sr
        self.mono = mono

        self._queue: Queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._loader_thread: Optional[threading.Thread] = None
        self._files_loaded = 0
        self._load_errors = 0

    def __enter__(self):
        """Start the prefetch thread."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the prefetch thread."""
        self.stop()
        return False

    def start(self):
        """Start the background loading thread."""
        if self._loader_thread is not None:
            return

        self._stop_event.clear()
        self._loader_thread = threading.Thread(
            target=self._load_files,
            daemon=True,
            name="AudioPrefetcher"
        )
        self._loader_thread.start()
        logger.debug(f"AudioPrefetcher started (buffer_size={self.buffer_size})")

    def stop(self):
        """Stop the background loading thread."""
        if self._loader_thread is None:
            return

        self._stop_event.set()

        # Drain the queue to unblock the loader thread
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Empty:
            pass

        self._loader_thread.join(timeout=5.0)
        self._loader_thread = None
        logger.debug(f"AudioPrefetcher stopped ({self._files_loaded} loaded, {self._load_errors} errors)")

    def _load_files(self):
        """Background thread: load files into the queue."""
        from core.file_utils import read_audio

        for file_path in self.file_list:
            if self._stop_event.is_set():
                break

            try:
                # Load audio file (supports m4a/aac via pydub fallback)
                audio, sr = read_audio(str(file_path), dtype='float32')

                # Convert to mono if requested
                if self.mono and audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Resample if target_sr specified
                if self.target_sr is not None and sr != self.target_sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr

                # Put in queue (blocks if full)
                self._queue.put((file_path, audio, sr))
                self._files_loaded += 1

            except Exception as e:
                logger.warning(f"Prefetch failed for {file_path.name}: {e}")
                # Put error marker in queue so consumer knows about it
                self._queue.put((file_path, None, None))
                self._load_errors += 1

        # Signal end of files
        self._queue.put((None, None, None))

    def __iter__(self) -> Iterator[Tuple[Path, Optional[np.ndarray], Optional[int]]]:
        """Iterate over prefetched files."""
        while True:
            try:
                file_path, audio, sr = self._queue.get(timeout=60.0)
            except Empty:
                logger.warning("Prefetcher queue timeout")
                break

            if file_path is None:
                # End of files
                break

            yield file_path, audio, sr

    def get(self, timeout: float = 60.0) -> Tuple[Optional[Path], Optional[np.ndarray], Optional[int]]:
        """
        Get the next prefetched file.

        Returns:
            (file_path, audio_array, sample_rate) or (None, None, None) if done
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None, None, None

    @property
    def stats(self) -> dict:
        """Get prefetch statistics."""
        return {
            'files_total': len(self.file_list),
            'files_loaded': self._files_loaded,
            'load_errors': self._load_errors,
            'queue_size': self._queue.qsize(),
        }


class PathPrefetcher:
    """
    Simpler prefetcher that just reads file bytes into RAM.

    For use with libraries that need file paths but benefit from
    having the file already in OS disk cache.

    This doesn't parse the audio, just reads raw bytes to warm the disk cache.
    """

    def __init__(self, file_list: List[Path], buffer_size: int = 8):
        self.file_list = [Path(f) for f in file_list]
        self.buffer_size = buffer_size
        self._cache: dict = {}
        self._stop_event = threading.Event()
        self._loader_thread: Optional[threading.Thread] = None
        self._current_idx = 0
        self._lock = threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        """Start background prefetching."""
        if self._loader_thread is not None:
            return

        self._stop_event.clear()
        self._loader_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name="PathPrefetcher"
        )
        self._loader_thread.start()

    def stop(self):
        """Stop background prefetching."""
        if self._loader_thread is None:
            return

        self._stop_event.set()
        self._loader_thread.join(timeout=2.0)
        self._loader_thread = None
        self._cache.clear()

    def _prefetch_loop(self):
        """Background thread: read files into cache."""
        while not self._stop_event.is_set():
            with self._lock:
                # Determine which files to prefetch
                start_idx = self._current_idx
                end_idx = min(start_idx + self.buffer_size, len(self.file_list))

            for idx in range(start_idx, end_idx):
                if self._stop_event.is_set():
                    break

                file_path = self.file_list[idx]
                cache_key = str(file_path)

                if cache_key not in self._cache:
                    try:
                        # Read file bytes into memory (warms OS disk cache)
                        with open(file_path, 'rb') as f:
                            _ = f.read()
                        self._cache[cache_key] = True
                    except Exception as e:
                        logger.debug(f"Prefetch failed: {file_path.name}: {e}")
                        self._cache[cache_key] = False

            # Wait a bit before checking again
            self._stop_event.wait(0.1)

    def mark_processed(self, file_path: Path):
        """Mark a file as processed, allowing cache cleanup."""
        with self._lock:
            self._current_idx += 1
            # Remove old entries from cache
            cache_key = str(file_path)
            self._cache.pop(cache_key, None)


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python audio_prefetcher.py <directory>")
        sys.exit(1)

    from core.file_utils import find_audio_files

    audio_dir = Path(sys.argv[1])
    files = find_audio_files(audio_dir)[:10]  # Test with first 10 files

    print(f"Testing prefetcher with {len(files)} files...")

    with AudioPrefetcher(files, buffer_size=3, mono=True) as prefetcher:
        for path, audio, sr in prefetcher:
            if audio is not None:
                print(f"  {path.name}: {audio.shape}, {sr}Hz")
            else:
                print(f"  {path.name}: FAILED")

    print("Done!")
