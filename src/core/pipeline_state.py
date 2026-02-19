"""
Persistent pipeline state for resumable runs.

Caches track lists, audio durations, stage completion, and per-pass progress
so that restarting a large pipeline run skips expensive directory scans and
resumes from where it left off.

State file: .pipeline_state.json in the working directory.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_FILENAME = '.pipeline_state.json'


class PipelineState:
    """Persistent state for resumable pipeline runs."""

    def __init__(self, working_dir: Path):
        self.state_path = working_dir / STATE_FILENAME
        self._state = self._load()

    def _load(self) -> dict:
        """Load existing state file or return empty state."""
        if not self.state_path.exists():
            return self._empty_state()

        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            if state.get('version') != STATE_VERSION:
                logger.info(f"State file version mismatch (got {state.get('version')}, "
                            f"expected {STATE_VERSION}), starting fresh")
                return self._empty_state()
            return state
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read state file: {e}")
            return self._empty_state()

    def _empty_state(self) -> dict:
        return {
            'version': STATE_VERSION,
            'config_hash': '',
            'working_dir': str(self.state_path.parent),
            'created': datetime.now(timezone.utc).isoformat(),
            'updated': '',
            'tracks': {},
            'total_duration': 0.0,
            'stages_completed': [],
            'crop_features': {},
        }

    def save(self):
        """Write state to disk atomically (temp file + rename)."""
        self._state['updated'] = datetime.now(timezone.utc).isoformat()
        try:
            dir_path = self.state_path.parent
            fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self._state, f, indent=2)
                os.replace(tmp_path, self.state_path)
            except Exception:
                os.unlink(tmp_path)
                raise
        except OSError as e:
            logger.warning(f"Could not save state file: {e}")

    # --- Config validation ---

    def is_valid_for(self, config_hash: str) -> bool:
        """Check if cached state matches current config and has track data."""
        return (
            self._state.get('config_hash') == config_hash
            and bool(self._state.get('tracks'))
        )

    def set_config_hash(self, config_hash: str):
        self._state['config_hash'] = config_hash

    # --- Track index ---

    def get_track_names(self) -> List[str]:
        """Return list of cached track folder names."""
        return list(self._state.get('tracks', {}).keys())

    def get_tracks(self) -> Dict[str, dict]:
        """Return full cached track dict."""
        return self._state.get('tracks', {})

    def set_tracks(self, folders: List[Path], durations: Dict[str, float]):
        """Cache the track list and per-track durations after a full scan."""
        tracks = {}
        for folder in folders:
            name = folder.name
            tracks[name] = {
                'duration': durations.get(name, 0.0),
            }
        self._state['tracks'] = tracks
        self._state['total_duration'] = sum(durations.values())

    def update_track(self, folder_name: str, **kwargs):
        """Update fields on a cached track entry."""
        tracks = self._state.setdefault('tracks', {})
        entry = tracks.setdefault(folder_name, {})
        entry.update(kwargs)

    def get_total_duration(self) -> float:
        """Return cached total audio duration in seconds."""
        return self._state.get('total_duration', 0.0)

    # --- Stage tracking ---

    def mark_stage_completed(self, stage: str):
        """Mark a top-level stage as done."""
        completed = self._state.setdefault('stages_completed', [])
        if stage not in completed:
            completed.append(stage)

    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage was already completed in a previous run."""
        return stage in self._state.get('stages_completed', [])

    def reset_stages(self):
        """Clear all stage completion markers (for config changes)."""
        self._state['stages_completed'] = []

    # --- Crop feature progress ---

    def get_pass_progress(self, pass_name: str) -> Tuple[int, int]:
        """Return (completed, total) for a given pass."""
        progress = self._state.get('crop_features', {}).get(pass_name, {})
        return progress.get('completed', 0), progress.get('total', 0)

    def update_pass_progress(self, pass_name: str, completed: int, total: int):
        """Update progress for a pass."""
        features = self._state.setdefault('crop_features', {})
        features[pass_name] = {'completed': completed, 'total': total}

    def is_pass_completed(self, pass_name: str) -> bool:
        """Check if a pass finished all its files."""
        completed, total = self.get_pass_progress(pass_name)
        return total > 0 and completed >= total

    # --- Invalidation ---

    def invalidate(self):
        """Reset state (config changed). Keep durations since those don't change."""
        durations = {name: t.get('duration', 0.0)
                     for name, t in self._state.get('tracks', {}).items()}
        total = self._state.get('total_duration', 0.0)

        self._state['stages_completed'] = []
        self._state['crop_features'] = {}
        self._state['config_hash'] = ''

        # Preserve duration cache — audio files don't change between config tweaks
        for name, track in self._state.get('tracks', {}).items():
            track_duration = durations.get(name, 0.0)
            self._state['tracks'][name] = {'duration': track_duration}
        self._state['total_duration'] = total

    def get_cached_durations(self) -> Dict[str, float]:
        """Return {folder_name: duration} from cache, even if config changed."""
        return {name: t.get('duration', 0.0)
                for name, t in self._state.get('tracks', {}).items()}

    # --- Config hash computation ---

    @staticmethod
    def compute_config_hash(config) -> str:
        """Hash the config fields that affect what work is done.

        Excludes volatile fields (verbose, workers, device, dry_run).
        """
        # Collect all fields that determine what work needs to happen
        hash_fields = {}

        # Skip flags
        for attr in dir(config):
            if attr.startswith('skip_'):
                hash_fields[attr] = getattr(config, attr)

        # Essentia sub-options
        for attr in ['essentia_genre', 'essentia_mood', 'essentia_instrument',
                      'essentia_voice', 'essentia_gender']:
            if hasattr(config, attr):
                hash_fields[attr] = getattr(config, attr)

        # Crop settings (change crop boundaries → different files)
        for attr in ['crop_length_samples', 'crop_mode', 'crop_overlap', 'crop_div4',
                      'crop_include_stems']:
            if hasattr(config, attr):
                hash_fields[attr] = getattr(config, attr)

        # Model choices
        for attr in ['separation_backend', 'flamingo_model']:
            if hasattr(config, attr):
                hash_fields[attr] = getattr(config, attr)

        # Overwrite flags
        if hasattr(config, 'overwrite'):
            hash_fields['overwrite'] = config.overwrite
        if hasattr(config, 'per_feature_overwrite'):
            hash_fields['per_feature_overwrite'] = dict(config.per_feature_overwrite)

        # Flamingo prompts and revision config
        if hasattr(config, 'flamingo_prompts'):
            hash_fields['flamingo_prompts'] = dict(config.flamingo_prompts) if config.flamingo_prompts else {}
        if hasattr(config, 'flamingo_revision'):
            hash_fields['flamingo_revision'] = dict(config.flamingo_revision) if config.flamingo_revision else {}

        # Deterministic JSON serialization for hashing
        canonical = json.dumps(hash_fields, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
