"""
Pipeline Statistics for MIR Project

Timing and statistics tracking for pipeline execution.

Usage:
    from core.pipeline_stats import TimingStats, PipelineStats
    
    stats = PipelineStats()
    stats.start_operation("demucs")
    # ... do work ...
    stats.end_operation("demucs", items_processed=10)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimingStats:
    """Statistics for a single timed operation."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    items_processed: int = 0
    items_skipped: int = 0
    
    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    @property
    def items_per_second(self) -> float:
        """Processing speed in items per second."""
        if self.elapsed > 0 and self.items_processed > 0:
            return self.items_processed / self.elapsed
        return 0.0
    
    @property
    def seconds_per_item(self) -> float:
        """Average time per item in seconds."""
        if self.items_processed > 0:
            return self.elapsed / self.items_processed
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'elapsed_seconds': round(self.elapsed, 2),
            'items_processed': self.items_processed,
            'items_skipped': self.items_skipped,
            'items_per_second': round(self.items_per_second, 3),
            'seconds_per_item': round(self.seconds_per_item, 3),
        }


@dataclass
class PipelineStats:
    """Statistics for pipeline execution with detailed timing."""
    # Organization
    files_found: int = 0
    files_organized: int = 0

    # Track analysis
    tracks_total: int = 0
    tracks_separated: int = 0
    tracks_analyzed: int = 0
    tracks_metadata_found: int = 0

    # Crops
    crops_created: int = 0
    crops_analyzed: int = 0

    # Legacy timing (for backward compatibility)
    time_organize: float = 0.0
    time_track_analysis: float = 0.0
    time_cropping: float = 0.0
    time_crop_analysis: float = 0.0

    # Detailed per-operation timing
    operation_timing: Dict[str, TimingStats] = field(default_factory=dict)
    
    # Run metadata
    run_start_time: float = field(default_factory=time.time)
    run_end_time: float = 0.0

    errors: List[str] = field(default_factory=list)
    
    def start_operation(self, name: str) -> TimingStats:
        """Start timing an operation."""
        timing = TimingStats(name=name, start_time=time.time())
        self.operation_timing[name] = timing
        return timing
    
    def end_operation(self, name: str, items_processed: int = 0, items_skipped: int = 0) -> float:
        """End timing an operation and return elapsed time."""
        if name in self.operation_timing:
            timing = self.operation_timing[name]
            timing.end_time = time.time()
            timing.items_processed = items_processed
            timing.items_skipped = items_skipped
            return timing.elapsed
        return 0.0
    
    def get_operation(self, name: str) -> Optional[TimingStats]:
        """Get timing stats for an operation."""
        return self.operation_timing.get(name)
    
    @property
    def total_time(self) -> float:
        """Total pipeline runtime."""
        if self.run_end_time > 0:
            return self.run_end_time - self.run_start_time
        return time.time() - self.run_start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all stats to dictionary for JSON serialization."""
        from datetime import datetime
        
        return {
            'run_timestamp': datetime.fromtimestamp(self.run_start_time).isoformat(),
            'total_runtime_seconds': round(self.total_time, 2),
            'total_runtime_minutes': round(self.total_time / 60, 2),
            'summary': {
                'files_found': self.files_found,
                'files_organized': self.files_organized,
                'tracks_total': self.tracks_total,
                'tracks_separated': self.tracks_separated,
                'tracks_analyzed': self.tracks_analyzed,
                'tracks_metadata_found': self.tracks_metadata_found,
                'crops_created': self.crops_created,
                'crops_analyzed': self.crops_analyzed,
            },
            'stage_timing': {
                'organize': round(self.time_organize, 2),
                'track_analysis': round(self.time_track_analysis, 2),
                'cropping': round(self.time_cropping, 2),
                'crop_analysis': round(self.time_crop_analysis, 2),
            },
            'operation_timing': {
                name: timing.to_dict() 
                for name, timing in self.operation_timing.items()
            },
            'errors': self.errors,
        }
