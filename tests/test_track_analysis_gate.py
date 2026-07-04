"""Regression test for the Stage 2 (track analysis) completeness gate.

Bug (caught 2026-07-04): master_pipeline.run() marked 'track_analysis' complete
UNCONDITIONALLY after _run_track_analysis() returned, even when stem separation
failed on every track (0 separated / N failed). The state gate then silently
no-op'd every subsequent re-run instead of retrying — a silent data-incompleteness
trap. The fix gates the completion mark on PipelineStats.track_analysis_succeeded().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.pipeline_stats import PipelineStats


def test_wholesale_separation_failure_is_not_success():
    """0 separated + >0 failed = the bug's signature — must NOT mark complete."""
    stats = PipelineStats()
    stats.tracks_separated = 0
    stats.tracks_separation_failed = 42
    assert stats.track_analysis_succeeded() is False


def test_real_progress_is_success():
    """Any successful separations → complete."""
    stats = PipelineStats()
    stats.tracks_separated = 42
    stats.tracks_separation_failed = 0
    assert stats.track_analysis_succeeded() is True


def test_partial_success_is_success():
    """Some succeeded, some failed → progress was made; mark complete."""
    stats = PipelineStats()
    stats.tracks_separated = 40
    stats.tracks_separation_failed = 2
    assert stats.track_analysis_succeeded() is True


def test_nothing_to_do_is_success():
    """All stems already present (all skipped) or separation disabled: 0/0 → complete."""
    stats = PipelineStats()
    stats.tracks_separated = 0
    stats.tracks_separation_failed = 0
    assert stats.track_analysis_succeeded() is True


def test_default_stats_are_success():
    """A fresh stats object (no separation attempted) is not a failure."""
    assert PipelineStats().track_analysis_succeeded() is True
