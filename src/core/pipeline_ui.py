"""
Pipeline TUI — btop-like static dashboard for MIR pipeline runs.

Redirects all log output to a rotating log file and displays a Rich full-screen
dashboard with: pipeline stage progress, per-feature status panel, current
operation + progress bar, statistics panel, and a scrolling recent-log panel.

Usage:
    ui = PipelineUI(stats=pipeline.stats)
    ui.start(config_name='master_pipeline.yaml', log_path=Path('pipeline.log'))
    ui.set_feature_config(features={'loudness': True, 'timbral': False, ...},
                          disabled_stages={'organize', 'cropping'})
    ui.set_stage('organize', 'running')
    ui.set_stage('organize', 'done')
    ui.set_current(file='Artist - Track', operation='Feature extraction',
                   done=5, total=200)
    ui.stop()  # prints final summary to stdout

Use --no-ui to skip the TUI and fall back to coloured stdout logging.
"""

import logging
import os
import re
import select
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── ANSI stripping ─────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mKJH]')

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)


# ── Stage definitions (display order) ──────────────────────────────────────────

STAGES: List[Tuple[str, str]] = [
    ('organize',        'Organize'),
    ('metadata_id3',    'ID3 Tags'),
    ('track_analysis',  'Track Analysis'),
    ('onset_analysis',  'Onsets'),
    ('metadata_lookup', 'Metadata Lookup'),
    ('first_features',  'First-Stage Features'),
    ('cropping',        'Cropping'),
    ('crop_analysis',   'Crop Analysis'),
    ('flamingo',        'Music Flamingo'),
    ('granite',         'Granite Revision'),
]

# ── Feature groups ──────────────────────────────────────────────────────────────
# (group_key, display_label, [feature_config_keys], {stage_ids_where_active})
FEATURE_GROUPS: List[Tuple[str, str, List[str], Set[str]]] = [
    ('loudness',    'Loudness',   ['loudness'],
                                  {'first_features', 'crop_analysis'}),
    ('spectral',    'Spectral',   ['spectral'],
                                  {'first_features', 'crop_analysis'}),
    ('saturation',  'Saturation', ['saturation'],
                                  {'first_features', 'crop_analysis'}),
    ('rms',         'RMS',        ['multiband_rms'],
                                  {'first_features', 'crop_analysis'}),
    ('rhythm',      'Rhythm',     ['syncopation', 'complexity'],
                                  {'first_features'}),
    ('per_stem',    'Per-Stem',   ['per_stem_rhythm', 'per_stem_harmonic', 'per_stem'],
                                  {'first_features', 'crop_analysis'}),
    ('chroma',      'Chroma',     ['chroma', 'hpcp_tiv'],
                                  {'crop_analysis'}),
    ('timbral',     'Timbral',    ['timbral'],
                                  {'crop_analysis'}),
    ('essentia',    'Essentia',   ['essentia'],
                                  {'crop_analysis'}),
    ('audiobox',    'AudioBox',   ['audiobox'],
                                  {'crop_analysis'}),
    ('flamingo',    'Flamingo',   ['music_flamingo'],
                                  {'flamingo'}),
    ('granite',     'Granite',    ['granite'],
                                  {'granite'}),
    ('metadata',    'Metadata',   ['metadata'],
                                  {'metadata_lookup'}),
]

# Short display names for individual config keys shown as sub-labels
_KEY_SHORT: Dict[str, str] = {
    'loudness':          'lufs/lra',
    'spectral':          'spectral',
    'saturation':        'sat',
    'multiband_rms':     'rms',
    'syncopation':       'syncopation',
    'complexity':        'complexity',
    'per_stem_rhythm':   'stem-rhythm',
    'per_stem_harmonic': 'stem-harm',
    'per_stem':          'per-stem',
    'chroma':            'chroma',
    'timbral':           'timbral',
    'essentia':          'essentia',
    'audiobox':          'audiobox',
    'music_flamingo':    'flamingo',
    'granite':           'granite',
    'metadata':          'metadata',
}

# Mapping from PipelineStats.operation_timing keys → feature group label
# Used to pull per-feature rates from PipelineStats into the features panel.
_OP_TO_GROUP_LABEL: Dict[str, str] = {
    'loudness':      'Loudness',
    'spectral':      'Spectral',
    'timbral':       'Timbral',
    'essentia':      'Essentia',
    'flamingo':      'Flamingo',
    'granite':       'Granite',
    'metadata':      'Metadata',
    'crop_features': 'Spectral',
    'beats':         'Rhythm',
    'onsets':        'Rhythm',
}


# ── Bevel panel ────────────────────────────────────────────────────────────────

class BevelPanel:
    """Panel with gradient bevel border (lofi metal effect).

    The border fades from ``#1B3639`` (dark corners) through ``#00838F``
    (mid-teal along the sides) to ``#4DB6AC`` (bright highlight at the
    centre of each edge), giving a subtle embossed/metallic look.
    """

    _CORNER = '#1B3639'
    _MID    = '#00838F'
    _HIGH   = '#4DB6AC'
    _TL, _TR, _BL, _BR = '╭', '╮', '╰', '╯'
    _H,  _V             = '─', '│'

    def __init__(self, renderable, title: str = '', padding: tuple = (0, 1)):
        self._renderable = renderable
        self._title      = title
        self._padding    = padding

    @classmethod
    def _lerp(cls, a: str, b: str, t: float) -> str:
        """Linear interpolate between two '#RRGGBB' hex colours."""
        t = max(0.0, min(1.0, t))
        ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
        br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
        return (f'#{int(ar+(br-ar)*t):02X}'
                f'{int(ag+(bg-ag)*t):02X}'
                f'{int(ab+(bb-ab)*t):02X}')

    def _hcolor(self, frac: float) -> str:
        """Colour for a horizontal border segment.
        frac=0 → corner (dark); frac=1 → centre (bright highlight)."""
        if frac < 0.45:
            return self._lerp(self._CORNER, self._MID, frac / 0.45)
        return self._lerp(self._MID, self._HIGH, (frac - 0.45) / 0.55)

    def _vcolor(self, frac: float) -> str:
        """Colour for a vertical border segment.
        frac=0 → corner (dark); frac=1 → centre of panel height."""
        return self._lerp(self._CORNER, self._MID, min(1.0, frac * 4.0))

    def __rich_console__(self, console, options):
        from rich.segment import Segment
        from rich.style   import Style
        from rich.text    import Text

        width     = options.max_width
        ph        = self._padding[1] if len(self._padding) > 1 else 0
        pv        = self._padding[0]
        content_w = max(1, width - 2 - ph * 2)
        inner_w   = width - 2          # chars between the two corner chars

        # Render inner content first (need row count for vertical gradient)
        rows       = console.render_lines(
            self._renderable, options.update_width(content_w), pad=True)
        total_rows = len(rows) + pv * 2

        # ── Parse title ──────────────────────────────────────────────────────
        title_t     = None
        title_cells = 0
        if self._title:
            title_t     = Text.from_markup(self._title)
            title_cells = len(title_t.plain) + 2   # space + text + space
            if title_cells >= inner_w - 2:
                title_t, title_cells = None, 0

        left_bars  = (inner_w - title_cells) // 2
        right_bars = inner_w - title_cells - left_bars

        def hfrac(i: int) -> float:
            """Symmetric frac for horizontal position i: 0 at corners, 1 at centre."""
            c = (inner_w - 1) / 2.0
            return 1.0 - abs(i - c) / max(1.0, c)

        # ── Top border ───────────────────────────────────────────────────────
        top = Text()
        top.append(self._TL, style=self._CORNER)
        for i in range(left_bars):
            top.append(self._H, style=self._hcolor(hfrac(i)))
        if title_t is not None:
            top.append(' ', style=self._hcolor(hfrac(left_bars)))
            top.append_text(title_t)
            top.append(' ', style=self._hcolor(hfrac(left_bars + title_cells - 1)))
        for i in range(right_bars):
            top.append(self._H, style=self._hcolor(hfrac(left_bars + title_cells + i)))
        top.append(self._TR, style=self._CORNER)
        yield from console.render(top, options.update_width(width))
        yield Segment.line()

        # ── Content rows (left/right vertical borders with vertical gradient) ─
        def _emit(segs, row_idx: int):
            frac = (1.0 - abs(2.0 * row_idx / max(1, total_rows - 1) - 1.0)
                    if total_rows > 1 else 1.0)
            vc = self._vcolor(frac)
            yield Segment(self._V, Style(color=vc))
            yield Segment(' ' * ph)
            yield from segs
            yield Segment(' ' * ph)
            yield Segment(self._V, Style(color=vc))
            yield Segment.line()

        blank = [Segment(' ' * content_w)]
        for r in range(pv):
            yield from _emit(blank, r)
        for r, segs in enumerate(rows):
            yield from _emit(segs, r + pv)
        for r in range(pv):
            yield from _emit(blank, len(rows) + pv + r)

        # ── Bottom border ────────────────────────────────────────────────────
        bot = Text()
        bot.append(self._BL, style=self._CORNER)
        for i in range(inner_w):
            bot.append(self._H, style=self._hcolor(hfrac(i)))
        bot.append(self._BR, style=self._CORNER)
        yield from console.render(bot, options.update_width(width))
        yield Segment.line()


# ── Logging handler ────────────────────────────────────────────────────────────

class TUILogHandler(logging.Handler):
    """Captures log records to a deque for the recent-log panel.  Does NOT emit
    to any stream — that role is handled by the FileHandler."""

    def __init__(self, log_deque: deque):
        super().__init__()
        self._deque = log_deque

    def emit(self, record: logging.LogRecord):
        try:
            ts = time.strftime('%H:%M:%S', time.localtime(record.created))
            level = record.levelname
            raw = _strip_ansi(record.getMessage())[:130]
            self._deque.append((ts, level, raw))
        except Exception:
            pass


# ── Module-level state (singleton reference + active flag) ────────────────────

_tui_active = False
_current_ui: Optional['PipelineUI'] = None  # set by PipelineUI.start()


def is_tui_active() -> bool:
    """Returns True while the TUI dashboard is running."""
    return _tui_active


@contextmanager
def suspend_tui_for_io() -> Generator:
    """Context manager: temporarily suspend the TUI so interactive terminal
    I/O (OAuth prompts, URLs, etc.) is visible to the user.

    Usage::

        from core.pipeline_ui import suspend_tui_for_io
        with suspend_tui_for_io():
            print("Visit: https://...")
            wait_for_user_input()
    """
    ui = _current_ui
    if ui is not None and ui.is_active:
        ui.suspend()
        try:
            yield
        finally:
            ui.resume()
    else:
        yield


# ── Main class ─────────────────────────────────────────────────────────────────

class PipelineUI:
    """
    Rich full-screen dashboard for MIR pipeline runs.

    Thread safety: all mutable state is protected by self._lock.  The refresh
    thread holds the lock only for the brief snapshot it needs to render; the
    pipeline's main thread holds it only for the brief duration of each setter.
    """

    def __init__(self, stats=None):
        """
        Args:
            stats: PipelineStats instance (from core.pipeline_stats).  The
                   refresh thread reads throughput / count data directly from
                   it — no extra calls needed for those fields.
        """
        self._stats = stats
        self._log_deque: deque = deque(maxlen=15)
        self._live: Optional['Live'] = None
        self._console: Optional['Console'] = None
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._active = False
        self._lock = threading.Lock()

        # Stage state: id → (status, elapsed_s)
        self._stages: Dict[str, Tuple[str, float]] = {
            s[0]: ('pending', 0.0) for s in STAGES
        }
        self._stage_start: Dict[str, float] = {}

        # Feature config
        self._feature_enabled: Dict[str, bool] = {}   # feature_key → enabled
        self._disabled_stages: Set[str] = set()        # stages disabled in config

        # Per-feature-group rate overrides (label → rate)
        self._group_rates: Dict[str, float] = {}

        # Current operation
        self._current_file: str = ''
        self._current_op: str = ''
        self._current_feature: str = ''   # sub-step within an operation
        self._progress_done: int = 0
        self._progress_total: int = 0
        self._progress_rate: float = 0.0
        self._progress_last_t: float = 0.0
        self._progress_last_done: int = 0

        # Live crop-progress source (dict reference from src/pipeline.py)
        self._crop_stats_dict: Optional[dict] = None
        self._crop_total: int = 0

        # Pipeline metadata
        self._config_name: str = ''
        self._log_path: str = 'pipeline.log'
        self._start_time: float = time.time()
        self._tracks_total: int = 0
        self._suspended: bool = False

        # Keyboard / graceful shutdown
        self._old_termios = None   # saved terminal settings restored on stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, config_name: str = '', log_path: Optional[Path] = None):
        """Start the TUI and redirect logging to *log_path* (overwritten each run)."""
        global _tui_active
        if not RICH_AVAILABLE:
            logger.warning('rich not installed — TUI disabled (pip install rich)')
            return

        self._config_name = config_name
        self._start_time = time.time()
        lp = str(log_path) if log_path else 'pipeline.log'
        self._log_path = lp

        # ── Logging: remove StreamHandlers, add FileHandler + TUIHandler ──
        root = logging.getLogger()

        file_handler = logging.FileHandler(lp, mode='w', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S')
        )
        file_handler.setLevel(logging.DEBUG)

        tui_handler = TUILogHandler(self._log_deque)
        tui_handler.setLevel(logging.INFO)

        for h in root.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)
        root.addHandler(file_handler)
        root.addHandler(tui_handler)

        # ── Rich Live (full-screen alternate buffer) ──────────────────────
        self._console = Console()
        self._live = Live(
            self._render(),
            console=self._console,
            screen=True,
            refresh_per_second=4,
        )
        self._live.start()

        # Put stdin in cbreak mode so single keypresses are readable immediately.
        # Do this AFTER Live.start() so we save whatever mode Rich left it in.
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            self._old_termios = None   # not a real tty (piped input, etc.)

        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name='PipelineUI-refresh'
        )
        self._refresh_thread.start()

        self._active = True
        _tui_active = True
        global _current_ui
        _current_ui = self

    def suspend(self):
        """Temporarily stop the Live display so the normal terminal is visible."""
        if not self._active or self._suspended:
            return
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=1.5)
        self._restore_terminal()
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
        self._suspended = True

    def resume(self):
        """Restart the Live display after suspend()."""
        if not self._active or not self._suspended:
            return
        if self._live:
            try:
                self._live.start()
            except Exception:
                pass
        # Re-enter cbreak mode after returning to the alternate screen
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            self._old_termios = None
        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name='PipelineUI-refresh'
        )
        self._refresh_thread.start()
        self._suspended = False

    def _restore_terminal(self):
        """Restore terminal to the settings saved before cbreak mode."""
        if self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
            self._old_termios = None

    def stop(self):
        """Stop the TUI.  The caller should print a final summary afterwards."""
        global _tui_active, _current_ui
        if not self._active:
            return

        _tui_active = False
        _current_ui = None
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)

        self._restore_terminal()

        if self._live:
            try:
                self._live.update(self._render())
                self._live.stop()
            except Exception:
                pass

        self._active = False

    # ── State setters ─────────────────────────────────────────────────────────

    def set_stage(self, stage_id: str, status: str):
        """Update a stage status.  status: 'running' | 'done' | 'skipped' | 'error'"""
        with self._lock:
            if status == 'running':
                self._stage_start[stage_id] = time.time()
                self._stages[stage_id] = ('running', 0.0)
            else:
                elapsed = time.time() - self._stage_start.get(stage_id, time.time())
                self._stages[stage_id] = (status, elapsed)

    def set_current(self, file: str = '', operation: str = '',
                    done: int = 0, total: int = 0, rate: float = 0.0,
                    feature: str = ''):
        """Update current file / operation and progress bar data.

        Args:
            feature: Optional sub-step label shown below the operation, e.g.
                     'Loudness', 'Spectral · Saturation', 'BPM (madmom)'.
        """
        with self._lock:
            if file:
                self._current_file = file
            if operation:
                self._current_op = operation
                self._current_feature = ''  # reset sub-step when operation changes
            if feature is not None:         # explicit '' clears it
                self._current_feature = feature
            self._progress_done = done
            self._progress_total = total
            if rate > 0:
                self._progress_rate = rate
            elif done > self._progress_last_done and self._progress_last_t > 0:
                dt = time.time() - self._progress_last_t
                if dt > 0.1:
                    self._progress_rate = (done - self._progress_last_done) / dt
            self._progress_last_t = time.time()
            self._progress_last_done = done

    def set_tracks_total(self, total: int):
        with self._lock:
            self._tracks_total = total

    def set_crop_progress_source(self, stats_dict: Optional[dict], total: int):
        """Point the refresh thread at a live crop-progress dict.
        Call with (None, 0) to detach after crop processing finishes."""
        with self._lock:
            self._crop_stats_dict = stats_dict
            self._crop_total = total

    def set_feature_config(self, features: Dict[str, bool],
                           disabled_stages: Optional[Set[str]] = None):
        """Set which features and stages are enabled/disabled in config.

        Call once after ui.start(), before the pipeline runs.

        Args:
            features: dict of feature_config_key → enabled_bool
                      e.g. {'loudness': True, 'timbral': False, ...}
            disabled_stages: set of stage_ids that are explicitly disabled in
                             the config (shown in red in the stages panel).
                             e.g. {'organize', 'track_analysis'}
        """
        with self._lock:
            self._feature_enabled = dict(features)
            self._disabled_stages = set(disabled_stages or ())

    def set_feature_rate(self, group_label: str, rate: float):
        """Manually set a per-feature-group throughput rate (items/s)."""
        with self._lock:
            self._group_rates[group_label] = rate

    # ── Refresh loop ──────────────────────────────────────────────────────────

    def _poll_keyboard(self):
        """Check for a graceful-shutdown keypress ('s') without blocking."""
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                ch = os.read(sys.stdin.fileno(), 1).decode('utf-8', errors='ignore')
                if ch.lower() == 's':
                    from core.graceful_shutdown import shutdown_requested
                    shutdown_requested.set()
                    logger.info(">>> Graceful shutdown requested ('s' pressed). "
                                "Finishing current file…")
        except Exception:
            pass

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._live and self._live.is_started:
                    self._live.update(self._render())
            except Exception:
                pass
            self._poll_keyboard()
            time.sleep(0.25)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> 'Layout':
        # Snapshot state under lock
        with self._lock:
            elapsed = time.time() - self._start_time
            stages = dict(self._stages)
            feature_enabled = dict(self._feature_enabled)
            disabled_stages = set(self._disabled_stages)
            group_rates = dict(self._group_rates)
            current_file = self._current_file
            current_op = self._current_op
            current_feature = self._current_feature
            prog_done = self._progress_done
            prog_total = self._progress_total
            prog_rate = self._progress_rate
            log_msgs = list(self._log_deque)
            config_name = self._config_name
            log_path = self._log_path
            tracks_total = self._tracks_total
            crop_stats = self._crop_stats_dict
            crop_total = self._crop_total

        # ── Derive crop progress ───────────────────────────────────────────
        # Once crop_stats is set (crop analysis phase), always use crop counts
        # so the features panel shows crop progress, not leftover track counts.
        if crop_stats is not None and crop_total > 0:
            prog_done  = crop_stats.get('crops_processed', 0)
            prog_total = crop_total

        # ── Derive stats from PipelineStats ───────────────────────────────
        tracks_processed = 0
        crops_analyzed = 0
        metadata_found = 0
        error_count = 0

        if self._stats is not None:
            tracks_processed = (
                getattr(self._stats, 'tracks_analyzed', 0)
                + getattr(self._stats, 'tracks_separated', 0)
            )
            crops_analyzed = getattr(self._stats, 'crops_analyzed', 0)
            # Also reflect crops being *created* (stage 3) before analysis begins
            crops_analyzed = max(crops_analyzed, getattr(self._stats, 'crops_created', 0))
            if crop_stats:
                # Use per-pass counter so the display resets at each pass boundary
                # and never exceeds the total (which counts unique crops, not pass × crops).
                pass_val = crop_stats.get('pass_crops_processed', 0)
                cumul_val = crop_stats.get('crops_processed', 0)
                crops_analyzed = max(crops_analyzed, pass_val if pass_val > 0 else cumul_val)
            metadata_found = getattr(self._stats, 'tracks_metadata_found', 0)
            error_count = len(getattr(self._stats, 'errors', []))

            # Pull per-feature rates from operation_timing
            op_timing = getattr(self._stats, 'operation_timing', {})
            for op_key, group_label in _OP_TO_GROUP_LABEL.items():
                timing = op_timing.get(op_key)
                if timing and timing.items_per_second > 0:
                    if group_label not in group_rates:
                        group_rates[group_label] = timing.items_per_second

        # ── Feature coverage (%) from pipeline's PASS 1 pre-scan ──────────
        feature_coverage: Dict[str, float] = {}
        active_crops: List[str] = []
        pass_rate: float = 0.0
        if crop_stats is not None:
            feature_coverage = crop_stats.get('feature_coverage', {})
            active_crops = crop_stats.get('active_crops', [])
            # Prefer crop-pipeline's active_feature over the track-level one
            crop_active_feat = crop_stats.get('active_feature', '')
            if crop_active_feat:
                current_feature = crop_active_feat
            pass_rate = crop_stats.get('pass_rate', 0.0)
            pass_rtf = crop_stats.get('pass_rtf', 0.0)
            last_file_rtf = crop_stats.get('last_file_rtf', 0.0)

        # ── Assemble layout ────────────────────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(name='header', size=3),
            Layout(name='body'),
            Layout(name='footer', size=1),
        )
        layout['body'].split_row(
            Layout(name='left', ratio=2),
            Layout(name='right', ratio=3),
        )
        layout['left'].split_column(
            Layout(name='stages', ratio=1),   # fixed-ish: 10 stages, smaller share
            Layout(name='features', ratio=2), # 13 feature rows — needs more room
        )
        layout['right'].split_column(
            Layout(name='current', ratio=3),
            Layout(name='stats', size=8),     # fixed: 4 data rows + border = 6 min
            Layout(name='log', ratio=5),      # taller log
        )

        layout['header'].update(
            self._render_header(config_name, elapsed, log_path))
        layout['stages'].update(
            self._render_stages(stages, disabled_stages))
        layout['features'].update(
            self._render_features(stages, feature_enabled, disabled_stages,
                                   prog_done, prog_total, prog_rate, group_rates,
                                   feature_coverage, current_feature, pass_rtf))
        layout['current'].update(
            self._render_current(current_file, current_op, current_feature,
                                  prog_done, prog_total, prog_rate,
                                  active_crops, pass_rtf, last_file_rtf))
        layout['stats'].update(
            self._render_stats(tracks_processed, tracks_total,
                                crops_analyzed, crop_total, metadata_found, error_count))
        layout['log'].update(
            self._render_log(log_msgs))
        layout['footer'].update(
            self._render_footer(log_path))
        return layout

    # ── Panel renderers ────────────────────────────────────────────────────────

    def _render_header(self, config_name: str, elapsed: float,
                        log_path: str) -> 'Panel':
        h = int(elapsed) // 3600
        m = int(elapsed) % 3600 // 60
        s = int(elapsed) % 60
        t = Text(justify='center')
        t.append('  MIR Feature Extraction Pipeline  ', style='bold green')
        if config_name:
            t.append(f'  {config_name}  ', style='dim cyan')
        t.append(f'  {h:02d}:{m:02d}:{s:02d}  ', style='bold cyan')
        return BevelPanel(t, padding=(0, 1))

    def _render_stages(self, stages: Dict[str, Tuple[str, float]],
                        disabled_stages: Set[str]) -> 'Panel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(width=2)
        table.add_column(min_width=20)
        table.add_column(width=7, justify='right')

        icon_map = {
            'pending': ('·', 'dim',       'dim'),
            'running': ('▶', 'bold yellow','bold yellow'),
            'done':    ('✓', 'bold green', 'dim green'),
            'skipped': ('─', 'dim',       'dim'),
            'error':   ('✗', 'bold red',  'bold red'),
        }

        for stage_id, stage_name in STAGES:
            status, elapsed = stages.get(stage_id, ('pending', 0.0))

            # Config-disabled stages get a distinct ✗ treatment
            if stage_id in disabled_stages and status in ('pending', 'skipped'):
                icon, icon_style, name_style = '✗', 'dim red', 'dim red'
                elapsed_str = 'disabled'
            else:
                icon, icon_style, name_style = icon_map.get(
                    status, ('·', 'dim', 'dim'))
                if elapsed > 0:
                    elapsed_str = (f'{int(elapsed)//60}:{int(elapsed)%60:02d}'
                                   if elapsed >= 60 else f'{elapsed:.0f}s')
                else:
                    elapsed_str = ''

            table.add_row(
                Text(icon, style=icon_style),
                Text(stage_name, style=name_style),
                Text(elapsed_str, style='dim red' if elapsed_str == 'disabled' else 'dim'),
            )

        return BevelPanel(table, title='[bold blue]Pipeline Stages[/]',
                          padding=(0, 1))

    def _render_features(self, stages: Dict[str, Tuple[str, float]],
                          feature_enabled: Dict[str, bool],
                          disabled_stages: Set[str],
                          prog_done: int, prog_total: int, prog_rate: float,
                          group_rates: Dict[str, float],
                          feature_coverage: Optional[Dict[str, float]] = None,
                          active_feature: str = '',
                          pass_rtf: float = 0.0) -> 'BevelPanel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(width=2)           # icon
        table.add_column(min_width=9)       # group name
        table.add_column(width=11, justify='right')  # N/M count
        table.add_column(min_width=7, justify='right')  # @ rate/s or status label

        active_stages = {sid for sid, (st, _) in stages.items() if st == 'running'}
        done_stages   = {sid for sid, (st, _) in stages.items() if st == 'done'}
        skipped_stages = {sid for sid, (st, _) in stages.items()
                          if st in ('skipped', 'pending')} | disabled_stages

        for group_key, label, config_keys, stage_ids in FEATURE_GROUPS:
            pct = (feature_coverage or {}).get(label)

            # Is the group enabled? (any of its config keys enabled)
            enabled = any(feature_enabled.get(k, True) for k in config_keys)

            if not enabled:
                lbl_t = Text(overflow='ellipsis', no_wrap=True)
                lbl_t.append(label, style='dim red')
                if pct is not None:
                    lbl_t.append(f' {pct:.0%}', style='dim')
                table.add_row(
                    Text('✗', style='dim red'),
                    lbl_t,
                    Text('', style=''),
                    Text('disabled', style='dim red'),
                )
                continue

            # Determine group status from relevant stages
            relevant_running = stage_ids & active_stages
            all_relevant_done = stage_ids and not (stage_ids - done_stages)
            stage_disabled = bool(stage_ids & disabled_stages) and not (stage_ids - disabled_stages)

            # During crop analysis, track-only feature groups should never show ▶.
            # If crop_analysis is active it means first_features has at minimum been
            # attempted; any remaining 'running' appearance is a TUI refresh artefact.
            if 'crop_analysis' in active_stages and 'crop_analysis' not in stage_ids:
                relevant_running = set()

            if relevant_running:
                # Currently processing — show progress + per-group rate if known
                icon, icon_style, name_style = '▶', 'bold yellow', 'yellow'
                count_str = f'{prog_done:,}/{prog_total:,}' if prog_total > 0 else ''
                # Use the group's own measured rate.
                # Fall back to pass_rate only for the currently-active feature group
                # (matched by checking if the label appears in active_feature string).
                # This prevents all groups showing the same global throughput.
                is_active_group = (active_feature and
                                   label.lower() in active_feature.lower())
                rtf = pass_rtf if is_active_group else group_rates.get(label, 0.0)
                if rtf == 0.0 and not active_feature and len(group_rates) == 0:
                    # Nothing active yet — fall back to global items/s as placeholder
                    rtf = prog_rate
                if rtf >= 100:
                    rate_str = f'@ {rtf:,.0f}s/s'
                elif rtf >= 10:
                    rate_str = f'@ {rtf:.1f}s/s'
                elif rtf > 0:
                    rate_str = f'@ {rtf:.2f}s/s'
                else:
                    rate_str = ''
                rate_style = 'cyan'
            elif all_relevant_done:
                icon, icon_style, name_style = '✓', 'bold green', 'dim green'
                count_str = ''
                # Badge features that only ran on tracks when crops are now running
                if 'crop_analysis' in active_stages and 'crop_analysis' not in stage_ids:
                    rate_str = 'track'
                    rate_style = 'dim green'
                else:
                    rate_str = ''
                    rate_style = ''
            elif stage_disabled:
                # The whole stage this feature runs in is config-disabled
                icon, icon_style, name_style = '─', 'dim red', 'dim red'
                count_str = ''
                rate_str = 'stage off'
                rate_style = 'dim red'
            else:
                icon, icon_style, name_style = '·', 'dim', 'dim'
                count_str = ''
                rate_str = ''
                rate_style = ''

            # Build label cell: main name + optional coverage %
            # Sub-keys shown as a dim suffix on the same line (no wrapping rows)
            lbl_t = Text(overflow='ellipsis', no_wrap=True)
            lbl_t.append(label, style=name_style)
            if pct is not None:
                lbl_t.append(f' {pct:.0%}', style='dim')
            if len(config_keys) > 1:
                short = ' · '.join(_KEY_SHORT.get(k, k) for k in config_keys)
                lbl_t.append(f'  {short}', style='dim')

            table.add_row(
                Text(icon, style=icon_style),
                lbl_t,
                Text(count_str, style='dim'),
                Text(rate_str, style=rate_style),
            )

        return BevelPanel(table, title='[bold blue]Features[/]',
                          padding=(0, 1))

    def _render_current(self, current_file: str, current_op: str,
                         current_feature: str,
                         done: int, total: int, rate: float,
                         active_crops: Optional[List[str]] = None,
                         pass_rtf: float = 0.0,
                         last_file_rtf: float = 0.0) -> 'Panel':
        lines = []

        if active_crops:
            # Multi-worker mode: show all in-flight crop stems
            lines.append(Text(current_op or 'Processing', style='cyan'))
            if current_feature:
                feat_t = Text(no_wrap=True)
                feat_t.append('  ↳ ', style='dim')
                feat_t.append(current_feature, style='bold white')
                lines.append(feat_t)
            for stem in active_crops:
                fn = stem if len(stem) <= 52 else '…' + stem[-51:]
                lines.append(Text(fn, style='bold yellow', no_wrap=True))
        else:
            fn = current_file or '—'
            if len(fn) > 52:
                fn = '…' + fn[-51:]
            lines.append(Text(fn, style='bold yellow', overflow='ellipsis', no_wrap=True))
            if current_op:
                op_t = Text(no_wrap=True)
                op_t.append(current_op, style='cyan')
                if current_feature:
                    op_t.append('  ↳  ', style='dim')
                    op_t.append(current_feature, style='bold white')
                lines.append(op_t)

        lines.append(Text(''))

        if total > 0:
            pct = min(1.0, done / total)
            filled = int(pct * 24)
            bar = '█' * filled + '░' * (24 - filled)
            # Show cumulative s/s next to the bar; fall back to items/s before first crop
            if pass_rtf >= 100:
                bar_rate = f'  {pass_rtf:,.0f}s/s'
            elif pass_rtf >= 10:
                bar_rate = f'  {pass_rtf:.1f}s/s'
            elif pass_rtf > 0:
                bar_rate = f'  {pass_rtf:.2f}s/s'
            elif rate > 0:
                bar_rate = f'  {rate:.1f} it/s'
            else:
                bar_rate = ''
            lines.append(Text(f'{bar}  {pct * 100:.0f}%{bar_rate}', style='green'))

            if rate > 0 and done < total:
                remaining = (total - done) / rate
                rh = int(remaining) // 3600
                rm = int(remaining) % 3600 // 60
                rs = int(remaining) % 60
                lines.append(Text(f'ETA  {rh:02d}:{rm:02d}:{rs:02d}', style='dim'))

            lines.append(Text(f'{done:,} / {total:,}', style='dim'))

            # Per-file RTF (sequential passes only; 0 in parallel/batch passes)
            if last_file_rtf > 0:
                if last_file_rtf >= 100:
                    lf_str = f'{last_file_rtf:,.0f}s/s'
                elif last_file_rtf >= 10:
                    lf_str = f'{last_file_rtf:.1f}s/s'
                else:
                    lf_str = f'{last_file_rtf:.2f}s/s'
                rtf_t = Text(no_wrap=True)
                rtf_t.append('Last  ', style='dim')
                rtf_t.append(lf_str, style='cyan')
                if pass_rtf > 0:
                    if pass_rtf >= 100:
                        cum_str = f'{pass_rtf:,.0f}s/s'
                    elif pass_rtf >= 10:
                        cum_str = f'{pass_rtf:.1f}s/s'
                    else:
                        cum_str = f'{pass_rtf:.2f}s/s'
                    rtf_t.append('   Avg  ', style='dim')
                    rtf_t.append(cum_str, style='cyan')
                lines.append(rtf_t)

        return BevelPanel(Group(*lines), title='[bold blue]Current[/]',
                          padding=(0, 1))

    def _render_stats(self, tracks_processed: int, tracks_total: int,
                       crops_analyzed: int, crops_total: int,
                       metadata_found: int, error_count: int) -> 'Panel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(min_width=12)
        table.add_column(justify='right')

        if tracks_processed > 0:
            t_str = (f'{tracks_processed:,} / {tracks_total:,}'
                     if tracks_total > 0 else f'{tracks_processed:,}')
        elif tracks_total > 0:
            t_str = f'{tracks_total:,}'
        else:
            t_str = '—'
        table.add_row(Text('Tracks', style='dim'), Text(t_str, style='white'))

        if metadata_found > 0:
            m_str = (f'{metadata_found:,} / {tracks_total:,}'
                     if tracks_total > 0 else f'{metadata_found:,}')
        else:
            m_str = f'{metadata_found:,}'
        table.add_row(Text('Metadata', style='dim'), Text(m_str, style='white'))

        if crops_analyzed > 0:
            c_str = (f'{crops_analyzed:,} / {crops_total:,}'
                     if crops_total > 0 else f'{crops_analyzed:,}')
        elif crops_total > 0:
            c_str = f'{crops_total:,}'
        else:
            c_str = '—'
        table.add_row(Text('Crops', style='dim'), Text(c_str, style='white'))

        err_style = 'bold red' if error_count > 0 else 'dim'
        table.add_row(Text('Errors', style='dim'),
                       Text(f'{error_count:,}', style=err_style))

        return BevelPanel(table, title='[bold blue]Statistics[/]',
                          padding=(0, 1))

    def _render_log(self, log_msgs: List[Tuple]) -> 'Panel':
        lines = []
        level_styles = {
            'WARNING': 'yellow',
            'ERROR': 'bold red',
            'CRITICAL': 'bold red',
        }
        for ts, level, msg in log_msgs[-12:]:
            style = level_styles.get(level, '')
            t = Text(overflow='ellipsis', no_wrap=True)
            t.append(f'{ts} ', style='dim')
            if style:
                t.append(f'[{level}] ', style=style)
            t.append(msg, style=style)
            lines.append(t)

        while len(lines) < 3:
            lines.append(Text(''))

        return BevelPanel(Group(*lines), title='[bold blue]Recent Log[/]',
                          padding=(0, 1))

    def _render_footer(self, log_path: str) -> 'Text':
        t = Text(justify='center')
        t.append(' [s] ', style='bold on dark_green')
        t.append(' graceful stop ', style='dim')
        t.append('   │   ', style='dim')
        t.append(f'Log → {log_path}', style='dim cyan')
        return t

    @property
    def is_active(self) -> bool:
        return self._active
