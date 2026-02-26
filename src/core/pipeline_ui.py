"""
Pipeline TUI — btop-like static dashboard for MIR pipeline runs.

Redirects all log output to a rotating log file and displays a Rich full-screen
dashboard with: pipeline stage progress, current operation + progress bar,
feature throughput table, statistics panel, and a scrolling recent-log panel.

Usage:
    ui = PipelineUI(stats=pipeline.stats)
    ui.start(config_name='master_pipeline.yaml', log_path=Path('pipeline.log'))
    ui.set_stage('organize', 'running')
    ui.set_stage('organize', 'done')
    ui.set_current(file='Artist - Track', operation='Stem separation',
                   done=5, total=200)
    ui.stop()  # prints final summary to stdout

Use --no-ui to skip the TUI and fall back to coloured stdout logging.
"""

import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rich_box
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

# Mapping from PipelineStats.operation_timing keys → display names for throughput table
OP_DISPLAY: Dict[str, str] = {
    'bs_roformer':   'BS-RoFormer',
    'demucs':        'Demucs',
    'beats':         'Beat / BPM',
    'onsets':        'Onsets',
    'loudness':      'Loudness',
    'spectral':      'Spectral',
    'timbral':       'Timbral',
    'essentia':      'Essentia',
    'flamingo':      'Flamingo',
    'granite':       'Granite',
    'metadata':      'Metadata',
    'crop_features': 'Crop Features',
    'crop_demucs':   'Crop Stems',
}


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


# ── Module-level TUI-active flag (read by ProgressBar) ─────────────────────────

_tui_active = False

def is_tui_active() -> bool:
    """Returns True while the TUI dashboard is running (ProgressBar uses this
    to suppress its own inline output)."""
    return _tui_active


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

        # Current operation
        self._current_file: str = ''
        self._current_op: str = ''
        self._progress_done: int = 0
        self._progress_total: int = 0
        self._progress_rate: float = 0.0
        self._progress_last_t: float = 0.0
        self._progress_last_done: int = 0

        # Live crop-progress source (dict reference from src/pipeline.py)
        self._crop_stats_dict: Optional[dict] = None
        self._crop_total: int = 0

        # Extra / manual feature rates
        self._feature_rates: Dict[str, float] = {}

        # Pipeline metadata
        self._config_name: str = ''
        self._log_path: str = 'pipeline.log'
        self._start_time: float = time.time()
        self._tracks_total: int = 0

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

        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name='PipelineUI-refresh'
        )
        self._refresh_thread.start()

        self._active = True
        _tui_active = True

    def stop(self):
        """Stop the TUI.  The caller should print a final summary afterwards."""
        global _tui_active
        if not self._active:
            return

        _tui_active = False
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)

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
                    done: int = 0, total: int = 0, rate: float = 0.0):
        """Update current file / operation and progress bar data."""
        with self._lock:
            if file:
                self._current_file = file
            if operation:
                self._current_op = operation
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

    def set_feature_rate(self, feature: str, rate: float):
        """Manually set a throughput rate (overrides operation_timing derivation)."""
        with self._lock:
            self._feature_rates[feature] = rate

    # ── Refresh loop ──────────────────────────────────────────────────────────

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._live and self._live.is_started:
                    self._live.update(self._render())
            except Exception:
                pass
            time.sleep(0.25)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> 'Layout':
        # Snapshot state under lock — minimize lock hold time
        with self._lock:
            elapsed = time.time() - self._start_time
            stages = dict(self._stages)
            current_file = self._current_file
            current_op = self._current_op
            prog_done = self._progress_done
            prog_total = self._progress_total
            prog_rate = self._progress_rate
            feat_rates = dict(self._feature_rates)
            log_msgs = list(self._log_deque)
            config_name = self._config_name
            log_path = self._log_path
            tracks_total = self._tracks_total
            crop_stats = self._crop_stats_dict
            crop_total = self._crop_total

        # ── Derive crop progress ───────────────────────────────────────────
        if crop_stats is not None and crop_total > 0:
            crops_done = crop_stats.get('crops_processed', 0)
            if crops_done >= prog_done:
                prog_done = crops_done
                prog_total = crop_total

        # ── Derive stats + throughput from PipelineStats ───────────────────
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
            if crop_stats:
                crops_analyzed = max(crops_analyzed, crop_stats.get('crops_processed', 0))
            metadata_found = getattr(self._stats, 'tracks_metadata_found', 0)
            error_count = len(getattr(self._stats, 'errors', []))

            # Build throughput rates from completed operations
            op_timing = getattr(self._stats, 'operation_timing', {})
            for op_key, display_name in OP_DISPLAY.items():
                timing = op_timing.get(op_key)
                if timing and timing.items_per_second > 0:
                    if display_name not in feat_rates:
                        feat_rates[display_name] = timing.items_per_second

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
            Layout(name='stages', ratio=3),
            Layout(name='throughput', ratio=2),
        )
        layout['right'].split_column(
            Layout(name='current', ratio=2),
            Layout(name='stats', ratio=2),
            Layout(name='log', ratio=3),
        )

        layout['header'].update(
            self._render_header(config_name, elapsed, log_path))
        layout['stages'].update(
            self._render_stages(stages))
        layout['throughput'].update(
            self._render_throughput(feat_rates))
        layout['current'].update(
            self._render_current(current_file, current_op,
                                  prog_done, prog_total, prog_rate))
        layout['stats'].update(
            self._render_stats(tracks_processed, tracks_total,
                                crops_analyzed, metadata_found, error_count))
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
        return Panel(t, style='green', box=rich_box.HEAVY_HEAD, padding=(0, 1))

    def _render_stages(self, stages: Dict[str, Tuple[str, float]]) -> 'Panel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(width=2)
        table.add_column(min_width=20)
        table.add_column(width=7, justify='right')

        icons = {
            'pending': ('·', 'dim'),
            'running': ('▶', 'bold yellow'),
            'done':    ('✓', 'bold green'),
            'skipped': ('─', 'dim'),
            'error':   ('✗', 'bold red'),
        }

        for stage_id, stage_name in STAGES:
            status, elapsed = stages.get(stage_id, ('pending', 0.0))
            icon, icon_style = icons.get(status, ('·', 'dim'))
            name_style = icon_style if status not in ('pending', 'skipped') else 'dim'

            if elapsed > 0:
                elapsed_str = (f'{int(elapsed)//60}:{int(elapsed)%60:02d}'
                               if elapsed >= 60 else f'{elapsed:.0f}s')
            else:
                elapsed_str = ''

            table.add_row(
                Text(icon, style=icon_style),
                Text(stage_name, style=name_style),
                Text(elapsed_str, style='dim'),
            )

        return Panel(table, title='[bold blue]Pipeline Stages[/]',
                     border_style='blue', box=rich_box.ROUNDED, padding=(0, 1))

    def _render_throughput(self, feat_rates: Dict[str, float]) -> 'Panel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(min_width=14)
        table.add_column(width=6, justify='right')
        table.add_column(min_width=10)

        active = {k: v for k, v in feat_rates.items() if v > 0}
        if not active:
            table.add_row(Text('No data yet', style='dim'), Text(''), Text(''))
        else:
            max_rate = max(active.values(), default=1.0)
            for name, rate in sorted(active.items(), key=lambda x: -x[1]):
                bar_len = max(1, int((rate / max_rate) * 10))
                bar = '█' * bar_len + '░' * (10 - bar_len)
                table.add_row(
                    Text(name, style='dim'),
                    Text(f'{rate:.1f}', style='cyan'),
                    Text(bar, style='green'),
                )

        return Panel(table, title='[bold blue]Throughput[/] [dim](it/s)[/]',
                     border_style='blue', box=rich_box.ROUNDED, padding=(0, 1))

    def _render_current(self, current_file: str, current_op: str,
                         done: int, total: int, rate: float) -> 'Panel':
        lines = []

        fn = current_file or '—'
        if len(fn) > 52:
            fn = '…' + fn[-51:]
        lines.append(Text(fn, style='bold yellow', overflow='ellipsis', no_wrap=True))

        if current_op:
            lines.append(Text(current_op, style='cyan'))

        lines.append(Text(''))

        if total > 0:
            pct = min(1.0, done / total)
            filled = int(pct * 24)
            bar = '█' * filled + '░' * (24 - filled)
            rate_str = f'  {rate:.1f} it/s' if rate > 0 else ''
            lines.append(Text(f'{bar}  {pct * 100:.0f}%{rate_str}', style='green'))

            if rate > 0 and done < total:
                remaining = (total - done) / rate
                rh = int(remaining) // 3600
                rm = int(remaining) % 3600 // 60
                rs = int(remaining) % 60
                lines.append(Text(f'ETA  {rh:02d}:{rm:02d}:{rs:02d}', style='dim'))

            lines.append(Text(f'{done:,} / {total:,}', style='dim'))

        return Panel(Group(*lines), title='[bold blue]Current[/]',
                     border_style='blue', box=rich_box.ROUNDED, padding=(0, 1))

    def _render_stats(self, tracks_processed: int, tracks_total: int,
                       crops_analyzed: int, metadata_found: int,
                       error_count: int) -> 'Panel':
        table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
        table.add_column(min_width=12)
        table.add_column(justify='right')

        t_str = (f'{tracks_processed:,} / {tracks_total:,}'
                 if tracks_total > 0 else f'{tracks_processed:,}')
        table.add_row(Text('Tracks', style='dim'), Text(t_str, style='white'))
        table.add_row(Text('Crops', style='dim'), Text(f'{crops_analyzed:,}', style='white'))
        table.add_row(Text('Metadata', style='dim'), Text(f'{metadata_found:,}', style='white'))
        err_style = 'bold red' if error_count > 0 else 'dim'
        table.add_row(Text('Errors', style='dim'),
                       Text(f'{error_count:,}', style=err_style))

        return Panel(table, title='[bold blue]Statistics[/]',
                     border_style='blue', box=rich_box.ROUNDED, padding=(0, 1))

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

        return Panel(Group(*lines), title='[bold blue]Recent Log[/]',
                     border_style='blue', box=rich_box.ROUNDED, padding=(0, 1))

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
