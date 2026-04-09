"""
MIR Pipeline Feature Benchmark Collector
=========================================
Activated by --benchmark flag in master_pipeline.py / pipeline.py.

Design
------
* Workers are subprocesses — they can't mutate shared state.
  Timings are embedded in the worker's return dict under '_bench_timings'
  and accumulated by the main process after each future completes.
* GPU passes run in the main thread, so timing is added inline.
* A BenchmarkCollector instance holds per-feature lists of elapsed seconds
  and the audio duration of each crop measured.
* call .report() at the end to get a Markdown table.

Usage (programmatic)
--------------------
    from core.benchmark import BenchmarkCollector
    bc = BenchmarkCollector(n_cpu_workers=10)
    # ...add timings...
    md = bc.report(hardware_tag="Ryzen9+RX9070XT")
    Path("benchmark.md").write_text(md)
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class BenchmarkCollector:
    """Thread-safe accumulator of per-feature wall-clock measurements."""

    def __init__(self, n_cpu_workers: int = 1):
        self.n_cpu_workers = n_cpu_workers
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._audio_s: List[float] = []          # duration of each measured crop
        self._lock = threading.Lock()
        self._t_start = time.time()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_worker_result(self, result: dict) -> None:
        """Pull '_bench_timings' and '_bench_audio_s' out of a worker result."""
        timings  = result.get("_bench_timings")
        audio_s  = result.get("_bench_audio_s", 0.0)
        if not timings:
            return
        with self._lock:
            for k, v in timings.items():
                self._records[k].append(float(v))
            if audio_s > 0:
                self._audio_s.append(float(audio_s))

    def add(self, feature: str, wall_s: float, audio_s: float = 0.0) -> None:
        """Record a single timing (from main-process GPU passes)."""
        with self._lock:
            self._records[feature].append(float(wall_s))
            if audio_s > 0:
                self._audio_s.append(float(audio_s))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def n_crops(self, feature: Optional[str] = None) -> int:
        with self._lock:
            if feature:
                return len(self._records.get(feature, []))
            return max((len(v) for v in self._records.values()), default=0)

    def mean_audio_s(self) -> float:
        with self._lock:
            return float(np.mean(self._audio_s)) if self._audio_s else 11.4

    def stats(self, feature: str) -> Optional[dict]:
        with self._lock:
            vals = list(self._records.get(feature, []))
        if not vals:
            return None
        a = np.array(vals)
        return {
            "n":    len(a),
            "mean": float(np.mean(a)),
            "std":  float(np.std(a)),
            "p10":  float(np.percentile(a, 10)),
            "p90":  float(np.percentile(a, 90)),
            "sum":  float(a.sum()),
        }

    def features(self) -> List[str]:
        with self._lock:
            return list(self._records.keys())

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _rtf(self, mean_s: float) -> str:
        audio = self.mean_audio_s()
        if mean_s <= 0:
            return "—"
        rtf = audio / mean_s
        if rtf >= 1000:
            return f"{rtf/1000:.1f}kx"
        return f"{rtf:.0f}x"

    def section_table(
        self,
        pass_name: str,
        feature_order: List[str],
        *,
        notes: str = "",
    ) -> str:
        """Render one pass as a Markdown table section."""
        n_c = self.n_crops()
        audio = self.mean_audio_s()
        lines = [
            f"## {pass_name}",
            f"crops measured: {n_c}  ·  avg audio duration: {audio:.1f} s  ·  CPU workers: {self.n_cpu_workers}",
        ]
        if notes:
            lines.append(f"_{notes}_")
        lines.append("")

        # Collect rows and compute total serial cost
        rows = []
        serial_total = 0.0
        for feat in feature_order:
            s = self.stats(feat)
            if s is None:
                continue
            serial_total += s["mean"]
            rows.append((feat, s))

        header = "| Feature | N | Mean (s) | Std (s) | p10–p90 (s) | RTF | % serial |"
        sep    = "|---------|---|----------|---------|-------------|-----|----------|"
        lines += [header, sep]

        for feat, s in rows:
            pct = f"{100*s['mean']/serial_total:.1f}" if serial_total > 0 else "—"
            p10p90 = f"{s['p10']:.3f}–{s['p90']:.3f}"
            lines.append(
                f"| {feat} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} "
                f"| {p10p90} | {self._rtf(s['mean'])} | {pct}% |"
            )

        if rows:
            # Summary row
            lines.append(f"| **Total serial / crop** | | **{serial_total:.4f}** | | | "
                         f"**{self._rtf(serial_total)}** | 100% |")
            if self.n_cpu_workers > 1:
                est_throughput = (audio / serial_total) * self.n_cpu_workers
                lines.append(f"| *Est. throughput ({self.n_cpu_workers} workers)* | | | | | "
                              f"*~{est_throughput:.0f}x* | |")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _hw_tag() -> str:
    """Best-effort one-liner describing the host hardware."""
    cpu = platform.processor() or platform.machine()
    try:
        # Try to get CPU model from /proc/cpuinfo
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    gpu = "unknown GPU"
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if "Card series" in line or "GPU[" in line:
                gpu = line.strip()
                break
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True,
            )
            gpu = out.strip().splitlines()[0]
        except Exception:
            pass

    return f"{cpu}  ·  {gpu}"


# ---------------------------------------------------------------------------
# Full-report helper
# ---------------------------------------------------------------------------

def _combined_table(
    sections: List[tuple],
    n_cpu_workers: int,
) -> str:
    """
    Render all passes as a single flat Markdown table, one row per feature.

    Each element of `sections` is (pass_name, BenchmarkCollector, feature_order, notes).
    The RTF column uses each collector's own mean_audio_s() so CPU and GPU features
    are compared correctly against their respective sample sets.
    """
    # First pass: collect all rows and grand serial total
    all_rows = []   # (feature, stats_dict, audio_s_for_rtf, pass_label)
    serial_total = 0.0
    for pass_name, collector, feature_order, _notes in sections:
        audio_s = collector.mean_audio_s()
        for feat in feature_order:
            s = collector.stats(feat)
            if s is None:
                continue
            serial_total += s["mean"]
            all_rows.append((feat, s, audio_s, pass_name))

    if not all_rows:
        return ""

    n_cpu_crops = max(
        (collector.n_crops() for _, collector, fo, _ in sections
         if any(collector.stats(f) for f in fo[:3])),  # heuristic: first pass = CPU
        default=0,
    )

    def _rtf(mean_s: float, audio_s: float) -> str:
        if mean_s <= 0:
            return "—"
        rtf = audio_s / mean_s
        return f"{rtf/1000:.1f}kx" if rtf >= 1000 else f"{rtf:.0f}x"

    # Normalise by audio duration so numbers are "cost per audio-second" (s/s)
    # Equivalent to 1/RTF — independent of crop length.
    def _norm(wall_s: float, audio_s: float) -> float:
        return wall_s / audio_s if audio_s > 0 else 0.0

    # Serial total in normalised units (use first section's audio_s as reference)
    ref_audio_s = all_rows[0][2] if all_rows else 1.0
    serial_total_norm = sum(_norm(s["mean"], audio_s) for _, s, audio_s, _ in all_rows)

    header = "| Feature | N | Mean (s/s) | Std (s/s) | p10–p90 (s/s) | RTF | % total cost |"
    sep    = "|---------|---|------------|-----------|---------------|-----|--------------|"
    lines  = [header, sep]

    prev_pass = None
    for feat, s, audio_s, pass_label in all_rows:
        if pass_label != prev_pass:
            lines.append(f"| ***{pass_label}*** | | | | | | |")
            prev_pass = pass_label
        mean_n = _norm(s['mean'], audio_s)
        std_n  = _norm(s['std'],  audio_s)
        p10_n  = _norm(s['p10'],  audio_s)
        p90_n  = _norm(s['p90'],  audio_s)
        pct    = f"{100 * mean_n / serial_total_norm:.1f}" if serial_total_norm > 0 else "—"
        p10p90 = f"{p10_n:.4f}–{p90_n:.4f}"
        lines.append(
            f"| {feat} | {s['n']} | {mean_n:.4f} | {std_n:.4f} "
            f"| {p10p90} | {_rtf(s['mean'], audio_s)} | {pct}% |"
        )

    lines.append(f"| **Total serial / audio-s** | | **{serial_total_norm:.4f}** | | | | 100% |")
    if n_cpu_workers > 1:
        # Est. throughput based on CPU pass (dominant cost)
        cpu_total = sum(s["mean"] for _, s, _, pl in all_rows if "CPU" in pl or "PASS 1" in pl)
        cpu_audio = next(
            (c.mean_audio_s() for _, c, _, _ in sections),
            11.4,
        )
        if cpu_total > 0:
            est = (cpu_audio / cpu_total) * n_cpu_workers
            lines.append(
                f"| *Est. CPU throughput ({n_cpu_workers} workers)* | | | | | "
                f"*~{est:.0f}x* | |"
            )

    lines.append("")
    return "\n".join(lines)


def build_full_report(
    sections: List[tuple],          # [(pass_name, collector, feature_order, notes), ...]
    project_root: Path,
    n_cpu_workers: int,
) -> Path:
    """
    Build and save benchmark_<date>.md to project_root.

    Each element of `sections` is (pass_name, BenchmarkCollector, feature_order, notes).
    """
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    date = datetime.now().strftime("%Y-%m-%d")
    hw   = _hw_tag()

    # Metadata summary per section
    meta_lines = []
    for pass_name, collector, feature_order, notes in sections:
        n_c   = collector.n_crops()
        a_s   = collector.mean_audio_s()
        extra = f"  _{notes}_" if notes else ""
        meta_lines.append(f"- **{pass_name}**: {n_c} crops, avg {a_s:.1f} s audio{extra}")

    header = "\n".join([
        "# MIR Pipeline Feature Benchmark",
        "",
        f"Generated: {now}",
        f"Hardware:  {hw}",
        f"CPU workers: {n_cpu_workers}",
        "",
        *meta_lines,
        "",
        "---",
        "",
    ])

    body = _combined_table(sections, n_cpu_workers)

    out_path = project_root / f"benchmark_{date}.md"
    out_path.write_text(header + body, encoding="utf-8")
    return out_path
