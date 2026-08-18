# src/agentic_vmm/engine/vram_monitor.py
"""
Real-time VRAM telemetry via nvidia-smi.

Provides both one-shot snapshots and a background sampling thread
that records VRAM usage over time for post-hoc analysis and live display.
"""

from __future__ import annotations

import subprocess
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VRAMSnapshot:
    """Single VRAM measurement."""
    timestamp: float          # time.time()
    used_mb: int
    total_mb: int
    free_mb: int
    gpu_util_pct: int         # GPU utilization %
    label: str = ""           # optional annotation (e.g. "after branch")

    @property
    def used_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb else 0.0

    def __repr__(self) -> str:
        return (
            f"VRAM: {self.used_mb}/{self.total_mb} MB "
            f"({self.used_pct:.1f}%) GPU:{self.gpu_util_pct}%"
            f"{f' [{self.label}]' if self.label else ''}"
        )


def vram_snapshot(label: str = "") -> Optional[VRAMSnapshot]:
    """Take a single VRAM snapshot via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        if len(parts) < 4:
            return None
        return VRAMSnapshot(
            timestamp=time.time(),
            used_mb=int(parts[0].strip()),
            total_mb=int(parts[1].strip()),
            free_mb=int(parts[2].strip()),
            gpu_util_pct=int(parts[3].strip()),
            label=label,
        )
    except Exception as e:
        logger.debug("vram_snapshot failed: %s", e)
        return None


class VRAMRecorder:
    """
    Background thread that samples VRAM at a fixed interval.
    Use mark(label) to annotate key moments (branch, reset, etc.).
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval = interval_sec
        self.samples: list[VRAMSnapshot] = []
        self._marks: list[VRAMSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def mark(self, label: str) -> Optional[VRAMSnapshot]:
        """Take an immediate labeled snapshot (for key events)."""
        snap = vram_snapshot(label)
        if snap:
            self._marks.append(snap)
        return snap

    @property
    def marks(self) -> list[VRAMSnapshot]:
        return list(self._marks)

    def peak_usage(self) -> int:
        """Peak VRAM used_mb across all samples + marks."""
        all_snaps = self.samples + self._marks
        return max((s.used_mb for s in all_snaps), default=0)

    def min_usage(self) -> int:
        all_snaps = self.samples + self._marks
        return min((s.used_mb for s in all_snaps), default=0)

    def delta_range(self) -> int:
        """Peak - Min across the entire recording."""
        return self.peak_usage() - self.min_usage()

    def summary_table(self) -> list[dict]:
        """Return marks as a list of dicts for tabular display."""
        return [
            {
                "label": m.label,
                "used_mb": m.used_mb,
                "total_mb": m.total_mb,
                "used_pct": f"{m.used_pct:.1f}%",
                "gpu_util": f"{m.gpu_util_pct}%",
            }
            for m in self._marks
        ]

    def _sample_loop(self) -> None:
        while self._running:
            snap = vram_snapshot()
            if snap:
                self.samples.append(snap)
            time.sleep(self.interval)
