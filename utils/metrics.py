"""
metrics.py — Episode metrics tracking with rolling statistics.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """Accumulates per-episode metrics and provides rolling-window stats."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self.history: Dict[str, List[float]] = collections.defaultdict(list)

    def record(self, episode: int, stats: Dict) -> None:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                self.history[k].append(float(v))

    def recent_avg(self, n: Optional[int] = None) -> Dict[str, float]:
        n = n or self.window
        return {k: float(np.mean(v[-n:])) for k, v in self.history.items() if v}

    def all(self) -> Dict[str, List[float]]:
        return dict(self.history)
