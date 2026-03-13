"""
logger.py — TensorBoard + CSV training logger.
"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Dict, Optional


class TrainingLogger:
    """
    Logs training metrics to:
        - TensorBoard (if tensorboard package is installed)
        - CSV file (always)
    """

    def __init__(self, config: dict, agent_type: str) -> None:
        paths = config["paths"]
        logs_dir = paths["logs_dir"]
        run_name = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(logs_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # CSV
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None

        # TensorBoard
        self._writer = None
        if config["logging"].get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self.run_dir)
            except ImportError:
                print("[Logger] TensorBoard not available — CSV only.")

        print(f"[Logger] Logging to {self.run_dir}")

    def log_episode(self, episode: int, stats: Dict) -> None:
        # Initialise CSV writer on first call
        if self._csv_writer is None:
            fieldnames = ["episode"] + list(stats.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()

        row = {"episode": episode, **stats}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._writer is not None:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(f"train/{k}", v, episode)

    def close(self) -> None:
        self._csv_file.close()
        if self._writer is not None:
            self._writer.close()
