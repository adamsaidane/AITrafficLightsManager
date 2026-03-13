"""
base_agent.py — Abstract base class for all RL agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseAgent(ABC):
    """All agents inherit from this and implement the core interface."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select an action (epsilon-greedy or stochastic)."""

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Update the agent's parameters. Returns a dict of training metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent from disk."""

    def get_greedy_action(self, state: np.ndarray) -> int:
        """Return the greedy (no-exploration) action. Useful at evaluation."""
        return self.select_action(state)
