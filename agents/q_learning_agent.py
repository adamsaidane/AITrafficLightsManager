"""
q_learning_agent.py — Tabular Q-Learning agent.

State discretisation:
    5 pressure levels (NS vs EW vehicle counts)  ×  num_phases  = 40 states
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Classic tabular Q-Learning with improved ε-greedy exploration.

    Kept as a baseline for comparison with deep RL agents.
    State is deliberately small: pressure_level × current_phase.
    """

    def __init__(self, config: dict) -> None:
        q_cfg = config["q_learning"]
        sim_cfg = config["simulation"]

        self.num_phases: int = len(sim_cfg["phases"])
        self.num_pressure: int = q_cfg["num_pressure_levels"]
        self.num_states: int = self.num_phases * self.num_pressure

        self.alpha: float = q_cfg["alpha"]
        self.gamma: float = q_cfg["gamma"]
        self.epsilon: float = q_cfg["epsilon_start"]
        self.epsilon_decay: float = q_cfg["epsilon_decay"]
        self.epsilon_min: float = q_cfg["epsilon_min"]

        self.lanes = sim_cfg["lanes"]

        # Q-table: (states × actions)
        self.q_table = np.zeros((self.num_states, self.num_phases))

    # ------------------------------------------------------------------ #

    def _pressure_level(self) -> int:
        """Discretise NS vs EW vehicle pressure into 5 buckets."""
        try:
            import traci
            existing = set(traci.lane.getIDList())
            ns = sum(traci.lane.getLastStepVehicleNumber(l)
                     for l in self.lanes if l.startswith(("N_", "S_")) and l in existing)
            ew = sum(traci.lane.getLastStepVehicleNumber(l)
                     for l in self.lanes if l.startswith(("E_", "W_")) and l in existing)
            diff = ns - ew
            if diff > 10:
                return 0
            elif diff > 3:
                return 1
            elif abs(diff) <= 3:
                return 2
            elif diff < -3:
                return 3
            else:
                return 4
        except Exception:
            return 2

    def state_index(self, phase: int) -> int:
        return self._pressure_level() * self.num_phases + phase

    # ------------------------------------------------------------------ #
    # BaseAgent interface
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy. state is an integer index for tabular agents."""
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_phases)
        q = self.q_table[state_idx]
        ties = np.flatnonzero(q == q.max())
        return int(np.random.choice(ties))

    def update(self, state, action, reward, next_state, done=False) -> dict:  # type: ignore[override]
        s = int(state) if np.isscalar(state) else int(state[0])
        ns = int(next_state) if np.isscalar(next_state) else int(next_state[0])
        target = reward + (0.0 if done else self.gamma * self.q_table[ns].max())
        self.q_table[s, action] += self.alpha * (target - self.q_table[s, action])
        return {"td_error": abs(target - self.q_table[s, action])}

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        data = {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.q_table = np.array(data["q_table"])
        self.epsilon = data["epsilon"]

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        q = self.q_table[state_idx]
        return int(np.argmax(q))
