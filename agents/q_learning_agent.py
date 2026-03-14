"""
q_learning_agent.py — Tabular Q-Learning agent (improved state space).

State discretisation v2:
    NS pressure (3)  ×  EW pressure (3)  ×  current_phase (8)  =  72 states

    NS/EW pressure buckets: LOW (0-4 veh) | MED (5-14) | HIGH (15+)
    This gives the agent independent NS/EW resolution, fixing the case
    where equal-but-large NS and EW loads appeared identical to the old
    single-difference scheme.

Improvements vs v1:
    - 72 states instead of 40 → finer discrimination
    - Optimistic initialisation (q_table = +50) → forces early exploration
    - Visit-count adaptive alpha: α(s,a) = α₀ / (1 + n(s,a)^0.7)
      prevents over-fitting early visits while still converging
    - UCB1 tie-breaking instead of random tie-breaking
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime

from agents.base_agent import BaseAgent

# Pressure thresholds (vehicles per direction group)
_PRESSURE_BINS = [0, 5, 15]   # → LOW < 5, MED 5–14, HIGH ≥ 15
_N_PRESS = 3                   # number of pressure bins


def _bin(vehicles: int) -> int:
    if vehicles < _PRESSURE_BINS[1]:
        return 0   # LOW
    elif vehicles < _PRESSURE_BINS[2]:
        return 1   # MED
    return 2       # HIGH


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning with richer state space and adaptive learning rate.
    Kept as interpretable baseline; use DQN/PPO for best performance.
    """

    def __init__(self, config: dict) -> None:
        q_cfg = config["q_learning"]
        sim_cfg = config["simulation"]

        self.num_phases: int = len(sim_cfg["phases"])
        # State = NS_bin × EW_bin × phase  →  3×3×8 = 72
        self.num_states: int = _N_PRESS * _N_PRESS * self.num_phases

        self.alpha0: float = q_cfg["alpha"]
        self.gamma: float  = q_cfg["gamma"]
        self.epsilon: float       = q_cfg["epsilon_start"]
        self.epsilon_decay: float = q_cfg["epsilon_decay"]
        self.epsilon_min: float   = q_cfg["epsilon_min"]
        self.lanes = sim_cfg["lanes"]

        # Optimistic init: encourages trying every action at least once
        self.q_table     = np.full((self.num_states, self.num_phases), 50.0)
        # Visit counts for adaptive alpha
        self.visit_count = np.zeros((self.num_states, self.num_phases), dtype=np.int32)

    # ------------------------------------------------------------------ #
    # State computation
    # ------------------------------------------------------------------ #

    def _get_pressures(self):
        """Return (ns_vehicles, ew_vehicles) from TraCI."""
        try:
            import traci
            existing = set(traci.lane.getIDList())
            ns = sum(traci.lane.getLastStepVehicleNumber(l)
                     for l in self.lanes
                     if l.startswith(("N_", "S_")) and l in existing)
            ew = sum(traci.lane.getLastStepVehicleNumber(l)
                     for l in self.lanes
                     if l.startswith(("E_", "W_")) and l in existing)
            return ns, ew
        except Exception:
            return 0, 0

    def state_index(self, phase: int) -> int:
        ns, ew = self._get_pressures()
        ns_bin = _bin(ns)
        ew_bin = _bin(ew)
        return (ns_bin * _N_PRESS + ew_bin) * self.num_phases + phase

    # ------------------------------------------------------------------ #
    # BaseAgent interface
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        """UCB1-enhanced ε-greedy."""
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_phases)

        q = self.q_table[state_idx].copy()
        # UCB bonus: unexplored actions get a small boost
        total_visits = self.visit_count[state_idx].sum() + 1
        ucb_bonus = np.sqrt(2.0 * np.log(total_visits + 1)
                            / (self.visit_count[state_idx] + 1))
        q_ucb = q + 0.5 * ucb_bonus
        ties = np.flatnonzero(q_ucb == q_ucb.max())
        return int(np.random.choice(ties))

    def update(self, state, action, reward, next_state, done=False) -> dict:
        s  = int(state)      if np.isscalar(state)      else int(state[0])
        ns = int(next_state) if np.isscalar(next_state) else int(next_state[0])

        self.visit_count[s, action] += 1
        # Adaptive alpha: decays with visit count for this (s,a) pair
        alpha = self.alpha0 / (1.0 + self.visit_count[s, action] ** 0.7)

        target = reward + (0.0 if done else self.gamma * self.q_table[ns].max())
        td_err = target - self.q_table[s, action]
        self.q_table[s, action] += alpha * td_err
        return {"td_error": abs(td_err), "alpha": alpha}

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        data = {
            "q_table":     self.q_table.tolist(),
            "visit_count": self.visit_count.tolist(),
            "epsilon":     self.epsilon,
            "timestamp":   datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.q_table     = np.array(data["q_table"])
        self.visit_count = np.array(data.get("visit_count",
                           np.zeros_like(self.q_table, dtype=np.int32)))
        self.epsilon = data["epsilon"]

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        return int(np.argmax(self.q_table[state_idx]))
