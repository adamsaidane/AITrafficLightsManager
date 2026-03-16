"""
sumo_env.py — SUMO Traffic Signal Environment
Compatible with SUMO/TraCI. Provides rich state vectors for deep RL.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import traci
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False


class SUMOEnvironment:
    """
    Gymnasium-style SUMO environment for single-intersection traffic control.

    State vector (per lane, flattened):
        - queue_length          (normalised by max_vehicles)
        - waiting_time          (normalised by 300 s)
        - mean_speed            (normalised by max_speed, inverted → lower is worse)
        - vehicle_density       (normalised by max_vehicles)
        + scalar: current_phase (one-hot encoded)
        + scalar: phase_duration (normalised by MAX_GREEN_TIME)

    Action:   integer ∈ [0, NUM_PHASES)
    Reward:   weighted combination of waiting-time reduction, throughput, queue
    """

    # ------------------------------------------------------------------ #
    def __init__(self, config: dict, use_gui: bool = False) -> None:
        self.cfg = config
        self.use_gui = use_gui

        sim_cfg = config["simulation"]
        self.tl_id: str = sim_cfg["tl_id"]
        self.lanes: List[str] = sim_cfg["lanes"]
        self.phases: Dict[int, str] = {int(k): v for k, v in sim_cfg["phases"].items()}
        self.num_phases: int = len(self.phases)
        self.min_green: int = sim_cfg["min_green_time"]
        self.max_green: int = sim_cfg["max_green_time"]
        self.yellow_time: int = sim_cfg["yellow_time"]
        self.total_steps: int = sim_cfg["total_steps"]
        self.sumo_cfg: str = sim_cfg["sumo_cfg"]

        rew_cfg = config["reward"]
        self.w_wait = rew_cfg["wait_multiplier"]
        self.w_thru = rew_cfg["throughput_multiplier"]
        self.w_queue = rew_cfg["queue_multiplier"]
        self.w_switch = rew_cfg["switch_penalty"]

        # Runtime state
        self.current_phase: int = 0
        self.time_on_phase: int = 0
        self.is_yellow: bool = False
        self.step_count: int = 0
        self._prev_waiting: float = 0.0
        self._pending_phase: Optional[int] = None

        # Phase logging — (phase_id, duration_steps) per completed phase
        self.phase_log: List[Tuple[int, int]] = []
        self._phase_start_step: int = 0

        # Composite action space (phase x duration):
        # Duration buckets in steps: 10s, 20s, 30s, 45s, 60s
        self.duration_buckets: List[int] = [10, 20, 30, 45, 60]
        self.num_duration_buckets: int = len(self.duration_buckets)
        self._committed_duration: int = self.min_green

        # Obs / action dimensions
        num_lanes = len(self.lanes)
        self.obs_dim: int = num_lanes * 4 + self.num_phases + 1
        # Extended: num_phases * num_duration_buckets  (8 * 5 = 40)
        self.action_dim: int = self.num_phases * self.num_duration_buckets

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> np.ndarray:
        """Launch SUMO and return initial observation."""
        if not SUMO_AVAILABLE:
            raise RuntimeError("TraCI / sumolib not installed.")
        binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        import random as _random
        seed = _random.randint(0, 99999)
        traci.start([
            binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--seed", str(seed),        # randomise traffic each episode
        ])
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[0])
        self.current_phase = 0
        self.time_on_phase = 0
        self.is_yellow = False
        self.step_count = 0
        self._prev_waiting = 0.0
        # Reset phase log for new episode
        self.phase_log = []
        self._phase_start_step = 0
        self._committed_duration = self.min_green
        return self._get_obs()

    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Core step
    # ------------------------------------------------------------------ #

    def decode_action(self, composite_action: int) -> Tuple[int, int]:
        """
        Decode composite action into (phase_id, duration_steps).
        composite_action in [0, num_phases * num_duration_buckets)
        """
        phase_id       = composite_action // self.num_duration_buckets
        duration_idx   = composite_action  % self.num_duration_buckets
        duration_steps = self.duration_buckets[duration_idx]
        return phase_id, duration_steps

    def step(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one decision step.

        action is a COMPOSITE integer encoding both phase and duration.
        It is decoded on the first call of a new phase; subsequent steps
        simply advance the simulation until the committed duration expires.
        Returns (obs, reward, done, info).
        """
        phase_changed = False

        # ---- Handle yellow → green transition ----
        if self.is_yellow:
            if self.time_on_phase >= self.yellow_time:
                assert self._pending_phase is not None
                self._apply_green(self._pending_phase)
                phase_changed = True
        else:
            # ---- Take action only when committed duration is met ----
            if action is not None and self.time_on_phase >= self._committed_duration:
                phase_id, duration = self.decode_action(action)
                # Clamp duration to [min_green, max_green]
                duration = max(self.min_green, min(duration, self.max_green))
                self._committed_duration = duration
                if phase_id != self.current_phase:
                    self._start_yellow(phase_id)
                # else: same phase, just extend with new duration

        # Advance simulation
        traci.simulationStep()
        self.time_on_phase += 1
        self.step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(phase_changed)
        done = (self.step_count >= self.total_steps
                or traci.simulation.getMinExpectedNumber() == 0)
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Phase control
    # ------------------------------------------------------------------ #

    def can_change_phase(self) -> bool:
        """True when the committed duration has been served (or min_green at minimum)."""
        return (not self.is_yellow
                and self.time_on_phase >= self._committed_duration)

    def _start_yellow(self, next_phase: int) -> None:
        yellow = self.phases[self.current_phase].replace("G", "y").replace("g", "y")
        traci.trafficlight.setRedYellowGreenState(self.tl_id, yellow)
        self.is_yellow = True
        self.time_on_phase = 0
        self._pending_phase = next_phase

    def _apply_green(self, phase: int) -> None:
        # Log the phase that just ended with its actual duration
        if self.step_count > 0:
            actual_duration = self.step_count - self._phase_start_step
            self.phase_log.append((self.current_phase, actual_duration))
            self._phase_start_step = self.step_count
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[phase])
        self.current_phase = phase
        self.is_yellow = False
        self.time_on_phase = 0
        self._committed_duration = self.min_green   # reset to minimum
        self._pending_phase = None

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """
        Returns a normalised float32 vector:
            [queue_0, wait_0, speed_0, density_0, ..., phase_onehot..., phase_duration]
        """
        lane_ids = set(traci.lane.getIDList()) if SUMO_AVAILABLE else set()
        feats: List[float] = []

        for lane in self.lanes:
            if lane in lane_ids:
                q = traci.lane.getLastStepHaltingNumber(lane) / 50.0
                w = traci.lane.getWaitingTime(lane) / 300.0
                spd = traci.lane.getLastStepMeanSpeed(lane)
                max_spd = traci.lane.getMaxSpeed(lane) if traci.lane.getMaxSpeed(lane) > 0 else 13.9
                spd_norm = 1.0 - min(spd / max_spd, 1.0)   # high value = congested
                dens = traci.lane.getLastStepOccupancy(lane)
            else:
                q, w, spd_norm, dens = 0.0, 0.0, 0.0, 0.0

            feats.extend([
                min(q, 1.0),
                min(w, 1.0),
                spd_norm,
                min(dens, 1.0),
            ])

        # Phase one-hot
        phase_oh = [0.0] * self.num_phases
        phase_oh[self.current_phase] = 1.0
        feats.extend(phase_oh)

        # Normalised phase duration
        feats.append(min(self.time_on_phase / self.max_green, 1.0))

        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _compute_reward(self, phase_changed: bool) -> Tuple[float, dict]:
        lane_ids = set(traci.lane.getIDList()) if SUMO_AVAILABLE else set()

        total_wait, total_queue = 0.0, 0.0
        for lane in self.lanes:
            if lane in lane_ids:
                total_wait += traci.lane.getWaitingTime(lane)
                total_queue += traci.lane.getLastStepHaltingNumber(lane)

        arrived = float(traci.simulation.getArrivedNumber())
        delta_wait = self._prev_waiting - total_wait   # positive = improvement
        self._prev_waiting = total_wait

        # Normalise raw values by number of lanes to keep reward scale stable
        n_lanes = max(len(self.lanes), 1)
        norm_wait  = total_wait  / n_lanes
        norm_queue = total_queue / n_lanes
        norm_delta = delta_wait  / n_lanes

        reward = (self.w_wait   * norm_wait
                  + self.w_thru  * arrived
                  + self.w_queue * norm_queue
                  + (self.w_switch if phase_changed else 0.0)
                  + 0.1 * norm_delta)

        # Hard clip: prevents extreme gradient magnitudes during early training
        reward = float(np.clip(reward, -200.0, 200.0))

        info = {
            "total_waiting":  total_wait,
            "total_queue":    total_queue,
            "arrived":        arrived,
            "phase_changed":  phase_changed,
        }
        return reward, info

    def get_phase_log(self) -> List[Tuple[int, int]]:
        """Return the full phase log for this episode. Call after env closes."""
        return list(self.phase_log)
