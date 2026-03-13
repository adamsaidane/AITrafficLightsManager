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
        - queue_length (normalised by max_vehicles)
        - waiting_time (normalised by 300 s)
        - mean_speed (normalised by max_speed, inverted → lower is worse)
        - vehicle_density (normalised by max_vehicles)
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

        # Obs / action dimensions
        num_lanes = len(self.lanes)
        self.obs_dim: int = num_lanes * 4 + self.num_phases + 1
        self.action_dim: int = self.num_phases

    def start(self) -> np.ndarray:
        """Launch SUMO and return initial observation."""
        if not SUMO_AVAILABLE:
            raise RuntimeError("TraCI / sumolib not installed.")
        binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        traci.start([
            binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
        ])
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[0])
        self.current_phase = 0
        self.time_on_phase = 0
        self.is_yellow = False
        self.step_count = 0
        self._prev_waiting = 0.0
        return self._get_obs()

    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass

    def step(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one decision step.

        If action is None the agent keeps the current phase.
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
            # ---- Take action if phase can change ----
            if action is not None and self.can_change_phase():
                if action != self.current_phase:
                    self._start_yellow(action)
                # else: keep current phase (no switch)

        # Advance simulation
        traci.simulationStep()
        self.time_on_phase += 1
        self.step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(phase_changed)
        done = (self.step_count >= self.total_steps
                or traci.simulation.getMinExpectedNumber() == 0)
        return obs, reward, done, info

    def can_change_phase(self) -> bool:
        return (not self.is_yellow
                and self.time_on_phase >= self.min_green)

    def _start_yellow(self, next_phase: int) -> None:
        yellow = self.phases[self.current_phase].replace("G", "y").replace("g", "y")
        traci.trafficlight.setRedYellowGreenState(self.tl_id, yellow)
        self.is_yellow = True
        self.time_on_phase = 0
        self._pending_phase = next_phase

    def _apply_green(self, phase: int) -> None:
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[phase])
        self.current_phase = phase
        self.is_yellow = False
        self.time_on_phase = 0
        self._pending_phase = None

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

        reward = (self.w_wait * total_wait
                  + self.w_thru * arrived
                  + self.w_queue * total_queue
                  + (self.w_switch if phase_changed else 0.0)
                  + 0.3 * delta_wait)

        info = {
            "total_waiting": total_wait,
            "total_queue": total_queue,
            "arrived": arrived,
            "phase_changed": phase_changed,
        }
        return float(reward), info
