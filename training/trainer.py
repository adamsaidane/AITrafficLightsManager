"""
trainer.py — Unified training loop for Q-Learning, DQN, and PPO.

Handles:
    - episode rollouts
    - agent updates
    - epsilon/exploration schedule
    - checkpoint saving
    - TensorBoard + CSV logging
"""

from __future__ import annotations

import os
import csv
import time
from typing import Optional

import numpy as np

from env.sumo_env import SUMOEnvironment
from agents.base_agent import BaseAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.logger import TrainingLogger
from utils.metrics import MetricsTracker


class Trainer:
    """
    Unified trainer that supports Q-Learning, DQN, and PPO agents.

    Usage:
        trainer = Trainer(config, agent_type="dqn")
        trainer.train()
    """

    def __init__(self, config: dict, agent_type: str = "dqn") -> None:
        self.cfg = config
        self.agent_type = agent_type.lower()
        sim_cfg = config["simulation"]

        self.num_episodes: int = sim_cfg["num_episodes"]
        self.total_steps: int = sim_cfg["total_steps"]
        log_cfg = config["logging"]
        self.log_interval: int = log_cfg["log_interval"]
        self.save_interval: int = log_cfg["save_interval"]

        paths = config["paths"]
        self.models_dir = paths["models_dir"]
        self.plots_dir = paths["plots_dir"]
        for d in [self.models_dir, self.plots_dir, paths["logs_dir"]]:
            os.makedirs(d, exist_ok=True)

        # Build environment once to get obs_dim
        self._sample_env = SUMOEnvironment(config)
        self.obs_dim = self._sample_env.obs_dim
        self.action_dim = self._sample_env.action_dim

        # Build agent
        self.agent = self._build_agent()

        self.logger = TrainingLogger(config, agent_type)
        self.metrics = MetricsTracker()

    # ------------------------------------------------------------------ #
    # Agent factory
    # ------------------------------------------------------------------ #

    def _build_agent(self) -> BaseAgent:
        if self.agent_type == "qlearning":
            return QLearningAgent(self.cfg)
        elif self.agent_type == "dqn":
            return DQNAgent(self.obs_dim, self.action_dim, self.cfg)
        elif self.agent_type == "ppo":
            return PPOAgent(self.obs_dim, self.action_dim, self.cfg)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    # ------------------------------------------------------------------ #
    # Episode runners
    # ------------------------------------------------------------------ #

    def _run_episode_qlearning(self, env: SUMOEnvironment, episode: int) -> dict:
        agent: QLearningAgent = self.agent  # type: ignore
        obs = env.start()
        state = agent.state_index(env.current_phase)

        ep_reward, ep_wait, ep_queue, ep_arrived = 0.0, 0.0, 0.0, 0.0
        step = 0

        while step < self.total_steps:
            action = agent.select_action(np.array([state]))
            obs, reward, done, info = env.step(action)
            next_state = agent.state_index(env.current_phase)

            agent.update(state, action, reward, next_state, done)

            ep_reward += reward
            ep_wait += info["total_waiting"]
            ep_queue += info["total_queue"]
            ep_arrived += info["arrived"]
            state = next_state
            step += 1
            if done:
                break

        agent.decay_epsilon()
        env.close()
        return dict(reward=ep_reward, waiting=ep_wait / max(step, 1),
                    queue=ep_queue / max(step, 1), arrived=ep_arrived,
                    epsilon=agent.epsilon)

    def _run_episode_dqn(self, env: SUMOEnvironment, episode: int) -> dict:
        agent: DQNAgent = self.agent  # type: ignore
        obs = env.start()

        ep_reward, ep_wait, ep_queue, ep_arrived = 0.0, 0.0, 0.0, 0.0
        losses = []
        step = 0

        while step < self.total_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(done))

            metrics = agent.update()
            if "loss" in metrics:
                losses.append(metrics["loss"])

            ep_reward += reward
            ep_wait += info["total_waiting"]
            ep_queue += info["total_queue"]
            ep_arrived += info["arrived"]
            obs = next_obs
            step += 1
            if done:
                break

        env.close()
        agent.on_episode_end()   # hard target sync every N episodes
        return dict(reward=ep_reward, waiting=ep_wait / max(step, 1),
                    queue=ep_queue / max(step, 1), arrived=ep_arrived,
                    epsilon=agent.epsilon,
                    loss=float(np.mean(losses)) if losses else 0.0)

    def _run_episode_ppo(self, env: SUMOEnvironment, episode: int) -> dict:
        agent: PPOAgent = self.agent  # type: ignore
        ppo_cfg = self.cfg["ppo"]
        n_steps = ppo_cfg["n_steps"]

        obs = env.start()
        ep_reward, ep_wait, ep_queue, ep_arrived = 0.0, 0.0, 0.0, 0.0
        step = 0
        last_obs = obs
        update_metrics: dict = {}

        while step < self.total_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, float(done))

            ep_reward += reward
            ep_wait += info["total_waiting"]
            ep_queue += info["total_queue"]
            ep_arrived += info["arrived"]
            last_obs = next_obs
            obs = next_obs
            step += 1

            # PPO update every n_steps
            if len(agent.buffer) >= n_steps or done:
                update_metrics = agent.update(last_state=last_obs)
                if done:
                    break

        env.close()
        result = dict(reward=ep_reward, waiting=ep_wait / max(step, 1),
                      queue=ep_queue / max(step, 1), arrived=ep_arrived)
        result.update(update_metrics)
        return result

    # ------------------------------------------------------------------ #
    # Main train loop
    # ------------------------------------------------------------------ #

    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from:
            self.agent.load(resume_from)
            print(f"[Trainer] Resumed from {resume_from}")

        print("\n" + "=" * 65)
        print(f"  Traffic RL Training  |  Agent: {self.agent_type.upper()}")
        print(f"  Episodes: {self.num_episodes}  |  Obs dim: {self.obs_dim}  |  Actions: {self.action_dim}")
        print("=" * 65)

        best_reward = float("-inf")

        for ep in range(1, self.num_episodes + 1):
            env = SUMOEnvironment(self.cfg)
            t0 = time.time()

            if self.agent_type == "qlearning":
                stats = self._run_episode_qlearning(env, ep)
            elif self.agent_type == "dqn":
                stats = self._run_episode_dqn(env, ep)
            else:
                stats = self._run_episode_ppo(env, ep)

            elapsed = time.time() - t0
            self.metrics.record(ep, stats)
            self.logger.log_episode(ep, stats)

            if ep % self.log_interval == 0:
                avg = self.metrics.recent_avg(10)
                print(f"  Ep {ep:>4}/{self.num_episodes} | "
                      f"R={stats['reward']:>8.1f} | Avg10={avg['reward']:>8.1f} | "
                      f"Wait={stats['waiting']:>6.1f}s | "
                      f"Arrived={stats['arrived']:>4.0f} | "
                      f"t={elapsed:.1f}s" +
                      (f" | ε={stats['epsilon']:.3f}" if "epsilon" in stats else ""))

            if ep % self.save_interval == 0 or ep == self.num_episodes:
                self._save_checkpoint(ep)

            if stats["reward"] > best_reward:
                best_reward = stats["reward"]
                self._save_checkpoint(ep, best=True)

        print("\n" + "=" * 65)
        print(f"  Training complete. Best reward: {best_reward:.2f}")
        print("=" * 65)
        self.logger.close()

    def _save_checkpoint(self, episode: int, best: bool = False) -> None:
        tag = "best" if best else f"ep{episode}"
        ext = ".json" if self.agent_type == "qlearning" else ".pt"
        path = os.path.join(self.models_dir, f"{self.agent_type}_{tag}{ext}")
        self.agent.save(path)
