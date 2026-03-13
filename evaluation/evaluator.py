"""
evaluator.py — Agent evaluation and head-to-head comparison.

Supports:
    - Greedy rollout for any trained agent
    - Fixed-time baseline (round-robin phases)
    - Comparison table with statistical metrics
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from env.sumo_env import SUMOEnvironment
from agents.base_agent import BaseAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


class FixedTimeBaseline:
    """Round-robin fixed-time traffic light controller (no learning)."""

    def __init__(self, num_phases: int, green_time: int = 30) -> None:
        self.num_phases = num_phases
        self.green_time = green_time
        self._timer = 0
        self._phase = 0

    def select_action(self, obs) -> int:
        self._timer += 1
        if self._timer >= self.green_time:
            self._timer = 0
            self._phase = (self._phase + 1) % self.num_phases
        return self._phase

    def get_greedy_action(self, obs) -> int:
        return self.select_action(obs)


class Evaluator:
    """Evaluate one or more agents over multiple episodes."""

    def __init__(self, config: dict, plots_dir: str = "plots") -> None:
        self.cfg = config
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _load_agent(self, agent_type: str, checkpoint: str) -> BaseAgent:
        env = SUMOEnvironment(self.cfg)
        if agent_type == "qlearning":
            agent = QLearningAgent(self.cfg)
        elif agent_type == "dqn":
            agent = DQNAgent(env.obs_dim, env.action_dim, self.cfg)
        elif agent_type == "ppo":
            agent = PPOAgent(env.obs_dim, env.action_dim, self.cfg)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        agent.load(checkpoint)
        return agent

    def _run_eval_episode(self, agent, use_greedy: bool = True) -> Dict:
        """Run a single evaluation episode."""
        env = SUMOEnvironment(self.cfg)
        obs = env.start()

        ep_reward, ep_wait, ep_queue, ep_arrived = 0.0, 0.0, 0.0, 0.0
        step = 0
        total_steps = self.cfg["simulation"]["total_steps"]

        while step < total_steps:
            if use_greedy:
                action = agent.get_greedy_action(obs)
            else:
                action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            ep_wait += info["total_waiting"]
            ep_queue += info["total_queue"]
            ep_arrived += info["arrived"]
            step += 1
            if done:
                break

        env.close()
        return dict(
            reward=ep_reward,
            mean_waiting=ep_wait / max(step, 1),
            mean_queue=ep_queue / max(step, 1),
            total_arrived=ep_arrived,
        )

    def evaluate_agent(self, agent_type: str, checkpoint: str,
                       n_episodes: int = 10) -> Dict:
        """Evaluate a trained agent over n episodes, return aggregated stats."""
        agent = self._load_agent(agent_type, checkpoint)
        results = [self._run_eval_episode(agent) for _ in range(n_episodes)]
        return self._aggregate(results)

    def compare(self, agents_info: List[Dict], n_episodes: int = 10,
                include_fixed: bool = True) -> pd.DataFrame:
        """
        Compare multiple agents.

        agents_info: list of dicts with keys 'name', 'type', 'checkpoint'
        """
        records = []

        if include_fixed:
            env = SUMOEnvironment(self.cfg)
            baseline = FixedTimeBaseline(env.action_dim, green_time=30)
            results = [self._run_eval_episode(baseline, use_greedy=False)
                       for _ in range(n_episodes)]
            agg = self._aggregate(results)
            agg["agent"] = "FixedTime"
            records.append(agg)

        for info in agents_info:
            agg = self.evaluate_agent(info["type"], info["checkpoint"], n_episodes)
            agg["agent"] = info["name"]
            records.append(agg)

        df = pd.DataFrame(records).set_index("agent")
        print("\n" + "=" * 65)
        print("  Evaluation Results")
        print("=" * 65)
        print(df.to_string())
        self._plot_comparison(df)
        return df

    # ------------------------------------------------------------------ #

    def _aggregate(self, results: List[Dict]) -> Dict:
        keys = results[0].keys()
        agg = {}
        for k in keys:
            vals = [r[k] for r in results]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        return agg

    def _plot_comparison(self, df: pd.DataFrame) -> None:
        metrics = ["reward_mean", "mean_waiting_mean", "total_arrived_mean"]
        titles = ["Mean Reward", "Mean Waiting Time (s)", "Total Vehicles Arrived"]
        colors = ["steelblue", "salmon", "mediumseagreen"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, metric, title, color in zip(axes, metrics, titles, colors):
            if metric not in df.columns:
                continue
            err_col = metric.replace("_mean", "_std")
            yerr = df[err_col].values if err_col in df.columns else None
            df[metric].plot(kind="bar", ax=ax, color=color, yerr=yerr,
                            capsize=4, edgecolor="black")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Agent Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(self.plots_dir, "agent_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Comparison plot saved → {out}")
