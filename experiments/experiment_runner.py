"""
experiment_runner.py — Run and compare multiple agent configurations.

Example usage:
    python experiments/experiment_runner.py --agents qlearning dqn ppo
"""

from __future__ import annotations

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.visualization import plot_training_curves


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(agent_type: str, config: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  Experiment: {agent_type.upper()}")
    print(f"{'='*55}")
    trainer = Trainer(config, agent_type=agent_type)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic RL Experiment Runner")
    parser.add_argument("--agents", nargs="+", default=["dqn"],
                        choices=["qlearning", "dqn", "ppo"],
                        help="Agent(s) to train")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--eval_episodes", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)

    for agent_type in args.agents:
        run_experiment(agent_type, config)

    # Post-training evaluation
    evaluator = Evaluator(config, plots_dir=config["paths"]["plots_dir"])
    agents_info = []
    for agent_type in args.agents:
        ext = ".json" if agent_type == "qlearning" else ".pt"
        ckpt = os.path.join(config["paths"]["models_dir"], f"{agent_type}_best{ext}")
        if os.path.exists(ckpt):
            agents_info.append({"name": agent_type.upper(), "type": agent_type,
                                 "checkpoint": ckpt})

    if agents_info:
        evaluator.compare(agents_info, n_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
