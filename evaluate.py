"""
evaluate.py — Evaluate and compare trained agents.

Usage:
    python evaluate.py --agent dqn --checkpoint models/dqn_best.pt
    python evaluate.py --compare --agents qlearning dqn ppo
"""

from __future__ import annotations

import argparse
import yaml
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic RL Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--agent", default="dqn",
                        choices=["qlearning", "dqn", "ppo"])
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (auto-detected if not given)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available checkpoints")
    parser.add_argument("--agents", nargs="+", default=["dqn"],
                        choices=["qlearning", "dqn", "ppo"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(config, plots_dir=config["paths"]["plots_dir"])

    if args.compare:
        agents_info = []
        for agent_type in args.agents:
            ext = ".json" if agent_type == "qlearning" else ".pt"
            ckpt = os.path.join(config["paths"]["models_dir"],
                                f"{agent_type}_best{ext}")
            if os.path.exists(ckpt):
                agents_info.append({"name": agent_type.upper(),
                                    "type": agent_type,
                                    "checkpoint": ckpt})
            else:
                print(f"[Warn] No checkpoint found for {agent_type}: {ckpt}")
        evaluator.compare(agents_info, n_episodes=args.episodes)
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            ext = ".json" if args.agent == "qlearning" else ".pt"
            ckpt = os.path.join(config["paths"]["models_dir"],
                                f"{args.agent}_best{ext}")
        stats = evaluator.evaluate_agent(args.agent, ckpt, args.episodes)
        print("\nEvaluation results:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
