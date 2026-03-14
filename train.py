"""
train.py — Entry point for training a traffic RL agent.

Usage examples:
    python train.py                          # DQN with default config
    python train.py --agent dqn
    python train.py --agent ppo
    python train.py --agent qlearning
    python train.py --agent dqn --resume models/dqn_ep100.pt
    python train.py --agent dqn --episodes 1000
"""

from __future__ import annotations

import os
import sys
import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic RL Training")
    parser.add_argument("--agent", default="dqn",
                        choices=["qlearning", "dqn", "ppo"],
                        help="RL algorithm to train")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes from config")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.episodes is not None:
        config["simulation"]["num_episodes"] = args.episodes

    # Generate SUMO config file
    sim_cfg = config["simulation"]
    sumocfg = f"""<configuration>
    <input>
        <net-file value="{sim_cfg['net_file']}"/>
        <route-files value="{sim_cfg['route_file']}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_cfg['total_steps']}"/>
    </time>
</configuration>"""
    with open(sim_cfg["sumo_cfg"], "w") as f:
        f.write(sumocfg)

    from training.trainer import Trainer
    trainer = Trainer(config, agent_type=args.agent)
    trainer.train(resume_from=args.resume)

    # Quick visualisation
    import glob
    logs = sorted(glob.glob(f"logs/{args.agent}_*/metrics.csv"))
    if logs:
        from utils.visualization import plot_training_curves
        plot_training_curves(logs[-1], out_dir=config["paths"]["plots_dir"])


if __name__ == "__main__":
    main()
