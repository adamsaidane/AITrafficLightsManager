# 🚦 Traffic RL — Adaptive Traffic Light Control with Deep Reinforcement Learning

A research-grade reinforcement learning project for adaptive traffic signal control
using the [SUMO](https://www.eclipse.org/sumo/) traffic simulator.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Algorithms](#algorithms)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Visualisation](#visualisation)
9. [Configuration](#configuration)
10. [Design Decisions](#design-decisions)

---

## Problem Statement

Traffic signal control at a single intersection is modelled as a **Markov Decision Process**:

| Component | Description |
|-----------|-------------|
| **State**  | Per-lane features: queue length, waiting time, mean speed, occupancy + current phase (one-hot) + phase duration |
| **Action** | Select one of 8 pre-defined signal phases |
| **Reward** | Weighted combination of: Δwaiting time (improvement signal), total waiting (penalty), vehicle throughput (bonus), queue length (penalty), phase-switch penalty |

The agent is trained inside a SUMO simulation via the TraCI Python API.

---

## Algorithms

Three algorithms are implemented, in increasing capability order:

### 1. Q-Learning (baseline)
Classic tabular Q-Learning. State is discretised into 40 buckets
(5 pressure levels × 8 phases). Good for quick experimentation.  
**Limitation**: cannot generalise to unseen states.

### 2. Double Dueling DQN + Prioritised Replay ⭐ (recommended)
- **Dueling architecture**: separates state-value V(s) from action-advantages A(s,a) — better generalisation across phases.
- **Double DQN**: decouples action selection from evaluation → reduces Q-value overestimation.
- **Prioritised Experience Replay (PER)**: samples high-TD-error transitions more frequently → up to 2× faster convergence.
- Rich 73-dimensional state vector.

### 3. PPO (Proximal Policy Optimisation)
- On-policy algorithm — handles non-stationary traffic distributions naturally.
- Clipped surrogate objective ensures stable updates.
- GAE (Generalised Advantage Estimation) for low-variance gradient estimates.
- Entropy bonus prevents premature phase-preference collapse.

---

## Architecture

```
traffic_rl_project/
├── configs/
│   └── config.yaml          # All hyperparameters & paths
├── env/
│   └── sumo_env.py          # SUMO/TraCI environment wrapper
├── agents/
│   ├── base_agent.py        # Abstract base class
│   ├── q_learning_agent.py  # Tabular Q-Learning
│   ├── dqn_agent.py         # Double Dueling DQN + PER
│   └── ppo_agent.py         # PPO with GAE
├── training/
│   └── trainer.py           # Unified training loop
├── evaluation/
│   └── evaluator.py         # Greedy rollout + comparison
├── utils/
│   ├── metrics.py           # Rolling-window metric tracker
│   ├── logger.py            # TensorBoard + CSV logger
│   └── visualization.py     # Training curve plots
├── experiments/
│   └── experiment_runner.py # Multi-agent experiment pipeline
├── train.py                 # Training entry point
└── evaluate.py              # Evaluation entry point
```

---

## Installation

### Prerequisites

```bash
# 1. Install SUMO
# Ubuntu/Debian:
sudo apt-get install sumo sumo-tools sumo-doc

# macOS:
brew install sumo

# 2. Install Python dependencies
pip install torch numpy pandas matplotlib pyyaml tensorboard
```

### Verify SUMO

```bash
sumo --version
python -c "import traci; print('TraCI OK')"
```

---

## Quick Start

```bash
cd traffic_rl_project

# Train the DQN agent (recommended)
python train.py --agent dqn

# Train PPO
python train.py --agent ppo

# Train baseline Q-Learning
python train.py --agent qlearning
```

---

## Training

```bash
# Full training with custom episode count
python train.py --agent dqn --episodes 500

# Resume from a checkpoint
python train.py --agent dqn --resume models/dqn_ep200.pt

# Run all three agents and compare automatically
python experiments/experiment_runner.py --agents qlearning dqn ppo
```

Training produces:
- `models/<agent>_best.pt` — best checkpoint
- `models/<agent>_ep<N>.pt` — periodic checkpoints
- `logs/<agent>_<timestamp>/metrics.csv` — per-episode metrics
- `logs/<agent>_<timestamp>/` — TensorBoard events

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Evaluation

```bash
# Evaluate best DQN checkpoint (10 episodes)
python evaluate.py --agent dqn --episodes 10

# Compare all agents vs fixed-time baseline
python evaluate.py --compare --agents qlearning dqn ppo --episodes 10
```

---

## Visualisation

```bash
# Generate training curves from a log file
python -c "
from utils.visualization import plot_training_curves
plot_training_curves('logs/<run>/metrics.csv', out_dir='plots')
"
```

Outputs saved to `plots/`:
- `training_curves.png` — reward, waiting time, throughput, queue, epsilon, loss
- `agent_comparison.png` — bar chart comparing all agents vs baseline

---

## Configuration

All hyperparameters live in `configs/config.yaml`. Key sections:

```yaml
simulation:
  num_episodes: 500
  total_steps: 3600        # 1 hour of simulated time

dqn:
  learning_rate: 0.0005
  gamma: 0.99
  double_dqn: true
  dueling: true
  batch_size: 64
  replay_buffer_size: 50000

ppo:
  learning_rate: 0.0003
  n_steps: 2048            # rollout length before PPO update
  update_epochs: 10

reward:
  wait_multiplier: -0.1
  throughput_multiplier: 1.0
  switch_penalty: -2.0
```

---

## Design Decisions

### State Representation
The 73-dim state vector (16 lanes × 4 features + 8 phase one-hot + 1 duration)
is far richer than the original 5-level pressure discretisation, enabling
neural networks to learn fine-grained traffic patterns.

### Reward Engineering
The reward combines:
- **Δwaiting** (0.3 weight): immediate improvement signal, dense feedback.
- **Total waiting** (−0.1): penalises congestion globally.
- **Throughput** (+1.0): rewards vehicles completing their journey.
- **Queue** (−0.05): additional pressure to keep queues short.
- **Switch penalty** (−2.0): discourages unnecessary phase oscillation.

### Why DQN over vanilla Q-Learning?
Q-Learning state space is O(pressure_levels × phases) = 40.  
DQN state space is continuous and 73-dimensional — exponentially more expressive,
enabling the agent to react to subtle lane-level patterns.

### Why not multi-agent?
This implementation targets a **single intersection**.  
For multi-intersection grids, extend with Independent DQN (each intersection
runs its own DQN) or QMIX for cooperative MARL.
