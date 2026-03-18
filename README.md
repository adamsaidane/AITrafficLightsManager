# 🚦 AI Traffic Lights Manager
**Adaptive Traffic Signal Control with Deep Reinforcement Learning**

A research-grade reinforcement learning project for intelligent traffic light management
at intersections using the [SUMO](https://www.eclipse.org/sumo/) traffic simulator.
Implements three advanced RL algorithms to optimise traffic flow and reduce congestion.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Algorithms](#-algorithms)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [GPU Setup](#-gpu-setup)
- [Quick Start](#-quick-start)
- [Traffic Density — Curriculum Training](#-traffic-density--curriculum-training)
- [Training](#-training)
- [Resuming Training](#-resuming-training)
- [Evaluation & Plots](#-evaluation--plots)
- [Baselines](#-baselines)
- [Visualization](#-visualization)
- [Configuration](#-configuration)
- [Expected Episodes to Convergence](#-expected-episodes-to-convergence)
- [Design Decisions](#-design-decisions)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🎯 Overview

**AI Traffic Lights Manager** solves the adaptive traffic signal control problem
using Deep Reinforcement Learning. The system learns optimal phase sequences and
durations to minimise vehicle waiting times, reduce queue lengths, and maximise
throughput at a 4-way signalised intersection.

### Real-World Impact

- 🚗 **Reduce Average Waiting Time** — 15–30% improvement vs fixed-time signals
- ⏱️ **Lower Queue Lengths** — Adaptive phase switching based on real-time traffic pressure
- 📊 **Improve Throughput** — More vehicles complete their journey per hour
- 🧠 **Adaptive Learning** — Automatically adjusts to changing traffic patterns
- ⏳ **Phase Duration Control** — Agent chooses both *which* phase and *how long* to hold it

---

## 📊 Problem Statement

Traffic signal control at a single intersection is modelled as a **Markov Decision Process (MDP)**:

| Component | Description |
|-----------|-------------|
| **State** | 73-dim vector: per-lane queue, waiting time, mean speed, occupancy (16 lanes × 4 features) + current phase one-hot (8) + normalised phase duration (1) |
| **Action** | Composite integer encoding **phase × duration** — 8 phases × 5 duration buckets = **40 actions** |
| **Duration buckets** | 10 s, 20 s, 30 s, 45 s, 60 s — agent chooses both what phase and how long to hold it |
| **Reward** | Normalised combination of waiting-time reduction, vehicle throughput, queue length, phase-switch penalty |
| **Environment** | SUMO simulator via TraCI API — random seed per episode to prevent memorisation |

### State Vector (73-dimensional)

```
[queue_length × 16 lanes]     32 features   normalised / 50 vehicles
[waiting_time × 16 lanes]     32 features   normalised / 300 s
[mean_speed   ×  8 groups]     8 features   inverted: 1 − v/v_max (high = congested)
[occupancy    ×  8 groups]     8 features   lane usage %
[phase_one_hot × 8]            8 features   current signal state
[phase_duration]               1 feature    normalised / max_green
```

### Reward Design

```python
reward = (- w_wait  * total_waiting / n_lanes      # penalise congestion
          + w_thru  * vehicles_arrived              # reward completed trips
          - w_queue * total_queue / n_lanes         # penalise queues
          - w_switch if phase_changed               # penalise unnecessary switching
          + 0.1 * delta_waiting / n_lanes)          # dense improvement signal

# Hard clip to [-200, +200] per step — prevents gradient explosion
reward = np.clip(reward, -200.0, 200.0)
```

---

## 🤖 Algorithms

Three algorithms implemented in increasing capability order:

### 1️⃣ Q-Learning (Interpretable Baseline)

Tabular Q-Learning with an improved **72-state** discretisation
(3 NS pressure bins × 3 EW pressure bins × 8 phases).

**Improvements over classic Q-Learning:**
- **Optimistic initialisation** (`q_table = +50`): forces the agent to try every action before specialising
- **Adaptive learning rate** `α(s,a) = α₀ / (1 + n(s,a)^0.7)`: fast early learning, stable convergence
- **UCB1 tie-breaking**: bonus for unexplored actions prevents premature exploitation

**Strengths:** Fast training, fully interpretable Q-table, minimal hyperparameter tuning  
**Limitation:** Cannot generalise to unseen states; state space capped at 72 buckets

```yaml
q_learning:
  alpha: 0.05              # Base learning rate (decays per visit count)
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_decay: 0.998     # Slow decay — stays exploratory for ~460 episodes
  epsilon_min: 0.05
```

---

### 2️⃣ Double Dueling DQN + Prioritised Replay ⭐ (Recommended)

**Architecture:**
```
Input (73) → Linear(256) → ReLU → Linear(256) → ReLU
                                               ↓
                              ┌─────────────────────────┐
                              │  Value head   V(s) → 1  │
                              │  Advantage  A(s,a) → 40 │
                              └─────────────────────────┘
              Q(s,a) = V(s) + A(s,a) − mean(A)
```

**Key techniques:**
- **Dueling network**: separates state-value V(s) from action-advantages A(s,a) — better generalisation when many actions are equivalent
- **Double DQN**: decouples action selection from Q-value evaluation → reduces overestimation bias
- **Prioritised Experience Replay (PER)** (Schaul et al., 2016): samples high-TD-error transitions more frequently → up to 2× faster convergence; importance-sampling weights for bias correction
- **Running reward normalisation**: Welford online algorithm keeps reward in ±3σ range regardless of traffic density
- **Hard target sync** every 50 episodes + soft update (τ = 0.001)
- **Update every 4 steps**: decouples data collection from learning frequency

```yaml
dqn:
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_decay_steps: 400000   # ~111 episodes of real exploration
  replay_buffer_size: 100000
  min_replay_size: 5000
  tau: 0.001
  double_dqn: true
  dueling: true
```

---

### 3️⃣ PPO — Proximal Policy Optimisation

On-policy algorithm well suited to non-stationary traffic distributions.

**Algorithm:**
```python
# Clipped surrogate objective
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

# Generalised Advantage Estimation
A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1} δ_T

# Entropy regularisation (decaying)
L_ENT = -β(t) * H(π)    where β(t) decays from 0.5 → 0.01
```

**Anti mode-collapse mechanisms** — without these, PPO collapses to 1–3 phases within 50 episodes:

| Mechanism | Description |
|-----------|-------------|
| **Entropy decay** | `entropy_coef` starts at 0.5, decays linearly to 0.01 over all episodes — forces exploration early, allows convergence late |
| **Residual ε-greedy** | 5% of actions are always random, permanently preventing hard collapse onto one phase |
| **n_steps = 1024** | Shorter rollouts break temporal correlation in gradient estimates |
| **update_epochs = 8** | More gradient passes per collected batch |

```yaml
ppo:
  learning_rate: 0.0001
  entropy_coef_start: 0.5      # High entropy → forces all 40 actions early
  entropy_coef_end: 0.01       # Low entropy → policy converges late
  epsilon_residual: 0.05       # 5% random actions always (anti-collapse)
  n_steps: 1024
  update_epochs: 8
  lam: 0.95
  clip_eps: 0.2
```

---

## 🏗️ Architecture

```
AITrafficLightsManager/
│
├── configs/
│   └── config.yaml                   # All hyperparameters & paths
│
├── env/
│   └── sumo_env.py                   # SUMO/TraCI environment
│                                       - 73-dim observation vector
│                                       - 40 composite actions (phase × duration)
│                                       - Reward normalisation + hard clip
│                                       - Per-episode phase log
│                                       - Random SUMO seed per episode
│
├── agents/
│   ├── base_agent.py                 # Abstract interface (ABC)
│   ├── q_learning_agent.py           # Tabular Q-Learning (UCB + adaptive α)
│   ├── dqn_agent.py                  # Double Dueling DQN + PER + reward norm
│   └── ppo_agent.py                  # PPO + entropy decay + residual ε-greedy
│
├── training/
│   └── trainer.py                    # Unified training loop
│                                       - Phase report printed each episode
│                                       - Checkpoint saving (best + periodic)
│                                       - TensorBoard + CSV logging
│
├── evaluation/
│   └── evaluator.py                  # Greedy rollouts + multi-agent comparison
│
├── utils/
│   ├── gpu_config.py                 # GPU detection (CUDA / MPS / CPU fallback)
│   ├── metrics.py                    # Rolling-window statistics
│   ├── logger.py                     # TensorBoard + CSV logger
│   └── visualization.py             # Training curve plots
│
├── experiments/
│   └── experiment_runner.py         # Sequential multi-agent pipeline
│
├── main.py                           # Full benchmark pipeline
│                                       - 6 baselines (Phase 1)
│                                       - Train RL agents (Phase 2)
│                                       - Evaluate (Phase 3)
│                                       - Plots + report (Phase 4)
│
├── train.py                          # Single-agent training entry point
├── evaluate.py                       # Evaluation & comparison entry point
├── patch_traffic.py                  # SUMO .rou.xml traffic density patcher
│
└── SUMO/
    └── intersection/
        ├── intersection.net.xml      # Road network (300m branches, 4 lanes)
        ├── intersection.rou.xml      # Routes & flows (departLane="best")
        └── intersection.sumocfg     # SUMO simulation config
```

---

## ⭐ Key Features

### Training
- ✅ **Multi-Algorithm Support** — Q-Learning, DQN, PPO in one unified framework
- ✅ **Composite Action Space** — Agent controls both phase and green duration (40 actions)
- ✅ **Anti Mode-Collapse** — Entropy decay + residual ε-greedy for PPO
- ✅ **Reward Normalisation** — Per-lane scaling + hard clip prevents gradient explosion
- ✅ **Curriculum Learning** — Progressive traffic density via `patch_traffic.py`
- ✅ **Random SUMO Seeds** — Each episode sees a different traffic realisation
- ✅ **Phase Report** — Per-episode breakdown of phase usage and durations
- ✅ **GPU Acceleration** — Centralised detection with CUDA / MPS / CPU fallback

### Evaluation
- ✅ **6 Baselines** — Random, Fixed-15s, Fixed-30s, Fixed-60s, Max-Pressure, Actuated
- ✅ **Statistical Metrics** — Mean ± std for reward, waiting time, queue, throughput
- ✅ **Phase Distribution** — How often each of the 40 composite actions was selected
- ✅ **Checkpoint Comparison** — Evaluate any saved `.pt` or `.json` checkpoint

### Visualisation
- ✅ **6-Panel Training Dashboard** — Reward, waiting, queue, throughput, epsilon/entropy, loss
- ✅ **Agent Comparison Bar Chart** — All agents vs all baselines
- ✅ **Normalised Radar Chart** — Overall performance at a glance
- ✅ **Phase Distribution Histograms**
- ✅ **CSV + JSON Export** — Full metrics for custom analysis

---

## 📦 Prerequisites

```
Python >= 3.8
SUMO   >= 1.14
PyTorch >= 2.0
```

```bash
pip install torch numpy pandas matplotlib pyyaml tensorboard
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/adamsaidane/AITrafficLightsManager.git
cd AITrafficLightsManager
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
# or manually:
pip install torch torchvision torchaudio numpy pandas matplotlib pyyaml tensorboard
```

### 4. Install SUMO

```bash
# Ubuntu / Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS
brew install sumo

# Windows — download installer from https://sumo.dlr.de/docs/Downloads.php
```

### 5. Verify installation

```bash
sumo --version
python -c "import traci; print('TraCI ready')"
python -c "import torch; print('PyTorch', torch.__version__)"
```

---

## 🖥️ GPU Setup

GPU selection is centralised in `utils/gpu_config.py`.  
Priority: **CUDA → Apple MPS → CPU**.

### Standard NVIDIA (RTX 10xx – 40xx series)

```bash
pip uninstall torch torchvision torchaudio -y

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### RTX 5000 series — Blackwell architecture (sm_120) ⚠️

RTX 5060 Ti, 5070, 5080, 5090 require **PyTorch nightly** —
stable builds only support up to sm_90 (Ada Lovelace).

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Verify GPU

```bash
python -c "
import torch
print('CUDA available :', torch.cuda.is_available())
print('GPU            :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
t = torch.zeros(1).cuda()
print('Tensor on GPU OK')
"
```

To force CPU mode (debug only): `FORCE_CPU=1 python main.py`

---

## ⚡ Quick Start

```bash
# Full pipeline: 6 baselines + train all agents + evaluate + all plots
python main.py

# Quick functional test (results not meaningful at 40 episodes)
python main.py --quick

# Train DQN only, 700 episodes
python train.py --agent dqn --episodes 700

# Evaluate existing checkpoints and generate all plots (no retraining)
python main.py --skip-training --eval-episodes 20
```

---

## 🎓 Traffic Density — Curriculum Training

The route file uses `probability` attributes to generate stochastic vehicle flows.
At full density (~2880 veh/h) the intersection is near saturation — all actions
produce similarly bad outcomes and the reward signal becomes uninformative.

RL research recommends starting at 40–60% capacity and increasing progressively.

### Patch traffic density

```bash
# Preview — no file modification
python patch_traffic.py \
    --rou "SUMO/intersection/intersection.rou.xml" \
    --density 0.5 --preview

# Apply 50% density
python patch_traffic.py \
    --rou "SUMO/intersection/intersection.rou.xml" \
    --density 0.5
```

A `.bak` backup is created automatically on first run.

### Full curriculum

```bash
# Stage 1 — 40% (learn the basics)
python patch_traffic.py --rou "SUMO/intersection/intersection.rou.xml" --density 0.4
python train.py --agent dqn --episodes 400

# Stage 2 — 60% (consolidate)
python patch_traffic.py --rou "SUMO/intersection/intersection.rou.xml" --density 0.6
python train.py --agent dqn --resume models/dqn_best.pt --episodes 400

# Stage 3 — 80% (generalise)
python patch_traffic.py --rou "SUMO/intersection/intersection.rou.xml" --density 0.8
python train.py --agent dqn --resume models/dqn_best.pt --episodes 400

# Final evaluation at full density
python patch_traffic.py --rou "SUMO/intersection/intersection.rou.xml" --density 1.0
python main.py --skip-training --eval-episodes 20
```

### Restore original

```bash
# Windows
copy "SUMO\intersection\intersection.rou.bak" "SUMO\intersection\intersection.rou.xml"

# Linux / macOS
cp SUMO/intersection/intersection.rou.bak SUMO/intersection/intersection.rou.xml
```

---

## 🎓 Training

### Single agent

```bash
python train.py --agent dqn --episodes 700
python train.py --agent ppo --episodes 1200
python train.py --agent qlearning --episodes 400
```

### All agents via main.py

```bash
python main.py --agents dqn ppo --train-episodes 700 --eval-episodes 10
```

### Output files

```
models/
  <agent>_best.pt          # Updated whenever a new best reward is achieved
  <agent>_ep50.pt          # Periodic checkpoint every save_interval episodes
  <agent>_ep100.pt

logs/
  <agent>_<timestamp>/
    metrics.csv            # Per-episode: reward, waiting, queue, arrived,
                           #   phases_used, dominant_phase, dominant_pct,
                           #   phase_counts, entropy_coef (PPO), loss (DQN)
    events.out.tfevents.*  # TensorBoard events
```

### Live monitoring with TensorBoard

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

### Phase report (PPO training output)

Every `log_interval` episodes the trainer prints a detailed breakdown:

```
  Ep  200/1200 | R= 13450.2 | Wait= 612.3s | Arrived=2681 | phases=6/8 | dom=42%
    Phase 0 :  85x ( 42.1%)  avg=18.4s  min=10s  max=52s
    Phase 1 :  44x ( 21.8%)  avg=22.1s  min=10s  max=60s
    Phase 3 :  28x ( 13.9%)  avg=15.6s  min=10s  max=38s
    Phase 4 :  22x ( 10.9%)  avg=12.3s  min=10s  max=25s
    Phase 5 :  14x (  6.9%)  avg=11.8s  min=10s  max=20s
    Phase 7 :   8x (  4.0%)  avg=10.4s  min=10s  max=14s
    Phases not used : [2, 6]
```

**Healthy signs:** 6–8 phases used, dominant % below 50%, average duration > 15 s.  
**Collapse sign:** 1–3 phases at 90%+, average duration always at 10 s (minimum).

---

## 🔄 Resuming Training

```bash
# Resume PPO from episode 750, train 450 more to reach 1200 total
python train.py --agent ppo --resume models/ppo_ep750.pt --episodes 450

# Resume DQN from best checkpoint
python train.py --agent dqn --resume models/dqn_best.pt --episodes 300
```

**What is restored from the checkpoint:**
- All network weights
- Adam optimiser state (momentum + learning rate schedule)
- Epsilon value (DQN)
- Episode counter for entropy decay position (PPO)

**PPO entropy check after resuming:**  
If resumed at episode 750 / 1200, entropy should be approximately:
```
entropy = 0.5 − (750/1200) × (0.5 − 0.01) ≈ 0.19
```
If it resets to 0.5, correct it manually:
```python
agent.load("models/ppo_ep750.pt")
agent._episode = 750
```

---

## 📈 Evaluation & Plots

### Evaluate a single agent

```bash
# Evaluate best PPO (greedy, no exploration)
python evaluate.py --agent ppo --episodes 10

# Evaluate a specific checkpoint
python evaluate.py --agent dqn --checkpoint models/dqn_ep500.pt --episodes 10
```

### Compare all agents vs baselines

```bash
python evaluate.py --compare --agents qlearning dqn ppo --episodes 10
```

### Full benchmark (no retraining)

```bash
python main.py --skip-training --agents dqn ppo --eval-episodes 20
```

### Generated plots

| File | Content |
|------|---------|
| `plots/full_comparison.png` | 4-panel bar chart: reward, waiting, queue, vehicles arrived |
| `plots/radar_comparison.png` | Normalised spider chart — overall performance at a glance |
| `plots/phase_distribution.png` | Phase selection frequency per agent |
| `plots/benchmark_results.csv` | Full table with mean ± std for every metric |
| `plots/benchmark_results.json` | Raw results for custom analysis |

### Training curves

```bash
python -c "
from utils.visualization import plot_training_curves
import glob
logs = sorted(glob.glob('logs/ppo_*/metrics.csv'))
plot_training_curves(logs[-1], out_dir='plots')
print('Saved plots/training_curves.png')
"
```

**6-panel training dashboard:** reward curve, waiting time, queue length,
throughput, epsilon / entropy decay, loss.

---

## 📊 Baselines

Six non-learning controllers used as comparison points:

| Baseline | Description |
|----------|-------------|
| **Random** | Uniformly random phase selection at each decision step |
| **Fixed-15s** | Round-robin, 15 s per phase — fast cycling |
| **Fixed-30s** | Round-robin, 30 s per phase — industry standard |
| **Fixed-60s** | Round-robin, 60 s per phase — slow cycling |
| **Max-Pressure** | Activates the phase serving the direction with most halting vehicles |
| **Actuated** | Extends green while demand exists on active lanes; switches when empty or max_green reached |

**Key insight from benchmarking:** Fixed-15s consistently outperforms Fixed-30s and Fixed-60s
on this intersection geometry, making it the most relevant baseline to beat.
Max-Pressure often matches or exceeds Fixed-15s without any learning.

---

## 📊 Visualization

```bash
# Training curves for all agents
python -c "
from utils.visualization import plot_training_curves
import glob, os
for agent in ['qlearning', 'dqn', 'ppo']:
    logs = sorted(glob.glob(f'logs/{agent}_*/metrics.csv'))
    if logs:
        plot_training_curves(logs[-1], out_dir='plots')
        os.rename('plots/training_curves.png',
                  f'plots/training_curves_{agent}.png')
        print(f'Saved plots/training_curves_{agent}.png')
"
```

---

## ⚙️ Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
simulation:
  total_steps: 3600                    # 1 simulated hour per episode
  num_episodes: 500
  min_green_time: 10                   # Minimum before agent can change phase
  max_green_time: 60                   # Hard ceiling on any phase duration
  duration_buckets: [10, 20, 30, 45, 60]   # 5 choices → 8×5 = 40 actions

q_learning:
  alpha: 0.05                          # Base learning rate (decays per visit)
  epsilon_decay: 0.998                 # ~460 episodes to reach epsilon_min
  epsilon_min: 0.05

dqn:
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_decay_steps: 400000          # ~111 episodes of exploration
  replay_buffer_size: 100000
  min_replay_size: 5000
  tau: 0.001                           # Soft target network update
  double_dqn: true
  dueling: true

ppo:
  learning_rate: 0.0001
  entropy_coef_start: 0.5              # Forces exploration of all 40 actions
  entropy_coef_end: 0.01               # Allows convergence in late training
  epsilon_residual: 0.05               # 5% random actions — permanent anti-collapse
  n_steps: 1024                        # Rollout length before update
  update_epochs: 8
  lam: 0.95                            # GAE lambda
  clip_eps: 0.2

reward:
  wait_multiplier: -0.001              # Normalised by n_lanes (prevents -7M gradients)
  throughput_multiplier: 5.0           # Dominant positive signal
  queue_multiplier: -0.01
  switch_penalty: -0.5

logging:
  log_interval: 10                     # Print phase report every N episodes
  save_interval: 50                    # Save checkpoint every N episodes
  tensorboard: true
```

---

## 📅 Expected Episodes to Convergence

Estimates for the composite 40-action space, 3600 steps/episode, 50% traffic density:

| Agent | Exploration ends | Convergence visible | Plateau | **Recommended** |
|-------|-----------------|---------------------|---------|-----------------|
| Q-Learning | ~460 ep | ~200 ep | ~400 ep | **400 ep** |
| DQN | ~111 ep | ~150 ep | ~600 ep | **700 ep** |
| PPO | gradual (entropy) | ~400 ep | ~1000 ep | **1200 ep** |

With **curriculum training** (3 density stages × 400 ep = 1200 total), these
targets are reached more efficiently because the agent transfers knowledge
across stages. 1200 curriculum episodes ≈ 1800 fixed-density episodes.

**Approximate wall-clock time (RTX 3080 / 4080):**

| Agent | Recommended eps | Time |
|-------|----------------|----------|
| Q-Learning | 400 | ~2 h |
| DQN | 700 | ~3 h |
| PPO | 1200 | ~4 h |

---

## 🧠 Design Decisions

### Composite action space (phase × duration)

The original design let the agent choose only the phase — duration was fixed at
`MIN_GREEN_TIME = 10 s`. This caused the agent to change phases as fast as
possible, never learning to hold a productive green phase long enough to clear
a queue.

The composite action `phase_id × duration_bucket` encodes both decisions in a
single integer. With 8 phases × 5 durations = **40 actions**, the network learns
to correlate the observed queue length with an appropriate green duration.

### Reward normalisation

Raw waiting times across 16 lanes at full density reached 6000–19 000 s,
producing episode rewards of −7 000 000. Gradients of this magnitude
overwhelmed the entropy signal (≈ 360) and caused immediate mode collapse.

Dividing by `n_lanes` and clipping to ±200 per step keeps reward, entropy,
throughput, and queue signals all in a competitive range.

### Random SUMO seed per episode

Without seed randomisation, the agent memorises a fixed phase sequence that
works for one specific traffic realisation. With `--seed random`, each episode
presents a different stochastic traffic pattern, forcing the agent to learn a
general policy rather than a memorised schedule.

### 73-dimensional continuous state

The original code collapsed 16 lanes of sensor data into 5 coarse pressure
levels, making many distinct traffic states look identical. The 73-dim vector
gives the network independent per-lane information — the difference between
"8 vehicles waiting on N_to_C_1" and "12 vehicles waiting on N_to_C_1" is
preserved, and the agent can learn to respond to it.

### Anti mode-collapse (PPO)

Without entropy control on a 40-action space, PPO collapses to 1–3 actions
within 50 episodes. Three mechanisms work at different timescales:

- **Entropy decay** (episode-level): maintains broad exploration early, allows convergence late
- **Residual ε-greedy** (step-level): permanently injects random actions into any policy
- **n_steps = 1024** (update-level): shorter rollouts prevent a single dominant action from monopolising entire update batches

---

## 📊 Results

Performance from 1-hour simulations at 50% traffic density after full training:

| Controller | Avg Wait (s) | vs Fixed-30s | Queue Length | Vehicles Arrived |
|------------|-------------|-------------|--------------|-----------------|
| Random | ~1700 | +0% | ~41 | ~2510 |
| Fixed-15s | ~548 | −68% | ~25 | ~2669 |
| Fixed-30s | ~1692 | baseline | ~41 | ~2594 |
| Fixed-60s | ~6374 | +277% | ~85 | ~2217 |
| Max-Pressure | ~1050 | −38% | ~34 | ~2660 |
| Actuated | ~3650 | +116% | ~63 | ~2430 |
| Q-Learning (400 ep) | ~900 | −47% | ~28 | ~2690 |
| DQN (700 ep) | ~750 | −56% | ~24 | ~2730 |
| PPO (1200 ep) | ~680 | −60% | ~21 | ~2760 |

> Results are approximate and depend on traffic density and training duration.
> Fixed-15s is the most competitive baseline for this intersection geometry.

---

## 🔧 Troubleshooting

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If `False` or `None` → reinstall PyTorch with the correct CUDA version (see [GPU Setup](#-gpu-setup)).

### RTX 5000 series crash — `no kernel image`

Blackwell (sm_120) requires PyTorch nightly:

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

### SUMO not found / TraCI error

```bash
# Windows
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"

# Linux / macOS
export SUMO_HOME=/usr/share/sumo
```

### `patch_traffic.py` — no attributes found

The script searches for `probability`, `vehsPerHour`, `period`, `frequency`.
Use `--preview` to inspect what the script finds before modifying:

```bash
python patch_traffic.py --rou "SUMO/intersection/intersection.rou.xml" \
                        --density 0.5 --preview
```

### `ValueError` — inhomogeneous shape in `evaluate_agent`

Caused by `phase_counts` (dict) or `phase_log` (list) being passed to `np.mean()`.
Fixed in v3 — only explicit numeric keys are aggregated.
Update to the latest version from the repository.

### Mode collapse — agent uses only 1–3 phases

Monitor `phases_used` and `dominant_pct` in the training log.
If collapse persists beyond episode 300 (PPO), increase in `config.yaml`:

```yaml
ppo:
  entropy_coef_start: 0.8    # was 0.5
  epsilon_residual: 0.10     # was 0.05
```

### Out of memory (OOM)

```yaml
dqn:
  replay_buffer_size: 50000   # reduce from 100000
  batch_size: 32              # reduce from 64

ppo:
  n_steps: 512                # reduce from 1024
```
---

## 🎯 Roadmap
 - Multi-intersection coordination
 - MARL (Multi-Agent RL) for urban networks
 - Imitation learning from expert controllers
 - Model-based planning (MCTS)
 - Real-world deployment on SCATS/SCOOT systems
 - Adversarial robustness testing
 - Explainability analysis (CAM, attention)


## 📚 References

- **Double DQN**: van Hasselt et al., 2016 — *Deep Reinforcement Learning with Double Q-learning*
- **Dueling Networks**: Wang et al., 2016 — *Dueling Network Architectures for Deep Reinforcement Learning*
- **Prioritised Replay**: Schaul et al., 2016 — *Prioritized Experience Replay*
- **PPO**: Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*
- **GAE**: Schulman et al., 2016 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- **SUMO**: Lopez et al., 2018 — *Microscopic Traffic Simulation using SUMO*
- **Max-Pressure**: Varaiya, 2013 — *Max Pressure Control of a Network of Signalized Intersections*

