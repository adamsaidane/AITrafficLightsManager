"""
dqn_agent.py — Double Dueling DQN with Prioritised Experience Replay.

Architecture
------------
    input → Linear(256) → ReLU → Linear(256) → ReLU
                                              ↓
                           ┌─────────────────────────────┐
                           │  Value stream  V(s)          │   Linear(256→1)
                           │  Advantage stream  A(s,a)    │   Linear(256→|A|)
                           └─────────────────────────────┘
                                   Q(s,a) = V(s) + A(s,a) - mean(A)

Why Double + Dueling + PER?
-  Double DQN: decouples action selection from evaluation → reduces overestimation.
-  Dueling: separates state-value from action-advantage → better generalisation
   across actions, especially when many actions share similar values.
-  PER: focuses replay on surprising/high-error transitions → faster learning.
"""

from __future__ import annotations

import os
import random
import math
import collections
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.gpu_config import DEVICE


# =========================================================================
# Neural Network
# =========================================================================

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage head
        self.adv_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        v = self.value_head(shared)
        a = self.adv_head(shared)
        return v + a - a.mean(dim=1, keepdim=True)


# =========================================================================
# Prioritised Replay Buffer
# =========================================================================

class PrioritisedReplayBuffer:
    """
    Proportional Prioritised Experience Replay (Schaul et al., 2016).
    Uses a simple list + segment-tree-style sampling via numpy.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
        self.priorities: collections.deque = collections.deque(maxlen=capacity)
        self.max_priority: float = 1.0

    def push(self, transition: Tuple) -> None:
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + 1e-6)
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return len(self.buffer)


# =========================================================================
# Agent
# =========================================================================

class DQNAgent(BaseAgent):
    """
    Double Dueling DQN with Prioritised Replay.

    Fixes applied vs v1:
    - Hard target-net sync every `target_hard_sync` episodes (prevents drift)
    - Learning is frozen for the first `min_replay_size` steps (no updates on
      tiny, biased buffer)
    - Running reward normalisation (mean/std tracked online) so the network
      sees a stable [-3, +3] reward range regardless of scaling
    - `update_every` flag: only update every N steps to decouple collection
      from learning speed
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict) -> None:
        dqn_cfg = config["dqn"]
        net_cfg  = config["network"]

        self.action_dim  = action_dim
        self.gamma       = dqn_cfg["gamma"]
        self.batch_size  = dqn_cfg["batch_size"]
        self.min_replay_size = dqn_cfg["min_replay_size"]
        self.grad_clip   = dqn_cfg["gradient_clip"]
        self.tau         = dqn_cfg["tau"]
        self.double_dqn  = dqn_cfg["double_dqn"]
        self.update_every = 4   # update network every 4 environment steps

        # Hard sync target net every N episodes (new: prevents slow drift)
        self.target_hard_sync = 50

        # Epsilon — linear decay over epsilon_decay_steps
        self.epsilon       = dqn_cfg["epsilon_start"]
        self.epsilon_end   = dqn_cfg["epsilon_end"]
        self.epsilon_decay_steps = dqn_cfg["epsilon_decay_steps"]
        self._step = 0

        # Beta for PER IS weights
        self._beta_start = 0.4
        self._beta_steps = dqn_cfg["epsilon_decay_steps"]

        self.device = DEVICE   # centralised GPU selection
        hidden = net_cfg["hidden_sizes"]

        self.online_net = DuelingDQN(obs_dim, action_dim, hidden).to(self.device)
        self.target_net = DuelingDQN(obs_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(),
                                    lr=dqn_cfg["learning_rate"], eps=1e-5)
        self.replay = PrioritisedReplayBuffer(dqn_cfg["replay_buffer_size"])

        # Running reward normalisation (Welford's algorithm)
        self._rew_mean = 0.0
        self._rew_var  = 1.0
        self._rew_n    = 0
        self._episode_count = 0

    # ------------------------------------------------------------------ #

    @property
    def beta(self) -> float:
        frac = min(self._step / max(self._beta_steps, 1), 1.0)
        return self._beta_start + frac * (1.0 - self._beta_start)

    def _update_epsilon(self) -> None:
        frac = min(self._step / max(self.epsilon_decay_steps, 1), 1.0)
        self.epsilon = max(
            self.epsilon_end,
            1.0 - frac * (1.0 - self.epsilon_end)
        )

    def _normalise_reward(self, r: float) -> float:
        """Online Welford normalisation — keeps reward in ~[-3, +3]."""
        self._rew_n += 1
        old_mean = self._rew_mean
        self._rew_mean += (r - old_mean) / self._rew_n
        self._rew_var  += (r - old_mean) * (r - self._rew_mean)
        std = max(np.sqrt(self._rew_var / max(self._rew_n, 1)), 1e-4)
        return float(np.clip((r - self._rew_mean) / std, -5.0, 5.0))

    def on_episode_end(self) -> None:
        """Call at end of each episode to trigger hard target sync if due."""
        self._episode_count += 1
        if self._episode_count % self.target_hard_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------ #
    # BaseAgent interface
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        self._step += 1
        self._update_epsilon()
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(s)
            return int(q.argmax(dim=1).item())

    def get_greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done) -> None:
        norm_r = self._normalise_reward(reward)
        self.replay.push((state, action, norm_r, next_state, done))

    def update(self, *args, **kwargs) -> dict:  # type: ignore[override]
        # Only update every N steps (decouples collection from learning)
        if self._step % self.update_every != 0:
            return {}
        if len(self.replay) < self.min_replay_size:
            return {}

        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay.sample(self.batch_size, self.beta)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        w  = torch.FloatTensor(weights).to(self.device)

        # Current Q
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q (Double DQN)
        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
                next_q = self.target_net(ns).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(ns).max(dim=1)[0]
            target_q = r + self.gamma * next_q * (1.0 - d)

        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        # Weighted Huber loss
        loss = (w * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft target network update
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        return {"loss": loss.item(), "mean_td_error": float(np.mean(np.abs(td_errors)))}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_state_dict":    self.online_net.state_dict(),
            "target_state_dict":    self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon":   self.epsilon,
            "step":      self._step,
            "rew_mean":  self._rew_mean,
            "rew_var":   self._rew_var,
            "rew_n":     self._rew_n,
            "ep_count":  self._episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon         = ckpt["epsilon"]
        self._step           = ckpt["step"]
        self._rew_mean       = ckpt.get("rew_mean", 0.0)
        self._rew_var        = ckpt.get("rew_var",  1.0)
        self._rew_n          = ckpt.get("rew_n",    0)
        self._episode_count  = ckpt.get("ep_count", 0)
