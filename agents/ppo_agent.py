"""
ppo_agent.py — Proximal Policy Optimisation (Schulman et al., 2017).

Uses Generalised Advantage Estimation (GAE) and a shared actor-critic backbone.

Why PPO for traffic?
-  On-policy: naturally handles the non-stationary traffic distribution.
-  Clipped objective: robust, stable updates without the brittle replay buffer.
-  Continuous improvement: good for long-horizon traffic optimisation.
-  Entropy bonus: prevents premature convergence to one dominant phase.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent


# =========================================================================
# Actor-Critic Network
# =========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)

        # Orthogonal init (common for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value

    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# =========================================================================
# Rollout buffer
# =========================================================================

class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, state, action, reward, log_prob, value, done) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.__init__()


# =========================================================================
# PPO Agent
# =========================================================================

class PPOAgent(BaseAgent):
    """Proximal Policy Optimisation with GAE."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict) -> None:
        ppo_cfg = config["ppo"]
        net_cfg = config["network"]

        self.gamma = ppo_cfg["gamma"]
        self.lam = ppo_cfg["lam"]
        self.clip_eps = ppo_cfg["clip_eps"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.value_coef = ppo_cfg["value_coef"]
        self.update_epochs = ppo_cfg["update_epochs"]
        self.n_steps = ppo_cfg["n_steps"]
        self.mini_batch_size = ppo_cfg["mini_batch_size"]
        self.grad_clip = ppo_cfg["gradient_clip"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, action_dim, net_cfg["hidden_sizes"]).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=ppo_cfg["learning_rate"],
                                    eps=1e-5)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.net.act(s)
        # Store for later update
        self._last = (log_prob.item(), value.item())
        return int(action.item())

    def store(self, state, action, reward, done) -> None:
        log_prob, value = self._last
        self.buffer.push(state, action, reward, log_prob, value, done)

    def get_greedy_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(s)
            return int(logits.argmax(dim=1).item())

    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        values_ext = np.append(values, last_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, last_state: np.ndarray = None, **kwargs) -> dict:  # type: ignore[override]
        if len(self.buffer) == 0:
            return {}

        last_value = 0.0
        if last_state is not None:
            s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, v = self.net(s)
                last_value = v.item()

        advantages, returns = self._compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        adv = torch.FloatTensor(advantages).to(self.device)
        ret = torch.FloatTensor(returns).to(self.device)

        n = len(states)
        total_loss, total_pg, total_val, total_ent = 0.0, 0.0, 0.0, 0.0
        update_count = 0

        for _ in range(self.update_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.mini_batch_size):
                idx = perm[start: start + self.mini_batch_size]
                logits, values = self.net(states[idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                val_loss = nn.functional.mse_loss(values, ret[idx])
                loss = pg_loss + self.value_coef * val_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_val += val_loss.item()
                total_ent += entropy.item()
                update_count += 1

        self.buffer.clear()
        denom = max(update_count, 1)
        return {
            "loss": total_loss / denom,
            "pg_loss": total_pg / denom,
            "val_loss": total_val / denom,
            "entropy": total_ent / denom,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
