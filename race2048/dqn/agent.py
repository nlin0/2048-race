"""DQN agent: ε-greedy exploration + periodic target-network sync."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from race2048.dqn.buffer import ReplayBuffer
from race2048.dqn.qnet import QNetwork


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class DQNAgent:
    """Deep Q-learning with a target network and experience replay."""

    def __init__(
        self,
        *,
        state_dim: int = 16,
        action_dim: int = 4,
        device: torch.device | str | None = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 100_000,
        target_update_every: int = 1000,
        grad_clip: float | None = 10.0,
    ) -> None:
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._gamma = gamma
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay_steps = max(1, eps_decay_steps)
        self._target_update_every = target_update_every
        self._grad_clip = grad_clip

        self.policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self._device
        )
        self.target_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self._device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.train_steps = 0

    def epsilon(self, global_step: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps."""
        t = min(1.0, global_step / self._eps_decay_steps)
        return self._eps_start + t * (self._eps_end - self._eps_start)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        global_step: int,
    ) -> int:
        """ε-greedy among legal moves only."""
        mask = np.asarray(legal_mask, dtype=np.bool_).reshape(4)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        eps = self.epsilon(global_step)
        if random.random() < eps:
            return int(random.choice(legal))

        x = torch.from_numpy(_flatten_obs(obs)).unsqueeze(0).to(self._device)
        self.policy_net.eval()
        q = self.policy_net(x).squeeze(0).clone()
        m = torch.from_numpy(mask).to(self._device)
        q[~m] = -float("inf")
        return int(q.argmax().item())

    def train_step(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
    ) -> float | None:
        """One gradient step. Returns loss or None if buffer too small."""
        if len(buffer) < batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones, next_masks = buffer.sample(
            batch_size, self._device
        )

        q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nq = self.target_net(next_states)
            nq = nq.masked_fill(~next_masks, float("-inf"))
            next_q = nq.max(1).values
            targets = rewards + (1.0 - dones) * self._gamma * next_q

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self._grad_clip
            )
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self._target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def sync_target(self) -> None:
        """Hard-copy weights to target network (e.g. end of episode)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
