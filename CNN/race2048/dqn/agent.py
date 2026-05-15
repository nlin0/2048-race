"""DQN agent: ε-greedy exploration + periodic target-network sync."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from race2048.dqn.buffer import ReplayBuffer
from race2048.dqn.qnet import BoardCNNQNetwork, QNetwork


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class DQNAgent:
    """Deep Q-learning with a target network and experience replay."""

    def __init__(
        self,
        *,
        state_dim: int = 16,
        action_dim: int = 4,
        qnet_type: str = "mlp",
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        layer_norm: bool = False,
        device: torch.device | str | None = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        target_update_every: int = 1000,
        grad_clip: float | None = 10.0,
        double_dqn: bool = True,
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
        self._double_dqn = double_dqn
        self._epsilon_step_offset = 0

        qkind = str(qnet_type).lower()
        if qkind == "cnn":
            self.policy_net = BoardCNNQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self._device)
            self.target_net = BoardCNNQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self._device)
        else:
            self.policy_net = QNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                layer_norm=layer_norm,
            ).to(self._device)
            self.target_net = QNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                layer_norm=layer_norm,
            ).to(self._device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.train_steps = 0

    def set_epsilon_offset(self, env_steps_at_load: int) -> None:
        """Treat ε-schedule as if global env step were reduced (e.g. resume fine-tune)."""
        self._epsilon_step_offset = max(0, int(env_steps_at_load))

    def epsilon(self, global_step: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps."""
        t = min(
            1.0,
            max(0, global_step - self._epsilon_step_offset) / self._eps_decay_steps,
        )
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
        states, actions, rewards, next_states, dones, next_masks, bootstrap = (
            buffer.sample(batch_size, self._device)
        )

        q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self._double_dqn:
                pq = self.policy_net(next_states)
                pq = pq.masked_fill(~next_masks, float("-inf"))
                best = pq.argmax(1, keepdim=True)
                nq = self.target_net(next_states)
                next_q = nq.gather(1, best).squeeze(1)
            else:
                nq = self.target_net(next_states)
                nq = nq.masked_fill(~next_masks, float("-inf"))
                next_q = nq.max(1).values
            has_legal = next_masks.any(dim=1)
            next_q = torch.where(has_legal, next_q, torch.zeros_like(next_q))
            targets = rewards + (1.0 - dones) * bootstrap * next_q

        loss = F.smooth_l1_loss(q_sa, targets)
        loss_f = float(loss.item())
        if not math.isfinite(loss_f):
            return None

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

        return loss_f

    @torch.no_grad()
    def replay_q_stats(
        self, buffer: ReplayBuffer, batch_size: int
    ) -> dict[str, float] | None:
        """One-batch Q diagnostics for logging (optional)."""
        if len(buffer) < batch_size:
            return None
        states, actions, _, _, _, _, _ = buffer.sample(batch_size, self._device)
        self.policy_net.eval()
        q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        return {
            "q_sa_mean": float(q_sa.mean().item()),
            "q_abs_max": float(q_sa.abs().max().item()),
        }

    def sync_target(self) -> None:
        """Hard-copy weights to target network (e.g. end of episode)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
