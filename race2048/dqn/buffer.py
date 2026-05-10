"""Experience replay buffer for DQN."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray  # (state_dim,), float32
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray  # (4,) bool — False for illegal moves at s'


class ReplayBuffer:
    """FIFO buffer; uniform random sampling for i.i.d. mini-batches."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._memory: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        self._memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).ravel(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).ravel(),
                done=bool(done),
                next_legal_mask=np.asarray(next_legal_mask, dtype=np.bool_).reshape(4),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self._memory, batch_size)
        states = torch.stack(
            [torch.from_numpy(t.state) for t in batch]
        ).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.stack(
            [torch.from_numpy(t.next_state) for t in batch]
        ).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        next_masks = torch.tensor(
            np.stack([t.next_legal_mask.astype(np.bool_) for t in batch]),
            dtype=torch.bool,
            device=device,
        )
        return states, actions, rewards, next_states, dones, next_masks
