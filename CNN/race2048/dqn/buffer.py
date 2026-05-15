"""Experience replay buffer for DQN (1-step or n-step)."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray  # (state_dim,), float32
    action: int
    reward: float  # n-step discounted return sum when n>1
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray  # (4,) bool — False for illegal moves at s'
    bootstrap_discount: float  # multiplier on Q(s'); γ for 1-step, γ^k for n-step


class ReplayBuffer:
    """FIFO buffer; uniform random sampling for i.i.d. mini-batches."""

    def __init__(self, capacity: int, *, gamma: float = 0.99) -> None:
        self._capacity = capacity
        self._gamma = float(gamma)
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
                bootstrap_discount=self._gamma,
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> tuple[
        torch.Tensor,
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
        bootstrap = torch.tensor(
            [t.bootstrap_discount for t in batch],
            dtype=torch.float32,
            device=device,
        )
        return states, actions, rewards, next_states, dones, next_masks, bootstrap


class NStepReplayBuffer(ReplayBuffer):
    """n-step returns with correct bootstrap γ^k (truncates at episode boundary)."""

    def __init__(self, capacity: int, *, gamma: float = 0.99, n: int = 3) -> None:
        super().__init__(capacity, gamma=gamma)
        self._n = max(1, int(n))
        self._pending: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque()

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        if self._n <= 1:
            return super().push(
                state, action, reward, next_state, done, next_legal_mask
            )
        self._pending.append(
            (
                np.asarray(state, dtype=np.float32).ravel(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).ravel(),
                bool(done),
                np.asarray(next_legal_mask, dtype=np.bool_).reshape(4),
            )
        )
        if done:
            while len(self._pending) > 0:
                self._emit_chunk()
        elif len(self._pending) >= self._n:
            self._emit_chunk()

    def _horizon(self) -> int:
        cap = min(self._n, len(self._pending))
        for i in range(cap):
            if self._pending[i][4]:
                return i + 1
        return cap

    def _emit_chunk(self) -> None:
        k = self._horizon()
        if k <= 0:
            return
        chunk = [self._pending[i] for i in range(k)]
        for _ in range(k):
            self._pending.popleft()

        s0, a0 = chunk[0][0], chunk[0][1]
        r_sum = sum((self._gamma**i) * chunk[i][2] for i in range(k))
        last_s2, last_m = chunk[-1][3], chunk[-1][5]
        done_any = any(chunk[i][4] for i in range(k))
        boot = self._gamma**k

        self._memory.append(
            Transition(
                state=s0,
                action=a0,
                reward=float(r_sum),
                next_state=last_s2,
                done=bool(done_any),
                next_legal_mask=last_m,
                bootstrap_discount=float(boot),
            )
        )
