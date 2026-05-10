"""Agent protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int: ...
