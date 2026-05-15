"""Q-networks: MLP on flat 16-dim board, or small CNN on 4×4 spatial layout."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP Q-function. Each hidden block: Linear → optional LayerNorm → ReLU."""

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm = layer_norm

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config_dict(self) -> dict[str, int | bool | str]:
        return {
            "qnet_type": "mlp",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "layer_norm": self.layer_norm,
        }

    @staticmethod
    def from_config(cfg: dict) -> "QNetwork":
        return QNetwork(
            state_dim=int(cfg.get("state_dim", 16)),
            action_dim=int(cfg.get("action_dim", 4)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_hidden_layers=int(cfg.get("num_hidden_layers", 2)),
            layer_norm=bool(cfg.get("layer_norm", False)),
        )


class BoardCNNQNetwork(nn.Module):
    """Spatial conv stack on log2 board (B,16) reshaped to (B,1,4,4)."""

    def __init__(
        self,
        *,
        state_dim: int = 16,
        action_dim: int = 4,
        channels: tuple[int, ...] = (64, 128, 128),
    ) -> None:
        super().__init__()
        if state_dim != 16:
            raise ValueError("BoardCNNQNetwork expects state_dim=16 (4×4 board)")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.channels = channels
        c0, c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c1, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(c2, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != 16:
            raise ValueError(f"expected (B,16), got {tuple(x.shape)}")
        h = x.view(x.shape[0], 1, 4, 4)
        h = self.conv(h)
        h = h.flatten(1)
        return self.head(h)

    def config_dict(self) -> dict[str, int | bool | str | tuple[int, ...]]:
        return {
            "qnet_type": "cnn",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cnn_channels": self.channels,
        }

    @staticmethod
    def from_config(cfg: dict) -> "BoardCNNQNetwork":
        ch = cfg.get("cnn_channels")
        if isinstance(ch, (list, tuple)) and len(ch) == 3:
            channels = (int(ch[0]), int(ch[1]), int(ch[2]))
        else:
            channels = (64, 128, 128)
        return BoardCNNQNetwork(
            state_dim=int(cfg.get("state_dim", 16)),
            action_dim=int(cfg.get("action_dim", 4)),
            channels=channels,
        )


def build_qnetwork_from_config(cfg: dict[str, Any]) -> nn.Module:
    qtype = str(cfg.get("qnet_type", "mlp")).lower()
    if qtype == "cnn":
        return BoardCNNQNetwork.from_config(cfg)
    return QNetwork.from_config(cfg)


def load_qnetwork_from_checkpoint(
    ck: dict[str, object] | object,
    device: torch.device | str,
) -> nn.Module:
    """Instantiate Q-network (MLP or CNN) from checkpoint dict or legacy raw state_dict."""
    dev = torch.device(device) if isinstance(device, str) else device
    if isinstance(ck, dict) and "policy_state" in ck:
        sd = ck["policy_state"]
        qcfg = ck.get("qnet_config")
    else:
        sd = ck  # type: ignore[assignment]
        qcfg = None
    if qcfg is not None and isinstance(qcfg, dict):
        net = build_qnetwork_from_config(qcfg)
    else:
        net = QNetwork()
    net.load_state_dict(sd)  # type: ignore[arg-type]
    net.to(dev)
    net.eval()
    return net
