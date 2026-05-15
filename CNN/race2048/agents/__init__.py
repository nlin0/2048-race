"""Baseline policies for comparison and scripted play."""

from race2048.agents.base import Agent
from race2048.agents.baselines import GreedyEmptyAgent, OrderedAgent, RandomAgent

__all__ = [
    "Agent",
    "RandomAgent",
    "OrderedAgent",
    "GreedyEmptyAgent",
]
