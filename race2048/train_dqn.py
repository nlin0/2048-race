"""Train CNN-based DQN agent for 2048.

This is a port of Tim's original training loop into the GUI repo.
It keeps the original simple training style but adds:
- legal action masking
- max move cap per episode
- graph output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from race2048.board import Game2048
from race2048.dqn.agent import DQNAgent


def max_tile(board: np.ndarray) -> int:
    return int(board.max())


def score_proxy(board: np.ndarray) -> int:
    return int(board.sum())


def train(
    num_episodes: int = 2000,
    save_interval: int = 200,
    model_path: str = "checkpoints/dqn_cnn_originalstyle.pt",
    load_path: str | None = None,
    graph_path: str = "training_curves.png",
    max_moves_per_episode: int = 1000,
):
    os.makedirs(Path(model_path).parent, exist_ok=True)

    agent = DQNAgent(
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.997,
    )

    if load_path is not None:
        print(f"Loading checkpoint from {load_path}")
        agent.load(load_path)

    env = Game2048()

    episode_rewards = []
    episode_max_tiles = []
    episode_lengths = []
    episode_losses = []

    win_count = 0

    print("Starting training...")
    print(f"Episodes: {num_episodes}")
    print(f"Save interval: every {save_interval} episodes")
    print(f"Max moves per episode: {max_moves_per_episode}\n")

    for episode in range(1, num_episodes + 1):
        board = env.reset()
        episode_reward = 0.0
        done = False
        moves = 0
        losses_this_episode = []

        while not done and moves < max_moves_per_episode:
            legal_actions = [int(a) for a in env.legal_actions()]
            if not legal_actions:
                break

            action = agent.select_action(
                board,
                training=True,
                legal_actions=legal_actions,
            )

            result = env.step(action)

            next_board = result.board
            reward = float(next_board.sum() - board.sum())
            done = bool(result.game_over)

            agent.store_experience(board, action, reward, next_board, done)
            loss = agent.train_step()

            if loss is not None:
                losses_this_episode.append(loss)

            episode_reward += reward
            board = next_board
            moves += 1

        agent.decay_epsilon()

        tile = max_tile(board)
        episode_rewards.append(episode_reward)
        episode_max_tiles.append(tile)
        episode_lengths.append(moves)

        if losses_this_episode:
            episode_losses.append(float(np.mean(losses_this_episode)))
        else:
            episode_losses.append(None)

        if tile >= 2048:
            win_count += 1

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_max_tile = np.mean(episode_max_tiles[-50:])
            avg_len = np.mean(episode_lengths[-50:])
            win_rate = (win_count / episode) * 100

            print(
                f"Episode {episode}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Avg Max Tile: {avg_max_tile:.1f} | "
                f"Avg Len: {avg_len:.1f} | "
                f"Win Rate: {win_rate:.2f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if episode % save_interval == 0:
            agent.save(model_path)
            print(f"  → Model saved at episode {episode}")

    agent.save(model_path)
    print(f"\nTraining complete! Final model saved to {model_path}")

    plot_training_curves(
        rewards=episode_rewards,
        max_tiles=episode_max_tiles,
        lengths=episode_lengths,
        losses=episode_losses,
        win_count=win_count,
        num_episodes=num_episodes,
        output_path=graph_path,
    )

    return agent


def moving_avg(values, window=50):
    return [
        np.mean(values[max(0, i - window + 1): i + 1])
        for i in range(len(values))
    ]


def plot_training_curves(
    rewards,
    max_tiles,
    lengths,
    losses,
    win_count,
    num_episodes,
    output_path,
):
    avg_rewards = moving_avg(rewards, 50)
    avg_max_tiles = moving_avg(max_tiles, 50)
    avg_lengths = moving_avg(lengths, 50)

    clean_losses = [x for x in losses if x is not None]
    avg_losses = moving_avg(clean_losses, 50) if clean_losses else []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(avg_rewards, linewidth=1.5)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].set_title("Training Reward Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(avg_max_tiles, linewidth=1.5)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average Max Tile")
    axes[0, 1].set_title("Max Tile Progress Over Time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(avg_lengths, linewidth=1.5)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Average Moves")
    axes[1, 0].set_title("Game Length Over Time")
    axes[1, 0].grid(True, alpha=0.3)

    if avg_losses:
        axes[1, 1].plot(avg_losses, linewidth=1.5)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Average Loss")
    axes[1, 1].set_title("Training Loss Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nTraining curves saved to {output_path}")
    print(f"Total wins/reached 2048: {win_count}/{num_episodes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--save", type=str, default="checkpoints/dqn_cnn_originalstyle.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--graph", type=str, default="training_curves.png")
    parser.add_argument("--max-moves", type=int, default=1000)

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        model_path=args.save,
        load_path=args.load,
        graph_path=args.graph,
        max_moves_per_episode=args.max_moves,
    )


if __name__ == "__main__":
    main()