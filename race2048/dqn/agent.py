import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from race2048.dqn.network import DQN, board_to_tensor
from race2048.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        replay_capacity=50000,
        batch_size=64,
        target_update_freq=250,
    ):
        self.device = torch.device("cpu")

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=replay_capacity)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

    def select_action(self, board, training=True, legal_actions=None):
        """
        Epsilon-greedy action selection.

        legal_actions is important for 2048 because invalid/no-op moves waste turns.
        If legal_actions is provided, the agent only chooses among valid moves.
        """
        if legal_actions is None:
            legal_actions = [0, 1, 2, 3]

        legal_actions = list(legal_actions)
        if len(legal_actions) == 0:
            return 0

        # Explore: random legal action
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # Exploit: best Q-value among legal actions
        state = board_to_tensor(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)

        mask = torch.full_like(q_values, -float("inf"))
        mask[legal_actions] = 0.0
        q_values = q_values + mask

        return int(q_values.argmax().item())

    def store_experience(self, state, action, reward, next_state, done):
        state_enc = board_to_tensor(state)
        next_state_enc = board_to_tensor(next_state)
        self.memory.push(state_enc, action, reward, next_state_enc, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Huber loss is usually more stable than plain MSE for DQN
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Prevent giant updates from destabilizing learning
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.steps = checkpoint.get("steps", self.steps)