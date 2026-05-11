import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Input: 4x4x16 (16 one-hot channels for each tile value)
        self.conv1 = nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
        
        # After conv layers: 2x2x128 = 512 features
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 outputs (one Q-value per action)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 16, 4, 4)
        x = self.relu(self.conv1(x))  # -> (batch, 128, 3, 3)
        x = self.relu(self.conv2(x))  # -> (batch, 128, 2, 2)
        x = x.view(x.size(0), -1)     # -> (batch, 512)
        x = self.relu(self.fc1(x))    # -> (batch, 256)
        x = self.fc2(x)                # -> (batch, 4)
        return x

def board_to_tensor(board):
    """Convert a 4x4 board into a 16-channel one-hot tensor"""
    # Tile values: 0, 2, 4, 8, 16, ..., 2048, 4096
    # Map to indices: 0, 1, 2, 3, 4,  ..., 11,   12
    tensor = np.zeros((16, 4, 4), dtype=np.float32)
    
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val == 0:
                channel = 0
            else:
                # log2(val) gives: 2->1, 4->2, 8->3, ..., 2048->11
                channel = int(np.log2(val))
            tensor[channel, i, j] = 1.0
    
    return tensor