import torch

data = torch.load('checkpoints/dqn.pt', map_location=torch.device('cpu'))

# Print keys or model state_dict
print(data.keys()) 