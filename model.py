import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network for trading.
    This neural network takes a state as input and outputs Q-values for each action.
    """
    
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN model.
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Define network architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input state
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)