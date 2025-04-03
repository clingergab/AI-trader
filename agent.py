import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQN


class DQNAgent:
    """
    Deep Q-Network Agent for trading.
    This agent uses a Deep Q-Network to learn optimal trading strategies.
    """
    
    def __init__(self, state_size: int, action_size: int) -> None:
        """
        Initialize the DQN agent.
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Determine device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Experience replay memory
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize Q-network and optimizer
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # For tracking inventory in evaluation
        self.inventory = []
    
        # Enable cudnn benchmarking for faster performance
        torch.backends.cudnn.benchmark = True
    
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store experience in memory.
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose an action based on the current state.
        Args:
            state (numpy.array): Current state
        Returns:
            int: Action to take
        """
        # Epsilon-greedy exploration strategy
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor for model input
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # print(f"Input tensor device: {state.device}")  # Debugging line
        
        # Get action values
        with torch.no_grad():
            action_values = self.model(state)
            # action = torch.argmax(action_values).item()
            # print(f"State shape: {state.shape}, Q-values: {action_values.cpu().numpy()[0]}, Selected action: {action}")
        
        # Return action with highest value
        return torch.argmax(action_values).item()
    
  
    def replay(self, batch_size: int) -> None:
        """
        Train the model using experiences from memory.
        Args:
            batch_size (int): Number of experiences to sample
        """
        # Check if we have enough experiences
        if len(self.memory) < batch_size:
            return
            
        # Sample random experiences
        batch = random.sample(self.memory, batch_size)
        
        # Pre-allocate arrays for better performance
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size)
        
        # Fill the arrays
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            
            if next_state is not None:
                next_states[i] = next_state
            
            dones[i] = float(done)
        
        # Convert to tensors in one go (much more efficient)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update model
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
    
    def save(self, filepath: str) -> None:
        """
        Save the model weights.
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load model weights.
        
        Args:
            filepath (str): Path to load the model from
        """
        # self.model.load_state_dict(torch.load(filepath))
         # Load the state dict to CPU first, then move to the correct device
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

