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
    
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience replay memory
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize Q-network and optimizer
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # For tracking inventory in evaluation
        self.inventory = []
    
    def remember(self, state, action, reward, next_state, done):
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
    
    def act(self, state):
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
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action values
        with torch.no_grad():
            action_values = self.model(state)
        
        # Return action with highest value
        return torch.argmax(action_values).item()
    
    def replay(self, batch_size):
        """
        Train the model using experiences from memory.
        
        Args:
            batch_size (int): Number of experiences to sample
        """
        # Check if we have enough experiences
        if len(self.memory) < batch_size:
            return
        
        # Sample random experiences
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state).unsqueeze(0)
            
            # If next_state is None (episode end), use zeros
            if next_state is not None:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Calculate target Q value
            target = reward
            if not done and next_state is not None:
                with torch.no_grad():
                    # Q(s', a') for all actions
                    next_q_values = self.model(next_state)
                    # Max Q(s', a')
                    max_next_q = torch.max(next_q_values).item()
                    # Q(s, a) = r + γ * max Q(s', a')
                    target = reward + self.gamma * max_next_q
            
            # Current Q values
            current_q = self.model(state)
            
            # Create target vector for all actions
            target_f = current_q.clone().detach()
            
            # Update target for the action we took
            target_f[0][action] = target
            
            # Zero gradients, perform backprop, and update weights
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_f)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """
        Save the model weights.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath):
        """
        Load model weights.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model.load_state_dict(torch.load(filepath))