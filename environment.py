from typing import Optional, Tuple, Dict, Union
import numpy as np
from pandas import DataFrame


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    This environment allows an agent to interact with historical stock market data
    and make trading decisions (buy, sell, or hold).
    """
    
    # Define possible actions
    HOLD = 0
    BUY = 1
    SELL = 2
    
    def __init__(self, data: DataFrame, window_size: int=1, initial_balance: int=10000) -> None:
        """
        Initialize the trading environment.
        Args:
            data (pandas.DataFrame): Historical stock data with technical indicators
            window_size (int): Number of time steps to include in the state
            initial_balance (float): Starting balance for the trading simulation
        """
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        self.balance = initial_balance
        self.holdings = 0
        self.index = 0
        self.done = False
        self.actions_history = []
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns:
            numpy.array: Initial state representation
        """
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        self.done = False
        self.actions_history = []
        
        return self.get_state(self.index)

    def step(self, action: int)-> Tuple[Optional[np.ndarray], float, bool, Dict[str, Union[float, int]]]:
        """Execute one step in the environment.
        Args:
            action (int): Action to take (0: HOLD, 1: BUY, 2: SELL)
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: Next state representation
                - reward: Reward for the action taken
                - done: Whether the episode is finished
                - info: Additional information (dictionary)"""
                
        # Record the action
        self.actions_history.append(action)
        
        # Get current price
        # price = float(self.data.iloc[self.index]['Close'].iloc[0])
        try:
        # Try direct access first (if Close is a column with values)
            price = float(self.data['Close'].iloc[self.index])
        except (TypeError, AttributeError):
            try:
                # Try iloc access if the above fails
                price = float(self.data.iloc[self.index, self.data.columns.get_loc('Close')])
            except:
                # Last resort - treat as a multi-index structure
                price = float(self.data.iloc[self.index]['Close'])
        reward = 0
        
        # Execute action
        if action == self.BUY and self.balance >= price:
            # Calculate maximum shares that can be purchased
            max_shares = self.balance // price
            self.holdings += max_shares
            self.balance -= max_shares * price
            # Small positive reward for valid BUY
            reward += 1
        elif action == self.SELL and self.holdings > 0:
            # Sell all holdings
            self.balance += self.holdings * price
            self.holdings = 0
            # Small positive reward for valid SELL
            reward += 1
        elif action == self.SELL and self.holdings <= 0:
            # Negative reward for invalid SELL attempt
            reward -= 5  # Penalize trying to sell with no holdings
        
        # Move to the next time step
        self.index += 1
        
        # Check if we've reached the end of the data
        self.done = self.index >= len(self.data) - 1
        
        # Calculate reward at the end of the episode
        if self.done:
            # Final portfolio value (cash + holdings)
            final_value = self.balance
            if self.holdings > 0:
                final_value += self.holdings * price
                
            # Reward is the profit/loss
            reward = final_value - self.initial_balance
        
        # Get the next state
        next_state = self.get_state(self.index) if not self.done else None
        
        # Additional info
        info = {
            'balance': self.balance,
            'holdings': self.holdings,
            'price': price
        }
        
        return next_state, reward, self.done, info

    def get_state(self, index: int) -> np.ndarray:
        """
        Create a state representation for the agent based on current market data.
        Args:
            index (int): Current time index
        Returns:
            numpy.array: State representation for the agent
        """
        if index - self.window_size + 1 < 0:
            # Not enough historical data, pad with zeros
            padding = -1 * (index - self.window_size + 1)
            market_state = np.zeros((self.window_size, 4))
            data_window = self.data.iloc[0:index+1]
        else:
            padding = 0
            data_window = self.data.iloc[index-self.window_size+1:index+1]
        
        # Fast vectorized approach to create market state
        close_values = data_window['Close'].values
        sma5_values = data_window['SMA_5'].values
        sma20_values = data_window['SMA_20'].values
        returns_values = data_window['Returns'].values
        
        # Create state using vectorized operations
        market_state = np.column_stack([
            close_values, sma5_values, sma20_values, returns_values
        ])
        
        # If padding was needed, create a properly padded array
        if padding > 0:
            padded_state = np.zeros((self.window_size, 4))
            padded_state[padding:] = market_state
            market_state = padded_state
        
        # Add position info
        position_info = np.array([float(self.holdings > 0), float(self.balance / self.initial_balance)])
        
        # Combine market data and position info
        combined_state = np.concatenate([market_state.flatten(), position_info])
        
        return combined_state
    
    def get_actions_history(self) -> list:
        """
        Get the history of actions taken during the episode.
        
        Returns:
            list: List of actions taken
        """
        return self.actions_history