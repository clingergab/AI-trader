import numpy as np

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
    
    def __init__(self, data, window_size=1, initial_balance=10000):
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
        
    def reset(self):
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
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take (0: HOLD, 1: BUY, 2: SELL)
            
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: Next state representation
                - reward: Reward for the action taken
                - done: Whether the episode is finished
                - info: Additional information (dictionary)
        """
        # Record the action
        self.actions_history.append(action)
        
        # Get current price
        price = float(self.data.iloc[self.index]['Close'].iloc[0])
        reward = 0
        
        # Execute action
        if action == self.BUY and self.balance >= price:
            # Calculate maximum shares that can be purchased
            max_shares = self.balance // price
            self.holdings += max_shares
            self.balance -= max_shares * price
            
        elif action == self.SELL and self.holdings > 0:
            # Sell all holdings
            self.balance += self.holdings * price
            self.holdings = 0
        
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
    
    def get_state(self, index):
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
            data_window = self.data.iloc[0:index+1]
            # Create state array with the correct shape
            state = np.zeros((self.window_size, 4))
            
            # Fill in the available data
            available_data = len(data_window)
            if available_data > 0:
                # Extract feature values
                close_values = data_window['Close'].values
                sma5_values = data_window['SMA_5'].values
                sma20_values = data_window['SMA_20'].values
                returns_values = data_window['Returns'].values
                
                # Fill in the state array
                for i in range(available_data):
                    pos = padding + i
                    state[pos, 0] = close_values[i]
                    state[pos, 1] = sma5_values[i]
                    state[pos, 2] = sma20_values[i]
                    state[pos, 3] = returns_values[i]
        else:
            # We have enough data
            data_window = self.data.iloc[index-self.window_size+1:index+1]
            
            # Create state directly from the values
            closes = data_window['Close'].values
            sma5s = data_window['SMA_5'].values
            sma20s = data_window['SMA_20'].values
            returns = data_window['Returns'].values
            
            # Ensure all arrays have the same length (should be equal to window_size)
            min_length = min(len(closes), len(sma5s), len(sma20s), len(returns), self.window_size)
            
            # Create state array
            state = np.zeros((self.window_size, 4))
            for i in range(min_length):
                state[i, 0] = closes[i]
                state[i, 1] = sma5s[i]
                state[i, 2] = sma20s[i]
                state[i, 3] = returns[i]
        
        return state.flatten()
    
    def get_actions_history(self):
        """
        Get the history of actions taken during the episode.
        
        Returns:
            list: List of actions taken
        """
        return self.actions_history