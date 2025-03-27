# AI-trader

An AI trading agent using Deep Q-Learning (DQN), a reinforcement learning algorithm that learns to make trading decisions (buy, sell, or hold) based on historical stock market data.  

### Trading Environment  
A simulated stock market environment where the agent executes trading decisions through buying, selling, or holding actions. The environment continuously monitors the agent's balance and holdings and dispenses performance-based rewards.
### Deep Q-Network (DQN)
The DQN is a neural network that learns to predict the expected future rewards (Q-values) for each action given a state. The agent uses these Q-values to make trading decisions.
### DQN Agent
The agent uses an epsilon-greedy strategy to balance exploration and exploitation. It stores its experiences in a replay memory and uses these experiences to train the DQN.

### Training the Agent
To train the agent, run the train.py script with appropriate arguments:  
```python train.py --symbol AAPL --start_date 2020-01-01 --end_date 2023-12-31 --episodes 500 --batch_size 32 --window_size 10```

Arguments:  
--symbol: Stock symbol (default: AAPL)  
--start_date: Training data start date (default: 2020-01-01)  
--end_date: Training data end date (default: 2023-12-31)  
--episodes: Number of training episodes (default: 500)  
--batch_size: Batch size for training (default: 32)  
--window_size: Window size for state representation (default: 10)  


To evaluate the agent, run the evaluate.py script with appropriate arguments:  
```python evaluate.py --symbol AAPL --model_path models/weights/{model name} --start_date 2024-01-01 --end_date 2024-12-31 --window_size 10```

Arguments:  
--symbol: Stock symbol (default: AAPL)  
--model_path: Path to the trained model (required)  
--start_date: Test data start date (default: 2024-01-01)  
--end_date: Test data end date (default: 2024-12-31)  
--window_size: Window size for state representation (default: 10)  

