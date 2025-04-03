import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from utils import download_stock_data, create_features
from environment import TradingEnvironment
from agent import DQNAgent


def train_agent(symbol: str, start_date: str, end_date: str, episodes: int, batch_size: int, window_size: int) -> DQNAgent:
    """
    Train the DQN agent on historical stock data.
    Args:
        symbol (str): Stock ticker symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        episodes (int): Number of training episodes
        batch_size (int): Batch size for training
        window_size (int): Window size for state representation
    Returns:
        DQNAgent: Trained agent
    """
    print(f"Training agent for {symbol} from {start_date} to {end_date}")
    
    # Download and preprocess data
    data = download_stock_data(symbol, start_date, end_date)
    data = create_features(data)
    
    # Create environment
    env = TradingEnvironment(data, window_size=window_size)
    
    # Create agent
    # state_size = window_size * 4  # 4 features per time step
    features_per_timestep = 4
    base_state_size = window_size * features_per_timestep
    position_features = 2  # For holdings flag and normalized balance
    state_size = base_state_size + position_features
    action_size = 3  # HOLD, BUY, SELL
    agent = DQNAgent(state_size, action_size)
    
    # Create directories for saving models
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/weights', exist_ok=True)
    
    # Training loop
    total_rewards = []
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Train model
            agent.replay(batch_size)
        
        total_rewards.append(total_reward)
        
        # Print progress
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        # Save model weights every 10 episodes
        if episode % 10 == 0:
            agent.save(f"models/weights/{symbol}_dqn_e{episode}.pth")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save(f"models/weights/{symbol}_dqn_final_{timestamp}.pth")
    
    # Plot training rewards
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    plt.title(f'Training Rewards for {symbol}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f"models/{symbol}_training_rewards.png")
    plt.close()
    
    return agent

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a DQN agent for stock trading')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for state representation')
    
    args = parser.parse_args()
    
    # Train agent
    agent = train_agent(
        args.symbol,
        args.start_date,
        args.end_date,
        args.episodes,
        args.batch_size,
        args.window_size
    )