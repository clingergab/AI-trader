import argparse

from pandas import DataFrame
from utils import download_stock_data, create_features, plot_trading_results, save_results
from environment import TradingEnvironment
from agent import DQNAgent

def evaluate_agent(agent: DQNAgent, data: DataFrame, window_size=10, initial_balance=10000) -> tuple[float, float, list]:
    """
    Evaluate a trained agent on historical stock data.
    Args:
        agent (DQNAgent): Trained agent
        data (pandas.DataFrame): Historical stock data with technical indicators
        window_size (int): Window size for state representation
        initial_balance (float): Initial balance for trading
    Returns:
        tuple: (final_balance, return_percentage, actions_history)
    """
    # Create environment
    env = TradingEnvironment(data, window_size=window_size, initial_balance=initial_balance)
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Turn off exploration
    agent.epsilon = 0
    
    # Simulate trading
    while not done:
        # Choose action
        action = agent.act(state)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Move to next state
        state = next_state
    
    # Get actions history
    actions_history = env.get_actions_history()
    
    # Get final portfolio value
    final_price = float(data['Close'].iloc[-1])
    final_balance = env.balance + env.holdings * final_price
    
    # Calculate return
    return_percentage = (final_balance - initial_balance) / initial_balance * 100
    
    return final_balance, return_percentage, actions_history

def benchmark_buy_and_hold(data: DataFrame, initial_balance=10000) -> tuple[float, float]:
    """
    Benchmark strategy: Buy and hold.
    Args:
        data (pandas.DataFrame): Historical stock data
        initial_balance (float): Initial balance for trading
    Returns:
        tuple: (final_balance, return_percentage)
    """
    
    # Buy as many shares as possible at the beginning
    initial_price = float(data['Close'].iloc[0])
    shares = initial_balance // initial_price
    cost = shares * initial_price
    remaining_balance = initial_balance - cost
    
    # Calculate final value
    final_price = float(data['Close'].iloc[-1])
    final_balance = remaining_balance + shares * final_price
    
    # Calculate return
    return_percentage = (final_balance - initial_balance) / initial_balance * 100
    
    return final_balance, return_percentage

def main(symbol, model_path, test_start_date, test_end_date, window_size=10) -> None:
    """
    Main evaluation function.
    Args:
        symbol (str): Stock ticker symbol
        model_path (str): Path to the trained model
        test_start_date (str): Start date for testing in format 'YYYY-MM-DD'
        test_end_date (str): End date for testing in format 'YYYY-MM-DD'
        window_size (int): Window size for state representation
    Returns:
        None
    """
    print(f"Evaluating model for {symbol} from {test_start_date} to {test_end_date}")
    
    # Download and preprocess data
    data = download_stock_data(symbol, test_start_date, test_end_date)
    data = create_features(data)
    
    # Create agent
    # state_size = window_size * 4  # 4 features per time step
    features_per_timestep = 4
    base_state_size = window_size * features_per_timestep
    position_features = 2  # For holdings flag and normalized balance
    state_size = base_state_size + position_features
    action_size = 3  # HOLD, BUY, SELL
    agent = DQNAgent(state_size, action_size)
    
    # Load trained model
    agent.load(model_path)

    print(f"Model loaded. Sample parameters: {next(agent.model.parameters())[0][0:5].detach().cpu().numpy()}")
    
    # Evaluate agent
    final_balance, return_percentage, actions_history = evaluate_agent(
        agent, data, window_size=window_size
    )
    
    # Plot trading results
    plot_trading_results(data, actions_history)
    
    # Compare to buy and hold strategy
    bh_final_balance, bh_return_percentage = benchmark_buy_and_hold(data)
    
    print("\nBuy and Hold Strategy:")
    print(f"Final Balance: ${bh_final_balance:.2f}")
    print(f"Return: {bh_return_percentage:.2f}%")
    
    print("\nDQN Agent vs. Buy and Hold:")
    print(f"DQN Return: {return_percentage:.2f}%")
    print(f"Buy and Hold Return: {bh_return_percentage:.2f}%")
    print(f"Difference: {return_percentage - bh_return_percentage:.2f}%")
    
    # Save results
    model_name = model_path.split('/')[-1].split('.')[0]
    save_results(model_name, symbol, test_start_date, test_end_date, final_balance, return_percentage)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent for stock trading')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--start_date', type=str, default='2024-01-02', help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-01-01', help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for state representation')
    
    args = parser.parse_args()
    
    # Evaluate agent
    main(
        args.symbol,
        args.model_path,
        args.start_date,
        args.end_date,
        args.window_size
    )