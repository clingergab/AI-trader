from sklearn.preprocessing import StandardScaler
from typing import Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def download_stock_data(symbol, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock ticker symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        pandas.DataFrame: Historical stock data
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def create_features(data):
    """
    Calculate technical indicators to use as features.
    
    Args:
        data (pandas.DataFrame): Historical stock data with OHLC prices
        
    Returns:
        pandas.DataFrame: Data with added technical indicators
    """
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Daily Returns
    data['Returns'] = data['Close'].pct_change()
    
    # Drop NaN values and reset index
    data.dropna(inplace=True)
    data.reset_index(drop=False, inplace=True)
    
    return data

def prepare_data(data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Prepare the final dataset for model training using StandardScaler"""
    # Select features to use
    features = data[['SMA_5', 'SMA_20', 'Returns', 'Close', 'Volume']].values
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features, scaler


def plot_trading_results(data, actions_taken, initial_balance=10000):
    """
    Plot trading results including price, actions, and portfolio value.
    
    Args:
        data (pandas.DataFrame): Historical data
        actions_taken (list): List of actions taken at each time step (0: HOLD, 1: BUY, 2: SELL)
        initial_balance (float): Starting balance for the trading simulation
        
    Returns:
        None (displays plot)
    """
    # Make sure actions_taken isn't longer than data
    if len(actions_taken) > len(data):
        actions_taken = actions_taken[:len(data)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # Plot 1: Price chart
    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Price')
    ax1.legend()
    
    # Plot 2: Buy/Sell Actions
    buy_indices = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_indices = [i for i, a in enumerate(actions_taken) if a == 2]
    
    ax2.plot(data.index, data['Close'], alpha=0.3, color='blue')
    
    # Check if there are any buy actions
    if buy_indices:
        ax2.scatter(
            data.index[buy_indices], 
            [float(data['Close'].iloc[i].iloc[0]) for i in buy_indices], 
            marker='^', color='green', label='Buy'
        )
    
    # Check if there are any sell actions
    if sell_indices:
        ax2.scatter(
            data.index[sell_indices], 
            [float(data['Close'].iloc[i].iloc[0]) for i in sell_indices], 
            marker='v', color='red', label='Sell'
        )
    
    ax2.set_ylabel('Price')
    ax2.set_title('Buy/Sell Actions')
    ax2.legend()
    
    # Plot 3: Portfolio Value (simplified simulation)
    portfolio_value = []
    balance = initial_balance
    holdings = 0
    
    for i in range(len(actions_taken)):
        price = float(data['Close'].iloc[i].iloc[0])
        
        if i < len(actions_taken):
            action = actions_taken[i]
            
            if action == 1 and balance >= price:  # BUY
                max_shares = balance // price
                holdings += max_shares
                balance -= max_shares * price
            elif action == 2 and holdings > 0:  # SELL
                balance += holdings * price
                holdings = 0
        
        current_value = balance + holdings * price
        portfolio_value.append(current_value)
    
    # Ensure portfolio_value matches data length
    if len(portfolio_value) < len(data):
        # Hold the last position for remaining time steps
        last_value = portfolio_value[-1]
        while len(portfolio_value) < len(data):
            portfolio_value.append(last_value)
    elif len(portfolio_value) > len(data):
        # Trim extra values
        portfolio_value = portfolio_value[:len(data)]
    
    ax3.plot(data.index, portfolio_value, color='purple')
    ax3.set_ylabel('Value ($)')
    ax3.set_title('Portfolio Value')
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate final return
    final_return = (portfolio_value[-1] - initial_balance) / initial_balance * 100
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${portfolio_value[-1]:.2f}")
    print(f"Return: {final_return:.2f}%")
    
    return portfolio_value[-1], final_return

def save_results(model_name, symbol, start_date, end_date, final_balance, return_pct):
    """
    Save trading results to a log file.
    
    Args:
        model_name (str): Name of the model used
        symbol (str): Stock ticker symbol
        start_date (str): Start date of evaluation period
        end_date (str): End date of evaluation period
        final_balance (float): Final portfolio value
        return_pct (float): Percentage return
        
    Returns:
        None
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'results/eval_{model_name}_{timestamp}.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Final Balance: ${final_balance:.2f}\n")
        f.write(f"Return: {return_pct:.2f}%\n")
        