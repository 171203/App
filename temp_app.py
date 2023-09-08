import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Define a function to download and process stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Define a function to calculate portfolio performance
def portfolio_performance(weights, returns):
    mean_returns = returns.mean()
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

# Define the Streamlit app
def main():
    st.title('Portfolio Optimization App')

    # Sidebar
    st.sidebar.header('Portfolio Configuration')
    selected_stocks = st.sidebar.multiselect('Select Stocks', ['AMZN', 'MSFT', 'NFLX', 'FDX'], default=['AMZN', 'MSFT'])
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2012-05-18'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))

    # Download stock data
    stock_data = download_stock_data(selected_stocks, start_date, end_date)

    # Display stock data
    st.subheader('Stock Data')
    st.write(stock_data.tail())

    # Calculate and display portfolio statistics
    log_returns = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    num_assets = len(selected_stocks)

    st.subheader('Portfolio Statistics')
    st.write(f'Number of Selected Stocks: {num_assets}')

    st.subheader('Expected Returns and Volatility')
    st.write('Annualized Expected Returns and Volatility:')
    
    def calculate_portfolio_statistics(weights):
        portfolio_return, portfolio_volatility = portfolio_performance(weights, log_returns)
        return portfolio_return, portfolio_volatility

    # Define an optimization function to maximize the Sharpe ratio
    def negative_sharpe(weights):
        portfolio_return, portfolio_volatility = calculate_portfolio_statistics(weights)
        rf_rate = 0.0  # Risk-free rate (you can adjust this)
        return -(portfolio_return - rf_rate) / portfolio_volatility

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Initial weights (equal weights)
    initial_weights = [1.0 / num_assets] * num_assets

    # Optimize portfolio
    result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=[(0, 1)] * num_assets, constraints=constraints)
    optimized_weights = result.x

    # Display portfolio statistics
    optimized_return, optimized_volatility = calculate_portfolio_statistics(optimized_weights)
    st.write(f'Optimal Portfolio Weights: {optimized_weights}')
    st.write(f'Expected Return: {optimized_return:.2%}')
    st.write(f'Volatility: {optimized_volatility:.2%}')

    # Efficient Frontier
    st.subheader('Efficient Frontier')
    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        ret, vol = calculate_portfolio_statistics(weights)
        ret_arr[i] = ret
        vol_arr[i] = vol

    # Calculate the Sharpe ratio for each portfolio
    sharpe_ratio = ret_arr / vol_arr

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = sharpe_ratio.argmax()
    max_sharpe_return = ret_arr[max_sharpe_idx]
    max_sharpe_volatility = vol_arr[max_sharpe_idx]

    # Plot the efficient frontier
    plt.figure(figsize=(12, 6))
    plt.scatter(vol_arr, ret_arr, c=sharpe_ratio, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')

    # Highlight the portfolio with the highest Sharpe ratio
    plt.scatter(max_sharpe_volatility, max_sharpe_return, c='red', marker='*', s=100, label='Max Sharpe Ratio')
    plt.legend()
    st.pyplot(plt)

if __name__ == '__main__':
    main()


