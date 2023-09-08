import streamlit as st
import requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime 

# Fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Log returns calculation
def log_returns(prices):
    return np.log(prices / prices.shift(1))

# Portfolio optimization
def optimize_portfolio(returns):
    n_assets = returns.shape[1]
    w0 = np.random.dirichlet(np.full(n_assets, 0.05))
    bounds = tuple((0, 1) for x in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opts = minimize(negative_sharpe_ratio, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return opts

# Objective function for minimizing negative Sharpe ratio
def negative_sharpe_ratio(weights):
    return -portfolio_return(weights) / portfolio_volatility(weights)

# Portfolio return calculation
def portfolio_return(weights):
    return np.sum(np.mean(log_returns(dataset).mean() * weights) * 252)

# Portfolio volatility calculation
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(log_returns(dataset).cov() * 252, weights)))

# Main Streamlit app
def main():
    st.title("Financial Data Analysis and Portfolio Optimization")
    
    # Sidebar
    st.sidebar.header("Settings")
    start_date = st.sidebar.date_input("Start Date", datetime(2012, 5, 18))
    end_date = st.sidebar.date_input("End Date", datetime(2023, 1, 1))
    symbol = st.sidebar.text_input("Stock Symbol (e.g., AMZN)", "AMZN")

    if st.sidebar.button("Fetch Data"):
        dataset = fetch_stock_data(symbol, start_date, end_date)

        if dataset is not None:
            st.write("Data loaded successfully.")
            st.subheader("Summary of Data")
            st.write(dataset.head())

            # Perform portfolio optimization
            st.subheader("Portfolio Optimization")
            portfolio_opts = optimize_portfolio(log_returns(dataset).dropna().T.values)

            st.write("Optimal Weights:", portfolio_opts.x.round(3))
            st.write("Expected Return:", portfolio_return(portfolio_opts.x))
            st.write("Expected Volatility:", portfolio_volatility(portfolio_opts.x))

            # Display portfolio optimization results
            st.subheader("Efficient Frontier and Portfolio Optimization")
            st.write("Note: For demonstration purposes, the efficient frontier plot is not included in this example.")
            # You can add your efficient frontier plot code here
    else:
        st.warning("Please select a valid date range and stock symbol.")

if __name__ == "__main__":
    main()
