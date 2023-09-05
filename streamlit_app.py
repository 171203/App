import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Function to download stock data
def download_stock_data():
    AMZN = yf.download("AMZN", start="2012-05-18", end="2023-01-01")
    MSFT = yf.download("MSFT", start="2012-05-18", end="2023-01-01")
    NFLX = yf.download("NFLX", start="2012-05-18", end="2023-01-01")
    FDX = yf.download("FDX", start="2012-05-18", end="2023-01-01")

    return AMZN['Adj Close'], MSFT['Adj Close'], NFLX['Adj Close'], FDX['Adj Close']

st.title("Portfolio Optimization using Markowitz Model")

# Sidebar
st.sidebar.header("Portfolio Inputs")
st.sidebar.write("Enter the details of your portfolio:")

# Risk-Free Rate Input
st.sidebar.subheader("Risk-Free Rate (%)")
risk_free_rate = st.sidebar.number_input("Enter risk-free rate (%)", 0.0, 10.0, 2.0)

# Main content
if st.button("Optimize"):
    # Download stock data
    AMZN_AJClose, MSFT_AJClose, NFLX_AJClose, FDX_AJClose = download_stock_data()
    
    dataset = pd.concat([AMZN_AJClose, MSFT_AJClose, FDX_AJClose, NFLX_AJClose], axis=1)
    
    # Calculate portfolio statistics
    expected_returns = dataset.pct_change().mean() * 252
    cov_matrix = dataset.pct_change().cov() * 252

    num_assets = len(dataset.columns)

    # Generate random weights for the portfolio
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Calculate portfolio returns and volatility
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

    # Display portfolio statistics
    st.subheader("Portfolio Statistics")
    st.write(f"Portfolio Expected Return: {portfolio_return:.2%}")
    st.write(f"Portfolio Volatility: {portfolio_stddev:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Display portfolio composition
    st.subheader("Portfolio Composition")
    st.write("Optimal Weights:")
    st.write(weights)

    # Plot portfolio composition
    plt.pie(weights, labels=dataset.columns, autopct='%1.1f%%', startangle=140)
    st.pyplot(plt)



