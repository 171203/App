import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Portfolio Optimization using Markowitz Model")

# Load the CSV file from GitHub
@st.cache
def load_data():
    url = "https://github.com/171203/App/blob/main/dataset.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Sidebar
st.sidebar.header("Portfolio Inputs")
st.sidebar.write("Enter the details of your portfolio:")

# Risk-Free Rate Input
st.sidebar.subheader("Risk-Free Rate (%)")
risk_free_rate = st.sidebar.number_input("Enter risk-free rate (%)", 0.0, 10.0, 2.0)

# Main content
if st.button("Optimize"):
    st.write("Optimizing...")

    # Extract stock data from the dataset
    stock_data = data.drop(columns=["Date"])
    symbols = stock_data.columns

    # Calculate expected returns and covariance matrix
    expected_returns = stock_data.pct_change().mean() * 252
    cov_matrix = stock_data.pct_change().cov() * 252

    num_assets = len(symbols)

    # Generate random portfolio weights
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
    plt.pie(weights, labels=symbols, autopct='%1.1f%%', startangle=140)
    st.pyplot(plt)






