import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Portfolio Optimization using Markowitz Model")

st.title("CSV File Viewer")

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Display the data in the DataFrame
    st.write("CSV Data:")
    st.dataframe(data)

# Main content
if st.button("Optimize"):
    st.write("Optimizing...")
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






