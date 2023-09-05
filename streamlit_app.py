import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
st.title("Portfolio Optimization using Markowitz Model")
# Main content
if st.button("Optimize"):
    st.write("Optimizing...")
    st.title("Display CSV File in Streamlit App")

    csv_url = "https://github.com/171203/App/blob/main/dataset.csv"
    
    dataset = pd.read_csv(csv_url, encoding='utf-8')
    

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




