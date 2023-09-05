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
    expected_returns = data.pct_change().mean() * 252
    cov_matrix = data.pct_change().cov() * 252

    np.random.seed(1)
    # Weight each security
    weights = np.random.random((4,1))
    # normalize it, so that some is one
    weights /= np.sum(weights)
    st.write("*****************   Markowitz Portfolio Optimization   **********************")
    st.write(f'Normalized Weights : {weights.flatten()}')

    # We generally do log return instead of return
    Markowitz_log_ret = np.log(data / data.shift(1))

    # Expected return (weighted sum of mean returns). Mult by 252 as we always do annual calculation and year has 252 business days
    Markowitz_exp_ret = Markowitz_log_ret.mean().dot(weights)*252
    st.write(f'\nExpected return of the portfolio is : {Markowitz_exp_ret[0]}')

    # Exp Volatility (Risk)
    Markowitz_exp_vol = np.sqrt(weights.T.dot(252*Markowitz_log_ret.cov().dot(weights)))
    st.write(f'\nVolatility of the portfolio: {Markowitz_exp_vol[0][0]}')

    # Sharpe ratio
    Markowitz_sr = Markowitz_exp_ret / Markowitz_exp_vol
    print(f'\nSharpe ratio of the portfolio: {Markowitz_sr[0][0]}')

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






