import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Portfolio Optimization using Markowitz Model")

# Sidebar
st.sidebar.header("Portfolio Inputs")
st.sidebar.write("Enter the details of your portfolio:")

# Stock Input
st.sidebar.subheader("Stock Data (Ticker, Number of Shares)")
stock_data = st.sidebar.text_area("Enter stock data (ticker, shares)", "AAPL,100\nGOOGL,50\nAMZN,75")

# Expected Returns Input
st.sidebar.subheader("Expected Annual Returns (%)")
expected_returns = st.sidebar.text_area("Enter expected annual returns (%)", "AAPL,10\nGOOGL,15\nAMZN,12")

# Risk-Free Rate Input
st.sidebar.subheader("Risk-Free Rate (%)")
risk_free_rate = st.sidebar.number_input("Enter risk-free rate (%)", 0.0, 10.0, 2.0)

# Main content
if st.button("Optimize"):
    # Parse stock data and expected returns
    stock_data = [line.split(',') for line in stock_data.split('\n')]
    stock_data = [(symbol.strip(), float(shares.strip())) for symbol, shares in stock_data if symbol and shares]

    expected_returns = [line.split(',') for line in expected_returns.split('\n')]
    expected_returns = [(symbol.strip(), float(returns.strip())) for symbol, returns in expected_returns if symbol and returns]

    if not stock_data or not expected_returns:
        st.warning("Please enter valid stock data and expected returns.")
    else:
        # Calculate portfolio statistics
        symbols, shares = zip(*stock_data)
        portfolio_data = pd.DataFrame({'Symbol': symbols, 'Shares': shares})
        portfolio_data.set_index('Symbol', inplace=True)

        expected_returns = dict(expected_returns)
        portfolio_data['Expected Return (%)'] = portfolio_data.index.map(expected_returns)
        portfolio_data['Weight'] = portfolio_data['Shares'] / portfolio_data['Shares'].sum()
        
        portfolio_returns = portfolio_data['Weight'] * portfolio_data['Expected Return (%)']
        portfolio_volatility = np.sqrt(np.dot(portfolio_returns, np.dot(np.cov(portfolio_returns, rowvar=False), portfolio_returns)))

        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_returns.sum() - risk_free_rate) / portfolio_volatility

        # Display portfolio data
        st.subheader("Portfolio Data")
        st.write(portfolio_data)

        # Display portfolio statistics
        st.subheader("Portfolio Statistics")
        st.write(f"Portfolio Expected Return: {portfolio_returns.sum():.2f}%")
        st.write(f"Portfolio Volatility: {portfolio_volatility:.2f}%")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Plot portfolio composition
        st.subheader("Portfolio Composition")
        plt.pie(portfolio_data['Weight'], labels=portfolio_data.index, autopct='%1.1f%%', startangle=140)
        st.pyplot(plt)


