import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pip install seaborn
import seaborn as sns
from datetime import datetime
import yfinance as yf
import scipy.optimize as sco

# Download stock data
AMZN = yf.download("AMZN", start="2012-05-18", end="2023-01-01", group_by="ticker") # Stock of Amazon
MSFT = yf.download("MSFT", start="2012-05-18", end="2023-01-01", group_by="ticker") # Stock of Microsoft
NFLX = yf.download("NFLX", start="2012-05-18", end="2023-01-01", group_by="ticker")
FDX = yf.download("FDX", start="2012-05-18", end="2023-01-01", group_by="ticker")

# Combine stock data into a single DataFrame
AMZN_AJClose = AMZN['Adj Close']
MSFT_AJClose = MSFT['Adj Close']
FDX_AJClose = FDX['Adj Close']
NFLX_AJClose = NFLX['Adj Close']
dataset = pd.concat([AMZN_AJClose, MSFT_AJClose, FDX_AJClose, NFLX_AJClose], axis=1)
dataset.columns = ['AMAZON', 'MICROSOFT', 'FDX', 'Netflix']

# Define Streamlit app
def main():
    st.title("Portfolio Optimization App")

    # Display boxplot
    st.subheader("Boxplot of Stock Prices")
    st.pyplot(plot_boxplot(dataset))

    # Display scatter matrix
    st.subheader("Scatter Matrix of Stock Prices")
    st.pyplot(plot_scatter_matrix(dataset))

    # Display daily close prices
    st.subheader("Daily Close Prices")
    st.pyplot(plot_daily_close_prices(dataset))

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    st.pyplot(plot_correlation_heatmap(dataset))

    # Portfolio optimization
    st.subheader("Portfolio Optimization Results")
    st.write("Portfolio performance metrics and optimization results go here.")

# Define functions for creating plots
def plot_boxplot(data):
    plt.figure(figsize=(8, 6))
    data.boxplot()
    plt.title("Boxplot of Stock Prices")
    return plt

def plot_scatter_matrix(data):
    plt.figure(figsize=(10, 8))
    pd.plotting.scatter_matrix(data, figsize=(10, 10))
    plt.title("Scatter Matrix of Stock Prices")
    return plt

def plot_daily_close_prices(data):
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.title('Daily Close Prices')
    plt.xlabel('Date: May 18th, 2012 - Dec. 30th, 2022')
    plt.ylabel('Values')
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.legend()
    return plt

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, annot_kws={'size': 12})
    plt.title("Correlation Heatmap")
    return plt

if __name__ == "__main__":
    main()

