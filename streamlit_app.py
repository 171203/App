pip install streamlit pandas numpy matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pip install cvxpy
import cvxpy as cp

# Load historical price data from a CSV file
@st.cache
def load_data():
    data = pd.read_csv('your_stock_data.csv')
    return data

data = load_data()

# Streamlit App
st.title('Portfolio Optimization App')

# Sidebar
st.sidebar.header('User Input')
selected_stocks = st.sidebar.multiselect('Select Stocks for Your Portfolio', data.columns[1:])

# Display selected stocks
st.write('You have selected the following stocks:')
st.write(selected_stocks)

# Load selected data
selected_data = data[['Date'] + selected_stocks]

# Calculate daily returns
returns = selected_data[selected_stocks].pct_change()
returns.dropna(inplace=True)

# Mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Display mean returns and covariance matrix
st.write('Mean Returns:')
st.write(mean_returns)
st.write('Covariance Matrix:')
st.write(cov_matrix)

# Portfolio Optimization
st.header('Portfolio Optimization')

# User-defined risk tolerance
st.sidebar.slider('Risk Tolerance', min_value=0.0, max_value=1.0, step=0.01, value=0.2)

# Portfolio Optimization
n_assets = len(selected_stocks)
weights = cp.Variable(n_assets)
risk_tolerance = st.sidebar.slider('Risk Tolerance', min_value=0.0, max_value=1.0, step=0.01, value=0.2)

# Portfolio Expected Return
portfolio_return = cp.sum(weights * mean_returns)
# Portfolio Risk (standard deviation)
portfolio_risk = cp.quad_form(weights, cov_matrix)

# Portfolio Optimization Problem
objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_risk)
constraints = [cp.sum(weights) == 1, weights >= 0]
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
problem.solve()

# Display portfolio weights
st.write('Portfolio Weights:')
st.write(weights.value)

# Visualization
st.header('Portfolio Performance')

# Portfolio Returns and Risk
portfolio_returns = np.sum(mean_returns * weights.value)
portfolio_volatility = np.sqrt(np.dot(weights.value.T, np.dot(cov_matrix, weights.value)))

# Display portfolio performance metrics
st.write('Expected Portfolio Return:', portfolio_returns)
st.write('Expected Portfolio Risk (Volatility):', portfolio_volatility)

# Plotting
st.header('Portfolio Allocation')
fig, ax = plt.subplots()
ax.pie(weights.value, labels=selected_stocks, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')
st.pyplot(fig)

