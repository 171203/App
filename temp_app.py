import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco
import requests
import io

def fetch_data_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error("Failed to fetch data file from GitHub.")
        return None

github_url = f"https://github.com/171203/App/blob/main/data.csv"
if st.button("Fetch Data from GitHub"):
    data_text = fetch_data_file(github_url)

    if data_text:
        # Load the data into a DataFrame (modify this part according to your data format)
        df = pd.read_csv(io.StringIO(data_text))

        # Display the data in your Streamlit app
        st.write("Data fetched from GitHub:")
        st.write(df)

# Sidebar with user input
st.sidebar.title("Portfolio Optimization")
st.sidebar.markdown("Adjust the portfolio weights and target return:")

# Define portfolio optimization function
def portfolio_optimization(returns, num_portfolios, risk_free_rate, target_return):
    num_assets = len(returns.columns)
    results = np.zeros((4, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Expected return
        portfolio_return = np.sum(returns.mean() * weights) * 252

        # Expected volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = weights[0]  # Weight of Amazon

    return results

# Sidebar inputs
num_portfolios = st.sidebar.slider("Number of Portfolios", 1, 10000, 1000)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.5)
target_return = st.sidebar.slider("Target Return (%)", 0.0, 20.0, 10.0)

# Calculate portfolio optimization results
returns = np.log(df / df.shift(1))
results = portfolio_optimization(returns, num_portfolios, risk_free_rate / 100, target_return / 100)

# Display portfolio optimization results
st.title("Portfolio Optimization Results")
st.markdown(f"**Number of Portfolios:** {num_portfolios}")
st.markdown(f"**Risk-Free Rate:** {risk_free_rate}%")
st.markdown(f"**Target Return:** {target_return}%")

# Create a DataFrame to display the results
columns = ["Return", "Volatility", "Sharpe Ratio", "Weight of Amazon"]
df_results = pd.DataFrame(results.T, columns=columns)

# Find the portfolio with the maximum Sharpe ratio
max_sharpe_idx = df_results["Sharpe Ratio"].idxmax()
max_sharpe_portfolio = df_results.iloc[max_sharpe_idx]

st.subheader("Maximum Sharpe Ratio Portfolio")
st.write(df_results)
st.write(f"Maximum Sharpe Ratio Portfolio:\n{max_sharpe_portfolio}")

# Plotting the efficient frontier
st.subheader("Efficient Frontier")
plt.figure(figsize=(10, 6))
plt.scatter(df_results["Volatility"], df_results["Return"], c=df_results["Sharpe Ratio"], cmap="viridis")
plt.title("Efficient Frontier")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.colorbar(label="Sharpe Ratio")
st.pyplot(plt)

# Display the weights of the maximum Sharpe ratio portfolio
st.subheader("Maximum Sharpe Ratio Portfolio Weights")
st.write("Weights of Assets in Maximum Sharpe Ratio Portfolio:")
st.write("Amazon:", max_sharpe_portfolio["Weight of Amazon"])
st.write("Microsoft:", 1 - max_sharpe_portfolio["Weight of Amazon"])



