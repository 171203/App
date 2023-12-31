import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
st.title("Portfolio Management Optimization")

# Upload a CSV file from your local computer
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Define the expected column names
expected_columns = ["AMAZON", "MICROSOFT", "FDX", "Netflix"]

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Check if the DataFrame contains the expected columns
    if not set(expected_columns).issubset(data.columns):
        st.error("Warning: The given dataset is not suitable for the Optimization")
        st.error("Error: Please upload the daatset named My_Portfolio.")
    else:
        # Display the uploaded data
        st.write("Uploaded CSV Data:")
        st.dataframe(data)

        # Portfolio Optimization
        st.header("Portfolio Optimization")
log_return = np.log(data / data.shift(1))

    
def expected_sharpe(weights, *args):
    # get the asset's returns
    returns = args[0]
    return - portfolio_return(weights, returns) / portfolio_volatility(weights, returns)

def taget_fun(weights, *args):
    # get the asset's returns
    returns = args[0]
    return portfolio_volatility(weights, returns)



if st.button("Daily Returns of the Portfolio"):
    if uploaded_file is None:
        st.error("**Please Upload the portfolio data file**")
    plt.figure(figsize=(20,8)) # Increases the Plot Size
    plt.grid(True)
    plt.title('Daily Close Prices of Amazon and Microsoft')
    plt.xlabel('Date: May 18th, 2012 - Dec. 30th, 2022')
    plt.ylabel('Values')
    plt.plot(data['AMAZON'], 'orange', label='Amazon')
    plt.plot(data['MICROSOFT'], 'blue', label='Microsoft')
    plt.plot(data['FDX'], 'green', label='FDX')
    plt.plot(data['Netflix'], 'red', label='Netflix')
    plt.legend()
    plt.legend()
    plt.legend()
    st.pyplot(plt)
if st.button("Box Plot Display"):
    plt.style.use("fivethirtyeight")
    data[['AMAZON','MICROSOFT','FDX', 'Netflix']].boxplot()
    plt.title("Boxplot of Stock Prices (Amazon, Microsoft, FDX, Netflix,)")
    st.pyplot(plt)
if st.button("Display the Distribution"):
    pd.plotting.scatter_matrix(data[['AMAZON','MICROSOFT','FDX', 'Netflix']], figsize=(10,10))
    st.pyplot(plt)
if st.button("Implementing the Optimization Algorithm"):
    st.write("**Model is loading...**")
    # Calculate expected returns and covariance matrix
    expected_returns = data.pct_change().mean() * 252
    cov_matrix = data.pct_change().cov() * 252
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)
    np.random.seed(1)
    # Weight each security
    weights = np.random.random((4,1))
    # normalize it, so that some is one
    weights /= np.sum(weights)
    st.header("Markowitz Portfolio Optimization")
    st.write(f'Normalized Weights : **{weights.flatten()}**')

    # We generally do log return instead of return
    Markowitz_log_ret = np.log(data / data.shift(1))

    # Expected return (weighted sum of mean returns). Mult by 252 as we always do annual calculation and year has 252 business days
    Markowitz_exp_ret = Markowitz_log_ret.mean().dot(weights)*252
    st.write(f'\nExpected return of the portfolio is : **{Markowitz_exp_ret[0]}**')

    # Exp Volatility (Risk)
    Markowitz_exp_vol = np.sqrt(weights.T.dot(252*Markowitz_log_ret.cov().dot(weights)))
    st.write(f'\nVolatility of the portfolio: **{Markowitz_exp_vol[0][0]}**')

    # Sharpe ratio
    Markowitz_sr = Markowitz_exp_ret / Markowitz_exp_vol
    st.write(f'\nSharpe ratio of the portfolio: **{Markowitz_sr[0][0]}**')


    
if st.button("Train the Model"):
    log_return = np.log(data / data.shift(1))
    num_ports = 5000
    all_weights = np.zeros((num_ports, len(data.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):
        weights = np.array(np.random.random(4))
        weights = weights/np.sum(weights)

        # save the weights
        all_weights[ind,:] = weights

        # expected return
        ret_arr[ind] = np.sum((log_return.mean()*weights)*252)

        # expected volatility
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    max_sr_ret = ret_arr[4632]
    max_sr_vol = vol_arr[4632]
    # plot the dataplt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.title("Visualization of the Portfolio")
    plt.xlabel('Volatility')
    plt.ylabel('Return')

    # add a red dot for max_sr_vol & max_sr_ret
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
    st.pyplot(plt)

  
    
    


    
    

   






