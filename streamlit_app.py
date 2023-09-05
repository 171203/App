import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

def log_returns(prices):
    return np.log(prices / prices.shift(1))

def arithmetic_returns(prices):
    return prices/prices.shift(1) - 1
    
def portfolio_return(weights, returns):
    return np.sum(np.mean(returns, axis=1) * weights) * 252

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(np.cov(returns) * 252, weights)))
    
def expected_sharpe(weights, *args):
    # get the asset's returns
    returns = args[0]
    return - portfolio_return(weights, returns) / portfolio_volatility(weights, returns)

def taget_fun(weights, *args):
    # get the asset's returns
    returns = args[0]
    return portfolio_volatility(weights, returns)

st.title("Portfolio Optimization using Markowitz Model")

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


# Main content
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

def portfolio_volatility(weight):
    return np.sqrt(np.dot(weight.T, np.dot(sigma,weight)))*np.sqrt(252)

def portfolio_return(weight):
    return np.sum(mean_returns*weight)*252

def portfolio_performance(weight):
    return_p = portfolio_return(weight)
    vol_p    = portfolio_volatility(weight)
    return return_p, vol_p
def negativeSR(weight):
    return_p, vol_p = portfolio_performance(weight)
    rf_rate         = 0.025
    return -(return_p - rf_rate)/vol_p

def max_sharpe_ratio():

    def sum_one(weight):
        w= weight
        return np.sum(weight)-1

    n_assets           = log_return.shape[1]
    weight_constraints = ({'type':'eq','fun': sum_one})
    w0                 = np.random.dirichlet(np.full(n_assets,0.05)).tolist()    # w0 is an initila guess

    return minimize(negativeSR,w0,method='SLSQP', bounds  =((0,1),)*n_assets, constraints = weight_constraints)

def min_vol():

    n_assets           = log_return.shape[1]
    weight_constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    w0                 = np.random.dirichlet(np.full(n_assets,0.05)).tolist()
    bounds             = ((0,1),)*n_assets

    return minimize(portfolio_volatility,w0,method='SLSQP',
                   bounds      = bounds,
                   constraints = weight_constraints)

def efficient_portfolio_target(target):

    constraints = ({'type':'eq','fun': lambda x: portfolio_return(x)- target},
                  {'type' :'eq','fun': lambda x: np.sum(x)-1})
    w0          = np.random.dirichlet(np.full(n_assets,0.05)).tolist()
    bounds      = ((0,1),)*n_assets

    return minimize(portfolio_volatility,w0, method = 'SLSQP',
                    bounds      = bounds,
                    constraints = constraints)

def efficient_frontier(return_range):
    return [efficient_portfolio_target(ret) for ret in return_range]
    


if st.button('Simulation'):
    # Portfolio Simulation
    sharpe_maximum      = max_sharpe_ratio()
    return_p,vol_p      = portfolio_performance(sharpe_maximum['x'])
    min_volatility      = min_vol()
    return_min,vol_min  = portfolio_performance(min_volatility['x'])
    portfolio        = 2673  # generation of a portfolio
    n_assets         = log_return.shape[1]
    weights          = np.random.dirichlet(np.full(n_assets,0.05),portfolio)
    mean_returns     = log_return.mean()
    sigma            = log_return.cov()
    expected_returns = np.zeros(portfolio)
    expected_vol     = np.zeros(portfolio)
    sharpe_ratio     = np.zeros(portfolio)
    rf_rate          = 0.0                          # risk free rate

    for i in range(portfolio):
        w  = weights[i,:]
        expected_returns[i] = np.sum(mean_returns @ w)*252
        expected_vol[i]  = np.sqrt(np.dot(w.T,sigma @ w))*np.sqrt(252)
        sharpe_ratio[i] = (expected_returns[i]-rf_rate)/expected_vol[i]

    
    plt.figure(figsize =(15,10))
    plt.style.use('ggplot')
    plt.scatter(expected_vol,expected_returns, c = sharpe_ratio)
    # plt.colorbar.sel(label = 'Sharpe Ratio',size=20)
    plt.colorbar().set_label('Sharpe Ratio', size= 20, color = 'g', family='serif',weight='bold')
    target               = np.linspace(return_min,1.02,100)
    efficient_portfolios = efficient_frontier(target)
    plt.plot([i.fun for i in efficient_portfolios], target, linestyle ='dashdot', color ='black', label='Efficient Frontier')
    plt.scatter(vol_p,return_p, c = 'r', marker='*', s = 500, label = 'Maximum Sharpe Ratio')
    plt.scatter(vol_min,return_min, c = 'g',  marker ='*', s = 500, label='Minimum Volatility Portfolio')
    font1 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
    font2 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
    plt.title('Portfolio Optimization based on Efficient Frontier',fontdict=font1)
    plt.xlabel('Annualised Volatility',fontdict=font2)
    plt.ylabel('Annualised Returns',fontdict=font2)
    plt.legend(labelspacing=0.8)
    st.pyplot(plt)

def calc_portfolio_perf(weights, mean_returns, cov, rf):# portfolio performance, calculate the annualised return, sharpe ratio
    portfolio_return = np.sum(mean_returns*weights)*252 #252 working days at the stock exchange
    portfolio_std = np.sqrt(np.dot(weights.T,np.dot(cov,weights)))*np.sqrt(252) # np.dot multiplication of matrices
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf): # random simulation
    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights=np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = sharpe_ratio
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
    return results_df

if st.button("Portfolio Optimization"):
    tickers = []
    for i in data[['AMAZON','MICROSOFT','FDX','Netflix']].columns:
        tickers.append(i)
    sharpe_maximum      = max_sharpe_ratio()
    return_p,vol_p      = portfolio_performance(sharpe_maximum['x'])
    min_volatility      = min_vol()
    return_min,vol_min  = portfolio_performance(min_volatility['x'])
    mean_returns = data[['AMAZON','MICROSOFT','FDX','Netflix']].pct_change().mean()
    cov = data[['AMAZON','MICROSOFT','FDX','Netflix']].pct_change().cov()
    num_portfolios = 10000
    rf = 0.025
    results_frame =simulate_random_portfolios(num_portfolios, mean_returns,cov, rf)
    results_frame.sum(axis=1)-results_frame["ret"]-results_frame["stdev"]-results_frame["sharpe"];
  
    max_sharpe_port=results_frame.iloc[results_frame["sharpe"].idxmax()] # max sharp ratio rouge
    
    min_vol_port = results_frame.iloc[results_frame["stdev"].idxmin()] # min volatility = min variance portfolio vert
   
    plt.subplots(figsize=(15,10)) # Number of rows/colums of the subplot grid
    plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='plasma') #Colormaps in Matplotlib
    font1 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
    font2 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
    plt.title('Optimization of the portfolio',fontdict=font1)
    plt.xlabel('Risk/Annualised Volatility',fontdict=font2)
    plt.ylabel('Annualised Returns',fontdict=font2)
    
    plt.colorbar().set_label('Sharpe Ratio', size= 20, color = 'g', family='serif',weight='bold')
    target  = np.linspace(return_min,1.02,100)
    
    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500, label = 'Maximum Sharpe Ratio')
    
    plt.scatter(min_vol_port[1] ,min_vol_port[0],marker=(5,1,0),color='g', s=500, label='Minimum Volatility Portfolio')
    plt.legend(labelspacing=0.8)
    st.pyplot(plt)


    
    

   






