import streamlit as st
import robin_stocks as r
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import date
from finquant.portfolio import build_portfolio
import matplotlib.pyplot as plt
import plotly.express as px
import math
from scipy import stats


# Log in to api
#login = r.login(username="dhutchings1998@gmail.com", password="Potomac_123",store_session = True)


########### Cached Functions #############

@st.cache
def fetch_build_holdings():
    return r.account.build_holdings()

@st.cache
def fetch_build_holdings():
    return r.account.build_holdings()

@st.cache
def fetch_transactions():
    transactions = {}
    for position in r.account.get_all_positions():
        symbol = r.stocks.get_instrument_by_url(position['instrument'])['symbol']
        date = position['created_at']
        transactions[symbol] = date
    return transactions


########## Backend Data Construction ###############

# List Stock Names
stock_names = []
for i in fetch_build_holdings():
    stock_names.append(i)

# Calculate total equity
total_equity = float(r.account.build_user_profile()['equity'])

# Identify portfolio allocations
allocation = {}
holdings = fetch_build_holdings()
for i in stock_names:
    allocation[i] = int((float(holdings[i]['equity'])/total_equity)*100)

d = {}
for index,name in enumerate(stock_names):
    d[index] = {"Name": name, "Allocation": allocation[name]}   
pf_allocation = pd.DataFrame.from_dict(d, orient="index")

names = pf_allocation["Name"].values.tolist()
start_date = datetime.datetime(2020, 9, 18)
end_date = date.today()

# Build portfolio
pf = build_portfolio(
    names=names, pf_allocation=pf_allocation, start_date=start_date, end_date=end_date, data_api="yfinance"
)

# Run monte carlo optimization
opt_w, opt_res = pf.mc_optimisation(num_trials=5000)

# Create table of stock info and allocations
pf_allocation.set_index('Name', inplace=True)
stock_info= pd.concat([pf.comp_mean_returns(),pf.skew, pf.kurtosis,pf_allocation, 
round(opt_w.T *100,0).astype(int)],axis=1)
stock_info.columns = ["Expected Annual Returns","Skewness", "Kurtosis","Current Weights(%)",
"Min Volatility Weights(%)", "Max Sharpe Ratio Weights(%)"]
stock_info['Expected Annual Returns'] = round(stock_info['Expected Annual Returns']*100,2)
stock_info['Expected Annual Returns'] = stock_info['Expected Annual Returns'].astype(str) + "%"

# Create plot of all equities
num_rows= 1+ math.floor(len(stock_names)/3)
fig = make_subplots(rows =2 , cols = 3, subplot_titles = stock_names)
row = 1
for count, i in enumerate(fetch_build_holdings()):
    history = r.stocks.get_stock_historicals(i, interval="day", span="year")
    df = pd.DataFrame(history)
    close_prices = [float(i) for i in list(df.close_price)]
    trans = fetch_transactions()
    col = (count % 3) + 1
    fig.add_trace(go.Scatter(x = df.begins_at, y = df.close_price, name = ""),row = row ,col=col)
    fig.add_shape(dict(type="line", x0 = trans[i], x1 = trans[i],
        y0 = float(min(close_prices)),
        y1 = float(max(close_prices)), line_width = 1.5, opacity = 0.5), row = row, col = col )  
    if count % 3 == 2 and count != 0:
        row = row + 1

fig.update_layout(showlegend = False, margin=dict(l=0,r=0,b=0,t=40))

## Calculate Alpha and Beta
weights = [weight/100 for weight in list(allocation.values())]

#spy represents market returns
spy = pd.DataFrame(r.stocks.get_stock_historicals("SPY", interval="day", span="year"))
spy['begins_at'] = pd.to_datetime(spy['begins_at']).apply(lambda x: x.replace(tzinfo=None))
spy = spy[spy['begins_at'] >= datetime.datetime(2020, 9, 18)][['begins_at','close_price']]
spy['close_price'] = spy['close_price'].astype(float)
spy = list(spy['close_price'].pct_change()[1:])

# p_returns are portfolio returns
p_returns = pf.comp_daily_returns()
p_returns = (p_returns * weights).sum(axis = 1)
p_returns = list(p_returns)

beta, alpha = stats.linregress(spy,
                p_returns)[:2]



########  Front End  #######

# Title
st.title('Robinhood Portfolio')

# 2 columns (portfolio stats)
left_column, right_column = st.beta_columns(2)
with left_column:
    st.markdown("**Portfolio Expected Return (Annual)**: " + str(round(float(pf.expected_return * 100),3)) + "%")
    st.markdown("**Portfolio Volatility**: " + str(round(float(pf.volatility * 100),3)) + "%")
    st.markdown("**Portfolio Beta**: " + str(round(beta,2)))


with right_column:
    st.markdown("**Portfolio Sharpe Ratio**: " + str(round(float(pf.sharpe),3)))
    st.markdown("**Portfolio Total Equity**: " + "$"+ str(round(total_equity,2)))
    st.markdown("**Portfolio Alpha**: " + str(round(alpha,5)*100))


# Plot all equity historicals
st.plotly_chart(fig)

# Plot cumulate returns for equities
return_fig = px.line(pf.comp_cumulative_returns(), labels={
                     "Date": "",
                     "value": "Cumulative Returns",
                     "variable": "Equity"
                 })
return_fig.update_layout(margin=dict(l=0,r=0,b=0))
st.write(return_fig)

# Plot stock info table
st.table(stock_info)



