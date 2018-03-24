
# coding: utf-8

# In[8]:


import os
print("running from {}".format(os.getcwd()))


# In[15]:


import os
import numpy as np
import pandas as pd
import pickle
import quandl
quandl.ApiConfig.api_key = 'qgcsveAh-6YsyazyaRee' 
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)


# In[11]:


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


# In[16]:


# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')


# In[18]:


os.listdir()


# In[106]:


axp_usd_price = get_quandl_data('EOD/AXP')


# In[91]:


# Rename volume column for BTC to compare volume more easily with stocks later
btc_usd_price_kraken.rename(columns={'Volume (Currency)': 'Volume'}, inplace=True)
btc_usd_price_kraken.head()


# In[107]:


axp_usd_price.head()


# In[22]:


# Chart the BTC pricing data
trace = go.Scatter(x=axp_usd_price.index, y=axp_usd_price['Close'])
py.iplot([trace])


# In[92]:


# Pull pricing data for 3 more BTC exchanges
stocks = ['AXP','MSFT','IBM', 'AAPL', 'CSCO']

# store daily price 
stock_prices = {}

stock_prices['BTC'] = btc_usd_price_kraken[(btc_usd_price_kraken).index > datetime(2017,1,1)]

for stock in stocks:
    code = 'EOD/{}'.format(stock)
    stock_df = get_quandl_data(code)
    stock_df = stock_df[(stock_df.index > datetime(2017,1,1))] # Kraken only has BTC data from 2015 on
    stock_prices[stock] = stock_df
    
# store monthly mean price
monthly_prices = {}
for ticker in stock_prices:
    monthly_mean = stock_prices[ticker].resample('M').mean()
    monthly_prices[ticker] = monthly_mean


# In[72]:


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)


# In[93]:


# Merge the BTC price dataseries' into a single dataframe
usd_datasets = merge_dfs_on_column(list(stock_prices.values()), list(stock_prices.keys()), 'Close') # daily price
volume_datasets = merge_dfs_on_column(list(stock_prices.values()), list(stock_prices.keys()), 'Volume') # daily volume
monthly_datasets = merge_dfs_on_column(list(monthly_prices.values()), list(monthly_prices.keys()), 'Close') # monthly mean price


# In[108]:


volume_datasets.tail()


# In[66]:


def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))
    
    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels= not seperate_y_axis,
            type=scale
        )
    )
    
    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale )
    
    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'
        
    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index, 
            y=series, 
            name=label_arr[index],
            visible=visibility
        )
        
        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config    
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.iplot(fig)


# In[99]:


# Chart all of the prices together
df_scatter(usd_datasets, 'Bitcoin Price vs IG Tech Stock Prices', seperate_y_axis=False, y_axis_label='Coin Value (USD)', scale='log')
df_scatter(monthly_datasets, 'Monthly: Bitcoin Price vs IG Tech Stock Prices', seperate_y_axis=False, y_axis_label='Coin Value (USD)', scale='log')
df_scatter(volume_datasets, 'Bitcoin Volume vs IG Tech Stock Volume', seperate_y_axis=False, y_axis_label='Volume ($)', scale='linear')


# In[98]:


# Remove "0" values
usd_datasets.replace(0, np.nan, inplace=True)
monthly_datasets.replace(0, np.nan, inplace=True)


# In[83]:


# Calculate the pearson correlation coefficients for cryptocurrencies in 2017
combined_df_2017 = usd_datasets[usd_datasets.index.year == 2017]
combined_df_2017.pct_change().corr(method='pearson')


# In[61]:


def correlation_heatmap(df, title, absolute_bounds=True):
    '''Plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient'),
    )
    
    layout = go.Layout(title=title)
    
    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0
        
    fig = go.Figure(data=[heatmap], layout=layout)
    py.iplot(fig)


# In[84]:


correlation_heatmap(combined_df_2017.pct_change(), "BTC correlation vs Tech Stocks in 2017")


# In[87]:


# Calculate the pearson correlation coefficients for monthly means in 2017
monthly_datasets.pct_change().corr(method='pearson')


# In[109]:


correlation_heatmap(monthly_datasets.pct_change(), "Monthly: BTC price correlation vs Tech Stocks in 2017")


# In[110]:


# Calculate the pearson correlation coefficients for monthly means in 2017
volume_datasets.pct_change().corr(method='pearson')


# In[101]:


correlation_heatmap(volume_datasets.pct_change(), "Volume: BTC correlation vs Tech Stocks in 2017")


# In[102]:


# Let's see if anything changed in 2018
combined_df_2018 = usd_datasets[usd_datasets.index.year == 2018]
combined_df_2018.pct_change().corr(method='pearson')


# In[111]:


correlation_heatmap(combined_df_2018.pct_change(), "Price: BTC correlation vs Tech Stocks in 2018")


# In[112]:


# A lot more correlated in 2018! Now, let's check volume
combined_df_2018 = volume_datasets[usd_datasets.index.year == 2018]
combined_df_2018.pct_change().corr(method='pearson')


# In[113]:


correlation_heatmap(combined_df_2018.pct_change(), "Volume: BTC correlation vs Tech Stocks in 2018")


# In[ ]:


# So simple conclusion is correlation of prices between BTC and big tech companies has increased in 2017, but there is no similar 
# correlation with volume

