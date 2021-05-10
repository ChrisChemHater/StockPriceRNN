

import pandas as pd
import quandl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsplots
import numpy as np
import scipy.stats as st

quandl.ApiConfig.api_key='HT4ys9sgAtiAwG_VUQiu'
pd.set_option("display.max_rows",10)
StockPrices=quandl.get_table("WIKI/PRICES",ticker=['IBM','MSFT','INTC','AAPL'],qopts={'columns':['ticker','date','adj_close','adj_volume']},
date={'gte':'1990-1-1','lte':'2017-12-31'},paginate=True)
# StockPrices=quandl.get_table("WIKI/PRICES",ticker=['TSLA'],qopts={'columns':['ticker','date','adj_close','adj_volume']},
# date={'gte':'2000-1-1','lte':'2017 -12-31'},paginate=True)
df=StockPrices.pivot(index='date',columns='ticker',values='adj_close')
df.fillna(method='ffill')
df.dtypes
df=df.astype('float')

StockReturns=df.pct_change()
stock_prices_monthly=df.resample('M').last()
stock_returns_monthly=stock_prices_monthly.pct_change()
print(stock_returns_monthly)

def auto_test(name):
    biased_acovf = sm.tsa.stattools.acovf(stock_returns_monthly[name], missing = 'drop')
    unbiased_acovf = sm.tsa.stattools.acovf(stock_returns_monthly[name], missing = 'drop', adjusted=True)
    biased_acf = sm.tsa.stattools.acf(stock_returns_monthly[name], missing = 'drop')
    unbiased_acf = sm.tsa.stattools.acf(stock_returns_monthly[name], missing = 'drop', adjusted=True)
    print("----------------------------------"+name+"start------------------------------")
    print("biased_acovf:",biased_acovf[:10])
    print("unbiased_acovf:",unbiased_acovf[:10])
    print("biased_acf:",biased_acf[:10])
    print("unbiased_acf:",unbiased_acf[:10])
    print("----------------------------------"+name+" end------------------------------")

# auto_test('AAPL')
# auto_test('TSLA') #特斯拉
# auto_test('IBM')
# auto_test('MSFT')
# auto_test('INTC')


def test_plot(name):
    tsplots.plot_acf(stock_returns_monthly[name][1:],lags=20,alpha=0.05)
    plt.show()
    tsplots.plot_pacf(stock_returns_monthly[name][1:].values,lags=20,alpha=0.05)
    plt.show()
# test_plot('AAPL')
# test_plot('TSLA')
# test_plot('IBM')
# test_plot('MSFT')
# test_plot('INTC')

def Jr_stat(data,q):
    data=data[len(data) % q:] #扔掉mod(q)的余数
    n=int(len(data)/q)
    data_tmp=[0]*n
    sigma_a2=sum(np.power(data,2))
    for i in range(q):
        data_tmp=data_tmp+data[i::q]
    sigma_b2=sum(np.power(data_tmp,2))
    return sigma_b2/sigma_a2-1,n

def VR_test(data,q):
    [Jr,n]=Jr_stat(data,q)
    return (1-st.norm.cdf(np.sqrt(n)*abs(Jr)))*2

print("IBM p_value for q=2 is ",VR_test(stock_returns_monthly['IBM'].values[1:],2))
print("MSFT p_value for q=2 is ",VR_test(stock_returns_monthly['MSFT'].values[1:],2))
print("INTC p_value for q=2 is ",VR_test(stock_returns_monthly['INTC'].values[1:],2))
print("AAPL p_value for q=2 is ",VR_test(stock_returns_monthly['AAPL'].values[1:],2))
# print("TSLA p_value for q=2 is ",VR_test(stock_returns_monthly['TSLA'].values[1:],2))
