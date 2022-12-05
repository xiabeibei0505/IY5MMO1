#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#for  reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
#for timestamps
from datetime import datetime
import tushare as ts
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# 拿到申请到的token - tushare.pro官网进行申请
token = '2ab1a504ad329acd2165f35bbf1a63cd795853aeabcfa24f669d2443'
# 设置token
pro = ts.pro_api(token)
#000001.SZ  平安银行    601166.SH  兴业银行    600900.SH 长江电力    600031.SH   三一重工 
df = pro.daily(ts_code='000001.SZ,601166.SH,600900.SH,600031.SH', start_date='20210101', end_date='20221231')
df


# In[7]:



import matplotlib.dates as dates


# In[5]:


df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')

df['date_k']=pd.to_datetime(df['trade_date'],format="%Y%m%d").apply(lambda x:dates.date2num(x)*1440)


# In[6]:


data = df[['ts_code','trade_date','open','high','low','close','vol','amount','pre_close','change','pct_chg','date_k','date']]   
data = data.sort_values(by='date',ascending=True)
data = data.set_index('date')  
data


# In[7]:


# Use name replace the code
data["ts_code"].replace("000001.SZ","PingAn",inplace = True)
data["ts_code"].replace("601166.SH","XingYe",inplace = True)
data["ts_code"].replace("600900.SH","ChangJiang",inplace = True)
data["ts_code"].replace("600031.SH","SanYi",inplace = True)
data


# In[8]:


data['ts_code'].value_counts()


# In[9]:


company_name =["PingAn","ChangJiang","XingYe","SanYi"]
company_list =["000001.SZ","601166.SH","600900.SH","603290.SH"]


# In[10]:


#let us see a historical view of the closing price
plt.figure(figsize=(25,20))
plt.subplots_adjust(top=1.25,bottom=1.2)

for i,company in enumerate(company_name, 1):
    plt.subplot(4,1,i)
    data.loc[(data['ts_code']== company)]['close'].plot() 
    plt.ylabel('close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {company}")
plt.tight_layout()


# In[11]:


# Now let us plot the total volume of stock being trades each day 
plt.figure(figsize=(25,20))
plt.subplots_adjust(top=1.25,bottom =1.2)
for i,company in enumerate(company_name, 1):
    plt.subplot(4,1,i)
    data.loc[(data['ts_code']==company)]['vol'].plot() 
    plt.ylabel('volume')
    plt.xlabel(None)
    plt.title(f"Sales volume of {company}")
plt.tight_layout()


# In[12]:


# What was the moving average of the various stocks?
ma_day =[10,15,20]
for ma in ma_day:
    for company in company_name:
        column_name = f"MA for {ma} days"
        data.loc[(data.ts_code==company),column_name] = data.loc[(data.ts_code==company)]['close'].rolling(ma).mean()
               


# In[13]:


fig,axes = plt.subplots(nrows=4, ncols=1)
fig.set_figheight(20)
fig.set_figwidth(20)
company_name =["PingAn","ChangJiang","XingYe","SanYi"]
data.loc[(data.ts_code=='PingAn')][['close','MA for 10 days','MA for 15 days','MA for 20 days']].plot(ax=axes[0])
axes[0].set_title('PingAn')

data.loc[(data.ts_code=='XingYe')][['close','MA for 10 days','MA for 15 days','MA for 20 days']].plot(ax=axes[1])
axes[1].set_title('XingYe')

data.loc[(data.ts_code=='SanYi')][['close','MA for 10 days','MA for 15 days','MA for 20 days']].plot(ax=axes[2])
axes[2].set_title('SanYi')

data.loc[(data.ts_code=='ChangJiang')][['close','MA for 10 days','MA for 15 days','MA for 20 days']].plot(ax=axes[3])
axes[3].set_title('ChangJiang')

fig.tight_layout()

#let us see a historical view of the closing price


# In[14]:


import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from datetime import datetime


# In[15]:


data_price =  df.loc[(df.ts_code=='600900.SH')][['ts_code','trade_date','open','high','low','close','vol','amount','pre_close','change','pct_chg','date_k','date']] 
data_price


# In[21]:


#2、数据处理
data_price = df.loc[(df.date>'2022-10-01')&(df.ts_code=='600900.SH')][['trade_date','open','high','low','close','vol']]            # 剔除非交易日
#data_price
#data_price.set_index('trade_date', inplace=True)       # 将日期作为索引
#data_price = data_price.astype(float)                    # 将价格数据类型转为浮点数
# 将日期格式转为 candlestick_ohlc 可识别的数值
data_price['Date'] = list(map(lambda x:mdates.date2num(datetime.strptime(x,'%Y%m%d')),data_price.trade_date.tolist()))


# In[24]:


#3、绘制K线图
ohlc = data_price[['Date','open','high','low','close']]
#ohlc.set_index('Date', inplace=True) 
# 提取绘图数据
f1, ax = plt.subplots(figsize = (25,15))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7, colorup='red', colordown='green')           # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('K Chart',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.show()


# In[25]:


data


# In[27]:


for company in company_name:
    data.loc[(data.ts_code==company),'dailyreturn'] = data.loc[(data.ts_code==company)]['pct_chg']
    
# Then we will plot the daily return percentage
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_figheight(8)
fig.set_figwidth(30)
#company_name =["PingAn","ChangJiang","XingYe","SiDa"]
data.loc[(data.ts_code=='PingAn')]['dailyreturn'].plot(ax=axes[0,0],legend=True, linestyle='--', marker='o')
axes[0,0].set_title('PingAn')

data.loc[(data.ts_code=='ChangJiang')]['dailyreturn'].plot(ax=axes[0,1],legend=True, linestyle='--', marker='o')
axes[0,1].set_title('ChangJiang')

data.loc[(data.ts_code=='XingYe')]['dailyreturn'].plot(ax=axes[1,0],legend=True, linestyle='--', marker='o')
axes[1,0].set_title('XingYe')

data.loc[(data.ts_code=='SanYi')]['dailyreturn'].plot(ax=axes[1,1],legend=True, linestyle='--', marker='o')
axes[1,1].set_title('SanYi')

fig.tight_layout()


# In[29]:


plt.figure(figsize=(12,7))
for i,company in enumerate(company_name, 1):
    plt.subplot(2,2,i)
    data.loc[(data.ts_code==company)]['dailyreturn'].hist(bins=50)
    plt.ylabel('Daily Return')
    plt.title(f'{company_name[i-1]}')
plt.tight_layout()


# In[31]:


from pandas.plotting import scatter_matrix


# In[32]:


#对股票数据的列名重新命名
ohlc = df[['open','high','low','close','vol']]
ohlc.columns=['open','high','low','close','vol']
#data=ohlc.loc['2018-01-02':'2018-12-28']  #获取某个时间段内的时间序列数据
scatter_matrix(ohlc[['open','high','low','close','vol']])
plt.figure(figsize=(25, 15)) 
plt.show()


# In[33]:


ohlc = df[['open','high','low','close','vol']]
ohlc.columns=['open','high','low','close','vol']
#data=ohlc.loc['2018-01-02':'2018-12-28']  #获取某个时间段内的时间序列数据
cov=np.corrcoef(ohlc[['open','high','low','close','vol']].T)
print(cov)


# In[34]:


data


# In[56]:


stock_data = data.loc[(data.ts_code=='ChangJiang')&(data.trade_date>'20220101')][['MA for 10 days','MA for 15 days','close']]  


# In[60]:


#
stock_data['close_m10-15']=stock_data['MA for 10 days']-stock_data['MA for 15 days']
stock_data['diff']=np.sign(stock_data['close_m10-15'])
stock_data['diff'].plot(ylim=(-2,2))


# In[61]:


stock_data['signal'] = np.sign(stock_data['diff'] - stock_data['diff'].shift(1))
stock_data['signal'].plot(ylim=(-2,2))


# In[62]:


trade = pd.concat([  pd.DataFrame({"price": stock_data.loc[stock_data["signal"] == 1, "close"],"operation": "Buy"}),
                     pd.DataFrame({"price": stock_data.loc[stock_data["signal"] == -1, "close"],"operation": "Sell"})  ])
trade.sort_index(inplace=True)
print(trade)


# In[63]:


stock_data['earn_rate'] = stock_data['close'].pct_change()
stock_data['earn_rate'].plot(grid=True,color='green',label='Shuoshi Stock')
plt.title('Earn rate of every day(from 2022-01)')
plt.ylabel('earn rate', fontsize='8')
plt.xlabel('date', fontsize='8')
#plt.ylim(0,0.3)  #可以绘制y轴的刻度范围
plt.legend(loc='best',fontsize='small')
plt.show()


# In[78]:


stock_data=stock_data.dropna(axis='index', how='any', subset=['signal','earn_rate'])


# In[79]:


# 计算股票的日平均收益
earn_mean_daily = np.mean(stock_data)
print("Averge Daily return：")
print(earn_mean_daily)


# In[80]:


# 绘制直方图
plt.hist(stock_data['earn_rate'], bins=75)
plt.show()


# In[81]:


stock_data['accumulate']=(1+stock_data['earn_rate']).cumprod()
stock_data['accumulate'].plot()
plt.legend()
plt.show()


# In[82]:


earn_rate_year=(1+np.mean(stock_data['earn_rate']))**252-1
print("Average annualized rate of return：",earn_rate_year)


# In[83]:


earn_rate_range=np.max(stock_data['earn_rate'])-np.min(stock_data['earn_rate'])
earn_rate_interquartile_range=stock_data['earn_rate'].quantile(0.75)-stock_data['earn_rate'].quantile(0.25)
earn_rate_var=np.var(stock_data['earn_rate'])
earn_rate_std=np.std(stock_data['earn_rate'])
earn_rate_coefficient=np.std(stock_data['earn_rate'])/np.mean(stock_data['earn_rate'])
print("Very poor daily rate of return：",earn_rate_range)
print("Daily yield quartile: ",earn_rate_interquartile_range)
print("Daily yield variance: ",earn_rate_var)
print("Standard deviation of daily rate of return: ",earn_rate_std)
print("Daily rate of return dispersion coefficient: ",earn_rate_coefficient)


# In[84]:


earn_mean_year=(1+np.mean(stock_data['earn_rate']))**252-1
earn_var_year=np.std(stock_data['earn_rate'])**2*252
earn_std_year=np.std(stock_data['earn_rate'])*np.sqrt(252)
earn_coefficient_year=earn_std_year/earn_mean_year
print("Average annual rate of return：",earn_mean_year)
print("Annual yield variance: ",earn_var_year)
print("Standard deviation of annual returns: ",earn_std_year)
print("Annual rate of return dispersion coefficient: ",earn_coefficient_year)


# In[85]:


from scipy import stats
# 计算收益分布的偏度
earn_rate_skew=stats.skew(stock_data['earn_rate'])
print("daily yield skewness：",earn_rate_skew)
print("daily yield skewness：",stock_data['earn_rate'].skew())


# In[86]:


from scipy import stats
# 计算收益分布的峰度
earn_rate_kurtosis=stats.kurtosis(stock_data['earn_rate'])
print("daily yield kurtosis：",earn_rate_kurtosis)
print("daily yield kurtosis：",stock_data['earn_rate'].kurt())


# In[87]:


# 模拟正态分布数据，其均值和标准差与文中的股票的日收益率相同。
mu=np.mean(stock_data['earn_rate'])
sigma=np.std(stock_data['earn_rate'])
norm=np.random.normal(mu,sigma,size=10000)
# 绘制正态分布的概率密度分布图
plt. hist(norm, bins=100, alpha=0.8, density=True, label='Normal Distribution')
 
# 绘制收益的概率密度分布图
plt.hist(stock_data['earn_rate'], bins=75, alpha=0.7, density=True,label='earn_rate Distribution')
plt.legend()
plt.show()


# In[88]:


# 从 scipy.stats 导入shapiro
from scipy.stats import shapiro
# 对股票收益进行Shapiro-Wilk检验
shapiro_results = shapiro(stock_data['earn_rate'])
print("Shapiro-Wilk test result: ", shapiro_results)
# 提取P值
p_value = shapiro_results[1]
print("P value: ", p_value)


# In[89]:


# 定义最小周期
min_periods = 20
# 计算波动率
vol = stock_data['earn_rate'].rolling(min_periods).std() * np.sqrt(min_periods)
# 绘制波动率曲线
vol.plot(grid=True)
# 显示绘图结果
plt.show()


# In[9]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from numpy import concatenate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt


# In[10]:


# 20000101-20220922  20220923-20221231
stock_data1 = pro.daily(ts_code='000001.SZ', start_date='20000101', end_date='20220922')
stock_data1['date'] = stock_data1.to_datetime(stock_data1['trade_date'], format='%Y%m%d', errors='coerce')
stock_data1.set_index('date', inplace=True)  
stock_data1.head(30)


# In[56]:


stock_data1= stock_data1.loc[(stock_data1.ts_code=='000001.SZ')][['open','high','low','vol','close']]
stock_data1.dropna()


# In[58]:


test_split=round(len(stock_data1)*0.20)
test_split


# In[61]:


df_for_training=stock_data1[:-1075]
df_for_testing=stock_data1[-1075:]


# In[62]:


scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)


# In[63]:


df_for_testing_scaled=scaler.transform(df_for_testing)


# In[64]:


df_for_training_scaled


# In[128]:


#获取DataFrame中的数据，形式为数组array形式
values=stock_data1.values
#确保所有数据为float类型
values=values.astype('float32')
 
# 特征的归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled)


# In[66]:


df_for_testing_scaled.shape


# In[67]:


df_for_training_scaled.shape


# In[68]:


def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)  


# In[69]:


trainX,trainY=createXY(df_for_training_scaled,30)


# In[70]:


trainX.shape


# In[71]:


testX,testY=createXY(df_for_testing_scaled,30)


# In[72]:


trainX[0]


# In[73]:


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


# In[74]:


def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)


# In[75]:


grid_search = grid_search.fit(trainX,trainY)


# In[77]:


grid_search.best_params_


# In[78]:


my_model=grid_search.best_estimator_.model


# In[79]:


my_model


# In[80]:


prediction=my_model.predict(testX)


# In[81]:


print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)


# In[82]:


prediction.shape


# In[83]:


scaler.inverse_transform(prediction)


# In[84]:


prediction_copies_array = np.repeat(prediction,5, axis=-1)


# In[85]:


prediction_copies_array.shape


# In[86]:


prediction_copies_array


# In[87]:


pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]


# In[88]:


pred


# In[89]:


original_copies_array = np.repeat(testY,5, axis=-1)

original_copies_array.shape

original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]


# In[90]:


pred


# In[91]:


print("Pred Values-- " ,pred)
print("\nOriginal Values-- ",original)


# In[92]:


plt.plot(original, color = 'red', label = 'Real  Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()


# In[93]:


df_30_days_past=df.iloc[-30:,:]


# In[94]:


df_30_days_past


# In[130]:


df_30_days_future=pd.read_csv("test.csv",parse_dates=["Date"],index_col=[0])
df_30_days_future.shape


# In[ ]:


df_30_days_future


# In[ ]:


df_30_days_future["Open"]=0
df_30_days_future=df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler.transform(df_30_days_past)
new_scaled_array=scaler.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)


# In[ ]:


full_df.shape


# In[ ]:


full_df.tail()


# In[ ]:


full_df.shape


# In[ ]:


full_df_scaled_array=full_df.values


# In[ ]:


full_df_scaled_array.shape


# In[ ]:


all_data=[]
time_step=30
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i,0]=prediction


# In[ ]:


all_data


# In[ ]:


new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,


# In[ ]:


y_pred_future_30_days


# In[95]:


df.to_csv('data_listnew.csv',index=None,encoding='utf-8-sig')


# In[105]:


df2 = pro.daily(ts_code='600030.SH', start_date='20210101', end_date='20221231')
df2.to_csv('data_list2.csv',index=None,encoding='utf-8-sig')


# In[107]:


# 创建空的DataFrame变量，用于存储股票数据
StockPrices = pd.DataFrame()
market_value_list=[]  #存储每支股票的平均市值
# 创建股票代码的列表
#'000001.SZ,601166.SH,600900.SH,600031.SH'
ticker_list = ['000001SZ', '601166SH', '600900SH', '600031SH','600030SH']
# 使用循环，挨个获取每只股票的数据，并存储每日收盘价
for ticker in ticker_list:
    stock_data = pd.read_excel('./'+ticker+'.xlsx', parse_dates=['date'], index_col='date')
    #stock_data=stock_data.loc['2016-03-01':'2017-12-29']
    StockPrices[ticker] = stock_data['close']  #获取每支股票的收盘价
    #将每支股票的市值均值存入列表中
    market_value_list.append(stock_data['vol'].mean()) 
StockPrices.index.name = 'date'  # 日期为索引列
# 输出数据的前5行
print(StockPrices.head())


# In[108]:


# 计算每日收益率，并丢弃缺失值
StockReturns = StockPrices.pct_change().dropna()
# 打印前5行数据
print(StockReturns.head())


# In[109]:


# 将收益率数据拷贝到新的变量 stock_return 中，这是为了后续调用的方便
stock_return = StockReturns.copy()
 
# 设置组合权重，存储为numpy数组类型

portfolio_weights = np.array([0.32, 0.15, 0.10, 0.18, 0.25])
# 计算加权的股票收益
WeightedReturns = stock_return.mul(portfolio_weights, axis=1)
# 计算投资组合的收益
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)
# 打印前5行数据
print(StockReturns.head())
 
# 绘制组合收益随时间变化的图
StockReturns.Portfolio.plot()
plt.show()


# In[110]:


# 定义累积收益曲线绘制函数
def cumulative_returns_plot(name_list):
    for name in name_list:
        CumulativeReturns = ((1+StockReturns[name]).cumprod()-1)
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()
# 计算累积的组合收益，并绘图
cumulative_returns_plot(['Portfolio'])


# In[111]:


# 设置投资组合中股票的数目
numstocks = 5
# 平均分配每一项的权重
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)
# 计算等权重组合的收益
StockReturns['Portfolio_EW'] = stock_return.mul(portfolio_weights_ew, axis=1).sum(axis=1)
# 打印前5行数据
print(StockReturns.head())
# 绘制累积收益曲线
cumulative_returns_plot(['Portfolio', 'Portfolio_EW'])


# In[112]:


#将上述获得的每支股票的平均市值转换为数组
market_values=np.array(market_value_list)
# 计算市值权重
market_weights = market_values / np.sum(market_values)
# 计算市值加权的组合收益
StockReturns['Portfolio_MVal'] = stock_return.mul(market_weights, axis=1).sum(axis=1)
# 打印前5行数据
print(StockReturns.head())
# 绘制累积收益曲线
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MVal'])


# In[113]:


# 计算相关矩阵
correlation_matrix = stock_return.corr()
# 输出相关矩阵
print(correlation_matrix)


# In[114]:


import seaborn as sns
#创建热图
sns.heatmap(correlation_matrix,annot=True,cmap='rainbow',linewidths=1.0,annot_kws={'size':8})
plt.xticks(rotation=0)
plt.yticks(rotation=75)
plt.show()


# In[115]:


# 计算协方差矩阵
cov_mat = stock_return.cov()
# 年化协方差矩阵
cov_mat_annual = cov_mat * 252
# 输出协方差矩阵
print(cov_mat_annual)


# In[116]:


# 计算投资组合的标准差
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print(portfolio_volatility)


# In[117]:


# 设置模拟的次数
number = 10000
# 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
random_p = np.empty((number, 7))
# 设置随机数种子，这里是为了结果可重复
np.random.seed(7)
 
#循环模拟10000次随机的投资组合
for i in range(number):
    #生成5个随机数，并归一化，得到一组随机的权重数据
    random5=np.random.random(5)
    random_weight=random5/np.sum(random5)
 
    #计算年平均收益率
    mean_return=stock_return.mul(random_weight,axis=1).sum(axis=1).mean()
    annual_return=(1+mean_return)**252-1
 
    #计算年化标准差，也成为波动率
    random_volatility=np.sqrt(np.dot(random_weight.T,np.dot(cov_mat_annual,random_weight)))
 
    #将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
    random_p[i][:5]=random_weight
    random_p[i][5]=annual_return
    random_p[i][6]=random_volatility
 
#将Numpy数组转化为DataF数据框
RandomPortfolios=pd.DataFrame(random_p)
#设置数据框RandomPortfolios每一列的名称
RandomPortfolios.columns=[ticker +'_weight' for ticker in ticker_list]+['Returns','Volatility']
 
#绘制散点图
RandomPortfolios.plot('Volatility','Returns',kind='scatter',alpha=0.3)
plt.show()


# In[118]:


# 找到标准差最小数据的索引值
min_index = RandomPortfolios.Volatility.idxmin()
 
# 在收益-风险散点图中突出风险最小的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[min_index,'Volatility']
y = RandomPortfolios.loc[min_index,'Returns']
plt.scatter(x, y, color='red')
#将该点坐标显示在图中并保留四位小数
plt.text(np.round(x,4),np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10)
plt.show()


# In[119]:


# 提取最小波动组合对应的权重, 并转换成Numpy数组
GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])
# 计算GMV投资组合收益
StockReturns['Portfolio_GMV'] = stock_return.mul(GMV_weights, axis=1).sum(axis=1)
#输出风险最小投资组合的权重
print(GMV_weights)


# In[120]:


# 设置无风险回报率为0
risk_free = 0
# 计算每项资产的夏普比率
RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility
# 绘制收益-标准差的散点图，并用颜色描绘夏普比率
plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()


# In[121]:


# 找到夏普比率最大数据对应的索引值
max_index = RandomPortfolios.Sharpe.idxmax()
# 在收益-风险散点图中突出夏普比率最大的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']
plt.scatter(x, y, color='red')
#将该点坐标显示在图中并保留四位小数
plt.text(np.round(x,4),np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10)
plt.show()


# In[122]:


# 提取最大夏普比率组合对应的权重，并转化为numpy数组
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])
# 计算MSR组合的收益
StockReturns['Portfolio_MSR'] = stock_return.mul(MSR_weights, axis=1).sum(axis=1)
#输出夏普比率最大的投资组合的权重
print(MSR_weights)


# In[ ]:




