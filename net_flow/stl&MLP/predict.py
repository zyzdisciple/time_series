# -*- coding: utf-8 -*-

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import csv


def parser(time_stamp):
    return time.strftime("%Y-%m-%d", 
                         time.localtime(int(time_stamp) // 1000))

def src_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
                      #将第0列当做索引使用,不再独自创建索引.
                      index_col=0,
                      #如果没有表头, 即数据从第0行开始, 使用header=None
                      header=None, 
                      # bool, default False
                      #If the parsed data only contains one column then return a Series.
                      squeeze=True,
                      #指定第几列需要进行日期转换, 以及方法.
                      parse_dates=[0], 
                      date_parser=parser)
    #需要指定x轴. y轴.
    # series.plot(x=0, y=1)
    # plt.show()
    return series

#STL 拆分
def seasonal_decompose_my(time_series):
    res = sm.tsa.seasonal_decompose(time_series.values, freq=7, two_sided=False)
    res.plot()
    plt.show()
    return res


'''
Dickey-Fuller检验：测试结果由测试统计量和一些置信区间的临界值组成。
如果“测试统计量”少于“临界值”，并认为序列是稳定的。
Test Statistic的值如果比Critical Value (5%)小则满足稳定性需求
p-value越低（理论上需要低于0.05）证明序列越稳定。
'''
def test_stationarity(timeseries):
    
    #336是滑动窗口长度. 目前是以一周的数据作为预测.
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()
 
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#加权平均
def expwighted_avg(timeseries):
    #336是指用过去一周的数据来做平滑
    expwighted_avg = timeseries.ewm(halflife=7).mean()
    plt.plot(timeseries)
    plt.plot(expwighted_avg, color='red')
    return expwighted_avg

#自动使用 arima计算合适的 pqd. 在 auto_arima中已经指定了 pqd的最大值.
def auto_arima_my(time_series, plot=False, all_data = True):
    if all_data:
        train = time_series
        valid = time_series
    else:
        train = time_series[:int(0.7*(len(time_series)))]
        valid = time_series[int(0.7*(len(time_series))):]
    # train.drop(0, axis=1, inplace=True)
    # train(test)
    # train.plot()
    
    model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train)
     
    forecast = model.predict(n_periods=len(valid))
    forecast = pd.DataFrame(forecast, index = valid.index)
    
    if plot:
        # plot the predictions for validation set
        plt.plot(train, label = 'Train')
        plt.plot(valid, label = 'Valid')
        plt.plot(forecast, label = 'Prediction')
        plt.show()
    
    test_score = mean_squared_error(valid, forecast)
    print('Test MSE: %.3f' % test_score)
    return forecast
    
def stl(time_series):
    
    sd = seasonal_decompose_my(time_series)
    #残差值的arima预测
    resid_data = pd.Series(sd.resid)
    #加上趋势, 季节值.
    trend_data = pd.Series(sd.trend)
    
    
    print(trend_data)
    
    seasonal_data = pd.DataFrame(sd.seasonal)
    seasonal_data = seasonal_data.shift(7)
    
    #删除 nan 并保存.
    resid_data.dropna(inplace=True)
    trend_data.dropna(inplace=True)
    seasonal_data.dropna(inplace=True)
    
    #将时间序列数据去除date列.
    date_to_index = pd.Series(time_series.values)
    
    
    #通过Arima 预测 残差和 trend值.
    predictions_arima = auto_arima_my(resid_data)
    predictions_arima_trend = auto_arima_my(trend_data)
    
    # print(predictions_arima_trend)
    # plt.plot(predictions_arima_trend)
    #一旦使用 show表示当前图完结.
    # plt.show()
    
    #绘制主图
    # plt.plot(date_to_index,color='blue')
    # plt.show()
    
    #分别预测 趋势 季节. 最后再加和.
    predictions_arima = predictions_arima.add(predictions_arima_trend, fill_value = 0).add(seasonal_data, fill_value = 0)
    plt.figure() 
    plt.plot(date_to_index, color='blue')
    plt.plot(predictions_arima,color='red')
    
    valid = pd.DataFrame(time_series.values)
    test_score = sqrt(mean_squared_error(valid[6:], predictions_arima))
    print('Test MSE: %.3f' % test_score)
    
def write_data_to_csv(list_data):
    with open('data/new_flow_prepare.csv', 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in list_data:
            csv_writer.writerow(value)

'''
#测试 残差值的 稳定性.
resid_data = pd.Series(seasonal_decompose_my(src_data()).resid)
resid_data.dropna(inplace=True)
test_stationarity(resid_data)
'''
stl(src_data())
