# -*- coding: utf-8 -*-

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import Hour,Minute, Day

def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
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

def predict(df):
    train = df
    test = df[int(0.7 * (len(df))):]
    model = ExponentialSmoothing(train, 
                                 trend = 'add',
                                 seasonal = 'additive', 
                                 seasonal_periods = 336,
                                 freq = Minute(30)
                                 ).fit(smoothing_level = 0.6, smoothing_slope = 0.05)
    
    pred = model.predict(start=test.index[0], end=test.index[-1])
    
    test_score = mean_squared_error(test, pred)
    print('Test MSE: %.3f' % test_score)
    
    
    # plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(pred, label='Holt-Winters')
    # plt.legend(loc='best')
    
    
predict(src_data())