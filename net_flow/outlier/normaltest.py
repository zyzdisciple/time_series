# -*- coding: utf-8 -*-

import scipy as sp
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn

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

def random_data():
    seed(1)
    # generate univariate observations
    data = 5 * randn(10000)
    return data


def equals_one():
    data = [1] * 50 + [2] * 50 + [3] * 50
    return data

data = src_data().values
# data = random_data()
# data = equals_one()
print(data)
result = sp.stats.kstest(data, 'norm')
print(result)
plt.figure()
plt.hist(data, bins=14, density=1)