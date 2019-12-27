# -*- coding: utf-8 -*-

import time
import pandas as pd
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.transformer import ClassicSeasonalDecomposition

from adtk.detector import InterQuartileRangeAD

import matplotlib.pyplot as plt
import statsmodels.api as sm

def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
                         time.localtime(int(time_stamp)))

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
    series.rename(columns={0:'timestamp', 1:'value'}, inplace=True)
    return series

#STL 拆分
def seasonal_decompose_my(time_series):
    res = sm.tsa.seasonal_decompose(time_series.values, freq=168, two_sided=False)
    res.plot()
    plt.show()
    return res

s_train = validate_series(src_data())
# print(s_train)

#根据STL数据进行时序分解.
# sd = seasonal_decompose_my(s_train)

#时序数据分解残差
# s_transformed = ClassicSeasonalDecomposition(trend=True).fit_transform(s_train).rename("Seasonal decomposition residual")

# plot(pd.concat([s_train, s_transformed], axis=1), ts_linewidth=1, ts_markersize=1);

#根据 季节性分解所得残差 进行时序数据异常检测
seasonal_ad = SeasonalAD(freq=168, trend=True)

anomalies = seasonal_ad.fit_detect(s_train, return_list=True)

plot(s_train, anomaly_pred=anomalies, ap_color='red', ap_marker_on_curve=True)

# 根据四分位法 进行检测
"""
即异常值通常被定义为小于QL-1.5IQR或大于QU+1.5IQR的值.
其中，QL称为下四分位数，表示全部观察值中有四分之一的数据取值比它小.
QU称为上四分位数，表示全部观察值中有四分之一的数据取值比它大.
IQR称为四分位数间距，是上四分位数QU与下四分位数QL之差，其间包含了全部观察值的一半。

这里指定的 c 就是 上式中的 1.5
"""

# iqr_ad = InterQuartileRangeAD(c=1.5)
# anomalies = iqr_ad.fit_detect(s_train)
# plot(s_train, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);

