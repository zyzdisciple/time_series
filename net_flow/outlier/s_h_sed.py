# -*- coding: utf-8 -*-

from pyculiarity import detect_ts
import pandas as pd
import time
import matplotlib.pyplot as plt


def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
                         time.localtime(int(time_stamp)))

def src_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
                      #如果没有表头, 即数据从第0行开始, 使用header=None
                      header=None,
                      #指定第几列需要进行日期转换, 以及方法.
                      parse_dates=[0], 
                      date_parser=parser)
    series.rename(columns={0:'timestamp', 1:'value'}, inplace=True)
    return series

df = src_data()

results = detect_ts(df, max_anoms=0.007,direction='both')

# print(results['anoms']['timestamp'])
# results['anoms']['timestamp']
plt.plot(df.timestamp, df.value)
plt.plot(results['anoms']['timestamp'], results['anoms']['anoms'], 'ro')
plt.grid(True)
plt.show()
