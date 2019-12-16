# -*- coding: utf-8 -*-

import pandas as pd
from fbprophet import Prophet
import time

def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
                         time.localtime(int(time_stamp) // 1000))

def src_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
                      #将第0列当做索引使用,不再独自创建索引.
                      # index_col=0,
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
    #在源数据上修改列名, 并保存.
    series.rename(columns={0:'ds', 1:'y'}, inplace=True)
    # print(len(series))
    drop_rows = []
    for index, raw in series.iterrows():
        if index % 2 == 0:
            drop_rows.append(index + 1)
            raw['y'] = raw['y'] + series.iat[(index + 1), 1]
    series.drop(drop_rows, inplace=True)
    series.reset_index(inplace=True)
    return series

def pridect(timeseries):
    timeseries['y'] = (timeseries['y'] - timeseries['y'].mean()) / (timeseries['y'].std())
    model = Prophet(weekly_seasonality=True, 
                    daily_seasonality=False)
    model.fit(timeseries)
    future = model.make_future_dataframe(freq='H',periods=168)
    forecast = model.predict(future)
    model.plot(forecast).show()#绘制预测效果图
    model.plot_components(forecast).show()#绘制成分趋势图
    
# print(src_data())
pridect(src_data())