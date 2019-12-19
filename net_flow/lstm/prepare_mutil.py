# -*- coding: utf-8 -*-

'''
提供多变量 LSTM预测.

目前是将时间特征作为特征值.
'''

import time
import csv
import os
import collections
from time_prepare import TimePrepare
from sklearn import preprocessing
import numpy as np

'''
当前时间的数据.
'''
def prepare_data_with_features(files):

    result = []
    for path in files:
        with open('../data/' + path, 'r') as src_file:
            #统计设备对应的端口数量.
            list_data = []
            csv_reader = csv.reader(src_file)
            for raw in csv_reader:
                list_data.append(raw)
            #将数据按照时间进行聚合.
            list_data = agg_data(list_data)
            #填充时间特征.
            list_data = fill_features(list_data)
            # list_data = one_hot(list_data)
            #填充数据. 按日期加和 暂时无需填充数据
            # list_data = fill_data_with_zero(list_data)
            result.extend(list_data)
     
    result = one_hot(result)
    return result

def fill_features(list_data):
    tp = TimePrepare()
    result = list()
    for item in list_data:
        features = tp.get_base_feature(item[-1])
        value = item[0]
        temp = list(features)
        temp.append(value)
        result.append(temp)
    return result
            
def one_hot(list_data):
    np_data = np.array(list_data)
    ohe = preprocessing.OneHotEncoder(sparse = False, categories='auto')
    ans = ohe.fit_transform(np_data[:, 0:2])
    new_data = np.c_[ans, np_data[:, 2:]]
    return new_data.tolist()
            
#将数据按照时间进行聚合.
def agg_data(list_data):
    dict_data = collections.OrderedDict()
    for item in list_data:
        item[4] = int(item[4][0:-1])
        #按日期统计加和.
        temp = get_time_0clock(item[4])
        item[4] = temp
        item[0] = int(item[0]) // 1000
        if temp in dict_data:
            dict_data[temp][0] = dict_data[temp][0] + item[0]
        else:
            dict_data[temp] = item
    result = []
    for k, v in dict_data.items():
        result.append(v)
    return result

#获取当天0点的时间戳
def get_time_0clock(timeMillis):
    t = time.localtime(float(timeMillis) / 1000)
    zero_time = time.mktime(time.strptime(time.strftime('%Y-%m-%d 00:00:00', t),'%Y-%m-%d %H:%M:%S'))
    return int(zero_time) * 1000

#对数据进行填充, 也即填充0值.
def fill_data_with_zero(list_data):
    result = []
    
    half_an_hour = 30 * 60 * 1000
    one_day = 24 * 60 * 60 * 1000
    current_time = get_time_0clock(list_data[0][4])
    end_time = current_time + one_day
    for raw in list_data:
        if (current_time >= end_time):
            break
        elif (int(raw[4]) != current_time):
            #print('current:%s, raw:%s' % (current_time, raw[0]))
            while (int(raw[4]) != current_time and current_time < end_time):
                temp = list(raw)
                temp[4] = current_time
                temp[0] = 0
                result.append(temp)
                current_time += half_an_hour
            #直到相等时, 将当前数据添加进去.
            result.append(raw)
            current_time += half_an_hour
            continue
        current_time += half_an_hour
        result.append(raw)
    while (current_time < end_time):
        temp = [0, '', '', current_time]
        result.append(temp)
        current_time += half_an_hour
    return result

def prepare_data():
    files = os.listdir('../data')
    list_data = prepare_data_with_features(files)
        
    return list_data

def write_data_to_csv(list_data):
    with open('data/new_flow_prepare_mutil.csv', 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in list_data:
            csv_writer.writerow(value)

#验证0值填充的结果         
def verify():
    half_an_hour = 30 * 60 * 1000
    with open('data/new_flow_prepare_mutil.csv', 'r') as new_file:
        csv_reader = csv.reader(new_file)
        pre = 0
        for raw in csv_reader:
            if pre == 0:
                pre = int(raw[0])
                continue
            else:
                pre += half_an_hour
                if pre != int(raw[0]):
                    print(raw[0])
            
# prepare_data()
write_data_to_csv(prepare_data())
# verify()