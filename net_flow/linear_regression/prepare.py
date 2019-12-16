# -*- coding: utf-8 -*-

import pandas as pd
import time
import csv

day_of_week = [1, 2, 3, 4, 5, 6, 7]

#one hot 独热编码, test
day_of_week = pd.get_dummies(day_of_week, prefix=None, prefix_sep='_', 
                   dummy_na=False, columns=None, sparse=False,
                   drop_first=False)

#是否是节假日
is_holidy = [0, 1]

#将时间分为早, 中, 下午, 晚上, 几部分.
time_of_one_day = [1, 2, 3, 4]

#将时间 按照 7 12 14 22 进行分片.
def get_period_of_day(timeMillis):
    hour = time.strftime('%H', time.localtime(float(timeMillis) / 1000))
    hour = int(hour)
    if (hour < 8 or hour >= 20):
        return 3
    elif (hour >= 7 and hour <= 14):
        return 1
    else:
        return 2

#获取是一周的第几天, 星期天是第0天.
def get_day_of_week(timeMillis):
    day_of_week = time.strftime('%w', time.localtime(float(timeMillis) / 1000))
    return int(day_of_week)

#获取是一月内的上中下旬.
def get_period_of_month(timeMillis):
    day = time.strftime('%d', time.localtime(float(timeMillis) / 1000))
    return int(day) // 10

#是否是工作日, 1表示工作日, 0 表示周末.
def is_week_day(timeMillis):
    day_of_week = get_day_of_week(timeMillis)
    return int(day_of_week == 6 or day_of_week == 0)

#获取当前时间的特征值.
def get_feature_of_time(timeMillis):
    return [get_period_of_day(timeMillis), get_day_of_week(timeMillis),
            get_period_of_month(timeMillis), is_week_day(timeMillis)]

#加载文件
#files = os.listdir('../data')

files = ['netflow_raw_1m_2019.11.22.csv', 'netflow_raw_1m_2019.11.23.csv', 
         'netflow_raw_1m_2019.11.24.csv', 'netflow_raw_1m_2019.11.25.csv']

prefix_files = [
        'netflow_raw_1m_2019.11.15.csv', 'netflow_raw_1m_2019.11.16.csv', 
         'netflow_raw_1m_2019.11.17.csv', 'netflow_raw_1m_2019.11.18.csv' ]

'''
接收两个参数, 当前时间的数据, 以及上周对应的数据.
'''
def prepare_data_with_features(files):

    result = []
    for path in files:
        with open('../data/' + path, 'r') as src_file:
            #统计设备对应的端口数量.
            port_map = {}
            list_data = []
            csv_reader = csv.reader(src_file)
            for raw in csv_reader:
                #统计端口数量. 仅统计一天内的数量.
                ip = raw[3]
                in_port = raw[1]
                out_port = raw[2]
                port_set = set()
                if (ip in port_map):
                    port_set = port_map[ip]
                else:
                    port_map[ip] = port_set
                port_set.add(in_port)
                port_set.add(out_port)
                list_data.append(raw)
            #将数据按照时间进行聚合.
            list_data = agg_data(list_data)
            list_data = fill_data_with_zero(list_data)
            
            #填充数据.
            
            #在当天, 总设备数, 总端口数.
            #统计当天的总的设备数, 端口数.
            device_num = len(port_map)
            port_num = 0
            for item in port_map.values():
                port_num += len(item)
            #数据处理
            for item in list_data:
                temp_list = []
                #设备数
                temp_list.append(device_num)
                #端口数
                temp_list.append(port_num)
                #时间特征值
                temp_list.extend(get_feature_of_time(item[4]))
                #流量, 也即y.
                temp_list.append(item[0])
                item.clear()
                item.extend(temp_list)
            result.extend(list_data)
    return result
            
#将数据按照时间进行聚合.
def agg_data(list_data):
    dict_data = {}
    for item in list_data:
        item[4] = int(item[4][0:-1])
        temp = item[4]
        item[0] = int(item[0]) // 1000
        if temp in dict_data:
            dict_data[temp][0] = dict_data[temp][0] + item[0]
        else:
            dict_data[temp] = item
    return list(dict_data.values())

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
    return result

#填充一周之前的数据.
def prepare_data(files, prefix_files):
    list_data = prepare_data_with_features(files)
    prefix_data = prepare_data_with_features(prefix_files)
    for index, val in enumerate(list_data):
        val.append(prefix_data[index][-1])
        val[-1] = val[-2]
        val[-2] = prefix_data[index][-1]
        
    return list_data

def write_data_to_csv(list_data):
    with open('data/new_flow_prepare.csv', 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in list_data:
            csv_writer.writerow(value)

write_data_to_csv(prepare_data(files, prefix_files))