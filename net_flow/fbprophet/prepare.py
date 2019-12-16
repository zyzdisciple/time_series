# -*- coding: utf-8 -*-

import time
import csv
import os
import collections

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
            #填充数据.
            list_data = fill_data_with_zero(list_data)
            result.extend(list_data)
    temp = []
    for item in result:
        temp.append([item[-1], item[0]])
    result = temp
    return result
            
#将数据按照时间进行聚合.
def agg_data(list_data):
    dict_data = collections.OrderedDict()
    for item in list_data:
        item[4] = int(item[4][0:-1])
        temp = item[4]
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

#填充一周之前的数据.
def prepare_data():
    files = os.listdir('../data')
    list_data = prepare_data_with_features(files)
        
    return list_data

def write_data_to_csv(list_data):
    with open('data/new_flow_prepare.csv', 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in list_data:
            csv_writer.writerow(value)

#验证0值填充的结果         
def verify():
    half_an_hour = 30 * 60 * 1000
    with open('data/new_flow_prepare.csv', 'r') as new_file:
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
            

write_data_to_csv(prepare_data())
# verify()