# -*- coding: utf-8 -*-

'''
合并数据, 将数据按时间进行合并, 生成基本序列.
'''
import os
import csv
import time

files = os.listdir('../data')
    
for path in files:
    with open('../data/' + path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        list_data = []
        for raw in csv_reader:
            list_data.append(raw)
    dict_data = {}
    for item in list_data:
        item[4] = item[4][0:-1]
        temp = item[4]
        item[0] = int(item[0]) // 1000
        if temp in dict_data:
            dict_data[temp][0] = dict_data[temp][0] + item[0]
        else:
            dict_data[temp] = item
    with open('data/new_flow_prepare.csv', 'a', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in dict_data.values():
            csv_writer.writerow([value[4], value[0]])


#目前均假定数据正常, 因此需要进行0值填充, 需要填充从15号到26的数据.
#且只填充到26号的最后一条数据, 并不继续向后.
start_time = '2019-11-15 00:00:00'
end_time = '2019-11-27 00:00:00'
half_an_hour = 30 * 60 * 1000
with open('data/new_flow_prepare.csv', 'r') as src_file:
    csv_reader = csv.reader(src_file)
    list_data = []
    current_time = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S")) * 1000
    end_time = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")) * 1000
    current_time = int(current_time)
    end_time = int(end_time)

#    验证数据缺失情况
#    count = 0
#    pre = 0
#    for raw in csv_reader:
#        print(float(raw[0]) - pre)
#        pre = float(raw[0])

    for raw in csv_reader:
        if (current_time > end_time):
            break
        elif (int(raw[0]) != current_time):
            #print('current:%s, raw:%s' % (current_time, raw[0]))
            while (int(raw[0]) != current_time and current_time < end_time):
                list_data.append([current_time, 0])
                current_time += half_an_hour
            #直到相等时, 将当前数据添加进去.
            list_data.append(raw)
            current_time += half_an_hour
            continue
        current_time += half_an_hour
        list_data.append(raw)


with open('data/new_flow_prepare.csv', 'w', newline='') as new_file:
        csv_writer = csv.writer(new_file, delimiter = ',')
        for value in list_data:
            csv_writer.writerow(value)
   