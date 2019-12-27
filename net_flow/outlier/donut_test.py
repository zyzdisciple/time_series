# -*- coding: utf-8 -*-

"""

donut 基于  VAE的实现.

代码在 tensorflow2.0上还是难以运行. 因此暂时放弃.
"""

import tensorflow as tf
import donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from donut import DonutTrainer, DonutPredictor
import pandas as pd
import numpy as np
from donut import complete_timestamp, standardize_kpi
import time


tf.compat.v1.disable_v2_behavior()

def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
                         time.localtime(int(time_stamp) // 1000))

def preapre_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
                      #如果没有表头, 即数据从第0行开始, 使用header=None
                      header=None)
                      #指定第几列需要进行日期转换, 以及方法.
                      # parse_dates=[0],
                      # date_parser=parser)
    series.rename(columns={0:'timestamp', 1:'value'}, inplace=True)
    labels = np.zeros_like(series.value, dtype=np.int32)
    return (series.timestamp, series.value, labels)


# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.

timestamp, values, labels = preapre_data()

# Complete the timestamp, and obtain the missing point indicators.
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

test_portion = 0.3

test_n = int(len(values) * test_portion)

train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]


# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)



with tf.compat.v1.variable_scope('model') as model_vs:
    model = donut.Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=10,
        z_dims=5,
    )

trainer = DonutTrainer(model=model, model_vs=model_vs)
predictor = DonutPredictor(model)

with tf.compat.v1.Session().as_default():
    trainer.fit(train_values, train_labels, train_missing, mean, std)
    test_score = predictor.get_score(test_values, test_missing)