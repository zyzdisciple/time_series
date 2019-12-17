# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def src_data():
    series = pd.read_csv('data/new_flow_prepare_base.csv',
                      #将第0列当做索引使用,不再独自创建索引.
                      index_col=0,
                      #如果没有表头, 即数据从第0行开始, 使用header=None
                      header=None, 
                      # bool, default False
                      #If the parsed data only contains one column then return a Series.
                      squeeze=True)
    #需要指定x轴. y轴.
    # series.plot(x=0, y=1)
    # plt.show()
    
    return series
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def predict(series, n_steps=7, n_features=1):
    X, y = split_sequence(series.values, n_steps)
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model = Sequential()
    # model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # demonstrate prediction
    
    y_hat_list = list()
    y_test_list = list()
    for i in range(len(X)):
        X_test = X[i]
        X_test = X_test.reshape((1, n_steps, n_features))
        yhat = model.predict(X_test, verbose=0)
        y_hat_list.append(yhat[0, 0])
        y_test_list.append(y[i])
    
    print(y_test_list)
    print(y_hat_list)
    plt.clf()
    plt.figure()
    plt.plot(pd.Series(y_test_list))
    plt.plot(pd.Series(y_hat_list))
    plt.show()

'''
CNN可以非常有效地从一维序列数据（例如单变量时间序列数据）中自动提取和学习特征。
n_seq 是指子序列的数量.


'''
def cnn_predict(series, n_steps=8, n_features=1, n_seq=2):
    
    X, y = split_sequence(series.values, n_steps)
    
    split = int(len(X) * 0.9)
    
    X_train = X[0:split]
    y_train = y[0:split]
    
    print(split)
    print(y)
    print(y_train)
    
    n_steps = 4
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
    X_train = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))
    
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(X_train, y_train, epochs=500, verbose=0)
    # demonstrate prediction
    y_hat_list = list()
    y_test_list = list()
    for i in range(len(X)):
        X_test = X[i]
        X_test = X_test.reshape((1, n_seq, n_steps, n_features))
        yhat = model.predict(X_test, verbose=0)
        y_hat_list.append(yhat[0, 0])
        y_test_list.append(y[i])
    plt.figure()
    y_t_s = pd.Series(y_test_list)
    y_h_s = pd.Series(y_hat_list)
    plt.plot(y_t_s, color='blue')
    plt.plot(y_h_s, color='orange')
    plt.show()
    test_score = mean_squared_error(y_t_s, y_h_s)
    print('Test MSE: %.3f' % test_score)
    

series = src_data()


# predict(series)

cnn_predict(series)
