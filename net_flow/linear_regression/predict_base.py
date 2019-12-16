# -*- coding: utf-8 -*-

import numpy as np
from csv import reader
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#这里是引用了交叉验证
from sklearn.model_selection import train_test_split

# Load a CSV file, 并将数据转换为 npArray
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return np.array(dataset)

#将上一期的数值进行标准化处理.
def standard_scaler(np_data):
    np_data[:, -2:-1] = preprocessing.StandardScaler().fit_transform(np_data[:, -2:-1])
    return np_data

#one-hot 编码
def one_hot(np_data):
    int_array = []
    for item in np_data:
        int_array.append(list(map(int, item)))
    int_array = np.array(int_array)
    ohe = preprocessing.OneHotEncoder(sparse = False, categories='auto')
    ans = ohe.fit_transform(int_array[:, 2:5])
    new_data = np.c_[int_array[:, 0:2], ans, int_array[:, 5:]]
    return new_data

#准备数据
def preapre_data():
    src_data = load_csv('data/new_flow_prepare.csv');
    src_data = one_hot(src_data)
    src_data = standard_scaler(src_data)
    return src_data

#在 sklearn中, 采取的方式是最小二乘法.
def liner_regression_with_least_square(src_data):
    x_train = src_data[:, 0:-1]
    x_train = np.c_[np.ones([len(x_train), 1]), x_train]
    y_train = src_data[:, -1:]
    linreg = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=0.1, 
                                                        random_state=7)
    model = linreg.fit(x_train, y_train)
    
    print('模型参数:')
    print(model)
    # 训练后模型截距
    print('模型截距:')
    print (linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    print('参数权重:')
    print (linreg.coef_)
    
    y_pred = linreg.predict(x_test)
    
    test_score = mean_squared_error(y_test, y_pred)
    print('Test MSE: %.3f' % test_score)
    
    # 做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()


#最小二乘法+线性回归.
liner_regression_with_least_square(preapre_data())  
#print(preapre_data())