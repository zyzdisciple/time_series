# -*- coding: utf-8 -*-

from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
import time

def parser(time_stamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", 
                         time.localtime(int(time_stamp) // 1000))

series = read_csv('data/new_flow_prepare.csv',
                  #将第0列当做索引使用,不再独自创建索引.
                  index_col=0,
                  #如果没有表头, 即数据从第0行开始, 使用header=None
                  header=None, 
                  # bool, default False
                  #If the parsed data only contains one column then return a Series.
                  squeeze=True, 
                  #指定第几列需要进行日期转换, 以及方法.
                  parse_dates=[0], date_parser=parser)
#需要指定x轴. y轴.
series.plot(x=0, y=1)
pyplot.show()


#也就是说，给定在t-1处的观察，预测在t + 1处的观察
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

#我们将保留观察结果的前66％用于“培训”，其余34％用于评估。
#在拆分期间，我们要小心排除NaN值的第一行数据。

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
#从第一行开始, 排除 NaN
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x

# walk-forward validation
#取数据集时存在滞后性, 即用前一位为 入参, 后一位为预测结果, 因此总会滞后一位.
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()


