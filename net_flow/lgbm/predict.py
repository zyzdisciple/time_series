# -*- coding: utf-8 -*-

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def src_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
                      #如果没有表头, 即数据从第0行开始, 使用header=None
                      header=None)
    #需要指定x轴. y轴.
    # series.plot(x=0, y=1)
    # plt.show()
    return series

def predict(df):
    x_train = df.iloc[:, 0:-1]
    y_train = df.iloc[:, -1:]
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=0.1, 
                                                        random_state=7)
    #转换为lgb数据, 并使用 特征分类. 无需自己 热编码.
    train_data = lgb.Dataset(x_train, label = y_train, feature_name=['dns', 'pns', 
                                                 'pod', 'dow',
                                                 'pom', 'isw', 
                                                 'pd'],
                          categorical_feature=['pod', 'dow',
                                               'pom', 'isw'])
    lgb_eval = lgb.Dataset(x_test,label = y_test,
                           reference=train_data,
                           feature_name=['dns', 'pns', 
                                                 'pod', 'dow',
                                                 'pom', 'isw', 
                                                 'pd'],
                           categorical_feature=['pod', 'dow',
                                               'pom', 'isw']
                           
                           )
    # lgb 参数
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression', # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,   # 叶子节点数
        'max_bin': 255, #最大数特征值
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9, # 建树的特征选择比例
        'bagging_fraction': 0.8, # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': -1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    
    lgbm = lgb.train(params, train_data, num_boost_round=100,
                     valid_sets=lgb_eval)
    
    # 预测数据集
    y_pred = lgbm.predict(x_test, num_iteration=lgbm.best_iteration)
    # 评估模型
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred))
    
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    
predict(src_data())