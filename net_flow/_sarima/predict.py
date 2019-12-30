# -*- coding: utf-8 -*-

# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pandas as pd
import itertools
 
# one-step sarima forecast
def sarima_forecast(history, config, n_step):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    end_index = len(history) + n_step - 1
    yhat = model_fit.predict(len(history), end_index)
    return yhat.tolist()
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg, n_step, verbose=0):
    # 根据n_step 预测不同的步数. 预测结果是list. 
    # one by one 预测, 例如当 n_step = 3时, 预测 5-7, 6-8, 7-9... 12-14 当n_step 小于
    # len test时, 则取len test.
    n_step = n_step if n_step <= n_test else n_test
    n_prediction_num = n_test - n_step + 1
    
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # 预测的目标值, 最终用来计算 rmse
    predict_y = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(n_prediction_num):
        temp_test = list()
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg, n_step)
		# store forecast in list of predictions
        predictions.extend(yhat)
        # add actual observation to history for the next loop
        for j in range(n_step):
            temp_test.append(test[i + j])
            predict_y.append(test[i + j])
        history.append(test[i])
        if (verbose==2):
            print(' > test:', temp_test, '\n', '> predict:', yhat)
    # estimate prediction error
    error = measure_rmse(predict_y, predictions) / len(predict_y)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, n_step = 1, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg, n_step)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg, n_step)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, n_step = 1, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg, n_step) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg, n_step) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2, 3, 4, 5]
    d_params = [0, 1, 2]
    q_params = [0, 1, 2, 3, 4, 5]
    # 用于将确定性趋势模型控制为“ n”，“ c”，“ t”，“ ct”之一的参数，分别表示无趋势，恒定，
    # 线性和具有线性趋势的常数。
    t_params = ['n', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    
    models = list(itertools.product(p_params, d_params, q_params, 
                                    P_params, D_params, Q_params, m_params,
                                    t_params))
    
    models = [[(c[0], c[1], c[2]), (c[3], c[4], c[5], c[6]), c[7]] for c in models]
    return models
 
def src_data():
    series = pd.read_csv('data/new_flow_prepare.csv',
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
    result = series.values.tolist()
    return result

if __name__ == '__main__':
    
    data = src_data()
    # define dataset
    # data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    print(data)
    # 从test数据中, 进行连续预测. 一次预测 n_step位. 
    # data split
    n_test = 7
    # 多步预测.
    n_step = 3
    
    # """
    # model configs
    cfg_list = sarima_configs(seasonal=[7])
    # grid search
    scores = grid_search(data, cfg_list, n_test, n_step, parallel=False)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
    # """
    """
    #在已经查找到 参数之后, 进一步处理.
    cfg_list = list()
    cfg_list.append([(0, 0, 0), (0, 1, 0, 7), 'n'])
    walk_forward_validation(data, n_test, [(0, 0, 0), (0, 1, 0, 7), 'n'], n_step, verbose=2)
    """
