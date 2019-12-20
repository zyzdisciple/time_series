# -*- coding: utf-8 -*-

"""
关于LSTM的代码已经是相当完善, 并且通过 加入参数的方式, 发现预测结果与CNN-LSTM相差不大.
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt, floor
from sklearn.preprocessing import MinMaxScaler


# load data set
#def load_dataset(file_path='dataset.csv', header_row_index=0, index_col_name =None, col_to_predict, cols_to_drop=None):
def _load_dataset(file_path, col_to_predict, header_row_index=None, 
                  index_col_name=None, cols_to_drop=None):
    
    """
    file_path: the csv file path
    header_row_index: the header row index in the csv file
    index_col_name: the index column (can be None if no index is there)
    col_to_predict: the column name/index to predict, can't be negative number
    cols_to_drop: the column names/indices to drop (single label or list-like)
    """
    # read dataset from disk
    dataset = pd.read_csv(file_path, header=header_row_index, 
                          index_col=index_col_name)
    
    # drop unused colums
    if cols_to_drop:
        dataset.drop(cols_to_drop, axis =1, inplace = True)
    
    # get rows and column names
    col_names = dataset.columns.values.tolist()
    values = dataset.values
    
    # move the column to predict to be the first col
    col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
    
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
        col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[col_to_predict_index+1:]
    values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)), values[:,:col_to_predict_index], values[:,col_to_predict_index+1:]), axis=1)
    # print(col_names, '\n values2\n', values)
    # ensure all data is float
    values = values.astype("float32")
    # print(col_names, '\n values3\n', values)
    return col_names, values, values.shape[1], output_col_name


# scale dataset
#def _scale_dataset(values, scale_range = (0,1)):
def _scale_dataset(values, scale_range):
    """
    values: dataset values
    scale_range: scale range to fit data in
    """
    # normalize features
    scaler = MinMaxScaler(feature_range=scale_range or (0, 1))
    scaled = scaler.fit_transform(values)
 
    return (scaler, scaled)


"""
convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2),
def _series_to_supervised(values, n_in=3, n_out=1, dropnan=True, col_names, verbose=True):
"""
def _series_to_supervised(values, n_in, n_out, col_names=None, dropnan=True, verbose=True):
    """
    values: dataset scaled values
    n_in: number of time lags (intervals) to use in each neuron, 与多少个之前的time_step相关,和后面的n_intervals是一样
    n_out: number of time-steps in future to predict，预测未来多少个time_step
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
    """
 
    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None: col_names = ["var%d" % (j+1) for j in range(n_vars)]
    df = pd.DataFrame(values)
    cols, names = list(), list()
 
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))         #这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]
 
    # put it all together
    agg = pd.concat(cols, axis=1)    #将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
    agg.columns = names
 
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
 
    if verbose:
        print("\nsupervised data shape:", agg.shape)
    return agg


"""split into train and test sets
def _split_data_to_train_test_sets(values, n_intervals=3, n_features, train_percentage=0.67, verbose=True):
"""
def _split_data_to_train_test_sets(values, n_intervals, n_features,
                                   train_percentage, verbose=True):
    """
    values: dataset supervised values
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    train_percentage: percentage of train data related to the dataset series size; (1-train_percentage) will be for test data
    verbose: whether to output some debug data
    """
    #ceil(x)->得到最接近的一个不大于x的整数, 防止 test中数据为0, 如floor(2.9)=2
    n_train_intervals = floor(values.shape[0] * train_percentage) #
    train = values[:n_train_intervals, :]
    test = values[n_train_intervals:, :]
 
    # split into input and outputs
    n_obs = n_intervals * n_features
    #train_Y直接赋值倒数第六列，刚好是t + n_out_timestep-1时刻的0号要预测列
    #train_X此时的shape为[train.shape[0], timesteps * features]
    train_X, train_y = train[:, :n_obs], train[:, -n_features]  
                                                                
    #print('before reshape\ntrain_X shape:', train_X.shape)
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
 
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_intervals, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))
 
    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)
 
    return (train_X, train_y, test_X, test_y)


"""
create the nn model
def _create_model(train_X, train_y, test_X, test_y, n_neurons=20,
 n_batch=50, n_epochs=60, is_stateful=False, has_memory_stack=False,
 loss_function='mse', optimizer_function='adam', draw_loss_plot=True,
 output_col_name, verbose=True):
"""
def _create_model(train_X, train_y, test_X, test_y, n_neurons,
                  n_batch, n_epochs, is_stateful, has_memory_stack,
                  loss_function, optimizer_function, activation,
                  output_col_name, draw_loss_plot=True, verbose=True):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    n_neurons: number of neurons for LSTM nn, units
    n_batch: nn batch size
    n_epochs: training epochs
    is_stateful: whether the model has memory states
    has_memory_stack: whether the model has memory stack
    loss_function: the model loss function evaluator
    optimizer_function: the loss optimizer function
    activation: 激活函数, 默认是 tanh
    draw_loss_plot: whether to draw the loss history plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """
 
    # design network
    model = Sequential()
 
    if is_stateful:
        # calculate new compatible batch size
        for i in range(n_batch, 0, -1):
            if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:
                if verbose and i != n_batch:
                    print ("\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
                n_batch = i
                break
 
        model.add(LSTM(n_neurons, activation=activation, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=has_memory_stack))
        if has_memory_stack:
            model.add(LSTM(n_neurons, activation=activation, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
    else:
        model.add(LSTM(n_neurons, activation=activation, input_shape=(train_X.shape[1], train_X.shape[2])))
 
    model.add(Dense(1))
 
    model.compile(loss=loss_function, optimizer=optimizer_function)
 
    if verbose:
        print("")
 
    # fit network
    losses = []
    val_losses = []
    if is_stateful:
        for i in range(n_epochs):
            history = model.fit(train_X, train_y, epochs=1, batch_size=n_batch, 
                                validation_data=(test_X, test_y), verbose=0, shuffle=False)
 
            if verbose:
                print("Epoch %d/%d" % (i + 1, n_epochs))
                print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))
 
            losses.append(history.history["loss"][0])
            val_losses.append(history.history["val_loss"][0])
 
            model.reset_states()
    else:
        history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, 
                            validation_data=(test_X, test_y), verbose=2 if verbose else 0, shuffle=False)
    
    
    if draw_loss_plot:
        plt.plot(history.history["loss"] if not is_stateful else losses, label="Train Loss (%s)" % output_col_name)
        plt.plot(history.history["val_loss"] if not is_stateful else val_losses, label="Test Loss (%s)" % output_col_name)
        plt.legend()
        plt.show()
    
    print(history.history)
    #model.save('./my_model_%s.h5'%datetime.datetime.now())
    return (model, n_batch)

# make a prediction
#def _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals=3, n_features, scaler=(0,1), draw_prediction_fit_plot=True, output_col_name, verbose=True):
def _make_prediction(model, train_X, train_y, test_X, test_y, 
                     compatible_n_batch, n_intervals, n_features, 
                     scaler, output_col_name, draw_prediction_fit_plot=True, verbose=True):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    compatible_n_batch: modified (compatible) nn batch size
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    scaler: the scaler object used to invert transformation to real scale
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """
 
    yhat = model.predict(test_X, batch_size=compatible_n_batch, verbose = 1 if verbose else 0)
    test_X = test_X.reshape((test_X.shape[0], n_intervals*n_features))
 
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, (1-n_features):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
 
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, (1-n_features):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
 
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
 
    # calculate average error percentage
    avg = np.average(inv_y)
    error_percentage = rmse / avg
 
    if verbose:
        print("")
        print("Test Root Mean Square Error: %.3f" % rmse)
        print("Test Average Value for %s: %.3f" % (output_col_name, avg))
        print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
        print("test Y:", inv_y, "\npredict y:", inv_yhat)
 
    if draw_prediction_fit_plot:
        plt.figure()
        plt.plot(inv_y, label="Actual (%s)" % output_col_name)
        plt.plot(inv_yhat, label="Predicted (%s)" % output_col_name)
        #显示图例
        plt.legend()
 
    return (inv_y, inv_yhat, rmse, error_percentage)

def main():
    #!input
    file_path = 'data/new_flow_prepare_mutil.csv'
    
    col_names, values,n_features, output_col_name = _load_dataset(file_path, 10)
    #default range is (0, 1)
    scaler, values = _scale_dataset(values, None)
    print('\nvalue shape:', values.shape)


    n_in_timestep = 7
    n_out_timestep = 1
    
    agg = _series_to_supervised(values, n_in_timestep,
                                 n_out_timestep)
    #agg = _series_to_supervised(values, 1, 2, dropnan, col_names, verbose)
    #agg = _series_to_supervised(values, 2, 1, dropnan, col_names, verbose)
    #agg = _series_to_supervised(values, 3, 2, dropnan, col_names, verbose)
    
    #agg1和agg1.value是不一样的，agg1是DataFrame，agg1.value是np.array
    print('\nagg.shape:', agg.shape)
    #print('\nagg1\n', agg1)
    
    train_percentage = 0.9
    train_X, train_Y, test_X, test_Y = _split_data_to_train_test_sets(agg.values, 
                                                                      n_in_timestep, 
                                                                      n_features, 
                                                                      train_percentage)
     
    n_neurons=50
    n_batch=1
    n_epochs=600
    is_stateful=False
    has_memory_stack=False
    loss_function='mse'
    optimizer_function='adam'
    activation='relu'
    model, compatible_n_batch = _create_model(train_X, train_Y,
                                              test_X,
                                              test_Y,
                                              n_neurons,
                                              n_batch,
                                              n_epochs,
                                              is_stateful,
                                              has_memory_stack,
                                              loss_function,
                                              optimizer_function,
                                              activation,
                                              output_col_name,
                                              False,
                                              False)
    
    
    test_X = np.concatenate((train_X, test_X), axis = 0)
    test_Y = np.concatenate((train_Y, test_Y), axis = 0)
    # test_Y = train_Y.append(test_Y)
    actual_target, predicted_target, \
        error_value, error_percentage = _make_prediction(model, train_X, 
                                                         train_Y, 
                                                         test_X, 
                                                         test_Y, 
                                                         compatible_n_batch, 
                                                         n_in_timestep, 
                                                         n_features, 
                                                         scaler,  
                                                         output_col_name)
    
    #model.save('./my_model_%s.h5'%datetime.datetime.now())
    # model.save('./my_model_in time step_%d_out_timestep_%d.h5'%n_in_timestep%n_out_timestep)

main()
