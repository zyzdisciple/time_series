在这一部分, 需要调整的参数在于:

expwighted_avg 的 halflife参数, 等于周期长度.

test_stationarity 的window参数, 也等于周期长度.

stl 方法中的shift参数.

stl 方法中, 如果 trend线性较强 可以采用线性回归的方式. 
不过一般来说使用 arima会更好.

其中包含一系列数据处理方法, 在其他地方可用.