# -*- coding: utf-8 -*-
import time

class TimePrepare():
    
    
    def __init__(self, **kwargs):
        self.dates = dict()
    
    
    """ 本方法是用来将 特殊节假日 加入 特征值中, 并赋予相应的权重值.
    # Arguments
        start_time: 起始时间. 某月某天. 形式要求为 11.11 
        alias: 别名, 将来可以通过别名来获取数据.
        freq: 发生频率, 默认是仅发生一次.
        duration: 持续时间, 从startTime开始算起, 持续几天.
        impact_factor: 影响因子. 接收list 或 单个值. 如果是list, 要求长度等于持续时间.
    """
    def add(self, alias, start_time, duration, freq=None, impact_facotr=1):
        date = Date(start_time, duration, freq, impact_facotr)
        self.dates[alias] = date
        
    #获取是一周的第几天, 星期天是第0天.
    def __get_day_of_week(self, datetime):
        day_of_week = time.strftime('%w', datetime)
        return int(day_of_week)
    
    #获取是一月内的上中下旬.
    def __get_period_of_month(self, datetime):
        day = time.strftime('%d', datetime)
        return int(day) // 10
    
    #是否是工作日, 1表示工作日, 0 表示周末.
    def __is_week_day(self, datetime):
        day_of_week = self.__get_day_of_week(datetime)
        return int(day_of_week == 6 or day_of_week == 0)
    
    """ 用来获取基本的时间特征值
        返回值包括: dayOfWeek, periodOfMonth(每个月的上中下旬), 是否是周内.
    """
    def get_base_feature(self, time_millis):
        #毫秒值转 时间.
        date = time.localtime(float(time_millis) / 1000)
        return self.__get_day_of_week(date), self.__get_period_of_month(date), \
            self.__is_week_day(date)
    
class Date():
    
    def __init__(self, start_time,  duration, freq=None,impact_facotr=1,  **kwargs):
        self.start_time = start_time
        self.freq = freq
        self.duration = duration
        self.impact_facotr = impact_facotr
            