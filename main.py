#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Created by Klaus Lee on 2021/6/20
-------------------------------------------------
"""

from functools import wraps
import os
import numpy as np
import pandas as pd
import datetime
import calendar
from scipy.interpolate import CubicSpline
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import chinese_calendar


# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'Data')
RESULT_PATH = os.path.join(ROOT_PATH, 'Result')


def time_decorator(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        end = datetime.datetime.now()
        print(f'“{func.__name__}” run time: {end - start}.')
        return result
    return timer


@time_decorator
def easy_read_csv(path, file_name, dedicated_filter=True, **kwargs):
    read_conds = {
        'encoding': 'utf-8',
        'engine': 'python',
        'index_col': None,
        'skiprows': None,
        'na_values': np.nan,
        **kwargs
    }
    data = pd.read_csv(os.path.join(path, file_name), **read_conds)
    if dedicated_filter:
        pass
    # 日期格式化
    data['DATE'] = pd.to_datetime(data['DATE'])
    # 按日期由近到远排序
    data.sort_values(by='DATE', ascending=False, inplace=True)
    # 重置index
    data.reset_index(inplace=True, drop=True)
    return data


data_fr007 = easy_read_csv(DATA_PATH, 'FR007.csv')
dic_deal = {'StartDate': datetime.datetime(2021, 1, 13),
            'EndDate': datetime.datetime(2026, 1, 13),
            'Life': '5Y',
            'NominalPrincipal': 30000000.0,
            'AdjPayDay': '经调整的下一营业日',
            'PayType': 'Float',
            'FixRate': 2.8525,
            'FixPayCycle': '3M',
            'FixIntType': '实际/365',
            'FixFirstPayDay': datetime.datetime(2021, 4, 13),
            'FloatRate': 'FR007',
            'FloatPayCycle': '3M',
            'FloatIntType': '实际/365',
            'FloatFirstPayDay': datetime.datetime(2021, 4, 13),
            'FloatFirstSetIntDay': datetime.datetime(2021, 1, 12),
            'FloatResetIntCycle': '7D',
            'FloatBPS': 0.0,
            }
data_deal = pd.DataFrame(dic_deal, index=[0])


class Swap:
    def __init__(self, deal=data_deal.iloc[0, :], _date=datetime.datetime(2021, 5, 28),
                 _time='16:30', _type='均值'):
        self._date, self._time, self._type = _date, _time, _type
        self.deal = deal.to_dict()
        self.deal_adj()
        # print(self.deal)
        self.data, self.database = self.__choose_data()
        self.fix_N, self.float_N, self.float_reset_N = self.N()
        self.fix_payday_list, self.float_payday_list, self.float_reset_list = self.__set_keydays()
        self.fix_df, self.float_df, self.fix_x_1, self.float_x_1, self.float_reset_df = self.set_interest()
        self.value = self._value()
        pass

    def deal_adj(self):
        key_list = ['StartDate', 'EndDate', 'FixFirstPayDay', 'FloatFirstPayDay', 'FloatFirstSetIntDay']
        for key in key_list:
            # self.deal[key] = datetime.datetime.fromtimestamp(self.deal[key])
            # 这里的timestamp是pd.timestamp而非原生的timestamp，因此要用pd里的函数转化格式
            self.deal[key] = self.deal[key].to_pydatetime()

    def __choose_data(self):
        # 选择浮动利率数据库
        if self.deal['FloatRate'] == 'FR007':
            data = data_fr007
        else:
            data = data_fr007
        # 根据日期 时刻 价格类型选择数据
        filter_list_0 = [i and j and k for i, j, k in zip(data['DATE'].eq(self._date),
                                                          data['TIME'].eq(self._time),
                                                          data['TYPE'].eq(self._type),)]
        filter_list_1 = [i and j for i, j in zip(data['TIME'].eq(self._time),
                                                 data['TYPE'].eq(self._type),)]
        database = data.loc[filter_list_1]
        # 向前填充空日期，遍历合约期间全部日期
        data_cache = pd.DataFrame
        date_range = pd.date_range(self.deal['FloatFirstSetIntDay'], self._date)
        # print(date_range)
        for i in range(len(date_range)):
            if date_range[i] in database['DATE'].to_numpy():
                data_cache = database.loc[database['DATE'] == date_range[i]]
            else:
                data_cache.loc[:, 'DATE'] = date_range[i]
                database = pd.concat([database, data_cache])
        # 按日期由近到远排序
        database.sort_values(by='DATE', ascending=False, inplace=True)
        # 重置index
        database.reset_index(inplace=True, drop=True)
        # print(database)
        data = data.loc[filter_list_0]
        # 重置index
        data.reset_index(inplace=True, drop=True)
        database.reset_index(inplace=True, drop=True)
        return data, database

    def N(self):
        # 计算支付次数
        def set_timedelta(paycycle):
            if paycycle[-1] == 'Y':
                time_delta = int(paycycle[:-1]) * datetime.timedelta(days=365)
            elif paycycle[-1] == 'M':
                time_delta = int(paycycle[:-1]) * datetime.timedelta(days=30)
            elif paycycle[-1] == 'D':
                time_delta = int(paycycle[:-1]) * datetime.timedelta(days=1)
            else:
                print('error')
                time_delta = int(paycycle[:-1]) * datetime.timedelta(days=1)
            return time_delta
        fix_timedelta = set_timedelta(self.deal['FixPayCycle'])
        float_timedelta = set_timedelta(self.deal['FloatPayCycle'])
        float_reset_timedelta = set_timedelta(self.deal['FloatResetIntCycle'])
        fix_N = np.trunc((self.deal['EndDate']-self.deal['StartDate'])/fix_timedelta).astype(np.int)
        float_N = np.trunc((self.deal['EndDate']-self.deal['StartDate'])/float_timedelta).astype(np.int)
        float_reset_N = -1 + np.trunc((self.deal['EndDate']-self.deal['FloatFirstSetIntDay'])/float_reset_timedelta).astype(np.int)
        return fix_N, float_N, float_reset_N

    def date_next(self, date_raw, date_delta):
        # 计算给定日期间隔的下一日
        delta_signal = date_delta[-1]
        delta_num = int(date_delta[:-1])
        next_year_raw = date_raw.year + delta_num
        next_month_raw = date_raw.month + delta_num
        next_day_raw = date_raw.day + delta_num
        # 只考虑存在性不考虑节假日调整
        if delta_signal == 'Y':
            if date_raw.month == 2 and date_raw.day == 29 and not calendar.isleap(next_year_raw):
                date_next_raw = date_raw.replace(year=date_raw.year+delta_num, month=3, day=1)
            else:
                date_next_raw = date_raw.replace(year=date_raw.year+delta_num)
        elif delta_signal == 'M':
            # 考虑到最多跨一年
            if delta_num >= 12:
                print('Are you OK?')
            month_list_raw = range(date_raw.month, next_month_raw+1, 1)
            month_list = [datetime.datetime(date_raw.year, i, 1) if i <= 12 else
                          datetime.datetime(date_raw.year + 1, i - 12, 1) for i in month_list_raw]
            # 先规划，后顺延
            month_list[0] = month_list[0].replace(day=date_raw.day)
            if date_raw.day == 29 and month_list[-1].month == 2:
                if calendar.isleap(month_list[-1].year):
                    month_list[-1] = month_list[-1].replace(day=date_raw.day)
                else:
                    month_list[-1] = month_list[-1].replace(month=3, day=1)
            elif date_raw.day == 30 and month_list[-1].month == 2:
                month_list[-1] = month_list[-1].replace(month=3, day=1)
            elif date_raw.day == 31 and month_list[-1].month in [2, 4, 6, 9, 11]:
                # 考虑了这里加一月不会出现跨年问题
                month_list[-1] = month_list[-1].replace(month=month_list[-1].month + 1, day=1)
            else:
                month_list[-1] = month_list[-1].replace(day=date_raw.day)
            date_next_raw = month_list[-1]
        elif delta_signal == 'D':
            # 不用考虑，直接timedelta
            if delta_num >= 31:
                print('Are you OK?')
            date_next_raw = date_raw + datetime.timedelta(days=delta_num)
        else:
            print('error')
            date_next_raw = date_raw
        # TODO:检验节假日
        return date_next_raw

    def __set_keydays(self):
        # 计算起息日和各个支付日
        fix_payday_list = [self.deal['StartDate']]
        float_payday_list = [self.deal['StartDate']]
        for i in range(self.fix_N):
            fix_payday_list.append(self.date_next(fix_payday_list[-1], self.deal['FixPayCycle']))
        fix_payday_list[1] = self.deal['FixFirstPayDay']
        fix_payday_list[-1] = self.deal['EndDate']
        for j in range(self.float_N):
            float_payday_list.append(self.date_next(float_payday_list[-1], self.deal['FloatPayCycle']))
        float_payday_list[1] = self.deal['FloatFirstPayDay']
        float_payday_list[-1] = self.deal['EndDate']
        # 浮动方季度支付但七日计息，存在复利计息的问题
        # 计算浮动利率报价日(每七日)和对应利率
        float_reset_list = [self.deal['FloatFirstSetIntDay']]
        for k in range(self.float_reset_N):
            float_reset_list.append(self.date_next(float_reset_list[-1], self.deal['FloatResetIntCycle']))
        # print(fix_payday_list)
        # print(float_payday_list)
        # print(float_reset_list)
        # 返回正确合规的详细日期
        return fix_payday_list, float_payday_list, float_reset_list

    @time_decorator
    def set_interest(self):
        # 根据估值当日关键点利用三次样条插值计算收益率曲线，并代入实际剩余天数计算FR007利率
        '''
        x为相应期限对应的实际剩余天数
        y为对应利率
        '''
        x_cycle = self.data.columns[4:].to_numpy()
        x_timedelta = np.array([(self.date_next(self._date, i) - self._date).days for i in x_cycle])
        y_interest = [self.data[i].to_numpy()[0] for i in x_cycle]
        # print(x_timedelta, y_interest)
        curve = CubicSpline(x_timedelta, y_interest)
        # 返回定价和折现所需的各个支付日实际剩余天数以及观测日当日的各支付日对应的浮动利率DF
        fix_x_raw = np.array([(i - self._date).days for i in self.fix_payday_list])
        fix_x = fix_x_raw[fix_x_raw > 0]
        # 补充上一支付日,用以计算当前所处的计息周期的期限
        fix_x_1 = fix_x_raw[-len(fix_x)-1:]
        fix_y = curve(fix_x)
        fix_df = pd.DataFrame({'fix_x': fix_x, 'fix_y': fix_y})
        float_x_raw = np.array([(i - self._date).days for i in self.float_payday_list])
        float_x = float_x_raw[float_x_raw > 0]
        # 补充上一支付日
        float_x_1 = float_x_raw[-len(float_x)-1:]
        float_y = curve(float_x)
        float_df = pd.DataFrame({'float_x': float_x, 'float_y': float_y})
        # 浮动利率计息，需要考虑估值日当日到前后两个支付日当日的应计利息和未记利息
        # 考虑关键点上一支付日，本支付周期内第一个计息周期的报价日，估值日前的各个报价日，估值日后的各个报价日，下一支付日
        # 本支付周期内第一个计息周期的报价日早于等于上一支付日
        # 若支付日与报价日在同一天，则由于用收盘价报价而导致实际上报了下个计息周期的价格，因此这里必须要用 > 和 <=
        float_reset_raw = np.array([(i - self._date).days for i in self.float_reset_list])
        index_cache = np.where((float_x_1[0] < float_reset_raw) & (float_reset_raw <= float_x_1[1]))
        index_cache = np.insert(index_cache[0], 0, index_cache[0][0]-1)
        float_reset = float_reset_raw[index_cache]
        cur_key_date = self._date + datetime.timedelta(days=int(float_reset[float_reset < 0][-1]))
        # 由时间差反推日期，用以查询数据库
        float_reset_date = np.array([self._date + datetime.timedelta(days=int(i)) for i in float_reset])
        # 根据日期查询当日浮动利率，由于在前面已经填充了空值，因此观测日前必有数据
        # 观测日之后的报价利率数据则需要预测，这里直接用过去来代替未来。
        float_reset_int = [self.database.loc[self.database['DATE'] == i, '5Y'].to_numpy()[0]
                           if i <= self._date else self.database.loc[self.database['DATE'] == cur_key_date, '5Y'].to_numpy()[0]
                           for i in float_reset_date]
        prev_pay_day, next_pay_day = float_x_1[0], float_x_1[1]
        second_reset_day, second_last_reset_day = float_reset[1], float_reset[-2]
        float_coupon_days = np.array([float_reset[i+1] - float_reset[i] for i in range(1, len(float_reset)-2)])
        float_coupon_days = np.insert(float_coupon_days, 0, second_reset_day - prev_pay_day)
        float_coupon_days = np.insert(float_coupon_days, -1, next_pay_day - second_last_reset_day)
        # insert不能用于连续在同一位置插入数据（如-1）（原因未知）
        float_coupon_days = np.append(float_coupon_days, 0)
        float_reset_df = pd.DataFrame({'float_reset_date': float_reset_date,
                                       'float_reset': float_reset,
                                       'float_coupon_days': float_coupon_days,
                                       'float_reset_int': float_reset_int
                                       })
        # print(float_reset_df)
        # Remaining, Interest
        return fix_df, float_df, fix_x_1, float_x_1, float_reset_df

    @time_decorator
    def _value(self):
        # 计算value_fix, value_float, value_swap
        # 这里以一般复利（而不是连续复利）计息和折现
        principle = self.deal['NominalPrincipal']
        paycycle = self.deal['FixPayCycle']
        coupon = principle * self.deal['FixRate'] * 0.01
        daydelta_list = [self.fix_x_1[i+1] - self.fix_x_1[i] for i in range(len(self.fix_x_1)-1)]
        # print(self.fix_df)
        # TODO:折现方式有待斟酌
        pv_factor = [1/(1+self.fix_df.loc[i, 'fix_y']*0.01*self.fix_df.loc[i, 'fix_x']/365) for i in range(self.fix_df.index.size)]
        # print(pv_factor)
        pv_fix_coupon = np.array([(coupon * daydelta_list[i]/365) * pv_factor[i] for i in range(self.fix_df.index.size)])
        value_fix = pv_fix_coupon.sum() + principle * pv_factor[-1]
        float_coupon_cache = np.array([1+self.float_reset_df.loc[i, 'float_reset_int']*0.01*self.float_reset_df.loc[i, 'float_coupon_days']/365
                                       for i in range(self.float_reset_df.index.size - 1)])
        float_coupon = principle * (float_coupon_cache.prod() - 1)
        value_float = (principle + float_coupon) * pv_factor[0]
        value = value_fix - value_float
        print('value_fix = ', value_fix)
        print('value_float = ', value_float)
        if self.deal['PayType'] == 'Float':
            print('value = ', value)
            return value
        elif self.deal['PayType'] == 'Fix':
            print('value = ', -value)
            return -value
        else:
            print('error')
            return None


x = Swap()
