#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Created by Klaus Lee on 2021/7/15
-------------------------------------------------
"""

from functools import wraps
import os
import numpy as np
import pandas as pd
import datetime
import calendar
from scipy import optimize
import scipy.interpolate as si
import matplotlib.pyplot as plt
import seaborn as sns


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

def read_csv_v1(path, file_name, dedicated_filter=0, **kwargs):
    read_conds = {
        'encoding': 'utf-8',
        'engine': 'python',
        'index_col': None,
        'skiprows': None,
        'na_values': np.nan,
        **kwargs
    }
    data = pd.read_csv(os.path.join(path, file_name), **read_conds)
    # 去重
    data.drop_duplicates(inplace=True)
    # 专门化清洗
    if dedicated_filter == 1:
        # 日期格式化
        data['DATE'] = pd.to_datetime(data['DATE'])
        # 按日期由近到远排序
        data.sort_values(by='DATE', ascending=False, inplace=True)
        # 重置index
        data.reset_index(inplace=True, drop=True)
    elif dedicated_filter == 2:
        # 日期格式化
        data['TransDate'] = pd.to_datetime(data['TransDate'])
        data['StartDate'] = pd.to_datetime(data['StartDate'])
        data['FirstStartDate'] = pd.to_datetime(data['FirstStartDate'])
        data['EndDate'] = pd.to_datetime(data['EndDate'])
        data['FixFirstPayDay'] = pd.to_datetime(data['FixFirstPayDay'])
        data['FloatFirstPayDay'] = pd.to_datetime(data['FloatFirstPayDay'])
        data['FloatFirstSetDay'] = pd.to_datetime(data['FloatFirstSetDay'])

    else:
        pass

    return data


data_fr007s = read_csv_v1(DATA_PATH, 'FR007S.csv', dedicated_filter=1)
data_frr = read_csv_v1(DATA_PATH, 'FRR.csv', dedicated_filter=1)
data_deal = read_csv_v1(DATA_PATH, 'IRS.csv', dedicated_filter=2, na_values=np.nan, skiprows=[1])


class IRS_FR007:
    def __init__(self, deal=data_deal.loc[0, :], _date=datetime.datetime(2021, 6, 30)):
        self._date = _date
        self.deal = deal.to_dict()
        self._time, self._type = '16:30', '均值'
        self.fr007s, self.frr, self.fr007s_today, self.frr_today = self.database()
        self.close_rate = self.close_rate_data()
        self.spot_rate = self.spot_rate_data()
        self.fix_payday, self.float_payday, self.float_reset = self.pay_day()
        self.value_float, self.value_float_data, self.value_reset_data = self.valuation_float()
        self.value_fix, self.value_fix_data = self.valuation_fix()

    def database(self):
        filter_fr007 = [i and j for i, j in zip(data_fr007s.loc[:, 'TIME'].eq(self._time),
                                                data_fr007s.loc[:, 'TYPE'].eq(self._type), )]
        fr007s_cache = data_fr007s.loc[filter_fr007]
        frr_cache = data_frr
        # 向前填充空日期，遍历合约期间全部日期，填充范围暂时设置为到观测日
        date_range = pd.date_range(self.deal['FloatFirstSetDay'], self._date)
        # 利率互换部分
        data_cache = pd.DataFrame
        for i in range(len(date_range)):
            if date_range[i] in fr007s_cache.loc[:, 'DATE'].to_numpy():
                data_cache = fr007s_cache.loc[fr007s_cache['DATE'] == date_range[i]]

            else:
                data_cache.loc[data_cache.index[0], 'DATE'] = date_range[i]
                fr007s_cache = pd.concat([fr007s_cache, data_cache])
        # 回购定盘利率部分
        data_cache = pd.DataFrame
        for i in range(len(date_range)):
            if date_range[i] in frr_cache.loc[:, 'DATE'].to_numpy():
                data_cache = frr_cache.loc[frr_cache['DATE'] == date_range[i]]

            else:
                data_cache.loc[data_cache.index[0], 'DATE'] = date_range[i]
                frr_cache = pd.concat([frr_cache, data_cache])
        # 按日期由近到远排序
        fr007s_cache.sort_values(by='DATE', ascending=False, inplace=True)
        frr_cache.sort_values(by='DATE', ascending=False, inplace=True)
        # 重置index
        fr007s_cache.reset_index(inplace=True, drop=True)
        frr_cache.reset_index(inplace=True, drop=True)
        # 提取估值当日数据用于计算曲线
        fr007s_today = fr007s_cache.loc[fr007s_cache['DATE'] == self._date]
        frr_today = frr_cache.loc[frr_cache['DATE'] == self._date]
        # 重置index
        fr007s_today.reset_index(inplace=True, drop=True)
        frr_today.reset_index(inplace=True, drop=True)
        return fr007s_cache, frr_cache, fr007s_today, frr_today

    def date_next(self, date_raw_0, date_delta_0):
        # 计算给定日期间隔的下一日
        delta_signal_0 = date_delta_0[-1]
        delta_num_0 = int(date_delta_0[:-1])
        # 先判断是否为0，若为0则直接return结束函数
        if delta_num_0 == 0:
            return date_raw_0

        def date_next_core(date_raw, date_delta):
            # 计算给定日期间隔的下一日
            delta_signal = date_delta[-1]
            delta_num = int(date_delta[:-1])
            next_year_raw = date_raw.year + delta_num
            next_month_raw = date_raw.month + delta_num
            next_day_raw = date_raw.day + delta_num
            # 只考虑存在性不考虑节假日调整
            if delta_signal == 'Y':
                if date_raw.month == 2 and date_raw.day == 29 and not calendar.isleap(next_year_raw):
                    date_next_raw = date_raw.replace(year=date_raw.year + delta_num, month=3, day=1)
                else:
                    date_next_raw = date_raw.replace(year=date_raw.year + delta_num)
            elif delta_signal == 'M':
                # 考虑到最多跨一年
                if delta_num >= 12:
                    print('Are you OK?')
                month_list_raw = range(date_raw.month, next_month_raw + 1, 1)
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
                date_next_raw = date_raw + datetime.timedelta(days=delta_num)
            else:
                print('Error: date_next()')
                date_next_raw = date_raw
            # TODO:经调整后的下一工作日
            # 下一工作日
            if date_next_raw.isoweekday() == 6:
                date_next_0 = date_next_raw + datetime.timedelta(days=2)
            elif date_next_raw.isoweekday() == 7:
                date_next_0 = date_next_raw + datetime.timedelta(days=1)
            else:
                date_next_0 = date_next_raw
            # 逾月则调整到上一工作日
                if date_next_0.month != date_next_raw.month:
                    if date_next_raw.isoweekday() == 6:
                        date_next_0 = date_next_raw - datetime.timedelta(days=1)
                    elif date_next_raw.isoweekday() == 7:
                        date_next_0 = date_next_raw - datetime.timedelta(days=2)
            # TODO:检验节假日
            return date_next_0

        # 对超过12M进行调整，为减少出错将调整部分放在外层，原本函数作为核心放在内层
        if delta_signal_0 == 'M' and delta_num_0 >= 12:
            part_1, part_2 = '{:}Y'.format(int(delta_num_0/12)), '{:}M'.format(int(delta_num_0 % 12))
            # print(date_delta_0, part_1, part_2)
            date_next = date_next_core(date_next_core(date_raw_0, part_1), part_2)
        else:
            date_next = date_next_core(date_raw_0, date_delta_0)

        return date_next

    def convert_y_m(self, delta):
        delta_signal = delta[-1]
        delta_num = int(delta[:-1])
        if delta_signal == 'Y':
            delta_new = '{:}M'.format(delta_num * 12)
        elif delta_signal == 'M' and delta_num >= 12:
            delta_new = '{:}Y'.format(delta_num / 12.0)
        else:
            delta_new = delta
        return delta_new

    def close_rate_data(self):
        # 获取标准期限列表
        life_array_cache = self.fr007s.columns.to_numpy()
        # np.where返回的值的结构很怪，print一下就明白为何要[0][0]了
        life_array_swap = life_array_cache[np.where(life_array_cache == '1M')[0][0]:]
        life_array_frr = ['1D', '7D', '14D']
        life_array = np.insert(life_array_swap, 0, life_array_frr)
        # TODO:这里默认为T+1机制，即非隔夜都为交易日第二日起息，隔夜为当日，以后添加其他机制
        life_n = len(life_array)
        start_date_array = np.array([self._date if i == 0 else self._date+datetime.timedelta(days=1)
                                     for i in range(life_n)])
        end_date_array = np.array([self.date_next(start_date_array[i], life_array[i]) for i in range(life_n)])
        days_array = np.array([(end_date_array[i] - start_date_array[i]).days for i in range(life_n)])

        def data_source(life):
            if life in ['1D', '7D']:
                source = 'FR00{:}'.format(life[0])
            elif life == '14D':
                source = 'FR0{:}'.format(life[:-1])
            else:
                source = 'FR007S_{:}'.format(life)
            return source
        data_source_array = np.array([data_source(life_array[i]) for i in range(life_n)])

        def close_rate_read(life):
            if life in ['1D', '7D']:
                close_rate = self.frr_today.loc[0, 'FR00{:}(%)'.format(life[0])]
            elif life == '14D':
                close_rate = self.frr_today.loc[0, 'FR0{:}(%)'.format(life[:-1])]
            else:
                close_rate = self.fr007s_today.loc[0, life]
            return close_rate
        close_rate_array = np.array([close_rate_read(life_array[i]) * 0.01 for i in range(life_n)])
        df_close_rate = pd.DataFrame({'start_date': start_date_array,
                                      'end_date': end_date_array,
                                      'life': life_array,
                                      'days': days_array,
                                      'source': data_source_array,
                                      'close_rate': close_rate_array,
                                      })
        return df_close_rate

    def days_count(self, basis, start_date, end_date):
        # TODO:完善其他日计数基准
        if basis == 'ACT/365':
            days_int = (end_date - start_date).days/365
        else:
            days_int = (end_date - start_date).days/365
            print('else?????')
            pass
        return days_int

    def spot_rate_data(self):
        # 基于收盘数据self.close_rate，运用bootstrap和线性插值计算
        # 线性插值时假定标准期限之间远期利率不变
        # df = e^(-sc*t)
        # print(self.close_rate)
        '''
        # 注意numpy.datetime64要先转换为datetime.datetime才能用传统方法计算，而两种转换方式中第一种给出的结果才是想要的
        start_date_array = self.close_rate.loc[:, 'start_date'].to_numpy(datetime.datetime)
        # start_date_array = self.close_rate.loc[:, 'start_date'].to_numpy().astype(datetime.datetime)
        end_date_array = self.close_rate.loc[:, 'end_date'].to_numpy(datetime.datetime)
        close_rate_array = self.close_rate.loc[:, 'close_rate'].to_numpy()
        '''
        life_array = self.close_rate.loc[:, 'life'].to_numpy()
        # 根据是否大于一个固定利息支付周期来选用计算方式
        index_cycle = np.where(life_array == self.deal['FixPayCycle'])[0][0]
        # 按照付息周期填充曲线x轴，即Life
        # 这里默认最小间隔为月(M)，为周和天暂不考虑
        index_1y = np.where(life_array == '1Y')[0][0]
        # 单位统一为月
        life_new_cache_array = np.array([self.convert_y_m(life_array[i]) for i in range(index_1y, len(life_array), 1)])
        # 按最小间隔填充次数
        n_filling = int((int(life_new_cache_array[-1][:-1])-int(life_new_cache_array[0][:-1]))/int(self.deal['FixPayCycle'][:-1]))
        # 数值填充
        life_new_cache = np.array([int(life_new_cache_array[0][:-1])+i*int(self.deal['FixPayCycle'][:-1])
                                   for i in range(n_filling+1)])
        # 带符号填充
        life_array_new_0 = np.array(['{:}M'.format(life_new_cache[i]) for i in range(len(life_new_cache))])
        life_array_new = np.concatenate([life_array[:index_1y], life_array_new_0])
        life_n_new = len(life_array_new)
        start_date_array_new = np.array([self._date if i == 0 else self._date+datetime.timedelta(days=1)
                                         for i in range(life_n_new)])
        end_date_array_new = np.array([self.date_next(start_date_array_new[i], life_array_new[i]) for i in range(life_n_new)])
        days_array_new = np.array([(end_date_array_new[i] - start_date_array_new[i]).days for i in range(life_n_new)])
        # 这一列涉及到贴现因子和即期利率的计算，小于一个付息周期的项必须设置为0(1D,7D,14D,1M)
        days_delta_array = np.array([0 if i <= index_cycle else int(days_array_new[i]-days_array_new[i-1])
                                     for i in range(life_n_new)])
        days_delta_array[index_cycle] = days_array_new[index_cycle]
        df_spot_rate_0 = pd.DataFrame({'start_date': start_date_array_new,
                                       'end_date': end_date_array_new,
                                       'life': life_array_new,
                                       'days': days_array_new,
                                       })
        df_close_rate_0 = self.close_rate.copy(deep=True)
        life_array_month = np.concatenate([life_array[:index_1y], life_new_cache_array])
        df_close_rate_0['life'] = life_array_month
        df_spot_rate = pd.concat([df_close_rate_0, df_spot_rate_0])
        # 去重排序清洗
        # 去重时好像自动优先去除NAN值,因此不用特意排序
        df_spot_rate.drop_duplicates(['days'], inplace=True)
        df_spot_rate.sort_values(by=['days', 'close_rate'], ascending=True, inplace=True)
        df_spot_rate.reset_index(inplace=True, drop=True)
        df_spot_rate['days_delta'] = days_delta_array
        # print(df_spot_rate)
        source_array_new = df_spot_rate.loc[:, 'source'].to_numpy()
        close_rate_array_new = df_spot_rate.loc[:, 'close_rate'].to_numpy()
        # 统计从最后一个周期契合的期限起(即1Y)，有固定利息数据的个数
        over_1y = life_array_month[index_1y:]
        '''['12M' '24M' '36M' '48M' '60M' '84M' '120M']'''
        over_1y_n = len(over_1y)
        # 标记这些数据在spot_rate大表中具体的index
        index_over_1y = np.array([np.where(life_array_new == over_1y[i])[0][0] for i in range(over_1y_n)])

        fill_stage = 0
        fill_stage_list = []
        for i in range(life_n_new):
            if i > index_over_1y[fill_stage]:
                fill_stage += 1
            fill_stage_list.append(fill_stage)
        fill_stage_array = np.array(fill_stage_list)
        df_spot_rate['fill_stage'] = fill_stage_array
        # 需要制定类型，不然以0生成会变被锁int型
        df_array = np.array([0 for i in range(life_n_new)], dtype=float)
        sc_array = np.array([0 for i in range(life_n_new)], dtype=float)

        # 小于等于一个付息周期的(1D,7D,14D,1M,3M)和在逐个递增的付息周期内都有数据的(6M,9M,1Y)
        for i in range(index_1y+1):
            # 小于等于一个付息周期的
            # 由于是季付，因此小于等于一个付息周期的就按照一次付息贴现
            if i <= index_cycle:
                df_cache = 1 / (1 + close_rate_array_new[i] *
                                self.days_count(self.deal['FixIntBasis'], start_date_array_new[i], end_date_array_new[i]))
                df_array[i] = df_cache
                sc_cache = - np.log(df_cache)/(days_array_new[i]/365)
                sc_array[i] = sc_cache
            elif index_cycle < i <= index_1y:
                # 有收盘价的且变动周期为逐级递增的付息周期(6M,9M,1Y)
                df_cache = (1-close_rate_array_new[i] * np.sum(df_array[:i]*days_delta_array[:i]/365))/\
                           (1+close_rate_array_new[i] * days_delta_array[i]/365)
                df_array[i] = df_cache
                sc_cache = - np.log(df_cache)/(days_array_new[i]/365)
                sc_array[i] = sc_cache
            else:
                # 逻辑发生变化，不再适合在大循环中运行，因此这部分单独拿出
                pass
        # 填充部分，由于期限很长，因此最好不要使用分段定常远期利率的方法，而是使用即期利率分段线性的假设
        # 统计有固定利息数据的个数，并以此分段over_1y_n
        # 在每段内，分割出每个具体付息周期index_over_1y
        # 根据fill_stage是否发生变化来判断是否跨期
        for j in range(1, over_1y_n):
            #
            # 直接单变量求解期末即期利率
            def func_sc(sc_final):
                # 此方法失败，似乎不能带着未知数进行列表操作，否则会直接运算出结果而不是成为方程
                # TODO:对比两种方法在数值上确实有差异，因此仍采用土法，保留通用法以备技术更新
                '''
                sc_cache_array = np.array([0 for m in range(index_over_1y[j-1]+1, index_over_1y[j])])
                df_cache_array = np.array([0 for m in range(index_over_1y[j-1]+1, index_over_1y[j])])
                for i, k in zip(range(index_over_1y[j-1]+1, index_over_1y[j]), range(index_over_1y[j]-index_over_1y[j-1]-1)):
                    sc_cache_array[k] = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                                        (days_array_new[i] - days_array_new[index_over_1y[j-1]]) /\
                                        (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    df_cache_array[k] = np.exp(-sc_cache_array[k]*days_array_new[i]/365)
                df_final = np.exp(-sc_final*days_array_new[index_over_1y[j]]/365)
                df_cache_array_extend = np.concatenate([df_array[:index_over_1y[j-1]+1], df_cache_array])
                return np.array(df_final*(1+close_rate_array_new[index_over_1y[j]]*days_delta_array[index_over_1y[j]]/365) +
                                close_rate_array_new[index_over_1y[j]] *
                                np.sum(df_cache_array_extend*days_delta_array[:index_over_1y[j]]/365)
                                - 1)
                '''
                # TODO：功力不够写不出通用形式，因此这里具体问题具体分析，直接枚举
                if index_over_1y[j]-index_over_1y[j-1]-1 == 3:
                    sc_1 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+1] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_2 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+2] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_3 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+3] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    df_final = np.exp(-sc_final*days_array_new[index_over_1y[j]]/365)
                    # 不能进行列表操作
                    return np.array(df_final*(1+close_rate_array_new[index_over_1y[j]]*days_delta_array[index_over_1y[j]]/365) +
                                    close_rate_array_new[index_over_1y[j]] *
                                    (np.sum(df_array[:index_over_1y[j-1]+1]*days_delta_array[:index_over_1y[j-1]+1]/365) +
                                    np.exp(-sc_1*days_array_new[index_over_1y[j-1]+1]/365)*days_delta_array[index_over_1y[j-1]+1]/365 +
                                    np.exp(-sc_2*days_array_new[index_over_1y[j-1]+2]/365)*days_delta_array[index_over_1y[j-1]+2]/365 +
                                    np.exp(-sc_3*days_array_new[index_over_1y[j-1]+3]/365)*days_delta_array[index_over_1y[j-1]+3]/365
                                    ) - 1)
                elif index_over_1y[j]-index_over_1y[j-1]-1 == 7:
                    sc_1 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+1] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_2 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+2] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_3 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+3] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_4 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+4] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_5 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+5] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_6 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+6] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_7 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+7] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    df_1 = np.exp(-sc_1*days_array_new[index_over_1y[j-1]+1]/365)
                    df_2 = np.exp(-sc_2*days_array_new[index_over_1y[j-1]+2]/365)
                    df_3 = np.exp(-sc_3*days_array_new[index_over_1y[j-1]+3]/365)
                    df_4 = np.exp(-sc_4*days_array_new[index_over_1y[j-1]+4]/365)
                    df_5 = np.exp(-sc_5*days_array_new[index_over_1y[j-1]+5]/365)
                    df_6 = np.exp(-sc_6*days_array_new[index_over_1y[j-1]+6]/365)
                    df_7 = np.exp(-sc_7*days_array_new[index_over_1y[j-1]+7]/365)
                    df_final = np.exp(-sc_final*days_array_new[index_over_1y[j]]/365)
                    # 不能进行列表操作
                    return np.array(df_final*(1+close_rate_array_new[index_over_1y[j]]*days_delta_array[index_over_1y[j]]/365) +
                                    close_rate_array_new[index_over_1y[j]] *
                                    (np.sum(df_array[:index_over_1y[j-1]+1]*days_delta_array[:index_over_1y[j-1]+1]/365) +
                                    df_1*days_delta_array[index_over_1y[j-1]+1]/365 +
                                    df_2*days_delta_array[index_over_1y[j-1]+2]/365 +
                                    df_3*days_delta_array[index_over_1y[j-1]+3]/365 +
                                    df_4*days_delta_array[index_over_1y[j-1]+4]/365 +
                                    df_5*days_delta_array[index_over_1y[j-1]+5]/365 +
                                    df_6*days_delta_array[index_over_1y[j-1]+6]/365 +
                                    df_7*days_delta_array[index_over_1y[j-1]+7]/365
                                    ) - 1)
                elif index_over_1y[j]-index_over_1y[j-1]-1 == 11:
                    sc_1 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+1] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_2 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+2] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_3 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+3] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_4 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+4] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_5 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+5] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_6 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+6] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_7 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+7] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_8 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+8] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_9 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+9] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_10 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+10] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    sc_11 = sc_array[index_over_1y[j-1]]+(sc_final-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[index_over_1y[j-1]+11] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    df_1 = np.exp(-sc_1*days_array_new[index_over_1y[j-1]+1]/365)
                    df_2 = np.exp(-sc_2*days_array_new[index_over_1y[j-1]+2]/365)
                    df_3 = np.exp(-sc_3*days_array_new[index_over_1y[j-1]+3]/365)
                    df_4 = np.exp(-sc_4*days_array_new[index_over_1y[j-1]+4]/365)
                    df_5 = np.exp(-sc_5*days_array_new[index_over_1y[j-1]+5]/365)
                    df_6 = np.exp(-sc_6*days_array_new[index_over_1y[j-1]+6]/365)
                    df_7 = np.exp(-sc_7*days_array_new[index_over_1y[j-1]+7]/365)
                    df_8 = np.exp(-sc_8*days_array_new[index_over_1y[j-1]+8]/365)
                    df_9 = np.exp(-sc_9*days_array_new[index_over_1y[j-1]+9]/365)
                    df_10 = np.exp(-sc_10*days_array_new[index_over_1y[j-1]+10]/365)
                    df_11 = np.exp(-sc_11*days_array_new[index_over_1y[j-1]+11]/365)
                    df_final = np.exp(-sc_final*days_array_new[index_over_1y[j]]/365)
                    # 不能进行列表操作
                    return np.array(df_final*(1+close_rate_array_new[index_over_1y[j]]*days_delta_array[index_over_1y[j]]/365) +
                                    close_rate_array_new[index_over_1y[j]] *
                                    (np.sum(df_array[:index_over_1y[j-1]+1]*days_delta_array[:index_over_1y[j-1]+1]/365) +
                                    df_1*days_delta_array[index_over_1y[j-1]+1]/365 +
                                    df_2*days_delta_array[index_over_1y[j-1]+2]/365 +
                                    df_3*days_delta_array[index_over_1y[j-1]+3]/365 +
                                    df_4*days_delta_array[index_over_1y[j-1]+4]/365 +
                                    df_5*days_delta_array[index_over_1y[j-1]+5]/365 +
                                    df_6*days_delta_array[index_over_1y[j-1]+6]/365 +
                                    df_7*days_delta_array[index_over_1y[j-1]+7]/365 +
                                    df_8*days_delta_array[index_over_1y[j-1]+8]/365 +
                                    df_9*days_delta_array[index_over_1y[j-1]+9]/365 +
                                    df_10*days_delta_array[index_over_1y[j-1]+10]/365 +
                                    df_11*days_delta_array[index_over_1y[j-1]+11]/365
                                    ) - 1)
                else:
                    print('I can\'t breathe!')
                    pass
            sc_solve = optimize.root(func_sc, np.array(sc_array[index_over_1y[j-1]]), tol=1e-10)
            sc = sc_solve.get('x')[0]
            # print(sc)
            # 每计算出一个端点sc，就要填充中间的部分，否则在后续计算中无法得到正确结论
            # 由于本质是单变量方程求解，因此解出端点值后，范围内数值迎刃而解。
            # 填充部分就不需要像解方程那样单独列示了，直接列表操作(且循环范围不同于解方程时，不可生搬硬套)
            for i in range(index_over_1y[j-1]+1, index_over_1y[j]+1):
                sc_cache = sc_array[index_over_1y[j-1]]+(sc-sc_array[index_over_1y[j-1]]) * \
                           (days_array_new[i] - days_array_new[index_over_1y[j-1]]) /\
                           (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                sc_array[i] = sc_cache
                df_cache = np.exp(-sc_cache*days_array_new[i]/365)
                df_array[i] = df_cache
            # 插值部分，直接完成远期（固定利率）和数据源两列，这里采用分段定常远期利率插值的假设
            # 在这个假设下，贴现系数是关于期限算子的指数函数，两个标准期限之间的任意贴现系数的对数与期限算子呈线性关系（而不是与期限呈线性关系）
            # 已完成数学推导，见README.md 提示见国债期货交易实务p67
            # 这个循环范围与解方程时的相同，因为是填充，面对的未知数范围是相同的
            for i in range(index_over_1y[j-1]+1, index_over_1y[j]):
                # numpy里 np.nan != np.nan，要用专用函数去判断
                # 空值才填充，非空值不填充
                if np.isnan(close_rate_array_new[i]):
                    source_array_new[i] = 'Forward'
                    # 但交易中心不是这么算的，*不！*
                    '''
                    close_rate_array_new[i] = (sc_array[index_over_1y[j]]*days_array_new[index_over_1y[j]] -
                                               sc_array[index_over_1y[j-1]]*days_array_new[index_over_1y[j-1]])/\
                                              (days_array_new[index_over_1y[j]] - days_array_new[index_over_1y[j-1]])
                    '''
                    # 每小段有各自的远期利率，按照计算出的贴现因子去算，而不是按照真实数据去算一大段。
                    close_rate_array_new[i] = (df_array[i-1]/df_array[i] - 1)/(days_delta_array[i]/365)
                else:
                    print('I cannot breath!')
        df_spot_rate['disc_factor'] = df_array
        df_spot_rate['spot_rate'] = sc_array
        df_spot_rate['source'] = source_array_new
        df_spot_rate['close_rate'] = close_rate_array_new
        order = ['start_date', 'end_date', 'life', 'days', 'days_delta', 'source', 'fill_stage',
                 'close_rate', 'disc_factor', 'spot_rate']
        df_spot_rate = df_spot_rate[order]
        return df_spot_rate

    def spot_curve(self, days):
        # 生成即期利率曲线，并提供函数对应关系
        x = self.spot_rate.loc[:, 'days'].to_numpy()
        y = self.spot_rate.loc[:, 'spot_rate'].to_numpy()
        # TODO:默认为三次样条插值，后续添加其他方法。
        curve = si.CubicSpline(x, y)
        return curve(days)

    def df_curve(self, days):
        df = np.exp(-self.spot_curve(days)*days/365)
        return df

    def pay_day(self):
        # 输出固定端和浮动端支付日列表
        # 先分别计算固定端和浮动端的付息次数
        delta_date = self.deal['EndDate'] - self.deal['StartDate']
        # 计算支付次数

        def set_timedelta(cycle):
            if cycle[-1] == 'Y':
                time_delta = int(cycle[:-1]) * datetime.timedelta(days=365)
            elif cycle[-1] == 'M':
                time_delta = int(cycle[:-1]) * datetime.timedelta(days=30)
            elif cycle[-1] == 'D':
                time_delta = int(cycle[:-1]) * datetime.timedelta(days=1)
            else:
                print('error')
                time_delta = int(cycle[:-1]) * datetime.timedelta(days=1)
            return time_delta
        fix_timedelta = set_timedelta(self.deal['FixPayCycle'])
        float_timedelta = set_timedelta(self.deal['FloatPayCycle'])
        float_reset_timedelta = set_timedelta(self.deal['FloatResetCycle'])
        fix_n = np.trunc((self.deal['EndDate'] - self.deal['StartDate']) / fix_timedelta).astype(int)
        float_n = np.trunc((self.deal['EndDate'] - self.deal['StartDate']) / float_timedelta).astype(int)
        float_reset_n = np.trunc((self.deal['EndDate'] - self.deal['FloatFirstSetDay'])
                                 / float_reset_timedelta).astype(int)
        # 计算起息日(0)和各个支付日(1:n)

        def times_cycle(cycle, n):
            if cycle[-1] == 'D' or 'M' or 'Y':
                delta_num = int(cycle[:-1])
                cycle_new = '{0}{1}'.format(delta_num*n, cycle[-1])
            else:
                print('尚不支持年月日之外的倍数化')
                cycle_new = cycle
            return cycle_new

        fix_payday_array = np.array([self.date_next(self.deal['StartDate'], times_cycle(self.deal['FixPayCycle'], i)) for i in range(fix_n+1)])
        float_payday_array = np.array([self.date_next(self.deal['StartDate'], times_cycle(self.deal['FloatPayCycle'], i)) for i in range(float_n+1)])
        # 浮动方季度支付但七日计息，存在复利计息的问题
        # 计算浮动利率报价日(每七日)和对应利率
        float_reset_array = np.array([self.date_next(self.deal['FloatFirstSetDay'], times_cycle(self.deal['FloatResetCycle'], i)) for i in range(float_reset_n)])
        # 细节修正
        fix_payday_array[1], fix_payday_array[-1] = self.deal['FixFirstPayDay'], self.deal['EndDate']
        float_payday_array[1], float_payday_array[-1] = self.deal['FloatFirstPayDay'], self.deal['EndDate']
        # TODO:工作日调整上仍有细微偏差，需要检查问题并调整
        # TODO:格式不统一，timedelta计算得到的类型为timedelta，年月调整得到的是datetime，
        return fix_payday_array, float_payday_array, float_reset_array

    def f_rate(self, start_day, end_day):
        return self.df_curve(start_day)/self.df_curve(end_day) - 1

    def f_days(self, start_day, end_day):
        return (end_day - start_day)/365

    def valuation_float(self):
        # 浮动端价值
        # 计算当前付息周期浮动端的具体情况
        # TODO:支付日是否重置计息周期，即上一个不完整的计息周期在付息后是否存在？
        # 这里假设存在，即每个计息周期是完整的，付息周期最开始的重置利息需要到上一个付息周期寻找
        float_days_pay_array = np.array([(i-self._date).days for i in self.float_payday])
        float_days_reset_array = np.array([(i-self._date).days for i in self.float_reset])
        n_cycle = np.where(float_days_pay_array > 0)[0][0]
        # 获取包含上一个周期端点在内的剩余所有周期的天数和反推出的对应日期
        float_days_pay_array_new = float_days_pay_array[n_cycle-1:]
        # 重置日定息定的是第二天开始的利息(T+1)，所以这里在区间端点上的选择有所区别。
        # 截取当前付息周期的定息天数
        next_pay_day = float_days_pay_array_new[np.where(float_days_pay_array_new > 0)[0][0]]
        # 由于最初的定息日早于最早的气息日，因此在>的基础上必须-1，若二者重合，则在-1的基础上不需要考虑>=
        # 定息日定的是从第二天(T+1)开始的周期的利息，若定息日与支付日重合，则那个定息日不影响上一周期，因此用<而不是<=
        float_days_reset_array_new = float_days_reset_array[
                                     np.where(float_days_reset_array > float_days_pay_array_new[0])[0][0]-1:
                                     np.where(float_days_reset_array < next_pay_day)[0][-1]+1]
        reset_n = len(float_days_reset_array_new)
        float_reset_date_array = np.array([self._date+datetime.timedelta(days=int(float_days_reset_array_new[i]))
                                           for i in range(reset_n)])
        # 分已知和未知，出于数据可得性的考虑，默认估值时已有当日数据
        n_known = len(float_days_reset_array_new[np.where(float_days_reset_array_new <= 0)])
        n_unknown = reset_n - n_known
        # 获取本付息周期已知的浮动利息
        float_rate_array_known = np.array([self.frr.loc[self.frr['DATE'] == float_reset_date_array[i], 'FR007(%)'].to_numpy()[0]
                                           * 0.01 for i in range(n_known)])
        # 计算远期利率以此代表本周期未知的浮动利率
        # 下一定息日
        next_reset_date = float_reset_date_array[np.where(float_days_reset_array_new > 0)][0]
        day_0 = (next_reset_date - self._date).days
        # 下一支付日
        next_pay_date = self.float_payday[np.where(float_days_pay_array > 0)][0]
        day_1 = (next_pay_date - self._date).days
        float_rate_array_unknown = self.f_rate(day_0, day_1)/self.f_days(day_0, day_1)
        # 计算每期的时间，需要考虑多种情况下的端点取值和函数构造方法
        # 思路：先计算每期绝对数值，再去区分已知和未知进行二次切片
        float_days_array = np.array([float_days_reset_array_new[i] - float_days_reset_array_new[i-1] for i in range(1, reset_n)])
        float_days_array[0] = float_days_reset_array_new[1] - float_days_pay_array_new[0]
        float_days_array = np.append(float_days_array, float_days_pay_array_new[1] - float_days_reset_array_new[-1])
        float_days_known = float_days_array[:n_known]
        float_days_unknown = float_days_array[-n_unknown:]
        operator_known = 1 + float_days_known/365 * float_rate_array_known
        if np.sum(float_days_unknown) != day_1 - day_0:
            print('np.sum(float_days_unknown)!=day_1-day_0, CHECK YOUR FUNC!!!!!!')
        operator_unknown = 1 + float_rate_array_unknown * ((day_1 - day_0)/365)
        c_1 = (operator_known.prod() * operator_unknown - 1) * self.deal['Principal']
        # 计算本付息周期以后付息周期的浮动利息的现值
        forward_date = self.float_payday[n_cycle:]
        forward_days_array = np.array([(i - self._date).days for i in forward_date])
        '''
        forward_rate_array = np.array([(self.df_curve(forward_days_array[i-1])/self.df_curve(forward_days_array[i])
                                        - 1)/((forward_days_array[i] - forward_days_array[i-1])/365)
                                       for i in range(1, len(forward_days_array))])
        '''
        # 两种写法得到的数值相同
        forward_rate_array = np.array([self.f_rate(forward_days_array[i-1], forward_days_array[i]) /
                                       self.f_days(forward_days_array[i-1], forward_days_array[i])
                                       for i in range(1, len(forward_days_array))])  # n-1
        forward_df_array = np.apply_along_axis(func1d=self.df_curve, axis=0, arr=forward_days_array)  # n
        forward_delta_array = np.array([self.f_days(forward_days_array[i-1], forward_days_array[i])  # n-1
                                        for i in range(1, len(forward_days_array))])
        forward_df_cache = forward_df_array[1:]  # n-1
        # 最终得到浮动端价值现值
        value_float = c_1 * forward_df_array[0] + self.deal['Principal'] * np.sum(
            forward_rate_array * forward_delta_array * forward_df_cache)
        dfa_forward_rate = np.insert(forward_rate_array, 0, np.nan)
        cash_fv_cache = forward_rate_array * forward_delta_array * self.deal['Principal']
        cash_fv = np.insert(cash_fv_cache, 0, c_1)
        cash_pv_cache = cash_fv_cache * forward_df_cache
        cash_pv = np.insert(cash_pv_cache, 0, c_1 * forward_df_array[0])
        value_float_data = pd.DataFrame({'pay_date': forward_date,
                                         'days': forward_days_array,
                                         'dis_factor': forward_df_array,
                                         'forward_rate': dfa_forward_rate,
                                         'cash_fv': cash_fv,
                                         'cash_pv': cash_pv,
                                         })
        float_rate = np.array([float_rate_array_known[i] if i < n_known else np.nan for i in range(reset_n)])
        value_reset_data = pd.DataFrame({'reset_date': float_reset_date_array,
                                         'days': float_days_array,
                                         'float_rate': float_rate
                                         })
        return value_float, value_float_data, value_reset_data

    def valuation_fix(self):
        fix_days_pay_array = np.array([(i-self._date).days for i in self.fix_payday])
        n_cycle = np.where(fix_days_pay_array > 0)[0][0]
        fix_date = self.float_payday[n_cycle - 1:]
        fix_days_array = np.array([(i - self._date).days for i in fix_date])
        fix_days_cache = fix_days_array[1:]
        fix_delta_array = np.array([self.f_days(fix_days_array[i-1], fix_days_array[i]) for i in range(1, len(fix_days_array))])
        fix_df_array = np.apply_along_axis(self.df_curve, 0, fix_days_cache)
        value_fix = np.sum(fix_delta_array * fix_df_array) * self.deal['Principal'] * self.deal['FixRate'] * 0.01
        cash_fv = fix_delta_array * self.deal['Principal'] * self.deal['FixRate'] * 0.01
        cash_pv = cash_fv * fix_df_array
        value_fix_data = pd.DataFrame({'pay_date': fix_date[1:],
                                       'days': fix_days_cache,
                                       'dis_factor': fix_df_array,
                                       'fix_rate': [self.deal['FixRate'] * 0.01 for i in range(len(fix_df_array))],
                                       'cash_fv': cash_fv,
                                       'cash_pv': cash_pv,
                                       })
        return value_fix, value_fix_data

    @time_decorator
    def result(self):
        self.spot_rate.to_csv(os.path.join(RESULT_PATH, 'spot_curve.csv'), index=False)
        self.value_reset_data.to_csv(os.path.join(RESULT_PATH, 'reset.csv'), index=False)
        self.value_fix_data.to_csv(os.path.join(RESULT_PATH, 'fix.csv'), index=False)
        self.value_float_data.to_csv(os.path.join(RESULT_PATH, 'float.csv'), index=False)
        value_swap = self.value_fix - self.value_float
        values_data = pd.DataFrame({'合约编号': self.deal['ID'],
                                    '估值日': self._date.date(),
                                    '固定利率支付方': self.deal['FixPayer'],
                                    '固定端现值(万元,T+1)': self.value_fix,
                                    '买方合约现值(万元,T+1)': -value_swap,
                                    '浮动利率支付方': self.deal['FloatPayer'],
                                    '浮动端现值(万元,T+1)': self.value_float,
                                    '卖方合约现值(万元,T+1)': value_swap,
                                    }, index=[0])
        dv01_data = pd.DataFrame({})
        result_data = pd.concat([values_data, dv01_data])
        result_data.T.to_csv(os.path.join(RESULT_PATH, 'result.csv'), encoding='gb18030')
        return result_data

    @time_decorator
    def draw(self):
        # 即期利率曲线
        xs = np.arange(1, self.spot_rate.loc[:, 'days'].to_numpy()[-1], 1)
        scs = np.array([self.spot_curve(xs[i]) for i in range(len(xs))])
        plt.xlabel('Days')
        plt.ylabel('Spot Rate')
        plt.title('Spot Curve in {0}'.format(self._date.date()))
        sns.lineplot(x=xs, y=scs, estimator=None)
        plt.savefig(os.path.join(RESULT_PATH, 'SpotCurve_{0}.png'.format(self._date.date())))
        plt.close('all')
        # 贴现因子曲线
        dfs = np.array([self.df_curve(xs[i]) for i in range(len(xs))])
        plt.xlabel('Days')
        plt.ylabel('Discount Factor')
        plt.title('DF Curve in {0}'.format(self._date.date()))
        sns.lineplot(x=xs, y=dfs, estimator=None)
        plt.savefig(os.path.join(RESULT_PATH, 'DFCurve_{0}.png'.format(self._date.date())))
        plt.close('all')


a = IRS_FR007(_date=datetime.datetime(2021, 6, 30))
a.result()
a.draw()
