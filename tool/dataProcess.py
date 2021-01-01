#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataProcess.py
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version
------------      -------    --------
2020/10/27 11:21 上午   Ferdinand      1.0
@Desciption
----------------
数据处理的一些函数
----------------
'''
import datetime
from chinese_calendar import is_workday, is_holiday, get_holiday_detail
import re
import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.preprocessing.text import Tokenizer
def extract_areaFeatures(array=[]):
    areaNames = ','.join(array[:,1])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(areaNames)
    vocab = tokenizer.word_index  # 得到每个词的编号
    areaFeatures=[]
    for line in array:
        level=line[0]
        areaName=line[1]
        ofareaid=line[2]

        levelVector=[0]*4
        levelVector[level]=1

        areaNameVector=[0]*len(vocab)
        areaNameVector[vocab[areaName]]=1

        ofareaidVector=[0]*5
        try:
            ofareaidVector[ofareaid]=1
        except:
            ofareaidVector[4]=1
        areaFeatures.append(levelVector+areaNameVector+ofareaidVector)
    return areaFeatures





def extract_timeFeatures(times):
    '''
    :param times: 时间一维数组，
    :return: 数据类型：二维np数组；维度：nx(24+3)；每一行：[24小时独热编码，是否工作日，是否休息日，是否节日]
    '''
    timeFeatures = []
    for time in times:
        time=str(time)
        year=time[:4]
        month=time[4:]
        #2000年-2030年
        yearFeature=[0]*31
        yearFeature[int(year[2:])]=1
        #01月-12月
        monthFeature=[0]*13
        monthFeature[int(month)]=1
        timeFeatures.append(yearFeature+monthFeature)
    return timeFeatures


# 更改为 过去->未来这种数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
