#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   handleFile.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/28 1:33 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
from fancyimpute import KNN

import pandas as pd
df = pd.read_excel('../data/historyData/zb_merge_kdywl_month.xlsx')
print('使用knn填充缺失数据:')
data=df.iloc[:,8:].values
train_Data=KNN(10).fit_transform(data)
print('填充缺失数据完成:')
df.loc[:,8:]=train_Data
df.to_excel('../data/historyData/zb_merge_kdywl_month_KNN.xlsx')
