#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   predictor.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/10/27 10:56 上午   Ferdinand      1.0  
@Desciption
----------------
面向对象
----------------     
'''
import re
import datetime
import numpy as np

from tool.dataProcess import extract_timeFeatures, series_to_supervised

from math import sqrt

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from predict.dataLoader import lstmModelDataLoader, lstmPredicterDataLoader
import sklearn
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from keras.models import load_model
from predict.lstmTrainer import lstmTrainer
from dateutil.relativedelta import relativedelta
import pandas as pd
class lstmPredicter(lstmTrainer):

    def __init__(self,paramDict={}):
        self.dataLoader=paramDict['dataLoader']
        self.modelPath=paramDict['modelPath']
        self.setModel()
        self.dataNeedToPredict=self.dataLoader.getDataNeedToPredict()
        self.past=self.dataLoader.past
        self.future=self.dataLoader.future
        self.predictFeature=self.dataLoader.predictFeature

    def setModel(self,modelPath='../data/modelFile/test_0.0419.h5'):
        self.modelPath=modelPath
        self.model = load_model(modelPath)

    def predict(self):
        self.predictData = self.model.predict(self.dataNeedToPredict)
        self.predictData=self.dataLoader.minMax.inverse_transform(self.predictData)




    def getPredictData(self):

        columnName=self.dataLoader.dict['日期']
        lastDate=self.dataLoader.originalData.loc[:,columnName].values[-1]
        lastDate=str(lastDate)
        year=int(lastDate[:4])
        month=int(lastDate[4:])
        futureTimes = []
        for i in range(self.future):
            futureTime = (datetime.date(year, month, 1) + relativedelta(months=i+1)).strftime(
                "%Y%m")
            futureTimes.append(futureTime)
        futureTimes = np.array(futureTimes).reshape(-1,1)
        data=np.concatenate((futureTimes,self.predictData.reshape(-1,1)),axis=1)
        df=pd.DataFrame(data,columns=['日期','预测%s'%self.dataLoader.predictColumnName])
        df.to_excel('../data/predictFile/预测%s_%s_%s->%s.xlsx'%(self.dataLoader.queryName,self.dataLoader.predictColumnName,
                                                              futureTimes[0][0],futureTimes[-1][0]))
        return data


        # originalData = self.__originalData
        # time = originalData[-1][0]
        # array = re.split("-| |:", time)
        # year = int(array[0])
        # month = int(array[1])
        # day = int(array[2])
        # hour = int(array[3])
        # futureTimes = []
        # for i in range(self.__future):
        #     futureTime = (datetime.datetime(year, month, day, hour, 00, 00) + datetime.timedelta(hours=i + 1)).strftime(
        #         "%Y-%m-%d %H:%M:%S")
        #     futureTimes.append([futureTime])
        # futureTimes = np.array(futureTimes)
        # self.__predictData = np.concatenate((futureTimes, data), axis=1)
        # return self.__predictData

def predict():
    dl_paramDict={}
    dl_paramDict['minMaxFilePath']='../data/modelFile/minMax/minMax_武汉市_快递业务总量.model'
    dl_paramDict['queryName']='武汉市'
    dl_paramDict['predictColumnName']='快递业务总量'
    dl_paramDict['context']='../'
    dl_paramDict['name']='no.1'
    dl_paramDict['filePath']='data/historyData/zb_merge_kdywl_month.xlsx'
    dl_paramDict['past']=5*12
    dl_paramDict['future']=2*12
    dl=lstmPredicterDataLoader(dl_paramDict)


    p_paramDict={}
    p_paramDict['modelPath']='../data/modelFile/武汉市_快递业务总量_test_0.0345.h5'
    p_paramDict['dataLoader']=dl
    predictor=lstmPredicter(p_paramDict)
    predictor.predict()
    predictor.getPredictData()
if __name__ == '__main__':
    predict()