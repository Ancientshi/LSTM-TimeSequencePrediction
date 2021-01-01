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
from predict.dataLoader import lstmModelDataLoader
import sklearn
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from keras.models import load_model


class lstmTrainer:
    '预测器'



    def __init__(self, paramDict={}):
        '''

        :param modelName: 预测器的名字
        :param originalData: [时间，揽收，投递，实名，面单]
        :param statue: trainAndValidate/testAndEvaluate/predict
        '''

        self.modelName =paramDict['modelName']
        self.dataLoader=paramDict['dataLoader']
        self.trainData=self.dataLoader.getTrainData()
        self.validData=self.dataLoader.getValidData()
        self.past=self.dataLoader.past
        self.future=self.dataLoader.future
        self.predictFeature=self.dataLoader.predictFeature
    def drawHistory(self):
        # plot history
        history=self.history
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='valid_loss')
        plt.legend()
        plt.show()
    def setModel(self,modelPath='../data/modelFile/test_0.0419.h5'):
        self.modelPath=modelPath
        self.model = load_model(modelPath)
    def startTrain(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(self.trainData['train_x'].shape[1], self.trainData['train_x'].shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(self.predictFeature * self.future))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        # fit network
        self.history = model.fit(self.trainData['train_x'], self.trainData['train_y'], epochs=140, batch_size=12,
                            validation_data=(self.validData['valid_x'], self.validData['valid_y']), verbose=2,
                            shuffle=False)

        self.model = model
        # 保存模型
        print("Saving model to disk \n")
        valid_lastLoss=round(self.history.history['val_loss'][-1],4)
        model.save('../data/modelFile/%s_%s_%s_%s.h5' % (self.dataLoader.queryName,self.dataLoader.predictColumnName,self.modelName,valid_lastLoss))


    def predict(self):
        self.predictData = self.model.predict(self.validData['valid_x'])
        self.predictData=self.dataLoader.minMax.inverse_transform(self.predictData)


    def showPredict(self):
        valid_y=self.dataLoader.minMax.inverse_transform(self.validData['valid_y'])
        plt.figure()
        for i in range(self.predictData.shape[0]):
            plt.subplot(self.predictData.shape[0],1,i+1)
            plt.plot(range(self.future),self.predictData[i],label='预测数据')
            plt.plot(range(self.future), valid_y[i], label='实际数据')
            plt.xlabel('时间点')
            plt.ylabel('数据量')
        plt.legend()
        plt.show()

def train():
    dl_paramDict={}
    dl_paramDict['queryName']='中国'
    dl_paramDict['predictColumnName']='快递业务总量'
    dl_paramDict['context']='../'
    dl_paramDict['name']='no.1'
    dl_paramDict['filePath']='data/historyData/zb_merge_kdywl_month.xlsx'
    dl_paramDict['past']=5*12
    dl_paramDict['future']=2*12
    dl=lstmModelDataLoader(dl_paramDict)

    p_paramDict={}
    p_paramDict['modelName']='test'
    p_paramDict['dataLoader']=dl
    test_predictor=lstmTrainer(p_paramDict)
    #训练并预测
    test_predictor.startTrain()
    test_predictor.drawHistory()
    #test_predictor.predict()
    #test_predictor.showPredict()
def predict():
    dl_paramDict={}
    dl_paramDict['queryName']='中国'
    dl_paramDict['predictColumnName']='快递业务总量'
    dl_paramDict['context']='../'
    dl_paramDict['name']='no.1'
    dl_paramDict['filePath']='data/historyData/zb_merge_kdywl_month.xlsx'
    dl_paramDict['past']=5*12
    dl_paramDict['future']=2*12
    dl=lstmModelDataLoader(dl_paramDict)


    p_paramDict={}
    p_paramDict['modelName']='test'
    p_paramDict['dataLoader']=dl
    test_predictor=lstmTrainer(p_paramDict)
    #训练并预测
    test_predictor.setModel('../data/modelFile/中国_快递业务总量_test_0.0258.h5')
    test_predictor.predict()
    test_predictor.showPredict()
if __name__ == '__main__':
    train()
    #predict()