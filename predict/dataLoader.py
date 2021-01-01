#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataLoader.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/17 1:23 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
from sklearn.preprocessing import MinMaxScaler

from predict.Import import *
from tool.getDict import getDict
from tool.dataProcess import extract_timeFeatures, extract_areaFeatures, series_to_supervised
import joblib

class AbstractDataLoader(metaclass=abc.ABCMeta):

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def setTrainData(self):
        self.x_train = []
        self.y_train = []
        pass

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def setValidData(self):
        self.x_valid = []
        self.y_valid = []
        pass

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def setTestData(self):
        self.x_test = []
        self.y_test = []
        pass

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def getTrainData(self):
        '''

        :return: 返回训练数据
        '''
        return self.x_train, self.y_train

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def getValidData(self):
        '''

        :return: 返回验证数据
        '''
        return self.x_valid, self.y_valid

    # @abc.abstractmethod #定义抽象方法，无需实现功能
    def getTestData(self):
        '''

        :return: 返回测试数据
        '''
        return self.x_test, self.y_test



class lstmModelDataLoader(AbstractDataLoader):
    dataInfoDict = {}
    past = 5 * 12
    future = 2 * 12
    # 特征数
    # 31年，1+12个月，1个指标
    feature = 45
    # 需要预测的一个指标
    predictFeature = 1
    # 验证集/测试集所占比例
    test_size = 0.2

    def __init__(self, paramDict={}):
        self.validData = {}
        self.trainData = {}
        # columnName = self.dict['地区名']
        # queryName = '东莞市'
        self.queryName=paramDict['queryName']
        self.predictColumnName=paramDict['predictColumnName']
        self.context = paramDict['context']
        self.name = paramDict['name']
        self.filePath = paramDict['filePath']
        self.past = paramDict['past']
        self.future = paramDict['future']
        self.dict = getDict()
        # 首先要读取xlsx
        self.getOriginalData()
        # 接着处理数据，提取特征
        self.extractFeature()
        # 数据标准化
        self.normalizeData()
        # 转换成可以喂给模型的数据
        self.transferDataToSupervisedData()
        # 将数据分为训练，验证
        self.divideData()

    def extractFeature(self):
        # 首先提取时间特征
        self.timeFeature = extract_timeFeatures(self.originalData.loc[:, self.dict['日期']].values)
        #然后提取地域特征
        # self.areaFeature=extract_areaFeatures(self.df.loc[:,[self.dict['地区层级'],
        #                                                      self.dict['地区名'],
        #                                                      self.dict['所属地域id']]].values)
        # 然后提取数值特征
        self.numFeature = self.originalData.loc[:, self.dict['快递业务总量']].values

    def normalizeData(self):
        self.minMax = MinMaxScaler()
        self.numFeature = self.minMax.fit_transform(self.numFeature.reshape(-1,1))
        #需要保存模型
        joblib.dump(value=self.minMax, filename='../data/modelFile/minMax/minMax_%s_%s.model'%(self.queryName,self.predictColumnName))
        # origin_data = self.mm.inverse_transform(mm_data)

    def transferDataToSupervisedData(self):
        data = np.concatenate((self.numFeature,self.timeFeature), axis=1)
        reframed = series_to_supervised(data, self.past, self.future)
        self.supervisedData = reframed.values



    def divideData(self):
        #44,89,
        valid_y_num = []
        for i in range(self.past, self.past + self.future):
            for l in range(self.predictFeature):
                valid_y_num.append(self.feature * i + l)

        self.trainData['train_x'], self.validData['valid_x'], self.trainData['train_y'], self.validData[
            'valid_y'] = sklearn.model_selection.train_test_split(
            self.supervisedData[:, :self.feature * self.past],
            self.supervisedData[:, valid_y_num],
            test_size=self.test_size, random_state=0)
        # reshape input to be 3D [samples, timesteps, features]
        # [样本，时间步长，特征]
        self.trainData['train_x'] = self.trainData['train_x'].reshape(
            (self.trainData['train_x'].shape[0], self.past, self.feature))
        self.validData['valid_x'] = self.validData['valid_x'].reshape(
            (self.validData['valid_x'].shape[0], self.past, self.feature))

    def getTrainData(self):
        return self.trainData
    def getValidData(self):
        return self.validData

    def getOriginalData(self):
        filePath = self.context + self.filePath
        self.df = pd.read_excel(filePath)
        self.df.fillna(0, inplace=True)
        columnName = self.dict['地区名']
        queryName = self.queryName
        self.originalData = self.df.loc[self.df[columnName] == queryName]

        self.originalData=self.originalData.sort_values(by=self.dict['日期'], ascending=True)
        #self.originalData = self.originalData.values

        dict = {'电子商务类业务': 'dzswlywl',
                '信件类业务': 'xjlkdywl',
                '快递其他业务': 'qtywl',
                '快递业务总量': 'kdywzl',
                '国内同城快递业务': 'gntckdywl',
                '国内异地快递业务': 'gnydkdywl',
                '国际及港澳台快递业务': 'gjjgatkdywl',
                '普服业务量': 'yzywzl',
                '函件业务量': 'hjywl',
                '报刊业务量': 'bkywl',
                '包裹业务量': 'bgywl',
                '其他业务量': 'hdywl',
                '从业人员数量': 'kd_cyry',
                '管理人员': 'kd_glry',
                '专业技术人员': 'kd_zyjsry',
                '技能人员': 'kd_jnry',
                '快递其他人员': 'kd_ryqt',
                '快递研究生及以上': 'kd_yjsjysxl',
                '快递本科': 'kd_dxbkxl',
                '快递大专': 'kd_dzxl',
                '快递高中': 'kd_gzxl',
                '快递初中及初中以下': 'kd_czjczyxxl',
                '快递全日制职工': 'kd_qrzzg',
                '快递非全日制职工': 'kd_fqrzzg',
                '快递劳务派遣职工': 'kd_lwpqzg',
                '快递其他职工': 'kd_ygfsqt',
                '普服全日制职工': 'pf_qrzzg',
                '普服非全日制职工': 'pf_fqrzzg',
                '普服劳务派遣职工': 'pf_qtzg',
                '普服其他职工': 'pf_lwpqzg',
                '营业人员': 'pf_yyry',
                '投递人员': 'pf_tdry',
                '普服其他人员': 'pf_qtry',
                '普服研究生及以上学历': 'pf_yjsjysxl',
                '普服大学本科学历': 'pf_dxbkxl',
                '普服大专学历': 'pf_dzxl',
                '普服高中学历': 'pf_gzxl',
                '普服初中及初中以下学历': 'pf_czjczyxxl',
                '申诉率': 'sssl',
                '满意度': 'myd',
                '业务量': 'sszl',
                '农村快递网点': 'nckdwd',
                '农村快递业务量': 'nckdywl',
                '快递服务乡镇覆盖数': 'kdfwxzfgs',
                '快递服务建制村覆盖数': 'kdfwjzcfgs',
                '营业场所数量': 'yycs',
                '房屋建筑面积': 'fwjzmj',
                '运输汽车数量': 'ysqc',
                '运输飞机数量': 'ysfj',
                '无人机数': 'wrj',
                '自动化立体仓库': 'zdhltck',
                '快递网路条数': 'kdwlts',
                '快递网路长度': 'kdwlcd',
                '邮政局所': 'yzjs',
                '建制村': 'jzc',
                '报刊亭': 'bkt',
                '信报箱': 'xbx',
                '信筒': 'xt',
                '车辆': 'cl',
                '邮路总条数': 'ylzts',
                '邮路总长度': 'ylzcd',

                '日期': 'date',
                '地区层级': 'level',
                '省份名': 'prov_name',
                '城市名': 'city_name',
                '邮政编码': 'area_code',
                '地区名': 'area_name',
                '所属地域id': 'ofareaid',
                '所属地域': 'ofareaname',
                '港澳台快递业务量': 'gatkdywl',
                '国际快递业务量': 'gjkdywl',
                '快递业务总收入': 'kdywzsr',
                '国内异地快递业务收入': 'gnydkdywsr',
                '港澳台快递业务收入': 'gatkdywsr',
                '国际快递业务收入': 'gjkdywsr',
                '其他快递业务收入': 'qtkdywsr'}

        # self.__savePath = '/Users/ferdinand/PycharmProjects/LSTM_predictionOfPackage_v2.0/model/h5Files/%s.h5' % modelName
        # self.__originalData = originalData
        # self.__extractTimeFeature()
        # self.__MaxminNormalize()
        # self.__toSupervisedData()
        # self.__divideDataIntoTrainAndValid()


class lstmPredicterDataLoader(lstmModelDataLoader):
    def __init__(self, paramDict={}):
        self.validData = {}
        self.trainData = {}
        self.minMaxFilePath=paramDict['minMaxFilePath']
        self.queryName=paramDict['queryName']
        self.predictColumnName=paramDict['predictColumnName']
        self.context = paramDict['context']
        self.name = paramDict['name']
        self.filePath = paramDict['filePath']
        self.past = paramDict['past']
        self.future = paramDict['future']
        self.dict = getDict()
        # 首先要读取xlsx
        self.getOriginalData()
        # 接着处理数据，提取特征
        self.extractFeature()
        # 数据标准化
        self.normalizeData()
        # 转换成可以喂给模型的数据
        self.transferDataToSupervisedData()
        # 将数据分为训练，验证
        self.divideData()

    def transferDataToSupervisedData(self):
        data = np.concatenate((self.numFeature,self.timeFeature), axis=1)
        self.supervisedData = data
    def getOriginalData(self):
        filePath = self.context + self.filePath
        self.df = pd.read_excel(filePath)
        self.df.fillna(0, inplace=True)
        columnName = self.dict['地区名']
        queryName = self.queryName
        self.originalData = self.df.loc[(self.df[columnName] == queryName)]

        self.originalData=self.originalData.sort_values(by=self.dict['日期'], ascending=True)
        self.originalData=self.originalData.iloc[-self.past:]
    def normalizeData(self):
        self.minMax = joblib.load(filename=self.minMaxFilePath)
        self.numFeature = self.minMax.transform(self.numFeature.reshape(-1,1))

    def divideData(self):
        self.dataNeedToPredict=self.supervisedData

    def getDataNeedToPredict(self):
        return self.dataNeedToPredict.reshape(-1,self.past,self.feature)

if __name__ == '__main__':
    paramDict={}
    paramDict['minMaxFilePath']=''
    paramDict['queryName']='东莞市'
    paramDict['predictColumnName']='快递业务总量'
    paramDict['context']='../'
    paramDict['name']='no.1'
    paramDict['filePath']='data/zb_merge_kdywl_month.xlsx'
    paramDict['past']=5*12
    paramDict['future']=2*12
    lstmModelDataLoader(paramDict)
