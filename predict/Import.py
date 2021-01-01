#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Import.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/17 1:40 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
from sklearn.model_selection import train_test_split


import abc #利用abc模块实现抽象类
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import builtins
import pandas as pd
import jieba
import sklearn
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow import *
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input, Lambda, Concatenate
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Model
import numpy as np
from gensim.models import word2vec
from  pandas import  DataFrame
from keras.optimizers import Adam
from matplotlib import pyplot
import time