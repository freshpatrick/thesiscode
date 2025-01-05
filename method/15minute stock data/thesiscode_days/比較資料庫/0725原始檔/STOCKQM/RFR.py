# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 04:10:30 2024

@author: User
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import requests
import json
import time
import csv
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import requests
import json
import time
import csv
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm #使用進度條

###############RFR套件#########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#載入資料集##
data=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')
bigdata=data.drop(['Unnamed: 0', '月份','公司代號','公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','賣出價'], axis=1)
x_bigdata=bigdata.drop(['return_rate'], axis=1)
y_bigdata=bigdata.iloc[:,21]
##合併成大資料先替除nan
concat_data=pd.concat([x_bigdata,y_bigdata], axis=1)
concat_data=concat_data.dropna()
x_bigdata=concat_data.iloc[:,:21]
#在這使用x的minmaxscalar
x_scaler = MinMaxScaler(feature_range = (-1, 1))
x_bigdata=pd.DataFrame(x_scaler.fit_transform(x_bigdata))
##跑模型## 80%訓練  20%測試
indexs=np.random.permutation(len(x_bigdata)) #隨機排序 49005以下的數字
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
test_indexs=indexs[int(len(x_bigdata)*0.6):]
#x部分
x_train=np.array(x_bigdata.loc[x_bigdata.index.intersection(train_indexs),:])  
x_test=np.array(x_bigdata.loc[x_bigdata.index.intersection(test_indexs),:])  
#y部分
y_bigdata=pd.DataFrame(y_bigdata)
y_train=np.array(y_bigdata.loc[y_bigdata.index.intersection(train_indexs),:]).reshape(-1)
y_test=np.array(y_bigdata.loc[y_bigdata.index.intersection(test_indexs),:]).reshape(-1)
# RandomFores 迴歸
rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
rfr.fit(x_train, y_train)
# Predict 預測y
y_pred = rfr.predict(x_test)
y_pred=np.reshape(y_pred,y_test.shape)
# 計算MAE
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
#meanmae_error   0.044
print(" 平均mae誤差: {:.2f}".format(meanmae_error))

















