# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:53:25 2024

@author: User
"""
####套件安裝###########
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import datetime
import requests
import json
import time
import csv
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median
from dateutil.relativedelta import relativedelta
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
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
import tqdm 
import tensorflow as tf
import os
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout



#############################所有股票#################
#AAPL= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AAPL.csv', encoding='utf_8_sig')

#AMZN= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AMZN.csv', encoding='utf_8_sig')

#= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/GOOG.csv', encoding='utf_8_sig')

#MSFT= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/MSFT.csv', encoding='utf_8_sig')

#TSLA= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/TSLA.csv', encoding='utf_8_sig')



AAPL= yf.download("AAPL", start="1980-01-01", end="2024-07-31")
TSLA= yf.download("TSLA", start="1980-01-01", end="2024-07-31")
MSFT= yf.download("MSFT", start="1980-01-01", end="2024-07-31")
IBM = yf.download("IBM ", start="1980-01-01", end="2024-07-31")



#####################合併大資料集 final_data_real##############################
final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)

stock_id=['AAPL','TSLA','MSFT','IBM']

stock_mae=[] #股票MSE
stock=[] #股票名稱

###複製表格
final_data_real_copy=final_data_real


###使用minmaxscalar
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))


for k in range(0,1):  #len(stock_id)
    print("第"+str(k)+"支股票")
    tsla_data =final_data_real[k]
    ##將Date移到index#########
    tsla_data.columns
    #tsla_data.set_index(['Date'], inplace=True)
    

    # Extracting the close price from the DataFrame
    tsla_close = tsla_data['Close'].values
    # Normalizing the TSLA stock data using MinMaxScaler
    tsla_data=tsla_data.drop('Adj Close', axis=1)
   
    ######## train 60% val 20% test 20%   ##############
    n = 10
    train =tsla_data[:int(len(tsla_data) *0.6)]
    val =tsla_data[:int(len(tsla_data) *0.8)]
    test =tsla_data[int(len(tsla_data) *0.8):]
    ##保留test10天候的數值
    y_testc=test['Close'][n:]
    feature_names = list(train.drop('Close', axis=1).columns)
    x_train = []
    y_train = []
    train_indexes = []
    #train 資料
    norm_data_xtrain = train[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(train)-n)):#range(0,len(train)-n)        
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainaddscalar=x_scaler.fit_transform(x_trainadd)

        x_train.append(np.transpose(x_trainaddscalar)) 
        y_train.append(train['Close'].iloc[i+n]) 
        train_indexes.append(train.index[i+n]) 
    print(x_train[0])
    
    x_train=np.array(x_train)
    y_train_dataframe=pd.DataFrame(y_train).iloc[:len(y_train)]
    y_train_tran=y_scaler.fit_transform(y_train_dataframe)
    y_train=np.array(y_train_tran).reshape(-1)
    print(x_train.shape)
      
    #val 資料
    x_val = []
    y_val = []
    val_indexes = []
    norm_data_xval = val[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(val)-n)):       
        x_valadd=norm_data_xval.iloc[i:i+n]. values
        x_valaddscalar=x_scaler.fit_transform(x_valadd)
        x_val.append(np.transpose(x_valaddscalar))
        y_val.append(val['Close'].iloc[i+n]) 
    ##轉成array
    print(x_val[0])
    x_val=np.array(x_val)
    y_val_dataframe=pd.DataFrame(y_val).iloc[:len(y_train)]
    y_val_val=y_scaler.fit_transform(y_val_dataframe)
    y_val=np.array(y_val_val).reshape(-1)
    print(x_val.shape)    
      
    ##test部分##
    x_test = []
    y_test = []
    test_indexes = []
    
    norm_data_xtest = test[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(test)-n)): 
        x_testadd=norm_data_xtest.iloc[i:i+n]. values
        x_testaddscalar=x_scaler.fit_transform(x_testadd)
        x_test.append(np.transpose(x_testaddscalar))
        y_test.append(test['Close'].iloc[i+n]) 
        test_indexes.append(test.index[i+n]) 

    #先備份
    x_test1=x_test
    y_test1=y_test

    x_test=np.array(x_test)

    y_test_dataframe=pd.DataFrame(y_test).iloc[:len(y_test)]
    y_test_tran=y_scaler.fit_transform(y_test_dataframe)
    y_test=np.array(y_test_tran).reshape(-1) 
    
    #開始跑模型
    n = 10
    n_steps = n 
    n_features = 4
    model = keras.models.Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_features,n_steps)))
    model.add(LSTM(20,activation='relu'))
    model.add(Dense(1))
    # 顯示網路模型架構
    model.summary()
    
    model.compile(keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()  loss=custom_mean_squared_error
    metrics=[keras.metrics.MeanAbsoluteError()])
    
    #設定回調函數
    model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock/model8/'
    #os.makedirs(model_dir)
    # TensorBoard回調函數會幫忙紀錄訓練資訊，並存成TensorBoard的紀錄檔
    log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock', 'model8')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    # ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
    #將模型儲存在C:/Users/User/lab2-logs/models/
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                                 monitor='val_mean_absolute_error', 
                                                 save_best_only=True, 
                                                 mode='min')
    
    
    history = model.fit(x_train,y_train,batch_size=32,epochs=30)
    
    #############預測x_test
    predictions = model.predict(x_test)
    predictions1=predictions.reshape(-1)
    ##minmax還原成正常的prediction
    predictions_orign = y_scaler.inverse_transform(predictions)
    
    # 顯示誤差百分比 顯示到小數點第二位
    meanmae_error=np.mean(abs(predictions_orign- np.array(y_testc)))
    
    ##將誤差和資料儲存起來
    stock_mae.append(meanmae_error) #股票MSE
    stock.append(stock_id[k]) #股票名稱

##合併大資料 ############
big_lstm_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)
#big_lstm_data.mean()
#big_lstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALAR CNN_LSTM.csv', encoding='utf_8_sig')
#big_lstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/pytorch資料/transformer_tensorflow/比較模型程式碼\比較資料庫/four company/實驗結果/CNN_LSTM.csv', encoding='utf_8_sig')


