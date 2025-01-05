# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:45:22 2024

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
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime

from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數

############################跑lstm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

###########神經元層
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention

# Adding the LSTM layer
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
import tensorflow as tf
from random import sample
import os
##############################0506從這開始跑##############################################
#final_data_real=pd.read_csv(r'C:\Users\2507\Desktop\遠端資料/0427擴充資料集bigfinal_data__0到405最後修正.csv', encoding='utf_8_sig')

#k原本只做到3月份
#final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0427擴充資料集bigfinal_data__0到405最後修正.csv', encoding='utf_8_sig')

#做到4月份
final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')



final_data_real=final_data_real.iloc[:,2:]

#final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','賣出價'], axis=1)
final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','return_rate'], axis=1)
##計算所有欄位平均值##
final_data_mean=final_data_real.iloc[:,2:]


##x變數先minscalar 
###使用minmaxscalar
x_scaler = MinMaxScaler(feature_range = (0, 1))
#使用x變數
x_variable=final_data_real.iloc[:,1:22]
x_variable=pd.DataFrame(x_scaler.fit_transform(x_variable))

#使用y變數
y_variable=pd.DataFrame(final_data_real.iloc[:,22])
#y_variable=pd.DataFrame(y_scaler.fit_transform(y_variable))
#company_name
company_name=pd.DataFrame(final_data_real.iloc[:,0])

#合併concat成大資料
columnname=final_data_real.columns
final_data_real=pd.concat([company_name,x_variable,y_variable],axis=1)
final_data_real.columns=columnname

##開始設變數跑資料
#只有一筆資料有na值
final_data_real=final_data_real.dropna()
print(final_data_real.isnull().sum())

##x_train 和 y_train
x_bigdata = []
y_bigdata = []

##wgangp需要新增的資料集
yc_data=[]


###############################################開始整理資料

stock_id=final_data_real['公司代號'].unique()


stock_mae=[] #股票MSE
stock=[] #股票名稱

###複製表格
final_data_real_copy=final_data_real

for k in range(0,len(stock_id)):  #len(stock_id)
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[final_data_real['公司代號']==stock_id[k]]    
    
    if(stock_id[k]==4142): #若為國光生計就跳過
        continue
    #final_data.set_index(['月份'], inplace=True)
    final_data=final_data.drop('公司代號', axis=1)
    
######製造資料集 ######  現有10天去預測11天 選10天因為val 只有16
#%%
#製造X跟Y(RD)
###########################TRAIN 資料#############
    #import tqdm #使用進度條
    n = 4 #n預設為10改n即可，資料1/4起，所以能預測的第一個Y為2/23，抓30天
    feature_names = list(final_data.drop('賣出價', axis=1).columns)

    train_indexes = []
    #train 資料
    norm_data_xtrain = final_data[feature_names]    
    #yc_train資料
    norm_data_yctrain = pd.DataFrame(final_data['賣出價'])
        

    
    for i in tqdm.tqdm_notebook(range(0,len(final_data)-n)):#range(0,len(train)-n) 
        ##加入minmax value        
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        #x_trainaddscalar=x_scaler.fit_transform(x_trainadd)
        x_bigdata.append(x_trainadd)  #x_train.append(norm_data_xtrain.iloc[i:i+n]. values)        
        #x_bigdata.append(x_trainaddscalar)

        ##yc的部分
        #yc_trainadd=norm_data_yctrain.iloc[i:i+n]. values  
        yc_trainadd=norm_data_yctrain[i:i+n]
        yc_data.append(yc_trainadd)  #x_train.append(norm_data_xtrain.iloc[i:i+n]. values)        
        #x_train.append(norm_data_xtrain.iloc[i:i+n]. values)
        #修改
        y_bigdata.append(final_data['賣出價'].iloc[i+n-1]) #現有資料+10天的Yy_train.append(train['return_rate'].iloc[i+n-1])
        #原本
        #y_train.append(train['return_rate'].iloc[i+n])
        #train_indexes.append(train.index[i+n])
    ##轉成array
    print(x_bigdata[0])
    #print(x_train.shape)
      
    
########最後隨機區分 80%訓練  20%測試  0520從這開始跑 ################
np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdata1', np.array(x_bigdata))
np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata1', np.array(y_bigdata))
#np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/yc_data', np.array(yc_data))

#x_bigdata1=np.array(x_bigdata)
#yc_data1=np.array(yc_data)




######################轉成資料結束


#############################2.CNN轉置資料##################
final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')
final_data_real=final_data_real.iloc[:,2:]

final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','return_rate'], axis=1)

#final_data_mean.mean()


##x變數先minscalar 
#使用x變數
x_variable=final_data_real.iloc[:,1:22]
x_variable=pd.DataFrame(x_scaler.fit_transform(x_variable))

#使用y變數
y_variable=pd.DataFrame(final_data_real.iloc[:,22])
#y_variable=pd.DataFrame(y_scaler.fit_transform(y_variable))
#company_name
company_name=pd.DataFrame(final_data_real.iloc[:,0])

#合併concat成大資料
columnname=final_data_real.columns
final_data_real=pd.concat([company_name,x_variable,y_variable],axis=1)
final_data_real.columns=columnname



##x變數先minscalar 暫時不用
##############################0220跑結束#################################################
##NAN值分布狀況 77家公司有na值
#只有一筆資料有na值
final_data_real=final_data_real.dropna()
print(final_data_real.isnull().sum())



##x_train 和 y_train

x_bigdataT = []
y_bigdata = []


###############################################開始整理資料

stock_id=final_data_real['公司代號'].unique()


stock_mae=[] #股票MSE
stock=[] #股票名稱

###複製表格
final_data_real_copy=final_data_real

for k in range(0,len(stock_id)):  #len(stock_id)
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[final_data_real['公司代號']==stock_id[k]]    
    
    if(stock_id[k]==4142): #若為國光生計就跳過
        continue
    #final_data.set_index(['月份'], inplace=True)
    final_data=final_data.drop('公司代號', axis=1)
    
######製造資料集 ######  現有10天去預測11天 選10天因為val 只有16
#%%
#製造X跟Y(RD)
###########################TRAIN 資料#############
    #import tqdm #使用進度條
    #n = 10 #n預設為10改n即可，資料1/4起，所以能預測的第一個Y為2/23，抓30天
    feature_names = list(final_data.drop('賣出價', axis=1).columns)

    train_indexes = []
    #train 資料
    norm_data_xtrain = final_data[feature_names]    
    #yc_train資料
    norm_data_yctrain = pd.DataFrame(final_data['賣出價'])
        

    
    for i in tqdm.tqdm_notebook(range(0,len(final_data)-n)):#range(0,len(train)-n)      
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        #轉置資料不一樣的地方
        x_bigdataT.append(np.transpose(x_trainadd) ) #x_train.append(norm_data_xtrain.iloc[i:i+n]. values)        
        #修改
        y_bigdata.append(final_data['賣出價'].iloc[i+n-1]) #現有資料+10天的Yy_train.append(train['return_rate'].iloc[i+n-1])
        #原本
        #y_train.append(train['return_rate'].iloc[i+n])
        #train_indexes.append(train.index[i+n])
    ##轉成array
    print(x_bigdataT[0])
    #print(x_train.shape)
      
    
########儲存資料################
np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdataT1', np.array(x_bigdataT))
#np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata', np.array(y_bigdata))













