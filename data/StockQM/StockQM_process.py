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
from pandas_datareader._utils import RemoteDataError
from numpy import median#
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
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


final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/405companydata.csv', encoding='utf_8_sig')
ten_days=pd.DataFrame()

for i in range(30,len(final_data_real)):
    print("******第"+str(i)+"筆資料****************")
    stockid=final_data_real['公司代號'][i]
    year=int(final_data_real['月份'][i][:4])
    month=int(final_data_real['月份'][i][5:7])
    #多加兩個月
    month_end=str(month+2).zfill(2)
    month_start=str(month+1).zfill(2)
    year_start=year
    year_end=year
    
    if(month==11):
        year_end=year_end+1   
        month_end=1
    
    if(month==12):
        year_start=year_start+1
        year_end=year_end+1  
        month_start=1
        month_end=2
        
        
    stock_data = yf.download(str(stockid)+'.TW', start=str(year_start)+'-'+str(month_start)+'-01', end=str(year_end)+'-'+str(month_end)+'-01')
    stock_tendays = stock_data['Close'][(len(stock_data)-11):(len(stock_data))]
    temp_ten=pd.DataFrame(stock_tendays)
    index_temp = pd.Index(list(range(len(temp_ten)))) 
    temp_ten = temp_ten.set_index(index_temp)
    ten_days=pd.concat([ten_days,temp_ten], axis=1)
    
ten_dayscopy=ten_days    
    
ten_days.columns=['前10天','前9天','前8天','前7天','前6天','前5天','前4天','前3天','前2天','前1天','新賣出價']

final_data_real2=pd.concat([final_data_real,ten_days], axis=1)
final_data_real2.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/1222擴充資料集bigfinal_data__0到405最終.csv', index = False, encoding='utf_8_sig')#輸出excel檔

#load data
final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/1222擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')
final_data_real=final_data_real.iloc[:,2:]
final_data_real=final_data_real.drop(['公司名稱','備註','上月營收','當月營收','去年累計營收','當月累計營收','去年當月營收','買入價','return_rate','賣出價'], axis=1)
final_data_real.columns
final_data_real['長期資金佔不動產']=final_data_real['長期資金佔不動產']*1000
max(final_data_real['Net_Income'])
min(final_data_real['Net_Income'])
max(final_data_real['長期資金佔不動產'])
min(final_data_real['長期資金佔不動產'])
final_data_mean=final_data_real.iloc[:,2:]
x_scaler = MinMaxScaler(feature_range = (0, 1))
#使用x變數
x_variable=final_data_real.iloc[:,1:32]
x_variable=pd.DataFrame(x_scaler.fit_transform(x_variable))
#使用y變數
y_variable=pd.DataFrame(final_data_real.iloc[:,32])

company_name=pd.DataFrame(final_data_real.iloc[:,0])

#concat
columnname=final_data_real.columns
final_data_real=pd.concat([company_name,x_variable,y_variable],axis=1)
final_data_real.columns=columnname

print(final_data_real.isnull().sum())
isnull=final_data_real.isnull()
null_locat=np.where(isnull)

for j in range(0,len(null_locat[0])):
    null_row=null_locat[0][j]
    null_col=null_locat[1][j]
    nullbool=pd.isnull(final_data_real.iloc[null_row,null_col])

    if(nullbool==True):
        #找有值的最後一個數值取代
        print("***第"+str(j)+"***筆有缺值資料***")
        notnull=np.where(final_data_real.iloc[null_row,:].notnull())[0]
        final_data_real.iloc[null_row,null_col]=final_data_real.iloc[null_row,notnull[len(notnull)-1]]
        
print(final_data_real.isnull().sum())#無空值

x_bigdata = []
y_bigdata = []
yc_data=[]
stock_id=final_data_real['公司代號'].unique()
stock_mae=[] 
stock=[] 
final_data_real_copy=final_data_real

for k in range(0,len(stock_id)): 
    print("*****************第"+str(k)+"********************支股票")
    #先拿台泥做比較
    final_data=final_data_real[final_data_real['公司代號']==stock_id[k]]    
    
    if(stock_id[k]==4142): 
        continue
    final_data=final_data.drop('公司代號', axis=1)
    n = 2 
    feature_names = list(final_data.drop('新賣出價', axis=1).columns)

    train_indexes = []
    norm_data_xtrain = final_data[feature_names]    
    norm_data_yctrain = pd.DataFrame(final_data['新賣出價'])
        

    for i in tqdm.tqdm_notebook(range(0,len(final_data)-n)):   
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainadd=np.transpose(x_trainadd)
        x_bigdata.append( x_trainadd)
        yc_trainadd=norm_data_yctrain[i:i+n]
        yc_data.append(yc_trainadd)     
        y_bigdata.append(final_data['新賣出價'].iloc[i+n-1]) 
    print(x_bigdata[0])

    
    
np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdata', np.array(x_bigdata))
np.save(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata', np.array(y_bigdata))


    
