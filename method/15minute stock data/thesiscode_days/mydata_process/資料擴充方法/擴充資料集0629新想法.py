# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:56:52 2024

@author: User
"""
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
import pandas as pd
import requests
from io import StringIO
import time
import pandas as pd
import requests
from io import StringIO
import time

#載入原本財務資料
#final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0601__0到405.csv', encoding='utf_8_sig')#輸出excel檔
final_data=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0601__0到405.csv', encoding='utf_8_sig')

#final_data2.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0601__0到405.csv', index = False, encoding='utf_8_sig')#輸出excel檔

#載入擴充的新資料
bigfinal_data=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')#輸出excel檔

stock_id=final_data['公司代號'].unique()

newfinal=pd.DataFrame()

#將final_data_real最後一個月合併到bigfinal_datal
for i in range(0,len(stock_id)):
    print("****************第"+ str(i) + "隻股票****************")
    temp_stockid=stock_id[i]
    #final_data
    temp_stockdataf=final_data[final_data['公司代號']==temp_stockid]
    #bigfinal_data
    temp_stockdatab=bigfinal_data[bigfinal_data['公司代號']==temp_stockid]  
    #找到財報為5月份的
    tempf=temp_stockdataf[temp_stockdataf['月份']=='2024-05']
    
    if(len(temp_stockdatab)==132 and len(tempf)==1):
        
        #更新tempf股價
        stock_data = yf.download(str(temp_stockid)+".TW", start='2024-06-15', end= '2024-07-01')
        #買入價
        tempf.iloc[0,31]=stock_data.Close[0]
        #賣出價
        tempf.iloc[0,32]=stock_data.Close[len(stock_data)-1]                  
        #報酬率
        tempf.iloc[0,30]=(tempf.iloc[0,32]-tempf.iloc[0,31])/tempf.iloc[0,31]
  
        
        #跟原先資料合併
        temp_stockdata=pd.concat([pd.DataFrame(temp_stockdatab),pd.DataFrame(tempf)], axis=0)
        #給予index
        index1 = pd.Index(list(range(133)))  # index1 = pd.Index(list(range(44))) 
        temp_stockdata = temp_stockdata.set_index(index1)
        temp_stockdata=temp_stockdata.drop(columns='Unnamed: 0')
        #合併資料集
        newfinal=pd.concat([pd.DataFrame(newfinal),pd.DataFrame(temp_stockdata)], axis=0)




#重設index
index2 = pd.Index(list(range(53998)))  # index1 = pd.Index(list(range(44))) 
newfinal = newfinal.set_index(index2)
##輸出最後結果
newfinal.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')


final_data_real1=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')
        

