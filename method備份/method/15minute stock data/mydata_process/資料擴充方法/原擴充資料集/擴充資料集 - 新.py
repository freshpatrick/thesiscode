# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:58:11 2024

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
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
#-------------------月營收爬蟲-------------------
#月爬蟲
#網站:
#https://mops.twse.com.tw/nas/t21/sii/t21sc03_111_1_0.html

import pandas as pd
import requests
from io import StringIO
import time


#-------------------月營收爬蟲-------------------
#月爬蟲
#網站:
#https://mops.twse.com.tw/nas/t21/sii/t21sc03_113_2_0.html

import pandas as pd
import requests
from io import StringIO
import time
def monthly_report(year, month):
    
    # 假如是西元，轉成民國
    if year > 1990:
        year -= 1911
    
    url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'_0.html'
    if year <= 98:
        url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'.html'
    
    # 偽瀏覽器
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    # 下載該年月的網站，並用pandas轉換成 dataframe
    r = requests.get(url, headers=headers)
    r.encoding = 'big5'

    dfs = pd.read_html(StringIO(r.text), encoding='big-5')

    df = pd.concat([df for df in dfs if df.shape[1] <= 11 and df.shape[1] > 5])
    
    if 'levels' in dir(df.columns):
        df.columns = df.columns.get_level_values(1)
    else:
        df = df[list(range(0,10))]
        column_index = df.index[(df[0] == '公司代號')][0]
        df.columns = df.iloc[column_index]
    
    df['當月營收'] = pd.to_numeric(df['當月營收'], 'coerce')
    df = df[~df['當月營收'].isnull()]
    df = df[df['公司代號'] != '合計']
    
    # 偽停頓
    time.sleep(5)

    return df


# 民國108年7月營收在8月10號之前公布
monthdata=monthly_report(109,10)


#########爬10年所有月營收(跑資料不用)##############
from dateutil.relativedelta import relativedelta
from datetime import datetime


#temp_month1="2013-05"

temp_month1="2024-04"

  ##如果12月10號就可以找全部資料庫  變成43
#while p < 128:  #128個月
p=0
while p < 2:  #2個月  先到2月 4/15之後再用3個月
    print(p)
    now_year=int(temp_month1[0:4])-1911 #轉成民國
    now_month=int(temp_month1[5:7])
    ##爬越資料
    monthdata=monthly_report(now_year,now_month)
    ##儲存月資料 
    #轉換資料
    #月份資料
    #月份的month轉成str
    #存檔
    monthdata.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/newest_allmonth/all_monthdata'+str(temp_month1[0:4])+'_'+str(temp_month1[5:7])+'.csv', index = False, encoding='utf_8_sig')#輸出excel檔

    
    # 偽停頓
    time.sleep(0.5)
    
    #找下個月
    temp_month1=datetime.strptime(temp_month1, '%Y-%m')
    temp_month1=temp_month1+ relativedelta(months=1) 
    #補二位數
    temp_month1=str(temp_month1.year)+'-'+str('{0:02d}'.format(temp_month1.month))

    #p=p+1
    p+=1

    #120  2023-05沒有資料








#########爬10年所有月營收結束#############








############原始資料0~864 1.17 從這跑##############################
#########合併0~864資料檔#########
final_data_1=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__0到250最新.csv', encoding='utf_8_sig')
final_data_2=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__250到400.csv', encoding='utf_8_sig')
final_data_3=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__400到864.csv', encoding='utf_8_sig')

final_data_real=pd.concat([final_data_1,final_data_2,final_data_3], axis=0)

###########0407 自己測試###############
#final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0407__0到2.csv', encoding='utf_8_sig')#輸出excel檔

final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0424__0到405.csv', encoding='utf_8_sig')#輸出excel檔

##0601載入新資料##
final_data_real=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0601__0到405.csv', encoding='utf_8_sig')#輸出excel檔




#備份到final data
final_data_test= final_data_real
final_data= final_data_real

#########找出43筆均沒有na 的股票###########
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


###########################################套件結束#####
final_data = final_data.dropna()
print(final_data.isnull().sum())

final_data.columns

#共706家廠商  0117從這開始執行
stock_id=final_data['公司代號'].unique()

#最終資料檔  #########0407從這開始跑#############
bigfinal_data= pd.DataFrame()

#查看每個股票的組合
stocklist=[]

#沒有完全資料的月營收股票 :1314 i=38 k=124這邊
target_stock_no=[]
target_stock_no_list=[]

##載入原先資料
bigfinal_data_first= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0427擴充資料集bigfinal_data__0到405最後修正.csv', encoding='utf_8_sig')
bigfinal_data_first=bigfinal_data_first.iloc[:,2:]

stock_ida=bigfinal_data_first['公司代號'].unique()


for i in range(0,len(stock_id)):  #len(stock_id)
    print("****************第"+ str(i) + "隻股票****************")
    temp_stockid=stock_id[i]
    temp_stockdata=final_data[final_data['公司代號']==temp_stockid]
    temp_stock_orign=bigfinal_data_first[bigfinal_data_first['公司代號']==temp_stockid]
    temp_stock_orign=temp_stock_orign.iloc[:130,:]
    ##重設temp data index   
    #備份
    #temp_stockdata2=temp_stockdata
    
    
    #若資料長度為45就繼續，否則就跳過  6/15後要改成45 這邊先改44
    if(len(temp_stockdata)==44): #原本if(len(temp_stockdata)==44):
        #重設temp_stockdata3 index
        index1 = pd.Index(list(range(44)))  #本來index1 = pd.Index(list(range(43)))
        temp_stockdata = temp_stockdata.set_index(index1)
        #備份
        temp_stockdata2=temp_stockdata
        
        #原本
        #temp_stockdata3=pd.DataFrame(temp_stockdata.iloc[0,:]).transpose()

        temp_stockdata3=pd.DataFrame(temp_stockdata.iloc[43,:]).transpose()


        #######新想法:##############
        #1.先建立空集合
        #temp_month=temp_stockdata['月份'][0]
        #月份
        #原temp_month=temp_stockdata.iloc[0,0]
        #temp_month=temp_stockdata.iloc[43,0]
        #temp_month=temp_stockdata['月份'][j]
        #temp_month=datetime.strptime(temp_month, '%Y-%m')
        for j in range(0,1):     ###130個月  到11211月   4/15後就要改130
            print(j)
            temp_month=temp_stockdata['月份'][43+j]
            temp_month=datetime.strptime(temp_month, '%Y-%m')
            #月份的month轉成str
            #str_month=str(temp_month.year)+'-'+str(temp_month.month)
            #多增加一個月
            temp_month=temp_month+ relativedelta(months=1)
            #補二位數
            str_month=str(temp_month.year)+'-'+str('{0:02d}'.format(temp_month.month))
            
            #擴充一個空集合
            #temp_stockdata4=pd.concat([pd.DataFrame(temp_stockdata3).transpose() ,pd.DataFrame(pd.Series(str_month)) ], axis=0)
            temp_stockdata3.loc[len(temp_stockdata3)] = pd.Series(dtype='float64') #dtype='float64'
            temp_stockdata3.iloc[len(temp_stockdata3)-1,0]=str_month
            
            #若為相同月份就帶入資料
            if(str_month in temp_stockdata['月份'].values):
                temp_monthdata=temp_stockdata[temp_stockdata2['月份']==str_month]
                temp_stockdata3.iloc[len(temp_stockdata3)-1,:]=temp_monthdata.values 
                             
                #7/1把這個省略，因為就可以抓6/15到6/31股價
                temp_stockdata3=temp_stockdata3.iloc[:2,:]
                
            index_temp = pd.Index(pd.Series([43,44]))  #本來index1 = pd.Index(list(range(43)))
            temp_stockdata3= temp_stockdata3.set_index(index_temp)
            #temp_stockdata3備份
            temp_stockdata4=temp_stockdata3 
            
           ############建空資料結束####################
          
           ##########for迴圈帶入股價資料############
        for k in range(0,len(temp_stockdata3)):     #len(temp_stockdata3)
            print(k)
            #若為空集合就帶入前一個資料
            if(np.isnan(temp_stockdata3.iloc[k,4])==True):

                
               temp_stockdata3.iloc[k,1:]=temp_stockdata3.iloc[k-1,1:].values
               #抓月份資料
               t_month=temp_stockdata3.iloc[k,0]
               
               ##補月營收資訊
               #讀月資料匯入
               monthly_data=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/newest_allmonth/all_monthdata'+str(t_month[0:4])+'_'+str(t_month[5:7])+'.csv', encoding='utf_8_sig')

               target_stock=monthly_data[monthly_data['公司代號']==str(temp_stockid)]
               
               if(len(target_stock)==0):
                   #註明哪支第幾個月找不到月營收
                   word_t="股票代號:"+str(temp_stockid)+'的'+str(t_month)+'無月資料'
                   #買入價設nan以利之後刪除
                   temp_stockdata3.iloc[k,31]=np.nan
                   #賣出價
                   temp_stockdata3.iloc[k,32]=np.nan

                  
                  ########把沒月營收的股票價格設na都列近來#############
                   target_stock_no.append(word_t)
                   target_stock_no_list.append(temp_stockid)
                   continue
                   
               index2 = pd.Index(list(range(1)))
               target_stock = target_stock.set_index(index2)
              
            ##########改四行置換每月營收結束###########
               temp_stockdata3.iloc[k,4]=pd.DataFrame(target_stock['上月比較增減(%)']).iloc[0,0]
               temp_stockdata3.iloc[k,5]=pd.DataFrame(target_stock['上月營收']).iloc[0,0]
               temp_stockdata3.iloc[k,6]=pd.DataFrame(target_stock['去年同月增減(%)']).iloc[0,0]
               temp_stockdata3.iloc[k,7]=pd.DataFrame(target_stock['去年當月營收']).iloc[0,0]
               temp_stockdata3.iloc[k,8]=pd.DataFrame(target_stock['當月營收']).iloc[0,0]
               temp_stockdata3.iloc[k,9]=pd.DataFrame(target_stock['前期比較增減(%)']).iloc[0,0]
               temp_stockdata3.iloc[k,10]=pd.DataFrame(target_stock['去年累計營收']).iloc[0,0]
               temp_stockdata3.iloc[k,11]=pd.DataFrame(target_stock['當月累計營收']).iloc[0,0]


            ####補股價
            t_month=temp_stockdata3.iloc[k,0]
            t_month=datetime.strptime(t_month, '%Y-%m')
                  
            #多增加一個月
            t_month=t_month+ relativedelta(months=1)
            
            ##後起點多增加一個月到個月月1號 這樣跑股價只會跑到月底
            t_month_end=t_month+ relativedelta(months=1)
            
               #補二位數
            str_month_stock=str(t_month.year)+'-'+str('{0:02d}'.format(t_month.month))
            
            str_month_stock_end=str(t_month_end.year)+'-'+str('{0:02d}'.format(t_month_end.month))
            
            stock_data = yf.download(str(temp_stockid)+".TW", start= str_month_stock+'-15', end= str_month_stock_end+'-01')
            
            
            
            #################0407結束#########################
            #買入價
            temp_stockdata3.iloc[k,31]=stock_data.Close[0]
            #賣出價
            temp_stockdata3.iloc[k,32]=stock_data.Close[len(stock_data)-1]
                  
            #報酬率
            temp_stockdata3.iloc[k,30]=(temp_stockdata3.iloc[k,32]-temp_stockdata3.iloc[k,31])/temp_stockdata3.iloc[k,31]

        
        #跟原先資料合併
        temp_stockdata3=pd.concat([pd.DataFrame(temp_stock_orign),pd.DataFrame(temp_stockdata3)], axis=0)
        index_final = pd.Index(list(range(132)))  #本來index1 = pd.Index(list(range(43)))
        temp_stockdata3 = temp_stockdata3.set_index(index_final)

        stocklist.append(temp_stockdata3)
        ##合併大資料
        bigfinal_data=pd.concat([pd.DataFrame(bigfinal_data),pd.DataFrame(temp_stockdata3)], axis=0)

#0601暫存 更新到2024 4月版
bigfinal_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0601擴充資料集bigfinal_data__0到405最終.csv', encoding='utf_8_sig')








#匯出csv檔
#bigfinal_data=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/擴充資料集bigfinal_data__0到870.csv', encoding='utf_8_sig')

bigfinal_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0427擴充資料集bigfinal_data__0到405最新也是最終.csv', encoding='utf_8_sig')

            
#####################################1.17擴充結束#############################
                 
 ############1.18開始合併資料集################
bigfinal_data_1= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/擴充資料集bigfinal_data__0到230.csv', encoding='utf_8_sig')
bigfinal_data_2= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/擴充資料集bigfinal_data__230到707.csv', encoding='utf_8_sig')
#共52578筆                       
fbigfinal_data=pd.concat([pd.DataFrame(bigfinal_data_1),pd.DataFrame(bigfinal_data_2)], axis=0)

fbigfinal_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/擴充資料集bigfinal_data__0到707.csv', encoding='utf_8_sig')
            
 
            
##########################0407 結束###############
    
 
    
 
    
 
    
 
            
            

        
        
    
    
    










