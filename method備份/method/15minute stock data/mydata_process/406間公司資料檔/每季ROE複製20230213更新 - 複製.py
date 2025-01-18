
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:02:59 2020

@author: User
"""
#########公開資訊觀測站###############
#5/14開始觀察 希望觀察每股盈餘和股東權益報酬率



#https://mopsfin.twse.com.tw/
#股東權益報酬率
#https://mopsfin.twse.com.tw/compare/data?compareItem=ROE&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId=2317
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

#-------------------------------月營收爬蟲結束--------------------------


stock_data = pd.read_csv(r'D:\20200508桌面大檔\20200506pyhton股票投資\20200822每季財務比率投資策略\股票代碼.csv', encoding='utf_8_sig')
stockid=pd.DataFrame(stock_data['股票代碼'])




###################爬ROE 營業利益率 稅後淨利率##############
#https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId=2034+%E5%85%81%E5%BC%B7+(%E4%B8%8A%E5%B8%82%E9%8B%BC%E9%90%B5%E5%B7%A5%E6%A5%AD)
#https://mopsfin.twse.com.tw/compare/data?compareItem=NetIncomeMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId=2034+%E5%85%81%E5%BC%B7+(%E4%B8%8A%E5%B8%82%E9%8B%BC%E9%90%B5%E5%B7%A5%E6%A5%AD)
#



stock_id=[]#id

ROE_data=[]
operate_margin=[]#營業利益率
Net_Income=[]#稅後淨利率

#爬蟲網址
#https://mopsfin.twse.com.tw/compare/data?compareItem=ROE&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId=2330

for i in range(834,len(stockid)):#len(stockid)
    print(i)


    now_stockid=str(stockid['股票代碼'][i])#現在股票代碼
    
    
    ##############ROE資料##################
    url_twse='https://mopsfin.twse.com.tw/compare/data?compareItem=ROE&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid

    res=requests.get(url_twse)
    time.sleep(0.5)
    s=json.loads(res.text)#將資料轉成json格式
  
    try:
        fiance_name=s['xaxisList']#季度

    except KeyError:
        continue
    
    
    ##############營業利益率############
    url_twse1='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid

    res1=requests.get(url_twse1)
    time.sleep(0.5)
    #print(res.text)

    s1=json.loads(res1.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name1=s1['xaxisList']#季度

        
    except KeyError:
        continue
    
    
    ###########稅後淨利率#################
    url_twse2='https://mopsfin.twse.com.tw/compare/data?compareItem=NetIncomeMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res2=requests.get(url_twse2)


    s2=json.loads(res2.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name2=s2['xaxisList']#季度

        
    except KeyError:
        continue
    

      
    #若有資料再存stockid
    stock_id.append(now_stockid)#股票代碼存起來
    

    #個股資料 #ROE資料
    stockfiance_data=pd.DataFrame(s['graphData'][0]['data'])
 
    finance_data=pd.DataFrame(stockfiance_data[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data.index=fiance_name
    
    ROE_data.append(finance_data)
    
    
    #個股資料 #稅後淨利率
    stockfiance_data1=pd.DataFrame(s1['graphData'][0]['data'])
  
    finance_data1=pd.DataFrame(stockfiance_data1[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data1.index=fiance_name1
    
    operate_margin.append(finance_data1)#將ROE資料存起來
    ###############################
   ##############稅後淨利率############

    #個股資料
    stockfiance_data2=pd.DataFrame(s2['graphData'][0]['data'])
  
    finance_data2=pd.DataFrame(stockfiance_data2[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data2.index=fiance_name2
    
    Net_Income.append(finance_data2)#將ROE資料存起來

    

#########################到一段落####################

########合併資料####################

ROE_data2 = pd.DataFrame()
#FOR迴圈 將list合併成dataframe
for k in range(0,len(ROE_data)):
    a=ROE_data[k].T
       #ROE_data1=pd.concat([a], axis=1)
    ROE_data2=pd.concat([ROE_data2,a], axis=0)
     
#合併成final_ROE
ROE_data2.index=pd.DataFrame(stock_id).index
   
final_ROE_data=pd.concat([pd.DataFrame(stock_id),pd.DataFrame(ROE_data2)], axis=1)    
       
final_ROE_data.columns




operate_margin2 = pd.DataFrame()
#FOR迴圈 將list合併成dataframe
for k in range(0,len(operate_margin)):
    a=operate_margin[k].T
       #ROE_data1=pd.concat([a], axis=1)
    operate_margin2=pd.concat([operate_margin2,a], axis=0)
     


Net_Income2 = pd.DataFrame()
#FOR迴圈 將list合併成dataframe
for k in range(0,len(Net_Income)):
    a=Net_Income[k].T
       #ROE_data1=pd.concat([a], axis=1)
    Net_Income2=pd.concat([Net_Income2,a], axis=0)

       
     
#合併成final_operate_margin_data ##營業利益率
operate_margin2.index=pd.DataFrame(stock_id).index
   
final_operate_margin_data=pd.concat([pd.DataFrame(stock_id),pd.DataFrame(operate_margin2)], axis=1)    
       
final_operate_margin_data.columns

#合併成final_Net_Income_data ##稅後淨利率
Net_Income2.index=pd.DataFrame(stock_id).index
   
final_Net_Income_data=pd.concat([pd.DataFrame(stock_id),pd.DataFrame(Net_Income2)], axis=1)    
       
final_Net_Income_data.columns


#############儲存檔案##############
final_ROE_data.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_ROE_data.csv', index = False, encoding='utf_8_sig')#輸出excel檔



final_operate_margin_data.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_operate_margin_data.csv', index = False, encoding='utf_8_sig')#輸出excel檔


final_Net_Income_data.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_Net_Income_data.csv', index = False, encoding='utf_8_sig')#輸出excel檔




####################讀取檔案2023.2.12####################
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

#-------------------------------月營收爬蟲結束--------------------------


stock_data = pd.read_csv(r'D:\20200508桌面大檔\20200506pyhton股票投資\20200822每季財務比率投資策略\股票代碼.csv', encoding='utf_8_sig')
stockid=pd.DataFrame(stock_data['股票代碼'])



final_ROE_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_ROE_data.csv', encoding='utf_8_sig')#輸出excel檔



final_operate_margin_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_operate_margin_data.csv', encoding='utf_8_sig')#輸出excel檔


final_Net_Income_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_Net_Income_data.csv', encoding='utf_8_sig')#輸出excel檔


#讀取檔案

all_monthdata=[]
for n in range(0,39):
    a=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/allmonth_data/all_monthdata'+str(n)+'.csv', encoding='utf_8_sig')
    all_monthdata.append(a)#輸出excel檔
    
    #pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/allmonth_data/all_monthdata'+str(n)+'.csv', encoding='utf_8_sig')#輸出excel檔



#################讀取檔案結束2023.2.12###################


    

#final_ROE_data = pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/finalroe.csv', encoding='utf_8_sig')#輸出excel檔
#### 來合併表ROE data和月營收資料 和股價資料#################

final_data2=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/newfinal_data2.csv', encoding='utf_8_sig')

from datetime import datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta
final_ROE_data.rename(columns = {'0':'stockid'}, inplace = True)
final_operate_margin_data.rename(columns = {'0':'stockid'}, inplace = True)
final_Net_Income_data.rename(columns = {'0':'stockid'}, inplace = True)

month=['2013-05','2013-08','2013-11','2014-03','2014-05','2014-08','2014-11','2015-03','2015-05','2015-08','2015-11', '2016-03','2016-05','2016-08','2016-11', '2017-03','2017-05','2017-08','2017-11', '2018-03','2018-05','2018-08','2018-11', '2019-03','2019-05','2019-08','2019-11', '2020-03','2020-05','2020-08','2020-11', '2021-03','2021-05','2021-08','2021-11', '2022-03','2022-05','2022-08','2022-11']            


#跑資料先反白
final_data2 = pd.DataFrame()

#先跑700家 2023/2/19
for p in  range(0,2):#len(final_ROE_data)
    id=final_ROE_data['stockid'][p]
    print(p)
    
    for q in range(0,len(all_monthdata)):
        seasondata=all_monthdata[q]
              
        id_monthdata=seasondata[seasondata['公司代號']==str(id)]#找到月營收公司資料
        
        #測試dataframe是否為空 為空代表月營收找不到要跳過
        if(id_monthdata.empty==True):
            print('股票代號'+str(id)+'的第'+str(q)+'號無法取得')

            #舉例:
            #id_monthdata=seasondata[seasondata['公司代號']==1201]#找到月營收公司資料
            
            temp_month=month[q]
            temp_month1=pd.Series(temp_month)
            a=pd.DataFrame(pd.Series(np.nan))
            id1=pd.DataFrame(pd.Series(id))

            
            a2=pd.concat([pd.DataFrame(temp_month1),id1,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a], axis=1) #兩個表水平整合
            a2.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income','return_rate','買入價','賣出價']
                               
            final_data2=pd.concat([final_data2,a2], axis=0) 
            continue
        
        
        
        #id_monthdata.iloc[0,1]
        
        id_ROEdata=final_ROE_data.iloc[p,(q+1):(q+2)]
        id_ROEdata.index=id_monthdata.index
        
        #營業利益率#
        id_operate_margindata=final_operate_margin_data.iloc[p,(q+1):(q+2)]
        id_operate_margindata.index=id_monthdata.index

        #稅後純益率
        id_Net_Incomedata=final_Net_Income_data.iloc[p,(q+1):(q+2)]
        id_Net_Incomedata.index=id_monthdata.index

        
        
        
        #final_data=pd.concat([id_monthdata,pd.DataFrame(id_ROEdata)], axis=1) #兩個表水平整合
        #final_data.columns=['公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE']
        #垂直疊加
        #final_data2=pd.concat([final_data2,final_data], axis=0) 
       
        #加入當月股價 
        try:
            #month=['2013-05','2013-08','2013-11','2014-03','2014-05','2014-08','2014-11','2015-03','2015-05','2015-08','2015-11', '2016-03','2016-05','2016-08','2016-11', '2017-03','2017-05','2017-08','2017-11', '2018-03','2018-05','2018-08','2018-11', '2019-03','2019-05','2019-08','2019-11', '2020-03','2020-05','2020-08','2020-11', '2021-03','2021-05','2021-08','2021-11', '2022-03','2022-05','2022-08','2022-11']
            
            temp_month=month[q]
            
            temp_month=datetime.strptime(temp_month, '%Y-%m')
            temp_month=temp_month+ relativedelta(months=1)
            str_month=str(temp_month.year)+'-'+str(temp_month.month)
            
            
            data = yf.download(str(id)+".TW", start=str_month+'-10', end=str_month+'-30')
        
        
        
        except RemoteDataError:
            continue
        except IndexError:
            continue
       
        
       
        ########這邊還要加買賣價格
        realmonth=pd.Series(month[q])
        try:
            buyprice=data.iloc[0,3:4]
        except RemoteDataError:
            continue
        except IndexError:
            continue
        
        sellprice=data.iloc[(len(data)-1),3:4]
        return_rate=round(data.iloc[(len(data)-1),3:4]-data.iloc[0,3:4],2)/round(data.iloc[0,3:4],2)*100
        return_rate.index=id_monthdata.index
        buyprice.index=id_monthdata.index
        sellprice.index=id_monthdata.index
        realmonth.index=id_monthdata.index
        
        
        final_data=pd.concat([pd.DataFrame(realmonth),id_monthdata,pd.DataFrame(id_ROEdata),pd.DataFrame(id_operate_margindata),pd.DataFrame(id_Net_Incomedata),pd.DataFrame(return_rate),pd.DataFrame(buyprice),pd.DataFrame(sellprice)], axis=1) #兩個表水平整合
        final_data.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income','return_rate','買入價','賣出價']
        
        
        #垂直疊加
          
        final_data2=pd.concat([final_data2,final_data], axis=0) 
        
        time.sleep(0.5)





#暫存final_data2
final_data3=final_data2

#儲存起來
final_data2.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/newfinal_data2.csv', index = False, encoding='utf_8_sig')#輸出excel檔


final_data2=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/newfinal_data2.csv', encoding='utf_8_sig')

#final_data2=final_data22

#############最後再加入每一股票ROE NETINCOME OPERATE 較上一季增加率#########

#查詢na位置
import numpy as np

final_data2_change_type = np.array(final_data2['公司代號'], dtype=np.int)


#a11=np.where(np.isnan(a1) == True)


final_id=np.unique(final_data2_change_type)




#判斷是否有INF 有INF就無法跑
#a=np.isinf(x_train)

#a2=np.where(a == True,a,0)

#################從這#############

#a3=x_train[np.isinf(x_train)]= 0


#a=final_data2['公司代號']
#aa=np.where(np.isnan(a) == True)


###########查詢結束


#final_id=np.unique(final_data2['公司代號'])
import numpy as np


#分割公司名稱做增減

#month=['2013-05','2013-08','2013-11','2014-03','2014-05','2014-08','2014-11','2015-03','2015-05','2015-08','2015-11', '2016-03','2016-05','2016-08','2016-11', '2017-03','2017-05','2017-08','2017-11', '2018-03','2018-05','2018-08','2018-11', '2019-03','2019-05','2019-08','2019-11', '2020-03','2020-05','2020-08','2020-11', '2021-03','2021-05','2021-08','2021-11', '2022-03','2022-05','2022-08','2022-11']            
#ROE_rate=pd.DataFrame()
#NET_INCOME_rate=pd.DataFrame()
#OPERATE_MARGIN_rate=pd.DataFrame()

#############資料集準備##############
train_data=pd.DataFrame()
val_data=pd.DataFrame()
test_data=pd.DataFrame()

real_final_unique=pd.DataFrame()

#跑842家
for s in range(0,842):#len(final_ROE_data)=842家
    unique_data=final_data2[final_data2['公司代號']==final_id[s]]
    #若股票代碼為空值就跳過
    if(unique_data.empty==True):
        continue
    
    #每次一開始為空資料
    ROE_rate=pd.DataFrame()
    NET_INCOME_rate=pd.DataFrame()
    OPERATE_MARGIN_rate=pd.DataFrame()

    for w in range(1,len(unique_data)):#len(unique_data)
        print(w)
        
        #ROE
        temproe=pd.DataFrame(unique_data['ROE'])
       
        #兩種情況
        #情況一:前一個為0就會變成inf 因為/0 因此若為0就把temproe變成nan
        if(temproe.iloc[w-1,0]==0):
            tempROE=np.nan
            
            
        #情況二:目前為正且前一個為負，這樣成長率要改成正
        elif(temproe.iloc[w,0]>=0 and temproe.iloc[w-1,0]<0):
            tempROE=-round((temproe.iloc[w,0]-temproe.iloc[w-1,0])/temproe.iloc[w-1,0],2)*100
        else:
            tempROE_rate=round((temproe.iloc[w,0]-temproe.iloc[w-1,0])/temproe.iloc[w-1,0],2)*100
        
        
        #tempROE_rate=pd.DataFrame(tempROE_rate)
        
        tempROE_rate2=pd.DataFrame(pd.Series(tempROE_rate))
              
        ROE_rate=pd.concat([ROE_rate,tempROE_rate2], axis=0) 
        
        #NET INCOME
        tempnetincome=pd.DataFrame(unique_data['Net_Income'])
        
        #兩種情況
        #情況一:前一個為0就會變成inf 因為/0 因此若為0就把temproe變成nan
        if(tempnetincome.iloc[w-1,0]==0):
            tempnetincome=np.nan
            
        #情況二:目前為正且前一個為負，這樣成長率要改成正
        elif(tempnetincome.iloc[w,0]>=0 and tempnetincome.iloc[w-1,0]<0):
            tempnetincome=-round((tempnetincome.iloc[w,0]-tempnetincome.iloc[w-1,0])/tempnetincome.iloc[w-1,0],2)*100
        else:
            tempnetincome_rate=round((tempnetincome.iloc[w,0]-tempnetincome.iloc[w-1,0])/tempnetincome.iloc[w-1,0],2)*100
    
        
        
        
        
        #tempnetincome_rate=round((tempnetincome.iloc[w,0]-tempnetincome.iloc[w-1,0])/tempnetincome.iloc[w-1,0],2)*100
        
        tempnetincome_rate2=pd.DataFrame(pd.Series(tempnetincome_rate))
              
        NET_INCOME_rate=pd.concat([NET_INCOME_rate,tempnetincome_rate2], axis=0) 
        
        #OPERATE_MARGIN
        tempoperatemargin=pd.DataFrame(unique_data['operate_margin'])
        

        #兩種情況
        #情況一:前一個為0就會變成inf 因為/0 因此若為0就把temproe變成nan
        if(tempoperatemargin.iloc[w-1,0]==0):
            tempoperatemargin=np.nan
            
        #情況二:目前為正且前一個為負，這樣成長率要改成正
        elif(tempoperatemargin.iloc[w,0]>=0 and tempoperatemargin.iloc[w-1,0]<0):
            tempoperatemargin=-round((tempoperatemargin.iloc[w,0]-tempoperatemargin.iloc[w-1,0])/tempoperatemargin.iloc[w-1,0],2)*100
        else:
            tempoperatemargin_rate=round((tempoperatemargin.iloc[w,0]-tempoperatemargin.iloc[w-1,0])/tempoperatemargin.iloc[w-1,0],2)*100
    
        
        
        
        #tempoperatemargin_rate=round((tempoperatemargin.iloc[w,0]-tempoperatemargin.iloc[w-1,0])/tempoperatemargin.iloc[w-1,0],2)*100
        
        tempoperatemargin_rate2=pd.DataFrame(pd.Series(tempoperatemargin_rate))
              
        OPERATE_MARGIN_rate=pd.concat([OPERATE_MARGIN_rate,tempoperatemargin_rate2], axis=0) 
               
          
     
    #垂直整合    
    a=pd.DataFrame(pd.Series(np.nan))
    #第一個是NAN
    ROE_rate =pd.concat([a,ROE_rate], axis=0) 
    
    NET_INCOME_rate =pd.concat([a, NET_INCOME_rate], axis=0) 

    OPERATE_MARGIN_rate =pd.concat([a,OPERATE_MARGIN_rate], axis=0) 

  
    #水平合併#################這邊要看
    unique_data.index=ROE_rate.index
    unique_data =pd.concat([unique_data,ROE_rate,NET_INCOME_rate,OPERATE_MARGIN_rate ], axis=1)
  
    #挑出想要的選項
    final_unique=unique_data.iloc[:,[0,1,4,6,9,15,18,19,20]]
    
    final_unique.columns='月份', '公司代號', '上月比較增減(%)', '去年同月增減(%)', '前期比較增減(%)','return_rate','ROE_rate','NETINCOME_rate','OPERATE_MARGIN_rate'
    
    #刪除na
    final_unique=final_unique.dropna(axis=0,how='any') 
    
    #重設index
    final_unique=final_unique.reset_index(level=None, drop=False, inplace=False, col_level=0,col_fill='')
    
    #借用
    #final_unique=final_unique2
    
    final_unique=final_unique.drop(['index'], axis=1)
    
    
    #分train vlid 和 test  6:2:2 
    data_num = final_unique.shape[0]
   ## 取得一筆與資料數量相同的亂數索引，主要目的是用於打散資料,最大最小值固定
    indexes = np.random.permutation(data_num)
    
  # 並將亂數索引值分為Train、validation和test分為，這裡的劃分比例為6:2:2
    train_indexes = indexes[:int(data_num *0.6)]
    val_indexes = indexes[int(data_num *0.6):int(data_num *0.8)]
    test_indexes = indexes[int(data_num *0.8):]
    
   # 透過索引值從data取出訓練資料、驗證資料和測試資料
    t_train_data = final_unique.loc[train_indexes] 
    t_val_data = final_unique.loc[val_indexes]
    t_test_data = final_unique.loc[test_indexes]
    

    t_train_data=t_train_data.reset_index(level=None, drop=False, inplace=False, col_level=0,col_fill='')
    t_val_data=t_val_data.reset_index(level=None, drop=False, inplace=False, col_level=0,col_fill='')
    t_test_data=t_test_data.reset_index(level=None, drop=False, inplace=False, col_level=0,col_fill='')

    t_train_data=t_train_data.drop(['index'], axis=1)
    t_val_data=t_val_data.drop(['index'], axis=1)
    t_test_data=t_test_data.drop(['index'], axis=1)

    
#######################最後資料合併#############    
    #訓練集(垂直合併)  
    train_data =pd.concat([train_data,t_train_data], axis=0) 
    
    #train_data.iloc[1296,:]

   #驗證集
    val_data =pd.concat([val_data,t_val_data], axis=0) 
    
    #測試集
    test_data =pd.concat([test_data,t_test_data], axis=0) 
    
    #最後大資料
    real_final_unique=pd.concat([real_final_unique,final_unique], axis=0) 

    #最後再變更空資料
    #ROE_rate=pd.DataFrame()
    #NET_INCOME_rate=pd.DataFrame()
    #OPERATE_MARGIN_rate=pd.DataFrame()


##############開始跑深度學習###############
#先刪除掉price
x_train = np.array(train_data.drop(['月份','公司代號','return_rate'], axis='columns'))/100#以columns排序

#判斷x_train是否有INF 有INF就無法跑
#a=np.isinf(x_train)

#a2=np.where(a == True,a,0)

#把INF變成0 用這個
a3=x_train[np.isinf(x_train)]= 0

#########################找出inf位置
a=np.isinf(x_train)
aa=np.where(a == True)#找出inf位置


a1=a=np.isnan(x_train)
aa1=np.where(a1 == True)#找出na位置 沒有
##########到這停########

y_train = np.array(train_data['return_rate'])/100


x_val = np.array(val_data.drop(['月份','公司代號','return_rate'], axis='columns'))/100#以columns排序
#用這個 把INF變成0

#a=np.isinf(x_val)
#aa=np.where(a == True)#找出inf位置

#a3=x_val[np.isinf(x_val)]= 0


y_val = np.array(val_data['return_rate'])/100


x_train.shape


#########建立並訓練網路模型#############
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# 建立一個Sequential型態的model
model = keras.Sequential(name='model-1')
# 第1層全連接層設為64個unit，將輸入形狀設定為(6, ) 代表6個欄位，而實際上我們輸入的數據形狀為(batch_size, 21)
model.add(layers.Dense(16, 
                       kernel_regularizer=keras.regularizers.l2(0.001), 
                       activation='relu', input_shape=(6,)))

model.add(layers.Dropout(0.3))

# 第2層全連接層設為64個unit
model.add(layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))

model.add(layers.Dropout(0.3))
# 最後一層全連接層設為1個unit
model.add(layers.Dense(1))
# 顯示網路模型架構
model.summary()

#設定訓練使用的優化器、損失函數和指標函數：
model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanAbsoluteError()])


#創建模型儲存目錄：
#在C:/Users/User/lab9-logs/models/建立模型目錄
model_dir = 'lab9-logs/models/'
#os.makedirs(model_dir)

#設定回調函數：
# TensorBoard回調函數會幫忙紀錄訓練資訊，並存成TensorBoard的紀錄檔
log_dir = os.path.join('lab9-logs', 'model-1')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
#將模型儲存在C:/Users/User/lab9-logs/models/
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')





#訓練網路模型：
history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=16,  # 批次大小設為16
               epochs=300,  # 整個dataset訓練300遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型


#訓練結果
history.history.keys()  # 查看history儲存的資訊有哪些

#在model.compile已經將損失函數設為均方誤差(Mean Square Error)
#所以history紀錄的loss和val_loss為Mean Squraed Error損失函數計算的損失值
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
#plt.ylim(0.02, 0.2)
plt.title('Mean square error')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')



#測試數據的誤差百分比：用測試數據預測房屋價格並與答案計算誤差百分比。
# 載入模型
model = keras.models.load_model('lab9-logs/models/Best-model-1.h5')
# 先將房屋價格取出
y_test = np.array(test_data['return_rate'])/100
# 標準化數據
#test_data = (test_data - mean) / std
# 將輸入數據存成Numpy 格式
x_test = np.array(test_data.drop(['月份','公司代號','return_rate'], axis='columns'))/100

# 預測測試數據
y_pred = model.predict(x_test)

# 將預測結果轉換回來(因為訓練時的訓練目標也有經過標準化)
#y_pred = np.reshape(y_pred * std['price'] + mean['price'], y_test.shape)
# 計算平均的誤差百分比
percentage_error = np.mean(np.abs(y_test - y_pred)) / np.mean(y_test) * 100
# 顯示誤差百分比 顯示到小數點第二位 

#注意:如果餵入資料為inf出來就是nan
print("Model_1 Percentage Error: {:.2f}%".format(percentage_error))

#因為validation的loss一直沒調降因此有過度擬合問題
#顯示準確度 

correct_rate=[]
for i in range(0,len(y_pred)):
    if ((y_pred[i]>0 and y_test[i]>0) or (y_pred[i]<0 and y_test[i]<0)):
        correct_rate.append(1)
    else:
        correct_rate.append(0)
        
        

        

        
correct_rate2=sum(correct_rate)/len(correct_rate)


print('準確率為 {:.2f}%'.format(correct_rate2*100))







###################訓練完成2023.2.13決數###############









# 原始資料
b = ["A", "B", "A", "C", "B", "C", "A", "A"]
numpy.unique(b)







final_data2.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/allmonth_data/final_data2.csv', index = False, encoding='utf_8_sig')#輸出excel檔


#最終資料index加上去
final_data2.index=month[0:39]*len(final_ROE_data)







##############################20230206結束####################
   

#############最後再加入每一股票ROE NETINCOME OPERATE 較上一季增加率#########
# 原始資料
b = ["A", "B", "A", "C", "B", "C", "A", "A"]
numpy.unique(b)
