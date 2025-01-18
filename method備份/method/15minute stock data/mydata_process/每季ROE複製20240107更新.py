

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:02:59 2020

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
from numpy import median
import requests
from io import StringIO
import time


#month data  web crawler
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


final_data_real123= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/0219最新擴充資料集bigfinal_data__0到414.csv')
unique_stockid=pd.DataFrame(np.unique(final_data_real123['公司代號']))##找414支股票



#quarter data  web crawler

#股票代碼
stock_data = pd.read_csv(r'D:\20200508桌面大檔\20200506pyhton股票投資\20200822每季財務比率投資策略\股票代碼.csv', encoding='utf_8_sig')
stockid=pd.DataFrame(stock_data['股票代碼'])
stock_id=[]
ROE_data=[]
operate_margin=[]#營業利益率
Net_Income=[]#稅後淨利率
ROA=[]#資產報酬率
GrossMargin=[]#毛利率

####經營能力指標#########
AccountsReceivableTurnover=[]  #應收款項週轉率

AccountsReceivableTurnoverDay=[] #平均收現日數

InventoryTurnover=[] #存貨周轉率

InventoryTurnoverDay=[]#平均銷貨天數

TotalAssetTurnover=[] #總資產週轉率

###償債能力##########
CurrentRatio=[] #流動比率
QuickRatio=[] #速動比率

InterestCoverage=[]  #利息保障倍數

#####財務結構#######
DebtRatio=[] #負債佔資產比率


LongTermLiabilitiesRatio=[]   #長期資金佔不動產


###現金流量#########
OperatingCashflowToCurrentLiability=[] #營業現金對流動負債比

OperatingCashflowToLiability=[] #營業現金對負債比####

OperatingCashflowToNetProfit=[]  #營業現金對稅後純益比


###########成長能力##############
RevenueYOY=[]  #營業收入年增率

GrossProfitYOY=[] #營業毛利年增率

OperatingIncomeYOY=[] #營業利益年增率

NetProfitYOY=[] #稅後純益年增率

EPSYOY=[] #每股盈餘年增率


for i in range(0,len(unique_stockid)):
    print(i)

    now_stockid=str(unique_stockid.iloc[i,0])
    
    #1.ROE
    url_twse='https://mopsfin.twse.com.tw/compare/data?compareItem=ROE&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid

    res=requests.get(url_twse)
    time.sleep(0.5)
    s=json.loads(res.text)#將資料轉成json格式
    try:
        fiance_name=s['xaxisList']#季度
    except KeyError:
        continue
    
    
    #2.營業利益率
    url_twse1='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid

    res1=requests.get(url_twse1)
    time.sleep(0.5)
    s1=json.loads(res1.text)

    try:
        fiance_name1=s1['xaxisList']
    except KeyError:
        continue
    
    
    #3.稅後淨利率
    url_twse2='https://mopsfin.twse.com.tw/compare/data?compareItem=NetIncomeMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res2=requests.get(url_twse2)
    s2=json.loads(res2.text)

    try:
        fiance_name2=s2['xaxisList']#季度
    except KeyError:
        continue
    
    
    #4.ROA
    url_twse3='https://mopsfin.twse.com.tw/compare/data?compareItem=ROA&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res3=requests.get(url_twse3)
    s3=json.loads(res3.text)
  
    try:
        fiance_name3=s3['xaxisList']#季度

        
    except KeyError:
        continue
    
    ######5.毛利率GrossMargin###############
    url_twse4='https://mopsfin.twse.com.tw/compare/data?compareItem=GrossMargin&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res4=requests.get(url_twse4)


    s4=json.loads(res4.text)#將資料轉成json格式
  
    try:
        fiance_name4=s4['xaxisList']#季度

        
    except KeyError:
        continue
    
     #經營能力指標開始
    
    #5.應收帳款周轉率 AccountsReceivableTurnover
    url_twse5='https://mopsfin.twse.com.tw/compare/data?compareItem=AccountsReceivableTurnover&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res5=requests.get(url_twse5)
    s5=json.loads(res5.text)

    try:
        fiance_name5=s5['xaxisList']#季度
        
    except KeyError:
        continue
    
    
    #6.AccountsReceivableTurnoverDay=[] #平均收現日數
    url_twse6='https://mopsfin.twse.com.tw/compare/data?compareItem=AccountsReceivableTurnoverDay&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res6=requests.get(url_twse6)

    s6=json.loads(res6.text)
    try:
        fiance_name6=s6['xaxisList']#季度        
    except KeyError:
        continue
    
    
    ######7.InventoryTurnover  ####I存貨周轉率
    url_twse7='https://mopsfin.twse.com.tw/compare/data?compareItem=InventoryTurnover&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res7=requests.get(url_twse7)

    s7=json.loads(res7.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name7=s7['xaxisList']#季度        
    except KeyError:
        continue
    
    
    ###8.InventoryTurnoverDay=[]#平均銷貨天數########
    url_twse8='https://mopsfin.twse.com.tw/compare/data?compareItem=InventoryTurnoverDay&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res8=requests.get(url_twse8)

    s8=json.loads(res8.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name8=s8['xaxisList']#季度        
    except KeyError:
        continue    
    
    
    ###9.TotalAssetTurnover=[] #總資產週轉率########
    url_twse9='https://mopsfin.twse.com.tw/compare/data?compareItem=TotalAssetTurnover&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res9=requests.get(url_twse9)

    s9=json.loads(res9.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name9=s9['xaxisList']#季度        
    except KeyError:
        continue        
    
     ###10.CurrentRatio=[] #流動比率########
    url_twse10='https://mopsfin.twse.com.tw/compare/data?compareItem=CurrentRatio&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res10=requests.get(url_twse10)

    s10=json.loads(res10.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name10=s10['xaxisList']#季度        
    except KeyError:
        continue     
    
    
     ###11.QuickRatio=[] #速動比率########
    url_twse11='https://mopsfin.twse.com.tw/compare/data?compareItem=QuickRatio&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res11=requests.get(url_twse11)

    s11=json.loads(res11.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name11=s11['xaxisList']#季度        
    except KeyError:
        continue  

    #12.InterestCoverage=[]利息保障倍數
    url_twse12='https://mopsfin.twse.com.tw/compare/data?compareItem=InterestCoverage&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res12=requests.get(url_twse12)

    s12=json.loads(res12.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name12=s12['xaxisList']#季度        
    except KeyError:
        continue  
    
    ###13.DebtRatio=[] 負債佔資產比率
    url_twse13='https://mopsfin.twse.com.tw/compare/data?compareItem=DebtRatio&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res13=requests.get(url_twse13)

    s13=json.loads(res13.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name13=s13['xaxisList']#季度        
    except KeyError:
        continue      


    ###14.LongTermLiabilitiesRatio=[]   #長期資金佔不動產
    url_twse14='https://mopsfin.twse.com.tw/compare/data?compareItem=LongTermLiabilitiesRatio&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res14=requests.get(url_twse14)

    s14=json.loads(res14.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name14=s14['xaxisList']#季度        
    except KeyError:
        continue   
    
    
    #######現金流量指標###########
    
    ##15.OperatingCashflowToCurrentLiability=[] #營業現金對流動負債比###########
    url_twse15='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingCashflowToCurrentLiability&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res15=requests.get(url_twse15)

    s15=json.loads(res15.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name15=s15['xaxisList']#季度        
    except KeyError:
        continue      


    ###16. OperatingCashflowToLiability=[] #營業現金對負債比####
    url_twse16='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingCashflowToLiability&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res16=requests.get(url_twse16)

    s16=json.loads(res16.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name16=s16['xaxisList']#季度        
    except KeyError:
        continue     
    
    
    #17. OperatingCashflowToNetProfit=[]  #營業現金對稅後純益比
    url_twse17='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingCashflowToNetProfit&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res17=requests.get(url_twse17)

    s17=json.loads(res17.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name17=s17['xaxisList']#季度        
    except KeyError:
        continue        
    
    
    #18.RevenueYOY=[]  #營業收入年增率
    url_twse18='https://mopsfin.twse.com.tw/compare/data?compareItem=RevenueYOY&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res18=requests.get(url_twse18)

    s18=json.loads(res18.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name18=s18['xaxisList']#季度        
    except KeyError:
        continue         
    
    ###19.GrossProfitYOY=[] #營業毛利年增率########
    url_twse19='https://mopsfin.twse.com.tw/compare/data?compareItem=GrossProfitYOY&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res19=requests.get(url_twse19)

    s19=json.loads(res19.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name19=s19['xaxisList']#季度        
    except KeyError:
        continue     
    
    
    #20.OperatingIncomeYOY=[] #營業利益年增率
    url_twse20='https://mopsfin.twse.com.tw/compare/data?compareItem=OperatingIncomeYOY&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res20=requests.get(url_twse20)

    s20=json.loads(res20.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name20=s20['xaxisList']#季度        
    except KeyError:
        continue    
   ###21.NetProfitYOY=[] #稅後純益年增率#########
    url_twse21='https://mopsfin.twse.com.tw/compare/data?compareItem=NetProfitYOY&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res21=requests.get(url_twse21)

    s21=json.loads(res21.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name21=s21['xaxisList']#季度        
    except KeyError:
        continue 
    
    
    ##22.EPSYOY=[] #每股盈餘年增率###
    url_twse22='https://mopsfin.twse.com.tw/compare/data?compareItem=EPSYOY&quarter=true&ylabel=%25&ys=0&revenue=true&bcodeAvg=true&companyAvg=true&qnumber=&companyId='+now_stockid
    time.sleep(0.5)
    res22=requests.get(url_twse22)

    s22=json.loads(res22.text)#將資料轉成json格式
  
    #將資料合併成dataframe
    
    #若爬不到財報資料就不會顯示xaxisList欄位
    try:
        fiance_name22=s22['xaxisList']#季度        
    except KeyError:
        continue 
    
      
###################開始將資料存起來######################

    #若有資料再存stockid
    #stock_id.append(now_stockid)#股票代碼存起來
    

    #個股資料 #ROE資料
    stockfiance_data=pd.DataFrame(s['graphData'][0]['data'])
 
    finance_data=pd.DataFrame(stockfiance_data[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data.index=fiance_name
    
    #如果沒有43筆ROEDATA就跳過
    if(len(finance_data)!=44):   ##原本是43 但3月份有第四季
      print("跳過")
      continue
    
    
    ROE_data.append(finance_data)#將ROE資料存起來
    
    #若有資料再存stockid
    stock_id.append(now_stockid)#股票代碼存起來
    
    #營業利益率
    stockfiance_data1=pd.DataFrame(s1['graphData'][0]['data'])
  
    finance_data1=pd.DataFrame(stockfiance_data1[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data1.index=fiance_name1
    
    operate_margin.append(finance_data1)#將營業利益率資料存起來
    
   
     ###稅後淨利率############

    #個股資料
    stockfiance_data2=pd.DataFrame(s2['graphData'][0]['data'])
  
    finance_data2=pd.DataFrame(stockfiance_data2[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data2.index=fiance_name2
    
    Net_Income.append(finance_data2)#將ROE資料存起來
    
    ####資產報酬率 ROA########
    stockfiance_data3=pd.DataFrame(s3['graphData'][0]['data'])
  
    finance_data3=pd.DataFrame(stockfiance_data3[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data3.index=fiance_name3
    
    ROA.append(finance_data3)#將ROE資料存起來  
    
    ####毛利率 GrossMargin###########
    stockfiance_data4=pd.DataFrame(s4['graphData'][0]['data'])
  
    finance_data4=pd.DataFrame(stockfiance_data4[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data4.index=fiance_name4
    
    GrossMargin.append(finance_data4)#將ROE資料存起來  
    
    ##################獲利指標結束#################
    
    ############經營能力指標開始######################
    
    ##5.應收帳款周轉率 AccountsReceivableTurnover
    stockfiance_data5=pd.DataFrame(s5['graphData'][0]['data'])
  
    finance_data5=pd.DataFrame(stockfiance_data5[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data5.index=fiance_name5
    
    AccountsReceivableTurnover.append(finance_data5)#將ROE資料存起來  
    
    
   ######6.AccountsReceivableTurnoverDay=[] #平均收現日數######
    stockfiance_data6=pd.DataFrame(s6['graphData'][0]['data'])
  
    finance_data6=pd.DataFrame(stockfiance_data6[1])#finance_data=pd.DataFrame(stockfiance_data['個股ROE'])
        
    finance_data6.index=fiance_name6
    
    AccountsReceivableTurnoverDay.append(finance_data6)#將AccountsReceivableTurnoverDay存起來 
    
    
   ######7.InventoryTurnover  ####I存貨周轉率
    stockfiance_data7=pd.DataFrame(s7['graphData'][0]['data'])
  
    finance_data7=pd.DataFrame(stockfiance_data7[1])
    finance_data7.index=fiance_name7
    
    InventoryTurnover.append(finance_data7)
       
    ###8.InventoryTurnoverDay#平均銷貨天數########
    stockfiance_data8=pd.DataFrame(s8['graphData'][0]['data'])
  
    finance_data8=pd.DataFrame(stockfiance_data8[1])
    finance_data8.index=fiance_name8
    
    InventoryTurnoverDay.append(finance_data8)
    
    ###9.TotalAssetTurnover=[] #總資產週轉率########
    stockfiance_data9=pd.DataFrame(s9['graphData'][0]['data'])
  
    finance_data9=pd.DataFrame(stockfiance_data9[1])
    finance_data9.index=fiance_name9
    
    TotalAssetTurnover.append(finance_data9)
    
    
   ######10.CurrentRatio 流動比率##########
    stockfiance_data10=pd.DataFrame(s10['graphData'][0]['data'])
  
    finance_data10=pd.DataFrame(stockfiance_data10[1])
    finance_data10.index=fiance_name10
    
    CurrentRatio.append(finance_data10)  
    
    ######11.QuickRatio 速動比率##########
    stockfiance_data11=pd.DataFrame(s11['graphData'][0]['data'])
  
    finance_data11=pd.DataFrame(stockfiance_data11[1])
    finance_data11.index=fiance_name11
    
    QuickRatio.append(finance_data11)  
    
    #12.InterestCoverage=[]利息保障倍數
    stockfiance_data12=pd.DataFrame(s12['graphData'][0]['data'])
  
    finance_data12=pd.DataFrame(stockfiance_data12[1])
    finance_data12.index=fiance_name12
    
    InterestCoverage.append(finance_data12)      
    
    #13.DebtRatio=[] #負債佔資產比率
    stockfiance_data13=pd.DataFrame(s13['graphData'][0]['data'])
  
    finance_data13=pd.DataFrame(stockfiance_data13[1])
    finance_data13.index=fiance_name13
    
    DebtRatio.append(finance_data13)  

    #14.LongTermLiabilitiesRatio=[]   #長期資金佔不動產
    stockfiance_data14=pd.DataFrame(s14['graphData'][0]['data'])
  
    finance_data14=pd.DataFrame(stockfiance_data14[1])
    finance_data14.index=fiance_name14
    
    LongTermLiabilitiesRatio.append(finance_data14)   
    
    #######現金流量###############
    
    #15.OperatingCashflowToCurrentLiability=[] #營業現金對流動負債比
    stockfiance_data15=pd.DataFrame(s15['graphData'][0]['data'])
  
    finance_data15=pd.DataFrame(stockfiance_data15[1])
    finance_data15.index=fiance_name15
    
    OperatingCashflowToCurrentLiability.append(finance_data15)    
    
    #16.OperatingCashflowToLiability=[] #應業現金對負債比########
    stockfiance_data16=pd.DataFrame(s16['graphData'][0]['data'])
  
    finance_data16=pd.DataFrame(stockfiance_data16[1])
    finance_data16.index=fiance_name16
    
    OperatingCashflowToLiability.append(finance_data16)


    #17. OperatingCashflowToNetProfit=[]  #營業現金對稅後純益比    
    stockfiance_data17=pd.DataFrame(s17['graphData'][0]['data'])
  
    finance_data17=pd.DataFrame(stockfiance_data17[1])
    finance_data17.index=fiance_name17
    
    OperatingCashflowToNetProfit.append(finance_data17)
    
    #######成長能力################
    
   ###18.RevenueYOY=[]  #營業收入年增率
    stockfiance_data18=pd.DataFrame(s18['graphData'][0]['data'])
  
    finance_data18=pd.DataFrame(stockfiance_data18[1])
    finance_data18.index=fiance_name18
    
    RevenueYOY.append(finance_data18)   
    
    ###19.GrossProfitYOY=[] #營業毛利年增率########
    stockfiance_data19=pd.DataFrame(s19['graphData'][0]['data'])
  
    finance_data19=pd.DataFrame(stockfiance_data19[1])
    finance_data19.index=fiance_name19
    
    GrossProfitYOY.append(finance_data19)     
    
    #20.OperatingIncomeYOY=[] #營業利益年增率
    stockfiance_data20=pd.DataFrame(s20['graphData'][0]['data'])
  
    finance_data20=pd.DataFrame(stockfiance_data20[1])
    finance_data20.index=fiance_name20
    
    OperatingIncomeYOY.append(finance_data20)   
    
    ###21.NetProfitYOY=[] #稅後純益年增率#########
    stockfiance_data21=pd.DataFrame(s21['graphData'][0]['data'])
  
    finance_data21=pd.DataFrame(stockfiance_data21[1])
    finance_data21.index=fiance_name21
    
    NetProfitYOY.append(finance_data21) 
    
    ##   22.EPSYOY=[] #每股盈餘年增率###
    stockfiance_data22=pd.DataFrame(s22['graphData'][0]['data'])
  
    finance_data22=pd.DataFrame(stockfiance_data22[1])
    finance_data22.index=fiance_name22
    
    EPSYOY.append(finance_data22) 
    
        
#########################到一段落11.12####################




########合併資料####################
###############用一個大迴圈包所有22個財務比率#####################
finance_ratio=[ROE_data,operate_margin,Net_Income,ROA,GrossMargin,AccountsReceivableTurnover,AccountsReceivableTurnoverDay,InventoryTurnover,InventoryTurnoverDay,TotalAssetTurnover,CurrentRatio,QuickRatio,InterestCoverage,DebtRatio,LongTermLiabilitiesRatio,OperatingCashflowToCurrentLiability,OperatingCashflowToLiability,OperatingCashflowToNetProfit,
GrossProfitYOY,OperatingIncomeYOY,NetProfitYOY,EPSYOY]
final_ratio=[]#最後比率


for m in range(0,len(finance_ratio)):
    print(m)
    temp_ratio = pd.DataFrame()
    ratio=finance_ratio[m] #第一個比率
#FOR迴圈 將list合併成dataframe
    for n in range(0,len(ratio)):
        temp=ratio[n].T
       #ROE_data1=pd.concat([a], axis=1)
        temp_ratio=pd.concat([temp_ratio,temp], axis=0)
     
    #合併成final_ROE  這裡匯出現問題
    #temp_ratio.index=pd.DataFrame(stock_id).index[0:49]
    temp_ratio.index=pd.DataFrame(stock_id).index
    
    ##final_ROE_data=pd.concat([pd.DataFrame(stock_id),pd.DataFrame(temp_ratio)], axis=1)    
    #final_ROE_data換inance_ratio[m]
    #data_merge=pd.concat([pd.DataFrame(stock_id[0:49]),pd.DataFrame(temp_ratio)], axis=1)
       
    data_merge=pd.concat([pd.DataFrame(stock_id),pd.DataFrame(temp_ratio)], axis=1)

    final_ratio.append(data_merge)
  
    final_ratiocopy=final_ratio #備份
  
##########################11.19月營收部分###################
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
##############################開始爬月營收########################
stock_data = pd.read_csv(r'D:\20200508桌面大檔\20200506pyhton股票投資\20200822每季財務比率投資策略\股票代碼.csv', encoding='utf_8_sig')
stockid=pd.DataFrame(stock_data['股票代碼'])


##final_ROE_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_ROE_data.csv', encoding='utf_8_sig')#輸出excel檔
#final_operate_margin_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_operate_margin_data.csv', encoding='utf_8_sig')#輸出excel檔
#final_Net_Income_data=pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/final_Net_Income_data.csv', encoding='utf_8_sig')#輸出excel檔





#final_data2.to_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/newfinal_data2.csv', index = False, encoding='utf_8_sig')#輸出excel檔

#爬月營收
month=['2013-05','2013-08','2013-11','2014-03','2014-05','2014-08','2014-11','2015-03','2015-05','2015-08','2015-11', '2016-03','2016-05','2016-08','2016-11', '2017-03','2017-05','2017-08','2017-11', '2018-03','2018-05','2018-08','2018-11', '2019-03','2019-05','2019-08','2019-11', '2020-03','2020-05','2020-08','2020-11', '2021-03','2021-05','2021-08','2021-11', '2022-03','2022-05','2022-08','2022-11', '2023-03','2023-05','2023-08','2023-11','2024-03']            


#################4月份要補2024-3  #######載入這邊先封起來#################


##########寫入資料######################
#for p in range(43,44):  ##如果4月10號就可以找全部資料庫  變成44
    #print(p)
    #now_year=int(month[p][0:4])-1911 #轉成民國
    #now_month=int(month[p][5:7])
    ##爬month資料
    #monthdata=monthly_report(now_year,now_month)
    ##儲存月資料 
    #monthdata.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/new_allmonthdata/all_monthdata'+str(p)+'.csv', index = False, encoding='utf_8_sig')#輸出excel檔
     # 偽停頓
    #time.sleep(0.5)
    
    
    
 ###讀取月營收########   

#讀取月營收 讀44
all_monthdata=[]
for n in range(0,44): 
    a=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/new_allmonthdata/all_monthdata'+str(n)+'.csv', encoding='utf_8_sig')
    all_monthdata.append(a)
    
    #pd.read_csv(r'D:/20200508桌面大檔/20200506pyhton股票投資/20230203想做的結合ROE用一季的去做tensorflow/allmonth_data/all_monthdata'+str(n)+'.csv', encoding='utf_8_sig')#輸出excel檔




#### 來合併財報和月營收資料和股價資料#################
from datetime import datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta



########之前做三個比率先反白###############
#final_ROE_data.rename(columns = {'0':'stockid'}, inplace = True)
#final_operate_margin_data.rename(columns = {'0':'stockid'}, inplace = True)
#final_Net_Income_data.rename(columns = {'0':'stockid'}, inplace = True)


#final_data2為最後資料檔  執行過一次就先不執行
final_data2 = pd.DataFrame()



#先跑700家 2023/2/19
for p in  range(0, len(final_ratio[0])): #先跑20家  len(final_ratio[0])
    #先把財務指標上的0換成stockid,所以我這邊就挑0
    choose_ratio=final_ratio[0] 
    choose_ratio.rename(columns = {0:'stockid'}, inplace = True)

    #挑第幾家
    id=choose_ratio['stockid'][p]
    #id=final_ROE_data['stockid'][p]


    print("****************第"+ str(p) + "隻股票****************")
    
    for q in range(0,len(all_monthdata)):  #先讀42  12.10號後
        #搜尋相同日期財務資料
        
        ########月營收資料
        seasondata=all_monthdata[q]
              
        id_monthdata=seasondata[seasondata['公司代號']==str(id)]#找到月營收公司資料
        
        #上月比較增減(%)
        monthYOY=id_monthdata['上月比較增減(%)']
        #去年同月增減(%)
        yearYOY=id_monthdata['去年同月增減(%)']
        #前期比較增減(%)
        periodYOY=id_monthdata['前期比較增減(%)']
        
        
        
        #測試dataframe是否為空 為空代表月營收找不到要跳過
        if(id_monthdata.empty==True):
            print('股票代號'+str(id)+'的第'+str(q)+'號無法取得所以跳過')
            continue
            
            print('股票代號'+str(id)+'的第'+str(q)+'號無法取得')

            #舉例:
            #id_monthdata=seasondata[seasondata['公司代號']==1201]#找到月營收公司資料
            
            temp_month=month[q]
            temp_month1=pd.Series(temp_month)
            a=pd.DataFrame(pd.Series(np.nan))
            id1=pd.DataFrame(pd.Series(id))

            #########這邊要修改##############
            #a2=pd.concat([pd.DataFrame(temp_month1),id1,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a], axis=1) #兩個表水平整合
            #a2.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income','return_rate','買入價','賣出價']
            a2=pd.concat([pd.DataFrame(temp_month1),id1,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a], axis=1) #兩個表水平整合             
            a2.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income',
'資產報酬率','毛利率','應收款項週轉率','平均收現日數','存貨周轉率','平均銷貨天數','總資產週轉率',
'流動比率','速動比率','利息保障倍數','負債佔資產比率','長期資金佔不動產','營業現金對流動負債比',
'營業現金對負債比','營業現金對稅後純益比','return_rate','買入價','賣出價']
            
            ##########成長能力4個指標拿掉#################
            #'營業收入年增率','營業毛利年增率','營業利益年增率','稅後純益年增率'
            
            
            final_data2=pd.concat([final_data2,a2], axis=0) 
            #continue
        
           #成長能力五個指標拿掉因為從2014Q1開始
        
        ##1.ROE
        id_ROEdata=final_ratio[0].iloc[p,(q+1):(q+2)]
        id_ROEdata.index=id_monthdata.index
        
        #2.營業利益率operate_margin
        id_operate_margindata=final_ratio[1].iloc[p,(q+1):(q+2)]
        id_operate_margindata.index=id_monthdata.index

        #3稅後純益率
        id_Net_Incomedata=final_ratio[2].iloc[p,(q+1):(q+2)]
        id_Net_Incomedata.index=id_monthdata.index
        #4.ROA
        id_ROA=final_ratio[3].iloc[p,(q+1):(q+2)]
        id_ROA.index=id_monthdata.index
        #5.GrossMargin
        id_GrossMargin=final_ratio[4].iloc[p,(q+1):(q+2)]
        id_GrossMargin.index=id_monthdata.index
        #6.AccountsReceivableTurnover
        id_AccountsReceivableTurnover=final_ratio[5].iloc[p,(q+1):(q+2)]
        id_AccountsReceivableTurnover.index=id_monthdata.index       
        #7.AccountsReceivableTurnoverDay
        id_AccountsReceivableTurnoverDay=final_ratio[6].iloc[p,(q+1):(q+2)]
        id_AccountsReceivableTurnoverDay.index=id_monthdata.index
        #8.InventoryTurnover
        id_InventoryTurnover=final_ratio[7].iloc[p,(q+1):(q+2)]
        id_InventoryTurnover.index=id_monthdata.index 
        #9.InventoryTurnoverDay
        id_InventoryTurnoverDay=final_ratio[8].iloc[p,(q+1):(q+2)]
        id_InventoryTurnoverDay.index=id_monthdata.index  
        #10.TotalAssetTurnover=[] #總資產週轉率
        id_TotalAssetTurnover=final_ratio[9].iloc[p,(q+1):(q+2)]
        id_TotalAssetTurnover.index=id_monthdata.index          
        #11.CurrentRatio=[] #總資產週轉率
        id_CurrentRatio=final_ratio[10].iloc[p,(q+1):(q+2)]
        id_CurrentRatio.index=id_monthdata.index          
        
        #12.QuickRatio=[] #速動比率
        id_QuickRatio=final_ratio[11].iloc[p,(q+1):(q+2)]
        id_QuickRatio.index=id_monthdata.index    
        #13.InterestCoverage=[]  #利息保障倍數
        id_InterestCoverage=final_ratio[12].iloc[p,(q+1):(q+2)]
        id_InterestCoverage.index=id_monthdata.index            
        #14.DebtRatio=[] #負債佔資產比率
        id_DebtRatio=final_ratio[13].iloc[p,(q+1):(q+2)]
        id_DebtRatio.index=id_monthdata.index   
        #15.LongTermLiabilitiesRatio
        id_LongTermLiabilitiesRatio=final_ratio[14].iloc[p,(q+1):(q+2)]
        id_LongTermLiabilitiesRatio.index=id_monthdata.index 
        #16.  OperatingCashflowToCurrentLiability=[] #營業現金對流動負債比      
        id_OperatingCashflowToCurrentLiability=final_ratio[15].iloc[p,(q+1):(q+2)]
        id_OperatingCashflowToCurrentLiability.index=id_monthdata.index 
        
        #17.OperatingCashflowToLiability=[] #營業現金對負債比####
        id_OperatingCashflowToLiability=final_ratio[16].iloc[p,(q+1):(q+2)]
        id_OperatingCashflowToLiability.index=id_monthdata.index       
        
        #18.OperatingCashflowToNetProfit=[]  #營業現金對稅後純益比
        id_OperatingCashflowToNetProfit=final_ratio[17].iloc[p,(q+1):(q+2)]
        id_OperatingCashflowToNetProfit.index=id_monthdata.index 
        
        
        ###########成長能力5個指標先不要因為是從2014Q1開始不適2013Q1##########
        
        #18.RevenueYOY=[]  #營業收入年增率
        #id_RevenueYOY=final_ratio[17].iloc[p,(q+1):(q+2)]
        #id_RevenueYOY.index=id_monthdata.index 
        #19.GrossProfitYOY=[] #營業毛利年增率
        #id_GrossProfitYOY=final_ratio[18].iloc[p,(q+1):(q+2)]
        #id_GrossProfitYOY.index=id_monthdata.index     
        #20.OperatingIncomeYOY=[] #營業利益年增率
        #id_OperatingIncomeYOY=final_ratio[19].iloc[p,(q+1):(q+2)]
        #id_OperatingIncomeYOY.index=id_monthdata.index     
        #21.NetProfitYOY=[] #稅後純益年增率
        #id_NetProfitYOY=final_ratio[20].iloc[p,(q+1):(q+2)]
        #id_NetProfitYOY.index=id_monthdata.index   
        #22.EPSYOY=[] #每股盈餘年增率
        #id_EPSYOY=final_ratio[21].iloc[p,(q+1):(q+2)]
        #id_EPSYOY.index=id_monthdata.index   


        


        
        #id_monthdata.iloc[0,1]
        
        ##########################11.19做到這#############
        
        #財務比率占存在final_tempratio
        #final_tempratio=final_ratio[p].iloc[p,(q+1):(q+2)]
        
        #加入當月股價 
        try:
            #month=['2013-04','2013-07','2013-10','2014-02','2014-04','2014-07','2014-10','2015-02','2015-04','2015-07','2015-10', '2016-02','2016-04','2016-07','2016-10', '2017-02','2017-04','2017-07','2017-10', '2018-02','2018-04','2018-07','2018-10', '2019-02','2019-04','2019-07','2019-10', '2020-02','2020-04','2020-07','2020-10', '2021-02','2021-04','2021-07','2021-10', '2022-03','2022-04','2022-07','2022-10', '2023-02','2023-04','2023-07','2023-10']            
            
            temp_month=month[q]
            
            temp_month=datetime.strptime(temp_month, '%Y-%m')
            
            ###看到月財報當月10號持有到月底#######
            #temp_month=temp_month+ relativedelta(months=1)看到月報10號持有到下月10號
            str_month=str(temp_month.year)+'-'+str(temp_month.month+1)
            #end_month=str(temp_month.year)+'-'+str(temp_month.month+1)
            #如果12月底就是31天,其他都30天
            
        #####################################這邊先改#####################
            #if(str(temp_month.month+1)=='12'):
                #data = yf.download(str(id)+".TW", start=str_month+'-10', end=str_month+'-31')
                

            #else:
                #data = yf.download(str(id)+".TW", start=str_month+'-10', end=str_month+'-30')
        
        
        
        except RemoteDataError:
            continue
        except IndexError:
            continue
        
        
        
        ########加買賣價格#######
        realmonth=pd.Series(month[q])
        try:
            #buyprice=data.iloc[0,3:4]
            buyprice= pd.DataFrame(pd.Series(1))  #先設1 原本是buyprice=data.iloc[0,3:4]
            
        except RemoteDataError:
            continue
        except IndexError:
            continue
        
        sellprice=pd.DataFrame(pd.Series(2))  #sellprice=data.iloc[(len(data)-1),3:4]

        
        return_rate= pd.DataFrame(pd.Series(1)) #round(data.iloc[(len(data)-1),3:4]-data.iloc[0,3:4],2)/round(data.iloc[0,3:4],2)*100
        return_rate.index=id_monthdata.index
        buyprice.index=id_monthdata.index
        sellprice.index=id_monthdata.index
        realmonth.index=id_monthdata.index
        
        
        ######合併 財務比率和月資料##########    
        #final_data=pd.concat([pd.DataFrame(realmonth),id_monthdata,pd.DataFrame(id_ROEdata),pd.DataFrame(id_operate_margindata),pd.DataFrame(id_Net_Incomedata),pd.DataFrame(return_rate),pd.DataFrame(buyprice),pd.DataFrame(sellprice)], axis=1) #兩個表水平整合
        #final_data.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income','return_rate','買入價','賣出價']
  
        final_data=pd.concat([pd.DataFrame(realmonth),pd.DataFrame(id_monthdata),
                              pd.DataFrame(id_ROEdata),pd.DataFrame(id_operate_margindata),pd.DataFrame(id_Net_Incomedata),pd.DataFrame(id_ROA),
                              pd.DataFrame(id_GrossMargin),pd.DataFrame(id_AccountsReceivableTurnover),pd.DataFrame(id_AccountsReceivableTurnoverDay),
                              pd.DataFrame(id_InventoryTurnover),pd.DataFrame(id_InventoryTurnoverDay),pd.DataFrame(id_TotalAssetTurnover),
                              pd.DataFrame(id_CurrentRatio),pd.DataFrame(id_QuickRatio),pd.DataFrame(id_InterestCoverage),                           
                              pd.DataFrame(id_DebtRatio),pd.DataFrame(id_LongTermLiabilitiesRatio),
                              pd.DataFrame(id_OperatingCashflowToCurrentLiability),pd.DataFrame(id_OperatingCashflowToLiability),pd.DataFrame(id_OperatingCashflowToNetProfit),
                              pd.DataFrame(return_rate),pd.DataFrame(buyprice),pd.DataFrame(sellprice)], axis=1) #兩個表水平整合
               
         
        final_data.columns=['月份','公司代號','公司名稱','備註','上月比較增減(%)','上月營收','去年同月增減(%)',    '去年當月營收',      '當月營收', '前期比較增減(%)',    '去年累計營收','當月累計營收','ROE','operate_margin','Net_Income',
'資產報酬率','毛利率','應收款項週轉率','平均收現日數','存貨周轉率','平均銷貨天數','總資產週轉率',
'流動比率','速動比率','利息保障倍數','負債佔資產比率','長期資金佔不動產','營業現金對流動負債比',
'營業現金對負債比','營業現金對稅後純益比','return_rate','買入價','賣出價']   
        
        #成長能力4個指標拿掉 因為是從2014Q1開始   營業收入年增率','營業毛利年增率','營業利益年增率','稅後純益年增率',
        #,pd.DataFrame(id_RevenueYOY),                              
                              #pd.DataFrame(id_GrossProfitYOY),pd.DataFrame(id_OperatingIncomeYOY),
                              #pd.DataFrame(id_NetProfitYOY),pd.DataFrame(id_EPSYOY),
        
        
        #垂直疊加
          
        final_data2=pd.concat([final_data2,final_data], axis=0) 
        
        time.sleep(0.5)
        


##0424自己暫存的資料        
final_data2.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0424__0到405.csv', index = False, encoding='utf_8_sig')#輸出excel檔
       
        
        
        
 #儲存檔案 
final_data2.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__600到864.csv', index = False, encoding='utf_8_sig')#輸出excel檔

#備份
final_data3=final_data2

#合併資料
final_data1=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__0到100.csv', encoding='utf_8_sig')


final_data4=pd.concat([final_data1,final_data2], axis=0) 

final_data4.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__0到150.csv', index = False, encoding='utf_8_sig')#輸出excel檔

########################11.19告一段落###########
#總資料2500~864    
final_data_1=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__250到400.csv', encoding='utf_8_sig')
final_data_2=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__400到600.csv', encoding='utf_8_sig')
final_data_3=pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__600到864.csv', encoding='utf_8_sig')
    

last_final_data=pd.concat([final_data_1,final_data_2,final_data_3], axis=0) 
        
        
last_final_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/final_data/final_data_0107__400到864.csv', index = False, encoding='utf_8_sig')#輸出excel檔
        
        
        
        
        
