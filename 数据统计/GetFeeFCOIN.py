# 拉数字货币的收盘数据

import xlrd
import datetime
import pyodbc
import copy
import xlwt
import os
import pandas as pd
import numpy as np
import tushare as ts
import time
import datetime as dt
from xml.etree import ElementTree as ET

# 连接数据库
database = "BtcStockPolicy"
host = "192.168.200.57"
user = "sa"
pwd = "sa123$%^"
account = "lch"
enter_date = '2018-08-08 00:00:00'
percent = 0


def get_stock_data(date):
    conn_info = 'DRIVER={SQL Server};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s'%(database, host, user, pwd)
    mssql_conn = pyodbc.connect(conn_info)
    date = dt.datetime.strptime(date, '%Y-%m-%d %X')
    date = date.strftime("%Y-%m-%d %X")
    sql = "select * from trade_hist where trade_time>='"+date+"' and trade_account = '" + account +"'"
    df = pd.read_sql(sql, mssql_conn)
    df = pd.DataFrame(df)
    df = df[df['trade_policyname'] == 'PolicyBtcSellFt']
    #df = df[df['trade_policyname'] == 'PolicyBtc0601ForFCOINftft1808']
    #df = pd.DataFrame(df, dtype={"tick_ask1": np.double, "tick_ask2": np.double, "tick_bid1": np.double, "tick_bid2": np.double})
    df = df.sort_values(by=['trade_time'])
    df.index = range(len(df))
    return df

def deal_data(data):
    ignoreId = ''
    data = data[data['入场成交数量'] != 0]
    data.index = range(len(data))
    for i in range(len(data)):
        id = data.loc[i, "备注"]
        if id != ignoreId:
            price = data.loc[i, "入场成交价格"]
            direction = data.loc[i, "入场委托方向"]
            count = data.loc[i, "入场成交数量"]
            open_time = data.loc[i, "入场委托时间"]
            open_time = open_time.replace("年", "-")
            open_time = open_time.replace("月", "-")
            open_time = open_time.replace("日", "")

            name = data.loc[i, "证券名称"]
            # poundage = np.double(data.loc[i, "入场手续费"])*0.15

            total_price = np.double(price) * np.double(count)
            data.loc[i, "实际盈利"] = np.double(total_price)

    stock_array = list(set(data["策略名称"]))
    for stock_name in stock_array:
        temp_data = data[data["策略名称"] == stock_name]
        save_string = "实盘结果-"+account+"-"+stock_name[13:]+"-"+enter_date.split(" ")[0]+".xls"
        writer = pd.ExcelWriter("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/"+save_string, index=False)
        temp_count = len(set(temp_data["备注"]))
        temp_prices = sum(temp_data["实际盈利"])
        temp_poundage = sum(temp_data['入场手续费'])
        temp_data.to_excel(writer,sheet_name='sheet1', index=False)
        temp_data2 = pd.DataFrame([[temp_count,temp_prices,temp_poundage]],columns=['总计','盈利','总手续费'])
        temp_data2.to_excel(writer,sheet_name='统计', index=False)
        writer.save()
        excels = file_name("实盘结果-"+account+"-"+stock_name[13:])
        writer2 = pd.ExcelWriter("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/统计/" + "实盘结果-"+account+"-"+stock_name[13:] + ".xls")
        dates = []
        data2 = np.array([])
        for excel in excels:
            df = pd.read_excel(excel,'统计')
            dates.append(excel[-14:-4])
            if (len(data2) > 0):
                data2 = np.append(data2,np.array(df),axis=0)
            else:
                data2 = np.array(df)
        dates.append('统计')
        array = np.zeros([len(dates),3],dtype=object)
        for i in range(len(data2)):
            array[i][0] = data2[i][0]
            array[i][1] = data2[i][1]
            array[i][2] = data2[i][2]
        array[len(dates)-1][0] = np.sum(data2[:,0],axis=0)
        array[len(dates)-1][1] = np.sum(data2[:,1],axis=0)
        array[len(dates)-1][2] = np.sum(data2[:,2],axis=0)
        df = pd.DataFrame(array,index=dates,columns=['总计','盈利','总手续费'])
        df.to_excel(writer2,sheet_name='统计')
        writer2.save()

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/"):
        if root == "C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/":
            for file in files:
                if os.path.splitext(file)[0][0:-11] == file_dir:
                    L.append(os.path.join(root, file))
    return L



all_data = get_stock_data(enter_date)
del all_data['trade_guid']
del all_data['trade_uid']
del all_data['trade_time']
del all_data['trade_account']
del all_data['trade_entertime']

all_data.columns = ['策略名称','证券代码','证券名称','策略持仓','策略盈利','实际盈利','入场委托时间','入场委托价格',
                    '入场委托数量','入场委托方向','入场成交价格','入场成交数量','入场手续费','出场委托时间',
                    '出场委托价格','出场委托数量','出场委托方向','出场成交价格','出场成交数量','出场手续费','备注']

deal_data(all_data)