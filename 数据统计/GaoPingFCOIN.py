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
host = "47.75.68.205"
# host = "47.75.178.163"
user = "sa"
pwd = "sa123$%^"
account = "lch"
enter_date = '2018-08-18 00:00:00'
percent = 0


def get_stock_data(date):
    conn_info = 'DRIVER={SQL Server};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s'%(database, host, user, pwd)
    mssql_conn = pyodbc.connect(conn_info)
    date = dt.datetime.strptime(date, '%Y-%m-%d %X')
    nextday = (date + dt.timedelta(days=1)).strftime('%Y-%m-%d %X')
    date = date.strftime("%Y-%m-%d %X")
    sql = "select * from trade_hist where trade_time>='"+date+"'and trade_time<'"+nextday+"' and trade_account = '" + account +"'"
    df = pd.read_sql(sql, mssql_conn)
    df = pd.DataFrame(df)
    #df = df[df['trade_policyname'] == 'PolicyBtc0601ForFCOINftbtc']
    #df = df[df['trade_policyname'] == 'PolicyBtc0601ForFCOINftft1808']
    #df = pd.DataFrame(df, dtype={"tick_ask1": np.double, "tick_ask2": np.double, "tick_bid1": np.double, "tick_bid2": np.double})
    df = df.sort_values(by=['trade_time'])
    df.index = range(len(df))
    return df

def deal_data(data):
    data = data[data['入场成交数量'] != 0]
    data.index = range(len(data))
    for i in range(len(data)):
        # if data.loc[i, "入场委托方向"] == "Buy":
        #     data.loc[i, "入场手续费"] = data.loc[i, "入场手续费"] * data.loc[i, "入场成交价格"]
        if data.loc[i, "出场委托方向"] == "Buy":
            # data.loc[i, "出场手续费"] = data.loc[i, "出场手续费"] * data.loc[i, "出场成交价格"]
            if data.loc[i, "策略持仓"] != 0:
                data.loc[i, "策略持仓"] = -data.loc[i, "策略持仓"]
        data.loc[i, "实际盈利"] = np.double(data.loc[i, "实际盈利"])

    stock_array = list(set(data["策略名称"]))
    for stock_name in stock_array:
        temp_data = data[data["策略名称"] == stock_name]
        temp_data.index = range(len(temp_data))
        coin = list(set(temp_data["证券代码"]))[0]
        save_string = "实盘结果-"+account+"-" + coin + "-"+stock_name+"-"+enter_date.split(" ")[0]+".xls"
        writer = pd.ExcelWriter("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/"+save_string, index=False)
        temp_count = len(set(temp_data["备注"]))
        temp_prices = sum(temp_data["实际盈利"])
        temp_poundage = 0
        temp_coins = 0
        for i in range(len(temp_data)):
            if temp_data.loc[i,'出场委托方向'] == "Buy":
                temp_poundage += temp_data.loc[i,'入场手续费']
                temp_coins += temp_data.loc[i,'出场手续费']
            else:
                temp_poundage += temp_data.loc[i,'出场手续费']
                temp_coins += temp_data.loc[i,'入场手续费']
        rest_coins = sum(temp_data["策略持仓"])
        temp_data.to_excel(writer,sheet_name='sheet1', index=False)
        temp_data2 = pd.DataFrame([[temp_count,temp_prices,temp_poundage,temp_coins,rest_coins]],columns=['总计','盈利','usdt手续费',coin + '手续费',"多余币"])
        temp_data2.to_excel(writer,sheet_name='统计', index=False)
        writer.save()
    excels = file_name(account,"PolicyBtc0801")
    writer2 = pd.ExcelWriter("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/统计/" + "实盘结果-"+account+"-PolicyBtc0801.xls")
    dates = dict()
    temp_name = ''
    sheet_array = []
    sheet_data = []
    i = -1
    for excel in excels:
        excel_data = np.array(pd.read_excel(excel,'统计'))[0]
        if excel[0:-15] != temp_name:
            i += 1
            sheet_dates = dict()
            temp_name = excel[0:-15]
            coin = excel[-37:-30]
            sheet_array.append(excel[-37:-15])
            sheet_data.append(sheet_dates)
        if(excel[-14:-4] in sheet_data[i].keys()):
            if coin == "btcusdt":
                sheet_data[i][excel[-14:-4]] = [sheet_data[i][excel[-14:-4]][0] + excel_data[0],sheet_data[i][excel[-14:-4]][1] + excel_data[1],sheet_data[i][excel[-14:-4]][2] + excel_data[2],sheet_data[i][excel[-14:-4]][3] + excel_data[3],sheet_data[i][excel[-14:-4]][4],sheet_data[i][excel[-14:-4]][5] + excel_data[4],sheet_data[i][excel[-14:-4]][6]]
            else:
                sheet_data[i][excel[-14:-4]] = [sheet_data[i][excel[-14:-4]][0] + excel_data[0],sheet_data[i][excel[-14:-4]][1] + excel_data[1],sheet_data[i][excel[-14:-4]][2] + excel_data[2],sheet_data[i][excel[-14:-4]][3],sheet_data[i][excel[-14:-4]][4] + excel_data[3],sheet_data[i][excel[-14:-4]][5],sheet_data[i][excel[-14:-4]][6] + excel_data[4]]
        else:
            if coin == "btcusdt":
                sheet_data[i][excel[-14:-4]] = [excel_data[0],excel_data[1],excel_data[2],excel_data[3],0,excel_data[4],0]
            else:
                sheet_data[i][excel[-14:-4]] = [excel_data[0],excel_data[1],excel_data[2],0,excel_data[3],0,excel_data[4]]

        if(excel[-14:-4] in dates.keys()):
            if coin == "btcusdt":
                dates[excel[-14:-4]] = [dates[excel[-14:-4]][0] + excel_data[0],dates[excel[-14:-4]][1] + excel_data[1],dates[excel[-14:-4]][2] + excel_data[2],dates[excel[-14:-4]][3] + excel_data[3],dates[excel[-14:-4]][4],dates[excel[-14:-4]][5] + excel_data[4],dates[excel[-14:-4]][6]]
            else:
                dates[excel[-14:-4]] = [dates[excel[-14:-4]][0] + excel_data[0],dates[excel[-14:-4]][1] + excel_data[1],dates[excel[-14:-4]][2] + excel_data[2],dates[excel[-14:-4]][3],dates[excel[-14:-4]][4] + excel_data[3],dates[excel[-14:-4]][5],dates[excel[-14:-4]][6] + excel_data[4]]
        else:
            if coin == "btcusdt":
                dates[excel[-14:-4]] = [excel_data[0],excel_data[1],excel_data[2],excel_data[3],0,excel_data[4],0]
            else:
                dates[excel[-14:-4]] = [excel_data[0],excel_data[1],excel_data[2],0,excel_data[3],0,excel_data[4]]

    for index in range(len(sheet_array)):
        excel_sheet_data = pd.DataFrame(list(sheet_data[index].values()),index=list(sheet_data[index].keys()),columns=['总计','盈利','usdt手续费','btc手续费','eth手续费','多余btc','多余eth'])
        excel_sheet_data.to_excel(writer2,sheet_name=sheet_array[index])
    df = pd.DataFrame(list(dates.values()),index=list(dates.keys()),columns=['总计','盈利','usdt手续费','btc手续费','eth手续费','多余btc','多余eth'])
    df.to_excel(writer2,sheet_name='统计')
    writer2.save()

def file_name(account,policyname):
    L=[]
    for root, dirs, files in os.walk("C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/"):
        if root == "C:/Users/Cila/Desktop/GainRate/数字货币实盘结果/":
            for file in files:
                filename = os.path.splitext(file)[0]
                info = filename.split("-")
                if info[1] == account and policyname in info[3]:
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