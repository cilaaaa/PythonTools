__author__ = 'Cila'

import pyodbc
import pandas as pd
import numpy as np
import datetime as dt

database = "Bitcoin"
host = "192.168.200.10"
user = "sa"
pwd = "sa123$%^"

enter_date = '2018-10-01'

def get_stock_data(date):
    conn_info = 'DRIVER={SQL Server};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s'%(database, host, user, pwd)
    mssql_conn = pyodbc.connect(conn_info)
    nextday = (date + dt.timedelta(days=1)).strftime('%Y-%m-%d %X')
    date2 = date.strftime("%Y-%m-%d %X")
    sql = "select * from tick_data_okex_future_quarter where tick_time>='"+ date2 +"'and tick_time<'"+nextday+"' and tick_code = 'eosusdt' order by tick_time asc"
    df = pd.read_sql(sql, mssql_conn)
    df = pd.DataFrame(df)
    #df = df[df['trade_policyname'] == 'PolicyBtc0601ht']
    #df = pd.DataFrame(df, dtype={"tick_ask1": np.double, "tick_ask2": np.double, "tick_bid1": np.double, "tick_bid2": np.double})
    array = np.zeros([len(df),6],dtype=object)
    for i in range(len(df)):
        array[i][0] = df.loc[i, "tick_time"]
        array[i][1] = "EOS-USD-190329"
        array[i][2] = df.loc[i, "tick_bids1"]
        array[i][3] = df.loc[i, "tick_bid1"]
        array[i][4] = df.loc[i, "tick_ask1"]
        array[i][5] = df.loc[i, "tick_asks1"]
    newdf = pd.DataFrame(array,columns=['timestamp','symbol','bidSize','bidPrice','askPrice','askSize'])
    newdf.to_csv("E:/Bitcoin/okex/" + date.strftime("%Y%m%d") + ".csv",index=False)
    return df

end_date = '2019-01-01'
sdate = dt.datetime.strptime(enter_date, '%Y-%m-%d')
edate = dt.datetime.strptime(end_date, '%Y-%m-%d')
while ((edate - sdate).days > 0):
    get_stock_data(sdate)
    sdate += dt.timedelta(days=1)