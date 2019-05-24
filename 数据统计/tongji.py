__author__ = 'Cila'
import tushare as ts
import numpy as np
import datetime
import decimal
import pymssql
import time


data = ts.get_hist_data('sh204001',start='2015-01-01',end='2015-06-09',str=True) #一次性获取全部日k线数据
data_time = np.array(data.index)
data = np.array(data)
high_arr = []
for i in range(0,len(data)):
    day_data = np.array(ts.get_tick_data('sh204001',date=data_time[i],retry_count=99999,pause=1))
    if day_data[0][0] != 'alert("当天没有数据");':
        # high_price = 0
        # high_time = []
        # for j in range(0,len(day_data)):
        #     if day_data[j][1] >= high_price:
        #         high_price = day_data[j][1]
        # index = np.where(high_price == day_data)
        # if len(index[0]) > 0:
        #     for k in range(0,len(index[0])):
        #         high_time.append(day_data[index[0][k]][0])
        # high_arr.append(high_time)
        for j in range(0,len(day_data)):
            if j % 1000 == 0:
                insert_sql = "insert into tickhist_mstr VALUES "
            insert_sql += "('204001',0,0,0,0,"+ str.replace(data_time[i],'-','') + str.replace(day_data[j][0],':','') + ","+str(day_data[j][1]) + ")"
            if (j+1) % 1000 == 0:
                conn = pymssql.connect("127.0.0.1","sa","sa123$%^","StockPolicy")
                cur = conn.cursor()
                cur.execute(insert_sql)
                conn.commit()
                conn.close()
            else:
                if j == len(day_data)-1:
                    conn = pymssql.connect("127.0.0.1","sa","sa123$%^","StockPolicy")
                    cur = conn.cursor()
                    cur.execute(insert_sql)
                    conn.commit()
                    conn.close()
                else:
                    insert_sql += ','
        time.sleep(1)
print(1)