__author__ = 'Cila'
# -*- coding: utf-8 -*-

import types
import urllib3
import json
import pymssql
import datetime
import numpy
import decimal

class GetCoinRate():
    def __init__(self):
        self.done = False
        self.updateNavcur()

    #利用urllib2获取网络数据
    def registerUrl(self,code,page):
        http = urllib3.PoolManager()
        r = http.request('GET',"http://fund.jrj.com.cn/json/archives/history/netvalue?fundCode="+code+"&obj=obj&date="+page)
        r = r.data.decode('utf-8')[8:]
        return r

    #解析从网络上获取的JSON数据
    def praserJsonFile(self,jsonData):
        value = json.loads(jsonData)
        if 'enddate' in value['fundHistoryNetValue'][0].keys():
            return value['fundHistoryNetValue']
        else:
            return None

    def InsertToDataBase(self,tick_code):
        self.cur.execute("select tick_time from tick_navcur where tick_code = '" + tick_code + "' order by tick_time desc")
        tick_info = numpy.array(self.cur.fetchall()).flatten()
        page = 2018
        flag = True
        while flag:
            online_data = self.registerUrl(tick_code,str(page))
            online_data = self.praserJsonFile(online_data)
            if online_data is None:
                flag = False
            else:
                # if '0' in online_data:
                #     temp_data = []
                #     for x in online_data:
                #         temp_data.append(online_data[x])
                #     online_data = temp_data
                label_time = online_data[len(online_data)-1]['enddate']
                if label_time not in tick_info:
                    insert_sql = "insert into tick_navcur VALUES "
                    for i in range(len(online_data)):
                        insert_sql += "('" + tick_code + "','" + online_data[i]['enddate'] + "'," + online_data[i]['yearYld'] + "," + online_data[i]['tenthouUnitIncm'] + ")"
                        if i != len(online_data) - 1:
                            insert_sql += ','
                    self.cur.execute(insert_sql)
                    self.conn.commit()
                else:
                    for j in range(len(online_data)):
                        label_time = online_data[j]['enddate']
                        if label_time not in tick_info:
                            insert_sql = "insert into tick_navcur VALUES ('" + tick_code + "','" + online_data[j]['enddate'] + "'," + online_data[j]['yearYld'] + "," + online_data[j]['tenthouUnitIncm'] + ")"
                            self.cur.execute(insert_sql)
                            self.conn.commit()
                        else:
                            flag = False
            page -= 1

    def updateNavcur(self):
        tick_data = ['511990','511660','519888','001621','004796','000540']
        self.conn = pymssql.connect("127.0.0.1","sa","sa123$%^","StockPolicy")
        self.cur = self.conn.cursor()
        for tick_code in tick_data:
            self.InsertToDataBase(tick_code)
        self.done = True

    def coinRate(self,tick_code,start_date = '1970-01-01',end_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        while True:
            if self.done:
                self.cur.execute("select * from tick_navcur where tick_code = '" + tick_code + "' and tick_time >= '" + start_date + "' and tick_time <= '" + end_date + "' order by tick_time asc")
                tick_info = self.cur.fetchall()
                # conn = pymssql.connect("192.168.200.10",'sa','sa123$%^','StockPolicy')
                # cur = conn.cursor()
                # cur.execute("SELECT Max(tick_open) as tick_open,convert(varchar,tick_time,23) as tick_time  FROM tickhist_mstr where tick_code = '" + tick_code + "' group by convert(varchar,tick_time,23) order by tick_time  desc")
                # tick_open = numpy.array(cur.fetchall())
                # if len(tick_info) == 0:
                #     return tick_open
                # else:
                #     for i,temp in enumerate(tick_info):
                #         index = numpy.where(temp[1] == tick_open)
                #         temp = list(temp)
                #         if len(index[0]) > 0:
                #             index = index[0][0]
                #             temp.append(decimal.Decimal(tick_open[index][0]))
                #         else:
                #             temp.append(decimal.Decimal(0))
                #         tick_info[i] = tuple(temp)
                tick_info = numpy.array(tick_info)
                return tick_info



