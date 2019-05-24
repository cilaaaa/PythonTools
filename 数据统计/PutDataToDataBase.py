__author__ = 'Cila'
import os
import gzip
from contextlib import closing
import requests
import pandas as pd
import pymssql
import datetime
import time

def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    #获取文件的名称，去掉
    g_file = gzip.GzipFile(file_name)
    #创建gzip对象
    open(f_name, "wb+").write(g_file.read())
    #gzip对象用read()打开后，写入open()建立的文件里。
    g_file.close()
    #关闭gzip对象


def file_name(file_dir,type):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == type:
                L.append(os.path.join(root, file))
    return L

class ProgressBar(object):

    def __init__(self, title,
                 count=0.0,
                 run_status=None,
                 fin_status=None,
                 total=100.0,
                 unit='', sep='/',
                 chunk_size=1.0):
        super(ProgressBar, self).__init__()
        self.info = "【%s】%s %.2f %s %s %.2f %s"
        self.title = title
        self.total = total
        self.count = count
        self.chunk_size = chunk_size
        self.status = run_status or ""
        self.fin_status = fin_status or " " * len(self.status)
        self.unit = unit
        self.seq = sep

    def __get_info(self):
        # 【名称】状态 进度 单位 分割线 总数 单位
        _info = self.info % (self.title, self.status,
                             self.count/self.chunk_size, self.unit, self.seq, self.total/self.chunk_size, self.unit)
        return _info

    def refresh(self, count=1, status=None):
        self.count += count
        # if status is not None:
        self.status = status or self.status
        end_str = "\r"
        if self.count >= self.total:
            end_str = '\n'
            self.status = status or self.fin_status
        print(self.__get_info(), end=end_str)

files = file_name('E:/Bitcoin/test/2019','.gz')
for temp_file in files:
    while True:
        try:
            un_gz(temp_file)
            os.remove(temp_file)
            print(temp_file)
            break
        except:
            name = str(os.path.split(temp_file)[1].split(".")[0])
            print(name)
            url = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/'+ name +'.csv.gz'
            #local = url.split('/')[-1]
            file_name = os.path.join('./Bitcoin/test',name + '.csv.gz')
            with closing(requests.get(url, stream=True)) as response:
                chunk_size = 1024 # 单次请求最大值
                content_size = int(response.headers['content-length']) # 内容体总大小
                progress = ProgressBar(file_name, total=content_size,
                                                 unit="KB", chunk_size=chunk_size, run_status="正在下载", fin_status="下载完成")
                with open(file_name, "wb") as file:
                    for data in response.iter_content(chunk_size=chunk_size):
                        file.write(data)
                        progress.refresh(count=len(data))
#
# files = file_name('./Bitcoin/test/2018','.csv')
# for temp_file in files:
#     print(temp_file)
#     sql = 'INSERT INTO tick_data_bitmex(TICK_CODE,tick_timestamp,TICK_TIME, TICK_ASK1, TICK_ASKS1,TICK_ASK2, TICK_ASKS2,TICK_ASK3, TICK_ASKS3,TICK_ASK4, TICK_ASKS4,TICK_ASK5, TICK_ASKS5,TICK_ASK6, TICK_ASKS6,' \
#           'TICK_ASK7, TICK_ASKS7,TICK_ASK8, TICK_ASKS8,TICK_ASK9, TICK_ASKS9,TICK_ASK10, TICK_ASKS10,' \
#           'TICK_BID1,TICK_BIDS1,TICK_BID2,TICK_BIDS2,TICK_BID3,TICK_BIDS3,TICK_BID4,TICK_BIDS4,TICK_BID5,TICK_BIDS5,TICK_BID6,TICK_BIDS6,TICK_BID7,TICK_BIDS7,TICK_BID8,TICK_BIDS8,TICK_BID9,TICK_BIDS9,TICK_BID10,TICK_BIDS10) ' \
#           'VALUES '
#     df = pd.read_csv(temp_file)
#     df = df[df['symbol'] == 'XBTUSD']
#     df = df.sort_values(by=['timestamp'])
#     df.index = range(len(df))
#     total = 0
#     index = 0
#
#     for i in range(len(df)):
#         askPrice = df.loc[i, "askPrice"]
#         bidPrice = df.loc[i, "bidPrice"]
#         timeStr = df.loc[i, "timestamp"]
#         askSize = df.loc[i, "askSize"]
#         bidSize = df.loc[i, "bidSize"]
#         dt = datetime.datetime.strptime(timeStr,"%Y-%m-%dD%H:%M:%S.%f000") + datetime.timedelta(hours=8)
#         tick_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
#         if len(tick_time) == 26:
#             tick_time = tick_time[:-3]
#         ms = int(dt.microsecond / 1000)
#         timestamp = int(round(time.mktime(time.strptime(timeStr, "%Y-%m-%dD%H:%M:%S.%f000")) * 1000)) + ms
#         sql += "('%s', '%s', '%s'," % ('XBTUSD', timestamp, tick_time)
#         for i in range(10):
#             sql += "'" + str(askPrice) + "','" + str(askSize) + "',"
#         for i in range(10):
#             sql += "'" + str(bidPrice) + "','" + str(bidSize) + "',"
#         sql = sql[:-1] + "),"
#         index += 1
#         if index == 1000:
#             total += 1000
#             index = 0
#             sql = sql[:-1]
#             con = pymssql.connect('192.168.200.11', 'sa', 'sa123$%^', 'Bitcoin')
#             # con = mysql.connector.connect(host='localhost', port=3306, user='root',
#             #                               password='1234', database='huobi', charset='utf8')
#
#             cursor = con.cursor()
#             try:
#                 # 执行sql语句
#                 cursor.execute(sql)
#                 # 提交到数据库执行
#                 con.commit()
#
#                 # logger.info('数据保存成功')
#             except pymssql.Error as e:
#                 print(' sql:' + sql + ' insert error!{}'.format(e))
#                 # 如果发生错误则回滚
#                 con.rollback()
#
#                 cursor.close()
#                 # 关闭数据库连接
#                 con.close()
#
#             sql = 'INSERT INTO tick_data_bitmex(TICK_CODE,tick_timestamp,TICK_TIME, TICK_ASK1, TICK_ASKS1,TICK_ASK2, TICK_ASKS2,TICK_ASK3, TICK_ASKS3,TICK_ASK4, TICK_ASKS4,TICK_ASK5, TICK_ASKS5,TICK_ASK6, TICK_ASKS6,' \
#                   'TICK_ASK7, TICK_ASKS7,TICK_ASK8, TICK_ASKS8,TICK_ASK9, TICK_ASKS9,TICK_ASK10, TICK_ASKS10,' \
#                   'TICK_BID1,TICK_BIDS1,TICK_BID2,TICK_BIDS2,TICK_BID3,TICK_BIDS3,TICK_BID4,TICK_BIDS4,TICK_BID5,TICK_BIDS5,TICK_BID6,TICK_BIDS6,TICK_BID7,TICK_BIDS7,TICK_BID8,TICK_BIDS8,TICK_BID9,TICK_BIDS9,TICK_BID10,TICK_BIDS10) ' \
#                   'VALUES '
#             print(str(total) + "/" + str(len(df)))
#     sql = sql[:-1]
#     con = pymssql.connect('192.168.200.11', 'sa', 'sa123$%^', 'Bitcoin')
#     # con = mysql.connector.connect(host='localhost', port=3306, user='root',
#     #                               password='1234', database='huobi', charset='utf8')
#
#     cursor = con.cursor()
#     try:
#         # 执行sql语句
#         cursor.execute(sql)
#         # 提交到数据库执行
#         con.commit()
#
#         # logger.info('数据保存成功')
#     except pymssql.Error as e:
#         print(' sql:' + sql + ' insert error!{}'.format(e))
#         # 如果发生错误则回滚
#         con.rollback()
#
#         cursor.close()
#         # 关闭数据库连接
#         con.close()