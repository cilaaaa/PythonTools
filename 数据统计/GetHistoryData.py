__author__ = 'Cila'
import pandas as pd
import requests
import os
from contextlib import closing
import datetime


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


startDateStr = '20190203'
endDateStr = '20190204'
sd = datetime.datetime.strptime(startDateStr,"%Y%m%d")
ed = datetime.datetime.strptime(endDateStr,"%Y%m%d")
days = (ed-sd).days
for i in range(days+1):
    name = (sd + datetime.timedelta(i)).strftime("%Y%m%d")
    url = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/'+ name +'.csv.gz'
    #local = url.split('/')[-1]
    file_name = os.path.join('E:\\Bitcoin',name + '.csv.gz')
    with closing(requests.get(url, stream=True)) as response:
        chunk_size = 1024  # 单次请求最大值
        content_size = int(response.headers['content-length']) # 内容体总大小
        progress = ProgressBar(file_name, total=content_size,
                                         unit="KB", chunk_size=chunk_size, run_status="正在下载", fin_status="下载完成")
        with open(file_name, "wb") as file:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.refresh(count=len(data))
