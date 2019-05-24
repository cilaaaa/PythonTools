__author__ = 'Cila'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import datetime

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        if root == file_dir:
            for file in files:
                if os.path.splitext(file)[1] == '.xls':
                    L.append(os.path.join(root, file))
    return L

excels = file_name('12-21')
writer = pd.ExcelWriter('12-21/统计.xlsx')
for excel in excels:
    args = excel.split("\\")[1]
    PingTai = "期货"
    if PingTai == "期货":
        fee = 0
    else:
        fee = 0
    df = pd.read_excel(excel)
    data = np.array(df)
    plotArray = []
    xArray = []
    i = 1
    for dataDetail in data:
        if type(dataDetail[3]) == str:
            if dataDetail[11] == 'Sell':
                if PingTai == "期货":
                    lirun = (1 / dataDetail[5] - 1 / dataDetail[9]) * dataDetail[8] - (1 / dataDetail[9] + 1 / dataDetail[5]) * dataDetail[8] * fee
                else:
                    lirun = (dataDetail[9] - dataDetail[5]) * dataDetail[8] - (dataDetail[9] + dataDetail[5]) * dataDetail[8] * fee
            else:
                if PingTai == "期货":
                    lirun = (1 / dataDetail[9] - 1 / dataDetail[5]) * dataDetail[8] - (1 / dataDetail[9] + 1 / dataDetail[5]) * dataDetail[8] * fee
                else:
                    lirun = (dataDetail[5] - dataDetail[9]) * dataDetail[8] - (dataDetail[9] + dataDetail[5]) * dataDetail[8] * fee
            plotArray.append(lirun)
            xArray.append(i)
            i += 1
    df = pd.DataFrame(plotArray,index=xArray,columns=['利润'])
    df.to_excel(writer,sheet_name=PingTai + args[1] + args[2] + args[3])
    fig = plt.figure(excel)
    plt.grid(True)
    ax = plt.gca()
    ax.plot(np.cumsum(plotArray))
writer.save()
plt.show()