__author__ = 'Cila'
import numpy as np
import pandas as pd
import os

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir + "/"):
        if root == file_dir + "/":
            for file in files:
                L.append(os.path.join(root, file))
    return L

def loadFile(filePath):
    txts = file_name(filePath)
    for txt in txts:
        data_txt = np.loadtxt(txt,dtype=str,delimiter=',',usecols=(44,1,23,22,24,25,4),skiprows=1)
        data_txtDF = pd.DataFrame(data_txt,columns=['timestamp','symbol','bidSize','bidPrice','askPrice','askSize','lastPrice'])

        data_txtDF.to_csv(txt.replace('txt','csv'),index=False)


loadFile('E:\\futureData\cfIF1811')