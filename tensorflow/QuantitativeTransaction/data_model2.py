import numpy as np
import os
import pandas as pd
import random
import time
import talib
import sklearn.preprocessing as prep
import pywt
import math
from statsmodels.robust import mad
from LearningDataHandle import DataHandle

random.seed(time.time())

class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 sheet='sheet1',
                 input_size=10,
                 num_steps=1,
                 valid_ratio=0.1,
                 state_param = 0,
                 initCsv=False,
                 normalized=True,
                 logisticRegression=True,
                 group=False):
        self.stock_sym = stock_sym
        self.sheet = sheet
        self.input_size = input_size
        self.num_steps = num_steps
        self.valid_ratio = valid_ratio
        self.state_param = state_param
        self.normalized = normalized
        self.group = group
        self.initCsv = initCsv
        self.lR = logisticRegression
        # Read csv file

        if self.initCsv:
            raw_df = pd.read_excel(os.path.join("data", "%s.xls" % self.stock_sym),self.sheet,header=None)
            RawPrice = np.array(raw_df.iloc[1:,4])
            RawVolum = np.array(raw_df.iloc[1:,5])
            RawHigh = np.array(raw_df.iloc[1:,2])
            RawLow = np.array(raw_df.iloc[1:,3])
            new_data = [RawPrice,RawVolum,RawHigh,RawLow]
            df = pd.DataFrame(new_data).T
            df.to_csv(os.path.join("data",'%s.csv' % self.stock_sym),header=None,index=None)

        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym),header=None)

        RawPrice = np.array(raw_df.iloc[:,0]).tolist()
        RawVolum = np.array(raw_df.iloc[:,1]).tolist()
        RawHigh = np.array(raw_df.iloc[:,2]).tolist()
        RawLow = np.array(raw_df.iloc[:,3]).tolist()
        [WavePrice,LenWavePrice] = self.XIAOBOQUZAO(RawPrice)
        [WaveVolumn,LenWaveVolumn] = self.XIAOBOQUZAO(RawVolum)
        [WaveHigh,LenWavehigh] = self.XIAOBOQUZAO(RawHigh)
        [WaveLow,LenWaveLow] = self.XIAOBOQUZAO(RawLow)
        self.raw_seq = DataHandle(WavePrice,WaveVolumn,WaveHigh,WaveLow,LenWavePrice,LenWaveVolumn)

        #计算斜率
        # ma_60 = WavePrice[0:59]
        # for i in range(59,LenWavePrice):
        #     ma_60.append(np.mean(WavePrice[i-59:i+1]))
        slop_5 = [0] * 4
        for i in range(4,LenWavePrice):
            slop_5.append(math.atan(100 * (WavePrice[i] - WavePrice[i-4]) / WavePrice[i-4]) * 180 / np.pi)
        state = np.zeros(LenWavePrice)
        for i in range(LenWavePrice):
            if slop_5[i] > self.state_param:
                state[i] = 1
            elif slop_5[i] >= -self.state_param and slop_5[i] <= self.state_param:
                state[i] = 2
            else:
                state[i] = 3
        self.state = state
        self.raw_price = RawPrice
        self.raw_volum = RawVolum
        self.raw_high = RawHigh
        self.raw_low = RawLow

        self.raw_seq = np.array(self.raw_seq[200:])
        sig = np.zeros([len(RawPrice)-5,3])
        for i in range(len(RawPrice)-5):
            temp = (np.mean(RawPrice[i+1:i+6]) - RawPrice[i]) / RawPrice[i] * 100
            if self.lR:
                if temp > 0:
                    sig[i] = [1,0,0]
                elif temp == 0:
                    sig[i] = [0,1,0]
                else:
                    sig[i] = [0,0,1]
            else:
                sig[i] = temp
        self.raw_sig = sig[200:]
        self.train_X, self.train_y,self.valid_X,self.valid_y,self.train_state,self.test_state = self._prepare_data(self.raw_seq,self.raw_sig)


    def GetTestData(self,test_sym):
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % test_sym),header=None)
        RawPrice = np.array(raw_df.iloc[:,0]).tolist()
        RawVolum = np.array(raw_df.iloc[:,1]).tolist()
        RawHigh = np.array(raw_df.iloc[:,2]).tolist()
        RawLow = np.array(raw_df.iloc[:,3]).tolist()
        test_data = []
        self.raw_test_price = RawPrice
        if self.lR:
            sig = np.zeros([len(RawPrice),3])
        else:
            sig = np.zeros([len(RawPrice),1])
        for i in range(len(RawPrice)-5):
            temp = (np.mean(RawPrice[i+1:i+6]) - RawPrice[i]) / RawPrice[i] * 100
            if self.lR:
                if temp > 0:
                    sig[i] = [1,0,0]
                elif temp == 0:
                    sig[i] = [0,1,0]
                else:
                    sig[i] = [0,0,1]
            else:
                sig[i] = temp
        if self.lR:
            sig[-5:] = [0,1,0]
        self.raw_test_sig = sig
        self.date = np.array(raw_df.iloc[:,4]).tolist()
        ma120_price = np.zeros(len(RawPrice)+199)
        ma120_volum = np.zeros(len(RawPrice)+199)
        ma120_high = np.zeros(len(RawPrice)+199)
        ma120_low = np.zeros(len(RawPrice)+199)
        for i in range(len(RawPrice)+199):
            if i < 199:
                ma120_price[i] = np.mean(self.raw_price[-199-120+i:-199+i])
                ma120_volum[i] = np.mean(self.raw_volum[-199-120+i:-199+i])
                ma120_high[i] = np.mean(self.raw_high[-199-120+i:-199+i])
                ma120_low[i] = np.mean(self.raw_low[-199-120+i:-199+i])
            elif i == 199:
                ma120_price[i] = np.mean(self.raw_price[-199-120+i:])
                ma120_volum[i] = np.mean(self.raw_volum[-199-120+i:])
                ma120_high[i] = np.mean(self.raw_high[-199-120+i:])
                ma120_low[i] = np.mean(self.raw_low[-199-120+i:])
            elif i < 319 and i >=200:
                ma120_price[i] = np.mean(self.raw_price[-319+i:] + RawPrice[:i - 199])
                ma120_volum[i] = np.mean(self.raw_volum[-319+i:] + RawVolum[:i - 199])
                ma120_high[i] = np.mean(self.raw_high[-319+i:] + RawHigh[:i - 199])
                ma120_low[i] = np.mean(self.raw_low[-319+i:] + RawLow[:i - 199])
            else:
                ma120_price[i] = np.mean(RawPrice[i-319:i-199])
                ma120_volum[i] = np.mean(RawVolum[i-319:i-199])
                ma120_high[i] = np.mean(RawHigh[i-319:i-199])
                ma120_low[i] = np.mean(RawLow[i-319:i-199])

        for i in range(len(RawPrice)):
            #使用MA120
            # price = ma120_price[i:i+200]
            # volum = ma120_volum[i:i+200]
            # high = ma120_high[i:i+200]
            # low = ma120_low[i:i+200]
            if i < 200 :
                price = self.raw_price[-199 + i:] + RawPrice[:i+1]
                volum = self.raw_volum[-199 + i:] + RawVolum[:i+1]
                high = self.raw_high[-199 + i:] + RawHigh[:i+1]
                low = self.raw_low[-199 + i:] + RawLow[:i+1]
            else:
                price = RawPrice[i-200:i]
                volum = RawVolum[i-200:i]
                high = RawHigh[i-200:i]
                low = RawLow[i-200:i]
            seq = DataHandle(price,volum,high,low,200,200,True)
            test_data.append(seq[:self.input_size])
        return test_data

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq,sig):
        # split into items of input_size
        seq = seq[:,:self.input_size]

        # split into groups of num_steps
        # if self.num_steps > 0:
        #     seq = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        #     sig = np.array([sig[i: i + self.num_steps] for i in range(len(sig) - self.num_steps)])

        train_size = int(len(seq) * (1 - self.valid_ratio))
        self.train_size = train_size
        train_X, valid_X = seq[:train_size-5], seq[train_size:]
        train_y, valid_y = sig[:train_size], sig[train_size:]
        # if self.normalized:
        #     train_X = self.standard_scale(train_X,train_X)
        #     test_X = self.standard_scale(train_X,test_X)
        train_state,test_state = self.state[:train_size],self.state[train_size:]
        return train_X, train_y, valid_X, valid_y,train_state,test_state

    def generate_one_epoch(self, i, batch_size):
        if(self.num_steps > 1 and self.group):
            x = []
            for j in range(self.num_steps,len(self.train_X) +1):
                train_x = self.train_X[j-self.num_steps:j]
                x.append(train_x)
            self.train_X = np.array(x)
            self.train_y = self.train_y[self.num_steps-1:]
            self.group = False

        if batch_size == 0:
            batch_size = len(self.train_X)
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1
        i = i % num_batches

        batch_X = self.train_X[i * batch_size: (i + 1) * batch_size]
        batch_y = np.array(self.train_y[i * batch_size: (i + 1) * batch_size])
        return batch_X, batch_y

    def standard_scale(self,fit,data):
        preprocessor = prep.StandardScaler().fit(fit)
        data = preprocessor.transform(data)
        return data

    def XIAOBOQUZAO(self,RawData):
        # w = pywt.Wavelet('db4')
        # noisy_coefs = pywt.wavedec(RawData,w,level=4,mode='per')
        # denoised = noisy_coefs[:]
        # sigma = mad(noisy_coefs[-1])
        # uthresh = sigma * np.sqrt(2 * np.log(len(RawData)))
        # denoised[1:] = (pywt.threshold(a,value=uthresh,mode='soft') for a in denoised[1:])
        # signal = pywt.waverec(denoised,w,mode='per')[0:len(RawData)]

        #ma120
        # signal = np.array(RawData)
        # signal[:119] = RawData[0:119]
        # for i in range(119,len(RawData)):
        #     signal[i] = np.mean(RawData[i-119:i+1])

        return RawData,len(RawData)
