__author__ = 'Cila'
import numpy as np
import talib
import pandas as pd
import tushare as ts
import pywt
import math
from statsmodels.robust import mad
import matplotlib.pyplot as plt

train_percent = 0.99
valid_percent = 0
test_percent = 0.01

def DataHandle(WavePrice,WaveVolum,WaveHigh,WaveLow,LenWavePrice,LenWaveVolum):
    LPR = [0] * 199
    for i in range(199,LenWavePrice):
        if WavePrice[i] != 0:
            lpr = min(WavePrice[(i-199):i+1]) / WavePrice[i]
        else:
            lpr = 0
        LPR.append(lpr)

    HPR = [0] * 199
    for i in range(199,LenWavePrice):
        if WavePrice[i] != 0:
            hpr = max(WavePrice[(i-199):i+1]) / WavePrice[i]
        else:
            hpr = 0
        HPR.append(hpr)

    SMA_P3 = [0] * 2
    for i in range(2,LenWavePrice):
        sma_p3 = sum(WavePrice[(i-2):i+1]) / 3
        SMA_P3.append(sma_p3)

    SMA_P15 = [0] * 14
    for i in range(14,LenWavePrice):
        sma_p15 = sum(WavePrice[(i-14):i+1]) / 15
        SMA_P15.append(sma_p15)

    SMA_P = []
    for i in range(LenWavePrice):
        if SMA_P15[i] != 0:
            sma_p = SMA_P3[i] / SMA_P15[i]
        else:
            sma_p = 0
        SMA_P.append(sma_p)

    SMA_V3 = [0] * 2
    for i in range(2,LenWaveVolum):
        sma_v3 = sum(WaveVolum[(i-2):i+1]) / 3
        SMA_V3.append(sma_v3)

    SMA_V15 = [0] * 14
    for i in range(14,LenWaveVolum):
        sma_v15 = sum(WaveVolum[(i-14):i+1]) / 15
        SMA_V15.append(sma_v15)

    SMA_V = []
    for i in range(LenWaveVolum):
        if SMA_V15[i] != 0:
            sma_v = SMA_V3[i] / SMA_V15[i]
        else:
            sma_v = 0
        SMA_V.append(sma_v)
    #求ADX(3)
    DM_PO = np.zeros(LenWavePrice)
    DM_NE = np.zeros(LenWavePrice)
    B = np.zeros(LenWavePrice)
    C = np.zeros(LenWavePrice)
    A = np.zeros(LenWavePrice)
    DM_PO[0],DM_NE[0],B[0],C[0] = 0,0,0,0
    for i in range(1,LenWavePrice):
        DM_PO[i] = WaveHigh[i] - WaveHigh[i-1]
        DM_NE[i] = WaveLow[i-1] - WaveLow[i]
        B[i] = abs((WaveHigh[i] - WavePrice[i-1]))
        C[i] = abs((WaveLow[i] - WavePrice[i-1]))

    for i in range(LenWavePrice):
        if DM_PO[i] > 0 and DM_PO[i] > DM_NE[i]:
            DM_PO[i] = DM_PO[i]
        else:
            DM_PO[i] = 0
        if DM_NE[i] > 0 and DM_NE[i] > DM_PO[i]:
            DM_NE[i] = DM_NE[i]
        else:
            DM_NE[i] = 0
    for i in range(LenWavePrice):
        A[i] = WaveHigh[i] - WaveLow[i]
    TR_r = np.max((A,B),axis=0)
    TR = np.max((TR_r,C),axis=0)
    PDI = []
    MDI = []
    for i in range(13,LenWavePrice):
        if sum(TR[i-13:i+1]) != 0:
            pdi = (sum(DM_PO[i-13:i+1]) / sum(TR[i-13:i+1])) * 100
        else:
            pdi = 0
        PDI.append(pdi)
        if sum(TR[i-13:i+1]) != 0:
            mdi = (sum(DM_NE[i-13:i+1]) / sum(TR[i-13:i+1])) * 100
        else:
            mdi = 0
        MDI.append(mdi)
    DX = []
    for i in range(LenWavePrice-13):
        if abs(PDI[i] + MDI[i]) != 0:
            dx = (100 * abs(PDI[i] - MDI[i])) / abs(PDI[i] + MDI[i])
        else:
            dx = 0
        DX.append(dx)
    ADX_3R = [DX[0] + 1e-6]
    for i in range(1,LenWavePrice - 13):
        ADX_3R.append((ADX_3R[i-1] * 2 + DX[i]) / 3)
    ADX_3 = [0] * 13
    ADX_3.extend(ADX_3R)

    ADX_15R = [DX[0]]
    for i in range(1,LenWavePrice-13):
        ADX_15R.append((ADX_15R[i-1] * 14 + DX[i]) / 15)
    ADX_15 = [0] * 13
    ADX_15.extend(ADX_15R)
    ATR_3 = [TR[0]]
    for i in range(1,LenWavePrice):
        ATR_3.append((ATR_3[i-1] * 2 + TR[i]) / 3)

    ATR_15 = [TR[0]]
    for i in range(1,LenWavePrice):
        ATR_15.append((ATR_15[i-1] * 14 + TR[i]) / 15)

    ATR_R = [0]
    for i in range(1,LenWavePrice):
        if ATR_15[i] != 0:
            atr_r = ATR_3[i] / ATR_15[i]
        else:
            atr_r = 0
        ATR_R.append(atr_r)
    # 计算STO(3)
    L3 = np.zeros(LenWavePrice)
    H3 = np.zeros(LenWavePrice)
    L3[0] = WaveLow[0]
    L3[1] = min(WaveLow[0:2])
    H3[0] = WaveHigh[0]
    H3[1] = max(WaveHigh[0:2])
    for i in range(2,LenWavePrice):
        L3[i] = min(WaveLow[i-2:i+1])
    for i in range(2,LenWavePrice):
        H3[i] = max(WaveHigh[i-2:i+1])
    STO_3 = []
    for i in range(LenWavePrice):
        if H3[i] - L3[i] != 0:
            sto3 = 100 * (WavePrice[i] - L3[i]) / (H3[i] - L3[i])
        else :
            sto3 = 0
        STO_3.append(sto3)
    #计算STO(15)
    L15 = np.zeros(LenWavePrice)
    H15 = np.zeros(LenWavePrice)
    for i in range(13):
        L15[i] = min(WaveLow[0:i+1])
    for i in range(14,LenWavePrice):
        L15[i] = min(WaveLow[i-14:i+1])
    for i in range(13):
        H15[i] = max(WaveHigh[0:i+1])
    for i in range(14,LenWavePrice):
        H15[i] = max(WaveHigh[i-14:i+1])
    STO_15 = []
    for i in range(LenWavePrice):
        if H15[i] - L15[i] != 0:
            sto15 = 100 * (WavePrice[i] - L15[i]) / (H15[i] - L15[i])
        else:
            sto15 = 0
        STO_15.append(sto15)
    #计算STO(ratio)
    STO_R = []
    for i in range(LenWavePrice):
        if STO_15[i] != 0:
            sto_r = STO_3[i] / STO_15[i]
        else:
            sto_r = 0
        STO_R.append(sto_r)
    #计算RSI(3)
    Wave_pd = [0]
    for i in range(1,LenWavePrice):
        Wave_pd.append(WavePrice[i] - WavePrice[i-1])
    RS_3 = [0] * 2
    for i in range(2,LenWavePrice):
        U = 0
        D = 0
        for j in range(i-2,i+1):
            if Wave_pd[j] > 0:
                U = U + Wave_pd[j]
            else:
                D = D - Wave_pd[j]
        if D != 0:
            ud = U / D
        else:
            ud = -1
        RS_3.append(ud)
    RSI_3 = []
    for i in range(LenWavePrice):
        if 1 + RS_3[i] != 0:
            rsi3 = 100 - 100 / (1 + RS_3[i])
        else:
            rsi3 = 100
        RSI_3.append(rsi3)
    RS_15 = [0] * 14
    for i in range(14,LenWavePrice):
        U = 0
        D = 0
        for j in range(i-14,i+1):
            if Wave_pd[j] > 0:
                U = U + Wave_pd[j]
            else:
                D = D - Wave_pd[j]
        if D != 0:
            ud = U / D
        else:
            ud = -1
        RS_15.append(ud)
    RSI_15 = []
    for i in range(LenWavePrice):
        if 1 + RS_15[i] != 0:
            rsi15 = 100 - 100 / (1 + RS_15[i])
        else:
            rsi15 = 100
        RSI_15.append(rsi15)
    RSI_D = []
    for i in range(LenWavePrice):
        RSI_D.append(RSI_3[i] - RSI_15[i])
    MACD = np.array(talib.MACD(np.array(WavePrice,dtype='f8')))[0]
    MACD[0:34] = 0
    MACD = MACD.tolist()
    MOM_3 = [0] * 2
    for i in range(2,LenWavePrice):
        MOM_3.append(WavePrice[i] - WavePrice[i-2])
    MOM_15 = [0] * 14
    for i in range(14,LenWavePrice):
        MOM_15.append(WavePrice[i] - WavePrice[i-14])
    MOM_R = []
    for i in range(LenWavePrice):
        MOM_R.append(MOM_3[i] - MOM_15[i])
    RETURN_ARR = [LPR,HPR,SMA_P,SMA_V,ATR_3,ATR_15,ATR_R,ADX_3,ADX_15,STO_3,STO_15,STO_R,RSI_3,RSI_15,RSI_D,MACD,MOM_3,MOM_15,MOM_R]
    RETURN_ARR = np.array(pd.DataFrame(RETURN_ARR).T).tolist()
    return RETURN_ARR

def ReadDataFromXml(path,state_param= np.pi / 4 , Swing = 1):
    data = pd.read_excel(path,'Sheet1',header=None)
    RawPrice = np.array(data.iloc[:,2]).tolist()
    RawVolum = np.array(data.iloc[:,3]).tolist()
    RawHigh = np.array(data.iloc[:,0]).tolist()
    RawLow = np.array(data.iloc[:,1]).tolist()
    [WavePrice,LenWavePrice] = XIAOBOQUZAO(RawPrice)
    [WaveVolumn,LenWaveVolumn] = XIAOBOQUZAO(RawVolum)
    [WaveHigh,LenWavehigh] = XIAOBOQUZAO(RawHigh)
    [WaveLow,LenWaveLow] = XIAOBOQUZAO(RawLow)
    ALL_ind = DataHandle(WavePrice,WaveVolumn,WaveHigh,WaveLow,LenWavePrice,LenWaveVolumn)
    # pd.DataFrame(ALL_ind).to_csv('Entropy_data/data60.csv')
    ma_60 = WavePrice[0:59]
    for i in range(59,LenWavePrice):
        ma_60.append(sum(WavePrice[i-59:i+1]) / 60)
    ma_20 = WavePrice[0:19]
    for i in range(19,LenWavePrice):
        ma_20.append(sum(WavePrice[i-19:i+1]) / 20)
    ma_5 = WavePrice[0:4]
    for i in range(4,LenWavePrice):
        ma_5.append(sum(WavePrice[i-4:i+1]) / 5)
    slop_5 = [0] * 4
    for i in range(4,LenWavePrice):
        slop_5.append(math.atan(100 * (ma_60[i] - ma_60[i-4]) / ma_60[i-4]) * 180 / np.pi)
    state = np.zeros(LenWavePrice)
    a,b,c = 0,0,0
    for i in range(LenWavePrice):
        if slop_5[i] > state_param:
            state[i] = 1
            a+=1
        elif slop_5[i] >= -state_param and slop_5[i] <= state_param:
            state[i] = 2
            b+=1
        else:
            state[i] = 3
            c+=1
    print(a,b,c)
    train_batches = int(len(ALL_ind) * train_percent)
    valid_batches = int(len(ALL_ind) * valid_percent)
    train_x,valid_x,test_x,train_y,valid_y,test_y = ALL_ind[0:train_batches],ALL_ind[train_batches:train_batches+valid_batches] \
                ,ALL_ind[train_batches+valid_batches:],state[0:train_batches],state[train_batches:train_batches+valid_batches] \
                ,state[train_batches+valid_batches:]
    train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_signal = YieldRate(RawPrice,state,ALL_ind)
    MergeTestPriceAndState = list(map(list,zip(*[RawPrice[train_batches+valid_batches:],state[train_batches+valid_batches:]])))
    test_x = np.concatenate((MergeTestPriceAndState,test_x),axis=1).tolist()
    return train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_x,test_signal

def YieldRate(RawPrice,state,All_ind):
    #计算收益率
    r = len(RawPrice)
    # R = np.zeros(r)
    # lnr = np.zeros(r)
    # R[0] = 0
    # ln_r = np.log(RawPrice)
    # lnr[0] = 0
    # for i in range(1,r):
    #     if RawPrice[i-1] != 0:
    #         R[i] = (RawPrice[i] - RawPrice[i-1]) / RawPrice[i-1]
    #     else:
    #         R[i] = 0
    #     lnr[i] = ln_r[i] - ln_r[i-1]
    # t_signal = np.zeros(r-1)
    # t = lnr[1:]
    # for i in range(r-1):
    #     if t[i] > 0:
    #         t_signal[i] = 1
    #     elif t[i] < 0:
    #         t_signal[i] = -1
    #     else:
    #         t_signal[i] = 0
    # t_operation = np.zeros(r-1)
    # t_position = np.zeros(r-1)
    # for i in range(1,r-1):
    #     if t_signal[i] == 1 and t_position[i-1] == 0:
    #         t_operation[i] = 1
    #     elif t_signal[i] == 1 and t_position[i-1] == 1:
    #         t_operation[i] = 0
    #     elif t_signal[i] == 1 and t_position[i-1] == -1:
    #         t_operation[i] = 1
    #
    #     elif t_signal[i] == -1 and t_position[i-1] == 1:
    #         t_operation[i] = -1
    #     elif t_signal[i] == -1 and t_position[i-1] == 0:
    #         t_operation[i] = -1
    #     elif t_signal[i] == -1 and t_operation[i-1] == -1:
    #         t_operation[i] = 0
    #     else:
    #         t_operation[i] = 0
    #
    #     if t_operation[i] == 0:
    #         t_position[i] = t_position[i-1]
    #     elif t_operation[i] == 1:
    #         t_position[i] = 1 + t_position[i-1]
    #     elif t_operation[i] == -1:
    #         t_position[i] = -1 + t_position[i-1]
    #
    # t_r = np.zeros(r)
    # t_r[0] = 0
    # for i in range(1,r):
    #     t_r[i] = lnr[i] * t_position[i-1]
    signal = []
    for i in range(r-5):
        if RawPrice[i] != 0:
            sig = 10 * (np.mean(RawPrice[i+1:i+6]) - RawPrice[i]) / RawPrice[i]
        else:
            sig = 0
        signal.append(sig)
    train_signal1 = []
    train_signal2 = []
    train_signal3 = []
    train_index1 = []
    train_index2 = []
    train_index3 = []
    for i in range(int(len(All_ind) * train_percent)):
        if state[i] == 1:
            train_signal1.append([signal[i]])
            train_index1.append(All_ind[i][:10])
        elif state[i] == 2:
            train_signal2.append([signal[i]])
            train_index2.append(All_ind[i][:10])
        else:
            train_signal3.append([signal[i]])
            train_index3.append(All_ind[i][:10])
    test_signal = signal[int(len(All_ind) * train_percent):-5]
    return train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_signal


def XIAOBOQUZAO(RawData,PlotShow = False):
    w = pywt.Wavelet('db4')
    noisy_coefs = pywt.wavedec(RawData,w,level=4,mode='per')
    denoised = noisy_coefs[:]
    sigma = mad(noisy_coefs[-1])
    uthresh = sigma * np.sqrt(2 * np.log(len(RawData)))
    denoised[1:] = (pywt.threshold(a,value=uthresh,mode='soft') for a in denoised[1:])
    signal = pywt.waverec(denoised,w,mode='per')[0:len(RawData)]
    # Wave_data
    # plt.figure(1)
    # plt.subplot(2,1,1)
    # plt.plot(RawData)
    # plt.title('Raw Data')
    # plt.grid(True)
    # plt.figure(1)
    # plt.subplot(2,1,2)
    # plt.plot(signal)
    # plt.title('Wave Data')
    # plt.grid(True)
    # plt.show()
    return signal.tolist(),len(signal)
#DataHandle(np.random.rand(500),np.random.rand(500),np.random.rand(500),np.random.rand(500),500,500)

def next_batch(data, batch_size,i):
    if (i+1)*batch_size > len(data):
        return data[i*batch_size:len(data)-1]
    else:
        return data[i*batch_size:(i+1)*batch_size]

def ReadDataByTuShare(code=0,start='2000-01-01',end='2016-12-31',ktype='D',autype='qfq',state_param=np.pi / 4,Swing=0.1):
    df = ts.get_k_data(code,start,end,ktype,autype)
    RawPrice = np.array(df.iloc[:,2]).tolist()
    RawVolum = np.array(df.iloc[:,5]).tolist()
    RawHigh = np.array(df.iloc[:,3]).tolist()
    RawLow = np.array(df.iloc[:,4]).tolist()
    [WavePrice,LenWavePrice] = XIAOBOQUZAO(RawPrice)
    [WaveVolumn,LenWaveVolumn] = XIAOBOQUZAO(RawVolum)
    [WaveHigh,LenWavehigh] = XIAOBOQUZAO(RawHigh)
    [WaveLow,LenWaveLow] = XIAOBOQUZAO(RawLow)
    ALL_ind = DataHandle(WavePrice,WaveVolumn,WaveHigh,WaveLow,LenWavePrice,LenWaveVolumn)
    # pd.DataFrame(ALL_ind).to_csv('Entropy_data/data60.csv')
    ma_120 = WavePrice[0:120]
    for i in range(120,LenWavePrice):
        ma_120.append(sum(WavePrice[i-120:i]) / 120)
    ma_60 = WavePrice[0:60]
    for i in range(60,LenWavePrice):
        ma_60.append(sum(WavePrice[i-60:i]) / 60)
    ma_20 = WavePrice[0:20]
    for i in range(20,LenWavePrice):
        ma_20.append(sum(WavePrice[i-20:i]) / 20)
    ma_5 = WavePrice[0:5]
    for i in range(5,LenWavePrice):
        ma_5.append(sum(WavePrice[i-5:i]) / 5)
    slop_5 = [0] * 4
    for i in range(4,LenWavePrice):
        slop_5.append(math.atan(100 * ((ma_120[i] - ma_120[i-4]) / ma_120[i-4]) * 180 / np.pi))
    state = np.zeros(LenWavePrice)
    a,b,c = 0,0,0
    for i in range(LenWavePrice):
        if slop_5[i] > state_param:
            state[i] = 1
            a+=1
        elif slop_5[i] >= -state_param and slop_5[i] <= state_param:
            state[i] = 2
            b+=1
        else:
            state[i] = 3
            c+=1
    print(a,b,c)
    train_batches = int(len(ALL_ind) * train_percent)
    valid_batches = int(len(ALL_ind) * valid_percent)
    train_x,valid_x,test_x,train_y,valid_y,test_y = ALL_ind[0:train_batches],ALL_ind[train_batches:train_batches+valid_batches] \
                ,ALL_ind[train_batches+valid_batches:],state[0:train_batches],state[train_batches:train_batches+valid_batches] \
                ,state[train_batches+valid_batches:]
    train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_signal = YieldRate(RawPrice,state,ALL_ind)
    MergeTestPriceAndState = list(map(list,zip(*[RawPrice[train_batches+valid_batches:],state[train_batches+valid_batches:]])))
    test_x = np.concatenate((MergeTestPriceAndState,test_x),axis=1).tolist()
    return train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_x,test_signal