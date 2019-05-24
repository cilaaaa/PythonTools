__author__ = 'Cila'
import numpy as np
import talib
import pandas as pd

def DataHandle(WavePrice,WaveVolum,WaveHigh,WaveLow,LenWavePrice,LenWaveVolum,Predict=False):
    if Predict:
        LPR = min(WavePrice) / WavePrice[-1]
    else:
        LPR = [0] * 199
        for i in range(199,LenWavePrice):
            if WavePrice[i] != 0:
                lpr = min(WavePrice[(i-199):i+1]) / WavePrice[i]
            else:
                lpr = 0
            LPR.append(lpr)
    if Predict:
        HPR = max(WavePrice) / WavePrice[-1]
    else:
        HPR = [0] * 199
        for i in range(199,LenWavePrice):
            if WavePrice[i] != 0:
                hpr = max(WavePrice[(i-199):i+1]) / WavePrice[i]
            else:
                hpr = 0
            HPR.append(hpr)

    if Predict:
        SMA_P3 = sum(WavePrice[-3:]) / 3
    else:
        SMA_P3 = [0] * 2
        for i in range(2,LenWavePrice):
            sma_p3 = sum(WavePrice[(i-2):i+1]) / 3
            SMA_P3.append(sma_p3)

    if Predict:
        SMA_P15 = sum(WavePrice[-15:]) / 15
    else:
        SMA_P15 = [0] * 14
        for i in range(14,LenWavePrice):
            sma_p15 = sum(WavePrice[(i-14):i+1]) / 15
            SMA_P15.append(sma_p15)

    if Predict:
        SMA_P = SMA_P3 / SMA_P15
    else:
        SMA_P = []
        for i in range(LenWavePrice):
            if SMA_P15[i] != 0:
                sma_p = SMA_P3[i] / SMA_P15[i]
            else:
                sma_p = 0
            SMA_P.append(sma_p)

    if Predict:
        SMA_V3 = sum(WaveVolum[-3:]) / 3
    else:
        SMA_V3 = [0] * 2
        for i in range(2,LenWaveVolum):
            sma_v3 = sum(WaveVolum[(i-2):i+1]) / 3
            SMA_V3.append(sma_v3)

    if Predict:
        SMA_V15 = sum(WaveVolum[-15:]) / 15
    else:
        SMA_V15 = [0] * 14
        for i in range(14,LenWaveVolum):
            sma_v15 = sum(WaveVolum[(i-14):i+1]) / 15
            SMA_V15.append(sma_v15)

    if Predict:
        SMA_V = SMA_V3 / SMA_V15
    else:
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
    if Predict:
        ADX_3 = ADX_3[-1]

    ADX_15R = [DX[0]]
    for i in range(1,LenWavePrice-13):
        ADX_15R.append((ADX_15R[i-1] * 14 + DX[i]) / 15)
    ADX_15 = [0] * 13
    ADX_15.extend(ADX_15R)
    if Predict:
        ADX_15 = ADX_15[-1]

    ATR_3 = [TR[0]]
    for i in range(1,LenWavePrice):
        ATR_3.append((ATR_3[i-1] * 2 + TR[i]) / 3)
    if Predict:
        ATR_3 = ATR_3[-1]

    ATR_15 = [TR[0]]
    for i in range(1,LenWavePrice):
        ATR_15.append((ATR_15[i-1] * 14 + TR[i]) / 15)
    if Predict:
        ATR_15 = ATR_15[-1]

    if Predict:
        ATR_R = ATR_3 / ATR_15
    else:
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
    if Predict:
        STO_3 = STO_3[-1]
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
    if Predict:
        STO_15 = STO_15[-1]
    #计算STO(ratio)
    if Predict:
        if STO_15 != 0:
            STO_R = STO_3 / STO_15
        else:
            STO_R = 0
    else:
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
    if Predict:
        if 1 + RS_3[-1] != 0:
            RSI_3 = 100 - 100 / (1 + RS_3[-1])
        else:
            RSI_3 = 100
    else:
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
    if Predict:
        if 1 + RS_15[-1] != 0:
            RSI_15 = 100 - 100 / (1 + RS_15[-1])
        else:
            RSI_15 = 100
    else:
        RSI_15 = []
        for i in range(LenWavePrice):
            if 1 + RS_15[i] != 0:
                rsi15 = 100 - 100 / (1 + RS_15[i])
            else:
                rsi15 = 100
            RSI_15.append(rsi15)
    if Predict:
        RSI_D = RSI_3 - RSI_15
    else:
        RSI_D = []
        for i in range(LenWavePrice):
            RSI_D.append(RSI_3[i] - RSI_15[i])
    MACD = np.array(talib.MACD(np.array(WavePrice,dtype='f8')))[0]
    MACD[0:34] = 0
    MACD = MACD.tolist()
    if Predict:
        MACD = MACD[-1]
    if Predict:
        MOM_3 = WavePrice[-1] - WavePrice[-3]
    else:
        MOM_3 = [0] * 2
        for i in range(2,LenWavePrice):
            MOM_3.append(WavePrice[i] - WavePrice[i-2])
    if Predict:
        MOM_15 = WavePrice[-1] - WavePrice[-15]
    else:
        MOM_15 = [0] * 14
        for i in range(14,LenWavePrice):
            MOM_15.append(WavePrice[i] - WavePrice[i-14])
    if Predict:
        MOM_R = MOM_3 - MOM_15
    else:
        MOM_R = []
        for i in range(LenWavePrice):
            MOM_R.append(MOM_3[i] - MOM_15[i])
    RETURN_ARR = [LPR,HPR,SMA_P,SMA_V,ATR_3,ATR_15,ATR_R,ADX_3,ADX_15,STO_3,STO_15,STO_R,RSI_3,RSI_15,RSI_D,MACD,MOM_3,MOM_15,MOM_R]
    if not Predict:
        RETURN_ARR = np.array(pd.DataFrame(RETURN_ARR).T).tolist()
    return RETURN_ARR