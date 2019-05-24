__author__ = 'Cila'
import pandas as pd
import numpy as np
import time

TickDataName = 'okex_EOS_future_quarter_1min.csv'
TradeDataName = 'WangQi.xlsx'
PolicyName = 'PolicyBtcFuture0111-180-24-0.05'
Buy = 'KaiDuo'
Sell = 'KaiKong'
BuyPing = 'PingDuo'
SellPing = 'PingKong'

def StrToTime(str):
    return time.strptime(str[0:19],"%Y-%m-%d %H:%M:%S")

TickData = np.array(pd.read_csv(TickDataName))
TradeData = np.array(pd.read_excel(TradeDataName))
SigData = np.zeros((len(TickData)-1,2))
MarkI = 0
TradeI = 0
for tradeDetail in TradeData:
    if tradeDetail[0] == PolicyName:
        InTime = StrToTime(tradeDetail[5])
        OutTime = StrToTime(tradeDetail[12])
        InType = tradeDetail[4]
        OutType = tradeDetail[11]
        InPrice = tradeDetail[8]
        OutPrice = tradeDetail[15]
        if InPrice > 0 and OutPrice > 0:
            for i in range(MarkI,len(TickData)-2):
                if StrToTime(TickData[i][0]) <= InTime and StrToTime(TickData[i+1][0]) >= InTime:
                    if i == TradeI:
                        TradeI = i + 1
                    else:
                        TradeI = i
                    if InType == Buy:
                        SigData[TradeI][0] = 1
                        SigData[TradeI][1] = InPrice
                    elif InType == Sell:
                        SigData[TradeI][0] = 2
                        SigData[TradeI][1] = InPrice
                    MarkI = i
                elif StrToTime(TickData[i][0]) <= OutTime and StrToTime(TickData[i+1][0]) >= OutTime:
                    if i == TradeI:
                        TradeI = i + 1
                    else:
                        TradeI = i
                    if OutType == BuyPing:
                        SigData[TradeI][0] = 3
                        SigData[TradeI][1] = OutPrice
                    elif OutType == SellPing:
                        SigData[TradeI][0] = 4
                        SigData[TradeI][1] = OutPrice
df = pd.DataFrame(SigData).to_csv(PolicyName + "Sig.csv",index=False,header=False)