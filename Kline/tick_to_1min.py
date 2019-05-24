# -*- coding: utf-8 -*-
import pandas as pd
import os
import time

WriteType = "new"
StartDate = time.strptime("2018-11-01","%Y-%m-%d")
EndDate = time.strptime("2018-11-30","%Y-%m-%d")

def load_file(file_name):
    # [tick_code],[tick_timestamp] ,[tick_time],[tick_ask1],[tick_asks1],[tick_bid1],[tick_bids1]
    # df = pd.read_excel(file_name)
    df = pd.read_csv(file_name,encoding='utf-8-sig')
    return df

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir + "/"):
        if root == file_dir + "/":
            for file in files:
                L.append(os.path.join(root, file))
    return L

def convert_to_1min(file_path,title):
    csvs = file_name(file_path)
    init = False
    for csv in csvs:
        date = time.strptime(csv.split('%')[-1].split('.')[0],"%Y%m%d")
        if date >= StartDate and date <= EndDate:
            raw_data = load_file(csv)
            if not init:
                init = True
                data = raw_data
            else:
                data = data.append(raw_data)
    last_minute = -1
    global open ,high, low, close,temp_dict,index1
    temp_dict = {}
    result = pd.DataFrame(columns=['datetime',"open", 'high', "low", 'close'])
    index1 = 0
    # high = -1
    # low = 999999
    data = data.values.tolist()
    for i in range(len(data)):
        timeStr = str(data[i][0])[:19].replace("D"," ")
        ts = time.mktime(time.strptime(timeStr, '%Y-%m-%d %H:%M:%S'))
        if i==0:
            last_minute = int(ts % 3600 / 60)
            close = round(data[i][6],4)
            open = close
            high = close
            low = close
            temp_dict['open'] = open
            temp_dict['close'] = close
            temp_dict['high'] = high
            temp_dict['low'] = low
            temp_dict['time'] = timeStr
        else:
            new_minute = int(ts % 3600 / 60)
            if new_minute != last_minute:
                last_minute = new_minute
                result.loc[index1] = {"datetime":temp_dict['time'], 'open':temp_dict['open'],
                                 "high":temp_dict['high'], 'low':temp_dict['low'], 'close':temp_dict['close']}
                index1 += 1

                close = round(data[i][6],4)
                temp_dict['open'] = close
                open = close
                high = close
                low = close
                temp_dict['open'] = open
                temp_dict['close'] = close
                temp_dict['high'] = high
                temp_dict['low'] = low
                temp_dict['time'] = timeStr
            else: # 同一分钟
                close = round(data[i][6],4)
                temp_dict['close'] = close
                if close > high:
                    high = close
                elif close < low:
                    low = close
                # temp_dict['open'] = open
                temp_dict['close'] = close
                temp_dict['high'] = high
                temp_dict['low'] = low
    if WriteType == "append":
        df = pd.read_csv(title,dtype={'datetime':str,'open':float,'high':float,'low':float,'close':float})
        result = df.append(result)
    result.to_csv(title,index=False)


if __name__ == '__main__':
    # convert_to_1min('spot_btc_1107-1206.csv',"okex_spot_1min.xlsx")
    convert_to_1min('E:\\futureData\cfIF1811',"cfIF1811_future_quarter_1min.csv")
    # df = pd.read_csv('okex_eos_future_quarter_1min.csv',dtype={'datetime':str,'open':float,'high':float,'low':float,'close':float})
    # df.to_csv('okex_eos_future_quarter_1min.csv',index=False)