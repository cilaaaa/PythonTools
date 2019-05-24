__author__ = 'Cila'
import pandas as pd
import os
import time
import datetime
def load_file(file_name):
    # [tick_code],[tick_timestamp] ,[tick_time],[tick_ask1],[tick_asks1],[tick_bid1],[tick_bids1]
    # df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)
    return df

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir + "/"):
        if root == file_dir + "/":
            for file in files:
                L.append(os.path.join(root, file))
    return L

def convert_to_1s(file_path):
    csvs = file_name(file_path)
    for csv in csvs:
        if csv.split('/')[-1][:6] != 'XBTUSD':
            raw_data = load_file(csv)
            raw_data = raw_data[raw_data['symbol'] == 'XBTUSD']
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], format="%Y-%m-%dD%H:%M:%S.%f")
            raw_data.index = raw_data['timestamp']
            raw_data = raw_data.resample('1S',).first().dropna()
            #
            # markSecond = ""
            # for i in range(len(raw_data)):
            #     strtime = str(raw_data.loc[i, "timestamp"])[:-10].replace("D"," ")
            #     sec = strtime[-2:]
            #     if sec != markSecond:
            #         markSecond = sec
            #         data.append([strtime,str(raw_data.loc[i, "symbol"]), float(raw_data.loc[i, "bidSize"]),float(raw_data.loc[i, "bidPrice"]), float(raw_data.loc[i, "askPrice"]), float(raw_data.loc[i, "askSize"])])
            # new_df = pd.DataFrame(data,columns=['timestamp','symbol','bidSize','bidPrice','askPrice','askSize'],index=False)
            name = csv[:-12] + 'XBTUSD-' + csv[-12:-4] + '.csv'
            raw_data.index = range(len(raw_data))
            raw_data.to_csv(name, index=False)
            os.remove(csv)

convert_to_1s('E:/Bitcoin/bitmex')