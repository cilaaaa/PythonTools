__author__ = 'Cila'
import pandas as pd
import os

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

def convert_to_1min(file_path,title):
    csvs = file_name(file_path)
    for csv in csvs:
        df = pd.read_csv(csv)
        df['bidPrice'] = list(map(lambda x:round(x,3),df['bidPrice']))
        df['askPrice'] = list(map(lambda x:round(x,3),df['askPrice']))
        df['lastPrice'] = list(map(lambda x:round(x,4),df['lastPrice']))
        df.to_csv(csv,index=False,encoding='utf-8')
        print(csv)

if __name__ == '__main__':
    # convert_to_1min('spot_btc_1107-1206.csv',"okex_spot_1min.xlsx")
    convert_to_1min('E:\Bitcoin\okex2019',"okex_eos_future_quarter_1min.csv")
    # df = pd.read_csv('okex_eos_future_quarter_1min.csv',dtype={'datetime':str,'open':float,'high':float,'low':float,'close':float})
    # df.to_csv('okex_eos_future_quarter_1min.csv',index=False)