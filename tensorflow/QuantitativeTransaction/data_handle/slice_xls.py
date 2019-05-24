__author__ = 'Cila'
import os
import pandas as pd
import numpy as np

stock_sym = 'HC.csv'
sheet = 'hc'
type = stock_sym.split('.')
test_radio = 0.2
if type[-1] == 'csv':
    raw_df = pd.read_csv(os.path.join("../data", "%s" % stock_sym))
else:
    raw_df = pd.read_excel(os.path.join("../data", "%s" % stock_sym),sheet,header=None)
RawPrice = np.array(raw_df.iloc[1:,4])
RawVolum = np.array(raw_df.iloc[1:,5])
RawHigh = np.array(raw_df.iloc[1:,2])
RawLow = np.array(raw_df.iloc[1:,3])
Date = np.array(raw_df.iloc[1:,0])


new_data = [RawPrice,RawVolum,RawHigh,RawLow,Date]

data = np.array(pd.DataFrame(new_data).T)
train_index =int(len(data) * (1-test_radio))
print(train_index)
train_data = data[0:train_index]
test_data = data[train_index:]
train_df = pd.DataFrame(train_data)
train_df.to_csv(os.path.join("../data",'train_%s' % stock_sym),header=None,index=None)
test_df = pd.DataFrame(test_data)
test_df.to_csv(os.path.join("../data",'test_%s' % stock_sym),header=None,index=None)