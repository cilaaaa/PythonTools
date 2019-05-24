__author__ = 'Cila'
import numpy as np
from data_model import StockDataSet
import os
import matplotlib.pyplot as plt

import tflearn

def gcd(feature,target):
    feature = feature.T
    delt = np.abs(target-feature).T
    min = np.min(delt)
    max = np.max(delt)
    r = (min + 0.5 * max) / (delt + 0.5 * max)
    r = np.mean(r,axis=0)
    return r

def normalize(fit,data):
    # min = np.min(fit,axis=0)
    # max = np.max(fit,axis=0)
    # data = -1 + 2 * (data - min) / (max - min)
    return data

def denormalize(fit,predict):
    # min = np.min(fit,axis=0)
    # max = np.max(fit,axis=0)
    # predict = (predict * (max - min) + max + min) / 2
    return predict

def accuracy(data,label):
    d = 0
    for i in range(len(data)):
        if data[i] * label[i] >= 0:
            d += 1
    return d / len(data) * 100.0

stock = StockDataSet('',input_size=10)
data = stock.train_X
x1 = normalize(data,data)
label = np.array(stock.raw_price[201:-4])
y = normalize(label,label)
test_x = stock.GetTestData('')
test_x = normalize(data,test_x)
test_label = stock.raw_test_price[1:]
gcd = gcd(x1,y)
y = y.reshape((-1,1))
x1 = x1 * gcd
test_x = test_x * gcd

input_layer = tflearn.input_data(shape=[None,10])
net = tflearn.batch_normalization(input_layer)
hidden_cur_cnt = 400
net = tflearn.fully_connected(net, hidden_cur_cnt, activation='relu',regularizer='L2',weights_init='xavier')
net = tflearn.dropout(net, 0.75)
for i in range(4 - 2):
    if hidden_cur_cnt > 2:
        hidden_next_cnt = int(hidden_cur_cnt / 2)
    else:
        hidden_next_cnt = 2
    net = tflearn.fully_connected(net, hidden_next_cnt, activation='relu',regularizer='L2',weights_init='xavier')
    net = tflearn.dropout(net, 0.75)
net = tflearn.fully_connected(net,1)
optimizer = tflearn.SGD(learning_rate=0.1, lr_decay=0.99, decay_step=1000)
net = tflearn.regression(net, optimizer=optimizer,metric='accuracy',loss='mean_square')
model = tflearn.DNN(net,tensorboard_verbose=0,tensorboard_dir='C:/Users/Cila/Desktop/tensorflow/QuantitativeTransaction/tensorboard')
#int(3e+4 / (len(x1) / FLAGS.batch_size))
model.fit(x1,y,n_epoch=100,
          show_metric=True,
          batch_size=32)
model.save(os.path.join("logs", 'GDC_ANN'))
model.load(os.path.join("logs", 'GDC_ANN'))
raw_test_pre = []
for i in range(len(test_x)):
    if i >= 5:
        test_batch_x = normalize(data,[test_x[i]])
        raw_test_pre.append(model.predict(test_batch_x)[0])
test = model.predict(x1)
train_prediction = denormalize(label,model.predict(x1))
prediction = denormalize(label,raw_test_pre)
plt.figure('capital')
plt.plot(prediction, 'b',test_label,'r',train_prediction,'g',label,'y')
plt.grid(True)
plt.show()

# def trade(price,raw_price):
#     position = np.zeros(len(price))
#     position[0] = 1
#     for i in range(len(price))