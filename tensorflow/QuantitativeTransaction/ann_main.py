__author__ = 'Cila'
import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

from data_model import StockDataSet
from ann_model import ANN
from rnn_model import LstmRNN

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 19, "Input size [10]")
flags.DEFINE_integer("num_steps", 1, "Num of steps [30]")
flags.DEFINE_integer("layer", 4, "Num of layer [1]")
flags.DEFINE_integer("hidden_size", 40, "Size of hidden_units")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 3e-5, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 100, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 20000, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", 'RB1805', "Target stock symbol [None]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_data(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.01):
    if target_symbol is not None:
        return [StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                valid_ratio=0.0,
                logisticRegression=True)]
    return [StockDataSet(stock_sym='data15min',
                     sheet='Sheet1',
                     input_size=input_size,
                     num_steps=num_steps,
                     logisticRegression=False)]

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    stock_data_list = load_data(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
        )
    train_signal1 = []
    train_signal2 = []
    train_signal3 = []
    train_index1 = []
    train_index2 = []
    train_index3 = []
    for i in range(len(stock_data_list[0].train_X)):
        if stock_data_list[0].train_state[i] == 1:
            train_index1.append(stock_data_list[0].train_X[i])
            train_signal1.append(stock_data_list[0].train_y[i])
        elif stock_data_list[0].train_state[i] == 2:
            train_index2.append(stock_data_list[0].train_X[i])
            train_signal2.append(stock_data_list[0].train_y[i])
        else:
            train_index3.append(stock_data_list[0].train_X[i])
            train_signal3.append(stock_data_list[0].train_y[i])

    test_x = stock_data_list[0].GetTestData('RB18051128')
    prediction = np.zeros(len(test_x))
    a = []
    b = []
    c = []
    a_index = 0
    b_index = 0
    c_index = 0
    for i in range(len(test_x)):
        if stock_data_list[0].test_state[i] == 1:
            a.append(test_x[i])
            prediction[i] = 1
        elif stock_data_list[0].test_state[i] == 2:
            b.append(test_x[i])
            prediction[i] = 2
        else:
            c.append(test_x[i])
            prediction[i] = 3
    x1 = standard_scale(train_index1,train_index1)
    x2 = standard_scale(train_index2,train_index2)
    x3 = standard_scale(train_index3,train_index3)
    if(len(a)>0):
        a = standard_scale(train_index1,a)
    if(len(b)>0):
        b = standard_scale(train_index2,b)
    if(len(c)>0):
        c = standard_scale(train_index3,c)

    name = "stock_rnn_Ann_%s_%s" % (FLAGS.stock_symbol,'model1')
    model1,graph1 = create_network()
    if FLAGS.train:
        with tf.Session(graph=graph1) as sess:
            model1.session = sess
            model1.fit(x1,train_signal1,n_epoch=int(1e+4 / (len(x1) / FLAGS.batch_size)),show_metric=True,batch_size=FLAGS.batch_size)
            model1.save(os.path.join("logs", name))
    if len(a)> 0:
        with tf.Session(graph=graph1) as sess:
            model1.session = sess
            model1.load(os.path.join("logs", name))
            prediction1 = model1.predict(a)
            prediction1 = np.argmax(prediction1,1)

    name = "stock_rnn_Ann_%s_%s" % (FLAGS.stock_symbol,'model2')
    model2,graph2 = create_network()
    if FLAGS.train:
        with tf.Session(graph=graph2) as sess:
            model2.session = sess
            model2.fit(x2,train_signal2,n_epoch=int(1e+4 / (len(x2) / FLAGS.batch_size)),show_metric=True,batch_size=FLAGS.batch_size)
            model2.save(os.path.join("logs", name))
    if len(b)> 0:
        with tf.Session(graph=graph2) as sess:
            model2.session = sess
            model2.load(os.path.join("logs", name))
            prediction2 = model2.predict(b)
            prediction2 = np.argmax(prediction2,1)

    name = "stock_rnn_Ann_%s_%s" % (FLAGS.stock_symbol,'model3')
    model3,graph3 = create_network()
    if FLAGS.train:
        with tf.Session(graph=graph3) as sess:
            model3.session = sess
            model3.fit(x3,train_signal3,n_epoch=int(1e+4 / (len(x3) / FLAGS.batch_size)),show_metric=True,batch_size=FLAGS.batch_size)
            model3.save(os.path.join("logs", name))
    if len(c)> 0:
        with tf.Session(graph=graph3) as sess:
            model3.session = sess
            model3.load(os.path.join("logs", name))
            prediction3 = model3.predict(c)
            prediction3 = np.argmax(prediction3,1)

    for i in range(len(test_x)):
        if prediction[i] == 1:
            prediction[i] = prediction1[a_index]
            a_index += 1
        elif prediction[i] == 2:
            prediction[i] = prediction2[b_index]
            b_index += 1
        else:
            prediction[i] = prediction3[c_index]
            c_index += 1
    trade(stock_data_list,prediction)

def create_network():
    graph = tf.Graph()
    with graph.as_default():
        input_layer = tflearn.input_data(shape=[None,FLAGS.input_size])
        hidden_cur_cnt = 512
        dense1 = tflearn.fully_connected(input_layer, hidden_cur_cnt, activation='leaky_relu',regularizer='L2')
        dropout1 = tflearn.dropout(dense1, 0.5)
        for i in range(FLAGS.layer - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            dense_middele = tflearn.fully_connected(dropout1, hidden_next_cnt, activation='leaky_relu',regularizer='L2')
            dropout1 = tflearn.dropout(dense_middele, 0.5)
        net = tflearn.fully_connected(dropout1,3,activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='mean_square')
        model = tflearn.DNN(net,tensorboard_verbose=0)
    return model,graph

def standard_scale(fit,data):
    preprocessor = prep.StandardScaler().fit(fit)
    data = preprocessor.transform(data)
    return data

def trade(dataset_list,signal):
    capital = 400000
    RawPrice = dataset_list[0].raw_price
    f60_position = np.zeros(len(signal))
    f60_position[0] = 0
    for i in range(1,len(signal)):
        if dataset_list[0].lR:
            if signal[i] <= 1 and f60_position[i-1] >= 0:
                f60_position[i] = 1
            elif  signal[i] >= 1 and f60_position[i-1] <= 0:
                f60_position[i] = -1
            elif  (1-signal[i]) * f60_position[i-1] < 0:
                f60_position[i] = 0
        else:
            if signal[i] >= 0 and f60_position[i-1] >= 0:
                f60_position[i] = 1
            elif  signal[i] <= 0 and f60_position[i-1] <= 0:
                f60_position[i] = -1
            elif  signal[i] * f60_position[i-1] < 0:
                f60_position[i] = 0
    tradetimes = 0
    f60_fee = np.zeros(len(signal))
    f60_slide = np.zeros(len(signal))
    bigpoint = 300
    for i in range(1,len(signal)):
        if(f60_position[i] != 0):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - RawPrice[i]* 1e-4 * bigpoint
            f60_slide[i] = - 0.2 * bigpoint
            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint

    f60_cost = np.add(f60_fee,f60_slide)

    print(tradetimes)

    chajia = np.diff(RawPrice,axis=0)
    d=0
    test_signal = dataset_list[0].raw_test_sig[FLAGS.num_steps-1:]
    chajia = chajia[-len(signal) + 1:]
    if dataset_list[0].lR:
        print(100.0 * np.sum(np.argmax(test_signal,1) == signal) / len(signal),'%')
    else:
        for i in range(len(test_signal)):
            if test_signal[i] * signal[i] < 0:
                d += 1
        print((1-d / len(test_signal)) *100,'%')

    f60_r = np.cumsum(np.multiply(f60_position[:-1],chajia) * bigpoint + f60_cost[1:]) + capital
    plt.figure(1)
    plt.plot(test_signal,'b',signal,'r')
    plt.grid(True)
    plt.figure(2)
    plt.plot(f60_r, color='b')
    plt.grid(True)
    plt.show()

if not os.path.exists("logs"):
    os.mkdir("logs")

if __name__ == '__main__':
    tf.app.run()