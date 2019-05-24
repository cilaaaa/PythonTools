__author__ = 'Cila'

import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import pymysql as mysql

from data_model import StockDataSet

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 10, "Input size [10]")
flags.DEFINE_integer("num_steps", 20 , "Num of steps [30]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate",0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("epoch", 50, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_string("stock_symbol", 'RB1805', "Target stock symbol [None]")
flags.DEFINE_string("type", 'RB', "Target stock type [None]")
flags.DEFINE_string("contract_multiplier", 10, "Target stock contract_multiplier [None]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("logisticRegression", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_data(input_size, num_steps, k=None, target_symbol=None, valid_ratio=0.0):
    if target_symbol is not None:
        return [StockDataSet(
                target_symbol,
                initCsv=False,
                input_size=input_size,
                num_steps=num_steps,
                logisticRegression=FLAGS.logisticRegression,
                valid_ratio=valid_ratio)]
    return [StockDataSet(stock_sym='data15min',
                     sheet='Sheet1',
                     input_size=input_size,
                     num_steps=num_steps,
                     logisticRegression=FLAGS.logisticRegression,
                     valid_ratio=valid_ratio)]


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    #tflearn lstm
    name = "stock_rnn_lstm_%s" % (FLAGS.stock_symbol)

    net = tflearn.input_data(shape=[None,FLAGS.num_steps,FLAGS.input_size])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.dropout(net, 0.75)
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.dropout(net, 0.75)
    net = tflearn.lstm(net, 32)
    net = tflearn.dropout(net, 0.75)
    net = tflearn.fully_connected(net,1)
    net = tflearn.regression(net, optimizer='adam',loss='mean_square')
    model = tflearn.DNN(net,tensorboard_verbose=0)

    stock_data_list = load_data(
                FLAGS.input_size,
                FLAGS.num_steps,
                k=FLAGS.stock_count,
                target_symbol=FLAGS.stock_symbol,
            )
    train_x = stock_data_list[0].train_X
    if FLAGS.train:
        x1 = standard_scale(train_x,train_x)
        x = []
        for j in range(FLAGS.num_steps,len(x1)+1):
            temp = x1[j-FLAGS.num_steps:j]
            x.append(temp)
        x1 = np.array(x)
        train_y = stock_data_list[0].train_y[FLAGS.num_steps-1:]
        model.fit(x1,train_y,n_epoch=FLAGS.epoch,validation_set=None,show_metric=True,snapshot_step=100,batch_size=FLAGS.batch_size)
        model.save(os.path.join("logs", name))
    model.load(os.path.join("logs", name))
    test_x = stock_data_list[0].GetTestData('RB18051128')
    test_x = standard_scale(train_x,test_x)
    x = []
    test_x = np.append(train_x[-FLAGS.num_steps+1:],test_x,axis=0)
    for j in range(FLAGS.num_steps,len(test_x)+1):
        temp = test_x[j-FLAGS.num_steps:j]
        x.append(temp)
    test_x = np.array(x)
    prediction = model.predict(test_x)
    trade(stock_data_list,prediction)
    plot(stock_data_list)

def standard_scale(fit,data):
    preprocessor = prep.StandardScaler().fit(fit)
    data = preprocessor.transform(data)
    return data

def trade(dataset_list,signal):
    RawPrice = dataset_list[0].raw_test_price
    Date = dataset_list[0].date
    predict_position = np.zeros(len(signal))
    predict_position[0] = 0
    for i in range(1,len(signal)):
        if signal[i] >= 0 and predict_position[i-1] >= 0:
            predict_position[i] = 1
        elif  signal[i] <= 0 and predict_position[i-1] <= 0:
            predict_position[i] = -1
        elif  signal[i] * predict_position[i-1] < 0:
            predict_position[i] = 0
    raw_sig = dataset_list[0].raw_test_sig
    raw_position = np.zeros(len(signal))
    raw_position[0] = 0
    for i in range(1,len(raw_sig)):
        if raw_sig[i] >= 0 and raw_position[i-1] >= 0:
            raw_position[i] = 1
        elif  raw_sig[i] <= 0 and raw_position[i-1] <= 0:
            raw_position[i] = -1
        elif  raw_sig[i] * raw_position[i-1] < 0:
            raw_position[i] = 0
    saveMysql(RawPrice,Date,signal,raw_sig,predict_position,raw_position)

def saveMysql(price,date,signal,raw_sig,pre_position,raw_position):
    db = mysql.connect("localhost","root","init1234","entropy")
    for i in range(len(price)):
        check_sql = "SELECT * FROM trade WHERE type = '%s' and date = '%s'" % (FLAGS.type,date[i])
        cursor = db.cursor()
        cursor.execute(check_sql)
        results = cursor.fetchone()
        if results:
            sql = "UPDATE trade SET price = '%.2f',position = '%d',sig = '%f',raw_sig = '%f',raw_position='%d' WHERE date = '%s'" % (price[i],pre_position[i],signal[i],raw_sig[i],raw_position[i],date[i])
        else:
            sql = "INSERT INTO trade(price,position,raw_position,sig,raw_sig, type, contract_multiplier,date) VALUES ('%.2f', '%d','%d','%f','%f' '%s', '%d','%s')" % (price[i],pre_position[i],raw_position[i],signal[i],raw_sig[i],FLAGS.type,FLAGS.contract_multiplier,date[i])
        try:
           # 执行sql语句
           cursor = db.cursor()
           cursor.execute(sql)
           # 提交到数据库执行
           db.commit()
        except:
           # 如果发生错误则回滚
           db.rollback()
    # 关闭数据库连接
    db.close()

def plot(dataset_list):
    Date = dataset_list[0].date
    db = mysql.connect("localhost","root","init1234","entropy",charset='utf8')
    cursor = db.cursor()
    delete_sql = "DELETE FROM trade_list WHERE type = '%s'" %(FLAGS.type)
    cursor.execute(delete_sql)
    db.commit()
    cursor = db.cursor()
    search_sql = "SELECT * FROM trade WHERE type = '%s' ORDER BY date ASC" % (FLAGS.type)
    cursor.execute(search_sql)
    results = cursor.fetchall()

    capital = 400000
    position = []
    raw_position = []
    price = []
    sig = []
    raw_sig = []
    for i in range(len(results)):
        position.append(results[i][2])
        price.append(float(results[i][1]))
        raw_position.append(results[i][6])
        sig.append(results[i][7])
        raw_sig.append(results[i][8])
    cm = results[0][4]
    tradetimes = 0
    f60_fee = np.zeros(len(position))
    f60_slide = np.zeros(len(position))
    bigpoint = 30 * cm
    j = 0
    per_trade_type = 0
    for i in range(1,len(position)):
        if(position[i] != position[i-1]):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - price[i]* 1e-4 * bigpoint
            f60_slide[i] = - 1 * bigpoint
            if position[i] > position[i-1]:
                trade_type = 1
            else:
                trade_type = -1
            if per_trade_type == 0:
                per_trade_type = trade_type
                j = i
            else:
                if per_trade_type == 1:
                    position_type = '买平'
                elif per_trade_type == -1:
                    position_type = '卖平'
                cost = (price[i] + price[j]) * 1e-4 * bigpoint + 2 * bigpoint
                profit = (price[i] - price[j]) * per_trade_type * bigpoint - cost
                cursor = db.cursor()
                insert_sql = "INSERT INTO trade_list (type,trade_type,in_price,out_price,in_date,out_date,amount,profit,position_type,cost) VALUES ('%s','%d','%f','%f','%s','%s','%d','%f','%s','%f')" %(FLAGS.type,trade_type,price[j],price[i],Date[j],Date[i],30,profit,position_type,cost)
                cursor.execute(insert_sql)
                db.commit()
                per_trade_type = 0

            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint

    f60_cost = np.add(f60_fee,f60_slide)
    print(tradetimes)
    db.close()

    chajia = np.diff(price,axis=0)

    f60_r = np.cumsum(np.multiply(position[:-1],chajia) * bigpoint + f60_cost[:-1]) + capital

    #原始信号
    tradetimes = 0
    f60_fee = np.zeros(len(raw_position))
    f60_slide = np.zeros(len(raw_position))
    for i in range(1,len(raw_position)):
        if(raw_position[i] != raw_position[i-1]):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - price[i]* 1e-4 * bigpoint
            f60_slide[i] = - 0.2 * bigpoint
            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint

    f60_cost = np.add(f60_fee,f60_slide)
    print(tradetimes)
    d = 0
    for i in range(len(raw_sig)):
        if raw_sig[i] * sig[i] <0:
            d += 1
    print(d / len(raw_sig) * 100,'%')

    f60_raw = np.cumsum(np.multiply(raw_position[:-1],chajia) * bigpoint + f60_cost[1:]) + capital

    plt.figure('capital')
    plt.plot(f60_r, 'b',f60_raw,'r')
    plt.grid(True)

    plt.figure('signal')
    plt.plot(sig, 'b',raw_sig,'r')
    plt.grid(True)
    plt.show()



if not os.path.exists("logs"):
    os.mkdir("logs")

if __name__ == '__main__':
    tf.app.run()