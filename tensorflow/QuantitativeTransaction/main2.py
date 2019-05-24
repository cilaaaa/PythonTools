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
flags.DEFINE_integer("input_size", 19, "Input size [10]")
flags.DEFINE_integer("num_steps",1, "Num of steps [30]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [64]")
flags.DEFINE_integer("layer", 4 ,'layer')
flags.DEFINE_float("keep_prob",0.75, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate",0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("epoch", 50, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_string("stock_symbol", 'train_HC', "Target stock symbol [None]")
flags.DEFINE_string("test_symbol", 'test_HC', "Target stock symbol [None]")
flags.DEFINE_string("type", 'HC', "Target stock type [None]")
flags.DEFINE_string("contract_multiplier", 10, "Target stock contract_multiplier [None]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("logisticRegression", True, "Use Logic")
flags.DEFINE_boolean("lstm", False, "Use lstm")

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
                     valid_ratio=valid_ratio)]


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    #tflearn lstm
    if FLAGS.lstm:
        name = "stock_rnn_lstm_%s_%dlayer" % (FLAGS.stock_symbol,FLAGS.layer)
    else:
        name = "stock_rnn_ann_%s_%dlayer" % (FLAGS.stock_symbol,FLAGS.layer)

    #ANN
    if not FLAGS.lstm:

        input_layer = tflearn.input_data(shape=[None,FLAGS.input_size])
        net = tflearn.batch_normalization(input_layer)
        hidden_cur_cnt = 400
        net = tflearn.fully_connected(net, hidden_cur_cnt, activation='relu',regularizer='L2',weights_init='xavier')
        net = tflearn.dropout(net, FLAGS.keep_prob)
        for i in range(FLAGS.layer - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            net = tflearn.fully_connected(net, hidden_next_cnt, activation='relu',regularizer='L2',weights_init='xavier')
            net = tflearn.dropout(net, FLAGS.keep_prob)
    #LSTM
    else:
        input_layer = tflearn.input_data(shape=[None,FLAGS.num_steps,FLAGS.input_size])
        net = tflearn.lstm(input_layer,256,dropout=FLAGS.keep_prob,return_seq=True,)
        for i in range(FLAGS.layer - 2):
            net = tflearn.lstm(net,256,dropout=FLAGS.keep_prob,return_seq=True)
        net = tflearn.lstm(net,256,dropout=FLAGS.keep_prob)
    if FLAGS.logisticRegression:
        softmax = tflearn.fully_connected(net, 3,activation='softmax')
    else:
        softmax = tflearn.fully_connected(net,1)
    if FLAGS.logisticRegression:
        optimizer = tflearn.Adam(learning_rate=1e-3,epsilon=0.001,beta1=0.9)
    else:
        optimizer = tflearn.SGD(learning_rate=0.1, lr_decay=0.99, decay_step=1000)
    net = tflearn.regression(softmax, optimizer=optimizer,metric='accuracy')
    model = tflearn.DNN(net,tensorboard_verbose=0,tensorboard_dir='C:/Users/Cila/Desktop/tensorflow/QuantitativeTransaction/tensorboard')

    stock_data_list = load_data(
                FLAGS.input_size,
                FLAGS.num_steps,
                k=FLAGS.stock_count,
                valid_ratio = 0.0,
                target_symbol=FLAGS.stock_symbol,
            )
    train_x = stock_data_list[0].train_X
    x1 = standard_scale(train_x,train_x)
    # x1 = train_x
    if FLAGS.lstm:
        x = []
        for j in range(FLAGS.num_steps,len(x1)+1):
            temp = x1[j-FLAGS.num_steps:j]
            x.append(temp)
        x1 = np.array(x)

    test_x = stock_data_list[0].GetTestData(FLAGS.test_symbol)
    test_y = stock_data_list[0].raw_test_sig
    if FLAGS.train:
        train_y = stock_data_list[0].train_y[FLAGS.num_steps-1:]
        #int(3e+4 / (len(x1) / FLAGS.batch_size))
        model.fit(x1,train_y,n_epoch=200,
                  validation_set=(test_x[5:105],test_y[0:100]),
                  show_metric=True,
                  batch_size=FLAGS.batch_size)
        model.save(os.path.join("logs", name))
    model.load(os.path.join("logs", name))
    raw_train_pre = model.predict(x1)
    raw_test_pre = []
    for i in range(len(test_x)):
        if i >= 5:
            if FLAGS.lstm:
                test_batch_x = []
                if i < FLAGS.num_steps-1 :
                    temp = np.append(train_x[-FLAGS.num_steps+i+1:],test_x[0:i+1],axis=0)
                else:
                    temp = test_x[i-FLAGS.num_steps+1:i+1]
                temp = standard_scale(train_x,temp)
                test_batch_x.append(temp)
            else:
                test_batch_x = standard_scale(train_x,[test_x[i]])
            raw_test_pre.append(model.predict(test_batch_x)[0])
    raw_test_pre = np.array(raw_test_pre)
    prediction = []
    if FLAGS.logisticRegression:
        for i in range(len(raw_test_pre)):
            temp = 1
            if raw_test_pre[i][0] >= 0.5:
                temp = 0
            if raw_test_pre[i][2] >= 0.5:
                temp = 2
            prediction.append(temp)
        train_prediction = np.argmax(raw_train_pre,1)
        train_y = np.argmax(stock_data_list[0].train_y[FLAGS.num_steps-1:],1)
        # prediction = np.argmax(raw_test_pre,1)
        test_y = np.argmax(test_y,1)
    else:
        train_prediction = raw_train_pre
        train_y = stock_data_list[0].train_y[FLAGS.num_steps-1:]
        prediction = raw_test_pre
    print("train accuracy %.2f,test accuracy %.2f" % (accuracy(train_prediction,train_y),accuracy(prediction,test_y)))

    trade(stock_data_list,prediction)
    plot(stock_data_list,train_prediction)

def accuracy(data,label):
    if FLAGS.logisticRegression:
        return 100.0 * np.sum(data == label) / len(data)
    else:
        d = 0
        for i in range(len(data)):
            if data[i] * label[i] >= 0:
                d += 1
        return d / len(data) * 100.0

def standard_scale(fit,data):
    #均值标准差
    mean = np.mean(fit,axis=0)
    std = np.std(fit,axis=0)
    data = (data - mean) / std

    # preprocessor = prep.Normalizer().fit(fit)
    # data = preprocessor.transform(data)

    # min_max_scaler = prep.MinMaxScaler()
    # min_max_scaler.fit(fit)
    # data = min_max_scaler.transform(data)

    #第三方方法
    # preprocessor = prep.StandardScaler().fit(fit)
    # data = preprocessor.transform(data)

    #最大最小值
    # merge = np.append(fit,data,axis=0)
    # max = np.max(merge,axis=0)
    # min = np.min(merge,axis=0)
    # data = (data-min) / (max - min)
    # merge = np.append(fit,data,axis=0)

    # mean = np.mean(fit,axis=0)
    # std = np.std(fit,axis=0)
    # data = (data - mean) / std
    return data

def trade(dataset_list,signal):
    RawPrice = dataset_list[0].raw_test_price
    Date = dataset_list[0].date
    predict_position = np.zeros(len(signal))
    predict_position[0] = 0
    for i in range(1,len(signal)):
        if FLAGS.logisticRegression:
            if signal[i] <= 1 and predict_position[i-1] >= 0:
                predict_position[i] = 1
            elif  signal[i] >= 1 and predict_position[i-1] <= 0:
                predict_position[i] = -1
            elif  (1-signal[i]) * predict_position[i-1] < 0:
                predict_position[i] = 0
        else:
            if signal[i] >= 0 and predict_position[i-1] >= 0:
                predict_position[i] = 1
            elif  signal[i] <= 0 and predict_position[i-1] <= 0:
                predict_position[i] = -1
            elif  signal[i] * predict_position[i-1] < 0:
                predict_position[i] = 0
    if FLAGS.logisticRegression:
        raw_sig = np.argmax(dataset_list[0].raw_test_sig,1)
    else:
        raw_sig = dataset_list[0].raw_test_sig
    raw_position = np.zeros(len(raw_sig))
    raw_position[0] = 0
    for i in range(1,len(raw_sig)):
        if FLAGS.logisticRegression:
            if raw_sig[i] <= 1 and raw_position[i-1] >= 0:
                raw_position[i] = 1
            elif  raw_sig[i] >= 1 and raw_position[i-1] <= 0:
                raw_position[i] = -1
            elif  (1-raw_sig[i]) * raw_position[i-1] < 0:
                raw_position[i] = 0
        else:
            if raw_sig[i] >= 0 and raw_position[i-1] >= 0:
                raw_position[i] = 1
            elif  raw_sig[i] <= 0 and raw_position[i-1] <= 0:
                raw_position[i] = -1
            elif  raw_sig[i] * raw_position[i-1] < 0:
                raw_position[i] = 0
    saveMysql(RawPrice,Date,signal,raw_sig,predict_position,raw_position)

def saveMysql(price,date,signal,raw_sig,pre_position,raw_position):
    db = mysql.connect("localhost","root","init1234","entropy")
    cursor = db.cursor()
    delete_sql = "DELETE FROM trade WHERE type = '%s'" %(FLAGS.type)
    cursor.execute(delete_sql)
    db.commit()
    sql = ''
    for i in range(len(price)):
        sql += "INSERT INTO trade (price,position,raw_position,sig,raw_sig, type, contract_multiplier,date) VALUES ('%.2f', '%d','%d','%f','%f', '%s', '%d','%s');" % (price[i],pre_position[i],raw_position[i],signal[i],raw_sig[i],FLAGS.type,FLAGS.contract_multiplier,date[i])
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

def plot(dataset_list,train_prediction):
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
    insert_sql = ''
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

                insert_sql += "INSERT INTO trade_list (type,trade_type,in_price,out_price,in_date,out_date,amount,profit,position_type,cost) VALUES ('%s','%d','%f','%f','%s','%s','%d','%f','%s','%f');" %(FLAGS.type,trade_type,price[j],price[i],Date[j],Date[i],30,profit,position_type,cost)
                per_trade_type = 0

            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint
    if insert_sql != '':
        cursor = db.cursor()
        cursor.execute(insert_sql)
        db.commit()
    db.close()

    f60_cost = np.add(f60_fee,f60_slide)
    print('测试集预测交易次数',tradetimes)

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
    print('测试集原始信号交易次数',tradetimes)
    f60_raw = np.cumsum(np.multiply(raw_position[:-1],chajia) * bigpoint + f60_cost[1:]) + capital

    #训练信号
    train_price = dataset_list[0].raw_price[200+FLAGS.num_steps-1:-5]
    chajia2 = np.diff(train_price,axis=0)
    if FLAGS.logisticRegression:
        raw_train_sig = np.argmax(dataset_list[0].train_y[FLAGS.num_steps-1:],1)
    else:
        raw_train_sig = dataset_list[0].train_y[FLAGS.num_steps-1:]
    raw_train_position = np.zeros(len(raw_train_sig))
    raw_train_position[0] = 0
    for i in range(1,len(raw_train_sig)):
        if FLAGS.logisticRegression:
            if raw_train_sig[i] <= 1 and raw_train_position[i-1] >= 0:
                raw_train_position[i] = 1
            elif  raw_train_sig[i] >= 1 and raw_train_position[i-1] <= 0:
                raw_train_position[i] = -1
            elif  (1-raw_train_sig[i]) * raw_train_position[i-1] < 0:
                raw_train_position[i] = 0
        else:
            if raw_train_sig[i] >=0 and raw_train_position[i-1] >= 0:
                raw_train_position[i] = 1
            elif  raw_train_sig[i] <=0 and raw_train_position[i-1] <= 0:
                raw_train_position[i] = -1
            elif  raw_train_sig[i] * raw_train_position[i-1] < 0:
                raw_train_position[i] = 0
    tradetimes = 0
    f60_fee = np.zeros(len(raw_train_position))
    f60_slide = np.zeros(len(raw_train_position))
    for i in range(1,len(raw_train_position)):
        if(raw_train_position[i] != raw_train_position[i-1]):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - train_price[i]* 1e-4 * bigpoint
            f60_slide[i] = - 0.2 * bigpoint
            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint
    f60_cost = np.add(f60_fee,f60_slide)
    print('训练集原始信号交易次数',tradetimes)
    train_raw = np.cumsum(np.multiply(raw_train_position[:-1],chajia2) * bigpoint + f60_cost[1:]) + capital

    train_position = np.zeros(len(train_prediction))
    train_prediction[0] = 0
    for i in range(1,len(train_prediction)):
        if FLAGS.logisticRegression:
            if train_prediction[i] <= 1 and train_position[i-1] >= 0:
                train_position[i] = 1
            elif  train_prediction[i] >= 1 and train_position[i-1] <= 0:
                train_position[i] = -1
            elif  (1-train_prediction[i]) * train_position[i-1] < 0:
                train_position[i] = 0
        else:
            if train_prediction[i] >=0 and train_position[i-1] >= 0:
                train_position[i] = 1
            elif  train_prediction[i] <=0 and train_position[i-1] <= 0:
                train_position[i] = -1
            elif  (1-train_prediction[i]) * train_position[i-1] < 0:
                train_position[i] = 0
    tradetimes = 0
    f60_fee = np.zeros(len(train_position))
    f60_slide = np.zeros(len(train_position))
    for i in range(1,len(train_position)):
        if(train_position[i] != train_position[i-1]):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - train_price[i]* 1e-4 * bigpoint
            f60_slide[i] = - 0.2 * bigpoint
            #股票
            # f60_fee[i] = - RawPriceNow[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint
    f60_cost = np.add(f60_fee,f60_slide)
    print('训练集预测信号交易次数',tradetimes)
    train = np.cumsum(np.multiply(train_position[:-1],chajia2) * bigpoint + f60_cost[1:]) + capital

    plt.figure('capital')
    plt.plot(f60_r, 'b',f60_raw,'r',train_raw,'g',train,'y')
    plt.grid(True)
    plt.show()



if not os.path.exists("logs"):
    os.mkdir("logs")

if __name__ == '__main__':
    tf.app.run()