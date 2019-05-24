__author__ = 'Cila'
import tensorflow as tf
import EntropyData
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.learn as learn
import neurolab as nl
import math
import tensorlayer as tl



def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

def standard_scale(fit,data):
    preprocessor = prep.StandardScaler().fit(fit)
    data = preprocessor.transform(data)
    return data


# train_x,valid_x,test_x = standard_scale(train_x,valid_x,test_x)

def get_random_block_from_data(train_x,train_y,batch_size):
    start_index = np.random.randint(0,len(train_x) - batch_size)
    return train_x[start_index:(start_index + batch_size)],train_y[start_index:(start_index + batch_size)]

def tf_better_nn(input,output,hidden_node_count,train_x,train_y,layer_cnt=2,keep_prob=0.5,batch_size=100,steps=30000):
    graph = tf.Graph()
    with graph.as_default():
        tf_train_x = tf. placeholder(tf.float32,[None,input])
        tf_train_y = tf.placeholder(tf.float32,[None,output])
        # tf_valid_x = tf.constant(valid_x,dtype=tf.float32)
        # tf_test_x = tf.constant(test_x,dtype=tf.float32)

        # start weight
        hidden_stddev = np.sqrt(2.0 / input)
        W1 = tf.Variable(xavier_init(input,hidden_node_count))
        b1 = tf.Variable(tf.truncated_normal([hidden_node_count]))
        # middle weight
        weights = []
        biases = []
        hidden_cur_cnt = hidden_node_count
        for i in range(layer_cnt - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            hidden_stddev = np.sqrt(2.0 / hidden_cur_cnt)
            weights.append(tf.Variable(tf.truncated_normal([hidden_cur_cnt,hidden_next_cnt],stddev=hidden_stddev)))
            biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
            hidden_cur_cnt = hidden_next_cnt
        # first layer
        y0 = tf.matmul(tf_train_x,W1) + b1
        hidden = tf.nn.relu(y0)

        # valid_y0 = tf.matmul(tf_valid_x,W1) + b1
        # valid_hidden = tf.nn.relu(valid_y0)
        #
        # test_y0 = tf.matmul(tf_test_x,W1) + b1
        # test_hidden = tf.nn.relu(test_y0)

        hidden_drop = tf.nn.dropout(hidden,keep_prob)

        # middle layer
        for i in range(layer_cnt - 2):
            y1 = tf.matmul(hidden_drop,weights[i]) + biases[i]
            hidden_drop = tf.nn.relu(y1)
            keep_prob += 0.5 * i / (layer_cnt + 1)
            hidden_drop = tf.nn.dropout(hidden_drop,keep_prob)

            y0 = tf.matmul(hidden, weights[i]) + biases[i]
            hidden = tf.nn.relu(y0)

            # valid_y0 = tf.matmul(valid_hidden, weights[i]) + biases[i]
            # valid_hidden = tf.nn.relu(valid_y0)
            #
            # test_y0 = tf.matmul(test_hidden, weights[i]) + biases[i]
            # test_hidden = tf.nn.relu(test_y0)

        W2= tf.Variable(xavier_init(hidden_cur_cnt,output))
        b2 = tf.Variable(tf.zeros([output]))
        # last wx + b
        logits = tf.matmul(hidden_drop, W2) + b2
        # logits_active = tf.nn.tanh(logits)
        logits_predict = tf.matmul(hidden,W2) + b2
        # valid_predict = tf.matmul(valid_hidden,W2) + b2
        # test_predict = tf.matmul(test_hidden,W2) + b2

        l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
        for i in range(len(weights)):
            l2_loss += tf.nn.l2_loss(weights[i])
        beta = 1e-5
        loss = tf.reduce_mean(tf.square(logits - tf_train_y)) + beta * l2_loss
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,100000,0.96,staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        # train_prediction = tf.nn.softmax(logits_predict)
        # valid_prediction = tf.nn.softmax(valid_predict)
        # test_prediction = tf.nn.softmax(test_predict)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        for step in range(steps):
            offset_range = len(train_x) - batch_size
            offset = (step * batch_size) % offset_range
            batch_data = train_x[offset:(offset + batch_size)]
            batch_labels = train_y[offset:(offset + batch_size)]
            return_w1,return_w2,return_weights,return_b1,return_b2,return_biases,_, l, predictions = sess.run([W1,W2,weights,b1,b2,biases,optimizer,loss,logits],feed_dict={tf_train_x:batch_data,tf_train_y:batch_labels})
            if (step+1) % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step+1, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, train_y))
        w = [return_w1,return_weights,return_w2]
        b = [return_b1,return_biases,return_b2]
        return w,b

def accuracy(predictions, labels):
    # return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
    # return np.sum(np.square(np.subtract(predictions,labels))) / predictions.shape[0]
    return 0

def sim(W,b,test_x,layer_cnt=2):
    tf_test_x = tf.placeholder(tf.float32,[None,len(test_x[0])])
    test_y0 = tf.matmul(tf_test_x,W[0]) + b[0]
    test_hidden = tf.nn.relu(test_y0)
    for i in range(layer_cnt - 2):
        test_y0 = tf.matmul(test_hidden, W[1][i]) + b[1][i]
        test_hidden = tf.nn.relu(test_y0)
    test_predict = tf.matmul(test_hidden,W[2]) + b[2]
    test_predict = tf.nn.tanh(test_predict)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        signal = sess.run(test_predict,{tf_test_x:test_x})
    return signal

def findBestHidden(input,output,train_x,train_y):
    number = int(np.sqrt(input+output))
    s = np.arange(number+1,number+11)
    res = np.zeros(len(s))
    for i in range(len(s)):
        w,b = tf_better_nn(input,output,s[i],train_x,train_y,steps=100)
        test_signal = sim(w,b,train_x)
        for j in range(len(train_x)):
            if test_signal[j] * train_y[j] < 0:
                res[i] = res[i] + 1
            else:
                res[i] = res[i]
    best_index = res.tolist().index(np.min(res))
    return s[best_index]

def tf_learn(train_x,train_y,steps=30000):
    feature_columns = learn.infer_real_valued_columns_from_input(train_x)
    regressor = learn.LinearRegressor(feature_columns=feature_columns)
    regressor.fit(train_x,train_y,steps=steps,batch_size=100)
    return regressor

def CaptialChart(state_param,Swing):
    train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_x,test_signal = EntropyData.ReadDataFromXml('Entropy_data/RB1805.xls',Swing=Swing,state_param=state_param)
    # train_index1,train_index2,train_index3,train_signal1,train_signal2,train_signal3,test_x,test_signal = EntropyData.ReadDataByTuShare(code='600486',start='2005-11-20',end='2017-11-20',ktype='5',state_param=state_param,Swing=Swing)


    x1 = standard_scale(train_index1,train_index1)
    x2 = standard_scale(train_index1,train_index2)
    x3 = standard_scale(train_index1,train_index3)

    Net1W,Net1B = tf_better_nn(10,1,findBestHidden(10,1,x1,train_signal1),x1,train_signal1,keep_prob=0.75)
    Net2W,Net2B = tf_better_nn(10,1,findBestHidden(10,1,x2,train_signal2),x2,train_signal2,keep_prob=0.75)
    Net3W,Net3B = tf_better_nn(10,1,findBestHidden(10,1,x3,train_signal3),x3,train_signal3,keep_prob=0.75)
    # net1 = tf_learn(x1,train_signal1)
    # net2 = tf_learn(x2,train_signal2)
    # net3 = tf_learn(x3,train_signal3)
    f60_singal = np.zeros(len(test_x))
    data60 = np.array(test_x)[:,2:12].tolist()
    state60 = np.array(test_x)[:,1:2].tolist()
    price60 = np.array(test_x)[:,0:1]
    # plt.plot(price60)
    # plt.title('Capital Line')
    # plt.grid(True)
    # plt.show()
    data60 = standard_scale(train_index1,data60)
    a = []
    b = []
    c = []
    a_index = 0
    b_index = 0
    c_index = 0
    for i in range(len(test_x)):
        if state60[i][0] == 1:
            a.append(data60[i])
            f60_singal[i] = 1
        elif state60[i][0] == 2:
            b.append(data60[i])
            f60_singal[i] = 2
        else:
            c.append(data60[i])
            f60_singal[i] = 3

    if(len(a) > 0):
        # s1 = list(net1.predict(np.array(a),as_iterable=True))
        s1 = sim(Net1W,Net1B,a)
    else:
        s1 = []
    if(len(b) > 0):
        # s2 = list(net2.predict(np.array(b),as_iterable=True))
        s2 = sim(Net2W,Net2B,b)
    else:
        s2 = []
    if(len(c) > 0):
        # s3 = list(net3.predict(np.array(c),as_iterable=True))
        s3 = sim(Net3W,Net3B,c)
    else:
        s3 = []
    for i in range(len(test_x)):
        if f60_singal[i] == 1:
            f60_singal[i] = s1[a_index]
            a_index += 1
        elif f60_singal[i] == 2:
            f60_singal[i] = s2[b_index]
            b_index += 1
        else:
            f60_singal[i] = s3[c_index]
            c_index += 1
    return f60_singal,test_signal
    d=0
    for i in range(len(test_signal[:-4])):
        if test_signal[i] * f60_singal[i] < 0:
            d += 1
    print(d / len(test_signal) *100,'%')
    f60_position = np.mat(np.zeros((len(f60_singal),1)))
    f60_position[0] = 0
    for i in range(1,len(f60_singal)):
        if f60_singal[i] >= 0 and f60_position[i-1] >= 0:
            f60_position[i] = 1
        elif  f60_singal[i] <= 0 and f60_position[i-1] <= 0:
            f60_position[i] = -1
        elif  f60_singal[i] * f60_position[i-1] < 0:
            f60_position[i] = 0
    tradetimes = 0
    f60_fee = np.zeros((len(f60_singal),1))
    f60_slide = np.zeros((len(f60_singal),1))
    bigpoint = 300
    for i in range(1,len(f60_singal)):
        if(f60_position[i] != 0):
            tradetimes = tradetimes +1
            #期货
            #  fee 1e-4 * 300
            #  slide  0.2 * 300
            f60_fee[i] = - price60[i]* 1e-4 * bigpoint
            f60_slide[i] = - 0.2 * bigpoint
            #股票
            # f60_fee[i] = - price60[i]*0.001*bigpoint
            # f60_slide[i] = - 0.1 * bigpoint
    f60_cost = np.add(f60_fee,f60_slide)

    print(tradetimes)

    chajia = np.diff(price60,axis=0)

    f60_r = np.multiply(f60_position[0:-1],chajia[0:]) * bigpoint + f60_cost[1:]

    return f60_r

def main():
    max = 0

    # capital = []
    # capital.append(CaptialChart(max,0.1))
    # best_i = 0
    # for i in range(10):
    #     capital.append(CaptialChart(max * 0.8,0.1))
    #     if (capital[i+1][-1] < capital[i][-1]):
    #         print(capital[i][-1])
    #         best_i = i
    #         break
    quxian,test_signal = CaptialChart(max,0.05)
    # quxian = list(np.diff(quxian[0:len(test_signal)]))
    # test_signal = list(np.diff(test_signal))
    d=0
    for i in range(len(test_signal)):
        if quxian[i] * test_signal[i] < 0:
            d += 1
    print(d / len(test_signal) *100,'%')
    plt.plot(quxian,'-b',test_signal,'-r')
    plt.title('Capital Line')
    plt.grid(True)
    plt.show()

main()