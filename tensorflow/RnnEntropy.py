__author__ = 'Cila'
import tensorflow as tf
import sklearn.preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts

def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

def standard_scale(fit,data):
    preprocessor = prep.StandardScaler().fit(fit)
    data = preprocessor.transform(data)
    return data

# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 4
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 1
# 每个隐含层的节点数
hidden_size = 10
# LSTM layer 的层数
layer_num = 2
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 1
# 学习速率
lr=6e-4

weights={
         'in':xavier_init(input_size,hidden_size),
         'out':xavier_init(hidden_size,class_num)
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[hidden_size,])),
        'out':tf.Variable(tf.constant(0.1,shape=[class_num,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X,keep_prob):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,hidden_size])  #将tensor转成3维，作为lstm cell的输入
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0,output_keep_prob=keep_prob)
    # mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layer_num,state_is_tuple=True)
    # init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(lstm_cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,hidden_size]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————训练模型——————————————————
def train_lstm(train_data,train_count):
    X=tf.placeholder(tf.float32, shape=[None,timestep_size,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,timestep_size,class_num])
    keep_prob = tf.placeholder(tf.float32)
    batch_index,train_x,train_y=get_train_data(train_data,train_count)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X,keep_prob)
    #损失函数
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.4
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,1e+6,0.96,staircase=True)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob:0.75})
            print("Number of iterations:",i+1," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save2\\modle.ckpt'))
        #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
        #在Linux下面用 'model_save2/modle.ckpt'
        print("The train has finished")


#————————————————预测模型————————————————————
def prediction(test_data,test_label,test_count):
    X=tf.placeholder(tf.float32, shape=[None,timestep_size,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    test_x,test_y=get_test_data(test_data,test_label,test_count)
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X,1)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        test_y=np.array(test_y)
        test_predict=np.array(test_predict)
        # acc=np.average(np.abs(test_predict[:len(test_y)]-test_y)/len(test_y))  #偏差程度
        # print("The accuracy of this predict:",acc)
        # 以折线图表示结果
        # plt.figure()
        # plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        # plt.plot(list(range(len(test_y))), test_y,  color='r')
        # plt.grid(True)
        # plt.show()
    return test_predict,test_y


#获取训练集
def get_train_data(train_data,train_count,batch_size=100):
    batch_index=[]
    data_train=train_data[0:train_count]
    normalized_train_data = (data_train-np.mean(data_train,axis=0)) / np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(data_train)-timestep_size):
        if i % batch_size==0:
            batch_index.append(i)
        x=normalized_train_data[i:i+timestep_size,:4]
        y=normalized_train_data[i:i+timestep_size,4,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(data_train)-timestep_size))
    return batch_index,train_x,train_y


#获取测试集
def  get_test_data(data,label,train_count):
    data_test=data[train_count:]
    test_label = label[train_count:]
    test_y = test_label
    normalized_test_data=(data_test-np.mean(data_test,axis=0))/np.std(data_test,axis=0)  #标准化
    size=(len(data_test)+timestep_size-1)//timestep_size  #有size个sample
    test_x = []
    for i in range(size):
        x=normalized_test_data[i*timestep_size:(i+1)*timestep_size,:]
        test_x.append(x.tolist())
    return test_x,test_y

def main():
    df = pd.read_excel('Entropy_data/data15min.xlsx','Sheet1',header=None)
    RawPrice = np.array(df.iloc[:,2]).tolist()
    RawVolum = np.array(df.iloc[:,3]).tolist()
    RawHigh = np.array(df.iloc[:,0]).tolist()
    RawLow = np.array(df.iloc[:,1]).tolist()
    train_label = []
    for i in range(len(RawPrice)-5):
        sig = (np.mean(RawPrice[i+1:i+6])-RawPrice[i]) / RawPrice[i] * 10
        train_label.append(sig)
    train_batches = int(len(RawPrice) * 0.7)
    train_data = np.array(list(map(list,zip(*[RawPrice,RawVolum,RawHigh,RawLow,train_label]))))

    train_lstm(train_data,train_batches)
    # df = pd.read_excel('Entropy_data/RB1805.xls','Sheet0',header=None)
    # RawPrice = np.array(df.iloc[2:,6])
    # RawVolum = np.array(df.iloc[1:,9]).tolist()
    # RawHigh = np.array(df.iloc[1:,4]).tolist()
    # RawLow = np.array(df.iloc[1:,5]).tolist()
    test_data = np.array(list(map(list,zip(*[RawPrice,RawVolum,RawHigh,RawLow]))))
    test_label = train_label

    test_predict,test_y = prediction(test_data,test_label,train_batches)
    # test_predict = test_y
    plt.figure(1)
    plt.plot(test_y, 'b',test_predict,'r')
    plt.grid(True)


    # df = ts.get_k_data('600230','1990-01-01','2017-11-01')
    # RawOpen = np.array(df.iloc[:-1,1]).tolist()
    # RawPrice = np.array(df.iloc[1:,2]).tolist()
    # RawVolum = np.array(df.iloc[:-1,5]).tolist()
    # RawHigh = np.array(df.iloc[:-1,3]).tolist()
    # RawLow = np.array(df.iloc[:-1,4]).tolist()
    # train_batches = int(len(RawPrice) * 0.7)
    # train_data = np.array(list(map(list,zip(*[RawOpen,RawVolum,RawHigh,RawLow]))))
    # train_label = RawPrice
    # train_lstm(train_data,train_label,train_batches)
    #
    # test_predict,test_y = prediction(train_data,train_label,train_batches)
    # RawPriceNow = np.array(df.iloc[:-1,2])
    # plt.figure(1)
    # plt.plot(test_y, 'b',test_predict,'r')
    # plt.grid(True)


    # df = ts.get_k_data('600230','2017-11-02')
    # RawOpen = np.array(df.iloc[:,1]).tolist()
    # RawPrice = np.array(df.iloc[1:,2]).tolist()
    # RawVolum = np.array(df.iloc[:,5]).tolist()
    # RawHigh = np.array(df.iloc[:,3]).tolist()
    # RawLow = np.array(df.iloc[:,4]).tolist()
    # test_data = np.array(list(map(list,zip(*[RawOpen,RawVolum,RawHigh,RawLow]))))
    # test_label = RawPrice
    # test_predict,test_y = prediction(test_data,test_label,0)
    # RawPriceNow = np.array(df.iloc[:,2])


    #交易模块

    capital = 400000
    # if (test_predict[0] >= RawPriceNow[0]):
    #     line.append(capital - (fee + slide) * RawPriceNow[0])
    #     position += 1
    # for i in range(1,len(test_predict)):
    #     print((RawPriceNow[i] - RawPriceNow[i-1]) * bigpoint * position)
    #     capital = capital + (RawPriceNow[i] - RawPriceNow[i-1]) * bigpoint * position
    #     if test_predict[i] > RawPriceNow[i]:
    #         capital = capital - (fee + slide) * RawPriceNow[i] * bigpoint
    #         position += 1
    #     elif test_predict[i] < RawPriceNow[i] and position > 0:
    #         capital = capital - (fee + slide) * RawPriceNow[i] * position
    #         position = 0
    #     line.append(capital)
    f60_position = np.zeros(len(test_predict))
    f60_position[0] = 0
    for i in range(1,len(test_predict)):
        if test_predict[i] >= 0 and f60_position[i-1] >= 0:
            f60_position[i] = 1
        elif  test_predict[i] <= 0 and f60_position[i-1] <= 0:
            f60_position[i] = -1
        elif  test_predict[i] * f60_position[i-1] < 0:
            f60_position[i] = 0
    tradetimes = 0
    f60_fee = np.zeros(len(test_predict))
    f60_slide = np.zeros(len(test_predict))
    bigpoint = 300
    for i in range(1,len(test_predict)):
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
    print(chajia)

    f60_r = np.cumsum(np.multiply(f60_position[0:-1],chajia[train_batches:]) * bigpoint + f60_cost[1:]) + capital
    plt.figure(2)
    plt.plot(f60_r, color='b')
    plt.grid(True)
    plt.show()

main()