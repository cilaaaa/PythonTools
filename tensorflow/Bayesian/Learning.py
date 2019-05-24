__author__ = 'Cila'
from Bayesian import DataModel
import tensorflow as tf
import numpy as np

input_size = 180
steps=300000
batch_size=1000
symbol = DataModel.DataTools(normalized=True,input_size=input_size)

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
    # return np.sum(np.square(np.subtract(predictions,labels))) / predictions.shape[0]
    # return 0
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder("float", [None, input_size])
    y_ = tf.placeholder("float", shape=[None, 3])
    W1 = tf.Variable(tf.zeros([input_size,90]))
    b1 = tf.Variable(tf.zeros([90]))
    W2 = tf.Variable(tf.zeros([90,3]))
    b2 = tf.Variable(tf.zeros([3]))
    keep_pro = tf.placeholder(tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1,keep_pro)
    y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 10e-5
    # learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,1000,0.96,staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    train_step = tf.train.AdagradOptimizer(10e-2).minimize(cross_entropy)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(steps):
        offset_range = len(symbol.train_X) - batch_size
        offset = (step * batch_size) % offset_range
        batch_data = symbol.train_X[offset:(offset + batch_size)]
        batch_labels = symbol.train_y[offset:(offset + batch_size)]
        _, l, predictions = sess.run([train_step,cross_entropy,y],feed_dict={x:batch_data,y_:batch_labels,keep_pro:0.75})
        if (step+1) % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step+1, l))
            print("Minibatch accuracy: %.2f%%" % accuracy(predictions, batch_labels))
    testY = sess.run(y,feed_dict={x:symbol.test_X,keep_pro:0.75})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval(feed_dict={x: symbol.test_X, y_: symbol.test_y,keep_pro:1}))

# print(optimizer.eval(feed_dict={x: symbol.test_X, y_: symbol.test_y}))