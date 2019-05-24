__author__ = 'Cila'
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)
in_units = 784
h1_units = 300
W1 = tf.Variable(xavier_init(in_units,h1_units))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(xavier_init(h1_units,10))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32,[None,in_units])
keep_pro = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_pro)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
# start_time = time.time()
num_steps_burn_in = 100
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    if i >= num_steps_burn_in:
        if not i % 100:
            print('step %d cross_entropy %.3f y %s y_ %s' %(i - num_steps_burn_in,sess.run(cross_entropy,{x:batch_xs,y_:batch_ys,keep_pro: 0.75}),sess.run(tf.argmax(y,1),{x:batch_xs,keep_pro: 0.75}),sess.run(tf.argmax(y_,1),{y_:batch_ys})))
    train_step.run({x:batch_xs,y_:batch_ys,keep_pro: 0.75})
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print(sess.run(tf.argmax(y,1),{x:[mnist.test.images[1]],keep_pro:1.0}))
# print(sess.run(tf.argmax(y_,1),{y_:[mnist.test.labels[1]]}))
print(accuracy.eval({x:mnist.train.images,y_:mnist.train.labels,keep_pro:1.0}))