__author__ = 'Cila'

import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector


class ANN(object):
    def __init__(self, stock_count,
                 name='',
                 num_layers=2,
                 num_steps=30,
                 input_size=1,
                 keep_prob=0.8,
                 layer_cnt=2,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"):
        """
        Args:
            sess:
            stock_count:
            lstm_size:
            num_layers
            num_steps:
            input_size:
            keep_prob:
            embed_size
            checkpoint_dir
        """
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        self.stock_count = stock_count
        self.name = name
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob
        self.hidden_size = int(input_size * 1.5)
        self.layer_cnt = layer_cnt
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.build_graph()

    def build_graph(self):
        with self.graph.as_default():

            # inputs.shape = (number of examples, number of input, dimension of each input).
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size], name="inputs_" + self.name)

            # first layer
            W1 = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_size]), name='w1')
            b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name='b1')

            # middle weight
            weights = []
            biases = []
            hidden_cur_cnt = self.hidden_size
            for i in range(self.layer_cnt - 2):
                if hidden_cur_cnt > 2:
                    hidden_next_cnt = int(hidden_cur_cnt / 2)
                else:
                    hidden_next_cnt = 2
                weights.append(tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt])))
                biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
                hidden_cur_cnt = hidden_next_cnt

            y0 = tf.matmul(self.inputs, W1) + b1
            hidden = tf.nn.relu(y0)
            hidden_drop = tf.nn.dropout(hidden, self.keep_prob)

            keep_prob = self.keep_prob

            # middle layer
            for i in range(self.layer_cnt - 2):
                y1 = tf.matmul(hidden_drop, weights[i]) + biases[i]
                hidden_drop = tf.nn.relu(y1)
                keep_prob += 0.5 * i / (self.layer_cnt + 1)
                hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)

            ws = tf.Variable(tf.truncated_normal([hidden_cur_cnt, 1]), name="w")
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.pred = tf.matmul(hidden_drop, ws) + bias

            if self.embed_size > 0:
                self.embed_matrix = tf.Variable(
                    tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                    name="embed_matrix"
                )

            else:
                self.inputs_with_embed = tf.identity(self.inputs)

            self.w_sum = tf.summary.histogram("w", ws)
            self.b_sum = tf.summary.histogram("b", bias)
            self.pred_sum = tf.summary.histogram("pred", self.pred)

            # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
            self.targets = tf.placeholder(tf.float32, [None, 1], name="targets_" + self.name)
            l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(ws)
            for i in range(len(weights)):
                l2_loss += tf.nn.l2_loss(weights[i])
                # l2_loss += tf.nn.l2_loss(biases[i])
            # beta = 0.25 / self.batch_size
            beta = 1e-5
            self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse") + beta * l2_loss
            self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate_" + self.name)
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

            self.loss_sum = tf.summary.scalar("loss_mse", self.loss)
            self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

            self.t_vars = tf.trainable_variables()
            self.saver = tf.train.Saver(tf.global_variables())


    def train(self, dataset_list, config):
        # hidden_size = self.findBestHidden(dataset_list, config)
        # self.hidden_size = hidden_size

        # Set up the logs folder
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        tf.global_variables_initializer().run()

        global_step = 0

        print("Start training for stocks:", [d.stock_sym for d in dataset_list])
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** ((epoch + config.init_epoch) // config.init_epoch)
            )
            for label_, d_ in enumerate(dataset_list):
                batch_X, batch_y = d_.generate_one_epoch(epoch, config.batch_size)
                global_step += 1
                epoch_step += 1
                train_data_feed = {
                    self.learning_rate: learning_rate,
                    self.inputs: batch_X,
                    self.targets: batch_y
                }

                train_loss, _, train_merged_sum = self.sess.run(
                    [self.loss, self.optim, self.merged_sum], train_data_feed)
                self.writer.add_summary(train_merged_sum, global_step=global_step)

                if global_step % 500 == 0:
                    # test_loss, test_pred = self.sess.run([self.loss, self.pred], test_data_feed)

                    print("Step:%d [Learning rate: %.6f] train_loss:%.6f" % (
                        global_step, learning_rate, train_loss))
        self.save(global_step)

    def prediction(self, prediction_data):
        predict = self.sess.run(self.pred, feed_dict={self.inputs: prediction_data})
        return predict

    @property
    def model_name(self):
        name = "stock_step%d_input%d_%s" % (
            self.num_steps, self.input_size, self.name)
        if self.embed_size > 0:
           name += "_embed%d" % self.embed_size
        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir


    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir


    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )


    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def plot_samples(self, preds, targets, figname, stock_sym=None):
        def _flatten(seq):
            return [x for y in seq for x in y]

        truths = _flatten(targets)[-200:]
        preds = _flatten(preds)[-200:]
        days = range(len(truths))[-200:]

        plt.figure(figsize=(12, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()