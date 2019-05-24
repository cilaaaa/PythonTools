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


class LstmRNN(object):
    def __init__(self, stock_count,
                 name='rnn',
                 lstm_size=20,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 keep_prob=0.8,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"):
        """
        Construct a RNN model using LSTM cell.
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
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        with self.graph.as_default():
            """
            The model asks for three things to be trained:
            - input: training data X
            - targets: training label y
            - learning_rate:
            """
            # inputs.shape = (number of examples, number of input, dimension of each input).
            self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

            self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
            self.targets = tf.placeholder(tf.float32, [None, 1], name="targets")

            def _create_one_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
                if self.keep_prob < 1.0:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                return lstm_cell

            cell = tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(self.num_layers)],
                state_is_tuple=True
            ) if self.num_layers > 1 else _create_one_cell()

            # Run dynamic RNN
            val, state_ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32, scope="dynamic_rnn")

            # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
            # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
            val = tf.transpose(val, [1, 0, 2])

            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
            ws = tf.Variable(tf.truncated_normal([self.lstm_size, 1]), name="w")
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.pred = tf.matmul(last, ws) + bias

            self.last_sum = tf.summary.histogram("lstm_state", last)
            self.w_sum = tf.summary.histogram("w", ws)
            self.b_sum = tf.summary.histogram("b", bias)
            self.pred_summ = tf.summary.histogram("pred", self.pred)

            # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
            self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse")
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

            self.loss_sum = tf.summary.scalar("loss_mse", self.loss)
            self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

            self.t_vars = tf.trainable_variables()
            self.saver = tf.train.Saver()

    def train(self, dataset_list, config):
        dataset_list[0].group = True
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        tf.global_variables_initializer().run()

        global_step = 0

        random.seed(time.time())

        print("Start training for stocks:", [d.stock_sym for d in dataset_list])
        for epoch in range(config.max_epoch):
            epoch_step = 0
            # learning_rate = config.init_learning_rate * (
            #     config.learning_rate_decay ** ((epoch + config.init_epoch) // config.init_epoch)
            # )
            learning_rate  = config.init_learning_rate
            for label_, d_ in enumerate(dataset_list):
                batch_X, batch_y = d_.generate_one_epoch(epoch, config.batch_size)

                global_step += 1
                epoch_step += 1
                train_data_feed = {
                    self.learning_rate: learning_rate,
                    self.inputs: batch_X,
                    self.targets: batch_y,
                }
                train_loss, _, train_merged_sum = self.sess.run(
                    [self.loss, self.optim, self.merged_sum], train_data_feed)
                self.writer.add_summary(train_merged_sum, global_step=global_step)

                if global_step % 500 == 0:
                    print("Step:%d [Learning rate: %.6f] train_loss:%.6f" % (
                        global_step, learning_rate, train_loss))

        # Save the final model
        self.save(global_step)

    @property
    def model_name(self):
        name = "stock_rnn_lstm%d_step%d_input%d_%s" % (
            self.lstm_size, self.num_steps, self.input_size, self.name)

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

    def prediction(self, prediction_data):
        if(self.num_steps > 1):
            prediction_data = np.array([prediction_data[i-self.num_steps: i] for i in range(len(prediction_data)  +1)])
        predict = self.sess.run(self.pred, feed_dict={self.inputs: prediction_data})
        return predict

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