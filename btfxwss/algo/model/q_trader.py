import numpy as np
import pandas as pd
import tensorflow as tf
import time
import btfxwss.algo.tests.reader as reader
from talib.abstract import *
from collections import defaultdict, namedtuple
from btfxwss.algo.backtest import Backtest
import btfxwss.algo.tests.util as util

DATA_CSV = 'bitfinex_public_ohlcv.csv'


class TradingEnv:
    def __init__(self):
        self.ohlcv_data = defaultdict(pd.DataFrame)
        self.data = None
        self.state = None
        self.terminal_state = False
        self.signal = None
        self.time_step = 0

    def load_market_data(self, filename):
        df = pd.read_csv(filename, names=['id', 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
                         parse_dates=['timestamp'], index_col='timestamp')
        df.sort_index(inplace=True)
        new_index = np.arange(0, df.index.size)
        df.index = new_index
        df.drop('id', axis=1, inplace=True)
        for symbol, data in df.groupby(['symbol']):
            data.reset_index(inplace=True, drop=True)
            self.ohlcv_data[symbol] = data.iloc[:, 1:]

    def init(self, symbol):
        # Using typical prices
        # price = (self.ohlcv_data['high'] + self.ohlcv_data['low'] + self.ohlcv_data['close']) / 3
        # price = price.values
        sma15 = SMA(self.ohlcv_data[symbol], timeperiod=15)
        sma60 = SMA(self.ohlcv_data[symbol], timeperiod=60)
        rsi = RSI(self.ohlcv_data[symbol], timeperiod=14)
        atr = ATR(self.ohlcv_data[symbol], timeperiod=14)
        self.data = np.column_stack((self.ohlcv_data[symbol], sma15, sma60, rsi, atr))
        self.state = self.data[0]
        self.signal = pd.Series(data=0, index=np.arange(self.data.shape[0]))
        self.time_step = 14
        return self.state

    def act(self, action):
        self.time_step += 1
        if self.time_step == self.data.shape[0]:
            self.terminal_state = True
            self.time_step = 14
        else:
            if action == 1:
                self.signal.loc[self.time_step] = 100
            if action == 2:
                self.signal.loc[self.time_step] = -100

        # bt = Backtest(pd.Series(data=self.data[self.time_step - 1:self.time_step + 1, 3],
        #                         index=[self.time_step - 1, self.time_step]),
        # #               self.signal.loc[self.time_step - 1:self.time_step + 1], signalType='shares')
        print(self.data[self.time_step, 3])
        # print (bt.data['shares'])
        # print (bt.data['pnl'])
        reward = ((self.data[self.time_step, 3] - self.data[self.time_step - 1, 3]) * self.signal.loc[self.time_step])
        next_state = self.data[self.time_step]
        return reward, next_state


def lstm(lstm_size):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # model.output_size = output_size
    # initial_state = (tf.zeros([batch_size, lstm_size]), tf.zeros([batch_size, lstm_size]))
    return cell


def run_epoch(model, sess):
    start_time = time.time()
    costs = 0.
    iters = 0
    print (model.final_state)
    print (model.initial_state)
    state, _ = sess.run([model.initial_state, model.train_op])
    fetches = {
        "loss": model.loss,
        "final_state": model.final_state,
        "train_opp": model.train_op
    }
    for step in range(model.data.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        # c, h = model.initial_state.c, model.initial_state.h
        # feed_dict[c] = state.c
        # feed_dict[h] = state.h
        vals = sess.run(fetches, feed_dict)
        loss = vals["loss"]
        state = vals["final_state"]
        iters += model.data.batch_size * model.data.num_steps
        costs += loss
        print("%.3f perplexity: %.3f speed: %.0f wps last loss: %.3f" %
              (step * 1.0 / model.data.epoch_size, np.exp(costs / iters),
               iters * model.data.batch_size /
               (time.time() - start_time), loss))


class PTBInput:
    def __init__(self, raw_data, config):
        self.raw_data = raw_data
        self.epoch_size = ((len(self.raw_data) // config.num_epochs) - 1) // config.num_steps
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps

    def get_data(self):
        x, y = reader.ptb_producer(self.raw_data, self.batch_size, self.num_steps)
        return (x, y)


class Model:
    def __init__(self, data, config, stage_flag="train"):
        self.stage_flag = stage_flag
        # x = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.num_features])
        # y = tf.placeholder(tf.float32, [config.batch_size, config.output_size])
        with tf.name_scope('my_graph'):
            self.data = data
            inputs, targets = data.get_data()
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [config.vocab_size, config.lstm_size], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, inputs)
            W = tf.Variable(tf.truncated_normal([config.lstm_size, config.vocab_size], stddev=0.1, dtype=tf.float32))
            b = tf.Variable(tf.constant(0.1, shape=[config.vocab_size]))
            outputs = []
            tf.summary.tensor_summary('logit_weights', W)
            tf.summary.tensor_summary('logit_biases', b)
            # initial_state = (tf.placeholder(tf.float32, [config.batch_size, config.lstm_size]),
            # tf.placeholder(tf.float32, [config.batch_size, config.lstm_size]))


            lstm1_output, self.final_state = self._build_rnn_graph_lstm(inputs, config, stage_flag)

            # lstm_1 = lstm(config.lstm_size)
            # # state = initial_state
            # with tf.variable_scope("lstm_layer") as scope:
            #     for i in range(config.num_steps):
            #         print(inputs.shape)
            #         h_lstm1, state = lstm_1(inputs[:, i, :], state)
            #         scope.reuse_variables()
            #         outputs.append(h_lstm1)
            # self.final_state = state
            # lstm1_output = tf.reshape(tf.concat(outputs, 1), [-1, config.lstm_size])
            print('output shape = ', lstm1_output.shape)
            logits = tf.matmul(lstm1_output, W) + b
            logits = tf.reshape(logits, [config.batch_size, config.num_steps, config.vocab_size])
            print(targets.shape)
            print(lstm1_output.shape)
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                tf.to_int32(targets),
                tf.ones([config.batch_size, config.num_steps], dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
            if stage_flag == "train":
                self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            self.summary = tf.summary.merge_all()

    def _get_lstm_cell(self, config, stage_flag):
        return tf.contrib.rnn.BasicLSTMCell(
            config.lstm_size, forget_bias=0.0, state_is_tuple=True,
            reuse=not stage_flag == "train")

    def _build_rnn_graph_lstm(self, inputs, config, stage_flag):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.


        cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(config, stage_flag) for _ in range(config.num_layers)], state_is_tuple=True)

        # cell = self._get_lstm_cell(config, stage_flag)
        # c = tf.zeros([config.batch_size, config.lstm_size],
        #              tf.float32)
        # h = tf.zeros([config.batch_size, config.lstm_size],
        #          tf.float32)
        # self.initial_state = tf.contrib.rnn.LSTMStateTuple(h=h, c=c)
        self.initial_state = cell.zero_state(config.batch_size, tf.float32)
        print(type(self.initial_state))

        # self.initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self.initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.lstm_size])
        return output, state

    def export_ops(self, name):
        """Exports ops to collections."""
        self.name = name
        ops = {util.with_prefix(self.name, "cost"): self.loss}
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self.initial_state_name = util.with_prefix(self.name, "initial")
        self.final_state_name = util.with_prefix(self.name, "final")
        print (type(self.initial_state))
        util.export_state_tuples((self.initial_state,), self.initial_state_name)
        util.export_state_tuples((self.final_state,), self.final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self.stage_flag == "train":
            self.train_op = tf.get_collection_ref("train_op")[0]
        self.loss = tf.get_collection_ref(util.with_prefix(self.name, "cost"))[0]
        num_replicas = 1
        self.initial_state = util.import_state_tuples(
            self.initial_state, self.initial_state_name, num_replicas)[0]
        print (self.initial_state)
        self.final_state = util.import_state_tuples(
            self.final_state, self.final_state_name, num_replicas)[0]


# env = TradingEnv()
# env.load_market_data(DATA_CSV)
# env.init('OMGUSD')
# for _ in range(100):
#     print(env.act(1))


# # For further testing of ohlcv batch sampling
# def input_pipeline(filenames):
#     reader = tf.TextLineReader()
#     assert isinstance(filenames, list), "filenames argument must be list"
#     filename_queue = tf.train.string_input_producer(filenames)
#     key, value = reader.read(filename_queue)
#     record_defaults = [[0.], [''], ['']] + [[0.] for _ in range(5)]
#     id, timestamp, symbol, o, h, l, c, v = tf.decode_csv(value, record_defaults=record_defaults)
#     features = tf.stack([o, h, l, c, v])
#     return features

#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(100):
#         state = sess.run([timestamp, symbol, features])
#     coord.request_stop()
#     coord.join(threads)
# print(state)

# square_matrix = tf.Variable(np.array([[1, 2], [4, 8]]), dtype=tf.int32)
# # rank = tf.rank(square_matrix)
# with tf.Session() as sess:
#     # t = tf.Print(square_matrix, [square_matrix])
#     # result = t
#     # output = result.eval()
#     sess.run(tf.global_variables_initializer())
#     output = sess.run(square_matrix)
# print(output)
