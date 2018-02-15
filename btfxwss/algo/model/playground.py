import numpy as np
import tensorflow as tf
import btfxwss.algo.tests.reader as reader
from btfxwss.algo.model.q_trader import run_epoch, PTBInput, Model
from collections import namedtuple

PTB_DATA_PATHNAME = '/home/robert/PyProjects/btfxwss/btfxwss/algo/tests'


def read_ptd_raw_data(pathname):
    train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(pathname)
    return train_data, valid_data, test_data, vocabulary


if __name__ == '__main__':
    data = read_ptd_raw_data(PTB_DATA_PATHNAME)
    Config = namedtuple('Config', 'batch_size, num_steps, lstm_size, num_epochs, vocab_size, init_scale, num_layers')
    config = Config(batch_size=10, num_steps=20, lstm_size=500, num_epochs=5,
                    vocab_size=10000, init_scale=0.1, num_layers=1)
    # writer = tf.summary.FileWriter('/home/robert/PyProjects/btfxwss/btfxwss/algo/model', graph=graph)
    # # writer.flush()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):
            train_input = PTBInput(data[0], config)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = Model(train_input, config)
                # models = {"Train": model}
                # for name, model in models.items():
                #         model.export_ops(name)
                #     metagraph = tf.train.export_meta_graph()
        soft_placement = False
    #
    # with tf.Graph().as_default():
        #     tf.train.import_meta_graph(metagraph)
        #     for model in models.values():
        #         model.import_ops()
        sv = tf.train.Supervisor(logdir="/home/robert/PyProjects/btfxwss/btfxwss/algo/model/")
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        print (tf.get_collection(tf.GraphKeys.INIT_OP))
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.num_epochs):
                run_epoch(model, session)
