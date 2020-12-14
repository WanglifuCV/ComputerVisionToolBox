# -*- coding:utf-8 -*-

import tensorflow as tf


def LSTM(num_hidden):
    lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden)
    return lstm_cell


def MultiLSTM(num_hidden, layer_num):

    cell = []
    for _ in range(layer_num):
        lstm_cell = tf.contrib.rnn.LSTMCell(
            num_hidden,
            state_is_tuple=True
        )
        cell.append(lstm_cell)

    multi_lstm = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)
    return multi_lstm



def BdLSTM(inputs, num_hidden, name_scope):
    """
    实现双向lstm (static)
    Args: 
        inputs: A tensor with shape [batch, max_time_step, channel]
    """
    with tf.name_scope(name_scope):
        lstm_cell_forward = LSTM(num_hidden=num_hidden)
        lstm_cell_backward = LSTM(num_hidden=num_hidden)

        shape = inputs.shape.as_list()

        inputs = tf.reshape(inputs, [-1, shape[2]])

        inputs = tf.split(inputs, shape[1], axis=0)

        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_cell_forward, lstm_cell_backward, inputs, dtype=tf.float32)

        outputs = [tf.expand_dims(out, 0) for out in outputs]

        outputs = tf.concat(outputs, axis=0)

        outputs = tf.transpose(outputs, [1, 0, 2])

        return outputs


def BdLSTM_dynamic(inputs, num_hidden, name_scope):
    """
    实现dynamic lstm
    Args:
        inputs: A tensor with shape [batch, max_time_step, channel]
        num_hidden: A integer of hidden units of lstm
        name_scop: name_scope of tf

    Return:
        output: A tensor of bidirectional lstm output
    """
    with tf.name_scope(name_scope):
        # 单层
        # lstm_cell_fw = LSTM(num_hidden)
        # lstm_cell_bw = LSTM(num_hidden)

        # 多层？
        lstm_cell_fw = MultiLSTM(num_hidden, layer_num=2)
        lstm_cell_bw = MultiLSTM(num_hidden, layer_num=2)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, 
            lstm_cell_bw, 
            inputs,
            dtype=tf.float32,
            scope=name_scope
        )

        outputs = tf.concat(outputs, 2)

        return outputs


if __name__ == '__main__':
    inputs = tf.placeholder(shape=[32, 80, 256], dtype=tf.float32)
    # outputs = BdLSTM(inputs=inputs, num_hidden=10, name_scope='lstm')
    outputs = BdLSTM_dynamic(inputs=inputs, num_hidden=10, name_scope='lstm_dynamic')
