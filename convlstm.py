import tensorflow as tf
from config import Config as config
import tensorflow.contrib.slim as slim
import numpy as np



def rnn_layer(bottom_sequence, sequence_length,scope):
    cell_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,input_shape=[4,4,256], output_channels = 256,kernel_shape =[3,3])
    cell_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,input_shape=[4,4,256], output_channels = 256,kernel_shape =[3,3])

    rnn_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)

    rnn_output_stack = tf.concat(rnn_output, -1, name='output_stack')

    return rnn_output_stack, enc_state


def rnn_layers(features, sequence_length):
    with tf.variable_scope("rnn"):
        rnn_sequence = tf.transpose(features, perm=[1, 0, 2,3,4], name='time_major')
        rnn1, _ = rnn_layer(rnn_sequence, sequence_length,'lstm1')
        rnn2, _ = rnn_layer(rnn1, sequence_length,'lsmt2')

        return rnn2

x = tf.placeholder(tf.float32, [None, 5, 4, 4,256])
inputs = np.ones([10,5,4,4,256])
sequence_len = np.ones([10])+4
y = tf.placeholder(tf.int32,[10])
# x = tf.ones([10,5,4,4,256])
# y = tf.ones([10],dtype=tf.int32)+5

rnn2 = rnn_layers(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(rnn2,feed_dict={x:inputs,y:sequence_len}))

