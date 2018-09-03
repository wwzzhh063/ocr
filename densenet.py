"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, widths, is_training,
             dropout_keep_prob=1,
             scope='densenet'):


    growth = 24




    with tf.variable_scope(scope, 'DenseNet', [images]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training)) as ssc:

            current =  slim.conv2d(images, 64, [3, 3],  padding='valid',
                                                            scope='pre_conv2')



            current = block(current, 6, growth, scope='lblock1')
            current = slim.conv2d(current,128,[1,1])
            current = slim.max_pool2d(current, [2, 2], stride=[2,2], scope='transition1_pool2')



            current =  block(current, 8, growth, scope='lblock2')
            current = slim.conv2d(current, 128, [1, 1])
            current =  slim.max_pool2d(current, [2, 1], stride=[2,1],scope='transition2_pool2')


            current =  block(current, 16, growth, scope='myblock3')
            current = slim.conv2d(current, 128, [1, 1])
            current =  slim.max_pool2d(current, [2, 1], stride=[2,1], scope='transition3_pool2')


            current =  block(current, 12, growth, scope='lblock3')
            current =  slim.max_pool2d(current, [3, 1],stride=[3,1], scope='lglobal_pool2')


            features =  tf.squeeze(current, axis=1, name='features')


            conv1_trim = tf.constant(2 * (3 // 2),
                                     dtype=tf.int32,
                                     name='conv1_trim')

            after_conv1 = widths - conv1_trim
            after_pool1 = tf.floor_div(after_conv1, 2)
            after_pool2 = after_pool1 - 1
            after_pool3 = after_pool2 - 1
            after_pool4 = after_pool3

            sequence_length = tf.reshape(after_pool4, [-1], name='seq_len')


    return features, sequence_length


def bn_drp_scope(is_training, keep_prob=0.8):
    with slim.arg_scope(
            [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
                [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):


    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False),
            activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
