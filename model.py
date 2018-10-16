import tensorflow as tf
from config import Config as config
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2_base
import utils
import cv2
import numpy as np
from glob import glob
import os
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import ScheduledEmbeddingTrainingHelper
from utils import DataSet
import random
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'




class CTC_Model():

    def __init__(self):
        self.a = 1

    def base_conv_layer(self,inputs,widths,is_training):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9
            , 'updates_collections': None}

        with slim.arg_scope([slim.conv2d],kernel_size = [3,3],weights_regularizer=slim.l2_regularizer(1e-4),
                            normalizer_fn= slim.batch_norm,normalizer_params = batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],kernel_size = [2,1],stride=[2,1],padding='VALID'):

                conv1 = slim.conv2d(inputs,64,padding='VALID',scope='conv1')
                conv2 = slim.conv2d(conv1,64,scope='conv2')
                poo1 = slim.max_pool2d(conv2,kernel_size=[2,2],stride=[2,2],scope='pool1')

                conv3 = slim.conv2d(poo1,128,scope='conv3')
                conv4 = slim.conv2d(conv3,128,scope='conv4')
                pool2 = slim.max_pool2d(conv4,scope='pool2')

                conv5 = slim.conv2d(pool2,256,scope='conv5')
                conv6 = slim.conv2d(conv5,256,scope='conv6')
                pool3 = slim.max_pool2d(conv6,scope='pool3')

                conv7 = slim.conv2d(pool3,512,scope='conv7')
                conv8 = slim.conv2d(conv7,512,scope='conv8')
                pool4 = slim.max_pool2d(conv8,kernel_size=[3,1],stride=[3,1],scope='pool4')

                features = tf.squeeze(pool4, axis=1, name='features')

                conv1_trim = tf.constant(2 * (3// 2),
                                         dtype=tf.int32,
                                      name='conv1_trim')


                after_conv1 = widths - conv1_trim
                after_pool1 = tf.floor_div(after_conv1, 2)
                after_pool2 = after_pool1 -1
                after_pool3 = after_pool2 -1
                after_pool4 = after_pool3

                sequence_length = tf.reshape(after_pool4, [-1], name='seq_len')
                sequence_length = tf.maximum(sequence_length,1)

                return features, sequence_length

    def base_conv_layer2(self,inputs,widths,is_training):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9
            , 'updates_collections': None}

        with slim.arg_scope([slim.conv2d],kernel_size = [3,3],weights_regularizer=slim.l2_regularizer(1e-4),
                            normalizer_fn= slim.batch_norm,normalizer_params = batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],kernel_size=[2,2],stride=[2,1],padding='VALID'):

                conv1 = slim.conv2d(inputs,64,padding='VALID',scope='conv1')
                conv2 = slim.conv2d(conv1,64,scope='conv2')
                poo1 = slim.max_pool2d(conv2,stride=[2,2],scope='pool1')

                conv3 = slim.conv2d(poo1,128,scope='conv3')
                conv4 = slim.conv2d(conv3,128,scope='conv4')
                pool2 = slim.max_pool2d(conv4,scope='pool2')

                conv5 = slim.conv2d(pool2,256,scope='conv5')
                conv6 = slim.conv2d(conv5,256,scope='conv6')
                pool3 = slim.max_pool2d(conv6,scope='pool3')

                conv7 = slim.conv2d(pool3,512,scope='conv7')
                conv8 = slim.conv2d(conv7,512,scope='conv8')
                pool4 = slim.max_pool2d(conv8,kernel_size=[3,1],stride=[3,1],scope='pool4')

                features = tf.squeeze(pool4, axis=1, name='features')

                conv1_trim = tf.constant(2 * (3// 2),
                                         dtype=tf.int32,
                                      name='conv1_trim')


                after_conv1 = widths - conv1_trim
                after_pool1 = tf.floor_div(after_conv1, 2)
                after_pool2 = after_pool1 -1
                after_pool3 = after_pool2 -1
                after_pool4 = after_pool3

                sequence_length = tf.reshape(after_pool4, [-1], name='seq_len')
                sequence_length = tf.maximum(sequence_length,1)

                return features, sequence_length


    def crnn_conv_layer(self,inputs,widths,is_training):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9
            , 'updates_collections': None}

        with slim.arg_scope([slim.conv2d],kernel_size = [3,3],weights_regularizer=slim.l2_regularizer(1e-4)):
            with slim.arg_scope([slim.max_pool2d],kernel_size = [2,1],stride=[2,1],padding='SAME'):

                conv1 = slim.conv2d(inputs,64,scope='conv1')
                poo1 = slim.max_pool2d(conv1,scope='pool1')

                conv2 = slim.conv2d(poo1,128,scope='conv2')
                pool2 = slim.max_pool2d(conv2,scope='pool2')

                conv3 = slim.conv2d(pool2,256,scope='conv3')
                conv4 = slim.conv2d(conv3,256,scope='conv4')
                pool3 = slim.max_pool2d(conv1,kernel_size=[2,2],stride=[2,2],scope='pool3')

                conv5 = slim.conv2d(pool3,512,scope='conv5',normalizer_fn= slim.batch_norm,normalizer_params = batch_norm_params)
                conv6 = slim.conv2d(conv5,512,scope='conv6',normalizer_fn= slim.batch_norm,normalizer_params = batch_norm_params)
                pool4 = slim.max_pool2d(conv1,kernel_size=[2,2],stride=[2,2],scope='pool4')

                conv7 = slim.conv2d(pool4,512,padding='SAME',scope='conv7')

                features = tf.squeeze(conv7, axis=1, name='features')


                conv1_trim = tf.constant(2 * (3// 2),
                                         dtype=tf.int32,
                                      name='conv1_trim')


                after_pool1 = tf.floor_div(widths, 2)
                after_pool2 = tf.floor_div(after_pool1, 2)
                after_pool3 = after_pool2
                after_pool4 = after_pool3
                after_conv7 = after_pool4-conv1_trim

                sequence_length = tf.reshape(after_conv7, [-1], name='seq_len')
                sequence_length = tf.maximum(sequence_length,1)

                return features, sequence_length




    def rnn_layer(self,bottom_sequence,sequence_length,rnn_size,scope):

        cell_fw = tf.contrib.rnn.LSTMBlockCell(rnn_size)
        cell_bw = tf.contrib.rnn.LSTMBlockCell(rnn_size)

        rnn_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, bottom_sequence,
            sequence_length=sequence_length,
            time_major=True,
            dtype=tf.float32,
            scope=scope)

        rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')

        return rnn_output_stack,enc_state


    def rnn_layers(self,features, sequence_length, num_classes,units):

        logit_activation = tf.nn.relu
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope("rnn"):

            rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
            rnn1 ,_ = self.rnn_layer(rnn_sequence, sequence_length, units, 'bdrnn1')
            rnn2 ,_= self.rnn_layer(rnn1, sequence_length, units, 'bdrnn2')
            rnn_logits = tf.layers.dense(rnn2, num_classes + 1,
                                         activation=logit_activation,
                                         kernel_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         name='logits')

            return rnn_logits


    def crnn(self,inputs, width,is_training):
        features, sequence_length = self.base_conv_layer(inputs, width,is_training)
        logits = self.rnn_layers(features, sequence_length, len(config.ONE_HOT),config.RNN_UNITS)
        return logits ,sequence_length


    def crnn2(self,inputs, width,is_training):
        features, sequence_length = self.base_conv_layer2(inputs, width,is_training)
        logits = self.rnn_layers(features, sequence_length, len(config.ONE_HOT),512)
        return logits ,sequence_length



    def ctc_loss_layer(self,rnn_logits, sequence_labels, sequence_length):
        """Build CTC Loss layer for training"""
        loss = tf.nn.ctc_loss(sequence_labels, rnn_logits, sequence_length,
                              time_major=True,ignore_longer_outputs_than_inputs = True)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
        return loss


    def error(self,logits,sequence_length,sequence_label,label_length,greedy_decoder=True):
        if greedy_decoder:
            predictions, _ = tf.nn.ctc_greedy_decoder(logits,sequence_length,merge_repeated=True)
        else:

            predictions, _ = tf.nn.ctc_beam_search_decoder(logits,sequence_length,beam_width=128,top_paths=1,merge_repeated=True)

        hypothesis = tf.cast(predictions[0], tf.int32)  # for edit_distance
        label_errors = tf.edit_distance(hypothesis, sequence_label, normalize=False)
        sequence_errors = tf.count_nonzero(label_errors, axis=0)
        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error,
                                 tf.cast(total_labels, tf.float32),
                                 name='label_error')
        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32),
                                    tf.shape(label_length)[0],
                                    name='sequence_error')
        tf.summary.scalar('label_error',label_error)
        tf.summary.scalar('sequence_error',sequence_error)
        return 1-label_error,1-sequence_error


    def train(self):
            inputs = tf.placeholder(tf.float32, [None, 32, None, 1])
            width = tf.placeholder(tf.int32, [None])
            sequence_label = tf.sparse_placeholder(tf.int32)
            label_length = tf.placeholder(tf.int32,[None])
            is_training = tf.placeholder(tf.bool)

            logits, sequence_length =self.crnn2(inputs,width,is_training)

            loss = self.ctc_loss_layer(logits, sequence_label, sequence_length)

            label_error,sequence_error = self.error(logits,sequence_length,sequence_label,label_length)

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #     optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(loss)
            tf.train.MomentumOptimizer
            # optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(loss)
            optimizer = tf.train.MomentumOptimizer(config.LEARN_RATE,0.9).minimize(loss)

            dataset = utils.DataSet()
            train_generator = dataset.train_data_generator(config.BATCH_SIZE)
            all_val_data = dataset.create_val_data()

            ctc_train_path = './ctc_train_path'
            ctc_val_path = './ctc_val_path'
            # saver = tf.train.Saver(tf.global_variables())
            saver = tf.train.Saver()

            i = 0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # if os.path.exists(config.MODEL_SAVE.replace('ctc.ckpt','')):
                #     saver.restore(sess,config.MODEL_SAVE)
                #     print("restore")

                merged = tf.summary.merge_all()
                writer_train = tf.summary.FileWriter(ctc_train_path, sess.graph)
                writer_val = tf.summary.FileWriter(ctc_val_path, sess.graph)

                while True:
                    images, labels, width_, length_,epoch= next(train_generator)

                    feeddict = {inputs: images, sequence_label: (labels[0], labels[1], labels[2]), width: width_,label_length:length_,is_training:True}

                    sess.run(optimizer, feed_dict=feeddict)

                    if i % 20 == 0:
                        images_val, labels_val, width_val, length_val = random.sample(all_val_data,1)[0]

                        feeddict_train = {inputs: images, sequence_label: (labels[0], labels[1], labels[2]), width: width_,
                                    label_length: length_,is_training:False}


                        feeddict_val = {inputs: images_val, sequence_label: (labels_val[0], labels_val[1], labels_val[2]),
                                        width: width_val,label_length:length_val,is_training:False}



                        train_loss,train_label_error,train_sequence_error, train_log = sess.run([loss,label_error,sequence_error, merged], feed_dict=feeddict_train)
                        label_error_val, sequence_error_val,val_log = sess.run([label_error, sequence_error,merged],feed_dict=feeddict_val)

                        writer_train.add_summary(train_log, i)
                        writer_val.add_summary(val_log, i)
                        print('loss:{}'.format(train_loss))
                        print('train_label_acc{}'.format(train_label_error))
                        print('train_seq_acc{}'.format(train_sequence_error))
                        print('val_label_acc{}'.format(label_error_val))
                        print('val_seq_acc{}'.format(sequence_error_val))
                        print('epoch{}'.format(epoch))
                        print('----------------------------------------------------------------------------------------------------------------')
                    #
                    if i % 100 == 0:
                        saver.save(sess,config.MODEL_SAVE)


                    if i%1000==0:
                        label_error_val_all, sequence_error_val_all = 0,0
                        j =0
                        for i in range(len(all_val_data)):
                            images_val, labels_val, width_val, length_val = all_val_data[i]

                            feeddict_val = {inputs: images_val,
                                            sequence_label: (labels_val[0], labels_val[1], labels_val[2]),
                                            width: width_val, label_length: length_val, is_training: False}

                            label_error_val, sequence_error_val, val_log = sess.run(
                                [label_error, sequence_error, merged], feed_dict=feeddict_val)
                            label_error_val_all = label_error_val_all+label_error_val
                            sequence_error_val_all = sequence_error_val_all+sequence_error_val
                            j=j+1
                        label_error_val_all = label_error_val/j
                        sequence_error_val_all = sequence_error_val_all/j
                        f = open('log.txt','a')
                        f.write('val_label_acc_all{}'.format(label_error_val_all))
                        f.write('val_seq_acc_all{}'.format(sequence_error_val_all))
                        f.write('----------------------------------------------------------------------------------------------------------------')
                        print('val_label_acc_all{}'.format(label_error_val_all))
                        print('val_seq_acc_all{}'.format(sequence_error_val_all))
                        print('----------------------------------------------------------------------------------------------------------------')
                        print(
                            '----------------------------------------------------------------------------------------------------------------')



                    i = i + 1


    def output(self,path):
        image,  wides = DataSet().get_imges([path])
        inputs = tf.placeholder(tf.float32, [None, 32, None, 1])
        width = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool)
        logits, sequence_length = self.crnn(inputs, width,is_training)

        decoder, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length, merge_repeated=True)
        decoder = decoder[0]

        dense_decoder = tf.sparse_to_dense(sparse_indices=decoder.indices, output_shape=decoder.dense_shape,
                                           sparse_values=decoder.values,default_value = -1)

        with tf.Session() as sess:

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess,config.MODEL_SAVE)

            sentence =sess.run(dense_decoder,feed_dict={inputs:image,width:wides,is_training:False})

            sentence = sentence.tolist()

            decode = dict(zip(config.ONE_HOT.values(),config.ONE_HOT.keys()))

            result = ''.join(list(map(lambda x:decode.get(x),sentence[0])))


        print(result)


    def analyze_result(self,paths):
        image_paths = sorted(glob(os.path.join(paths,'*')))
        image, wides = DataSet().get_imges(image_paths)
        def getlab(x):
            result = x.split('_')[-1].replace('.jpg','')
            result = result.replace('.png','')
            return result

        labels = map(getlab,image_paths)

        inputs = tf.placeholder(tf.float32, [None, 32, None, 1])
        width = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool)
        logits, sequence_length = self.crnn(inputs, width, is_training)

        decoder, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length, merge_repeated=True)
        decoder = decoder[0]

        dense_decoder = tf.sparse_to_dense(sparse_indices=decoder.indices, output_shape=decoder.dense_shape,
                                           sparse_values=decoder.values,default_value = -1)
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, config.MODEL_SAVE)

            result = []

            if image.shape[0] <= config.BATCH_SIZE:
                sentence = sess.run(dense_decoder, feed_dict={inputs: image, width: wides,is_training:False})

                sentence = sentence.tolist()

                decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

                result.extend(sentence)

            else:
                index = 0
                while (index+1)*config.BATCH_SIZE <= image.shape[0]:
                    if (index+1)*config.BATCH_SIZE > image.shape[0]:
                        end = image.shape[0]
                    else:
                        end = (index+1)*config.BATCH_SIZE

                    sentence = sess.run(dense_decoder, feed_dict={inputs: image[index*config.BATCH_SIZE:end,...], width: wides[index*config.BATCH_SIZE:end,...],is_training:True})

                    sentence = sentence.tolist()

                    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

                    result.extend(sentence)
                    index = index+1


            result = list(map(lambda y: ''.join(list(map(lambda x:decode.get(x), y))), result))

            result = dict(zip(labels,result))



        print(result)



#
#

# model = CTC_Model()
# model.train()
# model.output('/home/wzh/ocr/0.jpg')
# # model.analyze_result('/home/wzh/analyze')

# inputs = tf.Variable(tf.zeros([1,32,120,1]))




# 0gaussian_53+45×95=3.jpg
