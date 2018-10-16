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
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
import random


class CTC_Model():

    def __init__(self):
        print('a')
        self.a = 1

    def base_conv_layer(self,inputs,is_training):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9
            , 'updates_collections': None}

        with slim.arg_scope([slim.conv2d],kernel_size = [3,3],weights_regularizer=slim.l2_regularizer(1e-4),
                            normalizer_fn= slim.batch_norm,normalizer_params = batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],kernel_size = [2,2],stride=[2,1],padding='valid'):

                conv1 = slim.conv2d(inputs,64,padding='valid',scope='conv1')
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

                cnn_features = tf.squeeze(pool4, axis=1, name='features')

                return cnn_features

    def rnn_layers(self,cnn_features,num_units):
        cell = tf.contrib.rnn.GRUCell(num_units = num_units)
        outputs,output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=cnn_features,dtype=tf.float32)
        encoder_outputs = tf.concat(outputs,-1)

        return encoder_outputs,output_states

    def decode(self,helper, memory, scope, enc_state, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.A_UNITS, memory=memory)
            cell = tf.contrib.rnn.GRUCell(num_units=config.A_UNITS)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                            attention_layer_size=config.A_UNITS, output_attention=True)
            output_layer = Dense(units=config.ONE_HOT_SIZE)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=helper,
                initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=config.BATCH_SIZE).clone(
                    cell_state=enc_state[0]),
                output_layer=output_layer)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=config.SEQ_MAXSIZE)

            return outputs



    def build_network(self,inputs,train_output,target_output,sample_rate,is_training):
        cnn_features = self.base_conv_layer(inputs,is_training)
        outputs,output_states = self.rnn_layers(cnn_features,config.A_UNITS)

        output_embed = layers.embed_sequence(train_output,vocab_size=config.ONE_HOT_SIZE,embed_dim=config.ONE_HOT_SIZE,scope='embed')

        embeddings = tf.Variable(tf.truncated_normal(shape=[config.ONE_HOT_SIZE,config.ONE_HOT_SIZE],stddev=0.1),name='decoder_embedding')
        start_tokens = tf.zeros([config.BATCH_SIZE],dtype=tf.int64)

        train_length = np.array([config.SEQ_MAXSIZE]*config.BATCH_SIZE,dtype=np.int32)
        train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(output_embed,train_length,embeddings,sample_rate)

        train_outputs = self.decode(train_helper,outputs,'decode',output_states)

        pre_helper = seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens = tf.to_int32(start_tokens),end_token=1)

        pre_outputs = self.decode(pre_helper,outputs,'decode',output_states,reuse=True)

        pred_decode_result = pre_outputs[0].rnn_output

        mask = tf.cast(tf.sequence_mask(config.BATCH_SIZE * [train_length[0] - 1], train_length[0]),
                       tf.float32)
        att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, target_output, weights=mask)

        loss = tf.reduce_mean(att_loss)

        return loss,pred_decode_result





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



    def train(self):
        inputs = tf.placeholder(tf.float32, [None, 32, config.IMG_MAXSIZE, 1])
        train_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
        target_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
        sample_rate = tf.placeholder(tf.float32, shape=[], name='sample_rate')
        is_training = tf.placeholder(dtype=tf.bool)



        loss,  pred_decode_result = self.build_network(inputs, train_output, target_output,
                                                                            sample_rate,is_training)

        optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(loss)

        dataset = utils.DataSet()
        train_generator = dataset.train_data_generator(config.BATCH_SIZE)

        ctc_train_path = './ctc_train_path'
        ctc_val_path = './ctc_val_path'
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            i = 0

            if os.path.exists(config.MODEL_SAVE.replace('ctc.ckpt', '')):
                saver.restore(sess, config.MODEL_SAVE)
                print("restore")

            all_val_data = dataset.create_val_data()



            while True:
                images, labels_input, labels_output, train_label_list,epoch = next(train_generator)

                feedict = {inputs: images, train_output: labels_input, target_output: labels_output,
                            sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: True}
                sess.run(optimizer, feed_dict=feedict)



                if i %20 ==0:
                    val_image, val_labels_input, val_labels_output, val_label_list = random.sample(all_val_data,1)[0]
                    val_labels_input_ = val_labels_input[:config.BATCH_SIZE,...]
                    val_labels_output_ = val_labels_output[:config.BATCH_SIZE,...]


                    train_feedict = {inputs: images, train_output: labels_input, target_output: labels_output,
                                sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: False}

                    val_feedict = {inputs: val_image[:32,...], train_output: val_labels_input_, target_output: val_labels_output_,
                                sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: False}


                    train_result,loss_ = sess.run([pred_decode_result,loss ],feed_dict=train_feedict)
                    val_result = sess.run(pred_decode_result,feed_dict=val_feedict)

                    train_result = np.argmax(train_result,-1)
                    val_result = np.argmax(val_result,-1)

                    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

                    train_result = list(map(lambda y:''.join(list(map(lambda x:decode.get(x),y))).split('<EOS>')[0],train_result))
                    val_result = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))).split('<EOS>')[0], val_result))

                    print('loss:{}'.format(loss_))
                    print('train_label{}'.format(train_label_list[:5]))
                    print('train_pre{}'.format(train_result[:5]))
                    print('val_label{}'.format(val_label_list[:5]))
                    print('val_pre{}'.format(val_result[:5]))
                    # print('train_label_acc{}'.format(train_ctc_label_acc))
                    # print('train_seq_acc{}'.format(train_ctc_val_acc))
                    # print('val_label_acc{}'.format(val_ctc_label_acc_))
                    # print('val_seq_acc{}'.format(val_ctc_val_acc_))
                    print(
                        '----------------------------------------------------------------------------------------------------------------')

                if i % 100 == 0:
                    saver.save(sess, config.MODEL_SAVE)


                # if i%500 == 0:
                #     for i,val_data in enumerate(all_val_data):




                i= i+1











model = CTC_Model()
model.train()
# model.output('/home/wzh/1_4+6=78.png')
# model.analyze_result('/home/wzh/analyze')





# 0gaussian_53+45×95=3.jpg
