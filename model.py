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
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
import random
os.environ['CUDA_VISIBLE_DEVICES']='1'

class CTC_Model():

    def __init__(self):
        # print('a')
        self.a = 1

    def base_conv_layer(self,inputs,is_training):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9
            , 'updates_collections': None, 'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]}

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

    def decode(self,helper, memory, scope, enc_state,batch_size ,reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.A_UNITS, memory=memory)
            cell = tf.contrib.rnn.GRUCell(num_units=config.A_UNITS)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                            attention_layer_size=config.A_UNITS, output_attention=True)
            output_layer = Dense(units=config.ONE_HOT_SIZE)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=helper,
                    initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(
                    cell_state=enc_state[0]),
                output_layer=output_layer)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=config.SEQ_MAXSIZE)

            return outputs

    def accuracy(self,pred_decode_result,target_output,label_length):
        hypothesis = tf.contrib.layers.dense_to_sparse(tf.argmax(pred_decode_result,-1),1)
        hypothesis = tf.cast(hypothesis, tf.int32)  # for edit_distance

        target_output = tf.contrib.layers.dense_to_sparse(target_output,1)
        target_output = tf.cast(target_output,tf.int32)

        label_errors = tf.edit_distance(hypothesis, target_output, normalize=False)

        sequence_errors = tf.count_nonzero(label_errors, axis=0)


        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error,
                                 tf.cast(total_labels, tf.float32),
                                 name='label_error')

        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32),
                                    tf.shape(label_length)[0],
                                    name='sequence_error')


        tf.summary.scalar('label_error', label_error)
        tf.summary.scalar('sequence_error', sequence_error)

        return 1-label_error,1-sequence_error



    def build_network(self,inputs,train_output,target_output,sample_rate,is_training,label_length,batch_size):
        cnn_features = self.base_conv_layer(inputs,is_training)
        outputs,output_states = self.rnn_layers(cnn_features,config.A_UNITS)

        output_embed = layers.embed_sequence(train_output,vocab_size=config.ONE_HOT_SIZE,embed_dim=config.ONE_HOT_SIZE,scope='embed')

        embeddings = tf.Variable(tf.truncated_normal(shape=[config.ONE_HOT_SIZE,config.ONE_HOT_SIZE],stddev=0.1),name='decoder_embedding')
        start_tokens = tf.zeros(batch_size,dtype=tf.int64)

        train_length = np.array([config.SEQ_MAXSIZE]*config.BATCH_SIZE,dtype=np.int32)
        train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(output_embed,train_length,embeddings,sample_rate)

        train_outputs = self.decode(train_helper,outputs,'decode',output_states,config.BATCH_SIZE)

        pre_helper = seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens = tf.to_int32(start_tokens),end_token=1)

        pre_outputs = self.decode(pre_helper,outputs,'decode',output_states,batch_size,reuse=True)

        pred_decode_result = pre_outputs[0].rnn_output

        char_acc,seq_acc = self.accuracy(pred_decode_result,target_output,label_length)

        mask = tf.cast(tf.sequence_mask(config.BATCH_SIZE * [train_length[0] - 1], train_length[0]),
                       tf.float32)
        att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, target_output, weights=mask)

        loss = tf.reduce_mean(att_loss)

        return loss,char_acc,seq_acc,pred_decode_result




    def train(self):
        inputs = tf.placeholder(tf.float32, [None, 32, config.IMG_MAXSIZE, 1])
        train_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
        target_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
        sample_rate = tf.placeholder(tf.float32, shape=[], name='sample_rate')
        is_training = tf.placeholder(dtype=tf.bool)
        label_length = tf.placeholder(tf.int32, [None])
        batch_size = tf.placeholder(tf.int32,[1])



        loss, char_acc,seq_acc, pred_decode_result = self.build_network(inputs, train_output, target_output,
                                                                            sample_rate,is_training,label_length,batch_size)

        optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(loss)

        dataset = utils.DataSet()
        train_generator = dataset.train_data_generator(config.BATCH_SIZE)

        ctc_train_path = './ctc_train_path'
        ctc_val_path = './ctc_val_path'
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver()


        with tf.Session() as sess:
            i = 0

            if os.path.exists(config.MODEL_SAVE.replace('ctc.ckpt', '')):
                saver.restore(sess,config.MODEL_SAVE)
                print("restore")
            else:
                sess.run(tf.global_variables_initializer())

            all_val_data = dataset.create_val_data()




            while True:
                images, labels_input, labels_output, train_label_list,label_len,epoch = next(train_generator)

                feedict = {inputs: images, train_output: labels_input, target_output: labels_output,
                            sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: True,label_length:label_len,batch_size:[images.shape[0]]}
                sess.run(optimizer, feed_dict=feedict)



                if i %20 ==0:
                    val_image, val_labels_input, val_labels_output, val_label_list,label_len = random.sample(all_val_data,1)[0]


                    train_feedict = {inputs: images, train_output: labels_input, target_output: labels_output,
                                sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: False,label_length:label_len,batch_size:[images.shape[0]]}

                    val_feedict = {inputs: val_image, train_output: val_labels_input, target_output: val_labels_output,
                                sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: False,label_length:label_len,batch_size:[val_image.shape[0]]}


                    train_result,train_char_acc,train_seq_acc,loss_ = sess.run([pred_decode_result,char_acc,seq_acc,loss ],feed_dict=train_feedict)
                    val_result,val_char_acc,val_seq_acc = sess.run([pred_decode_result,char_acc,seq_acc],feed_dict=val_feedict)

                    train_result = np.argmax(train_result,-1)
                    val_result = np.argmax(val_result,-1)

                    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

                    train_result = list(map(lambda y:''.join(list(map(lambda x:decode.get(x),y))).split('<EOS>')[0],train_result))
                    val_result = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))).split('<EOS>')[0], val_result))

                    print('loss:{}'.format(loss_))
                    print('train_label_acc{}'.format(train_char_acc))
                    print('train_seq_acc{}'.format(train_seq_acc))
                    print('val_label_acc{}'.format(val_char_acc))
                    print('val_seq_acc{}'.format(val_seq_acc))
                    print('epoch{}'.format(epoch))
                    print('train_label{}'.format(train_label_list[:3]))
                    print('train_pre{}'.format(train_result[:3]))
                    print('val_label{}'.format(val_label_list[:3]))
                    print('val_pre{}'.format(val_result[:3]))
                    # print('train_label_acc{}'.format(train_ctc_label_acc))
                    # print('train_seq_acc{}'.format(train_ctc_val_acc))
                    # print('val_label_acc{}'.format(val_ctc_label_acc_))
                    # print('val_seq_acc{}'.format(val_ctc_val_acc_))
                    print(
                        '----------------------------------------------------------------------------------------------------------------')

                if i % 100 == 0:
                    saver.save(sess, config.MODEL_SAVE,global_step=int(i/100))


                if i%500 == 0:
                    label_error_val_all, sequence_error_val_all = 0, 0
                    for i in range(len(all_val_data)):
                        val_image, val_labels_input, val_labels_output, val_label_list, label_len = all_val_data[i]

                        val_feedict = {inputs: val_image, train_output: val_labels_input,
                                       target_output: val_labels_input,
                                       sample_rate: np.min([1., 0.2 * epoch + 0.2]), is_training: False,label_length:label_len ,batch_size:[val_image.shape[0]]}

                        val_char_acc, val_seq_acc = sess.run([char_acc, seq_acc],
                                                             feed_dict=val_feedict)
                        label_error_val_all = label_error_val_all + val_char_acc
                        sequence_error_val_all = sequence_error_val_all + val_seq_acc
                    j = len(all_val_data)
                    label_error_val_all = label_error_val_all / j
                    sequence_error_val_all = sequence_error_val_all / j
                    f = open('log.txt', 'a')
                    f.write('val_label_acc_all{}'.format(label_error_val_all))
                    f.write('val_seq_acc_all{}'.format(sequence_error_val_all))
                    f.write(
                        '----------------------------------------------------------------------------------------------------------------')
                    print('val_label_acc_all{}'.format(label_error_val_all))
                    print('val_seq_acc_all{}'.format(sequence_error_val_all))
                    print(
                        '----------------------------------------------------------------------------------------------------------------')
                    print(
                        '----------------------------------------------------------------------------------------------------------------')

                i= i+1



    def output(self,img):
        inputs = tf.placeholder(tf.float32, [None, 32, config.IMG_MAXSIZE, 1])
        is_training = tf.placeholder(dtype=tf.bool)
        batch_size = tf.placeholder(tf.int32,[1])

        cnn_features = self.base_conv_layer(inputs, is_training)
        outputs, output_states = self.rnn_layers(cnn_features, config.A_UNITS)


        embeddings = tf.Variable(tf.truncated_normal(shape=[config.ONE_HOT_SIZE, config.ONE_HOT_SIZE], stddev=0.1),
                                 name='decoder_embedding')
        start_tokens = tf.zeros(batch_size, dtype=tf.int64)


        pre_helper = seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)

        pre_outputs = self.decode(pre_helper, outputs, 'decode', output_states,batch_size)

        pred_decode_result = pre_outputs[0].rnn_output
        pred_decode_result = tf.argmax(pred_decode_result,-1)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            if os.path.exists(config.MODEL_SAVE.replace('ctc.ckpt', '')):
                saver.restore(sess, config.MODEL_SAVE)
                print("restore")

            print(sess.run(pred_decode_result,feed_dict={inputs:img,is_training:False,batch_size:[1]}))
















model = CTC_Model()

# img = cv2.imread('0.jpg')
# img = utils.image_normal(img)
# model.output(img)
model.train()
# model.output('/home/wzh/1_4+6=78.png')
# model.analyze_result('/home/wzh/analyze')





# 0gaussian_53+45×95=3.jpg
