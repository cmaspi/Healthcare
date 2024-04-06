import numpy as np
import tensorflow as tf
import random


class NeuralNet(object):
    """
    Build Neural Net with 64 sigmoid Layers and 1 hiddel sigmoid layer

    Input : self
            batch_size : (int) size of the batchs to learn on 10
            input_size : (int) length of input data
            output_size : (int) lenght of output 1
            is_training : (Bool) whether or not to train adam

    Output: self

    """
    def __init__(self,batch_size,input_size, output_size, is_training=False):
        # Placeholders for inputs.
        self.input_data = tf.placeholder(tf.float32, [batch_size,input_size])
        self.targets = tf.placeholder(tf.float32, [batch_size,output_size])

        inputs = self.input_data
        print(inputs)
        with tf.variable_scope('dense'):
            '''
            input_3d=tf.expand_dims(inputs, 2)
            conv = tf.layers.conv1d(input_3d, 3, 3, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, 64, name='fc')
            #fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            '''
            #hidden = tf.layers.dense(inputs, 64, tf.nn.relu)
            hidden1 = tf.layers.dense(inputs, 64, tf.nn.sigmoid)
            #hidden2 = tf.layers.dense(hidden1, 10, tf.sigmoid)
            output=tf.layers.dense(hidden1,output_size,activation=tf.sigmoid)

        predict = output
        target = self.targets
        loss=[]
        for i in range(batch_size):
            loss0=tf.losses.mean_squared_error(target[i],predict[i])
            #loss0 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target[i], logits=predict[i])
            loss.append(loss0)


        self.cost = tf.reduce_sum(loss)/batch_size
        self.outputs = predict#tf.nn.softmax(predict)



        if is_training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
            optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
