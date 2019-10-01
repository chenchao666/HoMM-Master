import tensorflow as tf
from numpy import *
import matplotlib.pyplot as plt
import warnings
import os
from scipy import io
import tensorflow.contrib.slim as slim
warnings.filterwarnings("ignore")
# os.environ['CUDA_VISIBLE_DEVICES']=''



class AlexNet():

    def __init__(self,images,num_class,keep_prob,Training_flag,scope_name,reuse=True):
        self.input=images
        self.NUM_class=num_class
        self.KEEP_prob=keep_prob
        self.scope=scope_name
        self.Regularizer_scale = 0.01
        self.Training_flag = Training_flag
        self.reuse=reuse
        self.BuildAlexNet()



    def BuildAlexNet(self):
        with tf.variable_scope("input_layer"):
            input_layer=self.input

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="VALID",
                                  activation=tf.nn.relu, name=self.scope+"conv1",reuse=self.reuse)
        norm1 = tf.layers.batch_normalization(conv1, training=self.Training_flag, name=self.scope+"norm1", reuse=self.reuse)
        pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[3,3], strides=2, padding="VALID", name=self.scope+"pool1")


        conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], strides=1, padding="SAME",
                                  activation=tf.nn.relu,name=self.scope+"conv2", reuse=self.reuse)
        norm2 = tf.layers.batch_normalization(conv2, training=self.Training_flag, name=self.scope+"norm_2", reuse=self.reuse)
        pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3, 3], strides=2, padding="VALID",name=self.scope + "pool2")


        conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], strides=1, padding="SAME",
                                  activation=tf.nn.relu,name=self.scope + "conv3", reuse=self.reuse)
        norm3 = tf.layers.batch_normalization(conv3, training=self.Training_flag, name=self.scope + "norm3", reuse=self.reuse)


        conv4 = tf.layers.conv2d(inputs=norm3, filters=384, kernel_size=[3, 3], strides=1, padding="SAME",
                                  activation=tf.nn.relu,name=self.scope + "conv4", reuse=self.reuse)
        norm4 = tf.layers.batch_normalization(conv4, training=self.Training_flag, name=self.scope + "norm4", reuse=self.reuse)

        conv5 = tf.layers.conv2d(inputs=norm4, filters=256, kernel_size=[3, 3], strides=1, padding="SAME",
                                  activation=tf.nn.relu,name=self.scope + "conv5", reuse=self.reuse)
        norm5 = tf.layers.batch_normalization(conv5, training=self.Training_flag, name=self.scope + "norm5", reuse=self.reuse)
        pool5 = tf.layers.max_pooling2d(inputs=norm5, pool_size=[3, 3], strides=2, padding="VALID", name=self.scope + "pool5")

        flatten_size=int(pool5.shape[1]*pool5.shape[2]*pool5.shape[3])
        flatten_layer=tf.reshape(pool5,[int(pool5.shape[0]),flatten_size])

        self.fc6 = tf.layers.dense(inputs=flatten_layer, units=4096, name=self.scope + "fc6", reuse=self.reuse)
        droplayer6 = tf.layers.dropout(inputs=self.fc6, rate=self.KEEP_prob, training=self.Training_flag, name=self.scope+"droplayer6")

        self.fc7 = tf.layers.dense(inputs=droplayer6, units=4096, name=self.scope + "fc7", reuse=self.reuse)
        droplayer7 = tf.layers.dropout(inputs=self.fc7, rate=self.KEEP_prob, training=self.Training_flag, name=self.scope + "droplayer7")

        self.fc8 = tf.layers.dense(inputs=droplayer7, units=self.NUM_class, name=self.scope + "fc8", reuse=self.reuse)
        self.softmax_output = tf.nn.softmax(logits=self.fc8, name=self.scope+"softlayer")



    def weights_initial(self,session):
        weights_dict=load("bvlc_alexnet.npy")
        dict=weights_dict.item()
        for op_name in dict:
            if op_name== u'fc8':
                continue
            else:
                with tf.variable_scope(self.scope + op_name, reuse=True):
                    for data in dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('bias', trainable=True)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('kernel', trainable=True)
                            if var.shape==data.shape:
                                session.run(var.assign(data))
                            else:
                                data=concatenate((data,data),axis=2)
                                session.run(var.assign(data))






    def Train(self):
        pass



    def Test(self):
        pass






