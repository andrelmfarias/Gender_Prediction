import tensorflow as tf
import numpy as np
import DataSets as ds
import Layers


def init_model(KeepProb_Dropout=1, input_dim=48*48):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, input_dim], name='x')
        y_desired = tf.placeholder(tf.float32, [None, 2], name='y_desired')
        ITM = tf.placeholder("bool", name='Is_Training_Mode')

    with tf.name_scope('CNN'):
        t = Layers.unflat(x, 48, 48, 1)
        nbfilter = 16
        nb_conv_per_block = 4
        for k in range(4):
            for i in range(nb_conv_per_block):
                d = Layers.conv(t, nbfilter, 3, 1, ITM,
                                'conv33_{}_{}'.format(k, i), KeepProb_Dropout)
                t = tf.concat([t, d], axis=3)
            t = Layers.maxpool(t, 2, 'pool')
            t = Layers.conv(t, 32, 3, 1, ITM, 'conv11_{}'.format(k), KeepProb_Dropout)
        t = Layers.flat(t)
        t = Layers.fc(t, 50, ITM, 'fc_1', KeepProb_Dropout)
        y = Layers.fc(t, 2, ITM, 'fc_2', KP_dropout=1.0, act=tf.nn.log_softmax)

    with tf.name_scope('cross_entropy'):
        diff = y_desired * y
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.75, staircase=True)

    Acc_Train = tf.placeholder("float", name='Acc_Train')
    Acc_Test = tf.placeholder("float", name='Acc_Test')

    return x, y
