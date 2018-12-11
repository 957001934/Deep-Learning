'''
import tensorflow-cpu
input px,labels
output pro_labels
author ambitious
date 18.12.5
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward#引入前向传播
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99#学习衰减率
REGULARIZER = 0.0001#正则化系数
STEPS = 50000#循环的轮数
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"


def backward(mnist):

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])#占位
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)#训练轮数
    #一般让模型的输出经过 sofemax 函数，以获得输出分类的概率分布，再与标准答案对比，求出交叉熵，得到损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)#求平均
    #.get_collection()方法将全部变量转换成列表
    loss = cem + tf.add_n(tf.get_collection('losses'))#交叉熵+所有参数正则化总损失
    #定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)
    #使用梯度下降优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所有待训练参数汇总为列表
    ema_op = ema.apply(tf.trainable_variables())
    #该函数实现将滑动平均和训练过程同步运行
#    with tf.control_dependencies([train_step, ema_op]):
#
    train_op = tf.group(train_step,ema_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()#将所有变量初始化
        sess.run(init_op)
        #断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载神经网络模型
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)#将mnist数据集输入到神经网络
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #保存神经网路模型，os.path.join拼接路径，标明并保存训练轮数
                #模型地址/模型名字/训练轮数
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)#加载数据集，如果没有则下载
    backward(mnist)

main()

