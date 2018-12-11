#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:#复现之前定义的计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)
        #加载参数的滑动平均值
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        #实例化具有滑动平均的 saver 对象
        saver = tf.train.Saver(ema_restore)
		#从第一维中提取最大值，即提取出最有可能的标签，进行比较
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #.cast转换成指定类型(将布尔型转换为实数型）,求准确率平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                #断点续训
                #从保存模型的路径中加载模型，
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载神经网络模型
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    #从保存路径中加载训练轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #计算出准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
