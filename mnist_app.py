'''
input picture_path
output predict_num
author ambitious
'''

import tensorflow as tf
import mnist_forward
import mnist_backward
import numpy as np
from PIL import Image


def restore_model(testPicArr):#输入图象像素处理后的矩阵
    #创建一个默认图，在图中操作
    with tf.Graph().as_default() as tg:
        x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y=mnist_forward.forward(x,None)
        preValue=tf.arg_max(y,1)#预测标签
        #滑动平均
        variable_averages=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            #模型加载
            ckpt=tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                #恢复当前会话
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue=sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print("没有发现文件")
                return -1

def pre_pic(picName):#对图片进行预处理
    img=Image.open(picName)
    reIm=img.resize((28,28),Image.ANTIALIAS)
    im_arr=np.array(reIm.convert('L'))#转变图象（1为二值图象，非黑（0）即白（255）；L为灰色图象，每个像素点是0-255的值）
    threshold=50#阈值
    for i in range(28):
        for j in range(28):
            im_arr[i][j]=255-im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j]=0#白色变黑色
            else:
                im_arr[i][j]=255#黑色变白色

    nm_arr=im_arr.reshape([1,784])
    nm_arr=nm_arr.astype(np.float32)#转换为浮点
    #转换为0.或1.# 的浮点数
    img_ready=np.multiply(nm_arr,1.0/255.0)
    #np.multiply()数组对于位置相乘
    return img_ready

def application():
    testNum = input("input the number of test pictures:")
    for i in range(int(testNum)):
        testPic =input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:{}".format(preValue))

application()